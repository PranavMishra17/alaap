"""
The mapper: description -> voice identity vector.

This is the only trained component in the architecture, and PHASE-02's
subject. Everything else in the pipeline is frozen.

DESIGN, driven by measured results rather than the literature:

  E1 measured how NOVEL each generation strategy is, in units of the
  manifold's own natural speaker spacing (nn_ratio, where 1.0 = as far from
  known speakers as a genuinely new real speaker would be):

      SLERP between nearest anchors   0.20x   safe, near-duplicate
      GMM k=5 over the PCA manifold   0.89x   novel, still in-distribution
      full-covariance Gaussian        1.29x   overshoots, off-manifold

  So retrieval and generation are not competing designs -- they are the two
  ends of one axis. `novelty` blends them. That is the adherence-vs-diversity
  dial from RESEARCH/01 section 4.2 option 3, made concrete and, unusually,
  calibrated against measured numbers rather than guessed.

  RESEARCH/01 also found that ZERO of eight surveyed papers, and no
  commercial API, exposes this as a user-facing control.

INVARIANTS OBSERVED:
  I1  all work happens in SpeakerSpace working coordinates; the raw vector is
      only produced at the very end, via decode() with shell projection
  I5  the mapper produces TIMBRE only. Never emotion, never per-line style.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import SpeakerSpace


# ------------------------------------------------------------- text encoder
class TextEncoder:
    """
    Frozen sentence embedder. Not trained (scope section 7 component 1).

    all-MiniLM-L6-v2: Apache-2.0, 384-d, ~90MB, runs on CPU in milliseconds.
    That matters because RESEARCH/11 established minting should be CPU-cheap
    -- the GPU cost belongs to rendering, not to designing a voice.
    """
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_id: str | None = None, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id or self.MODEL
        self.model = SentenceTransformer(self.model_id, device=device)

    def encode(self, texts: list[str] | str) -> np.ndarray:
        one = isinstance(texts, str)
        out = self.model.encode([texts] if one else texts,
                                normalize_embeddings=True,
                                show_progress_bar=False)
        return np.asarray(out, dtype=np.float64)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


# ------------------------------------------------------------------- mapper
@dataclass
class MintResult:
    vector: np.ndarray          # raw-space speaker vector, shell-projected
    working: np.ndarray         # working-space coordinates
    novelty: float
    anchors: list[int]
    anchor_similarity: float
    strategy: str


class RetrievalMapper:
    """
    Contrastive retrieval + calibrated interpolation.

    Fit on (caption, speaker_vector) pairs. At mint time:
        description -> text embedding -> top-k nearest captions
                    -> their speaker vectors as anchors
                    -> blend between SLERP(anchors) and a GMM sample
                       according to `novelty`
                    -> decode to raw space, project onto the shell
    """

    def __init__(self, space: SpeakerSpace, text_encoder: TextEncoder,
                 pca_dims: int = 50, gmm_components: int = 5):
        self.space = space
        self.text = text_encoder
        self.pca_dims = pca_dims
        self.gmm_components = gmm_components
        self.T: Optional[np.ndarray] = None      # (N, dt) caption embeddings
        self.P: Optional[np.ndarray] = None      # (N, k)  speaker PCA coords
        self.captions: list[str] = []
        self.gmm = None

    # ---------------------------------------------------------------- fit
    def fit(self, captions: list[str], vectors: np.ndarray) -> "RetrievalMapper":
        assert len(captions) == len(vectors), "captions and vectors must align"
        self.captions = list(captions)
        self.T = self.text.encode(self.captions)
        W = self.space.encode(np.asarray(vectors, dtype=np.float64))
        k = min(self.pca_dims, self.space.components.shape[0])
        self.P = self.space.to_pca(W)[:, :k]
        self.k = k
        from sklearn.mixture import GaussianMixture
        ncomp = min(self.gmm_components, max(1, len(self.P) // 10))
        self.gmm = GaussianMixture(ncomp, covariance_type="full", reg_covar=1e-4,
                                   random_state=0, max_iter=500).fit(self.P)
        return self

    # --------------------------------------------------------------- mint
    @staticmethod
    def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return (1 - t) * a + t * b
        ua, ub = a / na, b / nb
        om = np.arccos(np.clip(ua @ ub, -1, 1))
        if om < 1e-6:
            return (1 - t) * a + t * b
        v = (ua * np.sin((1 - t) * om) + ub * np.sin(t * om)) / np.sin(om)
        return v * ((1 - t) * na + t * nb)

    def mint(self, description: str, novelty: float = 0.5, top_k: int = 4,
             seed: int | None = None) -> MintResult:
        """
        novelty 0.0  pure retrieval/SLERP -- safest, near-duplicate of catalog
                     (E1: 0.20x natural speaker spacing)
        novelty 1.0  pure GMM sample -- most novel, still in-distribution
                     (E1: 0.89x natural spacing)

        Values above 1.0 are NOT exposed: E1 showed the next step up
        (full-covariance Gaussian) overshoots to 1.29x and lands off-manifold.
        """
        if self.T is None:
            raise RuntimeError("mapper not fitted")
        novelty = float(np.clip(novelty, 0.0, 1.0))
        rng = np.random.default_rng(seed)

        q = self.text.encode(description)[0]
        sims = self.T @ q
        order = np.argsort(-sims)[:max(top_k, 2)]
        anchor_sim = float(sims[order[0]])

        # retrieval end: SLERP among the top-k anchors, weighted by similarity
        w = sims[order]
        w = np.exp((w - w.max()) * 8.0); w = w / w.sum()
        cur = self.P[order[0]].copy()
        for j in range(1, len(order)):
            t = float(w[j] / (w[:j + 1].sum()))
            cur = self._slerp(cur, self.P[order[j]], t)
        retrieval = cur

        # generative end: GMM sample, nudged toward the retrieved region so the
        # description still steers it
        cand = self.gmm.sample(64)[0]
        d = cand - retrieval
        pick = cand[np.argsort((d * d).sum(1))[:8]]
        generative = pick[rng.integers(len(pick))]

        P_out = (1 - novelty) * retrieval + novelty * generative

        W = self.space.from_pca(
            np.pad(P_out[None, :], ((0, 0), (0, self.space.components.shape[0] - self.k))))
        vec = self.space.decode(W, project_to_shell=True)[0]
        return MintResult(vector=vec, working=W[0], novelty=novelty,
                          anchors=[int(i) for i in order],
                          anchor_similarity=anchor_sim,
                          strategy=f"retrieval({1-novelty:.2f})+gmm({novelty:.2f})")

    def mint_many(self, description: str, n: int = 5, **kw) -> list[MintResult]:
        """
        RESEARCH/11: every shipping commercial voice-design API returns
        SEVERAL candidates and lets a human pick. That is the product answer
        to the one-to-many problem, and it is worth copying.
        """
        return [self.mint(description, seed=i, **kw) for i in range(n)]

    # ------------------------------------------------------------- persist
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, T=self.T, P=self.P,
                            captions=np.array(self.captions, dtype=object),
                            k=self.k, pca_dims=self.pca_dims,
                            gmm_components=self.gmm_components,
                            text_model=self.text.model_id)
