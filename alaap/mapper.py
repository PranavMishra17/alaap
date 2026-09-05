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
                 pca_dims: int = 50, gmm_components: int = 5,
                 retrieval: str = "text"):
        """
        `retrieval` selects how a description is matched against the anchors:

          "text"    cosine over MiniLM sentence embeddings. The default, and
                    the only one with a rendering history behind it.
          "hybrid"  parse the description into target bins, score the parsed
                    axes by WEIGHTED bin distance, and fall back to the text
                    cosine in proportion to how little parsed.

        E15 measured why "hybrid" exists. Scored against true held-out voices,
        the text path is barely better than chance (rank 129.7 of 350 anchors,
        where chance is 175); weighted bin retrieval reaches 96.3 and improves
        cos-to-true-voice by 66%. E15d replicated it on a second corpus and a
        second encoder (+55%).

        It is NOT the default because all of those are retrieval quality in
        embedding space. Nothing has been rendered through this path, so its
        drift, consistency and adherence are unmeasured -- and E11 already
        caught one geometrically-better setting that rendered worse. Flip the
        default after a rendering arm, not before.
        """
        if retrieval not in ("text", "hybrid"):
            raise ValueError(
                f"retrieval must be 'text' or 'hybrid', got {retrieval!r}")
        self.space = space
        self.text = text_encoder
        self.pca_dims = pca_dims
        self.gmm_components = gmm_components
        self.retrieval = retrieval
        self.T: Optional[np.ndarray] = None      # (N, dt) caption embeddings
        self.P: Optional[np.ndarray] = None      # (N, k)  speaker PCA coords
        self.captions: list[str] = []
        self.gmm = None
        self.anchor_bins: Optional[np.ndarray] = None   # (N, A) bin indices
        self.axis_weights: Optional[np.ndarray] = None  # (A,) measured
        self.axes: list[str] = []

    # ---------------------------------------------------------------- fit
    def fit(self, captions: list[str], vectors: np.ndarray,
            anchor_bins: Optional[list[dict]] = None) -> "RetrievalMapper":
        """
        `anchor_bins` is each anchor's bin dict, i.e. what Binner.bin_one
        returns. Required for retrieval="hybrid", ignored otherwise.
        """
        assert len(captions) == len(vectors), "captions and vectors must align"
        if self.retrieval == "hybrid" and anchor_bins is None:
            raise ValueError(
                "retrieval='hybrid' needs anchor_bins -- pass "
                "[binner.bin_one(a) for a in attrs] alongside the captions. "
                "Without them there is nothing to measure a bin distance against.")
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
        # Cholesky factors, so mint() can draw from the GMM with ITS OWN rng.
        # sklearn's gmm.sample() calls check_random_state(self.random_state)
        # on every call, and random_state is a fixed int -- so it returns the
        # SAME points every time. See _sample_gmm.
        self._chol = [np.linalg.cholesky(
            c + 1e-9 * np.eye(c.shape[0])) for c in self.gmm.covariances_]
        # The typicality band of REAL speakers under this prior. E12 found that
        # both ends of the novelty range leave it: novelty 0 interpolates
        # BETWEEN anchors and lands in the gaps, novelty 1 lands out in the
        # tails, and both score 0-4% on a likelihood comparison against real
        # speakers. Distance metrics cannot tell those apart from novelty,
        # because a point far from all the data is also far from every other
        # point. Storing the band lets mint() aim at it directly.
        _ll = self.gmm.score_samples(self.P)
        self._ll_median = float(np.median(_ll))
        if anchor_bins is not None:
            self._fit_bin_index(anchor_bins)
        return self

    def _fit_bin_index(self, anchor_bins: list[dict]) -> None:
        """
        Build the anchor bin matrix and MEASURE each axis's weight.

        The weight is that axis's rank correlation between its own bin distance
        and voice distance -- how much it actually says about identity. E15 and
        E15d measured the same ordering independently on two corpora and two
        encoders: f0_mean 2.71/3.49, vtl_cm 1.44, spectral_tilt 0.66/0.69,
        hnr_db 0.56/0.43, f0_cv 0.33/0.23, speaking_rate 0.31/0.17, against a
        mean of 1.0.

        Pitch and vocal-tract length are anatomy. Rate and expressiveness are
        behaviour a speaker varies at will, and should not identify anyone --
        yet an equal-weight treatment, which is what the text path implicitly
        applies, gives rate the same say as pitch.
        """
        from .acoustics import BIN_LABELS, RECORDING_AXES
        # RECORDING_AXES (snr_db) describe the take, not the person. Leaving
        # them in cost 0.1493 cos-to-true against 0.1587 for the identity-only
        # axis set, and they are wrong in principle regardless of the number.
        axes = [a for a in BIN_LABELS
                if a not in RECORDING_AXES and all(a in b for b in anchor_bins)]
        if not axes:
            return
        idx = {a: {lbl: i for i, lbl in enumerate(BIN_LABELS[a])} for a in axes}
        B = np.array([[idx[a][b[a]] for a in axes] for b in anchor_bins], float)

        rng = np.random.default_rng(0)
        iu, ju = np.triu_indices(len(B), k=1)
        if len(iu) > 40000:
            sel = rng.choice(len(iu), 40000, replace=False)
            iu, ju = iu[sel], ju[sel]
        U = self.P / (np.linalg.norm(self.P, axis=1, keepdims=True) + 1e-12)
        dv = (1.0 - U @ U.T)[iu, ju]

        def _rho(x, y):
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            if rx.std() < 1e-12 or ry.std() < 1e-12:
                return 0.0
            return float(np.corrcoef(rx, ry)[0, 1])

        w = np.array([max(_rho(np.abs(B[iu, k] - B[ju, k]), dv), 0.0)
                      for k in range(len(axes))])
        if w.sum() <= 0:
            w = np.ones(len(axes))
        self.axes = axes
        self.anchor_bins = B
        self.axis_weights = w / w.sum() * len(w)

    def _sample_gmm(self, rng, n: int) -> np.ndarray:
        """
        Draw n points from the fitted mixture using the CALLER's rng.

        Why not `self.gmm.sample(n)`: sklearn re-derives its RNG from the
        fixed `random_state=0` on each call, so every call returns an
        identical batch. mint() then had at most 64 distinct generative
        outcomes available for the whole life of the mapper, and E12 measured
        the consequence -- nearest-neighbour distance 0.000 at novelty 1.0,
        i.e. minted voices with exact duplicates. Sampling here instead makes
        the draw depend on the per-description seed, as was always intended.
        """
        w = self.gmm.weights_
        comps = rng.choice(len(w), size=n, p=w)
        z = rng.standard_normal((n, self.gmm.means_.shape[1]))
        return np.stack([self.gmm.means_[c] + self._chol[c] @ z[i]
                         for i, c in enumerate(comps)])

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
             seed: int | None = None, typicality: bool = False) -> MintResult:
        """
        novelty 0.0  pure retrieval/SLERP -- safest, near-duplicate of catalog
                     (E1: 0.20x natural speaker spacing)
        novelty 1.0  pure GMM sample -- most novel, still in-distribution
                     (E1: 0.89x natural spacing)

        Values above 1.0 are NOT exposed: E1 showed the next step up
        (full-covariance Gaussian) overshoots to 1.29x and lands off-manifold.

        `typicality` selects, among several candidate blends, the one whose
        density under the speaker prior is closest to a real speaker's.

        DEFAULT OFF, and the reason is worth keeping. It was added on the
        strength of an on-manifold likelihood metric that turned out to be
        broken: a full-covariance GMM over 50 dimensions, fitted to ~1,250
        speakers, scores REAL held-out speakers no higher than Gaussian noise
        (both ~0-1%). It measures proximity to its own training points, not
        plausibility. Every number that justified turning this on was
        therefore invalid, and this stays off until a density estimator that
        passes that control says otherwise. See experiments/E12-novelty.
        """
        if self.T is None:
            raise RuntimeError("mapper not fitted")
        novelty = float(np.clip(novelty, 0.0, 1.0))
        rng = np.random.default_rng(seed)

        q = self.text.encode(description)[0]
        sims = self.T @ q

        if self.retrieval == "hybrid" and self.anchor_bins is not None:
            sims = self._hybrid_scores(description, sims)

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
        cand = self._sample_gmm(rng, 64)
        d = cand - retrieval
        pick = cand[np.argsort((d * d).sum(1))[:8]]

        if typicality and novelty > 0.0:
            # Blend against SEVERAL generative draws and keep the blend whose
            # density under the speaker prior best matches a real speaker's.
            # Taking one draw and hoping (the previous behaviour) is what put
            # novelty=1.0 at 4% on-manifold: nothing in the blend was aiming
            # at plausibility, so it drifted into the tails as novelty rose.
            C = (1 - novelty) * retrieval + novelty * pick
            P_out = C[int(np.argmin(np.abs(
                self.gmm.score_samples(C) - self._ll_median)))]
            strategy = f"retrieval({1-novelty:.2f})+gmm({novelty:.2f})+typical"
        else:
            generative = pick[rng.integers(len(pick))]
            P_out = (1 - novelty) * retrieval + novelty * generative
            strategy = f"retrieval({1-novelty:.2f})+gmm({novelty:.2f})"

        W = self.space.from_pca(
            np.pad(P_out[None, :], ((0, 0), (0, self.space.components.shape[0] - self.k))))
        vec = self.space.decode(W, project_to_shell=True)[0]
        return MintResult(vector=vec, working=W[0], novelty=novelty,
                          anchors=[int(i) for i in order],
                          anchor_similarity=anchor_sim,
                          strategy=strategy)

    def _hybrid_scores(self, description: str,
                       text_sims: np.ndarray) -> np.ndarray:
        """
        Blend the text cosine with a weighted bin distance, trusting the bins
        exactly as far as the description actually parsed.

        E15b measured why the blend has to be proportional rather than fixed:
        generated captions yield 5 of 6 axes, realistic user text yields 1.29
        (22%), and character-sheet prose -- "a gravelly old sailor,
        world-weary" -- yields 0.50. A fixed blend would be wrong in both
        directions.

        `alpha` is the parsed fraction. For a description carrying no acoustic
        words at all it is 0, and this returns the text scores untouched, so
        the caller can never be worse off than the text path.
        """
        from .acoustics import BIN_LABELS
        from .captions import target_bins_from_text
        want = target_bins_from_text(description)
        cols = [(i, a) for i, a in enumerate(self.axes) if a in want]
        if not cols:
            return text_sims
        # alpha is the share of identity INFORMATION the description pinned
        # down, not the share of axes it happened to name. Counting axes
        # treats "very deep" (f0_mean, weight 3.28) as worth the same as
        # "clean recording" (snr_db, 0.60), and it divides by every binned
        # axis including the recording-quality ones the description would
        # never mention -- which measured 0.1443 cos-to-true against 0.1587
        # for the weighted version.
        take0 = [i for i, _ in cols]
        alpha = float(self.axis_weights[take0].sum() /
                      max(self.axis_weights.sum(), 1e-12))

        idx = {a: {lbl: i for i, lbl in enumerate(BIN_LABELS[a])} for _, a in cols}
        take = [i for i, _ in cols]
        tgt = np.array([idx[a][want[a]] for _, a in cols], float)
        dist = (np.abs(self.anchor_bins[:, take] - tgt) *
                self.axis_weights[take]).sum(1)

        def _z(v):
            sd = v.std()
            return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)

        return (1.0 - alpha) * _z(text_sims) + alpha * _z(-dist)

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
