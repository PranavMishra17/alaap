"""
Speaker-embedding space geometry.

Everything here is a direct consequence of experiment E3
(see experiments/E3/RESULTS.md), measured on Qwen3-TTS-12Hz-0.6B-Base:

  - 79.5% of every vector is a constant offset shared by ALL speakers.
    Only 20.5% of its length carries speaker identity.
  - Raw cosine between DIFFERENT speakers is 0.9575. Raw cosine cannot
    discriminate speakers. Mean-centring moves that to ~0.003.
  - Not L2-normalised: ||z|| = 10.859 +/- 0.336 (CV 3.1%). Vectors live on
    a shell, so a generated vector must be scaled onto that shell.
  - Per-dim variance ratio 126.6x, so unweighted MSE badly over-weights
    a handful of dimensions.
  - Effective rank 50.7 / 1024. The manifold is ~50-D, not 1024-D.

Hence invariant I1, as amended by E3 and again by E4:

    MEAN-CENTRE FIRST, THEN PER-DIMENSION RESCALE,
    AND FIT THE TRANSFORM IN-DOMAIN.

E4 measured the third clause: on LibriTTS-R, a transform fit cross-domain
(on GLOBE-V2) gave EER 8.64%, while one fit in-domain gave 4.35% -- against
9.57% for raw vectors. A cross-domain transform buys almost nothing. The
tell is C_diff: 0.5545 cross-domain (a large residual similarity floor,
because the fit corpus mean does not cancel the target corpus mean) versus
-0.0017 in-domain.

A SpeakerSpace is therefore a per-DOMAIN artefact, not a per-encoder
constant. Persist it with every identity (invariant I2).

`SpeakerSpace` is the single place that transform lives. Nothing else in
the codebase should touch raw embeddings.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

__all__ = ["SpeakerSpace", "cosine_matrix", "pairwise_cosines"]


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return Xn @ Xn.T


def pairwise_cosines(X: np.ndarray, rng: np.random.Generator | None = None,
                     max_n: int = 600) -> np.ndarray:
    """Upper-triangle cosines, subsampled to keep it O(max_n^2)."""
    if len(X) > max_n:
        rng = rng or np.random.default_rng(0)
        X = X[rng.choice(len(X), max_n, replace=False)]
    C = cosine_matrix(X)
    iu = np.triu_indices(len(X), 1)
    return C[iu]


@dataclass
class SpaceStats:
    dim: int
    n_fit: int
    shell_radius: float          # mean ||z||
    shell_std: float
    norm_cv: float
    is_l2_normalised: bool
    var_ratio: float             # max/min per-dim variance
    effective_rank: float        # participation ratio of the PCA spectrum
    identity_fraction: float     # mean||z-mu|| / mean||z||


class SpeakerSpace:
    """
    Fitted transform between the backend's raw speaker-embedding space and a
    well-conditioned *working space* where the mapper operates.

        raw  --encode-->  working  (mean-centred, per-dim unit variance)
        working --decode-->  raw   (rescaled, mean restored, shell-projected)

    Fit once on a corpus of real embeddings; persist alongside every identity
    (invariant I2 -- an identity vector is meaningless without the transform
    that produced it).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 shell_radius: float, shell_std: float,
                 components: Optional[np.ndarray] = None,
                 explained: Optional[np.ndarray] = None,
                 stats: Optional[SpaceStats] = None,
                 version: str = "1"):
        self.mean = mean.astype(np.float64)
        self.std = np.maximum(std.astype(np.float64), 1e-8)
        self.shell_radius = float(shell_radius)
        self.shell_std = float(shell_std)
        self.components = components      # (k, D) PCA basis in WORKING space
        self.explained = explained        # (k,) explained variance ratio
        self.stats = stats
        self.version = version

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, Z: np.ndarray, n_components: int = 128) -> "SpeakerSpace":
        Z = np.asarray(Z, dtype=np.float64)
        n, d = Z.shape
        mean = Z.mean(0)
        std = Z.std(0)
        norms = np.linalg.norm(Z, axis=1)

        W = (Z - mean) / np.maximum(std, 1e-8)
        Wc = W - W.mean(0)
        k = int(min(n_components, min(Wc.shape) - 1))
        U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
        ev = (S ** 2) / max(n - 1, 1)
        ev_ratio = ev / ev.sum()

        # participation ratio on the FULL spectrum = effective dimensionality
        p = ev_ratio[ev_ratio > 0]
        eff_rank = float(np.exp(-(p * np.log(p)).sum()))

        var = std ** 2
        stats = SpaceStats(
            dim=int(d), n_fit=int(n),
            shell_radius=float(norms.mean()), shell_std=float(norms.std()),
            norm_cv=float(norms.std() / norms.mean()),
            is_l2_normalised=bool(norms.std() / norms.mean() < 0.01),
            var_ratio=float(var.max() / max(var.min(), 1e-12)),
            effective_rank=eff_rank,
            identity_fraction=float(
                np.linalg.norm(Z - mean, axis=1).mean() / norms.mean()),
        )
        return cls(mean, std, norms.mean(), norms.std(),
                   components=Vt[:k], explained=ev_ratio[:k], stats=stats)

    # ------------------------------------------------------------- transform
    def encode(self, Z: np.ndarray) -> np.ndarray:
        """raw -> working space (mean-centred, per-dim unit variance)."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        return (Z - self.mean) / self.std

    def decode(self, W: np.ndarray, project_to_shell: bool = True) -> np.ndarray:
        """
        working -> raw. Optionally project onto the ||z|| shell the encoder's
        real outputs occupy, which is what keeps a *generated* vector in
        distribution (E3: ||z|| = 10.859 +/- 0.336, CV 3.1%).
        """
        W = np.atleast_2d(np.asarray(W, dtype=np.float64))
        Z = W * self.std + self.mean
        if project_to_shell:
            n = np.linalg.norm(Z, axis=1, keepdims=True)
            Z = Z / np.maximum(n, 1e-12) * self.shell_radius
        return Z

    def to_pca(self, W: np.ndarray) -> np.ndarray:
        """working -> PCA coords. The generative prior should live here."""
        if self.components is None:
            raise RuntimeError("no PCA basis fitted")
        return np.atleast_2d(W) @ self.components.T

    def from_pca(self, P: np.ndarray) -> np.ndarray:
        if self.components is None:
            raise RuntimeError("no PCA basis fitted")
        return np.atleast_2d(P) @ self.components

    def n_components_for(self, frac: float) -> int:
        """How many PCA dims carry `frac` of the variance."""
        if self.explained is None:
            raise RuntimeError("no PCA basis fitted")
        return int(np.searchsorted(np.cumsum(self.explained), frac) + 1)

    # ------------------------------------------------------------ uniqueness
    def uniqueness_distance(self, z_new: np.ndarray, Z_existing: np.ndarray) -> float:
        """
        Cosine distance from a candidate identity to its nearest existing one,
        measured in WORKING space.

        E3: on RAW vectors the whole 1,025-speaker population has a maximum
        nearest-neighbour distance of 0.029, so the VoicePrivacy-B3 threshold
        of 0.3 is unreachable there. In working space the mean NN distance is
        0.550, so the check is meaningful. Never run this on raw vectors.
        """
        if len(Z_existing) == 0:
            return 1.0
        a = self.encode(z_new)
        B = self.encode(Z_existing)
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
        return float(1.0 - (B @ a.T).max())

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path, mean=self.mean, std=self.std,
            components=self.components if self.components is not None else np.zeros((0, 0)),
            explained=self.explained if self.explained is not None else np.zeros((0,)),
            shell_radius=self.shell_radius, shell_std=self.shell_std,
            version=self.version,
        )
        if self.stats is not None:
            with open(os.path.splitext(path)[0] + ".stats.json", "w") as f:
                json.dump(asdict(self.stats), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SpeakerSpace":
        d = np.load(path, allow_pickle=False)
        comp = d["components"]
        exp = d["explained"]
        return cls(d["mean"], d["std"], float(d["shell_radius"]), float(d["shell_std"]),
                   components=comp if comp.size else None,
                   explained=exp if exp.size else None,
                   version=str(d["version"]))

    def __repr__(self) -> str:
        s = self.stats
        if s is None:
            return f"SpeakerSpace(dim={len(self.mean)})"
        return (f"SpeakerSpace(dim={s.dim}, fit_on={s.n_fit}, "
                f"shell={s.shell_radius:.3f}+/-{s.shell_std:.3f}, "
                f"eff_rank={s.effective_rank:.1f}, "
                f"identity_frac={s.identity_fraction:.3f})")
