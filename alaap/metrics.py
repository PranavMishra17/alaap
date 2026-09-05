"""
Measurement primitives for the eval harness.

Axis 2 (identity consistency) and axis 3 (diversity / anti-mode-collapse)
from RESEARCH/06.

Two hard rules baked in here, both from the research:

  1. There is NO defensible universal cosine threshold. Published values are
     encoder-specific and non-comparable (SpeechBrain default 0.25, real
     same-speaker SIM-o 0.69-0.76, DIFFERENT real speakers 0.67 on another
     encoder). You must calibrate C_same / C_diff in-run. That is what
     `verification_stats` is for.

  2. Never score identity with the same encoder used for conditioning --
     that is marking your own homework. `verification_stats` is
     encoder-agnostic; pass it embeddings from an INDEPENDENT model when
     scoring a generated voice.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

__all__ = ["verification_stats", "VerificationStats", "vendi_score",
           "nn_distances", "cluster_separability"]


@dataclass
class VerificationStats:
    n_same: int
    n_diff: int
    same_mean: float
    same_std: float
    diff_mean: float
    diff_std: float
    separation: float        # same_mean - diff_mean
    d_prime: float           # separation / pooled std
    eer: float               # equal error rate from the cosine score
    eer_threshold: float     # cosine at EER -- the defensible operating point
    same_p1: float
    diff_p99: float
    overlap: float           # P(diff > same_p1): how often an impostor beats
                             # the 1st-percentile genuine score

    def to_dict(self):
        return asdict(self)


def _cos(X: np.ndarray) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return Xn @ Xn.T


def verification_stats(Z: np.ndarray, speaker_ids: list[str],
                       max_pairs: int = 400_000,
                       seed: int = 0) -> VerificationStats:
    """
    Calibrate C_same and C_diff on real data.

    Z            (N, D) embeddings
    speaker_ids  length-N labels

    Returns the genuine/impostor cosine distributions plus an EER and the
    cosine threshold at EER, which is the only defensible operating point
    (see rule 1 above).
    """
    Z = np.asarray(Z, dtype=np.float64)
    ids = np.asarray(speaker_ids)
    C = _cos(Z)
    n = len(Z)
    iu = np.triu_indices(n, 1)
    same_mask = ids[iu[0]] == ids[iu[1]]
    scores = C[iu]

    same = scores[same_mask]
    diff = scores[~same_mask]
    if len(same) == 0:
        raise ValueError("no same-speaker pairs -- need per_speaker >= 2")

    rng = np.random.default_rng(seed)
    if len(diff) > max_pairs:
        diff = diff[rng.choice(len(diff), max_pairs, replace=False)]

    # EER by sweeping the threshold over the pooled score range
    lo, hi = float(min(same.min(), diff.min())), float(max(same.max(), diff.max()))
    ths = np.linspace(lo, hi, 2000)
    # FRR: genuine below threshold. FAR: impostor at/above threshold.
    frr = (same[None, :] < ths[:, None]).mean(1)
    far = (diff[None, :] >= ths[:, None]).mean(1)
    i = int(np.argmin(np.abs(frr - far)))
    eer = float((frr[i] + far[i]) / 2)

    pooled = np.sqrt((same.var() + diff.var()) / 2)
    same_p1 = float(np.percentile(same, 1))

    return VerificationStats(
        n_same=int(len(same)), n_diff=int(len(diff)),
        same_mean=float(same.mean()), same_std=float(same.std()),
        diff_mean=float(diff.mean()), diff_std=float(diff.std()),
        separation=float(same.mean() - diff.mean()),
        d_prime=float((same.mean() - diff.mean()) / max(pooled, 1e-12)),
        eer=eer, eer_threshold=float(ths[i]),
        same_p1=same_p1, diff_p99=float(np.percentile(diff, 99)),
        overlap=float((diff > same_p1).mean()),
    )


def vendi_score(Z: np.ndarray, q: float = 1.0, normalise: bool = True) -> float:
    """
    Vendi score on the cosine kernel -- the number of "effectively distinct"
    items in a set. RESEARCH/06 sets the anti-mode-collapse target at
    normalised VS >= 0.35 (i.e. >= 7 effective voices out of 20 samples),
    alarm below 0.20, collapse below 0.10.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n = len(Z)
    if n < 2:
        return 0.0 if normalise else 1.0
    K = _cos(Z)
    K = (K + K.T) / 2.0
    K = K / n
    w = np.linalg.eigvalsh(K)
    w = np.clip(w, 0, None)
    w = w / max(w.sum(), 1e-12)
    p = w[w > 1e-12]
    if abs(q - 1.0) < 1e-9:
        vs = float(np.exp(-(p * np.log(p)).sum()))
    else:
        vs = float((p ** q).sum() ** (1.0 / (1.0 - q)))
    return vs / n if normalise else vs


def nn_distances(Z: np.ndarray) -> np.ndarray:
    """Cosine distance from each row to its nearest OTHER row."""
    C = _cos(np.asarray(Z, dtype=np.float64))
    np.fill_diagonal(C, -np.inf)
    return 1.0 - C.max(1)


def cluster_separability(Z: np.ndarray, labels: list[str]) -> float:
    """
    Silhouette on cosine distance. RESEARCH/06 wants >= 0.15 for
    "clusters from different descriptions are separable".
    """
    Z = np.asarray(Z, dtype=np.float64)
    lab = np.asarray(labels)
    D = 1.0 - _cos(Z)
    np.fill_diagonal(D, 0.0)
    sil = []
    for i in range(len(Z)):
        same = lab == lab[i]
        same[i] = False
        if same.sum() == 0:
            continue
        a = D[i, same].mean()
        b = min(D[i, lab == u].mean() for u in np.unique(lab) if u != lab[i])
        sil.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(sil)) if sil else 0.0
