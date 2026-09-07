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
           "nn_distances", "cluster_separability", "normalised_similarity",
           "CALIBRATION"]

# Measured on real human speech (LibriTTS-R, 400 clips / 21 speakers, E0).
# A raw similarity number is MEANINGLESS without these -- see E0 RESULTS
# section 2. WavLM x-vectors are so concentrated that two COMPLETELY
# DIFFERENT speakers score 0.6632, so "SECS >= 0.88" occupies only the top
# 0.29 of the scale, not the top 0.12 it appears to.
CALIBRATION = {
    #  encoder : (C_same, C_diff, EER)
    "wavlm": (0.9538, 0.6632, 0.0534),
    "ecapa": (0.6988, 0.2011, 0.0258),
}


def normalised_similarity(secs: float, encoder: str) -> float:
    """
    Put a raw cosine on a scale a human can read:

        1.0  indistinguishable from the target speaker
        0.0  indistinguishable from a DIFFERENT speaker

    E0 measured the same audio at WavLM 0.921 ("preserved") and ECAPA 0.560
    ("badly degraded"). Neither is wrong; they have different dynamic ranges.
    Reporting the forgiving one alone overstates the result -- and the
    forgiving one (WavLM) is also the WEAKER discriminator (EER 5.34% vs
    2.58%). Always report the floor alongside the number.
    """
    if encoder not in CALIBRATION:
        raise KeyError(f"no calibration for {encoder!r}; measure C_same/C_diff "
                       f"on real speech first (RESEARCH/06)")
    same, diff, _ = CALIBRATION[encoder]
    return float((secs - diff) / (same - diff))


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


# ------------------------------------------------------- typicality / manifold
def knn_radius(X: np.ndarray, R: np.ndarray, k: int = 5,
               self_exclude: bool = False) -> np.ndarray:
    """Cosine distance from each row of X to its k-th nearest row of R."""
    A = np.asarray(X, dtype=np.float64)
    B = np.asarray(R, dtype=np.float64)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - A @ B.T
    if self_exclude:
        np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, k - 1]


def isolation_pct(X: np.ndarray, R: np.ndarray, k: int = 5) -> float:
    """
    How isolated X is relative to how isolated REAL speakers are.

    Returns the percentile of X's median k-NN radius within the distribution
    of R's own k-NN radii. **50 = as typical as a median real speaker.
    100 = further from the data than any real speaker is.**

    WHY NOT A LIKELIHOOD. The obvious version of this test -- fit a
    full-covariance GMM to the reference and compare log-likelihoods -- does
    not work in this space and fails silently. Measured on GLOBE_V2: a
    24-component GMM fitted to 1,250 speakers in 50 PCA dimensions scores
    REAL held-out speakers at 0-1%, indistinguishable from Gaussian noise.
    With a few thousand points in 50 dimensions it is measuring proximity to
    its own training set, not plausibility, so anything evaluated on the SAME
    speakers the mapper was built from scores well for the wrong reason.

    A k-NN radius needs no density model, only a metric. Validated against the
    control that broke the likelihood version:

        held-out REAL speakers            54%      (must be near 50)
        gaussian noise                   100%      (must be near 100)
        dimension-shuffled speakers      100%      (must be near 100)

    The reference R must be speakers the thing under test was NOT built from.
    """
    r_self = knn_radius(R, R, k=k, self_exclude=True)
    return float((r_self < np.median(knn_radius(X, R, k=k))).mean() * 100.0)


# ------------------------------------------------------------ transcription
def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, iterative, O(min(|a|,|b|)) memory."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalise_transcript(s: str) -> str:
    """
    Strip what an ASR will never emit: the backend's own control tags and
    punctuation. Includes the Devanagari danda (U+0964) and double danda.
    """
    import re
    s = re.sub(r"<[a-z_]+>", " ", s)
    s = re.sub(r"[।॥.,!?;:'\"\-—‘’“”()]", " ", s)
    return " ".join(s.split())


def cer(reference: str, hypothesis: str) -> float:
    """
    Character error rate, after normalisation.

    CER rather than WER because Indic scripts are abugidas: word boundaries
    are less reliable than characters, and Devanagari/Tamil segmentation would
    add its own error term on top of the ASR's. Returns NaN for an empty
    reference rather than dividing by zero.
    """
    r, h = normalise_transcript(reference), normalise_transcript(hypothesis)
    if not r:
        return float("nan")
    return edit_distance(r, h) / len(r)


# ------------------------------------------------------------- naturalness
NATURALNESS_REAL_TYPICAL = 50.0     # held-out real speech, by construction
NATURALNESS_FLOOR = 75.0            # above this, flagged as synthetic-sounding


def naturalness_isolation(X: np.ndarray, reference: np.ndarray) -> float:
    """
    How far a clip's self-supervised features sit from real speech, 0-100.

    Thin wrapper over `isolation_pct` that exists to be NAMED, because the
    project had no naturalness measure at all until S20 and the absence was
    not visible -- drift, consistency, CER and uniqueness are every one of them
    identity or intelligibility gates, and all of them pass on audio a listener
    calls synthetic (S19).

    DELIBERATELY NOT A MOS PREDICTOR. RESEARCH/06 5.2: humans correlate with
    mean F0 at r = -0.059; DNSMOS at -0.788 and UTMOSv2 at -0.722. A MOS gate
    would reject high-pitched voices for a reason humans do not share, fighting
    the diversity axis this project exists to widen.

    `reference` must be features of REAL speech from the same domain, and
    `X` the same features for the clips being scored. Use **WavLM base+ layer
    5**, mean-pooled -- ECAPA is wrong here because it is trained to be
    INVARIANT to channel and quality.

    WHY LAYER 5 AND NOT ANOTHER. S20b swept all twelve at n=150 per side. Early
    layers encode pitch and are the DNSMOS trap in disguise: layer 0 separates
    the codec significantly (t = 2.04) at r(f0) = -0.578, which is DNSMOS
    territory and was rejected for it. Late layers see nothing. Layers 4 and 5
    both work; layer 5 is chosen because its pitch correlation is -0.054
    against humans' -0.059, where layer 4 sits at -0.211.

    VALIDATED (S20 at layer 5, 140 IndicVoices-R clips, half as reference):

        held-out real          51.6      typical, as isolation_pct requires
        S13 tagged renders     96.4
        S14 retimed            97.1
        S6 minted renders      97.3
        S7 catalog renders     97.5
        correlation with f0    r = -0.068      humans -0.059, DNSMOS -0.788
        real vs real           d = +0.17, a tie

    AND IT DOES SEE THE CODEC, which took three attempts to establish. S20 said
    it was blind, on 3 clips. S20b's first pass said layer 5 saw it, on 40
    clips and an effect-size cut with no test. At 150 per side the roundtrip
    sits +8.1 points above real at t = 2.59 -- so codec degradation IS
    detectable here, and this can track progress on it.
    """
    return isolation_pct(np.atleast_2d(X), np.asarray(reference, dtype=np.float64))
