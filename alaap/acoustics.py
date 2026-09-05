"""
Measure-first acoustic attributes — eval axis 1, and the annotation pipeline.

RESEARCH/05 corrected a load-bearing assumption here: Data-Speech computes
NINE columns from five tools, and SIX of the attributes the scope document
listed are not among them. jitter, shimmer, HNR, spectral tilt, formants and
vocal-tract length all have to be built. This module builds them.

The point of measuring first and captioning second is REVERSIBILITY:

    audio -> measure -> bin -> LLM writes prose from the bins -> caption
                          ^                                       |
                          +---------- verify by re-measuring ------+

A caption produced this way can be checked against a generated voice by
re-measuring that voice. A caption produced by asking an audio-LM to
describe a voice cannot -- which is exactly why VoicePersona v1's captions
are unusable as ground truth.

Deliberate dependency choices:
  * librosa.pyin for F0, not praat-parselmouth. parselmouth is GPLv3
    (RESEARCH/05 flags it dev-tool-only), and pyin needs no extra install.
  * Energy+ZCR VAD rather than silero, to avoid another torch model on a
    6GB card that is already busy.
  * Speaking rate needs NO forced aligner: in TTV you already know the text
    (RESEARCH/06 correction). Phones come from g2p when available, with a
    syllable-estimate fallback.

BINNING: bins are calibrated against a CORPUS DISTRIBUTION (percentiles),
never absolute thresholds. Data-Speech uses equal-width bins; percentile
bins are used here instead because they guarantee balanced occupancy, which
matters when the corpus is demographically skewed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

SR = 24000

# ---------------------------------------------------------------- primitives


def _frame(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    idx = np.arange(0, len(x) - n + 1, hop)
    return np.stack([x[i:i + n] for i in idx]) if len(idx) else np.zeros((1, n))


def voiced_mask(wav: np.ndarray, sr: int = SR, frame_ms: float = 25.0,
                hop_ms: float = 10.0, energy_pct: float = 40.0) -> np.ndarray:
    """Cheap energy+ZCR VAD. Returns a per-frame boolean."""
    n, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    F = _frame(wav, n, hop)
    e = (F ** 2).mean(1)
    zcr = (np.diff(np.sign(F), axis=1) != 0).mean(1)
    thr = np.percentile(e, energy_pct)
    return (e > max(thr, 1e-8)) & (zcr < 0.25)


def f0_track(wav: np.ndarray, sr: int = SR, fmin: float = 60.0,
             fmax: float = 500.0) -> np.ndarray:
    """F0 in Hz over voiced frames only (NaN elsewhere stripped)."""
    import librosa
    f0, vflag, _ = librosa.pyin(wav.astype(np.float32), sr=sr, fmin=fmin,
                                fmax=fmax, frame_length=2048)
    f0 = f0[np.isfinite(f0)]
    return f0 if len(f0) else np.array([0.0])


def spectral_tilt(wav: np.ndarray, sr: int = SR) -> float:
    """
    dB/octave slope of the long-term average spectrum. More negative = darker,
    breathier. Not in Data-Speech; built here.
    """
    import librosa
    S = np.abs(librosa.stft(wav.astype(np.float32), n_fft=2048, hop_length=512))
    m = S.mean(1) + 1e-10
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    ok = (f > 80) & (f < 8000)
    x, y = np.log2(f[ok]), 20 * np.log10(m[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def hnr(wav: np.ndarray, sr: int = SR) -> float:
    """
    Harmonics-to-noise ratio, dB, via frame autocorrelation peak.

    CAVEAT (RESEARCH/06): voice-quality measures like this are validated on
    SUSTAINED VOWELS FROM REAL SPEAKERS. Whether they transfer to short
    synthetic utterances is UNESTABLISHED. Treat as indicative until
    validated against a held-out real set.
    """
    n, hop = int(sr * 0.04), int(sr * 0.02)
    F = _frame(wav, n, hop)
    vals = []
    for fr in F:
        fr = fr - fr.mean()
        if (fr ** 2).sum() < 1e-9:
            continue
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        ac = ac / (ac[0] + 1e-12)
        lo, hi = int(sr / 500), min(int(sr / 60), len(ac) - 1)
        if hi <= lo:
            continue
        pk = ac[lo:hi].max()
        pk = min(max(pk, 1e-6), 0.999999)
        vals.append(10 * np.log10(pk / (1 - pk)))
    return float(np.median(vals)) if vals else 0.0


def jitter_shimmer(wav: np.ndarray, sr: int = SR) -> tuple[float, float]:
    """
    Period-to-period F0 instability (jitter) and amplitude instability
    (shimmer), both as relative fractions. Same sustained-vowel caveat as hnr.
    """
    f0 = f0_track(wav, sr)
    if len(f0) < 3:
        return 0.0, 0.0
    T = 1.0 / np.maximum(f0, 1e-6)
    jit = float(np.mean(np.abs(np.diff(T))) / max(np.mean(T), 1e-12))
    n, hop = int(sr * 0.03), int(sr * 0.015)
    A = np.sqrt((_frame(wav, n, hop) ** 2).mean(1)) + 1e-12
    shim = float(np.mean(np.abs(np.diff(A))) / max(np.mean(A), 1e-12))
    return jit, shim


def snr_estimate(wav: np.ndarray, sr: int = SR) -> float:
    """Crude speech/silence energy ratio in dB. Not Brouhaha, but free."""
    n, hop = int(sr * 0.025), int(sr * 0.010)
    e = (_frame(wav, n, hop) ** 2).mean(1) + 1e-12
    sp, no = np.percentile(e, 85), np.percentile(e, 10)
    return float(10 * np.log10(sp / max(no, 1e-12)))


def speaking_rate(wav: np.ndarray, text: str, sr: int = SR) -> float:
    """
    Phones per second of VOICED audio (silence excluded).

    NB: typical English is ~10-15 phones/s over TOTAL duration. Over voiced
    time only it reads ~18-22, because voiced_frac is typically ~0.6.
    Measured on LibriTTS-R: mean 20.7 phones/s voiced, i.e. ~12 overall.

    No forced aligner needed -- in TTV the text is known (RESEARCH/06).
    Uses g2p-en if installed, else a vowel-group syllable estimate x 2.5.
    """
    voiced = voiced_mask(wav, sr)
    voiced_s = float(voiced.sum() * 0.010)
    if voiced_s <= 0.05:
        return 0.0
    n_ph = None
    try:
        from g2p_en import G2p
        if not hasattr(speaking_rate, "_g2p"):
            speaking_rate._g2p = G2p()
        n_ph = len([p for p in speaking_rate._g2p(text) if p.strip() and p != " "])
    except Exception:
        pass
    if not n_ph:
        import re
        n_ph = int(2.5 * max(len(re.findall(r"[aeiouyAEIOUY]+", text)), 1))
    return float(n_ph / voiced_s)


# ---------------------------------------------------------------- attributes
@dataclass
class Attributes:
    f0_mean: float
    f0_std: float          # Hz -- kept for transparency, NOT binned (see f0_cv)
    f0_cv: float           # f0_std / f0_mean -- scale-free, this is what is binned
    f0_range: float
    speaking_rate: float
    hnr_db: float
    jitter: float
    shimmer: float
    spectral_tilt: float
    snr_db: float
    duration_s: float
    voiced_frac: float

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Attributes":
        """
        Tolerant loader for cached attribute dicts.

        f0_cv was added after the first corpora were measured (it replaced raw
        f0_std as the binned expressiveness axis, because f0_std in Hz
        correlates r=+0.72 with f0_mean). Rather than invalidate every cache
        and re-measure, derive it when absent -- it is exactly std/mean.
        """
        d = dict(d)
        if "f0_cv" not in d:
            d["f0_cv"] = d.get("f0_std", 0.0) / max(d.get("f0_mean", 1e-6), 1e-6)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def measure(wav: np.ndarray, text: str = "", sr: int = SR) -> Attributes:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    pk = float(np.abs(wav).max())
    if pk > 1.0:
        wav = wav / pk
    f0 = f0_track(wav, sr)
    jit, shim = jitter_shimmer(wav, sr)
    vm = voiced_mask(wav, sr)
    f0m, f0s = float(np.mean(f0)), float(np.std(f0))
    return Attributes(
        f0_mean=f0m, f0_std=f0s,
        f0_cv=float(f0s / max(f0m, 1e-6)),
        f0_range=float(np.percentile(f0, 95) - np.percentile(f0, 5)),
        speaking_rate=speaking_rate(wav, text, sr) if text else 0.0,
        hnr_db=hnr(wav, sr), jitter=jit, shimmer=shim,
        spectral_tilt=spectral_tilt(wav, sr), snr_db=snr_estimate(wav, sr),
        duration_s=float(len(wav) / sr), voiced_frac=float(vm.mean()))


# -------------------------------------------------------------------- bins
BIN_LABELS = {
    "f0_mean": ["very low-pitched", "low-pitched", "moderately pitched",
                "high-pitched", "very high-pitched"],
    "f0_cv": ["monotone", "slightly varied", "moderately expressive",
              "expressive", "highly animated"],
    "speaking_rate": ["very slow", "slow", "measured", "quick", "rapid"],
    "hnr_db": ["very rough", "rough", "slightly rough", "clear", "very clear"],
    "spectral_tilt": ["very dark", "dark", "balanced", "bright", "very bright"],
    "snr_db": ["very noisy", "noisy", "fair", "clean", "very clean"],
    "shimmer": ["very steady", "steady", "slightly unsteady", "unsteady",
                "very unsteady"],
    "jitter": ["very stable", "stable", "slightly unstable", "unstable",
               "very unstable"],
}


class Binner:
    """
    Percentile-calibrated bins, with two DECORRELATIONS applied first.

    Measured on GLOBE_V2 (2,500 clips), the naive attributes are badly
    entangled with pitch:

        f0_mean vs f0_std (Hz)   r = +0.723
        f0_mean vs hnr_db        r = +0.621   <- 39.9% of HNR variance is pitch

    The second is a MEASUREMENT ARTEFACT, not a property of voices: a high-F0
    signal has a stronger autocorrelation peak at its pitch period, so the HNR
    estimator reads it as "clearer". Left uncorrected, "gravelly" becomes
    entangled with "low-pitched", and a description like "a high, harsh voice"
    is far harder to satisfy than it should be -- which is exactly what S2
    observed (that description scored 0.00 in every run).

    Two fixes, both measured:

        f0_std -> f0_cv (std/mean)              r = +0.723 -> +0.197
        hnr_db -> residual after log-F0 regression  r = +0.621 -> +0.055

    The HNR regression is fit on the CORPUS at Binner.fit time and stored, so
    the correction travels with the binner rather than being recomputed per
    call.

    Bins are PERCENTILE, not equal-width as Data-Speech uses, because
    percentiles guarantee balanced occupancy even on a skewed corpus -- and
    RESEARCH/05 found VoicePersona's skew was a PROMPT artefact that an
    equal-width binner would have propagated.
    """

    def __init__(self, edges=None, n_bins: int = 5, hnr_fit=None):
        self.edges = edges or {}
        self.n_bins = n_bins
        # (slope, intercept) of hnr_db ~ a*log(f0_mean) + b, or None
        self.hnr_fit = hnr_fit

    def _hnr_corrected(self, a: "Attributes") -> float:
        """HNR with the pitch confound regressed out (see class docstring)."""
        if not self.hnr_fit:
            return a.hnr_db
        sl, ic = self.hnr_fit
        return float(a.hnr_db - (sl * np.log(max(a.f0_mean, 1e-6)) + ic))

    def _value(self, a: "Attributes", field: str) -> float:
        return self._hnr_corrected(a) if field == "hnr_db" else getattr(a, field)

    @classmethod
    def fit(cls, attrs: list[Attributes], n_bins: int = 5) -> "Binner":
        qs = np.linspace(0, 100, n_bins + 1)[1:-1]
        # fit the HNR pitch correction on the corpus first
        f0 = np.array([a.f0_mean for a in attrs], dtype=np.float64)
        hn = np.array([a.hnr_db for a in attrs], dtype=np.float64)
        ok = np.isfinite(f0) & np.isfinite(hn) & (f0 > 50)
        hnr_fit = None
        if ok.sum() >= 20:
            sl, ic = np.polyfit(np.log(f0[ok]), hn[ok], 1)
            hnr_fit = (float(sl), float(ic))
        tmp = cls({}, n_bins, hnr_fit)

        edges = {}
        for f in BIN_LABELS:
            v = np.array([tmp._value(a, f) for a in attrs], dtype=np.float64)
            v = v[np.isfinite(v)]
            if len(v) >= n_bins:
                edges[f] = [float(x) for x in np.percentile(v, qs)]
        return cls(edges, n_bins, hnr_fit)

    def bin_one(self, a: Attributes) -> dict[str, str]:
        out = {}
        for f, labels in BIN_LABELS.items():
            if f not in self.edges:
                continue
            i = int(np.searchsorted(self.edges[f], self._value(a, f)))
            # tilt and hnr: more negative / lower = "darker" / "rougher"
            out[f] = labels[min(i, len(labels) - 1)]
        return out

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        json.dump({"edges": self.edges, "n_bins": self.n_bins,
                   "hnr_fit": self.hnr_fit, "version": 2}, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        if d.get("version", 1) < 2:
            raise ValueError(
                f"{path} is a v1 binner: it bins raw f0_std and uncorrected "
                f"hnr_db, both of which are entangled with pitch "
                f"(r=+0.723 / +0.621). Re-fit it.")
        return cls(d["edges"], d["n_bins"], tuple(d["hnr_fit"]) if d.get("hnr_fit") else None)


def adherence_error(target_bins: dict[str, str], rendered: Attributes,
                    binner: Binner) -> dict:
    """
    Eval axis 1, objective half: re-measure a rendered voice and compare its
    bins to the ones its description promised.

    This is the whole point of measuring before captioning -- it makes
    description adherence checkable without an LLM judge.
    """
    got = binner.bin_one(rendered)
    per_attr, hits = {}, 0
    for f, want in target_bins.items():
        if f not in got:
            continue
        labels = BIN_LABELS[f]
        d = abs(labels.index(got[f]) - labels.index(want))
        per_attr[f] = {"want": want, "got": got[f], "bin_distance": d}
        hits += (d == 0)
    n = max(len(per_attr), 1)
    return {"per_attribute": per_attr, "exact_match_rate": hits / n,
            "mean_bin_distance": float(np.mean([v["bin_distance"]
                                                for v in per_attr.values()]) if per_attr else 0.0)}
