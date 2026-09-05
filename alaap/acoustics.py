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
    """
    Cheap energy+ZCR VAD. Returns a per-frame boolean.

    NOT a periodicity detector, and the difference matters when this is used
    as evidence rather than as a gate. `energy_pct=40` means at most ~60% of
    frames can ever pass, so `voiced_frac` reads ~0.60 for ordinary speech
    and cannot go higher. It is also BIMODAL in excitation: measured on
    synthetics it gives 0.597 for a harmonic signal, 0.601 for 70% harmonic
    + 30% noise, and 0.000 for 30/70 or pure noise.

    So a strong whisper collapses it to zero, but a mild breathy one reads
    0.60 and looks untouched. `hnr_db` is the graded probe for that question
    -- on the same synthetics it spans +7.8 dB to -10.1 dB. S13 originally
    leaned on voiced_frac to argue `<whisper>` was not whispering; the
    conclusion held, but hnr_db and f0_mean were the load-bearing evidence.
    """
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


# ------------------------------------------------------------------ formants
#
# E14 measured why these are here. Among REAL speakers, the distance between
# two captions in bin space predicts the distance between their voices at only
# rho = 0.260 -- and the mapper already transports 90% of that. The catalog
# saturates because five axes (pitch, tilt, HNR, expressiveness, rate) do not
# determine a speaker, not because the mapper loses information. The only way
# to raise the ceiling is to describe more.
#
# Vocal-tract length is the obvious missing axis: it is one of the strongest
# correlates of perceived speaker identity, it is largely INDEPENDENT of F0
# (which is why it adds rather than duplicates -- pitch is the larynx, VTL is
# the tube above it), and RESEARCH/05 already flagged formants as an attribute
# the project would have to build because Data-Speech does not supply them.
#
# Method: LPC on pre-emphasised voiced frames, roots -> formant frequencies.
# VTL from Fitch's formant dispersion, D = mean spacing between successive
# formants, VTL = c / (2D). This is the uniform-tube model -- an approximation
# a phonetician would call crude, but it is the standard one, it needs no
# corpus-specific calibration, and it is validated below against gender.

_C_CM_PER_S = 35000.0   # speed of sound in warm moist air, cm/s


def formants(wav: np.ndarray, sr: int = SR, n: int = 4,
             max_formant_hz: float = 5000.0, max_frames: int = 200) -> np.ndarray:
    """
    Median of the first `n` formants over voiced frames, in Hz.

    Follows the standard (Praat) recipe rather than running LPC at the native
    rate: **resample to 2 x max_formant first, then use ~2 poles per formant.**
    A first attempt used the full 24 kHz with order 26 and inverted the gender
    effect -- females measured LONGER vocal tracts than males, and F2/F3 came
    out lower for females than for males, both anatomically backwards. At 24 kHz
    an order-26 filter spends most of its poles modelling the 5-12 kHz region,
    which carries no formant information, and for low-F0 voices it locks onto
    individual harmonics instead.

    Returns NaN for any formant that could not be estimated, so a caller sees a
    missing value rather than a fabricated one.
    """
    import librosa
    out = np.full(n, np.nan)
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if len(wav) < sr // 20:
        return out
    vm = voiced_mask(wav, sr)
    idx = np.flatnonzero(vm)
    if len(idx) == 0:
        return out

    target_sr = int(2 * max_formant_hz)
    y = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)         if sr != target_sr else wav.astype(np.float32)
    scale = target_sr / sr

    hop, win = int(target_sr * 0.010), int(target_sr * 0.025)
    if len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).astype(int)]
    order = 2 * n + 2                       # 2 poles per formant, + a pair spare
    hamm = np.hamming(win)
    rows = []
    for i in idx:
        a = int(i * (sr * 0.010) * scale)
        b = a + win
        if b > len(y):
            continue
        fr = y[a:b].astype(np.float64)
        fr = np.append(fr[0], fr[1:] - 0.67 * fr[:-1])   # pre-emphasis
        fr = fr * hamm
        if not np.any(fr):
            continue
        try:
            A = librosa.lpc(fr, order=order)
        except Exception:
            continue
        r = np.roots(A)
        r = r[np.imag(r) > 0.01]
        if len(r) == 0:
            continue
        fr_hz = np.arctan2(np.imag(r), np.real(r)) * (target_sr / (2 * np.pi))
        bw = -0.5 * (target_sr / (2 * np.pi)) * np.log(np.maximum(np.abs(r), 1e-12))
        o = np.argsort(fr_hz)
        fr_hz, bw = fr_hz[o], bw[o]
        keep = (fr_hz > 90) & (fr_hz < max_formant_hz) & (bw < 400)
        fr_hz = fr_hz[keep]
        if len(fr_hz) >= n:
            rows.append(fr_hz[:n])
    if rows:
        out = np.nanmedian(np.vstack(rows), axis=0)
    return out


def vocal_tract_length(fmts: np.ndarray) -> float:
    """
    VTL in cm from formant dispersion (Fitch 1997), c / (2 * mean spacing).

    Typical adult values run ~13-15 cm for females and ~15-18 cm for males,
    which is the check that matters -- see the gender validation in tests.
    Returns NaN if fewer than two formants were estimated.
    """
    f = np.asarray(fmts, dtype=np.float64)
    f = f[np.isfinite(f)]
    if len(f) < 2:
        return float("nan")
    disp = float(np.mean(np.diff(np.sort(f))))
    if disp <= 1e-6:
        return float("nan")
    return float(_C_CM_PER_S / (2.0 * disp))


# ------------------------------------------------------- script-aware phones
#
# Speaking rate needs a phone COUNT. g2p-en covers Latin script only, and the
# vowel-group fallback below it counts `[aeiouy]+`, which finds ZERO matches in
# any Indic script. `max(..., 1)` then turns a whole Hindi sentence into 2.5
# phones, so every Indic clip would read ~0.5 phones/s and the entire
# speaking_rate axis would collapse into one bin -- silently, with no error.
#
# Brahmic scripts are abugidas: a consonant carries an inherent vowel (schwa)
# unless a vowel sign (matra) replaces it or a virama suppresses it. So phones
# are countable from graphemes without a lexicon:
#
#     consonant             -> 1 phone, plus 1 more for its inherent vowel
#                              UNLESS followed by a matra or a virama
#     matra (vowel sign)    -> 1 phone (it IS the vowel that replaced the schwa)
#     independent vowel     -> 1 phone
#     virama / halant       -> 0 (it only cancels the inherent vowel)
#     anusvara / candrabindu/ visarga -> 1 (nasal or /h/)
#
# Unicode allocates the nine Indic blocks in PARALLEL: the same offset inside
# each 128-point block means the same class of character. That is a property of
# the standard (each block was encoded on the ISCII layout), not a coincidence,
# so one offset table serves all nine scripts.
#
# SCHWA DELETION. Indo-Aryan languages delete the word-final inherent schwa:
# Hindi कमल is /kəmal/, three vowels, not /kəmələ/. Without this the count runs
# ~10-15% high on Hindi. Dravidian languages (Tamil, Telugu, Kannada,
# Malayalam) retain final vowels, so the deletion is applied per-script.
# Medial schwa deletion is real but lexically conditioned; it is not modelled
# here, which leaves a known small overcount. See RESEARCH/05.

_BLOCKS = {  # script -> base codepoint of its 128-point Unicode block
    "devanagari": 0x0900, "bengali": 0x0980, "gurmukhi": 0x0A00,
    "gujarati":   0x0A80, "oriya":   0x0B00, "tamil":    0x0B80,
    "telugu":     0x0C00, "kannada": 0x0C80, "malayalam": 0x0D00,
}
# Word-final inherent-vowel deletion. This is per-LANGUAGE, not per-family:
# Hindi/Marathi (Devanagari), Bengali, Punjabi and Gujarati delete it, but
# ODIA is the standard counterexample -- it retains the final vowel, which is
# one of the features that distinguishes it from Bengali. Dravidian scripts
# retain it too. Getting this wrong costs ~1 phone per word, i.e. 10-15% on
# short words, so it is worth being fussy about.
_SCHWA_DELETING = {"devanagari", "bengali", "gurmukhi", "gujarati"}

# offsets within a block (identical across all nine)
_VOWEL_IND = range(0x05, 0x15)   # independent vowels  अ..औ
_CONSONANT = range(0x15, 0x3A)   # consonants          क..ह
_MATRA     = range(0x3E, 0x4D)   # dependent vowels    ा..ौ
_VIRAMA    = 0x4D
_NASAL_H   = (0x01, 0x02, 0x03)  # candrabindu, anusvara, visarga


def detect_script(text: str) -> Optional[str]:
    """Dominant Indic script in `text`, or None if it is not Indic."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        for name, base in _BLOCKS.items():
            if base <= cp < base + 0x80:
                counts[name] = counts.get(name, 0) + 1
                break
    return max(counts, key=counts.get) if counts else None


def count_phones_indic(text: str, script: Optional[str] = None) -> int:
    """
    Estimate phones in Brahmic-script text. Returns 0 for non-Indic input so
    callers can fall back rather than trusting a bogus count.
    """
    script = script or detect_script(text)
    if script is None:
        return 0
    base = _BLOCKS[script]
    off = [ord(c) - base if base <= ord(c) < base + 0x80 else None for c in text]
    n = 0
    for i, o in enumerate(off):
        if o is None:
            continue
        nxt = off[i + 1] if i + 1 < len(off) else None
        if o in _CONSONANT:
            n += 1                                   # the consonant itself
            if nxt in _MATRA or nxt == _VIRAMA:
                pass                                 # vowel supplied, or killed
            else:
                n += 1                               # inherent schwa
        elif o in _VOWEL_IND or o in _MATRA:
            n += 1
        elif o in _NASAL_H:
            n += 1
    if script in _SCHWA_DELETING:
        # one schwa per word ending in a bare consonant
        import re as _re
        for w in _re.findall(r"[^\s\u0964\u0965.,!?;:]+", text):
            if not w:
                continue
            last = ord(w[-1]) - base
            if last in _CONSONANT:
                n -= 1
    return max(n, 0)


def speaking_rate(wav: np.ndarray, text: str, sr: int = SR) -> float:
    """
    Phones per second of VOICED audio (silence excluded).

    NB: typical English is ~10-15 phones/s over TOTAL duration. Over voiced
    time only it reads ~18-22, because voiced_frac is typically ~0.6.
    Measured on LibriTTS-R: mean 20.7 phones/s voiced, i.e. ~12 overall.

    No forced aligner needed -- in TTV the text is known (RESEARCH/06).

    Three counters, tried in order of how much they know about the text:
      1. Brahmic script  -> count_phones_indic (grapheme rules, see above)
      2. Latin + g2p-en  -> a real English lexicon
      3. anything else   -> vowel groups x 2.5
    The Indic branch comes FIRST because g2p-en does not fail loudly on
    Devanagari, it just returns junk.
    """
    voiced = voiced_mask(wav, sr)
    voiced_s = float(voiced.sum() * 0.010)
    if voiced_s <= 0.05:
        return 0.0
    n_ph = count_phones_indic(text)
    if not n_ph:
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
    # Added after E14. Defaults keep every cached corpus loadable -- they were
    # measured before these existed, and Attributes.from_dict drops unknowns.
    f1: float = float("nan")
    f2: float = float("nan")
    f3: float = float("nan")
    formant_dispersion: float = float("nan")
    vtl_cm: float = float("nan")

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


def _formant_fields(wav, sr, on: bool) -> dict:
    """Formant/VTL fields, or NaNs if disabled. LPC is the slowest step in
    measure(), so it is switchable for callers that do not need it."""
    if not on:
        return {}
    f = formants(wav, sr, n=4)
    d = np.diff(np.sort(f[np.isfinite(f)])) if np.isfinite(f).sum() >= 2 else []
    return {"f1": float(f[0]), "f2": float(f[1]), "f3": float(f[2]),
            "formant_dispersion": float(np.mean(d)) if len(d) else float("nan"),
            "vtl_cm": vocal_tract_length(f)}


def measure(wav: np.ndarray, text: str = "", sr: int = SR,
            formants_on: bool = True) -> Attributes:
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
        duration_s=float(len(wav) / sr), voiced_frac=float(vm.mean()),
        **_formant_fields(wav, sr, formants_on))


# -------------------------------------------------------------------- bins
#
# Axes that describe the RECORDING rather than the SPEAKER. They are measured
# and binned like the rest -- a caption may legitimately say "clean" -- but they
# must not enter an IDENTITY distance, because two takes of one person in
# different rooms are still one person. Excluded from the mapper's retrieval
# axes for that reason; E11's catalog sampler excludes them too, so that a
# minted voice is never described as "very noisy" on purpose.
RECORDING_AXES = {"snr_db"}

BIN_LABELS = {
    "f0_mean": ["very low-pitched", "low-pitched", "moderately pitched",
                "high-pitched", "very high-pitched"],
    "f0_cv": ["monotone", "slightly varied", "moderately expressive",
              "expressive", "highly animated"],
    "speaking_rate": ["very slow", "slow", "measured", "quick", "rapid"],
    "hnr_db": ["very rough", "rough", "slightly rough", "clear", "very clear"],
    "spectral_tilt": ["very dark", "dark", "balanced", "bright", "very bright"],
    # VTL: the size of the resonating tube, largely independent of pitch.
    # "small/large" rather than "child/adult" -- the measurement is anatomy,
    # not age, and a caption that claims age from a formant is overreaching.
    "vtl_cm": ["very small-throated", "small-throated", "medium-throated",
               "large-throated", "very large-throated"],
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
