"""
Watermarking — invariant I7, enforced from render #1.

NOT a launch task. EU AI Act Article 50(2) has applied since 2026-08-02, and
Article 2(12) carves Article 50 OUT of the open-source exemption, so
releasing openly does not exempt us (RESEARCH/09, RESEARCH/08).

AudioSeal is the choice, and its licence question is settled: MIT for code
AND weights, verified four ways in RESEARCH/09 (the LICENSE file, the README
changelog -- "We have updated our license to full MIT license (including the
license for the model weights)! Now you can use AudioSeal in commercial
application too!" -- the official docs, and the HF model card). It is also a
registered C2PA soft-binding algorithm, `com.aiwatermark.audioseal.1`.

WHAT THIS DOES NOT DO, stated plainly because overclaiming here is worse
than not shipping it (RESEARCH/09):

  * It is a GOOD-FAITH MARKER, not a robust one. Polarity inversion -- an
    inaudible one-line transform -- drives detection to 0.18/0.00.
  * Opus REMOVES it. Descript codec: 0.00. Real reverb: 0.22.
  * MP3 survives (1.00 @32kbps). OGG/Vorbis survives (0.95), which is the
    Unity/FMOD/Wwise default and therefore the format that matters for us.
  * PRESENCE BIT ONLY. Never put a render ID in the payload: AudioSeal's
    16-bit message averages 0.39 attribution and only 0.69 clean.
  * Short clips are UNVERIFIED. Every published evaluation is 10s / 5s / ~3s,
    the paper's prose contradicts its own Table 6 at 1s, and the best
    analogue (XAttnMark) has attribution collapsing to 81.2% at 1s. Game
    dialogue is 1-3s. That is experiment E5.

Provenance logging (IdentityStore.log_render) is the other half of I7 and is
NOT optional -- it is what makes a render traceable to
(identity, backend_version, description, timestamp).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

WM_SR = 16000   # AudioSeal operates at 16 kHz


@dataclass
class WatermarkResult:
    detected: bool
    probability: float
    n_samples: int
    duration_s: float


class Watermarker:
    """
    Thin wrapper over AudioSeal. Resamples in and out so callers can stay at
    the backend's native rate (24 kHz for Qwen3-TTS).
    """

    def __init__(self, generator: str = "audioseal_wm_16bits",
                 detector: str = "audioseal_detector_16bits",
                 device: str = "cpu"):
        import torch
        # AudioSeal's EnCodec path triggers torch.compile / inductor, which
        # needs MSVC `cl` and is not available on this Windows box. Fall back
        # to eager rather than requiring a Visual Studio toolchain.
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass
        from audioseal import AudioSeal
        self._torch = torch
        self.device = device
        self.gen = AudioSeal.load_generator(generator).to(device).eval()
        self.det = AudioSeal.load_detector(detector).to(device).eval()

    # ------------------------------------------------------------- embed
    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Returns the watermarked waveform at the SAME sample rate it was given."""
        import librosa
        torch = self._torch
        x = np.asarray(wav, dtype=np.float32).reshape(-1)
        orig_sr = sr
        if sr != WM_SR:
            x16 = librosa.resample(y=x, orig_sr=sr, target_sr=WM_SR)
        else:
            x16 = x
        t = torch.from_numpy(x16).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            wm = self.gen.get_watermark(t, WM_SR)
            out = (t + wm).squeeze().cpu().numpy()
        if orig_sr != WM_SR:
            out = librosa.resample(y=out.astype(np.float32),
                                   orig_sr=WM_SR, target_sr=orig_sr)
            # length can drift by a sample or two through two resamples
            out = out[:len(x)] if len(out) >= len(x) else \
                np.pad(out, (0, len(x) - len(out)))
        return out.astype(np.float32)

    # ------------------------------------------------------------ detect
    def detect(self, wav: np.ndarray, sr: int) -> WatermarkResult:
        import librosa
        torch = self._torch
        x = np.asarray(wav, dtype=np.float32).reshape(-1)
        if sr != WM_SR:
            x = librosa.resample(y=x, orig_sr=sr, target_sr=WM_SR)
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out = self.det.detect_watermark(t, WM_SR)
        # audioseal returns (detection_prob[1], message[1,16]).
        # We use the PRESENCE BIT ONLY and deliberately ignore the message --
        # RESEARCH/09: the 16-bit payload averages 0.39 attribution and only
        # 0.69 even on clean audio, so a render ID must never live in it.
        prob = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(prob, "detach"):
            prob = prob.detach().cpu().numpy()
        arr = np.asarray(prob, dtype=np.float64)
        p = float(arr) if arr.ndim == 0 else float(arr.mean())
        return WatermarkResult(detected=p > 0.5, probability=p,
                               n_samples=len(x), duration_s=len(x) / WM_SR)

    # ---------------------------------------------------------- integrity
    def perceptual_delta(self, orig: np.ndarray, marked: np.ndarray) -> dict:
        """How much did the watermark change the audio? SNR in dB, higher = subtler."""
        o = np.asarray(orig, dtype=np.float64).reshape(-1)
        m = np.asarray(marked, dtype=np.float64).reshape(-1)
        n = min(len(o), len(m))
        o, m = o[:n], m[:n]
        noise = m - o
        return {"snr_db": float(10 * np.log10((o ** 2).sum() /
                                              max((noise ** 2).sum(), 1e-12))),
                "max_abs_delta": float(np.abs(noise).max())}


# --------------------------------------------------------------- attacks
# The attack set that actually matters for us, from RESEARCH/09's robustness
# table. Deliberately includes the ones AudioSeal FAILS, so the failure is
# visible in our own numbers rather than only in a paper.
def attack_polarity_inversion(wav: np.ndarray, sr: int) -> np.ndarray:
    """Inaudible one-liner. Published to drive detection to 0.18/0.00."""
    return -np.asarray(wav, dtype=np.float32)


def attack_resample(wav: np.ndarray, sr: int, to: int = 16000) -> np.ndarray:
    import librosa
    x = librosa.resample(y=np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=to)
    return librosa.resample(y=x, orig_sr=to, target_sr=sr).astype(np.float32)


def attack_gaussian_noise(wav: np.ndarray, sr: int, snr_db: float = 30.0) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32)
    p = (x ** 2).mean()
    n = np.random.default_rng(0).normal(0, np.sqrt(p / (10 ** (snr_db / 10))), len(x))
    return (x + n).astype(np.float32)


def attack_amplitude(wav: np.ndarray, sr: int, gain: float = 0.5) -> np.ndarray:
    return (np.asarray(wav, dtype=np.float32) * gain).astype(np.float32)


def attack_highpass(wav: np.ndarray, sr: int, cutoff: float = 300.0) -> np.ndarray:
    from scipy.signal import butter, lfilter
    b, a = butter(4, cutoff / (sr / 2), btype="high")
    return lfilter(b, a, np.asarray(wav, dtype=np.float32)).astype(np.float32)


ATTACKS = {
    "none": lambda w, sr: w,
    "polarity inversion": attack_polarity_inversion,
    "resample 24k->16k->24k": attack_resample,
    "gaussian noise 30dB": attack_gaussian_noise,
    "amplitude x0.5": attack_amplitude,
    "highpass 300Hz": attack_highpass,
}
