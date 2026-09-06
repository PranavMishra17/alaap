"""
Signal-level speaking-rate control — the one direction axis that works here.

`S13`/`S13b` measured that Indic-Mio's text channel carries no direction: the
documented emotion tags are not tokens and perturb the model no more than
invented words of the same shape, and word stress is inert. What survived was
the harder property — **identity is robust to whatever the text does**.

`S12` measured `speaking_rate` as a genuine delivery axis (within-speaker /
between-speaker variance ratio 2.01 on CREMA-D) and near-worthless for identity
(weight 0.10-0.31 across three corpora). So drive it in the signal and stop
asking the model to cooperate.

A phase vocoder changes duration while preserving pitch, which is exactly the
required shape: move the delivery axis, leave the identity axis alone.
`ADR-012` puts `f0_mean` off-limits to direction, and S14 measured the shift at
**at most 2.3 Hz across a 3.3x range of rates**.

NAIVE RESAMPLING IS THE FAILURE THIS AVOIDS. Playing audio faster changes
duration AND pitch together -- it is what the Indic-Mio card's own
44100-vs-24000 bug does by accident. S14 ran it as a negative control: f0 moved
-54.7 / +91.3 Hz and ECAPA collapsed to 0.10, a completely different speaker.
That is what `retime` must not do, and the fact that the identity metric caught
it is why the vocoder's 0.80-0.88 is a real pass rather than an insensitive one.
"""
from __future__ import annotations

import numpy as np

# S14, measured not chosen: the range over which ECAPA identity stays >= 0.80
# and English CER stays <= 0.10. Expressed as a SPEED multiplier, so 1.43 is
# faster speech and 0.67 is slower.
RATE_MIN, RATE_MAX = 0.67, 1.43

# Where those numbers came from, so a caller on another backend is warned
# rather than silently inheriting a number measured somewhere else -- the same
# mistake ADR-011 records for the uniqueness floor.
RATE_BOUND = {
    "min": RATE_MIN,
    "max": RATE_MAX,
    "identity_retained_ecapa": 0.80,
    "cer_max_en": 0.10,
    "f0_shift_hz_max": 2.3,
    "measured_on": "indic-mio+miocodec-25hz-44.1khz-v2 (S14)",
    # A listener heard the slowest end as "60% just slowed down, 40% genuine
    # slow speech". So the bound is where identity and words survive, NOT where
    # the result stops sounding like an effect.
    #
    # THIS INNER RANGE IS A GUESS AND S14b FAILED TO MEASURE IT. That listening
    # test called an UNTOUCHED rate-1.00 control "processed", so it could not
    # separate the re-timing from the synthesis -- the baseline already sounds
    # processed to a native listener. Measuring it needs a comparative design
    # ("which of these sounds MORE processed?"), which is robust to a synthetic
    # baseline because both sides carry it.
    "listener_clean_range": (0.8, 1.25),
    "listener_clean_range_measured": False,
}


def retime(wav: np.ndarray, rate: float, sr: int | None = None) -> np.ndarray:
    """
    Change speaking rate by `rate` (a speed multiplier) without moving pitch.

    `rate` > 1 is faster speech, < 1 is slower. `sr` is unused -- a phase
    vocoder works in frames -- and is accepted so callers can pass it without
    thinking about whether it matters.

    Not clamped here: clamping is a policy decision that belongs to whoever can
    report the degradation to the caller. `clamp_rate` does that.
    """
    import librosa

    w = np.asarray(wav, dtype=np.float32).reshape(-1)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"rate must be a positive finite multiplier, got {rate!r}")
    if abs(rate - 1.0) < 1e-6 or len(w) < 2048:
        return w

    # Stretch factor is the reciprocal: to speak 1.5x faster the signal is
    # compressed to 1/1.5 of its length.
    factor = 1.0 / float(rate)

    # librosa's phase_vocoder raises `t_out values must be in the range
    # [0, D.shape[-1])` when the resampled frame index lands exactly on the
    # final frame. It depends on signal length and hit about one render in four
    # in S14. Padding first avoids it; truncating after is what keeps the
    # measurement honest, because `speaking_rate` is phones per second over the
    # WHOLE clip and trailing silence would depress the very axis being moved.
    pad = 4096
    for extra in (pad, 2 * pad):
        try:
            out = librosa.effects.time_stretch(y=np.pad(w, (0, extra)), rate=rate)
            break
        except Exception:
            out = None
    if out is None:
        return w

    want = int(round(len(w) * factor))
    return out[:want] if len(out) >= want else np.pad(out, (0, want - len(out)))


def clamp_rate(rate: float) -> tuple[float, list[str]]:
    """
    Clamp a requested rate to the measured bound, reporting what was lost.

    Returns (effective_rate, degradations). A request outside the bound is
    honoured up to the edge rather than refused, because the failure past it is
    gradual -- S14 measured ECAPA 0.78 at 2x rather than a collapse -- but the
    caller is always told.
    """
    r = float(rate)
    if r < RATE_MIN:
        return RATE_MIN, [
            f"rate {r:.2f} is below the measured bound {RATE_MIN} "
            f"(S14: identity degrades past it); clamped to {RATE_MIN}"]
    if r > RATE_MAX:
        return RATE_MAX, [
            f"rate {r:.2f} is above the measured bound {RATE_MAX} "
            f"(S14: identity degrades past it); clamped to {RATE_MAX}"]
    return r, []
