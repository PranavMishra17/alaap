# S14 — a direction channel that works, by not asking the model

**Run:** 2026-09-05 · `Indic-Mio` + `MioCodec-25Hz-44.1kHz-v2` · 2 voices × 2 lines × 10 factors × 2 methods
**Question:** if the text channel carries no direction, can the signal?

---

## Why this route

`S13`/`S13b` established that Indic-Mio's text channel carries no direction — the emotion
tags are not tokens, and word stress is inert. What survived was the harder property:
**identity is provably robust to whatever the text does.**

`S12` says `speaking_rate` is a real delivery axis (within/between ratio 2.01) and
near-worthless for identity (weight 0.10–0.31). So drive it directly and stop asking the
model to cooperate. A phase vocoder changes duration while preserving pitch — exactly the
shape needed: move the delivery axis, leave the identity axis alone.

## The three properties, all stated before the run

**1. Rate tracks the request.** Measured `speaking_rate` should scale as `1/factor`, or the
instrument is broken and nothing else is readable.

```
requested vs achieved rate ratio:  r = 0.9999
```

**2. Pitch does not move.** A phase vocoder preserves f0 by construction, and `ADR-012`
puts `f0_mean` off-limits to direction.

**3. The words survive.** CER at every factor.

| factor | rate × | f0 shift | ECAPA | normalised | CER hi | CER en |
|---|---|---|---|---|---|---|
| 0.60 | 1.66 | +2.3 Hz | 0.7256 | +1.054 | 0.075 | 0.016 |
| **0.70** | 1.42 | +0.6 | **0.8029** | +1.209 | 0.100 | 0.000 |
| 0.80 | 1.25 | +1.2 | 0.8636 | +1.331 | 0.075 | 0.000 |
| 0.90 | 1.11 | −0.7 | 0.8772 | +1.359 | 0.100 | 0.000 |
| 1.00 | 1.00 | +0.0 | 1.0000 | +1.605 | 0.075 | 0.000 |
| 1.15 | 0.87 | +0.6 | 0.8730 | +1.350 | 0.075 | 0.000 |
| 1.30 | 0.77 | +0.9 | 0.8521 | +1.308 | 0.100 | 0.000 |
| **1.50** | 0.67 | +0.6 | **0.8083** | +1.220 | 0.075 | 0.000 |
| 1.75 | 0.57 | +0.5 | 0.7823 | +1.168 | 0.075 | 0.000 |
| 2.00 | 0.50 | +0.4 | 0.7799 | +1.163 | 0.075 | 0.000 |

**Pitch moves by at most 2.3 Hz across a 3.3× range of speaking rates.** The Hindi CER
floats between 0.075 and 0.100 with no trend — that is `whisper-small`'s Devanagari
orthography, the same residual `S5b` measured on untouched audio, not damage from
stretching. English CER is **0.000 everywhere** except 0.016 at the most extreme
compression.

## The negative control, which is what makes the above readable

Naive resampling — playing the audio faster — changes duration *and* pitch together. It is
what the Indic-Mio card's own 44100-vs-24000 bug does by accident.

| factor | f0 shift | ECAPA | CER en |
|---|---|---|---|
| 0.70 | **−54.7 Hz** | **0.1042** | 0.000 |
| 1.00 | +0.0 | 1.0000 | 0.000 |
| 1.50 | **+91.3 Hz** | **0.1080** | 0.000 |

**ECAPA collapses to ~0.10 — a completely different speaker.** So the identity metric *can*
detect a rate change that destroys identity, and its verdict of 0.80–0.88 for the phase
vocoder is a real pass rather than an insensitive one.

Note also that CER stays 0.000 for the naive control: **intelligibility is not a proxy for
identity.** A render can be perfectly understandable and be someone else entirely — which
is the same lesson the wrong-codec failure taught from the other direction.

## The deliverable: a measured operating range

```
USABLE RANGE (ECAPA >= 0.80 and English CER <= 0.10):  x0.70 to x1.50
  = speech from 0.67x to 1.43x normal rate
```

That is a bound of the same kind as `DRIFT_FLOOR` and `CONSISTENCY_FLOOR` — measured, not
chosen. Outside it, ECAPA falls to 0.72–0.78 (still recognisably the same person, but
degrading) rather than failing sharply, so the range is a quality boundary, not a cliff.

**For comparison, Sarvam's Bulbul exposes `pace` at 0.5–2.0** and Indic-Mio exposes nothing
at all. This gets to 0.67–1.43× on a backend with no rate control, without touching the
model.

## What this does and does not give the product

**Gives:** one real, bounded direction axis, on the axis `S12` measured as the strongest
delivery carrier, with identity provably intact and a negative control proving the
identity check works.

**Does not give:** emotion. Rate is one axis of delivery. "Authoritative versus asking for
forgiveness" differ in pitch contour, emphasis placement and voice quality as well as rate,
and `S13`/`S13b` showed this backend offers no lever for any of those. A rate control alone
will not render the original product question.

## Not established

- **Nobody has listened.** Renders at ×0.70, ×1.00 and ×1.50 are saved in `out/`. Whether
  ×1.43 rate sounds *directed* rather than merely *fast* is exactly the question numbers
  cannot answer, and `S11`/`S13` both turned on a listener contradicting the metrics.
- **2 voices, 2 lines, one seed.** The ECAPA figures are means over 4 renders per factor.
- **Phase vocoding is audible.** The metrics say identity and words survive; they say
  nothing about artefacts, and a vocoder at ×2.0 usually sounds like one. `RESEARCH/10`'s
  0.6 intensity ceiling exists because of exactly this class of artefact.
- **CER is `whisper-small`.** The gated `ai4bharat/indic-conformer-600m-multilingual` is
  the better instrument and the Hindi floor of 0.075–0.100 is its orthography, not synthesis.
- **The range was measured with thresholds chosen here** (ECAPA ≥ 0.80, CER ≤ 0.10). The
  0.80 is not listener-validated — `S11` showed what happens when a borrowed threshold
  meets a listener.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S14-rate-control/run_rate_control.py
```

~10 minutes on a 6 GB card. Needs `S4`'s measurement cache and `S6`'s MioCodec embedding
cache.
