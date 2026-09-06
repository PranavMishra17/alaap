# S20 — a naturalness gate that is not a MOS predictor

**Run:** 2026-09-06 · IndicVoices-R Hindi · 140 real clips · 56 synthetic renders · WavLM base+ L6
**Question:** `S19` found nothing in this project can see what a listener hears. Can anything?

---

## The gap this fills, and the trap it had to avoid

`S19` established two things. Synthesis is distinguishable from real speech by a listener
(7 of 7 decisive judgements), and **no metric here can see it** — the codec roundtrip
differs from the real recording by 0.0 dB of SNR and 0.18 dB of HNR.

Drift, consistency, CER, uniqueness: every gate in the project is an **identity** or
**intelligibility** gate. All of them pass on audio a listener calls synthetic.

**The obvious fix was already ruled out.** `RESEARCH/06` §5.2 records Takagi et al.
perturbing F0 against human raters:

| | correlation with mean F0 |
|---|---|
| humans | **−0.059** |
| DNSMOS | **−0.788** |
| UTMOSv2 | −0.722 |

For a project whose point is spanning the pitch range, a MOS gate systematically rejects
high-pitched voices for a reason humans do not share — it would fight the diversity axis
directly. The same paper found all six predictors blind to prosody: pitch-accent corruption
cost humans 1.84 MOS points and moved every model under 0.1.

## What this is instead

Distance to the real-speech manifold in **WavLM base+ layer 6** features, mean-pooled,
scored with `metrics.isolation_pct` — machinery `E12` already validated by checking that
held-out *real* speakers score ~50%.

WavLM rather than ECAPA deliberately: **ECAPA is trained to be invariant to channel and
quality** so it can identify a speaker through them. That is exactly the wrong property.

## Result — every check passed

| group | n | isolation |
|---|---|---|
| **held-out REAL** | 70 | **49.8%** |
| S19 real (held-out) | 3 | 53.3% |
| S19 codec roundtrip | 3 | 53.3% |
| S14 retimed (rate 1.00) | 1 | 87.1% |
| S13 tagged renders | 4 | 92.1% |
| S7 catalog renders | 24 | **97.9%** |
| S6 minted renders | 21 | **98.0%** |

| check | | |
|---|---|---|
| 1 held-out real is typical | 49.8% | **PASS** |
| 2 synthetic scores higher | 98% vs 50% | **PASS** |
| 3 not a pitch detector | **r = −0.021** | **PASS** |
| 4 real vs real is a tie | d = +0.18 | **PASS** |

**Check 3 is the one that mattered.** At r = −0.021 the gate is closer to human behaviour
(−0.059) than to DNSMOS (−0.788) by a factor of forty. A gate that passed 1, 2 and 4 while
failing 3 would have been the documented trap in a new implementation.

The separation is not marginal: **real speech at 50%, this project's renders at 98%.**

## ⚠️ The limitation is as important as the result

**The codec roundtrip scores 53.3% — identical to real speech at 53.3%.** And a listener
told those two apart 2 times in 3 (`S19`).

So this gate detects **generated** speech and is **blind to the codec's own contribution.**

That has a direct consequence: `S19` found the naturalness deficit splits roughly evenly
between the codec and the model. **This gate can only see the model half.** Swap in a better
codec and it would not move — it would report no improvement on a real one.

It is therefore usable as an **output gate** ("is this render synthetic-sounding?") and
unusable as a **progress metric** for the codec. That distinction is written into the
docstring so the next person does not use it for the second thing.

## What was added

`alaap.metrics.naturalness_isolation`, with `NATURALNESS_REAL_TYPICAL = 50` and
`NATURALNESS_FLOOR = 75`. It is a thin wrapper over `isolation_pct` and exists mainly to be
**named** — the absence of any naturalness measure was invisible for nineteen experiments
precisely because nothing was called one.

Three invariant tests lock properties rather than numbers: typical and atypical separate, a
held-out reference scores mid-range rather than 0 or 100, and the documented floor sits
well above real speech.

## Not established

- **The floor of 75 is chosen, not measured.** It sits between real (50) and this
  project's renders (98) with room either side, and nothing has been rejected by it yet.
  `S11` is the standing warning about what happens when a chosen threshold meets a listener.
- **Reference is 70 Hindi clips from one corpus.** The gate is only meaningful against a
  reference from the same domain; scoring English renders against a Hindi reference would
  measure language, not naturalness.
- **Layer 6 was picked on general grounds** (middle layers carry phonetic and quality
  detail, top layers drift toward the pretext task) and not swept. Another layer might
  separate the codec roundtrip, which would remove the limitation above — that is the
  cheapest next test.
- **n = 1 to 4 for the S13, S14 and S19 groups.** Only the real (70), S6 (21) and S7 (24)
  numbers rest on a usable sample.
- **It has never rejected anything.** No render in the project has been gated on it, and a
  gate that has not yet blocked something is a proposal.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_gate.py --phase real
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_gate.py --phase synth
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_gate.py --phase validate
```

Three phases so features are cached and only one model is resident at a time. ~4 minutes.
