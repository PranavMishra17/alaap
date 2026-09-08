# S20 — a naturalness gate that is not a MOS predictor

**Run:** 2026-09-06 · IndicVoices-R Hindi · 140 real clips · 56 synthetic renders · WavLM base+ L6
**Superseded in part by `S20b`:** the gate now runs at **layer 5**, where it also detects codec degradation.
**Question:** `S19` found nothing in this project can see what a listener hears. Can anything?

---

## ⚠️ CORRECTED BY S23 — layer 5 is no longer the default

`S23` re-scored this sweep with two of its statistics repaired, and both errors are in the
checks rather than the data:

1. **The pitch correlation was POOLED over real and synthetic clips.** Synthetic clips score
   ~50 points higher *and* have a different F0 distribution, so the pooled number partly
   measures the group split. Within real speech — where naturalness is constant, so the
   correlation is unambiguous — **layer 5 reads −0.154 against a 0.165 bar, passing by
   0.011**, and **+0.323 on English, which it fails.**
2. **The real-vs-real control came from ONE arbitrary split.** Over 200 splits every layer
   lands at 0.11–0.13 here and 0.18–0.20 on English. The single-split values were noise.

Two consequences for what is written below:

- **Layer 4 is a pitch trap** at r = −0.370 within real speech. It is scored −0.211 here,
  passed, and then lost to layer 5 on that margin. The corrected statistic disqualifies it.
- **Layer 6 is cleaner than layer 5** (+0.004 vs −0.154) *and* separates synthetic better
  (t = 22.43 vs 21.51). It lost only on the codec question, at t = 1.98 against a bar of 2.

**What survives unchanged is the codec result**: layer 5 does detect MioCodec round-tripping
at +8.1 points, t = 2.59. That is why it is kept as `NATURALNESS_LAYER_CODEC`. But the gate's
default is now **layer 7**, which is clean on pitch in both languages and blind to the codec
(t = 0.28). **No layer does both** — the layers that see the codec are the ones that carry
pitch.

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

## ⚠️ WITHDRAWN — and then settled the other way

**The section that follows rested on 3 roundtrip clips.** Resampling 3 of 40 reproduces its
"identical" result in **41% of draws**, so it was an ordinary sampling outcome rather than a
finding.

`S20b` then settled it at **150 clips per side: the gate DOES see the codec** — the
roundtrip sits +8.1 points above real speech at `t = 2.59`, at **layer 5**. The gate has
been moved there and re-validated (all four checks pass, `r(f0) = −0.068` against humans'
−0.059). Layer 6, used below, comes in at `t = 1.98` and just misses.

So the limitation was not merely unproven — it was **wrong**. This gate can track codec
progress after all.

What survives unchanged is what the gate was validated for: **flagging generated speech**,
where the separation is 50 vs 98.

*(Original text kept below, struck through in substance, because the reasoning is the
instructive part.)*

## ~~The limitation is as important as the result~~

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
- ~~Layer 6 was picked on general grounds and not swept.~~ **`S20b` swept all 12.** None
  separates the codec roundtrip significantly; layer 6 is not beaten on evidence and stays.
  The sweep's own first pass declared three layers winners on an effect-size cut with no
  test behind it, and that is corrected there.
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
