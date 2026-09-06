# S20b — the layer sweep, and a claim of mine it withdraws

**Run:** 2026-09-06 · IndicVoices-R Hindi · 100 real / 40 codec roundtrips / 45 synthetic · WavLM base+, all 12 layers
**Question:** does some other layer see the codec degradation layer 6 missed?

---

## What this was meant to fix

`S20` shipped a working naturalness gate with one stated limitation: **the codec roundtrip
scored 53.3 against real speech's 53.3 — identical — while a listener told those apart 2
times in 3** (`S19`). So the gate saw generated speech and was blind to the codec's own
contribution, which `S19` measured as roughly half the naturalness deficit.

Layer 6 was chosen on general grounds and never swept. If another layer separated the
roundtrip, the limitation would disappear.

## The sweep

Every layer scored on four things at once, because a layer that separates the roundtrip by
tracking **pitch** is the DNSMOS trap in a new place, and one that separates two **real**
clips is measuring channel rather than naturalness.

| layer | real | round | synth | gap | **t** | r(f0) | ctrl d |
|---|---|---|---|---|---|---|---|
| 1 | 48.0% | 55.1% | 60.0% | +7.1 | 1.09 | −0.156 | 0.24 |
| **5** | 48.6% | 56.5% | 91.5% | **+7.9** | **1.19** | +0.206 | 0.07 |
| **6** | 52.0% | 59.2% | 94.0% | **+7.2** | **1.23** | +0.210 | 0.09 |
| 7 | 55.0% | 54.3% | 93.5% | −0.7 | −0.12 | +0.166 | 0.04 |
| 10 | 54.4% | 52.0% | 90.8% | −2.4 | −0.38 | +0.194 | 0.00 |

*(12 layers run; the informative rows shown.)*

**No layer separates the codec roundtrip significantly.** The largest gap is layer 5 at
+7.9 points, `t = 1.19` — not distinguishable from zero. Every layer's synth-vs-real
separation is large and obvious; every layer's roundtrip-vs-real separation is not.

## ⚠️ Which withdraws S20's stated limitation as well

`S20` reported the roundtrip at 53.3 against real's 53.3 — a gap of exactly zero — and
concluded the gate is *blind to the codec*. **That rested on 3 roundtrip clips.**

Resampling 3 of these 40 at random, 2000 times: **the gap falls under +2 points in 41% of
draws.** S20's zero was an ordinary outcome of drawing three samples, not evidence of
blindness.

So both statements fail:

| | status |
|---|---|
| "the gate is blind to the codec" (`S20`, n=3) | **not established** |
| "layer 5 sees the codec" (this sweep's first pass) | **not established** |

The honest position is that **this design cannot tell**, in either direction, and neither
write-up was entitled to say otherwise. `S20`'s `RESULTS.md` and the docstring in
`alaap.metrics.naturalness_isolation` both carried the blindness claim as fact; both are
corrected.

## ⚠️ And a methodological failure in this script's own first version

The sweep initially declared any layer with `gap >= 5` as **"sees the codec"** — an
effect-size cut with no test behind it. On that basis it named layers 1, 5 and 6 as
winners and printed *"S20's limitation is removable: switch the gate to layer 5."*

At this n the standard error on the difference is about 6 points. A +7.9 gap is `t = 1.19`.
**Three layers were declared winners on a difference that a test rejects.**

The verdict logic now computes `t` and requires `|t| >= 2`. This is the same error the
project has caught repeatedly in its own numbers — `E9`'s worst-case reversal, `S17`'s
mismatched adherence scoring — and it went in anyway, in a script written specifically to
be careful about the pitch trap. Being alert to one failure mode is not being careful.

## What would settle it

`~114 roundtrip clips against ~114 real`, from the observed effect size and variance, to
reach `t = 2`. That is 3× what was used here and entirely affordable — the audio phase
took under two minutes for 40.

Worth doing only if the codec half matters. `S19` says it is about half the deficit, so
probably yes, but the gate as it stands is still usable for what it was validated for:
**flagging generated output**, where the separation is 50 vs 94 and beyond argument.

## Not established

- **Whether any layer sees the codec.** That is the whole point of the above.
- **The layer choice is unchanged.** Layer 6 stays, not because it won, but because nothing
  beat it on evidence and it is what `S20`'s validation battery was run against.
- **`r(f0)` moved between runs** — −0.021 in `S20`, +0.210 here at the same layer — because
  the pooled sets differ (S20 included the S19 A/B clips, this includes only S6/S7 renders).
  Both are far from the −0.79 that would matter, but the figure is less stable than one
  number suggested.
- **One corpus, one language, one codec variant.**

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_layer_sweep.py --phase audio
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_layer_sweep.py --phase feats
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_layer_sweep.py --phase sweep
```

One forward pass yields all 12 layers, so sweeping them costs the same as scoring one.
