# S20b — the layer sweep: it does see the codec, at layer 5

**Run:** 2026-09-06 · IndicVoices-R Hindi · 300 real / 150 codec roundtrips / 45 synthetic · WavLM base+, all 12 layers
**Question:** does some layer see the codec degradation that layer 6 missed?

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


## Three answers to one question, and only the third is supported

This took three attempts, and the first two were both wrong in ways worth recording.

| attempt | n per side | claim | status |
|---|---|---|---|
| `S20` | **3** | the gate is *blind* to the codec | **not established** |
| `S20b` first pass | **40** | layer 5 *sees* it | **not established** |
| `S20b` final | **150** | layer 5 sees it, `t = 2.59` | **established** |

`S20` reported the roundtrip at 53.3 against real's 53.3 and concluded blindness. Resampling
3 of 40 clips reproduces that "identical" result in **41% of draws** — it was an ordinary
outcome of drawing three samples.

The first sweep then declared any layer with `gap ≥ 5` a winner — an effect-size cut with no
test behind it — and named three layers on that basis. At `n = 40` the standard error on the
difference is ~6 points, so a +7.9 gap is `t = 1.19`. The verdict logic now requires
`|t| ≥ 2`.

## The result at n = 150 per side

| layer | real | round | synth | gap | **t** | r(f0) | ctrl d | verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | 47.6% | 54.3% | 51.3% | +6.7 | **2.04** | **−0.578** | 0.05 | **pitch trap — rejected** |
| 1 | 47.3% | 53.7% | 66.8% | +6.4 | 1.85 | −0.445 | 0.07 | — |
| **4** | 45.9% | 54.6% | 95.8% | **+8.8** | **2.72** | −0.211 | 0.00 | sees the codec |
| **5** | 47.0% | 55.1% | 97.0% | **+8.1** | **2.59** | **−0.054** | 0.02 | **sees the codec** |
| 6 *(was in use)* | 46.0% | 52.4% | 98.5% | +6.5 | 1.98 | +0.058 | 0.06 | just misses |
| 8 | 47.6% | 47.5% | 93.4% | −0.2 | −0.05 | +0.078 | 0.09 | — |
| 10 | 48.3% | 44.7% | 92.8% | −3.6 | −1.01 | +0.020 | 0.06 | — |

**The codec's degradation is detectable.** `S19` measured it as roughly half the naturalness
deficit and a listener heard it 2 times in 3; layer 5 now sees it too.

### The pitch-trap check earned its keep

**Layer 0 separates the codec significantly — `t = 2.04` — at `r(f0) = −0.578`.** That is
DNSMOS territory (−0.788), and a sweep chasing the largest significant gap would have taken
it. It was rejected on the pitch check, which is the entire reason that check is scored
alongside the gap rather than after it.

The pattern is orderly: early layers encode pitch (−0.578, −0.445, −0.326), middle layers do
not (−0.054, +0.058), late layers see nothing at all.

### Why layer 5 rather than layer 4

Layer 4 has the bigger gap (+8.8 vs +8.1) and the higher `t` (2.72 vs 2.59). **Layer 5 is
chosen anyway**, because its pitch correlation is −0.054 against humans' −0.059, where layer
4 sits at −0.211 — four times further from human behaviour for a 0.7-point gain in
separation. Given what `RESEARCH/06` documents about pitch-correlated gates, that is not a
trade worth making.

## Re-validated at layer 5 before switching

The four-check battery from `S20`, re-run at the new layer:

| group | n | isolation |
|---|---|---|
| **held-out REAL** | 70 | **51.6%** |
| S13 tagged | 4 | 96.4% |
| S14 retimed | 1 | 97.1% |
| S6 minted | 21 | **97.3%** |
| S7 catalog | 24 | **97.5%** |

| check | | |
|---|---|---|
| 1 held-out real is typical | 51.6% | **PASS** |
| 2 synthetic scores higher | 97% vs 52% | **PASS** |
| 3 not a pitch detector | **r = −0.068** | **PASS** |
| 4 real vs real is a tie | d = +0.17 | **PASS** |

`r = −0.068` against a human reference of `−0.059`. The gate is behaving, on this axis,
like a person.

## The prediction, written down before the run

> *"Current SE ≈ 5.9 at n=40/50. At n=150/150 the SE falls to ≈3.2, so a genuine +7 gap
> would land at t ≈ 2.2. If it's noise, t stays near zero."*

Layer 6 came in at **t = 1.98**, layers 4 and 5 at **2.72** and **2.59**. The prediction was
right about the mechanism and slightly conservative about the size.

## Not established

- **One corpus, one language, one codec variant.** The layer choice may not transfer; it
  should be re-swept for English before the gate is used there.
- **The gate still has never rejected anything.** No render has been blocked by it. A gate
  that has not yet blocked something is a proposal.
- **`NATURALNESS_FLOOR = 75` remains chosen, not measured.** It sits between real (52) and
  renders (97) with room either side, and no listener has been asked whether a clip at 75
  sounds worse than one at 60.
- **The roundtrip effect is real but small** — 8 points against a 45-point synth-vs-real
  separation. It is detectable, not dominant, which is consistent with `S19` finding the
  codec responsible for about half a deficit a listener could hear only 2 times in 3.
- **`r(f0)` moves between runs and pooled sets** (−0.021, +0.210, −0.068 across three
  measurements at two layers). All far from −0.79, but the figure is less stable than any
  single number implies.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_layer_sweep.py --phase audio --n 300 --n-round 150
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_layer_sweep.py --phase feats --n 300 --n-round 150
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_layer_sweep.py --phase sweep --n 300 --n-round 150
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_gate.py --phase real --layer 5
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_gate.py --phase synth --layer 5
envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_gate.py --phase validate --layer 5
```

One forward pass yields all 12 layers, so sweeping them costs the same as scoring one.
~20 minutes end to end, most of it collecting audio.
