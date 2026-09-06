# S18 — auditing every axis, and finding the screen has a threshold

**Run:** 2026-09-06 · IndicVoices-R hi/bn/ta · 109–120 speakers with ≥2 clips each
**Question:** the within/between check found three things one at a time. What does it find run over everything?

---

## Why run it over everything

The same one-line check had already found three things, each because something else drew
attention to it:

| | |
|---|---|
| `speaking_rate` | within/between **1.32** — not an identity axis (`S16`) |
| `f0_cv` | **1.04** — not an identity axis (`S16`) |
| `vtl_cm` | at chance, and the **estimator was broken**; fixing it gave a fourth real axis (`S17`) |

Three for three. The base rate said run it over the lot.

## The audit

`ratio` = within-speaker variance / between-speaker variance over two takes.
`redund` = max |correlation| with an axis already in the identity set.

| axis | hi | bn | ta | mean | redund | verdict |
|---|---|---|---|---|---|---|
| **f0_mean** | 0.05 | 0.04 | 0.15 | **0.08** | 0.80 | in use |
| **spectral_tilt** | 0.14 | 0.22 | 0.20 | **0.19** | 0.54 | in use |
| **vtl_cm** | 0.18 | 0.19 | 0.26 | **0.21** | 0.56 | in use |
| f3 | 0.28 | 0.13 | 0.25 | 0.22 | 0.75 | redundant with `vtl_cm` |
| f1 | 0.20 | 0.19 | 0.32 | 0.23 | **0.90** | redundant |
| formant_dispersion | 0.45 | 0.21 | 0.27 | 0.31 | 0.55 | derivative |
| f2 | 0.54 | 0.30 | 0.49 | 0.44 | 0.63 | derivative |
| f0_std | 0.75 | 0.40 | 0.52 | 0.56 | 0.66 | restates `f0_mean` |
| **hnr_db** | 0.28 | 0.17 | **1.32** | **0.59** | 0.80 | **in use — and fails on Tamil** |
| shimmer | 0.49 | 0.43 | 0.95 | 0.62 | **0.50** | candidate |
| voiced_frac | 0.73 | 0.86 | 0.25 | 0.62 | 0.79 | candidate |
| f0_range | 0.85 | 0.37 | 0.87 | 0.69 | 0.66 | restates `f0_mean` |
| jitter | 0.81 | 0.62 | 0.77 | 0.73 | **0.36** | candidate |
| f0_cv | 1.04 | 0.55 | 0.63 | 0.74 | 0.20 | known dead |
| speaking_rate | 1.32 | 0.55 | 1.23 | 1.03 | 0.19 | known dead |
| duration_s | 2.22 | 1.66 | 1.55 | 1.81 | 0.17 | not identity |

**`jitter` and `shimmer` are the only survivors that are both separating and independent.**
Everything else that passes is a restatement of something already in the set — `f0_std` and
`f0_range` are `f0_mean` with more room to move, `f3` and `formant_dispersion` are what
`vtl_cm` is computed from.

## Then they were tested, and they made things worse

Passing the screen is not the same as helping. Adding them to the identity set:

| arm | cells | effective | of bound | adherence |
|---|---|---|---|---|
| **4-axis (identity)** | 625 | **22.6** | **59%** | 82.5% |
| 5-axis (+shimmer) | 3,125 | 19.1 | 50% | 82.1% |
| 6-axis (+jitter+shimmer) | 15,625 | 19.6 | 51% | 79.2% |

**Both hurt.** Diversity falls and adherence falls with them.

### The screen has a threshold, not a line

The four axes in use score **0.08, 0.19, 0.21, 0.59**. `jitter` and `shimmer` score **0.73**
and **0.62** — under 1.0, so they do separate speakers, but weakly. A weakly-separating axis
is one more thing the mapper must match on while carrying little identity, and it competes
with the strong axes for retrieval weight.

**`ratio < 1` is necessary and nowhere near sufficient.** What matters is `ratio ≪ 1`.

## Which indicts an axis already in use

`hnr_db` sits at **0.59** — the weakest of the four — and **fails outright on Tamil at
1.32**. If the threshold reading is right, removing it should help rather than hurt:

| arm | cells | effective | of bound | adherence | vtl adherence |
|---|---|---|---|---|---|
| 4-axis (with `hnr_db`) | 625 | **22.6** | 59% | 82.5% | 80.0% |
| **3-axis (without)** | 125 | 22.3 | 58% | **98.1%** | **87.5%** |

**`hnr_db` buys 0.3 effective voices and costs 15.6 points of adherence.** That is a bad
trade, and the prediction from the threshold reading held.

## ⚠️ This refutes something I claimed in S17

`S17` framed the 3-axis result as crippled by a **cell ceiling** — 125 describable
descriptions against 3,125 — and presented restoring `vtl_cm` as lifting that ceiling 5×.

Put every arm on one axis and that story does not survive:

| cells | effective voices |
|---|---|
| 125 | 22.3 |
| 625 | **22.6** |
| 3,125 | 19.6 |
| 15,625 | 19.6 |

**125× more describable cells moved effective voices by −2.7.** The size of the describable
space is not what limits capacity, at this n. **The number of *strong* identity axes is.**

That was worth being wrong about, because it changes what to do next: not more bins, not
more axes, but **better measurement of the axes already there.** `vtl_cm` went from noise to
the third-best axis in the set by fixing one formula.

*(The cell count must bind eventually — 125 descriptions cannot produce 500 distinct
voices. At n=80 it does not bind, and where it starts to is untested.)*

## Not established

- **Two takes per speaker.** The within-speaker variance estimate is noisy, and `hnr_db`
  swinging 0.17 → 1.32 across corpora may be that noise rather than a language effect. It
  is the same design used for every other axis, so the comparison holds even if the
  absolute ratios are soft.
- **Indic only.** The audit did not run on GLOBE_V2, whose cache lacks the per-speaker
  repeats this needs. `hnr_db` may behave differently in English, and it is used there.
- **Geometry only, nothing rendered or heard.** Every "effective voices" figure is a Vendi
  artefact, and `S11` showed what happens when one meets a listener.
- **`voiced_frac` was screened out on redundancy (0.79), not tested.** `S13` separately
  found it is an energy/ZCR measure capped near 0.60 and bimodal, so it is unlikely to be
  a good axis, but that is an argument rather than this experiment's measurement.
- **Nothing was changed in the shipped defaults.** `S6`/`S7` still write five axes. The
  4-axis and 3-axis sets are measured, not adopted.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S18-axis-audit/run_axis_audit.py
envs/qwen3/Scripts/python.exe experiments/S17-identity-axes/run_identity_axes.py
```

CPU only, a few minutes each.
