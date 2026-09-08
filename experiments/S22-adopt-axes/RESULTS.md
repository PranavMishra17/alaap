# S22 — the S17/S18 axis set is now the shipped default

**Run:** 2026-09-08 · IndicVoices-R Hindi · 141 real speakers · 80 mints
**What changed:** `alaap.catalog.CATALOG_AXES`, the one constant that drives both the
caption renderer and the cell sampler.

```
    was   f0_mean, spectral_tilt, hnr_db, f0_cv, speaking_rate     3,125 cells
    now   f0_mean, spectral_tilt, vtl_cm                             125 cells
```

---

## Why each axis moved

| axis | | evidence |
|---|---|---|
| `f0_cv`, `speaking_rate` | **out** | within-speaker spread *exceeds* between-speaker spread on real people — 1.04 and 1.32 (`S16`), reproducing `S12` on acted emotion and `E15` on three corpora. Three independent routes to the same conclusion. |
| `hnr_db` | **out** | weakest of the four at 0.59, and **fails outright on Tamil at 1.32**. Measured cost of keeping it: **0.3 effective voices bought for 15.6 points of adherence** (`S18`). |
| `vtl_cm` | **in** | never noise — the *estimator* was broken, averaging through the vowel-dependent F2. Repaired in `S17`; gender separation went from ~0 to +0.44…+1.03 on four corpora. |

## It reproduces the experiment to the published decimal

`S17`/`S18` measured this swap as one arm among three inside a single script. This runs **the
adopted constant** instead — a different claim, because a local variable in an experiment
agreeing with itself says nothing about what the library does by default.

| | S18 published | S22 measured |
|---|---|---|
| effective voices | 22.3 | **22.3** (0.4649 × 48) |
| adherence, `f0_mean` + `spectral_tilt` | 98.1% | **98.1%** |
| `vtl_cm` adherence | 87.5% | **87.5%** |

Three independent numbers to the published decimal. Four properties were asserted before any
of them was read, and all four passed.

**The headline adherence here is 94.6%, not 98.1%, and that is not a discrepancy.** `S18`
scored every arm on the axes *common to all arms* — `f0_mean` and `spectral_tilt` — because
grading one arm on a harder axis than another is a comparison of the scoring, not the arms.
There is only one arm here, so all three of its axes are scored:

| axis | adherence |
|---|---|
| `f0_mean` | 100.0% |
| `spectral_tilt` | 96.2% |
| `vtl_cm` | 87.5% |
| **mean of three** | **94.6%** |
| mean of the S18 pair | 98.1% |

Against the shipped five-axis baseline's **85.8%**, on the same corpus.

## What the adoption cost, and the guard added for it

Three axes × five bins is **125 describable descriptions**, down from 3,125. `sample_cells`
used to return 125 in silence when asked for more.

**That is a silent truncation that reads as the catalog saturating when really the sampler
exhausted**, and `HANDOFF` §10's product target is "a few hundred curated voices". It now
raises, naming the fix:

```
ValueError: asked for 300 distinct cells but 3 axes x 5 bins is only 125
describable descriptions (f0_mean, spectral_tilt, vtl_cm). Returning 125
silently would make a saturation curve read as the catalog saturating rather
than the sampler exhausting. Add bins or an axis.
```

`S18` flagged that the cell count must bind eventually and left *where* untested. This makes
the boundary announce itself rather than be absorbed into a result.

## Five experiments were pinned, not migrated

`CATALOG_AXES` is read by eight experiments. Changing it in place would leave every
`RESULTS.md` quoting an acceptance count or an adherence percentage describing a run nobody
can reproduce — **the same class of mistake as a cache key without a model id, which this
project has now made twice (`E0`, `S2`)**.

So `CATALOG_AXES_V1` stays, and `S8`, `S10`, `S16`, `E11` and `S17` import it explicitly.

`S17` is the sharp case: its arm 1 *is* "the five shipped axes", so following the new default
would have made it compare the identity set against itself and report no difference.

`S7`'s outputs now carry the axis set in the filename (`catalog_E_hi_id-f0+tilt+vtl.npz`), so
a re-run cannot overwrite the five-axis numbers its own write-up quotes.

## Two caveats that travel with this set

1. **`spectral_tilt` is not a clean brightness axis.** `S21` measured it reading F0 at
   `r = +0.918` where brightness is fixed by construction. It was kept because repairing it
   changes nothing measurable, but **count this set as ~2.6 independent axes, not 3.**
2. **This run samples 64% of its own describable space** (80 of 125). `S17` noted the
   acceptance rate is flattered by nearly running out of distinct things to ask for — so the
   *saturation curve* from a run this close to the ceiling measures the sampler, not the
   catalog. Effective voices and adherence are unaffected; the curve is not.

## Not established — and the half of item 1 still outstanding

**Nothing was rendered or heard.** This is geometry, exactly as `S17` and `S18` were.
Effective voices is a Vendi artefact, and `S11` is the standing record of what happens when a
geometric threshold meets a listener.

`HANDOFF` item 1 also asks for `S7`'s render pass — drift, consistency and CER **under the
audited pipeline rather than in geometry**. That needs the codec and the LM resident
together, and the box had **1.6 GB free of 15.7 GB** (Opera 5.9, Claude 3.8). Under the ~3 GB
floor in `HANDOFF` §8b that fails in ways that do not look like memory, so it was not
attempted. One command when there is room:

```bash
envs/qwen3/Scripts/python.exe experiments/S7-indic-catalog/run_indic_catalog.py --n 80
```

Also untested: **English**. `hnr_db` is used there and `S18`'s audit never ran on GLOBE_V2,
whose cache lacks the per-speaker repeats the within/between screen needs.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S22-adopt-axes/run_confirm.py   # ~3 min, no models
```
