# S17 — two dead caption axes, and one that was never dead

**Run:** 2026-09-06 · IndicVoices-R hi/bn/ta + GLOBE_V2 · 80 mints per arm
**Question:** `S16` showed two of the five caption axes distinguish nobody. Does dropping them help?

---

## The setup, and why the obvious fix might have backfired

`S16` found `f0_cv` and `speaking_rate` have within-speaker spread *larger* than
between-speaker spread on real people — 1.04 and 1.32. `S12` found the same two axes on
acted emotion. `E15` measured `speaking_rate` at identity weight 0.10–0.31 on three
corpora. Three independent routes to the same conclusion: **they are not identity axes.**

Captions are written over five axes, so two of every five words spent describing a voice
describe nothing that separates voices.

**But dropping them shrinks the describable space from 5⁵ = 3,125 cells to 5³ = 125.**
Fewer distinct descriptions can mean fewer distinct voices no matter what the dropped axes
carried. Two effects pushing opposite ways, which is why this was measured rather than
applied.

## Part 1 — dropping the dead axes helps, on 3 axes

Bound from `S9`: **~38 effective voices** in these 141 real speakers.

| arm | cells | accepted | **effective** | of bound | adherence |
|---|---|---|---|---|---|
| 5-axis, as shipped | 3,125 | 39 | 19.6 | 51% | 85.8% |
| **3-axis, identity only** | 125 | 50 | **22.3** | **58%** | **95.8%** |
| 5-axis, dead axes gated | 3,125 | 33 | 18.4 | 48% | 95.4% |

**+14% effective voices and +10 points of adherence, from 25× fewer describable cells.**
The "fewer cells means fewer voices" worry was wrong — the cells were never the binding
constraint.

### The third arm explains the mechanism

Arm 3 keeps all five axes in the caption but forces the mapper's hybrid weights to zero on
the two dead ones. If the damage were purely in retrieval scoring, it should have matched
arm 2.

**It gets the adherence benefit (95.4%) but not the diversity benefit (18.4, worse than
baseline).** So the dead axes hurt through two separate channels:

1. **Retrieval scoring** — they pull anchors toward matches on axes that identify nobody.
   Zeroing the weights fixes this: adherence 85.8% → 95.4%.
2. **The sampler** — with 5-axis cells and gated weights, two descriptions differing *only*
   in `speaking_rate` become indistinguishable, collide, and are rejected as duplicates.
   Acceptance falls to 33 of 80: **a third of the minting budget spent on descriptions the
   system can no longer tell apart.**

Fixing one channel without the other is worse than fixing neither. **Caption and sampler
must change together.**

## Part 2 — `vtl_cm` was never noise, and it lifts the cell ceiling

3 axes × 5 bins is **125 total descriptions**, and this run sampled 80 of them. A catalogue
of hundreds is arithmetically impossible there — so the 3-axis win came with a worse
ceiling than the one it removed.

The fix was not more bins but **a fourth real axis**. `vtl_cm` was being dropped by `S4`
and `S6` as noise (gender separation d = +0.11 / −0.17 / +0.06). It was not noise —
**the estimator was broken**, and in exactly the way `S16` had just diagnosed for
`speaking_rate`.

`vocal_tract_length` used Fitch's formant dispersion, `c / (2 × mean spacing)`, which
averages the gaps F2−F1 and F3−F2 — running the estimate **through F2**. F2 is the
vowel-dependent formant, swinging 800–2200 Hz between front and back vowels, so its
within-speaker spread swamps its between-speaker spread. Averaging it in cancelled what F1
and F3 knew:

| corpus | F1 | F2 | F3 | dispersion | **F1+F3** |
|---|---|---|---|---|---|
| hi | −0.62 | **+0.08** | −0.60 | +0.11 | **+0.70** |
| bn | −0.99 | **−0.13** | −0.59 | −0.17 | **+1.03** |
| ta | −0.60 | **−0.10** | −0.41 | +0.06 | **+0.44** |
| en | −0.71 | **−0.23** | −0.34 | +0.11 | **+0.68** |

*(Gender effect size; males should read positive. English measured on 219 GLOBE_V2 clips.)*

The estimator is now the mean of the two single-formant uniform-tube estimates,
`½(c/4F₁ + 5c/4F₃)`. It beats the old one on **all four corpora**, English included — a
fix that traded Indic for English would not have been one.

### With `vtl_cm` restored as a fourth identity axis

| arm | cells | accepted | effective | of bound | adherence | vtl adherence |
|---|---|---|---|---|---|---|
| 5-axis, as shipped | 3,125 | 39 | 19.6 | 51% | 85.8% | — |
| **4-axis, identity only** | **625** | 47 | **22.6** | **59%** | 82.5% | 80.0% |
| 5-axis, dead gated | 3,125 | 33 | 18.4 | 48% | 95.4% | — |

**Describable space rises 125 → 625 while the diversity gain holds (22.6 vs 22.3).** The
cell ceiling that made the 3-axis result unusable is lifted 5× by adding an axis that
actually identifies people.

Adherence on the three common axes falls to 82.5%, slightly under the shipped 85.8% —
`vtl_cm` competes for retrieval weight and is harder to hit (80.0% on its own).

> **A scoring flaw caught on the way.** The first version scored each arm on *its own*
> axis set, so the 5-axis arm was graded on three axes (its cells carry no `vtl_cm`) while
> the 4-axis arm was graded on four. Adding a harder axis to one side lowers its mean
> whether or not anything got worse — that is how a 95.8% and an 81.9% ended up in the same
> column. All arms are now scored on the three axes every arm writes.

## No arm dominates, and that is the result

| | diversity | adherence | describable space |
|---|---|---|---|
| 5-axis shipped | worst | middle | best |
| 3-axis identity | good | **best** | **worst** |
| **4-axis identity** | **best** | worst | good |

The 4-axis arm is the one worth shipping — **+15% effective voices at roughly shipped
adherence, with 5× less wasted description space** — but it is a choice among trade-offs,
not a free win, and nothing here has been listened to.

## What still is not solved

**59% of the bound is not the 87% `S8`'s retrieval reaches** on the same corpus. Dead axes
and a broken VTL estimator were both real costs; together they were worth 8 points. They
were not the main gap.

Two directions remain, in order of cost:

1. **More bins per axis.** 4 axes × 7 bins is 2,401 cells against 625. The thing to check
   first is whether percentile edges stay meaningful — 141 speakers over 7 bins is ~20 per
   bin, which is thin.
2. **More identity axes.** `vtl_cm` was found by fixing a broken measurement rather than by
   inventing anything. `jitter` and `shimmer` are measured but never tested for the
   within/between property `S16` used, and one of them may be a fifth real axis hiding
   behind the same kind of bug.

## Not established

- **Geometry only. Nothing was rendered or heard.** Effective voices is a Vendi artefact,
  and `S11` already showed what happens when a geometric threshold meets a listener.
- **Adherence is measured via the nearest real clip's bins**, which is a proxy for where a
  minted vector "landed". It is consistent across arms, so the comparison holds, but the
  absolute percentages are softer than they look.
- **One corpus, one language, one seed, n=80 per arm.** The gap between arms 1 and 2 (19.6
  vs 22.3) is not large against that.
- **The 3-axis arm samples 64% of its own space**, so its acceptance rate of 50/80 is
  flattered by having nearly run out of distinct things to ask for. At larger n it would
  saturate hard, and the 5-axis arm would not.
- **The VTL absolute values are a uniform-tube idealisation, not anatomy.** The migration
  moved cached means from ~15.5 cm to ~19.2 cm; real adult tracts are 14–18 cm. Only the
  ordering is used, since the axis is percentile-binned, but the number should not be
  quoted as a measurement of anyone's vocal tract.
- **Cached measurements were migrated, not re-measured.** `scripts/migrate_vtl.py`
  recomputes `vtl_cm` from the cached formants, which is exact — `vocal_tract_length` is a
  pure function of them — but the formants themselves were not re-estimated. Originals kept
  as `*.pre-vtl-fix`.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S17-identity-axes/run_identity_axes.py
```

CPU only, a few minutes. Needs `S4`'s measurement cache and `S6`'s MioCodec embedding cache.
