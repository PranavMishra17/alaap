# S17 — describing what cannot identify was costing capacity

**Run:** 2026-09-06 · IndicVoices-R Hindi · 141 speakers · 80 mints per arm
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

## Result — dropping them improves both diversity and adherence

Bound from `S9`: **~38 effective voices** in these 141 real speakers.

| arm | cells | accepted | **effective** | of bound | **adherence** |
|---|---|---|---|---|---|
| 5-axis, as shipped | 3,125 | 39 | 19.6 | 51% | 85.8% |
| **3-axis, identity only** | **125** | 50 | **22.3** | **58%** | **95.8%** |
| 5-axis, dead axes gated | 3,125 | 33 | 18.4 | 48% | 95.4% |

*(Adherence is scored on the identity axes only, so an arm that never writes
`speaking_rate` is not marked down for omitting it.)*

**+14% effective voices and +10 points of adherence, from 25× fewer describable cells.**
The "fewer cells means fewer voices" worry was wrong — the cells were never the binding
constraint.

## The third arm is the one that explains the mechanism

Arm 3 keeps all five axes in the caption but forces the mapper's hybrid weights to zero on
the two dead ones. If the damage were purely in retrieval scoring, that should have matched
arm 2.

**It gets the adherence benefit (95.4%) but not the diversity benefit (18.4, worse than
baseline).**

So the dead axes hurt through **two separate channels**:

1. **Retrieval scoring** — they pull anchors toward matches on axes that identify nobody.
   Zeroing the weights fixes this, and adherence jumps 85.8% → 95.4%.
2. **The sampler** — with 5-axis cells and gated weights, two descriptions differing *only*
   in `speaking_rate` become indistinguishable to the mapper and collide, so they are
   rejected as duplicates. Acceptance falls to 33 of 80: **a third of the minting budget is
   spent on descriptions the system can no longer tell apart.**

Fixing one channel without the other makes things worse overall. **The caption and the cell
sampler have to change together.**

## ⚠️ The win is real and the ceiling it creates is worse

Three axes at five bins is **125 distinct descriptions in total.** This run sampled 80 of
them — 64% of the entire describable space — at n=80.

**A catalogue of a few hundred voices is arithmetically impossible on three 5-bin axes.**
The 5-axis arm had 3,125 cells and was nowhere near exhausting them; the 3-axis arm is
close to running out at eighty.

So this result should not be read as "use three axes". It should be read as:

> **The description was carrying dead weight, and removing it helps — but the axis set now
> needs *more resolution*, not fewer axes.**

The obvious next test is **more bins per axis**: 3 axes × 7 bins is 343 cells, × 9 bins is
729, without reintroducing anything that fails the identity test. Whether the binner's
percentile edges stay meaningful at 7 or 9 bins on 141 speakers is the thing to check
first — that is roughly 15–20 speakers per bin, which is thin.

A second option is finding **more identity axes**. `vtl_cm` is the obvious candidate and is
currently dropped on this corpus because the formant estimate does not separate gender
(d = +0.11). Fixing that measurement would add a genuine anatomical axis rather than a
behavioural one.

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
- **58% of the bound is still not 87%**, which is what `S8`'s retrieval reaches on the same
  corpus. Dead axes were a real cost but they were not the main gap.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S17-identity-axes/run_identity_axes.py
```

CPU only, a few minutes. Needs `S4`'s measurement cache and `S6`'s MioCodec embedding cache.
