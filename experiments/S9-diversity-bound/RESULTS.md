# S9 — the diversity bound, and a claim it retracts

**Run:** 2026-09-05 · 432 real speakers, hi + bn + ta · MioCodec 128-d global embeddings
**Question:** how much diversity do the real speakers hold, before any method touches them?

---

## Why this had to be measured

`S7` minted an Indic catalog and got **~14 effective voices**. `S8` retrieved from a
library and got **~33**. Both numbers were reported against each other, and against
`E11`'s English ~20 — and none of them was reported against **the ceiling**.

Every method here draws from a space fitted on corpus speakers. Whatever diversity those
speakers hold is an upper bound: minting cannot invent diversity the space does not
contain, and retrieval certainly cannot. **Stating that bound is the difference between
"minting is weak" and "the space is small"** — and those two conclusions point at
completely different work.

## The bound

One point per speaker, so a talkative speaker is not counted twice. Normalised Vendi in
the same `SpeakerSpace` every other experiment uses:

| set | n | Vendi | **effective voices** |
|---|---|---|---|
| **real speakers, hi** | 141 | 0.270 | **38** |
| real speakers, bn | 138 | 0.266 | 37 |
| real speakers, ta | 153 | 0.272 | 42 |
| **real speakers, pooled hi+bn+ta** | **432** | 0.110 | **48** |
| | | | |
| S8 retrieved, hi | 82 | 0.405 | **33** |
| S7 minted, hi | 38 | 0.362 | **14** |
| S7 minted, pooled | 55 | 0.257 | **14** |

## Two things fall out, and one of them retracts a claim

### 1. Retrieval nearly saturates the space. Minting does not.

| | effective | share of the 38 available |
|---|---|---|
| retrieval | 33 | **87%** |
| minting | 14 | **37%** |

**Minting leaves nearly two thirds of the available diversity unused.** That is not a
property of the space — retrieval, drawing from the same 141 speakers through the same
embedding, reaches 87% of it.

### 2. ⚠️ Retracting "the ceiling is the space, not the method"

`S8`'s write-up claimed the ceiling belonged to the space rather than the method, on the
strength of text-retrieval scoring normalised Vendi **0.361** against minting's **0.362**.

**That inference was wrong, and the near-equality that motivated it was a coincidence.**
Normalised Vendi is a *fraction of the maximum diversity for a set of that size*. Two
sets with the same fraction and different n hold different numbers of voices — 0.361 on
95 retrieved is ~34 voices; 0.362 on 38 minted is ~14. Reading two normalised numbers as
if they were counts is the same error as `E11`'s "0.7 voices from 16".

The right comparison is effective count against the bound, and it says the opposite of
what was claimed: **the gap between minting and retrieval is real, large, and belongs to
the method.**

Filed as its own experiment rather than a correction inside `S8`, because the bound is
the more useful artefact: every future diversity number in this project should be
quoted against it.

## And pooling three languages barely helps

432 speakers hold **48** effective voices. 141 hold **38**.

Tripling the speaker count bought **26% more diversity**, not 200%. Hindi, Bengali and
Tamil speakers occupy heavily overlapping regions of MioCodec's identity space — which
is the expected result if that space encodes *voice* rather than *language*, and is
quiet evidence that it does.

It also explains `S7`'s pooled run directly: acceptance rose (47.5% → 68.8%, because
there were more distinct anchors to retrieve from) while effective voices stayed at
**exactly 14**. More data, same ceiling, because minting was never near the ceiling.

## The mechanism: blending contracts, and novelty contracts *harder*

Neither of the two obvious fixes works, and both were tested rather than argued:

| | effective voices |
|---|---|
| 141 speakers, novelty 0.00 | 14 |
| **432 speakers**, novelty 0.00 | 14 |
| 141 speakers, **novelty 0.35** | 12 |
| 141 speakers, **novelty 0.70** | 14 |

Pinned at 12–14 across a 3× corpus increase and the full novelty range. Retrieval
reaches 33 from the *same* anchors in the *same* space — so the loss is in what minting
**returns**, not in what it draws from.

The hypothesis had a shape statable in advance: minting blends anchors, and **any blend
of points on a shell lands inside it**. If that is the mechanism, minted vectors sit
nearer the centroid than real speakers do. Measured:

| | radius from corpus centroid | per-dim std |
|---|---|---|
| real speakers | 11.208 | — |
| minted, novelty 0.00 | 10.264 (**92%** of real) | 89% |
| minted, novelty 0.70 | 8.368 (**75%** of real) | 73% |

**Confirmed, with a twist that explains the whole table above: raising novelty makes
minted voices *more* central, not less.**

That is the opposite of what the knob is for. The generative branch samples from a GMM
fitted to the speakers and then keeps the draws nearest the retrieved region — and a
Gaussian's mass sits toward its mean, not on the shell where real speakers live. So
"novelty" trades interpolation between two real points for a draw from a distribution
that is itself more central. Both ends of the knob contract; the middle (0.35) is
measurably the worst of the three.

`mint` already calls `SpeakerSpace.decode(..., project_to_shell=True)`, which fixes the
radius in **raw** space. These numbers are in **working** space, and the contraction
survives the projection — so whatever that projection preserves, it is not the thing
Vendi is counting.

**This makes the fix a specific one rather than a search:** the diversity is lost at the
blend, so it has to be restored at the blend — not by feeding minting more anchors, and
not by turning novelty up.

## A fourth cause, found later by `S16`

The three candidates below were the ones visible at the time. `S16` added another, and it
is more promising than any of them.

**Two of the five caption axes carry no identity information.** `speaking_rate` and
`f0_cv` have within-speaker spread *larger* than between-speaker spread on real speakers —
1.32 and 1.04 — so two clips of one person differ on them more than two people do. `S12`
found the same on acted emotion; `S16` found it on plain read speech.

A description specifying rate and expressiveness is spending two of its five words on
things that distinguish nobody. That is a direct, mechanical explanation for a low
capacity ceiling, and the test is cheap: **re-fit the mapper on identity axes only
(`f0_mean`, `spectral_tilt`, `hnr_db`) and re-measure this bound.**

## What to do about it, in order of cost

1. **Restore the radius after blending.** Rescale a minted working-space vector to the
   real-speaker radius distribution before decoding. Directly targets the measured
   92%/75% contraction; a few lines, and it must be validated against the same bound —
   a change that raises Vendi without producing audibly distinct voices is not a fix.
2. ~~**Raise the uniqueness floor.**~~ **Done, and validated by a listener** (`S11`):
   0.30 admitted pairs heard as the same person half the time. Now 0.45 for MioCodec,
   costing 3% of effective diversity to remove 10 duplicate voices from 50.
3. ~~**Raise novelty.**~~ **Tested and ruled out** — it contracts harder (75% of real
   radius at 0.70). The knob does the opposite of its name in this space.
4. ~~**Widen the corpus.**~~ **Tested and ruled out** — +200% speakers, +0 effective
   voices.

## Not established

- **Vendi is one diversity metric.** It is the one `RESEARCH/06` specifies and the one
  E11 used, so the comparisons are internally consistent, but "effective voices" is a
  Vendi artefact, not a count of things a listener would distinguish.
- **The bound is the corpus's, not MioCodec's.** 432 read-speech speakers are not a
  sample of all human voices. A different corpus could hold more; nothing here says the
  128-d embedding itself tops out at 48.
- ~~**No listening.**~~ **`S11` ran it**, with controls 4/4. Two voices 0.30 apart *are*
  audibly the same person about half the time, which is what moved the floor to 0.45.
  Still a pilot: 8 pairs, 2 at the floor, one listener.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S9-diversity-bound/run_bound.py
```

CPU only, seconds. Needs the S4 measurement caches and S6's MioCodec embedding caches
for all three corpora.
