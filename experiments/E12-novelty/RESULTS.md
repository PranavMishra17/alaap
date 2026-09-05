# E12 — novelty, prior size, and two metrics that were wrong before one was right

**Run:** 2026-09-05 · 300 vectors per setting · GLOBE_V2 / 1.7B · **CPU only, nothing rendered**
**Question:** which `novelty` and `gmm_components` give the largest catalog of *plausible* distinct voices?

> **This experiment reversed itself twice.** The reversals are the most useful thing in it, so they are kept in order rather than tidied away. If you only read one section, read *The metric that was broken*.

---

## Finding 1 — the generative branch could only ever make 64 voices *(stands)*

The first sweep said novelty made everything monotonically worse, ending at a nearest-neighbour distance of **exactly 0.000** between 300 minted voices. That means the median minted voice had an *exact duplicate*, which 300 different descriptions cannot produce by chance.

```python
self.gmm = GaussianMixture(..., random_state=0, ...).fit(self.P)
...
cand = self.gmm.sample(64)[0]     # identical 64 points on EVERY call
```

`GaussianMixture.sample()` re-derives its RNG from `self.random_state` on every call. **The generative branch had at most 64 possible outcomes for the life of a mapper**, however many descriptions it was given — invisible at the default `novelty=0.0`, where that branch has zero weight.

Fixed via `_sample_gmm(rng, n)`, drawing from the fitted mixture with the *caller's* rng through precomputed Cholesky factors. Three regression tests. **This finding is mechanical and stands.**

It also means **E1's "novelty 1.0 collapsed to 0.097" was reading this bug.** Treat E1's novelty endpoints as withdrawn.

---

## The metric that was broken *(read this one)*

After the fix, spread rose monotonically with novelty, and "use `novelty=1.0`" was nearly the conclusion. It is wrong, because **spread cannot distinguish "novel" from "implausible"** — a point far from every real speaker is also far from every other sample, so a degenerate sampler wins on spread. That is the sparse region `RESEARCH/12` measured at **+60% relative WER**.

So a plausibility column was added: log-likelihood of minted voices against real speakers, under a 24-component full-covariance GMM.

**It reported that a *single* Gaussian — the crudest prior available — was the worst possible choice (0%), and that `novelty=0.60` peaked at 35%.** A write-up saying exactly that was committed.

Then it got the control it should have had first:

| scored set | likelihood metric says |
|---|---|
| **real held-out speakers** | **0–1%** |
| Gaussian noise | 0% |

**The metric could not tell a real human voice from noise.** A full-covariance GMM over 50 dimensions fitted to ~1,250 points measures proximity to its own training set, not plausibility. Every number it produced was invalid, including the ones already written up — and worse, because the reference had been fitted on the *same* speakers the mapper's anchors came from, retrieval-heavy settings scored well for a reason that had nothing to do with plausibility.

Everything that metric justified was reverted, including a `typicality` selection mode added to `mint()` on its evidence.

### The replacement, and its control

`metrics.isolation_pct` — non-parametric, no density model, only a metric: the percentile of a sample's median k-NN radius within the reference's own k-NN radii. **Validated before use**, on a reference of held-out speakers the mapper never saw:

| scored set | isolation | required |
|---|---|---|
| **real held-out speakers** | **54%** | near 50 ✅ |
| Gaussian noise | 100% | near 100 ✅ |
| dimension-shuffled real speakers | 100% | near 100 ✅ |

The control is now a unit test, so a future change breaks there rather than inside an experiment.

**Reading it:** 50 = as typical as a median real speaker. **Above 50** = out in the tails. **Below 50** = crowded into denser regions than real speakers occupy — the opposite failure, and a real one.

---

## Finding 2 — the corrected results

Corpus split in half: mapper, prior and PCA space fitted on one half; plausibility measured against the other. Control: held-out real speakers **54%**, nn median **0.653**.

### Novelty (`gmm_components=5`)

| novelty | nn median | vs real | below floor | **isolation** |
|---|---|---|---|---|
| 0.00 ← default | 0.412 | 0.63× | 19.0% | 25% |
| 0.30 | 0.398 | 0.61× | 19.7% | 24% |
| 0.45 | 0.422 | 0.65× | 12.7% | 23% |
| 0.60 | 0.437 | 0.67× | 5.7% | 23% |
| 0.75 | 0.457 | 0.70× | 2.3% | 25% |
| 0.90 | 0.476 | 0.73× | 0.7% | 26% |
| 1.00 | 0.487 | 0.75× | **0.7%** | 28% |

**Novelty barely affects plausibility at all** — isolation is flat at 23–28% across the entire range. The earlier "peak at 0.60" was an artefact of the broken metric. What novelty *does* buy is real and worth having: **collisions under the uniqueness floor fall from 19.0% to 0.7%**, and spread rises from 0.63× to 0.75× of real speaker spacing.

### Prior size (`novelty=1.0`)

| k | nn median | vs real | below floor | **isolation** |
|---|---|---|---|---|
| **1** | **0.546** | **0.84×** | **0.0%** | **45%** |
| 3 | 0.500 | 0.77× | 0.0% | 29% |
| 5 ← current | 0.487 | 0.75× | 0.7% | 28% |
| 12 | 0.473 | 0.72× | 3.3% | 27% |
| 24 | 0.444 | 0.68× | 10.0% | 26% |
| 48 | 0.403 | 0.62× | 24.0% | 24% |

**`gmm_components` is the lever, not novelty — and k=1 wins on every column at once**, including the one that was supposed to catch it. At 45% isolation it is the only setting that comes near real speakers' 54%; everything else sits at 24–29%, i.e. **crowded into denser regions than real people occupy.**

That is the honest reading of the whole experiment: the sampler's problem was never that it wandered into the tails. It is that **it huddles in the middle of the population and avoids the edges** — and more mixture components make that worse, because samples concentrate at mixture modes.

Note this contradicts E1's warning that a full-covariance Gaussian "overshoots to 1.29× and lands off-manifold". Measured here it *under*-shoots at 0.84×. E1's numbers came from the same code path as the 64-sample bug and should be re-run before either is trusted.

---

## What to change

**Nothing yet.** Every number here is geometry. Drift, consistency and adherence are unmeasured, and the confirming run — E11's `novelty=0.75` arm rendering against the `novelty=0.0` control — has not finished.

Candidates, in order of evidence:

1. **`gmm_components=1`, `novelty≈1.0`** — best on spread, collisions and plausibility simultaneously. Needs rendering confirmation most of all, because it is the setting E1 explicitly warned about.
2. **Raise `novelty` from 0.0 regardless** — the collision reduction (19.0% → 0.7%) is large and the plausibility cost is nil.

## Re-run at the fixed settings (2026-09-05)

`S9b`/`S10` found `pca_dims=50` truncates the basis and zero-pads the discarded
components. Re-run at the full basis, same control, same metric:

| | old (`pca_dims 50`) | **fixed (full basis)** |
|---|---|---|
| novelty 0.00 — nn median | 0.63× real | **0.72×** |
| novelty 1.00 — nn median | 0.75× real | **0.93×** |
| isolation across novelty | flat 23–28% | **35% → 49%** |
| `k=1`, novelty 1.0 — isolation | 45% | **72%** |
| held-out real speakers (control) | 54% | 54% |

Full new tables:

```
CONTROL  held-out REAL speakers   isolation 54%   nn median 0.653

NOVELTY (gmm_components=5)
novelty  nn med  vs real  <floor  isolation
   0.00   0.467    0.72x    8.7%        35%
   0.45   0.528    0.81x    4.7%        32%
   1.00   0.610    0.93x    0.0%        49%

COMPONENTS (novelty=1.0)
      k  nn med  vs real  <floor  isolation
      1   0.672    1.03x    0.0%        72%
      3   0.638    0.98x    0.0%        58%
      5   0.610    0.93x    0.0%        49%
     24   0.523    0.80x    1.3%        39%
     48   0.509    0.78x   11.7%        37%
```

**Everything improved, and one thing improved past its own reference.**

Spread rose across the board — minted voices now reach 0.93× real speaker spacing at
novelty 1.0 where they reached 0.75× before, and `k=1` reaches **1.03×**, i.e. slightly
*wider* than real speakers sit from each other.

### ⚠️ The `k=1` isolation number now overshoots, and that is not straightforwardly good

The old write-up said *"k=1 wins on every column at once"* at 45% isolation, praising it as
**the only setting that comes near real speakers' 54%**. It now reads **72%**, against the
same 54% control.

`isolation_pct` is the percentile of a sample's median k-NN radius within the reference's
own radii. 54% means "as typical as a real speaker". **72% means these voices sit in
*sparser* regions than real people do** — more distinct, and correspondingly further from
the dense part of the manifold. The old conclusion was "k=1 is the only setting that gets
close to real"; the honest new one is "k=1 now passes real and keeps going", and whether
that is better or worse is **not a question this metric answers.**

`E1` warned specifically about overshooting past real speaker spacing, and `k=1` at 1.03×
is now in the region it warned about. It was 0.75× when that warning looked satisfied.

**So the recommendation below needs a listener, not a re-read.** `S11` established the
instrument: pairs at known distances with positive and negative controls. Nothing here
says whether a voice at 1.03× real spacing sounds like a person.

## Not established

- One corpus, English only, `pca_dims=50` never swept.
- `isolation_pct` depends on k=5; the *ordering* across settings is the claim, not the absolute percentages.
- No rendering anywhere in this document. No drift, consistency, adherence, or listening evidence.
- k=1 winning on every axis is the pattern that should trigger suspicion, and it has now survived one metric replacement. It has not survived a rendering test.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E12-novelty/run_sweeps.py
```

~4 minutes on CPU. Prints its own control first — **if held-out real speakers are not near 50%, stop and fix the metric before reading anything below it.**
