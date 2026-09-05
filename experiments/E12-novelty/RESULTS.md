# E12 — novelty, the GMM defect, and why "more spread" was the wrong target

**Run:** 2026-09-05 · 300 vectors per setting · GLOBE_V2 / 1.7B mapper · **CPU only, nothing rendered**
**Question:** which `novelty` setting gives the largest catalog of genuinely distinct voices?

---

## Why

E11 found a catalog saturating fast at the default `novelty=0.0` — mean uniqueness fell 0.803 → 0.592 inside twenty voices. `novelty` is the one knob that plausibly changes that, and its endpoints were last measured by E1 **on the old, entangled acoustic axes**, before S2 run 4 decorrelated them. The knob was set by an invalidated number.

Minting a vector is linear algebra over cached embeddings — only *rendering* needs the GPU — so this ran on CPU while E11 held the card.

---

## Finding 1 — the generative branch could only ever make 64 voices

The first sweep said novelty made everything **monotonically worse**, ending at a nearest-neighbour distance of **exactly 0.000**. That means the median minted voice had an *exact duplicate*. 300 different descriptions cannot do that by chance; it is a defect, not a curve.

```python
self.gmm = GaussianMixture(..., random_state=0, ...).fit(self.P)
...
cand = self.gmm.sample(64)[0]     # identical 64 points on EVERY call
```

`GaussianMixture.sample()` calls `check_random_state(self.random_state)` each time, and `random_state` was a fixed int. **The generative branch had at most 64 possible outcomes for the entire life of a mapper**, however many descriptions it was given. Invisible at `novelty=0.0`, where that branch carries zero weight.

Fixed: `_sample_gmm(rng, n)` draws from the fitted mixture using the *caller's* rng via precomputed Cholesky factors — reproducible per seed, different across descriptions. Three regression tests pin it.

**E1's "novelty 1.0 collapsed to 0.097" was, on this evidence, reading this bug.** Treat E1's novelty endpoints as withdrawn until re-run.

---

## Finding 2 — spread was the wrong target, and it nearly fooled me

After the fix, spread rose monotonically with novelty and I wrote up `novelty=1.0` as the winner. **That conclusion was wrong**, and the thing that caught it was the component sweep (E12b), where a *single* Gaussian — the crudest possible prior — scored best on every diversity metric at once. A prior that cannot represent the data should not win. That is not a result, it is a warning.

The reason is that **"far from the data" and "novel" produce identical spread numbers.** A sampler placing points out in the tails gets high nearest-neighbour distance, high Vendi, and high distance-to-corpus, all while generating voices no real speaker resembles — precisely the sparse regions RESEARCH/12 measured at **+60% relative WER**.

So both sweeps gained a column that separates them:

> **`on_manifold_pct`** — where the median minted voice's log-likelihood falls within the distribution of *real speaker* log-likelihoods, under an **independent** reference mixture (24 components, fit once on the corpus, never on the samples — scoring samples under the model that generated them is circular).
>
> **50% = as typical as the median real speaker. 0% = less likely than every real speaker in the corpus.**

### Novelty, with the column that matters

Real speakers, same space and metric: **Vendi 0.397, nn median 0.660.**

| novelty | Vendi | nn median | vs real | below uniqueness floor | **on-manifold** |
|---|---|---|---|---|---|
| **0.00** ← *current default* | 0.102 | 0.413 | 0.63× | 17.3% | **0%** |
| 0.15 | 0.098 | 0.402 | 0.61× | 17.3% | 1% |
| 0.30 | 0.096 | 0.393 | 0.60× | 17.7% | 9% |
| 0.45 | 0.098 | 0.411 | 0.62× | 7.0% | 25% |
| **0.60** | 0.103 | 0.434 | 0.66× | 2.7% | **35%** ← best |
| 0.75 | 0.112 | 0.458 | 0.69× | **0.0%** | 28% |
| 0.90 | 0.121 | 0.482 | 0.73× | 0.0% | 14% |
| 1.00 | 0.126 | **0.496** | **0.75×** | 0.0% | 4% |

**On-manifold is unimodal and peaks at `novelty=0.60`, while spread rises monotonically.** Optimising spread alone would have selected 1.00, which sits at 4% — nearly as far off-manifold as the default.

**The most consequential line is the first one.** The current default, `novelty=0.0`, scores **0%**: the median minted voice is less likely than *every* one of 2,500 real speakers. Pure retrieval SLERP interpolates *between* anchors, and the midpoint between two real speakers is not generally a plausible speaker — which is exactly RESEARCH/12's finding about averaging in speaker space, showing up here as a property of the default setting.

So both ends fail, for opposite reasons: **0.0 lands in the gaps between real speakers, 1.0 lands outside them all.**

### Components (E12b), same lesson

At `novelty=1.0`, sweeping `gmm_components`:

| k | nn median | vs real | to-corpus | **on-manifold** |
|---|---|---|---|---|
| 1 | **0.543** | **0.82×** | **0.599** | **0%** |
| 3 | 0.517 | 0.78× | 0.573 | 2% |
| 5 ← current | 0.496 | 0.75× | 0.551 | 4% |
| 12 | 0.475 | 0.72× | 0.538 | 9% |
| 20 | 0.470 | 0.71× | 0.528 | 16% |
| 32 | 0.461 | 0.70× | 0.522 | 20% |
| 48 | 0.433 | 0.66× | 0.511 | **27%** |

**Perfectly anti-correlated.** Every diversity metric prefers k=1; the likelihood test says k=1 is the worst possible choice, generating below every real speaker. Note BIC prefers k=3 (290.3/sample) and also disagrees with on-manifold — BIC scores *fit to the anchors*, not plausibility of *samples*.

`to-corpus` did its narrower job: it falls monotonically as k rises, catching memorisation at high k. It simply cannot see the opposite failure, which is why the likelihood column was needed.

---

## What to change, and what to wait for

**Candidate, not a decision: `novelty≈0.60`, `gmm_components` higher than 5.** Both are geometry-only findings. Nothing here was rendered, so drift, consistency and adherence are unmeasured, and it remains possible that a more on-manifold vector renders no better.

The confirming run is **E11's `novelty=0.75` arm**, rendering against the 41-voice `novelty=0.0` control in `out/`. It is deliberately the *harder* case: 0.75 is further out than 0.60, so if drift and consistency hold there they hold at 0.60 too.

**Do not raise `gmm_components` on this evidence alone.** Even k=48 reaches only 27% on-manifold, so no setting tested makes the sampler typical — the ceiling may be the 50-dimensional PCA truncation or the mixture family itself, neither of which was swept.

## Not established

- One corpus (GLOBE_V2), English only, one `pca_dims=50`.
- `on_manifold_pct` depends on the reference mixture's own choice of 24 components. It was not sensitivity-tested; the *ordering* across settings is the claim, not the absolute percentages.
- No rendering, therefore no drift, consistency, adherence or listening evidence anywhere in this document.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E12-novelty/run_novelty_sweep.py --n 300
envs/qwen3/Scripts/python.exe experiments/E12-novelty/run_components_sweep.py --n 300
```

About 90 seconds each on CPU. Results in `out/results.json` and `out/results_components.json`.
