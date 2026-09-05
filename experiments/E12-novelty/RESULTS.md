# E12 — novelty vs catalog capacity

**Run:** 2026-09-05 · 300 vectors at each of 8 novelty settings · GLOBE_V2 / 1.7B mapper · **CPU only, nothing rendered**
**Question:** which `novelty` setting gives the largest catalog of genuinely distinct voices?

---

## Why

E11 found a catalog saturating fast at the default `novelty=0.0` — mean uniqueness fell 0.803 → 0.592 inside the first twenty voices. `novelty` is the one knob that plausibly changes that, and its two endpoints were last measured by E1 **on the old, entangled acoustic axes**, before S2 run 4 decorrelated them. E1's own note records that novelty 1.0 "collapsed to 0.097" afterwards. So the knob was set by an invalidated number.

Minting a vector is linear algebra over cached embeddings — only *rendering* needs the GPU. So this sweep ran on CPU while E11 held the GPU.

## The result that mattered was a bug

The first run said novelty makes everything **monotonically worse**:

| novelty | 0.00 | 0.30 | 0.45 | 0.60 | 0.75 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|
| nn median | 0.413 | 0.383 | 0.316 | 0.175 | 0.064 | 0.009 | **0.000** |

**A nearest-neighbour distance of exactly 0.000 means the median minted voice had an exact duplicate.** That is not a tuning curve, it is a defect — 300 different descriptions cannot produce identical vectors by chance.

**Cause.** `sklearn`'s `GaussianMixture.sample()` calls `check_random_state(self.random_state)` on *every* call, and `random_state` was a fixed int:

```python
self.gmm = GaussianMixture(..., random_state=0, ...).fit(self.P)
...
cand = self.gmm.sample(64)[0]     # <- identical 64 points, every single call
```

So the generative branch had **at most 64 possible outcomes for the entire life of the mapper**, no matter how many descriptions it was given. Drawing 300 voices from 64 fixed points guarantees duplicates, and the median lands on one.

It was invisible at the default `novelty=0.0`, because there `P_out = retrieval` and the generative branch carries zero weight.

**Fix.** `_sample_gmm(rng, n)` draws from the fitted mixture using the *caller's* rng via precomputed Cholesky factors — so the draw depends on the per-description seed, as always intended, while staying reproducible for a given seed. Three regression tests pin it.

## The result, after the fix

Real speakers in the same space, same metric, same n: **Vendi 0.397, nn median 0.660.** That is the target.

| novelty | Vendi | vs real | nn median | vs real | below uniqueness floor | 
|---|---|---|---|---|---|
| 0.00 | 0.102 | 0.26× | 0.413 | 0.63× | **17.3%** |
| 0.15 | 0.098 | 0.25× | 0.402 | 0.61× | 17.3% |
| 0.30 | 0.096 | 0.24× | 0.393 | 0.60× | 17.7% |
| 0.45 | 0.098 | 0.25× | 0.411 | 0.62× | 7.0% |
| 0.60 | 0.103 | 0.26× | 0.434 | 0.66× | 2.7% |
| 0.75 | 0.112 | 0.28× | 0.458 | 0.69× | **0.0%** |
| 0.90 | 0.121 | 0.30× | 0.482 | 0.73× | 0.0% |
| **1.00** | **0.126** | **0.32×** | **0.496** | **0.75×** | **0.0%** |

**The trend reverses.** Novelty now helps monotonically, and from 0.75 upward it eliminates uniqueness-floor collisions entirely — from 17.3% of minted voices colliding at `novelty=0.0` to none.

Two things follow:

1. **The project's `novelty=0.0` default was chosen because of the bug.** E1's "novelty 1.0 collapsed" was, on this evidence, measuring the fixed-sample defect rather than a property of the speaker manifold. E1's endpoint numbers should be treated as withdrawn until re-run.
2. **Synthesised voices are still only 0.75× as far apart as real people**, at the best setting. The catalog's ceiling is set by the manifold and the mapper, not only by this knob.

### A metric that measured nothing

`anchor_sim_mean` reads a constant **0.930** at every novelty. That is correct-by-construction and therefore useless: `anchor_similarity` is the retrieval similarity of the top anchor, computed *before* novelty is applied, so it cannot vary with novelty. It was included to show what novelty "spends", and it does not measure that. **The cost of novelty is unmeasured here.** A real version would compare the minted vector against the description's *target bins* after rendering — which is adherence, and needs the GPU.

## What this does NOT establish

**Nothing was rendered.** There is no drift and no consistency number here, so this is geometry only. A setting that spreads vectors further apart can push them off the manifold where they stop rendering into speech that matches the description — which is exactly what E1's off-manifold warning was about, and the reason `mint()` refuses novelty above 1.0.

**So `novelty=0.75` is a candidate, not a decision.** The confirming run is E11's `novelty=0.75` arm, rendering and measuring drift/consistency/adherence against the 41-voice `novelty=0.0` control in `out/`.

Also unestablished: one corpus (GLOBE_V2), one mapper configuration (`pca_dims=50`, `gmm_components=5`), English only. `gmm_components` was never swept and is a plausible second lever.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E12-novelty/run_novelty_sweep.py --n 300
```

Runs in about 90 seconds on CPU. Results in `out/results.json`.
