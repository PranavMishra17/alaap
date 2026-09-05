# E1 — Is the Qwen3-TTS speaker space navigable by synthesis?

> **Status:** ✅ RUN AND COMPLETE (after one discarded run — see §4) · **Date:** 2026-09-05
> **The decisive experiment.** Scope question **A2**; RESEARCH/12.
> **Model:** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` · **Data:** LibriTTS-R `train.clean.100`, 2,000 clips → **244 speakers**
> **Split:** 122 fit / 122 held out, disjoint speakers · **Space:** PCA-50 over centred+scaled
> **Artefacts:** `out/results.json`, `out/real_embeddings.npz`, `out/e1_navigability.png`

---

## 0. VERDICT

> ## The space IS navigable by synthesis. The two-tower design survives.
>
> Best generator **GMM k=5**: `nn_ratio = 0.89` against a pivot threshold of 1.5.
> **Even the worst generator (naive per-dim Gaussian) reaches 1.29× — still inside the threshold.**

**TacoSpawn's failure case does not reproduce here.** TacoSpawn measured a GMM over *d-vectors* sampling to `g2s 0.35` against `s2s 0.20` — a **1.75×** overshoot that put generated vectors nearly twice as far from any real speaker as real speakers sit from each other. On Qwen3-TTS the equivalent worst case is **1.29×**, and the best is **0.89×**.

That is almost certainly the mechanism RESEARCH/12 §9.2 predicted: Qwen3's speaker embedding has `enc_dim == hidden_size`, so it lives in **the backbone's own jointly-learned embedding space** — the "learned space" TacoSpawn found works — rather than a bolt-on discriminative ASV space, which is what failed for them.

---

## 1. Results

**Reference: a genuinely new real speaker sits `0.3818` from the fit set.** That is the natural spacing of the manifold, and the unit everything below is measured in.

| generator | nn_dist | **nn_ratio** | AUC | MMD | ‖z‖ ratio | reading |
|---|---|---|---|---|---|---|
| independent gauss | 0.4923 | **1.29×** | 0.613 | +0.0122 | 1.07 | overshoots — off-manifold |
| full-cov gauss | 0.4927 | **1.29×** | 0.567 | +0.0142 | 1.08 | overshoots — off-manifold |
| **GMM k=5** | 0.3398 | **0.89×** | 0.707 | +0.0139 | 1.08 | ⭐ **closest to natural spacing** |
| GMM k=10 | 0.2625 | 0.69× | 0.647 | +0.0125 | 1.06 | starting to memorise |
| GMM k=20 | 0.1886 | 0.49× | 0.557 | +0.0130 | 1.05 | memorising |
| SLERP (retrieval) | 0.0745 | **0.20×** | 0.556 | +0.0132 | 1.08 | heavily memorising |
| *resample real* | 0.0000 | 0.00× | 0.644 | +0.0132 | 1.07 | *(floor, by construction)* |

There is a clean monotone story: **generator capacity trades off against novelty.**

```
  naive Gaussian  ──────  too diffuse, lands off-manifold        1.29x
  GMM k=5         ──────  matches the manifold's natural spacing  0.89x   <-- sweet spot
  GMM k=10/20     ──────  tightens onto the training speakers     0.69x / 0.49x
  SLERP           ──────  sits essentially on top of its anchors  0.20x
  resample        ──────  memorisation floor                      0.00x
```

---

## 2. The finding that matters most for PHASE-02

> **SLERP interpolation between nearest-neighbour anchors produces voices that sit at `0.20×` the natural speaker spacing — i.e. very close to existing catalog entries.**

That is the exact strategy `PHASE-02` specifies for the retrieval mapper. It is **safe** (never off-manifold, never an artefact) but it is **low-novelty**: interpolants cluster tightly around the anchors they came from.

Two consequences:

1. **For a curated catalog, this is fine or even desirable** — you want new entries to sound plausible, not exotic.
2. **For "mint me a new voice from a description", this is a real limitation.** The retrieval mapper will keep returning near-duplicates of catalog voices. The catalog-uniqueness rule (min cosine distance) will start rejecting candidates quickly as the catalog grows.

**This sharpens the adherence↔diversity dial from RESEARCH/01 into something concrete and implementable:** interpolate toward anchors for high adherence, sample from a GMM (k≈5) for high novelty. The dial is literally a blend between two generators whose novelty is now measured — 0.20× vs 0.89×.

---

## 3. Honest caveats

- **The discriminator AUCs are noisy and not very informative.** With only 122 held-out real speakers, AUC ranges 0.556–0.707 with no clean ordering — `resample real` (which is real data by construction, so should score 0.5) came out at 0.644. Treat AUC here as "nothing is *grossly* detectable" rather than as a ranking. Needs more held-out speakers to be meaningful.
- **MMD is flat** (+0.0122 to +0.0142 across every generator, including resampled real data). At this sample size it does not discriminate. Not useful as run.
- **GMM k=5 has the *highest* AUC (0.707) while having the best nn_ratio.** These two metrics disagree. Given the AUC noise above I weight nn_ratio more heavily, but this deserves re-testing with more speakers.
- **This is embedding-space only.** It shows generated vectors are *in distribution*. It does **not** show they *render into good audio* — that is E2 (vocoder drift) and a listening test. A vector can be statistically in-distribution and still decode badly.
- **244 speakers is a small manifold sample.** LibriTTS-R `train.clean.100` has ~247 total, so this is nearly all of it. Scaling to `train.clean.360` would strengthen every number here.

---

## 4. Run 1 was discarded — the metric, not the generators, was broken

Recorded because the failure mode is subtle and easy to repeat.

Run 1 applied TacoSpawn's protocol literally: **median pairwise** cosine distance among real vs generated (`s2s`, `g2s`, `g2g`). It returned:

```
  independent gauss    g2s=1.0000 (0.97x)   [PASS]
  full-cov gauss       g2s=1.0003 (0.97x)   [PASS]
  GMM k=5              g2s=1.0332 (1.00x)   [PASS]
  ...every generator PASS, s2s reference = 1.0342
```

**Every generator passed, including the naive per-dim Gaussian that TacoSpawn specifically showed to be off-manifold.** That was the tell.

The cause: `s2s = 1.0342` means median cosine ≈ **−0.03**. After centring and PCA whitening, every pair of vectors is near-orthogonal, so **median pairwise cosine distance saturates at ~1.0 for any generator whatsoever.** The metric carried no information in this space. TacoSpawn's version works because it operates in a native d-vector space where median pairwise distance is genuinely informative.

Two further bugs in run 1's discriminator: **unbalanced classes** (122 real vs 500 generated) and **unstratified folds**, which produced sub-chance AUCs (0.107–0.302). A cross-validated AUC below 0.5 is a bug signature, not a finding.

**Fixes:** switched the primary metric to nearest-neighbour distance benchmarked against held-out real speakers; balanced and stratified the discriminator; added MMD as an independent check.

> **General lesson, same shape as E4's:** when every arm of an experiment passes — especially the arm that a published paper says should fail — suspect the measurement before believing the result.

---

## 5. What this changes

1. **A2 is answered: YES.** No pivot to Tier 2 is required on navigability grounds. `GRAND-PLAN` §6.1's pivot condition is not triggered.
2. **The mapper should target a GMM-like prior with few components (k≈5) in PCA-50 space** — not a full-covariance Gaussian (too diffuse, 1.29×) and not a high-k GMM (memorises).
3. **PHASE-02's SLERP retrieval is confirmed safe but low-novelty (0.20×).** Plan for the catalog to saturate, and expose the novelty trade as the CFG dial.
4. **RESEARCH/12's constraint 3 is empirically supported** — a jointly-learned embedding space really does behave better under a generative prior than a bolt-on ASV space. The scale of the difference (1.29× worst case here vs TacoSpawn's 1.75×) is the evidence.
5. **Still unproven: that these vectors render into good audio.** Everything here is embedding-space. E2 is next.

---

## 6. Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E1/run_e1.py --n-gen 500 --pca 50
```

Delete `out/real_embeddings.npz` to force re-extraction.
