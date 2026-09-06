# E14 — the catalog is limited by the description, not by the mapper

**Run:** 2026-09-05 · 400 held-out captions · 60,000 pairs · GLOBE_V2 / 1.7B · **CPU only, nothing rendered**
**Question:** when two descriptions differ, do the voices they mint differ correspondingly?

---

## Why this is the question under E11 and E12

E11 found the catalog saturating at forty voices. E12 found the sampling knobs cannot fix it without breaking drift. Both treated saturation as a *sampling* problem. Neither asked whether the mapper was transporting the description's information in the first place — which is the premise the whole two-tower design rests on.

**Measuring the description side.** Sentence-embedding distance between two captions is a bad independent variable: it measures the text encoder as much as the description, and two captions differing by one bin can sit arbitrarily close in MiniLM space. Difference is measured in **bin space** instead — L1 distance between bin indices — which is exactly the acoustic difference the caption was constructed to express, and is independent of any encoder.

**The reference, and what it is not.** The corpus is real (caption, voice) pairs, so the same correlation is computable on ground truth. That number is a **reference, not a ceiling**: a mapper can exceed it, because it constructs voices deterministically from captions while reality does not — two real speakers with identical bins are still different people. Reading it as an upper bound would be wrong, and an earlier version of this experiment did exactly that.

---

## Result

**Reference: ρ = 0.260** — among 400 held-out *real* speakers, bin distance vs voice-embedding distance across 60,000 pairs.

| novelty | ρ (bin distance vs minted voice distance) | vs reference | cos to the true voice | echoes its source anchor |
|---|---|---|---|---|
| **0.00** | **0.233** | **0.90×** | 0.114 | 0.0% |
| 0.45 | 0.225 | 0.86× | 0.114 | 0.0% |
| 0.75 | 0.164 | 0.63× | 0.092 | 0.0% |
| 1.00 | 0.091 | 0.35× | 0.066 | 0.0% |

### 1. The mapper is not the bottleneck

At `novelty=0.0` the mapper reproduces **90% of the caption→voice structure that exists between real speakers.** It is very nearly as description-consistent as reality is. And **`echoes source` is 0.0% at every setting** — a minted voice's nearest real speaker is essentially never the speaker its caption came from, so this is not retrieval wearing a mapper's clothes.

Whatever is limiting the catalog, it is not the mapper losing information.

### 2. The five-axis description is the bottleneck

**ρ = 0.260 is the reference itself.** Among real speakers, how far apart two voices are in embedding space is only weakly predicted by how far apart their captions are in bin space. This is a property of the *representation*, measured on ground truth with no mapper involved: pitch, tilt, HNR, expressiveness and rate simply do not determine a speaker.

That reframes E11's saturation. The catalog fills up because **five axes at five bins cannot specify more than a limited number of distinguishable voices**, and the mapper is already transporting 90% of what they do specify. Sampling knobs cannot manufacture distinctions the description never made.

*(A note on reading ρ = 0.260: it is a rank correlation over pairs, not a variance-explained figure. ρ² ≈ 0.07 is not the right summary either, since rank correlations over pairwise distances do not decompose that way. The claim is comparative — the mapper at 0.90× of the reference — not that "26% of identity is captioned".)*

### 3. Novelty destroys description fidelity

Transport falls 0.90× → 0.35× as novelty rises. This is the *same* trade E11 measured as drift, seen one step earlier in the pipeline: high-novelty vectors are further from what the description asked for, and then also render less faithfully. **Two independent measurements now say raising novelty costs description fidelity**, and only one of them needed the GPU.

`cos to true voice` (0.114 at `novelty=0`) is low and expected: many voices satisfy any given caption, which is the one-to-many problem `RESEARCH/11` describes. It falls with novelty too.

---

---

# E14b — VTL was built, and it pays; but the axes are badly weighted

E14 said the fix was to describe more. Vocal-tract length was built for that reason (`acoustics.formants` / `vocal_tract_length`) and validated against gender before use — females 15.53 cm, males 16.37 cm, all formant signs correct, r = −0.415 against F0 so it is not a restatement of pitch.

This is the test of whether it was worth it: the same reference correlation, on the **same 500 speakers and the same embeddings**, with and without the sixth axis.

> **The pairing was verified, not assumed.** Formants need the audio, which the cached corpus does not store. `stream_clips` is deterministic, so the first N clips of a fresh stream should be the cached rows — but on the first attempt they were not (**r = 0.011**), because S2 built the corpus with `min_dur=3.0, max_dur=12.0` and the `stream_clips` defaults are 2.0/15.0. The run aborted on its own alignment check rather than producing a plausible, meaningless number. With the bounds matched: **r = 1.0000, median relative difference 0.0000%.**

| description | ρ vs voice distance |
|---|---|
| `f0_mean` alone | **0.394** |
| `vtl_cm` alone | 0.188 |
| 5 axes | 0.281 |
| **6 axes (+ VTL)** | **0.323** |

**VTL adds +0.042, a +15.0% relative gain.** It carries identity information the other five axes do not, which is what it was built for.

### The unexpected result: `f0` alone beats all five axes together

**0.394 for pitch alone against 0.281 for the five-axis sum.** Adding four more measured axes to pitch *reduces* how well the description predicts voice distance.

That is not a bug, it is the distance metric. Bin-space L1 weights every axis equally, so four weakly-informative axes dilute the one strongly-informative one. The description is therefore not only **too narrow** — it is **badly weighted**, and the second problem is much cheaper to fix than the first.

This also puts the earlier numbers in perspective: the "reference" of 0.260 in E14 is a property of *equal-weight five-axis L1*, not of the information the captions contain. A weighted distance would raise it without measuring anything new.

---

# E14c — it replicates, but the ceiling was depressed by the corpus

E14 and E14b both ran on GLOBE_V2. "One corpus" was listed as unestablished, so this repeats the reference measurement on **LibriTTS-R train.clean.360** — a different corpus *and* a different encoder (0.6B, 1024-dim, against GLOBE's 1.7B/2048-dim). LibriTTS-R was cached at `per_speaker=2`, so it is deduplicated to one clip per speaker first; leaving the repeats in would have inflated ρ, since two clips of one speaker are close in both bin space and voice space.

| corpus / encoder | speakers | 5-axis ρ | `f0` alone ρ |
|---|---|---|---|
| GLOBE_V2 / 1.7B | 2,500 | 0.278 | **0.386** |
| **LibriTTS-R / 0.6B** | 883 | **0.374** | **0.616** |

**The qualitative finding replicates and is robust: `f0` alone beats the five-axis equal-weight sum on both corpora, across two different encoders.** The weighting defect is real and is not a GLOBE artefact.

**The quantitative ceiling does not replicate — it is far higher on the clean corpus**, and there is an already-documented reason. `data.py` records E4's measurement: ECAPA-TDNN scores **EER 20.0% on GLOBE_V2 against 2.46% on LibriTTS-R** through the identical code path — an 8× difference that isolates the corpus. GLOBE's speaker labels and/or its enhancement processing do not preserve identity reliably. Noisier speaker geometry means anything correlated *against* it reads lower.

So **E14's ρ = 0.260 should not be quoted as "the" limit of a five-axis description.** On a corpus whose speaker labels we trust, the same five axes reach 0.374 and pitch alone reaches 0.616. The description is still badly weighted — that is the robust part — but how much of a bottleneck it is depends on the corpus, and E14 measured it on the corpus the project already knows is the weaker one.

**A consequence worth acting on:** the mapper's anchors come from GLOBE_V2. If GLOBE's speaker geometry is the noisier one, then **the anchor corpus itself is a candidate for the next improvement**, alongside re-weighting. `libritts_r_360` is already wired in `data.py` and S2 already has a cached corpus for it.

## What this changes

**The roadmap item is a better description, not better sampling** — and E14b splits that into two jobs, the cheaper one first.

1. **Weight the axes.** `f0` alone (0.394) beats all five equally-weighted axes (0.281). Nothing needs to be measured to fix this — the retrieval distance and the caption's emphasis should reflect how much each axis actually carries. This is the largest available gain and it costs no new measurement.
2. **Put VTL into captions.** Built and validated; worth **+15% relative** on the six-axis reference. The `vtl_cm` bin axis exists (`very small-throated` … `very large-throated`); it is measured but **not yet written into `captions.ORDER`**, which is deliberate — adding an axis to captions invalidates every cached caption and mapper, so it should land together with the re-weighting and a full re-run.
3. **Consider moving the anchor corpus to LibriTTS-R** (E14c): the same five "
axes score 0.374 there against 0.278 on GLOBE_V2, and E4 already measured GLOBE's "
speaker labels as unreliable (EER 20.0% vs 2.46%).
4. **Then re-run E11** and see whether catalog capacity moved.
5. Only then revisit sampling.

**And leave `novelty` low.** Three experiments now agree: E14 (transport 0.90× → 0.35×), E11 (drift below floor 10% → 40%), and E1's original off-manifold warning. E12's geometric case for raising it is outvoted by everything that measured what the voice actually does.

## E14d — the drop came from `top_k`, not `pca_dims`

Re-running E14 at the fixed settings dropped transport from **0.90× to 0.62×**, which
looked like the `S9b`/`S10` fix costing description fidelity. But **two** things changed
in that commit — `pca_dims` and `top_k` (4 → 2) — and E14 varies neither, so its own
number could not attribute the drop.

### The obvious hypothesis, stated and then refuted

That raising `pca_dims` adds variation the caption never asked for. Sweeping it, same
mints, same metric:

| `pca_dims` | transport | effective voices |
|---|---|---|
| 10 | 1.33× | 8.4 |
| 25 | 1.43× | 17.9 |
| 50 | 1.46× | 29.3 |
| 100 | 1.46× | 43.0 |
| **150** | **1.46×** | **50.5** |

**Flat from 50 upward while diversity nearly doubles.** The `pca_dims` fix costs nothing;
the hypothesis is dead.

### `top_k` is the knob that trades

| retrieval | `top_k` | transport | effective voices |
|---|---|---|---|
| text | 1 | 0.65× | **57.5** |
| **text** | **2** | **0.94×** | **53.0** |
| text | 4 | **1.18×** | 47.5 |
| text | 6 | 1.30× | 43.0 |
| *hybrid* | *2* | *1.46×* | *50.5* |
| *hybrid* | *4* | *1.71×* | *42.6* |

Each step of `top_k` buys roughly **0.2× of transport for 5 effective voices.** That is a
real trade, it is monotonic, and it is what E14 was seeing.

> **A confound, named because it nearly went unnoticed.** Hybrid retrieval scores anchors
> partly *by* weighted bin distance — the very quantity `transport` measures — so its
> figures are inflated by construction. The **text** rows are the ones comparable to E14,
> which fits without `anchor_bins`. Reporting only the hybrid arm would have made the fix
> look free on both axes.

### Where that leaves `top_k = 2`

At `top_k=2` transport is **0.94×** — the mapper is very nearly *as* description-consistent
as reality is, neither more nor less. `top_k=4` sits at 1.18×, i.e. **more deterministic
than reality**, which this document's own note says is what a function must be, not
evidence of doing better.

So `top_k=2` costs 0.24× of transport against `top_k=4` and buys 5.5 effective voices,
landing at almost exactly the real-speaker reference. That is a defensible place to sit,
and it is now measured rather than assumed — `ADR-010` chose 2 on diversity alone.

**`top_k=1` is pure retrieval**, and its transport of 0.65× is the sanity check on the
metric: returning real speakers' own vectors should track the real-speaker reference, and
scoring *below* 1.0 there says the metric is not trivially satisfied.

## Not established

- **E14d's absolute transport figures are higher than E14's** because E14 fits on half
  the corpus and tests on the held-out half, while E14d fits and measures on all of it.
  Compare within a table, not across — the same warning this document already gives for
  E14 vs E14b.
- English only, one backend, one mapper configuration. **Two** corpora as of E14c, but the two differ in encoder as well, so corpus and encoder are not separated.
- E14b answers the "richer description" hypothesis for exactly one new axis (VTL, +15% relative). It says nothing about the others `RESEARCH/05` lists.
- Bin-space L1 treats every axis and every bin step as equal, and E14b shows that this **materially understates** the description — `f0` alone outscores the five-axis sum. Every ρ in this document is a property of that metric as much as of the captions.
- E14's reference (0.260, 400 clips) and E14b's 5-axis figure (0.281, 500 clips) differ because the two runs use different clip sets and duration bounds. Compare within a run, not across.
- No rendering. Transport is measured in embedding space; whether a listener hears the difference is unmeasured.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E14-transport/run_transport.py
```

~3 minutes on CPU. Mapper fitted on one half of the corpus, tested on held-out captions from the other.
