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

## What this changes

**The roadmap item is a better description, not better sampling** — and E14b splits that into two jobs, the cheaper one first.

1. **Weight the axes.** `f0` alone (0.394) beats all five equally-weighted axes (0.281). Nothing needs to be measured to fix this — the retrieval distance and the caption's emphasis should reflect how much each axis actually carries. This is the largest available gain and it costs no new measurement.
2. **Put VTL into captions.** Built and validated; worth **+15% relative** on the six-axis reference. The `vtl_cm` bin axis exists (`very small-throated` … `very large-throated`); it is measured but **not yet written into `captions.ORDER`**, which is deliberate — adding an axis to captions invalidates every cached caption and mapper, so it should land together with the re-weighting and a full re-run.
3. **Then re-run E11** and see whether catalog capacity moved.
4. Only then revisit sampling.

**And leave `novelty` low.** Three experiments now agree: E14 (transport 0.90× → 0.35×), E11 (drift below floor 10% → 40%), and E1's original off-manifold warning. E12's geometric case for raising it is outvoted by everything that measured what the voice actually does.

## Not established

- One corpus, English, one backend, one mapper configuration.
- E14b answers the "richer description" hypothesis for exactly one new axis (VTL, +15% relative). It says nothing about the others `RESEARCH/05` lists.
- Bin-space L1 treats every axis and every bin step as equal, and E14b shows that this **materially understates** the description — `f0` alone outscores the five-axis sum. Every ρ in this document is a property of that metric as much as of the captions.
- E14's reference (0.260, 400 clips) and E14b's 5-axis figure (0.281, 500 clips) differ because the two runs use different clip sets and duration bounds. Compare within a run, not across.
- No rendering. Transport is measured in embedding space; whether a listener hears the difference is unmeasured.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E14-transport/run_transport.py
```

~3 minutes on CPU. Mapper fitted on one half of the corpus, tested on held-out captions from the other.
