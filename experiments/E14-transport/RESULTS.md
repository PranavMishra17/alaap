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

## What this changes

**The roadmap item is more descriptive axes, not better sampling.**

`RESEARCH/05` already flagged that six of the attributes the scope assumed were unavailable and would have to be built — **formants and vocal-tract length among them**. VTL is one of the strongest correlates of perceived speaker identity and is currently not measured, not binned, and not in any caption. On this evidence, adding it would raise the reference ρ, which is the only thing that raises the catalog's ceiling.

Concretely, in priority order:
1. **Add formant / VTL axes** to `acoustics.py`, bin them, put them in captions.
2. Re-run E14. If the reference ρ rises, catalog capacity rises with it.
3. Only then revisit sampling.

**And leave `novelty` low.** Three experiments now agree: E14 (transport 0.90× → 0.35×), E11 (drift below floor 10% → 40%), and E1's original off-manifold warning. E12's geometric case for raising it is outvoted by everything that measured what the voice actually does.

## Not established

- One corpus, English, one backend, one mapper configuration.
- The reference is computed over the same five axes the mapper is given. It measures the limit of *this* description scheme, and says nothing about how much a richer one would recover — that is the hypothesis in "what this changes", not a result.
- Bin-space L1 treats all five axes as equally important and each bin step as equal. A perceptually-weighted distance would likely differ.
- No rendering. Transport is measured in embedding space; whether a listener hears the difference is unmeasured.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E14-transport/run_transport.py
```

~3 minutes on CPU. Mapper fitted on one half of the corpus, tested on held-out captions from the other.
