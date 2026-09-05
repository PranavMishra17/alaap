# S10 — the English pipeline has the same defect, worse

**Run:** 2026-09-05 · GLOBE_V2 · 2500 speakers · `Qwen3-TTS-12Hz-1.7B-Base`, 2048-d
**Questions:** does S9b's minting defect exist in English, and how does catalog *search* compare?

---

## The bound, and everything measured against it

`S9` established the discipline: a diversity number means nothing without the ceiling the
method was drawing from. For English that ceiling had never been measured either.

| | Vendi | effective voices | **share of the bound** |
|---|---|---|---|
| **REAL speakers (the bound)** | 0.081 | **203** | 100% |
| `E11`'s settings — `pca_dims 50, top_k 4` | 0.287 | 23 | **11%** |
| full basis, `top_k 4` | 0.443 | 35 | 17% |
| **full basis, `top_k 2`** | 0.518 | **41** | **20%** |
| pure retrieval, `top_k 1` | 0.559 | 45 | 22% |
| **library retrieval** over 184 reached voices | 0.525 | **97** | **48%** |

**The defect is real in English and it is worse than in Indic.** At E11's settings minting
reached **11%** of the diversity its own corpus held. Radial contraction measured **76%**
of real speaker radius, against Indic's 92%.

**The fix carries straight over: 23 → 41 effective voices, +81%.**

### What this does to E11's headline

`E11` reported "Vendi 0.482 → 20.2 effectively-distinct voices from 42 minted", checked
against `RESEARCH/06`'s anti-mode-collapse floor of 0.35 and called a pass. That check was
answering "is this mode-collapsed?" — a fair question, correctly answered.

It was not answering "how much of the available diversity did we reach?", and nothing in
that document could, because the bound was not measured. The answer is **11%**. E11's
number is not wrong; it is an understatement of what the English path can do, produced by
settings nobody had reason to question.

## Search: English is where retrieval really wins

Description → nearest library voice, exact bin match on all five axes:

| | Indic (S8, 141 voices) | **English (S10, 2500 voices)** |
|---|---|---|
| random control | 20.3% | 19.2% |
| text cosine | 39.0% | 51.4% |
| **hybrid (weighted bins)** | **65.6%** | **88.5%** |

**88.5% of descriptions retrieve a voice matching on every single axis.** That is not a
better method — it is the *same* method over a library 18× larger. Whatever description
you write, with 2500 speakers there is almost always someone who is genuinely that.

This is the clearest practical finding in the whole Indic/English comparison:

> **Retrieval quality is a function of library size, and it scales far more cheaply than
> minting quality does.**

Minting had to be fixed at the algorithm level to go from 11% to 20% of its bound.
Retrieval went from 65.6% to 88.5% by having more voices to choose from — no code change
at all.

### And retrieval still beats fixed minting by 2.4×

97 effective voices against minting's best 41. The same ordering as Indic (33 vs 22), and
a wider margin.

## What this means for the two paths

They are the same operation at different settings, which `S9b` established by making
`top_k=1` reachable:

```
top_k=1   retrieval  -- return the nearest real voice
top_k=2   minting    -- blend the two nearest
top_k=4   minting    -- blend the four nearest    <- every experiment before today
```

Each extra anchor blended costs diversity. So the product question is not "minting or
retrieval" but **how much blending to buy**, and what it buys is the ability to answer a
description no real voice matches — at a measured cost in catalog diversity.

With 2500 English voices reaching 88.5% exact adherence, that ability is worth
comparatively little. With 141 Indic voices at 65.6%, it is worth more. **The Indic path
needs minting more than the English path does, and is worse at it.**

## Not established

- **GLOBE_V2 is the anchor corpus, and `E14`'s roadmap wanted it replaced.** Its speakers
  are the ones whose diversity is being counted; a different English corpus gives a
  different bound.
- **`E11`'s numbers have NOT been re-run.** The settings changed today; E11's committed
  results were produced with `pca_dims=50, top_k=4` and stand as a record of that
  configuration. Re-running the whole English suite is a separate job.
- **The 203-voice bound is the corpus's, not Qwen3's.** 2500 read-speech speakers are not
  a sample of all human voices.
- **No audio was rendered or listened to anywhere in this experiment.** Every number is
  geometry and bin agreement. Whether 41 "effective voices" are 41 voices a listener would
  tell apart is untested, and is the check that matters most.
- **Adherence here is scored on generated captions.** `S8` measured the drop to sparse,
  user-style queries (65.6% → 53.8% on Indic); the equivalent was not run for English, so
  88.5% is a ceiling, not the operating point.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S10-english-bound/run_english_bound.py
```

CPU only, ~10 minutes (four full mapper fits over 2500 clips). Reads `S2`'s GLOBE_V2
corpus cache.
