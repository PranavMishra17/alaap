# PHASE 04 — Dataset v2 & the Measure-First Annotation Pipeline

> **Goal:** grounded, *reversible* captions; VoicePersona rebuilt under a licence that actually holds; corpus expanded beyond LibriTTS-P.
> **GPU:** local · **Depends on:** S0 (corpus GO-list, E7), S2
> **Evidence:** [`05`](../05-datasets-and-annotation.md) · [`08`](../08-licensing-propagation.md) · [`09`](../09-safety-and-watermarking.md)

---

## 0. Exit criteria

| # | Criterion | Threshold |
|---|---|---|
| **X4.1** | A generated voice's attributes are **verifiable by re-measurement** against its description's target bins | per-attribute error reported; the loop closes |
| **X4.2** | VoicePersona v2 published under a licence whose **upstream chain is traced and clean** | every source has a GO verdict with a licence URL |
| **X4.3** | Retrained mapper does not regress | S2/S3 metrics re-run on the expanded corpus |
| **X4.4** | Per-demographic slices reported and the skew's *cause* addressed | I10 |

---

## 1. The recipe, corrected

Scope §8.3 assumes Data-Speech extracts nine attribute families. **It computes 9 columns from 5 tools, and six of the attributes the brief lists are not among them.**

### 1.1 What Data-Speech actually computes

| Attribute | Tool | Licence |
|---|---|---|
| F0 mean / std | **`penn`** (FCNF0++), 16 kHz, hop 10 ms, 30–1000 Hz | MIT |
| SNR, C50, VAD duration | **Brouhaha** (`ylacombe/brouhaha-best`) | MIT |
| Speaking rate | **`g2p`** — and it is **IPA-characters/sec, not phones/sec** | MIT |
| STOI / SI-SDR / PESQ (optional) | torchaudio **SQUIM** | weights CC-BY-4.0 |

Captions: **Mistral-7B-Instruct-v0.2**, temp 0.6, 256 tokens, via four prompts extracted verbatim from `run_prompt_creation.py`.

**Binning:** **equal-width, not percentile.** Pitch is binned **per-speaker, per-gender**; everything else per-utterance.

### 1.2 What you must build yourself

Scope §8.3 step 1 lists these and Data-Speech provides none of them:

**jitter · shimmer · HNR · spectral tilt · formants F1–F3 · vocal-tract length**

Three licence traps in the obvious tooling:
- **`praat-parselmouth` is GPLv3** → dev-tool only, never linked into the served product
- **`audeering` gender/age model is CC-BY-NC-SA** → an Apache-2.0 replacement is identified in [`06`](../06-evaluation-harness.md); a linear probe on WavLM embeddings is the permissive swap-in
- **VoxSim has no licence at all**

**Reliability caveat to test, not assume:** jitter/shimmer/HNR are validated on *sustained vowels from real speakers*. Whether they transfer to short synthetic utterances is unestablished. Measure agreement against a held-out real set before trusting them as caption inputs.

---

## 2. The reversibility property — why this phase exists

Grounded captions make evaluation *cheap and objective*: generate a voice from a description, re-measure the audio, compare to the description's target bins. That only works if the caption was produced **from measurements** rather than from an audio-LM's impression.

```
audio → measure → bin → LLM writes prose from the bins → caption
                            ↑                                │
                            └────── verify by re-measuring ──┘
```

Keep an audio-LM impressionistic pass **only** as a supplementary layer for genuinely unmeasurable qualities (*menacing, warm, world-weary*), validated against a fixed tag taxonomy rather than free-form.

---

## 3. VoicePersona v2 — the rebuild

### 3.1 The v1 verdict, honestly

**The CC0 declaration does not hold.** ~79% of rows are defective:

| Source | Share | Problem |
|---|---|---|
| LAION's Got Talent | **52.6%** | **GPT-4o Audio output using OpenAI's eleven proprietary voices** — 483 tarballs named `alloy`/`ash`/`ballad`/…/`verse` |
| AnimeVox | 13.3% | **CC-BY-NC-4.0**, ripped from "official English-dubbed versions of popular anime series" |
| AniSpeech | 13.3% | Literal MIT *software* text applied to 18.8 GB of anime voices; no stated rights basis |
| GLOBE_V2 | 20.9% | ✅ clean |

The HF field says `cc` — a meaningless category tag, not CC0. **CC0 §4(c) expressly disclaims clearing others' rights.** And **HuggingFace disabled `ESpeech/ESpeech-igm` in May 2026 on a voice actor's complaint about exactly this pattern** — a permissive licence asserted over scraped voice work.

Also wrong on the card: **14,327 rows** (not 15,082), **English-only** (not 8+ languages), **307-char** descriptions (not ~500).

### 3.2 What v2 is

- **Drop LAION, AnimeVox, AniSpeech.** Keep GLOBE_V2.
- **Rebuild from the S0 GO-list**: GLOBE (CC0, 23,519 speakers), LibriTTS-R, VCTK, MLS, Common Voice CC0 subsets.
- **Re-caption with the measure-first pipeline**, not Qwen2-Audio direct description.
- **Ship as CC-BY-4.0, not CC0**, with a NOTICE file crediting every upstream corpus.
- Alternative if audio redistribution is undesirable: **ship annotations only**, keyed to public corpus IDs.

### 3.3 The skew is a prompt artefact, not a measurement

Scope §8.3 and §15.2 treat 62.6% female / 76.1% twenties as corpus properties to rebalance. **They are Qwen2-Audio *label distributions*.** The caption prompt hardcoded `GENDER: [male/female]`, capped age at `fifties+`, and never permitted "neutral."

**So: fix the prompt first, then re-measure, then decide whether rebalancing is even needed.** Rebalancing an artefact would bias the corpus in a new direction while appearing to correct one.

**What is lost:** dropping AnimeVox and AniSpeech removes exactly the character/anime voice coverage scope §8.3 called "real, non-replicable." That is a genuine loss. The honest replacements are MOSS-VoiceGenerator's cinematic training distribution (as a *renderer*, not a corpus) and, eventually, the deferred non-human workstream.

---

## 4. Expanding the Indic corpus

**RASMALAI was never released** — zero HF results; the 13,000-hr description corpus is unobtainable. **But the recipe is fully published and every input is CC-BY-4.0/MIT** (IndicVoices-R, Rasa, IndicTrans2).

**Reconstructing RASMALAI is the highest-leverage dataset action available**, and it is the prerequisite for option C in the Indic gate ([`PHASE-05`](PHASE-05-indic-gate.md)).

---

## 5. Corpus sizing

E7 answers this empirically. Until it reports, the working rule is **speaker count × session diversity**, not speaker count alone: at fixed 100 h, 60× more speakers halves ECAPA EER — **but one session each is worse than 100 speakers with many sessions.**

Practical consequence for GLOBE (23,519 speakers): do not assume more speakers is monotonically better if each contributes a single session.

---

*Phase spec v2 · 2026-09-02 · prev: [`PHASE-03`](PHASE-03-generative-mapper.md) · next: [`PHASE-05`](PHASE-05-indic-gate.md)*
