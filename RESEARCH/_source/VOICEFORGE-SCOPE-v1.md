# VoiceForge — Scope & Research Brief

**Status:** Scoping document. No code. This is the input to a later deep-research pass.
**Date:** 2026-09-02
**Author:** scoping session with Claude
**Prior art in this repo:** a CosyVoice wrapper (`cosyvoice/`, `wrapper/`, `main.py`). Treated here as *discarded* per instruction — see §2 for why it failed, because that diagnosis constrains the new design.

---

## 1. What we are building

**One sentence:** a system that turns a natural-language description of a character into a persistent, reusable *voice identity*, and then renders arbitrary dialogue in that voice — in English and Indian languages — at a quality good enough for games and dramatic content.

The reference diagram decomposes into three separable components:

```
  TEXTUAL              CUSTOM TEXT→VOICE-IDENTITY          VOICE            CUSTOM TTS           AUDIO
  DESCRIPTION   ──────►        EMBEDDER              ──────► IDENTITY ──────►  MODEL     ──────► MP3/WAV
                                                                                  ▲
  "His voice carries the weight                                                   │
   of mountains—gravelly, calm,                                              DIALOGUE
   deliberate..."                                                     "It's cold today."
```

| # | Component | What it is | Build or buy |
|---|-----------|-----------|--------------|
| **A** | Description encoder | text → voice-identity representation | **BUILD.** This is the project. |
| **B** | Voice identity | a persistent, storable, re-summonable handle for "this character's voice" | **BUILD.** This is the actual differentiator (see §4.3). |
| **C** | TTS renderer | (identity + dialogue text) → waveform | **BUY (frozen, off-the-shelf).** Do not train this. |

**The critical design property is that A and C are decoupled through B.** C is a frozen, swappable backend. If a better TTS ships in six months, we swap C and keep A and B. This is also what makes the Indic track possible as a *parallel backend* rather than a rewrite.

### 1.1 Non-goals (explicit)

- Not building a TTS acoustic model or vocoder from scratch.
- Not doing real-time / streaming latency work in phase 1.
- Not building the web app before the ML pipeline runs locally. **ML first.**
- Not cloning real, identifiable people from their audio. (Ethics + legal; see §15.)

---

## 2. Why the previous attempt failed (root-cause, not blame)

This matters because the same trap is easy to walk into twice.

The repo is built on **CosyVoice**. CosyVoice conditions on a **reference audio prompt consumed in-context** — the speaker information lives in a *sequence of acoustic/semantic tokens derived from a reference waveform*, not in a single fixed-length vector.

Three consequences that match the reported symptoms exactly:

1. **There is no vector to synthesize into.** The architecture in the diagram requires component A to *output a voice identity*. If the TTS only accepts reference audio, there is nothing for A to produce. The two-tower split is structurally impossible on that backend without first solving "generate a plausible reference waveform," which is a harder problem than the one being solved.
2. **"Customization" had no surface.** Attempting to steer voice via prompt text on a model whose speaker channel is audio-only produces weak, inconsistent, or ignored control — which reads as "the model is broken."
3. **Language settings were opaque.** CosyVoice's instruction/control path was Chinese-centric in its early releases; non-Chinese instruction following was materially weaker and under-documented. A "simple language setting" that silently degrades looks like a bug in your code when it is a capability gap in the model.

**Lesson encoded into this scope:** *the choice of TTS backend is not an implementation detail — it determines whether the architecture is expressible at all.* The single most important selection criterion for component C is **how it accepts speaker conditioning** (§6).

---

## 3. The field as of September 2026

The good news, and it is substantial: **this is now a named research subfield.** It is called **Text-to-Voice (TTV)** or **"voice design."** The problem is well-posed, benchmarked, and partially solved in the open.

**Timeline of directly relevant work:**

| Year | Work | Contribution |
|------|------|-------------|
| 2023 | **PromptSpeaker** | First clean formulation: prompt encoder → prior distribution → Glow → speaker vector → zero-shot VITS. Establishes the two-tower split. |
| 2024 | **Parler-TTS** (HF, after Lyth & King) | Description-conditioned TTS via cross-attention on a description encoder. 45k hrs. Fully open. |
| 2024 | **LibriTTS-P** | LibriTTS-R + speaker-identity and speaking-style prompts. First serious paired (description, voice) corpus. |
| 2024 | **IndicVoices-R** | 1,704 hrs, 10,496 speakers, 22 Indian languages. Unlocks Indic TTS scale. |
| 2024-12 | **Indic Parler-TTS** (AI4Bharat + HF) | **First open description-guided TTS for Indian languages.** 21 languages. Apache-2.0. |
| 2025 | **ParaSpeechCaps** | 2,769 hrs style-captioned speech, 59 style tags incl. abstract ones (*guttural, nasal, pained*). The scaling recipe for caption data. |
| 2025 | **InstructTTSEval** | Benchmark for natural-language instruction following in TTS. Gives us a scoring method. |
| 2025 | **IndexTTS-2** | Disentangles emotion from speaker identity — separate timbre prompt and emotion prompt. |
| 2026-01 | **Qwen3-TTS** | Apache-2.0. 0.6B/1.7B. Native **voice design from description** as a first-class mode. |
| 2026-04 | **VoxCPM2** | Apache-2.0. 2B, 30 languages, 48kHz. Native voice design + controllable cloning. |
| 2026 | **VoiceDesigner**, **VoiceSculptor**, **MOSS-VoiceGenerator**, **HiStyle** | Dedicated TTV systems; VoiceDesigner explicitly targets *fictional* voices via a DSP+generative augmentation pipeline. |

**The strategic consequence:** description → voice is no longer the hard part. A working demo of the *whole diagram* is now days of integration work, not months of research, because Qwen3-TTS and VoxCPM2 do it natively.

**Therefore the project's centre of gravity must move to the part that is still unsolved: identity persistence, catalogability, and controllable re-summoning (§4.3).**

---

## 4. Core ML concepts — is this conceptually possible?

**Verdict: yes, unambiguously.** It is a solved-in-principle problem with three specific difficulties that must be designed for rather than discovered.

### 4.1 The two-tower formulation

Learn a mapping `f: description_text → speaker_representation` such that a **frozen** TTS `G(speaker_representation, dialogue_text) → audio` produces a voice a human judges consistent with the description.

Training data is `(description, audio)` pairs. Get the target by running a frozen speaker encoder `E(audio) → z` over a speech corpus, and train `f` to hit `z`. Everything reduces to: **can we regress from text into the speaker-embedding manifold?**

### 4.2 Difficulty #1 — the one-to-many problem (THE central issue)

> "A gravelly old man's voice" describes **millions** of valid voices. It is one point in text space and a whole *region* in voice space.

This is the failure mode that kills naive implementations, and it fails *quietly*:

- Train `f` with **MSE/L2 loss** → the optimal prediction for a one-to-many mapping is the **conditional mean** of the region. The mean of many voices is a bland, averaged, characterless voice. Every description converges toward the same mushy centroid. Output sounds "fine but generic," and no amount of more data fixes it because the loss function is the bug.

**Mitigations, in order of preference:**

1. **Generative head.** Model `p(z | description)` rather than `E[z | description]`. Flow matching, a diffusion head, or a normalizing flow (PromptSpeaker used Glow for exactly this reason). Sample from the distribution → different valid voices per draw, all matching the description. **This is the correct answer.**
2. **Contrastive / retrieval.** Train a CLIP-style joint space between description text and speaker embeddings. Retrieve top-k real speakers, then perturb or interpolate. Cheap, extremely robust, gives immediate results, and doubles as the catalog search index. **Best first implementation.**
3. **Classifier-free guidance** on the generative head to trade diversity against description adherence at inference time — this becomes a user-facing "how literally should I take your description" dial.

**Never ship a plain-MSE mapper.** Note it explicitly in the plan.

### 4.3 Difficulty #2 — identity persistence (the real differentiator)

A game has an NPC with 400 lines recorded across 6 months of development. Every one must sound like the same character. Off-the-shelf voice design does **not** guarantee this: the same description prompted twice yields two different voices.

**Locked decision:** voice identity is a **fixed embedding vector**, persisted in a database, generated once and reused forever. *(Provisional — to be confirmed in deep research; see §17-A1.)*

This is precisely why component C's conditioning mechanism is the binding constraint. Three fallback tiers, weakest binding first:

| Tier | Mechanism | Reproducible? | Works on |
|------|-----------|--------------|----------|
| 1 (target) | Persist the **speaker embedding vector** | Exactly, by construction | Zonos, XTTS-v2, StyleTTS2, YourTTS |
| 2 (fallback) | Persist a **generated seed waveform** — mint a 10s clip once, then clone from that clip forever | Exactly, but storage is audio not a vector; no interpolation | Any zero-shot TTS: VoxCPM2, CosyVoice, F5, IndicF5 |
| 3 (weak) | Persist **description + RNG seed** | Only within one model version; breaks on any upgrade | VoxCPM2 (`seed=` param), Qwen3-TTS |

Tier 2 is a genuinely acceptable engineering answer and unblocks work on backends that expose no vector. Design the identity store so tiers 1 and 2 are both representable from day one.

### 4.4 Difficulty #3 — timbre vs. performance

Two orthogonal axes are routinely conflated:

- **Timbre / identity** — *who* is speaking. Physiology: pitch range, formants, vocal-tract length, breathiness, roughness. Stable across every line.
- **Style / performance** — *how* they say this line. Emotion, pace, emphasis, whisper vs. shout. Varies per line, and is the whole point of "dramatic content."

A character voice = **stable timbre + per-line performance direction**. Both must be addressable independently or the system is useless for drama. IndexTTS-2's explicit timbre/emotion disentanglement is the reference design here.

**Design implication:** the identity record stores timbre. The per-line API takes an optional performance instruction. Do not bake emotion into the identity vector.

### 4.5 Difficulty #4 — cross-lingual timbre transfer

Making one voice identity sound like the same person in Hindi and English is a genuinely hard open problem (accent leakage, phoneme coverage, speaker-embedding spaces that entangle language with identity).

**Locked decision: sidestep it.** A character gets **one voice profile per language**, linked by a shared `character_id`. Merging into a single cross-lingual identity is a later research question, not a phase-1 requirement.

This is the right call: it lets us pick the *best model per language family* instead of one compromise model, and it removes the hardest unsolved problem from the critical path.

---

## 5. Locked decisions

| Decision | Choice | Note |
|----------|--------|------|
| Voice identity representation | **Fixed embedding vector** | Provisional; confirm in deep research. Tier-2 seed-clip fallback designed in. |
| Licensing posture | **Research now, commercial-safe path preserved** | Prototype on the best model; document an Apache-2.0 swap-in for every non-commercial component. |
| Local hardware | **8–12 GB consumer GPU** | Load-bearing. See §9. |
| Cross-lingual | **One profile per language per character**, merge later | |
| Product shape | **Curated catalog + on-demand generation** | Catalog doubles as the retrieval index (§4.2 option 2). |
| First milestone | **Zero-training baseline first** | Wire the pipeline + build eval harness before training anything. |
| Voice range | **Human + heavy stylization** (aged, raspy, whispered, breathy, menacing, theatrical). Not non-human. | Non-human (goblin/dragon/robot) is a deliberate later workstream — no public dataset covers it; VoiceDesigner-2026 shows it needs a DSP augmentation pipeline. |
| Dataset | **Hybrid** — public corpora for core mapping + rebuilt VoicePersona for character coverage | See §8. |

---

## 6. TTS backend landscape (component C)

Selection criterion #1 is **how speaker conditioning is accepted**, because that determines whether the two-tower split is even expressible (§2).

| Model | Params | License | Conditioning mechanism | Exposes a speaker **vector**? | Indic? | Verdict |
|-------|--------|---------|------------------------|-------------------------------|--------|---------|
| **Zonos-v0.1** (Zyphra) | 1.6B | Apache-2.0 ⚠️*verify* | `make_speaker_embedding(wav, sr)` → vector, injected via `make_cond_dict` alongside explicit pitch/rate/quality/emotion dials | ✅ **YES — 128-d** (ResNet293-SimAM-ASP from VoxBlink2, 256→128 via LDA). Speaker encoder shipped as a **separate** checkpoint. | ❌ | ⭐ **Primary target for the two-tower / Tier-1 identity design.** The one model that makes Option A directly implementable. |
| **VoxCPM2** (OpenBMB) | 2B (also 0.6B / 0.5B) | Apache-2.0 | Three modes: **voice design from description text**, controllable cloning from ref wav, ultimate cloning (wav+transcript). `generate(text, reference_wav_path, cfg_value, seed)` | ❌ (ref audio + inline description) | 30 langs — ⚠️*verify Indic coverage* | ⭐ **Primary zero-training baseline.** 2B ≈ 8GB VRAM — fits the 12GB box; 0.6B variant ≈ 6GB as a safety margin. |
| **Qwen3-TTS** | 0.6B / 1.7B | Apache-2.0 | Learnable speaker encoder jointly trained with backbone; voice-design fine-tune; "thinking pattern" for complex descriptions | Internal, ⚠️*verify if externally addressable* | ❌ **No Hindi / Indian languages** | Strong baseline + quality reference. Rules itself out of the Indic track. |
| **Indic Parler-TTS** (AI4Bharat) | 0.9B | **Apache-2.0** | Description cross-attention (Parler architecture): `generate(input_ids=<description>, prompt_input_ids=<transcript>)` | ❌ (cross-attention, no vector) | ✅ **21 languages, 69 named voices** | ⭐ **The Indic track.** Commercially clean, description-native, already exactly our architecture. |
| **IndicF5** (AI4Bharat) | — | **CC-BY-NC** | Reference audio (F5/flow-matching) | ❌ | ✅ 11 languages, 1,417 hrs | Best Indic *quality*, but non-commercial. Research-only lane; not the commercial path. |
| **IndexTTS-2** | — | Restrictive / non-commercial | **Separate timbre prompt + emotion prompt.** Qwen3-finetuned soft instruction mechanism. Duration control. | ❌ | ❌ | Reference design for §4.4 disentanglement. Study it; don't ship it. |
| **Chatterbox** (Resemble) | — | **MIT** | Ref audio + emotion-exaggeration control | ❌ | ⚠️*verify* | Cleanest license. Fallback renderer. |
| **XTTS-v2** (Coqui) | — | **CPML (non-commercial)** | `gpt_cond_latent` + `speaker_embedding` | ✅ yes | Hindi ✅ | Vector-addressable and Indic-capable, but the license blocks the commercial path. Research lane only. |
| **F5-TTS** | — | **CC-BY-NC** (weights) | Reference audio | ❌ | via community ckpts | Popular but NC weights; would require retraining to ship. |
| **CosyVoice 2 / 3** | 0.5B+ | ⚠️*verify* | In-context reference audio tokens | ❌ | ⚠️ | **The previous dead end (§2).** Excluded unless deep research shows v3 exposes a vector. |
| **VibeVoice** (MSFT) | — | **Research-only** | — | — | — | Excluded. |
| **Sarvam Bulbul v3** | — | **API only, closed** | 35+ voices, 11 Indian langs → 22 planned; native Hinglish code-switching | n/a | ✅ best-in-class | **Not customizable.** Use strictly as the *quality ceiling* to benchmark our Indic output against. |

### 6.1 Resulting backend strategy

- **English / two-tower research track → Zonos.** Only backend where description → 128-d vector → frozen renderer is directly implementable.
- **English / zero-training baseline → VoxCPM2.** Native voice design, Apache-2.0, fits the GPU.
- **Indic track → Indic Parler-TTS.** Apache-2.0, description-native, 21 languages. IndicF5 as the NC-lane quality comparison.
- **Benchmarks (not shipped):** Sarvam Bulbul v3 (Indic ceiling), IndexTTS-2 (emotion-control ceiling), ElevenLabs Voice Design (product ceiling).

---

## 7. Recommended architecture

Two tracks that share an identity store, an eval harness, and an API surface.

```
                    ┌──────────────────────── SHARED ────────────────────────┐
                    │  Identity Store  ·  Eval Harness  ·  Catalog  ·  API    │
                    └────────────────────────────────────────────────────────┘
                           ▲                                    ▲
   ENGLISH TRACK           │                                    │        INDIC TRACK
   ───────────────         │                                    │        ────────────
   description ─► [ Description Encoder ]                       description(+lang) ─► [ Indic Parler-TTS ]
                       (frozen text LM)                                                (description cross-attn)
                           │                                                                 │
                           ▼                                                                 ▼
                  [ Generative Mapper ]  ◄── §4.2: flow-matching / contrastive,          Tier-2 identity:
                       (~10–50M params, THE trained part)                                 mint seed clip once
                           │
                           ▼
                  voice identity: 128-d vector  ──► [ Zonos, FROZEN ] ──► audio
                                                          ▲
                                              per-line performance direction
                                              (emotion / pace / pitch dials)
```

**Component inventory:**

1. **Description encoder** — frozen sentence/text embedder (candidates: T5, a sentence-transformer, a small Qwen). Not trained.
2. **Generative mapper** — the only trained component. ~10–50M params. Trains on a 12GB card. Outputs a *distribution* over speaker vectors, not a point (§4.2).
3. **Speaker encoder** — frozen. Zonos ships its speaker-embedding model as a separate checkpoint; use it to extract training targets over a corpus. Defines the target manifold.
4. **Renderer** — frozen, swappable. Zonos / VoxCPM2 / Indic Parler-TTS behind one interface.
5. **Identity store** — `character_id`, `language`, `description`, `embedding_vector` (Tier 1) *or* `seed_audio_ref` (Tier 2), `backend_id`, `backend_version`, tags, created_at. Must represent both tiers and must record which backend minted it.
6. **Catalog** — a few hundred pre-minted, human-audited voices with descriptions + tags. Ships value on day one and *is* the retrieval index for §4.2 option 2.
7. **Eval harness** — see §10. Build this before training anything.

---

## 8. Data requirements

### 8.1 What training the mapper actually needs

Paired `(natural-language voice description, audio of that voice)`. Nothing else. Estimated need: **500–3,000 hrs with diverse speakers** for a strong mapper; a usable prototype is trainable on **~50–100 hrs / ~1,000 distinct speakers**.

**Speaker count matters more than hours.** The mapper learns a map into speaker space; 1,000 speakers × 3 min beats 50 speakers × 1 hr.

### 8.2 Public corpora — what exists

| Dataset | Size | Has descriptions? | Use |
|---------|------|-------------------|-----|
| **ParaSpeechCaps** | 342 hrs human-labelled + 2,427 hrs auto = **~2,769 hrs**, 59 style tags incl. abstract (*guttural, nasal, pained*) | ✅ Native | ⭐ **Primary English training set.** Finetuning Parler-TTS on it gave +7.9% consistency MOS, +15.5% naturalness MOS. |
| **LibriTTS-P** | LibriTTS-R + speaker-identity & style prompts, hybrid human/synthetic | ✅ Native | ⭐ Core English pairs. Highest-quality human annotation. |
| **TextrolSpeech** | ~330 hrs | ✅ Native | Emotion/style control supplement. |
| **SpeechCraft** | large-scale expressive | ✅ Native | Fine-grained expressive supplement. |
| **IndicVoices-R** | **1,704 hrs, 10,496 speakers, 22 Indian languages**, 93.25% extempore | ❌ (must annotate) | ⭐ **Primary Indic training set.** Speaker count is outstanding. Extempore → natural prosody. |
| **Rasa** | ~400 hrs, 20 speakers, expressive | ❌ | Indic expressive supplement. |
| **RASMALAI** | Indic accents & intonations | ⚠️ verify | Indic accent coverage. |
| **LibriTTS-R / Emilia / MLS / VoxCeleb2** | very large | ❌ | Raw speaker diversity; annotate with our own pipeline. |

### 8.3 Your VoicePersona dataset — verdict and fix

**Current:** 15,082 samples · 10,179 speakers · **48.7 hrs** · 8+ languages · 702 accent variants. Sources: Laions Got Talent, GLOBE_V2, AniSpeech, AnimeVox. Descriptions from **Qwen2-Audio-7B-Instruct** (~500 chars). **CC0.**

**What's genuinely good:** 10,179 speakers in 48.7 hrs is an excellent *speaker-diversity* ratio — exactly the property §8.1 says matters. And AniSpeech + AnimeVox give **character/anime voices that no public corpus has**. That is real, non-replicable coverage.

**Why it isn't accurate enough — the root cause:** captions were produced by asking an audio-LM to describe the voice directly. Audio-LMs are unreliable at *quantitative* vocal attributes; they produce fluent, plausible, weakly-grounded prose. Your own card admits this ("AI predictions and may contain inaccuracies"). Training a mapper on ungrounded captions teaches it to reproduce the *caption model's* biases, not the voice.

**The fix — measure first, caption second** (this is the Data-Speech / ParaSpeechCaps recipe):

1. **Extract objective attributes with signal processing and specialist classifiers**, not an LLM: F0 mean/std/range, speaking rate (phones/sec), jitter, shimmer, HNR, spectral tilt, formants F1–F3, estimated vocal-tract length, SNR / reverberation, plus a dedicated speaker-attribute classifier for gender/age.
2. **Bin each measured value** into linguistic buckets (*very low-pitched · low · moderate · high*), calibrated against the corpus distribution — not absolute thresholds.
3. **Have an LLM write natural-language captions *from the measured bins*.** The LLM's job is fluency and phrasing, not perception. This makes captions grounded and, critically, **reversible** — you can verify a generated voice against the description by re-measuring it (§10).
4. **Keep the Qwen2-Audio pass** as a *supplementary* impressionistic layer for abstract, unmeasurable qualities (*menacing, warm, world-weary*) — validated against ParaSpeechCaps' 59-tag taxonomy rather than free-form.

**Also fix the skews:** female 62.6%, twenties 76.1%. Rebalance by sampling or reweighting, or the mapper inherits the bias and "old man's voice" lands off-manifold.

**Resulting dataset plan:** ParaSpeechCaps + LibriTTS-P for the core English mapping · **VoicePersona v2** (rebuilt per above) for character/stylized coverage · IndicVoices-R + our annotation pipeline for the Indic track.

---

## 9. Hardware reality check (8–12 GB)

This constraint is load-bearing and quietly determines the whole plan.

**Fits locally:**
- Inference: VoxCPM2-0.6B (~6 GB) comfortably; VoxCPM2-2B (~8 GB) on a 12 GB card; Zonos-1.6B (~6 GB); Indic Parler-TTS 0.9B.
- Training the **generative mapper** (10–50M params on pre-extracted frozen embeddings) — trivially. This is the entire trained surface of the project.
- Batch **feature extraction** and dataset annotation (CPU/GPU mix, time-bound not memory-bound).

**Does NOT fit locally — rented GPU events:**
- Any TTS backbone fine-tune (Parler-style description conditioning, Indic Parler-TTS finetune). A100/H100, measured in days.
- Large-scale embedding extraction over 2,000+ hrs (feasible locally but slow; rent for throughput).

**Design consequence:** pre-extract and cache **all** speaker embeddings and text embeddings to disk once. After that, mapper training reads vectors, never audio — turning a GPU-bound problem into a laptop-friendly one. **Make this caching layer a first-class part of the repo, not a script.**

---

## 10. Evaluation harness — build this FIRST

Absence of an eval harness is the second reason the last attempt stalled: with no measurement, "is it working?" is unanswerable and every change is a guess.

Four axes:

1. **Description adherence** — does the voice match the words?
   - *Objective, cheap:* re-measure the generated audio's attributes (§8.3 step 1) and compare against the description's target bins. This works **because** the captions are grounded in measurements. Report per-attribute error.
   - *Subjective:* LLM-judge, following **InstructTTSEval**'s three-tier protocol — Acoustic-Parameter Specification (12 attributes: pitch, speed, emotion, gender, age, clarity, fluency, accent, texture, tone, volume, personality), Descriptive-Style Directive, and **Role-Play** (closest to our use case).
2. **Identity consistency** — the property that makes this a *voice library*. Generate 20 different sentences from one identity; compute pairwise speaker-embedding cosine similarity with an **independent** speaker-verification model (not the one used for conditioning — otherwise it's marking its own homework). Also test across a session restart and a backend version bump.
3. **Diversity** — the anti-mode-collapse check that catches the §4.2 MSE failure. Sample N voices from one description; measure spread in speaker space. Sample from M different descriptions; confirm the clusters are separable. **A collapsed mapper scores well on adherence and fails here** — this test is what makes the failure visible.
4. **Intelligibility & quality** — WER via ASR round-trip (Whisper; use an Indic-capable ASR for the Indic track), plus UTMOS/DNSMOS for naturalness.

**Fixed evaluation set:** ~50 character descriptions spanning the intended range (age, gender, register, emotion, stylization) × a fixed dialogue script. Run it against every backend and every mapper checkpoint. Never change the set without versioning it.

---

## 11. The Grand Plan

Two tracks running to a public website where anyone can create voices.

**Track A (ML)** leads — the product is only as good as the voices.
**Track B (Platform)** starts in parallel from Stage 1, because the **identity store schema is the contract between the two tracks**, and changing it late is the expensive mistake.

```
 TRACK A — ML  (local-first, 8–12GB)
 ─────────────────────────────────────────────────────────────────────────
 S0 Baseline ──► S1 Identity ──► S2 Retrieval ──► S3 Generative ──► S4 Dataset ──► S5 Indic
    + harness      layer          mapper           mapper            v2             parity
                     │                │                                  │
                     │ identity       │ catalog                          │
                     │ schema         │ exists                           │
                     ▼                ▼                                  ▼
 ─────────────────────────────────────────────────────────────────────────
 S6 Inference ──► S7 Model ──► S8 Web app ──► S9 Multi-tenant ──► S10 Public
    service         hosting      (library)      platform             release
 TRACK B — PLATFORM
```

**Sequencing rule (non-negotiable):** do not start Stage N+1 until Stage N's exit criterion is *measured*, not felt. Every exit criterion below is a number or a demonstrable behaviour. This rule is the difference between this attempt and the last one.

### Stage summary

| # | Track | Goal | Exit criterion | GPU |
|---|-------|------|----------------|-----|
| **S0** | ML | Prove the diagram end-to-end with zero training | Baseline numbers table exists for EN + Indic | local |
| **S1** | ML | Voice identity persists | Same character across 20 lines + a restart | local |
| **S2** | ML | First "our model" — contrastive retrieval mapper | Beats S0 on adherence at equal-or-better diversity | local |
| **S3** | ML | Generative mapper (flow matching) | Beats S2 on adherence *without* losing diversity | local (+rented for sweeps) |
| **S4** | ML | VoicePersona v2, measure-first captions | Caption attributes verifiable by re-measurement | local |
| **S5** | ML | Indic at parity with EN | Indic identity consistency == EN; gap to Bulbul v3 quantified | rented |
| **S6** | Platform | Inference service behind a stable API | `mint` + `render` callable over HTTP, backend-swappable | local |
| **S7** | Platform | Models hosted on real GPU infra | p95 render latency + $/min-of-audio measured | cloud |
| **S8** | Platform | The website | A stranger mints a voice and downloads audio unaided | cloud |
| **S9** | Platform | Multi-tenant | Accounts, quotas, moderation, provenance live | cloud |
| **S10** | Both | Public release | Public site + open dataset + model card + writeup | cloud |

### Minimum shippable thing (name it now, protect it)

**S0 + S1 + S6 + S7 + a cut-down S8** is already a genuinely useful public product: *a browsable catalog of a few hundred curated voices, in English and Indian languages, with batch dialogue rendering and export.* No custom mapper required.

Ship that. It de-risks the entire platform track against real users while Track A is still doing research, and it gives the mapper (S2/S3) an honest baseline to beat *in production*, not just in a notebook.

### Stage detail

**S0 — Baseline & eval harness** *(no training)*
Stand up VoxCPM2 locally; render 50 test descriptions. Stand up Indic Parler-TTS; render the same in Hindi + 2 more languages using its 69 named voices. Build the eval harness (§10). Build the identity store schema supporting Tier 1 and Tier 2.
→ *Exit: a baseline numbers table. Every later improvement claim is measured against it.*

**S1 — Identity layer**
Stand up Zonos; verify the 128-d speaker embedding round-trips (extract → store → re-inject → same voice). Implement Tier-2 seed-clip identity on VoxCPM2 as the fallback. Measure identity consistency (§10 axis 2) for both tiers.
→ *Exit: a character sounds the same across 20 lines and a process restart.*

**S2 — Retrieval mapper** *(first real "our model")*
Extract speaker embeddings across ParaSpeechCaps + LibriTTS-P + VoicePersona; **cache to disk** (§9). Train a CLIP-style contrastive joint space (description ↔ speaker embedding). Ship description → retrieve top-k → interpolate/perturb → identity. Populate the catalog.
→ *Exit: beats S0 on description adherence at equal or better diversity.*

**S3 — Generative mapper**
Replace retrieval with a flow-matching / diffusion head over the speaker manifold (§4.2 option 1). Add classifier-free guidance as a user-facing adherence↔diversity dial.
→ *Exit: beats S2 on adherence without losing diversity — both measured.*

**S4 — Dataset rebuild**
Build the measure-first annotation pipeline (§8.3). Rebuild VoicePersona as **v2**; publish it. Annotate IndicVoices-R with the same pipeline. Retrain; re-measure.
→ *Exit: a generated voice's attributes can be verified against its description by re-measurement.*

**S5 — Indic parity**
Bring the Indic track to the same identity guarantees. Benchmark against Sarvam Bulbul v3 as the ceiling. Decide per-language whether Apache-licensed quality is shippable.
→ *Exit: Indic identity consistency matches EN; the quality gap to Bulbul v3 is a number.*

**S6 — Inference service** *(can start during S1)*
Wrap everything behind the renderer adapter interface (§12.1) and an HTTP API. Job queue for rendering. This is the layer that makes backends swappable, and it is the reason Track B can proceed while Track A is still changing models.
→ *Exit: `mint` and `render` work over HTTP against at least two different backends with no client change.*

**S7 — Model hosting & GPU infra**
Container images with weights baked or volume-cached. Warm-pool sizing. Audio cache. Observability.
→ *Exit: p95 render latency and $/minute-of-generated-audio are both measured under realistic load.*

**S8 — Web app: the voice library**
The four surfaces in §13. Design for the catalog-first experience.
→ *Exit: a person who has never seen the project mints a voice and downloads usable audio without help.*

**S9 — Multi-tenant platform**
Accounts, quotas, rate limits, moderation, provenance logging, watermarking, ToS. Billing only if it is actually needed.
→ *Exit: safe to put on the open internet.*

**S10 — Public release**
Public site. VoicePersona v2 published. Model card with honest limitations and eval numbers. Writeup or paper. Open-source what the licences allow.

---

## 12. Serving & hosting architecture

Requirement: *a way to clone/host the models we need, so anyone can use this to create voices.*

### 12.1 The renderer adapter — the single most important abstraction

Every backend hides behind one interface. This is the seam that keeps §2 from happening again: when a better TTS ships, only an adapter changes.

```python
class Renderer(Protocol):
    backend_id: str          # "voxcpm2" | "zonos" | "indic-parler"
    backend_version: str     # pinned; recorded on every minted identity
    identity_tier: int       # 1 = speaker vector, 2 = seed clip
    languages: list[str]
    public_servable: bool    # licence gate, see 12.5

    def mint_identity(description: str, lang: str, seed: int) -> Identity: ...
    def render(identity: Identity, text: str, direction: Direction | None) -> Audio: ...
```

`Direction` carries the **per-line performance** (emotion, pace, emphasis) that §4.4 insists must stay separate from identity. Backends that cannot honour it declare so and degrade explicitly rather than silently.

**Every identity record stores `backend_id` + `backend_version`.** A Tier-1 vector is only meaningful relative to the model that produced it; a Tier-2 seed clip is only meaningful relative to a model *version*. Without this field, a routine model upgrade silently changes every character's voice — a catastrophic and unrecoverable failure for anyone who already shipped a game with it.

### 12.2 Two workloads with opposite cost profiles

This split is the key economic insight of the whole platform:

| | **Mint a voice** | **Render dialogue** |
|---|---|---|
| Frequency | rare (once per character) | constant, bursty, batch |
| Compute | description encoder + mapper = **10–50M params** | full TTS backbone, 0.9–2B params |
| Runs on | **CPU, in milliseconds** | GPU |
| Cost | ~free | the entire GPU bill |

**Therefore: make minting and browsing free and instant; meter only rendering.** Users can design, name, tag, and organise a whole cast of characters at near-zero marginal cost, and pay only when they want to *hear* it. That is a genuinely good product shape and it falls directly out of the architecture — do not accidentally design it away by coupling mint to a GPU render.

(Caveat: Tier-2 seed-clip identities *do* need one GPU render at mint time. A reason to prefer Tier 1 that has nothing to do with ML quality.)

### 12.3 Runtime topology

```
  Browser ──► API (FastAPI, CPU, always-on)
                 ├── mint   ──► description encoder + mapper (CPU) ──► identity ──► Postgres
                 ├── search ──► pgvector similarity over identities ──► catalog
                 └── render ──► job queue ──► GPU worker pool ──► object storage ──► signed URL
                                                    │
                                     warm pool, weights on persistent volume
```

- **Database: Postgres with `pgvector`.** Stores identities *and* indexes them. One column for the 128-d speaker embedding, one for the description text embedding. This gives **"find me voices like this one"** and **semantic search over the catalog** for free — no separate vector DB. (Supabase and Neon are both already wired into this dev environment.)
- **Object storage with zero egress** (Cloudflare R2 or equivalent). Audio egress is the sleeper cost of any TTS product.
- **Aggressive render cache** keyed on `(identity_id, backend_version, text, direction)`. Game dialogue is re-rendered constantly during iteration, so a cache hit is a 100% cost saving on the single most common operation.

### 12.4 GPU hosting options

| Option | Cold start | Cost shape | When |
|--------|-----------|-----------|------|
| **Single always-warm GPU pod** (RunPod / Vast / Lambda) | none | flat ~$0.2–0.5/hr for a 24GB card | ⭐ **Start here.** Predictable, no cold-start tax, trivially debuggable. |
| **Serverless GPU** (Modal / RunPod serverless / Beam / Replicate) | 10–60s, dominated by pulling multi-GB weights | per-second | Once traffic is bursty and idle time dominates cost. |
| **Managed endpoints** (HF Inference Endpoints) | moderate | per-hour | Fastest to stand up; least control. |
| Self-host on owned hardware | none | capex | Not with an 8–12GB card. Later, if ever. |

**The cold-start cost is weight loading, not model init.** Bake weights into the image or mount a persistent volume. Getting this wrong turns a 2-second render into a 60-second one.

### 12.5 The licence gate — where §5 stops being theoretical

Hosting a model publicly *is* commercial-scale distribution of its output. The research/commercial split becomes a hard runtime boundary:

- **Publicly servable (Apache-2.0 / MIT):** VoxCPM2, Qwen3-TTS, **Indic Parler-TTS**, Chatterbox, Zonos *(⚠️ verify)*.
- **Research lane only, never in the public service:** IndicF5 (CC-BY-NC), XTTS-v2 (CPML), IndexTTS-2, F5-TTS, VibeVoice.

Enforce this **in code, not in a README**: each adapter declares `public_servable`, and the public deployment refuses to load a backend that is false. A licence rule that lives only in a document will eventually be violated by a config change.

---

## 13. The web application

Four surfaces. Catalog-first, because that is what works with a cold start and no users.

1. **Voice Library** — browse the curated catalog. Filter by language, gender, age, register, style tags. Play samples inline. Semantic search ("gravelly old mentor") over the pgvector index. *This is the landing experience.*
2. **Voice Studio** — type a description → mint → preview on a sample line → adjust (the CFG adherence↔diversity dial from §4.2, plus attribute nudges) → save to your library. Show **several candidates per description** — §4.2 makes multiple valid voices a feature, so surface it rather than hiding it behind one arbitrary sample.
3. **Character sheet** — a character is **one identity per language** (§4.5), linked by `character_id`. Makes the per-language decision visible and intentional in the UI instead of a hidden limitation.
4. **Script renderer** — paste or upload a dialogue script, assign voices per speaker, set per-line performance direction, batch render, download a zip **plus a manifest JSON** (line id → file → character → language) that a game engine can consume directly. This is the surface that makes the product usable in a real game pipeline rather than a toy.

**Export:** WAV (48kHz where the backend supports it) and MP3, plus the manifest. Consider Unity/Unreal-friendly file naming — a small detail that decides whether anyone uses this in practice.

---

## 14. Cost model

⚠️ **All figures below are order-of-magnitude estimates to size decisions, not measured facts.** They are replaced by real numbers at S7, which is exactly what S7's exit criterion demands.

**Development**
- Local work (S0–S4): **$0**. This is the whole point of the §9 cache-embeddings-once strategy.
- Rented fine-tune events (S5, and S3 sweeps): A100-class ≈ $1.5–2/hr. A Parler-style description-conditioned fine-tune is plausibly 2–5 days ≈ **$70–250 per run**. Budget for ~3 runs, not 1 — the first will be misconfigured.

**Public service (steady state)**
- 1× always-warm 24GB GPU ≈ $0.30/hr ≈ **~$220/month**.
- Postgres: free tier → ~$25/mo (Supabase / Neon).
- Object storage: ~$0.015/GB-month, **$0 egress** on R2.

**Unit economics — the number that matters**
A 0.9–2B TTS on a 24GB card plausibly generates somewhere between real-time and a few× real-time. That puts **$/minute-of-generated-audio in the fractions-of-a-cent range**, against commercial TTS priced at cents-to-dimes per minute. If that holds, the margin is enormous and generous free tiers are affordable — but *it is an assumption until S7 measures it*, and throughput on a batch of short game-dialogue lines is typically far worse than on one long paragraph. Measure with realistic line lengths, not a benchmark paragraph.

**The lever that dominates everything:** the render cache (§12.3). Iterating on game dialogue means re-rendering the same lines repeatedly; cache hits cost nothing.

---

## 15. Ethics, safety & licence propagation

Once "anyone can use this," these stop being footnotes.

### 15.1 A structural safety advantage — keep it

**VoiceForge generates novel voices from descriptions. It does not clone a voice from user-uploaded audio.** That means the core product has **no impersonation vector at all** — there is no way for a user to supply Person X's voice and get Person X back.

This is a real and defensible safety position, and it is a *consequence of the architecture rather than a policy bolted on top*. **Locked decision: the public product accepts no user audio upload.** If reference-audio cloning is ever added, it becomes a different product with a completely different risk profile and needs its own consent/verification design.

(Note the tension: Tier-2 seed-clip identity involves audio, but it is audio the *system* generated, never audio a user supplied. That distinction must be enforced at the API boundary.)

### 15.2 Remaining obligations

- **Named-real-person descriptions** ("sounds exactly like <celebrity>") — refuse at the description-validation layer. Cheap to implement, and the only realistic impersonation route left.
- **Watermarking** — embed an inaudible watermark in all output (AudioSeal, Perth, or equivalent). Add at S9, before public exposure.
- **Provenance logging** — every rendered file traceable to `(identity_id, backend_version, description, user, timestamp)`. This is already needed for the model-upgrade problem in §12.1; safety gets it for free.
- **Demographic bias** — VoicePersona is 62.6% female and 76.1% twenties. Rebalance (§8.3), and **report per-demographic eval slices** rather than a single aggregate number that hides the gap.
- **Disclosure** — synthetic audio labelled as such in the UI and in the export manifest.

### 15.3 Licence propagation — the non-obvious trap

There are **three separate licence questions**, and they are routinely conflated:

1. Can I *use* the model? (inference)
2. Can I *serve* the model publicly? (§12.5)
3. **Does training on a dataset impose obligations on the model I train and release?**

Question 3 is the one that bites late. If VoicePersona v2 or the mapper is trained on a corpus with share-alike or non-commercial terms, the released artefact may inherit them. Your own VoicePersona is **CC0**, which is maximally permissive and a genuine asset here — but ParaSpeechCaps, LibriTTS-P, IndicVoices-R and Rasa each need their licences read **specifically for the model-training-and-release case**, not just for use. This is a deep-research question (Q15), and the answer should be settled *before* S2 trains anything, because retraining to escape a licence is expensive.

---

## 16. Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Mapper mode-collapse via MSE loss** | **High** — silent, plausible-sounding failure | §4.2: generative head, never plain MSE. §10 axis 3 is the detector. |
| Picking a backend that exposes no speaker vector | High — repeats §2 exactly | §6 selection criterion is explicit; Tier-2 fallback designed in from the start. |
| Ungrounded captions poison the mapper | High | §8.3 measure-first pipeline. Reversible captions make it auditable. |
| Description adherence unmeasurable → can't tell if it works | High | §10 harness before any training. |
| **Model upgrade silently changes every existing voice** | **High** — unrecoverable for shipped games | §12.1: `backend_version` on every identity; pin versions; treat upgrades as migrations with re-render + diff. |
| Licence contamination reaching production | Medium-High | §12.5 `public_servable` enforced in code, not docs. |
| **Dataset licence propagates to the released model** | Medium-High | §15.3 / Q-E1 — settle *before* S2 trains. |
| Voice-cloning misuse | Medium-High | §15.1 no-audio-upload is structural; plus named-person refusal, watermarking, provenance. |
| 8–12GB blocks progress | Medium | §9 cache-embeddings-once; fine-tunes are scheduled rented events, not a daily dependency. |
| Indic quality below usable | Medium | Indic Parler-TTS is Apache + description-native; IndicF5 as research-lane comparison; Sarvam as ceiling. |
| GPU cost runaway at S7+ | Medium | Warm-pool sizing, render cache, free-mint/metered-render split (§12.2), hard quotas at S9. |
| Cold-start latency ruins UX | Medium | §12.4 — start always-warm; weights baked or volume-mounted. |
| Dataset demographic skew propagates | Medium | §8.3 rebalance; per-demographic eval slices. |
| Platform track builds on an ML contract that then changes | Medium | Identity schema frozen at S1 and treated as an interface; adapters absorb backend churn. |

---

## 17. Open questions for deep research

Labelled to match the question IDs in the deep-research prompt (§18). **A1–A3 are decisive** — they can invalidate the primary architecture.

### A. Decisive — architecture-defining

- **A1. Is Tier-1 (fixed embedding vector) actually right, or does Tier-2 (seed clip) win in practice?** Provisional answer is Tier 1. Settle empirically: does a Zonos 128-d vector re-injected across sessions and versions reproduce the voice, and is 128-d expressive enough for the full stylistic range in §5?
- **A2. Confirm Zonos-v0.1's licence, and the exact speaker-embedding contract** — dimension, normalisation, and critically: **do arbitrary/synthetic vectors in that space produce coherent voices, or do they fall off-manifold into artefacts?** *The entire two-tower design rests on the manifold being navigable by synthesis, not merely by extraction.* If off-manifold synthesis fails, the project pivots to Tier 2 wholesale.
- **A3. Which generative head for the mapper** — flow matching vs. diffusion vs. normalizing flow? Compare PromptSpeaker (Glow), HiStyle, VoiceDesigner (diffusion transformer), MOSS-VoiceGenerator, Unispeaker. Read all five before choosing.

### B. Backend selection

- **B1. Verify VoxCPM2's Indic coverage** within its 30 languages. If Hindi et al. are genuinely supported, it could unify both tracks on one Apache-2.0 backend — a major simplification.
- **B2. Does Qwen3-TTS expose its jointly-trained speaker encoder externally?** If yes it is a second Tier-1 candidate (Apache-2.0, stronger reported quality than Zonos), though still no Indic.
- **B3. How do VoiceDesigner and VoiceSculptor (2026) actually work, and are weights released?** VoiceSculptor claims full open-source incl. pretrained models and SOTA on InstructTTSEval-Zh; its "design a waveform, then clone it" approach is *exactly Tier 2* — meaningful independent evidence for that design.
- **B4. Indic Parler-TTS's real quality ceiling** vs. IndicF5 vs. Sarvam Bulbul v3, per language. Determines whether the Apache-2.0 Indic path is shippable or whether the commercial path needs a different answer.
- **B5. Emotion/performance control per line** — does IndexTTS-2's timbre/emotion disentanglement transfer onto Zonos's existing emotion dials, or does it need a separate style-token path?
- **B6. Confirm licences from primary LICENSE files** for CosyVoice 2/3, Chatterbox, F5-TTS, IndicF5, XTTS-v2, IndexTTS-2, VibeVoice — and re-verify that CosyVoice 3 still exposes no speaker vector, since that claim is the entire basis for abandoning the previous attempt (§2).

### C. Data & annotation

- **C1. Exact annotation recipe from ParaSpeechCaps / Data-Speech** — the attribute list, binning thresholds, and caption-generation prompts. Reuse rather than reinvent.
- **C2. Which speaker encoder defines the target manifold?** Zonos's ResNet293-SimAM-ASP vs. ECAPA-TDNN vs. WavLM-based vs. NeMo TitaNet. Must be one that (a) has a navigable synthesis manifold and (b) is *not* the same model used for eval axis 2, to avoid marking its own homework.
- **C3. Minimum viable speaker count** for the mapper. §8.1 asserts speaker count dominates hours — find the empirical curve if anyone has published one.
- **C4. Non-human voice augmentation** (deferred workstream): what DSP + generative chain does VoiceDesigner use to synthesize fictional-voice training data?

### D. Platform & serving

- **D1. Realistic throughput and $/min-of-audio** for VoxCPM2 / Zonos / Indic Parler-TTS on a 24GB card, measured on *short game-dialogue lines*, not long paragraphs. Underpins the entire §14 cost model.
- **D2. Cold-start mitigation** on serverless GPU for multi-GB TTS weights — who has published real numbers, and what actually works (baked images vs. volumes vs. snapshotting)?

### E. Legal & safety

- **E1. Licence propagation (§15.3):** for ParaSpeechCaps, LibriTTS-P, LibriTTS-R, IndicVoices-R and Rasa — what do their licences say about **training and releasing a model**, as distinct from using the data? Settle before S2.
- **E2. Audio watermarking** — AudioSeal vs. Perth vs. alternatives: robustness, licence, and whether embedding survives MP3 transcode and game-engine audio processing.

---

## 18. Deep-research prompt

Copy the block below verbatim into a deep-research agent, together with this file.

```
You are doing a deep technical research pass for VoiceForge, a Text-to-Voice (TTV)
system. Read the attached scope document (VOICEFORGE-SCOPE.md) first — it contains
the architecture, locked decisions, and the model landscape as currently understood.
Your job is to VERIFY that document's assumptions against primary sources and answer
its open questions, so that implementation can begin without re-deriving anything.

PROJECT IN ONE LINE
Natural-language character description -> a persistent, reusable voice identity ->
arbitrary dialogue rendered in that voice, in English and Indian languages, at
quality usable in games and dramatic content. Delivered ultimately as a public
website where anyone can create and use voices.

LOCKED CONSTRAINTS (do not relitigate these; work within them)
- Voice identity = a fixed embedding vector (PROVISIONAL — question A1 may overturn it;
  a seed-audio-clip fallback, "Tier 2", is designed in).
- Licensing posture: research now, but an Apache-2.0/MIT swap-in must be documented
  for every non-commercial component. Publicly hosting a model counts as commercial use.
- Local hardware: a single 8-12GB consumer GPU. Anything needing more is a discrete
  rented-GPU event, not a daily dependency.
- One voice profile per language per character. Cross-lingual identity merging is
  explicitly out of scope for now.
- Voice range: human + heavy stylization (aged, raspy, whispered, menacing, theatrical).
  Non-human/fictional (goblin, dragon, robot) is a deferred workstream.
- The TTS backend is frozen and swappable. We do not train a TTS acoustic model.
- No user audio upload in the public product (this is a deliberate safety property).

EVIDENCE STANDARD — this is the most important instruction
- PRIMARY SOURCES ONLY for any factual claim: the model's own repository, model card,
  LICENSE file, official docs, or the paper. Blog roundups and "best TTS of 2026"
  listicles are NOT acceptable evidence, especially for licences — several such claims
  in the scope doc are marked with a warning symbol precisely because they came from
  secondary sources.
- For every claim, cite the URL and say what kind of source it is.
- State confidence explicitly. If you cannot verify something, say "UNVERIFIED" and say
  what would verify it. Do not fill gaps with plausible-sounding inference.
- Where the scope document is WRONG, say so directly and show the evidence. Finding an
  error in it is a more valuable result than confirming it.

QUESTIONS, IN PRIORITY ORDER
Answer section A first and in the most depth; A2 is the single most decisive question.

A. DECISIVE — these can invalidate the primary architecture
  A1. Is a fixed speaker-embedding vector the right identity representation, versus
      persisting a generated seed audio clip? Find empirical evidence either way.
  A2. For Zonos-v0.1: confirm the exact licence from the LICENSE file. Then determine the
      speaker-embedding contract — dimension, normalisation, and CRITICALLY whether
      ARBITRARY or SYNTHESIZED vectors in that space produce coherent voices, or whether
      they fall off-manifold into artefacts. The entire two-tower design depends on that
      manifold being navigable by synthesis, not merely by extraction from real audio.
      Look for anyone who has interpolated, perturbed, or generated vectors in it.
  A3. Which generative head best models p(speaker_vector | description)? Compare
      PromptSpeaker (Glow/normalizing flow), HiStyle, VoiceDesigner (diffusion
      transformer), MOSS-VoiceGenerator, Unispeaker. Note specifically how each handles
      the one-to-many problem, and report any published comparison.

B. BACKEND SELECTION
  B1. VoxCPM2: does its 30-language support include Hindi or other Indian languages?
      Enumerate them from the official source. If yes, it could unify both tracks.
  B2. Qwen3-TTS: is its jointly-trained speaker encoder externally addressable — can one
      extract and re-inject a speaker vector? Confirm the licence.
  B3. VoiceDesigner and VoiceSculptor (2026): how do they work in detail, and are weights
      actually released? VoiceSculptor claims full open-source including pretrained models.
  B4. Indic Parler-TTS vs. IndicF5 vs. Sarvam Bulbul v3: quality per language, with any
      published MOS/CMOS/speaker-similarity numbers. Is the Apache-2.0 Indic path good
      enough to ship?
  B5. IndexTTS-2's timbre/emotion disentanglement: can the approach be transferred onto a
      different backend's emotion conditioning, or is it architecture-bound?
  B6. Confirm licences from LICENSE files for: CosyVoice 2/3, Chatterbox, F5-TTS,
      IndicF5, XTTS-v2, IndexTTS-2, VibeVoice. Also confirm whether CosyVoice 3 exposes
      a speaker vector (the scope doc excludes CosyVoice on the basis that it does not —
      verify this, as it is the reason the previous attempt was abandoned).

C. DATA & ANNOTATION
  C1. The exact ParaSpeechCaps / Data-Speech annotation recipe: attribute list, binning
      thresholds, caption-generation prompts. We want to reuse it, not reinvent it.
  C2. Which speaker encoder should define the target manifold — Zonos's ResNet293-SimAM-ASP,
      ECAPA-TDNN, a WavLM-based encoder, or NeMo TitaNet? Criteria: navigable synthesis
      manifold, and availability of a DIFFERENT encoder for evaluation so the system does
      not mark its own homework.
  C3. Is there published evidence on the minimum viable speaker count for training a
      description-to-speaker-embedding mapper, and on speaker-count vs. hours trade-off?
  C4. How does VoiceDesigner synthesize training data for fictional/non-human voices?
      Detail the DSP and generative augmentation chain. (Deferred workstream — brief answer.)

D. PLATFORM & SERVING
  D1. Real measured throughput and $/minute-of-generated-audio for VoxCPM2, Zonos and
      Indic Parler-TTS on a 24GB-class GPU. Prioritise numbers measured on SHORT
      utterances (game dialogue lines), which behave very differently from long paragraphs.
  D2. Cold-start mitigation for multi-GB TTS weights on serverless GPU platforms
      (Modal, RunPod, Replicate, Beam): published real-world numbers and which technique
      actually works — baked images, persistent volumes, or memory snapshotting.

E. LEGAL & SAFETY
  E1. For ParaSpeechCaps, LibriTTS-P, LibriTTS-R, IndicVoices-R and Rasa: what do the
      licences say about TRAINING AND RELEASING A MODEL, as distinct from using the data?
      Quote the relevant terms. This must be settled before any training begins.
  E2. Audio watermarking: AudioSeal vs. Perth vs. alternatives. Compare robustness,
      licence, and whether the watermark survives MP3 transcode and typical game-engine
      audio processing.

OUTPUT FORMAT
1. VERDICT ON THE ARCHITECTURE — does the two-tower design survive contact with A1-A3?
   If not, state the recommended pivot in one paragraph, up front. Lead with this.
2. CORRECTIONS TO THE SCOPE DOCUMENT — a table of every claim found to be wrong or
   unverifiable, with the correction and its primary source.
3. ANSWERS — one section per question, each ending with a confidence level
   (HIGH / MEDIUM / LOW / UNVERIFIED) and the sources used.
4. A REVISED MODEL SELECTION TABLE — replacing scope section 6, with every licence
   confirmed from a primary source and every warning symbol resolved.
5. RECOMMENDED STACK — one concrete recommendation for the English track, one for the
   Indic track, with the reasoning in three sentences each.
6. WHAT TO BUILD FIRST — the concrete Stage 0 setup, given everything found. Specific
   model checkpoints, specific versions, specific install constraints for an 8-12GB GPU.
7. OPEN AFTER RESEARCH — anything that could not be resolved from public sources and
   must be settled by running an experiment instead. For each, describe the cheapest
   experiment that would settle it.
```

---

## 19. Sources

**Models**
- [VoxCPM (OpenBMB)](https://github.com/OpenBMB/VoxCPM) · [Qwen3-TTS Technical Report](https://arxiv.org/html/2601.15621v1) · [Zonos (Zyphra)](https://github.com/Zyphra/Zonos) · [Zonos speaker-embedding checkpoint](https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding) · [Parler-TTS](https://github.com/huggingface/parler-tts) · [Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts) · [IndicF5](https://huggingface.co/ai4bharat/IndicF5) · [XTTS](https://edresson.github.io/XTTS/) · [Sarvam Bulbul v3](https://www.sarvam.ai/blogs/bulbul-v3) · [Sarvam Bulbul docs](https://docs.sarvam.ai/api-reference-docs/models/bulbul)

**Text-to-Voice / voice design research**
- [PromptSpeaker: Speaker Generation Based on Text Descriptions (2310.05001)](https://arxiv.org/abs/2310.05001) · [VoiceDesigner (2608.13613)](https://arxiv.org/abs/2608.13613) · [VoiceSculptor (2601.10629)](https://arxiv.org/pdf/2601.10629) · [HiStyle (2509.25842)](https://arxiv.org/pdf/2509.25842) · [MOSS-VoiceGenerator (2603.28086)](https://arxiv.org/pdf/2603.28086) · [Unispeaker (2501.06394)](https://arxiv.org/pdf/2501.06394) · [Generating Speakers by Prompting Listener Impressions (2406.08812)](https://arxiv.org/pdf/2406.08812) · [IndexTTS2 (2506.21619)](https://arxiv.org/abs/2506.21619)

**Datasets & benchmarks**
- [ParaSpeechCaps / Scaling Rich Style-Prompted TTS Datasets (2503.04713)](https://arxiv.org/abs/2503.04713) · [ParaSpeechCaps code](https://github.com/ajd12342/paraspeechcaps) · [LibriTTS-P (2406.07969)](https://arxiv.org/abs/2406.07969) · [LibriTTS-P code](https://github.com/line/LibriTTS-P) · [IndicVoices-R (2409.05356)](https://arxiv.org/pdf/2409.05356) · [IndicVoices-R on HF](https://huggingface.co/datasets/ai4bharat/indicvoices_r) · [RASMALAI (2505.18609)](https://arxiv.org/pdf/2505.18609) · [InstructTTSEval (2506.16381)](https://arxiv.org/abs/2506.16381) · [AI4Bharat TTS](https://ai4bharat.iitm.ac.in/areas/tts/)

**This project**
- [VoicePersona-Dataset (GitHub)](https://github.com/PranavMishra17/VoicePersona-Dataset) · [VoicePersona on HF](https://huggingface.co/datasets/Paranoiid/VoicePersona)

**Landscape surveys** (lower confidence — secondary sources; licence claims especially need primary verification)
- [Best Open Source Models for Voice Cloning 2026 (SiliconFlow)](https://www.siliconflow.com/articles/best-open-source-models-for-voice-cloning) · [Best Local TTS Models 2026](https://localaimaster.com/blog/best-local-tts-models) · [Open-Source Voice AI India 2026](https://caller.digital/blog/open-source-voice-ai-india-sarvam-ai4bharat-bhasini-2026)

---

## Appendix — verification status

Claims marked ⚠️ in §6 and §12.5 rest on secondary sources and **must be confirmed against primary sources** in the deep-research pass: Zonos licence; VoxCPM2 Indic language coverage; Qwen3-TTS speaker-encoder accessibility; Chatterbox Indic support; CosyVoice 2/3 licence and conditioning mechanism. Everything else in §6 was read from the model's own repo, model card, or technical report.

All figures in §14 are estimates, not measurements, and are superseded by S7.
