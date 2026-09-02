# 00 — Executive Verdict

> **Read this first.** Everything else in `RESEARCH/` is the evidence behind it.
> **Date:** 2026-09-02 · Research pass 1 · 12 parallel domain researchers · ~9,500 lines of primary-source findings
> **Verifies:** [`_source/VOICEFORGE-SCOPE-v1.md`](_source/VOICEFORGE-SCOPE-v1.md)
> **Next:** [`GRAND-PLAN.md`](GRAND-PLAN.md) → [`phases/`](phases/)

---

## 1. THE VERDICT ON THE ARCHITECTURE

**The two-tower design survives. Ship it.**

```
description ──[A: frozen text encoder + TRAINED mapper]──► B: voice identity ──[C: FROZEN TTS]──► audio
```

The decisive question (**A2**) was: *is the speaker-embedding manifold navigable by synthesis, or only by extraction?* The answer is **navigable, and this has been established since 2018** — the scope document treated as an open research risk something the literature had already settled:

- Uniformly random points on the unit hypersphere produce speech **"as natural as for seen or unseen real speakers"** — naturalness MOS **3.65** ([Jia et al., NeurIPS 2018](https://arxiv.org/abs/1806.04558) §3.6)
- WGAN-sampled synthetic embeddings are an **official VoicePrivacy 2026 challenge baseline** (B3)
- **1,225 pairwise speaker interpolations** render at WER 6.42–7.35% ([Gabryś et al., Interspeech 2021](https://arxiv.org/abs/2106.05762))
- **Zyphra ship SLERP speaker blending and additive emotion direction vectors** as ZONOS2 product features — the manifold is vendor-navigated, not hypothetical
- **PromptTTS++** is a working, Apache-2.0, downloadable description→256-d-speaker-embedding generator. The thing the project wants to build **already exists in open source**

**But every single component named in the scope document has changed.** The architecture is right; the parts list is wrong.

### 1.1 The parts list, then and now

| Role | Scope doc v1 | **After research** | Why it changed |
|---|---|---|---|
| **English Tier-1 renderer** | Zonos-v0.1 ⭐"the ONE model" | **Qwen3-TTS-12Hz-1.7B-Base** | Zonos-v0.1 is dead (last commit 2025-03-05) **and licence-poisoned** — its speaker encoder derives from VoxBlink2 (CC-BY-NC-SA). Qwen3 is Apache-2.0, exposes a `(2048,)` vector as the *sole* identity input, and that vector lives in the **backbone's own embedding space** [`03`](03-tts-backends-english.md) [`02`](02-identity-representation.md) |
| **Voice designer** | VoxCPM2 ⭐"primary baseline" | **Qwen3-TTS-1.7B-VoiceDesign** + **MOSS-VoiceGenerator** | VoxCPM2 has **no speaker vector at all**; its "voice design" is a text prefix. MOSS is Apache-2.0 and trained on *cinematic* rather than studio-clean audio [`03`](03-tts-backends-english.md) [`13`](13-description-to-embedding-prior-art.md) |
| **Mapper head** | flow matching / diffusion / normalizing flow | **MDN first**, then two-stage diffusion | The one open working system uses a **mixture density network** — a family the brief omitted entirely. It is also the cheapest [`13`](13-description-to-embedding-prior-art.md) |
| **Identity representation** | Tier 1 (vector), Tier 2 as fallback | **Store Tier 1 AND Tier 2 always. Tier 3 is dead.** | The v0.1→ZONOS2 encoder change orphaned every stored vector once in 16 months. Every commercial product is Tier 2. `seed` is non-deterministic even on one machine [`02`](02-identity-representation.md) [`11`](11-production-api-landscape.md) |
| **Primary English corpus** | ParaSpeechCaps ⭐ | **LibriTTS-P + LibriTTS-R + GLOBE + MLS + Common Voice** | ParaSpeechCaps is **CC-BY-NC-SA** — fails train-and-release on both counts. Its authors released their own fine-tune as NC-SA, which answers the question for us [`05`](05-datasets-and-annotation.md) [`08`](08-licensing-propagation.md) |
| **Indic renderer** | Indic Parler-TTS ⭐ + IndicF5 | **Indic Parler-TTS, alone and weakened** | Indic-Mio, DhVaani, IndicF5, SPRING_F5 are **all NO-GO** on upstream licence chains. See §4 — this is the worst news in the pass [`04`](04-indic-track.md) [`08`](08-licensing-propagation.md) |
| **Watermark** | AudioSeal vs Perth, undecided | **AudioSeal** (MIT, code + weights, verified 4 ways) | Resolved definitively. Also a registered C2PA soft-binding algorithm [`09`](09-safety-and-watermarking.md) |
| **Compliance timing** | S9, "before public exposure" | **Already legally live** | **EU AI Act Art. 50(2) applied 2 Aug 2026** — a month before this research. No grace period [`09`](09-safety-and-watermarking.md) [`08`](08-licensing-propagation.md) |

### 1.2 The one-line recommendation

> **Build the English track on Qwen3-TTS (Apache-2.0 end to end), copy PromptTTS++'s MDN mapper, store both a vector and a seed clip for every identity, train on LibriTTS-P/GLOBE rather than ParaSpeechCaps, watermark with AudioSeal from day one, and treat the Indic track as a separate, riskier project with its own go/no-go gate.**

---

## 2. THE FIVE THINGS THAT WOULD HAVE COST MONTHS

Ranked by how much time each saves.

### 2.1 ParaSpeechCaps is CC-BY-NC-SA — the primary training set was unusable

Scope §8.2 ranks it ⭐ first. It is **non-commercial *and* share-alike**, and its own authors released their fine-tune as NC-SA — which resolves the "does training propagate?" question in the worst direction, from the most authoritative possible source. Also blocked: CapSpeech, EARS, Expresso, Emilia, TextrolSpeech (no dataset licence), SpeechCraft (no licence at all), GigaSpeech, **VoxCeleb2** (never licensed, and Oxford has withdrawn every download).

**Discovering this at S2, after building the annotation pipeline around it, would have meant rebuilding the corpus and retraining.** [`05`](05-datasets-and-annotation.md) [`08`](08-licensing-propagation.md)

### 2.2 The mode-collapse failure is real, but not the failure the brief describes

Scope §4.2 predicts MSE → conditional mean → *"a bland, averaged, characterless voice… sounds fine but generic."*

**What actually happens is worse and easier to detect.** Averaging zero-centred embedding dimensions **shrinks the vector toward the origin**, and every identity **collapses onto one voice** — measured at **GVD −6.5 to −11.6 dB**. Meyer et al., verbatim: the system *"produces a very similar voice **regardless of** the anonymized speaker embedding fed to the TTS."*

**The fix is arithmetic, not architectural, and nearly free: per-dimension rescaling to each dimension's observed empirical range. GVD recovers from −6.50 to −0.14.**

This matters because per-dimension scales span **three orders of magnitude** — in a 704-d ECAPA+x-vector concat, the narrowest dimension spans `[−0.97, −0.11]` while the widest spans `[−72.37, 81.59]`. Any operation treating the space as isotropic — averaging, plain Gaussian noise, unweighted MSE, naive LERP — silently destroys diversity. **And Zonos's 128-d vector is confirmed *not* L2-normalised, so this applies directly.** [`12`](12-speaker-manifold-navigability.md)

### 2.3 An entire adjacent field has already solved much of this

**The VoicePrivacy Challenge** (4 editions, 2020–2026) is the same technical problem from the privacy side: generate a pseudo-speaker vector, render speech, require it to be natural, intelligible, and *distinct from other pseudo-speakers*. Six years of baselines, metrics, negative results, and a ready-made diversity metric (**GVD**, with a published scale).

**The scope document does not mention it once.** [`12`](12-speaker-manifold-navigability.md)

> **VPC 2026 results land 2026-09-26** (Sydney, SPSC/Interspeech) — three weeks after this research. Re-check then.

### 2.4 Your naturalness metric would have fought your diversity metric

[arXiv:2606.19951](https://arxiv.org/abs/2606.19951) (June 2026) perturbed F0 and measured predictor response against humans:

| | correlation with pitch |
|---|---|
| **Humans** | r = **−0.059** |
| **DNSMOS** | r = **−0.788** |
| **UTMOSv2** | r = **−0.722** |

A UTMOS/DNSMOS quality gate **systematically penalises high-pitched voices for a reason humans do not share.** Scope §10 axis 4 specifies exactly those metrics. The bias is systematic enough to read as a real finding, and it would have pushed the mapper away from vocal diversity while appearing to improve quality. Same paper: all six predictors moved **< 0.1** on prosodic corruption that cost humans **1.84 MOS**.

**Use TTSDS2** (`pip install ttsds`, MIT) — the only metric of 16 clearing Spearman 0.50 in every domain. [`06`](06-evaluation-harness.md)

### 2.5 You are already inside the EU compliance window

**EU AI Act Article 50(2) applied 2 August 2026.** The Digital Omnibus's grace covers only pre-existing systems, and **Art. 2(12) carves Art. 50 *out* of the open-source exemption** — so releasing openly does not help. The Code of Practice requires **two marking layers for audio (signed metadata AND watermark)** plus a **free public detector**.

**India's IT Rules (notified 10 Feb 2026, not limited to large platforms)** require a **"prominently prefixed audio disclosure."** For a 1.5-second game line that is product-destroying. **This is the single most important question to put to counsel**, and it is specific to serving Indian languages. [`09`](09-safety-and-watermarking.md) [`08`](08-licensing-propagation.md)

---

## 3. THE MASTER CORRECTIONS TABLE

Every claim in the scope document found to be wrong, unverifiable, or materially incomplete. Detail and primary sources in the linked file.

### 3.1 Architecture & identity

| § | Claim | Verdict | Correction | File |
|---|---|---|---|---|
| §6 | Zonos-v0.1: Apache-2.0, ⭐"the ONE model that makes Option A directly implementable" | **Wrong on both counts** | Dead since 2025-03-05; speaker encoder derives from **VoxBlink2 (CC-BY-NC-SA-4.0)**, challenged on the repo Feb 2025 and never answered. Also pulls **GPL-3.0** (phonemizer/eSpeak-NG). Superseded by **ZONOS2** (June 2026, MIT + Apache-2.0) which the brief never mentions | [`02`](02-identity-representation.md) [`08`](08-licensing-propagation.md) |
| §6 | Zonos embedding: 128-d, ResNet293-SimAM-ASP, 256→128 LDA | **Mostly right, one critical error** | Returns `(1,128)` **bfloat16** and is **NOT L2-normalised**. LDA confirmed by reading the checkpoint pickle: float64 `(128,256)`. Non-normalisation makes per-dim rescaling **mandatory** | [`02`](02-identity-representation.md) |
| §6 | Zonos "109 languages" implies broad coverage | **Misleading** | Those are **eSpeak phonemizer codes, not trained languages** | [`02`](02-identity-representation.md) |
| §4.3 | Tier 1 reproduces the voice **"exactly, by construction"** | **False** | **Vocoder drift**: the x-vector extracted from rendered audio "often differs substantially from the x-vector at the vocoder input." Reproducible at the *input*, not the *output*. Needs a closed loop | [`12`](12-speaker-manifold-navigability.md) |
| §4.3 | Tier 2 is a "fallback"; Tier 3 is "weak" | **Inverted / too generous** | **100% of shipping commercial voice-design products are Tier 2.** Tier 3 is not weak but **dead** — VoxCPM2's `retry_badcase` (default `True`) silently does `current_seed += 1` | [`11`](11-production-api-landscape.md) [`03`](03-tts-backends-english.md) |
| §4.2 | MSE → "bland, averaged, characterless voice" | **Mechanically wrong** | Collapse to **one voice**, not blandness. Cause is origin-shrinkage. Fix is per-dimension rescaling | [`12`](12-speaker-manifold-navigability.md) |
| §4.2 | Generative head "is the correct answer"; retrieval merely "best first implementation" | **Overconfident — the trade is bidirectional** | The only independent A/B ([2406.08812](https://arxiv.org/abs/2406.08812)): flow matching wins fidelity (FAD 3.559 vs 5.244) but **loses adherence** (SRCC 0.60 vs 0.74) and speaker similarity (0.36 vs 0.41). Hybrid best (FAD 3.126) | [`01`](01-ttv-landscape.md) |
| §17-A3 | Head candidates: flow / diffusion / normalizing flow | **Omits the winner** | **MDN** (per-dim 10-component GMM) is what the one open working system uses, and is cheapest | [`13`](13-description-to-embedding-prior-art.md) |
| §17-A3 | Unispeaker, HiStyle listed as generative-head rivals | **Both miscategorised** | Unispeaker is **contrastive/retrieval** (best prior art for S2). HiStyle is **stacked conditional diffusion + contrastive** | [`01`](01-ttv-landscape.md) |
| §3 | "PromptSpeaker — first clean formulation" | **Wrong priority** | **PromptTTS++** (2023-09-15) predates it, and unlike PromptSpeaker **released Apache-2.0 code and weights**. **InstructTTS** (2023-01-31) predates both | [`13`](13-description-to-embedding-prior-art.md) |
| §7 | Mapper 10–50M params | **Too high at the top** | Published working mappers: ~10M (Deep Dubbing), ~30M/stage (HiStyle). Use **10–30M** | [`01`](01-ttv-landscape.md) |
| §7 | Use a frozen ASV encoder to define the target manifold | **Risky as stated** | ASV encoders **suppress** intra-speaker variance — "misaligned with the objective of generation." TacoSpawn: GMM over d-vectors → **g2s 0.35 vs s2s 0.20** (off-manifold); learned space → 0.20/0.20 | [`12`](12-speaker-manifold-navigability.md) |

### 3.2 Backends

| § | Claim | Verdict | Correction | File |
|---|---|---|---|---|
| §6 | VoxCPM2 ⭐ primary baseline, native voice design | **No speaker vector exists** | "Voice design" is `final_text = f"({control}){text}"`. Zero matches for `speaker\|spk_\|embed\|design` in `voxcpm2.py`. Tier 2 only. *(It is the official API, not a demo hack — [`10`](10-performance-control.md) corrects [`03`](03-tts-backends-english.md) here)* | [`03`](03-tts-backends-english.md) |
| §17-B1 | VoxCPM2's 30 languages may unify both tracks | **Hindi only** | Best-in-class Hindi CER 0.79%, but **no other Indic language**. Unification hope is dead | [`03`](03-tts-backends-english.md) [`04`](04-indic-track.md) |
| §17-B2 | Qwen3-TTS speaker encoder "internal, verify if addressable" | **Fully addressable — the key find** | `extract_speaker_embedding()` → `(2048,)`; `x_vector_only_mode=True` sets `ref_code=None` so the vector is the **sole** identity input; `enc_dim == hidden_size` | [`03`](03-tts-backends-english.md) |
| §2, §17-B6 | CosyVoice exposes no speaker vector — the basis for abandoning it | **Right conclusion, wrong reason** | A 192-d CAMPPlus vector *does* exist, but `cosyvoice3.yaml` sets `use_spk_embedding: False`, v2/v3 ship no `spk2info.pt`, and zero-shot needs the prompt token sequence anyway | [`03`](03-tts-backends-english.md) |
| §6 | XTTS-v2 is CPML | **Code is MPL-2.0**; only weights are CPML | Also: `gpt_cond_latent` is a variable-length `[1,1024,T]` **sequence**, not a vector. Coqui defunct, `coqui.ai/cpml` 404s, the idiap fork ships the same registry | [`03`](03-tts-backends-english.md) [`02`](02-identity-representation.md) |
| §6 | IndexTTS-2 "restrictive / non-commercial" | **Not non-commercial** | bilibili MULA, Llama-style 100M MAU / RMB 1B threshold. **But §3.4(c)** (not §2(c)) bars using it *or its outputs* "to improve any AI model" — and §1.6 defines "Use" to include **running**. Hard-exclude from the training environment. Chinese text controlling; PRC law, Shanghai arbitration | [`10`](10-performance-control.md) [`08`](08-licensing-propagation.md) |
| §6 | VibeVoice "research-only" | **Withdrawn** | Microsoft pulled the TTS code 2025-09-05; Large/7B return HTTP 401. LICENSE file is MIT, but every render carries an **audible AI disclaimer** | [`03`](03-tts-backends-english.md) [`08`](08-licensing-propagation.md) |
| §6 | Chatterbox "cleanest license, fallback renderer" | **Confirmed MIT and servable** | But its `emotion_adv` reaches only T3, not S3Gen. Not in Resemble's hosted API | [`03`](03-tts-backends-english.md) [`10`](10-performance-control.md) |
| §6 | StyleTTS2, YourTTS listed as Tier-1 capable | **Both disqualified** | StyleTTS2: **no weights licence declared at all**. YourTTS: **CC-BY-NC-ND** | [`03`](03-tts-backends-english.md) |
| — | *(missed)* Kokoro-82M | **Second Tier-1 candidate** | `af_heart.pt` = `torch.Tensor (510,1,256)` float32, Apache-2.0 both sides, Hindi included, 54 voice tensors. **And an undocumented free timbre/prosody split: `ref_s[:,:128]`→decoder, `[128:]`→prosody predictor** | [`03`](03-tts-backends-english.md) [`10`](10-performance-control.md) |

### 3.3 Indic

| § | Claim | Verdict | Correction | File |
|---|---|---|---|---|
| §6 | IndicF5 is CC-BY-NC | **Declares MIT — right verdict, wrong route** | An AI4Bharat member confirmed commercial use in writing. **But** [2505.20693](https://arxiv.org/abs/2505.20693) shows IndicF5 = "IN-F5", a fine-tune of **English F5-TTS (CC-BY-NC-4.0)**. GitHub code additionally **unlicensed** | [`04`](04-indic-track.md) [`08`](08-licensing-propagation.md) |
| §6 | Sarvam Bulbul v3 = "best-in-class", the quality ceiling | **Wrong ceiling** | Sarvam's own blog: *"ElevenLabs v3 alpha leads on audio quality."* Bulbul tops only **8 kHz telephony** — the wrong condition for games | [`04`](04-indic-track.md) |
| §6 | Indic Parler-TTS is "description-native", ⭐ the Indic track | **Designer only, and weak** | **Zero speaker-embedding code** (grepped). Identity is a *name token in the same string, same cross-attention* as the emotion word. Description channel = style + a **closed set of 69 (68 unique) names**. No paper experiment mints a novel voice. Scores worst of all systems on InstructTTSEval-EN (RP **28.6**) | [`10`](10-performance-control.md) [`04`](04-indic-track.md) |
| §6 | "Native Hinglish code-switching" | **Half-right** | Mixed-*script* works; **Romanised Hindi degrades output** — Sarvam's docs say so. Needs an IndicXlit pre-pass | [`04`](04-indic-track.md) |
| §8.2 | RASMALAI available for Indic accents | **Never released** | Zero HF results; the 13,000-hr corpus is unobtainable. **But the recipe is published and every input is CC-BY-4.0/MIT — reconstructable.** Highest-leverage dataset action | [`04`](04-indic-track.md) |
| §8.2 | IndicVoices-R: 1,704 h / 10,496 spk / 22 langs / 93.25% extempore | **All four verified** | And **CC-BY-4.0 genuinely permits train-and-release** — the parent paper says the licence was chosen "allowing commercial usage." The brief's worry was misplaced | [`08`](08-licensing-propagation.md) |
| — | *(missed)* Indic-Mio / MioCodec / DhVaani | **Found, then disqualified** | Initially the best Indic Tier-1 hope. **Indic-Mio's own card says it trained on Expresso (CC-BY-NC-4.0)**; base `MioTTS-0.6B` and mandatory `MioCodec` both declare **Emilia** (gate says CC-BY-NC, tag says CC-BY-4.0). DhVaani's base `ZipVoice` has **no licence field** and also used Emilia | [`04`](04-indic-track.md) [`08`](08-licensing-propagation.md) |
| — | *(missed)* SPRING-INX | **Effectively unlicensed** | All 14 datasets: no licence field, no README terms; `Speech-Lab-IITM/SPRING-INX` **does not exist (404)**. Do not use beyond internal evaluation | [`08`](08-licensing-propagation.md) |
| — | *(missed)* IITM IndicTTS | **Not non-commercial — the received wisdom is wrong** | Two EULA versions recovered from Wayback. **V2 (in force ≥Oct 2024) grants perpetual, sub-licensable, royalty-free rights** with no NC and no research-only clause. **But §2.2 forbids your downstream recipients from onward sale** — incompatible with an Apache-2.0 weight release | [`08`](08-licensing-propagation.md) |

### 3.4 Data

| § | Claim | Verdict | Correction | File |
|---|---|---|---|---|
| §8.2 | ParaSpeechCaps ⭐ primary English training set | **Unusable** | **CC-BY-NC-SA-4.0.** Its authors released their own fine-tune as NC-SA | [`05`](05-datasets-and-annotation.md) [`08`](08-licensing-propagation.md) |
| §8.3 | VoicePersona is CC0 | **The declaration does not hold** | HF field says `cc` (a meaningless category tag). **~79% of rows defective:** LAION's Got Talent (52.6%) is **GPT-4o Audio output using OpenAI's eleven proprietary voices** — 483 tarballs named `alloy`/`ash`/…/`verse`; AnimeVox (13.3%) is CC-BY-NC and ripped from official anime dubs; AniSpeech (13.3%) applies literal MIT *software* text to 18.8 GB of anime voices. Only GLOBE_V2 (20.9%) is clean. **CC0 §4(c) expressly disclaims clearing others' rights.** HuggingFace disabled `ESpeech/ESpeech-igm` in May 2026 on a voice actor's complaint about exactly this pattern | [`05`](05-datasets-and-annotation.md) [`08`](08-licensing-propagation.md) |
| §8.3 | VoicePersona: 15,082 samples, 8+ languages, ~500-char descriptions | **All three wrong** | **14,327 rows**, **English-only**, **307-char** average | [`05`](05-datasets-and-annotation.md) |
| §8.3, §15.2 | Skew: 62.6% female, 76.1% twenties → "rebalance" | **Artefact, not measurement** | These are **Qwen2-Audio label distributions, not measurements**. The caption prompt hardcoded `GENDER: [male/female]`, capped age at `fifties+`, and never allowed "neutral". Fix the prompt, not the sampling | [`05`](05-datasets-and-annotation.md) [`09`](09-safety-and-watermarking.md) |
| §8.3 | Data-Speech step 1 extracts F0, rate, jitter, shimmer, HNR, tilt, formants, VTL, SNR | **It computes 9 columns and none of the last six** | Actual: F0 mean/std (`penn`), SNR + C50 + VAD (Brouhaha), speaking rate (`g2p`), optional STOI/SI-SDR/PESQ (SQUIM). **No jitter, shimmer, HNR, tilt, formants, or VTL.** Those must be built | [`05`](05-datasets-and-annotation.md) |
| §8.3 | "Bin each measured value… calibrated against the corpus distribution" | **Bins are equal-width, not percentile** | And pitch is binned **per-speaker, per-gender** | [`05`](05-datasets-and-annotation.md) |
| §8.3 | speaking rate = phones/sec | **IPA-characters/sec** | And it needs **no forced aligner** — in TTV you already know the text | [`05`](05-datasets-and-annotation.md) [`06`](06-evaluation-harness.md) |
| §8.1 | "Speaker count matters more than hours" | **Needs restating** | No published ablation exists for a description→embedding mapper. Nearest evidence: at fixed 100 h, 60× speakers halves ECAPA EER — **but one session each is worse than 100 speakers with many sessions**. Restate as **speaker count × session diversity** | [`05`](05-datasets-and-annotation.md) |
| §8.2 | ParaSpeechCaps finetune: +7.9% consistency MOS, +15.5% naturalness MOS | **Verifies exactly — but incomplete** | Same table shows **intelligibility regressed, WER 4.47 → 8.63** | [`05`](05-datasets-and-annotation.md) |
| §19 | "SpeakerVerse", "VccmDataset" cited | **Neither is a published dataset** | `VccmDataset/` is a directory in the ControlSpeech repo, which has **no LICENSE at all**. "SpeakerVerse" does not exist | [`05`](05-datasets-and-annotation.md) [`13`](13-description-to-embedding-prior-art.md) |

### 3.5 Evaluation, serving, safety

| § | Claim | Verdict | Correction | File |
|---|---|---|---|---|
| §10 | UTMOS/DNSMOS for naturalness | **Would fight the diversity axis** | DNSMOS r = −0.788 with pitch; humans −0.059. Use **TTSDS2** (MIT) | [`06`](06-evaluation-harness.md) |
| §10 | InstructTTSEval's 12 attributes | **Verified exactly, line by line** | But scoring is **binary true/false** judged by `gemini-2.5-pro`, **not locally runnable**, and **real human reference audio scores only 84.3 avg / 67.2 Role-Play**. Treat >85 as noise. ~$12.8/session EN | [`06`](06-evaluation-harness.md) [`13`](13-description-to-embedding-prior-art.md) |
| §10 axis 2 | Compare pairwise cosine with an independent encoder | **Right instinct, no universal threshold exists** | Values are encoder-specific and non-comparable (SpeechBrain default 0.25; real same-speaker SIM-o 0.69–0.76; *different* real speakers 0.67 on another encoder). **Must calibrate in-run** | [`06`](06-evaluation-harness.md) |
| §10 | ~50 descriptions is a sufficient eval set | **Under-powered, and pseudo-replication risk** | Aggregate to 50 description-level units before testing or you pseudo-replicate 20×. At N=50 you detect only **d ≈ 0.40**; the binary judge resolves **±10pp**. Objective re-measurement must be the primary gate | [`06`](06-evaluation-harness.md) |
| §14 | Short game lines behave much worse than paragraphs | **Backwards** | Short prompts show **higher** aggregate throughput at every concurrency ≥8 (112.6 vs 96.0 audio-s/GPU-s at c=32). The dangerous number is the README's single-stream RTF 0.13, which understates the card ~8× | [`07`](07-serving-and-cost.md) |
| §14 | Margin is enormous; meter rendering to cover GPU cost | **Margin holds; the cost model is wrong** | $/min lands **$0.00005–0.00017** vs ElevenLabs $0.075. But marginal cost recovers nothing — the bill is a **fixed warm-GPU floor**. Break-even for a $248/mo warm 4090: **55 h/mo vs ElevenLabs, 368 h/mo vs OpenAI**. Below that, self-hosting is *more* expensive per minute | [`07`](07-serving-and-cost.md) |
| §12.3 | Render cache is "the lever that dominates everything" | **A latency lever, not a cost lever** at this scale | [`07`](07-serving-and-cost.md) |
| §14 | Egress is the sleeper cost | **$0 on R2 — your own choice defuses it** | Residual: serve Opus (0.24 MB/min) not WAV (5.76 MB/min), 24× | [`07`](07-serving-and-cost.md) |
| §12.4 | Cold start 10–60s, mitigate with snapshotting | **Snapshots don't help our case** | Modal's docs: snapshots *"will generally not improve your cold start times — and may even worsen them"* when init is weight-loading-bound. Floor is **~10–30 s**. Irrelevant for batch, **fatal for interactive** | [`07`](07-serving-and-cost.md) |
| §12.2 | Minting is CPU-cheap and free | **Only if Tier 1 is default** | If Tier 2 is default (and it is, per [`11`](11-production-api-landscape.md)), every mint is a GPU render — and §13.2 wants *several candidates per description* | [`11`](11-production-api-landscape.md) [`07`](07-serving-and-cost.md) |
| §17-E2 | Does the watermark survive MP3? | **MP3 is solved (1.00 @32kbps) — wrong worry** | Real failures: **polarity inversion → 0.18/0.00** (an inaudible one-liner), **Opus removes it**, Descript codec 0.00, real reverb 0.22. **OGG/Vorbis survives 0.95** — the Unity/FMOD/Wwise default. Never put a render ID in the payload (16-bit message: 0.39 attribution, 0.69 clean) | [`09`](09-safety-and-watermarking.md) |
| §15.1 | "No impersonation vector at all" — a structural safety advantage | **Does not survive. Restate.** | Correct claim: *"no **cloning** vector — we never accept reference audio."* ParaSpeechCaps trains description→voice on **594 named celebrities**; generated master voices match **69% of women at FAR 1%**; *Midler*/*Waits* make imitation actionable **without copying**; periphrasis defeats name gates at **>90% ASR** | [`09`](09-safety-and-watermarking.md) |
| §15.2 | Watermark at S9, before public exposure | **Already required** | EU Art. 50(2) live since **2 Aug 2026**; Art. 2(12) carves it out of the open-source exemption | [`09`](09-safety-and-watermarking.md) |
| §4.4 | "Do not bake emotion into the identity vector" | **Right goal, false description of the backend** | [2606.05367](https://arxiv.org/abs/2606.05367)'s `full_swap` proves the x-vector **dominates** the codec tokens on Qwen3-TTS Base. Enforce by **discipline**: mint from neutral, store neutral, add `α·τ` at render | [`10`](10-performance-control.md) |
| §17-B5 | Can IndexTTS-2's disentanglement transfer? | **No — and the brief was right to ask** | It is a **Gradient Reversal Layer at training time only**; GRL is a no-op at inference, there is no artefact to lift. Emotion is *summed into* the speaker slot anyway | [`10`](10-performance-control.md) |

---

## 4. THE INDIC TRACK NEEDS A DECISION

**This is the worst news in the research pass, and it deserves to be stated plainly rather than buried.**

The scope document treats Indic as a parallel track at rough parity with English. After research, it is not.

### 4.1 What's left after the licence audit

| Model | Status | Why |
|---|---|---|
| **Indic Parler-TTS** | ✅ servable — **the only one** | Apache-2.0, clean chain, 21 languages |
| Indic-Mio + MioCodec | ❌ NO-GO | Trained on **Expresso (CC-BY-NC)**; base + codec declare **Emilia** |
| DhVaani-0.5 | ❌ NO-GO | Base `ZipVoice` has **no licence field**; also Emilia |
| IndicF5 / SPRING_F5 | ❌ NO-GO | Fine-tunes of **CC-BY-NC F5-TTS** |
| SPRING-INX (data) | ❌ NO-GO | Effectively unlicensed |
| MMS-TTS / SeamlessM4T | ❌ NO-GO | **CC-BY-NC-4.0**, and single-voice-per-language |
| VoxCPM2 | ⚠️ Hindi only | No other Indic language |

### 4.2 And the survivor is structurally weak

**Indic Parler-TTS has zero speaker-embedding code.** Identity is a name token in the same description string, through the same cross-attention, as the emotion word. So:

- **No Tier-1 identity is possible.** No vector exists.
- **Identity and performance share one channel** — changing the emotion word can change the voice. This breaks scope §4.4's core requirement.
- **The voice space is effectively a closed set of 68 names.** No published experiment mints a novel voice.
- It scores **worst of every system tested** on InstructTTSEval-EN (Role-Play 28.6).

### 4.3 The three honest options

| Option | What it means | Cost |
|---|---|---|
| **A. Ship Indic as a curated catalog only** | 68 named voices × 21 languages, seed-clip identity, no custom voice design in Indic. Honest, shippable, useful. | Low. Recommended for v1. |
| **B. Research-lane Indic** | Use IndicF5/Indic-Mio privately for quality benchmarking; never serve them. Ship nothing Indic publicly until a clean Tier-1 option exists. | Low cost, zero Indic product |
| **C. Build the missing piece** | Reconstruct RASMALAI from CC-BY-4.0 inputs (recipe published, all inputs clean), then train an Indic description→voice model. | High — this is a research project, not integration |

**Recommendation: A for v1, C as a funded workstream after English ships.** Do not let the Indic track block the English track — that is what §11's parallel-tracks diagram implies but does not enforce.

**Per-language shippability (option A):** ship 8 (Hindi, Telugu, Bengali, Marathi, Kannada, Malayalam, Odia, Assamese) · conditional 2 (Tamil, Gujarati) · **hold Punjabi** — officially "unofficial", no published quality number, 11 h of training data.

---

## 5. RECOMMENDED STACK

### 5.1 English track

**Qwen3-TTS-12Hz-1.7B-Base** (renderer, Tier-1 vector) + **Qwen3-TTS-1.7B-VoiceDesign** (designer) + **MDN mapper** copied from PromptTTS++ + **AudioSeal** (watermark).

*Three sentences:* Qwen3-TTS is the only Apache-2.0 model whose speaker vector is both externally addressable and the **sole** identity input, and because `enc_dim == hidden_size` that vector lives in the backbone's own jointly-trained embedding space — precisely the "learned space" TacoSpawn found works where bolt-on ASV spaces fail. PromptTTS++ gives a working, Apache-2.0, 256-d reference implementation of the exact description→embedding contract, using an MDN head that is cheaper than flow matching and pairs naturally with the unit-sphere geometry the manifold evidence says is safe. The whole chain — model, weights, training corpora, watermark — is Apache-2.0/MIT/CC-BY with no upstream defect found, which after five discovered licence traps is worth more than a marginal quality edge.

**Caveat carried forward:** Qwen3-TTS Base has **no instruct channel in its API**. Per-line Direction comes from [2606.05367](https://arxiv.org/abs/2606.05367)'s training-free emotion direction vectors: `x_new = x(target,neutral) + α·τ_emo`, ΔEECS +0.29 at SECS_W 0.912, **τ averaged over ≥4 speakers** (single-speaker τ leaks source timbre: 0.810). ~64 KB artefact, an afternoon's work.

### 5.2 Indic track

**Indic Parler-TTS as a curated catalog**, seed-clip identity, 8 confirmed languages, no custom voice design in v1.

*Three sentences:* It is the only Indic model that survives the licence audit, and it turns a description into speech — but it has no speaker vector, so Tier-1 identity is impossible and voice and style share one channel. Shipping its 68 named voices as a curated, seed-clip-backed catalog is honest, immediately useful, and does not over-promise. Custom Indic voice design is a separate research project gated on reconstructing RASMALAI from its CC-BY-4.0 inputs — worth doing, but not on the English track's critical path.

### 5.3 Everything else

| Layer | Choice | Licence |
|---|---|---|
| Watermark | **AudioSeal** | MIT (code + weights) |
| Naturalness | **TTSDS2** (`pip install ttsds`) | MIT |
| Speaker verification (eval) | `pyannote/wespeaker-voxceleb-resnet34-LM` | Apache-2.0 / CC-BY-4.0 |
| Diversity | **Vendi score** + **GVD** | MIT |
| ASR (EN) | `whisper-large-v3` | Apache-2.0 |
| ASR (Indic) | `ai4bharat/indic-conformer-600m-multilingual` | MIT |
| F0 | `penn` (FCNF0++) | MIT |
| SNR/C50/VAD | Brouhaha | MIT |
| Timbre encoder (candidate) | `amphion/naturalspeech3_facodec` | Apache-2.0 |
| Corpora | LibriTTS-P, LibriTTS-R, GLOBE, MLS, Common Voice, VCTK, IndicVoices-R, Rasa | CC-BY-4.0 / CC0 |
| DB | Postgres + pgvector (HNSW) | — |
| Object storage | Cloudflare R2 (zero egress) | — |
| First GPU | RunPod Community RTX 4090, $0.34/hr ≈ **$248/mo warm** | — |

---

## 6. WHAT CHANGES IN THE PLAN

Four structural changes to scope §11. Full detail in [`GRAND-PLAN.md`](GRAND-PLAN.md).

1. **Dataset work moves from S4 → S0/S1.** ParaSpeechCaps is gone; the corpus is now an open question, and S2 cannot start until it is answered. This is the single biggest sequencing change.
2. **Compliance moves from S9 → S0.** EU Art. 50(2) is already live. Watermarking, disclosure, and provenance logging are day-one, not pre-launch.
3. **A new S0 experiment block, before anything is built.** Six experiments, all ≤1 day, that together de-risk the entire architecture. Two of them (`E0` direction vectors, `E3` per-dimension rescaling) are afternoon-scale and unblock everything downstream.
4. **The Indic track gets its own go/no-go gate** rather than running as an assumed parallel.

---

## 7. OPEN AFTER RESEARCH — settle by experiment, not by reading

The ten questions that could not be resolved from public sources, in priority order. Each has a specified experiment in its source file.

| # | Question | Experiment | Cost | Blocks |
|---|---|---|---|---|
| **E0** | Do training-free emotion direction vectors work on Qwen3-TTS Base? | Build τ over ≥4 speakers; measure ΔEECS and SECS_W | **1 afternoon** | The entire `Direction` channel |
| **E1** | Is Qwen3-TTS's `(2048,)` space a *learned* space (good) or a *d-vector* space (bad)? | 500 speakers → GMM → 200 samples → **s2s / g2s / g2g** per TacoSpawn §6.2 | ~4 GPU-h | **Everything.** The most important experiment in the project |
| **E2** | How large is vocoder drift on Qwen3-TTS? | 100 identities: render → re-extract → cosine(in,out) | ~2 GPU-h | Whether Tier 1 needs a closed loop; the mint cost model |
| **E3** | Are the embedding dimensions heterogeneously scaled? | Dump 1,000 real embeddings; plot per-dim min/max/σ | **~20 min, CPU** | The mapper's output layer. **Do this first** |
| **E4** | What are `C_same` / `C_diff` for the eval encoder? | 50 speakers × 20 utterances; intra/inter cosine distributions | ~1 GPU-h | Every identity-consistency threshold |
| **E5** | Does AudioSeal survive 1–3 s game-dialogue clips? | Fork Sony's `raw_bench`, sweep duration | ~2 GPU-h | Whether watermarking is viable at all for our output |
| **E6** | Does sampling PromptTTS++'s MDN give diverse voices from one prompt? | 20 samples from one prompt → render → Vendi + GVD | ~1 GPU-h | The core A3 question, on real weights |
| **E7** | Minimum viable speaker count × session diversity | 4 speaker arms × 2 session conditions × 3 seeds, cached `z` | **<10 GPU-h** | Corpus sizing; how much of GLOBE to annotate |
| **E8** | Real $/min on short lines for our actual stack | Benchmark sweep, spec in [`07`](07-serving-and-cost.md) §9 | **<$5, one afternoon** | The entire cost model |
| **E9** | Does identity drift under *heavy stylisation* (aged/raspy/whispered)? | Sweep stylisation intensity, measure SECS | ~3 GPU-h | Whether §5's voice range is reachable. **Nobody has measured this** |

**Plus two that are not experiments:**

- **Counsel question #1:** does India's "prominently prefixed audio disclosure" apply to a 1.5-second game dialogue line delivered via API to a developer? This is potentially product-destroying and has no technical workaround.
- **Free intelligence:** **VPC 2026 results land 2026-09-26.** Re-check — it will contain the current best answer to "how do you generate a good synthetic speaker vector."

---

## 8. Confidence and caveats on this verdict

- Every factual claim traces to a primary source cited in one of the 13 domain files. Where a claim could not be verified it is marked **UNVERIFIED** there, and those are not laundered into confidence here.
- **Two domain files disagree in one place** and it is flagged in situ: [`03`](03-tts-backends-english.md) calls VoxCPM2's parenthetical control a "demo hack"; [`10`](10-performance-control.md) reads the source and finds it is the official API. Take [`10`](10-performance-control.md)'s reading — it read the API surface directly.
- **All throughput and cost figures are derived from published benchmarks, not measured on our stack.** E8 replaces them.
- **All licence readings are engineering research, not legal advice.** [`08`](08-licensing-propagation.md) marks every NEEDS-LAWYER row.
- **Nothing here has been run.** This pass verified what is true in the literature and the repos. The ten experiments in §7 are what convert that into knowledge about *our* system.

---

*Pass 1 · 2026-09-02 · 12 parallel domain researchers · primary sources only · corrections scored above confirmations.*
