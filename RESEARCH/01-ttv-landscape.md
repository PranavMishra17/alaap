# 01 — The Text-to-Voice Field & the Generative Mapper Head

> **Domain:** TTV research landscape; the one-to-many problem; generative head selection
> **Answers:** A3 (decisive), B3
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [02-identity-representation.md](02-identity-representation.md) · [05-datasets-and-annotation.md](05-datasets-and-annotation.md) · [06-evaluation-harness.md](06-evaluation-harness.md)

## 0. Bottom line

- **All seven arXiv IDs in the brief resolve to real papers with the claimed names.** The brief's own suspicion ("several look suspicious") was unfounded. 2608.13613 = VoiceDesigner, 2603.28086 = MOSS-VoiceGenerator, 2601.10629 = VoiceSculptor all exist and match. **[HIGH]**
- **The single most important result in this whole field for Alaap is arXiv 2406.08812 (Chen et al., Interspeech 2024)** — the only published head-to-head of a *regression* mapper vs a *flow-matching* mapper on the identical two-tower setup. It shows the tradeoff is real and bidirectional: flow matching wins fidelity (FAD 3.559 vs 5.244) but **loses** attribute adherence (SRCC 0.60 vs 0.74) and speaker similarity (0.36 vs 0.41). The **hybrid** (discriminative + flow) got the best FAD (3.126). **[HIGH]**
- **This is a correction to the project's framing.** The brief treats "MSE → conditional mean → mush" as settled, and it is directionally right, but the published evidence says a pure generative head *costs you adherence*. The answer is not "replace MSE with a flow"; it is "flow head + an adherence-preserving auxiliary term, with a sampling-temperature dial." **[HIGH]**
- **VoiceSculptor's Apache-2.0 claim is almost certainly ineffective for the weights.** The repo LICENSE.txt really is Apache-2.0 and the README really does grant it over "codes and model weights" — but VoiceSculptor-VD is fine-tuned from `HKUSTAudio/Llasa-3B` (**CC-BY-NC-4.0**, whose card says it "prohibits free commercial use") and inference requires `HKUSTAudio/xcodec2` (**CC-BY-NC-4.0**). You cannot relicense a NC derivative as Apache-2.0. **This blocks VoiceSculptor for public hosting.** **[HIGH on facts, MEDIUM on legal conclusion]**
- **VoiceSculptor is NOT current SOTA on InstructTTSEval-Zh.** It claimed *open-source* SOTA in Jan 2026, but **Qwen3-TTS-VoiceDesign** (Apache-2.0, open weights, released 2026-01-22) scores **84.3 / 82.9 / 77.4** (APS/DSD/RP) on the ZH split vs VoiceSculptor's **75.7 / 64.7 / 61.5**, per MOSS-VoiceGenerator's own comparison table. **[HIGH]**
- **VoiceDesigner has no code and no weights**, is CC-BY-4.0 on the *paper only*, and is **not a two-tower design** — it is a 1.0B end-to-end MM-DiT with no exportable speaker vector. Its value to Alaap is its **DSP augmentation recipe**, which is directly reusable and independently validates the stylization strategy. **[HIGH]**
- **Two published param budgets for the mapper bracket the brief's 10–50M estimate and land at its low end:** HiStyle uses ~30M per stage (12 layers × 512 hidden), Deep Dubbing's Text-to-Timbre uses a 4-layer DiT (4 heads, 392 hidden — order 10M). The brief's budget is correct but should be read as **10–30M, not 50M**. **[HIGH]**
- **Mode collapse IS measurable and one paper already publishes the right metric:** PromptSpeaker's **`gen2gen-near`** = nearest-neighbour cosine distance between speakers generated from the *same* prompt (they report 0.088). UniSpeaker publishes **SSD** (Speaker Similarity Diversity, lower = more diverse). Adopt both. **[HIGH]**
- **The contrastive/retrieval-first plan has direct prior art** (Speaker-Text Retrieval arXiv 2312.06055; UniSpeaker's KV-Former + soft contrastive loss; HiStyle's contrastive alignment stage). UniSpeaker is the important one: it explicitly frames soft contrastive labels as the *mechanism for preserving one-to-many*, and it is the only system that reports a diversity metric alongside adherence. **[HIGH]**
- **Nobody in this field trains on Indian languages.** Qwen3-TTS covers 10 languages, none Indian. VoiceSculptor/MOSS are ZH+EN. This is an unfilled gap and a genuine risk to the project's stated scope — no off-the-shelf description→voice system covers it. **[HIGH]**

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| 1 | "Several arXiv IDs look suspicious" | **WRONG** | All 7 resolve correctly to the named systems. No fabricated IDs. | arXiv abs pages (all 7) | HIGH |
| 2 | VoiceSculptor "full open-source incl. pretrained weights" | **PARTLY WRONG** | Code + VD weights are published under an Apache-2.0 file, but derive from Llasa-3B (CC-BY-NC-4.0) and depend on xcodec2 (CC-BY-NC-4.0). The permissive grant is upstream-invalid. Unusable for public hosting. | repo LICENSE.txt; HF cards for Llasa-3B and xcodec2 | HIGH (facts) / MEDIUM (legal) |
| 3 | VoiceSculptor "SOTA on InstructTTSEval-Zh" | **STALE** | Was open-source SOTA Jan 2026. Superseded by Qwen3-TTS-VoiceDesign (Apache-2.0). Gemini-TTS-Pro leads overall. | arXiv 2603.28086 comparison table | HIGH |
| 4 | VoiceDesigner "claims a diffusion transformer" | **CORRECT** | 1.0B single-stream MM-DiT, flow-matching objective, DAC-VAE latents at 48 kHz. | arXiv 2608.13613 | HIGH |
| 5 | VoiceDesigner "DSP + generative augmentation targeting fictional voices" | **CORRECT AND UNDERSTATED** | Pitch/formant shift, reverb, EQ, band-pass, DRC, SiFi-GAN pitch-contour manipulation → dragons, demons, robots, miniature creatures. Directly reusable. | arXiv 2608.13613 §data | HIGH |
| 6 | Implied: VoiceDesigner is a usable component | **WRONG** | No code, no weights, no repo, no release commitment. Demo page only. Paper is CC-BY-4.0; that licenses the *paper*, not a model. | voicedesigner-demo.github.io; arXiv 2608.13613 | HIGH |
| 7 | Implied: VoiceDesigner supports the two-tower/Tier-2 seed-clip thesis | **NO** | It is end-to-end; no separable, exportable speaker vector. Identity persists only via reference *audio* (clone/edit mode) — which is a seed-clip argument, but not a vector argument. | arXiv 2608.13613 | HIGH |
| 8 | PromptSpeaker is "the canonical two-tower formulation" | **CORRECT** | BERT → FFT×2 → GRU → 10 tokens → linear → Gaussian prior → 12-block Glow → 256-d speaker vector → frozen zero-shot VITS. Exactly the A→B→C shape. | arXiv 2310.05001 | HIGH |
| 9 | Implied: PromptSpeaker is buildable from released artifacts | **NO** | Demo site only. No repo, no weights, no param counts published. | promptspeaker.github.io/demo | HIGH |
| 10 | Framing: "generative head fixes the one-to-many problem" | **INCOMPLETE** | The only published A/B shows the generative head *trades* adherence for diversity/fidelity. Needs a hybrid or an adherence-preserving auxiliary loss. | arXiv 2406.08812 Tables 1–3 | HIGH |
| 11 | Mapper budget "~10–50M params" | **HIGH END TOO HIGH** | Published working mappers are ~10M (Deep Dubbing TTT) and ~30M (HiStyle per stage). Budget 10–30M. | arXiv 2509.25842; 2509.15845 | HIGH |
| 12 | Unispeaker as a *generative* head candidate | **MISCATEGORISED** | It is contrastive/attention-retrieval (KV-Former + soft contrastive loss), not a generative sampler. It belongs in the *retrieval* bucket — which makes it the best prior art for the planned S2 implementation, not a rival to the flow head. | arXiv 2501.06394 | HIGH |
| 13 | HiStyle as a distinct non-generative alternative | **WRONG** | HiStyle IS generative — two stacked conditional **diffusion** predictors (~30M each) plus a contrastive alignment objective. It is a hybrid, and its ablation is the closest thing to the recommended design. | arXiv 2509.25842 | HIGH |

---

## 2. Paper existence audit

**Result: 7/7 resolve. Zero fabricated IDs. The brief's identifiers are clean.**

| Claimed work | arXiv ID in brief | Resolves? | Actual title / authors / date | Confidence |
|---|---|---|---|---|
| PromptSpeaker | 2310.05001 | **YES — exact match** | *PromptSpeaker: Speaker Generation Based on Text Descriptions* — Zhang, Liu, Lei, Chen, Yin, Xie, Li — 2023-10-08 | HIGH |
| HiStyle | 2509.25842 | **YES — exact match** | *HiStyle: Hierarchical Style Embedding Predictor for Text-Prompt-Guided Controllable Speech Synthesis* — Zhang, Li, Hu, Li, Xie — 2025-09-30 | HIGH |
| VoiceDesigner | 2608.13613 | **YES — exact match** | *VoiceDesigner: Text-to-Voice Generation and Editing via Unified Diffusion Modeling and Data Augmentation* — Hai, Thakkar, Chen, Wang, Su, Kumar, Elhilali, Jin — 2026-08-12 | HIGH |
| MOSS-VoiceGenerator | 2603.28086 | **YES — exact match** | *MOSS-VoiceGenerator: Create Realistic Voices with Natural Language Descriptions* — Huang, Fan, Jiang et al. (Qiu group) — 2026-03-30 | HIGH |
| UniSpeaker | 2501.06394 | **YES — exact match** | *UniSpeaker: A Unified Approach for Multimodality-driven Speaker Generation* — Sheng, Du, Lu, Zhang, Ling — 2025-01-11 | HIGH |
| Listener Impressions | 2406.08812 | **YES — exact match** | *Generating Speakers by Prompting Listener Impressions for Pre-trained Multi-Speaker TTS Systems* — Chen, Liu, Cooper, Yamagishi, Qian — 2024-06-13, Interspeech 2024 | HIGH |
| VoiceSculptor | 2601.10629 | **YES — exact match** | *VoiceSculptor: Your Voice, Designed By You* — Hu, Chen, Ma, Guo et al. (ASLP lab, 22 authors) — 2026-01-15 v1, v2 2026-01-20 | HIGH |
| Parler-TTS / Lyth & King | (not given) | **YES** | *Natural language guidance of high-fidelity text-to-speech with synthetic annotations* — Lyth & King — arXiv 2402.01912, 2024-02-02 | HIGH |

**Additional systems found that the brief does not mention and should:**

| System | ID / URL | Why it matters | Confidence |
|---|---|---|---|
| **Qwen3-TTS-VoiceDesign** | huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | Apache-2.0, open weights, 1.7B, current open-source InstructTTSEval leader. The strongest *legally usable* baseline in the field. | HIGH |
| **Deep Dubbing (Text-to-Timbre)** | arXiv 2509.15845 | A published two-tower description→**speaker-embedding** module using OT-CFM flow matching into Cam++ 192-d space. This is the closest published analogue to Alaap's intended mapper. | HIGH |
| **Speaker-Text Retrieval via Contrastive Learning** | arXiv 2312.06055 | Direct prior art for the planned S2 retrieval-first approach. Liu, Wang, Cooper, Miao, Yamagishi, 2023-12-11. EN + JA. | HIGH |
| **Sub-Center Speaker Embeddings** | arXiv 2407.04291 | Argues recognition-trained speaker embeddings *destroy* intra-speaker variance — i.e. the target space B may itself be a diversity bottleneck. Publishes intra/inter-class variance ratio as the metric. | HIGH |

---

## 3. The one-to-many problem: mechanisms compared

### 3.1 PromptSpeaker (2310.05001) — normalizing flow, canonical two-tower

**Mechanism.** BERT → 2 FFT blocks (hidden 256, filter 1024) → GRU (hidden 256) → 10-token layer → linear head emitting **mean and variance of a Gaussian semantic prior**. A **Glow** model (12 flow blocks: actnorm, invertible 1×1 conv, affine coupling with kernel 1, 4 layers each) maps a *sample* from that prior to a 256-d speaker vector. That vector drives a **frozen zero-shot VITS**.

**How it beats the mean.** Sampling happens in the prior; the flow is a bijection, so it cannot collapse the distribution to a point the way an MSE regressor does. Diversity is a property of the prior's learned variance.

- Two-tower: **YES** — textbook. Speaker vector is separable and reusable.
- Mapper params: **NOT PUBLISHED.** Only dims (256/256) and depth (12 blocks). **UNVERIFIED.**
- Data: 74 internal stylistic speakers (100 samples each, 13 annotators) + AISHELL-3 (218 spk × 20) + DiDiSpeech (500 spk × 20). VITS pre-trained on 500 h / 250 spk. Small — under 1000 speakers total.
- Weights: **NOT RELEASED.** Demo page only.
- CFG: **not used, not mentioned.**
- **Diversity metric — the valuable bit:** `gen2gen-near` = 0.088, the cosine distance between the two *closest* speakers generated from the **same prompt** (d-vectors from WeSpeaker). Also `gen2syn-near` 0.085 vs `syn2syn-near` 0.113 vs `syn2syn-same` 0.024, used to argue generated speakers are genuinely novel rather than training-set copies.
- Results: gender accuracy 96% (vs 46% baseline), speaker-timbre MOS 3.23±0.20 (vs 1.87 baseline), naturalness MOS 3.53±0.09 (GT 4.25).

**Verdict for Alaap:** the architecture is right and the evaluation methodology is the single most reusable artifact in this paper. But there is nothing to clone — it must be reimplemented. Confidence **HIGH**.

### 3.2 Listener Impressions (2406.08812) — THE decisive experiment

This is the paper the project should read end-to-end. It builds *both* heads on one two-tower substrate and compares them.

**Discriminative head:** RoBERTa (+LoRA rank 8, AdapterHub) → [CLS] → 4-layer linear projection → speaker embedding. Loss = **L2 + cosine**. This is precisely the failure mode the brief describes.

**Generative head:** **Conditional Flow Matching** over speaker embeddings conditioned on the prompt, sampling from a Gaussian prior at inference. The authors state the problem in their own words — that prompt↔speaker-embedding is not one-to-one and a single prompt can describe different speakers — and argue FM captures the multimodal distribution instead of averaging it.

**The numbers.**

| System | FAD ↓ | Naturalness MOS | Attribute SRCC ↑ (avg, seen) | Speaker similarity ↑ |
|---|---|---|---|---|
| Discriminative, no LoRA | 11.217 | 3.15 ± 0.25 | — | 0.42 |
| Discriminative + LoRA | 5.244 | 3.45 ± 0.19 | **0.74** | **0.41** |
| **Flow Matching + LoRA** | 3.559 | **3.52 ± 0.26** | 0.60 | 0.36 |
| **Discriminative + Flow Matching** | **3.126** | 3.50 ± 0.24 | 0.37 | 0.37 |

Per-attribute, discriminative dominates on the ones users actually name: pitch 0.94 vs 0.86, voice depth 0.87 vs 0.76.

**Reading it honestly.** The generative head buys *distributional realism* (FAD nearly halved) and naturalness. It **costs** ~19% relative attribute adherence and ~12% relative speaker similarity. The authors' conclusion is explicitly hybrid: combining both lets the system "capture speaker-related information from prompts better and generate speech with higher fidelity."

- Two-tower: **YES**, and explicitly motivated as such — the prompt module is decoupled so it can swap into different pre-trained multi-speaker TTS by replacing only the speaker encoder. This is independent validation of Alaap's core architecture from a Yamagishi-lab paper. **[HIGH]**
- Params: not published beyond RoBERTa-base + LoRA r=8 + 4-layer projection. **UNVERIFIED.**
- Data: Corpus of Spontaneous Japanese, 2,672 training speakers / 30 eval. Prompts derived from CSJ's existing **26-question listener impression survey** rather than hand-written annotations — a cheap annotation trick worth stealing (see 05-datasets).
- Weights/code: **not released.** **[HIGH]**
- CFG: not used.
- Diversity metrics: FAD (reference-free fidelity), SRCC of 9 attributes rated by 100 native listeners, EMD over MOS distributions in the appendix.

### 3.3 HiStyle (2509.25842) — stacked diffusion + contrastive, and the best param reference

**Mechanism.** Two sequential **conditional diffusion** predictors over transformer blocks. Stage 1: text-prompt embedding → speaker (timbre) embedding. Stage 2: predicted speaker embedding fused by **residual connection** with the prompt embedding → fine-grained style embedding. Text encoder is frozen BERT + linear projection. A **contrastive cosine-similarity objective** (positives = predicted/reference match, negatives = other batch samples) aligns the text and audio embedding spaces.

The hierarchy is motivated by an empirical observation worth carrying into 02-identity-representation: style embeddings **cluster hierarchically — timbre first, then finer style attributes.** That is an argument for a two-level identity vector.

- Two-tower: **YES** (it predicts embeddings consumed by a separate synthesiser).
- Mapper params: **~30M per stage** — 12 layers, hidden 512. Two stages ≈ 60M total, or 30M if you only need timbre. **This is the most directly citable budget number in the field.** **[HIGH]**
- Data: 2,000 h expressive speech, **only ~20 distinct timbres.** Note the mismatch — dense style coverage, very sparse *speaker* coverage. HiStyle is really a style predictor, not a speaker generator.
- Weights: **not released.** Anonymous 4open.science samples link implies it was under double-blind review. **[HIGH]**
- CFG: not mentioned.
- Diversity: **not measured.** Evaluation is controllability accuracy + Style-MOS only.

**Ablation table (this is the closest published head-comparison after 2406.08812):**

| Approach | Speed acc | Volume | Pitch | Fluctuation | Style-MOS |
|---|---|---|---|---|---|
| **HiStyle (2-stage diffusion + contrastive)** | **90.98** | **95.56** | **92.87** | **88.02** | **3.71** |
| Text prompt only | 89.21 | 85.32 | 85.69 | 82.32 | 3.45 |
| Discriminative model | 85.21 | 88.43 | 90.65 | 83.65 | 3.48 |
| Variation network | 92.56 | 93.33 | 88.21 | 86.58 | 3.52 |
| Query encoder | 90.48 | 91.58 | 91.86 | 83.31 | 3.68 |

Note the discriminative baseline is *not catastrophically* worse here (85–91% vs 88–96%). The mush failure is real but it is a quality gradient, not a cliff — consistent with the brief's warning that it "fails quietly." **PromptSpeaker is cited as [17] but never compared against.** No flow-based baseline appears. **[HIGH]**

### 3.4 UniSpeaker (2501.06394) — contrastive, not generative; the diversity-metric source

**Mechanism.** A **KV-Former** "unified voice aggregator": learnable key-value vectors cross-attended by multimodal (text / face / speech) queries, retrieving the most informative representation in a shared voice subspace, which then conditions an existing CFM synthesiser.

**How it handles one-to-many — explicitly and by design:**
1. **Soft contrastive loss (SoftCL)** uses intra-modal relationships as *soft labels* rather than hard one-to-one pairs, on the stated grounds that the mapping from description to voice characteristics **is** one-to-many.
2. **Speech-anchoring:** reference speech is provided with 50% probability in training, which enables alignment across modalities without needing parallel data in every pair.

- Two-tower: **YES** — produces a unified speaker embedding consumed by a separate CFM model.
- Params: **NOT DISCLOSED. UNVERIFIED.**
- Data: ~1,000 h across LRS3-TED, LibriTTS-P, VCTK-R, plus an internal speaker-description set.
- Weights/code: **not released** — samples page only; paper is CC-BY-4.0. **[HIGH]**
- CFG: not used.
- **Diversity metric: SSD (Speaker Similarity Diversity)** — speaker similarity between speech generated from *different* speakers' descriptions; lower = more diverse. UniSpeaker reports significantly lower SSD than baselines, with t-SNE showing a "significantly richer" voice-space distribution. Also introduces the **MVC benchmark** (5 tasks; 600 faces, 600 text descriptions, 200 editing samples, all unseen) with a clean three-axis metric design: **suitability (SST/SSC/MOS-Match) · diversity (SSD) · quality (WER/MOS-Nat)**. Copy this structure into 06-evaluation-harness. **[HIGH]**

### 3.5 VoiceDesigner (2608.13613) — see §4 for full B3 treatment

Not a two-tower system. 1.0B single-stream **MM-DiT**, flow-matching over DAC-VAE latents, conditioned jointly on transcript (F5-TTS tokenizer), instruction (T5Gemma-XL), and audio reference via token-level AdaLN and 3D-RoPE. One-to-many is handled only *implicitly* — different noise seeds give different outputs. **No diversity metric is reported and mode collapse is never discussed.**

### 3.6 MOSS-VoiceGenerator (2603.28086) — autoregressive discrete tokens

**Mechanism.** Decoder-only causal LM over **MOSS-Audio-Tokenizer** codec tokens with delay-pattern generation; voice description concatenated with target text. Next-token cross-entropy — **no flow, no diffusion.** One-to-many is handled by the sampler (temperature/top-p), which is the standard AR answer.

- Two-tower: **NO.** End-to-end.
- Params: **1.7B released** (8B also trained), initialised from Qwen3.
- Data: ~25,000 h (ZH 18,025 h / EN 7,047 h). Phase 1 ≈ 5,000 h **cinematic** (film/TV/episodic); Phase 2 ≈ 10,000 h style-mined from internal TTS corpora + ~10,000 h crowdsourced dubbing. DNSMOS ≥ 3.0 filtering.
- Weights: **YES — `OpenMOSS-Team/MOSS-VoiceGenerator`, Apache-2.0, ~2.4B params BF16.** Base is Qwen3 (Apache-2.0), so the licence chain is **clean**. **[HIGH]**
- CFG: not mentioned.
- **Diversity — the one useful scaling datum in the field:** they chose the 1.7B over the 8B because it "achieves comparable instruction-following quality to the 8B model, **while demonstrating better generation diversity**." Bigger was *less* diverse. Consistent with the mode-collapse thesis and an argument for keeping Alaap's mapper small. **[HIGH]** — though no quantitative diversity number backs the claim, only a qualitative sunburst over age/emotion/texture. **[MEDIUM]**

### 3.7 VoiceSculptor (2601.10629) — see §4 for full B3 treatment

Discrete-token LLM (LLaSA-3B / XCodec2), **no flow or diffusion**, with Chain-of-Thought attribute tokens. Design-then-clone: generates a prompt waveform, then CosyVoice2 clones from it. **Independent validation of the Tier-2 seed-clip identity design** — this is the brief's strongest correct intuition. **[HIGH]**

### 3.8 Parler-TTS (2402.01912) — the honest negative control

Lyth & King's contribution is the **data recipe**, not the head: automatic annotation of speaker characteristics and recording conditions across 45,000 h, so that description-conditioning becomes learnable at all. Conditioning is cross-attention on description embeddings inside an end-to-end model. **No separable speaker vector, no explicit distributional head** — description → audio directly, one-to-many left entirely to the AR sampler. Checkpoints: Mini v1 880M, Large v1 2.3B, **Apache-2.0**, repo actively maintained. **[HIGH]**

Worth being blunt: Parler-TTS is the archetype of the thing Alaap is *not* building. It is useful as a licence-clean baseline and as an annotation-pipeline reference, not as an architecture.

### 3.9 Summary matrix

| System | Head mechanism | Two-tower? | Mapper params | Data | Weights | Licence | CFG dial | Diversity metric |
|---|---|---|---|---|---|---|---|---|
| PromptSpeaker | **Glow (normalizing flow)** | **YES** | Unpublished (256-d, 12 blocks) | ~790 spk, 500 h VITS | No | — | No | **gen2gen-near** ✅ |
| Listener Impressions | **Flow matching** + discriminative | **YES** | Unpublished (RoBERTa+LoRA r8) | CSJ, 2,672 spk | No | — | No | FAD, SRCC, EMD |
| HiStyle | **2× conditional diffusion** + contrastive | **YES** | **~30M/stage** | 2,000 h, ~20 timbres | No | — | No | None |
| UniSpeaker | **Contrastive (KV-Former, SoftCL)** | **YES** | Unpublished | ~1,000 h | No | CC-BY-4.0 (paper) | No | **SSD** ✅ |
| VoiceDesigner | Flow-matching MM-DiT (implicit) | NO | 1.0B (whole model) | 56,165 h + DSP aug | **No** | CC-BY-4.0 (paper) | No | None |
| MOSS-VoiceGenerator | **AR discrete tokens** | NO | 1.7B | ~25,000 h | **YES** | **Apache-2.0** ✅ | No | Qualitative only |
| VoiceSculptor | **AR discrete tokens + CoT** | Partly (seed clip) | 3B (card says 4B) | 9,000 h CPT | **YES** | **Apache-2.0 — CONTESTED** ⚠️ | No | None |
| Qwen3-TTS-VD | AR discrete tokens | NO | 1.7B | Undisclosed | **YES** | **Apache-2.0** ✅ | No | None |
| Deep Dubbing TTT | **OT-CFM flow matching** | **YES** | 4-layer DiT, 392 hidden (~10M) | 4,000 h, 300K descriptions | No | — | No | None |
| Parler-TTS | AR, cross-attention | NO | 880M / 2.3B | 45,000 h | **YES** | **Apache-2.0** ✅ | No | None |

**The single most striking pattern: not one system in this field exposes classifier-free guidance as an adherence-vs-diversity dial.** CFG is not mentioned in any of the eight papers. If Alaap ships that dial, it is genuinely novel — and given §4's evidence that the tradeoff is real and steep, it is also the *right* product feature. **[HIGH — verified by absence across all 8 papers; MEDIUM that no unread appendix mentions it.]**

---

## 4. Head-to-head evidence

### 4.1 B3 — How VoiceDesigner and VoiceSculptor actually work

**VoiceDesigner (Adobe-affiliated author list: Su, Kumar, Jin + JHU's Hai, Thakkar, Elhilali).**

Architecture: single-stream **MM-DiT**, 1.0B params, operating on **DAC-VAE** latents at 48 kHz / 25 Hz latent frame rate. Three conditioning modalities share one token space: transcript (F5-TTS tokenizer), instruction (**T5Gemma-XL** encoder), audio reference. Injection via token-level **AdaLN** + **3D-RoPE**. A separate sentence-level duration predictor initialised from **Qwen3-0.6B** with a Dasheng encoder.

Training data: 56,165 h pretraining (Emilia, Common Voice, LibriTTS-R, HiFiTTS-2-44.1k), then fine-tuning on ESD, RAVDESS, SAVEE, Expresso, EARS, CapSpeech-Agent, VCTK, DreamVoice, plus **an internal 16-hour character-voice set: 20 character identities performed by 12 professional voice actors.**

**The DSP augmentation pipeline — the reusable part.** Pitch shifting, formant shifting, reverberation, EQ, band-pass filtering, dynamic range compression, and pitch-contour manipulation via **SiFi-GAN**, chained to turn ordinary speech into "dragons, demons, possessed entities, robots, astronauts, and miniature creatures." This is a cheap, licence-free, training-data-free recipe for the heavy-stylization half of Alaap's scope, and it can be applied at *inference* as post-processing even without VoiceDesigner's model. **[HIGH]**

Results: Style-ACC **0.66** (best; vs Qwen3-TTS-VoiceDesign 0.50), WER 1.22%, MOS-C 3.93±0.06 (vs ElevenLabs-TTV 4.00±0.06 — i.e. **it does not beat ElevenLabs on MOS**). Cloning SIM-o 0.757 (vs CosyVoice-3 0.718, F5-TTS 0.670). Editing MOS-E 4.129±0.054 vs Step-Audio-EditX 3.333±0.064. Benchmarks: TTV-Traits (75 samples / 15 emotions) and TTV-Character (150 samples / 50 characters).

**Weights genuinely released? NO.** No GitHub repo, no HuggingFace, no release commitment anywhere in the paper or on the demo page. CC-BY-4.0 covers the arXiv paper. **[HIGH]** — To flip this verdict, the thing to check is a repo appearing under the Adobe or JHU CLSP org after camera-ready.

**VoiceSculptor (ASLP lab, NWPU — same Lei Xie group as PromptSpeaker and HiStyle).**

Two-stage **design-then-clone**:
1. **Voice Design (VD):** LLaSA-3B (Llama-3.2-3B-Instruct base) generates discrete **XCodec2** tokens representing the target voice, with **Chain-of-Thought fine-grained attribute tokens** (pitch, rate, loudness, emotion, style) emitted as auxiliary intermediate outputs. Joint cross-entropy over text and speech tokens. Stochastic attribute-token dropout p=0.2. Retrieval augmentation over **500K in-domain instructions** in a Milvus vector DB.
2. **Voice Cloning (VC):** the designed voice is rendered to a prompt waveform and fed to **CosyVoice2** for high-fidelity timbre transfer.

Data ladder: SFT 1,000 h → 3,700 h → 4,000 h → CPT 9,000 h (adding VoxBox emotion-filtered samples). Pipeline: denoise, VAD, multi-speaker detection, ASR (FireRedASR/Whisper/SenseVoice), forced alignment, multi-level annotation, human verification.

**Weights genuinely released? YES — with a licence problem.**

| Artifact | Stated licence | Reality |
|---|---|---|
| `github.com/ASLP-lab/VoiceSculptor` LICENSE.txt | **Apache License 2.0** (verbatim standard text) | ✅ verified |
| README grant | "We use the Apache 2.0 license. Researchers and developers are free to use the codes **and model weights**" | ✅ verified verbatim |
| `ASLP-lab/VoiceSculptor-VD` HF card | Apache-2.0; 4B params BF16, safetensors present | ✅ weights real |
| **Base: `HKUSTAudio/Llasa-3B`** | **CC-BY-NC-4.0** — card states it "prohibits free commercial use because of ethics and privacy concerns" | ⚠️ **conflict** |
| **Required: `HKUSTAudio/xcodec2`** | **CC-BY-NC-4.0** | ⚠️ **conflict** |
| CosyVoice2 (cloning stage) | Apache-2.0 | ✅ clean |

**Conclusion: the Apache-2.0 grant over VoiceSculptor-VD's weights is very likely ineffective.** A fine-tune of a CC-BY-NC-4.0 model is a derivative work and inherits the NonCommercial restriction; the downstream author cannot grant more than they received. Public hosting counts as commercial use under Alaap's own locked constraint, so **VoiceSculptor is disqualified for the public product.** It remains usable for local research and as a quality reference. **[HIGH on every licence fact; MEDIUM on the legal conclusion — this warrants a real lawyer, not a research agent.]**

Note also a **param discrepancy**: the paper says the VD model is 3B (LLaSA-3B), the HF card says 4B. Probably vocabulary expansion for XCodec2 tokens, but unresolved. **[MEDIUM]**

### 4.2 InstructTTSEval — current standings

Benchmark: Kexin Huang et al., **arXiv 2506.16381**. Three hierarchical tasks — **APS** (Acoustic-Parameter Specification: pitch, speed, volume, timbre), **DSD** (Descriptive-Style Directive: high-level style descriptions), **RP** (Role-Play). 1,000 test cases per task per language, **6,000 total across English and Chinese only**. Scored by **Gemini as an automatic LLM judge**. Repo `github.com/KexinHUANG19/InstructTTSEval`; dataset `CaasiHUANG/InstructTTSEval` on HF. **No Indian-language coverage.** **[HIGH]**

Standings, as tabulated by MOSS-VoiceGenerator (2603.28086), the most recent source:

| Model | EN APS/DSD/RP | ZH APS/DSD/RP | Open weights? |
|---|---|---|---|
| **Gemini-TTS-Pro** | **87.6 / 86.0 / 67.2** | **89.0 / 90.1 / 75.5** | No (commercial) |
| **Qwen3-TTS-VD** | 78.4 / 78.8 / **72.0** | **84.3 / 82.9 / 77.4** | **YES, Apache-2.0** |
| MIMO-Audio-7B-Instruct | 80.6 / 77.6 / 59.5 | 75.7 / 74.3 / 61.5 | Yes |
| MOSS-VoiceGenerator | 68.2 / 82.0 / 68.7 | 78.0 / 80.0 / 74.0 | **YES, Apache-2.0** |
| GPT-4o-mini-TTS | 76.4 / 74.3 / 54.8 | 54.9 / 52.3 / 46.0 | No |
| VoiceSculptor-VD | — | 75.7 / 64.7 / 61.5 | Yes (contested licence) |
| VoxInstruct | 54.9 / 57.0 / 39.3 | 47.5 / 52.3 / 42.6 | Yes |

**Answer to "who holds SOTA": Gemini-TTS-Pro overall; Qwen3-TTS-VoiceDesign among open-weights, on both splits.** VoiceSculptor's Jan-2026 open-source-SOTA-on-Zh claim was true when published (its own table showed it beating MiMo-Audio-7B 67.6 vs 64.5 AVG) but was superseded within days by Qwen3-TTS's 2026-01-22 release. **[HIGH]**

Caveat worth carrying into 06-evaluation-harness: InstructTTSEval measures **instruction adherence only**. It has no diversity axis at all. A model that returns the same bland voice for every prompt in a category can score well. **Do not use it as Alaap's primary metric.** **[HIGH]**

### 4.3 Diversity measurement in speaker space — the complete published toolkit

| Metric | Source | Definition | Use for Alaap |
|---|---|---|---|
| **`gen2gen-near`** | PromptSpeaker 2310.05001 | Cosine distance between the two closest speaker vectors generated from the **same prompt** (WeSpeaker d-vectors). Reported 0.088. | **The mode-collapse detector.** Sample N=50 per prompt, measure; if it trends toward 0 the mapper has collapsed. Primary regression gate. |
| `gen2syn-near` / `syn2syn-near` / `syn2syn-same` | PromptSpeaker | Distances from generated to training speakers, and among training speakers. 0.085 / 0.113 / 0.024. | **Novelty** check — proves the mapper isn't memorising the training set. |
| **SSD** | UniSpeaker 2501.06394 | Speaker similarity between outputs from *different* descriptions. Lower = more diverse. | **Between-prompt separation.** Catches the opposite failure: every prompt landing on one centroid. |
| Intra/inter-class variance ratio | Sub-Center 2407.04291 | Normalised intra-speaker variance of the embedding space itself. | Diagnoses whether **space B is the bottleneck** rather than the mapper. See §7. |
| FAD | 2406.08812 | Fréchet Audio Distance, reference-free. | Distributional realism of the *audio*, not the vector. |
| EMD over MOS | 2406.08812 appendix | Earth Mover's Distance between rating distributions. | Human-eval distribution matching. |
| t-SNE of voice space | UniSpeaker | Qualitative coverage picture. | Cheap visual smoke test. |

**Important negative result: HiStyle, VoiceDesigner, VoiceSculptor, MOSS-VoiceGenerator, Qwen3-TTS and Parler-TTS report NO diversity metric whatsoever.** The 2026 systems optimise adherence exclusively. The field is measuring the thing that is easy to measure, and the failure mode the brief identifies is largely unmeasured in current work. **This is a real gap and a defensible contribution.** **[HIGH]**

### 4.4 Contrastive / retrieval prior art (the planned S2 implementation)

- **Speaker-Text Retrieval via Contrastive Learning** — arXiv 2312.06055, Liu, Wang, Cooper, Miao, Yamagishi, 2023-12-11. Frames speaker-description ↔ audio as a **cross-modal retrieval task** with pre-trained speaker and text encoders and a simple contrastive framework; evaluated in **English and Japanese**. This is exactly the S2 plan, already published. **Detailed recall@k numbers, dataset sizes and stated limitations: UNVERIFIED** — the HTML render 404s and the PDF did not extract cleanly. Read the PDF manually; it is 560 KB and cached locally from this session.
- **UniSpeaker** — the strongest contrastive result, and the one that argues soft labels are *how* you keep one-to-many alive rather than a limitation to escape.
- **HiStyle** — uses contrastive cosine alignment as an auxiliary to a diffusion head, i.e. the hybrid shape.
- **CLASP** (arXiv 2412.13071) and **SPARCLE** (arXiv 2607.01238) exist as CLAP-style language-speech contrastive spaces but are aimed at retrieval/IR, not voice design. **[MEDIUM — abstract-level only]**

**Measured limitation of pure retrieval, inferred from the field rather than stated:** retrieval can only ever return a voice that exists in the bank. It cannot produce a novel identity, so `gen2gen-near` is structurally bounded by the bank's internal spacing and `gen2syn-near` is zero by construction — it fails PromptSpeaker's novelty test by definition. Top-k + perturb/interpolate mitigates this but interpolating two speaker vectors is not guaranteed to land on a *valid* voice (the space is not convex). **[MEDIUM — this is reasoned from the metric definitions, not from a paper that measured it. Flagged in §7 as an experiment.]**

---

## 5. Recommendation for Alaap's mapper

**Build a conditional flow-matching head over a frozen speaker-embedding space, with a discriminative auxiliary loss and a CFG-style temperature dial. Ship retrieval first.**

### Why flow matching over Glow, diffusion, VAE, or discrete tokens

- **vs Glow (PromptSpeaker):** flows require bijectivity and coupling-layer plumbing; OT-CFM is a plain MSE-on-velocity regression that trains in a few lines and samples in ~10 steps. Deep Dubbing (2509.15845) already demonstrates OT-CFM working on exactly this task into a 192-d Cam++ space. Same distributional guarantee, far less machinery. **[HIGH]**
- **vs diffusion (HiStyle):** works, but HiStyle needs two stacked stages and reports no diversity gain for the cost. Flow matching gets there in fewer sampling steps.
- **vs discrete tokens (VoiceSculptor / MOSS / Qwen3):** requires a 1.7–3B LM, blows the 12GB budget for *training*, and produces no separable identity vector — it breaks the two-tower constraint outright.
- **vs pure contrastive/retrieval (UniSpeaker):** cannot invent novel voices. Correct as a **baseline and a bootstrap**, not as the destination.

### The design, concretely

```
description
  └─ frozen text encoder (A)              ~0 trained params
       └─ projection + FiLM/SALN conditioning
            └─ OT-CFM DiT head            ← THE TRAINED MAPPER, 10-30M
                 ├─ velocity-field MSE loss              (distributional)
                 ├─ + λ · cosine-to-reference aux loss   (adherence — the 2406.08812 hybrid)
                 └─ + contrastive InfoNCE vs batch       (text-audio alignment, per HiStyle/UniSpeaker)
                      └─ speaker vector (B)   192-d Cam++ or 256-d
                           └─ frozen TTS (C)
```

**Param budget: 10–30M, not 50M.** Anchors: Deep Dubbing's TTT is a 4-layer DiT with 4 attention heads and 392 hidden dims — call it ~10M — and it works on 4,000 h / 300K descriptions. HiStyle is ~30M per stage (12 layers, 512 hidden). Start at Deep Dubbing's size; it trains comfortably on 12GB. MOSS's finding that their 1.7B was *more diverse* than their 8B is a further argument against scaling up. **[HIGH]**

**Target space B: 192-d Cam++ speaker embeddings.** Two independent reasons: Deep Dubbing's flow-matching TTT targets exactly this space and works, and CosyVoice2 (Apache-2.0) natively consumes Cam++ embeddings — so A→B→C composes with no adapter. **[HIGH]** Caveat in §7: whether CosyVoice2 preserves quality on speaker-embedding-only conditioning without a reference utterance is **unverified and is the project's #1 risk.**

**Backbone C: CosyVoice2 / Fun-CosyVoice3-0.5B, Apache-2.0.** Clean licence, 0.5B, fits 12GB inference easily. Qwen3-TTS-VoiceDesign (Apache-2.0) is the reference *ceiling* to beat on adherence and the honest baseline to benchmark against. **[HIGH]**

**Expose CFG as the product's signature control.** No published system in this field does this. Given 2406.08812's demonstrated adherence↔diversity tradeoff, a user-facing "how literally should I take your description" slider is both technically motivated and a differentiator. Implement as conditional/unconditional velocity interpolation with the null condition trained via 10–20% description dropout. **[HIGH on novelty, MEDIUM on it working first try]**

### S2 vs S3

**S2 — Retrieval baseline (build first, ~1 week).** CLIP-style dual encoder: frozen text encoder + frozen Cam++, InfoNCE over (description, speaker) pairs. Retrieve top-k, perturb with Gaussian noise scaled to the local neighbourhood, or interpolate between two retrieved vectors. Cheap, robust, impossible to mode-collapse, and it **immediately gives you the evaluation harness and the embedding bank you need for S3.** Prior art: 2312.06055, UniSpeaker. Ship this as the fallback path in production regardless of what S3 does.

**S3 — Flow-matching head (the real system).** Train the OT-CFM DiT on the same pairs. **Gate promotion on `gen2gen-near` and SSD, not on adherence alone** — otherwise you will select a mode-collapsed model that scores beautifully on InstructTTSEval-style metrics. Keep the retrieval path as a live A/B.

---

## 6. What this means for the build

**Clone / install (all licence-clean for public hosting):**

| Artifact | URL | Licence | Role | Fits 12GB? |
|---|---|---|---|---|
| CosyVoice / CosyVoice2 / Fun-CosyVoice3-0.5B | github.com/FunAudioLLM/CosyVoice | Apache-2.0 | **Backbone C** | Yes, comfortably (0.5B) |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | Apache-2.0 | Adherence ceiling / baseline | Yes (~4.5GB BF16) |
| Qwen3-TTS-12Hz-0.6B-Base | huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base | Apache-2.0 | Lighter baseline (~2.5GB) | Yes |
| MOSS-VoiceGenerator | huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator | Apache-2.0 (Qwen3 base — clean) | Quality reference, cinematic-data recipe | Inference yes (~2.4B BF16 ≈ 5GB) |
| Parler-TTS Mini v1 (880M) | github.com/huggingface/parler-tts | Apache-2.0 | Annotation-pipeline reference | Yes |
| InstructTTSEval | github.com/KexinHUANG19/InstructTTSEval + `CaasiHUANG/InstructTTSEval` | — (verify) | Adherence benchmark (EN/ZH only) | N/A — needs Gemini API for judging |
| WeSpeaker / Cam++ | (speaker encoders) | Apache-2.0 (verify) | **Space B** + `gen2gen-near` scoring | Yes |

**Do NOT ship:**

- **VoiceSculptor** — Apache-2.0 claim broken by CC-BY-NC-4.0 upstream (Llasa-3B, xcodec2). Local research only.
- **VoiceDesigner** — nothing to ship. No code, no weights. Take the DSP recipe, leave the model.
- **PromptSpeaker / HiStyle / UniSpeaker / Deep Dubbing** — no weights released; these are architecture papers to reimplement, not dependencies.

**12GB GPU constraints:**
- Training the 10–30M mapper is trivial — it will not be the bottleneck. The bottleneck is **precomputing speaker embeddings** over the corpus (one forward pass of Cam++ per utterance, embarrassingly parallel, cacheable) and **evaluation**, which needs the full TTS in the loop.
- Because A and C are frozen, **precompute and cache all text embeddings and all speaker embeddings to disk once.** Mapper training then never loads either tower — this is what makes the 12GB budget comfortable and is the main practical advantage of the two-tower design.
- Running CosyVoice2 (0.5B) + the mapper simultaneously for eval fits in 12GB. Running Qwen3-TTS-1.7B alongside for A/B does not — sequence them.

**Steal these, they are free:**
1. **VoiceDesigner's DSP chain** (pitch/formant shift, reverb, EQ, band-pass, DRC, SiFi-GAN contour) for the heavy-stylization scope — works as offline augmentation *or* inference post-processing.
2. **The listener-impression annotation trick** (2406.08812): derive prompts from existing multi-question perceptual surveys instead of hand-writing descriptions.
3. **MOSS's cinematic-data insight**: studio TTS corpora produce voices that sound artificial; film/TV audio carries the acoustic variation dramatic content needs.
4. **UniSpeaker's three-axis evaluation** (suitability / diversity / quality) as the harness skeleton.

---

## 7. Open — must be settled by experiment

| Question | Cheapest experiment | Est. cost/time | What it blocks |
|---|---|---|---|
| **Does CosyVoice2 render acceptable speech from a speaker embedding ALONE, with no reference utterance?** This is the load-bearing assumption of the entire two-tower architecture. | Extract Cam++ embeddings from 50 held-out speakers, feed them to CosyVoice2 with reference audio ablated, measure SIM-o and WER against the reference-audio path. | 1 day, local GPU | **Everything.** If it fails, B must become a seed *clip* (Tier-2, VoiceSculptor-style) rather than a vector, and the whole mapper target changes. |
| Is the Cam++ / WeSpeaker space itself too collapsed to carry diversity? (Sub-Center 2407.04291 argues recognition-trained embeddings destroy intra-speaker variance) | Compute intra/inter-class variance ratio on the corpus; compare against a non-recognition-trained alternative. | 2 days | Choice of space B; possibly forces a sub-center or VAE-derived embedding instead. |
| Does the hybrid loss (CFM + cosine aux) actually recover 2406.08812's adherence loss, or just reproduce it? | Train three 10M heads on identical data — pure CFM, pure discriminative, hybrid at λ ∈ {0.1, 0.5, 1.0}. Score `gen2gen-near` + attribute SRCC. | 3–4 days, single GPU | The λ setting, and whether the CFG dial is even needed. |
| Does CFG genuinely trade adherence for diversity here, monotonically? | Sweep guidance scale 1.0→5.0 on the trained head; plot SRCC vs `gen2gen-near`. | 1 day after head exists | The product's signature control. If non-monotonic, don't ship it as a slider. |
| Is interpolation between two retrieved speaker vectors valid (does the midpoint sound like a real voice)? | Interpolate 100 random pairs at α=0.5, render, human-rate naturalness vs endpoints. | 2 days | Whether the S2 retrieval fallback can produce novel identities at all. |
| Do descriptions in Indian languages, or of Indian-accented voices, have ANY coverage in available speaker corpora? | Audit corpora for Indian-language speaker counts and description availability. | 3 days | The multilingual half of the product scope. **No system in this field covers it** — this is currently unaddressed. |
| What are 2312.06055's actual retrieval ceilings (recall@k) and stated limitations? | Read the cached PDF manually — HTML renders 404. | 1 hour | How good the S2 baseline can be expected to get. Currently **UNVERIFIED.** |
| Does VoiceDesigner ever release weights? | Watch Adobe Research / JHU CLSP GitHub orgs post-camera-ready. | passive | Whether a strong character-voice model becomes available (CC-BY-4.0 paper implies research-friendly intent). |

---

## 8. Sources

| # | URL | Type | Used for | Confidence in source |
|---|---|---|---|---|
| 1 | https://arxiv.org/abs/2310.05001 | paper (arXiv abs) | PromptSpeaker existence, authors, date | HIGH |
| 2 | https://arxiv.org/html/2310.05001v1 | paper (arXiv HTML) | Glow architecture, gen2gen-near metric, data, results | HIGH |
| 3 | https://arxiv.org/abs/2509.25842 · https://arxiv.org/html/2509.25842v1 | paper | HiStyle two-stage diffusion, ~30M params, ablation table | HIGH |
| 4 | https://arxiv.org/abs/2608.13613 · https://arxiv.org/html/2608.13613v1 | paper | VoiceDesigner MM-DiT, DSP pipeline, no release, results | HIGH |
| 5 | https://voicedesigner-demo.github.io/ | official demo page | Confirms no code/weights; DSP + character examples | HIGH |
| 6 | https://arxiv.org/abs/2603.28086 · https://arxiv.org/html/2603.28086v1 | paper | MOSS-VoiceGenerator AR architecture, 25k h, InstructTTSEval table, 1.7B-vs-8B diversity | HIGH |
| 7 | https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator | model card | Apache-2.0, 2.4B BF16, weights present | HIGH |
| 8 | https://arxiv.org/abs/2501.06394 · https://arxiv.org/html/2501.06394v1 | paper | UniSpeaker KV-Former, SoftCL, SSD metric, MVC benchmark | HIGH |
| 9 | https://arxiv.org/abs/2406.08812 · https://arxiv.org/html/2406.08812v1 | paper | **The decisive discriminative-vs-flow-matching head-to-head tables** | HIGH |
| 10 | https://arxiv.org/abs/2601.10629 · https://arxiv.org/html/2601.10629v2 | paper | VoiceSculptor design-then-clone, LLaSA-3B, data ladder, InstructTTSEval-Zh claim | HIGH |
| 11 | https://github.com/ASLP-lab/VoiceSculptor | repo | README licence grant, deps (xcodec2, CosyVoice2), file list | HIGH |
| 12 | https://raw.githubusercontent.com/ASLP-lab/VoiceSculptor/main/LICENSE.txt | LICENSE file | Verbatim Apache-2.0 text confirmed | HIGH |
| 13 | https://huggingface.co/ASLP-lab/VoiceSculptor-VD | model card | Apache-2.0 tag, 4B BF16, base = Llasa-3B | HIGH |
| 14 | https://huggingface.co/HKUSTAudio/Llasa-3B | model card | **CC-BY-NC-4.0, "prohibits free commercial use"** — the licence conflict | HIGH |
| 15 | https://huggingface.co/HKUSTAudio/xcodec2 | model card | **CC-BY-NC-4.0** — second licence conflict | HIGH |
| 16 | https://github.com/FunAudioLLM/CosyVoice | repo | Apache-2.0, model sizes (0.3B/0.5B) | HIGH |
| 17 | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | model card | Apache-2.0, end-to-end, 10 languages (no Indian), streaming latency | HIGH |
| 18 | https://github.com/QwenLM/Qwen3-TTS | repo | Qwen3-TTS open-source series | MEDIUM (via search) |
| 19 | https://arxiv.org/abs/2506.16381 | paper | InstructTTSEval definition, APS/DSD/RP, Gemini judge, 6,000 cases EN+ZH | HIGH |
| 20 | https://github.com/KexinHUANG19/InstructTTSEval · https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval | repo / dataset | Benchmark availability | MEDIUM (README not fully readable) |
| 21 | https://arxiv.org/abs/2402.01912 | paper | Parler-TTS / Lyth & King, 45k h annotation | HIGH |
| 22 | https://github.com/huggingface/parler-tts | repo | Apache-2.0, Mini 880M / Large 2.3B, maintained | HIGH |
| 23 | https://arxiv.org/html/2509.15845v1 | paper | **Deep Dubbing Text-to-Timbre: OT-CFM → Cam++ 192-d, 4-layer DiT** | HIGH |
| 24 | https://arxiv.org/abs/2312.06055 | paper | Speaker-Text Retrieval contrastive prior art (abstract level only) | MEDIUM — details UNVERIFIED |
| 25 | https://arxiv.org/abs/2407.04291 | paper | Sub-center embeddings; intra/inter-class variance as diversity metric | MEDIUM (via search + abstract) |
| 26 | https://arxiv.org/html/2412.13071 (CLASP) · https://arxiv.org/html/2607.01238 (SPARCLE) | paper | Adjacent CLAP-style contrastive spaces | LOW-MEDIUM — abstract only |
| 27 | https://simonwillison.net/2026/Jan/22/qwen3-tts/ | secondary | Qwen3-TTS release date corroboration only | LOW (secondary — used only for date) |
