# 13 — Description → Speaker-Embedding: Prior Art & the Reference Implementation

> **Domain:** every published system that maps a natural-language description into an addressable speaker/style vector; which are open; which are buildable
> **Answers:** **A3** (which generative head), and supplies a working reference implementation for the mapper
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [01-ttv-landscape.md](01-ttv-landscape.md) · [02-identity-representation.md](02-identity-representation.md) · [03-tts-backends-english.md](03-tts-backends-english.md) · [06-evaluation-harness.md](06-evaluation-harness.md) · [12-speaker-manifold-navigability.md](12-speaker-manifold-navigability.md)

---

## 0. Bottom line

- **⭐ PromptTTS++ is the reference implementation VoiceForge should read before writing a line of mapper code.** It is the *only* system surveyed that gives a permissively-licensed, downloadable, description→speaker-embedding generator with a directly addressable vector space: **256-d, L2-normalized, Apache-2.0, weights real and downloadable**. [HIGH]
- **Its head is an MDN (mixture density network), not diffusion.** A **per-dimension 10-component Gaussian mixture** over the 256-d vector. The scope document's §17-A3 framing (flow vs diffusion vs normalizing flow) **omits the family that the one working open system actually uses**, and MDN is by far the cheapest of the four. [HIGH]
- **Generating a usable speaker embedding from a non-audio modality is a settled result** — faces since 2022, text descriptions since 2023. **The gap is not feasibility; it is openness.** [HIGH]
- **HiStyle supplies the taxonomy of the whole design family**, ranked, which is exactly the comparison scope §17-A3 asked for — and it finds **hierarchical two-stage diffusion** best. But it has no code, an internal dataset, and an internal backbone. [HIGH]
- **The strongest *currently usable* open voice-design model is MOSS-VoiceGenerator** (Apache-2.0, ~74k downloads, trained on cinematic rather than studio-clean content) — **but it is Tier 2 and gives you no addressable vector.** [HIGH]
- **ControlSpeech does not do what its framing implies.** It has a separable numeric timbre vector, but **no description→timbre path exists anywhere in the model** — timbre comes only from reference audio. Its many-to-many story is about *style*, not speaker. And its own model code was never released. [HIGH]
- **Three headline systems are closed and one is now gone:** Voicebox ("we are not making the Voicebox model or code publicly available"), Audiobox (demo **shut down February 2026**, no weights ever), NaturalSpeech 2 (no official Microsoft release). [HIGH]
- **Read the InstructTTSEval ceiling carefully: real human reference audio scores only 84.3, and 67.2 on Role-Play.** Absolute scores above ~85 are meaningless; the metric is a noisy binary LLM proxy. The meaningful signal is the ~30-point open-vs-commercial gap. [HIGH]

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Conf. |
|---|---|---|---|---|---|
| 13-1 | §17-A3 lists the candidate heads as flow matching / diffusion / normalizing flow | **Incomplete — omits MDN** | The only *open, working, downloadable* description→speaker-embedding generator uses a **mixture density network** (per-dim 10-component GMM). It is also the cheapest to train. MDN belongs at the top of the A3 comparison, not absent from it. | [`line/promptttspp`](https://github.com/line/promptttspp) source | HIGH |
| 13-2 | §3 timeline: "PromptSpeaker — first clean formulation… Establishes the two-tower split" | **Wrong priority** | **PromptTTS++** (arXiv [2309.08140](https://arxiv.org/abs/2309.08140), v1 **2023-09-15**) predates PromptSpeaker ([2310.05001](https://arxiv.org/abs/2310.05001), 2023-10) — and unlike PromptSpeaker it **released code and weights under Apache-2.0**. PromptSpeaker has no released weights. | arXiv v1 dates | HIGH |
| 13-3 | §3 timeline omits **InstructTTS** (2301.13662, 2023-01-31) | **Missing the earliest entry** | InstructTTS predates every system in the table by 8 months. Worth citing for the style-encoder + InfoNCE + CLUB MI-minimisation design, even though its speaker path is a closed-set LUT. | [arXiv:2301.13662](https://arxiv.org/abs/2301.13662) | HIGH |
| 13-4 | §17-A3 lists VoiceDesigner as "diffusion transformer" among mapper candidates | **Category error** (compounding [`01`](01-ttv-landscape.md)'s finding) | Several systems in the A3 list generate a **style** vector, not a **speaker** vector. PromptVC, InstructTTS and ControlSpeech's SMSD are all style predictors; the speaker channel is separate and audio-only. Comparing them as speaker-embedding generators is comparing different things. | per-paper, §3 below | HIGH |
| 13-5 | §6 benchmark row implies open models are near commercial quality | **~30-point gap on the only public benchmark** | InstructTTSEval EN average: best open (VoxInstruct) **50.4** vs gemini-flash **88.7**, hume **71.1**, gpt-4o-mini-tts **68.5**. Parler-TTS-mini **46.9**. The open field is not close. | [InstructTTSEval](https://arxiv.org/abs/2506.16381) | HIGH |
| 13-6 | §10 proposes InstructTTSEval as an adherence metric without caveat | **Ceiling caveat missing** | **Real human reference audio scores 84.3 avg and only 67.2 on Role-Play** — the task closest to VoiceForge's use case. Scoring is **binary true/false** per item judged by `gemini-2.5-pro`, macro-averaged. Treat >85 as noise. Judge cost ≈ **$12.8/session EN**. | ibid. | HIGH |
| 13-7 | §8.2 / [`05`](05-datasets-and-annotation.md) noted "VccmDataset does not exist" | **Partial reconciliation** | `VccmDataset/` **is** a directory in the ControlSpeech repo. It is not an independently published dataset with a card or licence — and the repo has **no LICENSE at all** — so it remains unusable, but the name is not fabricated. | [`jishengpeng/ControlSpeech`](https://github.com/jishengpeng/ControlSpeech) | HIGH |
| 13-8 | §17-C4 / §5: non-human voices deferred because "no public dataset covers it" | **A third path exists the brief didn't consider** | **BatonVoice** ([2509.26514](https://arxiv.org/abs/2509.26514)) has an LLM emit an explicit **textual vocal-feature plan** — neither an embedding nor raw audio. That is a route to stylization that needs no paired corpus at all. Worth a look before committing to DSP augmentation. | [arXiv:2509.26514](https://arxiv.org/abs/2509.26514) | MEDIUM |

---

## 2. The survey table

| System | arXiv | v1 | Tier | Code + licence | Conf. |
|---|---|---|---|---|---|
| **⭐ PromptTTS++** (LY Corp) | [2309.08140](https://arxiv.org/abs/2309.08140) | 2023-09-15 | **TIER 1** — 256-d, sampled | [`line/promptttspp`](https://github.com/line/promptttspp) **Apache-2.0** (LICENSE verified); weights on HF Space, Apache-2.0 | HIGH |
| **ControlSpeech** | [2406.01205](https://arxiv.org/abs/2406.01205) | 2024-06-03 | TIER 1 for **style** (512-d) / **TIER 2 for timbre** | **NO LICENCE** (404, `license: null`); **model code never released** | HIGH |
| **PromptVC** (NWPU/Ximalaya) | [2309.09262](https://arxiv.org/abs/2309.09262) | 2023-09-17 | TIER 1 — style vector via latent diffusion | **No code, no weights** — demo page only | HIGH |
| **InstructTTS** (Tencent) | [2301.13662](https://arxiv.org/abs/2301.13662) | 2023-01-31 | TIER 1 style encoder / speaker = **closed-set LUT** | Demo page only, no licence, no code/weights | HIGH |
| **Audiobox** (Meta) | [2312.15821](https://arxiv.org/abs/2312.15821) | 2023-12-25 | TIER 2 | **Closed.** Demo **shut down Feb 2026** | HIGH |
| **Voicebox** (Meta) | [2306.15687](https://arxiv.org/abs/2306.15687) | 2023-06-23 | TIER 2 | **Closed**, on the record | HIGH |
| **VoiceLDM** | [2309.13664](https://arxiv.org/abs/2309.13664) | 2023-09-24 | TIER 2, but a 512-d CLAP conditioning slot | [`glory20h/VoiceLDM`](https://github.com/glory20h/VoiceLDM) **Apache-2.0** | HIGH |
| **VoxInstruct** | [2408.15676](https://arxiv.org/abs/2408.15676) | 2024-08-28 | TIER 2 | [`thuhcsi/VoxInstruct`](https://github.com/thuhcsi/VoxInstruct) **MIT**; ckpts on Drive; inference only | HIGH |
| **NaturalSpeech 2** (MSFT) | [2304.09116](https://arxiv.org/abs/2304.09116) | 2023-04-18 | TIER 2 | **No official release.** Amphion reimpl, MIT | HIGH |
| **HiStyle** (NWPU ASLP) | [2509.25842](https://arxiv.org/abs/2509.25842) | 2025-09-30 | **TIER 1** — 2-stage diffusion | **No code** (anonymised demo → under review) | HIGH |
| **MOSS-VoiceGenerator** (Fudan) | [2603.28086](https://arxiv.org/abs/2603.28086) | 2026-03-30 | TIER 2 | HF `OpenMOSS-Team/MOSS-VoiceGenerator` **Apache-2.0**, ~74k downloads | HIGH |
| **FleSpeech** | [2501.04644](https://arxiv.org/abs/2501.04644) | 2025-01-08 | TIER 1 family — multimodal prompt→representation | No code found | MEDIUM |
| **InstructTTSEval** | [2506.16381](https://arxiv.org/abs/2506.16381) | 2025-06-19 | benchmark | `CaasiHUANG/InstructTTSEval` **MIT** | HIGH |
| **MINT-Bench** | [2604.17958](https://arxiv.org/abs/2604.17958) | 2026-04-20 | benchmark | toolkit released; licence unchecked | MEDIUM |
| **VoiceMe** | [2203.15379](https://arxiv.org/abs/2203.15379) | 2022-03-29 | TIER 1 — navigates SpeakerNet space | not checked | HIGH (existence) |
| **Face-TTS** | [2302.13700](https://arxiv.org/abs/2302.13700) | 2023-02-27 | TIER 1-ish — face→speaker-embedding space | facetts.github.io | HIGH (existence) |
| "Face2Voice" | — | — | — | **DOES NOT EXIST** — arXiv `all:"Face2Voice"` returns 0 results | HIGH |

---

## 3. PromptTTS++ — the reference implementation ⭐

**Verified from repo source code, not paper prose.** From `egs/proposed/bin/conf/model/prompttts_mdn_v2_wo_erg_final*.yaml` and `promptttspp/modules/`:

| Component | What it is |
|---|---|
| `prompt_encoder` | **BERT-base-uncased**, `[CLS]` token (768) → MLP adaptor **768 → 512 → 512 → 256** |
| `style_mdn` | `MDNLayer(in_dim=256, out_dim=256, num_gaussians=10, dim_wise=True)` → a **per-dimension 10-component GMM** over a **256-d** vector |
| `reference_encoder` | GST-style `StyleEncoder` (mel 80 → GRU 256) — the MDN's regression target |
| `norm_style_emb` | `true` → embeddings **L2-normalized to the unit 256-sphere** |

**Correction to the brief and to common summaries:** the speaker-embedding generator is an **MDN, not diffusion**. The diffusion (`GaussianDiffusion`, out_dim 80) is the *mel decoder*. The abstract is consistent: "diffusion-based acoustic model with mixture density networks to model speaker variations."

The decisive artefact, from `promptttspp/models/prompttts_mdn_v2_final/model.py`:

```python
def generate_style_emb(self, style_prompt, reference_mel, ...):
    prompt_emb = self.prompt_encoder(style_prompt, ...)
    log_pi, log_sigma, mu = self.style_mdn(prompt_emb.transpose(-1, -2))
    prompt_emb = self.sample_style_emb(log_pi, log_sigma, mu, ...)
    ref_emb = self.reference_encoder(reference_mel, ref_lengths)
```

`infer()` accepts **either** a text prompt (→ MDN sample) **or** a reference mel (→ reference encoder) into the *same* 256-d slot, then `x = x + style_emb` and passes it as `g` to the decoder.

> **The 256-d vector is the sole carrier of speaker + style identity, and you can address that space directly with an arbitrary 256-d unit vector.** This is precisely the two-tower contract scope §7 specifies — already built, already open, already Apache-2.0.

**Note the `norm_style_emb: true` detail.** It means PromptTTS++ operates on the **unit hypersphere** — exactly the condition under which [`12-speaker-manifold-navigability.md`](12-speaker-manifold-navigability.md) §6.1 found random sampling *works* (Jia et al.'s L2-normalized d-vectors, MOS 3.65). PromptTTS++ has independently arrived at the geometry the manifold evidence says is safe. That is a strong convergent signal.

### 3.1 Usability: marginal, but real

| | |
|---|---|
| Last push | **2024-10-11** · 86★ / 6 forks · not archived |
| Stack | Python 3.8, `torch==1.11.0+cu113` — **old**; expect an isolated venv (compare [`04-indic-track.md`](04-indic-track.md)'s parler-tts pinning problem) |
| Weights | **Real and downloadable** from the HF Space: `pretrained_model/checkpoint/proposed/last.ckpt` = **1.32 GB**; `bigvgan_f0_full/last.ckpt` = **660 MB** |
| HF Space | currently **`RUNTIME_ERROR`** — the Space is broken, the weights are not |
| Training data | LibriTTS-R, English, 24 kHz — **CC BY 4.0**, clean for train-and-release (see [`08-licensing-propagation.md`](08-licensing-propagation.md)) |
| Caveat | The released config uses the plain `PromptEncoder` (one concatenated prompt string), **not** the `SepPromptEncoder` that splits style\|speaker prompts on `"\|"`. That variant is in the code but **not in the shipped checkpoint.** |

> **The `SepPromptEncoder` gap is directly relevant to scope §4.4** (timbre vs performance must be independently addressable). PromptTTS++ *designed* a split style/speaker prompt encoder and then shipped a checkpoint without it. That is a warning about how easily the two channels collapse back into one — see [`10-performance-control.md`](10-performance-control.md).

---

## 4. ControlSpeech — reads like Tier 1, is not

From the v3 HTML (ACL 2025 Main). Title changed in v3 to *"…Simultaneous and **Independent** Zero-shot Speaker Cloning and Zero-shot Language Style Control"* — the "Decoupled Codec" suffix was dropped.

- **Timbre `Y_t`**: a **global vector** from the **FACodec** (NaturalSpeech 3) timbre extractor, applied to the **reference audio**. Fused by conditional layer-norm: linear layers `W_γ, W_β` map `Y_t` → scale/bias over `Y_codec`.
- **Style `Y_s'`**: SMSD = BERT `[CLS]` → **Gaussian MDN** (3/5/7 components) → **sampled global style vector ∈ ℝ⁵¹²**.

> **So: yes there is a separable numeric timbre representation, but there is NO description→timbre path anywhere in the model.** Timbre comes only from reference audio. SMSD is a *style* predictor, not a speaker predictor. The many-to-many framing in the paper is explicitly about style, and the paper contrasts itself with PromptTTS 2 on exactly this point.

**Release status is worse than it looks.** No LICENSE at any path (`main/LICENSE`, `master/LICENSE`, `main/LICENSE.md` all 404); GitHub API `license: null` → **all rights reserved by default**. More importantly the **model code is not in the repo**: root contents are `VccmDataset/`, `baseline/`, `emotion_acc/`, `pitch_energy_speed_acc/`, `spk_sv.py`, `wer.py`, `README.md`. The README calls it the "ControlToolkit" — dataset + eval metrics + reproduction code for *baselines* (PromptTTS, PromptStyle). The Google Drive is labelled "Models(**Baseline**)". Last push 2024-11-22.

**Actionable consolation:** FACodec itself — the component yielding the decoupled global timbre vector — **is** openly available: [`amphion/naturalspeech3_facodec`](https://huggingface.co/amphion/naturalspeech3_facodec), **Apache-2.0**. That is a usable timbre extractor with a clean licence, and a candidate for the C2 encoder question.

---

## 5. The rest, briefly

**PromptVC** ([2309.09262](https://arxiv.org/abs/2309.09262), ICASSP 2024) — a clean Tier-1 precedent, unbuildable. Abstract states it directly: *"employs a latent diffusion model to generate a style vector driven by natural language prompts… the style vector is extracted by a style encoder during training, and then the latent diffusion model is trained independently to sample the style vector from noise."* Text encoder **ChatGLM2-6B**, cross-attention conditioning. Dimension not stated. Internal **400-hour Mandarin** corpus. **No repo exists** (`yaoxunji/prompt-vc` → 404). Generates a *style* vector; source timbre is preserved (it is VC).

**InstructTTS** ([2301.13662](https://arxiv.org/abs/2301.13662)) — style vector yes, speaker identity no. **RoBERTa** `[CLS]` → sentence embedding, **frozen**, then an adaptor into a style latent, aligned to an audio style encoder via **InfoNCE**, with **CLUB** MI-minimisation against speaker and content. **Speaker identity is a `Speaker ID → LUT`** over a closed set — no description→speaker path, no zero-shot speaker capability. Discrete diffusion over VQ acoustic tokens (12-layer, 8-head, dim 256) → Mel-VQ-VAE → HiFi-GAN. Internal Tencent **NLSpeech** dataset, not released. It is an *encoder* (deterministic text→style), not a *generator* — you cannot sample diverse voices from one description. *(The commonly-cited TASLP 2024 vol. 32 pp. 2913–2925 citation could not be confirmed from IEEE Xplore — treat as unverified.)*

**Audiobox** ([2312.15821](https://arxiv.org/abs/2312.15821)) — T5-base maps the description to a **sequence** of word embeddings; the voice prompt is embedded by a light Transformer; both concatenated as keys/values for cross-attention in the flow-matching transformer. **No global speaker vector anywhere.** No weights ever released. The metademolab page now reads: *"As of February 2026, the Audiobox Demo is no longer available."* HF search returns only `facebook/audiobox-aesthetics` — a *quality-assessment* model, unrelated to generation.

**Voicebox** ([2306.15687](https://arxiv.org/abs/2306.15687)) — Meta, on the record: **"we are not making the Voicebox model or code publicly available at this time."** Architecturally not description-driven at all — audio-context infilling with flow matching, Tier 2 by construction. `lucasnewman/voicebox-small` is a third-party reimplementation.

**VoiceLDM** ([2309.13664](https://arxiv.org/abs/2309.13664)) — Tier 2 output, but a genuinely addressable global slot. From `voiceldm/pipeline.py`: `c_desc` is a single global **512-d CLAP embedding** (`laion/clap-htsat-unfused`), obtained from **either** `clap_model.text_projection(text)` **or** `clap_model.audio_projection(audio)` — interchangeable. But it lives in a **CLAP audio-scene space**, and the paper's "description" is about **environmental context**, not voice identity. Apache-2.0, 194★, last push 2024-08-09.

**VoxInstruct** ([2408.15676](https://arxiv.org/abs/2408.15676), ACM MM 2024) — unified multilingual codec **language model**: instruction → speech **semantic tokens** → acoustic tokens → audio, with multiple CFG strategies. No global speaker/style vector. **MIT**, ckpts on Drive, inference only (training listed as future work), last push 2024-11-09. **Top open-source system on InstructTTSEval (50.4% avg EN).**

**NaturalSpeech 2** ([2304.09116](https://arxiv.org/abs/2304.09116)) — RVQ latents + latent diffusion; speaker identity via in-context **speech prompting**, not a global embedding. **No official Microsoft code or weights.** Community reimpl `amphion/naturalspeech2_libritts` (MIT). *Relevant for PromptTTS 2 due diligence: PromptTTS 2's backbone is itself unreleased, so PromptTTS 2 is not reproducible end-to-end from official artefacts.*

---

## 6. HiStyle — the taxonomy scope §17-A3 actually asked for

[arXiv:2509.25842](https://arxiv.org/abs/2509.25842), 2025-09-30, NWPU ASLP. **The most directly on-point Tier-1 paper found, and it was not on the brief's list in its correct role.**

It runs t-SNE on global style embeddings from ECAPA-TDNN / a pretrained voiceprint model / a CNN-GRU encoder and finds the space **clusters first by timbre, then subdivides by style attribute** — an important geometric fact that corroborates [`12`](12-speaker-manifold-navigability.md) §7.

Architecture: a **two-stage predictor** —
1. **Speaker Embedding Predictor** → global speaker embedding, conditioned on a BERT text-prompt embedding
2. **Style Embedding Predictor** → conditioned on the prompt embedding **plus a residual connection from stage 1**

Both are **conditional diffusion models on transformer blocks: 12 layers, hidden 512, ~30M params each**, trained with **MSE + a cosine-similarity contrastive loss** (in-batch negatives).

Reported controllability: gender **98.88%**, volume **95.56%**, pitch **92.87%**, speed **90.84%**, fluctuation **88.02%**, WER **3.32%**.

### The design-family taxonomy, ranked by HiStyle

| Family | Exemplar | HiStyle's verdict |
|---|---|---|
| Lightweight projection net ("Discriminative Model") | PromptTTS | **weakest** |
| Diffusion variation network | PromptTTS 2 | middle |
| Query encoder | FleSpeech ([2501.04644](https://arxiv.org/abs/2501.04644)) | middle |
| **MDN / GMM** | **PromptTTS++**, ControlSpeech SMSD | middle |
| **Hierarchical two-stage diffusion** | **HiStyle** | **best in their comparison** |

> **This is the answer to A3, with a caveat.** The ranking comes from HiStyle's own paper — self-reported, and it has **no code, an internal dataset, and an internal SingleCodec+LLaMA backbone**, so it is not independently reproducible. Weigh it against [`01-ttv-landscape.md`](01-ttv-landscape.md)'s finding from [arXiv:2406.08812](https://arxiv.org/abs/2406.08812) — the only *independent* A/B — which found flow matching wins fidelity but **loses adherence** to a regression head. Two sources, two different conclusions, both plausible. **This is why S2 must measure rather than assume.**
>
> **Practical sequencing that satisfies both:** build the MDN head first (PromptTTS++ gives you working code and a checkpoint), measure it, then try two-stage diffusion (HiStyle's recipe is ~30M params/stage — well inside the 12 GB budget). The residual connection from stage 1 to stage 2 is the cheap, high-value idea to steal regardless of which head wins.

---

## 7. MOSS-VoiceGenerator — the best usable open voice-design model today

[arXiv:2603.28086](https://arxiv.org/abs/2603.28086), 2026-03-30, Fudan (same group as InstructTTSEval).

Framed exactly as VoiceForge's problem — "voice design from natural language… generate speaker timbres directly from free-form textual descriptions" — but architecturally **Tier 2**: a **Qwen3**-initialised causal LM, delay-pattern, emitting **MOSS-Audio-Tokenizer RVQ tokens** (first 16 codebook layers) from `[description ‖ transcript]`, decoded to waveform. **No intermediate timbre vector.**

- **Open weights, Apache-2.0**, `OpenMOSS-Team/MOSS-VoiceGenerator`, **~74k downloads**
- **Trained on cinematic content specifically to escape studio-clean timbres** — directly relevant to scope §5's "human + heavy stylization" range, and to the character-voice coverage lost when the anime sources are dropped from VoicePersona (see [`05`](05-datasets-and-annotation.md))

> **Recommended role: the Tier-2 English baseline for S0.** It is Apache-2.0, purpose-built for voice design, and trained on exactly the dramatic register VoiceForge targets. Compare against VoxCPM2 (whose "voice design" [`03`](03-tts-backends-english.md) found to be a text-prefix hack) — MOSS is the more honest implementation of the same idea.

---

## 8. InstructTTSEval — read the ceiling before trusting the metric

[arXiv:2506.16381](https://arxiv.org/abs/2506.16381), 2025-06-19, Kexin Huang / Xipeng Qiu, Fudan. 3 tasks × 1k cases × EN+ZH = 6k, each with reference audio, mined from movies/TV with DVA-toolkit expressiveness filtering. Dataset `CaasiHUANG/InstructTTSEval`, **MIT**.

- **APS** — Acoustic-Parameter Specification: explicit control of 12 low-level acoustic features
- **DSD** — Descriptive-Style Directive: free-form descriptions with attributes randomly omitted
- **RP** — **Role-Play**: abstract scenario, e.g. "elderly storyteller" — the model must infer the vocal style. **This is VoiceForge's actual use case.**

**Scoring:** each item judged **binary true/false** by `gemini-2.5-pro` ("true" = primary style attributes align without conflict). Reported number = **macro-average % true** over 1,000 items.

| System | APS | DSD | RP | Avg |
|---|---|---|---|---|
| **reference_audio (real human)** | 96.2 | 89.4 | **67.2** | **84.3** |
| gemini-flash | 92.3 | 93.8 | 80.1 | 88.7 |
| gemini-pro | 87.6 | 86.0 | 67.2 | 80.3 |
| hume | 83.0 | 75.3 | 54.3 | 71.1 |
| gpt-4o-mini-tts | 76.4 | 74.3 | 54.8 | 68.5 |
| **VoxInstruct** | 54.9 | 57.0 | 39.3 | **50.4** |
| PromptTTS (repro) | 64.3 | 47.2 | 31.4 | 47.6 |
| Parler-TTS-mini / -large | 63.4 / 60.0 | 48.7 / 45.9 | 28.6 / 31.2 | 46.9 / 45.7 |
| PromptStyle (repro) | 57.4 | 46.4 | 30.9 | 38.2 |

> **Three things to take from this table:**
> 1. **Real human audio scores 84.3, and only 67.2 on Role-Play.** Anything above ~85 is measuring judge noise, not quality. Two systems already exceed the human reference — that is a metric artefact, not superhuman TTS.
> 2. **Role-Play is the hardest task for everyone**, and it is exactly VoiceForge's use case. Open systems score 28.6–39.3 there. This is where the real work is.
> 3. **Parler-TTS scores 46.9** — and Indic Parler-TTS is the brief's Indic backbone. Calibrate expectations for the Indic track accordingly ([`04`](04-indic-track.md)).
>
> Judge cost ≈ **$12.8/session EN** — cheap enough to run per-checkpoint, expensive enough not to run per-commit. Feed to [`06-evaluation-harness.md`](06-evaluation-harness.md)'s budget.

**Also verified as existing (2026):** MINT-Bench ([2604.17958](https://arxiv.org/abs/2604.17958)), FlexiVoice ([2601.04656](https://arxiv.org/abs/2601.04656)), CapTalk ([2604.08363](https://arxiv.org/abs/2604.08363), "Unified Voice Design"), OV-InstructTTS ([2601.01459](https://arxiv.org/abs/2601.01459)), Poly-InstructTTS ([2608.20387](https://arxiv.org/abs/2608.20387), Interspeech 2026), **BatonVoice** ([2509.26514](https://arxiv.org/abs/2509.26514) — LLM emits an explicit *textual* vocal-feature plan; a third path that is neither embedding nor raw audio).

---

## 9. Face→voice: independent proof the space is navigable

**No arXiv paper titled "Face2Voice" exists** (`all:"Face2Voice"` → 0 results), and **no PromptTTS 2 face extension was found** — treat any such claim as **UNVERIFIED**. What does exist:

- **⭐ VoiceMe** ([2203.15379](https://arxiv.org/abs/2203.15379), Interspeech 2022) — **the most useful data point here.** Conditions TTS on **SpeakerNet speaker-verification embeddings** and uses a **human sampling paradigm to explore the speaker latent space**. Users create voices fitting photos of faces, art portraits *and cartoons*; independent raters confirm the match; gender apparent from the face is recovered; participants **converge consistently** toward the real voice prototype. **Direct empirical evidence that an off-the-shelf speaker-verification embedding space is smooth, navigable and semantically meaningful** — which is the A2 question, answered from a completely independent direction. See [`12`](12-speaker-manifold-navigability.md) §5.
- **Face-TTS / "Imaginary Voice"** ([2302.13700](https://arxiv.org/abs/2302.13700), ICASSP 2023) — face image → cross-modal biometric features condition a diffusion TTS, with a **speaker feature binding loss enforcing similarity in speaker embedding space**. Zero-shot, no per-speaker fine-tuning. LRS3.
- **Zero-shot personalized Lip2Speech** ([2305.14359](https://arxiv.org/abs/2305.14359), ICASSP 2023) — explicit **face-based speaker embeddings (FSE)** controlling voice for unseen speakers, VAE-disentangled from content.
- **FleSpeech** ([2501.04644](https://arxiv.org/abs/2501.04644)) — a **multimodal prompt encoder unifying text, audio and visual prompts into one representation**, explicitly supporting "generating a voice that matches a character's visual appearance."

> **Bottom line:** generating a usable speaker embedding from a non-audio modality is a settled result — faces 2022–2023, text descriptions 2023–2026. **The gap is not feasibility; it is openness.** Of everything surveyed, exactly one system gives a permissively-licensed, downloadable, description→speaker-embedding generator with an addressable vector space: **PromptTTS++**.
>
> **A speculative but cheap idea worth logging:** FleSpeech's visual prompt path suggests a future VoiceForge feature — *upload a character portrait, get a matching voice*. It fits the product (game developers have character art before they have voice direction), and it carries **no impersonation risk from a drawn character**, preserving the §15.1 safety property. Not phase 1; worth not designing out.

---

## 10. What this means for the build

1. **Read PromptTTS++'s source before writing the mapper.** It is the working reference for the exact contract scope §7 specifies. Clone it, get the 1.32 GB checkpoint, run inference, and confirm the 256-d slot behaves as documented. This is a **day-one S0 task**, not an S2 task.
2. **Add MDN to the A3 comparison at position one.** Cheapest head, only open working implementation, and it pairs naturally with the L2-normalized unit-sphere geometry that [`12`](12-speaker-manifold-navigability.md) found safe.
3. **Steal HiStyle's residual connection** (stage-1 speaker embedding → stage-2 style predictor) regardless of head family. It is the concrete mechanism for scope §4.4's timbre/performance separation, at ~30M params/stage.
4. **Use MOSS-VoiceGenerator as the S0 Tier-2 English baseline** rather than VoxCPM2's text-prefix hack.
5. **Adopt InstructTTSEval, but report Role-Play separately and quote the 84.3/67.2 human ceiling next to every number.** Never report an InstructTTSEval score without it.
6. **`amphion/naturalspeech3_facodec` (Apache-2.0) is a candidate C2 timbre encoder** with a clean licence and a documented decoupled global timbre vector.
7. **Treat the `SepPromptEncoder` gap as a warning**: the one team that built a split style/speaker prompt encoder shipped without it. Budget real effort for keeping the two channels separate ([`10-performance-control.md`](10-performance-control.md)).

---

## 11. Open

| # | Question | Cheapest resolution | Blocks |
|---|---|---|---|
| 13-O1 | Does the PromptTTS++ checkpoint still load on a modern stack? | Build the py3.8/torch-1.11 venv, load the 1.32 GB ckpt, run one inference | Whether the reference impl is usable or only readable |
| 13-O2 | Does sampling the PromptTTS++ MDN give diverse voices from one prompt? | Sample 20 style embeddings from one prompt; render; compute Vendi + GVD per [`06`](06-evaluation-harness.md) / [`12`](12-speaker-manifold-navigability.md) | **The core A3 question, answered on real weights for ~1 GPU-hr** |
| 13-O3 | Is HiStyle's code released yet? | Re-check the 4open.science link and search for a de-anonymised repo | Whether two-stage diffusion is buildable or must be reimplemented |
| 13-O4 | ControlSpeech licence — will the authors grant one? | Open a GitHub issue asking for a LICENSE file | Only matters if FACodec alone proves insufficient |
| 13-O5 | InstructTTSEval judge cost at our cadence | Price 50 descriptions × 3 tasks against the $12.8/session EN figure | [`06`](06-evaluation-harness.md) eval budget |

---

*Pass 1 · 2026-09-02 · arXiv IDs fetched and confirmed. PromptTTS++ and ControlSpeech claims read from repository source code and LICENSE endpoints, not paper prose.*
