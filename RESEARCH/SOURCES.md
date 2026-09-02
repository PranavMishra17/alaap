# SOURCES — Master Bibliography

> Load-bearing primary sources across the corpus. Each domain file carries its own full source table; this is the index to the ones that **changed a decision**.
> **All fetched 2026-09-02.** Licences and prices are snapshots — re-verify before relying on them.

**Source-type key:** `LICENSE` = licence file fetched directly · `CARD` = model/dataset card · `REPO` = repository source code · `SPEC` = official OpenAPI/API reference · `PAPER` = arXiv/proceedings · `DOCS` = official documentation · `REG` = regulator publication

---

## A. The decisive evidence (A1 / A2 / A3)

| Source | Type | What it settled | File |
|---|---|---|---|
| [arXiv:1806.04558](https://arxiv.org/abs/1806.04558) §3.6 — Jia et al., NeurIPS 2018 | PAPER | **Random unit-hypersphere vectors → naturalness MOS 3.65**, "as natural as for seen or unseen real speakers." The manifold is navigable by synthesis | [`12`](12-speaker-manifold-navigability.md) |
| [arXiv:2207.04834](https://arxiv.org/abs/2207.04834) §5.1.1 — Meyer et al., Interspeech 2022 | PAPER | **The collapse mechanism and its fix.** Per-dim rescaling takes GVD −6.50 → −0.14. Dimension ranges span `[−0.97,−0.11]` to `[−72.37,81.59]` | [`12`](12-speaker-manifold-navigability.md) |
| [arXiv:2111.05095](https://arxiv.org/abs/2111.05095) §6.2 — TacoSpawn | PAPER | **Discriminative d-vector spaces resist parametric priors** — g2s 0.35 vs s2s 0.20. Learned spaces hit 0.20/0.20. Standard-normal priors "performed worse" | [`12`](12-speaker-manifold-navigability.md) |
| [arXiv:2406.08812](https://arxiv.org/abs/2406.08812) — Interspeech 2024 | PAPER | **The only independent A/B**: flow matching wins fidelity (FAD 3.559 vs 5.244) but **loses adherence** (SRCC 0.60 vs 0.74). Hybrid best (3.126) | [`01`](01-ttv-landscape.md) |
| [`line/promptttspp`](https://github.com/line/promptttspp) | REPO + LICENSE | **The reference implementation.** MDN head (`MDNLayer(256,256,num_gaussians=10,dim_wise=True)`), `norm_style_emb: true`, Apache-2.0, weights downloadable (1.32 GB) | [`13`](13-description-to-embedding-prior-art.md) |
| [arXiv:2606.05367](https://arxiv.org/abs/2606.05367) | PAPER | **Training-free Direction channel on our exact checkpoint**: `x + α·τ`, ΔEECS +0.29 at SECS_W 0.912, τ over ≥4 speakers. LM steering ≈0.000 S-SIM cost; decoder −0.064 | [`10`](10-performance-control.md) |
| Panariello et al., Interspeech 2023 — [DOI 10.21437/Interspeech.2023-448](https://doi.org/10.21437/Interspeech.2023-448) | PAPER | **Vocoder drift** — the vector in ≠ the vector out. Kills "exactly, by construction" | [`12`](12-speaker-manifold-navigability.md) |
| [VPC 2026 Evaluation Plan](https://www.voiceprivacychallenge.org/vp2026/docs/VPC_2026_sept01.pdf) | DOCS | **B1 (averaging) retired**; B3 (WGAN synthetic embedding) is now a baseline. **Results land 2026-09-26** | [`12`](12-speaker-manifold-navigability.md) |
| [arXiv:2407.04291](https://arxiv.org/abs/2407.04291) | PAPER | **Discriminative ≠ generative** — ASV encoders suppress the intra-speaker variance generation needs | [`12`](12-speaker-manifold-navigability.md) |
| [arXiv:2106.05762](https://arxiv.org/abs/2106.05762) — Interspeech 2021 | PAPER | **1,225 pairwise interpolations**, WER 6.42–7.35%. Interpolation works | [`12`](12-speaker-manifold-navigability.md) |
| [arXiv:2310.03538](https://arxiv.org/abs/2310.03538) — ICASSP 2024 | PAPER | **"Extrapolation is not particularly meaningful"**; same-language pairing is crucial | [`12`](12-speaker-manifold-navigability.md) |
| [arXiv:2508.19210](https://arxiv.org/abs/2508.19210) — APSIPA 2025 | PAPER | **SLERP not LERP** (preserves unit norm); nearest-neighbour same-gender pairing | [`12`](12-speaker-manifold-navigability.md) |
| [arXiv:2005.08601](https://arxiv.org/abs/2005.08601) — Interspeech 2020 | PAPER | **Sparse-region averaging costs +60% relative WER** (10.94% vs 6.83%). The space is not uniformly navigable | [`12`](12-speaker-manifold-navigability.md) |

---

## B. Backends — licence and mechanism ground truth

| Source | Type | What it settled | File |
|---|---|---|---|
| Qwen3-TTS repo — `extract_speaker_embedding()`, `x_vector_only_mode` | REPO | **`(2048,)` vector as the sole identity input**; `enc_dim == hidden_size`. Apache-2.0. **The Tier-1 backend** | [`03`](03-tts-backends-english.md) |
| [`OpenBMB/VoxCPM`](https://github.com/OpenBMB/VoxCPM) — `voxcpm2.py` | REPO | **No speaker vector.** "Voice design" is `final_text = f"({control}){text}"`. Hindi only among Indic | [`03`](03-tts-backends-english.md) |
| [`Zyphra/Zonos`](https://github.com/Zyphra/Zonos) + speaker-embedding card | REPO + CARD | v0.1 dead (last commit 2025-03-05); encoder derives from **VoxBlink2 (CC-BY-NC-SA-4.0)**; returns `(1,128)` **bfloat16, not L2-normalised**; 109 "languages" are eSpeak codes | [`02`](02-identity-representation.md) |
| ZONOS2 (June 2026) | CARD + DOCS | MIT code + Apache-2.0 weights; **Zyphra ship SLERP blending and additive emotion vectors**; bf16 **15.34 GB — does not fit 12 GB** | [`02`](02-identity-representation.md) |
| `cosyvoice3.yaml` — `use_spk_embedding: False` | REPO | CosyVoice: right conclusion, wrong reason. Vector exists, is disabled, no `spk2info.pt` shipped | [`03`](03-tts-backends-english.md) |
| `hexgrad/Kokoro-82M` — `af_heart.pt` | REPO | `torch.Tensor (510,1,256)`; **`ref_s[:,:128]`→decoder, `[128:]`→prosody** — an undocumented free timbre/prosody split | [`03`](03-tts-backends-english.md) [`10`](10-performance-control.md) |
| bilibili MULA §3.4(c), §1.6, §9 | LICENSE | IndexTTS-2 is **not** non-commercial — but bars using it *or its outputs* to improve any AI model; "Use" includes **running**; Chinese text controlling | [`10`](10-performance-control.md) [`08`](08-licensing-propagation.md) |
| Coqui CPML / `coqui.ai/cpml` (404) | LICENSE | XTTS-v2: **code MPL-2.0**, weights CPML. Coqui defunct; the idiap fork cannot relicense | [`03`](03-tts-backends-english.md) |
| VibeVoice README | REPO | **TTS withdrawn 2025-09-05**; Large/7B return HTTP 401; renders carry an audible AI disclaimer | [`03`](03-tts-backends-english.md) |
| `OpenMOSS-Team/MOSS-VoiceGenerator` | CARD + LICENSE | Apache-2.0, ~74k downloads, **trained on cinematic content**. Best usable open voice-design model | [`13`](13-description-to-embedding-prior-art.md) |

---

## C. Indic

| Source | Type | What it settled | File |
|---|---|---|---|
| [arXiv:2505.20693](https://arxiv.org/abs/2505.20693) | PAPER | **IndicF5 = "IN-F5", a fine-tune of English F5-TTS (CC-BY-NC-4.0)** — despite its MIT card | [`04`](04-indic-track.md) [`08`](08-licensing-propagation.md) |
| `ai4bharat/indic-parler-tts` card + Parler source (grepped) | CARD + REPO | **Zero speaker-embedding code.** Identity is a name token in the same string as style. 68 unique names. "Random voice" is the card's own term | [`10`](10-performance-control.md) [`04`](04-indic-track.md) |
| `SPRINGLab/Indic-Mio`, `MioTTS-0.6B`, `MioCodec` cards | CARD | Trained on **Expresso (CC-BY-NC-4.0)**; base + codec declare **Emilia**. **NO-GO** | [`08`](08-licensing-propagation.md) |
| IITM IndicTTS EULA **V2** (Wayback `20241021015937`, md5 `04f70731…`) | LICENSE | **Not non-commercial** — perpetual, sub-licensable, royalty-free. **But §2.2 bars downstream onward sale.** URL 404'd Feb 2026; byte-identical copy in `thennal/indic_tts_ml` | [`08`](08-licensing-propagation.md) |
| `Speech-Lab-IITM/SPRING-INX` → **404** | REPO | SPRING-INX (14 datasets) is **effectively unlicensed** | [`08`](08-licensing-propagation.md) |
| [Sarvam Bulbul v3 blog](https://www.sarvam.ai/blogs/bulbul-v3) | DOCS | Sarvam's own words: *"ElevenLabs v3 alpha leads on audio quality."* Bulbul tops only 8 kHz telephony | [`04`](04-indic-track.md) |
| [`ai4bharat/indicvoices_r`](https://huggingface.co/datasets/ai4bharat/indicvoices_r) + [arXiv:2409.05356](https://arxiv.org/abs/2409.05356) | CARD + PAPER | 1,704 h / 10,496 spk / 22 langs / 93.25% extempore — **all verified**. CC-BY-4.0 chosen "allowing commercial usage" | [`08`](08-licensing-propagation.md) |

---

## D. Data & annotation

| Source | Type | What it settled | File |
|---|---|---|---|
| ParaSpeechCaps dataset card | CARD | **CC-BY-NC-SA-4.0.** Authors released their own fine-tune as NC-SA — answering the propagation question | [`05`](05-datasets-and-annotation.md) [`08`](08-licensing-propagation.md) |
| [`huggingface/dataspeech`](https://github.com/huggingface/dataspeech) — `run_prompt_creation.py` | REPO | **9 columns from 5 tools**; equal-width bins; IPA-chars/sec; **no jitter/shimmer/HNR/tilt/formants/VTL** | [`05`](05-datasets-and-annotation.md) |
| [arXiv:2503.04713](https://arxiv.org/abs/2503.04713) | PAPER | +7.9%/+15.5% MOS verified — **but WER regressed 4.47 → 8.63** | [`05`](05-datasets-and-annotation.md) |
| AnimeVox / AniSpeech / LAION Got Talent cards | CARD | VoicePersona's upstream: **CC-BY-NC**, MIT-over-anime-audio, and **GPT-4o Audio output using OpenAI's proprietary voices** (483 tarballs named `alloy`…`verse`) | [`05`](05-datasets-and-annotation.md) [`08`](08-licensing-propagation.md) |
| `ESpeech/ESpeech-igm` disabled, May 2026 | — | HuggingFace acted on a voice actor's complaint about a permissive licence over scraped voice work | [`08`](08-licensing-propagation.md) |
| Vaessen & van Leeuwen 2022 | PAPER | At fixed 100 h, 60× speakers halves ECAPA EER — **but one session each is worse than 100 speakers**. "Speaker count × session diversity" | [`05`](05-datasets-and-annotation.md) |

---

## E. Evaluation

| Source | Type | What it settled | File |
|---|---|---|---|
| [arXiv:2606.19951](https://arxiv.org/abs/2606.19951) | PAPER | **MOS predictors anti-correlated with pitch**: DNSMOS r = −0.788, UTMOSv2 −0.722, **humans −0.059**. All six moved <0.1 on corruption costing humans 1.84 MOS | [`06`](06-evaluation-harness.md) |
| [arXiv:2506.16381](https://arxiv.org/abs/2506.16381) | PAPER + dataset (MIT) | InstructTTSEval: 12 attributes verified. **Binary** judge (`gemini-2.5-pro`). **Human ceiling 84.3 avg / 67.2 Role-Play.** ~$12.8/session EN | [`06`](06-evaluation-harness.md) [`13`](13-description-to-embedding-prior-art.md) |
| Noé et al., Interspeech 2020 — [DOI 10.21437/Interspeech.2020-2720](https://doi.org/10.21437/Interspeech.2020-2720) | PAPER | **GVD** — a diversity metric with a published scale and baselines | [`12`](12-speaker-manifold-navigability.md) |
| `ttsds` 2.1.3 (MIT) / TTSDS2 | REPO | Only metric of 16 clearing Spearman 0.50 in **every** domain | [`06`](06-evaluation-harness.md) |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | CARD | 256-d, 0.723% EER, fbank→ResNet — **zero lineage overlap** with WavLM/ECAPA conditioning | [`06`](06-evaluation-harness.md) |
| [arXiv:2604.26347](https://arxiv.org/abs/2604.26347) | PAPER | emotion2vec cosine drops **below chance** under speaker distractors — caveat on every emotion number | [`10`](10-performance-control.md) |

---

## F. Serving & economics

| Source | Type | What it settled | File |
|---|---|---|---|
| `nanovllm-voxcpm` README benchmark; vLLM-Omni measurement | REPO | Two published figures disagree **3.2×**. **Short prompts are faster in aggregate** (112.6 vs 96.0 audio-s/GPU-s at c=32) | [`07`](07-serving-and-cost.md) |
| [Modal cold-start docs](https://modal.com/docs) | DOCS | Snapshots *"will generally not improve your cold start times — and may even worsen them"* when weight-loading-bound. Their 45s→5s win was a **0.5B** model | [`07`](07-serving-and-cost.md) |
| RunPod / Beam pricing pages | DOCS | Community 4090 $0.34/hr → **$248.20/mo warm**; Beam 4090 $306.60; cheapest with SLA = Secure L4 $357.70 | [`07`](07-serving-and-cost.md) |
| ElevenLabs / Cartesia / OpenAI pricing | DOCS | $0.075 / $0.028 / $0.01125 per min → **break-even 55 h/mo and 368 h/mo** | [`07`](07-serving-and-cost.md) |
| [ElevenLabs OpenAPI](https://api.elevenlabs.io/openapi.json) | SPEC | 300 paths, **zero voice vectors**. `generated_voice_id` addresses stored audio. `played_not_selected_voice_ids` "used for RLHF" | [`11`](11-production-api-landscape.md) |
| [Cartesia API changes](https://docs.cartesia.ai/build-with-cartesia/tts-models/api-changes.md) | SPEC | **192-d `Embedding` schema and `/voices/mix` sunset 2026-06-01, no replacement.** *"Embeddings are not accepted in this API version"* | [`11`](11-production-api-landscape.md) |
| [Resemble voice-design docs](https://docs.resemble.ai/voice-creation/voice-design/generate) | SPEC | Tier 2, named as such: *"Create a voice clone from a selected voice design candidate."* **4-hour preview TTL** | [`11`](11-production-api-landscape.md) |
| [Hume voice design](https://dev.hume.ai/docs/voice/voice-design) | DOCS | *"stores both the speech and the prompt that shaped it"* — the Tier-2 identity record, described by a vendor | [`11`](11-production-api-landscape.md) |

---

## G. Safety, watermarking, regulation

| Source | Type | What it settled | File |
|---|---|---|---|
| AudioSeal LICENSE + README changelog + model card | LICENSE | **MIT for code *and* weights** — *"including the license for the model weights… you can use AudioSeal in commercial application too!"* Verified four ways | [`09`](09-safety-and-watermarking.md) |
| C2PA soft-binding registry — `com.aiwatermark.audioseal.1` (2026-03-08) | REG | AudioSeal is the standards-endorsed choice | [`09`](09-safety-and-watermarking.md) |
| AudioMarkBench / RAW-Bench | PAPER | **Polarity inversion → 0.18/0.00**; **Opus removes it**; Descript 0.00; reverb 0.22; **OGG/Vorbis 0.95 ✓**; MP3 1.00 @32kbps | [`09`](09-safety-and-watermarking.md) |
| XAttnMark duration ablation | PAPER | Detection 98.6–99.3% at 1–10 s; **attribution collapses to 81.2% at 1 s** | [`09`](09-safety-and-watermarking.md) |
| **EU AI Act Art. 50(2); Art. 2(12); Code of Practice** | REG | **Applied 2026-08-02.** No grace for new systems. Art. 50 **excluded** from the open-source exemption. **Two marking layers for audio + a free public detector** | [`09`](09-safety-and-watermarking.md) [`08`](08-licensing-propagation.md) |
| **India IT Rules, notified 2026-02-10** | REG | Not SSMI-limited. Prohibited-use blocking, permanent non-removable metadata, **"prominently prefixed audio disclosure."** The %-of-display-area rule did **not** survive | [`09`](09-safety-and-watermarking.md) |
| *Lehrman v. Lovo* | REG | Copyright claims dismissed; **right-of-publicity claims allowed through** | [`08`](08-licensing-propagation.md) |
| *Midler*, *Waits*; *Arijit Singh* ¶18; ELVIS Act; NO FAKES §2(c)(2)(B) | REG | Imitation actionable **without copying**. No-upload keeps us outside the core of the statutes but not the tort | [`09`](09-safety-and-watermarking.md) [`08`](08-licensing-propagation.md) |

---

## H. Explicitly UNVERIFIED

Carried forward so nothing is silently upgraded to fact.

| Item | Status | Settled by |
|---|---|---|
| Zonos-specific manifold navigability | No published test found (250 issues, 858 comments, 51 HF discussions, 24 Spaces swept; Reddit unreachable) | Moot — v0.1 is out |
| AudioSeal on 1–3 s clips | **Nobody has published it.** Paper prose contradicts its own Table 6 | **E5** |
| Identity drift under *heavy stylisation* | Nobody has measured our actual use case | **E9** |
| Speaker-count ablation for a description→embedding mapper | No publication across ten systems checked | **E7** |
| Indic Parler-TTS / Parler-TTS speed | **None published anywhere** | **E8** |
| TTS render-cache hit rates | No published data for any product | measure in production |
| Google Cloud TTS / Azure Speech list prices | Pages render client-side; third-party numbers refused | direct quote |
| VoiceMOS 2026 results | Released to participants 2026-08-31; not public | re-check after 2026-09-16 |
| VPC 2026 results | Workshop **2026-09-26** | re-check |
| India prefixed-audio-disclosure applicability to a 1.5 s API-delivered line | Legal, not research | **counsel** |

---

*Compiled 2026-09-02 from 13 domain research files. Each file carries its full source table; this indexes what changed a decision.*
