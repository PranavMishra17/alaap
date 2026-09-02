# 03 — TTS Backend Landscape (Component C) & Licence Verification

> **Domain:** the frozen renderer; speaker-conditioning mechanisms; licence ground truth
> **Answers:** B1, B2, B6
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [02-identity-representation.md](02-identity-representation.md) · [04-indic-track.md](04-indic-track.md) · [07-serving-and-cost.md](07-serving-and-cost.md) · [08-licensing-propagation.md](08-licensing-propagation.md)

## 0. Bottom line

- **Qwen3-TTS Base is the only verified Tier-1 backend.** Its inference API exposes `model.extract_speaker_embedding(audio, sr) -> Tensor (D,)` and an `x_vector_only_mode=True` path in which the reference audio token sequence is set to `None` and the speaker vector is the *sole* identity carrier, injected as a single token embedding via `speaker_embed.view(1, 1, -1)`. `generate_voice_clone(voice_clone_prompt={...})` accepts a hand-built dict, so a trained mapper can write its output straight into `ref_spk_embedding` with no audio anywhere in the loop. Apache-2.0 code *and* weights. **HIGH** — read from source. [S12][S13][S14][S15]
- **VoxCPM2 has no addressable speaker vector at all.** Its "Voice Design" is a *text-prefix string hack*: the demo builds `final_text = f"({control}){text}"` and passes it as ordinary text. The strings `speaker`, `spk_`, `embed`, `design`, `description` appear **zero times** in `model/voxcpm2.py`. Identity conditioning is a VAE-encoded latent *patch sequence* `(T, P, D)` spliced in as a prefix — structurally the same failure that killed the CosyVoice attempt. **HIGH** — read from source. [S5][S6][S7]
- **The brief is wrong about why CosyVoice failed, but right that it fails.** CosyVoice 2/3 *does* compute a 192-d CAMPPlus speaker embedding (`_extract_spk_embedding`, `llm_embedding`/`flow_embedding`). But `cosyvoice3.yaml` ships `use_spk_embedding: False  # change to True during sft`, CosyVoice2/3 ship **no `spk2info.pt`** (only v1's `CosyVoice-300M-SFT` does), and zero-shot inference additionally requires `flow_prompt_speech_token` + `prompt_speech_feat`. So the vector exists but is inert and non-load-bearing. Conclusion stands; stated reason needs correcting. **HIGH**. [S16][S17][S18][S19]
- **VoxCPM2 does support Hindi** — verbatim in the official 30-language list, and it is the *best* model in OpenBMB's own Hindi benchmark (CER 0.79%). It is Apache-2.0 for **both** code and weights. This genuinely does let one Apache-2.0 backend cover the English and Indic tracks — but only at Tier 2, never Tier 1. **HIGH**. [S2][S3][S4]
- **Voice design and speaker-vector cloning are in *different Qwen3-TTS checkpoints*.** `Qwen3-TTS-12Hz-1.7B-VoiceDesign` has `"speaker_encoder_config": null` in its `config.json` — no speaker encoder at all — and `generate_voice_design()` raises unless `tts_model_type == "voice_design"`. You cannot do description→vector→render inside one Qwen checkpoint. **HIGH**. [S13][S20]
- **Best description-driven voice design ≠ best Tier-1 vector.** They are different models, and for Qwen they are different *checkpoints of the same family*. See §7.
- **Licence corrections that change the answer:** XTTS-v2 code is **MPL-2.0** (not CPML) but weights are **CPML non-commercial** → disqualified. IndexTTS-2 is **not** non-commercial — it is a Llama-style licence with a 100M-MAU / RMB 1B-revenue threshold → *is* servable, with one dangerous clause. OmniVoice is **Apache-2.0 code + CC-BY-NC weights** → disqualified. **HIGH**. [S21][S22][S23][S28]
- **Microsoft did withdraw VibeVoice-TTS.** Verbatim from the repo README: "we have removed the VibeVoice-TTS code from this repository." `microsoft/VibeVoice-Large` and `-7B` return HTTP 401. A successor, **VibeVoice-Realtime-0.5B** (MIT), exists but ships voices "in an embedded format" specifically to prevent cloning. **HIGH**. [S24][S25][S26]
- **`seed` is not a coherent identity anchor.** VoxCPM2's seed path is bare `torch.manual_seed` + `cuda.manual_seed_all`, and `retry_badcase` (default `True` in `core.py`) silently does `current_seed += 1` on retry. Tier 3 is incoherent even *within* one machine. **HIGH**. [S8][S9]
- **Kokoro is the second verified Tier-1 backend, and the brief under-rated it to a one-line footnote.** Apache-2.0 code **and** weights, an explicitly commercial-friendly model card, 82M params, **Hindi included**, and a real style-vector pack — `voices/af_heart.pt` was downloaded and loaded: `torch.Tensor`, shape **`(510, 1, 256)`**, float32. `load_voice()` accepts a raw tensor directly and already blends packs with `torch.mean`. **HIGH — verified by download, not inference.** [S38]
- **Three licence facts that invalidate common assumptions:** **Fish-Speech changed licence** — the code is now a bespoke "Fish Audio Research License Agreement" (dated 2026-03-07) that names hosted APIs as commercial use; any note saying "CC-BY-NC-SA" is stale on the code side. **Higgs Audio v2 is not Apache** despite the repo badge — weights are a Llama-3-derived "Boson Higgs Audio 2 Community License" with a 100k-active-user cliff. **YourTTS weights are CC BY-NC-ND 4.0** per Coqui's own `.models.json` — the ND term also forbids fine-tuning. **HIGH.** [S39][S40][S41]
- **Two models the brief missed:** **OmniVoice** (k2-fsa, 600+ languages incl. 9 Indic, first-class `instruct=` voice-design parameter — but CC-BY-NC weights) and **Parler-TTS** (Apache-2.0 code and weights, the canonical description-conditioned architecture). **HIGH**. [S27][S28][S29]

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| 1 | VoxCPM2 API is `generate(text, reference_wav_path, cfg_value, seed)` | **Partly wrong** | Real signature is `_generate(text, prompt_wav_path, prompt_text, reference_wav_path, cfg_value, inference_timesteps, min_len, max_len, normalize, denoise, retry_badcase, retry_badcase_max_times, retry_badcase_ratio_threshold, streaming, seed)`. There is **no voice-description parameter**. | `src/voxcpm/core.py` L183-199 | HIGH [S6] |
| 2 | VoxCPM2 has a "voice design from description text" mode | **Technically true, mechanically not what you think** | Implemented as string concatenation in the *demo app*, not the model API: `final_text = f"({control}){text}"`. The description is tokenised as ordinary text. | `app.py` L318-322 | HIGH [S7] |
| 3 | VoxCPM2 = Apache-2.0 | **Correct, and stronger than claimed** | Apache-2.0 for code (LICENSE file) **and** weights (HF YAML `license: apache-2.0`). README: "Weights and code released under the Apache-2.0 license, free for commercial use." | LICENSE + HF YAML + README L54 | HIGH [S1][S3][S4] |
| 4 | VoxCPM2 "30 languages" — Indic status unknown | **Hindi IS included** | Hindi is in the verbatim 30-language list and benchmarked at CER 0.79% (beats Fish S2-Pro's 0.91%). No other Indic language is present (no Bengali/Tamil/Telugu/Marathi/Gujarati/Urdu). | README L57, L539 | HIGH [S2] |
| 5 | VoxCPM2 variants: 2B / 0.6B / 0.5B | **Correct** | `openbmb/VoxCPM2` (2B), `openbmb/VoxCPM1.5` (0.6B), `openbmb/VoxCPM-0.5B` (0.5B). All Apache-2.0. | README L381 | HIGH [S2] |
| 6 | Qwen3-TTS has a jointly-trained speaker encoder | **Correct AND externally addressable** | ECAPA-TDNN speaker encoder; `extract_speaker_embedding()` is a public method; `enc_dim` = 2048 (1.7B) / 1024 (0.6B), equal to talker `hidden_size`. Injected as one token embedding. | `modeling_qwen3_tts.py` L1941-1954, L2171; `config.json` | HIGH [S14][S20] |
| 7 | Qwen3-TTS is Apache-2.0 | **Correct** | LICENSE file = Apache-2.0; HF YAML `license: apache-2.0` on Base, VoiceDesign and 0.6B. Paper abstract: "we release both tokenizers and models under the Apache 2.0 license." | LICENSE + HF YAML + arXiv | HIGH [S11][S12][S20] |
| 8 | Qwen3-TTS: no Hindi/Indian languages | **Correct** | 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian. No Indic. | README + HF card | HIGH [S12] |
| 9 | arXiv 2601.15621 = Qwen3-TTS Technical Report | **Correct, exists** | Submitted 2026-01-22, Hu/Zhu/He/…/Lin. | arxiv.org/abs/2601.15621 | HIGH [S11] |
| 10 | CosyVoice 3 "conditions on a reference-audio token sequence with **no addressable speaker vector**" | **Reason wrong, conclusion right** | A 192-d CAMPPlus vector *does* exist and *is* plumbed. But `cosyvoice3.yaml` has `use_spk_embedding: False`, CosyVoice2/3 ship no `spk2info.pt`, and zero-shot needs `flow_prompt_speech_token` + `prompt_speech_feat` regardless. The vector is present but inert. | `frontend.py` L108-118, L168-184; `cosyvoice3.yaml` L11/L164 | HIGH [S16][S17][S19] |
| 11 | CosyVoice 2/3 licence | **Apache-2.0, code and weights** | LICENSE = Apache-2.0; HF YAML `apache-2.0` for `CosyVoice2-0.5B` and `Fun-CosyVoice3-0.5B-2512`. | LICENSE + HF YAML | HIGH [S18] |
| 12 | "CosyVoice 3" as a downloadable model | **Name is wrong** | The released artefact is **`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`** ("Fun-CosyVoice 3.0"), 0.5B, 9 languages + 18 Chinese dialects, **no Hindi**. Class `CosyVoice3` exists in `cosyvoice/cli/cosyvoice.py` L189. | README L5; HF card | HIGH [S18][S31] |
| 13 | Chatterbox = MIT, ref audio + emotion exaggeration | **Correct** | LICENSE = "MIT License, Copyright (c) 2025 Resemble AI". `exaggeration` is a real parameter on `prepare_conditionals`. | LICENSE; `tts.py` L182 | HIGH [S32][S33] |
| 14 | Chatterbox Indic support unknown | **Hindi supported, with a dedicated checkpoint** | Chatterbox Multilingual covers 23 languages incl. Hindi, plus a single-language pack `ResembleAI/Chatterbox-Multilingual-hi`. Relevant to [04-indic-track.md](04-indic-track.md). | GitHub README | MEDIUM [S32] |
| 15 | F5-TTS = CC-BY-NC weights | **Correct, but code and weights differ** | `LICENSE` = "MIT License, Copyright (c) 2024 Yushen CHEN"; HF YAML `license: cc-by-nc-4.0`. **MIT code, non-commercial weights.** Disqualified. | LICENSE + HF YAML | HIGH [S34][S46] |
| 16 | XTTS-v2 = CPML non-commercial | **Correct for weights; code claim wrong** | Code is **MPL-2.0** (`LICENSE.txt`, both `coqui-ai/TTS` and the maintained `idiap/coqui-ai-TTS`). Weights are **Coqui Public Model License 1.0.0**: "This license allows only non-commercial use." | LICENSE.txt; HF YAML; CPML text | HIGH [S21][S22] |
| 17 | XTTS-v2 is "vector-addressable" via `gpt_cond_latent` + `speaker_embedding` | **Half wrong** | `speaker_embedding` is a fixed vector, but **`gpt_cond_latent` is a variable-length sequence `[1, 1024, T]`** built from `self.gpt.get_style_emb(mel).transpose(1, 2)`. Both are required by `inference()`. Not a clean Tier-1 vector. | `TTS/tts/models/xtts.py` | HIGH [S23] |
| 18 | Coqui shut down — who maintains it? | **`idiap/coqui-ai-TTS`** | Maintained by Idiap Research Institute, PyPI package `coqui-tts`, code MPL-2.0. **The fork does not and cannot change the weights licence** — CPML is Coqui's grant on the checkpoint. XTTS stays disqualified. | idiap/coqui-ai-TTS | HIGH [S22] |
| 19 | IndexTTS-2 = "restrictive/non-commercial" | **WRONG** | "bilibili Model Use License Agreement" is Llama-style: commercial use permitted **unless** >100M MAU or >RMB 1B revenue (§2.2). We are far below both. It *is* publicly servable. **But** §2(c) forbids using it "to improve any AI model" except itself/derivatives/non-commercial models — a real hazard for our trained mapper. | LICENSE §2.2, §2(c) | HIGH [S35] |
| 20 | VibeVoice = research-only; "check whether Microsoft withdrew it" | **Licence MIT; TTS genuinely withdrawn** | LICENSE = MIT (Microsoft). README, verbatim: "we have removed the VibeVoice-TTS code from this repository." `VibeVoice-Large`/`-7B` → HTTP 401. `VibeVoice-1.5B` → HTTP 200. Successor `VibeVoice-Realtime-0.5B` is MIT but ships voices "in an embedded format" to block cloning, and the docs say "not recommend using VibeVoice in commercial or real-world applications." | LICENSE; README; HF API | HIGH [S24][S25][S26] |
| 21 | Model list is complete | **Two omissions** | **OmniVoice** (k2-fsa, 600+ langs, first-class `instruct=`, Apache code / **CC-BY-NC weights**) and **Parler-TTS** (Apache-2.0 code+weights, FLAN-T5 description cross-attention). See §6. | See §6 | HIGH [S27][S28][S29] |
| 22 | Kokoro treated as a one-line also-ran | **Badly under-rated** | Apache-2.0 code **and** weights, ungated, card explicitly welcomes commercial deployment. Ships a real style-vector pack `(510, 1, 256)` float32, `load_voice()` takes a raw tensor, blending via `torch.mean` is upstream. Has **Hindi** (`h='hi'`, 4 voices). **Second Tier-1 backend.** | `pipeline.py` L31/167-176/242; `af_heart.pt` loaded | HIGH [S38] |
| 23 | Fish-Speech/OpenAudio = CC-BY-NC-SA | **Stale** | Weights are `cc-by-nc-sa-4.0`, but the **code** is now a bespoke "FISH AUDIO RESEARCH LICENSE AGREEMENT" (Last Updated 2026-03-07); GitHub reports `NOASSERTION`. §III names *"a hosted service or application programming interface"* as Commercial Purpose. Both halves disqualify. | LICENSE §III, §V | HIGH [S39] |
| 24 | Higgs Audio implied permissive | **Wrong** | Code Apache-2.0, but weights are the **"Boson Higgs Audio 2 Community License Agreement"**, built on the Meta Llama 3 Community License; HF YAML `license: other`. §2: >100,000 annual active users requires a discretionary expanded licence. Repo renamed → `bosonai/higgs-tts-2-3b-base`. | HF LICENSE file | HIGH [S40] |
| 25 | YourTTS assumed usable | **Wrong** | Code MPL-2.0, but Coqui's own manifest `TTS/.models.json` gives `tts_models/multilingual/multi-dataset/your_tts` → `"license": "CC BY-NC-ND 4.0"`. Non-commercial **and** no-derivatives (blocks fine-tuning). Technically an ideal 512-d d-vector model; legally dead. | `TTS/.models.json` | HIGH [S41] |
| 26 | StyleTTS2 weights assumed MIT like the code | **Wrong** | Code MIT. **Weights have NO declared licence at all** — `yl4579/StyleTTS2-LJSpeech` and `-LibriTTS` both have `cardData: None`. The only term is a README disclosure obligation, not an OSI grant. Absence of a grant is worse than a known-bad one. | HF API `cardData` | HIGH [S42] |
| 27 | Orpheus assumed cleanly Apache | **Needs a legal read** | HF YAML says `apache-2.0`, but the repo is `gated: auto` and `base_model: meta-llama/Llama-3.2-3B-Instruct` — the Llama 3.2 Community Licence arguably propagates. Also has a gated Hindi research release. | HF card YAML | MEDIUM [S43] |

## 2. THE REVISED MODEL TABLE (replaces scope section 6)

| Model | Params | Code licence | Weights licence | Licence source URL | Conditioning mechanism | Speaker vector? | Voice design from text? | Indic? | VRAM fp16 | Fits 12GB? | Publicly servable? | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B-Base** | 1.7B | Apache-2.0 [S12] | Apache-2.0 [S20] | [LICENSE](https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/LICENSE) | ECAPA-TDNN x-vector (single token embed) **or** ICL ref tokens | **YES — `(2048,)`, `x_vector_only_mode`** | No (separate ckpt) | No | ~4.5 GB weights, ~7 GB run | Yes | **YES** | **Tier-1 backend. Primary pick.** |
| **Qwen3-TTS-12Hz-0.6B-Base** | 0.6B | Apache-2.0 | Apache-2.0 [S20] | same | same | **YES — `(1024,)`** | No | No | ~2.5 GB weights, ~4 GB run | Yes (and 8GB) | **YES** | Tier-1 fallback for 8GB. |
| **Qwen3-TTS-12Hz-1.7B-VoiceDesign** | 1.7B | Apache-2.0 | Apache-2.0 [S20] | same | Natural-language `instruct=` string | **NO** (`speaker_encoder_config: null`) | **YES — first-class** | No | ~4.5 GB weights | Yes | **YES** | **Best voice-design model. Use as Tier-2 minter.** |
| **VoxCPM2** | 2B | Apache-2.0 [S1] | Apache-2.0 [S3] | [LICENSE](https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/LICENSE) | VAE latent patch sequence `(T,P,D)` as prefix | **NO** | Yes, but as text prefix `"(desc)text"` | **Hindi only** | ~8 GB (README) | Yes | **YES** | Tier-2 only. Best Apache Hindi option. |
| **VoxCPM1.5** | 0.6B | Apache-2.0 | Apache-2.0 [S4] | same | ref wav + transcript (continuation) | **NO** | No | Unstated | ~6 GB (README) | Yes (and 8GB) | **YES** | Tier-2, 8GB fallback. |
| **VoxCPM-0.5B** | 0.5B | Apache-2.0 | Apache-2.0 [S4] | same | ref wav + transcript | **NO** | No | No | ~5 GB (README) | Yes | **YES** | Legacy. |
| **Fun-CosyVoice3-0.5B-2512** | 0.5B | Apache-2.0 [S16] | Apache-2.0 [S18] | [LICENSE](https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/LICENSE) | prompt speech tokens + mel; 192-d CAMPPlus vector present but `use_spk_embedding: False` | **Inert** — exists, unused | `instruct` (emotion/dialect, not identity) | **No** | ~2 GB | Yes | **YES** | Rejected: identity not vector-addressable. |
| **CosyVoice2-0.5B** | 0.5B | Apache-2.0 | Apache-2.0 [S18] | same | same | **Inert** | `inference_instruct2` | No | ~2 GB | Yes | **YES** | Superseded by Fun-CosyVoice3. |
| **Chatterbox (Multilingual)** | ~0.5B | MIT [S33] | MIT [S30] | [LICENSE](https://raw.githubusercontent.com/resemble-ai/chatterbox/master/LICENSE) | `Conditionals{T3Cond(speaker_emb 256-d, cond_prompt_speech_tokens), s3gen_ref_dict(ref mel + tokens + x-vector)}` | **Partial** — 256-d vector drives T3, but S3Gen needs ref mel | No | **Hindi + `-hi` pack** | ~3 GB | Yes | **YES** | **Tier-1.5:** `Conditionals.save()` is a persistable identity blob. |
| **XTTS-v2** | ~0.5B | **MPL-2.0** [S21] | **CPML 1.0.0 (non-commercial)** [S22] | [HF LICENSE.txt](https://huggingface.co/coqui/XTTS-v2/raw/main/LICENSE.txt) | `gpt_cond_latent [1,1024,T]` (sequence) + `speaker_embedding` (vector) | Partial — latent is a sequence | No | Hindi (17 langs) | ~4 GB | Yes | **NO** | **Disqualified: CPML.** |
| **IndexTTS-2** | ~1.5B | bilibili MULA [S35] | bilibili MULA (§1.4 covers weights *and* code) | [LICENSE](https://raw.githubusercontent.com/index-tts/index-tts/main/LICENSE) | ref wav (timbre) + separate emotion ref/vector | UNVERIFIED | Emotion description | UNVERIFIED | UNVERIFIED | UNVERIFIED | **YES** (under threshold) | Usable, but §2(c) "improve any AI model" clause is a mapper hazard. |
| **VibeVoice-TTS (1.5B / Large)** | 1.5B / 7B | MIT [S24] | MIT (1.5B); Large **gone** (HTTP 401) | [LICENSE](https://raw.githubusercontent.com/microsoft/VibeVoice/main/LICENSE) | ref wav, multi-speaker | NO | No | No | UNVERIFIED | — | **Effectively no** | **Withdrawn by Microsoft 2025-09-05.** |
| **VibeVoice-Realtime-0.5B** | 0.5B | MIT | MIT [S26] | [HF YAML](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | **Pre-embedded fixed voice set** (cloning deliberately blocked) | Closed set only | No | No (EN only) | ~1.5 GB | Yes | Legally yes; docs discourage | Rejected: closed voice set. |
| **OmniVoice** | 0.6B | Apache-2.0 [S27] | **CC-BY-NC** [S28] | [HF card §License](https://huggingface.co/k2-fsa/OmniVoice) | ref wav, **or** `instruct=` attribute string | UNVERIFIED | **YES — first-class `instruct=`** | **9 Indic langs** | ~2 GB | Yes | **NO** | **Disqualified: CC-BY-NC weights** (Emilia data). |
| **Parler-TTS** | 0.88B (large-v1) | Apache-2.0 [S29] | Apache-2.0 [S29] | [LICENSE](https://raw.githubusercontent.com/huggingface/parler-tts/main/LICENSE) | FLAN-T5 description encoder → cross-attention | No (sequence) | **YES — native architecture** | No | ~2.5 GB | Yes | **YES** | Only fully-Apache native description-conditioned model. |
| **F5-TTS** | 0.34B | MIT [S34] | **CC-BY-NC-4.0** [S46] | [HF YAML](https://huggingface.co/SWivid/F5-TTS) | ref wav + transcript (in-context) | **NO** | No | No | ~2 GB | Yes | **NO** | Rejected twice: NC weights **and** no vector. |
| **Kokoro** | **82M** | Apache-2.0 [S38] | **Apache-2.0** [S38] | [HF card](https://huggingface.co/hexgrad/Kokoro-82M) | **Pure style-vector lookup — no reference audio in the inference path** | **YES — pack `(510,1,256)`, `[1,256]` per call** | No | **Hindi (4 voices)** | UNVERIFIED (82M → trivial) | Yes | **YES** | **Tier-1 backend. Cheapest by far.** |
| **StyleTTS2** | 0.15B | MIT [S36] | **NONE DECLARED** [S42] | [LICENSE](https://raw.githubusercontent.com/yl4579/StyleTTS2/main/LICENSE) | ref wav → `compute_style()`; or style diffusion from sentence BERT | **YES — `ref_s` 256-d (128 timbre + 128 prosody)** | No | No | UNVERIFIED | Yes | **NO — no grant** | Kokoro's base model, but weights carry no licence. |
| **YourTTS** | ~0.09B | MPL-2.0 [S21] | **CC BY-NC-ND 4.0** [S41] | [`TTS/.models.json`](https://raw.githubusercontent.com/coqui-ai/TTS/dev/TTS/.models.json) | external H/ASP **d-vector** | **YES — 512-d `d_vectors`** | No | No (en/fr/pt-br) | UNVERIFIED | Yes | **NO** | Technically ideal, legally dead (NC **and** ND). |
| **Orpheus** | 3B | Apache-2.0 [S43] | apache-2.0, **gated**, Llama-3.2-3B-derived [S43] | [HF card](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) | **voice name string prepended**: `f"{voice}: {prompt}"` | **NO** | No (fixed name set) | Hindi research release (gated) | UNVERIFIED | Likely no (3B) | Yes* | No vector → rejected. |
| **Higgs Audio v2 / Higgs TTS 2** | 3.6B + 2.2B | Apache-2.0 [S40] | **Boson Community Licence (100k AAU cap)** [S40] | [HF LICENSE](https://huggingface.co/bosonai/higgs-tts-2-3b-base) | ChatML: `<\|scene_desc_start\|>` description, `[SPEAKER0]` tags, or ref audio → codec tokens | **NO** | **YES — free-form scene/voice description** | No (en/zh/de/ko) | **≥24 GB stated** | **NO** | UNCLEAR | Fails VRAM **and** licence ceiling. |
| **Fish-Speech / OpenAudio** | UNVERIFIED | **Fish Audio Research Licence (NC)** [S39] | **CC-BY-NC-SA-4.0** [S39] | [LICENSE](https://raw.githubusercontent.com/fishaudio/fish-speech/main/LICENSE) | ref audio bytes + text → codec tokens | **NO** | No | No | UNVERIFIED | — | **NO** | §III names hosted APIs as commercial. Disqualified twice. |
| **Dia** | 1.6B | Apache-2.0 [S44] | **Apache-2.0** [S44] | [HF card](https://huggingface.co/nari-labs/Dia-1.6B-0626) | `[S1]`/`[S2]` dialogue tags + optional audio prompt | **NO** — *"different voices every time you run"* | No | No (EN only) | **~4.4 GB fp16 (stated)** | Yes | Yes | Legally clean, but **no addressable identity at all**. |
| **Sesame CSM** | 1B | Apache-2.0 [S45] | **Apache-2.0** [S45] | [HF card](https://huggingface.co/sesame/csm-1b) | `generate(text, speaker: int, context: List[Segment])` | **NO** — the int is string-formatted into the prompt: `encode(f"[{speaker}]{text}")` | No | No | UNVERIFIED | Yes | Yes | The `speaker: int` is a decoy, not an embedding index. |

> Every non-`UNVERIFIED` cell traces to a numbered source in §9. Cells marked `UNVERIFIED` are carried to §8 as experiments/follow-ups rather than guessed.

## 3. B1 — VoxCPM2 in depth

### 3.1 Full language list (verbatim from official source)

From `README.md` L56-59 of `OpenBMB/VoxCPM` @ `main` [S2]:

> **🌍 Supported Languages (30)**
> Arabic, Burmese, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese
>
> Chinese Dialect: 四川话, 粤语, 吴语, 东北话, 河南话, 陕西话, 山东话, 天津话, 闽南话

**Answer to the unification question: partially yes.** Hindi is present and *strong* — OpenBMB's internal 30-language ASR benchmark (README L539) reports `hi (Hindi) | CER | 0.79% | 0.91%` (VoxCPM2 vs Fish S2-Pro), i.e. VoxCPM2 wins. A separate comparison table (L470, L503) gives Hindi WER 5.827 and speaker-similarity 85.6 as best-in-column.

**But**: Hindi is the *only* Indian language. There is no Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi or Urdu. If the Indic track means "Hindi", VoxCPM2 unifies it on one Apache-2.0 backend. If it means "Indian languages", it does not — and OmniVoice (§6.1) is the only model found with broad Indic coverage, but its weights are CC-BY-NC. See [04-indic-track.md](04-indic-track.md).

### 3.2 API surface & conditioning

Actual signature, `src/voxcpm/core.py` L183-199 [S6] (`generate()` is a thin wrapper: `return next_and_close(self._generate(*args, streaming=False, **kwargs))`):

```python
def _generate(
    self, text: str, prompt_wav_path: str = None, prompt_text: str = None,
    reference_wav_path: str = None, cfg_value: float = 2.0,
    inference_timesteps: int = 10, min_len: int = 2, max_len: int = 4096,
    normalize: bool = False, denoise: bool = False,
    retry_badcase: bool = True, retry_badcase_max_times: int = 3,
    retry_badcase_ratio_threshold: float = 6.0,
    streaming: bool = False, seed: Optional[int] = None,
) -> Generator[np.ndarray, None, None]
```

**There is no voice-description parameter.** The three documented modes map onto argument combinations:

| README mode | Arguments | What actually happens |
|---|---|---|
| Voice Design | *(none — text only)* | Description is **prepended to the text string** |
| Controllable Cloning | `reference_wav_path` | VAE-encode ref → prefix token block |
| Ultimate Cloning | `prompt_wav_path` + `prompt_text` | Continuation-mode prefix |

**Voice Design is string formatting.** `app.py` L318-322 [S7]:

```python
control = re.sub(r"[()（）]", "", control).strip()
final_text = f"({control}){text}" if control else text
```

The parenthesised description is tokenised by the ordinary text tokenizer and prepended to the sentence to be spoken. Consequences: (a) the description is re-interpreted on **every** utterance, so voice drifts line to line; (b) there is nothing to persist but the string; (c) `reference_wav_path` and voice design are **mutually exclusive** in the UI, and `_generate` rejects `reference_wav_path` on non-v2 models (`core.py` L243).

**No speaker vector exists.** Grepping `src/voxcpm/model/voxcpm2.py` (1279 lines) for `speaker|spk_|voice_embed|design|description|instruct` returns **zero matches**. The identity path is `_encode_wav()` → `(T, P, D)` latent patch tensor → `_make_ref_prefix()` → `[ref_start, ref_audio…, ref_end]` token/feature block concatenated ahead of the text (`voxcpm2.py` L401-460) [S5]. This is a **variable-length sequence**, not an addressable vector — architecturally the same shape of problem as CosyVoice.

`build_prompt_cache()` / `merge_prompt_cache()` (L695-787) look like an identity-persistence hook but are not: they require a `prompt_wav_path` or `reference_wav_path`, and return a model-internal cache dict. That is a per-version speedup artefact, not a portable, interpolatable identity.

**Verdict: VoxCPM2 supports Tier 2 only.** Mint a 10s clip once (voice design, description in the text prefix), then clone from that clip forever via `reference_wav_path`. Tier 1 is not expressible.

### 3.3 Determinism of `seed`

From `src/voxcpm/model/utils.py` L27-37 [S9]:

```python
def materialize_generation_seed(seed: Optional[int]) -> int:
    if seed is not None: return int(seed)
    return int(torch.seed() & 0xFFFFFFFF)

def apply_generation_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
```

This is bare PyTorch seeding. Assessment:

- **Same process, same machine, same versions: probably reproducible.** Requires no non-deterministic CUDA kernels; not guaranteed, and `torch.use_deterministic_algorithms` is never set. **MEDIUM**.
- **Across machines: NO.** Different GPU architecture, CUDA/cuDNN version or kernel autotuning changes floating-point reduction order. Nothing in this code pins that. **HIGH**.
- **Across versions: NO.** Any weight or architecture change re-maps the noise → voice function entirely. **HIGH**.
- **Worse — the seed silently mutates.** `voxcpm2.py` L642-685 [S8]: `current_seed = materialize_generation_seed(seed)`, then in the retry loop `current_seed += 1` on each bad case, with `self.last_successful_seed` recording what actually worked. `retry_badcase` defaults to **`True`** in `core.py`. So the seed you passed is not necessarily the seed that produced your audio, *on the same machine*. The demo app has to read the value back: `last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)` (`app.py` L345).

**Verdict: Tier 3 is not coherent on VoxCPM2**, and not merely because of version bumps — it breaks within a single machine due to retry-seed drift. If Tier 3 is ever used, it must persist `last_successful_seed`, not the requested seed, and must set `retry_badcase=False`.

### 3.4 VRAM & throughput

From README L378-381 [S2] (a single official table) and HF blob sizes [S10]:

| | VoxCPM2 | VoxCPM1.5 | VoxCPM-0.5B |
|---|---|---|---|
| Params | 2B | 0.6B | 0.5B |
| **VRAM (official)** | **~8 GB** | **~6 GB** | **~5 GB** |
| Weights on disk (bf16) | 4.58 GB | 1.60 GB | — |
| RTF (RTX 4090) | ~0.30 | ~0.15 | ~0.17 |
| RTF w/ Nano-vLLM (RTX 4090) | ~0.13 | ~0.08 | ~0.10 |
| HF repo | `openbmb/VoxCPM2` | `openbmb/VoxCPM1.5` | `openbmb/VoxCPM-0.5B` |
| Tech report | [arXiv 2606.06928](https://arxiv.org/abs/2606.06928) | — | [arXiv 2509.24650](https://arxiv.org/abs/2509.24650) |

- **Does 2B fit 12GB? YES** — ~8 GB claimed, 4.58 GB of weights, comfortable headroom. **HIGH**.
- **Does 0.6B fit 8GB? YES** — VoxCPM1.5 is ~6 GB claimed. **HIGH**.
- **Does 2B fit 8GB? Borderline** — the official figure *is* ~8 GB, leaving no room for the denoiser (`load_denoiser=True` by default pulls in ZipEnhancer). Pass `load_denoiser=False`.
- RTF is quoted only for **RTX 4090** (24GB, much faster than an 8-12GB consumer card). Expect materially worse on the target hardware — see §8.
- The VoxCPM2 technical report (arXiv 2606.06928, submitted 2026-06-05, Zhou et al.) exists and confirms "natural-language voice design" and "style-controllable voice cloning", but its abstract does not describe the conditioning implementation. **HIGH that it exists**; the mechanism claim rests on source, not paper.

## 4. B2 — Qwen3-TTS in depth

**Existence: confirmed.** arXiv **2601.15621**, "Qwen3-TTS Technical Report", Hangrui Hu, Xinfa Zhu, …, Junyang Lin; submitted **2026-01-22** [S11]. The brief's citation is correct.

**Licence: Apache-2.0, code and weights.** `LICENSE` in `QwenLM/Qwen3-TTS` is Apache-2.0 [S12]. HF YAML `license: apache-2.0` on `Qwen3-TTS-12Hz-1.7B-Base`, `-0.6B-Base`, and `-1.7B-VoiceDesign` [S20]. The paper abstract states: "we release both tokenizers and models under the Apache 2.0 license." **HIGH.**

**Languages: 10, no Indic.** Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian [S12]. Brief correct.

**Released checkpoints** [S12]: `Qwen/Qwen3-TTS-Tokenizer-12Hz`, `-12Hz-1.7B-Base`, `-12Hz-1.7B-CustomVoice`, `-12Hz-1.7B-VoiceDesign`, `-12Hz-0.6B-Base`, `-12Hz-0.6B-CustomVoice`.

### 4.1 The speaker encoder IS externally addressable — this is the key finding

`modeling_qwen3_tts.py` L1941-1954 [S14]:

```python
def extract_speaker_embedding(self, audio, sr):
    assert sr == 24000, "Only support 24kHz audio"
    mels = mel_spectrogram(torch.from_numpy(audio).unsqueeze(0), n_fft=1024,
        num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024,
        fmin=0, fmax=12000).transpose(1, 2)
    speaker_embedding = self.speaker_encoder(mels.to(self.device).to(self.dtype))[0]
    return speaker_embedding
```

`qwen3_tts_model.py` L41-52 types the result explicitly [S13]:

```python
@dataclass
class VoiceClonePromptItem:
    ref_spk_embedding: torch.Tensor                  # (D,)
    x_vector_only_mode: bool
```

**The decisive detail** — in `create_voice_clone_prompt` (L435-458) [S13], when `x_vector_only_mode=True` the reference token sequence is dropped entirely:

```python
items.append(VoiceClonePromptItem(
    ref_code=None if xvec_only else code,   # <-- ref audio tokens DISCARDED
    ref_spk_embedding=spk_emb,
    x_vector_only_mode=bool(xvec_only),
    icl_mode=bool(not xvec_only),
    ref_text=rtext,
))
```

Docstring, verbatim: *"x_vector_only_mode=True: Only speaker embedding is used to clone voice; ref_text/ref_code are ignored."*

Injection is a **single token embedding** (`modeling_qwen3_tts.py` L2166-2172) [S14]:

```python
codec_input_emebdding = torch.cat([codec_input_emebdding_0,
                                   speaker_embed.view(1, 1, -1),
                                   codec_input_emebdding_1], dim=1)
```

**And the injection point is public.** `generate_voice_clone(..., voice_clone_prompt=...)` accepts a raw dict [S13], and `_prompt_items_to_voice_clone_prompt` shows its exact shape:

```python
dict(ref_code=[...], ref_spk_embedding=[...], x_vector_only_mode=[...], icl_mode=[...])
```

So a trained mapper can produce a `(D,)` tensor and hand it directly to `generate_voice_clone(text=..., voice_clone_prompt={"ref_code": [None], "ref_spk_embedding": [v], "x_vector_only_mode": [True], "icl_mode": [False]})` — **no audio anywhere in the loop**. This is exactly the addressable slot the two-tower split requires. **HIGH — read from source.**

### 4.2 Dimensions (from shipped `config.json`, not the docstring)

| Checkpoint | `speaker_encoder_config.enc_dim` | talker `hidden_size` | Layers |
|---|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-Base` | **2048** | 2048 | 28 |
| `Qwen3-TTS-12Hz-0.6B-Base` | **1024** | 1024 | 28 |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | **`null` — no speaker encoder** | 2048 | 28 |

Source: `https://huggingface.co/<repo>/raw/main/config.json` [S20]. **HIGH.**

Two notes:
1. **`enc_dim == hidden_size` is not a coincidence** — the speaker embedding *is* a token embedding in the talker's own space. This is a much better mapper target than a bolted-on 192-d SV vector: it lives in the space the backbone already reasons in. The architecture is ECAPA-TDNN (`configuration_qwen3_tts.py` L25) [S15].
2. **The docstring is stale.** `Qwen3TTSSpeakerEncoderConfig`'s docstring says *"enc_dim … defaults to 192"* while the actual `__init__` default is `1024` and the shipped configs are 1024/2048. Trust `config.json`, not the docstring.

### 4.3 Voice design lives in a different checkpoint

`generate_voice_design(text, instruct, language, non_streaming_mode=True, **kwargs)` [S13] takes a free-form natural-language `instruct` string — a genuine first-class API parameter, unlike VoxCPM2's text-prefix hack. But it **hard-raises** unless `self.model.tts_model_type == "voice_design"`, and the VoiceDesign checkpoint has `speaker_encoder_config: null`.

**Therefore, within Qwen3-TTS you cannot go description → vector → render in one model.** The pipeline must be:
- **Tier 1 (preferred):** mapper → `(2048,)` vector → `Base` + `x_vector_only_mode=True`.
- **Tier 2 (minting):** `VoiceDesign` + `instruct` → 10s clip → `extract_speaker_embedding()` on `Base` → store the `(2048,)` vector → thereafter Tier 1.

That second path is genuinely valuable: it gives a **description-to-vector bootstrap** using only Apache-2.0 components, and it produces mapper training data (description, vector) pairs at one GPU render each. See [02-identity-representation.md](02-identity-representation.md).

### 4.4 The "thinking pattern" claim

The brief claims a "thinking pattern for complex descriptions". I found **no `thinking`/`think` parameter** in `generate_voice_design`, `generate_custom_voice`, or `generate_voice_clone` signatures in `qwen_tts/inference/qwen3_tts_model.py`. It may be described in the paper body (not the abstract). **UNVERIFIED — carried to §8.**

### 4.5 VRAM

From HF blob sizes [S10] (no official VRAM table exists in the README):

| Checkpoint | Weights (bf16) | Est. inference VRAM | Fits 12GB? | Fits 8GB? |
|---|---|---|---|---|
| `-1.7B-Base` | 3.86 GB + 0.68 GB tokenizer = **4.54 GB** | ~6-7 GB | Yes | Tight, likely yes |
| `-1.7B-VoiceDesign` | 3.83 + 0.68 = **4.52 GB** | ~6-7 GB | Yes | Tight |
| `-0.6B-Base` | 1.83 + 0.68 = **2.51 GB** | ~4 GB | Yes | Yes |

Weights are measured; the VRAM column is an **estimate** (weights + KV + activations), not a vendor claim — **LOW/MEDIUM confidence**, must be measured (§8). Running Base and VoiceDesign **simultaneously** needs ~9 GB of weights alone and will not fit 12GB comfortably — load them sequentially.

## 5. B6 — licence audit, one subsection per model

### 5.1 CosyVoice 2 / Fun-CosyVoice 3 (FunAudioLLM)

- **Code:** Apache-2.0. `LICENSE` @ `FunAudioLLM/CosyVoice` `main` opens with the standard "Apache License / Version 2.0, January 2004" header [S16]. **HIGH.**
- **Weights:** Apache-2.0. HF YAML `license: apache-2.0` on both `FunAudioLLM/CosyVoice2-0.5B` and `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` [S18]. **HIGH.**
- **Naming correction:** the released "CosyVoice 3" is **`Fun-CosyVoice3-0.5B-2512`** ("Fun-CosyVoice 3.0"), 0.5B, README L5 [S31]. Class `CosyVoice3(CosyVoice2)` at `cosyvoice/cli/cosyvoice.py` L189 [S17].
- **Languages:** 9 (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian) + 18 Chinese dialects. **No Hindi, no Indic** [S18].

**Re-verifying the speaker-vector claim (the basis for abandoning the previous attempt):**

A speaker vector **does** exist. `cosyvoice/cli/frontend.py` L108-118 [S17]:

```python
def _extract_spk_embedding(self, prompt_wav):
    ...
    embedding = self.campplus_session.run(None, {...})[0].flatten().tolist()
    embedding = torch.tensor([embedding]).to(self.device)
    return embedding
```

It is 192-d (`cosyvoice3.yaml` L11: `spk_embed_dim: 192`) [S19], passed as `llm_embedding` and `flow_embedding`, persisted in `spk2info.pt`, and there is even a pure-vector API: `inference_sft(tts_text, spk_id)` → `frontend_sft` builds `{'text', 'text_len', 'llm_embedding', 'flow_embedding'}` with **no prompt tokens at all** (L162-166). `add_zero_shot_spk()` / `save_spkinfo()` persist identities.

**So the brief's stated reason is wrong. But its conclusion survives, for three concrete reasons:**

1. **The flow decoder ignores it by default.** `cosyvoice3.yaml` L164, verbatim: `use_spk_embedding: False # change to True during sft` [S19]. In the shipped zero-shot config the acoustic decoder does not consume the speaker embedding at all.
2. **No speaker table ships with v2/v3.** Blob listings [S10]: `CosyVoice-300M-SFT` has `spk2info.pt` (0.01 MB); **`CosyVoice2-0.5B` and `Fun-CosyVoice3-0.5B-2512` have none.** `inference_sft` therefore has nothing to look up, and `frontend_sft` would `KeyError`.
3. **Zero-shot needs the sequence regardless.** `frontend_zero_shot` (L168-184) always emits `llm_prompt_speech_token`, `flow_prompt_speech_token` and `prompt_speech_feat` alongside the embedding. The 192-d vector is a *hint* riding on top of reference-audio conditioning, not the identity carrier.

Additionally, the embedding is produced by a **frozen external ONNX speaker-verification model (CAMPPlus)**, not a jointly-trained encoder — so it encodes verification-discriminative features, not generation-controlling ones. Mapping a description into CAMPPlus space would not reliably control the output even if the flow used it.

**Verdict: the decision to abandon CosyVoice was correct; record the corrected reason.** **HIGH.**

### 5.2 Chatterbox (Resemble AI)

- **Code:** MIT. `LICENSE` @ `resemble-ai/chatterbox` `master`, verbatim: "MIT License / Copyright (c) 2025 Resemble AI" [S33]. **HIGH.**
- **Weights:** MIT. HF YAML `license: mit` on `ResembleAI/chatterbox` [S30]. **HIGH.**
- **Indic: Yes — Hindi, with a verified MIT checkpoint.** Chatterbox Multilingual covers 23 languages including Hindi [S32]. The dedicated single-language pack **`ResembleAI/Chatterbox-Multilingual-hi` is HTTP 200, `cardData.license: mit`, ungated** [S46] — **HIGH**. Note the umbrella repo `ResembleAI/Chatterbox-Multilingual-TTS` returns **HTTP 401** (gated or renamed), so pin the `-hi` pack, not the umbrella.
- **Conditioning — genuinely interesting, "Tier 1.5".** `src/chatterbox/tts.py` defines a serialisable `Conditionals` dataclass with `save(fpath)` / `load(fpath)` (L64-102) [S33]. Contents:
  - `T3Cond.speaker_emb` — a **256-d fixed vector** (`voice_encoder/config.py`: `speaker_embed_size = 256`), projected into the backbone by `nn.Linear(hp.speaker_embed_size, hp.n_channels)` in `T3CondEnc` — i.e. a real addressable slot for the text→token stage.
  - `T3Cond.cond_prompt_speech_tokens` — **optional** (`Optional[Tensor] = None`), so T3 can run on the vector alone.
  - `s3gen_ref_dict` — **not optional**. `s3gen.py::embed_ref` (L118-158) builds `ref_mels_24` (variable-length mel), `ref_speech_tokens` (sequence) *and* `ref_x_vector`. The token→waveform stage requires the mel sequence.
- **Verdict:** not a clean Tier 1 — the vocoder stage still needs a reference mel sequence, which does not interpolate meaningfully. But `Conditionals.save()` gives a **persistable, audio-free identity blob**, which is strictly better than Tier 2's "keep a wav around". Call it **Tier 1.5**. Also carries `exaggeration` (emotion) as claimed. **HIGH** on mechanism.

### 5.3 XTTS-v2 (Coqui)

- **Code: MPL-2.0, NOT CPML.** `https://raw.githubusercontent.com/coqui-ai/TTS/dev/LICENSE.txt` is the Mozilla Public License Version 2.0 [S21]. The maintained fork `idiap/coqui-ai-TTS` is likewise MPL-2.0 [S22]. **The brief conflates code and weights here.** **HIGH.**
- **Weights: Coqui Public Model License 1.0.0 — non-commercial.** HF YAML: `license: other`, `license_name: coqui-public-model-license`, `license_link: https://coqui.ai/cpml` [S30]. That link now **404s** (Coqui is gone); the text survives at `https://huggingface.co/coqui/XTTS-v2/raw/main/LICENSE.txt` [S22]. Operative clauses, verbatim:
  > "This license allows only non-commercial use of a machine learning model and its outputs."
  > "Use of the model to train other models for commercial use is not a non-commercial purpose."
  > "Use for revenue-generating activity, including projects directly funded by government grants, is not a non-commercial purpose."

  The middle clause is fatal twice over: it bars public hosting **and** bars training our mapper against XTTS even if we never served XTTS itself. **HIGH.**
- **Maintainer after shutdown:** **Idiap Research Institute**, `idiap/coqui-ai-TTS`, PyPI `coqui-tts` [S22]. **A fork cannot relicense the checkpoint** — CPML is Coqui's grant on the weights, and Idiap has no authority to widen it. The maintenance situation therefore **does not** change the licence calculus. **HIGH.**
- **Conditioning — brief is half wrong.** From `TTS/tts/models/xtts.py` [S23]:
  ```python
  def inference(self, text, language, gpt_cond_latent, speaker_embedding, ...)
  def get_conditioning_latents(self, audio_path, max_ref_length=30, gpt_cond_len=6,
                               gpt_cond_chunk_len=6, librosa_trim_db=None,
                               sound_norm_refs=False, load_sr=22050)
  ```
  - `speaker_embedding` — fixed vector, from `self.hifigan_decoder.speaker_encoder.forward(audio_16k, l2_norm=True).unsqueeze(-1)`. **Vector-addressable.**
  - `gpt_cond_latent` — **variable-length sequence `[1, 1024, T]`**, from `cond_latent = self.gpt.get_style_emb(mel); return cond_latent.transpose(1, 2)`. **Not** a vector.

  Both are required. So XTTS is *partially* addressable, not the clean two-tensor vector store the brief implies. **HIGH.**
- **Hindi:** XTTS-v2 covers 17 languages including Hindi [S22]. **MEDIUM.**
- **Verdict: DISQUALIFIED on weights licence.** Public hosting is commercial use; CPML forbids it explicitly.

### 5.4 IndexTTS-2 — the brief is wrong here

- **Licence: "bilibili Model Use License Agreement"**, `LICENSE` @ `index-tts/index-tts` `main` (10,554 bytes; Chinese version at `LICENSE_ZH.txt`) [S35]. **HIGH.**
- **Scope covers code AND weights.** §1.4, verbatim: *"'Model': means the artificial-intelligence model named 'bilibili indextts2', including but not limited to **model weights and final code**…"* There is no separate permissive code licence in the repo — only `LICENSE` and `LICENSE_ZH.txt`. **HIGH.**
- **It is NOT non-commercial.** §2.2, verbatim:
  > "If You intend to Use … and either (i) your or any of your Affiliates' products or services had more than 100 million monthly active users in the immediately preceding calendar month, or (ii) your or any of your Affiliates' annual revenue in the immediately preceding calendar year exceeded RMB 1 billion, You must request a separated license from us…"

  This is a Llama-2-style scale threshold. Below 100M MAU and RMB 1B revenue, commercial use — including public hosting — **is granted**. The brief's "restrictive/non-commercial" is **wrong**. **HIGH.**
- **The clause that actually threatens us.** §2(c), verbatim:
  > "You may not Use the bilibili indextts2 or any Derivative Work to improve any AI model, except for the bilibili indextts2 itself, its Derivative Works, or non-commercial AI models."

  We train a **mapper** whose supervision could come from IndexTTS-2 outputs or embeddings. Whether that counts as "using the Model to improve an AI model" is a genuine legal question, and §1.5 defines "Derivative Work" broadly enough to include "any work based on … model outputs". **This is a licence-propagation issue for [08-licensing-propagation.md](08-licensing-propagation.md), not a settled fact.** Flagging as a hazard, not a blocker. **HIGH** on the text, **UNVERIFIED** on interpretation.
- **HF card:** `IndexTeam/IndexTTS-2` YAML has **no `license:` field** — only `language:` and `pipeline_tag:` [S30]. The repo LICENSE governs.
- **Conditioning / speaker vector / Indic: UNVERIFIED** — not reached this pass. Carried to §8.

### 5.5 VibeVoice (Microsoft)

- **Code licence: MIT.** `LICENSE` verbatim: "MIT License / Copyright (c) 2025 Microsoft" [S24]. **HIGH.**
- **Microsoft DID withdraw the TTS model.** README news entry, verbatim [S25]:
  > "2025-09-05: VibeVoice is an open-source research framework intended to advance collaboration in the speech synthesis community. After release, we discovered instances where the tool was used in ways inconsistent with the stated intent. Since responsible use of AI is one of Microsoft's guiding principles, we have removed the VibeVoice-TTS code from this repository."

  HF API status [S26]: `microsoft/VibeVoice-1.5B` → **200**; `microsoft/VibeVoice-Large` → **401**; `microsoft/VibeVoice-7B` → **401**. So the large checkpoints are gated/removed and the TTS code is gone from the repo. **HIGH.**
- **The successor is real but useless to us.** `VibeVoice-Realtime-0.5B` (announced 2025-12-03) is MIT (`license: mit` in HF YAML), English-only, single-speaker, built on Qwen2.5-0.5B, ~200 ms first-packet [S26]. Its docs say, verbatim:
  > "To mitigate deepfake risks and ensure low latency for the first speech chunk, **voice prompts are provided in an embedded format**. For users requiring voice customization, please reach out to our team."

  That is a deliberate closed set of pre-embedded voices — no arbitrary cloning, no vector you can mint. Also, verbatim: *"We do not recommend using VibeVoice in commercial or real-world applications without further testing and development. This model is intended for research and development purposes only."* That is a **recommendation in documentation, not a licence term** — MIT remains MIT legally — but it signals the maintainers' posture and the withdrawal precedent. **HIGH.**
- **Verdict:** unusable. The one thing it has that we want (a fixed set of persisted voice embeddings) is exactly the thing they refuse to let you extend.

### 5.6 F5-TTS

- **Code: MIT.** `LICENSE` verbatim: "MIT License / Copyright (c) 2024 Yushen CHEN" [S34]. **HIGH.**
- **Weights: CC-BY-NC-4.0 — the brief is correct.** `SWivid/F5-TTS` HF card YAML: `license: cc-by-nc-4.0` [S46]. **CODE AND WEIGHTS DIFFER** (MIT code, non-commercial weights) — the same Emilia-dataset constraint that forces OmniVoice's weights to CC-BY-NC (§6.1). **HIGH.**
- **Speaker vector: NO** — pure in-context reference-wav + transcript conditioning. Rejected on mechanism as well as licence.

### 5.7 Kokoro — the second Tier-1 backend

- **Code: Apache-2.0** (`hexgrad/kokoro` LICENSE). **Weights: Apache-2.0** (`hexgrad/Kokoro-82M` HF card YAML). **They match** — one of only four models in this survey where both halves are permissive and ungated. The card goes out of its way to say so, verbatim: *"This is an Apache-licensed model, and Kokoro has been deployed in numerous projects and commercial APIs. We welcome the deployment of the model in real use cases."* **HIGH** [S38].
- **Conditioning: pure style-vector lookup, with no reference audio anywhere in the inference path.** 54 per-voice `.pt` files under `voices/`.
- **The vector is real — verified by download, not by inference.** `voices/af_heart.pt` was fetched and loaded: `torch.Tensor`, shape **`(510, 1, 256)`**, `float32`. Used at `pipeline.py` L242 as `model(ps, pack[len(ps)-1], speed)` → a **`[1, 256]`** vector per call. Split inside `model.py`: `s = ref_s[:, 128:]` (prosody predictor, L104) and `self.decoder(..., ref_s[:, :128])` (timbre, L118). `style_dim: 128` in `config.json`. **HIGH** [S38].
- **Injection and interpolation are first-class upstream features.** `load_voice()` is typed `Union[str, torch.FloatTensor]` — **you can hand it a raw tensor**, which is precisely the mapper output slot. Multi-voice blending already exists: `torch.mean(torch.stack(packs), dim=0)` (`pipeline.py` L167-176). Nothing needs to be reverse-engineered.
- **Indic: YES.** `LANG_CODES` at `pipeline.py` L31 includes `h='hi'`, with four Hindi voices: `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`. **HIGH.**
- **No description-based voice design.** Kokoro is a renderer only; the description→vector step must come from our mapper (or from a bootstrap using a different model).
- **82M params.** VRAM is **UNVERIFIED** (no primary source states a figure), but at 82M it is not a constraint on any card under discussion.
- **Verdict: a serious Tier-1 candidate, and materially cheaper than Qwen3-TTS.** The one structural caveat: the pack is `(510, 1, 256)` **indexed by phoneme count**, not a single 256-d vector — identity is a length-conditioned table. A mapper must either emit the whole pack or emit one 256-d vector and broadcast across all 510 slots, which is untested. See §8.

### 5.8 StyleTTS2 — good vector, no licence

- **Code: MIT.** `LICENSE` verbatim: "MIT License / Copyright (c) 2023 Aaron (Yinghao) Li" [S36]. **HIGH.**
- **Weights: NO LICENCE DECLARED — this is worse than a bad licence.** Both `yl4579/StyleTTS2-LJSpeech` and `yl4579/StyleTTS2-LibriTTS` return **`cardData: None`** from the HF API — no README, no YAML `license:` field. The only stated term is a bespoke README clause: *"Pre-Trained Models: Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized…"* That is a disclosure obligation, not a grant of rights. **With no licence, the default is "all rights reserved".** **HIGH** [S42].
- **Speaker vector: YES, 256-d.** `compute_style(path)` returns `torch.cat([ref_s, ref_p], dim=1)` from `model.style_encoder` (128-d, timbre/decoder) + `model.predictor_encoder` (128-d, prosody); `style_dim: 128` in `Configs/config_libritts.yml` L44, corroborated by `torch.randn((1, 256))` in the sampler. Interpolation is demonstrated upstream: `ref = alpha * ref + (1 - alpha) * ref_s[:, :128]`. **HIGH.**
- Also supports **style diffusion with no reference audio** — but the style is sampled from the *sentence's* BERT embedding, not from a voice description. **Not** description-based voice design.
- **Indic: NO** for released weights (English LJSpeech/LibriTTS; PL-BERT is English-only).
- **GPL contamination risk at inference**: the README notes the inference path depends on a GPL-licensed package (phonemizer/espeak-ng). Relevant to [08-licensing-propagation.md](08-licensing-propagation.md).
- **Verdict: rejected on licence.** Architecturally it is Kokoro's base model, and Kokoro has the same vector idea with an actual Apache grant. Use Kokoro.

### 5.9 YourTTS — technically ideal, legally dead

- **Code: MPL-2.0** (inside `coqui-ai/TTS`, archived) [S21].
- **Weights: CC BY-NC-ND 4.0**, from Coqui's own manifest `TTS/.models.json`: `tts_models/multilingual/multi-dataset/your_tts` → `"license": "CC BY-NC-ND 4.0"` [S41]. **HIGH.**
- Non-commercial kills public hosting; **NoDerivatives additionally forbids fine-tuning or redistributing modifications**, which is stricter than CPML.
- **Speaker vector: YES — a textbook 512-d d-vector.** `_set_cond_input`: `g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)`; `d_vector_dim=512`, `use_d_vector_file=True` (`recipes/vctk/yourtts/train_yourtts.py` L129-130). An alternative lookup mode exists: `self.emb_g = nn.Embedding(num_speakers, 256)` (`vits.py` L779).
- **Languages:** en, fr-fr, pt-br only. No Indic.
- **Verdict: exactly the architecture we want, and completely unusable.** Worth recording as the clearest illustration that mechanism and licence must be checked independently.

### 5.10 Orpheus — no vector, and a licence-stacking question

- **Code: Apache-2.0.** **Weights: HF YAML `apache-2.0`** — but the repo is `gated: auto` (login + terms) and `base_model: meta-llama/Llama-3.2-3B-Instruct`. The Apache tag is the publisher's claim; whether the Llama 3.2 Community Licence propagates through a fine-tune is a live legal question. **MEDIUM** on the weights position [S43].
- **Conditioning: the voice name is prepended to the text.** `engine_class.py` L80: `adapted_prompt = f"{voice}: {prompt}"`, with a fixed vocabulary at L16: `["zoe","zac","jess","leo","mia","julia","leah"]` (README adds `tara`, `dan`).
- **Speaker vector: NO.** Identity is literally tokenised text. Nothing to extract, store or interpolate. **HIGH.**
- **Indic: yes but gated** — `canopylabs/3b-hi-ft-research_release`, `language: ["hi"]`, apache-2.0, research release.
- 3B params. **Verdict: rejected on mechanism** regardless of the licence question.

### 5.11 Higgs Audio v2 / Higgs TTS 2 — fails VRAM and licence

- **Code: Apache-2.0.** **Weights: "BOSON HIGGS AUDIO 2 COMMUNITY LICENSE AGREEMENT"** (release date 2025-06-20), a LICENSE file inside the HF repo; HF YAML says only `license: other`. It is explicitly *based on and incorporates the Meta Llama 3 Community License*. **CODE AND WEIGHTS DIFFER.** **HIGH** [S40].
- Not non-commercial, but §2 imposes a ceiling: *">100,000 annual active users… you must request an expanded license from Boson AI, which Boson AI may grant… in its sole discretion."* A discretionary future permission is not a frozen grant — poor fit for a component we want to never think about again.
- Repo **renamed**: `bosonai/higgs-audio-v2-generation-3B-base` → `bosonai/higgs-tts-2-3b-base`. A v2.5 also exists.
- **Conditioning:** ChatML multimodal. Three routes: free-form **scene/voice description** in `<|scene_desc_start|>…<|scene_desc_end|>`, `[SPEAKER0]` tags, or reference audio → codec tokens (variable length).
- **Voice design from description: YES** — `examples/voice_prompts/profile.yaml` holds free-form descriptions, e.g. `male_en_british: "He speaks with a clear British accent and a conversational, inquisitive tone…"`. This makes it a genuine voice-design contender on mechanism.
- **Speaker vector: NO** — reference audio becomes variable-length codec tokens; everything else lives in text. **HIGH.**
- **VRAM: ≥24 GB stated** (`README_V2.md` L121: *"a machine equipped with GPU with at least 24GB memory"*). **This alone disqualifies it on our locked 8-12GB constraint.**
- Languages: en, zh, de, ko. No Indic.

### 5.12 Fish-Speech / OpenAudio — the licence changed; disqualified twice

- **Code: "FISH AUDIO RESEARCH LICENSE AGREEMENT", Last Updated 2026-03-07.** GitHub reports `NOASSERTION`. This is **not** CC-BY-NC-SA and **no longer any permissive code licence** — a change from what most notes record. §V makes it cover code *and* weights: *"'Fish Audio Materials' means, collectively, Fish Audio's proprietary Models, Software and Documentation."* **HIGH** [S39].
- **The clause that ends it**, §III defines Commercial Purpose as including: *"(i) creating, modifying, or distributing Your product or service, including via a hosted service or application programming interface."* Public hosting is named explicitly — exactly our constraint.
- **Weights: `cc-by-nc-sa-4.0`** (`fishaudio/fish-speech-1.5` HF YAML). `fishaudio/openaudio-s1-mini` now redirects to `fishaudio/s1-mini`, also CC-BY-NC-SA and **access-restricted**.
- **Conditioning:** `ServeTTSRequest.references: list[ServeReferenceAudio]` = `audio: bytes` + `text: str` (`fish_speech/utils/schema.py` L60-89). **Speaker vector: NO.** `reference_id` is a fish.audio *cloud* voice ID, not a local tensor.
- **Verdict: DISQUALIFIED on both code and weights.** Note this is the model VoxCPM2 benchmarks against as "Fish S2-Pro" [S31] — it is competitive, and unusable.

### 5.13 Dia — clean licence, no identity

- **Code: Apache-2.0. Weights: Apache-2.0** (`nari-labs/Dia-1.6B`, `Dia-1.6B-0626`), ungated. **They match.** The brief's Apache claim is confirmed from primary sources. **HIGH** [S44].
- **Conditioning: essentially none.** `generate()` takes `text` + optional `audio_prompt` (DAC frames). `[S1]`/`[S2]` are byte-substituted dialogue tags (`model.py` L257), not identities.
- **Speaker vector: NO**, and README L165 is decisive: *"The model was not fine-tuned on a specific voice. Hence, you will get different voices every time you run the model. You can keep speaker consistency by either adding an audio prompt, or fixing the seed."* Voice is **random per run** — this is Tier 3 as an architecture, which is precisely what we rejected.
- **VRAM (stated, primary): `float16` ~4.4 GB**, bf16 ~4.4 GB, fp32 ~7.9 GB. 1.6B params. Fits 12GB.
- English only. **Verdict: legally clean, operationally useless for a persistent identity.**

### 5.14 Sesame CSM — the `speaker: int` is a decoy

- **Code: Apache-2.0. Weights: Apache-2.0** (`sesame/csm-1b`). **They match.** Runtime additionally requires gated `meta-llama/Llama-3.2-1B` access. **HIGH** [S45].
- **Conditioning:** `generate(self, text: str, speaker: int, context: List[Segment], max_audio_length_ms, temperature, topk)` (`generator.py` L109), with `@dataclass class Segment: speaker: int; text: str; audio: torch.Tensor` (L15).
- **Speaker vector: NO — and the reason matters.** The `speaker: int` looks like an embedding-table index but is not. `generator.py` L64: `text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")` — **the integer is string-formatted into the text prompt** and tokenised by the Llama-3 tokenizer. There is no `nn.Embedding` speaker table. All real identity comes from `context` audio → variable-length RVQ tokens. **HIGH.**
- Indic: NO (README L141: *"some capacity for non-English languages due to data contamination… but it likely won't do well"*).
- **Verdict: rejected on mechanism.** A useful cautionary case — an integer speaker argument is not evidence of an addressable vector.

## 6. Models the brief missed

### 6.1 OmniVoice (k2-fsa) — the near-miss

- **Repo:** `https://github.com/k2-fsa/OmniVoice` (HTTP 200); HF `k2-fsa/OmniVoice`; paper *"OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models"*. Built on `Qwen/Qwen3-0.6B`. **HIGH** [S27].
- **Why it matters:** README, verbatim: *"supporting over 600 languages … supporting voice cloning and voice design"* and *"**Voice Design**: Control voices via assigned speaker attributes (gender, age, pitch, dialect/accent, whisper, etc.)."* Unlike VoxCPM2, `instruct=` is a **first-class API parameter**:
  ```python
  audio = model.generate(text="...", instruct="female, young adult, high pitch, british accent")
  ```
- **Indic coverage is the best found anywhere** (`docs/languages.md`, with training hours) [S27]: Bengali (271.76 h), Gujarati (91.18), **Hindi (117.17)**, Kannada (128.06), Malayalam (166.57), Marathi (156.71), Tamil (423.09), Telugu (230.21), Urdu (211.27). Nine Indic languages. Directly relevant to [04-indic-track.md](04-indic-track.md).
- **DISQUALIFIED — and this is exactly the code/weights split the evidence standard warns about.** HF model card §License, verbatim [S28]:
  > "Our code is released under the Apache 2.0 License. **The pre-trained model is licensed under the CC-BY-NC** due to constraints from its training data (e.g., Emilia)."

  A secondary blog described OmniVoice as "Apache-2.0" [S37]. It is not, for the part that matters. **Public hosting is commercial use → excluded.** **HIGH.**
- **Caveat even if it were licensed:** the README warns *"Voice design is trained on Chinese and English data only. It can generalize to other languages, but may produce unstable results."* And `instruct` is a **closed attribute vocabulary** (one value per category: gender/age/pitch/style/accent/dialect), not free-form natural language — a narrower target than the brief's "natural-language character description".

### 6.2 Parler-TTS (HuggingFace) — the only fully-Apache native description model

- **Code: Apache-2.0** (`LICENSE` @ `huggingface/parler-tts` `main`) [S29]. **Weights: Apache-2.0** (`parler-tts/parler-tts-large-v1` HF YAML `license: apache-2.0`) [S29]. Both permissive. **HIGH.**
- **Architecture is the canonical description-conditioned design:** a frozen FLAN-T5 text encoder turns the description into hidden states which condition an audio-token decoder by **cross-attention** (reproduction of Lyth & King, *"Natural language guidance of high-fidelity text-to-speech with synthetic annotations"*). Gender, pitch, speaking rate and reverberation are controlled purely through the description string. **MEDIUM** — architecture from the model card/paper, not re-derived from source this pass.
- **No speaker vector** — the conditioning is a variable-length encoder-output sequence, so it is not a Tier-1 slot as-is. But it is the only model in the survey where description-conditioning is the **native trained interface** rather than a bolt-on, and it is fully Apache for code *and* weights. Worth considering for [02-identity-representation.md](02-identity-representation.md): a mapper could target a fixed-length prefix of the description-encoder output space.
- English-only; ~0.88B (large-v1); no Indic.

## 7. What this means for the build

**Two different models for two different roles — they are not the same model.**

| Role | Model | Why |
|---|---|---|
| **Tier-1 vector — quality track** | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | A `(2048,)` vector is the *sole* identity input (`x_vector_only_mode=True`, `ref_code=None`); injection point is a public API; Apache-2.0 code and weights; and it has a description-conditioned sibling checkpoint for bootstrapping. |
| **Tier-1 vector — cheap track** | `hexgrad/Kokoro-82M` | 82M params, Apache-2.0 code and weights, `load_voice()` accepts a raw tensor, blending is upstream, **and it has Hindi**. Runs anywhere. Caveat: the pack is length-indexed `(510,1,256)`, not one vector. |
| **Description-driven voice design** | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Free-form `instruct=` is a first-class parameter. Apache-2.0. Runner-ups: **Parler-TTS** (native description architecture, fully Apache) and **Higgs Audio v2** (free-form scene descriptions — but ≥24 GB VRAM and a 100k-AAU licence ceiling, so excluded). |
| **Hindi / Indic** | `openbmb/VoxCPM2` (Tier 2) or `hexgrad/Kokoro-82M` (Tier 1) | VoxCPM2: Apache-2.0 both halves, best-in-class Hindi CER 0.79% — but Tier 2 only. Kokoro: Hindi *and* a real vector, at 82M. OmniVoice has far better Indic breadth but CC-BY-NC weights. |

**Two Tier-1 backends now exist, and they trade off cleanly.** Qwen3-TTS gives a single clean `(2048,)` vector in the backbone's own embedding space plus an Apache description-conditioned sibling for bootstrapping — the better research target. Kokoro gives an Apache-licensed vector on an 82M model that fits anywhere and already ships Hindi — the better engineering target. **Prototype the mapper against Kokoro first** (iteration is minutes, not hours), then port to Qwen3-TTS for quality. The adapter interface must therefore abstract *vector dimension* (256 vs 1024 vs 2048) and *whether identity is length-indexed*.

**The recommended architecture** (all Apache-2.0):

1. **Mint:** `VoiceDesign` + `instruct="<character description>"` → 10 s clip. One GPU render.
2. **Distil to a vector:** load `Base`, call `extract_speaker_embedding(clip_24k, 24000)` → `(2048,)` tensor. CPU-cheap once loaded.
3. **Persist the vector.** This is the voice identity. Interpolatable, tiny, no audio retained.
4. **Render:** `Base.generate_voice_clone(text=..., voice_clone_prompt={"ref_code": [None], "ref_spk_embedding": [v], "x_vector_only_mode": [True], "icl_mode": [False]})`.
5. **Train the mapper** on (description, vector) pairs harvested from step 1-2. This bootstrap is the whole reason to keep VoiceDesign around.

Steps 1-2 give Tier 2 on day one and generate the training set for Tier 1 — the same pipeline serves both, which is the main practical argument for Qwen3-TTS over everything else.

**Repo IDs and revisions to pin:**

| Purpose | HF repo ID | Revision |
|---|---|---|
| Tier-1 renderer (quality) | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | **PIN — commit SHA not captured; must pin before any mapper training** |
| Tier-1 renderer (8GB) | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | PIN (note: `enc_dim` 1024, **not** interchangeable with 1.7B's 2048) |
| **Tier-1 renderer (prototype)** | `hexgrad/Kokoro-82M` | PIN — 256-d style space; pin the `voices/*.pt` pack revision too, it is the identity format |
| Voice-design minter | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | PIN |
| Speech tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | PIN |
| Hindi / Tier-2 | `openbmb/VoxCPM2` | PIN |
| Hindi on 8GB | `openbmb/VoxCPM1.5` | PIN |
| Indic alt (Tier 1.5, MIT) | `ResembleAI/Chatterbox-Multilingual-hi` | PIN |
| Description-conditioned alt | `parler-tts/parler-tts-large-v1` | PIN |

> **Pinning is not optional.** A `(2048,)` vector is only meaningful against the exact speaker-encoder weights that produced it. Any checkpoint update silently invalidates every stored identity. Record commit SHAs, not `main`. This is the single largest operational risk in the Tier-1 design and belongs in [07-serving-and-cost.md](07-serving-and-cost.md).

**Install constraints for a 12GB card:**

- Do **not** hold `Base` and `VoiceDesign` resident together (~9 GB of weights before activations). Load sequentially; minting is a batch job, rendering is the hot path.
- VoxCPM2: pass `load_denoiser=False` to `from_pretrained` — the default `True` pulls in ZipEnhancer (`iic/speech_zipenhancer_ans_multiloss_16k_base`) on top of the ~8 GB budget.
- VoxCPM2 quotes `attn_implementation="flash_attention_2"` in the Qwen examples; verify the wheel exists for the local CUDA before depending on it.
- If Tier-1 must run on 8GB, use `Qwen3-TTS-12Hz-0.6B-Base` and accept a 1024-d vector space — and note the two dimensions are **not** transferable, so the choice of card fixes the identity format.

**Licence posture for public hosting.** Clean on **both** code and weights: Qwen3-TTS (all variants), VoxCPM2/1.5/0.5B, CosyVoice2 / Fun-CosyVoice3, **Kokoro**, Chatterbox, Parler-TTS, Dia, Sesame CSM. **Excluded:** XTTS-v2 (CPML non-commercial), YourTTS (CC BY-NC-ND), Fish-Speech/OpenAudio (research licence naming hosted APIs + CC-BY-NC-SA weights), OmniVoice (CC-BY-NC weights), StyleTTS2 (**no weights grant at all**), VibeVoice-TTS (withdrawn). **Conditional / discretionary:** IndexTTS-2 (servable under the 100M-MAU / RMB 1B threshold, but §2(c) is a live hazard for the mapper), Higgs Audio v2 (100k-AAU cliff — and fails the 24 GB VRAM floor anyway), Orpheus (Llama-base propagation unresolved). Detail belongs in [08-licensing-propagation.md](08-licensing-propagation.md).

**The general lesson, stated once:** of the 20 models audited, **six have a code licence that differs materially from their weights licence**, and in four of those the code is permissive while the weights are not (OmniVoice, YourTTS, XTTS-v2, Higgs Audio). A seventh (StyleTTS2) has permissive code and *no weights licence at all*. Never quote a single licence for a TTS model.

## 8. Open — must be settled by experiment

| Question | Cheapest experiment | Est. cost/time | What it blocks |
|---|---|---|---|
| Does `x_vector_only_mode=True` with a **hand-constructed** `ref_spk_embedding` (not from audio) actually render intelligible, identity-stable speech? | Extract a vector from a clip, perturb it, re-render via the `voice_clone_prompt` dict path. | 1 GPU-hour | **Everything.** This is the load-bearing assumption of the whole Tier-1 design. |
| Is the Qwen speaker-embedding space smooth/interpolatable? | Extract vectors from 20 clips, render at 10 interpolation midpoints, listen + measure speaker similarity. | 2 GPU-hours | Whether the mapper can be trained with an L2/cosine objective at all. |
| What is real VRAM and RTF for `1.7B-Base` on the actual 8-12GB card? | Load fp16, render 30 s, log `torch.cuda.max_memory_allocated()`. | 30 min | Card sizing; 1.7B-vs-0.6B choice, which **fixes the identity dimension**. |
| **Kokoro: can one 256-d vector be broadcast across all 510 length-slots and still render correctly?** | Take `af_heart.pt`, replace all 510 slots with `pack[100]`, render sentences of varying length, listen. | 30 min, trivial GPU | Whether the mapper emits `(256,)` or the full `(510,1,256)` pack — i.e. the entire identity format for the cheap track. **Highest-value open item.** |
| Is Kokoro's 256-d style space smooth under interpolation across *different* voices (not just the shipped blend)? | Blend `af_heart` × `hf_alpha` at 10 alphas; listen for artefacts. | 30 min | Whether Kokoro supports interpolatable identities or only nearest-voice lookup. |
| Does the Llama base licence propagate to Orpheus / Higgs weights? | Legal read, not an experiment. | — | Only matters if we revisit either; both currently rejected on mechanism/VRAM. |
| IndexTTS-2: conditioning mechanism, speaker vector, Indic support. | Read `index-tts` inference source. | 1 hour, no GPU | Whether the §2(c) hazard is even worth taking. |
| Does Qwen3-TTS have a "thinking pattern" for complex descriptions? | Read arXiv 2601.15621 body (not abstract). | 30 min | Brief claim 4.4; affects description-handling design. |
| Is VoxCPM2 `seed` reproducible run-to-run on one machine with `retry_badcase=False`? | Render the same text+seed 10x, hash the waveform. | 20 min | Whether Tier 3 has *any* residual use as a debugging aid. |
| Does IndexTTS-2's §2(c) forbid training our mapper on its outputs? | Legal read, not an experiment. | — | Whether IndexTTS-2 is usable at all. |
| Chatterbox: can `s3gen_ref_dict` be synthesised from a vector, or must it come from audio? | Read `s3gen.py` flow-decoder inputs; attempt a vector-only render. | 2 GPU-hours | Whether Chatterbox is Tier 1 or only Tier 1.5. |

## 9. Sources

| # | URL | Type | Used for | Confidence in source |
|---|---|---|---|---|
| S1 | `https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/LICENSE` | LICENSE file | VoxCPM code licence = Apache-2.0 | HIGH |
| S2 | `https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/README.md` | Official README | 30-language list, Hindi benchmark, VRAM/RTF table, variants | HIGH |
| S3 | `https://huggingface.co/openbmb/VoxCPM2` (`/raw/main/README.md`) | HF model card YAML | VoxCPM2 weights licence = apache-2.0 | HIGH |
| S4 | `https://huggingface.co/openbmb/VoxCPM1.5`, `.../VoxCPM-0.5B` | HF model card YAML | Variant weights licences | HIGH |
| S5 | `https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/src/voxcpm/model/voxcpm2.py` | Source code | `_encode_wav`, `_make_ref_prefix`, no speaker vector, `build_prompt_cache` | HIGH |
| S6 | `https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/src/voxcpm/core.py` | Source code | Real `_generate` signature; `retry_badcase=True` default | HIGH |
| S7 | `https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/app.py` | Source code | `final_text = f"({control}){text}"` — voice design is a text prefix | HIGH |
| S8 | `voxcpm2.py` L642-685 | Source code | `current_seed += 1` retry drift; `last_successful_seed` | HIGH |
| S9 | `https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/src/voxcpm/model/utils.py` | Source code | `materialize_generation_seed`, `apply_generation_seed` | HIGH |
| S10 | `https://huggingface.co/api/models/<repo>?blobs=true` | HF API | Weight sizes; presence/absence of `spk2info.pt` | HIGH |
| S11 | `https://arxiv.org/abs/2601.15621` | Paper | Qwen3-TTS report exists; authors; date; Apache-2.0 statement | HIGH |
| S12 | `https://github.com/QwenLM/Qwen3-TTS` + `/main/LICENSE` | README + LICENSE | Checkpoint list, languages, code licence Apache-2.0 | HIGH |
| S13 | `https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/qwen_tts/inference/qwen3_tts_model.py` | Source code | `VoiceClonePromptItem`, `x_vector_only_mode`, `create_voice_clone_prompt`, `generate_voice_clone`, `generate_voice_design` | HIGH |
| S14 | `https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/qwen_tts/core/models/modeling_qwen3_tts.py` | Source code | `extract_speaker_embedding`, `speaker_embed.view(1,1,-1)` injection | HIGH |
| S15 | `https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/qwen_tts/core/models/configuration_qwen3_tts.py` | Source code | ECAPA-TDNN, `enc_dim`, stale docstring | HIGH |
| S16 | `https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/LICENSE` | LICENSE file | CosyVoice code licence = Apache-2.0 | HIGH |
| S17 | `https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/cli/frontend.py`, `.../cli/cosyvoice.py` | Source code | `_extract_spk_embedding`, `frontend_sft`, `frontend_zero_shot`, `class CosyVoice3` | HIGH |
| S18 | `https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, `.../CosyVoice2-0.5B` | HF model cards | Weights licence apache-2.0; languages; 0.5B | HIGH |
| S19 | `https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/raw/main/cosyvoice3.yaml` | Model config | `spk_embed_dim: 192`; `use_spk_embedding: False` | HIGH |
| S20 | `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-{1.7B,0.6B}-{Base,VoiceDesign}/raw/main/{config.json,README.md}` | Model config + card | `enc_dim` 2048/1024; VoiceDesign `speaker_encoder_config: null`; weights apache-2.0 | HIGH |
| S21 | `https://raw.githubusercontent.com/coqui-ai/TTS/dev/LICENSE.txt` | LICENSE file | Coqui code = MPL-2.0 (not CPML) | HIGH |
| S22 | `https://huggingface.co/coqui/XTTS-v2/raw/main/LICENSE.txt` + `https://github.com/idiap/coqui-ai-TTS` | Licence text + repo | CPML 1.0.0 non-commercial clauses; Idiap maintainership | HIGH |
| S23 | `https://raw.githubusercontent.com/idiap/coqui-ai-TTS/main/TTS/tts/models/xtts.py` | Source code | `get_conditioning_latents`/`inference` signatures; `gpt_cond_latent [1,1024,T]` is a sequence | HIGH |
| S24 | `https://raw.githubusercontent.com/microsoft/VibeVoice/main/LICENSE` | LICENSE file | VibeVoice code = MIT | HIGH |
| S25 | `https://raw.githubusercontent.com/microsoft/VibeVoice/main/README.md` | Official README | 2025-09-05 withdrawal statement (verbatim) | HIGH |
| S26 | `https://huggingface.co/api/models/microsoft/VibeVoice-*` + `docs/vibevoice-realtime-0.5b.md` | HF API + docs | Large/7B → 401; Realtime-0.5B MIT; embedded-voice policy | HIGH |
| S27 | `https://github.com/k2-fsa/OmniVoice` (`/main/LICENSE`, `/main/README.md`, `/main/docs/languages.md`, `/main/docs/voice-design.md`) | LICENSE + README + docs | Apache-2.0 code; 600+ langs; 9 Indic langs w/ hours; `instruct=` API | HIGH |
| S28 | `https://huggingface.co/k2-fsa/OmniVoice/raw/main/README.md` §License | HF model card | **"pre-trained model is licensed under the CC-BY-NC"** | HIGH |
| S29 | `https://raw.githubusercontent.com/huggingface/parler-tts/main/LICENSE` + `https://huggingface.co/parler-tts/parler-tts-large-v1` | LICENSE + HF card | Parler-TTS Apache-2.0 code AND weights | HIGH |
| S30 | `https://huggingface.co/{ResembleAI/chatterbox,coqui/XTTS-v2,IndexTeam/IndexTTS-2}/raw/main/README.md` | HF card YAML | Chatterbox mit; XTTS `license: other`/CPML; IndexTTS-2 no YAML licence | HIGH |
| S31 | `https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/README.md` | Official README | Fun-CosyVoice3-0.5B-2512 naming/links; benchmark table | HIGH |
| S32 | `https://github.com/resemble-ai/chatterbox` | Official README | 23 languages incl. Hindi; `Chatterbox-Multilingual-hi`; repo IDs | MEDIUM |
| S33 | `https://raw.githubusercontent.com/resemble-ai/chatterbox/master/{LICENSE,src/chatterbox/tts.py,src/chatterbox/mtl_tts.py,src/chatterbox/models/t3/modules/cond_enc.py,src/chatterbox/models/voice_encoder/config.py,src/chatterbox/models/s3gen/s3gen.py}` | LICENSE + source | MIT; `Conditionals.save/load`; `speaker_embed_size=256`; `embed_ref` needs ref mel | HIGH |
| S34 | `https://raw.githubusercontent.com/SWivid/F5-TTS/main/LICENSE` | LICENSE file | F5-TTS code = MIT | HIGH |
| S35 | `https://raw.githubusercontent.com/index-tts/index-tts/main/LICENSE` | LICENSE file | bilibili MULA §1.4 scope, §2.2 threshold, §2(c) AI-model clause | HIGH |
| S36 | `https://raw.githubusercontent.com/yl4579/StyleTTS2/main/LICENSE` | LICENSE file | StyleTTS2 code = MIT | HIGH |
| S37 | Web search result summaries (BentoML, Pinggy, Hyperstack roundups) | **Secondary — NOT relied upon** | Used only to *discover* OmniVoice and Parler-TTS as candidates; every claim then re-verified against S27-S29. Note the blog claim "OmniVoice … Apache-2.0" is **contradicted** by S28. | LOW (discovery only) |
| S38 | `https://github.com/hexgrad/kokoro` (`/main/LICENSE`), `https://huggingface.co/hexgrad/Kokoro-82M` (card + `config.json`), `.../resolve/main/voices/af_heart.pt` (**downloaded and loaded**), `kokoro/pipeline.py`, `kokoro/model.py` | LICENSE + HF card + **binary weight file** + source | Apache-2.0 code AND weights; `(510,1,256)` float32 pack; `load_voice()` accepts a tensor; `torch.mean` blending; `h='hi'` + 4 Hindi voices; `style_dim: 128` | HIGH |
| S39 | `https://raw.githubusercontent.com/fishaudio/fish-speech/main/LICENSE`; `https://huggingface.co/fishaudio/fish-speech-1.5` | LICENSE + HF card | "Fish Audio Research License Agreement" (2026-03-07), §III hosted-API clause, §V scope; weights `cc-by-nc-sa-4.0` | HIGH |
| S40 | `https://huggingface.co/bosonai/higgs-tts-2-3b-base` (LICENSE + card); `https://github.com/boson-ai/higgs-audio` (`/main/LICENSE`, `README_V2.md`) | LICENSE + card + README | Apache code; "Boson Higgs Audio 2 Community License" (Llama-3-derived), 100k-AAU §2 clause; ≥24 GB VRAM; repo rename | HIGH |
| S41 | `https://raw.githubusercontent.com/coqui-ai/TTS/dev/TTS/.models.json` | Vendor manifest | `your_tts` → `"license": "CC BY-NC-ND 4.0"`; 512-d `d_vectors` | HIGH |
| S42 | `https://huggingface.co/api/models/yl4579/StyleTTS2-{LJSpeech,LibriTTS}`; StyleTTS2 README §License; `Configs/config_libritts.yml` | HF API + README + config | `cardData: None` → **no declared weights licence**; `ref_s` 256-d (128+128); GPL phonemizer note | HIGH |
| S43 | `https://github.com/canopyai/Orpheus-TTS` (`/main/LICENSE`); `https://huggingface.co/canopylabs/orpheus-3b-0.1-ft`; `engine_class.py` | LICENSE + card + source | Apache code; weights apache-2.0 but `gated: auto` + Llama-3.2-3B base; `f"{voice}: {prompt}"` — no vector | HIGH code / MEDIUM weights |
| S44 | `https://github.com/nari-labs/dia` (`/main/LICENSE`, README); `https://huggingface.co/nari-labs/Dia-1.6B-0626` | LICENSE + README + card | Apache-2.0 code AND weights; "different voices every time you run"; fp16 ~4.4 GB | HIGH |
| S46 | `https://huggingface.co/SWivid/F5-TTS` (card YAML); `https://huggingface.co/api/models/ResembleAI/Chatterbox-Multilingual-{TTS,hi}` | HF card YAML + API | F5-TTS weights `cc-by-nc-4.0`; Chatterbox `-hi` = mit/ungated; `-TTS` umbrella = HTTP 401 | HIGH |
| S45 | `https://github.com/SesameAILabs/csm` (`/main/LICENSE`, `generator.py`); `https://huggingface.co/sesame/csm-1b` | LICENSE + source + card | Apache-2.0 both; `encode(f"[{speaker}]{text}")` — the int is text, not an embedding index | HIGH |
