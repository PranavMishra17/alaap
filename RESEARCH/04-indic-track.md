# 04 — The Indic Track

> **Domain:** Indic TTS backends, quality ceiling, per-language shippability, Indic data
> **Answers:** B4, and the Indic half of B1/E1
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [03-tts-backends-english.md](03-tts-backends-english.md) · [05-datasets-and-annotation.md](05-datasets-and-annotation.md) · [06-evaluation-harness.md](06-evaluation-harness.md) · [08-licensing-propagation.md](08-licensing-propagation.md)

## 0. Bottom line

- **The Indic track CAN reach Tier 1.** The brief's implicit assumption that Indic is permanently Tier-2/3 is **wrong**. `SPRINGLab/Indic-Mio` (Apache-2.0, 609M, all 22 scheduled Indian languages + English) is built on `Aratako/MioCodec-25Hz-24kHz` (MIT), whose documented API exposes a **continuous `global_embedding` vector** representing speaker identity, and a `decode(content_token_indices, global_embedding)` entry point. That is an addressable, persistable, interpolatable speaker vector on a clean Apache-2.0/MIT chain. **HIGH** for the API's existence; **UNVERIFIED** that it round-trips cleanly through Indic-Mio's generation path (§10, E1).
- **Indic Parler-TTS's description channel is NOT a voice designer.** It controls *style and acoustics* (pitch, rate, expressivity, reverberation, background noise, quality, emotion) plus a **closed set of 69 named speaker strings**. The RASMALAI paper never evaluates minting a novel voice from a description. Arbitrary new voice identity from text is **not supported**; coarse voice variation via gender/age/accent adjectives is. **HIGH**.
- **The brief's IndicF5 licence claim is wrong, but the correct answer is worse than "MIT".** The HF card says `license: mit` and an AI4Bharat org member states on the record that commercial use is fine. **But** IndicF5 = "IN-F5", a **fine-tune of the English F5-TTS checkpoint**, whose weights are **CC-BY-NC-4.0**. AI4Bharat cannot relicense a third party's NC-restricted weights. IndicF5 is therefore **legally contested, not clean** — research lane until cleared. **HIGH** on the facts, **UNVERIFIED** on the legal outcome.
- **The same defect afflicts `SPRINGLab/SPRING_F5`** (Apache-2.0 tag, 23 Indian languages, released 2026-08-17) which *explicitly declares* `base_model: SWivid/F5-TTS`. This is a **systemic problem across the Indic F5 family**, not a one-off. **HIGH**.
- **Indic Parler-TTS is the only major Indic TTS with a fully clean licence chain end to end** — Apache-2.0 code (`huggingface/parler-tts`), Apache-2.0 base weights (`parler-tts-mini-v1.1`), Apache-2.0 released weights, and all four training corpora CC-BY-4.0/CC-V1. **HIGH**.
- **Bulbul v3 is not the quality ceiling the brief assumes.** Sarvam's own blog states: *"General (full-band) evaluations: ElevenLabs v3 alpha leads on audio quality."* Bulbul v3 is the top performer **only in the 8 kHz telephony condition**. For a games/dramatic-content product rendering full-band audio, the ceiling is ElevenLabs v3, not Bulbul. **HIGH**.
- **Sarvam has released zero TTS weights.** Their HF org holds 14 repos — LLMs (`sarvam-105b`, `sarvam-30b`, `sarvam-m` all Apache-2.0) and eval datasets, **no TTS model**. API-only is confirmed, not assumed. **HIGH**.
- **`parler-tts` the library is effectively unmaintained** — last commit 2024-12-10, 130 open issues, and it hard-pins `transformers>=4.46.1,<=4.46.1` (exact). It must run in its own isolated process. **HIGH**.
- **RASMALAI, the 13,000-hour description-annotated corpus, was never released.** HF search for `rasmalai` returns **zero models and zero datasets**. The paper promises release; only the trained model (`indic-parler-tts`) shipped. **The only Indic corpus with natural-language voice descriptions attached is not obtainable.** **HIGH**.
- **Script handling is a hard requirement, not a nicety.** Every credible Indic backend expects **native script**. Roman-script Hindi ("aapka order confirm ho gaya hai") degrades output on Sarvam by their own docs, and needs an IndicXlit (MIT) transliteration pre-pass for open models. Mixed *script* code-switching (Devanagari Hindi + Latin English words) is well supported; Romanised Hindi is not. **HIGH**.

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| 1 | IndicF5 is **CC-BY-NC** | **WRONG — and the truth is messier** | Card declares `license: mit`; AI4Bharat org member affirms commercial use on the record. **But** IndicF5 is a fine-tune of CC-BY-NC-4.0 F5-TTS weights, so the MIT grant is contested upstream. Neither "CC-BY-NC" nor "clean MIT" is correct. | [HF API cardData](https://huggingface.co/api/models/ai4bharat/IndicF5) · [discussion #34](https://huggingface.co/ai4bharat/IndicF5/discussions/34) · [arXiv 2505.20693](https://arxiv.org/abs/2505.20693) · [SWivid/F5-TTS card](https://huggingface.co/SWivid/F5-TTS) | HIGH (facts) / UNVERIFIED (legal) |
| 2 | Indic Parler-TTS: **21 languages** | **CORRECT** | 20 Indic + English officially; plus 3 unofficial (Chhattisgarhi, Kashmiri, Punjabi). Note only **18 languages have named speakers**. | [model card](https://huggingface.co/ai4bharat/indic-parler-tts) | HIGH |
| 3 | Indic Parler-TTS: **69 named voices** | **CORRECT — with a catch** | Enumerated table sums to exactly 69 name-slots, but "Riya" appears under both Bengali and English → **68 unique strings**. Konkani, Santali, Sindhi, Urdu, Kashmiri have **no named voices at all**. | model card speaker table | HIGH |
| 4 | Indic Parler-TTS: **0.9B, Apache-2.0** | **CORRECT** | `model.safetensors` = 3.751 GB F32 → 937,803,241 params. Apache-2.0 verified on code, base weights and released weights. | [HF tree API](https://huggingface.co/api/models/ai4bharat/indic-parler-tts/tree/main) | HIGH |
| 5 | Indic Parler-TTS: **no speaker vector** | **CORRECT** | Voice is addressed only by a name *string* inside the description. No embedding input, no reference-audio cloning. | model card usage code | HIGH |
| 6 | Description cross-attention `generate(input_ids=<desc>, prompt_input_ids=<transcript>)` | **CORRECT** | Verbatim in the card's usage examples. Two separate tokenizers (description uses `model.config.text_encoder`). | model card | HIGH |
| 7 | IndicF5: **11 languages, 1417 hrs, reference-audio F5/flow-matching** | **CORRECT** | as, bn, gu, hi, kn, ml, mr, or, pa, ta, te. Requires ref audio **plus its transcript**. | [IndicF5 card](https://huggingface.co/ai4bharat/IndicF5) | HIGH |
| 8 | Bulbul v3: **best-in-class quality** | **WRONG** | Sarvam's own blog: *"ElevenLabs v3 alpha leads on audio quality"* at full-band. Bulbul v3 tops only the 8 kHz telephony track. | [sarvam.ai/blogs/bulbul-v3](https://www.sarvam.ai/blogs/bulbul-v3) (via Wayback) | HIGH |
| 9 | Bulbul v3: **API-only closed** | **CORRECT** | sarvamai HF org: 14 repos, zero TTS models. | [HF org API](https://huggingface.co/api/models?author=sarvamai) | HIGH |
| 10 | Bulbul v3: **35+ voices, 11 languages → 22 planned** | **CORRECT** | 35 v3 speakers; 11 languages (hi, bn, ta, te, gu, kn, ml, mr, pa, od, en). | [docs.sarvam.ai](https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert) | HIGH |
| 11 | Bulbul v3: **native Hinglish code-switching** | **PARTLY WRONG** | Mixed-*script* code-switching works (demo: `मैं Suresh बोल रहा हूँ ABC Finance से`). But Sarvam's docs warn *"Transliterated input … significantly reduces output quality"* and advise "Always use native script for Indic words." Romanised Hindi is **not** supported. | docs.sarvam.ai + blog | HIGH |
| 12 | IndicVoices-R: **1704 hrs / 10,496 spk / 22 langs / 93.25% extempore** | **CORRECT** | All four figures verified. Licence **CC-BY-4.0** — permits training *and releasing* a model, with attribution. | [arXiv 2409.05356](https://arxiv.org/abs/2409.05356) · [dataset card](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | HIGH |
| 13 | (Implied) Indic track is Tier-2/3 only | **WRONG** | `SPRINGLab/Indic-Mio` + MioCodec expose a continuous speaker `global_embedding`. Tier 1 is reachable on an Apache-2.0/MIT chain. | [Indic-Mio card](https://huggingface.co/SPRINGLab/Indic-Mio) · [MioCodec card](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz) | HIGH (API) / UNVERIFIED (works end-to-end) |
| 14 | (Omission) Brief lists only 3 Indic candidates | **INCOMPLETE** | Missed **Indic-Mio**, **DhVaani-0.5**, **SPRING_F5**, **vits_rasa_13**, and the MIT Indic ASR needed for the eval harness. | §6 | HIGH |
| 15 | "VoxCPM2's Indic coverage" | **OVERSTATED** | VoxCPM-0.5B is **Chinese/English only**. VoxCPM2's 30-language list includes **Hindi and no other Indian language**. | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | HIGH |

## 2. The Indic backend table

| Model | Params | Code lic. | Weights lic. | Licence URL | Conditioning | Speaker vector? | Voice design from text? | Languages | Code-switch? | VRAM | Servable? | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ai4bharat/indic-parler-tts** | 937.8M | Apache-2.0 (`huggingface/parler-tts`) | **Apache-2.0** | [card](https://huggingface.co/ai4bharat/indic-parler-tts) | Description cross-attn + transcript | **No** | **Partial** — style/acoustics yes; identity = closed set of 69 names | 20 Indic + En official; +3 unofficial | Mixed-script only | 3.75 GB fp32 / ~1.9 GB bf16 | **Yes** (isolated venv) | **SHIP** — the only clean-chain Indic model. Use as voice *designer*. |
| **SPRINGLab/Indic-Mio** | 608.9M (BF16) | Apache-2.0 (via base) | **Apache-2.0** | [card](https://huggingface.co/SPRINGLab/Indic-Mio) | Text + codec global embedding; emotion tags | **YES** — MioCodec `global_embedding` | No (emotion tags only) | **All 22 scheduled + En** | **Yes** (card widget is Hinglish) | ~1.2 GB bf16 + 132M codec | **Yes** (vLLM) | **SHIP (pending E1)** — the Tier-1 route. Not gated. |
| **ARTPARK-IISc/DhVaani-0.5** | 122.8M (F32) | Apache-2.0 (`k2-fsa/ZipVoice`) | **Apache-2.0** | [card](https://huggingface.co/ARTPARK-IISc/DhVaani-0.5) | Ref waveform + its transcript | No | No | **27** incl. bho, mag, raj, hne | Unverified | <2 GB; CPU-capable | Yes (gated: auto) | **SHIP as fallback** — widest coverage, smallest model. |
| **ai4bharat/IndicF5** | ~350M (1.40 GB) | **NONE** (GitHub `license: null`) | `mit` tag, **contested** (derived from CC-BY-NC-4.0 F5-TTS) | [card](https://huggingface.co/ai4bharat/IndicF5) | Ref audio **+ transcript** | No | No | 11 (exactly our target set) | Yes (per Phir Hera Fairy) | ~2–3 GB | Technically yes | **RESEARCH LANE ONLY** until upstream NC cleared. |
| **SPRINGLab/SPRING_F5** | n/a (no safetensors index) | — | `apache-2.0` tag, **contested** (declares `base_model: SWivid/F5-TTS`) | [card](https://huggingface.co/SPRINGLab/SPRING_F5) | Ref audio + transcript | No | No | 23 Indic + En | **Yes** (CodeMix widgets) | ~2–3 GB | Not gated | **RESEARCH LANE ONLY** — same defect, more explicit. |
| **ai4bharat/vits_rasa_13** | ~40M (161 MB) | — (gated) | **CC-BY-4.0** (commercial OK w/ attribution) | [card](https://huggingface.co/ai4bharat/vits_rasa_13) | `speaker_id` int + `emotion_id` int | **Partial** — internal embedding table, addressed by integer | No | 13 — **no Hindi, no Gujarati, no Odia** | No | ~1 GB | Yes (gated) | **NICHE** — no Hindi kills it as a primary. |
| **facebook/mms-tts-\*** | ~36M each | — | **CC-BY-NC-4.0** | [mms-tts-hin](https://huggingface.co/facebook/mms-tts-hin) | Text only | **No** (`num_speakers: 1`) | No | ~15 Indic repos | No | <1 GB | — | **DISQUALIFIED** — NC licence, single voice per language. |
| **Sarvam Bulbul v3** | closed | closed | closed, API-only | [docs](https://docs.sarvam.ai/api-reference-docs/models/bulbul) | 35 preset voices | No | No | 11 | Mixed-script yes, Romanised no | n/a | API | **CEILING REFERENCE ONLY** — ₹30/10k chars. |

## 3. Indic Parler-TTS in depth

### 3.1 What the description channel actually controls

This is the single most important question in the brief, and the answer is **narrower than hoped**.

The model card enumerates six controllable axes, all *acoustic/stylistic*:

| Control | Range |
|---|---|
| Background Noise | clear → slightly noisy |
| Reverberation | close-sounding → distant-sounding |
| Expressivity | monotone → highly expressive |
| Pitch | high / low / moderate |
| Speaking Rate | slow → fast |
| Speech Quality | basic → refined |

Plus **emotion** for 10 officially-supported languages (Command, Anger, Narration, Conversation, Disgust, Fear, Happy, Neutral, Proper Noun, News, Sad, Surprise), and **accent** ("A male British speaker", "A female American speaker").

Voice *identity* enters by a completely different mechanism: **the speaker's name is a literal token in the description string**. From the card: *"this checkpoint was also trained on pre-determined speakers, characterized by name (e.g. Rohit, Karan, Leela, Maya, Sita, ...)"*, invoked as `"Divya's voice is monotone yet slightly fast in delivery..."`.

The RASMALAI paper confirms this is by construction. Captions were LLM-generated from structured attribute tags, and the attribute set includes a literal **Speaker ID** field (Table 2). The worked example is:

> *"Jaya, a female speaker, delivers speech in a slightly roomy environment with a high-pitched, expressive tone."*

**Can an arbitrary NEW voice be described into existence?** **No — not reliably.** Evidence:

1. Every evaluation in the RASMALAI paper conditions on either a **named seen speaker** (§4.1, §4.3, §4.4) or on **acoustic attributes** (§4.2). There is **no experiment** synthesising a novel, unnamed, repeatable voice identity from a description.
2. Attribute-level accuracy (Table 6) is reported for c50, F0 std, F0 mean, SNR, speaking rate and PESQ — i.e. *acoustics*, not identity.
3. The model card's own framing for unnamed descriptions is the section titled **"🎲 Random voice"** — an explicit admission that without a name, the voice is a draw, not a design.

So the voice channel is best modelled as: **`voice = named_speaker (closed set of 69) × {gender, age, accent} adjectives × 6 acoustic axes × emotion`**. That is a rich *style* space wrapped around a *closed identity* space. For Alaap this means Indic Parler-TTS is a **voice designer for style, and a voice selector for identity**.

### 3.2 Route to a stable identity (Tier 1/2/3?)

- **Tier 1 — No.** No speaker embedding is exposed on any input path. The only continuous representation is the description encoder's hidden states, which are a deterministic function of the description text; caching them is equivalent to storing the string, and they live in a *caption* space, not a *speaker* space. Interpolating there interpolates wording, not timbre. Not a Tier-1 vector.
- **Tier 2 — Yes, and this is the recommended use.** Fix a description + seed, generate once, keep the **waveform**, then clone from that waveform forever using a zero-shot backend. This is the "mint once" pattern in the brief, and it works because the minting model and the rendering model can differ. **Constraint:** the renderer's language set gates this. IndicF5 covers 11 languages, Indic-Mio 22, DhVaani 27 — so pairing Parler (designer) with Indic-Mio (renderer) keeps the chain clean *and* wide.
- **Tier 3 — Yes, and unusually strong for Tier 3.** A *named* speaker plus a fixed description is far more reproducible than the brief's "description + RNG seed" characterisation implies: the paper reports **speaker similarity 0.95** across generations, described as *"exceptionally consistent in preserving speaker identity across generated utterances."* But it is still Tier 3: it breaks on a version bump, and it is limited to 69 identities.

**Reproducibility of description+seed for an *unnamed* voice: UNVERIFIED.** No published experiment measures voice-identity variance across seeds for a fixed unnamed description. This is cheap to measure ourselves (§10).

### 3.3 Language tiers and quality

**Yes, Indic Parler-TTS has explicit tiers** — three of them, and they do not coincide:

1. **Officially supported (21):** Assamese, Bengali, Bodo, Dogri, English, Gujarati, Hindi, Kannada, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu.
2. **Unofficial (3):** Chhattisgarhi, Kashmiri, **Punjabi**.
3. **Has named speakers (18):** the speaker table — which *includes* Punjabi and Chhattisgarhi but *excludes* Konkani, Santali, Sindhi, Urdu, Kashmiri.

Note the trap: **Punjabi is "unofficial" yet has two recommended voices (Divjot, Gurpreet); Urdu is "official" yet has none.**

**Published quality (model card, Native Speaker Score %, finetuned):**

| Language | NSS (finetuned) | | Language | NSS (finetuned) |
|---|---|---|---|---|
| Sanskrit | 99.79 ± 0.34 | | Malayalam | 86.54 ± 1.67 |
| Maithili | 95.36 ± 2.52 | | Bengali | 86.16 ± 1.85 |
| Bodo | 94.47 ± 4.12 | | Hindi | **84.79 ± 2.09** |
| Odia | 88.94 ± 3.26 | | Nepali | 80.02 ± 5.75 |
| Dogri | 88.80 ± 3.57 | | Urdu | 77.75 ± 3.82 |
| Telugu | 88.54 ± 1.86 | | Marathi | **76.96 ± 1.45** |
| Kannada | 88.17 ± 2.81 | | Konkani | 76.60 ± 4.14 |
| Assamese | 87.36 ± 1.81 | | Sindhi | 76.46 ± 1.29 |
| Manipuri | 85.63 ± 2.60 | | **Tamil** | **75.48 ± 2.18** |
| | | | **Gujarati** | **75.36 ± 1.78** |

**RASMALAI paper, independent metrics** ([arXiv 2505.18609v2](https://arxiv.org/abs/2505.18609)):

- **MUSHRA, Rasa-13 average: 81.7 ± 2.7** vs **human 89.7 ± 1.8** vs prior SOTA IndicVoiceCraft **73.0 ± 3.4**.
- Near-human in four languages: Bodo 85.4 (human 93.4), Maithili 84.8 (87.7), **Marathi 88.0 (91.8)**, **Telugu 85.9 (92.4)**.
- Biggest gains over baseline: Assamese 61.6→82.1, Bengali 73.2→83.4, Bodo 71.0→85.4, Nepali 55.0→75.4.
- **CER 12% · WER 24% · Noresqa-MOS 4.14 · speaker-similarity 0.95 · IF-BLEU 93.18**.
- Emotion perceptual accuracy: fear 85.09%, happy 84.74%, surprise 78.83%, anger 72.54%, **sad 65.35%** (15.18% confused with fear).
- Zero-shot expressive transfer: Native 86.73, Proximal 80.86, Distal 76.01.

**Read the WER honestly: 24% WER / 12% CER is high** for a shippable TTS. It is a *consolidated* figure across 24 languages including very low-resource ones, but it is the only published intelligibility number and it is not a strong one. Measure per-language WER ourselves before shipping (§10).

**Caution on cross-reading NSS vs MUSHRA:** NSS is a percentage from a "MOS-like framework"; MUSHRA is a 0–100 hidden-reference scale. Marathi is *low* on NSS (76.96) but *near-human* on MUSHRA (88.0 vs 91.8). **The two tables disagree.** Do not rank languages on NSS alone.

**Maintenance:** the weights were last touched 2025-09-24; there is **no v2 and no successor**. AI4Bharat's only 2026 TTS release is `bhili-tts` (2026-09-01, MIT), for Bhili — outside our language set. The `parler-tts` library's last commit is **2024-12-10** with 130 open issues.

## 4. IndicF5 & the non-commercial lane

**The brief says CC-BY-NC. The card says MIT. Both are wrong in different directions.**

**What is verified:**

- `cardData.license = "mit"`, repo last modified **2026-03-03**. There is no NC clause anywhere in the card; the only restriction is a Terms of Use: *"you agree to only clone voices for which you have explicit permission."*
- On **2026-03-02** a user opened discussion #34, *"Commercial Use Inquiry – IndicF5 License Clarification"*, asking specifically about paid SaaS. On **2026-03-03** an **AI4Bharat org member** replied: *"The model is released in MIT license. So its completely open for commercial usage."* The repo's `lastModified` is that same timestamp.
- Training data is clean: Rasa (CC-BY-4.0), IndicVoices-R (CC-BY-4.0), IndicTTS, LIMMITS.

**What breaks it:**

- The method paper is **"Phir Hera Fairy: An English Fairytaler is a Strong Faker of Fluent Speech in Low-Resource Indian Languages"** ([arXiv 2505.20693](https://arxiv.org/abs/2505.20693)), by the same lab. Its abstract compares *"(i) training from scratch, (ii) fine-tuning English F5 on Indian data, and (iii) fine-tuning on both"*, and concludes: **"Fine-tuning with only Indian data proves most effective and the resultant IN-F5 is a near-human polyglot."**
- **`SWivid/F5-TTS` weights are `cc-by-nc-4.0`** (card YAML, verified directly). The *code* is MIT; the *weights* are NC. This is exactly the code/weights split the brief warns about.
- A CC-BY-NC-4.0 licensor's NonCommercial restriction binds adapted material. AI4Bharat can license their own contribution as MIT; they cannot extinguish SWivid's restriction on the weights they started from.

**Two further defects:**

1. **The GitHub code has no licence at all.** `api.github.com/repos/AI4Bharat/IndicF5` returns `license: null`, and `raw.githubusercontent.com/.../LICENSE` returns 404. The `pip install git+https://github.com/ai4bharat/IndicF5.git` you are told to run installs **unlicensed** code (default: all rights reserved). Mitigation: the HF repo vendors the same `f5_tts/` tree under the repo's MIT tag, and upstream F5-TTS code is MIT — so the code side is re-derivable. The **weights** side is not.
2. **Quality complaints are public and numerous.** Community discussions include *"Extremely poor and unintelligible Hindi output despite correct reference audio and transcript"*, *"IndicF5 Hindi TTS Skipping Words During Generation"*, and *"Random garbled fragments of non-existent word output, with same input different gibberish on different runs"*. **MEDIUM** — self-selected complaints, but the non-determinism report is a direct threat to a *persistent identity* product and must be reproduced before relying on it.

**Verdict:** IndicF5 is the **research-lane / quality-reference** model. Do not ship it publicly until either (a) SWivid grants a commercial exception, (b) AI4Bharat states in writing that IN-F5 was trained from scratch, or (c) we retrain the same recipe from a permissive base. Confidence that the derivation happened: **HIGH**. Confidence in the legal outcome: **UNVERIFIED** — this is genuinely unsettled law and AI4Bharat asserts otherwise.

`SPRINGLab/SPRING_F5` (Apache-2.0 tag, 2026-08-17, 23 Indian languages + English, trained on IndicVoices + Rasa) has the **identical defect, declared even more plainly** — its card literally lists `base_model: SWivid/F5-TTS`. Treat the whole Indic-F5 family as one contaminated lane.

## 5. Sarvam Bulbul v3 — the ceiling

**Verified facts** ([docs.sarvam.ai](https://docs.sarvam.ai), [blog](https://www.sarvam.ai/blogs/bulbul-v3) — the blog 403s to direct fetch; retrieved via Wayback snapshot 2026-06-11 of the 2026-02-05 post):

- **API-only, confirmed by absence:** the `sarvamai` HF org holds 14 repos — `sarvam-105b`, `sarvam-30b`, `sarvam-m` (+GGUF/FP8 variants, all **Apache-2.0**), `shuka-1` (llama3 licence), `sarvam-translate` (**GPL-3.0**), `OpenHathi-7B`. **Zero TTS models.** Sarvam *does* open-weight LLMs; they have never open-weighted a TTS.
- **35 v3 voices** (shubh default, aditya, ritu, priya, neha, rahul, pooja, rohan, simran, kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa, kabir, aayan, ashutosh, advait, anand, tanya, tarun, sunny, mani, gokul, vijay, shruti, suhani, mohit, kavitha, rehan, soham, rupali). v2 has 8.
- **11 languages:** hi, bn, ta, te, gu, kn, ml, mr, pa, od, en (all `-IN`).
- **Pricing: ₹30 per 10,000 characters** = ₹0.003/char ≈ **₹3,000 (~US$34) per 1M chars**, "charged per character, rounded up." ₹100 free credits. For scale: Google Chirp3-HD is $30/1M chars, ElevenLabs v3 is $100/1M. **Sarvam is not dramatically cheaper than Western TTS.**

**The quality claim, corrected.** Sarvam commissioned an **independent third-party blind A/B study by Josh Talks** across 11 languages, 50–70 annotators per language, ~2,000 votes per language, **20,000+ total votes from 500+ annotators**, against ElevenLabs v3 alpha, ElevenLabs Flash v2.5, and Cartesia Sonic-3, in two conditions. Their own summary:

> **"General (full-band) evaluations: ElevenLabs v3 alpha leads on audio quality; Bulbul V3 outperforms Cartesia Sonic-3 and all other competitors."**
> **"8 kHz (telephony) evaluations: Bulbul V3 is the clear top performer across all competitors."**

So the true ranking at full-band is **ElevenLabs v3 alpha > Bulbul v3 > Cartesia Sonic-3 > rest**. Bulbul wins outright only at telephony bandwidth. **For Alaap — games and dramatic content, full-band — the ceiling is ElevenLabs v3, and Bulbul v3 is the best *Indian-language-specialist* second.** Set the target accordingly.

> ⚠️ A secondary extraction of the blog's charts produced per-competitor win-rates (63.14% vs EL Flash v2.5; 36.87% vs Sonic-3; 28.11% vs EL v3 alpha at 48 kHz). Those figures **contradict the blog's own prose** ("outperforms Cartesia Sonic-3"). The numbers live in chart images I could not read directly. **Treat the percentages as UNVERIFIED; trust the prose.**

Sarvam also publishes: error rate 8.60% (vs Cartesia 8.98%, EL v3 10.86%, EL Flash 31.90%), word-skip 7.47%, mispronunciation 7.84%, CER 0.03 numerics / 0.02 STEM. **MEDIUM** — vendor-reported, chart-derived.

### How to benchmark against the ceiling cheaply and legally

**Sarvam released their benchmark prompt sets.** This is the cheapest legitimate route:

- **[`sarvamai/tts-general-benchmark`](https://huggingface.co/datasets/sarvamai/tts-general-benchmark)** — 1,815 text prompts, 11 languages, two tracks: `high_quality` (1,265 — Conversational Bots 275, Audiobook 132, News 121, General 110, Education 110, AI Assistants 110, Content Creation 110, Culture 77, Announcements 110, **Indianisms 55**, Insane Repetition 55) and `8khz_telephony` (550).
- **[`sarvamai/tts-robustness-benchmark`](https://huggingface.co/datasets/sarvamai/tts-robustness-benchmark)** — 959 prompts across 7 stress domains: Numerics 267, STEM 245, Indian Named Entities 215, Abbreviations 71, **Code-mixed 70**, **Romanized 53**, URLs/Emails 38.

These are **text-only** — no audio, no voices. That is ideal: we synthesise with *our* stack, run ASR, and compute CER/WER on identical prompts, producing numbers directly comparable to Sarvam's published CER. The `Code-mixed` and `Romanized` categories are precisely the Hinglish question.

**Licence caveat:** both are tagged `license: other` with **no licence text in the card** — UNVERIFIED. The general card states: *"This is an evaluation-only benchmark dataset intended for testing and comparison — not for model training."* Use for internal evaluation with attribution; **do not train on them and do not redistribute.**

For the listening half, budget roughly **₹100–₹500 of Bulbul credits** (₹30/10k chars) to render the same prompts through the ceiling model for side-by-side MUSHRA. That is a rounding error and completely legitimate paid API use.

## 6. Models & datasets the brief missed

**⭐ SPRINGLab/Indic-Mio — the most important find in this report.**
Apache-2.0, **not gated**, 608,894,976 params BF16, 44 kHz, **RTF < 0.1**, last modified 2026-05-27. Supports **all 22 scheduled Indian languages + English**. Trained on IndicTTS, Rasa and Syspin — on *a single A6000 ADA in under 6 hours*, which means we can realistically fine-tune it ourselves. Card states: *"Zero-shot voice cloning supported via speaker embeddings in the codec. Also works well for code-mixed sentences."* Emotion tags (`<happy> <sad> <angry> <disgust> <fear> <surprise>`, plus `<enunciated> <confused> <whisper>` for English) placed at end of sentence; word stress via `*asterisks*`.

**Licence chain, fully verified and fully clean:**
`SPRINGLab/Indic-Mio` (Apache-2.0) ← `Aratako/MioTTS-0.6B` (Apache-2.0) ← `Qwen/Qwen3-0.6B-Base` (Apache-2.0); codec `Aratako/MioCodec-25Hz-24kHz` (**MIT**); training data `ai4bharat/Rasa` (CC-BY-4.0), `libritts_r`, `expresso`, `amphion/Emilia-Dataset` (**CC-BY-4.0**, not NC). **No NC anywhere in the chain.**

**⭐ The Tier-1 mechanism.** MioCodec's card documents the decomposition explicitly:

> *"**Content Tokens:** Discrete representations that primarily capture linguistic information and phonetic content ('what' is being said) at a low frame rate (25 Hz). **Global Embeddings:** A continuous vector representing broad acoustic characteristics ('how') — including speaker identity, recording environment, and microphone traits."*

And the API takes it as an argument:
```python
features = model.encode(waveform)          # -> features.global_embedding
resynth  = model.decode(content_token_indices=..., global_embedding=...)
vc_wave  = model.voice_conversion(source, reference)   # zero-shot VC
```

**This is a persistable, addressable, interpolatable speaker vector on an MIT codec.** Two caveats, both material:
1. The vector **entangles speaker identity with recording environment and microphone traits** — by the card's own wording. It is not a clean d-vector/x-vector. Interpolation will move room acoustics along with timbre.
2. MioCodec's own language list is **en, ja, nl, fr, de, it, pl, pt, es, ko, zh — no Indian language.** Its WavLM-base+ encoder never saw Indic speech. Indic-Mio fine-tuned the *LM*, not necessarily the codec. Speaker fidelity for Indic voices is **UNVERIFIED**.
3. **No published MOS/WER/CER of any kind** on the Indic-Mio card, and no paper — only a HF-repo citation by Advait Joglekar. Quality is completely unmeasured. This is the biggest risk on an otherwise ideal candidate.

**ARTPARK-IISc/DhVaani-0.5.** Apache-2.0, 122,798,800 params F32 (~491 MB), 24 kHz, flow-matching, base `k2-fsa/ZipVoice` (Apache-2.0), last modified **2026-07-26** — the newest credible Indic TTS. **27 Indian languages**, uniquely including Bhojpuri, Magahi, Rajasthani and Chhattisgarhi. Reference-waveform + transcript only, **no speaker vector**. Runs on CPU. Gated `auto`. No published quality numbers; the card self-declares low-resource languages are markedly weaker.

**SPRINGLab/SPRING_F5.** Apache-2.0 tag, 2026-08-17, 23 Indian languages + English, `ai4bharat/IndicVoices` + `Rasa`, **not gated**. Its widget examples are explicitly **CodeMix-Hindi / CodeMix-Tamil / CodeMix-Telugu** — the clearest published evidence of code-switching in any open Indic model. **But** it declares `base_model: SWivid/F5-TTS` → same CC-BY-NC contamination as IndicF5.

**ai4bharat/vits_rasa_13.** CC-BY-4.0 (commercially usable with attribution), ~40M params, 161 MB. Exposes `model(input_ids, speaker_id=16, emotion_id=0)` — 20 speaker IDs × 14 style IDs. The internal VITS speaker-embedding table is the closest thing to a lookup-table identity, but it is a **fixed roster addressed by integer**, with no zero-shot cloning. **Fatally, it has no Hindi, no Gujarati and no Odia** (languages: as, bn, brx, doi, kn, mai, ml, mr, ne, pa, sa, ta, te). *I could not read `modeling_vits.py` or `config.json` — the repo is gated and returns HTTP 401 — so whether an arbitrary continuous vector can be injected in place of the integer lookup is **UNVERIFIED**.*

**Meta MMS-TTS — DISQUALIFIED.** `facebook/mms-tts-*` is **CC-BY-NC-4.0** (public hosting = commercial use → excluded), and the configs show `num_speakers: 1`, `speaker_embedding_size: 0` — **one fixed voice per language, no embedding**. Useless for a voice-identity product even if the licence were clean. ~15 Indic-family repos.

**VoxCPM — the brief's claim does not hold.** `VoxCPM-0.5B` states it is *"trained primarily on Chinese and English data"* with `language: [en, zh]`. `VoxCPM2` (2B, Apache-2.0) lists 30 languages including **Hindi and no other Indian language**. It does offer text-driven "Voice Design", so it is worth a look **for Hindi only** — but it is not an Indic model.

**Also checked and disqualified:** XTTS-v2 (Coqui CPML, non-commercial), OuteTTS 1.0 (CC-BY-NC-SA-4.0), IndexTTS-2 (bilibili licence, no Indic), Kokoro-82M (English only), Llasa-3B (CC-BY-NC), Orpheus/MOSS-TTS/Higgs Audio v2/Zonos (no Indic). `kenpath/svara-tts-v1` (Apache-2.0, 19 Indic languages, 198k downloads) is an **Orpheus-3B derivative whose base is a `research_release`** — flag before shipping. **MEDIUM.**

**Commercial landscape (for the ceiling only):** **Deepgram Aura has zero Indian languages** — rule it out entirely. **Krutrim has no TTS** (Dhwani is speech-to-*text*, under a non-OSI "Krutrim Community License"). **Navana has no TTS** (ASR only). ElevenLabs v3 covers 10 Indian languages at $0.10/1k chars (Multilingual v2 and Flash v2.5 cover **only Hindi and Tamil**). Cartesia Sonic covers 10. Google covers 12 Indic locales, Chirp3-HD at $30/1M chars. **Gnani.ai** is the only vendor explicitly advertising *"Urban Hinglish"* and code-switched speech (claimed MOS 4.23, p95 latency <250 ms) but publishes no pricing. **Bhashini** (MeitY) serves six TTS families including `Bhashini/IITM/TTS` across 24 languages, but **licences are not stated on the model list page — UNVERIFIED**.

**⭐ Two MIT tools that the evaluation harness needs (§06):**
- **`ai4bharat/indic-conformer-600m-multilingual`** — **MIT**, 600M, Conformer CTC+RNNT ASR for **all 22 official Indian languages**. This is how we compute WER/CER on our own Indic TTS output. Essential, and permissively licensed.
- **`ai4bharat/Cadence`** — MIT *tag*, punctuation restoration for English + 22 Indic languages. **⚠️ Its `base_model` is `google/gemma-3-1b-pt`, licensed `gemma` (gated, not OSI-open).** The MIT tag on a Gemma derivative is questionable. Prefer `Cadence-Fast` only if the Gemma terms are reviewed and accepted — or avoid both.

## 7. Per-language shippability verdict

"Best Apache-2.0 option" = clean-chain permissive only (Indic Parler-TTS, Indic-Mio, DhVaani). IndicF5/SPRING_F5 are excluded as licence-contested. "Quality vs ceiling" compares published Indic Parler-TTS numbers against Bulbul v3 / ElevenLabs v3.

| Language | Best Apache-2.0 option | Quality vs ceiling | Ship? | Confidence |
|---|---|---|---|---|
| **Hindi** | Parler (Rohit, Divya, Aman, Rani) + Indic-Mio | NSS 84.79 ± 2.09; below ceiling but solid. Most-contested language — every commercial rival is strongest here | **YES** | HIGH (licence), MEDIUM (quality) |
| **Tamil** | Parler (Kavitha, Jaya — only Jaya recommended) + Indic-Mio | NSS **75.48** — joint-lowest. Only 1 recommended voice | **CONDITIONAL** — ship with reduced voice range; re-measure before promising Tamil | HIGH / MEDIUM |
| **Telugu** | Parler (Prakash, Lalitha, Kiran) + Indic-Mio | NSS 88.54; **MUSHRA 85.9 vs human 92.4 — near-human** | **YES — strongest tier** | HIGH |
| **Bengali** | Parler (Arjun, Aditi + 4) + Indic-Mio | NSS 86.16; MUSHRA 83.4 (vs IndicVC 73.2). 6 voices | **YES** | HIGH |
| **Marathi** | Parler (Sanjay, Sunita + 4) + Indic-Mio | NSS 76.96 **but MUSHRA 88.0 vs human 91.8 — near-human.** Tables disagree | **YES** — trust MUSHRA | MEDIUM (conflicting metrics) |
| **Kannada** | Parler (Suresh, Anu + 2) + Indic-Mio | NSS 88.17 | **YES** | HIGH |
| **Malayalam** | Parler (Anjali, Anju, Harish) + Indic-Mio | NSS 86.54 | **YES** | HIGH |
| **Gujarati** | Parler (Yash, Neha) + Indic-Mio | NSS **75.36** — joint-lowest. Only 2 voices, only 21.24 hrs in finetune set | **CONDITIONAL** — weakest of the majors | HIGH / MEDIUM |
| **Punjabi** | Parler (Divjot, Gurpreet) + Indic-Mio | **No NSS row published.** Language is officially "unofficial" yet has 2 recommended voices. Only 11.07 hrs training data — the least of any language | **NO — not on published evidence.** Must measure first | LOW |
| **Odia** | Parler (Manas, Debjani) + Indic-Mio | NSS 88.94 — one of the highest | **YES** | HIGH |
| **Assamese** | Parler (Amit, Sita + 2) + Indic-Mio | NSS 87.36; MUSHRA 82.1 (vs IndicVC 61.6 — biggest gain) | **YES** | HIGH |

**Summary: ship 8 (Hindi, Telugu, Bengali, Marathi, Kannada, Malayalam, Odia, Assamese), conditional 2 (Tamil, Gujarati), hold 1 (Punjabi).**

Two caveats that apply to the whole table: (1) all NSS/MUSHRA numbers describe **Indic Parler-TTS**, i.e. the *designer*; if Indic-Mio is the renderer, **its** per-language quality is entirely unmeasured. (2) The consolidated **24% WER** is high enough that per-language intelligibility must be re-measured before any public launch.

## 8. Indic data: IndicVoices-R, Rasa, RASMALAI

| Dataset | Size | Speakers | Languages | Licence | Descriptions? | Gated |
|---|---|---|---|---|---|---|
| **IndicVoices-R** | **1,704 hrs** ✓ | **10,496** ✓ | **22** ✓ | **CC-BY-4.0** | No | auto |
| **Rasa** | ~400 hrs (paper); 288 hrs used in Parler | 20 | 13 | **CC-BY-4.0** | No — but has `style`, `gender`, `language`, `duration` columns | auto |
| **IndicVoices** (parent) | 7,200+ hrs | 16,237 | 22 | **CC-BY-4.0** | No | auto |
| **RASMALAI** | 13,000 hrs | — | 23 Indic + En | Paper CC-BY-4.0 | **YES — 24M annotations** | **NOT RELEASED** |
| **IndicTTS** (IITM) | 382 hrs (in Parler) | 2/lang | 12–13 | **CC-BY-4.0** *claimed by AI4Bharat*; IITM site unreachable | No | — |
| **LIMMITS** | 568 hrs (in Parler) | 2/lang | 7–9 | **CC-BY-4.0** *claimed by AI4Bharat* | No | — |
| **GLOBE** | 535 hrs | 23,519 | English (164 accents) | "CC V1" per Parler card | Accent labels | — |

**IndicVoices-R: all four brief claims verified.** 1,704 hours, 10,496 speakers, 22 Indian languages (arXiv 2409.05356 abstract), 93.25% extempore (dataset card). **Licence CC-BY-4.0** — and yes, **this permits training a model and releasing it**, including commercially, subject only to attribution. It is a NeurIPS 2024 D&B paper and the authors state *"We open-source all data and code."* Repo is `gated: auto` (accept terms + HF token, auto-approved).

**Rasa: CC-BY-4.0**, ~400 hrs, 20 speakers across 13 languages, with **style and emotion labels** (ALEXA, ANGER, BB, BOOK, CONV, DIGI, DISGUST, FEAR, HAPPY, NEWS, SAD, SURPRISE, UMANG, WIKI) and speaker identity. Not natural-language descriptions, but **structured tags — which is exactly the input RASMALAI's pipeline consumed.**

**RASMALAI: the answer to the brief's key data question is "yes it exists, no you can't have it."**
The paper describes 13,000 hours and **24 million text-description annotations** covering speaker identity, accent, emotion, style and background, in three flavours per utterance (Descriptive, Concise, Attribute-Robust) plus a **Native Prompt** translated into the target language via IndicTrans2. Its attribute coverage table beats AudioBox, PromptTTS2 and LibriTTS-P, and it is the only multilingual one. The conclusion says *"We release Rasmalai, along with our models and code."*

**They did not.** HF search for `rasmalai` returns **zero datasets and zero models**. The AI4Bharat org's 146 models contain no RASMALAI checkpoint. **Only the trained model shipped** — `ai4bharat/indic-parler-tts` is the released artefact of RASMALAI. Confidence: **HIGH** (exhaustive API search).

**So: is there ANY Indic corpus with natural-language voice descriptions attached? No — none obtainable.** We must annotate from scratch. The good news is that **the recipe is fully published and cheap to reproduce**:

1. Take IndicVoices-R (CC-BY-4.0, 1,704 hrs, 10,496 speakers) and/or Rasa (CC-BY-4.0, style + emotion + speaker labels already present).
2. Extract acoustic attributes — pitch (penn/torchcrepe), C50 + SNR (Brouhaha/pyannote), speaking rate, PESQ — then **discretise into bins**.
3. Feed binned attributes to an instruct LLM to emit Descriptive / Concise / Attribute-Robust captions (they used Llama-3.1-8B-Instruct — fits our 12GB budget quantised).
4. Optionally translate to native script with IndicTrans2 (`ai4bharat/indictrans2-en-indic-1B`, **MIT**).

Every input is CC-BY-4.0 and every tool is permissive. **This is the single highest-leverage dataset action available to the project** — it reconstructs the missing corpus on a licence chain we can actually ship. See [05-datasets-and-annotation.md](05-datasets-and-annotation.md).

**IITM IndicTTS licence is UNVERIFIED.** `iitm.ac.in/donlab/indictts/database` was unreachable and its `license.pdf` returned no response. AI4Bharat's Parler model card asserts **CC BY 4.0**, and an HF mirror (`SPRINGLab/IndicTTS_Tamil`) is tagged cc-by-4.0 — but donlab documentation elsewhere references signing a "License For Use of IndicTTS" agreement. **Do not assume commercial clearance; obtain the PDF.** This matters because IndicTTS is in the training mix of Indic Parler-TTS, IndicF5 *and* Indic-Mio. **LOW confidence.**

## 9. What this means for the build

**The recommended Indic stack — two models, two roles:**

| Role | Model | Revision | Licence |
|---|---|---|---|
| **Voice designer** (description → seed waveform) | `ai4bharat/indic-parler-tts` | `7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca` | Apache-2.0, clean chain |
| **Voice renderer** (identity → arbitrary dialogue) | `SPRINGLab/Indic-Mio` | pin at load; last mod 2026-05-27 | Apache-2.0, clean chain |
| **Codec / speaker vector** | `Aratako/MioCodec-25Hz-24kHz` | pin at load; last mod 2026-02-03 | MIT |
| **Wide-coverage fallback** | `ARTPARK-IISc/DhVaani-0.5` | last mod 2026-07-26 | Apache-2.0 |
| **Eval ASR** (WER/CER) | `ai4bharat/indic-conformer-600m-multilingual` | last mod 2026-02-07 | MIT |
| **Transliteration** (Roman → native) | `AI4Bharat/IndicXlit` | — | MIT |
| **Text normalisation** | `indic_nlp_library` | — | MIT |
| **Quality reference only** | `ai4bharat/IndicF5` @ `ba85abedf18dc479a447eaa0eccbd76ab78a47d5` | — | **contested — do not ship** |

**Why this shape.** Indic Parler-TTS is the only Indic model that turns *a description* into speech, and Indic-Mio is the only clean-licence Indic model that exposes *a speaker vector*. Chaining them gives description → seed waveform → `global_embedding` → persisted identity → unlimited dialogue. That is **Tier 2 guaranteed, Tier 1 if E1 passes** — and every link is Apache-2.0 or MIT.

> ### ⚠️ SUPERSEDED ON LICENCE — read [08 §4.4](08-licensing-propagation.md) and [08 §4.7](08-licensing-propagation.md) before building on this table
>
> The sentence above — *"every link is Apache-2.0 or MIT"* — was written from the **declared** licences. The deeper chain trace in `08` reached the opposite conclusion on **both** halves of this stack, and `08` wins: it traced the training inputs, this table read the YAML.
>
> | Role | This table said | `08` found | Status |
> |---|---|---|---|
> | Voice **designer** — `ai4bharat/indic-parler-tts` | Apache-2.0, clean chain | §4.7 **CONDITIONAL**: 382 of 1,806 h are IITM IndicTTS, whose recovered EULA §2.2 forbids onward sublicensing; repo is gated and the gate terms are unread | `public_servable = False` |
> | Voice **renderer** — `SPRINGLab/Indic-Mio` | Apache-2.0, clean chain | §4.4 **BLOCKED, HIGH confidence**: `ylacombe/expresso` (CC-BY-NC-4.0) is a *declared, direct* training input in Indic-Mio's own YAML; Emilia (NC) enters twice transitively via MioTTS-0.6B and MioCodec; MioCodec derives from the unlicensed `kanade-tokenizer`. MioCodec is unavoidable at inference. | `public_servable = False`, **needs lawyer** |
>
> **Consequence, stated plainly: as of 2026-09-05 there is no publicly-servable Indic path.** Both halves of the recommended stack fail the audit, and `Qwen3-TTS` — the backend every experiment E0–E10 actually runs on — supports ten languages, none of them Indian (`LANG_ALIAS` in `alaap/renderer.py`).
>
> This blocks **shipping**, not **building**. Corpus access, captioning, measurement and the mapper are all unaffected and are proceeding on `SPRINGLab/IndicVoices-R_*` mirrors. See `NEEDS-FROM-YOU.md` §1–2 for the two cheap actions that would unblock the designer half, and `DECISIONS.md` ADR-006 for the four routes out.

**Install constraints for one 8–12GB GPU:**

- **Three mutually incompatible Python environments. Run them as separate processes/services — this is not optional.**
  - `parler-tts` hard-pins **`transformers>=4.46.1,<=4.46.1`** (exact equality) and also needs `descript-audio-codec` + `descript-audiotools` from a git URL.
  - `Indic-Mio` wants **vLLM** and a modern `transformers` (`AutoModelForCausalLM`, chat templates) plus `miocodec` from GitHub.
  - `IndicF5` uses `trust_remote_code=True` and has an open issue titled *"Fix transformers 5.0.0 compatibility"*.
- **VRAM is not the constraint.** Parler 3.75 GB fp32 → ~1.9 GB bf16; Indic-Mio 1.2 GB bf16 + 132M codec; IndicF5 1.4 GB; DhVaani 0.49 GB. **All four fit simultaneously in 12GB.** Note Indic-Mio's own card launches vLLM with `--gpu-memory-utilization 0.5`.
- **All AI4Bharat repos are `gated: auto`** — `indic-parler-tts`, `IndicF5`, `vits_rasa_13`, and the datasets. CI needs an `HF_TOKEN` with terms accepted. **`SPRINGLab/Indic-Mio` and `SPRING_F5` are NOT gated** — a real operational advantage. Note `/raw/` returns 401 on gated repos while `/resolve/` serves the card; use `/resolve/`.
- **Throughput:** Indic-Mio publishes **RTF < 0.1** at 44 kHz. Parler publishes no RTF; HF's `INFERENCE.md` (SDPA, `torch.compile`, batching, streaming) is the optimisation path. **Parler RTF on consumer hardware: UNVERIFIED.**

**Text normalisation and script — mandatory pre-pass:**

1. **Native script is required.** Every backend's examples are Devanagari/Gurmukhi/Tamil/Telugu script. Sarvam's docs say plainly that transliterated input *"significantly reduces output quality"*; the same holds for open models, whose tokenizers are Indic-script-trained (Parler uses an expanded Llama2 tokenizer with byte fallback — it will *encode* Roman Hindi, but read it as English phonetics).
2. **Roman-script Hindi must be transliterated before synthesis**, via IndicXlit (MIT). Expect to need an override table for English loanwords and short Hindi function words that transliterators systematically misread.
3. **Mixed-script code-switching is fine and is the intended input form.** Indic-Mio's own widget: `प्लान तो बढ़िया है, but wait... Have you checked the hotel bookings? Last minute पे रूम मिलना is next to impossible on weekends.` SPRING_F5 ships CodeMix widgets for Hindi/Tamil/Telugu. **Rule: Indic words in Indic script, English words in Latin script.**
4. **Punctuation drives prosody.** Parler's card: *"Punctuation can be used to control the prosody of the generations, e.g. use commas to add small breaks."* For ASR-derived or user-typed text, restore punctuation first — but note the Gemma-licence caveat on `Cadence`.
5. **Include "very clear audio" in every Parler description** to pin recording quality; the card states this explicitly.
6. **Numerics, abbreviations and named entities are the known failure class** — that is what Sarvam's robustness benchmark exists to measure (267 numerics, 245 STEM, 215 Indian named entities). Expand digits, currency, dates and units to words in-language before synthesis.

**Licence hygiene to propagate to [08-licensing-propagation.md](08-licensing-propagation.md):**
- Never ship anything descending from `SWivid/F5-TTS` weights (CC-BY-NC-4.0): rules out IndicF5, SPRING_F5, and any F5 derivative.
- MMS-TTS (CC-BY-NC-4.0), XTTS-v2 (CPML), OuteTTS (CC-BY-NC-SA), Llasa (CC-BY-NC) are all excluded by the public-hosting-is-commercial rule.
- `Cadence` and `Cadence-Fast` carry an MIT tag over a **Gemma-licensed** base — review before use.
- CC-BY-4.0 (IndicVoices-R, Rasa, vits_rasa_13) **is shippable** — attribution only. Do not treat CC-BY as a blocker.

## 10. Open — must be settled by experiment

| # | Question | Cheapest experiment | Est. cost/time | What it blocks |
|---|---|---|---|---|
| **E1** | **Does MioCodec's `global_embedding` survive Indic-Mio's generation path?** Can we `encode()` a reference, keep the vector, generate content tokens for *new* text, and `decode()` with the stored vector to get the same voice? | Accept HF terms; `encode` 20 Indic reference clips → store vectors; generate 5 new sentences each; `decode` with stored vs default embedding; measure speaker similarity with WavLM-base-plus-sv | **1 day, 1 GPU** | **The entire Tier-1 claim for Indic.** Highest-value experiment in this document. |
| **E2** | Is the `global_embedding` **interpolatable**? Does a 50/50 blend of two voices sound like a coherent third voice, or like artefacts? | Take 10 vector pairs, interpolate at α ∈ {0, .25, .5, .75, 1}, render fixed text, human A/B for "is this one coherent voice" | 1 day + small listening test | Whether Indic identities can be *blended* (a headline Alaap feature) and whether a mapper can regress into this space |
| **E3** | How badly does the room/mic entanglement hurt? MioCodec's vector mixes speaker with recording environment | Render the same speaker vector extracted from clean vs noisy reference clips; measure timbre drift | 0.5 day | Whether seed waveforms need studio-clean minting, and whether Tier-2 seeds must be quality-gated |
| **E4** | **Indic-Mio has zero published quality numbers.** Is it actually good? | Synthesise `sarvamai/tts-general-benchmark` high_quality track (1,265 prompts × 11 langs); ASR with `indic-conformer-600m-multilingual`; report per-language CER/WER | 2 days, mostly GPU time | Whether the recommended renderer is shippable at all — and the whole §7 table |
| **E5** | Per-language WER — is the consolidated **24%** hiding disasters? | Same harness as E4, run on Indic Parler-TTS across all 11 target languages | Shares E4's harness | §7 verdicts, especially Tamil / Gujarati / **Punjabi** (which has no published number at all) |
| **E6** | For an **unnamed** description, how stable is voice identity across seeds? | Fix one description, generate 50 samples at different seeds, compute pairwise speaker similarity; repeat for a *named* speaker as control | 0.5 day | Whether Parler descriptions can ever be Tier-3 identities, or must always be Tier-2 (mint + freeze) |
| **E7** | Do the 69 named voices stay stable across the pretrained → finetuned checkpoints? | Render identical descriptions on `indic-parler-tts` and `indic-parler-tts-pretrained`; measure similarity | 0.5 day | How brittle Tier-3 really is; sizes the version-pinning risk |
| **E8** | Does **Romanised Hindi** work at all, or is IndicXlit mandatory? | Run the 53 `Romanized` + 70 `Code-mixed` prompts from `tts-robustness-benchmark` through Parler and Indic-Mio, raw vs IndicXlit-transliterated; CER both | 1 day | The Hinglish product decision and the text-pipeline architecture |
| **E9** | **Is IndicF5 legally usable?** | Email AI4Bharat (authors of arXiv 2505.20693) asking directly: was the released checkpoint initialised from `SWivid/F5-TTS` weights? Ask SWivid for a commercial exception | 1 email, days–weeks latency | Whether the best-known-quality Indic model is available; also unblocks SPRING_F5 |
| **E10** | Is `vits_rasa_13`'s speaker embedding injectable as a continuous vector? | Accept gate, read `modeling_vits.py`, try replacing the `nn.Embedding` lookup with an arbitrary tensor | 0.5 day | A CC-BY-4.0 Tier-1 backup if E1 fails — but only for its 13 languages (no Hindi) |
| **E11** | Obtain the **IITM IndicTTS licence PDF** | Contact IITM DONLab directly; site was unreachable | 1 email | Whether the training-data chain under Parler/Indic-Mio is genuinely clean |

## 11. Sources

| # | URL | Type | Used for | Confidence in source |
|---|---|---|---|---|
| 1 | https://huggingface.co/ai4bharat/indic-parler-tts | Official model card | Licence, 21 languages, 69 voices, controls, NSS table, training data | HIGH |
| 2 | https://huggingface.co/api/models/ai4bharat/indic-parler-tts | HF API (cardData) | `license: apache-2.0`, gated status, sha, param count via tree | HIGH |
| 3 | https://arxiv.org/abs/2505.18609 (v2) | Peer-reviewed paper (Interspeech 2025) | RASMALAI corpus, MUSHRA/CER/WER/MOS/S-SIM/IF-BLEU, description construction | HIGH |
| 4 | https://huggingface.co/ai4bharat/IndicF5 | Official model card | `license: mit`, 11 languages, 1417 hrs, Terms of Use | HIGH |
| 5 | https://huggingface.co/ai4bharat/IndicF5/discussions/34 | Official maintainer statement | AI4Bharat org member: "MIT license… completely open for commercial usage" (2026-03-03) | HIGH |
| 6 | https://arxiv.org/abs/2505.20693 | Peer-reviewed paper | **IN-F5 is a fine-tune of English F5-TTS** — the licence-contamination evidence | HIGH |
| 7 | https://huggingface.co/SWivid/F5-TTS | Official model card | Weights `cc-by-nc-4.0` (vs MIT code) — the code/weights split | HIGH |
| 8 | https://api.github.com/repos/SWivid/F5-TTS | GitHub API | F5-TTS **code** licence = MIT | HIGH |
| 9 | https://api.github.com/repos/AI4Bharat/IndicF5 | GitHub API | IndicF5 code `license: null` — unlicensed | HIGH |
| 10 | https://huggingface.co/SPRINGLab/Indic-Mio | Official model card | Apache-2.0, 22 languages, 44kHz, RTF<0.1, speaker-embedding cloning, code-mix widget | HIGH |
| 11 | https://huggingface.co/Aratako/MioCodec-25Hz-24kHz | Official model card | **MIT; `global_embedding` continuous speaker vector; `decode()`/`voice_conversion()` API** | HIGH |
| 12 | https://huggingface.co/Aratako/MioTTS-0.6B | HF API | Apache-2.0, base = Qwen3-0.6B-Base, data = Emilia (CC-BY-4.0) | HIGH |
| 13 | https://huggingface.co/ARTPARK-IISc/DhVaani-0.5 | HF API + card | Apache-2.0, 27 languages, 122.8M params, ZipVoice base | HIGH |
| 14 | https://huggingface.co/SPRINGLab/SPRING_F5 | Official model card | Apache-2.0 tag over `base_model: SWivid/F5-TTS`; CodeMix widgets | HIGH |
| 15 | https://huggingface.co/ai4bharat/vits_rasa_13 | Official model card | CC-BY-4.0, 13 languages (no Hindi), 20 speaker IDs, 14 style IDs | HIGH |
| 16 | https://www.sarvam.ai/blogs/bulbul-v3 (via web.archive.org 20260611133516) | Official vendor blog | Josh Talks study design; **"ElevenLabs v3 alpha leads on audio quality"**; Hinglish demos | HIGH (content) / MEDIUM (chart numbers unreadable) |
| 17 | https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert | Official API docs | 35 voices, 11 languages, transliteration warning | HIGH |
| 18 | https://docs.sarvam.ai/api-reference-docs/pricing | Official pricing page | ₹30 per 10,000 characters | HIGH |
| 19 | https://huggingface.co/api/models?author=sarvamai | HF API | **Zero TTS models** — API-only confirmed; LLMs Apache-2.0 | HIGH |
| 20 | https://huggingface.co/datasets/sarvamai/tts-general-benchmark | Official dataset card | 1,815 prompts, 11 languages, two tracks, "evaluation-only" | HIGH (content) / UNVERIFIED (licence "other") |
| 21 | https://huggingface.co/datasets/sarvamai/tts-robustness-benchmark | Official dataset card | 959 prompts, 7 domains incl. Code-mixed 70 / Romanized 53 | HIGH |
| 22 | https://arxiv.org/abs/2409.05356 | Peer-reviewed paper (NeurIPS 2024 D&B) | IndicVoices-R: 1,704 hrs / 10,496 spk / 22 langs | HIGH |
| 23 | https://huggingface.co/datasets/ai4bharat/indicvoices_r | Official dataset card | CC-BY-4.0, 93.25% extempore | HIGH |
| 24 | https://huggingface.co/api/datasets/ai4bharat/Rasa | HF API | CC-BY-4.0; per-language splits; style/gender columns | HIGH |
| 25 | https://api.github.com/repos/huggingface/parler-tts | GitHub API | Apache-2.0; **last push 2024-12-10**; 130 open issues | HIGH |
| 26 | https://raw.githubusercontent.com/huggingface/parler-tts/main/setup.py | Source file | **`transformers>=4.46.1,<=4.46.1`** hard pin | HIGH |
| 27 | https://huggingface.co/api/models?author=ai4bharat (146 models) | HF API | No TTS successor; `bhili-tts` 2026-09-01 is the only 2026 TTS | HIGH |
| 28 | https://huggingface.co/api/models?search=rasmalai + datasets | HF API | **RASMALAI not released — zero results** | HIGH |
| 29 | https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual | Official model card | MIT, 600M, 22 languages ASR — eval harness | HIGH |
| 30 | https://huggingface.co/ai4bharat/Cadence | Official model card | MIT tag; **base `google/gemma-3-1b-pt` (`license: gemma`)** | HIGH |
| 31 | https://api.github.com/repos/AI4Bharat/IndicXlit | GitHub API | MIT — transliteration | HIGH |
| 32 | https://api.github.com/repos/anoopkunchukuttan/indic_nlp_library | GitHub API | MIT — normalisation | HIGH |
| 33 | https://huggingface.co/facebook/mms-tts-hin | Official model card + config | CC-BY-NC-4.0; `num_speakers: 1`; `speaker_embedding_size: 0` | HIGH |
| 34 | https://huggingface.co/openbmb/VoxCPM2 | Official model card | 30-language list — Hindi only among Indian languages | HIGH |
| 35 | https://developers.deepgram.com/docs/tts-models | Official docs | Zero Indian languages — ruled out | HIGH |
| 36 | https://elevenlabs.io/docs/models · /pricing/api | Official docs | v3 = 10 Indian languages; $0.10/1k chars | HIGH |
| 37 | https://docs.cartesia.ai/api-reference/tts/tts · https://cartesia.ai/pricing | Official docs | 10 Indian languages; credit pricing | HIGH |
| 38 | https://cloud.google.com/text-to-speech/pricing | Official pricing | Chirp3-HD $30/1M chars; 12 Indic locales | HIGH |
| 39 | https://huggingface.co/krutrim-ai-labs/Dhwani | Official model card | Krutrim has **no TTS**; Dhwani is STT; non-OSI licence | HIGH |
| 40 | https://dibd-bhashini.gitbook.io/bhashini-apis/available-models-for-usage | Official docs | 6 TTS families served; **licences not stated** | HIGH (list) / UNVERIFIED (licences) |
| 41 | https://www.gnani.ai/text-to-speech-api | Vendor page | Only vendor explicitly advertising "Urban Hinglish"; claimed MOS 4.23 | MEDIUM |
| 42 | https://www.iitm.ac.in/donlab/indictts/database | Official site | **UNREACHABLE — licence PDF not obtained** | UNVERIFIED |
