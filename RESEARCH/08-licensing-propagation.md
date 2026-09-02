# 08 — Licence Propagation & Legal Constraints

> **Domain:** the three licence questions; upstream chains; train-and-release obligations; the servability gate
> **Answers:** E1; scope section 15.3; the `public_servable` gate in scope section 12.5
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [01-ttv-landscape.md](01-ttv-landscape.md) · [03-tts-backends-english.md](03-tts-backends-english.md) · [04-indic-track.md](04-indic-track.md) · [05-datasets-and-annotation.md](05-datasets-and-annotation.md) · [09-safety-and-watermarking.md](09-safety-and-watermarking.md)

> ⚠️ **NOT LEGAL ADVICE.** Engineering research against primary licence text. Every NEEDS-LAWYER row must go to counsel before commercial launch.

---

## 0. Bottom line

- **The paradigm trap is real and I verified all three instances plus two new ones.** VoiceSculptor ships Apache-2.0 over a fine-tune of `HKUSTAudio/Llasa-3B` (CC-BY-NC-4.0, whose card says the licence *"prohibits free commercial use"*). `SPRINGLab/SPRING_F5` declares `license: apache-2.0` and `base_model: SWivid/F5-TTS` in the same YAML block, where F5-TTS's weights are CC-BY-NC-4.0. **IndicF5 is confirmed as IN-F5, a fine-tune of English F5-TTS**, from the paper's own footnote, which links `IN-F5` directly to `https://huggingface.co/ai4bharat/IndicF5`. **HIGH**
- **New finding — `SPRINGLab/Indic-Mio` has the defect, and it was on the critical path.** Its own card says *"For American English, LibriTTS and **Expresso**"* were used. `ylacombe/expresso` is `license: cc-by-nc-4.0`. Its base model `Aratako/MioTTS-0.6B` (Apache-2.0) and its codec `Aratako/MioCodec-25Hz-24kHz` (MIT) both declare `amphion/Emilia-Dataset` as training data, and Emilia's binding terms say CC-BY-**NC**. **Do not put Indic-Mio on the commercial critical path without counsel.** **HIGH** on the facts, **MEDIUM** on the legal consequence (see §5).
- **New finding — `amphion/Emilia-Dataset` declares `license: cc-by-4.0` in machine-readable metadata while its own gate agreement says *"The researcher shall use the Emilia dataset under the CC-BY-NC license"*.** Every licence scanner in existence reads Emilia as permissive. It is not. This single mislabel propagates into F5-TTS, MioCodec, MioTTS, ZipVoice and therefore DhVaani. **HIGH**
- **New finding, and the one that hurts most — `Zonos-v0.1`'s speaker encoder is CC-BY-NC-**SA**-4.0 upstream.** Zonos declares Apache-2.0, but voice cloning routes through a *third* repo, `Zyphra/Zonos-v0.1-speaker-embedding`, whose own card says it is *"based on the ResNet293-SimAM-ASP models from VoxBlink2"*. VoxBlink2's LICENSE reads: *"The data falls under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license."* **This was raised publicly on the Zyphra repo in February 2025 and has never been answered.** Zonos is the scope document's primary target for the entire two-tower design (§6.1, A2). **HIGH on the facts; NEEDS LAWYER on the conclusion.**
- **Two more models load an NC checkpoint at inference time, invisible from the top-level tag.** IndexTTS-2's `model_download.py` pulls `amphion/MaskGCT` (`license: cc-by-nc-4.0`) as a *mandatory* inference component. This and the Zonos encoder were only findable by reading dependency-resolution code — **add "grep the inference path for `hf_hub_download` / `from_pretrained`" to the audit procedure**. **HIGH**
- **VoicePersona's CC0 declaration does not hold.** ~79% of its 15,082 samples come from sources whose declarers did not hold the rights: `laion/laions_got_talent` (52.6%) is ~110 hours of **OpenAI GPT-4o Audio output using OpenAI's eleven proprietary voices** — I confirmed 483 tarballs named `english_alloy_*`, `..._ash_*`, `..._ballad_*`, `..._coral_*`, `..._echo_*`, `..._fable_*`, `..._nova_*`, `..._onyx_*`, `..._sage_*`, `..._shimmer_*`, `..._verse_*` — over which LAION has placed an Apache-2.0 LICENSE file. `AnimeVox` (13.3%) is ripped from *"official English-dubbed versions of popular anime series"*. `AniSpeech` (13.3%) applies the literal MIT software licence text, *"Copyright 2023 ShoukanLabs"*, to 18.8 GB of anime voice recordings. Only `GLOBE_V2` (20.9%, CC0 via Common Voice) is clean. **HIGH**
- **Creative Commons has published its own answer to §15.3's question 3, and it is adverse.** From CC's AI-training guidance: NC means *"all stages, from copying the data during training to **sharing the trained model**, must not be for commercial gain"*; SA means *"AI models **or outputs** … would require AI developers to use the same CC license as the original works"*. CC concedes this is the conservative reading and that copyright exceptions may mean the licence never attaches — but **this is the page a rights-holder will cite**, and it is the reason the recommendation below is "design so the question never has to be answered". **HIGH** that this is CC's published position; **UNSETTLED** as law.
- **The clean English training path is the LibriVox chain**: LibriVox (public domain, US) → LibriSpeech → LibriTTS → LibriTTS-R (the only link with a verified standalone `LICENSE.txt`) → LibriTTS-P, plus MLS and Common Voice. Everything expressive — ParaSpeechCaps, Expresso, EARS, Emilia, ESD, SpeechCraft, TextrolSpeech — is non-commercial or research-only. **That is the real product gap, and it is a licensing gap, not a technical one.** **HIGH**
- **The Indic dataset picture is better than the brief assumed.** IndicVoices, IndicVoices-R and Rasa are all genuinely CC-BY-4.0, and the IndicVoices paper says the licence was chosen *"allowing commercial usage"* in terms. **HIGH**
- **I recovered the IITM IndicTTS End User License Agreement that a sibling could not reach — and it is not what anyone assumed.** It is *not* non-commercial. §2.1 grants a *"perpetual, non-exclusive, worldwide, transferable, sub-licensable, royalty-free"* licence and says *"Licensee shall be allowed to freely distribute the Derivative Work."* **But §2.2 forbids your downstream recipients from onward sale or sub-licensing, and §5 mandates a specific copyright notice.** §2.2 is flatly incompatible with releasing open weights under Apache-2.0/MIT. **MEDIUM-HIGH** (see §3.9 for the provenance caveat).
- **`ai4bharat/IndicF5` is now a gated repo and its licence claim is unchanged.** An AI4Bharat member wrote on 2026-03-03: *"The model is released in MIT license. So its completely open for commercial usage."* That statement is made about a fine-tune of CC-BY-NC-4.0 weights. **A licensor's assurance about their own downstream grant does not cure a defect in the upstream chain.** NEEDS LAWYER.
- **Google's Crowdsourced Indian-language corpora (OpenSLR SLR63/64/65/66/78/79) are CC BY-**SA** 4.0 — the only ShareAlike term in the Indic mix — and IndicF5's paper lists them as training data.** ShareAlike is copyleft-for-adaptations. **HIGH** on the licence; the model-as-adaptation question is unsettled (§5).
- **The servability answer is short.** **Clean: VoxCPM2, Qwen3-TTS (whole family including `VoiceDesign`, all Apache-2.0), Chatterbox, Parler-TTS, CosyVoice 2/3.** Conditional: Indic Parler-TTS (IndicTTS §2.2, and it is now *gated*), Zonos (speaker-encoder taint). **`False`: IndicF5, SPRING_F5, Indic-Mio, DhVaani, VoiceSculptor, Llasa-3B, xcodec2, F5-TTS, XTTS-v2, IndexTTS-2, VibeVoice.**
- **Two compliance deadlines are already past, and one of them requires a change to the render path.** **EU AI Act Art. 50(2)** — machine-readable marking and detectability of synthetic audio — **applied from 2 August 2026**; the Digital Omnibus's four-month grace covers *only* systems already on the market before that date, so a new launch gets none, and Art. 2(12) expressly carves Art. 50 *out* of the open-source exemption. **India's IT Amendment Rules 2026** have required, since ~20 Feb 2026, "**a prominently prefixed audio disclosure**" plus embedded permanent metadata including a unique identifier, non-removable. *(Correction: the percentage-of-display-area figure the brief expected was in the October 2025 draft and did **not** survive into the notified rules.)* **HIGH** on the EU text and dates; **MEDIUM** on the Indian effective date; the scope of the Indian rule for a non-cloned synthetic voice is **UNVERIFIED**. See §6.2–6.3.
- **The best news in the file is a negative finding: VoiceForge's no-audio-upload architecture already sits outside the core of every voice-specific liability regime surveyed.** Tennessee's ELVIS Act tool provision needs a *"particular, identifiable individual"*; NO FAKES §2(c)(2)(B) needs a product *"primarily designed to produce … digital replicas of a specifically identified individual"*; the Bombay High Court in *Arijit Singh* condemned tools that *"enable the conversion of any voice into that of a celebrity"*. **A description-to-novel-voice system with no audio input matches none of them.** Scope §15.1 is doing more work than it was given credit for — do not trade it away. **HIGH**
- **XTTS-v2's CPML restricts the generated audio itself, and there is no longer anyone who can sell you a commercial licence.** Its first substantive line: *"This license allows only non-commercial use of a machine learning model **and its outputs**."* Coqui Inc. dissolved in January 2024; the `license_link` on the model card (`https://coqui.ai/cpml`) is now a dead URL. **Remove it from the roadmap entirely rather than carrying it as a "research lane" option.**

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| 1 | §15.3: *"Your own VoicePersona is **CC0**, which is maximally permissive and a genuine asset here"* | **WRONG — and this is the most consequential correction in the file** | VoicePersona's HF metadata declares `license: cc` (HF's meaningless generic Creative Commons catch-all), not `cc0-1.0`; only a README badge says CC0. More importantly, ~79% of its samples derive from sources whose declarers lacked the rights to grant anything. It is a **liability**, not an asset. | [HF API](https://huggingface.co/api/datasets/Paranoiid/VoicePersona) `cardData.license = "cc"`; upstream analysis §3.13 | HIGH |
| 2 | §6 table: **IndicF5 · License: CC-BY-NC** | **WRONG as to the declaration; RIGHT as to the conclusion** | IndicF5's card declares **MIT**, not CC-BY-NC, and an AI4Bharat member confirmed commercial use in writing. The scope doc reached the correct *verdict* by the wrong *route*. The actual reason it is not servable is the upstream F5-TTS CC-BY-NC-4.0 weights, plus a `license: null` GitHub repo. | [HF API](https://huggingface.co/api/models/ai4bharat/IndicF5) `cardData.license = "mit"`; [discussion #34](https://huggingface.co/ai4bharat/IndicF5/discussions/34) | HIGH |
| 3 | §6 table: **IndexTTS-2 · Restrictive / non-commercial** | **PARTLY WRONG** | It is not non-commercial. It is the **bilibili Model Use License Agreement**, which permits commercial use below 100M MAU / RMB 1bn revenue. The *actual* blocker is §3.4(c): you may not use it or its outputs *"to improve any AI model … except … non-commercial AI models"*, and §1.5 defines model **outputs** as Derivative Work. That kills it as a data source for our mapper. | [LICENSE](https://github.com/index-tts/index-tts/blob/main/LICENSE) §1.5, §2.2, §3.4(c) | HIGH |
| 4 | §6 table: **VibeVoice · Research-only** | **RIGHT for the wrong reason — and the exclusion is stronger than the scope doc knew** | The licence *file* is MIT. But Microsoft **removed the VibeVoice-TTS code from GitHub** on 2025-09-05 and `microsoft/VibeVoice-Large` now returns **HTTP 401**; the surviving card says the model is *"limited to research purpose use"* and *"not intended or licensed for"* a list of scenarios; and **every synthesized file carries an automatically embedded *audible* "This segment was generated by AI" disclaimer**, with an explicit anti-circumvention clause on the Realtime variant. Product-killing independent of the licence. | [GitHub README](https://raw.githubusercontent.com/microsoft/VibeVoice/main/README.md); [VibeVoice-1.5B card](https://huggingface.co/microsoft/VibeVoice-1.5B/raw/main/README.md) | MEDIUM-HIGH |
| 5 | §6 table: **F5-TTS · CC-BY-NC (weights)** | **CORRECT, and worth sharpening** | Weights CC-BY-NC-4.0; **GitHub code is MIT**. The split matters: you may reuse the F5-TTS *code* commercially, just not the checkpoint. | [HF API](https://huggingface.co/api/models/SWivid/F5-TTS) `cc-by-nc-4.0`; `gh api repos/SWivid/F5-TTS` → `MIT` | HIGH |
| 6 | §6 table: **CosyVoice 2/3 · ⚠️ verify** | **VERIFIED — Apache-2.0** | Both the `FunAudioLLM/CosyVoice` repo and the `CosyVoice2-0.5B` weights declare Apache-2.0. | `gh api repos/FunAudioLLM/CosyVoice` → `Apache-2.0`; [HF API](https://huggingface.co/api/models/FunAudioLLM/CosyVoice2-0.5B) | HIGH |
| 7 | §6 table: **Zonos-v0.1 · Apache-2.0 ⚠️ verify** — *"⭐ Primary target for the two-tower / Tier-1 identity design. The one model that makes Option A directly implementable."* | **THE ⚠️ WAS JUSTIFIED. Apache-2.0 is declared on the two repos everyone checks — and the speaker encoder, which is the whole reason Zonos was chosen, is a third repo derived from CC-BY-NC-**SA**-4.0 data.** | Transformer and hybrid weights *are* `apache-2.0` and the GitHub `LICENSE` *is* Apache 2.0. But `speaker_cloning.py` hard-codes `hf_hub_download(repo_id="Zyphra/Zonos-v0.1-speaker-embedding", filename="ResNet293_SimAM_ASP_base.pt")`, and that repo's card says it is *"based on the ResNet293-SimAM-ASP models from VoxBlink2"* — VoxBlink2 being **CC BY-NC-SA 4.0**. Challenged on the repo Feb 2025, unanswered 18 months later. Repo last pushed 2025-03-05. **This is the single highest-impact correction for the architecture, because §7's entire English track routes through this encoder.** | [speaker_cloning.py](https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/speaker_cloning.py); [VoxBlink2 LICENSE](https://raw.githubusercontent.com/VoxBlink2/ScriptsForVoxBlink2/main/LICENSE); [discussion #2](https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding/discussions/2) | HIGH on facts |
| 8 | §6 table: **Qwen3-TTS · ❌ No Hindi / Indian languages** | **CORRECT on languages; the scope doc understates the opportunity** | Qwen3-TTS covers 10 languages, none Indic. **But `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` exists as open Apache-2.0 weights** — a description-conditioned voice-design model that is exactly VoiceForge's shape, with the cleanest licence in the field. It belongs in §6 as a first-class English/multilingual candidate. | [HF search](https://huggingface.co/api/models?search=Qwen3-TTS) — the whole family tags `license:apache-2.0` | HIGH |
| 9 | §15.3: *"ParaSpeechCaps, LibriTTS-P, IndicVoices-R and Rasa each need their licences read"* | **The framing is right; the risk ranking is wrong** | LibriTTS-P, IndicVoices-R and Rasa are all fine (CC-BY-4.0). **ParaSpeechCaps is the one that blocks**: it is CC-BY-NC-**SA**-4.0, and its authors released their own trained checkpoints as CC-BY-NC-SA-4.0 — i.e. the dataset owners themselves treat a model trained on it as encumbered. | [HF raw README](https://huggingface.co/datasets/ajd12342/paraspeechcaps/raw/main/README.md); [ajd12342/parler-tts-mini-v1-paraspeechcaps](https://huggingface.co/api/models/ajd12342/parler-tts-mini-v1-paraspeechcaps) | HIGH |
| 10 | §12.5: *"Publicly servable (Apache-2.0 / MIT): … **Indic Parler-TTS**"* | **CONDITIONAL, not settled** | Indic Parler-TTS is Apache-2.0 and its card lists its data licences openly (the best practice in the field). But 382 of its 1,806 hours are IITM **IndicTTS**, whose EULA §2.2 restricts what your downstream recipients may do. An Apache-2.0 weight release lets them do exactly what §2.2 forbids. | [indic-parler-tts card](https://huggingface.co/ai4bharat/indic-parler-tts) training table; IITM EULA §2.2 | MEDIUM |
| 11 | §5: *"an Apache-2.0 swap-in must be documented for every non-commercial component"* | **CORRECT, and now actionable — but the swap-in list is shorter than the scope assumes** | English renderers: **VoxCPM2, Qwen3-TTS-VoiceDesign, Chatterbox, Parler-TTS** (Zonos is *not* a swap-in; it is now one of the components needing one — row 7). Indic: only Indic Parler-TTS, conditionally. **Speaker encoder: no swap-in is currently documented, and §8.7 argues one is now required.** | §8 | HIGH |
| 12 | Implicit throughout: *a permissive top-level tag means the artefact is clean* | **WRONG — this is the single structural correction** | Six artefacts in this audit declare Apache-2.0/MIT/CC-BY over upstreams that are NC, unlicensed, or contractually restricted. The declaration is the *last* thing to check, not the first. | §2, §4 | HIGH |

---

## 2. The three questions, and the upstream-chain method

Three questions, routinely conflated, that have different answers for the same artefact:

1. **USE** — may I run inference locally, for research?
2. **SERVE** — may I host it publicly and charge for its output? *(Hosting a model publicly is commercial use. This is the project's locked posture and it is the correct reading.)*
3. **TRAIN-AND-RELEASE** — does training on this impose obligations on the weights I then publish?

Q3 is the one that bites late, because the cost of discovering it is a full retrain.

### The method, stated as a procedure

For any artefact `X`:

1. Read the **machine-readable** declaration: `https://huggingface.co/api/models/<id>` → `cardData.license`, `cardData.base_model`, `cardData.datasets`. For GitHub: `gh api repos/<owner>/<repo> --jq .license.spdx_id`. **A `null` licence is not "permissive by default" — it is all rights reserved.**
2. Read the **human-readable** declaration: the card body's licence section, and any `LICENSE`/`LICENSE.txt`/`license.pdf` file in the repo. **These disagree with the metadata often enough that you must read both.**
3. Read the **gate**: `cardData.extra_gated_prompt` on a gated repo is a *contract*, and it overrides the tag. Emilia is the canonical case.
4. **Recurse into every upstream**: `base_model`, the codec/vocoder dependency, and every entry in `datasets`. Then read the *paper's* dataset section, because cards under-declare.
5. Check **code vs weights vs data** separately. F5-TTS is MIT code + CC-BY-NC weights. ParaSpeechCaps is MIT code + CC-BY-NC-SA data. TextrolSpeech is MIT over a repo containing no data at all.
6. Apply **nemo dat quod non habet** — no one gives what they do not have. A downstream grant is void to the extent it exceeds upstream rights. This is the whole game.

7. **Grep the inference path.** `hf_hub_download(...)`, `from_pretrained(...)`, `snapshot_download(...)` — a model can pull a *differently-licensed checkpoint at runtime* that appears nowhere in its metadata. This step found the two most serious model-side findings in the audit (Zonos, IndexTTS-2) and nothing else would have.

### The six failure modes this audit found

| Mode | Example found | Why a scanner misses it |
|---|---|---|
| Permissive grant over NC weights | VoiceSculptor → Llasa-3B; SPRING_F5 → F5-TTS | The `base_model` field is *right there* and simply not cross-checked |
| Permissive grant over NC **data** | Indic-Mio → Expresso; MioCodec → Emilia | Requires reading the card body, not just YAML |
| **NC checkpoint pulled at runtime** | **Zonos → VoxBlink2 encoder; IndexTTS-2 → MaskGCT** | **Invisible from every card and every API field — only the code shows it** |
| Tag contradicts the binding terms | Emilia: tag `cc-by-4.0`, gate says CC-BY-NC | The gate text is not in the licence field |
| Meaningless tag treated as a grant | VoicePersona `cc`; AnimeVox `cc`; Indic Parler's `"CC V1"` | `cc` is a *category*, not a licence; it grants nothing |
| Software licence applied to recordings | AniSpeech MIT over anime audio; TextrolSpeech MIT over a code-only repo | SPDX says "MIT" and stops |

**One structural observation worth acting on: nine of the twelve model weight repos audited here ship no LICENSE file at all.** The grant is a single mutable YAML line. Only XTTS-v2 and IndexTTS-2 ship actual licence text — and those are the two restrictive ones. IndexTTS-2 also proves the field changes: it had *no* licence file for its first seven months. **Snapshot the card and API response, timestamped, at ingest** (§8.4 rule 4).

---

## 3. E1 — dataset licence audit (train-and-release)

| Dataset | Declared licence | Upstream chain | Use? | Train? | Release trained model? | Redistribute? | Operative clause | Confidence |
|---|---|---|---|---|---|---|---|---|
| **ParaSpeechCaps** | CC-BY-NC-**SA**-4.0 (data); MIT (code) | VoxCeleb1/2 + Expresso (NC) + EARS (NC) + Emilia (NC) | ✅ | research only | ❌ **NO** | ❌ NO (SA) | "The dataset and models are licensed under the CC-BY-NC-SA 4.0 license." | HIGH |
| **LibriTTS-P** | CC-BY-4.0 (README only; repo `license: null`) | LibriTTS-R | ✅ | ✅ | ✅ | ✅ + attribution | "## License / [CC BY 4.0]" | MEDIUM-HIGH |
| **LibriTTS-R** | CC-BY-4.0 (`LICENSE.txt` in doc.tar.gz) | LibriTTS | ✅ | ✅ | ✅ | ✅ + attribution | "made available by Google LLC under a Creative Commons Attribution 4.0 International License" | HIGH |
| **LibriTTS** | CC-BY-4.0 (OpenSLR SLR60) | LibriSpeech | ✅ | ✅ | ✅ | ✅ + attribution | `License: CC BY 4.0` | HIGH |
| **LibriSpeech** | CC-BY-4.0 (OpenSLR SLR12; no LICENSE file) | LibriVox | ✅ | ✅ | ✅ | ✅ + attribution | `License: CC BY 4.0` | HIGH |
| **LibriVox** | Public domain (US) | — | ✅ | ✅ | ✅ | ✅ | "all our recordings are public domain… anyone can use all our recordings however they wish (even to sell them)" | HIGH |
| **IndicVoices-R** | CC-BY-4.0 | IndicVoices | ✅ | ✅ | ✅ | ✅ + attribution | "released under the same CC BY 4.0 license" | HIGH |
| **IndicVoices** | CC-BY-4.0 | own collection, consented | ✅ | ✅ | ✅ | ✅ + attribution | "the dataset will be released with CC-BY-4.0 license, allowing commercial usage" | HIGH |
| **Rasa** | CC-BY-4.0 (HF); GitHub repo `license: null` | own collection | ✅ | ✅ | ✅ | ✅ + attribution | card License section: "CC-BY-4.0" | HIGH |
| **RASMALAI** | n/a | — | ❌ | ❌ | ❌ | ❌ | not released as data | MEDIUM |
| **IITM IndicTTS** | Custom **IITM EULA** (not CC) | IITM Speech Technology Consortium / TDIL / MeitY | ✅ | ✅ | ⚠️ **§2.2 conflict** | ⚠️ §5 notice | "Such third party shall not be allowed to further sell, lease, license, sub-license…" | MEDIUM-HIGH |
| **Google Crowdsourced Indic** (SLR63/64/65/66/78/79) | **CC BY-SA 4.0** | own collection | ✅ | ✅ | ⚠️ ShareAlike | ⚠️ must relicense SA | `License: Attribution-ShareAlike 4.0 International` | HIGH |
| **TextrolSpeech** | MIT — *on a code-only repo* | LibriTTS + VCTK + **ESD** + TESS/MEAD/SAVEE/MESS | ✅ | research only | ❌ **NO** | ❌ NO | ESD: "This database can only be used for research purpose." | HIGH |
| **SpeechCraft** | **none — all rights reserved** | Zhvoice + AISHELL-3 + **GigaSpeech-M** + LibriTTS-R | EULA only | ❌ | ❌ **NO** | ❌ NO | "solely for academic, non-commercial research purposes" | HIGH |
| **Emilia** | tag `cc-by-4.0`; **terms say CC-BY-NC** | scraped in-the-wild web audio | gated | ❌ | ❌ **NO** | ❌ NO | "The researcher shall use the Emilia dataset under the CC-BY-NC license" | HIGH |
| **VoxCeleb2** | Oxford: metadata CC-BY-**SA**-4.0; KAIST mirror: CC-BY-4.0 "for research purposes"; audio withdrawn | YouTube, copyright retained by uploaders | impaired | ❌ | ❌ **NO** | ❌ NO | "available to download for research purposes"; "The copyright remains with the original owners of the video." | HIGH |
| **Common Voice** | CC0-1.0 + platform contract | donated voice | ✅ | ✅ | ✅ | ⚠️ contractually discouraged | "we ask that you not post, distribute, or mirror any Common Voice dataset" | HIGH / MEDIUM |
| **MLS** | CC-BY-4.0 (OpenSLR SLR94) | LibriVox | ✅ | ✅ | ✅ | ✅ + attribution + identity term | "You agree to not attempt to determine the identity of speakers in this dataset." | HIGH / MEDIUM |
| **VoicePersona** (ours) | HF tag `cc` (meaningless); README badge CC0 | LAION-Got-Talent + GLOBE_V2 + AniSpeech + AnimeVox | ⚠️ | ⚠️ | ❌ **NO as-is** | ❌ **NO as-is** | see §3.13 | HIGH |

---

### 3.1 ParaSpeechCaps — BLOCKED, and the authors say so themselves

This is the most important dataset finding, because it is the one the scope document most wants to use, and because the dataset's authors have already answered our Q3 for us.

**Declared licence.** [HF raw README frontmatter](https://huggingface.co/datasets/ajd12342/paraspeechcaps/raw/main/README.md): `license: cc-by-nc-sa-4.0`. Confirmed in [API metadata](https://huggingface.co/api/datasets/ajd12342/paraspeechcaps).

**The code/data split, stated by the authors.** [GitHub `LICENSE`](https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/LICENSE) is MIT ("Copyright (c) 2025 Anuj Jitendra Diwan"), and `gh api repos/ajd12342/paraspeechcaps` reports `MIT`. The README separates them explicitly:

> "**LICENSE:** This code repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. **The dataset and models are licensed under the [CC-BY-NC-SA 4.0] license.**"

**"and models."** That is the explicit statement about models trained on the data, and it is not aspirational — both released checkpoints carry it. [`ajd12342/parler-tts-mini-v1-paraspeechcaps`](https://huggingface.co/api/models/ajd12342/parler-tts-mini-v1-paraspeechcaps) has `cardData.license: cc-by-nc-sa-4.0`, as does the `-only-base` variant. **The dataset owners took a Parler-TTS checkpoint that was Apache-2.0, fine-tuned it on ParaSpeechCaps, and released the result as CC-BY-NC-SA-4.0.** Whatever the abstract legal question in §5, the people whose consent would matter have publicly declined to treat the weights as unencumbered.

**Audio is not included** — the card, verbatim:

> "**NOTE**: We release style captions and a host of other useful style-related metadata, but not the source audio files. Please refer to our codebase for setup instructions on how to download them from their respective datasets (VoxCeleb, Expresso, EARS, Emilia)."

And the paper (arXiv 2503.04713, §3.4 footnote):

> "We only provide textual annotations for existing datasets. **Their speech data is subject to their own licenses.**"

**Upstream chain — every audio source is NC or research-only:**

| Source | Licence | Evidence |
|---|---|---|
| VoxCeleb 1+2 (594 celebrities, PSC-Base) | CC-BY(-SA) "for research purposes"; audio withdrawn by Oxford | §3.11 |
| **Expresso** | **CC-BY-NC-4.0** | [speechbot.github.io/expresso](https://speechbot.github.io/expresso/): "The Expresso dataset is distributed under the CC BY-NC 4.0 license." |
| **EARS** | **CC-BY-NC-4.0** | [LICENSE](https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/LICENSE) opens "Attribution-NonCommercial 4.0 International" |
| **Emilia** (all of PSC-Scaled, 2,427 hr ≈ 86% by hours) | **CC-BY-NC-4.0** | §3.12 |

The Emilia pin is decisive: the [dataset setup guide](https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/README.md) instructs use of a specific pinned revision `fc71e07e…`, whose metadata at that commit reads `license: cc-by-nc-4.0`. The `cc-by-4.0` tag now on Emilia's `main` is a later artefact of the YODAS merge and does not reach back.

**Verdicts.** USE ✅ (research). TRAIN-AND-RELEASE ❌. REDISTRIBUTE ❌ — NC blocks commercial redistribution and **SA** would force any rebuilt derivative (i.e. VoicePersona v2, if it inherited from here) to CC-BY-NC-SA-4.0 permanently. **HIGH.**

---

### 3.2–3.6 The LibriVox chain — the clean spine

**LibriVox (root).** [librivox.org/pages/public-domain](https://librivox.org/pages/public-domain/):

> "LibriVox records only texts that are in the public domain (in the USA…), and **all our recordings are public domain**… This means anyone can use all our recordings however they wish (even to sell them)."

> "The recordings are free, and **there is no need to credit LibriVox**, although of course we much prefer if you do credit us."

Two independent mechanisms make this work: the *text* is PD by copyright expiry; the *recording* is PD by volunteer dedication ("if you record for LibriVox, all your recordings will be donated to the public domain"). The [LibriVox wiki](https://wiki.librivox.org/index.php?title=Copyright_and_Public_Domain) lists **"Used LibriVox recordings to train AI models"** among accepted uses and explains the deliberate refusal of CC: *"we didn't want to add any restrictions to the recordings we make."*

**Territorial caveat, in LibriVox's own words:** *"all our recordings are public domain in the USA, but not necessarily in other countries."* A bare PD dedication has uncertain effect in civil-law jurisdictions where moral rights are inalienable. **This is a NEEDS-LAWYER row for EU/India distribution, not for the US.**

**The chain, from OpenSLR licence lines:** [SLR12 LibriSpeech](https://www.openslr.org/12/), [SLR60 LibriTTS](https://www.openslr.org/60/), [SLR141 LibriTTS-R](https://www.openslr.org/141/), [SLR94 MLS](https://www.openslr.org/94/) all read `License: CC BY 4.0`. I fetched all four pages directly.

**LibriTTS-R is the strongest link** — the only one with a standalone licence file, extracted from [openslr.org/resources/141/doc.tar.gz](https://www.openslr.org/resources/141/doc.tar.gz), complete contents:

> "The LibriTTS-R corpus is made available by Google LLC under a Creative Commons Attribution 4.0 International License. See <http://creativecommons.org/licenses/by/4.0/>."

This is the right place for it to appear: Miipher *regenerates* waveforms neurally, so LibriTTS-R audio carries a genuine fresh Google copyright, and Google filed an explicit licence over it. Consistent.

**One honest gap.** Nobody in the chain states *what* the CC-BY-4.0 covers — the PD audio, or only the contributor's own segmentation/alignment/restoration work. Nobody can licence PD material under CC-BY; a licensor can only licence their own contribution. The defensible reading is the latter, but it is asserted nowhere. **UNVERIFIED.** Practically this is benign: it means the chain *adds* an attribution obligation to material that had none, so complying is strictly conservative.

**Trap to avoid.** The [HF `mythicinfinity/libritts_r` card](https://huggingface.co/datasets/mythicinfinity/libritts_r/raw/main/README.md) declares `license: cc-by-4.0` but its BibTeX contains `copyright = "http://creativecommons.org/licenses/by-nc-nd/4.0/"`. **That NC-ND string is the arXiv posting licence of the paper, not the data licence.** An automated scanner will flag LibriTTS-R as non-commercial. It is not. Note this in the compliance tooling.

**LibriTTS-P** ([github.com/line/LibriTTS-P](https://github.com/line/LibriTTS-P), LINE Corp): `gh api repos/line/LibriTTS-P` → **`license: null`**, and no `LICENSE` file exists on either branch. The only grant is the README:

> "## License
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)"

The repo contains no code — only CSV/TXT annotation files — so the single statement reads onto the data. Derivation is clean (annotations over LibriTTS-R audio, which is CC-BY-4.0). **MEDIUM-HIGH**, downgraded only because the grant is README-only. *Action: email LINE for a proper LICENSE file.*

---

### 3.7 IndicVoices-R and IndicVoices — clean, with an express commercial grant

The hypothesis that AI4Bharat declares CC-BY-4.0 downstream over a more restrictive parent **does not hold here**. Verified across five concordant primary sources.

**IndicVoices-R** — [HF API](https://huggingface.co/api/datasets/ai4bharat/indicvoices_r) `cardData.license: cc-by-4.0`; [GitHub `LICENSE.md`](https://raw.githubusercontent.com/AI4Bharat/IndicVoices-R/master/LICENSE.md) is the full CC-BY-4.0 legal text; `gh api repos/AI4Bharat/IndicVoices-R` → `CC-BY-4.0`; Zenodo record 11636050 `license: cc-by-4.0`, `access_right: open`.

arXiv 2409.05356v1 §6, verbatim:

> "The dataset leverages anonymized speech from its parent dataset IndicVoices and **is released under the same CC BY 4.0 license.** IndicVoices underwent rigorous ethical review and approval by the Institute Ethics Committee. It obtained explicit consent from each participant for the use of their speech data."

**Verified negative:** IV-R does *not* ingest IndicTTS, LIMMITS or Google-CS. Those appear in the paper only as benchmark rows and separate fine-tuning experiments.

**IndicVoices (parent)** — arXiv 2403.01926v1 §10, the decisive clause:

> "All tools will be released with a MIT license and **the dataset will be released with CC-BY-4.0 license, allowing commercial usage.**"

An express, unambiguous, first-party grant of commercial use.

**Two caveats.** (a) `gh api repos/AI4Bharat/IndicVoices` → **`license: null`** — the *code* repo is all rights reserved (12 filename/branch probes all 404). Do not copy code from it; the data is fine. (b) All AI4Bharat speech datasets are `gated: "auto"`. `cardData` contains **no** `extra_gated_prompt` for `indicvoices_r`, `IndicVoices` or `Rasa` — consistent with contact-info-only gating. **But nobody in this audit could authenticate to HF and read what sits behind the login wall.** That is the cheapest open item in the file: log in, screenshot the gate.

---

### 3.8 Rasa — released, CC-BY-4.0, and the brief's doubt was misplaced

[`ai4bharat/Rasa`](https://huggingface.co/datasets/ai4bharat/Rasa): 388 GB, 1,145.44 hours / 640,950 utterances across 22 languages × F+M, `license: cc-by-4.0`, `gated: auto`, last modified 2026-06-06. Card License section reads simply "CC-BY-4.0". Citation is Interspeech **2024** (not 2023).

**The GitHub repo is a decoy.** `gh api repos/AI4Bharat/Rasa` → **`license: null`**. Its contents are training code (`TTS`, `configs`, `main.py`, `trainer`, `vocoder.py`), not data. Do not conclude "Rasa is unlicensed" from it — that is the trap running in reverse.

**Downstream practice confirms the reading:** Rasa is the declared training set for `ai4bharat/IndicF5` (MIT), `SPRINGLab/SPRING_F5` (Apache-2.0) and `SPRINGLab/Indic-Mio` (Apache-2.0). **MEDIUM** on recording provenance — I did not verify whether the recordings were newly commissioned.

---

### 3.9 IITM IndicTTS — the EULA recovered, and it is not what anyone assumed

A sibling researcher could not reach `iitm.ac.in/donlab`. I recovered the licence by a different route: the HF mirror [`thennal/indic_tts_ml`](https://huggingface.co/datasets/thennal/indic_tts_ml) (whose card says *"The license is given in the repository"*) ships a 7-page `license.pdf`. I downloaded and read it.

**It is an "END USER LICENSE AGREEMENT", Licensor: "The Indian Institute of Technology Madras (IITM)".** Operative clauses, verbatim:

**§2.1 — the grant, which is far more permissive than assumed:**

> "The Licensor hereby grants the Licensee a **perpetual, non-exclusive, worldwide, transferable, sub-licensable, royalty-free license** to a) make copies of the Licensed Software in source and object code and data; b) modify copies of the Licensed Software and data to create derivative works thereof. **The Licensee will exclusively own all software, files, documentation, discoveries, ideas, inventions, improvements, processes, materials and data ("Derivative Work") acquired/prepared/generated/developed in any medium by Licensee using the Licensed Software.** To the extent necessary to vest such sole and exclusive ownership in the Licensee, Licensor and/or its personnel hereby irrevocably assign to License(and, as applicable, its successors and assigns) any and all rights in and to such Derivative Work. **Notwithstanding anything to the contrary, Licensee shall be allowed to freely distribute the Derivative Work.**"

**There is no non-commercial clause anywhere in the agreement.** On its face, a model trained on IndicTTS is a "Derivative Work" that the Licensee **exclusively owns** and **may freely distribute**. That is a strikingly favourable reading for train-and-release, and it corrects a widely-repeated assumption.

**§2.2 — the clause that actually blocks an open-weights release:**

> "The Licensee agrees not to remove any copyright, trademark or patent notices that appear in the Licensed Software. The Licensee shall ensure that the third party to whom the Derivative Work is sold is made aware that the Derivative Work has few open source component along with Licensee's IP. **Such third party shall not be allowed to further sell, lease, license, sub-license, decompile, disassemble or reverse engineer any portion of the Licensed Software or the Derivative Work.**"

Read that against an Apache-2.0 weight release. Apache-2.0 §2 grants every recipient the right to "reproduce, prepare Derivative Works of, publicly display, publicly perform, **sublicense**, and distribute". **§2.2 requires you to forbid your recipients precisely what Apache-2.0 grants them.** You cannot both comply with §2.2 and release IndicTTS-derived weights under a permissive licence. This is the sharpest single finding in the Indic track.

**§5 — mandatory notice, which nobody downstream is carrying:**

> "REDISTRIBUTORS MUST RETAIN THE FOLLOWING COPYRIGHT NOTICE:
> *"COPYRIGHT 2016 TTS Consortium, TDIL, Meity represented by Hema A Murthy & S Umesh, DEPARTMENT OF Computer Science and Engineering and Electrical Engineering, IIT Madras. ALL RIGHTS RESERVED"*"

§3: terminable by the Licensee only, on complete destruction of copies. §4: governed by the laws of India.

**On SPRINGLab's re-declaration.** §5 names **S Umesh** as one of two representatives of the copyright — and S. Umesh is the SPRING Lab PI at IIT Madras. That makes SPRINGLab's re-hosting of IndicTTS subsets plausibly *authorised*, which is a meaningful update. But re-hosting is a different act from relicensing, and their own tagging is incoherent: of 14 `SPRINGLab/IndicTTS_*` datasets, **only 3 carry `license:cc-by-4.0`** (Bengali, Tamil, Punjabi); the other 11 carry no licence tag at all. Ad-hoc per-repo labelling, not an institutional relicensing decision.

**Provenance caveat, stated plainly.** This PDF came from a third-party HF mirror, not from IITM directly. It is internally consistent, names the correct people and bodies, and is dated to the 2016 TTS Consortium/TDIL/MeitY programme. **Confidence MEDIUM-HIGH on authenticity; the §2.2 conflict must be confirmed against IITM before shipping anything derived from IndicTTS.**

**Verdicts.** USE ✅. TRAIN ✅. RELEASE TRAINED MODEL ⚠️ **only under a licence that restricts onward sublicensing** — i.e. not Apache-2.0/MIT. REDISTRIBUTE ⚠️ with the §5 notice. **NEEDS LAWYER.**

---

### 3.10 Google Crowdsourced Indic (OpenSLR) — the only ShareAlike in the mix

I fetched all six OpenSLR pages directly. SLR63 (Malayalam), SLR64 (Marathi), SLR65 (Tamil), SLR66 (Telugu), SLR78 (Gujarati), SLR79 (Kannada) each read:

> `License: Attribution-ShareAlike 4.0 International`

**CC BY-SA 4.0 is copyleft for adaptations.** Its §3(b) requires that if you Share Adapted Material, the "Adapter's License" must be the same or a compatible ShareAlike licence.

This matters because the IN-F5 paper (arXiv 2505.20693 §3.1) lists these as training data:

> "we incorporate studio-quality speech from **IndicTTS**, **LIMMITS**, and **Rasa**… we integrate **Google Crowdsourced TTS**, enabling IN-F5 to generalize across different voices… Finally, we leverage **IndicVoices-R**"

So any model trained on the Google Crowdsourced Indic corpora has a ShareAlike question attached, in addition to whatever else. Whether trained weights are "Adapted Material" is the unsettled question in §5 — but note that ShareAlike is a *worse* problem than NC for us, because NC blocks a use while SA compels a licence choice on the artefact we most want to release permissively.

**And Creative Commons' own AI guidance is unhelpfully direct on this point** (§5, verbatim): *"If AI models **or outputs** are based on ShareAlike content and they will be shared publicly, following the ShareAlike condition would require AI developers to use the same CC license as the original works."* Applied literally, a model trained on SLR63–66/78/79 would have to ship under CC-BY-SA-4.0 rather than Apache-2.0 — and arguably so would its generated audio. **That is a licence outcome incompatible with the project's stated posture, and it is triggered by six corpora that are easy to leave out.**

**Verdict.** USE ✅. TRAIN ✅. RELEASE ⚠️ **ShareAlike risk — NEEDS LAWYER; default to excluding.** REDISTRIBUTE ⚠️ must relicense derived data CC-BY-SA-4.0. **HIGH** on the licence; the propagation question is unsettled but CC's published reading is adverse.

---

### 3.11 VoxCeleb2 — blocked three ways, and no longer distributed

Three primary documents make three different claims.

**(a) Oxford VGG** ([vox2.html](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html), "Terms and Conditions"):

> "**The provided VoxCeleb2 metadata is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.**"

Link target is literally `creativecommons.org/licenses/by-sa/4.0/` — **BY-SA, not BY** — and the scope is *"the provided VoxCeleb2 **metadata**"*, not the audio. Then:

> "**Audio files** — The audio files for the VoxCeleb 2 dataset are no longer available from this website."

Same for video and identifying metadata. Wayback shows the withdrawal in place by 15 Jan 2024 and persisting through Nov 2025. **The original distributor has withdrawn everything.**

**(b) KAIST mirror** ([mm.kaist.ac.kr/datasets/voxceleb](https://mm.kaist.ac.kr/datasets/voxceleb/)):

> "**The VoxCeleb dataset is available to download for research purposes under a Creative Commons Attribution 4.0 International License. The copyright remains with the original owners of the video.**"

Note this says CC-BY-4.0 (contradicting Oxford's BY-SA) *and* adds a field-of-use limit that CC-BY does not contain.

**(c) The EULA** ([license.txt](https://mm.kaist.ac.kr/datasets/voxceleb/files/license.txt)):

> "**The copyright of both the original and cropped versions of the videos remains with the original owners.**"
> "**Downloading this dataset implies agreement to follow the same conditions for any modification and/or re-distribution of the dataset in any form.**"

That last sentence is a contractual copyleft-by-agreement: your redistribution must carry the same conditions, *including* "for research purposes".

**(d) The legal basis is a research exemption.** The [VGG Dataset Privacy Notice](https://www.robots.ox.ac.uk/~vgg/terms/url-lists-privacy-notice.html) grounds the collection in UK GDPR Art. 14(5)(b), for processing *"for … scientific or historical research purposes"*, and notes *"it is not possible to notify data subjects that content may have been used"*. **Commercial exploitation sits outside the exemption the collection relies on.** Add to that: this is biometric voice data of thousands of named living individuals.

**Verdicts.** USE impaired. TRAIN-AND-RELEASE ❌. REDISTRIBUTE ❌. **HIGH.**

---

### 3.12 Emilia — the canonical "tag lies" case

**What the metadata says.** [API](https://huggingface.co/api/datasets/amphion/Emilia-Dataset): `cardData.license = "cc-by-4.0"`, tag `license:cc-by-4.0`.

**What the binding terms say.** The repo is `gated: "auto"`; its `extra_gated_prompt`, which I retrieved in full:

> "**Terms of Access:** The researcher has requested permission to use the Emilia dataset, the Emilia-Pipe preprocessing pipeline, and the Emilia-Yodas dataset. In exchange for such permission, the researcher hereby agrees to the following terms and conditions:
> 1. **The researcher shall use the Emilia dataset under the CC-BY-NC license and the Emilia-YODAS dataset under the CC-BY license.**
> 2. The authors make no representations or warranties regarding the datasets, including but not limited to warranties of non-infringement or fitness for a particular purpose.
> 3. The researcher accepts full responsibility for their use of the datasets and shall defend and indemnify the authors … against any and all claims arising from the researcher's use of the datasets, **including but not limited to the researcher's use of any copies of copyrighted content that they may create from the datasets.**
> 4. The researcher may provide research associates and colleagues with access to the datasets, provided that they first agree to be bound by these terms and conditions.
> 5. **The authors reserve the right to terminate the researcher's access to the datasets at any time.**
> 6. **If the researcher is employed by a for-profit, commercial entity, the researcher's employer shall also be bound by these terms and conditions**, and the researcher hereby represents that they are fully authorized to enter into this agreement on behalf of such employer."

Required gate fields include *"Your Supervisor/manager/director"* — an academic-access posture.

**Corroborated at source.** [Amphion `preprocessors/Emilia/README.md`](https://raw.githubusercontent.com/open-mmlab/Amphion/main/preprocessors/Emilia/README.md):

> "*Please note that Emilia does not own the copyright to the audio files; the copyright remains with the original owners of the videos or audio. **Users are permitted to use this dataset only for non-commercial purposes under the CC BY-NC-4.0 license.***"

**Why the tag says CC-BY.** The Feb 2025 release note explains: Emilia-Large = the original 101k-hour **Emilia** (`CC BY-NC 4.0`) + the new 114k-hour **Emilia-YODAS** (`CC BY 4.0`). **Two datasets, two licences, one `cc-by-4.0` tag.** Only the YODAS half is permissive.

**Why this is the highest-leverage finding in the model audit.** Emilia is declared training data for `SWivid/F5-TTS`, `Aratako/MioCodec-25Hz-24kHz`, `Aratako/MioTTS-0.6B` and `k2-fsa/ZipVoice` — which means it reaches IndicF5, SPRING_F5, Indic-Mio and DhVaani. **A single mislabelled tag is upstream of most of the Indic track.**

**Verdicts.** USE gated. TRAIN-AND-RELEASE ❌ (Emilia half). REDISTRIBUTE ❌. **HIGH.**

---

### 3.13 VoicePersona — the CC0 declaration does not hold

This is our own dataset. An honest finding matters more than a comfortable one, so here it is in full.

**The declaration is already internally inconsistent.** [HF API](https://huggingface.co/api/datasets/Paranoiid/VoicePersona) → `cardData.license = "cc"`. HuggingFace's `cc` is a **generic category tag, not a licence** — it identifies no specific instrument and grants nothing. The README carries a badge linking CC0 1.0, and the scope document treats the dataset as CC0. **The machine-readable field and the human-readable badge disagree, and the machine-readable field is the one tooling reads.**

**The dataset redistributes the audio, not just the captions.** `dataset_info` declares an `audio` feature at 16 kHz across ~5.27 GB / 15,082 rows. This is not an annotations-only release like ParaSpeechCaps, so upstream audio rights flow through directly and unavoidably.

**The four sources, by sample count:**

| Source | Samples | % | Declared | Actual status |
|---|---:|---:|---|---|
| `laion/laions_got_talent` | 7,937 | **52.6%** | **no HF licence field**; an Apache-2.0 `LICENSE` file in the repo | OpenAI GPT-4o Audio output, OpenAI's proprietary voices |
| `MushanW/GLOBE_V2` | 3,146 | 20.9% | `cc0-1.0`, `source_datasets: mozilla-foundation/common_voice_14_0` | **Clean** |
| `ShoukanLabs/AniSpeech` | 2,000 | 13.3% | `mit` | MIT text applied to anime voice recordings |
| `taresh18/AnimeVox` | 1,999 | 13.3% | `cc` (meaningless) | Ripped from official anime dubs |

**(a) LAION's Got Talent — the largest source, and the most serious problem.** The [card](https://huggingface.co/datasets/laion/laions_got_talent/raw/main/README.md) states its construction:

> "The dataset was constructed wiht a diverse menu of prompts **the OpenAI Voice API** via Hyprlab (https://docs.hyprlab.io/browse-models/model-list/openai/chat#gpt-4o-audio-models)."

> "( currently 110 hours, will grow soon )"

I enumerated the repo's 494 files. **483 tarballs are named after OpenAI's eleven proprietary voice identities**, with these counts: `alloy` 58, `ash` 55, `sage` 56, `echo` 43, `ballad` 40, `coral` 40, `fable` 40, `nova` 40, `onyx` 40, `verse` 40, `shimmer` 31. Filenames such as `english_alloy_intense_anger_rage_fury_hatred_and_annoyance.tar` make the provenance unambiguous.

Three distinct problems stack here:

1. **Contract.** OpenAI's Terms of Use restrict using Output to develop models that compete with OpenAI. *(I could not obtain the verbatim clause — openai.com returns HTTP 403 to every automated fetch I attempted, including via the Wayback Machine. The restriction's existence is confirmed by search-surfaced excerpts of OpenAI's Terms of Use, May 2025 Business Terms and Services Agreement, but I am not quoting text I could not fetch. **MEDIUM confidence; NEEDS LAWYER to read the current operative text.**)* A description→voice TTS product is squarely a competing model.
2. **Access route.** The audio was obtained "via Hyprlab", a third-party API reseller — so the terms under which it was generated may not even be OpenAI's own customer terms.
3. **The Apache-2.0 LICENSE file.** LAION placed the Apache 2.0 text in the repo (I fetched and confirmed it) while the `cardData` carries no licence field at all. LAION cannot grant Apache-2.0 rights over OpenAI's model output. And Apache-2.0 is not CC0, so **even taking LAION's grant at face value, VoicePersona's CC0 claim over this 52.6% is wrong.**

**(b) AnimeVox** — the [card](https://huggingface.co/datasets/taresh18/AnimeVox/raw/main/README.md) is explicit:

> "### Source
> Audio clips were sourced from **official English-dubbed versions of popular anime series**. The clips were selected to capture diverse emotional tones and vocal characteristics unique to each character."

11,020 clips, 19 named characters, 15 anime series, with `character_name` and `anime` as dataset fields. The provenance is not merely inferable, it is *labelled*. This is copyrighted audiovisual work, and separately it is the recorded performances of identifiable working voice actors — a personality-rights exposure distinct from copyright (§6). Declared `license: cc`.

**(c) AniSpeech** — declares MIT and links a `license` file. I fetched it. It is the **literal MIT software licence text**, beginning:

> "Copyright 2023 ShoukanLabs
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software")…"

ShoukanLabs is asserting copyright over 18.8 GB of "captioned anime voices" and licensing it as *software*. The card contains **no provenance statement whatsoever** about where the audio came from. The absence is itself the finding.

**(d) GLOBE_V2** — `license: cc0-1.0`, `source_datasets: [mozilla-foundation/common_voice_14_0]`, derived by filtering and enhancement. Common Voice is CC0. **This one is clean**, subject only to carrying Common Voice's no-re-identification obligation forward.

**⭐ This exact failure mode already produced a takedown, three months ago, and Hugging Face acted on it.** `ESpeech/ESpeech-igm` was a speech dataset built from audio extracted from YouTube videos and republished under **Apache-2.0**. A professional voice actor complained; the notice is published verbatim in HF's public takedown log (`huggingface-legal/takedown-notices`, `2026/2026-05-26-IGM.md`):

> "The dataset appears to have been created by a third-party user from **audio extracted from IGM YouTube videos**. These videos contain **my professional voice-over work**."
>
> "**The dataset page also lists the dataset under the Apache-2.0 license. This license was not authorized by me and cannot lawfully apply to my voice recordings, vocal performance, or the underlying audio/video content.**"

The HF API confirms the outcome: `{'id': 'ESpeech/ESpeech-igm', 'disabled': True, 'tags': [… 'license:apache-2.0' …]}` — **the repo is disabled and still carries the Apache-2.0 tag it was never entitled to.** The hook was the **performer's** rights, not copyright in the underlying video.

Read that against AnimeVox (ripped official anime dubs, 19 named characters) and AniSpeech (18.8 GB of "anime voices" under an MIT text): **this is the same fact pattern, with the same defect, from a complainant class that has already demonstrated it will file.** *(The log also records a May 2026 Maluma / Vermillio notice against `chilombo/rvcmodels` asserting rights over voice-clone **model weights** — the encumbrance-reaches-the-model theory is already being asserted in practice.)*

**Why "but we declared CC0" does not help.** CC0 1.0 §4(c), verbatim:

> "Affirmer disclaims responsibility for clearing rights of other persons that may apply to the Work or any use thereof, including without limitation any person's Copyright and Related Rights in the Work. Further, Affirmer disclaims responsibility for obtaining any necessary consents, permissions or other rights required for any use of the Work."

CC0 is a **waiver of the affirmer's own rights**. It cannot waive OpenAI's contract rights, a Japanese animation studio's copyright, or a voice actor's personality rights, and it expressly says so. *Nemo dat quod non habet.* A CC0 stamp on content you did not clear does not launder it; it just makes the downstream user's reliance unreasonable in hindsight.

**Verdicts for VoicePersona as it stands today.** USE ⚠️ (research, at risk). TRAIN ⚠️. **RELEASE A MODEL TRAINED ON IT ❌.** **REDISTRIBUTE ❌.** **HIGH.**

**What to do about it** — see §8. The short version: **VoicePersona v2 must be rebuilt from GLOBE_V2 plus the LibriVox chain plus the CC-BY-4.0 Indic corpora, and VoicePersona v1 should be re-tagged from `cc` to reflect reality rather than left standing as a CC0 claim.**

---

## 4. Model licence audit (serve publicly)

| Model | Code lic. | Weights lic. | Upstream chain | Output clause? | AUP rider? | PUBLICLY SERVABLE | Conf. |
|---|---|---|---|---|---|---|---|
| **VoxCPM2** | Apache-2.0 | Apache-2.0 | MiniCPM-4 (Apache-2.0); 2M+ hr **undisclosed** | no | advisory only | ✅ **YES** | HIGH |
| **Qwen3-TTS** family (Base / CustomVoice / **VoiceDesign** / Tokenizer, 0.6B & 1.7B) | Apache-2.0 | Apache-2.0 | self-contained; own 12Hz tokenizer, also Apache-2.0 | no | **none** | ✅ **YES — cleanest in the field** | HIGH |
| **Chatterbox** | MIT | MIT | Llama *architecture* only (verified in code, not Meta weights); CosyVoice (Apache-2.0), HiFT-GAN (MIT), Perth (MIT) | no | "Don't use this model to do bad things." | ✅ **YES** | HIGH |
| **Parler-TTS** | Apache-2.0 | Apache-2.0 | LibriTTS-R + MLS-Eng, both CC-BY-4.0; DAC codec MIT | no | none | ✅ **YES** | HIGH |
| **CosyVoice 2 / 3** | Apache-2.0 | Apache-2.0 (YAML only, no LICENSE file) | Qwen2.5-0.5B (Apache-2.0), HiFT-GAN (MIT); ⚠️ optional proprietary `ttsfrd` wheel — **do not install it** | no | none | ✅ **YES** | MEDIUM-HIGH |
| **Zonos-v0.1** (transformer + hybrid) | Apache-2.0 | Apache-2.0 *(declared)* | **speaker encoder ← VoxBlink2 (CC-BY-NC-SA-4.0)**; DAC codec MIT; eSpeak-NG GPL-3.0 | no | none | ⚠️ **NEEDS LAWYER → effectively NO for the cloning path** | HIGH facts |
| **VibeVoice** | MIT *(but the TTS code was removed from the repo)* | MIT *(1.5B; **Large withdrawn, HTTP 401**)* | Qwen2.5-1.5B / 0.5B (Apache-2.0) | **YES — a mandatory *audible* AI disclaimer is embedded in every output; Realtime variant bars circumventing it** | "limited to research purpose use"; "not intended or licensed for…" | ⚠️ **NEEDS LAWYER → practically NO** | MEDIUM |
| **Indic Parler-TTS** | Apache-2.0 | Apache-2.0 | GLOBE (CC0) + **IndicTTS** + LIMMITS + Rasa | no | none | ⚠️ **NEEDS LAWYER** (IndicTTS §2.2) | MEDIUM |
| **ai4bharat/indic-conformer-600m-multilingual** (ASR) | — | MIT | undeclared | no | none | ⚠️ **NEEDS LAWYER** (no declared provenance) | LOW |
| **IndicF5** | **`license: null`** (GitHub) | MIT *(declared)* | **F5-TTS (CC-BY-NC-4.0)** ← decisive | no | none | ❌ **NO** | HIGH |
| **SPRINGLab/SPRING_F5** | — | Apache-2.0 *(declared)* | **F5-TTS (CC-BY-NC-4.0)**, declared in its own YAML | no | none | ❌ **NO** | HIGH |
| **SPRINGLab/Indic-Mio** | — | Apache-2.0 *(declared)* | MioTTS-0.6B ← Emilia (NC); MioCodec ← Emilia (NC); **direct: Expresso (CC-BY-NC-4.0)**, IndicTTS, SYSPIN, SPICOR | no | none | ❌ **NO** | HIGH |
| **Aratako/MioCodec-25Hz-24kHz** | MIT | MIT *(declared)* | **Emilia (NC)**; Kanade-Tokenizer (**unlicensed**); WavLM-base+ | no | none | ❌ **NO** | HIGH |
| **ARTPARK-IISc/DhVaani-0.5** | — | Apache-2.0 *(declared)* | **ZipVoice** (HF weights carry **no licence field**; code Apache-2.0) ← Emilia (NC); + IndicTTS, SYSPIN, Rasa | no | *"Please also respect the licenses of the training corpora"* | ❌ **NO** | MEDIUM-HIGH |
| **F5-TTS** | MIT | **CC-BY-NC-4.0** | Emilia (NC) | no | none | ❌ **NO** | HIGH |
| **VoiceSculptor-VD** | Apache-2.0 | Apache-2.0 *(declared)* | **Llasa-3B (CC-BY-NC-4.0)** → Llama-3.2-3B-Instruct (Llama 3.2 Community Licence); requires **xcodec2 (CC-BY-NC-4.0)** | no | misuse disclaimer | ❌ **NO** | HIGH |
| **Llasa-3B** | NOASSERTION | **CC-BY-NC-4.0** | Llama-3.2-3B-Instruct | no | "strictly prohibited … for any illegal purposes" | ❌ **NO** | HIGH |
| **xcodec2** / `xcodec2-hf` | MIT (X-Codec-2.0 repo) | **CC-BY-NC-4.0** | — | no | none | ❌ **NO** | HIGH |
| **XTTS-v2** | MPL-2.0 (TTS repo) | **CPML** (`license_name: coqui-public-model-license`; `license_link` is a **dead URL**) | — | **YES — "only non-commercial use of a machine learning model *and its outputs*"** | Notices clause forces CPML terms onto every recipient of the output | ❌ **NO — and unfixable; no licensor exists** | HIGH |
| **IndexTTS-2** | **bilibili Model Use License Agreement** (added 2026-01-20; GitHub reports `NOASSERTION`) | same | **`amphion/MaskGCT` (CC-BY-NC-4.0) downloaded at runtime as a mandatory inference component**; Qwen3-0.6B emotion module (Apache-2.0); w2v-BERT (MIT); BigVGAN (MIT) | **YES — §1.5 defines model outputs as Derivative Work; §3.4(c) bars using them to improve any commercial AI model** | §4.1–4.3 compliance + high-risk bans; DISCLAIMER file bars unauthorised commercial use of synthesized voices | ❌ **NO** — three independent blockers | HIGH |

---

### 4.1 Paradigm case A — VoiceSculptor → Llasa-3B → Llama 3.2, plus xcodec2

The full chain, each link from its own primary source.

**Link 1 — VoiceSculptor declares Apache-2.0 and names its own defect.** [`ASLP-lab/VoiceSculptor-VD` raw README](https://huggingface.co/ASLP-lab/VoiceSculptor-VD/raw/main/README.md) frontmatter:

```yaml
license: apache-2.0
base_model:
- HKUSTAudio/Llasa-3B
pipeline_tag: text-to-speech
```

Card body, verbatim:

> "We use the Apache 2.0 license. Researchers and developers are free to use the codes and model weights of our VoiceSculptor."

`gh api repos/ASLP-lab/VoiceSculptor` → `Apache-2.0`.

**Link 2 — Llasa-3B is CC-BY-NC-4.0 and says so in terms.** [`HKUSTAudio/Llasa-3B` README](https://huggingface.co/HKUSTAudio/Llasa-3B/raw/main/README.md), `license: cc-by-nc-4.0`, and the Disclaimer, verbatim:

> "This model is licensed under the CC BY-NC 4.0 License, **which prohibits free commercial use because of ethics and privacy concerns; detected violations will result in legal consequences.**"

> "This codebase is strictly prohibited from being used for any illegal purposes in any country or region. Please refer to your local laws about DMCA and other related laws."

**Link 3 — the layer the brief missed.** Llasa-3B's own `base_model` is `meta-llama/Llama-3.2-3B-Instruct`, whose licence is `llama3.2` (the **Llama 3.2 Community License**, `gated: manual`). That adds a *third* set of obligations Apache-2.0 does not carry: the "Built with Llama" attribution requirement, the derivative-model naming requirement, the 700M-MAU threshold, and Meta's Acceptable Use Policy. **A downstream Apache-2.0 stamp erases all three.**

**Link 4 — the codec.** Llasa/VoiceSculptor generation requires XCodec2. [`HKUSTAudio/xcodec2`](https://huggingface.co/HKUSTAudio/xcodec2/raw/main/README.md): `license: cc-by-nc-4.0`. The 2026-06-25 Transformers-native re-release [`HKUSTAudio/xcodec2-hf`](https://huggingface.co/api/models/HKUSTAudio/xcodec2-hf) is **also `cc-by-nc-4.0`** — checked for relicensing, none occurred. The *code* repo `zhenye234/X-Codec-2.0` is MIT; the *weights* are NC. Note the shape: you cannot produce audio at all without loading NC weights, so the NC term is unavoidable at inference time, not merely at training time.

**Verdict: `public_servable = False`.** The Apache-2.0 grant is void to the extent it exceeds upstream rights, and here it exceeds them at two independent points (Llasa weights, xcodec2 weights) plus a third with different obligations (Llama 3.2). **HIGH.**

---

### 4.2 Paradigm case B — IndicF5 = IN-F5 = a fine-tune of English F5-TTS

The brief called this "a decisive finding". It is. Here is every link, verified independently.

**(a) The declaration.** [`https://huggingface.co/api/models/ai4bharat/IndicF5`](https://huggingface.co/api/models/ai4bharat/IndicF5):

```json
"cardData": { "license": "mit",
              "datasets": ["ai4bharat/indicvoices_r", "ai4bharat/Rasa"] }
"lastModified": "2026-03-03T03:06:36.000Z"
```

Note there is **no `base_model` field** — the fine-tune relationship is simply not declared.

**(b) The written commercial-use assurance.** [Discussion #34, "Commercial Use Inquiry – IndicF5 License Clarification"](https://huggingface.co/ai4bharat/IndicF5/discussions/34), opened 2026-03-02 by `AbhishekDelMundu`, answered 2026-03-03T03:06:12Z by `safikhan` (Mohammed Safi Ur Rahman Khan, AI4Bharat), verbatim in full:

> "Hi
>
> The model is released in MIT license. So its completely open for commercial usage."

The model's `lastModified` is `2026-03-03T03:06:36Z` — **24 seconds after that reply**. The MIT tag and the assurance were the same act.

**(c) The paper says IndicF5 is a fine-tune of English F5-TTS.** arXiv 2505.20693, *"Phir Hera Fairy: An English Fairytaler is a Strong Faker of Fluent Speech in Low-Resource Indian Languages"* (Praveen Srinivasa Varadhan, Srija Anand, Soma Siddhartha, Mitesh M. Khapra — IIT Madras / AI4Bharat). Abstract, verbatim:

> "We evaluate how **the English F5-TTS model** adapts to 11 Indian languages… We compare: (i) training from scratch, (ii) **fine-tuning English F5 on Indian data**, and (iii) fine-tuning on both Indian and English data to prevent forgetting. **Fine-tuning with only Indian data proves most effective and the resultant IN-F5** … is a near-human polyglot."

**(d) The paper links IN-F5 to the IndicF5 repo by footnote.** From the arXiv HTML (v1), the abstract's footnote 2 on "IN-F5" resolves to:

> `IN-F5 ² https://huggingface.co/ai4bharat/IndicF5`

Footnote 1 on "English Fairytaler" reads: *"Reference to English F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech With Flow Matching"*. **IN-F5 and IndicF5 are the same artefact, stated by the authors.**

**(e) Corroborated by the repo's own file listing.** The HF repo's siblings include the entire F5-TTS source tree — `f5_tts/model/`, `f5_tts/infer/`, `f5_tts/eval/eval_librispeech_test_clean.py`, `f5_tts/configs/F5TTS_Base_train.yaml`, and the F5-TTS example audio `basic_ref_en.wav` / `basic_ref_zh.wav`. It is F5-TTS.

**(f) F5-TTS weights are CC-BY-NC-4.0.** [`https://huggingface.co/api/models/SWivid/F5-TTS`](https://huggingface.co/api/models/SWivid/F5-TTS): `cardData.license = "cc-by-nc-4.0"`, `cardData.datasets = ["amphion/Emilia-Dataset"]`, 777,802 downloads. The **code** at `gh api repos/SWivid/F5-TTS` is **MIT** — the split matters and is easy to conflate.

**(g) The AI4Bharat GitHub repo is unlicensed.** `gh api repos/AI4Bharat/IndicF5 --jq .license` → **`null`**, created 2025-03-11, last pushed 2025-09-24. **No licence = all rights reserved** for the code in that repo. The brief's claim is confirmed exactly.

**(h) New as of this audit: the HF repo is now gated.** `"gated": "auto"` — fetching `raw/main/README.md` returns *"Access to model ai4bharat/IndicF5 is restricted. You must have access to it and be authenticated to access it."* 27,389 downloads, 189 likes. This is a change worth recording; it does not alter the licence analysis.

**(i) And the training data adds a ShareAlike question.** Per §3.1 of the paper, the IN11 mix (1,417 hr) is IndicTTS + LIMMITS + Rasa + **Google Crowdsourced TTS (CC BY-SA 4.0)** + IndicVoices-R. See §3.10.

**Verdict: `public_servable = False`.** The MIT grant on IndicF5 exceeds upstream rights in the F5-TTS weights. A licensor's sincere assurance about their own grant is not a warranty of upstream title, and AI4Bharat is not in a position to relicense SWivid's checkpoint. **HIGH on the facts. NEEDS LAWYER on whether AI4Bharat obtained a separate permission not visible in any public artefact — that is the one thing that would change the answer, and it is worth one email.**

---

### 4.3 Paradigm case C — SPRINGLab/SPRING_F5

The shortest case in the file, because the defect is in the same YAML block as the grant. [`SPRINGLab/SPRING_F5` raw README](https://huggingface.co/SPRINGLab/SPRING_F5/raw/main/README.md):

```yaml
license: apache-2.0
base_model:
- SWivid/F5-TTS
datasets:
- ai4bharat/IndicVoices
- ai4bharat/Rasa
```

Card body: *"**SPRING_F5** is a multilingual text-to-speech (TTS) model **based on F5-TTS**, fine-tuned to support 23 Indian Language & English."* Last modified 2026-08-17 — recent, and still declared this way.

Apache-2.0 over CC-BY-NC-4.0 weights. **`public_servable = False`. HIGH.**

---

### 4.4 The Indic-Mio / MioCodec / MioTTS chain — CRITICAL FINDING

A sibling researcher provisionally put this chain on the Indic critical path on the strength of its Apache-2.0/MIT declarations. **Traced properly, it has the defect.** The mechanism is subtler than IndicF5's — the base *weights* really are Apache-2.0 — but the defect is real and it appears at three independent points.

**Layer 1 — `SPRINGLab/Indic-Mio`, Apache-2.0.** [Raw README](https://huggingface.co/SPRINGLab/Indic-Mio/raw/main/README.md): `license: apache-2.0`, `base_model: Aratako/MioTTS-0.6B`, `datasets: [ai4bharat/Rasa, mythicinfinity/libritts_r, ylacombe/expresso]`. Training section, verbatim:

> "For Indian languages, **IndicTTS, Rasa and Syspin** datasets were used. For American English, **LibriTTS and Expresso**, while for Indian English, **SPICOR** dataset was used."

**`ylacombe/expresso` is `license: cc-by-nc-4.0`** — confirmed from the [raw dataset card](https://huggingface.co/datasets/ylacombe/expresso/raw/main/README.md) frontmatter, and independently from Meta's own project page (*"The Expresso dataset is distributed under the CC BY-NC 4.0 license."*). **Indic-Mio declares an NC corpus as a direct training input in its own YAML frontmatter, and declares Apache-2.0 in the line below it.** Plus IndicTTS (§3.9's §2.2 conflict), SYSPIN and SPICOR (licences UNVERIFIED).

**Layer 2 — `Aratako/MioTTS-0.6B`, Apache-2.0.** [Raw README](https://huggingface.co/Aratako/MioTTS-0.6B/raw/main/README.md): `license: apache-2.0`, `base_model: [Qwen/Qwen3-0.6B-Base]` (Apache-2.0 — clean), `datasets: [nvidia/hifitts-2, amphion/Emilia-Dataset]`, ~100k hours. **Emilia's binding terms say CC-BY-NC** (§3.12). Licence section: *"This model is released under the Apache 2.0."* The ethical section is advisory only, not a licence condition.

*Sizing note for the adapter table:* the MioTTS family is **not uniformly Apache-2.0**. The card's own table shows MioTTS-0.1B under the **Falcon-LLM License**, and 0.4B / 1.2B / 2.6B under the **LFM Open License v1.0**. Only the 0.6B and 1.7B (Qwen3-based) are Apache-2.0. Indic-Mio uses 0.6B, so the base is fine — but do not generalise across the family.

**Layer 3 — `Aratako/MioCodec-25Hz-24kHz`, MIT.** [Raw README](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz/raw/main/README.md): `license: mit`, `datasets: [sarulab-speech/mls_sidon, mythicinfinity/Libriheavy-HQ, nvidia/hifitts-2, amphion/Emilia-Dataset]`. **Emilia again.** Two further provenance issues: it is *"Based on the [Kanade-Tokenizer](https://github.com/frothywater/kanade-tokenizer) implementation"*, and `gh api repos/frothywater/kanade-tokenizer --jq .license` → **`null`** (unlicensed code). Its SSL encoder is WavLM-base+, whose HF `cardData.license` is also **null**. `gh api repos/Aratako/MioCodec` → MIT for MioCodec's own code.

**This layer is unavoidable at inference.** Indic-Mio's own quickstart does `MioCodec.from_pretrained("Aratako/MioCodec-25Hz-24kHz")` to decode tokens to waveform. **You cannot serve Indic-Mio without loading MioCodec**, exactly as you cannot serve Llasa without xcodec2.

**Verdict: `public_servable = False`.** Three independent NC/unlicensed exposures — Expresso directly, Emilia twice transitively, and an unlicensed codec implementation — under a stack of Apache-2.0/MIT declarations. **HIGH on the facts.** The legal consequence turns on §5, which is unsettled, but Expresso as a *direct, declared* training input of the released model is the least deniable of the three. **NEEDS LAWYER before any reinstatement.**

---

### 4.5 DhVaani-0.5 — Apache-2.0 with a disclaimer that gives the game away

[`ARTPARK-IISc/DhVaani-0.5`](https://huggingface.co/ARTPARK-IISc/DhVaani-0.5), `gated: auto`, `license: apache-2.0`, `base_model: k2-fsa/ZipVoice`, 27 Indian languages. Card, verbatim:

> "Apache-2.0, following the base model. Built on [ZipVoice](https://github.com/k2-fsa/ZipVoice) (k2-fsa). **Please also respect the licenses of the training corpora (IndicTTS, Rasa, IISc SYSPIN).**"

> "Trained pooled from multiple TTS datasets like **IISc SYSPIN, IndicTTS (SPRINGLab) and Rasa** at 16 kHz."

Two things. First, *"Apache-2.0, following the base model"* — but `k2-fsa/ZipVoice` on HF has **no licence field at all** (`cardData.license = null`; the tag list contains no `license:*` entry). The GitHub repo `k2-fsa/ZipVoice` *is* Apache-2.0, so the code is fine, but the **weights carry no declaration**, and those weights declare `amphion/Emilia-Dataset` as training data. So "following the base model" follows a base model that declared nothing.

Second, *"Please also respect the licenses of the training corpora"* is a disclaimer that **pushes the obligation downstream while keeping the permissive tag**. It is honest, and it is also an admission that the Apache-2.0 stamp does not resolve the question. IndicTTS carries the §2.2 conflict (§3.9). SYSPIN's licence is **UNVERIFIED**.

**Verdict: `public_servable = False` pending counsel.** **MEDIUM-HIGH.**

---

### 4.6 ai4bharat/indic-conformer-600m-multilingual

`license: mit`, `gated: auto`, 94,446 downloads, ASR (not TTS — relevant to us as an eval/WER component, not a renderer). **`cardData` declares no `base_model` and no `datasets`.** For an AI4Bharat ASR model the training mix is almost certainly IndicVoices/Kathbath/Shrutilipi (all CC-BY-4.0), which would be fine — but that is inference, not evidence.

**Verdict: ⚠️ NEEDS LAWYER / needs the authors asked.** **LOW confidence** — the MIT tag is real, the provenance is simply undeclared. If used only offline in the eval harness (never served), the exposure is small.

---

### 4.7 Indic Parler-TTS — best practice in the field, one unresolved dependency

This is the model whose card behaves the way every card should. [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts) (`license: apache-2.0`, `gated: auto`), licence statement verbatim:

> "This model is permissively licensed under the Apache 2.0 license."

And — uniquely in this entire audit — **it ships a per-dataset licence table**:

> | Dataset | Duration (hrs) | Languages Covered | No. of Utterances | License |
> |---|---|---|---|---|
> | GLOBE | 535.0 | 1 | 581,725 | CC V1 |
> | IndicTTS | 382.0 | 12 | 220,606 | CC BY 4.0 |
> | LIMMITS | 568.0 | 7 | 246,008 | CC BY 4.0 |
> | Rasa | 288.0 | 9 | 155,734 | CC BY 4.0 |

~1,806 hours, no NC corpus anywhere in the mix. **That is what a clean chain looks like when someone actually does the work.**

**The one problem is the IndicTTS row.** AI4Bharat asserts CC-BY-4.0 over IIT Madras's corpus. The actual IITM EULA I recovered (§3.9) is *not* CC-BY-4.0 — it is a bespoke agreement with a §2.2 no-onward-sublicensing term and a §5 mandatory notice. **If §2.2 binds, then Indic Parler-TTS's Apache-2.0 weight release is itself non-compliant**, and building on it inherits the problem.

**Two further facts, both new as of this audit.** (1) **The repo is now gated** (`gated: "auto"`, `lastModified: 2025-09-24`); `raw/main/README.md` returns *"Access to model ai4bharat/indic-parler-tts is restricted."* The base model `ai4bharat/indic-parler-tts-pretrained` is gated too. **A gate is a click-through agreement whose terms are not visible in public metadata and may restrict hosting independently of the Apache-2.0 field.** You cannot responsibly host weights whose gate terms you have not read. (2) The card's `"CC V1"` entry for GLOBE is **not a real licence identifier**; upstream `MushanW/GLOBE` and `GLOBE_V2` are `cc0-1.0`, so it most likely means CC0 1.0 — but that is inference, not evidence. Separately, `ai4b-hf/GLOBE-annotated` (the only dataset in the YAML) declares `license: None`. **The base chain is clean**: fine-tune of `ai4bharat/indic-parler-tts-pretrained`, itself a *"multilingual Indic extension of Parler-TTS Mini"* (`parler-tts/parler-tts-mini-v1.1`, Apache-2.0). Note IndicVoices-R is **not** in the training table.

**Verdict: ⚠️ CONDITIONAL — the strongest Indic candidate, and the one worth spending a lawyer hour on.** Two cheap actions settle it: **log in and capture the gate agreement text**, and get IITM to confirm the IndicTTS CC-BY-4.0 re-designation. If both clear, this flips to `public_servable = True` and the Indic track is solved. **MEDIUM.**

*(Note: the model is the RASMALAI paper's `IndicParlerTTS` — arXiv 2505.18609 abstract: "Using RASMALAI, we develop IndicParlerTTS, the first open-source, text-description-guided TTS for Indian languages." The **RASMALAI dataset itself is not released**: it is absent from all 76 datasets in the `ai4bharat` HF namespace and from HF dataset search entirely. The brief's suspicion is confirmed. **MEDIUM.** The card's 1,806-hour table, not the paper's 13,000 hours, is what the released checkpoint documents.)*

---

### 4.8 Parler-TTS — the real precedent for what an open TTS release actually ships

The brief asks: in practice, what does an openly-released TTS model do about its data's terms? Parler-TTS is the cleanest case study, because it is the most deliberately "fully open" release in the field.

- **Weights:** `parler-tts/parler-tts-large-v1`, `license: apache-2.0`. Card: *"This model is permissively licensed under the Apache 2.0 license."*
- **Code:** `gh api repos/huggingface/parler-tts` → `Apache-2.0`.
- **Data:** `parler-tts/mls_eng` (CC-BY-4.0), `parler-tts/libritts_r_filtered` (CC-BY-4.0), `parler-tts/mls-eng-speaker-descriptions` (CC-BY-4.0). **Every training corpus is CC-BY-4.0. No NC anywhere.** 45K hours.
- **Card claim:** *"Parler-TTS is a fully open-source release"* with *"datasets, pre-processing, training code and weights released publicly under permissive license."*

**What it actually ships for attribution:** I enumerated the weights repo's files. There is **no `NOTICE`, no `ATTRIBUTION.md`, no `LICENSE` file** — only `README.md`, config, tokenizer and safetensors. Attribution is handled entirely by (a) the linked `datasets:` metadata and (b) two BibTeX citation requests in the card body (Lacombe/Srivastav/Gandhi 2024; Lyth & King, arXiv 2402.01912).

**The lesson, stated honestly.** The field's practice for CC-BY training data is: *pick only CC-BY/CC0 corpora, name them in the model card's metadata and prose, request citation, and ship no separate NOTICE file.* Whether that satisfies CC-BY-4.0 §3(a) is arguable (§5), but it is **the observed norm even among the most licence-conscious releases** — and note that HuggingFace's own flagship, trained entirely on CC-BY-4.0 data, provided no attribution to those corpora at all. **Parler-TTS's conduct is therefore not a defence we can lean on.**

**The best precedent in the field is `hexgrad/Kokoro-82M`** (Apache-2.0 weights), which ships a dedicated model-card section:

> "### Creative Commons Attribution
> The following CC BY audio was part of the dataset used to train Kokoro v1.0."
>
> | Audio Data | Duration Used | License | Added to Training Set After |
> |---|---|---|---|
> | Koniwa `tnc` | <1h | CC BY 3.0 | v0.19 / 22 Nov 2024 |
> | SIWIS | <11h | CC BY 4.0 | v0.19 / 22 Nov 2024 |

plus a provenance statement: *"Kokoro was trained exclusively on **permissive/non-copyrighted audio data** and IPA phoneme labels… **No synthetic audio from open TTS models or 'custom voice clones'**"*.

**That is the pattern to copy**: per-source table, licence links, duration, *and a version marker for when each source entered the training set* — the last of which makes the claim auditable across releases, and is exactly the discipline scope §12.1 already applies to `backend_version`. It satisfies §3(a)(1)(A)(i), (iii), (v) and (C) at corpus granularity; add a `NOTICE` carrying copyright notices and the warranty disclaimer to close (ii) and (iv).

**A second pattern worth stealing — and a trap inside it.** `rhasspy/piper-voices` ships a `MODEL_CARD` file **colocated with every individual voice checkpoint**, each naming the dataset URL, its licence, and what was fine-tuned from what. That is the right granularity for a *multi-voice catalogue*, which is exactly what VoiceForge is. **The trap: the repo root declares `license: mit` while individual voices are trained on CC-BY-NC-SA-4.0 data, visible only in the per-voice file.** Anyone pulling "Piper voices, MIT" wholesale ships NC-derived weights. **Per-artefact attribution is only protective if the top-level declaration does not contradict it** — so if our catalogue ever mixes licence classes, the catalogue-level licence must be the most restrictive one present, not the most convenient. *(Aside: Piper's Hindi voices point at the IIT Madras licence PDF at `iitm.ac.in/donlab/indictts/downloads/license.pdf`, which no longer resolves. **That is the document recovered in §3.9** — worth sending back to them.)*

**And a sobering data point on how far per-work attribution scales.** MLCommons People's Speech is the only corpus found that attempts it: a `credits.csv` of **16,769 rows across 474 distinct creators**, built because *"To comply with the CC-BY and CC-BY-SA licenses, we must attribute the original creators of the work."* In practice it lives at an unversioned bucket path, is not linked from the dataset card, has an **empty `credits` column on essentially every row**, and ~57% of its rows are **ShareAlike**. **Corpus-level attribution done well beats per-work attribution done badly** — which is also what CC's own guidance asks for.

**Combine Kokoro's table format, Piper's per-artefact colocation, and Indic Parler-TTS's per-dataset licence column, and we are comfortably ahead of the field.**

---

### 4.9 XTTS-v2 — the CPML restricts the *output*, and no one can sell you a way out

[`coqui/XTTS-v2`](https://huggingface.co/api/models/coqui/XTTS-v2): `license: other`, `license_name: "coqui-public-model-license"`, `license_link: "https://coqui.ai/cpml"`. **That link is now HTTP 404** — Coqui Inc. dissolved in January 2024 and the domain serves a GitHub Pages "Site not found" shell. The only surviving primary text is the in-repo copy at [`LICENSE.txt`](https://huggingface.co/coqui/XTTS-v2/raw/main/LICENSE.txt) (4,014 bytes). The `coqui-ai/TTS` code repo is MPL-2.0 — a separate artefact under separate terms, and the code is genuinely reusable.

**CPML 1.0.0, the operative clauses, verbatim.** The very first substantive line answers the output question the brief asked about:

> "# Coqui Public Model License 1.0.0
> **This license allows only non-commercial use of a machine learning model and its outputs.**"

The grant:

> "The licensor grants you a copyright license to do everything you might do with the model that would otherwise infringe the licensor's copyright in it, **for any non-commercial purpose**."

The definition, which forecloses our use twice over:

> "**Non-commercial purposes** include any of the following uses of the model **or its output**, but **only so far as you do not receive any direct or indirect payment arising from the use of the model or its output**."

> "Use by commercial or for-profit entities for testing, evaluation, or non-commercial research and development. **Use of the model to train other models for commercial use is not a non-commercial purpose.**"

The passthrough obligation, which would be operationally absurd for a hosted service:

> "## Notices — You must ensure that anyone who gets a copy of any part of the model, or any modification of the model, **or their output**, from you also gets a copy of these terms or the URL for them above."

And "Use" is defined to sweep outputs in: *"**Use** means anything you do with the model **or its output** requiring one of your licenses."*

**There is no commercial-licence path.** From the project's own [discussion #4304](https://github.com/coqui-ai/TTS/discussions/4304), answered by the maintainer of the active `idiap` fork: *"You can only use XTTS under the CPML now, there is no one to sell a commercial license anymore."* and *"there is no way of obtaining a commercial license."*

> 🚩 **A misinformation warning that vindicates the primary-source rule.** Several licence-summary sites claim CPML permits commercial use below "$1 million annual revenue or 10,000 end users." **No such threshold exists anywhere in the CPML 1.0.0 text.** CPML is a flat, unconditional non-commercial licence. If anyone cites that threshold in a planning discussion, the citation is wrong.

**Verdict: `public_servable = False`, permanently and unfixably.** The output clause encumbers even the generated audio, the Notices clause would force CPML terms onto every listener, and the licensor no longer exists. **Remove XTTS-v2 from the roadmap rather than keeping it as a "research lane" option** — a research lane whose artefacts can never be commercialised has limited value here. **HIGH.**

---

### 4.10 IndexTTS-2 — the only licence in this audit that explicitly restricts what you may do with the *audio*

The brief asks whether any TTS licence restricts generated audio. **This one does, and it does so in a way that specifically forecloses our use case.** The `LICENSE.txt` is the **bilibili Model Use License Agreement**.

**§1.5 — "Derivative Work" is defined to include model outputs:**

> "**"Derivative Work"**: means any derivative of the Model, including without limitation: **(i) any modification of the Model, model outputs, or their derivatives;** (ii) any work based on the Model, model outputs, or their derivatives; (iii) any other machine learning model which is created by re-training, fine-tuning, quantizing, LoRA, parameter-efficient fine-tuning, or any other method involving incremental weights or merged checkpoints…"

**§3.4(c) — the operative restriction:**

> "**You may not Use the bilibili indextts2 or any Derivative Work to improve any AI model, except for the bilibili indextts2 itself, its Derivative Works，or non-commercial AI models.**"

Read together: **the audio IndexTTS-2 generates is "Derivative Work", and you may not use it to improve any commercial AI model.** Using IndexTTS-2 to generate reference audio, style targets, or distillation data for VoiceForge's mapper is expressly prohibited. Studying the architecture is fine; touching its outputs is not.

**§2.2 — the commercial threshold (which is *not* a blanket NC bar):**

> "If You intend to Use … the Model or any Derivative Work, and either (i) your or any of your Affiliates' products or services had more than **100 million monthly active users** in the immediately preceding calendar month, or (ii) your or any of your Affiliates' annual revenue in the immediately preceding calendar year exceeded **RMB 1 billion**, You must request a separated license from us."

**§3.4(a)–(b) — downstream flow-through and notice retention:**

> "You must ensure that any downstream recipient of the Model or any Derivative Work that you distribute complies with this Agreement, and you must impose appropriate contractual terms on such downstream recipients."
> "You must retain all original copyright notices and a copy of this Agreement in every copy of the Model or any Derivative Work that you Use."

**§4.1(a) — a mandatory disclaimer string on any derivative distribution:**

> "…you must clearly state in the distribution page or accompanying documentation: *"Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work."*"

**§4.2** bans high-risk deployment (medical, autonomous driving, military, biometric surveillance, automated credit/employment decisions). **§5.3** is a patent-style retaliation clause terminating all rights if you sue bilibili over the model *or any output*. **§6** is PRC law + Shanghai Arbitration Commission. **§9**: the Chinese version prevails on any conflict.

**And a third blocker, found only by reading the inference path.** `indextts/utils/model_download.py` (`ensure_models_available`) hard-codes a runtime fetch:

```python
("semantic_codec", "amphion/MaskGCT", "semantic_codec/model.safetensors", …)
```

[`amphion/MaskGCT`](https://huggingface.co/amphion/MaskGCT/raw/main/README.md) frontmatter: `license: cc-by-nc-4.0`, `datasets: [amphion/Emilia-Dataset]`. `infer_v2.py` loads it as a mandatory component (`build_semantic_codec` → `safetensors.torch.load_model`). Note the history: the HF repo originally *bundled* the MaskGCT weights and **deleted them on 2025-09-08**, switching to runtime download. Removing the redistribution does not remove the NC obligation from the production inference path.

**Licence-file history worth recording:** `LICENSE.txt` was added to the HF repo on **2026-01-20** — for its first seven months the weights shipped with **no licence file at all**. `cardData.license` is still `null`; GitHub reports `NOASSERTION`. The newer `IndexTeam/IndexTTS-2.5` carries `license_name: "bilibili-model-license"`.

**Verdict: `public_servable = False`, and — separately and more importantly — `usable_as_a_data_source = False`.** Three independent blockers: the NC MaskGCT codec in the inference path, the DISCLAIMER's bar on unauthorised commercial use of synthesized voices, and §3.4(c)'s bar on using outputs to improve a commercial model. The scope document is right to say "study it; don't ship it", and should add **"and don't train on its output"**. **HIGH.**

---

### 4.11 VibeVoice — MIT on paper, withdrawn in practice, and it talks over your audio

The scope table says "research-only" and the licence file says MIT. **The scope table is closer to right.**

**What Microsoft actually did.** [GitHub README](https://raw.githubusercontent.com/microsoft/VibeVoice/main/README.md), verbatim:

> "2025-09-05: VibeVoice is an open-source research framework intended to advance collaboration in the speech synthesis community. After release, we discovered instances where the tool was used in ways inconsistent with the stated intent. Since responsible use of AI is one of Microsoft's guiding principles, **we have removed the VibeVoice-TTS code from this repository**."

`microsoft/VibeVoice-Large` now returns **HTTP 401** and is absent from the `microsoft` model listing; the repo has pivoted to ASR plus `VibeVoice-Realtime-0.5B`. Third-party re-uploads exist (`aoi-ot/VibeVoice-Large` and others) — **these are redistributions by parties with no rights to grant and must not be used.** `microsoft/VibeVoice-1.5B` survives with `license: mit`, but the TTS code that licence would cover is gone from the repo.

**The card's own restrictions, which read as licence language rather than advice:**

> "The VibeVoice model is **limited to research purpose use** exploring highly realistic audio dialogue generation detailed in the tech report."

> "Furthermore, **this release is not intended or licensed for any of the following scenarios**: Voice impersonation without explicit, recorded consent… Disinformation or impersonation… Real-time or low-latency voice conversion…"

> "**We do not recommend using VibeVoice in commercial or real-world applications** without further testing and development. **This model is intended for research and development purposes only.**"

**The product-killing clause is about the output:**

> "**Embedded an audible disclaimer (e.g. \"This segment was generated by AI\") automatically into every synthesized audio file.** Added an imperceptible watermark to generated audio so third parties can verify VibeVoice provenance…"

An **audible spoken disclaimer inside every render** disqualifies this for game dialogue regardless of the licence. And `VibeVoice-Realtime-0.5B` closes the obvious workaround with an anti-circumvention clause:

> "**Any act to circumvent, disable, or otherwise interfere with any technical or procedural safeguards implemented in this release, including but not limited to security controls, watermarking and other transparency mechanisms.**"

Upstream is clean (Qwen2.5-1.5B / 0.5B, both Apache-2.0). **Verdict: `public_servable = False`. NEEDS LAWYER on the MIT-vs-card-text conflict, but practically excluded on the audible-disclaimer ground alone. MEDIUM.**

---

### 4.12 The clean set, briefly

**VoxCPM2** — [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2/raw/main/README.md), `license: apache-2.0`, 2B params, 30 languages including Hindi, 48 kHz. Card, verbatim: *"📜 **Fully Open-Source & Commercial-Ready** — Apache-2.0 license, free for commercial use"* and *"Released under the Apache-2.0 license, free for commercial use."* Backbone: *"Based on MiniCPM-4, totally 2B parameters"* (`openbmb/VoxCPM-0.5B` declares `base_model: [openbmb/MiniCPM4-0.5B]`; MiniCPM4 verified Apache-2.0). GitHub `OpenBMB/VoxCPM` → Apache-2.0, pushed today. **No Emilia dependency found; the corpus is described only as "2M+ hours multilingual speech" and is not disclosed** — an upstream we cannot trace (§9). No licence-level output clause; Limitations carries an advisory worth honouring anyway: *"**Strictly forbidden** to use for impersonation, fraud, or disinformation. AI-generated content should be clearly labeled."*

⚠️ **Use VoxCPM2, not v1/1.5.** The `VoxCPM-0.5B` and `VoxCPM1.5` cards contain, *outside* their licence sections: *"This model is released for research and development purposes only. We do not recommend its use in production or commercial applications…"* — a line **absent from VoxCPM2**, which replaced it with "Commercial-Ready". Both also say the weights are Apache-2.0, so the operative grant is Apache-2.0 either way and the line reads as a recommendation — but there is no reason to inherit the argument. Note too that **no VoxCPM HF repo contains a LICENSE file**; mirror the GitHub Apache-2.0 text into our own distribution. ✅ **YES.**

**Qwen3-TTS family — the cleanest licence position in the entire audit.** Every open repo tags `license:apache-2.0`: `Qwen3-TTS-12Hz-{0.6B,1.7B}-Base`, `-CustomVoice`, **`Qwen3-TTS-12Hz-1.7B-VoiceDesign`**, and `Qwen3-TTS-Tokenizer-12Hz`. `gh api repos/QwenLM/Qwen3-TTS` → Apache-2.0, with the full 11,343-byte Apache text in `LICENSE`. **The key negative finding: there is no "Qwen Research" or "Tongyi Qianwen" licence anywhere on Qwen3-TTS** — Alibaba has used bespoke licences on other lines, but not here. No `base_model`, no declared `datasets`, no external codec (its own 12Hz tokenizer is also Apache-2.0), **no AUP, no output clause, no restriction list**. The VoiceDesign variant is a description-conditioned voice-design model — **the single cleanest licence-to-capability match for VoiceForge that exists today.** No Indic coverage (10 languages: zh/en/ja/ko/de/fr/ru/pt/es/it). ✅ **YES.**

**Chatterbox** — `ResembleAI/chatterbox` `license: mit`; [`LICENSE`](https://raw.githubusercontent.com/resemble-ai/chatterbox/master/LICENSE) is the MIT text, *"Copyright (c) 2025 Resemble AI"*, granting rights *"without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell"*. 23 languages including **Hindi**. **No AUP file exists** — `NOTICE`, `ACCEPTABLE_USE.md`, `AUP.md` and `CODE_OF_CONDUCT.md` all return 404; only `LICENSE` resolves. The sole ethical statement imposes no condition: *"Don't use this model to do bad things. Prompts are sourced from freely available data on the internet."*

**The "Llama backbone" question is resolved, and it is fine.** The card says *"0.5B Llama backbone"* and acknowledges Llama 3, which raised the possibility that Meta's Community Licence applies on top of MIT. It does not: `src/chatterbox/models/t3/t3.py` constructs `LlamaModel(LlamaConfig(**config_dict))` from a local config with `hidden_size=1024, num_hidden_layers=30` — matching **no** Meta-released checkpoint (Llama 3.2 1B is 2048/16). This is Llama *architecture* via `transformers`, trained from scratch; no Meta weights are loaded. Other deps clean: CosyVoice (Apache-2.0), HiFT-GAN (MIT). ✅ **YES.**

**Zonos-v0.1 — ⚠️ NOT in the clean set. See §1 row 7.** `Zyphra/Zonos-v0.1-transformer` and `-hybrid` are both `license: apache-2.0` and `gh api repos/Zyphra/Zonos/contents/LICENSE` is the Apache 2.0 text — but the speaker encoder that makes Zonos the two-tower target is a **third repo** loaded at runtime. [`zonos/speaker_cloning.py`](https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/speaker_cloning.py):

```python
spk_model_path = hf_hub_download(
    repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
    filename="ResNet293_SimAM_ASP_base.pt",
```

That repo's own card states its provenance:

> "The speaker embedding models are based on the [ResNet293-SimAM-ASP](https://github.com/VoxBlink2/ScriptsForVoxBlink2/tree/main/asv) models from VoxBlink2. **We use the pretrain models** as we found the finetunes performed worse."

And [VoxBlink2's LICENSE](https://raw.githubusercontent.com/VoxBlink2/ScriptsForVoxBlink2/main/LICENSE), verbatim:

> "**License:** The data falls under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (**CC BY-NC-SA 4.0**) license… **By downloading or using this dataset, you agree to abide by the terms of the CC BY-NC-SA 4.0 license, including any modifications or redistributions in any form.**"

The README adds: *"you can share and adapt the dataset for **non-commercial purposes** as long as you provide appropriate attribution and distribute your contributions under the same license."*

**This was raised publicly and never answered.** [`Zyphra/Zonos-v0.1-speaker-embedding` discussion #2](https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding/discussions/2), "Issues on licence", 13 Feb 2025:

> "VoxBlink2 models are licenced under CC BY-NC-SA 4.0 which prevent Zonos to change it and propose this model under Apache 2.0."

**No Zyphra response; the thread is still open 18 months later, and the GitHub repo was last pushed 2025-03-05.** Note the extra sting relative to a plain NC term: CC-BY-NC-**SA**'s ShareAlike would, if it reaches the derivative, try to pull our released weights toward NC-SA rather than merely blocking a use. Other deps are clean (Descript DAC = MIT; eSpeak-NG = GPL-3.0, separate process — check linkage). `Zyphra/ZONOS2` also declares `apache-2.0`; **whether it inherits the same encoder lineage is UNVERIFIED**.

⚠️ **NEEDS LAWYER — treat as `public_servable = False` for the cloning/speaker-embedding path until answered.** This is a live risk to scope §7's English track, not a footnote: the two-tower design's *"description → 128-d vector → frozen renderer"* contract runs entirely through this encoder. Mitigations, cheapest first: (a) open a discussion demanding a written Apache-2.0 warranty on the encoder; (b) **substitute a cleanly-licensed speaker encoder** — WeSpeaker/3D-Speaker ECAPA or ResNet variants trained on VoxCeleb-free data — and re-derive the manifold; (c) train a replacement encoder on our own GO-list corpora. Option (b) is the one to cost out at S1, because it also decouples the identity layer from a repo that has been unmaintained for 18 months.

**Chatterbox's built-in watermarking, which is an asset for us:**

> "Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker] - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy."

There is **no clause requiring retention** and no anti-circumvention term (contrast VibeVoice, §4.11); Perth itself is separately MIT. **Keep it on anyway** — it is free provenance defence, its removal would look bad in any dispute, and see §6.2 on EU AI Act Art. 50(2). Cross-ref [09-safety-and-watermarking.md](09-safety-and-watermarking.md).

**CosyVoice 2 / 3** — `gh api repos/FunAudioLLM/CosyVoice` → Apache-2.0; `FunAudioLLM/CosyVoice2-0.5B` and `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (CosyVoice 3 ships under the "Fun-CosyVoice" name) both `license: apache-2.0`. **No code-vs-weights split was found**, contrary to the scope doc's suspicion — though note the weights grant is a bare YAML field with no LICENSE file, and the ModelScope mirrors return an empty `License` field (no conflicting declaration, but no corroboration either). The bundled LM is `Qwen2ForCausalLM`, `hidden_size 896 / 24 layers / vocab 151936` = Qwen2.5-0.5B (Apache-2.0). ⚠️ **One action item: do not install the optional `ttsfrd` text frontend** — it is a closed-source Alibaba binary wheel distributed with no declared licence. The README is explicit that it is optional: *"If you do not install `ttsfrd` package, we will use wetext by default."* ✅ **YES** on licence (MEDIUM-HIGH) — the reason to exclude CosyVoice remains architectural (no speaker vector, scope §2), not legal.

---

## 5. The CC-BY-NC-trained-model question, honestly

Everything above turns on one unresolved question: **is a set of trained weights a "derivative" of its training data for licence purposes?** Here is where that stands in 2026, stated with the uncertainty intact.

### What the licence text actually says

CC-BY-NC-4.0's operative definitions, verbatim from [the legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en):

> "**NonCommercial** means not primarily intended for or directed towards commercial advantage or monetary compensation."

> "**Adapted Material** means material subject to Copyright and Similar Rights that is derived from or based upon the Licensed Material and in which the Licensed Material is **translated, altered, arranged, transformed, or otherwise modified in a manner requiring permission under the Copyright and Similar Rights held by the Licensor**."

> "**Share** means to provide material to the public by any means or process that requires permission under the Licensed Rights…"

Read carefully, the NC restriction attaches to **exercising the Licensed Rights** — reproducing and sharing the material. The "Adapted Material" definition is *self-limiting*: it only bites where the modification is "**in a manner requiring permission under the Copyright**". So the honest formulation of the question is:

**Does training a model on a work require permission under copyright at all?** If it does not, CC-BY-NC's conditions arguably never attach to the weights, and the NC term restricts only the copying you did to obtain the data — not the artefact you produce. If it does, the weights are plausibly Adapted Material and NC/SA propagate.

### Which way the arguments cut

**Arguments that weights are NOT encumbered:**
- The "requiring permission" qualifier above.
- Weights are statistical parameters, not a reproduction of any work; they generally fail a substantial-similarity test.
- Text-and-data-mining exceptions exist in several jurisdictions (EU DSM Art. 3 for research, Art. 4 with an opt-out; Japan Art. 30-4; Singapore) and TDM exceptions are *statutory* — a licence's NC term does not override an exception, though the EU's Art. 4 opt-out complicates this.
- US fair-use decisions in 2025 moved in favour of training being transformative in at least some postures.

**Arguments that they ARE encumbered:**
- The act of *acquiring and copying* the dataset for training is unambiguously a reproduction requiring permission, and the only permission you have is the NC licence. Breach of the condition terminates the licence — which makes your copies infringing regardless of the weights' status.
- Contract, not copyright, does much of the work here. Emilia's gate agreement, VoxCeleb's EULA, GigaSpeech's gate and the IITM EULA are **contracts you accepted**. A TDM exception does not release you from a contract you signed.
- **The strongest evidence is behavioural, and it is in this document.** ParaSpeechCaps' authors — the people best placed to say — released *their own* Parler-TTS fine-tune as CC-BY-NC-SA-4.0. The Emilia authors say *"Users are permitted to use this dataset only for non-commercial purposes."* Meta ships Expresso and EARS as NC. **Dataset authors overwhelmingly behave as though the encumbrance propagates.**
- **And so do model authors, including in the direction that costs them something.** Meta's **VoxPopuli** distributes its *data* as **CC0** and its *pre-trained models* as **CC BY-NC 4.0** — permissive data in, non-commercial weights out. **NVIDIA** propagates the licence explicitly onto the weights: `nvidia/parakeet-tdt-0.6b-v2` states *"GOVERNING TERMS: Use of this model is governed by the CC-BY-4.0 license"*, while `nvidia/canary-1b` ships CC-BY-**NC**-4.0 — the choice tracking each model's training mix. **The people with the most to lose from the encumbrance theory are pricing it in.**
- **The USCO has a name for the workaround, and it is not a compliment.** Part 3 report: *"in assessing whether the transfer of training datasets, synthetic data, or model weights is **obscuring a commercial benefit and constitutes 'data laundering'**, the financial relationships between the actors are relevant."* Splitting "NC research entity trains" from "commercial entity serves" is a structure the Office has already flagged. **Do not plan around it.**

### What the courts have actually said — and it is not one story

**No court anywhere has held that model weights are Adapted Material under a CC licence.** But the surrounding case law has moved a long way, and it now points in two directions.

**US — the derivative theory was rejected at the pleading stage.** *Kadrey v. Meta*, N.D. Cal. (Chhabria J., 20 Nov 2023):

> "The plaintiffs allege that the 'LLaMA language models are themselves infringing derivative works'… **This is nonsensical.** … **There is no way to understand the LLaMA models themselves as a recasting or adaptation of any of the plaintiffs' books.**"

**The US Copyright Office ties it to memorisation** — *Copyright and Artificial Intelligence, Part 3: Generative AI Training* (pre-publication, 9 May 2025), p. 29:

> "**Whether a model's weights implicate the reproduction or derivative work rights turns on whether the model has retained or memorized substantial protectable expression from the work(s) at issue.** … the use of those works in preparing a training dataset and training a model implicates the reproduction right, but **copying the resulting weights will only infringe where there is substantial similarity.**"

**Germany — the opposite conclusion, twice, and the second one is about audio.** *GEMA v. OpenAI*, LG München I, 11 Nov 2025 (42 O 14139/24), from the court's own press release:

> "Durch die Memorisierung sei eine Verkörperung als Voraussetzung der urheberrechtlichen **Vervielfältigung** … durch Daten in den spezifizierten Parametern des Modells gegeben."
> *(Through memorisation there is an embodiment, as the prerequisite for copyright **reproduction**, by data in the specified parameters of the model.)*

**⭐ *GEMA v. Suno*, LG München I, 31 July 2026 (42 O 763/25) — five weeks before this research date, and the most on-point decision that exists.** The first European ruling on a **generative AI audio** tool. The court prohibited four distinct acts, including **communication to the public through the offering of the model itself**, alongside training reproduction, memorisation in the parameters, and the outputs. On the TDM exception:

> "Diese Vervielfältigung in den Modellen sei **nicht durch die Schrankenbestimmungen des Text und Data Mining des § 44b UrhG gedeckt**."

And, applying US fair use extraterritorially to the US-side training:

> "Sämtliche im Rahmen der fair use Prüfung … zu prüfende Faktoren **sprächen gegen die Beklagte**."

Both are expressly **"nicht rechtskräftig"** — not final, appeals expected. *(Court press releases — HIGH for the quoted findings; the four-limb breakdown is MEDIUM.)*

**Read what that means for us specifically: a European court has held that *offering an audio-generation model to the public* is itself an infringing act, separate from training on the data.** That is our exact business model, and it is the configuration least protected by the "weights aren't a derivative" argument.

**US fair use — still zero binding precedent, and the two district rulings conflict.**

| Case | Court / date | Holding | Status as of 2026-09-02 |
|---|---|---|---|
| **Bartz v. Anthropic** | N.D. Cal., Alsup, 23 June 2025 | Training on **lawfully acquired** books = fair use; retaining **pirated** copies = not | **Settled.** Final judgment 20 July 2026, **$1.5bn** fund. **The fair-use holding will never be appellate-tested.** |
| **Kadrey v. Meta** | N.D. Cal., Chhabria, 25 June 2025 | SJ for Meta on training — expressly because plaintiffs argued badly | partial SJ; torrenting claim alive |
| **Thomson Reuters v. Ross** | D. Del., Bibas, 11 Feb 2025 | **No** fair use; factor four counts harm to a *potential* AI-training-data market | **3d Cir. No. 25-2153, argued 11 June 2026, no decision** (MEDIUM) |

Two quotes worth keeping. Bartz, on acquisition — the most durable finding in the area:

> "**Such piracy of otherwise available copies is inherently, irredeemably infringing even if the pirated copies are immediately used for the transformative use and immediately discarded.**"
> "**There is no carveout, however, from the Copyright Act for AI companies.**"

And Kadrey's own disclaimer, to quote at anyone who says US courts have blessed training:

> "**this ruling does not stand for the proposition that Meta's use of copyrighted materials to train its language models is lawful. It stands only for the proposition that these plaintiffs made the wrong arguments and failed to develop a record in support of the right one.**"
> "**In cases involving uses like Meta's, it seems like the plaintiffs will often win, at least where those cases have better-developed records on the market effects of the defendant's use.**"

**EU TDM — the exception we would be relying on, and Art 3 is unavailable to us.** Directive (EU) 2019/790 Art 3 covers only "**research organisations and cultural heritage institutions**". **Art 4** is our only route and requires **lawful access** *and* no reservation: it applies "on condition that the use … **has not been expressly reserved by their rightholders in an appropriate manner, such as machine-readable means**". **"Publicly reachable" is not "lawfully accessible."**

**And the European authority for "scraping to build a training set is lawful" is genuinely unstable right now.** *Kneschke v. LAION* was dismissed at LG Hamburg (27 Sept 2024) on the **§60d scientific-research** exception, with the court leaning *toward* a natural-language reservation counting as machine-readable. OLG Hamburg (5 U 104/24, 10 Dec 2025) affirmed the outcome **but on §44b grounds and reversed that lean**, holding the reservation *"nicht die gesetzlich vorgesehene Form (Maschinenlesbarkeit)"* — not in the legally prescribed machine-readable form. It expressly allowed revision. **⏰ The BGH hears it as `I ZR 281/25` — *"Erstellen eines Datensatzes für KI-Training"* — on 3 September 2026, i.e. tomorrow.** Two non-final lower-court judgments that disagree on reasoning, with the supreme-court hearing this week. **Re-check this within days; it is the cheapest high-value update available.**

**AI Act Art 53** (in force since 2 Aug 2025) requires a copyright policy honouring Art 4(3) reservations and a public "sufficiently detailed summary about the content used for training", and **Art 53(2)'s open-source exemption expressly does not cover those two obligations**. But Art 53 binds providers of **general-purpose AI models** (Art 3(63): "significant generality … a wide range of distinct tasks"), and **a single-purpose TTS or description-mapper model very likely is not one** — textual, not judicially tested, **UNVERIFIED**. **Art 50 reaches us regardless (§6.2).**

**India — and this cuts *our* way.** *ANI Media Pvt. Ltd. v. OpenAI*, Delhi HC, CS(COMM) 1028/2024, judgment **24 July 2026** (Bansal J.), ¶271:

> "**I am of the prima facie view that Open AI's act of storing ANI's original literary works for training LLMs underlying ChatGPT falls under Section 52(1)(a) of the Copyright Act and therefore, does not amount to infringement**…"

Interim injunction **refused**. ¶262 is the part that should shape our scraping policy: the court leaned heavily on the fact that ANI *could* have blocked crawlers and did not — "**Despite having an option of opt-out, evidently ANI has not exercised the same.**" ¶274 limits it to the application. Separately, **DPIIT's *Working Paper on Generative AI and Copyright (Part I)*** (8 Dec 2025) states "**There is currently no specific exception under copyright law for text and data mining**" and records that the Committee rejected a TDM exception in favour of a **mandatory blanket licence with statutory remuneration payable on commercialisation**. That is a proposal, not law — but it is the direction of Indian policy and it would attach a royalty to exactly what we plan to do.

**The practical read across all of this:** the acquisition question (how you got the data) is where courts have actually imposed liability — $1.5bn in Bartz. The weights-as-derivative question is unresolved and split by jurisdiction. And **publicly offering a generative audio model is the newest and least-tested exposure of all**, per GEMA v. Suno.

### What Creative Commons itself says — and it is more specific than I expected

CC published *"Using CC-Licensed Works for AI Training"* (with a May 2025 legal primer) which addresses this question **by licence element**. Verbatim from [creativecommons.org/using-cc-licensed-works-for-ai-training-2/](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/):

**On NonCommercial — this is the sentence that decides our posture:**

> "**NonCommercial (NC)** CC BY-NC and CC BY-NC-SA give permission for NonCommercial uses only. **If AI training data includes the NonCommercial restriction, then following the NC restriction would require that all stages, from copying the data during training to sharing the trained model, must not be for commercial gain.**"

**On ShareAlike — worse for us, and it names outputs as well as models:**

> "**ShareAlike (SA)** CC BY-SA and CC BY-NC-SA require that adaptations be shared under the same license. **If AI models or outputs are based on ShareAlike content and they will be shared publicly, following the ShareAlike condition would require AI developers to use the same CC license as the original works.**"

**On Attribution — usefully permissive about the *form*:**

> "**Attribution (BY)** All CC licenses require attribution to the creator(s) of the licensed material. **For AI model training, attribution could be a simple link to the source of the dataset used to train the model.**"

**And on NoDerivatives**, which matters if an ND corpus ever appears in a mix: *"Following the NoDerivatives restriction would require that ND-licensed content not be used as training data."*

**CC's own framing of how binding this is — quote it honestly, because it cuts both ways:**

> "Keep in mind: following this guidance is likely to lead to **overcompliance** with both copyright law and CC license terms. **It assumes the most restrictive legal interpretation** for those who wish to take a conservative approach and minimize risk."

> "**CC licenses apply only when copyright permission is required. If exceptions or limitations apply, then the CC license terms don't apply.**"

So CC is explicitly *not* asserting that the law compels this reading — they are describing what compliance looks like if you take the conservative view, and they concede that copyright exceptions may mean the licence never attaches at all. **That is exactly the two-sided honesty this question deserves.** But note what it means practically: the steward of these licences has published, in writing, that following NC means *"sharing the trained model must not be for commercial gain"* and that following SA means using *"the same CC license as the original works"*. **A rights-holder in a dispute will cite this page.** We should not plan to be on the other side of it.

**Confidence: HIGH** that this is CC's published guidance; **UNSETTLED** that it states the legal position, which CC itself does not claim.

### The engineering conclusion

**Treat NC and SA as propagating, and design so the question never has to be answered.**

The reasoning is asymmetric-cost, not legal:

- If NC does not propagate and we avoided NC data anyway, we lost some expressive coverage.
- If NC does propagate and we trained on it, we retrain from scratch, take down a shipped model, and possibly face a claim from a rights-holder who has already published their view.

The second outcome is orders of magnitude worse, and the mitigation is cheap: **use only CC-BY / CC0 / public-domain / explicitly-commercial corpora for anything whose weights get released.** Keep an NC research lane if it is scientifically useful, and enforce the separation with the same `public_servable` mechanism that guards the backends — a `train_releasable: bool` on every dataset record, checked by the training script.

**And note the two places the debate is irrelevant.**

**First, contract terms.** Whatever copyright says, Emilia clause 6 binds our employer, VoxCeleb's EULA propagates its conditions to our redistribution, GigaSpeech's gate says "non-commercial research and educational purposes" only, IITM §2.2 restricts our recipients, and IndexTTS-2 §3.4(c) forbids using its output to improve a commercial model. **Those are enforceable as contracts regardless of how the copyright question resolves**, and a TDM exception does not release you from an agreement you clicked through. They are the harder constraint and the one this project must actually design around.

**Second, acquisition.** Bartz is the only case in this area that has produced a nine-figure number, and it was about *how Anthropic got the books*, not about training on them. **Log provenance and TDM-reservation state (robots.txt, `noai` headers, gate agreements accepted) per source, at fetch time, permanently.** That record is cheap now and unreconstructable later, and it is the difference between the Bartz fair-use holding and the Bartz settlement.

**Confidence: the underlying legal question is genuinely UNSETTLED and is now split between US and German courts. The engineering recommendation is HIGH confidence, precisely because it does not depend on resolving it.**

---

## 6. Voice-likeness, personality rights & synthetic-media regulation

> Still not legal advice — but this section is now built on court judgments, official gazettes and Commission documents read directly, not on summaries. Where an item is secondary-sourced or unconfirmed it says so.

### 6.0 Why this is not the same question as copyright — and the case that proves it

Copyright protects a *recording*. Personality and publicity rights protect a *person's identity*, including the distinctive sound of their voice, independently of who owns the recording. **Clearing copyright in a corpus does not clear the personality rights of the speakers in it.**

The clearest demonstration is **Lehrman v. Lovo, Inc.**, No. 1:24-cv-03770 (S.D.N.Y., Oetken J., 10 July 2025) — voice actors suing an AI voice-over company. The court's framing:

> "**What Plaintiffs are essentially asking for is copyright protection for their voices *qua* voices.**"
> "**Copyright protection does not extend to this kind of imperfect mimicry, even when accomplished using advanced technology rather than more traditional techniques like musical covers or impersonations.**"

**The copyright and Lanham Act claims were dismissed. The New York Civil Rights Law §§ 50/51 right-of-publicity claims and the consumer-protection claims survived.** *(Docket confirmed via CourtListener RECAP — HIGH; holdings quoted from law-firm reporting of the opinion, primary text not retrieved — MEDIUM.)*

**The lesson for VoiceForge: our copyright analysis (§3–§5) is not our main exposure. Publicity and personality rights are.** Two immediate consequences for data choices already made in §7:

- **VoxCeleb** is 6,000+ named celebrities. A perfect copyright licence would not clear it, and the VGG privacy notice's reliance on a *research* exemption makes that explicit.
- **AnimeVox** labels 19 named characters from 15 series. Behind each is a working voice actor — precisely the *Lehrman* plaintiff class.

Both are already excluded on copyright grounds. **The publicity analysis independently confirms the exclusion**, which is a useful redundancy.

### 6.1 United States

**There is no federal right of publicity.** U.S. Copyright Office, *Copyright and AI, Part 1: Digital Replicas* (July 2024):

> "**While no federal statute focuses solely on the use of an individual's image, likeness, or voice**, several serve to limit the creation or use of digital replicas in particular circumstances."
> "…the result is **a patchwork of protections, with the availability of a remedy dependent on where the affected individual lives or where the unauthorized use occurred.**"

The Office notes Alaska, Kansas, Maryland and North Carolina have *neither* statutory nor common-law publicity rights, while 27 states provide postmortem rights ranging from 20 years to **indefinite in Tennessee**.

#### Tennessee ELVIS Act — the first enacted US law that attaches liability to the *tool*

SB 2096 / HB 2091, **Public Chapter 588**, signed 26 Mar 2024, **effective 1 July 2024**. Operative text is amendment HA0578, which deleted everything after the enacting clause.

The definition, which expressly covers synthesis:

> "'**Voice**' means a sound in a medium that is readily identifiable and attributable to a particular individual, **regardless of whether the sound contains the actual voice or a simulation of the voice** of the individual;"

**§ 47-25-1105(a)(3) — the tool provision, and the one to read against our product:**

> "A person is liable to a civil action if the person **distributes, transmits, or otherwise makes available an algorithm, software, tool, or other technology, service, or device, the primary purpose or function** of such algorithm, software, tool, or other technology, service, or device **is the production of a particular, identifiable individual's photograph, voice, or likeness, with knowledge** that distributing, transmitting, or otherwise making available the photograph, voice, or likeness was not authorized by the individual…"

**Two limiting elements save a general TTS product: "particular, identifiable individual" and a knowledge requirement.** A multi-speaker description-to-voice system is not within it. A per-celebrity cloned voice offered as such is squarely within it. **This is the strongest legal argument yet for scope §15.2's named-person refusal gate — it is not merely good practice, it is what keeps us outside the ELVIS Act's tool provision.**

#### NO FAKES Act — **PENDING, not enacted, as of 2026-09-02**

| Bill | Introduced | Status |
|---|---|---|
| S.4875 (118th) | 2024-07-31 | died in committee |
| S.1367 / H.R.2794 (119th) | 2025-04-09 | never advanced past referral |
| **S.4591 "NO FAKES Act of 2026"** | 2026-05-20 | **reported by Senate Judiciary 2026-06-24; on the Senate Legislative Calendar, General Orders, Calendar No. 446. Awaiting floor action.** |
| H.R.8915 (119th) | 2026-05-20 | referred, no action |

*(govinfo BILLSTATUS XML — HIGH; the 2026-08-10 → 2026-09-02 window is MEDIUM.)*

From S.4591 **as reported** — the definition:

> "'digital replica' … **means a newly created, computer-generated, highly realistic electronic representation that is readily identifiable as the voice or visual likeness of an individual** that — (i) is embodied in a sound recording, image, audiovisual work … in which the actual individual did not actually perform or appear…"

**§2(c)(2)(B) — a Grokster-style three-prong tool test drafted for products like ours:**

> "Distributing, importing, transmitting, or otherwise making available to the public a product or service that — (i) **is primarily designed to produce 1 or more digital replicas of a specifically identified individual** … without … authorization; (ii) **has only limited commercially significant purpose or use other than** to produce a digital replica of a specifically identified individual …; or (iii) **is marketed, advertised, or otherwise promoted** … as a product or service designed to produce a digital replica of a specifically identified individual…"

Three product-design consequences if this passes, all of which we should build for now because they are free:

1. **Prong (iii) means our marketing copy alone can create liability.** "Make any celebrity's voice" as a tagline would do it even if the product could not. Write the copy as if the Act were already law.
2. **The tool safe harbour is forfeited by (c)(2)(B).** §2(d)(1)(A) grants no liability for distributing a tool "*unless the product or service is a product or service described in subsection (c)(2)(B)*." So (c)(2)(B) is the load-bearing question, not notice-and-takedown.
3. **The online-service safe harbour has real engineering requirements** — a **registered designated agent**, a **repeat-violator termination policy users are informed of**, prompt removal on notice, a **14-day counter-notice restoration window**, and for some service classes **digital-fingerprint-based staydown**. §2(h)(1) makes it an IP law for §230(e)(2) purposes, so **Section 230 immunity would not apply.** Cross-ref scope §9 (S9 multitenant/moderation) — this is the design brief for it.

§2(g) preemption **expressly preserves** pre-2 Jan 2025 state causes of action for "making available to the public a product or service capable of producing 1 or more digital replicas" — **so the Tennessee tool provision survives NO FAKES rather than being displaced by it.**

#### FTC impersonation rule — **does NOT cover individuals**

The Government and Business Impersonation Rule, 16 CFR Part 461 (89 FR 15017), is effective 1 April 2024 and prohibits falsely posing as a *government entity or a business*. The SNPRM (89 FR 15072) proposed §461.5, a "means and instrumentalities" provision that would have caught a hosted cloning tool:

> "It is a violation of this part … to **provide goods or services with knowledge or reason to know that those goods or services will be used to** … materially and falsely pose as … a government entity or officer thereof, a business or officer thereof, **or an individual**…"

But 89 FR 104905 (26 Dec 2024):

> "**The Commission has decided not to proceed with the SNPRM's proposed means and instrumentalities provision at this time.**"

Verified against the **eCFR as of 2026-08-31**: Part 461 still contains only §§461.1, 461.2, 461.3. **No §461.4, no §461.5.** The FTC's 2026 Regulatory Agenda still lists the rule as a current rulemaking, so this can change. *(FTC Act §5 case-by-case authority is unaffected.)*

#### State statutes worth tracking

- **Washington SSB 5886, Ch. 69, Laws of 2026 — effective 11 June 2026, and the broadest yet.** New RCW 63.60.020(3) defines "forged digital likeness" to include "**an audio recording which is either persistent or transmitted in real-time** of an actual and identifiable individual's **voice**" that is indistinguishable from genuine, misrepresents the individual, and is likely to deceive. RCW 63.60.050 as amended: "**An infringement may occur under this section without regard to whether the use or activity is for profit or not for profit.**" **Note "transmitted in real-time" — this reaches live voice conversion, not just rendered files.**
- **California AB 1836** (Civ. Code §3344.1, eff. 1 Jan 2025) — liability for one who "**produces, distributes, or makes available the digital replica of a deceased personality's voice**"; minimum $10,000. **AB 2602** (Lab. Code §927) voids digital-replica contract clauses absent specific description of uses plus counsel/union representation.
- **Illinois** 765 ILCS 1075 as amended (eff. 1 Jan 2025) — "digital replica" covers "the **voice**, image, or likeness … created using a computer, algorithm, software, tool, artificial intelligence, or other technology", and extends liability to those who "**materially contribute to, induce, or otherwise facilitate**" a violation after actual knowledge. **A contributory hook aimed at hosts.**
- **New York** GOL §5-302 and Civ. Rights Law §50-f.

*(State wording MEDIUM — official sites blocked direct download; re-verify exact text before quoting in a compliance document. **No fifty-state survey was run**; Montana and others are UNVERIFIED.)*

### 6.2 EU AI Act Article 50 — **already in force, and it applies to us**

#### The text

**Article 50(2)**, verbatim from Regulation (EU) 2024/1689:

> "**Providers of AI systems, including general-purpose AI systems, generating synthetic audio, image, video or text content, shall ensure that the outputs of the AI system are marked in a machine-readable format and detectable as artificially generated or manipulated. Providers shall ensure their technical solutions are effective, interoperable, robust and reliable as far as this is technically feasible**, taking into account the specificities and limitations of various types of content, the costs of implementation and the generally acknowledged state of the art, as may be reflected in relevant technical standards. This obligation shall not apply to the extent the AI systems perform an assistive function for standard editing or do not substantially alter the input data provided by the deployer or the semantics thereof…"

**Article 50(4)**, verbatim:

> "**Deployers of an AI system that generates or manipulates image, audio or video content constituting a deep fake, shall disclose that the content has been artificially generated or manipulated.** … Where the content forms part of an evidently artistic, creative, satirical, fictional or analogous work or programme, the transparency obligations set out in this paragraph are limited to disclosure of the existence of such generated or manipulated content in an appropriate manner that does not hamper the display or enjoyment of the work."

Also operative: **Art 50(1)** (tell people they are interacting with an AI system unless obvious); **Art 50(5)** — the information must be provided "**at the latest at the time of the first interaction or exposure**" and "shall conform to the applicable accessibility requirements"; **Art 3(60)** defines "deep fake" as content "that resembles existing persons, objects, places, entities or events and would falsely appear to a person to be authentic or truthful."

**The Art 50(4) creative-works carve-out is directly useful to us**: game dialogue is "an evidently artistic, creative, … fictional or analogous work", so the *deployer's* disclosure obligation is limited to disclosing the existence of generated content "in an appropriate manner that does not hamper the display or enjoyment of the work" — i.e. a credits-screen or manifest disclosure, not an in-game announcement. **This does not soften Art 50(2), which is ours as *provider* and has no such carve-out.**

#### The dates — verified, including the Digital Omnibus

**Article 50 applied from 2 August 2026. It was not delayed. That date is one month in the past.**

Art 113 sets general application at 2 August 2026 with derogations in points (a)–(c), **none of which mentions Chapter IV** (where Art 50 sits).

**Digital Omnibus on AI = Regulation (EU) 2026/1744**, of 8 July 2026, published OJ 24 July 2026, **in force 27 July 2026**. Its effect on Art 50 is narrow and specific:

1. **Art 50(1)–(6) are unchanged.** The only amendment is to paragraph 7 (removing a Commission implementing-act empowerment).
2. **A four-month transitional, for legacy systems only** — new **Art 111(4)**:
   > "**Providers of AI systems, including general-purpose AI systems, generating synthetic audio, image, video or text content, that have been placed on the market before 2 August 2026 shall take the necessary steps in order to comply with Article 50(2) by 2 December 2026.**"

   **Anything placed on the market on or after 2 August 2026 complies immediately, with no transition. A new product gets nothing.** VoiceForge is a new product. **The obligation attaches on the day we launch in, or to, the Union.**
3. Art 113's high-risk dates were pushed (2 Dec 2027 / 2 Aug 2028). **Chapter IV is untouched.**

⚠️ **Version warning:** both pre- and post-Omnibus renderings of Art 113 circulate. Check which you are reading.

#### It applies to us, and open-sourcing does not escape it

**Art 2(1)(a)/(c)** is extraterritorial: it reaches providers "irrespective of whether … established or located within the Union or in a third country", and third-country providers "**where the output produced by the AI system is used in the Union**."

**Art 2(12) — the open-source exemption expressly carves *out* Article 50:**

> "This Regulation does not apply to AI systems released under free and open-source licences, **unless they are placed on the market or put into service as high-risk AI systems or as an AI system that falls under Article 5 or 50**."

The **Commission Guidelines on Article 50, C(2026) 5054 final, 20 July 2026** draw the line we care about exactly:

> "**(23)** … **providers and deployers of open-source AI systems within the scope of Article 50 AI Act still need to ensure compliance** with their respective transparency obligations."
> "**(24)** By contrast, **providers of free and open-source AI components that cover software, data, models, tools, services or processes** for customisation and integration into an AI system **and that do not constitute in themselves (in part or in whole) an AI system, are not subject to the transparency obligations in Article 50**."

**Read against our plan that is a clean, actionable split: releasing the mapper weights and VoicePersona v2 as components is outside Art 50; running the hosted service that emits audio is inside it.**

And on our "frozen third-party backends" posture — Guidelines para (11):

> "a company provides a generative or interactive AI application … **on the Union market under its own name or trademark** … **The company is a provider responsible for compliance with the transparency obligations in Article 50(1) and/or (2)**, **regardless of whether the AI system is provided for free or for payment and regardless of whether the provider is established or located in the Union or in a third country.**"

> "If a company takes an already existing generative AI system placed on the market by another provider and **modifies that system (e.g. with new training data)**, which it afterwards puts into service **under its own name or trade mark, then that company becomes a provider of the new system**…"

**So "we just host someone else's frozen model" is not a defence.** Serving VoxCPM2 under the VoiceForge brand makes VoiceForge the provider for Art 50(2). Two further Guidelines points close the remaining escape routes:

- para **(58)**: "**Article 50(2) AI Act may apply to generative AI systems designed with a narrow intended purpose to produce specific outputs.**" → a single-purpose TTS system is in scope.
- para **(60)**: "**Audio** refers to a time-varying signal encoding sound … **This may cover speech**…"
- para **(428)**, on deep fakes: "'Persons' is to be understood as realistic, human beings (**including digital replicas of real persons**, realistic AI-generated human avatars or personas, and **personal characteristics or expressions, such as image, voice**, behaviour, performances etc.)." → **voice cloning is a deep fake for Art 50(4)**; our no-cloning architecture keeps us out of 50(4)'s core.

**Penalties: Art 99(4)** — up to **EUR 15,000,000 or 3% of total worldwide annual turnover**; Art 99(6) applies the *lower* of the two for SMEs and start-ups.

#### What "machine-readable marking" means in practice

**Recital 133** names the acceptable techniques: "**watermarks, metadata identifications, cryptographic methods for proving provenance and authenticity of content, logging methods, fingerprints or other techniques**", implementable "at the level of the AI system or at the level of the AI model."

**Guidelines para (73)** sets a helpful limit: "**providers are not required to record or keep a full provenance chain**."

**Guidelines para (76)** sets the interoperability bar — and concedes the standards are not there yet:

> "To ensure full interoperability, **providers must rely on publicly-available industry standard detection solutions** that allow any third party to implement detection… **Where such standards are not available, in particular at the initial stage of the implementation of Article 50(2) AI Act for watermarking technologies, the provider may rely on its own detection solution** … so long as those solutions ensure interoperability. That possibility should be limited in time…"

**Code of Practice on Transparency of AI-generated Content**, published **10 June 2026**, drafted by six independent experts with input from 180+ stakeholders; assessed adequate by the Commission and the AI Board; ~190 organisations signed by end-July 2026. It uses "a revised **two-layered marking approach involving secured metadata and watermarking**, optional fingerprinting and logging." It is **voluntary** — and note the Omnibus recital: codes of practice "**have limited legal effect, and in particular do not grant a presumption of conformity**." Signing demonstrates compliance; it does not confer it.

⚠️ **A practical gap to plan around: C2PA has no dedicated audio guidance.** C2PA Technical Specification 2.2 (May 2025) lists ID3 and RIFF among embeddable formats but contains **no audio-specific hard-binding/soft-binding guidance** comparable to its image and video sections *(verified negative against the spec text — HIGH)*. **For audio-only output we will need a watermark (soft binding) alongside container metadata**, and para (76) currently permits our own detection solution precisely because the standard is missing. Cross-ref [09-safety-and-watermarking.md](09-safety-and-watermarking.md) — this makes the AudioSeal/Perth decision a *compliance* decision, not just a safety one.

### 6.3 India — the fastest-moving jurisdiction, and already binding

#### Personality rights: a judge-made right, built entirely on interim orders

**There is no statutory right of publicity in India.** *Digital Collectibles Pte Ltd v. Galactus Funware Technology*, 2023:DHC:2796 (Bansal J., 26 Apr 2023):

> "¶53. **In the absence of a specific legislation, the right to publicity cannot be an absolute right in India.**"
> "¶55. … **the violation of the right of publicity in India has to be considered on the touchstone of the common law wrong of passing off, as also weighed against the 'right to freedom of speech and expression' enshrined under Article 19(1)(a).**"

**Anil Kapoor v. Simply Life India**, CS(COMM) 652/2023, Delhi HC (Prathiba M. Singh J., 20 Sept 2023) — *ex parte ad interim*:

> "¶39. **Using a person's name, voice, dialogues, images in an illegal manner, that too for commercial purposes, cannot be permitted.**"
> "¶48. … **restrained from utilizing the Plaintiff-Anil Kapoor's name, likeness, image, voice, personality or any other aspects of his persona** … **or in any other manner misuse the said attributes using technological tools such as Artificial Intelligence, Machine Learning, deep fakes, face morphing and/or GIFs** either for monetary gains or otherwise…"

**Jackie Shroff v. The Peppy Store**, 2024:DHC:4046 (Narula J., 15 May 2024) — restrained commercial use of "**voice**" and separately restrained an "**unlicensed AI chatbot that uses attributes of the Plaintiff's persona**". **But the Court refused ex parte relief against a "Thug Life" YouTube compiler**, warning at ¶21 that enjoining such creators "could set a precedent that stifles freedom of expression." **The right is not absolute even in the cases that grant it.**

**⭐ Arijit Singh v. Codible Ventures LLP**, Bombay HC (R.I. Chagla J., 26 July 2024) — **the AI voice-cloning case, and the one that most directly describes a product like ours.** The defendants ran "Real Voice Cloning (RVC)" with 456 of the plaintiff's songs uploaded without authorisation; `huggingface.co` was among the named defendants.

> "¶17. … prima facie … **the Plaintiff's personality traits … including the Plaintiff's name, voice, photograph / caricature, image, likeness, persona … are protectable elements of the Plaintiff's personality rights and right to publicity.**"

> "**¶18. Making AI tools available that enable the conversion of any voice into that of a celebrity without his/her permission constitutes a violation of the celebrity's personality rights. Such tools facilitate unauthorized appropriation and manipulation of a celebrity's voice, which is a key component of their personal identity and public persona.**"

The operative injunction ¶24(a) restrains "**voice / vocal style and technique**" and expressly "**creating or using artificial intelligence voice models, or voice conversion tool, synthesized voices or digital avatars … that imitate or mimic or represent the Plaintiff**". It may operate as a **dynamic injunction** (¶22). The test applied is: celebrity status + **identifiability from the defendant's use** + **commercial gain**; ¶21 adds that free expression "does not grant the license to exploit a celebrity's persona for commercial gain."

**¶18 is the sentence to design against.** Note precisely what it condemns: *tools that enable conversion of a voice into that of a celebrity*. **VoiceForge's locked no-audio-upload architecture (scope §15.1) means we do not build that tool** — there is no input by which a user supplies a target voice. That is a genuine, structural distinction from the Codible defendants, and it is the single best reason to keep §15.1 locked.

⚠️ **A caveat that materially weakens reliance on this case: ¶32 states "This order will continue till 3rd September, 2024." Whether it was extended, confirmed, or modified is UNVERIFIED** — Indian Kanoon is Cloudflare-blocked and the Bombay HC portal was unreachable. **The flagship holding sits in an order that by its own terms expired.** Get this checked before citing it in any decision document.

**Asha Bhosle v. Mayk Inc**, Bombay HC (Arif S. Doctor J., 29 Sept 2025) adopts Arijit ¶18 near-verbatim, restraining use of "**voice / vocal style and technique**" including "through the use of any technology such as **AI Voice Models**" — so the reasoning has been repeated, which helps.

**Aishwarya Rai Bachchan v. Aishwaryaworld.com**, Delhi HC (Tejas Karia J., 9 Sept 2025) — ⚠️ **precision point**: voice appears in the reasoning, but the **operative clause ¶39(i) enumerates only name/acronym, image and likeness, and "any other attributes of her persona"**. Voice falls under the residual, unlike Arijit and Asha Bhosle where it is expressly enumerated.

*(MEDIUM/LOW, order texts not retrieved: Abhishek Bachchan, Sadhguru, Kumar Sanu, Hrithik Roshan, Karan Johar. **Nagarjuna Akkineni: no primary source found — UNVERIFIED.**)*

> ⚠️ **Precedential weight — read this before relying on any of the above.** **Every** Indian voice/personality decision found is an **interim or ad-interim order, most of them ex parte**, expressly on a *prima facie* view. **No final judgment after trial on AI voice cloning exists in India.** They bind named defendants; they are persuasive, not binding, and provisional. And *Digital Collectibles* — which requires misrepresentation and subordinates publicity to Art 19(1)(a) — sits in unresolved tension with the *Anil Kapoor* / *Arijit Singh* line. Two single-judge High Court lines, both interim, pointing different ways.

#### Indian performers' rights — a rights layer nobody's copyright analysis covers

Copyright Act 1957:

> **§2(qq):** "'performer' includes an actor, singer, musician, dancer, acrobat, juggler, conjurer, snake charmer, **a person delivering a lecture or any other person who makes a performance**"
> **§38A(1)(a):** the performer's exclusive right "to make a sound recording … including … **reproduction of it in any material form including the storing of it in any medium by electronic or any other means**…"
> **§39:** no performer's right is infringed by "(a) the making of any sound recording … for the private use of the person making such recording, or **solely for purposes of bona fide teaching or research**; or … (c) such other acts … **which do not constitute infringement of copyright under section 52**."

Whether §39(c) carries the *ANI* §52(1)(a) reasoning across to performers' rights, and whether an ordinary person reading prompts into a speech corpus is a "performer", are **open questions with no authority — UNVERIFIED**. **For a speech product this is a distinct rights layer from copyright in the underlying text, and it applies to Indic corpora recorded in India.** *(It does not disturb our GO list — IndicVoices obtained written consent from each participant and passed an Institute Ethics Committee review, which is the right posture under this head too.)*

#### DPDP Act 2023 — and the "publicly available" exemption is narrower than assumed

The Act never uses the words "voice", "biometric" or "audio". The relevant definitions:

> **§2(t):** "'**personal data**' means any data about an individual who is identifiable by or in relation to such data;"
> **§6(1):** consent "shall be **free, specific, informed, unconditional and unambiguous with a clear affirmative action**…"
> **§6(4):** "**right to withdraw her consent at any time**, with the ease of doing so being comparable to the ease with which such consent was given."

**⭐ §3(c)(ii) — the exemption people over-read:**

> "it shall not apply to — … (ii) **personal data that is made or caused to be made publicly available by — (A) the Data Principal to whom such personal data relates; or (B) any other person who is under an obligation under any law for the time being in force in India to make such personal data publicly available.**"

**This is not a general "it was on the internet" exemption.** Either the speaker published it, or an Indian statutory obligation required its publication. **A third-party upload, a fan mirror, or a dataset aggregator satisfies neither limb.** That is directly adverse to any scraped-audio corpus — and it is a second, independent reason (alongside §3.12) to keep Emilia-class data out of anything touching Indian speakers. *(No Indian judicial interpretation of §3(c)(ii) exists; whether a voice recording is "personal data" under §2(t) has no Board or judicial authority — **UNVERIFIED/LOW**.)*

**DPDP Rules 2025 (G.S.R. 846(E), 13 Nov 2025) — the parts that would bite are NOT yet in force:**

| Rules | Subject | In force 2026-09-02? |
|---|---|---|
| 1, 2, 17–21 | Board constitution, definitions | ✅ yes |
| 4 | Consent Managers | ❌ ~13 Nov 2026 |
| **3, 5–16, 22, 23** | **notice, security safeguards, breach notification, retention, data-principal rights, children's data, cross-border transfer** | ❌ **~13 May 2027** |

**We have roughly eight months before consent-manager obligations and twenty before the substantive compliance regime.** That is enough runway to build consent/erasure into the identity store from the start rather than retrofitting it. *(⚠️ the gazette is dated 13 Nov 2025 while the official PIB backgrounder says 14 Nov — treat May 2027 as ±1 day.)*

#### ⭐ IT Rules synthetic-content labelling — **IN FORCE, and it is an audio requirement**

Information Technology (Intermediary Guidelines and Digital Media Ethics Code) **Amendment Rules, 2026, G.S.R. 120(E), dated 10.02.2026**, effective **~20 Feb 2026** *(MEDIUM on the date; **HIGH** on the rule text, read from the official MeitY consolidated PDF)*.

> **Rule 2(1)(wa):** "'**synthetically generated information**' means audio, visual or audio-visual information which is artificially or algorithmically created, generated, modified or altered using a computer resource, **in a manner that such information appears to be real, authentic or true and depicts or portrays any individual or event in a manner that is, or is likely to be perceived as indistinguishable from a natural person or real-world event**;"

> **Rule 3(3)(a)(ii):** "every such information … is **prominently labelled** in a manner that ensures prominent visibility in the visual display … **or, in the case of audio content, through a prominently prefixed audio disclosure**, that can be used to immediately identify that such information is synthetically generated information … and **such information shall be embedded with a permanent metadata or other appropriate technical provenance mechanisms, to the extent technically feasible, including a unique identifier**, to identify the computer resource of the intermediary used to create, generate, modify or alter such information;"

> **(b)** "the intermediary **shall not enable the modification, suppression or removal of the label, permanent metadata, including the unique identifier**…"

**⭐⭐ CORRECTION TO THE BRIEF (and to §6.3 as first drafted).** The brief anticipated "a visible label covering a defined percentage of the display area … and an audible/embedded marker for audio." **The percentage requirement was in the October 2025 draft and did NOT survive into the notified rules.** The official consolidated text was searched for "per cent", "percent", "10%", "surface area", "display area" and "duration" — **there is no percentage or size threshold anywhere in the synthetic-generated-information provisions** *(verified negative against the official text — HIGH)*. The final standard is qualitative: prominent visibility for visual, and "**a prominently prefixed audio disclosure**" for audio.

**A prefixed spoken disclosure is a hard product problem for game dialogue** — it is not a metadata flag, it is audible content at the head of the file. Two mitigations, both needing counsel:

- **Rule 4(1A)** (user declaration + intermediary verification) applies **only to significant social media intermediaries**, which a TTS API is unlikely to be. **Rule 3(3) is the one that reaches us.**
- **The scope of Rule 2(1)(wa) is genuinely arguable.** The information must "appear to be real, authentic or true" **and** "depict or portray any individual or event". A synthetic voice for a fictional game character arguably portrays *no individual and no event*. Proviso (c) also excludes computer-resource use "**solely for improving accessibility, clarity, quality, translation, description, searchability, or discoverability**". **Whether a generic, non-cloned synthetic voice falls inside Rule 2(1)(wa) at all is UNVERIFIED and there is no authority. Do not treat either reading as settled** — but note that the argument for exclusion is much stronger for VoiceForge than for a voice-cloning product, which is another dividend of the §15.1 architecture.

Also note: **the takedown clock was cut from thirty-six hours to three hours** (rule 3(1)(d)). That is a moderation-SLA requirement for S9, not a labelling one.

### 6.4 What this means concretely

| Obligation | Source & status | Where it lands in the build |
|---|---|---|
| **No user audio upload** | Locked (scope §15.1) — and it is what keeps us outside Arijit ¶18, the ELVIS Act tool provision, and NO FAKES §2(c)(2)(B) | API boundary — enforce in code, not docs |
| **Refuse named-real-person descriptions** | Committed (scope §15.2) — the ELVIS Act's "particular, identifiable individual" element makes this load-bearing | Description-validation layer; hard gate, fail closed |
| **Marketing copy must not promise celebrity voices** | NO FAKES §2(c)(2)(B)(iii) — *promotion* alone creates liability | Website, docs, app-store copy. Free to comply with today |
| **Machine-readable watermark on every output** | **EU Art 50(2) — in force since 2 Aug 2026; no grace period for a new product** | Render path, every backend, before any EU exposure |
| **Prefixed audio disclosure + embedded permanent metadata with a unique identifier; label non-removable** | **India IT Rules r.3(3)(a)(ii)–(b) — in force since ~20 Feb 2026**; scope arguable | Render path + export manifest; **needs Indian counsel before Indian launch** |
| **Visible synthetic-content disclosure** | EU Art 50(4) — creative-works carve-out applies to game dialogue | UI + export manifest, not in-audio |
| **Provenance log per render** | Already required by scope §12.1 | Free; also the audit trail for a takedown |
| **Designated agent, repeat-violator policy, 14-day counter-notice, staydown** | NO FAKES §2(d)(1)(B) — *if* enacted; §230 would not apply | S9 moderation design brief. Build the hooks now |
| **3-hour takedown SLA** | India IT Rules r.3(1)(d) | S9 moderation |
| **Consent + erasure on speaker data** | DPDP §6; Rules 3, 5–16 from ~May 2027 | Identity store schema — cheap now, expensive later |
| **Speaker-similarity screen vs. a known-persons index** | Recommended, not required | Mint time; reuses the eval harness |
| **Exclude VoxCeleb / AnimeVox-class data** | Required on copyright grounds anyway | §7 decision table |

**The single most valuable thing in this section is a negative finding: VoiceForge's architecture already sits outside the core of every voice-specific liability regime surveyed.** The ELVIS Act needs a "particular, identifiable individual"; NO FAKES needs a product "primarily designed to produce … digital replicas of a specifically identified individual"; Arijit ¶18 condemns tools that "enable the conversion of any voice into that of a celebrity"; EU Art 50(4) deep-fake obligations attach to content resembling "existing persons". **A description-to-novel-voice system with no audio input matches none of these.** That is not luck — it is scope §15.1 doing exactly what it was designed to do. **Do not trade it away**, and note that the moment reference-audio cloning is added, every row above changes character.

---

## 7. THE DECISION TABLE

| Artefact | Intended action | Verdict | Confidence | Lawyer needed? |
|---|---|---|---|---|
| **LibriTTS-R** | train mapper, release weights | ✅ **GO** | HIGH | no |
| **LibriTTS-P** | train mapper (style captions), release weights | ✅ **GO** | MEDIUM-HIGH | no — but email LINE for a LICENSE file |
| **LibriTTS / LibriSpeech** | train, release | ✅ **GO** | HIGH | no |
| **MLS** | train, release | ✅ **GO** | HIGH | no — carry the no-re-identification term forward |
| **Common Voice** | train, release | ✅ **GO** | HIGH | ⚠️ yes, only if we *mirror* it (MDC platform terms) |
| **GLOBE / GLOBE_V2** | train, release, redistribute | ✅ **GO** | HIGH | no |
| **IndicVoices** | train, release | ✅ **GO** | HIGH | no |
| **IndicVoices-R** | train, release | ✅ **GO** | HIGH | no |
| **Rasa** | train, release | ✅ **GO** | HIGH | no |
| **Shrutilipi / Kathbath** | train, release | ✅ GO | MEDIUM | no |
| **IndicSUPERB** | train, release | ⚠️ CAUTION — CC0 over web-crawled prompt text AI4Bharat disclaims owning | MEDIUM | no (eval use only) |
| **Google Crowdsourced Indic** (SLR63–66, 78, 79) | train, release | ⚠️ **HOLD — ShareAlike** | HIGH lic. / UNVERIFIED propagation | **YES** |
| **IITM IndicTTS** | train, release open weights | ⚠️ **HOLD — EULA §2.2 forbids what an open licence grants** | MEDIUM-HIGH | **YES** |
| **SYSPIN / LIMMITS / SPICOR** | train, release | ⛔ **UNVERIFIED — treat as blocked** | UNVERIFIED | **YES** |
| **ParaSpeechCaps** | train mapper, release weights | ⛔ **NO-GO** | HIGH | no — the answer is clear |
| **Expresso / EARS** | train, release | ⛔ **NO-GO** (CC-BY-NC-4.0) | HIGH | only to negotiate a commercial licence from Meta / U. Hamburg |
| **Emilia** (non-YODAS) | train, release | ⛔ **NO-GO** | HIGH | no |
| **Emilia-YODAS** (isolated) | train, release | ⚠️ CAUTION — CC-BY, but must be deliberately separated | MEDIUM | yes if used |
| **VoxCeleb 1/2** | train, release | ⛔ **NO-GO** — research-only + publicity rights + GDPR basis | HIGH | no |
| **TextrolSpeech** | train, release | ⛔ **NO-GO** (ESD research-only) | HIGH | no |
| **SpeechCraft** | train, release | ⛔ **NO-GO** (unlicensed + NC EULA) | HIGH | no |
| **VoicePersona v1 (ours)** | train mapper, release weights, keep published as CC0 | ⛔ **NO-GO as-is — 79% defective; re-tag it** | HIGH | **YES** |
| — `laion/laions_got_talent` | any use | ⛔ **NO-GO** — OpenAI ToU + no valid grant | MEDIUM-HIGH | **YES** |
| — `AnimeVox` | any use | ⛔ **NO-GO** — ripped anime dubs + performer rights | HIGH | no |
| — `AniSpeech` | any use | ⛔ **NO-GO** — MIT over unowned recordings | HIGH | no |
| — `GLOBE_V2` | keep | ✅ **GO** — the salvageable 20.9% | HIGH | no |
| **VoxCPM2** | host publicly, serve output commercially | ✅ **SERVE** (use v2, not v1/1.5) | HIGH | no |
| **Qwen3-TTS + VoiceDesign** | host publicly | ✅ **SERVE** — cleanest in the field | HIGH | no |
| **Chatterbox** | host publicly | ✅ **SERVE** — keep Perth watermark on | HIGH | no |
| **Parler-TTS** | host publicly | ✅ **SERVE** — add CC-BY attribution we ship, which they didn't | HIGH | no |
| **CosyVoice 2/3** | host publicly | ✅ SERVE — **do not install `ttsfrd`** (excluded on architecture, not licence) | MEDIUM-HIGH | no |
| **Zonos-v0.1** — cloning / speaker-embedding path | host publicly | ⚠️ **HOLD — speaker encoder ← VoxBlink2 (CC-BY-NC-SA-4.0), challenged Feb 2025, unanswered** | HIGH facts | **YES** |
| **Zonos-v0.1** — TTS without speaker embedding | host publicly | ✅ SERVE, but it is not the product | MEDIUM | no |
| **VibeVoice** | host publicly | ⛔ **NO-SERVE** — Large withdrawn; audible AI disclaimer in every output | MEDIUM | no |
| **Indic Parler-TTS** | host publicly | ⚠️ **HOLD — the one worth a lawyer hour** (also: read the gate terms) | MEDIUM | **YES** |
| **ai4bharat/indic-conformer-600m** | eval harness only, never served | ⚠️ CAUTION — undeclared provenance | LOW | only if served |
| **IndicF5** | host publicly | ⛔ **NO-SERVE** | HIGH | **YES** (one email may resolve it) |
| **SPRING_F5** | host publicly | ⛔ **NO-SERVE** | HIGH | no |
| **Indic-Mio + MioCodec** | host publicly | ⛔ **NO-SERVE** | HIGH | **YES** |
| **DhVaani-0.5** | host publicly | ⛔ **NO-SERVE** | MEDIUM-HIGH | **YES** |
| **F5-TTS** | host publicly | ⛔ **NO-SERVE** (weights); code MIT is reusable | HIGH | no |
| **VoiceSculptor / Llasa-3B / xcodec2** | host publicly | ⛔ **NO-SERVE** | HIGH | no |
| **XTTS-v2** | host publicly | ⛔ **NO-SERVE** — CPML covers *outputs*; no licensor exists to sell a commercial licence. **Remove from roadmap.** | HIGH | no |
| **IndexTTS-2** | host publicly **or use its output as training data** | ⛔ **NO — both.** Also loads CC-BY-NC-4.0 MaskGCT at inference | HIGH | no |
| **Mapper weights (ours)** | release publicly under Apache-2.0 | ✅ GO **iff** trained only on GO-rows above | HIGH | no |
| **VoicePersona v2 (ours)** | publish as CC0 | ✅ GO **iff** rebuilt per §8 | HIGH | **YES** (one review) |

---

## 8. What this means for the build

### 8.1 The S2 training set — what goes in, what stays out

**Use (English):** LibriTTS-R · LibriTTS-P (style captions over LibriTTS-R audio) · LibriTTS · LibriSpeech · MLS-English · Common Voice · GLOBE / GLOBE_V2.

**Use (Indic):** IndicVoices · IndicVoices-R · Rasa. *(Shrutilipi, Kathbath for auxiliary tasks.)*

**Hold pending counsel:** Google Crowdsourced Indic (ShareAlike) · IITM IndicTTS (§2.2) · SYSPIN / LIMMITS / SPICOR (unverified).

**Exclude, permanently, from anything whose weights ship:** ParaSpeechCaps · Expresso · EARS · Emilia · VoxCeleb 1/2 · TextrolSpeech · SpeechCraft · everything derived from them.

**The consequence, stated plainly.** The excluded set is *exactly* the expressive/emotional/situational-style data. LibriTTS-P gives us rich **intrinsic** speaker traits (pitch, timbre, age, gender, clarity) — which is most of what a character-voice mapper needs — but almost nothing on **performance style** (whispered, shouted, breathy, menacing). The scope doc's §4.4 `Direction` channel is therefore **licence-constrained, not just technically hard**. Three honest options: (a) negotiate a commercial licence for Expresso/EARS directly from Meta and Universität Hamburg; (b) record a small purpose-built expressive corpus; (c) synthesise style data using a backend whose licence permits it — **VoxCPM2 or Qwen3-TTS-VoiceDesign, both Apache-2.0 with no output restriction and no AUP** — which is cheap, legally clean, and the option I would take first.

**Which backends may lawfully be used to generate training data — this is its own gate.** ✅ VoxCPM2, Qwen3-TTS/VoiceDesign, Chatterbox, Parler-TTS, CosyVoice (no output restrictions anywhere). ⛔ **XTTS-v2** (CPML: *"only non-commercial use of a machine learning model **and its outputs**"*, plus *"Use of the model to train other models for commercial use is not a non-commercial purpose"*), **IndexTTS-2** (§3.4(c)), **VibeVoice** (audible disclaimer baked into every render), and **anything on the `False` list**. Add a `usable_as_training_source: bool` alongside `public_servable` — the two are not the same flag, and XTTS-v2/IndexTTS-2 are exactly where they diverge. Cross-ref [10-performance-control.md](10-performance-control.md).

### 8.2 What VoicePersona v2 ships under

**Rebuild, do not patch.** Concretely:

1. **Drop all three defective sources.** LAION-Got-Talent (7,937), AniSpeech (2,000), AnimeVox (1,999) — 11,936 of 15,082 rows.
2. **Keep GLOBE_V2** (3,146 rows) as the seed.
3. **Rebuild to scale from the GO list:** LibriTTS-R + LibriTTS-P for English intrinsic traits; GLOBE_V2 for accent breadth; IndicVoices-R + Rasa for Indic. Re-run the measure-first caption pipeline (scope §8.3) over that audio.
4. **License v2 as CC-BY-4.0, not CC0.** This is a deliberate downgrade in permissiveness and it is the right call: our sources are CC-BY-4.0, and **CC-BY-4.0 §3(a) obligations flow to anyone who Shares the material.** A CC0 stamp on CC-BY-derived audio would repeat the v1 error in a milder form. Use CC0 only for the caption text we author ourselves, and say so explicitly with a dual declaration:

   > *Captions and metadata: CC0-1.0. Audio: CC-BY-4.0, inherited from LibriTTS-R (Google LLC), LibriTTS-P (LINE Corp), GLOBE_V2, IndicVoices-R and Rasa (AI4Bharat). See ATTRIBUTION.md.*

5. **Set the HF `license:` field to a real identifier.** `cc-by-4.0`, not `cc`.
6. **Deal with v1.** It is published and downloadable. At minimum: change the tag from `cc` to `other`, add a prominent card notice describing the upstream defects, and remove the CC0 badge. Whether to unpublish is a lawyer question — but leaving a CC0 claim standing over OpenAI output and ripped anime is the worst of the available options, and it is the one currently in force.

### 8.3 What the mapper ships under

**Apache-2.0**, conditional on the training set being drawn only from §7 GO rows.

The mapper is 10–30M parameters trained on (description, audio) pairs. If every pair comes from CC-BY-4.0 / CC0 / PD sources, there is no NC or SA term anywhere in the chain and Apache-2.0 is clean. **The moment one NC row enters, that is no longer true** — which is why this needs to be a mechanical gate, not a habit (§8.5).

Ship alongside the weights:
- `ATTRIBUTION.md` — every CC-BY corpus, its licensor, a licence link, and **a statement of what we changed** (CC-BY-4.0 §3(a)(1)(B) requires "indicate if You modified the Licensed Material").
- A model card with a **per-dataset licence table**, following Indic Parler-TTS's format rather than Parler-TTS's citation-only norm.
- The eval slices by demographic that scope §15.2 already commits to.

### 8.4 The `public_servable` values, as code

```python
# adapters/registry.py — the licence gate of scope §12.5.
# Rationale for every False is in RESEARCH/08-licensing-propagation.md §4.
# Verified 2026-09-02. Re-verify on every backend version bump: these fields
# are mutable, and IndexTTS-2 proves they change (see §4.10).

PUBLIC_SERVABLE: dict[str, bool] = {
    # ── Apache-2.0 / MIT, full upstream chain traced clean ─────────────
    "voxcpm2":        True,   # Apache-2.0; MiniCPM-4 backbone. NOT VoxCPM-0.5B/1.5.
    "qwen3-tts":      True,   # Apache-2.0 across the family; no AUP, no output clause
    "qwen3-tts-vd":   True,   # VoiceDesign variant, Apache-2.0
    "chatterbox":     True,   # MIT; Llama arch only, no Meta weights. Keep Perth on.
    "parler-tts":     True,   # Apache-2.0; CC-BY-4.0 data only
    "cosyvoice2":     True,   # Apache-2.0. Do NOT install the `ttsfrd` wheel.

    # ── Blocked: upstream chain defect ─────────────────────────────────
    "zonos-v0.1":     False,  # §4.12 speaker encoder <- VoxBlink2 (CC-BY-NC-SA-4.0);
                              #   challenged on-repo Feb 2025, never answered.
                              #   Flip only with a written Zyphra warranty, or after
                              #   substituting a cleanly-licensed speaker encoder.
    "indic-parler":   False,  # §4.7 CONDITIONAL - IITM EULA 2.2 + unread gate terms
    "indicf5":        False,  # §4.2 fine-tune of F5-TTS (CC-BY-NC-4.0)
    "spring-f5":      False,  # §4.3 declares base_model: SWivid/F5-TTS
    "indic-mio":      False,  # §4.4 Expresso (NC) direct; Emilia (NC) via base + codec
    "dhvaani":        False,  # §4.5 ZipVoice weights undeclared; IndicTTS; Emilia
    "f5-tts":         False,  # weights CC-BY-NC-4.0 (code is MIT)
    "voicesculptor":  False,  # §4.1 Llasa-3B (NC) + xcodec2 (NC) + Llama 3.2 terms
    "llasa-3b":       False,  # CC-BY-NC-4.0, "prohibits free commercial use"
    "xcodec2":        False,  # CC-BY-NC-4.0 (both xcodec2 and xcodec2-hf)
    "xtts-v2":        False,  # CPML covers OUTPUTS; no licensor exists. Remove.
    "indextts2":      False,  # bilibili MULA; loads CC-BY-NC-4.0 MaskGCT at inference;
                              #   3.4(c) also bars using its OUTPUT as training data
    "vibevoice":      False,  # §4.11 Large withdrawn; audible AI disclaimer per render

    # ── Recorded rejections: the obvious Indic fallbacks, so nobody re-proposes them
    "mms-tts":        False,  # facebook/mms-tts-* — CC-BY-NC-4.0
    "seamless-m4t":   False,  # facebook/seamless-m4t-v2-large — CC-BY-NC-4.0
}
```

**Four hard rules to enforce in code, not prose:**

1. **Fail closed.** A backend absent from `PUBLIC_SERVABLE` must raise on load in the public deployment, never default to `True`. An unlicensed artefact and an unknown artefact carry the same risk.
2. **Assert at load, not at config-parse.** The scope doc is right that "a licence rule that lives only in a document will eventually be violated by a config change" — so put the check in the adapter's `__init__`, where a config change cannot route around it.
3. **Gate the *checkpoint*, not the adapter.** Zonos and IndexTTS-2 both prove that the blocking artefact can be a *sub-checkpoint fetched at runtime*, invisible from the top-level model id. Maintain an allowlist of permitted HF repo ids and **assert on every `hf_hub_download` / `from_pretrained` call in the serving path**, so a model that quietly starts pulling a new codec fails loudly rather than silently.
4. **Snapshot the evidence at ingest.** Nine of the twelve weight repos audited here ship **no LICENSE file at all** — the grant is a single mutable YAML line. Store the model card and API response, timestamped, alongside the pinned revision hash. If a licence changes under us, we need to be able to show what it said when we adopted it.

### 8.5 The gate the scope document is missing

`public_servable` guards Q2. **Nothing currently guards Q3** — and Q3 is the one that costs a retrain. Add the mirror:

```python
@dataclass(frozen=True)
class Corpus:
    name: str
    licence: str            # SPDX id, or "other:<name>"
    train_releasable: bool  # may weights trained on this ship publicly?
    attribution: str | None # required NOTICE line, if any
    source_url: str
```

The training script asserts `all(c.train_releasable for c in mix)` before the first optimiser step, and **emits `ATTRIBUTION.md` from the mix itself** — so the attribution file cannot drift from what was actually trained on. Record the resolved mix in the checkpoint metadata alongside `backend_version`, for the same reason scope §12.1 records `backend_id`: without it, a routine data change silently alters what the released weights may be licensed under.

### 8.6 Attribution — what actually has to ship, and where

CC-BY-4.0 §3(a)(1), verbatim, is the requirement:

> "If You Share the Licensed Material (including in modified form), You must: retain the following if it is supplied by the Licensor with the Licensed Material: identification of the creator(s)…; a copyright notice; a notice that refers to this Public License; a notice that refers to the disclaimer of warranties; a URI or hyperlink to the Licensed Material…; **indicate if You modified the Licensed Material and retain an indication of any previous modifications**; and indicate the Licensed Material is licensed under this Public License, and include the text of, or the URI or hyperlink to, this Public License."

And §3(a)(2):

> "You may satisfy the conditions in Section 3(a)(1) in any reasonable manner based on the medium, means, and context in which You Share the Licensed Material. For example, it may be reasonable to satisfy the conditions by providing a URI or hyperlink to a resource that includes the required information."

**Two distinct obligations, two different places:**

- **Sharing the *dataset* (VoicePersona v2) → §3(a) applies squarely.** Ship `ATTRIBUTION.md` in the dataset repo with all seven elements per corpus, including "indicate if You modified" — for us that means naming the resampling, filtering, and caption-generation steps.
- **Sharing the *weights* → only if weights are "Adapted Material" (§5, unsettled).** The field's practice is to name datasets in the card and request citation. **Do §3(a) properly anyway** — it costs one generated file and means the §5 question never has to be litigated for our artefact.

**The good news is that CC tells us the bar is low for models.** Per their AI-training guidance: *"For AI model training, attribution could be **a simple link to the source of the dataset** used to train the model."* So a model card that links each training corpus, plus a generated `ATTRIBUTION.md`, is comfortably above what CC themselves describe as sufficient. There is no reason to skip it.

**Where the files go:**

| Artefact | File | Content |
|---|---|---|
| VoicePersona v2 (HF dataset) | `ATTRIBUTION.md` + card table | Per-corpus: creator, copyright notice, licence link, warranty-disclaimer notice, source URI, modifications made |
| Mapper weights (HF model) | `ATTRIBUTION.md` + card table | Same, generated from the training mix |
| Mapper weights | `LICENSE` | Apache-2.0 |
| Web app | `/licenses` page | Per-backend: model, licence, link; plus the dataset attributions |
| Rendered audio export | manifest JSON | `backend_id`, `backend_version`, synthetic-content declaration, watermark id |

The `/licenses` page is not decoration: CC-BY §3(a)(2) explicitly blesses satisfying the conditions by hyperlink, and a public service is the "medium, means, and context" in which we Share.

### 8.7 The English track — the Zonos problem is architectural, not cosmetic

Scope §6.1 says: *"English / two-tower research track → Zonos. Only backend where description → 128-d vector → frozen renderer is directly implementable."* **That plan now has a licence dependency it did not know about**, and the dependency is on the exact component that makes Zonos special.

The speaker encoder is not incidental to the design — it *is* the design. Tier-1 identity is "a 128-d vector in the encoder's space", and §12.1 records that a Tier-1 vector *"is only meaningful relative to the model that produced it"*. So the encoder is the most expensive thing in the system to swap: every minted identity is defined relative to it.

**Which argues for settling this at S1, before any identity is minted, not at S7 when the service goes public.** Three routes:

1. **Get a written warranty from Zyphra** that the speaker-embedding checkpoint is Apache-2.0 notwithstanding VoxBlink2's CC-BY-NC-SA-4.0 terms — or that the encoder was retrained on other data. Cheapest, but the existing 18-month-old unanswered thread is not encouraging, and the repo has been unmaintained since March 2025.
2. **Substitute a cleanly-licensed speaker encoder** and re-derive the manifold. This is the option to cost out. It decouples the identity layer from an abandoned repo *and* from an NC upstream, and the two-tower design is agnostic to which encoder defines the space as long as one is chosen and pinned. The cost is that the encoder must be at least as navigable-by-synthesis as ResNet293-SimAM-ASP — which is scope question A2 and must be measured either way.
3. **Fall back to Tier-2 seed-clip identity** on a clean backend (VoxCPM2 or Qwen3-TTS-VoiceDesign, both of which do description-conditioned voice design natively). The scope doc already designs this fallback in. Note that Qwen3-TTS-VoiceDesign did not exist as a known option when §6 was written and is arguably a better *product* fit than Zonos regardless of licence — description-native, Apache-2.0, zero riders.

**The honest summary: the licence audit has weakened the case for the Zonos-specific two-tower design and strengthened the case for description-native Apache-2.0 backends.** That is a §00/§02 conversation, not a §08 one, but it originates here.

### 8.8 The Indic track, restated

This is the part of the plan the audit changes most, so it is worth saying directly.

**The Indic *data* is fine** — IndicVoices, IndicVoices-R and Rasa are genuinely CC-BY-4.0 with an express commercial grant in the parent paper. **The Indic *models* are the problem.** Every high-quality Indic TTS checkpoint in the field — IndicF5, SPRING_F5, Indic-Mio, DhVaani — traces back to an NC artefact, and the only one that does not (Indic Parler-TTS) has an unresolved IndicTTS dependency.

So the Indic track has three routes, in order of preference:

1. **Clear Indic Parler-TTS.** One question to IITM/AI4Bharat about §2.2 and the CC-BY-4.0 re-designation. If it clears, the scope doc's plan works as written. **Cheapest by a wide margin — do this first.**
2. **Use a clean multilingual backend that already has Hindi.** VoxCPM2 (30 languages incl. Hindi) and Chatterbox (23 incl. Hindi) are both unambiguously servable. Quality against Indic-specialist models is an empirical question for [04-indic-track.md](04-indic-track.md), not a licence one.
3. **Train our own Indic renderer** on IndicVoices-R + Rasa from a clean base (Qwen3-TTS or Parler-TTS architecture, both Apache-2.0). Expensive, and only justified if 1 and 2 both fail.

**What is no longer available:** treating Indic-Mio + MioCodec + DhVaani as the Apache-2.0/MIT critical path. That was the sibling's provisional plan and this audit removes it.

**And two obvious-looking fallbacks are also closed**, so nobody re-proposes them: **`facebook/mms-tts-*`** (1,100+ languages, the standard route to broad Indic coverage) and **`facebook/seamless-m4t-v2-large`** are both **CC-BY-NC-4.0**. They are the first thing anyone reaches for when an Indic gap appears, and hosting either commercially is a direct breach on Meta's own terms — no unsettled legal theory required. Add them to `PUBLIC_SERVABLE` as explicit `False` entries rather than leaving them absent, so the rejection is documented rather than rediscovered.

---

## 9. Open — must be settled by counsel or by asking the authors

| Question | Who to ask | Exactly what to ask | What it blocks |
|---|---|---|---|
| Does IITM IndicTTS EULA §2.2 bar an Apache-2.0 weight release? | **Counsel**, then IIT Madras Speech Technology Consortium (Hema A. Murthy / S. Umesh) | "Your EULA §2.2 requires that third parties receiving a Derivative Work not further sell or sub-license it. Does releasing model weights trained on IndicTTS under Apache-2.0 breach §2.2? If AI4Bharat/SPRINGLab have re-designated the corpus CC-BY-4.0, can you confirm that in writing?" | Indic Parler-TTS · Indic-Mio · DhVaani · the whole Indic track |
| Is the `license.pdf` I recovered the authentic, current IITM EULA? | IIT Madras Donlab | Send the PDF hash and ask for confirmation + the current version | Everything in the row above |
| Did AI4Bharat obtain a separate permission from the F5-TTS authors? | AI4Bharat (`safikhan`), via HF discussion #34 | "IndicF5 is described in arXiv 2505.20693 as a fine-tune of English F5-TTS, whose weights are CC-BY-NC-4.0. On what basis is the MIT grant made? Was a separate licence obtained from the F5-TTS authors?" | IndicF5 servability (a "yes" flips it) |
| Does a model trained on CC-BY-NC data inherit NC? | **Counsel** | Show them CC's own guidance (*"all stages, from copying the data during training to sharing the trained model, must not be for commercial gain"*) and ask how much weight it carries. Frame around the actual exposure: contract terms (Emilia cl. 6, VoxCeleb EULA, IITM §2.2, IndexTTS-2 §3.4(c)) are enforceable regardless; the copyright question is secondary | Whether the NC research lane can ever be commercialised; Zonos; Indic-Mio |
| Does CC-BY-SA propagate from Google Crowdsourced Indic — and from **VoxBlink2** via the Zonos encoder — to our weights? | **Counsel** | Same question, SA flavour, and CC's published reading is adverse: *"would require AI developers to use the same CC license as the original works"* | Any Indic model trained on SLR63–66/78/79; **and the Zonos speaker encoder** |
| **OpenAI ToU verbatim text** — is redistributing/training on GPT-4o Audio output a breach? | **Counsel** | openai.com returns 403 to automated fetch. Get the current operative Terms of Use + Business Terms text and read the competing-models clause and Output-ownership clause against our facts | VoicePersona v1's status; whether v1 must be unpublished |
| Must VoicePersona v1 be unpublished, or is re-tagging enough? | **Counsel** | Present §3.13 in full | Reputational and legal exposure that is accruing today |
| SYSPIN / LIMMITS / SPICOR licences | IISc SYSPIN project; LIMMITS organisers; SPRING Lab | "What licence governs the corpus, and may models trained on it be released commercially?" | Indic-Mio and DhVaani reinstatement; any use of these corpora |
| **Does India's IT Rules r.2(1)(wa) reach a generic, non-cloned synthetic voice?** | **Counsel (Indian)** | The definition needs the audio to "appear to be real, authentic or true" **and** "depict or portray any individual or event". Does a fictional game character's voice qualify? Does proviso (c) (accessibility/clarity/quality) help? | **Whether every rendered file needs a prefixed spoken disclosure in India.** Product-defining |
| **Was the *Arijit Singh* injunction extended past 3 Sept 2024?** | **Counsel (Indian)** / Bombay HC records | ¶32 says the order runs only to that date. It is the flagship AI-voice-cloning holding and by its own terms it expired | How much weight ¶18 carries in an Indian risk assessment |
| Do Indian **performers' rights** (Copyright Act §§2(qq), 38A, 39) attach to speech-corpus contributors, and does §39(c) carry *ANI*'s §52(1)(a) reasoning across? | **Counsel (Indian)** | No authority found either way | A rights layer separate from copyright on every Indic corpus |
| Is a **voice recording "personal data" under DPDP §2(t)**, and does §3(c)(ii) exempt third-party-uploaded audio? | **Counsel (Indian)** | §3(c)(ii) requires the *data principal* to have published it, or an Indian statutory obligation. No judicial interpretation exists | Whether any scraped Indic audio is usable at all |
| Does Art. 50(2) marking need C2PA, a watermark, or both — and which detector? | **engineering + counsel** | Guidelines para (76) permits our own detection solution *only while* no standard exists; C2PA 2.2 has no audio-specific binding guidance | The AudioSeal/Perth decision in [09](09-safety-and-watermarking.md) — now a compliance decision, not just safety |
| Should we sign the **Code of Practice on Transparency of AI-generated Content**? | **Counsel** | It is voluntary and confers no presumption of conformity, but ~190 orgs signed and it demonstrates compliance | Regulatory posture; cheap signalling |
| NO FAKES Act — floor action after 2026-08-10 | **Counsel (US)** | S.4591 sits on the Senate calendar; confirm status before US launch | Whether §2(d)(1)(B)'s designated-agent/staydown machinery becomes mandatory |
| ⏰ ***Kneschke v. LAION*, BGH `I ZR 281/25` — hearing 3 September 2026, i.e. TOMORROW** | watch, this week | *"Erstellen eines Datensatzes für KI-Training"*. Germany's supreme court on whether scraping to build a training set is lawful, and what counts as a machine-readable reservation | **The EU scraping analysis in §5. Cheapest high-value update in this file** |
| *Thomson Reuters v. Ross*, 3d Cir. No. 25-2153 | watch | Argued 11 June 2026, no decision. **First federal appellate ruling on AI-training fair use** | The whole §5 analysis |
| *GEMA v. Suno* appeal (LG München I, 31 July 2026, not final) | watch | A European court held that **offering an audio-generation model to the public** is itself infringing | Direct precedent for our business model |
| **Zonos speaker encoder: does Apache-2.0 survive VoxBlink2's CC-BY-NC-SA-4.0?** | **Counsel**, then Zyphra (bump [discussion #2](https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding/discussions/2)) | "Your card states the encoder is based on VoxBlink2's ResNet293-SimAM-ASP pretrained models, and VoxBlink2 is CC-BY-NC-SA-4.0. On what basis is the Apache-2.0 grant made? Was the encoder retrained on other data?" | **Scope §7's entire English two-tower track.** Settle at S1, before identities are minted |
| Does `Zyphra/ZONOS2` inherit the same encoder lineage? | check the code | Grep ZONOS2's inference path for `hf_hub_download` | Whether ZONOS2 is an escape hatch or the same problem |
| Which cleanly-licensed speaker encoder is navigable-by-synthesis? | **experiment, not counsel** | Run scope A2's off-manifold test against 2–3 candidates (WeSpeaker / 3D-Speaker ECAPA variants) | The Zonos substitution route (§8.7 option 2) |
| Undisclosed training corpora: VoxCPM2 (2M hr), Zonos (200k hr), Chatterbox (0.5M hr) | the vendors | "Can you disclose or warrant the licence status of the training data?" | Nothing today — but it is the residual risk under every clean Apache-2.0 backend, and worth asking |
| Common Voice Data Consumer License inside MDC | **Counsel** | Read the specific licence attached to CV in Mozilla Data Collective before mirroring | Whether we may mirror CV audio |
| HF gate text behind login for AI4Bharat datasets **and models** | anyone with an HF account | Log in, screenshot the gate for `indicvoices_r`, `IndicVoices`, `Rasa`, **`indic-parler-tts`**, `indic-parler-tts-pretrained`, `IndicF5`, `DhVaani-0.5`, `indic-conformer-600m` | **10-minute check, and it blocks Indic Parler-TTS servability.** Do it first |
| `ai4b-hf/GLOBE-annotated` declares `license: None`; the Indic Parler card's `"CC V1"` is not a real identifier | AI4Bharat | "What licence governs GLOBE-annotated, and does 'CC V1' in the training table mean CC0 1.0?" | Indic Parler-TTS chain completeness |
| LibriTTS-P proper LICENSE file | LINE Corporation | "Your README grants CC-BY-4.0 but the repo has no LICENSE file — can you add one?" | Nothing; raises confidence MEDIUM-HIGH → HIGH |
| What does CC-BY-4.0's grant actually cover in the LibriVox chain — PD audio or the contributor's own work? | — | Probably unanswerable; note it and comply conservatively | Nothing; complying is strictly conservative |

---

## 10. Sources

| # | URL | Type | Used for | Confidence in source |
|---|---|---|---|---|
| 1 | https://huggingface.co/api/models/HKUSTAudio/Llasa-3B | HF API (primary metadata) | Llasa-3B CC-BY-NC-4.0; base_model Llama-3.2-3B-Instruct | HIGH |
| 2 | https://huggingface.co/HKUSTAudio/Llasa-3B/raw/main/README.md | raw model card | "prohibits free commercial use" disclaimer, verbatim | HIGH |
| 3 | https://huggingface.co/HKUSTAudio/xcodec2/raw/main/README.md | raw model card | xcodec2 `license: cc-by-nc-4.0` | HIGH |
| 4 | https://huggingface.co/api/models/HKUSTAudio/xcodec2-hf | HF API | xcodec2-hf also CC-BY-NC-4.0 (no relicensing) | HIGH |
| 5 | https://huggingface.co/ASLP-lab/VoiceSculptor-VD/raw/main/README.md | raw model card | Apache-2.0 over `base_model: HKUSTAudio/Llasa-3B` | HIGH |
| 6 | `gh api repos/ASLP-lab/VoiceSculptor` | GitHub API | code Apache-2.0 | HIGH |
| 7 | https://huggingface.co/api/models/meta-llama/Llama-3.2-3B-Instruct | HF API | `license: llama3.2`, `gated: manual` | HIGH |
| 8 | https://huggingface.co/api/models/ai4bharat/IndicF5 | HF API | MIT declaration; datasets; `gated: auto`; lastModified | HIGH |
| 9 | https://huggingface.co/ai4bharat/IndicF5/discussions/34 | HF discussion (via API) | AI4Bharat member's verbatim commercial-use assurance, 2026-03-03 | HIGH |
| 10 | https://arxiv.org/abs/2505.20693 + /html/2505.20693v1 | paper (primary) | "fine-tuning English F5 on Indian data"; footnote 2 links IN-F5 → ai4bharat/IndicF5; IN11 data mix | HIGH |
| 11 | `gh api repos/AI4Bharat/IndicF5` | GitHub API | `license: null` — unlicensed | HIGH |
| 12 | https://huggingface.co/api/models/SWivid/F5-TTS | HF API | weights CC-BY-NC-4.0; trained on Emilia | HIGH |
| 13 | `gh api repos/SWivid/F5-TTS` | GitHub API | code MIT | HIGH |
| 14 | https://huggingface.co/SPRINGLab/SPRING_F5/raw/main/README.md | raw model card | `license: apache-2.0` + `base_model: SWivid/F5-TTS` | HIGH |
| 15 | https://huggingface.co/SPRINGLab/Indic-Mio/raw/main/README.md | raw model card | Apache-2.0; Expresso/IndicTTS/Syspin/SPICOR training statement | HIGH |
| 16 | https://huggingface.co/datasets/ylacombe/expresso/raw/main/README.md | raw dataset card | Expresso `license: cc-by-nc-4.0` | HIGH |
| 17 | https://speechbot.github.io/expresso/ | project page (Meta) | "distributed under the CC BY-NC 4.0 license" | HIGH |
| 18 | https://huggingface.co/Aratako/MioTTS-0.6B/raw/main/README.md | raw model card | Apache-2.0; Emilia + hifitts-2; Qwen3-0.6B-Base; family licence table | HIGH |
| 19 | https://huggingface.co/Aratako/MioCodec-25Hz-24kHz/raw/main/README.md | raw model card | MIT; Emilia; Kanade-Tokenizer; WavLM-base+ | HIGH |
| 20 | `gh api repos/frothywater/kanade-tokenizer` | GitHub API | `license: null` — unlicensed | HIGH |
| 21 | https://huggingface.co/api/datasets/amphion/Emilia-Dataset | HF API | tag `cc-by-4.0` **and** the full `extra_gated_prompt` saying CC-BY-**NC** | HIGH |
| 22 | https://raw.githubusercontent.com/open-mmlab/Amphion/main/preprocessors/Emilia/README.md | source repo | "only for non-commercial purposes under the CC BY-NC-4.0 license" | HIGH |
| 23 | https://huggingface.co/ARTPARK-IISc/DhVaani-0.5 | model card (page) | "Apache-2.0, following the base model… respect the licenses of the training corpora" | MEDIUM-HIGH |
| 24 | https://huggingface.co/api/models/k2-fsa/ZipVoice | HF API | **no licence field**; trained on Emilia | HIGH |
| 25 | `gh api repos/k2-fsa/ZipVoice` | GitHub API | code Apache-2.0 | HIGH |
| 26 | https://huggingface.co/ai4bharat/indic-parler-tts | model card (page) | Apache-2.0 + the per-dataset licence table | MEDIUM-HIGH |
| 27 | https://huggingface.co/datasets/thennal/indic_tts_ml/resolve/main/license.pdf | **IITM End User License Agreement (primary)** | §2.1 grant, §2.2 downstream bar, §5 notice, §3, §4 | MEDIUM-HIGH (mirror-sourced) |
| 28 | https://www.openslr.org/{63,64,65,66,78,79}/ | OpenSLR pages | Google Crowdsourced Indic = **CC BY-SA 4.0** | HIGH |
| 29 | https://www.openslr.org/{12,60,141,94}/ | OpenSLR pages | LibriSpeech / LibriTTS / LibriTTS-R / MLS = CC BY 4.0 | HIGH |
| 30 | https://www.openslr.org/resources/141/doc.tar.gz → `LICENSE.txt` | licence file (primary) | LibriTTS-R CC-BY-4.0, Google LLC, verbatim | HIGH |
| 31 | https://librivox.org/pages/public-domain/ | project page | PD dedication; "even to sell them"; US-only caveat | HIGH |
| 32 | https://wiki.librivox.org/index.php?title=Copyright_and_Public_Domain | project wiki | AI training listed as accepted use; refusal of CC explained | HIGH |
| 33 | https://raw.githubusercontent.com/line/LibriTTS-P/main/README.md | raw README | the only CC-BY-4.0 grant; repo `license: null` | MEDIUM-HIGH |
| 34 | https://huggingface.co/datasets/ajd12342/paraspeechcaps/raw/main/README.md | raw dataset card | CC-BY-NC-SA-4.0; "The dataset **and models**…"; audio-not-included note | HIGH |
| 35 | https://huggingface.co/api/models/ajd12342/parler-tts-mini-v1-paraspeechcaps | HF API | authors' own trained checkpoint released CC-BY-NC-SA-4.0 | HIGH |
| 36 | https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/LICENSE | licence file | EARS CC-BY-NC-4.0 | HIGH |
| 37 | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html | terms page | metadata CC-BY-**SA**-4.0; audio/video/metadata withdrawn | HIGH |
| 38 | https://mm.kaist.ac.kr/datasets/voxceleb/files/license.txt | EULA | "copyright … remains with the original owners"; conditions propagate on redistribution | HIGH |
| 39 | https://www.robots.ox.ac.uk/~vgg/terms/url-lists-privacy-notice.html | privacy notice | UK GDPR Art. 14(5)(b) research basis | HIGH |
| 40 | https://raw.githubusercontent.com/HLTSingapore/Emotional-Speech-Data/master/README.md | README | ESD: "can only be used for research purpose" | HIGH |
| 41 | https://raw.githubusercontent.com/thuhcsi/SpeechCraft/master/Emphasis-SpeechCraft-EULA.pdf | EULA | "solely for academic, non-commercial research purposes"; derivative-ownership claim | HIGH |
| 42 | https://arxiv.org/html/2308.14430v1 | paper | TextrolSpeech upstream list incl. ESD | MEDIUM-HIGH |
| 43 | https://raw.githubusercontent.com/mozilla/legal-docs/prod/en/common_voice_terms.md | terms (primary) | CC0 grant; "we ask that you not post, distribute, or mirror" | HIGH |
| 44 | https://datacollective.mozillafoundation.org/terms | platform ToS | §2.b(viii)–(ix) re-identification and mirroring bars | MEDIUM-HIGH |
| 45 | https://huggingface.co/datasets/facebook/multilingual_librispeech/raw/main/README.md | raw dataset card | MLS licence + no-re-identification term | MEDIUM |
| 46 | https://arxiv.org/abs/2409.05356 | paper | IV-R "released under the same CC BY 4.0 license" | HIGH |
| 47 | https://arxiv.org/abs/2403.01926 | paper | IndicVoices "CC-BY-4.0 license, **allowing commercial usage**" | HIGH |
| 48 | https://raw.githubusercontent.com/AI4Bharat/IndicVoices-R/master/LICENSE.md | licence file | full CC-BY-4.0 text | HIGH |
| 49 | https://huggingface.co/api/datasets/ai4bharat/Rasa | HF API | CC-BY-4.0; scale; gated auto | HIGH |
| 50 | https://arxiv.org/abs/2505.18609 | paper | RASMALAI → IndicParlerTTS; dataset not released | MEDIUM |
| 51 | https://huggingface.co/api/datasets/Paranoiid/VoicePersona | HF API | **`license: "cc"`** — not CC0; audio feature; splits | HIGH |
| 52 | https://huggingface.co/datasets/laion/laions_got_talent/raw/main/README.md | raw dataset card | "the OpenAI Voice API via Hyprlab"; 110 hours | HIGH |
| 53 | https://huggingface.co/api/datasets/laion/laions_got_talent | HF API | 483 tarballs across 11 OpenAI voice names; Apache-2.0 LICENSE file present; **no licence field** | HIGH |
| 54 | https://huggingface.co/datasets/taresh18/AnimeVox/raw/main/README.md | raw dataset card | "sourced from official English-dubbed versions of popular anime series"; `license: cc` | HIGH |
| 55 | https://huggingface.co/datasets/ShoukanLabs/AniSpeech/raw/main/license | licence file | literal MIT text, "Copyright 2023 ShoukanLabs", over anime audio | HIGH |
| 56 | https://huggingface.co/datasets/MushanW/GLOBE_V2/raw/main/README.md | raw dataset card | `cc0-1.0`; `source_datasets: common_voice_14_0` | HIGH |
| 57 | https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt | CC0 legal code | §4(c) "Affirmer disclaims responsibility for clearing rights of other persons" | HIGH |
| 58 | https://creativecommons.org/licenses/by-nc/4.0/legalcode.en | CC legal code | "NonCommercial means…", "Adapted Material means…", "Share means…" | HIGH |
| 59 | https://creativecommons.org/licenses/by/4.0/legalcode.en | CC legal code | §3(a)(1)–(2) attribution requirements, verbatim | HIGH |
| 60 | https://github.com/index-tts/index-tts/blob/main/LICENSE | bespoke licence (primary) | bilibili MULA §1.5, §2.2, §3.4(a)(b)(c), §4.1–4.3, §5.3, §6, §9 | HIGH |
| 61 | https://huggingface.co/openbmb/VoxCPM2/raw/main/README.md | raw model card | Apache-2.0 "free for commercial use"; MiniCPM-4; 2M+ hr undisclosed; impersonation advisory | HIGH |
| 62 | https://huggingface.co/api/models?search=Qwen3-TTS | HF API search | whole Qwen3-TTS family incl. **VoiceDesign** tagged `license:apache-2.0` | HIGH |
| 63 | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/raw/main/README.md | raw model card | `license: apache-2.0`; voice-design capability | HIGH |
| 64 | `gh api repos/Zyphra/Zonos/contents/LICENSE` | licence file | Apache 2.0 text | HIGH |
| 65 | https://huggingface.co/api/models/Zyphra/Zonos-v0.1-{transformer,hybrid} | HF API | both `apache-2.0` | HIGH |
| 66 | https://huggingface.co/ResembleAI/chatterbox/raw/main/README.md | raw model card | MIT; Perth watermarking statement; disclaimer; Llama 3 acknowledgement | HIGH |
| 67 | https://huggingface.co/api/models/coqui/XTTS-v2 | HF API | `license: other`, `license_name: coqui-public-model-license` | HIGH |
| 68 | https://huggingface.co/api/models/microsoft/VibeVoice-1.5B + `gh api repos/microsoft/VibeVoice` | HF API + GitHub API | **MIT**, contradicting the scope table's "research-only" | HIGH |
| 69 | https://huggingface.co/api/models/FunAudioLLM/CosyVoice2-0.5B + `gh api repos/FunAudioLLM/CosyVoice` | HF API + GitHub API | Apache-2.0 code and weights | HIGH |
| 70 | https://huggingface.co/parler-tts/parler-tts-large-v1 + its API file listing | model card + HF API | Apache-2.0; CC-BY-4.0 data only; **no NOTICE/ATTRIBUTION file shipped** | HIGH |
| 71 | https://huggingface.co/api/datasets/parler-tts/{mls_eng,libritts_r_filtered,mls-eng-speaker-descriptions} | HF API | all `cc-by-4.0` | HIGH |
| 72 | https://huggingface.co/api/datasets?author=SPRINGLab | HF API | 3 of 14 IndicTTS mirrors carry a licence tag; 11 carry none | HIGH |
| 73 | https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil/raw/main/README.md | raw dataset card | "derived from the Indic TTS Database… Speech Technology Consortium at IIT Madras" | HIGH |
| 74 | https://huggingface.co/api/models/ai4bharat/indic-conformer-600m-multilingual | HF API | MIT; **no base_model, no datasets declared** | HIGH |
| 75 | https://openai.com/policies/{row-terms-of-use,may-2025-business-terms,services-agreement} | vendor terms | competing-models restriction — **fetch blocked (HTTP 403); NOT quoted** | **UNVERIFIED** |
| 76 | https://huggingface.co/datasets/mythicinfinity/libritts_r/raw/main/README.md | raw dataset card | the NC-ND BibTeX trap (arXiv posting licence ≠ data licence) | HIGH |
| 77 | https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/speaker_cloning.py | **source code (primary)** | hard-coded `hf_hub_download("Zyphra/Zonos-v0.1-speaker-embedding", "ResNet293_SimAM_ASP_base.pt")` | HIGH |
| 78 | https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding/blob/main/README.md | model card | "based on the ResNet293-SimAM-ASP models from VoxBlink2… We use the pretrain models" | HIGH |
| 79 | https://raw.githubusercontent.com/VoxBlink2/ScriptsForVoxBlink2/main/LICENSE | **licence file (primary)** | VoxBlink2 = **CC BY-NC-SA 4.0**, binding on "modifications or redistributions in any form" | HIGH |
| 80 | https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding/discussions/2 | HF discussion | the licence conflict raised 13 Feb 2025, **never answered** | HIGH |
| 81 | https://huggingface.co/coqui/XTTS-v2/raw/main/LICENSE.txt | **CPML 1.0.0 text (primary)** | "only non-commercial use of a machine learning model **and its outputs**"; Notices passthrough; no-training-for-commercial clause | HIGH |
| 82 | https://github.com/coqui-ai/TTS/discussions/4304 | project discussion (maintainer) | "there is no way of obtaining a commercial license" | HIGH |
| 83 | https://raw.githubusercontent.com/index-tts/index-tts/main/indextts/utils/model_download.py | **source code (primary)** | mandatory runtime fetch of `amphion/MaskGCT` | HIGH |
| 84 | https://huggingface.co/amphion/MaskGCT/raw/main/README.md | raw model card | `license: cc-by-nc-4.0`, trained on Emilia | HIGH |
| 85 | https://raw.githubusercontent.com/microsoft/VibeVoice/main/README.md | repo README | "we have removed the VibeVoice-TTS code from this repository" (2025-09-05) | HIGH |
| 86 | https://huggingface.co/microsoft/VibeVoice-1.5B/raw/main/README.md | raw model card | "limited to research purpose use"; "not intended or licensed for"; **audible AI disclaimer per output** | HIGH |
| 87 | https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/LICENSE | licence file | Apache-2.0, 11,343 bytes; **no Tongyi/Qwen-Research licence anywhere** | HIGH |
| 88 | https://raw.githubusercontent.com/resemble-ai/chatterbox/master/LICENSE | licence file | MIT, "Copyright (c) 2025 Resemble AI" | HIGH |
| 89 | `src/chatterbox/models/t3/{t3.py,llama_configs.py}` | source code | `LlamaModel(LlamaConfig(**config_dict))`, 1024/30 — architecture only, no Meta weights | HIGH |
| 90 | https://huggingface.co/openbmb/{VoxCPM-0.5B,VoxCPM1.5}/raw/main/README.md | raw model cards | the "research and development purposes only" line **absent from VoxCPM2** | HIGH |
| 91 | https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/raw/main/README.md | raw model card | CosyVoice 3 = `license: apache-2.0`; optional proprietary `ttsfrd` wheel | MEDIUM-HIGH |
| 92 | https://huggingface.co/api/models/IndexTeam/IndexTTS-2/commits/main | HF commit log | `LICENSE.txt` added **2026-01-20**; repo had no licence file for 7 months | HIGH |
| 93 | https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/ | **Creative Commons' own guidance (primary)** | NC: *"all stages … to sharing the trained model, must not be for commercial gain"*; SA: *"the same CC license as the original works"*; BY: *"a simple link to the source of the dataset"*; and the overcompliance caveat | HIGH |
| 94 | https://creativecommons.org/2025/05/15/understanding-cc-licenses-and-ai-training-a-legal-primer/ + the [May 2025 PDF](https://creativecommons.org/wp-content/uploads/2025/05/Using-CC-licensed-Works-for-AI-Training.pdf) | CC legal primer | *"making copies … as well as **subsequent use and distribution of the trained model**, would need to be for noncommercial purposes"*; and *"in many cases, neither the AI model nor its outputs would be considered to be derivative works"* — **read before the counsel meeting** | HIGH |
| 95 | https://huggingface.co/datasets/huggingface-legal/takedown-notices — `2026/2026-05-26-IGM.md` | **published takedown notice (primary)** | `ESpeech/ESpeech-igm`: a voice actor's complaint that Apache-2.0 *"cannot lawfully apply to my voice recordings, vocal performance"*; repo now `disabled: True` | HIGH |
| 96 | https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf | US government report | weights infringe *"only … where there is substantial similarity"*; and the **"data laundering"** passage | HIGH |
| 97 | https://www.justiz.bayern.de/gerichte-und-behoerden/landgericht/muenchen-1/presse/2026/16.php | **official court press release** | *GEMA v. Suno*, 31 July 2026 — first EU ruling on a generative **audio** model; **offering the model held to be communication to the public**; not final | HIGH |
| 98 | https://www.justiz.bayern.de/gerichte-und-behoerden/landgericht/muenchen-1/presse/2025/11.php | official court press release | *GEMA v. OpenAI* — memorisation in model parameters = reproduction, not covered by §44b TDM; not final | HIGH |
| 99 | https://delhihighcourt.nic.in/app/showFileJudgment/ABL24072026SC10282024_171649.pdf | **official court judgment** | *ANI v. OpenAI*, 24 July 2026 — training prima facie within §52(1)(a); ¶262 opt-out reasoning; ¶274 limits it to the application | HIGH |
| 100 | https://www.dpiit.gov.in/static/uploads/2025/12/ff266bbeed10c48e3479c941484f3525.pdf | Government of India working paper | *"There is currently no specific exception under copyright law for text and data mining"*; TDM exception rejected in favour of a mandatory blanket licence | HIGH |
| 101 | Regulation (EU) 2024/1689 Arts. 2(12), 3(60), 50, 99(4), 113; Recital 133 | **Official Journal text** | Art. 50(2)/(4) verbatim; open-source carve-out; penalties; application dates | HIGH |
| 102 | Regulation (EU) 2026/1744 ("Digital Omnibus on AI"), OJ 24 July 2026 | **Official Journal text** | Art. 50(1)–(6) unchanged; new Art. 111(4) four-month grace **for pre-2-Aug-2026 systems only** | HIGH |
| 103 | Commission Guidelines on Article 50, **C(2026) 5054 final**, 20 July 2026 | official Commission document | paras (11), (23), (24), (58), (60), (73), (76), (428) — narrow-purpose TTS in scope; component-vs-system split; voice = deep fake | HIGH |
| 104 | Tennessee Public Chapter 588 (ELVIS Act), amendment [HA0578](https://www.capitol.tn.gov/Bills/113/Amend/HA0578.pdf) | official legislature PDF | "Voice" definition incl. simulation; § 47-25-1105(a)(3) tool provision | HIGH |
| 105 | https://www.govinfo.gov/content/pkg/BILLS-119s4591rs/xml/BILLS-119s4591rs.xml | official bill text | NO FAKES Act of 2026 §2(a)(2), §2(c)(2)(B), §2(d), §2(g), §2(h)(1); **reported, on Senate calendar, not enacted** | HIGH |
| 106 | 16 CFR Part 461 (eCFR, checked 2026-08-31); 89 FR 15017; 89 FR 15072; 89 FR 104905 | Federal Register + eCFR | impersonation rule covers government/business only; **the individuals + "means and instrumentalities" extension was NOT adopted** | HIGH |
| 107 | Washington SSB 5886, Ch. 69, Laws of 2026 (RCW 63.60.020(3), .050) | state statute | "forged digital likeness" incl. **real-time** voice; liability regardless of for-profit | HIGH |
| 108 | Delhi HC 2023:DHC:2796 (*Digital Collectibles*); CS(COMM) 652/2023 (*Anil Kapoor*); 2024:DHC:4046 (*Jackie Shroff*); Bombay HC IA(L) 23560/2024 (*Arijit Singh*); IA(L) 30382/2025 (*Asha Bhosle*); CS(COMM) 956/2025 (*Aishwarya Rai*) | court orders | Indian personality-rights line on **voice** — all interim; *Arijit* ¶18 is the key holding; ¶32 expiry **UNVERIFIED** | HIGH text / MEDIUM status |
| 109 | DPDP Act 2023 (MeitY gazette) §§2(t), 3(c)(ii), 6(1), 6(4); DPDP Rules 2025 G.S.R. 846(E) | official gazette | staged commencement — substantive rules ~May 2027 | HIGH |
| 110 | IT Amendment Rules 2026, **G.S.R. 120(E)** — [MeitY consolidated PDF](https://www.meity.gov.in/static/uploads/2026/02/550681ab908f8afb135b0ad42816a1c9.pdf) | official notification | r.2(1)(wa), r.3(3)(a)(ii)–(b): "**prominently prefixed audio disclosure**" + permanent metadata + unique identifier; **no percentage threshold survived from the draft** | HIGH text / MEDIUM date |
| 111 | https://huggingface.co/hexgrad/Kokoro-82M ; https://huggingface.co/rhasspy/piper-voices ; MLCommons People's Speech `credits.csv` | model/dataset cards | the three real-world attribution patterns compared in §4.8 | HIGH |
| 112 | https://laion.ai/notes/laion-maintenance/ and /blog/relaion-5b/ | dataset author's statement | LAION-5B withdrawal + *"we advise strongly AGAINST using datasets in their original form for creating end products"* | HIGH |
| 113 | Copyright Act 1957 (India) §§2(qq), 38A, 38B, 39 — copyright.gov.in | official statute | performers' rights; §38A(1)(a)(i) covers **electronic storage**; §39 exception is research/teaching only | HIGH |
| 114 | https://github.com/facebookresearch/voxpopuli ; https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 ; /canary-1b | repo + model cards | CC0 data → **CC-BY-NC-4.0 models** (Meta); licence propagated onto weights per training mix (NVIDIA) | HIGH |
| 115 | https://huggingface.co/api/models/facebook/mms-tts-hin ; /facebook/seamless-m4t-v2-large | HF API | both **CC-BY-NC-4.0** — the two obvious Indic fallbacks, both closed | HIGH |

---

*Pass 1 complete. Every claim above rests on a primary source fetched during this pass — a LICENSE file, a raw model/dataset card, a gate agreement, a court judgment or press release, an official gazette, or the licence text itself. Where a source could not be reached, the file says **UNVERIFIED** rather than paraphrasing. The two known gaps are the **OpenAI Terms of Use** verbatim text (openai.com returns HTTP 403 to every automated fetch attempted, including via the Wayback Machine) and the **Coqui CPML's** canonical URL (dead — the in-repo copy was used instead).*

**The five highest-value follow-ups, in order of cost-to-value:**

1. **Log in to HuggingFace and screenshot the gate terms** on `ai4bharat/indic-parler-tts`, `indic-parler-tts-pretrained`, `indicvoices_r`, `IndicVoices`, `Rasa`, `IndicF5`, `DhVaani-0.5`. **Ten minutes, and it gates the entire Indic track.**
2. **Check the BGH result in *Kneschke v. LAION* (`I ZR 281/25`)** — heard 3 September 2026, the day after this file was written.
3. **Put the Zonos speaker-encoder question to counsel before S1 mints a single identity.** A Tier-1 vector is only meaningful relative to the encoder that produced it; substituting the encoder later invalidates every voice already created.
4. **Decide what happens to VoicePersona v1**, which stands published under a CC0 claim today, over 79% of material its declarer had no right to license. This exposure is accruing now, not at launch.
5. **Ask IITM to confirm the IndicTTS EULA** recovered in §3.9 and whether §2.2 permits an Apache-2.0 weight release. One email; it decides whether the scope document's Indic plan works as written.
