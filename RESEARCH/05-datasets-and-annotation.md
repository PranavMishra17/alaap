# 05 — Training Data & the Measure-First Annotation Pipeline

> **Domain:** paired (description, audio) corpora; the Data-Speech/ParaSpeechCaps recipe; VoicePersona v2
> **Answers:** C1, C3, C4
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [01-ttv-landscape.md](01-ttv-landscape.md) · [02-identity-representation.md](02-identity-representation.md) · [04-indic-track.md](04-indic-track.md) · [06-evaluation-harness.md](06-evaluation-harness.md) · [08-licensing-propagation.md](08-licensing-propagation.md)

## 0. Bottom line

- **The measure-first, caption-second thesis is correct and is now the industry consensus.** Three independent teams converged on it: Data-Speech/Parler-TTS (2024), ParaSpeechCaps (2025) and VoiceDesigner (2026) all measure with signal processing, bin, then use an LLM only for fluency. No shipped corpus lets an audio-LM produce quantitative attributes. **[HIGH]**
- **But the brief describes a pipeline that does not exist.** Data-Speech computes **nine** columns from five tools — F0 mean, F0 std, SNR, C50, speech duration, speaking rate, phonemes, and optionally STOI/SI-SDR/PESQ. It computes **no jitter, no shimmer, no HNR, no spectral tilt, no formants, no VTL, and no F0 range.** Those are VoiceForge's own extension, not a reuse. **[HIGH]**
- **Two concrete bugs in the recipe the brief plans to copy.** (1) `speaking_rate` is `len(ipa_string)/duration` — **IPA *characters* per second, not phones per second**. (2) Bins are **equal-WIDTH histogram bins**, not percentiles — so the extreme labels ("very low-pitch") are nearly empty and a mapper trained on them will under-generate exactly the character voices the product needs. The brief's own instinct (percentiles) is right and VoiceDesigner 2026 confirms it; Data-Speech's implementation is what's wrong. Both are small fixes, but fixing #1 invalidates every published bin edge. **[HIGH]**
- 🔴 **ParaSpeechCaps cannot be the primary training set. It is triply blocked.** Annotations are **CC-BY-NC-SA-4.0**; it ships **no audio**; and all four audio sources are unusable — VoxCeleb (**Oxford has withdrawn every download: audio, video, URLs, and metadata**), Expresso (NC), EARS (NC), Emilia-EN (NC + a gate that binds a for-profit employer). The scope doc marks it "⭐ Primary English training set". **This is the largest correction in this document.** **[HIGH]**
- **LibriTTS-P is the corpus the plan should rest on.** CC BY 4.0, 2,443 speakers with human-annotated identity adjectives (44-term vocabulary with `very`/`slightly` qualifiers, 3 annotators each), plus a reusable template bank of style prompts, over CC BY 4.0 LibriTTS-R audio. It is the **only permissively-licensed rich intrinsic vocabulary in existence**. ⚠️ Its CC BY 4.0 claim rests on one README line — **there is no LICENSE file**. Get one. **[HIGH on the data; MEDIUM on the licence]**
- **Three hidden non-commercial dependencies are inside the recipe itself**, and the scope doc budgets for none of them: the gender classifier `audeering/wav2vec2-large-robust-24-ft-age-gender` is **CC-BY-NC-SA-4.0**; VoxSim (ParaSpeechCaps' tag-propagation model) has **no licence at all**; Praat/parselmouth (the obvious jitter/shimmer tool) is **GPLv3**. A permissive gender replacement exists (`alefiury/...-gender-recognition-librispeech`, Apache-2.0). **[HIGH]**
- **The +7.9% Consistency MOS / +15.5% Naturalness MOS claims are verified exactly** (3.55→3.83 and 3.10→3.58, Table 3, arXiv 2503.04713v2) — but they are *relative* percentages of a 5-point Likert mean against the `+LibriTTS-P,Expresso,EARS` baseline, not vanilla Parler-TTS, and **intelligibility got worse** (IMOS 4.44→4.07, WER 4.47→8.63). The most transferable number is that **human intrinsic annotation alone lifted Intrinsic Tag Recall 40.7%→63.6%** before any scaling. **[HIGH]**
- **VoicePersona's CC0 declaration does not hold, on four independent grounds.** 13.3% is CC BY-**NC** (AnimeVox), another 13.3% is MIT-with-attribution over anime audio with no stated rights basis (AniSpeech), 52.6% is Apache-2.0 synthetic gpt-4o-audio (LAION), and the HF card does not even say CC0 — it says `license: cc`. **26.5% of the corpus is anime audio ripped from commercial dubs and VoicePersona redistributes the audio itself.** Only GLOBE_V2 (20.9%) is genuinely clean. **[HIGH on documents; legal conclusion needs counsel]**
- **The demographic skew is largely a prompt artefact, not a property of the audio.** VoicePersona's live prompt hard-codes `GENDER: [male/female]`, caps age at `fifties+`, and instructs the model `never "neutral"` on accent — for a 52.6% slice that is synthetic TTS with no real speaker demographics at all. Rebalancing the existing labels would optimise against noise. Re-measure first. **[HIGH]**
- **The audio-LM criticism is right for magnitudes and wrong for categories.** Qwen-Omni (Qwen2-Audio's successor) scores **32% on pitch** and **50% on speed** on StepEval-Audio-Paralinguistic, but **76% on emotion**; GPT-4o Audio gets 40%/58%/82%. Keep the audio-LM — demoted from *describer* to *verifier of closed-vocabulary abstract tags*, which is exactly ParaSpeechCaps' design and what its ablation shows is necessary. **[HIGH numbers / MEDIUM transfer to Qwen2-Audio specifically]**
- **"Speaker count beats hours" is TRUE and better supported than the brief knows — but the brief states it wrong.** Nobody has published this ablation for a description→embedding mapper (**UNVERIFIED**, §5.1). But a fixed-100-hour VoxCeleb2 ablation (Vaessen & van Leeuwen, Interspeech 2022) shows 60× more speakers **halves** ECAPA EER (12.19%→6.04%) at ~58 s/speaker — *and* that the same 60× speakers with **one recording session each is worse than 100 speakers** (12.19%→15.97%). The operative variable is **speakers × session diversity**. Separately, arXiv 2512.17356 finds WER/quality saturate at ~50 speakers but **speaker-similarity keeps improving to ~500**. Restate the belief accordingly, and note that **VoicePersona's 52.6% synthetic-TTS slice has effectively zero session diversity.** **[HIGH]**
- 🚨 **The brief missed the largest permissively-licensed style-captioned corpus that exists: RASMALAI** (arXiv 2505.18609, **CC BY 4.0, 13,000 h, 24M descriptions, 24 languages including English**) — it is filed in the plan as an Indic-only item. It also missed **GLOBE** (535 h from **23,519 speakers**, 164 accents, **CC0**, ~82 s/speaker) — the single best public match to the speaker-count hypothesis, and already a VoicePersona upstream. **[HIGH]**
- **VoiceDesigner (arXiv 2608.13613) resolves and its data recipe is the answer to the non-human workstream** — a DSP rack (pitch shift, formant shift, reverb, EQ, band-pass, DRC, SiFi-GAN pitch-contour manipulation) over *clean, licensed* LibriTTS-R and VCTK, plus voice-cloning/VC generative expansion. **This is a licence-clean replacement for the anime data**, and it should be built during the dataset phase rather than deferred. **[HIGH]**

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| 1 | ParaSpeechCaps is the "⭐ **Primary English training set**" | 🔴 **WRONG — hard blocker** | Annotations are CC-BY-NC-**SA**-4.0; it ships **no audio**; its four audio sources are VoxCeleb (withdrawn entirely by Oxford), Expresso (NC), EARS (NC), Emilia-EN (NC + employer-binding gate). Unusable for a released commercial model. | [HF card YAML](https://huggingface.co/datasets/ajd12342/paraspeechcaps/raw/main/README.md); [dataset/README.md](https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/README.md); [vox2.html](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) | HIGH |
| 2 | "ParaSpeechCaps: 342 hrs human + 2,427 hrs auto = ~2,769 hrs" | ✅ **CORRECT, but state it precisely** | 342 h = **all** human-annotated splits (train 282.54 + dev 26.29 + holdout 33.04). The arXiv **v2** body says **282 h** (train only) while the arXiv abstract metadata still says 342. Total 2,769.03 h is right. | HF card split table; arXiv 2503.04713 v1 vs v2 | HIGH |
| 3 | "59 style tags incl. abstract (guttural, nasal, pained)" | ✅ **CORRECT** | 59 = 28 rich intrinsic + 5 basic intrinsic + 23 rich situational + 3 basic situational, across 11 style factors. Full enumeration in §2.4. | arXiv 2503.04713 Appendix A | HIGH |
| 4 | "+7.9% consistency MOS, +15.5% naturalness MOS" | ✅ **CORRECT, with caveats** | 3.55→3.83 and 3.10→3.58 vs the `+LTTSP,Exp,EARS` baseline. Relative % of a Likert mean. **Intelligibility regressed** (IMOS 4.44→4.07, WER 4.47→8.63). | arXiv 2503.04713v2 Table 3 | HIGH |
| 5 | The attribute list to extract: "jitter, shimmer, HNR, spectral tilt, formants F1–F3, estimated VTL" | ⚠️ **NOT A REUSE** | Data-Speech computes **none** of these. It has four modules total. These are a VoiceForge extension — a good one (they are the correlates of *raspy*/*breathy*/*deep*) but they must be built and validated, not installed. | [`dataspeech/`](https://github.com/huggingface/dataspeech) source, read in full | HIGH |
| 6 | "speaking rate (phones/sec)" | 🔴 **WRONG** | Code is `len(transducer(text).output_string) / duration` — **IPA characters per second**. Not comparable across languages; fixing it invalidates every published bin edge. | [`rate.py`](https://raw.githubusercontent.com/huggingface/dataspeech/main/dataspeech/cpu_enrichments/rate.py) | HIGH |
| 7 | "Bin each measured value... calibrated against the corpus distribution" | ⚠️ **HALF RIGHT** | The *range* is corpus-derived but the partition is `np.histogram(values, bins=N)` = **equal-width**, not percentile. Extreme bins end up nearly empty. VoiceDesigner (2026) does use percentiles — the brief's instinct is right, Data-Speech's implementation is not. Use `np.quantile`. | [`metadata_to_text.py`](https://raw.githubusercontent.com/huggingface/dataspeech/main/scripts/metadata_to_text.py) L57–97; arXiv 2608.13613 §IV-A1 | HIGH |
| 8 | Implied: the ParaSpeechCaps pipeline can be reused wholesale | ⚠️ **PARTLY UNRELEASED** | Situational-tag scaling and the style-prompt generator are marked "**coming soon**" and do not exist in the repo (95-entry tree, `truncated: false`). Reusable: basic tags, intrinsic VoxSim propagation, and the appendix prompts. | [automatic_annotation/README.md](https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/automatic_annotation/README.md); repo tree | HIGH |
| 9 | Implied: the extraction stack is licence-clean | 🔴 **THREE NC/GPL DEPENDENCIES** | Gender classifier `audeering/wav2vec2-large-robust-24-ft-age-gender` = **CC-BY-NC-SA-4.0**; VoxSim (`kaistmm/voxsim_trainer`) has **no LICENSE file**; Praat/parselmouth = **GPLv3**. penn, brouhaha, g2p, SQUIM are all fine. | model card YAML; repo tree (LICENSE 404); Praat licence | HIGH |
| 10 | VoicePersona is "**CC0**, maximally permissive and a genuine asset here" | 🔴 **WRONG — it is the most exposed input in the plan** | HF YAML says `license: cc`, not CC0. 13.3% is CC BY-NC; 26.5% is anime audio ripped from commercial dubs with no rights basis; 52.6% is Apache-2.0 with an unresolved OpenAI-ToS question. And it **redistributes the audio**. | HF card YAMLs of VoicePersona and all four upstreams | HIGH |
| 11 | VoicePersona "15,082 samples, 8+ languages, ~500-char descriptions" | 🔴 **THREE FACTUAL ERRORS** | The published artefact has **14,327 rows**. The card YAML declares **`en` only** — "8+ languages" is Qwen2-Audio's capability restated as a dataset property. The repo's own `analysis_report.txt` measures **307 characters**, not 500. | datasets-server size API; HF card YAML; analysis_report.txt | HIGH |
| 12 | VoicePersona's "702 accent variants" | ⚠️ **MISLEADING** | Unnormalised free-text. The top 10 contains five separate labels for American English, plus "Valley girl accent". 792 rows are `unknown`. | statistics.json | HIGH |
| 13 | "Rebalance the 62.6% female / 76.1% twenties skew" | ⚠️ **WOULD OPTIMISE AGAINST NOISE** | The skew is substantially generated: the live prompt hard-codes `GENDER: [male/female]`, caps age at `fifties+`, and orders the model to `never` say "neutral" — over a 52.6% synthetic-TTS slice with no real demographics. Re-measure before rebalancing. | [`src/dataset_processor.py`](https://raw.githubusercontent.com/PranavMishra17/VoicePersona-Dataset/main/src/dataset_processor.py) | HIGH |
| 14 | "AniSpeech + AnimeVox give character/anime voices that no public corpus has… real, non-replicable coverage" | ⚠️ **REPLICABLE, AND LEGALLY** | VoiceDesigner's DSP rack over CC-BY LibriTTS-R and VCTK synthesizes non-human/character voice variation from clean licensed audio. That is the same coverage without the exposure. | arXiv 2608.13613 §III-C | HIGH |
| 15 | "LibriTTS-P — ⭐ Core English pairs. Highest-quality human annotation." | ✅ **CORRECT, and understated** | It is now the *foundation*, not a supplement: CC BY 4.0, 2,443 speakers, 44-adjective human vocabulary with strength qualifiers, 373,868 prompts, plus a reusable paraphrase bank. ⚠️ No LICENSE file exists — the CC BY 4.0 claim is one README line. | [repo README](https://raw.githubusercontent.com/line/LibriTTS-P/main/README.md); GitHub API `license: null` | HIGH data / MEDIUM licence |
| 16 | "LibriTTS-R / Emilia / MLS / VoxCeleb2 — raw speaker diversity; annotate with our own pipeline" | ⚠️ **TWO OF FOUR ARE OUT** | VoxCeleb2 is **no longer distributed** and never had an audio licence. Emilia-proper is CC BY-NC (its HF YAML wrongly says `cc-by-4.0`). LibriTTS-R and MLS are fine (both CC BY 4.0). **Add Common Voice SS 26.0 (CC0, 294 langs) and People's Speech (CC BY).** | vox2.html; Emilia gate text; openslr.org/94 & /141 | HIGH |
| 17 | "TextrolSpeech (~330 hrs)" as a candidate | 🔴 **BLOCKED** | Repo MIT covers *code*; **no dataset licence is stated anywhere**, and it is built on ESD/MEAD/SAVEE/TESS/MESS, which are research-only. | repo LICENSE; arXiv 2308.14430 | HIGH |
| 18 | "SpeechCraft" as a candidate | 🔴 **UNLICENSED** | No LICENSE file, no statement in README or paper, GitHub API `license: null`. The emphasis subset is behind a signed non-commercial EULA. Absence of a licence is not permission. | [repo README](https://raw.githubusercontent.com/thuhcsi/SpeechCraft/master/README.md) (branch `master`) | HIGH |
| 19 | Sequencing: dataset rebuild at **S4**, after both mappers | ⚠️ **TOO LATE** | With ParaSpeechCaps out, S2's only permissive paired data is LibriTTS-P's 2,443 speakers unless the annotation pipeline exists first. And the measured features *are* the adherence metric that S2/S3 need. **Split S4: move measure+bin to run alongside S0/S1.** | this document §8.4 | HIGH (reasoning) |
| 20 | VoiceDesigner "explicitly targets fictional voices via a DSP+generative augmentation pipeline" | ✅ **CORRECT** | Verified in full: DSP rack + generative (cloning + VC) expansion; 16 h / 20 characters / 12 actors → 360 voice variations. No code, no weights. | arXiv 2608.13613 §III-C, §IV-A | HIGH |
| 21 | "**Speaker count matters more than hours.** 1,000 speakers × 3 min beats 50 speakers × 1 hr." | ⚠️ **RIGHT DIRECTION, INCOMPLETE — AND UNVERIFIED FOR THIS TASK** | No published ablation exists for a description→embedding mapper. Adjacent fixed-budget evidence supports it strongly, **but only with recording-session diversity**: 5,994 speakers at 1 session each scores *worse* than 100 speakers (ECAPA EER 15.97 vs 12.19), while the same 5,994 with 7.8 sessions each scores 6.04. Restate as **"speaker count × session diversity"**. Also: the knee for speaker-similarity is ~**500** speakers, not 50. | [Vaessen & van Leeuwen, Interspeech 2022](https://www.isca-archive.org/interspeech_2022/vaessen22_interspeech.pdf) Tables 1–2; [arXiv 2512.17356](https://arxiv.org/abs/2512.17356) §4.2; [arXiv 2309.14838](https://arxiv.org/abs/2309.14838) §2.1 | HIGH (evidence) / HIGH (the absence) |
| 22 | RASMALAI filed as an Indic-track corpus only | ⚠️ **UNDER-USED** | **CC BY 4.0, 13,000 h, 24M natural-language style descriptions, 24 languages including English.** The largest permissively-licensed style-captioned corpus in existence, and a fourth independent implementation of measure-first captioning. Belongs in the English plan too. | [arXiv 2505.18609](https://arxiv.org/abs/2505.18609) | HIGH |
| 23 | Corpus list omits GLOBE, CapSpeech, VoxBlink2, NonverbalTTS | ⚠️ **INCOMPLETE** | **GLOBE**: 535 h / **23,519 speakers** / 164 accents / **CC0** — best public match to the speaker-count hypothesis, and already a VoicePersona upstream. **CapSpeech** (2506.02863): the most-cited 2025 style corpus — but **CC BY-NC**. **VoxBlink2**: 111,284 speakers. **NonverbalTTS**: laughs/sighs/gasps, which nothing else covers. | §4 | HIGH |
| 24 | Reference list implies "SpeakerVerse", "VccmDataset", "Audiobox caption data" are usable corpora | 🔴 **TWO DO NOT EXIST** | "SpeakerVerse" and "VccmDataset" return no arXiv and no HF match. No Meta caption corpus was ever released. Do not cite. | direct search, **UNVERIFIED/absent** | HIGH (absence) |

---

## 2. C1 — the annotation recipe, in reusable detail

Everything below is read from the **actual source files**, not from paper prose. Two repos matter and they are not the same pipeline:

| | Data-Speech (HF) | ParaSpeechCaps (Diwan et al.) |
|---|---|---|
| Repo | `github.com/huggingface/dataspeech` | `github.com/ajd12342/paraspeechcaps` |
| Repo licence | **MIT** ([LICENSE](https://raw.githubusercontent.com/huggingface/dataspeech/main/LICENSE), "Copyright (c) 2024 The Hugging Face team") | **MIT** ([LICENSE](https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/LICENSE), "Copyright (c) 2025 Anuj Jitendra Diwan") |
| Dataset licence | n/a (tool only) | **CC-BY-NC-SA-4.0** — see §3 |
| Scope | basic tags only, fully automatic | basic tags (vendored Data-Speech) **+** rich tags via human annotation and two scaling pipelines |
| Bin style | **equal-width** histogram bins over a filtered range | **fixed hard-coded thresholds**, derived once from Data-Speech's v02 edges |

ParaSpeechCaps **vendors** Data-Speech: `dataset/automatic_annotation/basic_tags/dataspeech/` is a copy of the HF package with `--load_from_disk` support added. Its own README says so ([basic_tags/README.md](https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/automatic_annotation/basic_tags/README.md)). So there is exactly **one** measurement stack to install, and two threshold configurations to choose between.

> ⚠️ **Two parts of ParaSpeechCaps' pipeline are NOT released.** `dataset/automatic_annotation/README.md` marks *Situational Tag Scaling* and *style_prompts/* as **"(coming soon)"**, and the top-level README still lists as open TODOs: "Release code for our human annotation pipeline" and "Release code for our automatic annotation pipeline". The repo tree (95 entries, `truncated: false`) confirms neither directory exists. **[HIGH]** What you *can* reuse verbatim: basic tags, the intrinsic-tag VoxSim propagation, and the prompts printed in the paper appendix. What you must reimplement: situational scaling and the style-prompt generator.

### 2.1 Attribute list and the tool computing each

This is the **complete** list. Data-Speech computes **nine** columns, from five tools. There is nothing else in the package.

| Attribute (column) | Exact definition, as coded | Tool / library | Licence of tool | Notes |
|---|---|---|---|---|
| `utterance_pitch_mean` | `pitch.mean()` over the whole utterance, Hz | **penn** (FCNF0++), `penn.from_audio` | MIT ([LICENSE](https://raw.githubusercontent.com/interactiveaudiolab/penn/master/LICENSE)) | `hopsize=0.01`, `fmin=30.`, `fmax=1000.`, `center='half-hop'`, `interp_unvoiced_at=0.065`, `checkpoint=None` (default FCNF0++ trained on MDB-stem-synth + PTDB). Audio cast to **16 kHz** first (`main.py` does `cast_column(Audio(sampling_rate=16_000))`). |
| `utterance_pitch_std` | `pitch.std()` over the utterance, Hz | penn | MIT | Later relabelled **"speech monotony"**. Std of the *linear-Hz* pitch track, **not** log-F0. |
| `snr` | mean of Brouhaha's frame-level SNR over VAD-active, non-zero frames, dB | **Brouhaha** VAD (`RegressiveActivityDetectionPipeline`), checkpoint `ylacombe/brouhaha-best` | brouhaha-vad MIT ([LICENSE](https://raw.githubusercontent.com/marianne-m/brouhaha-vad/main/LICENSE), CNRS); checkpoint card says `license: mit` | Frame ratio hard-coded `ratio = 16000/270`. Mask = VAD-active **AND NOT** (`snr==0 AND c50==0`). |
| `c50` | mean Brouhaha C50 (clarity index) over the same mask, dB | Brouhaha | MIT | Later relabelled **"reverberation"**. C50 = early/late energy ratio at 50 ms; **higher C50 = drier/closer**, so the bin vocabulary runs distant → close as C50 rises. |
| `speech_duration` | sum of VAD segment durations, seconds | Brouhaha annotation tracks | MIT | This is what makes speaking rate silence-corrected. |
| `speaking_rate` | **`len(phonemes) / audio_duration`** where `phonemes = transducer(text).output_string` | **g2p** (`make_g2p('eng','eng-ipa')`) | MIT ([LICENSE](https://raw.githubusercontent.com/roedoejet/g2p/main/LICENSE), Aidan Pine / NRC) | See the correction below — this is **not** phones/sec. Uses `speech_duration` when Brouhaha ran, else raw clip length. |
| `phonemes` | the IPA string itself | g2p | MIT | Retained in the released datasets. |
| `stoi`, `si-sdr` (`sdr`), `pesq` | reference-free quality estimates | **torchaudio `SQUIM_OBJECTIVE`** | torchaudio BSD-2; **SQUIM weights CC-BY-4.0** per [pipeline docs](https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.SQUIM_OBJECTIVE.html) | Optional (`--apply_squim_quality_estimation`). Resampled to SQUIM's rate; **truncated to the first 15 s** (`max_audio_length = 15 * SQUIM_OBJECTIVE.sample_rate`). |
| `gender` | argmax of male vs female class probability | **`audeering/wav2vec2-large-robust-24-ft-age-gender`** (ParaSpeechCaps `extract_gender.py`) | 🔴 **CC-BY-NC-SA-4.0** — see the [model card YAML](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender/raw/main/README.md) | Data-Speech itself has **no** gender step; it expects a `gender` column to already exist (`per_dataset_script/add_gender_to_libritts_r.py` reads corpus metadata). |

**🔴 CORRECTION — the brief's attribute list is substantially larger than what Data-Speech actually computes.** The brief asks for "F0 mean/std/range, speaking rate (phones/sec), **jitter, shimmer, HNR, spectral tilt, formants F1–F3, estimated vocal-tract length**, SNR/reverberation". **Data-Speech computes none of the bolded items.** There is no jitter, no shimmer, no HNR, no spectral tilt, no formant tracking and no VTL anywhere in the repo — `dataspeech/` contains exactly four modules (`rate.py`, `pitch.py`, `snr_and_reverb.py`, `squim.py`) and I have read all four. It also computes no F0 *range* (only mean and std). **[HIGH]** Those extra features are a genuinely good idea for a project that cares about *raspy / breathy / aged* — jitter, shimmer and HNR are exactly the correlates of vocal roughness and breathiness — but they are **VoiceForge's own extension**, not something to be "reused" from Data-Speech. Budget for them (Praat/Parselmouth, GPL-adjacent — check; or `librosa` + `pysptk`, both permissive) and validate them yourself.

**🔴 CORRECTION — `speaking_rate` is not phonemes per second.** The brief and the ParaSpeechCaps paper both say "the number of phonemes per second". The code says:

```python
phonemes = transducer(text).output_string       # a single IPA STRING
speaking_rate = len(phonemes) / audio_length    # len() of a str == CHARACTER count
```

`len()` of a Python string counts **characters**, and an IPA transcription contains multi-character segments (`tʃ`, `aɪ`, `ɔː`) plus, depending on the mapping, length marks and diacritics. So the measured quantity is **IPA-characters per second**, which is a monotone-ish but not linear proxy for phones/sec. This is why the v01 bin edges top out around 22 and v02 around 27 — far above any real phone rate (English runs ~10–15 phones/s). It is *internally consistent* (the same transform is applied everywhere, so the bins are still meaningful), but it is **not** the quantity the brief names, it is **not comparable across languages** with different IPA orthographies, and any VoiceForge re-implementation that "fixes" it to true phone counts **must recompute the bin edges from scratch**. **[HIGH — read directly from [`dataspeech/cpu_enrichments/rate.py`](https://raw.githubusercontent.com/huggingface/dataspeech/main/dataspeech/cpu_enrichments/rate.py)]**

**Note on "reverberation":** it is C50, a *clarity* index, not RT60. And `c50` is measured on the **same** frames as SNR. For synthetic or heavily-processed audio the Brouhaha estimate is out of domain; ParaSpeechCaps found VoxCeleb's median SNR was 31.76 dB vs 59.49 / 50.42 / 61.70 dB for Expresso / EARS / LibriTTS-R and ran **Voicefixer** over all VoxCeleb audio before annotating (paper §C.1) **[HIGH]**.

### 2.2 Binning thresholds and bin vocabulary

**🔴 CORRECTION — Data-Speech does NOT use percentiles.** The brief says bins are "calibrated against the CORPUS DISTRIBUTION, not absolute thresholds". Half right. The code is:

```python
values = values[~np.isnan(values)]
if std_tolerance is not None:                                        # outlier rejection
    values = values[np.abs(values - np.mean(values)) < std_tolerance * np.std(values)]
hist, bin_edges = np.histogram(values, bins=len(text_bins),
                               range=(lower_range, values.max()) if lower_range else None)
...
index_bins = np.searchsorted(bin_edges, batch, side="left")
batch_bins = [text_bins[min(max(i-1, 0), len(text_bins)-1)] for i in index_bins]
```

`np.histogram(values, bins=N)` produces **N equal-WIDTH bins** spanning `[min, max]` of the filtered values — not equal-count quantiles. The *range* is corpus-derived; the *partition* is uniform. Consequence: on a roughly Gaussian attribute the extreme bins ("very low pitch", "very fast") end up nearly empty and the middle bins hold most of the mass. This is a **real weakness** and it directly matters to VoiceForge, because a mapper trained on these captions will almost never see "very low-pitched" and will therefore under-generate it. **[HIGH — [`scripts/metadata_to_text.py`](https://raw.githubusercontent.com/huggingface/dataspeech/main/scripts/metadata_to_text.py) L57–97]**

**Use `np.quantile` instead.** Note that a 2026 primary source agrees with the brief and not with Data-Speech: VoiceDesigner (arXiv 2608.13613 §IV-A1) states *"we discretize the values using percentile-based bins computed from the Emilia and HiFiTTS-2-44.1k datasets."* **[HIGH]** The brief's instinct is right; Data-Speech's implementation is what's wrong. Fixing this is a ~3-line change.

Two further mechanics worth copying:

1. **Pitch is binned per-speaker AND per-gender.** `speaker_level_relative_to_gender()` first does `groupby(speaker_id).agg({pitch: "mean"})`, then builds **separate** male and female histograms. Every utterance of a speaker gets that speaker's single pitch label. This is correct and important — pitch is an *identity* attribute, not an utterance attribute. Everything else (rate, SNR, C50, monotony) is binned per-**utterance** on the pooled distribution.
2. **Outlier rejection before binning**, with per-attribute σ tolerances (defaults: pitch 2.0, speaking rate 4.0, SNR 3.5, reverberation 4, monotony 4; the published v02 run used pitch **1.5**, reverberation **8.0**, monotony **2.0**, speaking rate **5.5**, SNR **3.5** — see [`run_metadata_to_text_10k_v02.sh`](https://raw.githubusercontent.com/huggingface/dataspeech/main/examples/tags_to_annotations/run_metadata_to_text_10k_v02.sh)).

#### Copy-pasteable: Data-Speech v02 bin edges (the Parler-TTS Mini/Large v1 configuration)

Computed over `mls-eng-10k` + `libritts_r` (clean + other), train splits. Source: [`examples/tags_to_annotations/v02_bin_edges.json`](https://raw.githubusercontent.com/huggingface/dataspeech/main/examples/tags_to_annotations/v02_bin_edges.json) **[HIGH]**

```json
{
  "speaking_rate":     [0.0, 3.8258038258038254, 7.651607651607651, 11.477411477411476,
                        15.303215303215302, 19.129019129019127, 22.95482295482295, 26.78062678062678],
  "noise":             [17.12751579284668, 25.4012325831822, 33.67494937351772, 41.94866616385323,
                        50.22238295418875, 58.49609974452427, 66.76981653485979, 75.04353332519531],
  "reverberation":     [10, 35, 45, 55, 59, 60],
  "speech_monotony":   [0.0, 20.37920924595424, 40.75841849190848, 70, 90, 142.6544647216797],
  "pitch_bins_male":   [64.6531982421875, 81.66683959960938, 98.68048095703125, 115.69412231445312,
                        132.707763671875, 149.72140502929688, 166.73504638671875, 183.74868774414062],
  "pitch_bins_female": [120.17855072021484, 141.6242690945264, 163.06998746883795, 184.51570584314953,
                        205.96142421746106, 227.40714259177264, 248.8528609660842, 270.29857934039575],
  "si-sdr":            [-17.804332733154297, -0.40644073486328125, 10, 20, 25, 28, 34.38934326171875],
  "pesq":              [1, 1.7, 2.4, 3.1, 3.6, 4, 4.499948978424072]
}
```

Units: `speaking_rate` IPA-chars/s · `noise` SNR dB · `reverberation` C50 dB · `speech_monotony` Hz (std of linear-Hz F0) · pitch Hz · `si-sdr` dB · `pesq` 1–4.5.

Note `reverberation` and `speech_monotony` in v02 were **hand-adjusted**, not histogram output — they are short, irregular, round-numbered lists. The authors overrode the automatic bins where the automatic bins were bad. Expect to do the same.

#### Copy-pasteable: Data-Speech v02 bin vocabulary

Source: [`v02_text_bins.json`](https://raw.githubusercontent.com/huggingface/dataspeech/main/examples/tags_to_annotations/v02_text_bins.json) **[HIGH]**

```json
{
  "speaker_rate_bins":        ["very slowly", "slowly", "slightly slowly", "moderate speed",
                               "slightly fast", "fast", "very fast"],
  "snr_bins":                 ["very noisy", "noisy", "slightly noisy", "balanced in clarity",
                               "slightly clean", "clean", "very clean"],
  "reverberation_bins":       ["very distant-sounding", "distant-sounding", "slightly distant-sounding",
                               "slightly close-sounding", "very close-sounding"],
  "utterance_level_std":      ["very monotone", "monotone", "slightly expressive and animated",
                               "expressive and animated", "very expressive and animated"],
  "speaker_level_pitch_bins": ["very low-pitch", "low-pitch", "slightly low-pitch", "moderate pitch",
                               "slightly high-pitch", "high-pitch", "very high-pitch"]
}
```

The older **v01** vocabulary (used by `parler-tts-mini-v0.1`) is different and worth knowing because some published Parler datasets use it: rate `["very slowly","quite slowly","slightly slowly","moderate speed","slightly fast","quite fast","very fast"]`; SNR `["very noisy","quite noisy","slightly noisy","moderate ambient sound","slightly clear","quite clear","very clear"]`; reverb `["very roomy sounding",...,"very confined sounding"]` (7 bins); monotony `["very monotone",...,"very expressive"]` (7 bins). Two extra vocabularies live only in `metadata_to_text.py` as Python constants: `SI_SDR_BINS = ["extremely noisy","very noisy","noisy","slightly noisy","almost no noise","very clear"]` and `PESQ_BINS = ["very bad speech quality","bad speech quality","slightly bad speech quality","moderate speech quality","great speech quality","wonderful speech quality"]`.

#### Copy-pasteable: ParaSpeechCaps thresholds (coarser — 3 bins for pitch and rate)

ParaSpeechCaps deliberately collapses to 3 levels, and hard-codes the edges rather than recomputing per corpus. Paper Appendix C.3 and [`bin_edges.json`](https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/automatic_annotation/basic_tags/bin_edges.json) agree exactly **[HIGH]**:

| Attribute | Rule |
|---|---|
| Pitch, male | `low-pitched < 115.7 Hz` · `high-pitched > 149.7 Hz` · else `medium-pitched` |
| Pitch, female | `low-pitched < 141.6 Hz` · `high-pitched > 184.5 Hz` · else `medium-pitched` |
| Speed | `slow speed < 11.5 PPS` · `fast speed > 19.1 PPS` · else `measured speed` |
| Noise (SNR dB) | edges `17.1, 25.4, 33.7, 42.0, 50.2, 58.5, 66.8, 75.0` → 7 labels `"very noisy environment" … "very clean environment"` |

These are literally the **v02 Data-Speech edges, subsampled**: male 115.69/149.72 are v02 edges #4 and #6; female 141.62/184.52 are v02 edges #2 and #4; 11.477/19.129 are v02 speaking-rate edges #4 and #6. So ParaSpeechCaps inherits the MLS+LibriTTS-R distribution, applied unchanged to VoxCeleb, Expresso, EARS and Emilia. **That is a real methodological weakness to note**: the thresholds are *not* recalibrated to each corpus, so a corpus with a different pitch distribution gets systematically skewed labels. **[HIGH]**

### 2.3 Caption generation prompts

**Model:** `mistralai/Mistral-7B-Instruct-v0.2` in both pipelines. Data-Speech runs it via `accelerate launch ... run_prompt_creation.py --per_device_eval_batch_size 64 --attn_implementation sdpa`, optionally `--load_in_4bit`. ParaSpeechCaps (paper §E.4): per-device batch 32, **temperature 0.6, top-p 1.0, max 256 new tokens**. Prompt is wrapped with `tokenizer.apply_chat_template([{"role":"user","content": prompt}])`. **[HIGH]**

#### Data-Speech `NEW_PROMPT` — the one used for Parler-TTS Mini/Large v1

Verbatim from [`scripts/run_prompt_creation.py`](https://raw.githubusercontent.com/huggingface/dataspeech/main/scripts/run_prompt_creation.py) L332–353. Placeholders `[gender] [reverberation] [sdr_noise] [speech_monotony] [speaking_rate] [pitch]` are string-replaced with the bin labels.

```
You will be given six descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The gender (male, female)
2. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
3. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
4. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
5. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)
6. The pitch of the speaker's voice (very low-pitch, low-pitch, slightly low-pitch, moderate pitch, slightly high-pitch, high-pitch, very high-pitch)

Your task is to create a text description using these keywords that accurately describes the speech sample.
If the amount of noise is 'very noisy' and the level of reverberation is 'very distant-sounding', you must include terms such as 'very poor recording' or `very bad recording` in the description.
Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very close-sounding', you must include terms like 'very good recording' or `excellent recording` in the description.
You can randomly omit the following terms, as they are default terms: 'moderate speed' and 'moderate pitch'.
Do not add extra details beyond what has been provided above. You can change the order of keywords, and replace synonymous terms.

For example, given the following keywords: 'female', 'slightly distant-sounding', 'noisy', 'very expressive and animated', 'very slowly', 'moderate pitch', a valid description would be: 'A woman speaks very slowly but has a very animated delivery. The recording is noisy and there is some roominess.'
Another valid description would be: 'In a noisy room, a female speaker delivers a very animated and expressive speech, at a very slow pace.'
Another valid description would be: 'A woman enunciates a very expressive speech. Her voice is slightly distant-sounding, with some background noise present. She speaks very slowly with a moderate pitch but a very expressive tone.'

Ensure that the generated description is grammatically correct, easy to understand, and concise. Only return one and only one description.

For the keywords: '[gender]', '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', '[pitch]', the corresponding description is:
```

Three sibling prompts exist in the same file and are worth knowing:
- **`NEW_PROMPT_WITH_ACCENT`** — identical plus `7. The accent of the speaker.` and `'[accent]'`. Selected automatically when the accent column is not `"Unindentified"` (sic — the typo is in the code, `run_prompt_creation.py` L562).
- **`NEW_SINGLE_SPEAKER_PROMPT`** / **`SINGLE_SPEAKER_PROMPT`** — drop gender and pitch, inject a `[speaker_name]`. This is how Parler-TTS learned named voices ("Jon", "Lea", …) via `--speaker_ids_to_name_json`. **Directly relevant to VoiceForge's persistent-identity goal**: the mechanism for a *named, reusable* voice in Parler-TTS is literally putting the name in the caption.
- **`PROMPT`** — the v01 vocabulary version.

Note the prompt's own bug: `PROMPT`'s final line ends `... the corresponding description is:"` — a stray double-quote that ships in every v01-generated caption run.

#### ParaSpeechCaps style-prompt generator

Paper §E.4, verbatim. `{all_tags_str}` is a comma-separated list of *all* tags (basic + intrinsic + situational). **[HIGH]**

```
An audio sample of a person's speech can be described in several ways using descriptive keywords. These keywords may include demographic data about the person (e.g. gender, name, accent) and voice characteristics (e.g. related to pitch, gender, texture and rhythm, volume, clarity, speaking rate, emotion, expressiveness).

You will be provided several keywords that describe the speech sample. Your task is to create a simple text description using the provided keywords that accurately describes the speech sample. Ensure that the description remains grammatically correct, easy to understand, and concise. You can rearrange the keyword order as necessary, and substitute synonymous terms where appropriate. After you are provided the keywords, generate only the description and do not output anything else.

An example is provided below.

female, confused, hesitant, slightly noisy environment
Description: A woman's speech sounds confused and hesitant, recorded in a slightly noisy environment.

Now, generate a description for the following example:
{all_tags_str}
Description:
```

Note how much simpler this is than Data-Speech's — one exemplar, no conditional rules. It is a **tag-list-to-sentence** transducer, nothing more. That is the right shape: *the LLM supplies fluency, not perception*, exactly as the brief says.

#### The two other LLM prompts in ParaSpeechCaps

**Acoustic Matching** (Gemini 1.5 Flash, `gemini-1.5-flash-002`, temp 1.0, top-p 0.95) — used to verify situational/emotion tags. The key line for VoiceForge is the anti-content-bias instruction:

```
Analyze the provided speech clip to evaluate how effectively it conveys the emotion {emotion}, focusing on tone of voice and delivery rather than the spoken content.
Key Instructions:
- Focus on Tone: Analyze pitch, tempo, loudness, intonation, and rhythm to judge emotional expression.
- Strength of Emotion: Rate how strongly the tone conveys the emotion on a scale of 1 to 5 (1 = not at all, 5 = very strongly).
- Ignore Content Bias: Evaluate tone and delivery only, disregarding the meaning of the spoken words.
Aspects to Consider:
- Does the pitch and intonation match the energy level of the emotion?
- Is the tempo, rhythm, and loudness appropriate for the emotion?
- Are the tone and delivery consistent with typical characteristics of the emotion?
In your output, start by describing the tone and manner of speaking in the clip. Then, analyze how well the tone aligns with the provided emotion. Finally, rate how strongly the emotion is conveyed on a scale of 1 to 5. To make it easier to parse, format your final answer as follows: "Rating: X/5", where X is the number of your choice.
```

Only clips scoring **5/5** were kept. **[HIGH]**

**Celebrity-prior seeding** (GPT-4 `gpt-4-0125-preview`) — used only to *find candidate speakers* for rare tags, never as ground truth (the paper's own §E.1 heading is "**Imperfectly** labelling celebrities with style tags"). Notably, the attribute list in this prompt is a **larger, earlier** vocabulary than the final 59 — it includes `Hushed`, `Lisp`, `Pitchy`, `Staccato`, `Enunciated` under Rhythm — which is why those strings appear in the raw annotation dumps but not in the final taxonomy.

### 2.4 The ParaSpeechCaps 59-tag taxonomy

Full enumeration, Appendix A of arXiv 2503.04713. **11 style factors. 59 = 28 rich intrinsic + 5 basic intrinsic + 23 rich situational + 3 basic situational.** All four sub-counts verified to sum correctly. **[HIGH]**

**INTRINSIC — speaker-level, persists across utterances (33)**

| Factor | Tags | n |
|---|---|---|
| Pitch *(rich)* | Shrill, Nasal, Deep | 3 |
| Texture *(rich)* | Silky, Husky, Raspy, **Guttural**, Vocal-fry | 5 |
| Clarity *(rich)* | Crisp, Slurred, Stammering | 3 |
| Volume *(rich)* | Booming, Authoritative, Loud, Soft | 4 |
| Rhythm *(rich)* | Flowing, Monotonous, Punctuated, Hesitant, Singsong | 5 |
| Accent *(rich)* | American, British, Scottish, Canadian, Australian, Irish, Indian, Jamaican | 8 |
| Pitch Levels *(basic)* | High-pitched, Medium-pitched, Low-pitched | 3 |
| Gender *(basic)* | Male, Female | 2 |

**SITUATIONAL — utterance-level (26)**

| Factor | Tags | n |
|---|---|---|
| Emotion *(rich)* | Enthusiastic, Happy, Angry, Saddened, Awed, Calm, Anxious, Disgusted, Scared, Confused, Bored, Sleepy, **Pained**, Guilt, Sarcastic, Sympathetic, Admiring, Desirous | 18 |
| Expressiveness *(rich)* | Animated, Laughing, Passive, Whispered, Enunciated | 5 |
| Speed Levels *(basic)* | Fast, Measured, Slow | 3 |

Manually written definitions ship in the paper's Table 5, e.g. *Guttural: "A deep, throaty, gravelly voice."* · *Vocal-fry: "A creaky, breathy voice that occurs when vocal cords flutter and produce a sizzling, popping sound at ends of sentences."* · *Nasal: "A whiny voice that sounds like someone is speaking through their nose."* Use these verbatim — they were shown to annotators, so they define the operational meaning of each label.

**Relevance check for VoiceForge's stated stylization targets (aged, raspy, whispered, breathy, menacing, theatrical):**

| VoiceForge target | Covered by PSC? |
|---|---|
| raspy | ✅ `Raspy` (intrinsic, texture) |
| whispered | ✅ `Whispered` (situational, expressiveness) |
| breathy | ⚠️ only via `Vocal-fry`'s definition; no standalone tag |
| aged | ❌ **not in the taxonomy at all** — PSC has no age axis |
| menacing | ❌ nearest are `Angry`, `Authoritative` |
| theatrical | ❌ nearest is `Animated` |

**Three of six VoiceForge stylization targets have no PSC tag.** LibriTTS-P covers `old` / `young` / `middle-aged` / `mature` and `raspy` / `thick` / `muffled` / `nasal` (§4); the rest need either VoicePersona-v2 vocabulary or a new axis.

### 2.5 Human annotation protocol

Paper §3.1 and Appendix B. **[HIGH]**

- **Platform:** Amazon Mechanical Turk. **Qualification gate:** a screening task; only the **38** workers who passed ≥5 of 6 examples were admitted.
- **Stimulus:** per *speaker*, not per utterance — one concatenated audio file of **3–8 clips totalling 20–40 seconds**, plus the speaker's **name** when available (a leak: annotators knew whose voice it was for VoxCeleb celebrities).
- **Task:** shown the rich intrinsic tag list **with definitions**, write **at least 3 distinct** tags. Free-text entry, not forced choice.
- **Redundancy:** **5 annotations per speaker.**
- **Agreement rule:** keep tags with **≥2 annotators** agreeing for train/dev; **≥3** for holdout.
- **Speaker recruitment (VoxCeleb, 594 celebrities total):** 302 from an IMDb list + a ChatGPT list of "distinctive voices" + the 200 longest Wikipedia pages; then, finding the tag distribution imbalanced, GPT-4 was asked to name celebrities likely to have each of the **12 rarest tags** (*lisp, hushed, pitchy, staccato, monotonous, punctuated, vocal fry, guttural, singsong, soft, stammering, shrill*), ≤40 per tag → **187** more; then **105** random. **This targeted-recruitment step is the single most transferable idea in the paper** — it is how you get coverage of rare voice qualities without annotating a million speakers.
- **Accent and gender were NOT annotated by humans** — taken from metadata for Expresso/EARS, and from **GPT-4 prompted with the celebrity's name** for VoxCeleb (§E.3). That is a knowledge-lookup, not a listening judgement, and it is a quality caveat on the accent labels.
- **Situational tags were not newly annotated at all** — they are Expresso's and EARS's existing style labels remapped to the PSC vocabulary (paper Table 6), with `default`, `narration`, `non-verbal`, `interjection` and `vegetative` styles discarded.

**What the raw annotation dump actually looks like — and why this matters.** I counted tag strings in the two released JSONs. [`pscbase_name_to_intrinsictags.json`](https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/pscbase_name_to_intrinsictags.json) (707 speakers, all annotations before the ≥2 filter) contains **551 distinct raw strings**, including typos that survived to release (`Enuciated`, `Gutteral`, `Hogh-pitched`, `Heaitant`, `Stmamering`, `Flowingnjmmmmmh`), multi-tag cells (`Flowing. measured`), and a long tail of ~380 singletons (`Clairvoyant`, `Motherly`, `Angelic`, `sexy`). The **published 33-tag ontology is a curated head of a free-text distribution**, not a closed choice set the annotators were restricted to. **[HIGH — computed directly from the released files]**

Two consequences for VoiceForge:
1. **Free-text collection then curation is the right protocol** — it discovers vocabulary you would not have listed (the frequency table's head is: Flowing 1147, Measured 1078, Deep 750, Crisp 731, Fast 562, Loud 452, Nasal 445, Booming 434, Authoritative 429, Silky 412, Enunciated 391, Husky 378, Hesitant 360, Raspy 352, Soft 293, Singsong 278, Shrill 258, Slurred 255, Stammering 210, Slow 208, Monotonous 202, Punctuated 183, Vocal-fry 136, Pitchy 135, Guttural 117, Staccato 91).
2. **Budget for a normalisation pass.** ~30% of raw strings need merging or discarding.

**Scaling pipeline 1 — intrinsic tags via perceptual speaker similarity (released code, reusable).** Median **VoxSim** embedding over 10 random utterances per speaker; for each annotated VoxCeleb speaker, copy tags to every Emilia speaker with **cosine ≥ 0.8** (= VoxSim rating 5/6). Only **18 of 28** rich intrinsic tags propagate — all *except* clarity tags (Crisp/Slurred/Stammering), accent, and the ones later dropped: the README lists exactly *Shrill, Nasal, Deep, Silky, Husky, Raspy, Guttural, Vocal-fry, Booming, Authoritative, Loud, Soft, Flowing, Monotonous, Punctuated, Hesitant, Singsong, Enunciated*. 9× data multiplication → 2,427 hrs. Ablation: swapping VoxSim for a **standard WavLM-Large ECAPA-TDNN** embedder (threshold tuned to 0.41 for equal speaker count) **degrades** tag recall — perceptual similarity ≠ verification similarity. **[HIGH]** *This is a directly relevant finding for VoiceForge's identity space: a verification-trained embedder does not preserve the perceptual axes that descriptions refer to.*

> 🔴 **Blocker on reusing this step:** the VoxSim model is required, and [`github.com/kaistmm/voxsim_trainer`](https://github.com/kaistmm/voxsim_trainer) has **no LICENSE file** (tree listed, 404 on LICENSE) — i.e. all rights reserved by default. Weights are distributed via a Google Drive link. Unusable for a commercially released artefact without permission from the authors. **[HIGH]**

**Scaling pipeline 2 — situational tags, 3-stage cascade (code NOT released).** (a) **Expressivity filtering** with the [audEERING dominance-valence-arousal model (Wagner et al. 2023)](https://arxiv.org/abs/2203.07378): keep utterances with any DVA value `< 0.35` or `> 0.75`, then apply emotion-specific direction constraints. (b) **Semantic matching** with SFR-Embedding-Mistral over transcripts, query template `Instruct: Given an emotion, retrieve relevant transcript lines whose overall style/emotions matches the provided emotion. Query: {emotion}`, with keyword-blocklist filtering to stop the retriever cheating on literal emotion words. (c) **Acoustic matching** with Gemini 1.5 Flash on the top 100k per emotion, keep only 5/5. 3× multiplication → 215 hrs. The paper's Table 2 ablation shows removing **any** of the three stages degrades quality. **[HIGH]**

**Verified reported gains — the brief's numbers are correct, with important context.** Paper Table 3 (arXiv 2503.04713v2), human evaluation, 246-example tag-balanced holdout, Parler-TTS Mini v1 finetunes, CFG scale 1.5:

| Model | CMOS ↑ | Intr Tag Recall | Sit Tag Recall | NMOS ↑ | IMOS ↑ | WER ↓ |
|---|---|---|---|---|---|---|
| Ground truth | 4.42 ±0.07 | 88.7% | 88.6% | 4.36 ±0.07 | 4.28 ±0.06 | 7.93 |
| Parler-TTS (base) | 3.05 ±0.08 | 33.0% | 21.2% | 2.85 ±0.07 | 4.31 ±0.07 | 4.62 |
| +LibriTTS-R | 3.07 ±0.08 | 33.7% | 22.4% | 2.95 ±0.07 | **4.44 ±0.06** | **4.47** |
| +LibriTTS-P,Expresso,EARS *(best baseline)* | 3.55 ±0.08 | 40.7% | 69.7% | 3.10 ±0.07 | 4.19 ±0.07 | 7.14 |
| **PSC-Base** | 3.75 ±0.08 | 63.6% | 68.1% | 3.27 ±0.08 | 4.05 ±0.07 | 9.14 |
| **PSC-Scaled** | **3.83 ±0.08** | **69.5%** | **75.4%** | **3.58 ±0.07** | 4.07 ±0.07 | 8.63 |

- **+7.9% Consistency MOS** = 3.55 → 3.83, i.e. **+0.28 absolute** on a 5-point Likert, expressed as a relative percentage. ✅ **verified**
- **+15.5% Naturalness MOS** = 3.10 → 3.58, i.e. **+0.48 absolute**. ✅ **verified**
- Baseline is `+LTTSP,Exp,EARS`, **not** vanilla Parler-TTS. Against vanilla Parler-TTS the gains would be much larger (+25.6% CMOS, +25.6% NMOS).
- **Caveats the brief should carry:** (i) these are *relative percentages of a Likert mean*, a flattering presentation — the CMOS delta of 0.28 is ~3.5× the 95% CI half-width, so real but modest; (ii) **intelligibility got worse** (IMOS 4.44 → 4.07, WER 4.47 → 8.63) and the paper argues, credibly, that this is partly *faithfulness* — a model that actually renders `slurred` and `stammering` should score lower on intelligibility; (iii) the biggest single win is **Intrinsic Tag Recall 40.7% → 63.6% from human data alone**, before any scaling. Human intrinsic annotation is where the value is. **[HIGH]**
- One more useful ablation: **inference-time classifier-free guidance at scale 1.5 improves style consistency for every model**, even though none were trained with CFG dropout (paper §5.5, Table 4). Free win; adopt it.

---

## 3. Corpus audit

Every licence below was read from a primary artefact — the raw HF card YAML, a `LICENSE` file fetched directly, the OpenSLR resource page, the official terms page, or the paper. "Train-and-release OK?" means: *may we train a model on this and release that model for commercial use?*

| Corpus | Hours | Speakers | Langs | Descriptions? | SR | Licence | Licence source | Train-and-release OK? | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **ParaSpeechCaps** | 2,769 total (2,427 auto + 342 human across splits; **282 human in *train*** — see note) | 39,002 scaled / 641 base / 624 dev / 167 holdout | en | ✅ **native**, 1–2 free-form captions/utt, 59 tags | n/a — **no audio shipped** | **CC-BY-NC-SA-4.0** (annotations); code MIT | [HF card YAML](https://huggingface.co/datasets/ajd12342/paraspeechcaps/raw/main/README.md) `license: cc-by-nc-sa-4.0`; repo README | 🔴 **NO** — NC **and** ShareAlike | **Read it, learn from it, do not train a shipped model on it.** Best rich-tag data in existence; legally unusable for VoiceForge's stated commercial goal. Its audio chain is worse still (below). |
| **LibriTTS-P** | 585 (all of LibriTTS-R) | **2,443** | en | ✅ **native** — 373,868 prompts: templated style prompts + human speaker-identity adjective lists | 24 kHz | **CC BY 4.0** | [repo README §License](https://raw.githubusercontent.com/line/LibriTTS-P/main/README.md) — ⚠️ **no LICENSE file**, GitHub API reports `license: null` | 🟢 **YES**, with attribution | ⭐ **The single most important corpus for this project.** Only permissively-licensed rich *intrinsic* vocabulary that exists. 2,443 speakers with human-annotated identity adjectives. |
| **LibriTTS-R** | 585 | 2,456 | en | ❌ | 24 kHz | **CC BY 4.0** | `LICENSE.txt` inside [openslr.org/resources/141/doc.tar.gz](https://www.openslr.org/141/): *"made available by Google LLC under a Creative Commons Attribution 4.0 International License"* | 🟢 **YES** | ⭐ The audio under LibriTTS-P. Clean, restored, permissive. Download first. |
| **TextrolSpeech** | 330 | 1,324 | en | ✅ 236,220 GPT-written prompts, 5 style factors | 24 kHz | Repo `LICENSE` = **MIT (code only)**. **Dataset licence: UNVERIFIED — none stated anywhere** | [repo LICENSE](https://raw.githubusercontent.com/jishengpeng/TextrolSpeech/main/LICENSE); paper arXiv 2308.14430 | 🔴 **NO / blocked** | Built from LibriTTS+VCTK (fine) **+ ESD, MEAD, SAVEE, TESS, MESS** (research-only). The MIT file is a trap — it covers code. |
| **SpeechCraft** | 2,391 (Tab. 2) / 2,381.54 (Tab. 4) / "~2,000" (abstract) — inconsistent in its own paper | >3,200 | en + zh | ✅ 2,108,710 LLM-written descriptions, 8 attributes incl. word-level emphasis | mixed 16–24 kHz, not standardised | **UNVERIFIED — no LICENSE file, no statement in README or paper.** Emphasis subset gated by signed EULA: *"non-commercial research and/or educational purposes"* | [repo README](https://raw.githubusercontent.com/thuhcsi/SpeechCraft/master/README.md) (note: branch is **`master`**); GitHub API `license: null` | 🔴 **NO** — absence of a licence is not permission | Large and bilingual, but unlicensed. Do not use. |
| **Emilia** | 101.7k (Emilia) + 113.9k (Emilia-YODAS) = 215.6k | not stated (diarised in-the-wild) | zh en ja fr de ko | ❌ (transcripts + DNSMOS + speaker ids) | 24 kHz | ⚠️ **HF YAML says flat `cc-by-4.0` but the card body and gate say: Emilia = CC BY-**NC**; Emilia-YODAS = CC BY.** Gate binds a for-profit employer. | [HF card + `extra_gated_prompt`](https://huggingface.co/datasets/amphion/Emilia-Dataset) | Emilia 🔴 **NO** · YODAS 🟡 nominally yes | **If you trust the HF licence tag you will misclassify 101.7k NC hours as CC-BY.** YODAS is CC BY 4.0 but the underlying YouTube copyright was never the authors' to license. |
| **MLS** | 50,686.74 train (44.5k en) | 6,052 train (5,490 en) | en de nl fr es it pt pl | ❌ | 16 kHz | **CC BY 4.0** | [openslr.org/94](https://www.openslr.org/94/); [HF card YAML](https://huggingface.co/datasets/facebook/multilingual_librispeech/raw/main/README.md) | 🟢 **YES** | Public-domain LibriVox source. Large but audiobook-narrow. Data hosted at `dl.fbaipublicfiles.com/mls/`, not OpenSLR. |
| **VoxCeleb2** | 2,442 | **6,112** | multi (uncontrolled) | ❌ | UNVERIFIED | **Metadata only: CC BY-SA 4.0. Audio: no licence granted, ever.** | [vox2.html terms](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) | 🔴 **NO** | 🚨 **AND THE DOWNLOAD IS GONE.** The official page now states the URLs, timestamps, audio files, video files **and** identifying metadata "are no longer available from this website." Only VoxCeleb1 trial lists remain. **[HIGH]** |
| **Common Voice** | **Scripted Speech 26.0** (2026-06-12): 42,388 total / 28,893 validated | 378,025 contributor-accounts (counted per-language) | **294** | ❌ (age band / gender / accent tags) | MP3, rate UNVERIFIED | **CC0-1.0** | [cv-dataset release JSON](https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/cv-corpus-26.0-2026-06-12.json); Mozilla Data Collective listings | 🟢 **YES**, no attribution required | ⭐ **Best speaker-count-per-licence-dollar in existence.** ⚠️ HF `mozilla-foundation` is stale — only v13 and v17; v18–26 are 404. Fetch from Mozilla Data Collective. Also note Spontaneous Speech is now a separate v4.0 (78 langs). |
| **GigaSpeech** | 10,000 transcribed / 33,005 total | **not provided** (`"speaker": "N/A"`) | en | ❌ | 16 kHz Opus | Repo Apache-2.0 (**scripts only**). Audio gate: *"Researcher shall use the Database only for **non-commercial research and educational purposes**"*, clause 6 binds the employer | the actual [Google Form Terms of Access](https://docs.google.com/forms/d/e/1FAIpQLSfmLDh0gxzST7n-os72-ZmgyRPot67RFvhwJNIngBYILYveIQ/viewform) | 🔴 **NO** | Also useless here: **it has no speaker labels at all.** |
| **People's Speech** | 30,000+ | not stated | en (+23 minor) | ❌ | 16 kHz | **CC-BY-4.0 / CC-BY-SA-4.0** (YAML also lists cc-by-2.0/2.5/3.0, cc-by-sa-3.0) | [HF card YAML](https://huggingface.co/datasets/MLCommons/peoples_speech/raw/main/README.md); [arXiv 2111.09344](https://arxiv.org/abs/2111.09344) | 🟢 **YES** — designed for it | Paper *explicitly* excluded CC-BY-NC "because our dataset is intended for downstream commercial usage", and ships a CSV to strip the 3,100 SA hours. Use `clean`/`dirty`, avoid `clean_sa`/`dirty_sa`. |
| **EARS** | 100 | 107 | en | ❌ but **7 reading styles × 22 emotions** + 6-field speaker demographics | **48 kHz anechoic** | 🔴 **CC BY-NC 4.0** | [LICENSE](https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/LICENSE) line 1: `Attribution-NonCommercial 4.0 International` | 🔴 **NO** | Richest *structured* style metadata of any corpus here, and the best audio quality. Entirely blocked by NC. Painful. |
| **Expresso** | 45.9 (11.5 read + 34.4 improv) — abstract says "40 h", the table says 45.9 | **4** (2M/2F) | en | ❌ but **26 categorical styles** | **48 kHz / 24-bit** | 🔴 **CC BY-NC 4.0** | [dataset README](https://raw.githubusercontent.com/facebookresearch/textlesslib/main/examples/expresso/dataset/README.md) | 🔴 **NO** | 4 speakers — near-zero value for speaker diversity anyway. |
| **VCTK 0.92** | not stated (110 × ~400 sents) | 110 | en (many accents) | ❌ | **48 kHz / 16-bit** (from 96/24) | **CC BY 4.0** | DSpace item metadata `dc.rights` for [handle 10283/3443](https://datashare.ed.ac.uk/handle/10283/3443) **and** the corpus's own `README.txt` | 🟢 **YES** | ⭐ Studio-grade, accent-diverse, permissive. **The right base for DSP stylization augmentation** (§7). Note p280/p315 had mic faults. 11.7 GB. |
| **DailyTalk** | 20 | **2** | en | ❌ (emotion/act/topic inherited from DailyDialog) | 22.05 kHz (pipeline config) | ⚠️ **CC BY-SA 4.0 badge vs. "freely available for *academic use*" in both README and paper** — genuine conflict; repo `LICENSE` is MIT and covers the code | [repo README](https://github.com/keonlee9420/DailyTalk); [arXiv 2207.01063](https://arxiv.org/abs/2207.01063) | 🟡 unclear | 2 speakers. Not worth resolving the ambiguity for. **Skip.** |
| **VoicePersona** (own) | 48.7 claimed | 10,179 claimed | **en only** (card YAML says `en`) | ✅ Qwen2-Audio free-text | 16 kHz | **`license: cc`** on HF (not CC0); upstreams Apache-2.0 / CC0 / MIT / **CC BY-NC** | [HF card YAML](https://huggingface.co/datasets/Paranoiid/VoicePersona/raw/main/README.md) | 🔴 **NO as it stands** | See §6. Ships 5.0 GB of audio, 26.5% of it anime-derived with no rights basis. |

**Cross-refs:** IndicVoices-R, RASMALAI and the Indic corpora are [04-indic-track.md](04-indic-track.md)'s domain and are deliberately not audited here. Licence-propagation-to-weights is [08-licensing-propagation.md](08-licensing-propagation.md)'s.

### 3.1 The finding that reshapes the plan

**ParaSpeechCaps ships no audio, and every one of its four audio sources is unusable.** The dataset card is explicit: it provides annotations plus `relative_audio_path`, and you must obtain audio from **VoxCeleb, Expresso, EARS and Emilia** yourself. Their status:

| PSC audio source | Status |
|---|---|
| VoxCeleb1 + 2 | 🚨 **No longer distributed by Oxford at all.** No licence for the audio ever existed. |
| Expresso | 🔴 CC BY-NC 4.0 |
| EARS | 🔴 CC BY-NC 4.0 |
| Emilia (EN) | 🔴 CC BY-NC 4.0, gated, binds a for-profit employer |

So ParaSpeechCaps is **triply blocked**: NC-SA annotations, NC audio, and one source that is no longer obtainable. The scope doc marks it "⭐ **Primary English training set**". **That is not viable and it is the biggest single correction in this document.** [HIGH]

**What replaces it.** The permissively-licensed core is:

| Role | Corpus | Licence | Hours | Speakers |
|---|---|---|---|---|
| Rich intrinsic descriptions | **LibriTTS-P** | CC BY 4.0 | 585 | 2,443 |
| Its audio | **LibriTTS-R** | CC BY 4.0 | 585 | 2,456 |
| Studio quality + accents + DSP base | **VCTK 0.92** | CC BY 4.0 | ~44 | 110 |
| Raw speaker diversity | **Common Voice SS 26.0** | CC0 | 28,893 valid | ~100k+ (en: 100,172 accounts) |
| Bulk English | **MLS** | CC BY 4.0 | 44,500 (en) | 5,490 (en) |
| Bulk English, commercial-intent | **People's Speech** (`clean`/`dirty`) | CC BY 4.0 | ~26,900 | not stated |

Descriptions exist natively only for LibriTTS-P's 2,443 speakers. **Everything else must be annotated by VoiceForge's own pipeline (§2, §6.3) — which is precisely why C1 matters more than the corpus shortlist does.**

---

## 4. Corpora the brief missed

The brief's corpus list stops at 2025. Six things it should have.

| Corpus | What it is | Hours / speakers | Langs | Licence | Why it matters here |
|---|---|---|---|---|---|
| ⭐ **RASMALAI** ([arXiv 2505.18609](https://arxiv.org/abs/2505.18609), AI4Bharat, 2025) | **13,000 h with 24M natural-language style descriptions**; finetune subset 1,804 h. Built over IndicVoices (16,237 spk), RASA, IndicTTS, LIMMITS and **GLOBE**. Three prompt styles: Descriptive / Concise / Attribute-Robust. Ships `ai4bharat/indic-parler-tts`. | 13,000 h | **24 (23 Indic + EN)** | **CC BY 4.0** | 🚨 **The brief lists RASMALAI only as an Indic-track item. It is also, by a wide margin, the largest permissively-licensed style-captioned corpus in existence — and it includes English.** It should be in the *English* plan too, not just [04-indic-track.md](04-indic-track.md). Its 24M-description generation is a fourth independent implementation of measure-first captioning. |
| ⭐ **GLOBE** ([described in 2505.18609](https://arxiv.org/abs/2505.18609); the `GLOBE_V2` VoicePersona already uses) | Common Voice refined for TTS | **535 h from 23,519 speakers**, 164 accents | en | **CC0-1.0** (per [GLOBE_V2 card](https://huggingface.co/datasets/MushanW/GLOBE_V2/raw/main/README.md)) | **~82 seconds per speaker — the single best public match to the project's hypothesised regime** (§5). CC0. VoicePersona already ships 3,146 samples of it; the full corpus is 7× larger. **Download this early.** |
| **CapSpeech** ([arXiv 2506.02863](https://arxiv.org/abs/2506.02863), 2025) | >10M machine-annotated + ~0.36M human-annotated caption–audio pairs; separates pretraining from finetuning. Cited by VoiceDesigner as the current best instruction-following corpus. | very large | en | ⚠️ **CC BY-NC 4.0** | The most-cited 2025 style-captioned corpus and the brief does not mention it. **NC, so same blocker as ParaSpeechCaps** — read it, do not train on it. HF org `OpenSound/CapSpeech-*`. |
| **VoxBlink2** ([arXiv 2407.11510](https://arxiv.org/abs/2407.11510), Interspeech 2024) | **111,284 speakers**, 9,904,382 utterances | — | multi | check before use | No captions, but the **largest speaker-count corpus that exists**. Relevant only if the frozen speaker encoder is ever retrained ([02-identity-representation.md](02-identity-representation.md)). |
| **NonverbalTTS** ([arXiv 2507.13155](https://arxiv.org/abs/2507.13155)) | 17 h, 10 nonverbal-vocalization types + 8 emotions, auto-detected then human-validated | 17 h | en | open-access | Covers laughs, sighs, gasps — directly relevant to *dramatic content* and to VoiceForge's stylization targets, which no other corpus here covers. Small enough to use immediately. Derived from VoxCeleb + Expresso, so **check the upstream chain**. |
| **EmoVoice-DB** ([arXiv 2504.12867](https://arxiv.org/abs/2504.12867), ACM MM 2025) | 40 h, freestyle natural-language *emotion* descriptions | 40 h | en | see repo | Notable as a **counter-example**: its captions are distilled from GPT-4o-audio, i.e. exactly the audio-LM-as-describer approach §6.2 argues against. Worth reading as a comparison point for the v2 ablation. |

**Also verified as real but lower priority:** LibriTTS-VI ([2509.15626](https://arxiv.org/abs/2509.15626), 11-dim numerical voice-impression vectors rather than text — interesting as a *target representation*, see 02), VoxInstruct ([2408.15676](https://arxiv.org/abs/2408.15676), model not corpus), Audio-FLAN ([2502.16584](https://arxiv.org/abs/2502.16584)), OV-InstructTTS ([2601.01459](https://arxiv.org/abs/2601.01459)).

**Checked and NOT real — do not cite:** "**SpeakerVerse**" (no arXiv, no HF dataset — **UNVERIFIED, likely a hallucination**), "**VccmDataset**" (**UNVERIFIED**, no arXiv or HF match), **Audiobox/Voicebox caption data** (**UNVERIFIED that Meta ever released any caption corpus**; both ParaSpeechCaps and Parler-TTS discuss Audiobox as concurrent closed work).

---

## 5. C3 — speaker count vs hours

### 5.1 The direct answer: nobody has published this for a mapper

> **UNVERIFIED — stated plainly.** No published work runs a speaker-count-vs-hours ablation for a description→speaker-embedding mapper. Every system in that exact family reports its corpus and runs **no** ablation on the number of training speakers: PromptSpeaker, TacoSpawn, VoiceLens, PromptTTS 2, Parler-TTS, LibriTTS-P, CapSpeech, VoiceDesigner, MOSS-VoiceGenerator, CapTalk. Nor is there any 2024–2026 *scaling law* study for zero-shot or prompt-based TTS that varies speaker count as an independent axis — MaskGCT (100k h), MegaTTS 3 (600k h), IndexTTS (34k h), XTTS (27k h) all report **hours only**. **[HIGH confidence in the absence — checked directly]**

What the family actually trained on, for calibration:

| System | Data | Hours/speaker |
|---|---|---|
| PromptSpeaker ([2310.05001](https://arxiv.org/abs/2310.05001)) | 792 speakers, 21,760 utterances (~25–30 h) across Internal Stylistic (74 spk × 100), AISHELL-3 (218 × 20), DiDiSpeech (500 × 20) | **~2 min** |
| TacoSpawn ([2111.05095](https://arxiv.org/abs/2111.05095)) | libriclean 1,230 spk / 240 h · **enus1100: 1,100 spk × 30 min** · en1468 1,468 spk / 717 h | 12–30 min |
| VoiceLens ([2309.14094](https://arxiv.org/abs/2309.14094)) | DiDiSpeech-2, 1,489 speakers with >100 utterances | ~few min |
| LibriTTS-P ([2406.07969](https://arxiv.org/abs/2406.07969)) | 2,443 speakers / 585 h | ~14 min |
| **Parler-TTS** ([2402.01912](https://arxiv.org/abs/2402.01912)) | MLS English **44,659.74 h / 5,490 speakers** + LibriTTS-R | **~8.1 h** |

**Note a common misreading**: Parler-TTS's "34 speakers, characterized by name" refers to the set of *consistently-nameable* voices, not its training speaker inventory. The real inventory is 5,490 MLS speakers at ~8.1 h each — **Parler-TTS sits at the opposite end of the design space from VoiceForge's hypothesis.** Whether that explains its limited voice diversity is *inference, not published*. **[MEDIUM]**

### 5.2 The belief is nonetheless SUPPORTED — by fixed-budget ablations in adjacent literature

Three independent fixed-budget experiments, none in the mapper family, all pointing the same way.

**(a) The decisive one — Vaessen & van Leeuwen, "Training speaker recognition systems with limited data", Interspeech 2022** ([ISCA proceedings PDF](https://www.isca-archive.org/interspeech_2022/vaessen22_interspeech.pdf)). Three VoxCeleb2 subsets at a **fixed ~100-hour / ~50k-utterance budget**, varying speaker count 60×. Vox1-O EER (%), lower is better:

| Training set | Hours | Speakers | Sessions/spk | X-vector | **ECAPA-TDNN** | wav2vec2 |
|---|---|---|---|---|---|---|
| vox2 (full reference) | 2,314 | 5,994 | 22.8 | 6.30 | 2.91 | 2.40 |
| tiny-**few-speakers** | 113 | **100** | 50.7 | 12.91 | **12.19** | 7.46 |
| tiny-**few-sessions** | 100 | **5,994** | **1.0** | 21.70 | **15.97** | 15.72 |
| tiny-**many-sessions** | 97 | **5,994** | 7.8 | 9.75 | **6.04** | 3.60 |

**At constant hours, 60× more speakers halves EER (12.19 → 6.04 ECAPA; 7.46 → 3.60 wav2vec2).** Per-speaker audio in that winning arm is **~58 seconds** — almost exactly the brief's "1,000 speakers × 3 min" regime. **[HIGH — numbers read from the proceedings PDF]**

> 🚨 **AND HERE IS THE QUALIFIER THE BRIEF OMITS.** The same table shows that 60× more speakers *without session diversity* is **worse than 100 speakers** (ECAPA 12.19 → **15.97**). The operative variable is **speakers × recording sessions**, not speakers alone. **[HIGH]**
>
> **This changes the corpus strategy.** "1,000 speakers × 3 min" only wins if those 3 minutes span several distinct recordings. Check every candidate corpus for it:
> - **Common Voice / GLOBE** — each clip is an independent recording session. ✅ Ideal.
> - **LibriTTS-R** — chaptered audiobooks; a speaker's clips span multiple chapters. ✅ Usable, and the chapter field lets you *control* the variable.
> - **VCTK** — 110 speakers, one studio session each. ⚠️ Session-poor by construction.
> - **VoicePersona's LAION slice (52.6%)** — synthetic TTS, effectively **zero** session diversity. ⚠️ On this evidence it may actively hurt.

**(b) Truong et al., ICASSP 2024** ([arXiv 2309.14838](https://arxiv.org/abs/2309.14838)), §2.1 verbatim: *"We trained the x-vector model using a fixed 100,000 training utterances of the VoxCeleb 2 dev set… increasing the number of speakers will lead to fewer training utterances per speaker. As depicted in Figure 1, the performance of the x-vector model consistently improves with an increasing number of speakers."* Direction **[HIGH]**; the per-point values are plot-only and **UNVERIFIED**.

**(c) The most on-point result — [arXiv 2512.17356](https://arxiv.org/abs/2512.17356)**, TTS trained on purely synthetic data, fixed utterance budget, speakers swept to 1,000. Verbatim (§4.2): *"when the number of speakers exceeds 50, improvements in Word Error Rate (WER) and Unweighted Task Mean Opinion Score (UTMOS) begin to plateau… Interestingly, the turning point for the SIM metric occurs much later, at around 500 speakers, implying that speaker diversity plays a more critical role in accurately modeling speaker-specific characteristics than in capturing speech pronunciation."* **[HIGH]**

**This is the number to plan against: intelligibility and quality saturate at ~50 speakers; speaker *similarity* keeps improving to ~500 and beyond.** VoiceForge's whole product is the metric with the late knee.

**Do not cite** [arXiv 2204.06450](https://arxiv.org/abs/2204.06450) as fixed-budget evidence, even though its numbers are attractive (EER 5.19% at n=50 → 0.90% at n=3,000, log fit R²=0.95). That experiment *expands* the speaker pool, so total data grows with speaker count. It shows "more speakers → logarithmically better", not "speakers beat hours". **[HIGH on the numbers, but the framing matters]**

### 5.3 Two 2026 results on data composition beating data volume

Not speaker-count studies, but the same lesson at corpus level, and both are strong:

- **CapTalk** ([arXiv 2604.08363](https://arxiv.org/abs/2604.08363)) §5.2.1, verbatim: *"1500-hour conversational data underperforms 300-hour acted data, whereas 5000-hour conversational data improves substantially, and adding 300 hours of acted speech yields the best result (AVG 73.73)."* **5× more hours lost to a more diverse 300 h.** **[HIGH]**
- **MOSS-VoiceGenerator** ([arXiv 2603.28086](https://arxiv.org/abs/2603.28086)), verbatim: *"we attempted to mix in approximately 10,000 hours of TTS-base data (without instructions…). However, we observed no significant gain in either objective benchmarks or human evaluation."* **[HIGH]**
- Note also that **ParaSpeechCaps' own 8× scaling confounds the two axes** — PSC-Scaled grew speakers *and* hours together (641 → 39,002 speakers, 282 → 2,427 h), so its 3.73 → 3.83 CMOS gain says nothing about which axis mattered. The paper is explicit about the mechanism; the confound is not hidden, but it is a confound.

**Summary verdict on the brief's claim.** *"Speaker count matters more than hours"* is **directionally correct and better supported than the brief itself knows** — but it should be restated as:

> **Speaker count × recording-session diversity dominates hours, at fixed budget, for speaker-identity metrics specifically. The knee is around 500 speakers, not 50. Many speakers with one session each is worse than few speakers.**

### 5.4 The cheapest ablation that settles it for VoiceForge

Since the literature does not answer the question in the form the project needs, design the experiment that does. It is small enough to run in one evening on one 8–12 GB card.

**The claim under test:** *for a description→speaker-embedding mapper, N speakers × (T/N) minutes beats (N/k) speakers × (kT/N) minutes at fixed total budget T — and this holds only with session diversity.*

**Why it is nearly free:** targets are `z = E(audio)` from a **frozen** encoder. Encode the audio **once**, cache `z` (~1 KB/utterance; 500k utterances ≈ 500 MB), and every arm below trains on a table of vectors with zero audio I/O.

**Corpus:** **LibriTTS-P** descriptions over **LibriTTS-R** audio (both CC BY 4.0, so the result is publishable). 2,443 speakers is enough to sweep two decades, the prompts are **human-annotated** rather than templated, and the audio is **chaptered — so session diversity can be varied**, which §5.2 proves is the confound that matters. Optional second arm: **GLOBE** (23,519 speakers, CC0) for the extreme-speaker-count end.

**The ladder — fix total speaker-minutes B, sweep N:**

| Arm | Speakers | Min/speaker | B |
|---|---|---|---|
| A | 50 | 30 | 1,500 |
| B | 150 | 10 | 1,500 |
| C | 500 | 3 | 1,500 |
| D | 1,500 | 1 | 1,500 |

**Plus the cross-arm that actually matters:** run C and D twice — (i) all audio from **one chapter** per speaker, (ii) audio from **≥4 distinct chapters**. This is the direct test of the Vaessen confound and the arm most likely to overturn a naive reading of the belief. 3 seeds per cell.

**Metrics** — all in speaker-embedding space, no vocoder needed:
1. **Distinct-voice count ω(G)** — VoiceLens's clique number: how many sampled speakers are mutually ≥ d cosine apart, with d = the typical nearest-training-speaker distance. *Primary metric — it measures "how many different voices can this thing actually make".* ([arXiv 2309.14094](https://arxiv.org/abs/2309.14094))
2. **TacoSpawn distance battery** — `gen2gen-near` vs `train2train-near` vs `gen2train-near` ([arXiv 2111.05095](https://arxiv.org/abs/2111.05095)). Catches mode collapse *and* memorisation of training speakers, which are different failures.
3. **Attribute adherence** — freeze a linear probe (gender / pitch tercile / rate tercile) on real `z`; check sampled `z` matches the caption's bins.
4. **Distribution match** — MMD or Fréchet distance between generated `z` and **held-out real speaker** `z`.

**Evaluate on held-out SPEAKERS, not held-out utterances.** An utterance-level split lets a memorising model score well and would invert the conclusion.

**Cost:** one-time encoding of the LibriTTS-R subset ≈ **3–6 GPU-hours**; the full 4-arm × 2-session × 3-seed sweep (24 runs on cached vectors) ≈ **<2 GPU-hours**. **Under 10 GPU-hours total.**

**What falsifies the belief:**
- If ω(G) and MMD are flat or *better* for arm A (50 spk × 30 min) at equal B → the belief is false as stated; optimise for audio quality and caption richness instead.
- If C-single-session ≈ A, the gain is **entirely session diversity**, not speaker count → re-specify the data-collection strategy around recording diversity, and reconsider the 52.6% synthetic slice of VoicePersona.
- If ω(G) saturates below N=500, the §5.2(c) knee does not transfer and hours-per-speaker should be re-prioritised.

**Publish it.** A clean iso-budget speaker-count × session-diversity ladder for a description→speaker-embedding mapper does not exist in the literature (§5.1), it costs one evening, and it converts the project's central data assumption from an assertion into a result.

---

## 6. VoicePersona: audit and the v2 rebuild spec

### 6.0 What the artefact actually is

| Claim in scope doc | Verified value | Verdict | Source | Conf |
|---|---|---|---|---|
| 15,082 samples | `statistics.json` says 15,082. **The published HF artefact has 14,327 rows** (train 9,862 + val 2,465 + test 2,000) | **CONTRADICTED** — 755 samples (5.0%) claimed but not shipped | [datasets-server size API](https://datasets-server.huggingface.co/size?dataset=Paranoiid%2FVoicePersona); [statistics.json](https://raw.githubusercontent.com/PranavMishra17/VoicePersona-Dataset/main/analyse/statistics.json) | HIGH |
| 10,179 speakers | `"unique_speakers": 10179` — on the 15,082 base | Plausible, unverified independently | statistics.json | MEDIUM |
| 48.7 hrs | `48.67201394053917` — on the 15,082 base | Same caveat | statistics.json | MEDIUM |
| **8+ languages** | HF YAML declares **`language: - en`** only. GLOBE_V2 = English; the AniSpeech split is literally named `ENGLISH`; AnimeVox = English dubs; LAION tarballs are 428 `english_*` vs 5 `german_*` | **FALSE.** The "8+ languages" line describes *Qwen2-Audio's* capability and was restated as a dataset property | [HF raw card](https://huggingface.co/datasets/Paranoiid/VoicePersona/raw/main/README.md) | HIGH |
| 702 accent variants | `"unique_accents": 702` confirmed as a raw count | **MISLEADING** — unnormalised free-text. Top-10 contains "General American", "United States English", "American English", "US accent" and "English" as five separate entries, plus "Valley girl accent" | statistics.json | HIGH |
| ~500-char descriptions | Card says ~500. **The repo's own `analyse/analysis_report.txt` measures 307** | **CONTRADICTED by its own output** | analysis_report.txt | HIGH |
| Qwen2-Audio-7B-Instruct | `src/config.py`: `MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"` | CONFIRMED | src/config.py | HIGH |
| Female 62.6% / twenties 76.1% | 9,448/15,082 and 11,481/15,082 | CONFIRMED — but see §6.2, these are **generated labels, not observed demographics** | statistics.json | HIGH |
| **CC0** | HF YAML says **`license: cc`**, not `cc0-1.0`. The CC0 text exists only in the GitHub repo's `LICENSE`; the HF repo (which ships the 5.0 GB of audio) has **no LICENSE file** | **NOT AS CLAIMED** | HF raw card; HF API `cardData.license`; HF repo tree | HIGH |

Also present in the shipped statistics: an age bucket `"techarties": 2` — a hallucinated category that survived to release. `unknown` gender/age = 275; `unknown` accent = 792. Generation is `TEMPERATURE = 0.7`, `TOP_P = 0.95`, `DO_SAMPLE = True`, `MAX_NEW_TOKENS = 256` — **non-deterministic, so the labels are not reproducible**.

One structural fact that makes everything below binding rather than academic: **VoicePersona redistributes the source audio**, not just derived text. `download_size: 4,999,234,370` (~5.0 GB), `audio` feature at 16 kHz.

### 6.1 Upstream licence chain (the real risk)

**Verdict: the CC0 declaration does not hold. Four independent grounds, each sufficient alone.**

| Upstream | Share | Exact licence string | Provenance | CC0-compatible? |
|---|---|---|---|---|
| `laion/laions_got_talent` | 7,937 (52.6%) | YAML `license:` **absent**; repo contains an undeclared `LICENSE` = **Apache-2.0** | **Synthetic** — "the OpenAI Voice API via Hyprlab", gpt-4o-audio. Tarballs named `english_alloy_*` (`alloy` = an OpenAI TTS voice) | **NO** — §4 notice/attribution conditions; OpenAI ToS position unresolved |
| `MushanW/GLOBE_V2` | 3,146 (20.9%) | `license: cc0-1.0` | Common Voice 14 derivative (CC0), resampled to 44.1 kHz, ~5% removed | **YES** |
| `ShoukanLabs/AniSpeech` | 2,000 (13.3%) | `license: mit`; licence file reads "Copyright 2023 ShoukanLabs" and governs "the Software" | "captioned anime voices" — **no source anime, studio or rights basis named anywhere** | **NO** — attribution condition; no rights basis |
| `taresh18/AnimeVox` | 1,999 (13.3%) | YAML `license: cc`; **card body: "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"** | Card, verbatim: *"Audio clips were sourced from official English-dubbed versions of popular anime series."* 15 series, 19 characters, extracted with TTSizer | **NO** — NonCommercial **and** infringing |

1. **NonCommercial contamination.** AnimeVox is CC BY-NC 4.0. 13.3% of the corpus is NC. CC0 is a total waiver permitting unlimited commercial use. Irreconcilable — and the VoicePersona README markets "✅ Commercial use" to "Game developers", i.e. it actively invites downstream violation. **[HIGH]**
2. **Attribution/notice conditions.** AniSpeech is MIT and LAION is Apache-2.0; both require notice retention. CC0's whole proposition is that *no* conditions attach. Even ignoring anime rights, CC0 would be unavailable. **[HIGH]**
3. **Anime audio with no rights basis — the real exposure.** **26.5% of the corpus (3,999 samples) is anime-derived and the audio itself is redistributed.** AnimeVox states outright it was ripped from official commercial dubs; AniSpeech asserts a bare "Copyright 2023 ShoukanLabs" over voice performances it did not commission. Neither can convey rights it never held. Beyond copyright this implicates performers' rights and, in several US states, right-of-publicity over vocal likeness — and a voice-cloning dataset is precisely the use that makes those claims live. **[HIGH on the documentary facts; the legal conclusion needs counsel]**
4. **The metadata does not even say CC0.** `license: cc` is HF's generic bucket; it grants nothing. The artefact that ships the audio carries no operative licence. **[HIGH]**

**Consequence for VoiceForge.** The scope doc (§15.3) treats VoicePersona's CC0 as "maximally permissive and a genuine asset here". That is exactly backwards: **VoicePersona is currently the *most* legally exposed input in the plan**, because unlike ParaSpeechCaps (honestly labelled NC-SA, ships no audio) it *ships infringing audio under a public-domain declaration*. This needs to be fixed before anything is trained on it, not after.

**Remediation, in order:**
1. **Immediately** change the HF YAML off `cc`, drop the CC0 badge and the "✅ Commercial use" line. The current state misleads downstream users into infringement.
2. **Drop AnimeVox and AniSpeech entirely** (3,999 samples). This resolves grounds 1 and 3 at a stroke. Remaining 11,083 samples = LAION (Apache-2.0) + GLOBE_V2 (CC0) → a defensible **Apache-2.0 release with a NOTICE file**. Cost: you lose the anime/character coverage the scope doc calls "real, non-replicable coverage" — see §6.3 for how to replace it legitimately.
3. If anime data is retained, **CC BY-NC 4.0 is the floor**, with a prominent unverified-rights notice. Mitigation, not cure.
4. **Strongly consider the annotations-only release**: ship `voice_description` / `gender` / `age` / `accent` keyed to upstream sample IDs, **without audio**. This is exactly what ParaSpeechCaps does and it is why ParaSpeechCaps could be released at all. The captions are your own work product; the copyright problem lives almost entirely in the redistributed audio.
5. Resolve the LAION→OpenAI question with counsel. 52.6% of the corpus is gpt-4o-audio output obtained via a third-party reseller, and the downstream purpose is training a competing speech-synthesis model. That is the exact fact pattern provider terms target. **[UNVERIFIED — needs legal input, not more research]**

### 6.2 Is audio-LM captioning as bad as claimed?

**Short answer: yes for quantitative attributes, and the VoicePersona pipeline makes it worse than the model's own ceiling.**

**Evidence 1 — direct benchmark numbers.** StepEval-Audio-Paralinguistic (Step-Audio 2 tech report, [arXiv 2507.16632](https://arxiv.org/abs/2507.16632) Table 2) evaluates open-set paralinguistic QA over 11 dimensions, 550 clips. Accuracy (%):

| Model | Avg | Gender | Age | Timbre | **Pitch** | Rhythm | **Speed** | Emotion |
|---|---|---|---|---|---|---|---|---|
| GPT-4o Audio | 43.45 | 18 | 42 | 34 | **40** | 60 | **58** | 82 |
| Kimi-Audio | 49.64 | 94 | 50 | 10 | **56** | 40 | **44** | 66 |
| **Qwen-Omni** | 44.18 | 40 | 50 | 16 | **32** | 54 | **50** | 76 |
| Step-Audio-AQAA | 36.91 | 70 | 66 | 18 | **38** | 48 | **54** | 40 |
| Step-Audio 2 | 76.55 | 98 | 92 | 78 | **78** | 70 | **78** | 72 |

Qwen-Omni — the **successor** to Qwen2-Audio — scores **32% on pitch** and **50% on speed**. Emotion is the one dimension audio-LMs do well (76–82%). **[HIGH for the numbers; MEDIUM as evidence about Qwen2-Audio specifically, since (a) Qwen-Omni ≠ Qwen2-Audio-7B-Instruct and is the later, stronger model, and (b) the clips are Chinese podcast audio.]** The direction is unambiguous: **audio-LMs are strong on emotion and category, weak on continuous acoustic magnitudes** — which is precisely the measure-first / caption-second split the brief proposes.

**Evidence 2 — corroborating, on a different axis.** [PitchBench (arXiv 2605.26176)](https://arxiv.org/abs/2605.26176) finds pitch hearing "highly unreliable" across ALMs, varying sharply with sound source, note duration and response format. **[HIGH for the finding; LOW–MEDIUM transfer — it measures *musical* pitch, not voice F0.]**

**Evidence 3 — convergent design choices by everyone building this data.** Nobody who has shipped a large style-captioned corpus lets an audio-LM produce quantitative attributes. Data-Speech: signal processing only. ParaSpeechCaps: signal processing for basic tags, uses Gemini 1.5 Flash **only as a verifier of a proposed emotion label**, never as a describer. VoiceDesigner (2026): PENN for pitch, phoneme counting for rate, AudioBox-Aesthetics for quality, percentile bins, then *templates* with an LLM only for synonym variation. **Three independent teams, same architecture.** **[HIGH]**

**Evidence 4 — VoicePersona's own prompts are the bigger problem.** This is the finding that matters most, and it is not about Qwen2-Audio at all. The LAION-branch prompt in [`src/dataset_processor.py`](https://raw.githubusercontent.com/PranavMishra17/VoicePersona-Dataset/main/src/dataset_processor.py) asks the model to *invent* demographics with a constrained, biased schema:

```
GENDER: [male/female]
AGE:    [teens/twenties/thirties/forties/fifties+]
ACCENT: [specific accent like "General American", "British RP", "Australian" - never "neutral"]
```

- `[male/female]` is a forced binary → the "other: 65" bucket is parse noise, not observation.
- The age ladder tops out at "fifties+" → **the 76.1% "twenties" skew is partly a prompt artefact**, and the corpus structurally cannot represent *aged* voices, which is one of VoiceForge's six named stylization targets.
- `never "neutral"` **forbids the model from declining to guess** → it manufactures accent labels for the 52.6% of the corpus that is synthetic OpenAI TTS with no real accent at all. That is where the 702 "accents" come from.

**So the demographic skew the scope doc plans to "rebalance" is substantially a labelling artefact, not a property of the audio.** Rebalancing by resampling the existing labels would optimise against noise. Fix the measurement first.

Also note `src/prompts.py` — which looks like the prompt file — is **dead code**; nothing imports it. The live prompts are the per-processor overrides. Worth knowing before anyone tries to "just tweak the prompt".

**Where the brief is too harsh:** an audio-LM pass is *not* worthless. On emotion and on categorical/abstract qualities it is the best cheap tool available (76–82% on emotion above), and ParaSpeechCaps' own ablation shows the Gemini acoustic-verification stage is **necessary** — removing it degrades tag quality. Keep it, but demote it from *describer* to *verifier and abstract-tag proposer*, constrained to a closed vocabulary. That is exactly what the brief's step 4 says; the evidence supports it.

### 6.3 Concrete v2 pipeline spec

**Design rule: every attribute in a v2 caption must be re-derivable from the audio by a deterministic measurement, so a generated voice can be scored by re-measuring it.** Anything that fails that test goes in a separate, clearly-marked impressionistic layer.

**Stage 0 — fix the corpus before touching the pipeline.**
- Drop AniSpeech + AnimeVox (§6.1). Keep GLOBE_V2 (CC0) and LAION (Apache-2.0, with NOTICE).
- Replace the lost character/stylized coverage with legitimately-licensed sources: **VCTK** (CC BY 4.0, 110 speakers, 48 kHz, hemi-anechoic) and **LibriTTS-R** (CC BY 4.0, 2,456 speakers) put through the **VoiceDesigner DSP chain** (§7) to synthesize stylized/character variants. This is the same move VoiceDesigner made and it produces *licence-clean* character voices, unlike ripped anime.
- Publish **annotations separately from audio**, ParaSpeechCaps-style, so the caption work is releasable regardless of how the audio question lands.

**Stage 1 — measure (deterministic, no LLM).** All tools MIT/BSD/CC-BY:

| # | Attribute | Tool | Config |
|---|---|---|---|
| 1 | `f0_mean`, `f0_std`, `f0_p05`, `f0_p95` (→ range) | **penn** (MIT) | `hopsize=.01, fmin=30, fmax=1000, center='half-hop', interp_unvoiced_at=.065`, 16 kHz. Aggregate **log-F0**, not linear Hz — Data-Speech's linear-Hz std is a defect. |
| 2 | `snr`, `c50`, `speech_duration` | **Brouhaha** + `ylacombe/brouhaha-best` (MIT) | as Data-Speech |
| 3 | `speaking_rate` | **g2p** (MIT) + **a real phone segmenter** | ⚠️ Do **not** copy `len(ipa_string)/dur`. Count IPA *segments*, or better, use forced alignment (MFA, MIT) for true phones/sec over `speech_duration`. **Recompute all bin edges after this change.** |
| 4 | `stoi`, `si-sdr`, `pesq` | **torchaudio SQUIM** (weights CC-BY-4.0) | drop the 15 s truncation if clips are longer |
| 5 | **`jitter`, `shimmer`, `hnr`** *(VoiceForge extension)* | Praat via **parselmouth** — ⚠️ Praat is **GPLv3**; parselmouth is GPLv3. **This contaminates a released pipeline.** Permissive alternatives: implement jitter/shimmer from a pitch-period track (penn gives you periodicity) and HNR via autocorrelation in `librosa`/`numpy`. Budget ~1 day. | These are the direct correlates of *raspy* and *breathy*. Not in any existing pipeline; this is genuinely VoiceForge's own contribution. |
| 6 | **`spectral_tilt`, `F1–F3`, `VTL`** *(VoiceForge extension)* | tilt = LPC/long-term-average-spectrum slope (librosa, ISC); formants via LPC roots; VTL from formant dispersion | Correlates of *deep/booming* and of body size — the thing "a big burly voice" actually means. |
| 7 | `gender` (and `age`) | 🔴 **NOT `audeering/wav2vec2-large-robust-24-ft-age-gender` — it is CC-BY-NC-SA-4.0.** Use a permissive alternative or train your own small head on a CC-BY corpus (VCTK + Common Voice both carry gender/age metadata). | This is a **hard swap-in requirement** under the project's Apache/MIT constraint, and it silently affects ParaSpeechCaps' basic tags too. |

**Stage 2 — bin, using percentiles, per corpus.**
- Replace `np.histogram(values, bins=N)` with **`np.quantile(values, np.linspace(0,1,N+1))`**. Keep the σ-based outlier rejection first.
- **Pitch: per-speaker mean, then percentile *within gender***, as Data-Speech does. Everything else: per-utterance, pooled.
- Recompute edges **per corpus** — do not inherit ParaSpeechCaps' MLS/LibriTTS-R-derived constants (§2.2). Persist them as a JSON alongside the dataset so captions are reversible.
- **Store the raw continuous value next to the bin label in every row.** This is what makes the dataset an evaluation harness as well as a training set — see 06.
- Vocabulary: start from Data-Speech v02 (§2.2), extend with LibriTTS-P's **44 speaker-identity adjectives** (CC BY 4.0 — the only permissively-licensed rich intrinsic vocabulary that exists) and the ParaSpeechCaps 33 for tag *names* only. LibriTTS-P also gives you `old` / `young` / `middle-aged` / `mature`, filling PSC's missing age axis, plus its `{very, slightly, ∅}` strength qualifier scheme.

**Stage 3 — caption from bins, template-first.**
- **Build a template bank, not a free-running LLM.** Both LibriTTS-P (54 keys × ~20–30 paraphrases in `style_prompt_candidates_v230922.csv`, CC BY 4.0) and VoiceDesigner (*"we design a set of caption templates and employ an LLM to introduce synonym variations"*) do this. It is cheaper, deterministic, and eliminates a whole class of hallucination. Reuse LibriTTS-P's bank directly for the basic layer.
- Use an LLM (ParaSpeechCaps' §E.4 prompt is the right shape) only to **paraphrase and to fuse the rich tags in**, at temperature ~0.6, 256 max tokens.
- Generate **2–3 captions per sample** at different specificity levels (terse / medium / rich). VoiceForge's users will type short prompts; training only on 500-char descriptions teaches the mapper a distribution it will never see at inference.
- Include the speaker **name** variant of the prompt (Data-Speech's `SINGLE_SPEAKER_PROMPT`) if persistent named identities are a product goal.

**Stage 4 — impressionistic layer, verified not generated.**
- Closed vocabulary = ParaSpeechCaps' 33 intrinsic ∪ LibriTTS-P's 44, plus VoiceForge's own missing targets (`aged`, `menacing`, `theatrical`, `breathy`) with **written definitions** — PSC's Table 5 definitions verbatim where they overlap.
- Use the audio-LM as a **rater**, not a describer: for each candidate tag, the PSC Acoustic-Matching prompt shape (§2.3) → keep only 5/5. Never let it emit free text into a caption.
- **Only accept a tag if it is consistent with the measurements**: `raspy` should co-occur with high jitter/shimmer and low HNR; `deep` with low F0 *and* low spectral tilt. Log every disagreement — that log is your annotation-quality metric.

**Stage 5 — rebalance, on measurements not on labels.**
- Do **not** rebalance the current 62.6%/76.1% label distribution (§6.2 — it is a prompt artefact). Rebalance after re-measuring gender and re-deriving age from acoustics + true metadata.
- Balance on **speaker count per stratum**, not hours (§5). Target strata: gender × pitch-tercile × age-band × rate-tercile.
- Steal ParaSpeechCaps' **targeted recruitment** trick for rare qualities: identify the 10–15 rarest bins, then go find speakers for them specifically rather than hoping uniform sampling covers them.

**Stage 6 — publish the reversibility check.** For a held-out slice, re-measure generated audio and report per-attribute agreement with the caption's bins. This is the single number that justifies "measure first, caption second" over the v1 approach, and no published dataset reports it. It is cheap, and it is a genuine contribution.

---

## 7. C4 — non-human voice data synthesis (deferred, brief)

**arXiv 2608.13613 resolves.** *VoiceDesigner: Text-to-Voice Generation and Editing via Unified Diffusion Modeling and Data Augmentation* — Hai, Thakkar, Chen, Y. Wang, Su, Kumar, Elhilali, Jin (JHU + Adobe Research), 2026-08-12. **[HIGH]** See [01-ttv-landscape.md](01-ttv-landscape.md) for the architecture; this section covers only the data pipeline. No code, no weights.

**Two augmentation pipelines (paper §III-C):**

**(a) DSP simulation** — a chained effects rack over ordinary recordings:
> *"a standard speech recording is transformed through a sequence of audio effects, including **pitch shifting, formant shifting, reverberation, equalization, band-pass filtering, and dynamic range compression**. These transformations are implemented using DSP libraries such as **Pedalboard**, **librosa**, and **Parselmouth**. In addition, we incorporate **SiFi-GAN** to perform pitch contour manipulation, such as flattening intonation toward monotonic speech or adjusting pitch variation ranges."*

Applied to a **bandwidth-extended LibriTTS-R and VCTK** to synthesize non-human character variants. Note the licence composition: Pedalboard (GPLv3, Spotify), librosa (ISC), Parselmouth/Praat (GPLv3), SiFi-GAN (check). **A VoiceForge re-implementation should avoid Pedalboard and Parselmouth** — the effects are all reimplementable permissively (`torchaudio.functional`, `pedalboard`→`pyroomacoustics`/`scipy` for reverb+EQ, `librosa`/`pyworld` for pitch/formant shift). **[HIGH on the recipe; the licence read is mine]**

**(b) Generative simulation** — zero-shot voice cloning + pitch-controlled voice conversion, used to expand a small, stylistically-rich seed set: cloning gives new *content* at fixed timbre/style, VC gives new *timbre* at fixed content/prosody contours. Applied to ESD/RAVDESS/SAVEE (E1), Expresso/EARS/CapSpeech-Agent (E2), VCTK + Common Voice (accents), DreamVoice (timbre labels), and an **internal 16-hour set of 20 character identities performed by 12 professional voice actors → 360 distinct voice variations**. Captions written by **Qwen3-30B-A3B-Instruct**. Ablation (Fig. 5) shows generative augmentation improves both Style-ACC and WER over real-data-only. **[HIGH]**

**Also from the same paper, and directly relevant to §2.2 and §6.3:** VoiceDesigner uses **PENN** for speaker-level mean pitch and utterance-level pitch std (silence/unvoiced excluded, **log-scale before quantization**), phonemes ÷ silence-removed duration for rate, **AudioBox-Aesthetics** for quality, and **percentile-based bins** computed over Emilia + HiFiTTS-2-44.1k. Pretraining corpus: Emilia + Common Voice + LibriTTS-R + HiFiTTS-2-44.1k subset = **56,165 hours**. **[HIGH]**

**Takeaways for VoiceForge's deferred non-human track:**
1. The character/non-human data problem is solved by **augmentation of clean licensed speech**, not by scraping character audio. This is a direct, evidence-backed alternative to the AniSpeech/AnimeVox exposure in §6.1.
2. The DSP rack is ~200 lines and runs on CPU. It is the cheapest possible expansion of stylization coverage and should be built **during** the dataset phase, not deferred to a separate workstream.
3. **A 16-hour, 20-character professional recording produced 360 voice variations.** If VoiceForge ever commissions original audio, that is the published ratio to plan against.
4. Even a 1.0B model with 56k hours of pretraining needed augmentation to cover fictional voices. Do not expect coverage to emerge from scale.

---

## 8. What this means for the build

### 8.1 Download order and disk footprint

Sizes are from the HF datasets-server `size` API or from `content-length` on the official artefact — measured, not estimated.

| # | Get | Why first | Size | Where |
|---|---|---|---|---|
| 1 | **LibriTTS-P annotations** | 3 CSVs, ~1 MB. Gives you 2,443 speakers × 3 annotators of human intrinsic tags **today**, CC BY 4.0. Nothing else in the world does this permissively. | **~82 MB** (81.7 MB is `metadata_w_style_prompt_tags_v230922.csv`) | `github.com/line/LibriTTS-P/data/` |
| 2 | **ParaSpeechCaps annotations** | Read-only reference: the 59-tag ontology, the tag→caption mapping, and 1.07M worked examples of good captions. **Do not train on it.** | **368 MB** (1,068,136 rows, no audio) | `ajd12342/paraspeechcaps` |
| 3 | **VCTK 0.92** | Smallest high-quality multi-speaker corpus. Enough to build and debug the whole measure→bin→caption pipeline end to end in a day. Also the DSP-augmentation base. | **11.7 GB** (`VCTK-Corpus-0.92.zip`) | datashare.ed.ac.uk handle 10283/3443 |
| 4 | **LibriTTS-R** | The audio under #1. Turns LibriTTS-P into real (description, audio) pairs. | **~80 GB** raw; the HF `parler-tts/libritts_r_filtered` parquet mirror is **98.6 GB** | openslr.org/141 |
| 5 | **GLOBE** (full, not just VoicePersona's 3,146-sample slice) | **535 h / 23,519 speakers / 164 accents / CC0**, ~82 s per speaker — the closest public match to the §5 regime, and each clip is an independent recording session (the variable §5.2 says actually matters). | `MushanW/GLOBE_V2` parquet: **76 GB** | HF `MushanW/GLOBE_V2` |
| 6 | **RASMALAI** | **CC BY 4.0, 13,000 h, 24M style descriptions, 24 languages including English.** The largest permissive style-captioned corpus in existence, and the brief has it filed as Indic-only. | large — take the **1,804 h finetune subset** first | AI4Bharat, see [04-indic-track.md](04-indic-track.md) |
| 7 | **Common Voice SS 26.0 (en first)** | Extends the speaker-count play beyond GLOBE. CC0, ~100k English contributor accounts. Subsample aggressively — you want *speakers*, not hours. | en full is large; **plan to keep ≤2 clips/speaker** → a few hundred GB down to tens | Mozilla Data Collective (**not** HF — v18–26 are 404 there) |
| 8 | **MLS / People's Speech** | Only if the above proves insufficient. | MLS en ~44.5k h; `parler-tts/mls_eng_10k` parquet mirror is **158 GB** | dl.fbaipublicfiles.com/mls/ ; `MLCommons/peoples_speech` |
| — | VoicePersona v1 | Do **not** re-download. 5.0 GB, 14,327 rows, and §6.1 must be resolved first. | 5.0 GB | — |

**Practical footprint for phase 1 (items 1–4): ~93 GB.** That fits a single consumer drive. Do not pull MLS or Emilia yet.

**Streaming note:** every corpus above except VCTK and LibriTTS-R is available as HF parquet and can be `load_dataset(..., streaming=True)`. For the *measurement* pass you never need the whole corpus on disk — you need the feature vectors, which are ~200 bytes/utterance. **Measure while streaming, persist features, discard audio.** That converts a 158 GB problem into a 300 MB one for everything except the final embedding-extraction pass.

### 8.2 The exact feature-extraction stack to install

Licences checked; every item is permissive except where flagged.

```bash
# core (all MIT/BSD/Apache)
pip install "datasets[audio]" transformers accelerate torch torchaudio
pip install penn                                                    # MIT — pitch (FCNF0++)
pip install g2p                                                     # MIT — IPA transduction
pip install https://github.com/marianne-m/brouhaha-vad/archive/main.zip   # MIT — SNR + C50 + VAD
pip install librosa soundfile numpy scipy                           # ISC/BSD — tilt, formants, HNR
# optional
pip install bitsandbytes                                            # 4-bit Mistral for captioning
```

Model weights to cache:

| Weight | Purpose | Licence |
|---|---|---|
| `ylacombe/brouhaha-best` | SNR / C50 | **MIT** ✅ |
| penn default checkpoint (FCNF0++) | F0 | **MIT** ✅ |
| `torchaudio.pipelines.SQUIM_OBJECTIVE` | STOI / SI-SDR / PESQ | **CC BY 4.0** ✅ |
| `mistralai/Mistral-7B-Instruct-v0.2` | caption fluency | Apache-2.0 ✅ (fits 8–12 GB in 4-bit) |
| ~~`audeering/wav2vec2-large-robust-24-ft-age-gender`~~ | gender/age | 🔴 **CC-BY-NC-SA-4.0 — MUST BE REPLACED** |
| ~~VoxSim `wavlm_ecapa.model`~~ | perceptual similarity | 🔴 **no licence at all** (`kaistmm/voxsim_trainer` has no LICENSE file) |
| ~~Praat / parselmouth~~ | jitter/shimmer/HNR/formants | 🔴 **GPLv3 — contaminates a released pipeline** |

**Three hard swap-ins, all of which the scope doc does not currently budget for.** The gender classifier is the urgent one: it is a dependency of *both* Data-Speech's binning (pitch bins are gender-relative) and ParaSpeechCaps' basic tags, and its NC licence silently attaches to any pipeline that copies the recipe as-is. Options: derive gender from corpus metadata where it exists (VCTK, Common Voice, LibriTTS-R all have it), and train a small permissive classifier on that metadata for corpora that lack it. That is a half-day job and removes the blocker permanently.

### 8.3 Compute shape

- **Measurement is embarrassingly parallel and mostly CPU-bound.** penn and Brouhaha want GPU; g2p, tilt, formants and HNR are CPU. Data-Speech's `main.py` shards with `num_proc = n_gpus * workers_per_gpu`, `with_rank=True`, and uses the `remove_columns=[audio_column_name]` trick to avoid rewriting audio into the cache — copy that pattern, it is the difference between a 200 GB and a 1 TB cache.
- **Captioning is the GPU-hour cost.** Mistral-7B in 4-bit at batch 32–64 on one 8–12 GB card. Budget: at ~1.5 s/caption sequential for a 12 GB card and 2–3 captions per sample, 15k samples ≈ **20 GPU-hours**; 400k samples ≈ 500 GPU-hours. **This is why the template bank matters** — a template + synonym-substitution pass costs ~0 and covers the basic layer entirely, leaving the LLM for only the rich-tag fusion.
- **Everything in §2 fits the locked one-8–12 GB-GPU constraint.** Nothing in the annotation pipeline needs more. The only step that would not fit is VoxSim intrinsic-tag propagation over 45k hours, and that step is licence-blocked anyway.

### 8.4 Sequencing correction

The scope doc puts the dataset rebuild at **S4**, after the retrieval mapper (S2) and generative mapper (S3). Two reasons to move measurement earlier:

1. **S2 needs `(description, audio)` pairs and the only permissively-licensed source is LibriTTS-P's 2,443 speakers.** If ParaSpeechCaps is off the table (§3.1), S2's training set is *much* smaller than planned unless the annotation pipeline exists first. The pipeline is the unblocker, not a later polish step.
2. **The measured features are also the evaluation harness.** Re-measuring generated audio against caption bins is the reversibility check (§6.3, Stage 6) and it is what [06-evaluation-harness.md](06-evaluation-harness.md) needs. Building it at S4 means S2 and S3 are trained without an objective adherence metric.

**Recommendation: split S4.** Move *Stage 1–2 (measure + bin)* to run alongside S0/S1 — it is CPU work, needs no trained model, and unblocks both the training set and the metric. Leave *Stage 3–6 (caption, verify, rebalance, publish)* at S4.

---

## 9. Open — must be settled by experiment

| Question | Cheapest experiment | Est. cost/time | What it blocks |
|---|---|---|---|
| **Does speaker count actually dominate hours for the mapper?** (§5 — the project's central data belief, UNVERIFIED in the literature) | The §5.4 iso-hours ladder: fix total hours, sweep speaker count 100/300/1k/3k/10k over Common Voice EN (CC0) + LibriTTS-R. Metrics: retrieval R@10, MMD to the real z-distribution, gen2gen-near. | ~30 GPU-hours after a one-time embedding pass; **≈3 days wall-clock** on one 12 GB card | The entire corpus-selection strategy, the VoicePersona v2 rebalancing target, and how much of Common Voice to download. **Run this first.** |
| **Do equal-width bins actually hurt vs percentile bins?** (§2.2 — Data-Speech does one, the brief and VoiceDesigner assume the other) | Caption the same 2,443 LibriTTS-P speakers twice, train the same mapper on each, compare per-bin adherence — especially recall of the extreme bins ("very low-pitch"). | ~8 GPU-hours; **1 day** | Whether to reuse Data-Speech's published edges or recompute. Cheap, and the answer is probably "percentiles win at the tails", which is where character voices live. |
| **Is `len(ipa_string)/dur` good enough, or does true phones/sec matter?** (§2.1) | Compute both over LibriTTS-R, correlate. If Spearman ρ > 0.95 the shortcut is fine and you inherit Data-Speech's edges for free; if not, recompute everything. | **2 hours**, CPU only | Whether §2.2's published bin edges are reusable at all. Do this before any captioning. |
| **Does the reversibility claim hold — can a generated voice be verified by re-measuring it?** (§6.3 Stage 6 — the justification for the whole measure-first thesis) | For 200 held-out captions, synthesize with the frozen TTS, re-run Stage 1, report per-attribute bin agreement and Spearman ρ on the continuous values. | ~4 GPU-hours; **1 day** | The evaluation harness (06), and the honest claim in any writeup. **No published dataset reports this number.** Genuine contribution if it works. |
| **Do jitter / shimmer / HNR / spectral tilt actually predict the tags we care about?** (§2.1 — VoiceForge's own extension, unvalidated by anyone) | Measure all four over LibriTTS-P's 2,443 speakers; test separation of `raspy` / `thick` / `muffled` / `nasal` speakers vs the rest (AUC per tag). | **1 day**, CPU only | Whether the extra features earn their implementation cost, or whether F0+rate+SNR already carry the signal. Cheap and decisive. |
| **How much does audio-LM abstract tagging actually add, once measurement is in place?** (§6.2) | Ablate: captions from measurements alone vs measurements + verified audio-LM tags. Compare adherence on *abstract* prompts only. Mirrors ParaSpeechCaps' own Table 2 ablation design. | ~10 GPU-hours; **2 days** | Whether to keep any audio-LM in the v2 pipeline, and how much of the 500-GPU-hour captioning budget to spend. |
| **Is a permissive gender classifier accurate enough off-domain?** (§8.2 — hard swap-in) | Score `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech` (Apache-2.0) against VCTK and Common Voice metadata. Its published F1 of 0.9993 is in-domain on LibriSpeech-clean. | **3 hours** | Every pitch bin, since pitch bins are gender-relative. Blocks Stage 1. |
| **Does the DSP stylization chain produce voices the speaker encoder treats as distinct identities?** (§7) | Apply the VoiceDesigner rack to 50 VCTK speakers; check whether `E(audio)` for the stylized variants lands *off* the original speaker and *inside* the manifold rather than outside it. | **1 day** | Whether DSP augmentation is a legitimate replacement for the dropped anime data, or produces out-of-distribution embeddings the mapper cannot reach. |
| **Do the CC-BY-SA subsets of People's Speech contaminate released weights?** | Legal question, not experimental. Use the shipped CSV to exclude the 3,100 SA hours and sidestep it entirely. | 0 | Nothing, if you just exclude them. |
| **Can LibriTTS-P's CC-BY-4.0 claim be relied on with no LICENSE file?** (§3) | Email the LINE authors and ask for a LICENSE file in the repo. | 0 | The one corpus the whole plan now rests on. **Do this on day one** — it is a one-line email with a large downside if ignored. |

---

## 10. Sources

| # | URL | Type | Used for | Confidence in source |
|---|---|---|---|---|
| 1 | https://github.com/huggingface/dataspeech | repo (code read in full) | §2.1–2.3 attribute list, binning code, prompts | HIGH |
| 2 | https://raw.githubusercontent.com/huggingface/dataspeech/main/LICENSE | LICENSE file | Data-Speech = MIT | HIGH |
| 3 | https://raw.githubusercontent.com/huggingface/dataspeech/main/dataspeech/cpu_enrichments/rate.py | source file | `speaking_rate = len(ipa_str)/dur` correction | HIGH |
| 4 | https://raw.githubusercontent.com/huggingface/dataspeech/main/dataspeech/gpu_enrichments/pitch.py | source file | penn config | HIGH |
| 5 | https://raw.githubusercontent.com/huggingface/dataspeech/main/dataspeech/gpu_enrichments/snr_and_reverb.py | source file | Brouhaha SNR/C50/VAD | HIGH |
| 6 | https://raw.githubusercontent.com/huggingface/dataspeech/main/dataspeech/gpu_enrichments/squim.py | source file | SQUIM SI-SDR/PESQ/STOI, 15 s truncation | HIGH |
| 7 | https://raw.githubusercontent.com/huggingface/dataspeech/main/scripts/metadata_to_text.py | source file | equal-width binning correction; per-gender pitch | HIGH |
| 8 | https://raw.githubusercontent.com/huggingface/dataspeech/main/scripts/run_prompt_creation.py | source file | all four verbatim caption prompts | HIGH |
| 9 | https://raw.githubusercontent.com/huggingface/dataspeech/main/examples/tags_to_annotations/v02_bin_edges.json | data file | copy-pasteable bin edges | HIGH |
| 10 | https://raw.githubusercontent.com/huggingface/dataspeech/main/examples/tags_to_annotations/v02_text_bins.json | data file | copy-pasteable bin vocabulary | HIGH |
| 11 | https://raw.githubusercontent.com/huggingface/dataspeech/main/examples/prompt_creation/run_prompt_creation_45k.sh | shell script | Mistral-7B-Instruct-v0.2 as the captioner | HIGH |
| 12 | https://arxiv.org/abs/2503.04713 · https://arxiv.org/html/2503.04713v2 | paper (v1 + v2) | 59-tag taxonomy, thresholds, prompts, protocol, MOS table; the 342-vs-282 discrepancy | HIGH |
| 13 | https://huggingface.co/datasets/ajd12342/paraspeechcaps/raw/main/README.md | HF card raw YAML | `license: cc-by-nc-sa-4.0`; per-split hours/speakers | HIGH |
| 14 | https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/README.md | repo doc | audio not shipped; VoxCeleb/Expresso/EARS/Emilia sourcing | HIGH |
| 15 | https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/automatic_annotation/basic_tags/README.md | repo doc | vendored Data-Speech; gender model identity | HIGH |
| 16 | https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/automatic_annotation/basic_tags/bin_edges.json + text_bins.json | data files | PSC 3-bin thresholds | HIGH |
| 17 | https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/automatic_annotation/intrinsic/README.md | repo doc | VoxSim propagation, 18 propagated tags, 0.8 threshold | HIGH |
| 18 | https://raw.githubusercontent.com/ajd12342/paraspeechcaps/main/dataset/pscbase_name_to_intrinsictags.json | data file (551 distinct strings counted) | free-text annotation protocol evidence | HIGH |
| 19 | https://github.com/line/LibriTTS-P + `data/df{1,2,3}_en.csv` + `style_prompt_candidates_v230922.csv` | repo + data (tag vocabulary extracted) | 2,443 speakers, 44 adjectives, template bank, CC BY 4.0 | HIGH (data) / MEDIUM (licence — README only, no LICENSE file) |
| 20 | https://arxiv.org/abs/2406.07969 | paper | LibriTTS-P 373,868 prompts | HIGH |
| 21 | https://www.openslr.org/141/ + `doc.tar.gz`→`LICENSE.txt` | official resource + LICENSE | LibriTTS-R CC BY 4.0 | HIGH |
| 22 | https://www.openslr.org/94/ · https://arxiv.org/html/2012.03411v2 | official resource + paper | MLS hours/speakers/CC BY 4.0 | HIGH |
| 23 | https://huggingface.co/datasets/amphion/Emilia-Dataset (card + `extra_gated_prompt`) | HF card + gate text | Emilia NC vs YODAS BY split; YAML/body contradiction | HIGH |
| 24 | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html | official terms page | VoxCeleb2 metadata-only CC BY-SA; **downloads withdrawn** | HIGH |
| 25 | https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/cv-corpus-26.0-2026-06-12.json | official release JSON | Common Voice SS 26.0 stats, CC0 | HIGH |
| 26 | https://docs.google.com/forms/d/e/1FAIpQLSfmLDh0gxzST7n-os72-ZmgyRPot67RFvhwJNIngBYILYveIQ/viewform | official Terms of Access | GigaSpeech non-commercial clause 1 + employer-binding clause 6 | HIGH |
| 27 | https://huggingface.co/datasets/MLCommons/peoples_speech/raw/main/README.md · https://arxiv.org/abs/2111.09344 | HF card + paper | People's Speech CC-BY/CC-BY-SA split, commercial intent | HIGH |
| 28 | https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/LICENSE | LICENSE file | EARS CC BY-NC 4.0 | HIGH |
| 29 | https://raw.githubusercontent.com/facebookresearch/textlesslib/main/examples/expresso/dataset/README.md | official README | Expresso CC BY-NC 4.0, 26 styles, 4 speakers | HIGH |
| 30 | https://datashare.ed.ac.uk/handle/10283/3443 (DSpace REST `dc.rights` + corpus `README.txt`) | repository metadata + LICENSE | VCTK CC BY 4.0, 48 kHz/16-bit | HIGH |
| 31 | https://raw.githubusercontent.com/jishengpeng/TextrolSpeech/main/LICENSE · https://arxiv.org/html/2308.14430v1 | LICENSE + paper | TextrolSpeech MIT-code-only; dataset licence absent | HIGH (absence) |
| 32 | https://raw.githubusercontent.com/thuhcsi/SpeechCraft/master/README.md | repo README (`master` branch) | SpeechCraft has no licence; emphasis EULA is non-commercial | HIGH (absence) |
| 33 | https://github.com/keonlee9420/DailyTalk · https://arxiv.org/abs/2207.01063 | repo + paper | DailyTalk CC BY-SA badge vs "academic use" prose | HIGH |
| 34 | https://huggingface.co/datasets/Paranoiid/VoicePersona/raw/main/README.md | HF card raw YAML | `license: cc`; `language: en`; schema; 5.0 GB audio | HIGH |
| 35 | https://raw.githubusercontent.com/PranavMishra17/VoicePersona-Dataset/main/src/dataset_processor.py | source file | the live Qwen2-Audio prompts; forced-binary gender, capped age, `never "neutral"` | HIGH |
| 36 | https://raw.githubusercontent.com/PranavMishra17/VoicePersona-Dataset/main/analyse/statistics.json | data file | 15,082 / 10,179 / 48.7 h / 702 accents / `techarties` | HIGH |
| 37 | https://datasets-server.huggingface.co/size?dataset=Paranoiid%2FVoicePersona | HF API | **14,327 rows actually shipped**, 5.0 GB | HIGH |
| 38 | https://huggingface.co/datasets/laion/laions_got_talent (card + LICENSE file) | HF card + LICENSE | Apache-2.0 undeclared in YAML; gpt-4o-audio via Hyprlab provenance | HIGH |
| 39 | https://huggingface.co/datasets/MushanW/GLOBE_V2/raw/main/README.md | HF card raw YAML | `license: cc0-1.0`, Common Voice 14 upstream | HIGH |
| 40 | https://huggingface.co/datasets/ShoukanLabs/AniSpeech (card + `license` file) | HF card + licence file | `license: mit`, no provenance stated | HIGH |
| 41 | https://huggingface.co/datasets/taresh18/AnimeVox/raw/main/README.md | HF card raw | YAML `cc` vs body **CC BY-NC 4.0**; "official English-dubbed versions of popular anime series" | HIGH |
| 42 | https://arxiv.org/abs/2608.13613 · https://arxiv.org/html/2608.13613v1 | paper | C4 DSP + generative augmentation chain; percentile bins; 56,165 h pretraining | HIGH |
| 43 | https://arxiv.org/abs/2507.16632 (Step-Audio 2, Table 2) | paper | StepEval-Audio-Paralinguistic: audio-LM pitch/speed accuracy | HIGH |
| 44 | https://arxiv.org/abs/2605.26176 | paper | PitchBench — ALM pitch unreliability (musical pitch; transfer is indirect) | HIGH (paper) / LOW–MED (transfer) |
| 45 | https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender/raw/main/README.md | HF card raw YAML | `license: cc-by-nc-sa-4.0` — the hidden NC dependency | HIGH |
| 46 | https://raw.githubusercontent.com/interactiveaudiolab/penn/master/LICENSE · https://raw.githubusercontent.com/marianne-m/brouhaha-vad/main/LICENSE · https://raw.githubusercontent.com/roedoejet/g2p/main/LICENSE | LICENSE files | penn/brouhaha/g2p all MIT | HIGH |
| 47 | https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.SQUIM_OBJECTIVE.html | official docs | SQUIM weights CC BY 4.0 | HIGH |
| 48 | https://github.com/kaistmm/voxsim_trainer (tree listed; LICENSE 404) | repo | VoxSim has **no licence** | HIGH |
| 49 | https://raw.githubusercontent.com/tiantiaf0627/vox-profile-release/main/LICENSE | LICENSE file | Vox-Profile is RAIL, not Apache/MIT | HIGH |
| 50 | https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech/raw/main/README.md | HF card raw | Apache-2.0 gender classifier, F1 0.9993 in-domain | HIGH |
| 51 | https://arxiv.org/abs/2406.06926 | paper | VoxSim perceptual similarity (cited by PSC) | HIGH |
| 52 | https://arxiv.org/abs/2406.06185 | paper | EARS 100 h / 107 speakers / 48 kHz | HIGH |
| 53 | https://www.isca-archive.org/interspeech_2022/vaessen22_interspeech.pdf | ISCA proceedings (peer-reviewed) | **§5.2 the decisive fixed-budget speaker-count ablation + the session-diversity confound** | HIGH |
| 54 | https://arxiv.org/abs/2512.17356 | paper | fixed-utterance speaker sweep: WER/UTMOS knee ~50 spk, **SIM knee ~500 spk** | HIGH |
| 55 | https://arxiv.org/abs/2309.14838 | paper (ICASSP 2024) | fixed-100k-utterance speaker sweep, monotone improvement (values plot-only → UNVERIFIED) | HIGH direction / LOW magnitude |
| 56 | https://arxiv.org/abs/2204.06450 | paper | speaker-pool expansion EER curve — **explicitly NOT fixed-budget; do not cite as such** | HIGH numbers / framing caveat |
| 57 | https://arxiv.org/abs/2310.05001 | paper | PromptSpeaker data: 792 spk / 21,760 utts; no speaker-count ablation | HIGH |
| 58 | https://arxiv.org/abs/2111.05095 | paper | TacoSpawn data (enus1100 = 1,100 spk × 30 min); the gen2gen/train2train/gen2train metric battery | HIGH |
| 59 | https://arxiv.org/abs/2309.14094 | paper | VoiceLens — closest architecture; distinct-voice count ω(G) metric | HIGH |
| 60 | https://arxiv.org/abs/2402.01912 | paper | Parler-TTS (Lyth & King) — **no ablation section at all**; MLS 45k h | HIGH |
| 61 | https://arxiv.org/abs/2505.18609 | paper | **RASMALAI: CC BY 4.0, 13,000 h, 24M descriptions, 24 langs** + GLOBE stats | HIGH |
| 62 | https://arxiv.org/abs/2506.02863 | paper | CapSpeech — large, most-cited 2025 style corpus, **CC BY-NC 4.0** | HIGH |
| 63 | https://arxiv.org/abs/2604.08363 | paper | CapTalk §5.2.1 — 300 h acted beats 1,500 h conversational | HIGH |
| 64 | https://arxiv.org/abs/2603.28086 | paper | MOSS-VoiceGenerator — +10,000 h of low-diversity data bought nothing | HIGH |
| 65 | https://arxiv.org/abs/2407.11510 · https://arxiv.org/abs/2507.13155 · https://arxiv.org/abs/2504.12867 · https://arxiv.org/abs/2509.15626 | papers | VoxBlink2 (111,284 spk) · NonverbalTTS · EmoVoice-DB · LibriTTS-VI | HIGH |
| 66 | https://arxiv.org/abs/2203.07378 | paper | audEERING dominance-valence-arousal model used in PSC expressivity filtering | HIGH |
| — | "SpeakerVerse", "VccmDataset", Meta Audiobox caption data | **searched, not found** | recorded as **UNVERIFIED / do not cite** | HIGH (absence) |
