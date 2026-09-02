# 06 — The Evaluation Harness (build this FIRST)

> **Domain:** measurement — adherence, identity consistency, diversity, intelligibility
> **Answers:** scope section 10; supports A1/A3 verification
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [01-ttv-landscape.md](01-ttv-landscape.md) · [02-identity-representation.md](02-identity-representation.md) · [05-datasets-and-annotation.md](05-datasets-and-annotation.md)

---

## 0. Bottom line

- **InstructTTSEval is real, the code is public, and the judge prompt is retrievable verbatim** — but it is **NOT locally runnable**. The judge is `gemini-2.5-pro-preview-05-06` called through the Google GenAI API with the audio file uploaded per item. There is no open-weight judge in the repo, no local fallback, and no leaderboard. Adopt its *protocol*; do not expect to adopt its *judge* under the 12GB / permissive-licence constraint. **[HIGH]** ([repo](https://github.com/KexinHUANG19/InstructTTSEval/blob/main/eval/gemini_eval.py), source: repo code)
- **The brief's "12 attributes" claim is exactly right** — verified against the actual judge prompt file, which enumerates 性别/音高/语速/音量/年龄/清晰度/流畅度/口音/音色质感/情绪/语调/性格 = gender, pitch, speed, volume, age, clarity, fluency, accent, texture, emotion, tone, personality. The brief's list is correct and complete. **[HIGH]** ([eval_prompt.txt](https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/eval_prompt.txt), source: repo file)
- **The single biggest measurement trap: MOS predictors are strongly, wrongly anti-correlated with pitch.** Takagi et al. (arXiv 2606.19951, June 2026) perturbed F0 and measured six predictors against humans: humans showed **r = −0.059** with mean F0 (i.e. no relationship); **DNSMOS showed r = −0.788 and UTMOSv2 r = −0.722**. For a project whose entire point is spanning the pitch range (children, aged men, whispered, raspy), a UTMOS/DNSMOS quality gate will **systematically select against high-pitched voices for a reason humans do not share** — the naturalness metric actively fights the diversity axis. **[HIGH]** ([arXiv](https://arxiv.org/html/2606.19951v1), source: paper)
- Same paper, second trap: **all six MOS predictors are blind to prosodic error.** Pitch-accent corruption dropped human MOS by **1.84 points** (4.00 → 2.16) while every model moved **< 0.1**. MOS predictors measure *signal cleanliness*, not *delivery*. Do not use them as a proxy for "does this sound like a good performance." **[HIGH]**
- **Objective acoustic re-measurement is the right primary adherence gate**, and it is also the statistically powerful one. It is continuous, deterministic, free, and CPU-cheap. The LLM-judge route produces a *binary* verdict; at 50 descriptions that resolves only ~10 percentage-point differences. Make re-measurement the gate and the judge the tiebreaker. **[HIGH — derived from standard power analysis, see §7]**
- **Encoder independence is achievable and cheap.** WeSpeaker ResNet34-LM (Apache-2.0 code, CC-BY-4.0 weights, 256-dim, 0.723% EER on VoxCeleb1-O) is architecturally and data-independent of any WavLM- or ECAPA-derived conditioning encoder. Use it for scoring; never score with the family you conditioned on. **[HIGH]** ([WeSpeaker recipe](https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/examples/voxceleb/v2/README.md), source: repo)
- **Absolute cosine thresholds are not portable across encoders and the literature does not supply a universal one.** Published same-speaker real-speech SIM-o on WavLM-large is **0.69–0.76**; mean pairwise cosine between *genuinely different* real speakers on wavlm-base-plus-sv is **0.67 ± 0.18**. Those numbers are not comparable. Every threshold in this harness must be **calibrated in-run against real speakers on the same encoder**. This is the correction that saves you from a fake pass/fail line. **[HIGH]**
- **A ready-made, MIT-licensed, actively-maintained distributional metric already exists and you should install it**: `pip install ttsds` (v2.1.3, pushed 2026-07-07). TTSDS2 was the **only metric of 16 compared to exceed Spearman 0.50 with human judgement in every domain** (average 0.67). Its 2-Wasserstein-between-feature-distributions machinery is also exactly the machinery the diversity axis needs. **[HIGH]** ([arXiv 2506.19441](https://arxiv.org/html/2506.19441v1) · [repo](https://github.com/ttsds/ttsds), source: paper + repo)
- **Licence landmines in the obvious choices.** The de-facto standard SIM-o encoder ships from `microsoft/UniSpeech`, which is **CC BY-SA 3.0** (share-alike), and its checkpoint is a **Google Drive link**. The obvious age/gender classifier (`audeering/wav2vec2-large-robust-24-ft-age-gender`) is **CC-BY-NC-SA-4.0** (non-commercial). `praat-parselmouth` is **GPLv3**. MMS and SeamlessM4T-v2 are **CC-BY-NC-4.0**. ParaSpeechCaps is **CC-BY-NC-SA-4.0**. None of these are blockers for an internal dev harness, but every one of them is a blocker for anything that ships. **[HIGH — all verified from licence fields/LICENSE files]**
- **Indic is a different calibration regime, not a different pipeline.** `ai4bharat/indic-conformer-600m-multilingual` is **MIT**, covers all 22 scheduled languages, and is the correct choice. But its reported Hindi WER is **13.2** on the Vaani benchmark versus ~2% English WER floors — you must set per-language baselines or you will read ASR weakness as TTS failure. **[HIGH]** ([model card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), source: model card)

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| 1 | InstructTTSEval = arXiv 2506.16381 | **Correct** | Exists. "InstructTTSEval: Benchmarking Complex Natural-Language Instruction Following in Text-to-Speech Systems", Huang et al., submitted 2025-06-19, CC-BY-4.0. Dataset MIT on HF. | [arXiv abs](https://arxiv.org/abs/2506.16381) · [HF dataset](https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval) | HIGH |
| 2 | "12 attributes: pitch, speed, emotion, gender, age, clarity, fluency, accent, texture, tone, volume, personality" | **Correct — all 12, exact** | Verified line-by-line against the shipped judge prompt. No additions, no omissions. Rare case of a brief being precisely right. | [eval_prompt.txt](https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/eval_prompt.txt) | HIGH |
| 3 | Implied: we can "follow InstructTTSEval's protocol" locally | **WRONG in one respect** | The protocol is adoptable; the *judge* is not. It requires `google-genai`, a paid API key, per-item audio upload, and a specific preview model. No open-weight judge is provided. Budget for API cost or substitute an open audio-LLM and re-validate. | [gemini_eval.py](https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/gemini_eval.py), [requirements.txt](https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/requirements.txt) | HIGH |
| 4 | Implied: InstructTTSEval scoring is graded/scalar | **WRONG** | Scoring is **binary** — the judge emits a JSON dict and only the `一致性` (consistency) field is read, `true`/`false`. Reported as % true. Low resolution; see §7 for the statistical consequence. | [gemini_eval.py](https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/gemini_eval.py) L~"if result and '一致性' in result" | HIGH |
| 5 | "English set as well as InstructTTSEval-Zh" | **Correct** | 1,000 EN + 1,000 ZH, each with APS/DSD/RP instructions = 6,000 instructions. Both splits public. Note: **the judge prompt is Chinese for both splits** — the repo README states "`eval_prompt.txt` is used both for EN and ZH subsets." | [eval/README.md](https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/README.md) | HIGH |
| 6 | Axis 2: use an INDEPENDENT speaker-verification model | **Correct and important** | Endorsed. Achievable at zero cost with WeSpeaker ResNet34-LM. See §3.1 for the independence matrix. | [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | HIGH |
| 7 | Axis 3: "a mode-collapsed mapper scores WELL on adherence and fails only here" | **Correct — and this is the load-bearing insight of the whole harness** | Confirmed by construction: a mapper that emits one canonical "old gravelly man" voice satisfies every re-measured attribute bin and every judge verdict. Only spread in speaker space exposes it. No published TTV benchmark tests this. | reasoning + absence of counter-evidence in §4 | HIGH |
| 8 | Implied: a defensible pass/fail cosine threshold exists in the literature | **WRONG** | No universal threshold exists. Values are encoder-specific and non-comparable. SpeechBrain's shipped default is `threshold=0.25` for ECAPA; real same-speaker SIM-o on WavLM-large is 0.69–0.76; different real speakers on wavlm-base-plus-sv average 0.67. Must calibrate in-run. | [speechbrain/inference/speaker.py](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/speechbrain/inference/speaker.py) L62 | HIGH |
| 9 | Implied: UTMOS/DNSMOS are safe naturalness gates | **WRONG for this project specifically** | They are anti-correlated with pitch in a way humans are not, and blind to prosody. For stylized/aged/whispered voices they are actively misleading. See §5.2. | [arXiv 2606.19951](https://arxiv.org/html/2606.19951v1) | HIGH |
| 10 | Speaking rate needs a forced aligner (MFA/charsiu/whisper-timestamped) | **WRONG — you don't need one** | In TTV **you already know the text**. Phones/sec = (G2P phone count of the known text) ÷ (voiced duration from VAD). Deterministic, exact, no alignment error. An aligner is only needed for *per-phone* durations, which no adherence axis requires. | reasoning from task structure; g2p-en + silero-vad | HIGH |

---

## 2. Axis 1 — description adherence

### 2.1 Objective re-measurement stack (runnable)

This is the cheapest and most powerful adherence metric, and it is the one to build first. It works because captions are grounded in measurements and therefore reversible — you re-measure the generated audio and check it lands in the bin the description named.

| Attribute | Library | Call | Reliable on synthetic? | Licence |
|---|---|---|---|---|
| **F0 mean / median / range** | `praat-parselmouth` 0.4.7 | `snd.to_pitch(pitch_floor=60, pitch_ceiling=500).selected_array['frequency']` | **Yes — high.** Autocorrelation pitch is the best-validated measure here. Widen ceiling to 600 for child/falsetto; a 500 Hz default silently clips high voices. | **GPLv3** |
| **F0 (neural, for hard cases)** | `torchcrepe` 0.0.24 / `penn` 1.0.0 | `torchcrepe.predict(...)` / `penn.from_audio(...)` | **Yes.** PENN reports 97.0% RPA and **11.2× realtime on CPU / 408× on GPU** — faster and more accurate than CREPE. Use as a cross-check on breathy/creaky output where autocorrelation fails. | **MIT** (both) |
| **F0 (vocoder-grade)** | `pyworld` 0.3.6 | `pw.harvest(x, fs)` | Yes. Also what TTSDS2 uses for its prosody factor. | (permissive, no SPDX declared — **verify before shipping**) |
| **Jitter (local)** | `praat-parselmouth` | `call([snd, pp], "Get jitter (local)", 0,0, 1e-4, 0.02, 1.3)` via `PointProcess` | **⚠️ LOW — report but do not gate on it.** See caveat below. | GPLv3 |
| **Shimmer (local)** | `praat-parselmouth` | `call([snd, pp], "Get shimmer (local)", ...)` | **⚠️ LOW.** Same caveat. | GPLv3 |
| **HNR** | `praat-parselmouth` | `snd.to_harmonicity_cc()` → `.values.mean()` | **MEDIUM — the most usable of the three.** HNR was identified as one of the most influential predictors of perceptual voice grade, where jitter/shimmer "contributed minimally." Best available proxy for *raspy vs clear*. | GPLv3 |
| **Formants F1–F3** | `praat-parselmouth` | `snd.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)` | MEDIUM. **Must set `maximum_formant` by expected gender** (5500 Hz female / 5000 Hz male) or F-numbering flips. Vowel-conditioned — compare only across the *same fixed script*. | GPLv3 |
| **Vocal-tract length (eVTL)** | derived from formants | formant dispersion = mean successive ΔF | **⚠️ LOW as absolute inference, MEDIUM as relative feature.** eVTL correlates with actual speaker height at only **r ≈ .25–.32** among adults. Use to rank two synthesized voices, never to claim a body size. | — |
| **Speaking rate (phones/sec)** | `g2p-en` + `silero-vad` | `len(G2p()(text))` ÷ Σ(voiced segment durations) | **HIGH — exact.** Text is known; no aligner needed, no alignment error. Report *articulation rate* (excluding internal pauses) and *speaking rate* (including) separately. | `g2p-en` Apache-2.0 · `silero-vad` MIT |
| **Speaking rate (transcript-free fallback)** | Praat syllable-nuclei | de Jong & Wempe (2009) intensity-peak method | MEDIUM. Only needed if text is unavailable. Validated against human syllable counts on Dutch corpora. | GPLv3 |
| **Volume / loudness** | `praat-parselmouth` | `snd.to_intensity().values.mean()`; or LUFS via `pyloudnorm` | HIGH *in relative terms only.* Absolute dB is meaningless post-normalisation — **measure before any loudness normalisation, or the volume attribute is destroyed by your own pipeline.** | GPLv3 |
| **SNR + reverberation (C50)** | Brouhaha | `brouhaha-vad` (pyannote-based) | MEDIUM. Jointly predicts VAD + speech-to-noise ratio + C50. Designed for real recordings; on clean TTS output expect saturation at the high end — useful as an *anomaly detector* (has the backend started hallucinating room tone?) rather than a graded score. Not on PyPI — install from git. | **MIT** (CNRS) |
| **SNR/STOI/PESQ (reference-free)** | `torchaudio` SQUIM Objective | `SQUIM_OBJECTIVE.get_model()` | MEDIUM. Note `SquimSubjective` needs a **non-matching clean reference** — it is NORESQA-MOS, not a standalone predictor. Easy to misuse. | BSD |
| **Age + gender (perceived)** | `audeering/wav2vec2-large-robust-24-ft-age-gender` | HF `AutoModel`, outputs age∈[0,1]→×100 yrs, and P(child/female/male) | MEDIUM. **MAE 7.1–10.8 years**, gender ≥91.1% acc. Too coarse for a year figure; **adequate for 4 coarse bins** (child / young adult / middle / elderly), which is all the descriptions need. Trained on aGender+CommonVoice+TIMIT+VoxCeleb2 — real speech only; transfer to synthetic is **UNVERIFIED**. | **CC-BY-NC-SA-4.0 ⚠️ non-commercial** |
| **Age/gender (permissive swap-in)** | train a linear probe on WavLM/ECAPA embeddings using a permissive labelled corpus | — | The honest Apache/MIT path. Costs a day. | depends on corpus |

**The jitter/shimmer caveat, stated honestly.** The brief's suspicion is correct and you should act on it. Jitter and shimmer are *cycle-to-cycle* perturbation measures defined on a `PointProcess` of glottal pulse instants. Their validation literature is built on **sustained vowels from real speakers**, and Praat's own algorithm verification was done using **synthesized vowels with exactly zero jitter and shimmer** produced by a formant synthesizer — i.e. the tool was checked against synthetic *steady* signals, not connected synthetic speech. Two failure modes bite this project specifically:

1. **Connected speech breaks the assumption.** Pitch-period marking across voiced/unvoiced transitions, creak, and phrase-final devoicing injects spurious perturbation. Values become a function of the *sentence*, not the *voice*.
2. **Neural vocoders are not glottal sources.** A HiFiGAN/BigVGAN/flow-matching decoder produces waveforms with periodicity statistics unlike a real larynx. A "raspy" stylization may be rendered as spectral noise without any period-to-period irregularity — jitter reads clean while the voice sounds rough.

**Ruling:** compute jitter/shimmer, log them, but **do not gate on them and do not put them in a scorecard cell that implies a pass/fail**. Use **HNR** as the raspiness proxy, gate on that, and validate HNR against your own ears once (§10). This is the difference between a metric and a number.

**A second trap worth naming:** measure volume, SNR and F0 **before** any loudness normalisation or resampling in your own pipeline. A harness that normalises then measures loudness is measuring its own normaliser.

### 2.2 InstructTTSEval protocol

Verified from the repository, not from the paper's prose.

**Three tiers**, increasing in abstraction — each item carries the same `text` with three alternative instructions:

| Tier | What it is | Example (verbatim, `en_0`) |
|---|---|---|
| **APS** — Acoustic-Parameter Specification | A structured, attribute-by-attribute spec across the 12 dimensions, one per line as `key: value.` | `gender: Male.` / `pitch: Mid-range male pitch, slightly elevated at the beginning then stabilizing.` / `speed: Initially brisk, then slightly relaxing…` / … / `personality: Appears direct and confident in their statement.` |
| **DSD** — Descriptive-Style Directive | One flowing prose sentence that *implies* the attributes rather than listing them. | "Incorporate the nuances of General American English by inflecting your voice with a slightly elevated pitch at the beginning before settling into steady clarity, achieving fluency with an informal yet assertive emotional progression." |
| **RP** — Role-Play | A scenario/persona framing from which the voice must be inferred. | "Imagine a context where there's a clear explanation when talking to a child about a complex idea. This TTS voice should have the texture of a smooth young adult voice in General American English, initially quick with a slightly elevated pitch for engagement…" |

**The 12 dimensions, verbatim from `eval_prompt.txt`** (Chinese source, English gloss):

| # | Source | Gloss | Prompt's own definition (condensed) |
|---|---|---|---|
| 1 | 性别 | gender | vocal-fold differences + socialised speech patterns |
| 2 | 音高 | pitch | perceived frequency; **expressed relative to gender** ("female high pitch", "male low steady") |
| 3 | 语速 | speed | rate, may vary within an utterance; note rhythmic patterns |
| 4 | 音量 | volume | whisper / conversational / shout |
| 5 | 年龄 | age | child / teen / young adult / middle-aged / elderly |
| 6 | 清晰度 | clarity | crisp articulation vs mumbling/slurring |
| 7 | 流畅度 | fluency | absence of hesitation, repetition, fillers ("um", "like", "you know") |
| 8 | 口音 | accent | dialect region as precisely as identifiable |
| 9 | 音色质感 | texture/timbre | sweet, hoarse, deep, bright, warm, nasal, soft, coarse, thin |
| 10 | 情绪 | emotion | may shift mid-utterance |
| 11 | 语调 | tone | sarcasm, formality, enthusiasm, indifference |
| 12 | 性格 | personality | extrovert/introvert, confident, decisive, anxious — **only if consistently evident** |

**Judge:** `models/gemini-2.5-pro-preview-05-06`, `temperature=0`, all four safety categories set to `BLOCK_NONE`, audio uploaded via `client.files.upload()` per item and deleted after. 5 retries. 10 worker processes by default.

**Output contract:** JSON dict with one key per dimension plus `一致性` (consistency) as `true`/`false`. **Only `一致性` is scored.** Three JSON-extraction fallbacks (fenced block → regex object search → whole-text parse).

**The four judging instructions that do the real work** (these are the transferable part — note how adversarial they are):

1. If any dimension objectively conflicts — *especially speaker gender or age* — return `false` immediately.
2. "The description has a high probability of conflicting with the speech… **do not readily trust the given description**; form your own understanding of the audio first." (An explicit anti-sycophancy instruction.)
3. "Speaker gender is especially likely to be the opposite of the description — pay particular attention."
4. When the description uses intensity words ("excited", "strong"), the audio very likely **under-delivers**; judge `false` if the degree is insufficient.

Plus: ignore non-style factors (pronunciation accuracy, naturalness); judge only against the description; unmentioned attributes are unconstrained.

**Reported results** (% consistent; APS/DSD/RP). English: human reference 96.2/89.4/67.2 — note the reference *itself* only scores 67.2 on Role-Play, which is the ceiling, not 100. Gemini-flash 92.3/93.8/80.1; Hume 83.0/75.3/54.3; **Parler-TTS-mini 63.4/48.7/28.6**, **Parler-TTS-large 60.0/45.9/31.2**, VoxInstruct 54.9/57.0/39.3, PromptTTS 64.3/47.2/31.4, PromptStyle 57.4/46.4/30.9. **[MEDIUM — extracted from paper HTML tables, not re-derived]**

**The number that should anchor your expectations:** the best *open* system scores **~63/49/29**. If your S0 baseline lands near that, you are at the open-source state of the art, not failing.

**Can we run it on a 12GB GPU?** The eval itself needs **no GPU at all** — `google-genai`, `requests`, `tqdm`. All compute is remote. The constraint is not VRAM, it is **API access and cost**: 3 calls × N items, each with an audio upload, on a Pro-tier model. For a 50-description set × 3 tiers = 150 calls per pass, which is affordable. For the full 6,000-instruction benchmark it is not, and you do not need it.

### 2.3 LLM-judge design

**Recommendation: adopt the protocol, replace the judge, and validate the replacement — or skip the judge for S0 entirely.**

Three options, in order of preference:

1. **Objective re-measurement only for S0.** The judge adds nothing you can act on at baseline, costs money, and introduces a closed dependency. Ship §2.1, get numbers, add the judge when you have two systems to compare.
2. **Gemini judge, InstructTTSEval-faithful, small N.** 50 descriptions × 3 tiers = 150 calls per eval pass. Keep the prompt **byte-identical** to `eval_prompt.txt` (including the Chinese — it is used for the English split too, so translating it breaks comparability with published numbers). Add your own 13th key only *after* the `一致性` field so parsing stays compatible.
3. **Open audio-LLM judge.** Qwen3-Omni (2025-09, arXiv 2509.17765) and Kimi-Audio are open-weight and paralinguistically capable; layer-wise analyses of paralinguistic understanding have been published for Qwen2.5-Omni and Kimi-Audio. **But no published work validates any open audio-LLM as an InstructTTSEval-protocol judge, and agreement with the Gemini judge is UNVERIFIED.** If you go this way, the validation experiment in §10 is mandatory, not optional.

**Design rules regardless of judge:**

- **Keep the binary verdict.** Do not "improve" it to a 1–5 scale — you lose comparability with the only published numbers, and a scale invites the judge to hedge at 3.
- **Keep the anti-sycophancy instructions.** They are the reason the judge is not a rubber stamp. A judge told "here is a description and its audio" will agree ~90% of the time.
- **Score gender and age as hard fails.** Objective, verifiable, and the failure mode most likely to be silently tolerated.
- **Never let the judge see the identity ID, the checkpoint name, or the system name.** Blind it, and randomise item order per run.
- **Log the full per-dimension dict, not just the verdict.** It costs nothing and gives you a free diagnostic: which of the 12 dimensions your mapper drops.

---

## 3. Axis 2 — identity consistency

### 3.1 Encoder independence matrix

| Encoder | Checkpoint | Code licence | Weights licence | Dim | VoxCeleb1-O EER | SSL frontend | Training data |
|---|---|---|---|---|---|---|---|
| **WeSpeaker ResNet34-LM** ⭐ | `pyannote/wespeaker-voxceleb-resnet34-LM` | Apache-2.0 | **CC-BY-4.0** | 256 | **0.723%** (LM+AS-Norm) / 0.659% (+QMF) | none — fbank80 | VoxCeleb2 |
| WeSpeaker ResNet221-LM | `voxceleb_resnet221_LM` | Apache-2.0 | CC-BY-4.0 | 256 | **0.505%** | none | VoxCeleb2 |
| WeSpeaker ResNet293-LM | `voxceleb_resnet293_LM` | Apache-2.0 | CC-BY-4.0 | 256 | **0.425%** (+QMF) — best in zoo | none | VoxCeleb2 |
| WeSpeaker CAM++ | `voxceleb_CAM++` | Apache-2.0 | CC-BY-4.0 | 192 | 0.803% (0.654% w/ AS-Norm per CAM++ paper) | none | VoxCeleb2 |
| WeSpeaker ECAPA1024-LM | `voxceleb_ECAPA1024_LM` | Apache-2.0 | CC-BY-4.0 | 192 | 0.728% | none | VoxCeleb2 |
| SpeechBrain ECAPA-TDNN | `speechbrain/spkrec-ecapa-voxceleb` | Apache-2.0 | **Apache-2.0** | 192 | 0.80% | none | VoxCeleb1+2 |
| NeMo TitaNet-L | `nvidia/speakerverification_en_titanet_large` | Apache-2.0 (toolkit) | **CC-BY-4.0** | 192 | ~0.66% (vendor-reported, **UNVERIFIED here**) | none | VoxCeleb + others |
| pyannote/embedding | `pyannote/embedding` | MIT | **MIT** (gated: auto) | 512 | — | none | VoxCeleb |
| Resemblyzer (GE2E) | `resemble-ai/Resemblyzer` | **Apache-2.0** | Apache-2.0 | 256 | ~5–6% (legacy, **UNVERIFIED**) | none | VoxCeleb+LibriSpeech |
| **WavLM-large SV** (the SIM-o standard) | `wavlm_large_finetune.pth` from `microsoft/UniSpeech` | **CC BY-SA 3.0 ⚠️** | same | 512 | 0.383% (reported) | **WavLM** | Mix6/VoxCeleb |
| `microsoft/wavlm-base-plus-sv` | HF | — (no licence field ⚠️) | **none declared** | 512 | — | **WavLM** | VoxCeleb |
| 3D-Speaker ERes2Net / CAM++ | modelscope | **Apache-2.0** | per-model | 192 | — | none | VoxCeleb/CNCeleb |

**The independence ruling.**

The contaminating families are (a) **WavLM-derived** — anything whose frontend is WavLM shares representations with a WavLM-conditioned TTS, and (b) **ECAPA-TDNN** — the most common conditioning architecture in open TTS.

- **Scoring encoder (primary): WeSpeaker ResNet34-LM.** fbank80 → ResNet. No SSL frontend at all, therefore no representational overlap with a WavLM conditioning encoder; different architecture family from ECAPA. Apache-2.0 code, CC-BY-4.0 weights, sub-1% EER, 256-dim, loads in three lines through `pyannote.audio`. This is the clean choice.
- **Scoring encoder (secondary, for agreement): NeMo TitaNet-L** — different toolkit, different training recipe, different vendor. Agreement between ResNet34-LM and TitaNet-L is your evidence that a result is encoder-independent rather than an artefact.
- **Do NOT use for scoring** if you condition on WavLM or ECAPA: the WavLM-large SV checkpoint, `wavlm-base-plus-sv`, or SpeechBrain ECAPA. Reserve **WavLM-large SV for reporting SIM-o only**, purely so your numbers are comparable to the published literature — and label it as such.
- **Report both.** One column "SIM-o (WavLM-large, for literature comparability)" and one column "identity consistency (WeSpeaker ResNet34-LM, independent)". They answer different questions and mixing them is how you fool yourself.

**Two operational hazards.** The WavLM-large SV checkpoint is distributed as a **Google Drive link** inside `cal_sim.sh` — pin a local copy and hash it, or your eval becomes irreproducible the day that link rots. And `microsoft/wavlm-base-plus-sv` has **no declared licence** on HF; do not build a shipping dependency on it.

**Is there published work on how much SV-model choice changes TTS similarity rankings?** Not directly. TTSDS2 uses *two* speaker extractors (d-Vector and WeSpeaker) precisely because no single one is trusted, which is indirect evidence that the community treats this as a real risk — but a head-to-head study of ranking stability across SV encoders is **UNVERIFIED / apparently unpublished**. Worth the cheap experiment in §10.

### 3.2 Defensible thresholds

**The honest position first: there is no universal cosine threshold, and any number you quote without naming the encoder is meaningless.** Three published anchors, all on *different* encoders, all mutually incomparable:

| Anchor | Encoder | Value | Source |
|---|---|---|---|
| SpeechBrain shipped same-speaker default | ECAPA-TDNN (192-d) | `threshold = 0.25` | [speaker.py L62](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/speechbrain/inference/speaker.py) (repo code, HIGH) |
| **Real speech, same speaker, different utterance (the ceiling)** | WavLM-large SV | **0.69** (LibriSpeech-PC), **0.73** (Seed-TTS test-en), **0.76** (test-zh) | F5-TTS paper, "Ground Truth" row ([arXiv](https://arxiv.org/html/2410.06885v3), HIGH) |
| **Genuinely different real speakers, mean pairwise** | wavlm-base-plus-sv | **0.67 ± 0.18** | Chen et al. 2025 ([arXiv 2511.07135](https://arxiv.org/html/2511.07135), HIGH) |
| Best open TTS SIM-o vs reference | WavLM-large SV | 0.66–0.76 | F5-TTS / E2-TTS tables (HIGH) |
| Generated-speaker **intra-identity** consistency ("Stability") | wavlm-base-plus-sv | **0.85 ± 0.10** (FACodec), **0.90 ± 0.06** (CosyVoice2) | Chen et al. 2025 (HIGH) |

Look at rows 3 and 5 together: on the *same* encoder, different real people score **0.67** and the same generated identity scores **0.85–0.90**. That ~0.20 gap is the entire signal. And note that 0.67 for *different people* would look like a pass against a naive 0.25 threshold — which is exactly how a mode-collapsed system passes an identity test.

**The protocol that is actually defensible:**

> **Calibrate three reference distributions in-run, on your scoring encoder, from the same fixed script, every single eval pass. Never hardcode a threshold.**
>
> - **C_same** — 20 utterances from one *real* held-out speaker → mean pairwise cosine. This is the achievable ceiling.
> - **C_diff** — 20 utterances from 20 *different* real speakers → mean pairwise cosine. This is the floor.
> - **C_id(d)** — 20 utterances from one generated identity → mean pairwise cosine. The thing under test.
>
> **Identity consistency passes when `C_id(d) ≥ C_same − 0.05`**, evaluated per description, reported as the fraction of descriptions passing.
>
> Report the normalized form **`(C_id − C_diff) / (C_same − C_diff)`** — 1.0 = as consistent as a real person, 0.0 = no more consistent than a room full of strangers. This is portable across encoders in a way raw cosine is not.

**The three consistency conditions the brief asks for**, all using the same statistic:

1. **Within-session:** 20 sentences, one identity, one process. → `C_id`
2. **Across session restart:** re-load the identity in a fresh process, regenerate. → `C_restart`, compare to `C_id`. Any drop >0.03 means your identity is not actually being persisted — it is being partly re-derived from RNG state. This is a **real and common bug** and this test is the only thing that catches it.
3. **Across backend version bump:** same identity, new TTS weights. → `C_version`. Expect a drop; the point is to *quantify* it so "we upgraded the vocoder and all the voices changed" is a number rather than a complaint.

**SIM-o and SIM-r, defined precisely.**

- **SIM-o** ("original"): cosine similarity between the speaker embedding of the **generated** audio and that of the **original, real** reference/prompt audio.
- **SIM-r** ("resynthesized"): cosine similarity against a **codec-resynthesized** version of the reference — i.e. the reference passed through the system's own audio codec/vocoder round-trip. It removes the codec's own fidelity ceiling from the score.
- **SIM-r is not comparable across systems using different vocoders/codecs** and is therefore the weaker number; SIM-o is the one to report. **[HIGH on the definitions; MEDIUM on VALL-E as the specific originating paper — widely attributed there but not re-verified against arXiv 2301.02111 in this pass.]**
- **The known inconsistency you must not reproduce:** VALL-E excluded the prompt from the scored segment (0.754) while VALL-E 2 included it (0.905) — **a 0.151 absolute difference from a definitional choice alone** ([arXiv 2510.06927](https://arxiv.org/html/2510.06927v1), HIGH). **Write down which convention you use, in the scorecard, forever.** For Alaap the prompt-inclusion question is moot (there is no audio prompt — identity comes from a vector), which is a genuine simplification. State that explicitly so nobody later "corrects" your numbers to a convention that doesn't apply.

**Published work on intra-identity consistency across many generations:** the closest is Chen et al. 2025's "Stability" metric (0.85–0.90). It measures embedding similarity across utterances converted *from different source speakers* to the same generated target — structurally the same statistic as ours. Beyond that, **the literature is thin: TTS eval overwhelmingly measures similarity-to-a-reference, not self-consistency-across-generations.** The position paper confirms the gap: "existing metrics rarely account for speaker consistency over extended durations." **This axis is genuinely under-served, and building it is a real contribution rather than a re-implementation. [HIGH]**

---

## 4. Axis 3 — diversity / anti-mode-collapse

**State of the literature: nobody has built this properly, and no TTV benchmark includes a diversity axis.** InstructTTSEval has no diversity component. TTSDS2 measures distributional distance *to real speech*, which is a realism check, not a spread check — a mode-collapsed system with one very realistic voice scores well. ClonEval evaluates voice cloning. The closest prior art is speaker-generation work (TacoSpawn, arXiv 2111.05095, which proposes objective metrics for novel-speaker generation but whose exact formulations I could **not extract from the PDF — UNVERIFIED**) and Chen et al. 2025, which does give usable, concrete numbers. **The brief's claim that this test is what makes the silent failure visible is correct, and it is correct partly because nobody else is running it.**

**Available metric machinery, verified:**

| Metric | What it gives you | Package | Licence | Used in speech? |
|---|---|---|---|---|
| **Vendi Score** | "Effective number of distinct elements" — exp(Shannon entropy of eigenvalues of the similarity matrix). **No reference set needed.** Explicitly demonstrated for measuring GAN mode collapse. | `vendi-score` 0.0.3 | **MIT** | Not in speech that I found — **novel application here** |
| Mean pairwise cosine | Crude but directly interpretable spread | numpy | — | **Yes** — Chen et al. 2025 "Pairwise Diversity" |
| Density & Coverage | Fidelity/diversity split, reference-based | `prdc` 0.2 | **MIT** | Rare in speech |
| Improved Precision/Recall | Manifold overlap | `prdc` | MIT | Rare in speech |
| Silhouette / Davies-Bouldin | Cluster separability given labels | `scikit-learn` | BSD | Standard, general |
| 2-Wasserstein between feature sets | TTSDS2's own machinery | `ttsds` 2.1.3 | **MIT** | **Yes** — TTSDS2 |

### The concrete metric, with a target number

**Two numbers, both cheap, both calibrated in-run. Do not use absolute cosine.**

**Setup.** For each of M = 50 descriptions, sample **N = 20 identities** and render **one fixed short line** with each (same line for all — this removes text as a confound). Embed all 1,000 with **WeSpeaker ResNet34-LM** (256-d, L2-normalised). In the same pass, embed 20 utterances from 20 **real, distinct** held-out speakers speaking that same line — this is your calibration set.

**Metric 1 — Normalized Vendi Score (nVS). Primary.**

```
nVS(d) = VendiScore(embeddings for description d, cosine kernel, q=1) / N
```

Vendi Score is the effective number of distinct voices among the N samples. Dividing by N puts it on [1/N, 1].

- **Target: nVS ≥ 0.35** — i.e. **≥ 7 effective distinct voices out of 20** for a typical description.
- **Alarm: nVS < 0.20** (< 4 effective voices).
- **Mode collapse declared: nVS < 0.10** (≤ 2 effective voices).
- Report `nVS_real` for the 20 real distinct speakers in the same run as the achievable ceiling — expect it near 0.7–0.9. **Do not target 1.0: a description legitimately constrains the space.** "An elderly gravel-voiced man" *should* be less diverse than 20 random humans. The question is whether it is more diverse than one voice.

**Metric 2 — Collapse tripwire. Hard fail, no interpretation needed.**

```
mean pairwise cosine within description d  >  C_same − 0.05
```

where `C_same` is the measured same-real-speaker consistency from §3.2. **If N samples that are supposed to be different people score as similar as one person talking to themselves, they are one person.** This is a binary, un-arguable, un-gameable failure signal, and it is grounded: Chen et al. measured deliberately-same-identity generated audio at **0.85–0.90** and genuinely-different real speakers at **0.67** on the same encoder. If your "diverse samples" land at 0.85+, the mapper has collapsed.

**Metric 3 — Cluster separability. Confirms descriptions are distinguishable, not just noisy.**

Silhouette score over all M×N embeddings labelled by description ID.

- **Target: silhouette ≥ 0.15.** Modest by clustering standards, and deliberately so — descriptions overlap semantically ("gruff old man" and "weathered elderly sailor" *should* be near each other, and forcing them apart would be a worse system, not a better one).
- **The diagnostic that matters more than the number:** mean *between*-description distance must exceed mean *within*-description distance, with a bootstrap 95% CI on the difference excluding zero. If it doesn't, your mapper is ignoring the description entirely — and note this is the exact failure that **adherence metrics cannot see** if the single collapsed voice happens to sit in a plausible bin.

**Why this combination is defensible:** Metric 1 is a published mode-collapse detector (Vendi Score, demonstrated on GANs) applied to a published embedding space with published EER. Metric 2 is grounded in published intra- vs inter-speaker cosine values on real data. Metric 3 is standard clustering practice. All three are calibrated against real speakers measured in the same run on the same encoder, so encoder swaps and version bumps don't silently move the goalposts. Total cost: ~1,000 extra renders and about two minutes of embedding compute.

---

## 5. Axis 4 — intelligibility & naturalness

### 5.1 ASR (EN + Indic)

| Model | Checkpoint | Licence | Use | Notes |
|---|---|---|---|---|
| **Whisper large-v3** ⭐ | `openai/whisper-large-v3` | **Apache-2.0** | English WER — **the standard** | Used by seed-tts-eval, F5-TTS, CosyVoice, E2-TTS. Use it for comparability even if something is marginally better. ~10GB fp16 — fits 12GB. |
| Whisper large-v3-turbo | `openai/whisper-large-v3-turbo` | **MIT** | Fast iteration loop | Use for inner-loop checks, large-v3 for reported numbers. |
| **IndicConformer 600M** ⭐ | `ai4bharat/indic-conformer-600m-multilingual` | **MIT** (gated: auto) | **All 22 scheduled Indian languages** | Hybrid CTC+RNNT, 600M. Hindi WER **13.2** on Vaani. The right choice on licence *and* coverage. |
| IndicConformer per-language | `ai4bharat/indicconformer_stt_{hi,ta,te,bn,ml,mr,ur,as,ne}_hybrid_ctc_rnnt_large` | MIT | Per-language, if the multilingual model underperforms | Nine+ languages published separately. |
| Paraformer-zh | `funasr/paraformer-zh` | (FunASR — verify) | Mandarin, only if you add it | seed-tts-eval's ZH engine. |
| ~~MMS~~ | `facebook/mms-1b-all` | **CC-BY-NC-4.0 ⛔** | — | **Non-commercial. Excluded by the locked constraint.** |
| ~~SeamlessM4T v2~~ | `facebook/seamless-m4t-v2-large` | **CC-BY-NC-4.0 ⛔** | — | Same. Note `ai4bharat/indic-seamless` is a derivative — inherit-check before use. |
| IndicWhisper | — | — | — | **UNVERIFIED as a usable artifact.** No official `ai4bharat/indicwhisper` repo resolves on HF; only third-party re-uploads of varying provenance. The Vistaar paper exists; the distributable checkpoint does not, cleanly. **Do not plan around it.** |

**Baselines — what "good" actually is:**

| Test set | ASR | Ground-truth (real speech) WER | Best open TTS WER | Source |
|---|---|---|---|---|
| LibriSpeech-PC test-clean | whisper-large-v3 | **2.23%** | 2.42–2.95% | F5-TTS Table (HIGH) |
| Seed-TTS test-en | whisper-large-v3 | **2.06%** | **1.83%** (F5-TTS, *below* GT) | F5-TTS Table (HIGH) |
| Seed-TTS test-zh | Paraformer-zh | 1.26% | 1.56% | F5-TTS Table (HIGH) |
| Vaani Hindi (real, conversational) | IndicConformer 600M | **13.2%** | — | model card (HIGH) |

**Three calibration facts to internalise:**

1. **Real speech does not score 0% WER.** The floor is ~2%, set by the ASR. Anything at 2–3% on English is *at parity with human recordings*. Chasing 1% is chasing ASR noise.
2. **TTS can beat ground truth** (1.83% vs 2.06%) because synthetic speech is *cleaner and more canonically pronounced* than real speech. **A very low WER is therefore not unambiguously good** — it can indicate a flat, over-articulated, under-stylized voice. For a project whose goal is heavy stylization, **WER dropping is a yellow flag as often as a green one.** Report WER alongside the stylization axes, never alone.
3. **The Indic floor is ~5× higher and unmeasured for synthetic input.** 13.2% is on real noisy conversational speech; clean synthetic will be lower but is **UNVERIFIED**. You must measure your own per-language floor by running IndicConformer over *real* held-out clean recordings before you can interpret a single Indic TTS WER number.

**Normalisation:** replicate seed-tts-eval's `process_one()` exactly — strip all ASCII + CJK punctuation (keeping apostrophes), collapse double spaces, lowercase for English, character-split for Chinese, then `jiwer.compute_measures`. Reuse the code rather than writing your own; text normalisation differences alone move WER by tenths of a point and will make your numbers incomparable.

**The metric worth adding, from the position paper:** WER's correlation with perception is non-linear and saturating — going 1.61% → 1.47% had "minimal impact on user perception." **Below ~3%, stop treating WER as a quality signal and start treating it as a regression tripwire.** Its job is to catch catastrophic failure (the voice became unintelligible), not to rank good systems.

### 5.2 MOS predictors, and where they lie to you

| Predictor | Package / repo | Licence | Status 2026 | Verdict for Alaap |
|---|---|---|---|---|
| **UTMOSv2** | `sarulab-speech/UTMOSv2` | **MIT** | Maintained (pushed 2026-04-02, 365★) | Best-maintained of the classic family. **Still pitch-biased (r = −0.722).** |
| UTMOS (22) | `sarulab-speech/UTMOS22` | **MIT** | Stale (last push 2024-04) | Still the most-cited. Use only for comparability with older papers. |
| **Distill-MOS** | `pip install distillmos` 0.9.1 | **MIT** (verified in LICENSE) | Active (2025-05) | **Cheapest credible option.** ConvTransformer distilled from a wav2vec2 XLS-R teacher; 16 kHz mono in, 1–5 out, 3 lines of code, tiny. Good default. |
| NISQA | `gabrielmittag/NISQA`, `pip install nisqa` | **MIT** | Widely used (968★) | Gives dimension sub-scores (noisiness, coloration, discontinuity, loudness) — **more diagnostic than a scalar MOS**, which is worth more to you than the MOS itself. |
| DNSMOS P.835 | `pip install speechmos` | MIT (wrapper) | Standard | **Trained on speech *enhancement* data, applied to synthesis** — the position paper names this domain mismatch explicitly. **Worst pitch bias measured (r = −0.788).** |
| torchaudio SQUIM | `torchaudio` | BSD | Maintained | `SQUIM_OBJECTIVE` (PESQ/STOI/SI-SDR) is useful. **`SquimSubjective` requires a non-matching clean reference** — it is NORESQA-MOS. Frequently misused as reference-free. |
| **TTSDS2** ⭐ | `pip install ttsds` 2.1.3 | **MIT** | Active (2026-07-07) | **The one to actually trust.** See below. |

**VoiceMOS Challenge status, verified:** editions ran **2022, 2023, 2024**; **AudioMOS Challenge 2025** (ASRU 2025, Honolulu) broadened to general audio; **VoiceMOS Challenge 2026 is live — results were released to participants on 2026-08-31**, two days ago, and are **not yet public** (papers due 2026-09-16 for ICASSP 2027). Its tracks are directly relevant: Track 2 is emotional-TTS quality *and emotional similarity*, Track 3 is **speaker and accent similarity prediction**. **Re-check this in ~4 weeks; the 2026 results may hand you a better predictor for free.** Baselines named for 2026: URGENT-MOS, UTMOS, Emotion2vec. **[HIGH on status; the results themselves are UNVERIFIED because unpublished]**

#### Where they lie to you — the evidence

**Trap 1 — pitch bias (the one that matters most for this project).** Takagi et al., arXiv 2606.19951 (2026-06-18), scaled F0 by factors {0.5 … 2.0} and compared six predictors (SHEET-MB, SHEET-BV, UTMOS, UTMOSv2, NISQA, DNSMOS) against human ratings:

| | Correlation with mean F0 |
|---|---|
| **Humans** | **r = −0.059** (essentially none) |
| DNSMOS | **r = −0.788** |
| UTMOSv2 | **r = −0.722** |

**Every one of these models penalises higher-pitched voices for reasons human listeners do not share.** Alaap deliberately spans children, elderly, falsetto, and heavily stylized voices. Using any of these as a quality gate installs a **systematic bias against high-pitched identities** — and because it is systematic, it will look like a consistent, believable finding ("our child voices score worse on naturalness") rather than an artefact. **This is the trap most likely to silently corrupt this project's conclusions.**

**Trap 2 — prosodic blindness.** Same paper: pitch-accent corruption dropped **human** MOS from 4.00 → 3.19 → 2.16 (**−1.84**). All six models moved **< 0.1**. They measure signal cleanliness, not delivery. For dramatic/game content where delivery *is* the product, MOS predictors are close to irrelevant.

**Trap 3 — inverted rate and variability sensitivity.** Humans correlated **r = −0.520** with speaking rate; models ≈ 0. Humans **r = +0.477** with F0 variability (they *like* expressive pitch movement); models ≈ 0. The models are indifferent to exactly the two dimensions this project manipulates on purpose.

**Trap 4 — saturation and OOD failure.** VoiceMOS 2024 found most predictors cannot reliably rank strong systems. The position paper: performance "degrades significantly when applied to out-of-domain data," and predictors "lack uncertainty estimation… only point estimates without confidence intervals." Its recommendation, which you should adopt verbatim as a house rule: **"small differences in predicted MOS should not be interpreted as genuine performance gains."**

**Trap 5 — language coverage.** UTMOS/UTMOSv2 are trained on BVCC/BC-style English and Chinese listening-test data. **Applying them to Hindi, Tamil or Telugu output is UNVERIFIED and probably not defensible.** No MOS predictor validated on Indic TTS was found. For Indic, report WER + TTSDS2 and **omit the MOS column rather than fabricate one.**

#### The ruling

1. **TTSDS2 (`pip install ttsds`) is the primary naturalness metric.** It was the **only** metric of 16 to exceed Spearman 0.50 with human judgement in **every** domain (avg **0.67**), it is MIT, actively maintained, and multilingual across 14 languages. Its factors — Generic (WavLM/HuBERT/wav2vec2 activations), Speaker (d-Vector, WeSpeaker), Prosody (PyWORLD F0, HuBERT + Allosaurus speaking rate), Intelligibility (wav2vec2 + Whisper-small activations) — are combined via **2-Wasserstein distance**, normalised as `S(X) = 100 × W_noise/(W_real + W_noise)`, unweighted mean across factors. Being distributional, **it needs a set, not a clip** — which fits our fixed-set design exactly. Caveat: its multilingual path swaps in **mHuBERT-147, which is CC-BY-NC-SA-4.0**, so the multilingual mode carries a non-commercial dependency the English mode does not.
2. **Distill-MOS as the cheap secondary scalar** (MIT, trivial API).
3. **NISQA for its sub-dimensions**, not its MOS — noisiness/coloration/discontinuity are actionable in a way "3.7" is not.
4. **Never gate on UTMOS/DNSMOS**, and if you report them, report them **stratified by target pitch bin** so the bias is visible rather than baked into an average.
5. **Report a naturalness delta as significant only with a CI**, per the position paper.

---

## 6. The fixed eval set — concrete design spec

**Does something reusable already exist?** Partly, and none of it is a drop-in:

| Candidate | Verdict |
|---|---|
| **InstructTTSEval** (1k EN + 1k ZH × APS/DSD/RP) | **Adopt the *format*, not the content.** MIT-licensed dataset, and the three-tier structure is exactly right. But it is EN/ZH only, sourced from NCSSD + film/TV, and has no Indic and no game-dialogue register. Take ~30 items as an **anchor subset** so you have one number comparable to published work. |
| **ParaSpeechCaps** (59 style tags, 342h human-labelled + 2,427h scaled) | Rich and well-matched in vocabulary, held-out sets exist — but **CC-BY-NC-SA-4.0 ⛔ non-commercial.** Usable to *inform* your tag vocabulary; not usable as a shipped eval asset. |
| **TTSDS2 benchmark set** | Auto-regenerated quarterly from recent YouTube specifically to avoid leakage; 14 languages; 50 speaker-matched pairs per language. **Use as-is for the naturalness axis**, and inherit its anti-leakage discipline. |
| **IndicVoices-R** (1,704h, 10,496 speakers, 22 languages, NeurIPS 2024 D&B) | The right source for **real-speaker Indic calibration audio** (your `C_same`/`C_diff` reference sets), not for descriptions. |
| Seed-TTS-eval | Reference-audio-driven; **structurally inapplicable** — Alaap has no audio prompt. Borrow its WER normalisation code only. |

**Conclusion: build your own, borrow the structure.** No existing set covers description→voice for Indic languages with game-dialogue registers.

### Design spec

**Size: 50 descriptions.** This is the right number, and §7 shows why: it detects a paired effect size of **d ≈ 0.40** at 80% power, which is the granularity at which "S2 beats S0" is a real claim. Going to 100 buys you d ≈ 0.28; going to 25 leaves you at d ≈ 0.57, which is too coarse to see anything but a rewrite.

**Stratification — 50 descriptions as a crossed design, not a wishlist:**

| Facet | Levels | n |
|---|---|---|
| Age | child, young adult, middle-aged, elderly | balanced |
| Gender | male, female, ambiguous/non-binary | balanced |
| Register | game-combat bark, narrative/exposition, intimate/quiet dialogue, villain monologue | balanced |
| Emotion | neutral, angry, fearful, joyful, sorrowful | balanced |
| Stylization | none, raspy, whispered, breathy, nasal, theatrically aged | **~40% of set** |
| Language | English ×30, Hindi ×8, Tamil ×4, Telugu ×4, Bengali ×4 | 50 total |

Plus **5 deliberate adversarial items** inside the 50: mutually-conflicting attributes ("a booming whisper"), extreme-edge attributes ("the highest-pitched voice you can produce"), and near-duplicate pairs that differ in exactly one attribute — the near-duplicate pairs are your sharpest single-attribute-sensitivity probe and cost nothing.

**Every description gets all three InstructTTSEval tiers (APS / DSD / RP)** so adherence degrades measurably with abstraction, exactly as the published reference does (96.2 → 89.4 → 67.2).

**Dialogue script — 20 lines per description, and the length distribution is load-bearing:**

| Bucket | Lines | Words | Why |
|---|---|---|---|
| **Ultra-short barks** | **6** | **1–3** ("Behind you!", "Reload.", "No.") | **The brief is right and this is the most-neglected case.** Short lines have no time to establish prosody, expose vocoder attack artefacts, and are where identity consistency breaks first. Most TTS evals use paragraphs and never see this. |
| Short exchanges | 6 | 4–10 | Standard game dialogue. |
| Medium | 5 | 11–25 | Where most published evals live. |
| Long | 3 | 26–60 | Long-form drift and identity decay. |

= 20 lines. **The same 20 lines for every description** — this is essential. Varying text across descriptions confounds text with identity and destroys the diversity metric.

**A separate 1-line diversity script:** one medium line (~12 words), used for all M×N = 1,000 diversity renders.

**Versioning:**

```
eval-sets/
  v1.0.0/
    descriptions.jsonl      # id, tier{APS,DSD,RP}, target bins, facet labels, lang
    script.jsonl            # 20 lines, fixed ids, length bucket
    diversity_line.txt
    calibration/            # real-speaker refs for C_same / C_diff, per language
    MANIFEST.sha256         # hash of every file
    CHANGELOG.md
```

Rules: **semver, append-only, never edit an item in place.** Patch = typo. Minor = added items (old scores stay valid on the old subset). Major = changed/removed items (**all prior scores are void and must be re-run** — say so loudly in the changelog). Every scorecard row records `eval_set_version` + `MANIFEST.sha256` + the git SHA of the harness. Without this, six months of scores become uncomparable and you are back to guessing, which is the exact failure this whole harness exists to prevent.

**Total per eval pass:** 50 × 20 = 1,000 adherence/consistency renders + 1,000 diversity renders = **2,000 clips, ≈ 2.8 h of audio.**

---

## 7. Statistical protocol

**The pseudo-replication trap, first, because it is the one that will otherwise produce a false win.** You will have 1,000 utterances but only **50 independent units**. The 20 lines within a description are *not* independent samples — they share an identity, a description, and a mapper draw. Computing a t-test over 1,000 utterances inflates N twenty-fold and will manufacture p < 0.001 for noise. **Aggregate to the description level first (50 numbers), then test.** Nothing else in this section matters if this is wrong.

**Detectable effect size at N = 50 paired descriptions** (α = 0.05 two-sided, standard paired-design power):

| Power | Detectable Cohen's d | Interpretation |
|---|---|---|
| 0.80 | **d ≈ 0.40** | medium effect — this is your resolution |
| 0.90 | d ≈ 0.46 | |

| To detect | Descriptions needed |
|---|---|
| d = 0.20 (small) | **~196** |
| d = 0.30 | ~87 |
| **d = 0.40** | **~50** ✓ |
| d = 0.50 | ~32 |

**Read this as a design constraint, not a footnote: at 50 descriptions you can honestly claim medium-or-larger improvements and nothing smaller.** If S2 improves adherence by d = 0.25, your harness cannot tell — and the correct response is to say so, not to slice the data until something is significant.

**The binary-judge resolution problem.** InstructTTSEval-style verdicts are binary. At 50 descriptions × 3 tiers = 150 binary judgements, McNemar's paired test resolves roughly **±10 percentage points**. Detecting a 5 pp change needs ~600 items. **Therefore: the continuous objective re-measurement is the primary gate and the binary judge is a confirmatory secondary.** This inverts the intuitive ordering and it is the single most important statistical consequence of the design.

**Recommended protocol:**

1. **Primary: paired cluster bootstrap over descriptions.** Resample the 50 descriptions **with replacement**, 10,000 iterations; carry all 20 lines of a sampled description together (this is what makes it a *cluster* bootstrap and what respects the dependence). Report the **95% percentile CI of the paired mean difference**. Claim an improvement only when the CI excludes 0. Report the CI, not just the verdict.
2. **Secondary: Wilcoxon signed-rank** on the 50 paired description-level scores. Non-parametric, no normality assumption, ~95% ARE relative to the paired t-test, and the convention used in Blizzard-style TTS comparisons.
3. **Multiplicity: Holm–Bonferroni across the four axes.** Four families (adherence, consistency, diversity, intelligibility). Correct within, and pre-declare which axis is primary — otherwise you are running four lotteries and reporting the winner.
4. **Pre-register the comparison.** Before running S2, write down the axis, the direction, and the threshold. This is a two-line file and it is the difference between measurement and post-hoc storytelling.
5. **Report effect sizes with CIs, always. Never report a bare p-value**, and never report a MOS-predictor delta without a CI (the position paper's explicit instruction).
6. **Seed discipline:** fix and record RNG seeds; run **3 seeds** and report mean ± sd across seeds. **Variance across seeds is itself a headline result** — if seed variance exceeds your S0→S2 delta, you have no result, and finding that out early is cheap.

---

## 8. The S0 baseline scorecard — exact template

Every cell must be filled by the S0 stage. `—` is an acceptable value; a blank is not. Copy this table verbatim into the S0 report.

```
RUN METADATA
  harness_git_sha      : ________            eval_set_version : v1.0.0
  eval_set_sha256      : ________            date_utc         : ________
  backend / version    : ________            gpu / vram       : ________
  seeds                : [__, __, __]        total_render_time: ______ min
  loudness_norm_applied_before_measurement : NO   <- must be NO
```

| # | Metric | Encoder / tool + version | Value (mean ± sd over 3 seeds) | 95% CI | Reference / ceiling | Pass? |
|---|---|---|---|---|---|---|
| **AXIS 1 — ADHERENCE (objective re-measurement)** ||||||
| 1.1 | F0 mean in target bin (% of 50) | parselmouth 0.4.7 | `__._% ± _._` | `[__._, __._]` | 100% | ☐ |
| 1.2 | F0 mean abs error vs bin centre (Hz) | parselmouth | `__._ ± _._` | | — | ☐ |
| 1.3 | Speaking rate in target bin (%) | g2p-en + silero-vad | `__._% ± _._` | | 100% | ☐ |
| 1.4 | Articulation rate (phones/s) | g2p-en + silero-vad | `__._ ± _._` | | real: `__._` | — |
| 1.5 | Volume/intensity in target bin (%) | parselmouth (pre-norm) | `__._%` | | 100% | ☐ |
| 1.6 | Gender agreement (%) | audeering w2v2-24 | `__._%` | | ≥91.1% (model acc) | ☐ |
| 1.7 | Age bin agreement, 4 bins (%) | audeering w2v2-24 | `__._%` | | MAE 7.1–10.8 y | ☐ |
| 1.8 | HNR in target bin, raspy items (%) | parselmouth | `__._%` | | — | ☐ |
| 1.9 | Jitter (local) / Shimmer (local) | parselmouth | `__.__ / __.__` | | **DIAGNOSTIC ONLY — do not gate** | — |
| 1.10 | Formant dispersion / eVTL (cm) | parselmouth | `__._ ± _._` | | **relative use only** | — |
| 1.11 | **Composite adherence score** | mean of 1.1/1.3/1.5/1.6/1.7 | `__._%` | `[__._, __._]` | 100% | ☐ |
| **AXIS 1b — ADHERENCE (LLM judge, optional at S0)** ||||||
| 1.12 | APS consistency (%) | gemini-2.5-pro-prev-05-06 | `__._%` | | ref 96.2 / best-open 63.4 | ☐ |
| 1.13 | DSD consistency (%) | " | `__._%` | | ref 89.4 / best-open 48.7 | ☐ |
| 1.14 | RP consistency (%) | " | `__._%` | | ref 67.2 / best-open 31.4 | ☐ |
| 1.15 | Per-dimension failure histogram (12) | " | `[…]` | | — | — |
| **AXIS 2 — IDENTITY CONSISTENCY** (encoder: WeSpeaker ResNet34-LM, 256-d) ||||||
| 2.1 | `C_same` real speaker, 20 utts | WeSpeaker R34-LM | `0.___` | | **calibration** | — |
| 2.2 | `C_diff` 20 different real speakers | " | `0.___` | | **calibration** | — |
| 2.3 | `C_id` within-session, median over 50 | " | `0.___ ± _.___` | `[_.___, _.___]` | ≥ `C_same` − 0.05 | ☐ |
| 2.4 | Normalized `(C_id−C_diff)/(C_same−C_diff)` | " | `0.___` | | 1.0 | ☐ |
| 2.5 | `C_restart` after process restart | " | `0.___` | | drop < 0.03 | ☐ |
| 2.6 | `C_version` after backend bump | " | `0.___` | | record, don't gate | — |
| 2.7 | Descriptions passing 2.3 (%) | " | `__._%` | | 100% | ☐ |
| 2.8 | Cross-encoder agreement (Spearman) | + NeMo TitaNet-L | `0.___` | | > 0.8 | ☐ |
| 2.9 | SIM-o *(literature comparability only)* | WavLM-large SV | `0.___` | | GT 0.69–0.76 | — |
| **AXIS 3 — DIVERSITY / ANTI-MODE-COLLAPSE** (N=20 samples × 50 descriptions) ||||||
| 3.1 | **nVS — normalized Vendi Score** | vendi-score, cosine, q=1 | `0.___ ± _.___` | `[_.___, _.___]` | **≥ 0.35** | ☐ |
| 3.2 | `nVS_real` (20 real distinct speakers) | " | `0.___` | | **calibration ceiling** | — |
| 3.3 | Mean intra-description cosine | WeSpeaker R34-LM | `0.___` | | **TRIPWIRE: fail if > `C_same`−0.05** | ☐ |
| 3.4 | Descriptions with nVS < 0.20 (alarm) | " | `__ / 50` | | 0 | ☐ |
| 3.5 | Descriptions with nVS < 0.10 (collapsed) | " | `__ / 50` | | **0 — hard fail** | ☐ |
| 3.6 | Silhouette by description ID | sklearn | `0.___` | | ≥ 0.15 | ☐ |
| 3.7 | Between − within mean distance | " | `0.___` | `[_.___, _.___]` | **CI must exclude 0** | ☐ |
| **AXIS 4 — INTELLIGIBILITY & NATURALNESS** ||||||
| 4.1 | WER English (%) | whisper-large-v3 + seed-tts norm | `_.__% ± _.__` | `[_.__, _.__]` | GT 2.06–2.23% | ☐ |
| 4.2 | WER by length bucket (1–3 / 4–10 / 11–25 / 26–60 w) | " | `_.__ / _.__ / _.__ / _.__` | | **short bucket is the risk** | ☐ |
| 4.3 | WER Hindi / Tamil / Telugu / Bengali (%) | indic-conformer-600m | `__._ / __._ / __._ / __._` | | **own real-speech floor** | ☐ |
| 4.4 | Real-speech WER floor per Indic language | " | `__._ …` | | **calibration — measure first** | — |
| 4.5 | **TTSDS2 overall (0–100)** | ttsds 2.1.3 | `__._ ± _._` | `[__._, __._]` | real ≈ 100 | ☐ |
| 4.6 | TTSDS2 per factor (Generic/Speaker/Prosody/Intelligibility) | " | `__._ / __._ / __._ / __._` | | | — |
| 4.7 | Distill-MOS (1–5) | distillmos 0.9.1 | `_.__ ± _.__` | `[_.__, _.__]` | — | — |
| 4.8 | **Distill-MOS stratified by target pitch bin** | " | `low __._ / mid __._ / high __._` | | **bias check — expect artificial high-pitch penalty** | — |
| 4.9 | NISQA sub-dims (noise/color/discont/loud) | nisqa 2.0 | `_._ / _._ / _._ / _._` | | | — |
| 4.10 | UTMOSv2 *(reported, never gated)* | UTMOSv2 | `_.__` | | **pitch-biased r=−0.72** | — |
| **AXIS 5 — RUN HEALTH** ||||||
| 5.1 | Renders attempted / succeeded | — | `____ / ____` | | 100% | ☐ |
| 5.2 | Seed-to-seed sd of composite adherence | — | `_.___` | | **must be < S0→S2 delta** | ☐ |
| 5.3 | Wall-clock, eval only (excl. TTS render) | — | `__ min` | | < 60 min | ☐ |

---

## 9. What this means for the build

**Install (one environment, CPU+GPU, ~12GB VRAM peak):**

```bash
# core
pip install torch torchaudio transformers jiwer numpy scipy scikit-learn pandas

# adherence — acoustic re-measurement
pip install praat-parselmouth==0.4.7      # GPLv3 — dev-tool only, see licence note
pip install torchcrepe penn pyworld        # MIT/MIT/permissive — F0 cross-checks
pip install g2p-en silero-vad              # Apache-2.0 / MIT — speaking rate
pip install pyloudnorm                     # loudness, measured PRE-normalisation

# identity + diversity
pip install pyannote.audio                 # loads WeSpeaker ResNet34-LM
pip install vendi-score prdc               # MIT / MIT

# intelligibility + naturalness
pip install ttsds==2.1.3                   # MIT — TTSDS2, the primary naturalness metric
pip install distillmos nisqa               # MIT / MIT

# adherence — LLM judge (optional; API, no GPU)
pip install google-genai requests tqdm

# system deps required by ttsds
sudo apt-get install ffmpeg automake autoconf unzip sox gfortran subversion libtool
# ttsds pins numpy<2 — install into its OWN venv to avoid dependency hell
```

**Checkpoints to pin, hash, and vendor locally** (never let the harness hot-download — a rotated checkpoint silently rewrites your history):

| Purpose | Checkpoint | Licence | ~Size |
|---|---|---|---|
| Identity scoring (independent) | `pyannote/wespeaker-voxceleb-resnet34-LM` | CC-BY-4.0 | ~26 MB |
| Cross-check encoder | `nvidia/speakerverification_en_titanet_large` | CC-BY-4.0 | ~100 MB |
| English ASR | `openai/whisper-large-v3` | Apache-2.0 | ~3.1 GB |
| Fast ASR loop | `openai/whisper-large-v3-turbo` | MIT | ~1.6 GB |
| Indic ASR | `ai4bharat/indic-conformer-600m-multilingual` | MIT (gated: auto) | ~2.4 GB |
| Age/gender | `audeering/wav2vec2-large-robust-24-ft-age-gender` | **CC-BY-NC-SA-4.0 ⚠️** | ~1.2 GB |
| SIM-o (comparability only) | `wavlm_large_finetune.pth` (UniSpeech) | **CC-BY-SA-3.0 ⚠️**, Google-Drive-hosted | ~1.2 GB |
| Naturalness | ttsds internal (WavLM/HuBERT/wav2vec2/Whisper-small/WeSpeaker/d-Vector) | mixed; **mHuBERT-147 is CC-BY-NC-SA-4.0** in multilingual mode | ~5 GB |

**VRAM.** Nothing runs concurrently; **stage the pipeline and unload between stages.** Peak is Whisper-large-v3 at ~10 GB fp16 — fits 12 GB with room, does **not** fit alongside anything else. Run order: render → (unload TTS) → ASR → (unload) → embeddings + TTSDS2 → MOS → CPU acoustics. On an 8 GB card, substitute `whisper-large-v3-turbo` (~1.6 GB) and accept a small WER offset — but then **never mix turbo and large-v3 numbers in the same table.**

**Runtime per eval pass** (2,000 clips ≈ 2.8 h audio, single consumer GPU) — estimates, **[LOW confidence, measure and replace these]**:

| Stage | Est. | Device |
|---|---|---|
| TTS render (not eval) | *backend-dependent* | GPU |
| Whisper-large-v3 WER | 15–25 min | GPU |
| IndicConformer (Indic subset) | 3–5 min | GPU |
| Speaker embeddings (2,000) | 2–3 min | GPU |
| TTSDS2 full suite | 20–30 min | GPU |
| Distill-MOS / NISQA | 3–5 min | GPU |
| parselmouth + rate + VAD | 5–10 min | CPU (parallelise) |
| Vendi / silhouette / bootstrap | < 1 min | CPU |
| **Total eval, excl. render** | **≈ 50–75 min** | |
| LLM judge (150 calls, if enabled) | +10–15 min | API |

Cache aggressively by `(clip_sha256, metric, model_version)` — reruns after a metric change should recompute only that metric.

**Build order (this is the dependency chain, not a wishlist):**

1. **Eval set v1.0.0 + manifest** — nothing is measurable until this is frozen.
2. **Calibration harness** (`C_same`, `C_diff`, `nVS_real`, per-language real-speech WER floors) — **before any TTS scoring.** Every threshold in this document depends on it, and it is the step most likely to be skipped and most expensive to skip.
3. **Axis 1 objective re-measurement** — cheapest, most powerful, CPU-only.
4. **Axis 2 + 3** — they share the embedding pass; build together.
5. **Axis 4** — WER + TTSDS2.
6. **Scorecard emitter + cluster bootstrap.**
7. *Only then* the LLM judge.

**The licence position, stated plainly.** For an internal measurement harness that never ships, GPL (parselmouth, phonemizer), CC-BY-SA (UniSpeech), and CC-BY-NC (audeering, mHuBERT-147, MMS, Seamless, ParaSpeechCaps) are all usable — none of them touch the product. **But the S0 report must carry a licence column so nobody later promotes a measurement dependency into the runtime.** The permissive-only subset (Whisper Apache-2.0, IndicConformer MIT, WeSpeaker Apache-2.0/CC-BY-4.0, ttsds MIT, distillmos MIT, vendi-score MIT, g2p-en Apache-2.0, silero-vad MIT) already covers **every axis except age/gender classification** — which is the one genuine gap, and the linear-probe swap-in in §2.1 closes it in about a day.

---

## 10. Open — must be settled by experiment

| Question | Cheapest experiment | Est. cost/time | What it blocks |
|---|---|---|---|
| Do parselmouth jitter/shimmer/HNR mean anything on **neural-vocoder** output? | Synthesize 20 clips deliberately spanning clean→raspy with your S0 backend; measure all three; rank by ear (n=3 listeners, forced-choice). Correlate. | **2 h** | Whether "raspy/gravelly" is measurable at all — a whole stylization family in Axis 1 |
| Does the audeering age/gender model transfer to **synthetic** speech? | Run it over 50 synthetic clips with known intended age/gender bins + 50 real clips as control. Compare confusion matrices. | **1 h** | Cells 1.6/1.7 — two of five composite-adherence components |
| Do ResNet34-LM and TitaNet-L **agree** on identity rankings? | Embed the same 1,000 clips with both; Spearman-correlate per-description consistency scores. | **1 h** | Whether Axis 2/3 results are encoder-independent (cell 2.8) — and it answers a question the literature has not |
| What is `nVS_real` on **our** encoder and script? | Embed 20 real distinct speakers on the fixed line; compute Vendi. | **30 min** | The **entire Axis 3 target number**. Do this first. |
| Is the **0.35 nVS target** actually right? | After the above, sample from a deliberately-collapsed toy mapper (constant output + noise) and a deliberately-diverse one (random real speakers). Confirm the target separates them. | **2 h** | Whether Axis 3 has any discriminative power at all |
| Can an **open audio-LLM** (Qwen3-Omni / Kimi-Audio) replace the Gemini judge? | 100 items scored by both; Cohen's κ. Accept only if κ > 0.6. | **4 h + API** | Whether the LLM-judge route is permissively licensed and locally runnable at all |
| Real-speech **WER floor per Indic language** with IndicConformer | Run over 100 clean IndicVoices-R held-out clips per language. | **2 h** | Cell 4.3 — without it Indic WER is uninterpretable |
| Does **short-line** (1–3 word) synthesis break identity consistency? | Compute `C_id` separately per length bucket. Free — the data already exists in the standard run. | **0** (falls out of the main run) | Whether the game-dialogue use case works at all — the highest-value free result here |
| Do MOS predictors show the **pitch bias** on *our* outputs? | Distill-MOS on 50 clips stratified by target pitch bin; regress score on measured F0. | **1 h** | Cell 4.8; confirms/refutes the §5.2 trap locally |
| Does **TTSDS2 saturate** on our S0 output? | Run it; check whether factor scores land near 100 or spread. | **30 min** | Whether Axis 4 can rank checkpoints or only detect disasters |
| **Seed variance vs expected effect size** | 3 seeds, same config; sd of composite adherence. | **1 eval pass** | Whether *any* S0→S2 claim is possible — check before trusting a single comparison |
| Do **VoiceMOS 2026** results supersede this stack? | Re-check the challenge site after 2026-09-16 (papers due). | **30 min, in ~4 weeks** | Possible free upgrade to §5.2 |

---

## 11. Sources

| # | URL | Type | Used for | Confidence in source |
|---|---|---|---|---|
| 1 | https://arxiv.org/abs/2506.16381 | Paper (arXiv abs) | InstructTTSEval existence, title, authors, abstract, 2025-06-19 date | HIGH |
| 2 | https://arxiv.org/html/2506.16381v1 | Paper (HTML) | 12 attributes grouped in 4 tiers; judge model; results tables; dataset sizes | MEDIUM (table extraction) |
| 3 | https://github.com/KexinHUANG19/InstructTTSEval | Repo | Repo structure, eval/ directory | HIGH |
| 4 | https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/eval_prompt.txt | Repo file (verbatim) | **The exact 12 dimensions and full judge prompt** | HIGH |
| 5 | https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/gemini_eval.py | Repo code | Judge model id, temperature=0, binary `一致性` parsing, upload flow, retries | HIGH |
| 6 | https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/README.md | Repo file | Data format, output statistics format, EN/ZH share one prompt | HIGH |
| 7 | https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/example_en.jsonl | Repo data | Verbatim APS/DSD/RP examples | HIGH |
| 8 | https://raw.githubusercontent.com/KexinHUANG19/InstructTTSEval/main/eval/requirements.txt | Repo file | Deps = google-genai/requests/tqdm → no local GPU need, no local judge | HIGH |
| 9 | https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval | Model/dataset card | MIT licence, 2k samples, 634 MB, 16 kHz | HIGH |
| 10 | https://arxiv.org/html/2606.19951v1 | Paper (HTML) | **Pitch bias r=−0.788/−0.722 vs human −0.059; prosody blindness Δ<0.1 vs human −1.84** | HIGH |
| 11 | https://arxiv.org/html/2510.06927v1 | Position paper (HTML) | WER saturation 1.61→1.47; SIM-o convention gap 0.754 vs 0.905; DNSMOS domain mismatch; uncertainty recommendation | HIGH |
| 12 | https://arxiv.org/html/2506.19441v1 | Paper (HTML) | TTSDS2 factors, 2-Wasserstein formula, 0.67 avg Spearman, 14 languages, 16 metrics | HIGH |
| 13 | https://github.com/ttsds/ttsds + pyproject.toml | Repo + build file | **MIT licence**, v2.1.3, deps, pushed 2026-07-07 | HIGH |
| 14 | https://pypi.org/pypi/ttsds/json | Package index | Version 2.1.3, Python support | HIGH |
| 15 | https://arxiv.org/html/2410.06885v3 | Paper (HTML) | **Ground-truth SIM-o 0.69/0.73/0.76 and WER 2.23/2.06/1.26; F5-TTS/E2-TTS numbers** | HIGH |
| 16 | https://github.com/BytedanceSpeech/seed-tts-eval (README, cal_sim.sh, cal_wer.sh, run_wer.py) | Repo code | De-facto standard: whisper-large-v3 + Paraformer-zh; WavLM-large SV via UniSpeech; exact text normalisation | HIGH |
| 17 | https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/examples/voxceleb/v2/README.md | Repo file | **VoxCeleb1-O EERs: ResNet34-LM 0.723, ResNet221-LM 0.505, ResNet293-LM 0.425, CAM++ 0.803, ECAPA1024-LM 0.728** | HIGH |
| 18 | https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/docs/pretrained.md | Repo file | **Pretrained weights follow VoxCeleb licence = CC-BY-4.0**; model zoo | HIGH |
| 19 | https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/LICENSE | LICENSE file | WeSpeaker code Apache-2.0 | HIGH |
| 20 | https://raw.githubusercontent.com/microsoft/UniSpeech/main/LICENSE | LICENSE file | **UniSpeech = CC BY-SA 3.0** (hosts the SIM-o checkpoint) | HIGH |
| 21 | https://raw.githubusercontent.com/speechbrain/speechbrain/develop/speechbrain/inference/speaker.py | Repo code | **Default same-speaker cosine threshold = 0.25** | HIGH |
| 22 | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb | Model card | Apache-2.0, EER 0.80, VoxCeleb1+2 | HIGH |
| 23 | https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM | Model card | CC-BY-4.0, pyannote.audio loading snippet | HIGH |
| 24 | https://huggingface.co/api/models/{...} (audeering, whisper, mms, seamless, ai4bharat, pyannote, nvidia, mHuBERT-147) | API (licence field) | **Verified licences**: audeering CC-BY-NC-SA-4.0; whisper-large-v3 Apache-2.0; v3-turbo MIT; MMS CC-BY-NC-4.0; Seamless CC-BY-NC-4.0; indic-conformer MIT; titanet CC-BY-4.0; mHuBERT-147 CC-BY-NC-SA-4.0 | HIGH |
| 25 | https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual | Model card | MIT, 22 languages, CTC+RNNT, **Hindi WER 13.2 on Vaani** | HIGH |
| 26 | https://arxiv.org/html/2511.07135 | Paper (HTML) | **Pairwise diversity: real 0.67±0.18 vs generated 0.71–0.74; Stability 0.85–0.90; wavlm-base-plus-sv** | HIGH |
| 27 | https://arxiv.org/abs/2210.02410 | Paper (arXiv) | Vendi Score definition (exp Shannon entropy of similarity-matrix eigenvalues), reference-free, GAN mode-collapse use | HIGH |
| 28 | https://pypi.org/pypi/vendi-score/json | Package index | MIT, v0.0.3 | HIGH |
| 29 | https://sites.google.com/view/voicemos-challenge (+ /voicemos-challenge-2026) | Official challenge site | **Editions 2022/2023/2024, AudioMOS 2025, VMC2026 results to participants 2026-08-31, not public**; 2026 tracks | HIGH |
| 30 | https://github.com/sarulab-speech/UTMOSv2 · /UTMOS22 | Repos (API) | Both MIT; UTMOSv2 pushed 2026-04-02, UTMOS22 stale 2024-04 | HIGH |
| 31 | https://github.com/microsoft/Distill-MOS (LICENSE + README) | Repo | **MIT verified in LICENSE file**; usage; 16 kHz, 1–5 output | HIGH |
| 32 | https://pypi.org/pypi/{distillmos,nisqa,praat-parselmouth,torchcrepe,penn,g2p-en,silero-vad,phonemizer,prdc,...}/json | Package index | **Licence verification**: distillmos MIT, nisqa MIT, praat-parselmouth **GPLv3**, torchcrepe MIT, penn MIT, g2p-en Apache-2.0, silero-vad MIT, phonemizer **GPLv3**, prdc MIT | HIGH |
| 33 | https://arxiv.org/abs/2301.12258 (PENN) | Paper | PENN 97.0% RPA; **11.2× realtime CPU / 408× GPU** | MEDIUM |
| 34 | https://arxiv.org/abs/2210.13248 · https://github.com/marianne-m/brouhaha-vad | Paper + repo LICENSE | Brouhaha joint VAD+SNR+C50; **MIT (CNRS)** | HIGH |
| 35 | https://docs.pytorch.org/audio/stable/tutorials/squim_tutorial.html | Official docs | SQUIM Objective vs Subjective; **SquimSubjective needs non-matching reference (NORESQA-MOS)** | HIGH |
| 36 | https://link.springer.com/article/10.3758/BRM.41.2.385 | Paper (de Jong & Wempe 2009) | Transcript-free syllable-nuclei speech rate; validated vs human counts | HIGH |
| 37 | https://link.springer.com/article/10.3758/s13428-023-02288-x · https://www.sciencedirect.com/science/article/pii/S0095447023000797 | Papers | **eVTL–height correlation only r ≈ .25–.32 in adults**; VTL 13–20 cm adult range | MEDIUM |
| 38 | https://www.sciencedirect.com/science/article/pii/S2212017313002788 | Paper | Jitter/shimmer/HNR validation; Praat algorithm verified on **synthesized steady vowels**; HNR most predictive, jitter/shimmer contribute minimally | MEDIUM |
| 39 | https://arxiv.org/abs/2503.04713 · https://github.com/ajd12342/paraspeechcaps | Paper + repo | ParaSpeechCaps 59 tags, 342h+2427h, **CC-BY-NC-SA-4.0** | HIGH |
| 40 | https://arxiv.org/html/2409.05356v2 · https://github.com/AI4Bharat/IndicVoices-R | Paper + repo | IndicVoices-R 1,704h / 10,496 speakers / 22 languages; IV-R benchmark uses S-SIM | HIGH |
| 41 | https://arxiv.org/abs/2111.05095 | Paper (abs) | TacoSpawn speaker generation; objective metrics claimed to correlate with human similarity perception — **exact formulations UNVERIFIED (PDF unextractable)** | LOW |
| 42 | https://arxiv.org/pdf/2509.17765 | Paper | Qwen3-Omni open-weight omni model, paralinguistic capability | MEDIUM |
| 43 | https://arxiv.org/pdf/2603.24430 | Paper | Objective TTS metrics saturate; I2D raises system-level correlation 0.118 → 0.464 with UTMOSv2 | MEDIUM |
| 44 | https://ieeexplore.ieee.org/document/11119540/ | Journal (IEEE Access) | Three years of VoiceMOS: SSL encoders most reliable; data diversity > architecture; ensembling diminishing returns | MEDIUM |
| 45 | https://arxiv.org/pdf/2303.00332 | Paper | CAM++ VoxCeleb-O EER 0.654 w/ AS-Norm; 51% fewer params than ECAPA | MEDIUM |

### Explicitly UNVERIFIED

- **InstructTTSEval leaderboard** — none found. The paper's tables are the only ranking.
- **TacoSpawn's exact diversity/novelty formulations** — PDF not machine-extractable in this pass.
- **VoiceMOS 2026 results** — released to participants 2026-08-31, not public.
- **Cosine value at the EER operating point** for WeSpeaker/ECAPA on VoxCeleb — EER *percentages* are published; the *decision thresholds* are not. This is why §3.2 calibrates in-run instead of quoting one.
- **Whether any MOS predictor is valid on Indic languages** — no validation found; assume not.
- **audeering age/gender transfer to synthetic speech** — trained on real speech only.
- **NeMo TitaNet-L EER (~0.66%)** and **Resemblyzer EER (~5–6%)** — vendor/folklore figures, not re-verified against a primary table.
- **Official AI4Bharat IndicWhisper checkpoint** — does not resolve on HF; only third-party re-uploads.
- **Runtime estimates in §9** — engineering estimates, not measured. Replace with real numbers after the first pass.
