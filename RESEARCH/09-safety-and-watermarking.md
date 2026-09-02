# 09 — Safety, Watermarking, Provenance & Bias

> **Domain:** watermarking, provenance, synthetic-media regulation, description-layer gating, fairness
> **Answers:** E2; scope section 15
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [05-datasets-and-annotation.md](05-datasets-and-annotation.md) · [07-serving-and-cost.md](07-serving-and-cost.md) · [08-licensing-propagation.md](08-licensing-propagation.md) · [11-production-api-landscape.md](11-production-api-landscape.md)

> ⚠️ **NOT LEGAL ADVICE.** Regulatory sections are engineering research against primary sources.

---

## 0. Bottom line

- **AudioSeal is the answer to E2, and its licence is now unambiguous: MIT code AND MIT weights**, verified from the LICENSE file, the README changelog (*"We have updated our license to full MIT license (including the license for the model weights)! Now you can use AudioSeal in commercial application too!"*), the official docs, and the HF model card — with a clean dependency tree (`numpy, omegaconf, torch, einops`) and no CC-BY-NC upstream. Actively maintained (v0.2.0, Dec 2025). It is also a **registered C2PA soft-binding algorithm** (`com.aiwatermark.audioseal.1`, added 2026-03-08), which makes it the standards-endorsed choice, not merely a defensible one. · **HIGH** · §2.1.1, §3.3

- 🔴 **The short-clip question is UNVERIFIED for AudioSeal and nobody has published the number.** Every evaluation is on 10 s (paper detection), 5 s (paper attribution, AudioMarkBench) or ~3 s (RAW-Bench). The paper's own §5.3 prose ("IoU 0.99 at one second") **contradicts its own Table 6 (IoU 0.802)** — and localisation is the wrong quantity for us anyway. The best analogical evidence (XAttnMark's duration ablation) says **detection holds at 98.6–99.3 % from 1–10 s while attribution falls to 81.2 % at 1 s.** **Presence is probably fine; payload is not. Test it (X1).** · §2.y

- 🔴 **MP3 is a solved problem (1.00 at 32 kbps) — the brief is worrying about the wrong attack.** The real failures are **Opus** (AudioMarkBench: removes the watermark at ViSQOL ≥ 3), **neural codecs** (Descript Audio Codec → full-message 0.00), **real reverb** (message 0.22 strict), **aggressive high-pass** (0.61), and — most cheaply exploitable — **polarity inversion, an inaudible one-line operation that drives bitwise accuracy to 0.18 and message accuracy to 0.00.** Good news for games: **OGG/Vorbis survives (0.95)**, which is the Unity/FMOD/Wwise default path. · **HIGH** · §2.1.2–2.1.3

- **Treat the watermark as a presence bit, never as an identity.** Attribution averages **0.39** under edits and only **0.69 on unmodified audio** (independent, ICML 2025), corroborated by AudioSeal's own Table 4. Identity belongs in the provenance log and the export manifest. · **HIGH** · §2.1.6

- ❌ **"No impersonation vector at all" does not survive scrutiny. Restate it as "no *cloning* vector — we never accept reference audio."** No documented incident exists, but the mechanism is published: ParaSpeechCaps trains description→voice on **594 named celebrities** using GPT-4 as a name→voice-attribute oracle; Google's own novel-voice paper defines success as **g2s = s2s** (a generated voice sits as close to its nearest real speaker as two real speakers do); generated-not-cloned master voices already match **69 % of women at FAR 1 %**; and *Midler*/*Waits* make **imitation without copying actionable** — copyright never enters it. **The coincidental-collision path survives perfect description-layer moderation.** · **HIGH** · §5.2

- **Two abuse vectors dwarf the one the brief budgets for, and neither is in the scope document: voice-biometric defeat (master voices) and industrial-scale fraud/vishing** — the latter with a **$7 M** FCC enforcement precedent from a single incident, and **no permissively-licensed vishing classifier in existence.** · **HIGH** · §7

- **EU AI Act Art. 50(2) applies from 2 August 2026 and the Digital Omnibus did NOT delay it. We get no grace period** (Art. 111(4) relief covers only systems on the market before that date). The **Code of Practice on Transparency (final 10 June 2026) requires TWO machine-readable marking layers for audio — signed metadata *and* an imperceptible watermark** — plus **a free detection solution** for regulators, media and researchers. Metadata alone is explicitly insufficient; so is logging alone. Penalty cap for an SME is **3 % of turnover**. · **HIGH** · §4.1

- **India is the most prescriptive regime and it binds generation tools directly** (IT Rules r.3(3), notified 10 Feb 2026 — **not** SSMI-limited). It mandates automated prohibited-use blocking, permanent non-removable metadata with a unique identifier, and — the collision with our product — **"a prominently prefixed audio disclosure" for audio content.** *(The widely-repeated "10 % of audio duration" requirement was **dropped** from the final text.)* Whether it reaches a *novel* voice is genuinely arguable and is the highest-value counsel question here. · **HIGH** text / **MEDIUM** application · §4.3

- **Skip C2PA manifests for v1.** The spec is ready (2.4, audio is first-class, `c2pa.ai-disclosure` exists) but the implementation is not (no Ogg/Opus, soft binding unimplemented, `ai-disclosure` absent from the SDK), **no storefront reads it**, and the game-audio pipeline destroys embedded manifests before a player hears anything. **Steam's requirement is a text box a human types into.** Ship watermark + signed log now; add manifests to *web downloads* in ~2 quarters. · **HIGH** · §3.9

- **The brief's bias prescription ("rebalance by sampling or reweighting") is under-powered and probably aimed at the wrong component.** Because the TTS is **frozen**, an "old man" failure could be conditioning failure (rebalancing helps) or **decoder failure (rebalancing cannot help at all)** — a distinction the brief does not draw, and one a few hours of experiment would settle. Worse, **the 62.6 % / 76.1 % skew figures are Qwen2-Audio-7B label distributions, not measured demographics.** · **HIGH** · §6.1, §6.4

- 🔴 **Three independent literatures all disadvantage female voices**: watermark removal is significantly easier on female-attributed audio (**p ≈ 2.4×10⁻⁶**, all methods), master-voice attacks match **69 % F vs 38 % M**, and speaker-verification cost is **2.58×** worse for Indian females. **Our corpus is 62.6 % female.** Calibrate the watermark threshold and the ASV screen **per gender**, not globally. · **HIGH** · §6.1

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| **C1** | §15.1: *"the core product has **no impersonation vector at all**"* | ❌ **Overstated** | True claim is **"no *cloning* vector"**. Description→identity is a *trained objective* in the open data (ParaSpeechCaps: 594 named celebrities, GPT-4 as name→voice oracle); novel-voice generation targets **g2s = s2s** by design; generated master voices match **69 % of women at FAR 1 %**; *Midler* makes imitation actionable **without copying**. The architecture defends against the wrong tort. | arXiv:2503.04713 §3.1–3.2 · arXiv:2111.05095 Table 2 · arXiv:2204.11304 · 849 F.2d 460 | **HIGH** |
| **C2** | §15.2: *"Named-real-person descriptions… **the only realistic impersonation route left**"* | ❌ **Wrong** | **Two routes survive a perfect name gate.** (a) **Periphrasis** — implicit-reference attacks reach **> 90 % ASR** with inverse scaling and defeat SmoothLLM / PerplexityFilter / Erase-and-Check. (b) **Coincidental collision** — needs no attacker at all. The gazetteer is the weakest of the three controls, not the sufficient one. | arXiv:2410.03857 · §5.2 | **HIGH** |
| **C3** | Q-E2: *"whether embedding survives **MP3 transcode** and game-engine audio processing"* | ⚠️ **Wrong attack** | **MP3 is solved: 1.00 at 32 kbps** (and ~0.9 at 16 kbps). The real failures are **Opus**, **neural codecs (DAC → 0.00)**, **real reverb (0.22)**, **high-pass (0.61)** and **polarity inversion (0.18 / 0.00)** — an inaudible one-line strip. **OGG/Vorbis, the actual game default, survives at 0.95.** | ICML 2024 Table 3 + App D.2 · Interspeech 2025 (RAW-Bench) Table 5 · NeurIPS 2024 D&B | **HIGH** |
| **C4** | §15.2: *"embed an inaudible watermark in all output (**AudioSeal, Perth, or equivalent**)"* | ⚠️ **Only one of these works** | **AudioSeal: yes.** **Perth: presence-only, and the open-source version is the weak one** (vendor's own benchmark: pitch-shift **10 %**; the fixed PerTh V2 is **commercial and closed**). **Timbre is GPL-3.0** → disqualified. **SynthID audio is unobtainable** — Google does not even watermark its own Chirp TTS. **WavMark has a hard ≥1 s floor** and 0.38× real-time CPU detection. The "or equivalent" set is nearly empty. | LICENSE files fetched · resemble.ai/benchmarks · ai.google.dev/responsible/docs/safeguards/synthid | **HIGH** |
| **C5** | §15.2: watermarking as a **single** measure, *"Add at S9"* | ⚠️ **Insufficient** | The EU Code of Practice (final 10 Jun 2026) requires **at least two layers of machine-readable marking for audio** — signed metadata **and** an imperceptible watermark — and states *"relying on fingerprinting or logging alone is not considered sufficient."* It also requires **a free detection solution** for regulators, media, researchers and civil society. **S9 needs three deliverables, not one.** | EC CoP PDF, Measure 1.1 + Commitment 2 | **HIGH** |
| **C6** | §8.3 / §15.2: *"**female 62.6 %, twenties 76.1 %**"* | ⚠️ **Not measurements** | These are **Qwen2-Audio-7B-Instruct label distributions**, not annotated demographics, and audio-LLM age estimation regresses toward the mode — which would inflate an apparent twenties concentration. Publishing them as demographics is a Datasheets documentation defect. **Audit first (X6, ~2 hours).** | VoicePersona dataset card | **HIGH** |
| **C7** | §15.2 / §8.3: *"**Rebalance** by sampling or reweighting"* as the bias mitigation | ⚠️ **Under-powered and mis-targeted** | Because the TTS is **frozen**, failure could be **(A) conditioning** (rebalancing helps) or **(B) decoder** (rebalancing cannot help *at all*). Nothing in the literature distinguishes these for a frozen decoder. Balancing demonstrably fixes *calibration* disparities (FDR 0.989) but **fails where a group is absent rather than rare.** Present rebalancing as one of three, behind targeted data acquisition and honest reporting. **Run X4 before funding any rebalancing.** | arXiv:2204.12649 · FAccT 2022 · arXiv:2307.02009 | **HIGH** |
| **C8** | §16 risk table: *"Voice-cloning misuse"* as the sole abuse risk | ❌ **Incomplete** | Two vectors with larger expected harm are absent: **voice-biometric defeat** (our product is a ready-made master-voice generator; 69 % F / 38 % M at FAR 1 %) and **fraud/vishing at scale** (**$6 M** FCC forfeiture + **$1 M** settlement from one incident; **no permissively-licensed vishing classifier exists**). **Child-voice generation** is also unaddressed and is a legal duty in India. | arXiv:2204.11304 · FCC DOC-402762A1 · §7 | **HIGH** |
| **C9** | §15.2: *"Disclosure — synthetic audio labelled as such in the UI and in the export manifest"* | ⚠️ **Necessary, not sufficient** | The binding disclosure lands on our **users**: Steam's Content Survey names *"sound"* explicitly, and itch.io **delists** untagged asset pages. **We must ship them five artefacts** (§3.8), the highest-value being the **runtime-vs-build-time warning** — runtime generation additionally triggers Microsoft Store 11.16, Google Play, Apple 5.1.2(i) and Steam's Live-Generated guardrails question. India may additionally require **a prefixed audible disclosure**. | partner.steamgames.com contentsurvey · itch.io quality-guidelines · MeitY IT Rules r.3(3) | **HIGH** |
| **C10** | §15 has **no dates and no jurisdictions** | ⚠️ **Gap** | **EU Art. 50(2) applies 2 August 2026**; the Digital Omnibus (Reg. 2026/1744) postponed high-risk rules but **not** Art. 50; **we get no grace period.** SME fine cap **3 % of turnover**. India's IT Rules amendment was notified **10 Feb 2026**. California SB 942 is operative **2 Aug 2026** above **1 M monthly users**. **NO FAKES is not law** (S.4591, Senate Calendar No. 446). **Colorado's AI Act never took effect** — repealed and replaced by SB 26-189. | EC Guidelines ¶2 · EC omnibus news · MeitY PDF · CA Leg Counsel · govinfo BILLSTATUS | **HIGH** |
| **C11** | §15.1: *"enforced at the API boundary"* (the Tier-2 seed tension) | ✅ **Right — and needs a second invariant** | The §5.2 ASV public-figure screen introduces a **second** system-held audio path. Both must be enforced identically and **structurally** (§8.2 Invariants 1–2), per the brief's own §12.5 "in code, not in a README" principle. | — | **HIGH** |
| **C12** | §12.2: *free mint / metered render* | ⚠️ **Backwards for one threat** | Every `mint()` returns a (description, voice) pair — **exactly the distillation training set for a clone of our mapper**, the one artefact that is genuinely ours. Making the extraction-relevant call the free one inverts the defence. Quota mint per account. | §7 B2 | **MEDIUM** |
| **C13** | `PHASE-10`: open dataset + model card + writeup as a goal | ⚠️ **Undecided decision** | **If mapper weights ship, the watermark is voided for every downstream user**, and Art. 50(2) provider duties transfer to whoever deploys them. Releasing inference code with watermarking baked in is weak — the code is trivially editable. **Frame this as a decision at S10, not a default.** | AI Act Art. 2 · §7 B1 | **HIGH** |
| **C14** | *(Premise the project may be carrying)* Hume AI as the "declines cloning on principle" exemplar | ❌ **Stale** | **Hume reversed.** OCTAVE (Jan 2025) *"can generate a voice and personality from prompts or recordings as brief as 5 seconds."* Their Dec 2024 framing was product-quality, not safety. **We cannot cite anyone as having concluded description-only is the safe path — keeping the no-upload rule makes us more restrictive than every major vendor, and we own that argument alone.** | hume.ai/blog/how-to-clone-your-voice-with-ai | **HIGH** |

---

## 2. E2 — watermarking comparison

### 2.0 The comparison table

Ratings are **for our workload**: 1–3 s English/Indic speech dialogue lines, rendered by a public commercial service, shipped into game engines.

| Tool | Code lic. | Weights lic. | Payload | MP3/Opus? | Game-engine DSP? | 1–3 s clips? | Perceptual cost | FPR/FNR | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **AudioSeal** (Meta) | **MIT** (verified) | **MIT** (verified, explicit) | 16-bit optional; presence bit is the reliable channel | MP3 32 kbps **1.00**; OGG/Vorbis **✓**; **Opus → high FNR (fails)** | Echo ✓ 1.00; real reverb: bitwise .87, message .22; lowpass ✓; **highpass 1.5 kHz 0.61 ✗**; **polarity inversion .18 ✗**; mix w/ music 0.9787 | Presence: **UNVERIFIED below 5 s** (theory + analogue say fine); payload: **degrades sharply** | PESQ 4.470, ViSQOL 4.829, STOI 0.997, SI-SNR 26.0, MUSHRA 77.07 vs GT 80.49 | AudioSeal τ=0.15 → FPR<0.01, FNR<0.01 (10 s clips) | ✅ **ADOPT** |
| Perth (Resemble) | *see §2.2* | *see §2.2* | — | — | — | — | — | — | see §2.2 |
| WavMark (Microsoft) | *see §2.3* | *see §2.3* | 16/32-bit, 1 s patches + sliding search | MP3 0.99 AUC; **Gaussian noise & MP3 remove it** (AudioMarkBench) | Echo 0.93; **lowpass 0.50**; EnCodec 0.51 | 1 s patch granularity; brute-force sync search | PESQ 4.302, SI-SNR 38.25 | Least robust of the four benchmarked | ❌ Reject |
| SilentCipher (Sony) | *see §2.4* | *see §2.4* | 23.8 bits @ 5.33 bps | MP3/OGG/AAC ✓; **DAC/EnCodec ✗** | RV msg .45–.67; **PI .00**, **PS .00** | ~4.5 s at nominal capacity | Best clean quality: SI-SNR 49.13, MCD 0.25, MOS-LQO 4.98 | 0.999/0.993 clean acc | ⚠️ Second choice |
| Timbre WM (NDSS'24) | *see §2.5* | *see §2.5* | 30 bits @ 5.00 bps | MP3 ✓ / OGG ✓ / AA ✓ | Highest overall robustness in RAW-Bench | ~6 s at nominal capacity | Worst quality: MCD 1.74, MOS-LQO 4.59 | 1.000/1.000 clean | ⚠️ Robust but audible-risk |
| **Google SynthID (audio)** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | *see §2.6* |
| IDEAW | *see §2.7* | *see §2.7* | dual-embedding, localisation | — | — | — | — | — | see §2.7 |
| XAttnMark (Dolby/Lehigh, ICML'25) | **no public release found** | none | 16-bit | MP3 0.995; EnCodec 0.965 | Echo 0.995; Speed 0.995 (only method that survives) | **Detection 98.6–99.3% at 1–10 s; attribution 81.2% @1 s** | SI-SNR 29.00, PESQ 4.43, ViSQOL 4.56 | 0.9919 (0.9856/0.0019) avg | ❌ Not obtainable |
| VocBulwark (2026), WaveVerify, SyncGuard, WMCodec | research-stage | — | — | — | — | — | — | — | 👀 Watch, do not adopt |

---

### 2.1 AudioSeal (Meta) — the recommendation

#### 2.1.1 LICENCE — RESOLVED DEFINITIVELY ✅

This was the single decisive question in E2, and the brief's caution was **historically justified but is now obsolete**.

**Code licence — `LICENSE` file fetched directly from the repo:**

> "MIT License
> Copyright (c) Meta Platforms, Inc. and affiliates.
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software")…"
> — <https://raw.githubusercontent.com/facebookresearch/audioseal/main/LICENSE> (**LICENSE file, fetched**) · **HIGH**

**Weights licence — the changelog entry in the repo README, quoted verbatim:**

> "2024-04-02: We have updated our license to full MIT license (including the license for the model weights)! Now you can use AudioSeal in commercial application too!"
> — <https://raw.githubusercontent.com/facebookresearch/audioseal/main/README.md> (**official repo README**) · **HIGH**

**Confirmed independently in the official documentation site:**

> "AudioSeal is available under the MIT license and can be used in both research and commercial applications."
> — <https://facebookresearch-audioseal.mintlify.app/> (**official docs**) · **HIGH**

**And in the Hugging Face model-card metadata:** `license: MIT` — <https://huggingface.co/facebook/audioseal> (**official model card**) · **HIGH**

**Upstream-contamination check (the trap that voided VoiceSculptor/Llasa-3B, IndicF5/F5-TTS and SPRING_F5/F5-TTS):**

| Vector | Finding | Verdict |
|---|---|---|
| Runtime dependencies | `pyproject.toml` declares only `numpy`, `omegaconf`, `torch>=1.13.0`, `einops`. **No `audiocraft`, no `encodec` package.** The CC-BY-NC AudioCraft/EnCodec *weights* are never pulled in. | ✅ Clean |
| Architecture provenance | The generator is "derived from EnCodec's design" (paper §3, Fig. 4) — an **architectural** derivation, re-implemented and re-trained inside this MIT repo. No EnCodec checkpoint is loaded. | ✅ Clean |
| Training data | "trained on a 4.5K hours subset from the VoxPopuli dataset" (paper §3.4). VoxPopuli is European-Parliament proceedings, released CC0 for the unlabelled/transcribed speech. | ✅ Clean — and note this is the *watermarker's* training data, which does not touch our TTS or mapper licence chain (see [08](08-licensing-propagation.md)) |
| Package classifier | PyPI `audioseal` 0.2.0 declares `License :: OSI Approved :: MIT License` | ✅ Clean |

> **VERDICT: AudioSeal is usable in a commercial, publicly-hosted service. Code MIT, weights MIT, no restrictive upstream. Confidence HIGH.**
>
> ⚠️ **Historical note so nobody re-litigates this:** before 2024-04-02 the AudioSeal *weights* were released under CC-BY-NC while the code was MIT. Any source, blog post, or memory dated before April 2024 that says "AudioSeal weights are non-commercial" **was correct at the time and is wrong now.** Pin this quote in the repo so the question is not re-opened.

**Maintenance status (currency check):** PyPI `audioseal` 0.2.0 released **2025-12-17**; prior releases 0.1.8 (2025-08-11), 0.1.7/0.1.6/0.1.5 (April 2025). Repo not archived. Not withdrawn, unlike Microsoft VibeVoice-TTS.
— <https://pypi.org/pypi/audioseal/json> (**official package index**) · **HIGH**

#### 2.1.2 Robustness — the paper's own table, with exact attack parameters

From San Roman et al., *Proactive Detection of Voice Cloning with Localized Watermarking*, **ICML 2024**, Table 3 + Appendix D.2. Evaluation on **10 k ten-second audios** from the VoxPopuli validation split; threshold chosen to maximise accuracy on a balanced set. `Acc (TPR/FPR)` and AUC.

| Attack | Eval parameter (App. D.2) | AudioSeal Acc (TPR/FPR) | AUC | WavMark Acc | WavMark AUC |
|---|---|---|---|---|---|
| None | — | 1.00 (1.00/0.00) | 1.00 | 1.00 | 1.00 |
| Bandpass | 500 Hz – 5000 Hz | 1.00 (1.00/0.00) | 1.00 | 1.00 | 1.00 |
| **Highpass** | **cut below 1500 Hz** | **0.61 (0.82/0.60)** | **0.61** | 1.00 | 1.00 |
| Lowpass | cut above 500 Hz | 0.99 (0.99/0.00) | 0.99 | 0.50 | 0.50 |
| Boost audio | ×10 | 1.00 (1.00/0.00) | 1.00 | 1.00 | 1.00 |
| Duck audio | ×0.1 | 1.00 (1.00/0.00) | 1.00 | 1.00 | 1.00 |
| Echo | 0.5 s delay, vol 0.5 | 1.00 (1.00/0.00) | 1.00 | 0.93 | 0.98 |
| Pink noise | std 0.1 | 1.00 (1.00/0.00) | 1.00 | 0.88 | 0.93 |
| White noise | std 0.05 | 0.91 (0.86/0.04) | 0.95 | 0.50 | 0.50 |
| Speed / "Fast" | ×1.25 | 0.99 (0.99/0.00) | 1.00 | 0.50 | 0.15 |
| Smooth | moving-avg window 40 | 0.99 (0.99/0.00) | 1.00 | 0.94 | 0.98 |
| Resample | →32 kHz→back | 1.00 (1.00/0.00) | 1.00 | 1.00 | 1.00 |
| AAC | 64 kbps | 1.00 (1.00/0.00) | 1.00 | 1.00 | 1.00 |
| **MP3** | **32 kbps** | **1.00 (1.00/0.00)** | **1.00** | 1.00 | 0.99 |
| EnCodec | 24 kHz, nq=16, →16 kHz | 0.98 (0.98/0.01) | 1.00 | 0.51 | 0.50 |
| **Average** | | **0.96 (0.98/0.04)** | **0.97** | 0.85 (0.85/0.14) | 0.84 |

— <https://arxiv.org/pdf/2401.17264> (**peer-reviewed paper, ICML 2024**) · **HIGH**

**Three things this table tells us that the brief did not anticipate:**

1. **MP3 at 32 kbps is a non-issue.** Acc 1.00. Figure 10 shows AudioSeal ≈0.9 even at 16 kbps and 1.00 from 32 kbps up. The brief's framing ("whether embedding survives MP3 transcode") treats this as an open risk; **it is not.** MP3 is solved.
2. **High-pass filtering is AudioSeal's specific weakness — and the paper says why.** §C.5: *"Our system's TF-loudness loss embeds the watermark where human speech carries the most energy, typically lower frequencies, due to auditory masking. This contrasts with WavMark, which places the watermark in higher frequency bands."* The paper's own head-to-head (§C.5): **highpass @1500 Hz — AudioSeal 0.7, WavMark 1.0; lowpass @1500 Hz — AudioSeal 1.0, WavMark 0.7.** They are complementary failures.
   *Why this matters for games:* occlusion/muffling filters are **low-pass** (AudioSeal survives, 0.99–1.00). Telephone/radio/walkie-talkie voice effects are **band-pass with a high-pass corner around 300 Hz** — bandpass 500–5000 Hz tested at 1.00, so a standard telephony effect is probably fine; an aggressive high-pass above ~1.5 kHz is not. **Flag as a per-effect risk, not a blanket one.**
3. **Mixing with a non-watermarked bed is fine.** Appendix C.3, Table 7: watermarked speech loudness-normalised and summed with non-watermarked music → **Acc 0.9787 (FPR 0.0310 / TPR 0.9883), AUC 0.9961**. With both watermarked, 0.9996. This directly answers "mixing with other audio" in the brief. · **HIGH**

**Multilingual / out-of-domain generalisation (Table 8):** average accuracy 0.95–0.98 across Seamless-translated Expresso in Mandarin, Spanish, French, Italian, German; Voicebox English; AudioGen; MusicGen. The **highpass column is the weak one in every language (0.52–0.71)** — the weakness is systematic, not language-specific. Additional datasets (Table 10): AudioSet 0.9992 (TPR 0.9996 / FPR 0.0011), ASVspoof 1.00, FakeAVCeleb 1.00. · **HIGH**

#### 2.1.3 Third-party robustness — where it looks worse

Independent evaluation matters more than the authors' own table. Three independent sources:

**(a) AudioMarkBench (Liu et al., NeurIPS 2024 Datasets & Benchmarks).** 20 k samples from Common Voice (25 languages incl. **Bengali and Tamil**) + LibriSpeech, **5-second clips at 16 kHz**, 15 perturbation types, no-box / black-box / white-box.

- Operating point they had to pick to hit FPR<0.01 and FNR<0.01: **AudioSeal τ = 0.15**, *not* the library default of 0.5. **This is a concrete configuration finding we must reproduce.**
- Headline: *"state-of-the-art audio watermarks are robust against several common no-box watermark-removal perturbations such as time stretch, low-pass, high-pass, and echo… current audio watermarking methods are not robust against no-box removal perturbations that are unseen during adversarial training. For instance, while ViSQOL is no smaller than 3, **EnCodec, SoundStream, and Opus achieve high FNRs**, indicating that those perturbations can remove watermarks from watermarked audios while preserving the audio quality."*
- Ranking: *"AudioSeal is the most robust against watermark removal and forgery among the evaluated watermarking methods. In contrast, WavMark is the least robust."*
- **Forgery** is hard: FPRs near 0 for all methods except quantisation (FPR > 0.2, but quality compromised).
- **White-box removal is total:** *"FNRs reach 1 for all watermarking methods when the SNR of the perturbations is 20"* (ViSQOL 3.2–3.9 — still acceptable quality). The AudioSeal authors reach the same conclusion: *"the detector's weights have to be kept private — otherwise adversarial attacks might be easily forged."*
- **⚠️ FAIRNESS GAP (this belongs in §6 as much as here):** *"watermarked audios with attribute 'female' are less robust to watermark-removal Gaussian noise perturbations (i.e., have higher FNRs) than those with attribute 'male' for all the evaluated watermarking methods… we conduct a two-tailed t-test… the calculated p-value ≈ 2.4 × 10⁻⁶ < α = 0.05. Thus, the robustness gap between 'female' and 'male' groups is statistically significant."* The gap also appears under **EnCodec, Opus and quantisation** perturbations, and under black-box (Square attack) and white-box attacks. **Age groups (teens/twenties/thirties/forties): "we observe no consistently significant differences across age groups"** — but note the dataset has **no speakers above their forties**, so this says nothing about elderly voices.
- Language variation in FNR is "noticeable"; the benchmark's own limitation statement concedes the language and age coverage is thin.
— <https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf> (**peer-reviewed, NeurIPS 2024 D&B**) · **HIGH**

**(b) RAW-Bench — Özer et al. (Sony AI), Interspeech 2025, "A Comprehensive Real-World Assessment of Audio Watermarking Algorithms: Will They Survive Neural Codecs?"** This is the most operationally relevant paper published, because it uses **raw uncompressed ≥44.1 kHz source audio**, **20 real-world distortions** (vs AudioMarkBench's 12), and includes **reverberation from a binaural room-impulse-response database** and **OGG/Vorbis** — i.e. the game-audio cases.

Model characteristics (their Table 1): AudioSeal — 16 kHz, **16-bit message, capacity 5.33 bits/s**, 4 500 h training, waveform domain. SilentCipher — 16 kHz, 23.8 bits, 5.33 bps, 372 h. Timbre — 22.05 kHz, 30 bits, 5.00 bps, 100 h. WavMark — 16 kHz, 16 bits, 5.28 bps, 5 000 h.

Clean quality (their Table 3): **AudioSeal SI-SNR 22.73, MCD 0.53, MOS-LQO 4.93, bitwise/full-message accuracy 0.997 / 0.962.** SilentCipher 49.13 / 0.25 / 4.98 / 0.999 / 0.993. Timbre 21.91 / 1.74 / **4.59** / 1.000 / 1.000. WavMark 35.89 / 0.62 / 4.91 / 0.998 / 0.993.

AudioSeal under their attacks — **bitwise accuracy** (top) and **full-message accuracy** (bottom), strict / loose settings:

| Attack | Bitwise (strict) | Full-message (strict) | Full-message (loose) | Reading |
|---|---|---|---|---|
| Gaussian noise (SNR 20–60) | ✓ ≥.99 | .93 | .95 | fine |
| Background noise | ✓ | .95 | .96 | fine |
| **Reverberation (BRIR, SNR 0–12)** | **.87** | **.22** | **.72** | **presence probably OK, message destroyed** |
| Dynamic-range compression | ✓ | .92 | .96 | fine |
| DR expansion | ✓ | .89 | .94 | fine |
| Limiter | .98 | .88 | .96 | fine |
| Lowpass (3.5–8 kHz) | ✓ | .93 | .96 | fine |
| **Highpass (10–500 Hz)** | .96 | **.57** | .88 | consistent with the known weakness |
| **Equalisation (±0.75 dB)** | .91 | **.39** | .90 | strict EQ hurts the message badly |
| Time stretch (0.75–1.25) | .97 | .72 | .71 | message degraded |
| Time jitter | ✓ | .96 | .95 | fine |
| **Polarity inversion** | **.18** | **.00** | **.00** | **CATASTROPHIC — see below** |
| Gain adjustment | ✓ | .91 | .91 | fine |
| Quantisation (8–16 bit) | ✓ | .95 | .95 | fine |
| **Phase shift (±0.10 s)** | **.62** | **.06** | **.02** | **near-total failure** |
| EnCodec (24 kHz) | .96 | .65 | .78 | degraded |
| **Descript Audio Codec (44.1 kHz)** | **.52** | **.00** | **.00** | **CATASTROPHIC** |
| MP3 (64/128/256 kbps) | ✓ | .92 | .93 | fine |
| **OGG / Vorbis (48–256 kbps)** | **✓** | **.95** | **.95** | **fine — good news for Wwise/FMOD** |
| AAC (64/128/256 kbps) | ✓ | .96 | .96 | fine |

Paper's conclusion, verbatim: *"Bitwise accuracies are generally below 0.5, and full-message accuracies are around 0 for almost all approaches in both EN and DA… audio watermarking models fail under neural compression, highlighting a critical weakness."* And: *"if we consider the limit situation where both algorithms successfully achieve their purpose, we believe that neural codecs will end up removing imperceptible watermarks."*

Code: `github.com/SonyResearch/raw_bench` — **use this as our robustness harness rather than writing one** (verify its licence before vendoring).
— <https://www.isca-archive.org/interspeech_2025/ozer25_interspeech.pdf> · also <https://arxiv.org/pdf/2505.19663> (**peer-reviewed, Interspeech 2025**) · **HIGH**

> 🔴 **THE POLARITY-INVERSION FINDING IS THE MOST ACTIONABLE THING IN THIS WHOLE SECTION.**
> Multiplying a waveform by −1 is **perceptually inaudible** for a mono source, costs nothing, is a one-line change in any DAW or `ffmpeg -af "volume=-1"`, and drives AudioSeal's bitwise accuracy to **0.18** and full-message accuracy to **0.00**. Phase shift by ±0.1 s is nearly as bad (.62 / .06).
> **Implication for the build: our detector MUST test the polarity-inverted signal as well as the original**, and ideally a small grid of circular time shifts. This is a ~2× detection-cost mitigation for a total-failure mode, and it is cheap because AudioSeal detection is 3.30 ms/sample.
> **Confidence HIGH** that the vulnerability exists (published table). **UNVERIFIED** that testing `-x` recovers it — that is a 30-minute experiment (§9).

**(c) XAttnMark (Liu et al., ICML 2025, Dolby + Lehigh)** — independent third-party numbers on AudioSeal, on MusicCaps, 5 s, 16-bit messages, threshold by Youden's index, attribution pools of {100, 1000, 10000}:

AudioSeal, `Det. (TPR/FPR)` / `Att.`: Identity 1.00 (0.99/0.00) / **0.69**; Bandpass 1.00 / 0.31; Boost 1.00 / 0.50; Duck 1.00 / 0.56; Echo 1.00 / 0.38; Highpass 1.00 / 0.31; Lowpass 1.00 / 0.56; MP3@128k 1.00 / 0.38; Pink 1.00 / 0.75; White 1.00 / 0.56; Smooth 1.00 / 0.19; **Speed (0.8–1.2×) 0.61 (0.36/0.15) / 0.00**; Resample 1.00 / 0.56; AAC 1.00 / 0.12; EnCodec 1.00 / 0.31; Crop 1.00 / 0.12. **Average 0.971 (0.950/0.010) detection, 0.39 attribution.**

Two conclusions:
- **Presence detection holds up independently** (avg 0.971 across 16 edits, on out-of-domain music).
- 🔴 **The 16-bit message does not.** Attribution averages **0.39**, and is only **0.69 on completely unmodified audio.** This matches AudioSeal's own Table 4 (attribution accuracy 68.2 % at N=1 falling to 56.4 % at N=10⁴, FAR 2.5–11.8 %) and RAW-Bench's full-message figures. **Do not architect around the AudioSeal payload.**
- The Speed disagreement (0.61 here vs 0.99 in AudioSeal's own paper at 1.25×) is a genuine inter-paper conflict; different domain (music vs speech) and different threshold selection. Treat speed/pitch change as **MEDIUM-confidence risk, not solved.**
— <https://arxiv.org/abs/2502.04230> (**peer-reviewed, ICML 2025**) · **HIGH** for the numbers, **MEDIUM** for transfer to speech

#### 2.1.4 Imperceptibility — including the stylized-voice question

Published numbers (AudioSeal paper Table 1, on 10 s VoxPopuli speech):

| Method | SI-SNR ↑ | PESQ ↑ | STOI ↑ | ViSQOL ↑ | MUSHRA ↑ |
|---|---|---|---|---|---|
| WavMark | 38.25 | 4.302 | 0.997 | 4.730 | 71.52 ± 7.18 |
| **AudioSeal** | 26.00 | **4.470** | 0.997 | **4.829** | **77.07 ± 6.35** |
| *(ground truth reference)* | — | — | — | — | *80.49* |
| *(low anchor: EnCodec @1.5 kbps)* | — | — | — | — | *53.21* |

MUSHRA protocol: 100 speech samples × 10 s, ≥20 raters each, low-anchor screening at 80 %. AudioSeal deliberately trades SI-SNR (26.0 vs WavMark's 38.25) for perceptual quality: *"AudioSeal is not optimized for SI-SNR but rather for perceptual quality of speech… our goal is to hide as much watermark power as possible while keeping it perceptually indistinguishable from the original."*

On AI-generated speech (Table 9): SI-SNR 23.35–25.23, PESQ 4.199–4.449, STOI 0.998–0.999, ViSQOL 4.669–4.800. RAW-Bench independently measures MOS-LQO **4.93** and MCD **0.53** on a 44.1 kHz raw corpus. XAttnMark independently measures PESQ **4.51**, STOI **0.990**, ViSQOL **4.72**. **Three independent groups agree the perceptual cost is small.** · **HIGH**

**⚠️ Does it audibly degrade stylized voices — raspy, whispered, breathy?**

**UNVERIFIED. No paper breaks this out.** What exists, and why it is *not* an answer:

- AudioSeal's Table 8 generalisation set is built from **Expresso** speech translated by SeamlessExpressive. Expresso *is* an expressive corpus, and AudioSeal averages 0.97–0.98 detection on it. But (i) that measures **detection**, not audibility, and (ii) the translated output is re-synthesised, so the original whisper/breathy phonation is not preserved intact. **This is weak evidence, not a pass.**
- RAW-Bench covers speech, music and environmental sound but does not slice by phonation type.
- No listening test in any of the four papers isolates whispered, creaky, growled or breathy speech.

**Why we should genuinely worry.** AudioSeal's imperceptibility comes from a **TF-loudness loss based on auditory masking** — the watermark is deliberately placed *where the speech has the most energy* so the speech masks it. The paper is explicit: it embeds *"where human speech carries the most energy, typically lower frequencies."* Whispered speech has **no voiced excitation, no harmonic stack, and hence no strong low-frequency energy to mask with**; its spectrum is a shaped noise floor. The masker the loss relies on is largely absent. That predicts **two** failure directions simultaneously — the watermark becomes *more audible* (less masking) **and** *weaker* (less energy budget). The same argument applies to breathy phonation and to very quiet lines.

Our product deliberately generates "raspy", "whispered", "breathy" voices from descriptions. **This is a first-class risk for us and a non-issue for Meta's use case.** Design the test in §9.

#### 2.1.5 Detection: threshold, key, and who can run it

- **Detection is public.** The detector checkpoint `audioseal_detector_16bits` is on the Hugging Face Hub under MIT. Anyone can download it. There is **no secret key**. Consequences:
  - ✅ Third parties (a platform, a journalist, a games storefront) can verify our audio without asking us — which is exactly what EU AI Act Art. 50 "detectable" pressure wants, and what the *Watermarking Without Standards Is Not AI Governance* critique says is missing elsewhere.
  - 🔴 An adversary has white-box access to the detector by default. AudioMarkBench: white-box FNR → 1.0 at SNR 20. AudioSeal's authors: *"the detector's weights should be kept confidential."* **We cannot keep them confidential — they are already public.** Our threat model must assume watermark removal is available to any motivated adversary.
- **Threshold.** Library default 0.5. AudioMarkBench had to use **τ = 0.15** to reach FPR<0.01 and FNR<0.01 on 5 s clips. **We must calibrate our own τ on our own output distribution and clip-length distribution — do not ship the default.**
- **Reported operating points:** paper — average 0.96 acc (TPR 0.98 / FPR 0.04) across 15 edits on 10 s clips; AudioSet FPR 0.0011 at TPR 0.9996. The paper's own scale framing: *"on a platform processing 1 billion samples daily, an FPR of 10⁻³ and a TPR of 0.5 means that 1 million samples require manual review each day."*

#### 2.1.6 Payload capacity — the honest answer

- Nominal: **16-bit optional message**, capacity **5.33 bits/second** (RAW-Bench Table 1). 16 bits therefore occupies a **~3.0 second** nominal window.
- **Reliability of that payload is poor even before our short-clip problem:** attribution 0.69 clean / 0.39 average under edits (XAttnMark); 0.962 full-message clean but .00–.72 under reverb/DAC/polarity (RAW-Bench); 56–68 % attribution accuracy with FAR 2.5–11.8 % at realistic pool sizes (AudioSeal Table 4).
- ➡️ **Architecture consequence: the watermark carries a PRESENCE BIT, not an identity.** Do not attempt to embed `render_id` or `identity_id` in the watermark. Provenance/identity lives in (a) the server-side provenance log and (b) the export manifest / C2PA sidecar. The watermark's job is exactly one thing: *"this audio came from a generative system"*, survivable after the metadata is stripped.
- If we *do* set a message, set it to a **low-cardinality, slowly-changing value** — e.g. a 16-bit `(service_id, model_epoch)` code with ≤64 live values — where 56–69 % per-clip accuracy still aggregates to a confident answer over several clips. Never a per-render unique ID.

#### 2.1.7 Compute cost at render time

AudioSeal paper Table 5, single Nvidia Quadro GP100, segments 1–10 s:

| | Generation (ms) | Detection (ms) |
|---|---|---|
| **AudioSeal** | **7.41 ± 4.52** (14× faster than WavMark) | **3.30 ± 2.03** watermarked · **3.25 ± 1.99** unwatermarked (**485×** faster than WavMark when no watermark present) |
| WavMark | 104.58 ± 65.66 | 106.21 ± 66.95 / 1710.70 ± 1314.02 |

For a 2-second render this is **single-digit milliseconds on GPU** — utterly negligible next to TTS inference (see [07](07-serving-and-cost.md)). It is a `wav + model.get_watermark(wav)` addition; there is no reason to make it optional, batched, or deferred. **Watermark unconditionally, in the render path, before the bytes ever leave the process.**

---

### 2.2 Perth (Resemble AI) — MIT, but presence-only and the good version is closed

**Licence — `LICENSE` fetched directly:** standard MIT text, **copyright Resemble AI, 2025**.
> *"Permission is hereby granted, free of charge, to any person obtaining a copy of this software"*
— <https://raw.githubusercontent.com/resemble-ai/perth/master/LICENSE> (**LICENSE file, fetched**) · **HIGH**

**Upstream check:** PyPI `resemble-perth` 1.0.1 (2025-05-23), **34.4 MB wheel** — the size confirms weights are bundled inside the MIT distribution. No separate weights licence, no gated HF repo, no upstream checkpoint dependency. Dependencies `librosa` (ISC) and `soundfile` (BSD), both permissive. **No restrictive upstream found** — materially cleaner than the VoiceSculptor/Llasa-3B and IndicF5/F5-TTS traps. · **MEDIUM-HIGH**

**Verdict: licence YES. Function NO.**
- 🔴 **Presence-only.** VoxWatermark characterises Perth as using *"an implicit signature"* rather than explicit bit payloads. It cannot carry any message. (For us this is survivable — see C1 in §2.z — but it removes all optionality.)
- 🔴 **The open-source version is the weak one.** Resemble's own published benchmark reports **open-source Perth: pitch-shift 10 %, clipped Gaussian noise 45 %**, while the fixed version — **PerTh V2: pitch-shift 94 %, Gaussian 100 %** — is **commercial and closed**. A 10 % survival rate under pitch shift is a non-starter for game audio, where per-instance pitch randomisation is routine. · **MEDIUM** (vendor self-reported)
- **Zero peer-reviewed evaluation.** The Multiplexing paper (arXiv:2511.02278) states plainly that for Perth *"peer-reviewed evaluation remains scarce."* No published minimum duration, no duration ablation, no PESQ. The vendor claim that *"any non-silent segment is enough to recover it"* is marketing, not evidence. · **LOW**

**Where Perth still earns a place:** if we use Resemble **Chatterbox** anywhere in the stack ([03](03-tts-backends-english.md)), its output is already Perth-marked, so consistency argues for keeping it. And the Multiplexing paper finds **AudioSeal + Perth complementary — 0.974 AUC / 0.905 TPR@0.05 averaged across 11 attacks** — so layering two independent MIT marks raises the removal bar at no licence cost. **Consider as a second layer, never as the primary.**

### 2.3 WavMark (Microsoft) — MIT, and rejected on the merits

**Licence:** `LICENSE` at `raw.githubusercontent.com/wavmark/wavmark/main/LICENSE` is MIT; GitHub sidebar reads "MIT license". **Weights independently checked:** HF `M4869/WavMark` model card metadata reads **`license: mit`**. No gating, no acceptable-use addendum. · **HIGH**
⚠️ **Cosmetic wrinkle:** the LICENSE file appears to retain unfilled `[year]` / `[fullname]` placeholders — an uncustomised MIT template with **no identified licensor**. This does not invalidate the grant in practice but should be recorded in our licence register. · **MEDIUM**

**Verdict: licence YES, but reject.** Reasons, all independently sourced:

| Problem | Evidence |
|---|---|
| 🔴 **Hard ≥ 1 s floor** | The encoding unit is exactly 1 second at 16 kHz (32 bits via an invertible network over STFT). The SoK confirms WavMark *"only works with audio >= 1 second in length."* **Any Alaap line under 1.0 s cannot be watermarked at all** — and game barks routinely include sub-second interjections. |
| 🔴 **No redundancy at 1.5 s** | The headline **0.48 % utterance BER depends on repeated encoding across 10–20 s** with majority voting. A 1.5 s file holds **one** complete unit. The applicable numbers are the segment-level ones: 0.65 % clean, **6.41 % under median filtering**, 3.40 % under quantisation — with no votes to correct them. |
| 🔴 **Short-file false positives** | Detection is **Brute Force Detection**: a 1-second window slides at a 5 % stride ≈ **20 decode attempts per second**, accepting on a **10-bit pattern match → 1/1 024 per attempt**. Naive per-file FPR for a 1.5 s clip is on the order of **3 %** unless acceptance is tightened. · **MEDIUM** (derivation from the paper's own stated numbers; validate empirically) |
| 🔴 **Detection is slower than real time on CPU** | Encoding 7.7× RT on CPU, but **BFD localisation runs at 0.38× real time — ~2.6 s of compute per 1 s of audio.** The authors call this *"generally tolerable"*; for a service that must scan at scale it is not. |
| 🔴 **Least robust of the four benchmarked** | AudioMarkBench: *"WavMark is the least robust… watermarks embedded by WavMark can even be removed by Gaussian noise and MP3 compression without compromising the watermarked audios' quality."* Black-box removal succeeds at **SNR ≈ 40 / ViSQOL ≈ 4.5** — near-transparent audio, mark gone. RAW-Bench: **0.00 bitwise accuracy under Descript Audio Codec.** |
| **Silence is toxic** | The paper warns of *"noticeable noise"* on muted segments and recommends *"omitting silent segments"* — exactly what short game barks are full of. |

### 2.4 SilentCipher (Sony) — the credible runner-up, with a game-specific disqualifier

**Licence:** `LICENSE` at `raw.githubusercontent.com/sony/silentcipher/master/LICENSE` — **MIT, copyright Sony Research Inc. 2024**, granting permission to *"use, copy, modify, merge, publish, distribute, sublicense, and/or sell"*. **Weights:** HF `Sony/SilentCipher` — MIT, no non-commercial clause, dual-hosted on GitHub Releases and HF. **A corporate MIT grant from Sony Research is the strongest licence provenance in this set.** · **HIGH**
⚠️ Training code is still being released — you get inference + weights, not reproducibility. Acceptable for deployment; we cannot retrain on our own voice distribution. · **MEDIUM**

**Strengths — genuinely better than AudioSeal on two axes:**
- **Best imperceptibility headroom in the field: SDR 47.24 dB**, and RAW-Bench independently measures **SI-SNR 49.13, MCD 0.25, MOS-LQO 4.98** vs AudioSeal's 22.73 / 0.53 / 4.93. It also exposes a **tunable message-SDR knob** — a genuine robustness/quality dial we do not get with AudioSeal.
- **1 302× real time** — the fastest in the set.
- **32 bits @ 16 kHz, 40 bits @ 44.1 kHz**, with the message **repeated across spectrogram frames** rather than packed into discrete units — architecturally the right shape for short clips, and it explains 100 % recovery under 50 % cropping.
- Its own paper reports **OGG Vorbis 100 % at 64/128/256 kbps** and MP3 96 %/100 %/100 % — on 6-second clips, on its own attack list.

**Why it is not the primary:**

| Problem | Evidence |
|---|---|
| 🔴 **Independent evaluation is far worse than its own paper.** | RAW-Bench, evaluating the released checkpoint under *strict* real-world settings, reports materially lower **full-message** accuracy for SilentCipher under codec attacks than the SilentCipher paper reports for the same codec families. ⚠️ **We could not resolve every per-cell value reliably from text extraction — treat the exact figures as MEDIUM and re-read RAW-Bench Table 5 directly before relying on them.** The direction of the discrepancy (own-paper optimistic vs independent pessimistic) is the part we are confident about. |
| 🔴 **Gain adjustment appears to be a failure mode.** | RAW-Bench's SilentCipher row shows near-zero full-message accuracy on at least one of the dynamics/gain attacks. **If gain adjustment genuinely breaks it, that is disqualifying for game audio**, where distance attenuation changes level continuously. ⚠️ **MEDIUM — verify before considering SilentCipher at all.** |
| 🔴 **6 s is the shortest length ever evaluated** — 4× our median clip. Trained at a fixed 12 s. | Its 50 %-crop result covers 3 s, not 1.5 s. |
| **Opus untested**; no pitch-shift, no reverb, no DRC in its own attack list. | Same blind spots as everyone. |
| **Shares AudioSeal's fatal generic weaknesses.** | HarmonicAttack defeats it at ~100 % ASR; neural codecs destroy it. |

**Verdict: keep as the documented fallback.** If our §2.y experiment shows AudioSeal failing on short clips or on stylized phonation, SilentCipher's SDR headroom and frame-repeated message make it the first thing to try. **But run the gain-adjustment and MP3/OGG checks first — the independent numbers do not match the vendor's.**

### 2.5 Timbre Watermarking (NDSS 2024) — ❌ disqualified by licence

**Licence — fetched:** `raw.githubusercontent.com/TimbreWatermarking/TimbreWatermarking/main/LICENSE` is verbatim the **GNU General Public License, Version 3, 29 June 2007**. GitHub sidebar confirms "GPL-3.0 license". · **HIGH**

> *"GNU GENERAL PUBLIC LICENSE / Version 3, 29 June 2007"*
> *"You may convey verbatim copies of the Program's source code as you receive it, in any medium, provided that you conspicuously and appropriately publish on each copy an appropriate copyright notice."*

**Analysis, and the nuance does not save it.** GPL-3.0 is not AGPL, so pure server-side SaaS use does not by itself trigger conveying obligations — a narrow reading permits running it on our server. But:
- The moment we ship **anything** to a customer — a desktop tool, a Unity/Unreal plugin, an on-prem or self-hosted tier, a Docker image, an SDK — that is **conveying**, and copyleft attaches to the **combined work**, forcing GPL release of our pipeline. Our roadmap contains exactly such deliverables.
- Python import-linking into a proprietary service is a contested derivative-work boundary the FSF reads aggressively.
- ⚠️ **Additional suspected upstream trap (UNVERIFIED):** the repo root contains a `voice.clone/` directory carrying third-party TTS/voice-cloning implementations for the paper's attack experiments — bundled third-party code under its own possibly research-only terms. **The exact pattern that voided three components elsewhere in this project.** Not audited, because the GPL verdict makes the audit moot.

**Verdict: REJECT.** Doubly unfortunate, because **Timbre scores the highest overall robustness of the four systems in RAW-Bench** — and simultaneously the **lowest imperceptibility** (MCD 1.74, MOS-LQO 4.59) and **~65 % accuracy on singing/music** (SoK). Even licence-clean it would be the wrong trade for a product whose output *is* the voice.

### 2.6 Google SynthID (audio) — ❌ RESOLVED: unobtainable, at any price

**The brief lists SynthID as a candidate. It is not one, and this is now settled with four independent citations.** · **HIGH**

1. **Google's own developer documentation covers TEXT ONLY.** *"SynthID Text has been open sourced to make watermarking for text generation available to developers."* The entire technical page — HF Transformers v4.46.0+ integration, reference implementation, demo Space — is text watermarking. **No audio, image or video embedding tool is offered to developers.** — <https://ai.google.dev/responsible/docs/safeguards/synthid>
2. **DeepMind confines audio to Google products:** the audio watermark operates *"within our AI music generation model Lyria or the podcast generation feature of Notebook LM."* No external embedding endpoint.
3. 🔑 **The decisive data point:** Vertex AI documentation states *"all media generated by Imagen, Veo, and Lyria (**but not Chirp**) are watermarked."* **Chirp is Google's own speech/TTS family, and it is explicitly NOT SynthID-watermarked.** If Google does not watermark its own TTS output, there is plainly no speech-watermarking SDK for anyone else.
4. **SynthID Detector** (launched I/O, 2026-05-19) is **detect-only and access-gated**: *"Journalists, media professionals and researchers can join our waitlist."* Not a public API, not self-serve, and it reads **only Google-ecosystem watermarks**.

On the 2026 "cross-vendor SynthID" headlines (OpenAI, ElevenLabs, NVIDIA Cosmos, Kakao): these are **negotiated bilateral partnerships** where Google integrates SynthID into a named partner's stack — not a published SDK. Google's own framing is asymmetric: *"we open sourced SynthID **text** watermarking, so any developer can build with this technology."* The word "text" is doing all the work; **no equivalent sentence exists for audio.**

**Verdict: third parties may eventually DETECT via a gated waitlist; they can never EMBED. Remove SynthID from the candidate list.**

### 2.7 IDEAW, and the 2025–2026 field

**IDEAW** (EMNLP 2024) — `github.com/PecholaL/IDEAW` is **Apache-2.0** (ideal licence), reports **< 1 % BER** and PESQ 4.33. **Disqualified on deployability:** *"no pretrained weights are released"*, and the authors' own disclaimer states *"Due to the company's security regulations, the final version of the experimental code is debugged and run on company's server but cannot be copied out. **There may be some errors in the current version.**"* Shipping this means funding a from-scratch training run against a self-admittedly buggy reference implementation. · **HIGH**

**The 2025–2026 successors — all "watch, do not adopt":**

| Method | Why interesting | Why unusable |
|---|---|---|
| **XAttnMark** (ICML 2025, Lehigh + **Dolby**) | Best published numbers in the field: detection avg **99.19 %**, attribution avg **93 %** (vs AudioSeal's 39 %), TPR 98.56 % @ FPR 0.19 %, PESQ 4.43 / SI-SNR 29.0. Runs the **only published duration ablation** (§2.y). | **No code or weights released.** A Dolby-authored watermarker is unlikely to ship permissively. |
| **SyncGuard** (2025) | 🔑 **0.5 s → 99.63 %** — the only published sub-second result, and the only method surviving **pitch scaling 0.9–1.1 at 99.83–99.92 %**. Frame-wise broadcast embedding with time-dimension averaging is **architecturally exactly what Alaap needs.** | **No code repository or licensing information provided in the paper.** Baselines are FSVC/FDLM/DeAR/DRAW, not AudioSeal/WavMark, so cross-comparison is unsound. **Monitor for release — this is the one to watch.** |
| **WaveVerify** (2025) | MIT badge; best published BER (0.00 across MP3 64k/128k, highpass 3500 Hz, bandpass); best ViSQOL **4.76**; the **only method evaluated on an emotional-speech corpus (RAVDESS)**. | ⚠️ **Weights licence UNVERIFIED** — the checkpoint *"will be automatically downloaded on first use"* with no stated host or terms. **Untested against Opus and every neural codec** — the axis that kills watermarks. 10 s evaluation only. |
| **WAKE** (Interspeech 2025) | 🔑 **The only KEYED design found** — an 8-bit key; wrong key → ~50 % BER. That is the correct structural answer to the forgery problem in §2.1.5. Supports re-embedding without destroying the first mark. | Licence and code availability **unclear** (demo page only). **Monitor.** |
| **VocBulwark** (Jan 2026) | In-model watermarking via adapter injection with frozen generative weights; claims resilience to codec regeneration and variable-length manipulation. | Research-stage. **And architecturally incompatible with us:** in-model watermarking contradicts our **frozen, swappable TTS backend** contract ([03](03-tts-backends-english.md)). |
| **VoxWatermark** (Jun 2026), **StreamMark** (Apr 2026) | New large-scale benchmarks; VoxWatermark is the **only benchmark that includes Perth**. | Benchmarks, not watermarkers. Useful for our eval harness. |

**AudioMarkBench** repo (`moyangkuo/AudioMarkBench`) is **MPL-2.0** — fine as an *evaluation harness* (we do not ship it), but ⚠️ **do not vendor its files into the product tree, and note that it pulls GPL-3.0 Timbre as a submodule.** Keep it in an isolated eval container. **RAW-Bench** (`github.com/SonyResearch/raw_bench`) is the better harness for us (§2.y) — verify its licence before vendoring.

> **Do not build a bespoke watermarker.** Every 2025–2026 improvement is unreleased, research-stage, or in-model. Post-hoc watermarking is architecturally correct for Alaap — it is the weaker family, but it is the one compatible with a frozen, swappable backend.

---

### 2.y The short-clip problem — the highest-priority sub-question

**The brief is right to flag this, and the literature does not answer it for AudioSeal.**

**What the evaluation lengths actually are:**

| Source | Clip length used |
|---|---|
| AudioSeal paper — detection results (Table 3) | **10 s** ("10k ten-seconds audios") |
| AudioSeal paper — attribution (Table 4) | **5 s** ("each consisting of 5 seconds of speech (not 10s to reduce compute needs)") |
| AudioSeal paper — runtime (Fig. 9, Table 5) | 1–10 s, but **runtime only, no accuracy** |
| AudioMarkBench | **5 s** (Common Voice), ≤5 s (LibriSpeech) |
| RAW-Bench | ~**3 s** effective (16 bits ÷ 5.33 bps) |
| XAttnMark | 5 s default, **with a 1–10 s ablation** |

**Nobody publishes AudioSeal detection accuracy on a fully-watermarked 1–3 s file. That number does not exist.** Anyone who tells you otherwise is reading the localisation result, which is a different quantity.

**What the localisation result actually says (and the discrepancy inside the paper).**
AudioSeal §5.3 prose claims: *"AudioSeal achieves an IoU of 0.99 when just one second of speech is AI-manipulated, compared to WavMark's 0.35."* But the paper's **own Table 6**, whose caption reads *"The IoU is computed for 1s of watermark in 10s audios (corresponding to the leftmost point in Fig. 5)"*, reports **IoU = 0.802** for the EnCodec architecture (0.796 for the DPRNN variant). Figure 5's leftmost point agrees with ~0.8, not 0.99.
➡️ **The paper contradicts itself; 0.802 is the number backed by a table, 0.99 is the number in the prose.** Cite 0.802. **Confidence HIGH that the discrepancy exists; MEDIUM on which is correct.**
➡️ **And in any case this measures the wrong thing for us.** Localisation IoU answers *"can you find a 1 s watermarked island inside a 10 s ocean?"*. Our question is *"is this entire 1.4 s file watermarked?"* — a strictly easier problem with a different failure mode.

**The best available evidence that short clips are fine — from a different tool.**
XAttnMark (ICML 2025) runs the only published duration ablation, on MusicCaps, averaged over all standard edits:
> *"the detection accuracy remains robust (98.6–99.3%) across all tested durations from 1–10 s, demonstrating our method's effectiveness even with short audio clips. Attribution accuracy shows greater sensitivity, improving from 81.2% at 1s to a peak of 93.0% at 5s, then stabilizing at 91.5–92.6% for longer durations… reliable detection (>98.5%) is achievable with clips as short as 1s."*
XAttnMark's inference also **pads and splits input into 1 s chunks** and watermarks each chunk independently — i.e. 1 s is its native unit.
➡️ This is **strong analogical evidence for the general principle**: *presence detection is duration-robust to ~1 s; payload recovery is not* (81.2 % vs 93.0 %). It is **not** evidence about AudioSeal, and it is on music, not speech. · **MEDIUM** as transfer evidence.

**The theoretical argument, stated so we know what we are testing.**
AudioSeal's detector emits a per-sample probability and the detection score is the **mean over samples**. For a *fully* watermarked clip the expected per-sample score does not depend on duration; only the **variance of the mean** does. A 1.5 s clip at 16 kHz still contains **24 000 samples** — far more than statistical sufficiency requires. So the naive worry ("too few samples") is wrong. The real short-clip risks are different and specific:

1. **Silence fraction.** The TF-loudness loss puts watermark energy where speech energy is. A 1.2 s game line with 0.4 s of leading/trailing silence and a plosive onset has proportionally **much less watermarked signal** than a 10 s continuous read. Effective watermarked duration ≪ file duration.
2. **Boundary effects.** Convolutional encoder/decoder with strides (2,4,5,8) → receptive-field edge effects at clip start and end are a **larger fraction** of a short clip.
3. **Threshold calibration drift.** τ was chosen on 10 s audio. Score variance rises as duration falls, so a threshold tuned at 10 s will have a **different, uncharacterised FPR/FNR** at 1.5 s. AudioMarkBench already had to move τ from 0.5 to 0.15 going to 5 s.
4. **Per-line vs per-corpus decisions.** For enforcement we usually get *many* lines from one game, not one. Aggregating the presence decision over N lines collapses the error rate geometrically. **This changes the product design, not just the test.**

> ### THE TEST WE MUST RUN (§9 line item)
> **Cost: ~2 GPU-hours, one afternoon. It blocks nothing upstream but must complete before S9.**
> 1. Generate 2 000 lines from our own pipeline, stratified over: duration bins {0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0 s} × phonation {modal, whispered, breathy, raspy, shouted} × gender × age-band × language {en, hi, ta, bn}. Use the *real* mint→render path, not LibriSpeech.
> 2. Watermark all of them; keep a matched unwatermarked control set of equal size (needed for FPR).
> 3. Sweep τ ∈ [0.02, 0.9]; plot **ROC per duration bin** and report **TPR at FPR = 10⁻³** per bin. Pick τ per duration bucket if the curves separate.
> 4. Repeat the whole sweep after each attack: MP3 32/128k, **Opus 24/32/64k** ⚠️, Vorbis 64/128k, AAC 128k, EnCodec, DAC, resample 44.1↔16k, ±10 % speed, RIR reverb (small room / hall / cathedral), lowpass 500/1500/3500 Hz, highpass 300/1500 Hz, **polarity inversion**, ±0.1 s phase shift, −20 dB gain, mix with a music bed at 0/−6/−12 dB SNR.
> 5. Report **per-demographic-slice** TPR (AudioMarkBench found a statistically significant female/male gap; we must know whether it reproduces on our output). This slice **is** the §6.2 eval slice — do not build two harnesses.
> 6. Measure **audibility** on the stylized subset: PESQ/ViSQOL vs unwatermarked, plus a small forced-choice ABX listening test on ~40 whispered/breathy pairs. This is the only way to answer §2.1.4.
> 7. **Do not write the harness.** Fork `github.com/SonyResearch/raw_bench`, add Opus + polarity-inversion + our duration/phonation stratification.
>
> **Exit criterion:** *TPR ≥ 0.99 at FPR ≤ 10⁻³ for every duration bin ≥ 1.0 s under MP3/AAC/Vorbis/resample/gain/reverb, with a documented, accepted failure list (Opus, DAC, polarity-without-mitigation, aggressive highpass).*

---

### 2.z RECOMMENDATION

**Adopt AudioSeal. Ship it at S9 as specified in the brief. Nothing else on the market is both permissively licensed and this well-validated.**

The reasoning, ranked:

1. **Licence is the gate, and only AudioSeal clears it unambiguously** with MIT on *both* code and weights, verified from the LICENSE file, the README changelog, the official docs, and the HF model card, with a clean dependency tree. For a public commercial service this is decisive. (§2.1.1)
2. **It is the most robust of the benchmarked post-hoc watermarkers** by the judgement of an independent NeurIPS benchmark, not by its authors' claim.
3. **It is the only one with a public detector**, which is what makes third-party verification — and therefore any credible transparency claim under EU AI Act Art. 50 — actually possible.
4. **The cost is nil**: 7.4 ms to embed, 3.3 ms to detect.

**Adopt it with these four non-negotiable engineering conditions:**

| # | Condition | Because |
|---|---|---|
| **C1** | **Watermark carries a presence bit only.** Identity/render-ID lives in the provenance DB and the export manifest, never in the payload. | Attribution 0.39 avg / 0.69 clean (§2.1.6) |
| **C2** | **Detector tests `x` and `−x`**, plus a small circular-shift grid. | Polarity inversion → bitwise 0.18, message 0.00 (§2.1.3b) |
| **C3** | **Calibrate τ on our own output**, per duration bucket, targeting FPR ≤ 10⁻³. Never ship the 0.5 default. | AudioMarkBench needed τ=0.15 at 5 s |
| **C4** | **Publish the accepted failure list** in the model card and ToS rather than claiming robustness we do not have: Opus, Descript Audio Codec, neural re-synthesis, aggressive high-pass, and any white-box adversary. | Honesty is cheaper than a broken claim, and §4 shows nobody is required to have an unbreakable watermark |

**What AudioSeal does NOT buy us — state this plainly in the model card:**

- **It is not a defence against a determined adversary.** White-box removal drives FNR to 1.0 (AudioMarkBench). The AAAI 2026 overwriting attack — *"Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking"* — reports a **nearly 100 % attack success rate** against AudioSeal across white-, grey- and black-box settings, with post-attack BER *"tightly centered around 0.5 (mean µ = 0.504–0.506), indicating a complete corruption of the original watermark."* An adaptive-attack follow-up (*Learning to Evade*, June 2026) reports **0 % detection after removal, 3–17 % after replacement**, at SNR 23.9–26.6 dB and ViSQOL 4.26–4.70 — i.e. **removal at unimpaired audio quality**. — <https://arxiv.org/pdf/2509.05835> (AAAI 2026) · <https://arxiv.org/html/2606.22310> · **HIGH**
- **It is not durable through a neural codec.** RAW-Bench: DAC → full-message 0.00. AudioMarkBench: EnCodec/SoundStream/Opus → high FNR at ViSQOL ≥ 3. As neural codecs enter normal delivery pipelines this degrades over time.
- **It is therefore a signal for good-faith downstream verification and a compliance artefact — not an enforcement mechanism.** Design the product accordingly: the watermark tells an honest platform "this is synthetic"; the **provenance log** is what answers "who made this", and it cannot be stripped by anyone outside our infrastructure.

**Do not adopt Perth, WavMark, SilentCipher or Timbre as the primary** (per-tool licence and robustness detail in §2.2–2.5). **SilentCipher is the credible second choice on quality grounds** (SI-SNR 49.13, MCD 0.25, MOS-LQO 4.98 — materially better than AudioSeal, and it shares the same 0.00 polarity/phase failure) *if and only if* its licence clears; **Timbre is the most robust in RAW-Bench but has the worst perceptual scores** (MCD 1.74, MOS-LQO 4.59), which is exactly the wrong trade for a product whose output *is* the voice.

**Do not build a bespoke watermarker.** Every 2025–2026 improvement (XAttnMark, VocBulwark, WaveVerify, SyncGuard, WMCodec) is either unreleased, research-stage, or in-model — and in-model watermarking is incompatible with our **frozen, swappable TTS backend** contract ([03](03-tts-backends-english.md)). Post-hoc is architecturally correct for us even though it is the weaker family.

**Re-evaluate when:** (a) an Opus-robust permissively-licensed watermarker appears, (b) C2PA soft-binding for audio has a working open implementation (§3), or (c) a harmonised EU standard names a specific technique (§4.1).

---

## 3. Provenance: C2PA, content credentials & platform rules

### 3.1 C2PA for audio in 2026 — the specification is ready, the software is not

**Current version: C2PA Technical Specification 2.4, dated 2026-04-01.** (2.3 was 2026-01-05.) Anything citing 2.2 is 16 months stale.
— <https://spec.c2pa.org/specifications/specifications/2.4/specs/_attachments/C2PA_Specification.pdf> (**official specification, downloaded**) · **HIGH**

**Audio is a first-class asset type.** Embedding is defined per container (Appendix A.1):

| Format | Spec § | Mechanism |
|---|---|---|
| **WAV / BWF** (+ AVI, WebP, other RIFF) | A.3.7 | RIFF chunk with identifier `C2PA` |
| **MP3, FLAC** | A.3.4 | ID3v2 GEOB (General Encapsulated Object) |
| **AAC, ALAC**, MP4, MOV, HEIF | A.5 | BMFF `uuid` box |
| **OGG Vorbis** | A.3.5 | dedicated logical bitstream, 5-byte id `\x00c2pa` |
| **Opus** | — | **not in the spec at all** |

Verbatim, §A.3.7: *"The C2PA Manifest Store shall be embedded into a RIFF-compatible file (i.e., WAV, AVI or WebP) as the data of a chunk with an identifier of `C2PA`. For compatibility reasons, this `C2PA` chunk shall appear as the last sub-chunk of the first RIFF header chunk."* · **HIGH**

**New in 2.4 and aimed precisely at us — the `c2pa.ai-disclosure` assertion (§18.28):**
> *"a new AI Disclosure assertion is introduced to provide a means for a Claim Generator to provide machine-readable AI transparency information. This assertion enables verifiable AI transparency, automated compliance verification, and trustworthy AI content at scale."*

Fields: `modelType` (required), `modelName`, `modelIdentifier`, `contentProfile.humanOversightLevel` ∈ {`fully_autonomous`, `prompt_guided`, `human_validated`}, `scientificDomain`. For Alaap the canonical manifest is `c2pa.actions` → `c2pa.created` with `digitalSourceType: http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia`, plus `c2pa.ai-disclosure` with `humanOversightLevel: prompt_guided`. · **HIGH**

### 3.2 Does it survive transcode? No — and that is the whole problem for our users

The spec has no blanket disclaimer; it addresses the failure structurally. The **hard binding** is a hash over the asset bits, so any re-encode invalidates it, and in practice the chunk is dropped outright — nothing in ffmpeg, LAME, libsndfile, Wwise or FMOD preserves a `C2PA` RIFF chunk or an ID3 GEOB.

C2PA's answer is **soft binding** (§9.3.1), verbatim: *"These soft bindings enable digital content to be matched even if the underlying bits differ… if a C2PA manifest is removed from an asset, but a copy of that manifest remains in a provenance store elsewhere, the manifest and asset may be matched using available soft bindings."* §2.4.1 names the result a **Durable Content Credential**. §9.3.1 also warns: *"a soft binding shall not be used as a hard binding."* · **HIGH**

> **Concretely for our users:** a WAV we hand a game developer, imported into Unity/Wwise, compressed to Vorbis/Opus/ADPCM and packed into a `.pak`, has **no manifest left** by the time a player hears it. This is not an edge case; it is the default path for our primary audience.

### 3.3 🔑 AudioSeal is a REGISTERED C2PA soft-binding algorithm

This is the finding that reconciles §2 and §3 and settles the "complementary or replacement?" question.

The C2PA soft-binding algorithm registry (`softbinding-algorithm-list.json`, `c2pa-org/specifications`) holds **53 entries, 21 of them audio-capable**, including:

| Algorithm id | Type | Added | Note |
|---|---|---|---|
| **`com.aiwatermark.audioseal.1`** | watermark | **2026-03-08** | **Meta FAIR AudioSeal**; `encodedMediaTypes: audio/mpeg, audio/wav, audio/flac, audio/mp4`; ships a Soft Binding Resolution API |
| `com.adobe.hiermark.A` | watermark | 2026-08-13 | Adobe Hierarchical Audio watermark |
| `com.microsoft.wavmark.1` | watermark | 2025-10-12 | Microsoft Responsible AI |
| `me.deepmark.audio.vigil.128` | watermark | 2026-06-05 | 128 bit/s neural, real-time CPU |
| `io.iscc.v0` | fingerprint | 2024-05-17 | ISO 24138 ISCC |

— <https://github.com/c2pa-org/specifications> (**official registry**) · **HIGH**
⚠️ The spec cites this list at `https://spec.c2pa.org/softbinding-alg-list`, which **302s to the specifications index and returns no JSON.** Build against the GitHub artefact, not the published URL. · **HIGH**

The soft-binding assertion schema is audio-aware: `soft-binding-timespan-map` scopes a binding by `start`/`end` in milliseconds, so bindings can be per-line.

> **This means the choice in §2.z is not merely defensible, it is the standards-endorsed one.** Picking AudioSeal is picking the algorithm C2PA itself lists as the way to make an audio Content Credential durable. Our watermark becomes the soft binding; our provenance DB becomes the manifest repository. **Nothing built now is thrown away later.**

### 3.4 Working implementations — the gap between spec and code

`contentauth/c2pa-rs` ships **both `LICENSE-MIT` and `LICENSE-APACHE`** (dual MIT / Apache-2.0 — GitHub's API reports `NOASSERTION` because of the dual licence, not because it is unlicensed). `contentauth/c2pa-python` is **Apache-2.0**. Both clear our licensing bar. Releases are healthy: `c2patool v0.27.16`, `c2pa v0.90.16`, `c2pa-python v0.37.8`, all 2026-08-27. · **HIGH**

**Audio formats actually implemented** (`docs/supported-formats.md`): `wav`, `mp3`, `flac`, `m4a`. **No Ogg, no Opus** — PR #2073 *"add OGG Vorbis and Opus C2PA manifest support"* opened 2026-04-18, last touched 2026-08-05, **still open**.

**Two things we would need are not implemented at all:**
- **Soft binding**: `sdk/src/assertions/soft_binding.rs` is only the serde struct. The resolution API is unmerged (PRs #1299 Aug 2025 and #2399 Aug 2026, both open); #2399's own description says *"local watermark/fingerprint embedding and extraction are deliberately out of scope here."*
- **`c2pa.ai-disclosure`**: zero code hits in c2pa-rs, five months after the assertion was published.

· **HIGH**

### 3.5 Signing and cost

- A C2PA manifest needs an X.509 v3 certificate meeting the C2PA **Certificate Policy**. The conformance programme and official Trust List launched mid-2025; the **Interim Trust List was frozen on 2026-01-01** ("No new certificates will be added").
- Conformance requires signing a legal agreement with C2PA, providing architecture evidence, and working with programme staff. Two assurance levels.
- CAs on the trust list as of April 2026: **DigiCert, SSL.com, Tauth Labs, Trufo**.
- **Cost:** SSL.com publishes a **free tier — one Level 1 Claim Signing Certificate (1 year) + 10,000 trusted timestamps/year** — but it requires a valid C2PA conformance record ID, i.e. conformance first. Premium is quote-only. **The dollar cost of conformance itself is UNVERIFIED.**
- **Self-signing is worse than nothing.** c2pa-rs defaults `trust.anchors` to `null`. Any conforming validator emits `signingCredential.untrusted` — *"The signing credential is not listed on any of the validator's applicable trust lists"* — and §15.7 says *"the claim shall be rejected."* A self-signed Alaap manifest reads as "unknown signer" everywhere it matters.

· **HIGH** (except conformance cost)

### 3.6 ISO status — do not call it a standard yet

**ISO/CD 22144 "Authenticity of information — Content credentials"** is at stage **30.99** (CD approved for registration as DIS) — **under development, not published**. · <https://www.iso.org/standard/90726.html> · **MEDIUM** (site refused direct fetch; re-confirm in a browser before external use)

### 3.7 Platform and storefront AI-disclosure rules

#### 3.7.1 🎯 Steam — the one that actually binds our users

Fetched from `partner.steamgames.com/doc/gettingstarted/contentsurvey`. The Steamworks Content Survey has three mandatory sections; the third is **Generative Artificial Intelligence Content**. Verbatim:

> *"We are aware that many modern game development environments have AI powered tools built into them. **Efficiency gains through the use of these tools is not the focus of this section. Instead, it is concerned with the use of AI in creating content that ships with your game, and is consumed by players. This includes content such as artwork, sound, narrative, localization, etc.**"*

**"sound" is named explicitly. Alaap output is squarely in scope.** · **HIGH**

**Pre-Generated** (build-time — our normal case), verbatim:
> *"Any kind of content that ships with your game and is consumed by players that is created with the help of AI tools during development. Under the Steam Distribution Agreement, **you promise Valve that your game will not include illegal or infringing content, and that your game will be consistent with your marketing materials.** In our prerelease review, we will evaluate the output of AI generated content in your game the same way we evaluate all non-AI content."*

**Live-Generated** (runtime API calls — the expensive case), verbatim:
> *"Any kind of content created with the help of AI tools while the game is running. In addition to following the same rules as Pre-Generated AI content, this comes with an additional requirement — in the Content Survey, **you'll need to tell us what kind of guardrails you're putting on your AI to ensure it's not generating illegal content.**"*

Also: *"Products on Steam must adhere to the content rules, regardless of whether it is disclosed"* — disclosure is not absolution. Some survey answers cannot be edited post-approval without contacting Steam Support. Adult-Only sexual content via Live-Generated AI is refused outright.

**What appears publicly on the store page** (verified from a live Steam product page):
> *"**AI Generated Content Disclosure** — The developers describe how their game uses AI Generated Content like this: …"* — free-text, developer-authored, public.

**Is there an AI-voice-specific Steam rule? No — none exists.** Synthesised voice is covered as "sound", nothing more. · **HIGH**

#### 3.7.2 itch.io

`itch.io/docs/creators/quality-guidelines`, verbatim:
> *"We ask that you accurately tag your project if it contains materials produced by generative AI by utilizing the **AI Disclosure** section on your project's edit page."*
> *"…we are **strictly enforcing disclosure for all game asset pages due to legal ambiguity around rights associated with Generative AI content. Failure to tag your asset page may result in delisting.**"*

Selecting yes yields an `AI Generated` tag plus sub-tags — **`AI Generated Graphics`, `AI Generated Sound`, `AI Generated Text & Dialog`, `AI Generated Code`**. Untagged AI assets *"will no longer be eligible for indexing on our browse pages."* Traditional game AI (pathfinding, procgen) is exempt.
**Alaap output → `AI Generated Sound`** (plus `AI Generated Text & Dialog` if we generated the script). · **HIGH**

#### 3.7.3 Consoles — mostly UNVERIFIED, and honestly so

| Platform | Published AI-disclosure requirement? | Confidence |
|---|---|---|
| **Microsoft / Xbox** | **YES.** Microsoft Store Policies v7.19 (published 2025-09-10, effective 2025-10-14) §**11.16 Live Generative AI Content**: products with *"dynamic content created by generative AI models in response to user inputs"* must disclose in metadata, declare in Partner Center, comply with all Store Policies, and *"provide a means for users to report inappropriate content to the developer."* Footnote 1 confirms "Store" includes the Xbox Store. **On its published text this does NOT capture pre-rendered voice lines** — the trigger is *dynamic* content generated *in response to user inputs*. Xbox Requirements (XR v16.3, 2026-07-01) contain **no AI XR**. | **HIGH** |
| **Sony / PlayStation** | **UNVERIFIED — no public policy exists.** `partners.playstation.net` exposes no public policy body. A developer must check the PS5 **Technical Requirements Checklist (TRC)**, the store-submission questionnaire, and the Global Developer & Publisher Agreement — all NDA-gated. **Do not represent Sony's position from public sources.** | **UNVERIFIED** |
| **Nintendo** | **UNVERIFIED for certification.** Lotcheck requirements are NDA-gated. The only Nintendo primary source on genAI is the 84th AGM shareholder Q&A (June 2024, Furukawa) — a statement about Nintendo's own practice that imposes no third-party duty. | **UNVERIFIED** |
| **Epic Games Store** | **No requirement found.** `legal.epicgames.com/epicgames/content-guidelines` has seven sections and zero mentions of AI/generative/synthetic content. Residual gap: the EGS Distribution Agreement is inside the Dev Portal, not publicly posted. | **MEDIUM** |

#### 3.7.4 Mobile stores

- **Google Play — two separate instruments; do not conflate them.** The *AI-Generated Content policy* scopes to **generative AI apps** and explicitly excludes apps that *"merely host AI-generated content and are unable to create content using AI"* — so a game shipping pre-rendered voice lines is **out of scope**; its requirement is in-app reporting/flagging, not disclosure. Two exceptions bite: apps that *"create voice and/or video recordings of real-life individuals using AI"*, and runtime TTS as a central feature. The separate *Declaring AI-generated content in Play Console* flow yields an **"AI-labeled"** designation but is scoped to **store-listing and promotional assets only** — an AI-voiced trailer is declarable; the same lines inside the game are not. · **HIGH**
- **Apple — no AI-content disclosure requirement for in-game audio.** The only operative clause is App Review Guideline **5.1.2(i)**: *"You must clearly disclose where personal data will be shared with third parties, including with third-party AI, and obtain explicit permission before doing so."* That binds **runtime** calls to our API, not build-time rendering. The 2025 age-rating overhaul treats AI features as a **rating input, not a disclosure duty**. · **HIGH**
- **IARC / ESRB:** no generative-AI Content Descriptor or Interactive Element found. **UNVERIFIED** — worth rechecking, since IARC drives ratings on Microsoft Store, Play, Epic and eShop simultaneously.

#### 3.7.5 Social & music platforms

| Platform | Requirement for synthetic AUDIO | Confidence |
|---|---|---|
| **TikTok** | **The only explicit safe harbour for generic TTS.** Disclosure **required** when *"AI-generated audio mimics the voice of a real person"*; **not needed** when *"Using generic text-to-speech (TTS) narration, when the TTS isn't a recognizable voice of a known individual."* AIGC defined as *"Any image, video, or audio made or changed by AI."* Reads/writes Content Credentials (first video platform, May 2024); **audio-only Credentials were "coming soon" and remain UNVERIFIED as shipped.** | **HIGH** |
| **Meta** | **Strictest; no real-person carve-out.** *"We require people to disclose, using our AI-disclosure tool, whenever they post organic content with photorealistic video or **realistic-sounding audio that was digitally created or altered**"* — *"we may apply penalties if they fail to do so."* Per the Oversight Board's Iraqi Kurdistan audio decision (2025-06-24) Meta stated it can *"only automatically identify and label static images, not video or audio content"* → **uploader self-disclosure is the only functioning mechanism for audio.** | **HIGH** |
| **YouTube** | Disclosure required *"when they use AI to meaningfully alter or generate photorealistic content"* under three triggers, all about depicting real people/events/scenes. Exemptions include *"Cloning one's own voice to create voice overs or dubs."* Generic synthetic narration fails all three triggers and **appears** out of scope — but YouTube never says so → **UNVERIFIED**. YouTube **auto-labels content containing C2PA metadata**, and those labels *"can not be adjusted."* Likeness detection is **face-only**; *"We aim to extend likeness detection to audio in the near future."* | **HIGH** / gap UNVERIFIED |
| **Spotify** | Supports *"the new industry standard for AI disclosures in music credits, developed through DDEX"*, letting rights holders indicate *"whether that's AI-generated vocals, instrumentation, or post-production"* (newsroom, 2025-09-25). **Not punitive.** Impersonation is consent-gated: *"Vocal impersonation is only allowed in music on Spotify when the impersonated artist has authorized the usage."* April 2026 beta surfaces AI credits in-app, caveating *"the absence of a credit doesn't mean AI wasn't used."* **DDEX detail:** there is no separately named "AI disclosure standard" — it is an extension inside **ERN v4.3.1**, a coarse recording-level signal. Exact XML element name **UNVERIFIED**. | **HIGH** / DDEX detail MEDIUM |

#### 3.7.6 SAG-AFTRA 2025 Interactive Media Agreement — relevant, but not to us

Ratified July 2025 (**95.04 % to 4.96 %**), ending the video-game strike. Includes *"consent and disclosure requirements for A.I. digital replica use"*; consent must be *"set forth in writing in a clear and conspicuous manner"* and is invalidated if the use no longer fits the description given at consent time. Scope: *"The Visual Digital Replica covers scripted cinematic scenes only… All material, whether in-game or cinematic, is covered by a Vocal Digital Replica."* **Real Time Generation** carries a **7.5× scale** minimum.

**Critically: this is a collective bargaining agreement binding signatory companies — not a law, and not binding on an indie studio using Alaap with no union talent.** It bites only when a Alaap voice is *derived from* a union performer's recorded performance (which our architecture forbids), or when a signatory studio uses us. **It is a boundary condition on features we have already ruled out, not a constraint on from-description synthesis.**
— sagaftra.org (**official pages, but DataDome-blocked to direct fetch; gathered via search index**) · **MEDIUM — re-verify in a browser before any customer-facing use**

### 3.8 What a Steam-shipping developer must do, and the five artefacts we owe them

**They must:** answer the Steamworks Content Survey's Generative-AI section, declare **Pre-Generated** (or **Live-Generated** if they call our API at runtime), write the free-text store-page disclosure, and stand behind the Distribution Agreement promise that the content is not illegal or infringing. On itch.io, set AI Disclosure → yes → **Sound**. If they call us at runtime they additionally trigger **Microsoft Store 11.16**, **Google Play's AI-Generated Content policy**, and **Apple 5.1.2(i)**.

**We must ship:**

1. **A paste-ready Steam disclosure paragraph**, auto-filled per project, in the register real store pages use — e.g. *"Character voice lines in this game were generated using Alaap, a text-description-to-speech synthesis service. Voices are synthetic and are not derived from, and do not replicate, any identifiable real person's voice. All generated dialogue was written and reviewed by the development team."*
2. **A rights attestation** the developer can rely on for the "not illegal or infringing" promise: a per-render statement that the voice is model-synthesised, not cloned from an identifiable person, with our training-data basis. Without this we silently push their Distribution Agreement risk onto them.
3. **A per-render provenance record** with a stable ID — render UUID, timestamp, model + version, description prompt hash, output SHA-256, account, licence terms — retrievable by ID and exportable as JSON/CSV for a whole project.
4. **The itch.io tag mapping** and a one-click "AI Generated Sound" checklist.
5. 🔑 **A runtime-vs-build-time warning in the API docs.** This is the highest-value thing we can tell a user: **build-time rendering triggers almost nothing; runtime generation triggers Microsoft 11.16, Google Play, Apple 5.1.2(i), and Steam's Live-Generated guardrails question.** Most developers do not know this line exists. Putting it in the docs is free and prevents a whole class of certification surprise.

### 3.9 Engineering verdict on C2PA

> **Do not build C2PA manifest signing for v1. Build watermark + server-side provenance log. Revisit in ~12 months.** · **Confidence HIGH**

**Against C2PA now:**
- **The delivery path destroys it.** Our audio goes into Unity/Wwise/FMOD, is transcoded to Vorbis/Opus/ADPCM and packed into an archive. The RIFF `C2PA` chunk and the ID3 GEOB survive none of it.
- **No storefront reads it.** Steam, itch, Epic, PlayStation, Nintendo, Play, Apple — zero read Content Credentials. **Steam's requirement is a text box a human types into**; a manifest satisfies nothing there.
- **Trust costs money and organisational effort**, and self-signing yields `signingCredential.untrusted` — arguably worse than no manifest at all.
- **The libraries lack exactly the parts we would need**: Opus/Ogg unmerged, soft binding an unimplemented struct, `c2pa.ai-disclosure` absent five months after publication.

**For watermark + log now:** AudioSeal is MIT, **registry-listed as a C2PA soft binding**, and survives transcode by design — the only property that matters for game shipping. A watermark plus a signed server-side render log is a complete provenance story on its own: detect → recover → look up. That is what a Steam moderator, a DMCA claimant or a journalist actually needs.

**Where C2PA does earn its keep — and the v1.5 scope: the web download, not the game build.** The WAV/MP3 a creator downloads from voiceforge.com is exactly the artefact YouTube auto-labels from. One `c2pa-python` call gets our audio auto-labelled without the creator touching a disclosure toggle. Sequence:

| When | Do |
|---|---|
| **Now (S9)** | AudioSeal on every render · signed provenance log · public lookup endpoint · the five developer artefacts above |
| **+1 quarter** | Apply for C2PA conformance (Level 1); SSL.com's free tier then costs nothing |
| **+2 quarters** | Emit signed manifests on **web-download** WAV/MP3/M4A via `c2pa-python`: `c2pa.created` + `digitalSourceType: trainedAlgorithmicMedia`, `c2pa.soft-binding` referencing `com.aiwatermark.audioseal.1`, `c2pa.ai-disclosure` with `humanOversightLevel: prompt_guided` (hand-rolled CBOR until the SDK catches up) |
| **Watch** | c2pa-rs PR **#2073** (Ogg/Opus) and **#2399** (soft-binding resolution) — merging either is the signal for deeper investment |

**Complementary, not a replacement.** C2PA's own FAQ: *"C2PA supports durable credentials via soft bindings—such as invisible watermarking or fingerprinting—that can help rediscover the associated Content Credential even if it's removed from the file."* The two layers do different jobs: the manifest carries rich, signed, machine-readable claims and is fragile; the watermark carries one bit and is durable. **We need the durable bit first.**

---

## 4. Regulation: what we must actually do

> ⚠️ **NOT LEGAL ADVICE.** This is compliance-*engineering* scoping against primary sources — statute text, regulator PDFs, court orders. It tells you what to build. It does not tell you whether you are safe. Get counsel before public launch, and specifically before EU or India availability.

**Access caveat carried through from the research pass:** EUR-Lex sits behind an AWS WAF challenge and could not be fetched directly. EU statutory text below comes from (a) **official European Commission PDFs on ec.europa.eu that reproduce the operative text verbatim** (HIGH) and (b) the Future-of-Life AI Act Explorer, a **secondary reproduction** (MEDIUM-HIGH unless corroborated by (a)).

### 4.1 EU AI Act Article 50

#### The obligation on us, verbatim

**Article 50(2)** — corroborated HIGH by the Commission's own Guidelines and Code of Practice PDFs:

> *"Providers of AI systems, including general-purpose AI systems, generating synthetic audio, image, video or text content, shall ensure that the outputs of the AI system are **marked in a machine-readable format and detectable as artificially generated or manipulated**. Providers shall ensure their technical solutions are **effective, interoperable, robust and reliable** as far as this is technically feasible, taking into account the specificities and limitations of various types of content, the costs of implementation and the generally acknowledged state of the art…"*

**We are squarely in scope.** No exemption applies: the assistive-editing carve-out does not reach us, and the Commission Guidelines list *"Synthesis of realistic speech in a specific person's voice"* as a semantic change requiring marking. The only audio carve-out is accessibility technology (AAC devices, customised neural voices for disabled users) — not a general TTS product. · **HIGH**

**Article 50(4)** puts a *separate* duty on **deployers** — i.e. our game-developer customers:
> *"Deployers of an AI system that generates or manipulates image, audio or video content constituting a deep fake, shall disclose that the content has been artificially generated or manipulated… Where the content forms part of an **evidently artistic, creative, satirical, fictional** or analogous work or programme, the transparency obligations… are limited to disclosure of the existence of such generated or manipulated content in an appropriate manner **that does not hamper the display or enjoyment of the work**."*

**Article 50(5):** information must be given *"in a clear and distinguishable manner at the latest at the time of the first interaction or exposure"* and *"shall conform to the applicable accessibility requirements."*

#### ⚠️ Does a NOVEL voice count as a "deep fake"? Probably yes — and this corrects a likely assumption

The Commission's Article 50 Guidelines (official, 20 July 2026) read the Art 3(60) definition broadly:

> *"it is sufficient for simulated persons, objects, places, entities or events to resemble someone or something that exists, **can plausibly exist** or could have plausibly existed in reality to be considered a deep fake"*

and *"'Persons' is to be understood as realistic, human beings (including digital replicas of real persons, **realistic AI-generated human avatars or personas, and personal characteristics or expressions, such as image, voice, behaviour, performances** etc.)."*

> **A novel-but-realistic synthetic voice can be a "deep fake" for the purposes of the deployer duty.** Our "it's nobody's voice" argument does not exit Art 50(4). For game developers the artistic/fictional-work limitation softens it to "disclosure of the existence" in a non-intrusive manner — a credits line or store-page note, not an in-game interruption. **This makes deployer guidance a product deliverable, not a courtesy** (§3.8). · **HIGH**

#### Dates — verified, and the Digital Omnibus did NOT delay Article 50

| Fact | Date | Confidence |
|---|---|---|
| **Article 50 applies from** | **2 August 2026** — Art 113 general date; Chapter IV is in **no** exception list | MEDIUM-HIGH (secondary text) / **HIGH** via Commission Guidelines ¶2: *"These transparency obligations apply two years after the entry into force of the AI Act, i.e. as from 2 August 2026."* |
| **Digital Omnibus on AI = Regulation (EU) 2026/1744** | published OJ 24 July 2026, in force 27 July 2026 | **HIGH** (Commission news page) |
| What the Omnibus postponed | **Annex III high-risk → 2 Dec 2027; Annex I → 2 Aug 2028.** **It did not postpone Article 50.** | **HIGH** |
| Transitional relief, new Art 111(4) | Providers of synthetic-content systems **placed on the market before 2 August 2026** have until **2 December 2026** to comply with Art 50(2) | MEDIUM-HIGH |

> **Operational conclusion: Alaap is not yet on the market, so no grace period applies. Marking must work on day one of EU availability.** · **HIGH**

#### Territorial scope — being outside the EU does not help

**Art 2(1)(a):** applies to *"providers placing on the market or putting into service AI systems… in the Union, **irrespective of whether those providers are established or located within the Union or in a third country**."*
**Art 2(1)(c):** also to third-country providers *"where the output produced by the AI system is used in the Union."*
**→ An India- or US-hosted Alaap serving EU game developers is in scope.** · **MEDIUM-HIGH**

#### 🔑 What technical form must the marking take? Two layers, and metadata alone is not enough

**Recital 133** is technology-neutral on its face — *"watermarks, metadata identifications, cryptographic methods for proving provenance and authenticity of content, logging methods, fingerprints or other techniques."*

**But the Code of Practice on Transparency of AI-Generated Content — final version published 10 June 2026, ~190 signatories by end July 2026 — raises the bar specifically for audio.** Measure 1.1, verbatim:

> *"So long as no single marking technique can, under the state of the art, ensure by itself compliance with the four requirements in Article 50(2) AI Act of effectiveness, interoperability, robustness, and reliability **for audio, images, video, and containerised text**… Signatories will implement a **multi-layered marking approach** to ensure that the outputs of their generative AI systems are marked with **at least two layers of machine-readable marking**"*

- **Sub-measure 1.1.1 — digitally signed metadata:** *"All recorded information will be digitally signed and time-stamped… in a secure and tamper-evident manner."*
- **Sub-measure 1.1.2 — imperceptible watermark:** embedded *"in a manner that is difficult for it to be separated from the content."*
- **Sub-measure 1.1.3 — fingerprinting/logging is optional and insufficient alone:** *"relying on fingerprinting or logging alone is not considered sufficient."*

> **This settles the §2-vs-§3 question at the regulatory level: for audio you need signed provenance metadata AND an inaudible watermark. Our provenance log alone does not satisfy the Code. Neither does the watermark alone.** · **HIGH**

The Code is voluntary, but the Commission states non-adherents *"will have to demonstrate compliance… through alternative equivalently adequate means."* · **HIGH**

#### 🔑 Commitment 2 — we must ship a public DETECTOR

> *"Signatories will make available a detection solution… to enable deployers, users… end-users exposed to the content, and other legitimate parties (such as competent authorities, independent researchers, civil society and media organisations) to verify whether content has been generated or manipulated by their AI system."*

Delivery may be a public specification, downloadable software, or a cloud API. It must be **free of charge**, except: *"Signatories with fewer than 1,000,000 monthly users of their generative AI system, whose detection solution incurs substantial operational costs… may charge a fee… in cases where requests from a single user exceed a reasonable threshold."* Free access to regulators, media, fact-checkers, researchers and civil society is **always** required. · **HIGH**

> **This is a direct architectural requirement, and it is why AudioSeal's public detector is an asset rather than only a liability (§2.1.5).** We can point third parties at the public AudioSeal detector *and* run a hosted `POST /detect` endpoint. Our small size buys a fee allowance, not an exemption.

**SME proportionality** exists but is soft — CoP Recital (g): *"simplified ways of compliance for SMEs and SMCs, including startups, should be possible, in a proportionate manner."*

#### Harmonised standards — none yet

**No published CEN-CENELEC JTC 21 harmonised standard on marking synthetic content as of Sept 2026.** CEN-CENELEC announced (23 Oct 2025) an accelerated route targeting availability **Q4 2026**. **The Code of Practice is the de facto compliance route today.** · **MEDIUM**

This matches the argument in *Watermarking Without Standards Is Not AI Governance* (arXiv 2505.23814): watermarking without standards, verification infrastructure, and detection access does not constitute governance. · **MEDIUM**

#### Penalties

**Art 99(4)(g)** — breach of Art 50: fines *"up to EUR 15 000 000 or, if the offender is an undertaking, up to 3 % of its total worldwide annual turnover… whichever is higher."*
**Art 99(6)** — for SMEs and start-ups the fine is capped at *"whichever thereof is **lower**."*
> **For us the practical cap is 3 % of turnover, not €15 m.** · **MEDIUM-HIGH**

### 4.2 United States

| Instrument | Status as of 2026-09-02 | Binds Alaap? | Confidence |
|---|---|---|---|
| **NO FAKES Act** | **S.4591 (2026)**, introduced 20 May 2026, reported favourably 18/24 Jun 2026, **placed on Senate Legislative Calendar, Calendar No. 446**. Identical House bill **H.R. 8915**. *(S.1367 of 2025 is superseded.)* **Not passed either chamber. Not law.** | **Not yet — but design for it** | **HIGH** (govinfo BILLSTATUS) |
| **TAKE IT DOWN Act** (PL 119-12) | Law. Every operative term is *"intimate **visual** depiction"*. **No audio in scope.** | **No** | **HIGH** |
| **TN ELVIS Act** | In force **1 July 2024**. | **Conditionally — see below** | **HIGH** |
| **FCC / TCPA** | Declaratory Ruling **FCC 24-17, CG Docket 23-362**, adopted 2 Feb 2024: *"the TCPA's restrictions on the use of 'artificial or prerecorded voice' encompass current AI technologies that generate human voices."* NPRM **FCC 24-84** adopted 7 Aug 2024 proposed AI-call disclosure. **No final rules adopted in that docket through Aug 2026** (Federal Register query returns nothing). | **No — binds the caller, not the TTS vendor** | **HIGH** |
| **CA SB 942 (AI Transparency Act), as amended by AB 853** | Chaptered 13 Oct 2025; **operative 2 August 2026**. | **Only above 1,000,000 monthly users** | **HIGH** |
| **Colorado AI Act** | ⚠️ **SB 24-205 was repealed and replaced by SB 26-189 (signed 14 May 2026); it never took effect.** New ADMT framework effective 1 Jan 2027; a voice generator makes no "consequential decisions". | **No** | **MEDIUM** |
| **Utah AIPA** | Disclosure duty attaches to consumer-facing *interactions* on request and to regulated occupations. | **No** | **MEDIUM** |

#### 🔑 The two US provisions that actually name tool providers

**NO FAKES S.4591 § 2(c)(2)(B)** creates liability for making available a product or service that:
> *"(i) is **primarily designed** to produce 1 or more digital replicas of a specifically identified individual… (ii) has **only limited commercially significant purpose or use** other than to produce a digital replica of a specifically identified individual…; or (iii) is **marketed, advertised, or otherwise promoted**… as a product or service designed to produce a digital replica of a specifically identified individual…"*

with a safe harbour at § 2(d)(1)(A) for products **not** described in (c)(2)(B). "Digital replica" (§ 2(a)(2)(A)) means a representation *"**readily identifiable as the voice or visual likeness of an individual**."*

**TN ELVIS Act, new § 47-25-1105(a)(3):**
> *"A person is liable to a civil action if the person **distributes, transmits, or otherwise makes available an algorithm, software, tool, or other technology, service, or device, the primary purpose or function of which is the production of an individual's photograph, voice, or likeness** without authorization…"*

Note also that TN's definition of "voice" expressly covers simulation: *"a sound in a medium that is readily identifiable and attributable to a particular individual, **regardless of whether the sound contains the actual voice or a simulation of the voice**."* And § 47-25-1107(c) was amended to *"had knowledge or reasonably should have known"* — a low scienter bar for secondary actors.

> **Both statutes turn on the same hinge: PRIMARY PURPOSE and MARKETING.** Our architecture (novel voices, no upload, no target speaker) lands on the safe side of "primarily designed" and "limited commercially significant purpose". **The entire remaining exposure is prong (iii): how we market.**
>
> 🔴 **Therefore a hard, permanent product rule: never market, name, tag, preset, or template a voice by reference to a real person. No "sounds like X". No celebrity-inspired presets. No "our Morgan-Freeman-style narrator". Not in copy, not in a preset name, not in an example prompt, not in a demo video.** One marketing page can move us from the safe harbour into the liability clause. This is the cheapest compliance control in this document and the easiest to violate by accident.
>
> **HIGH** that the statutes turn on this. **MEDIUM** on how a court applies it — **no interpreting case law on § 47-25-1105(a)(3) was located.**

**California SB 942 is the one to engineer for**, because it converges with the EU Code of Practice. "Covered provider" = a GenAI system with *"**over 1,000,000 monthly visitors or users**"* publicly accessible in California. Requirements: a **free public AI detection tool** (accepts uploads **and** URLs, exposes an **API**), an optional **manifest disclosure**, and a mandatory **latent disclosure** conveying provider name, system name/version, creation timestamp and a unique identifier, *"permanent or extraordinarily difficult to remove"* where technically feasible. **Covers audio.** · **HIGH**

### 4.3 India — the most prescriptive regime, and it binds generation tools directly

**Primary source: the official MeitY consolidated text, "The Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Rules, 2021 [updated as on 10.02.2026]" (meity.gov.in PDF). Amending instrument G.S.R. 120(E) dated 10.02.2026; in force 20 February 2026.** · **HIGH** on text, **MEDIUM** on the in-force date.

**New Rule 2(1)(wa) — "synthetically generated information":**
> *"audio, visual or audio-visual information which is artificially or algorithmically created, generated, modified or altered using a computer resource, in a manner that such information appears to be real, authentic or true and **depicts or portrays any individual or event in a manner that is, or is likely to be perceived as indistinguishable from a natural person or real-world event**"*

**New Rule 3(3) — and note it is NOT limited to significant social media intermediaries:**
> *"**Where an intermediary offers a computer resource which may enable, permit, or facilitate the creation, generation, modification… of information as synthetically generated information**, it shall ensure that,—
> (i) it deploys reasonable and appropriate technical measures, including automated tools… to not allow any user to create… any such synthetically generated information that violates any law… including… (I) child sexual exploitative and abuse material, non-consensual intimate imagery, or obscene content; (II) …any false document or false electronic record; (III) …explosive material, arms or ammunition; or (IV) **falsely depicts or portrays a natural person or real-world event by misrepresenting, in a manner that is likely to deceive, such person's identity, voice, conduct, action, statement**…; and
> (ii) every such information not covered under sub-clause (i)… is prominently labelled… **or, in the case of audio content, through a prominently prefixed audio disclosure**, that can be used to immediately identify that such information is synthetically generated information… and such information **shall be embedded with a permanent metadata or other appropriate technical provenance mechanisms, to the extent technically feasible, including a unique identifier, to identify the computer resource of the intermediary** used to create such information;
> (b) the intermediary **shall not enable the modification, suppression or removal of the label, permanent metadata, including the unique identifier**."*

**New Rule 3(1)(ca)** additionally requires informing users that misuse for prohibited synthetic content *"may attract penalty or punishment"* under the IT Act, BNS 2023, POCSO, the Representation of the People Act 1951, IRWA 1986, POSH 2013 and the Immoral Traffic (Prevention) Act 1956.

#### Three corrections to widely-repeated claims

| Claim in circulation | Verdict |
|---|---|
| "India requires a label covering 10 % of display area / the first 10 % of audio duration" | ❌ **DROPPED from the final rules.** The phrase "per cent" appears nowhere in this context in the notified consolidated text. The requirement is a *"prominently prefixed audio disclosure"* with **no quantified duration**. · **HIGH** (exhaustive grep of the official consolidated PDF) |
| "It only binds significant social media intermediaries" | ❌ **Rule 3(3) binds any intermediary offering a generation-capable computer resource.** Only Rule 4(1A) (user self-declaration before display) is SSMI-only. · **HIGH** |
| "It's still a draft" | ❌ Notified 10 Feb 2026. · **HIGH** |

#### 🔴 The India requirement that collides with our product

> **A "prominently prefixed audio disclosure" on a 1.5-second game dialogue line is product-destroying.** A spoken "this audio is AI-generated" preamble is longer than the line itself, and shipping it inside a game would be absurd.
>
> **Residual ambiguity that may rescue us — and it is genuinely arguable both ways:** Rule 3(3)(a)(ii) applies to information *"not covered under sub-clause (i)"* that is **synthetically generated information** as defined in 2(1)(wa) — which requires the content to *"depict or portray **any individual** or event"* in a way perceived as indistinguishable from a natural person or real event. A novel voice speaking invented game dialogue arguably depicts **no individual and no real-world event**. Against that: "indistinguishable from a natural person" reads as a **realism** test rather than an **identity** test, and our output is realistic by construction.
> · **MEDIUM — genuinely unresolved. This is a counsel question, not an engineering one, and it is the single highest-value legal question in this document (§9).**
>
> **Engineering posture until it is settled:** make the audible prefix a **per-jurisdiction, per-delivery-channel toggle**, default OFF for game-asset export and default ON for any consumer-facing playback surface we operate in India. Build the switch now; it is trivial at design time and expensive to retrofit. The metadata + unique-identifier + non-removability requirements we should simply implement unconditionally — they cost nothing and satisfy EU, California and India at once.

Also note: whether a pure generation service is an "intermediary" under IT Act s.2(1)(w) at all is contestable, but **Rule 3(3) is plainly drafted to catch generation tools.** Extraterritorial reach over a foreign-hosted service serving Indian users: **MEDIUM** (IT Act s.75).

#### Personality rights — an Indian court has already enjoined a tool provider

**Asha Bhosle v. Mayk Inc. & Ors.**, Bombay High Court, IA(L) 30382/2025 in Comm IP Suit (L) 30262/2025, **order of 29 September 2025**. Defendant No. 2 was an AI voice platform. ¶15, verbatim:

> *"**In my prima facie view, making AI tools available to enable the conversion of any voice into that of a celebrity without his/her permission would constitute a violation of the celebrity's personality rights. Such tools facilitate the unauthorized appropriation and manipulation of a celebrity's voice**, which is a key component of their personal identity and public persona…"*

· **HIGH** (court-uploaded order read in full)

> **The ratio is narrow in our favour: it concerns tools "to enable the conversion of any voice INTO THAT OF a celebrity."** Our no-upload, no-target-speaker design is exactly the posture this order does not reach. But it establishes that **Indian courts will enjoin a tool provider, not merely a user** — so the architecture is doing real defensive work, and any feature that erodes it (reference-audio cloning, similarity search, celebrity presets) carries direct injunction risk in India. · **HIGH** on the quote, **MEDIUM** on extension.

Supporting line, all interim orders with no final adjudication: **Anil Kapoor v. Simply Life India** (Delhi HC, CS(COMM) 652/2023, 20 Sep 2023); **Arijit Singh v. Codible Ventures** (Bombay HC, 26 Jul 2024 — first Indian AI voice-cloning order); **Aishwarya Rai Bachchan** (Delhi HC, CS(COMM) 956/2025, 9 Sep 2025); **Jackie Shroff** (2024). Doctrine: common-law personality/publicity rights + passing off + dilution + Copyright Act s.38-B performers' moral rights. **India has no statutory right of publicity.** · **MEDIUM**

#### DPDP Act 2023 + DPDP Rules 2025

- **Rules notified 14 November 2025**, with *"an eighteen-month period for phased compliance"* (PIB). · **HIGH**
- Phase-in: notification provisions Nov 2025; **Consent Managers (Rule 4) 13 Nov 2026**; **Rules 3, 5–16, 22–23 on 13 May 2027**. · **MEDIUM**
- Penalties: up to **₹250 crore** (security safeguards), **₹200 crore** (breach notification; children's data), **₹50 crore** otherwise. · **HIGH**
- **What binds us:** Alaap is a **Data Fiduciary** for account data *and for user-submitted voice descriptions and dialogue text*. Obligations: standalone plain-language consent notice, purpose limitation, minimisation, security safeguards, breach notification to the Board and affected principals, a published contact point, and **verifiable parental consent for users under 18** plus a ban on behavioural advertising to children. **Age assurance is a real product requirement**, and one a game-developer-facing tool will genuinely encounter. · **MEDIUM-HIGH**

### 4.4 THE COMPLIANCE CHECKLIST

**MUST** = obligation with a citation. **SHOULD** = risk reduction. Stage tags map to the phase plan in [`GRAND-PLAN.md`](GRAND-PLAN.md).

#### Pre-launch (build during S9, before any public exposure)

| # | Item | Basis | Notes for the build |
|---|---|---|---|
| 1 | **MUST** Embed an **imperceptible audio watermark** in every generated file, unconditionally, inside the render process. | EU Art 50(2); CoP 1.1.2; CA §22757.3 latent disclosure | AudioSeal, §2.z. 7.4 ms. No opt-out, no flag. |
| 2 | **MUST** Embed **digitally signed, time-stamped provenance metadata**: AI-generated flag, provider name, system name + version, creation timestamp, unique identifier. **Two layers, not one.** | EU CoP 1.1.1 (*"relying on fingerprinting or logging alone is not considered sufficient"*); CA §22757.3; India r.3(3)(a)(ii) | This is the manifest sidecar of §3.9. Sign it even before C2PA conformance. |
| 3 | **MUST** Embed a **permanent unique identifier identifying Alaap as the generating resource**, and **do not expose any UI that removes the label or metadata**. | India r.3(3)(a)(ii)+(b) | Note (b) forbids *enabling* removal — so no "strip metadata" export option. |
| 4 | **MUST** Ship a **detection mechanism** for every marking layer — public spec, downloadable library, or API — free to regulators, media, fact-checkers, researchers and civil society without volume limits. | EU CoP Commitment 2; CA §22757.3 | `POST /detect` + point at the public AudioSeal detector. Fee allowance below 1M monthly users, but only for excessive single-user volume. |
| 5 | **MUST** Deploy **automated prohibited-use controls** on generation: CSAM/NCII/obscene, false documents/records, arms & explosives, and **audio that misrepresents a real person's identity, voice, conduct or statements**. | India r.3(3)(a)(i)(I)–(IV) | This is §5.3, and India makes it mandatory rather than merely prudent. |
| 6 | **MUST** Put the **statutory misuse warning** in ToS and in the generation UI, naming IT Act, BNS 2023, POCSO, RPA 1951, IRWA 1986, POSH 2013, ITPA 1956. | India r.3(1)(ca) | Literal list; copy it. |
| 7 | **MUST** DPDP: standalone consent notice, purpose limitation, minimisation, security safeguards, breach-notification runbook, published contact point. | DPDP Act 2023 + Rules 2025 | Build now; Rule 3 binding 13 May 2027. |
| 8 | **MUST** DPDP: **age gate + verifiable parental consent under 18**; no behavioural advertising to children. | DPDP Act 2023 | Real product work — do not defer. |
| 9 | 🔴 **MUST NOT** market, name, tag, preset, or template any voice by reference to a real person. No "sounds like X" anywhere: copy, preset names, example prompts, demo videos, SEO. | NO FAKES S.4591 §2(c)(2)(B)(iii); TN §47-25-1105(a)(3); *Asha Bhosle* ¶15 | **Permanent.** Add to the marketing review checklist, not just engineering. |
| 10 | **SHOULD** Build the **India audible-prefix toggle** — per-jurisdiction, per-delivery-channel, default OFF for game-asset export, default ON for any consumer surface we operate in India. | India r.3(3)(a)(ii) | Cheap now, expensive later. See the §4.3 ambiguity. |
| 11 | **SHOULD** Keep the **no-upload architecture** and document it as a **design invariant enforced at the API boundary**. | Strongest defence under TN "primary purpose", NO FAKES safe harbour §2(d)(1)(A), *Asha Bhosle* | §8. Enforce in code, per the brief's own §12.5 principle. |
| 12 | **SHOULD** Run a **similarity screen** of every minted identity against an ASV index of high-risk public figures; block or re-mint on near-match; **log the check**. | Mitigates TN, NO FAKES, Indian personality rights, Midler/Waits | §5.2. The log is the evidence of good faith. |
| 13 | **SHOULD** Publish an **Article 50 provider statement** and a **deployer guidance page** telling game developers their Art 50(4) duty (with the artistic-work limitation) and their TCPA exposure if output is repurposed into calls. | AI Act Art 50(4); FCC 24-17 | Merge with the five artefacts in §3.8. |
| 14 | **SHOULD** Document marking performance by **internal testing against CoP Commitment 3** (effectiveness / interoperability / robustness / reliability). The Code expressly permits self-testing pending recognised benchmarks. | CoP Measure 4.2 | **This is exactly the §2.y experiment.** Run it once; it serves engineering *and* compliance. |

#### At launch

| # | Item | Basis |
|---|---|---|
| 15 | **MUST** Marking live from **day one** of EU availability. Art 111(4)'s 2 Dec 2026 relief applies only to systems on the market **before 2 Aug 2026** — not to us. | Art 111(4) |
| 16 | **SHOULD** **Sign the Code of Practice on Transparency of AI-Generated Content.** Non-signatories must demonstrate *"alternative equivalently adequate means"* — a worse position for a small team than simply adhering. | Commission CoP page |
| 17 | **SHOULD** Adopt the **EU icon** for labelling AI-generated content in the UI and export manifest. | Commission EU-icons policy |
| 18 | **SHOULD** Notice-and-takedown channel + repeat-infringer termination policy. | Mirrors NO FAKES §2(d)(1)(B) safe-harbour conditions; Indian orders routinely require 72-hour takedown |
| 19 | **SHOULD** Ship the **storefront disclosure artefacts** of §3.8 (Steam paragraph, rights attestation, provenance export, itch tag map, runtime-vs-build-time warning). | Steam / itch.io policy |

#### At scale

| # | Trigger | Item |
|---|---|---|
| 20 | **> 1,000,000 monthly visitors/users, publicly accessible in California** | **MUST** Full **CA AI Transparency Act** compliance: free public AI detection tool accepting **uploads and URLs with an API**; manifest disclosure option; latent disclosure. Operative since 2 Aug 2026. |
| 21 | **> 1,000,000 monthly users** | **MUST** Remove any detection-tool fee — the CoP fee allowance ends at this threshold. |
| 22 | If notified as a Significant Data Fiduciary | **MUST** DPIA, audits, India-resident DPO. |
| 23 | Ongoing | **SHOULD** Track: CEN-CENELEC JTC 21 harmonised standards (expected Q4 2026); **NO FAKES S.4591 floor vote (Calendar No. 446)**; FCC CG 23-362; Colorado SB 26-189 (1 Jan 2027); CA §§22757.3.1 (1 Jan 2027) / 22757.3.3 (1 Jan 2028); DPDP Rule 4 (13 Nov 2026) and Rules 3, 5–16 (13 May 2027). |

> ### The single biggest engineering item
> **The two-layer marking stack — signed provenance metadata + inaudible audio watermark — plus a public detection endpoint.** It satisfies **EU Art 50(2) and the Code of Practice, California SB 942, and India's metadata/unique-identifier rule simultaneously.** Build it once, before launch. Everything else in this checklist is policy text, a toggle, or a log.

---

## 5. Description-layer gating

### 5.1 Named-person refusal: options and false-positive risk

#### What the vendors actually publish — nobody has solved this

| Vendor | Gates *cloning*? | Gates *describing a real person*? |
|---|---|---|
| **ElevenLabs** | Yes — *"blocking the cloning of celebrity and other high risk voices, and requiring technological verification for access to our Professional Voice Cloning tool"* | **Policy text arguably yes; technical safeguards no; Voice Design docs entirely silent** |
| **Hume AI** | ⚠️ **No longer — they reversed** | No |
| **OpenAI** | Declined to release Voice Engine broadly | N/A (never shipped) |
| **Microsoft** | Yes — Azure Custom Neural Voice is **Limited Access**, *"only customers managed by Microsoft"*, use confined to registered use cases; VALL-E 2 never released | N/A |
| **Google** | Yes — Chirp 3 Instant Custom Voice *"restricted to allow-listed users"*; Gemini TTS output SynthID-watermarked | No published policy found |
| **Resemble AI** | Yes — consent workflow for Professional Clone | No — *and* they open-sourced **Chatterbox**, an **MIT-licensed** zero-shot cloner |

> **Nobody in the market has published a description-layer named-person policy. We would be writing one from scratch — a differentiator and a liability at once.** · **HIGH**

**ElevenLabs' policy drafting is worth copying**, because it is modality-neutral. Prohibited Use Policy §5 forbids *"creating or using ElevenLabs audio output to intentionally **replicate the voice of another person**: (a) without consent or legal right"* — it does **not** say "clone from audio", so on its face it already covers description-based replication. §6 additionally forbids impersonating *"political candidates or elected government officials, **regardless of whether authorization was obtained**."* Enforcement is described as *"a combination of automated systems, user reports, and human review."* · **HIGH**
— <https://elevenlabs.io/use-policy> · <https://elevenlabs.io/safety> · <https://elevenlabs.io/docs/eleven-creative/voices/voice-design>

> ⚠️ **CORRECTION TO A PREMISE THE PROJECT MAY BE CARRYING: Hume AI has reversed its no-cloning stance.** Voice Control (Dec 2024) framed cloning as *"riskier, take more time, and often compromise on quality"* — a **product-quality** argument, not a safety one. By OCTAVE (Jan 2025) Hume ships both: OCTAVE *"can generate a voice and personality from prompts or recordings as brief as 5 seconds."*
> **We cannot cite Hume as "a serious lab concluded description-only is the safe path."** The one company that took that position abandoned it within ~12 months. Keeping the no-upload rule makes us **more restrictive than every major vendor** — a real differentiator, but we own the argument alone. · **HIGH** (visible in Hume's own product docs)

#### Implementation options, and why the obvious one fails

| Option | Published prior art | Verdict |
|---|---|---|
| **NER + gazetteer** (spaCy/GLiNER + Wikidata/Pantheon) | Gazetteer utility is *"undermined by pervasive spurious entity matching"*; filtering spurious mentions is worth ~**+3.70 % F1** — i.e. the baseline is bad enough that spurious-match filtering is a headline result. Entity–noun homograph ambiguity is a named unsolved class. **No published study measures celebrity-name-collision FP rates for ordinary names** — *no evidence found.* | **Necessary, not sufficient.** Scope to **Pantheon (~48 k globally-notable people)**, not all of Wikidata, or "a gravelly old sailor named Morgan" gets refused. |
| **LLM classifier gate** (Llama Guard 4, ShieldGemma) | Meta's own numbers: **English recall 69 %, FPR 11 %, F1 61 %; multilingual recall 43 %, FPR 3 %, F1 51 %.** And Meta's own warning: *"Some hazard categories may require factual, up-to-date knowledge to be evaluated fully (for example, **[S5] Defamation, [S8] Intellectual Property**, and [S13] Elections)."* | **A 31 % English / 57 % multilingual miss rate on exactly the categories we care about is a triage layer, not a gate.** The multilingual figure is disqualifying on its own for our Indic track. |
| **Embedding similarity to known-person descriptions** | *No published prior art for voice.* | Unstudied. |
| 🔴 **Periphrasis / implicit reference** — the evasion that breaks all of the above | **"You Know What I'm Saying: Jailbreak Attack via Implicit Reference" (AIR), Wu et al., arXiv:2410.03857.** *"AIR decomposes a malicious objective into permissible objectives and links them through implicit references… by introducing the discussion subject using a harmless objective and then incorporating the malicious objective with implicit references **that omit the subject**… the model fails to identify potential malicious objectives"* — **ASR > 90 %** on GPT-4o, Claude-3.5-Sonnet and Qwen-2-72B, with an **inverse scaling phenomenon, where larger models are more vulnerable**, and *"current detection methods were unable to effectively defend against this attack method"* (SmoothLLM, PerplexityFilter, Erase-and-Check all failed). | 🔴 **A name gazetteer is defeated by construction.** Any user who cannot type "Morgan Freeman" writes *"the narrator of March of the Penguins, warm, unhurried, deep."* Zero named entities. · **HIGH** |

The only paper on prompt-level gating for speech generation — *Synthetic Voices, Real Threats* (arXiv:2511.10913, Nov 2025) — red-teams five commercial TTS systems and finds *"semantic obfuscation techniques (Concat, Shuffle)"* **substantially reduce refusal rates**, with proactive moderation catching only **57–93 %**. It explicitly studies **content** harm and positions itself as distinct from speaker impersonation. **The impersonation-by-description gate remains unstudied in the literature.** · **HIGH**

#### Recommended design

Defence in depth, none of these load-bearing alone:
1. **Gazetteer over Pantheon-scale notable persons** — catches the lazy 80 %, cheap, auditable. Tune for **high precision**, accept misses.
2. **LLM classifier as triage**, not gate — flag for review, never silently refuse on its verdict alone in a non-English language.
3. 🔑 **The audio-side check (§5.2) is the only defence that survives periphrasis**, because it inspects the *output* rather than the *input*. Prioritise it over prompt classification.
4. **Rate-limit regenerations per description lineage, and log the lineage** — this is what defeats the human-in-the-loop attack.
5. **Human review queue + prompt retention.** Every vendor that publishes enforcement details relies on this combination.

---

### 5.2 CHALLENGE: is "no impersonation vector" airtight?

> # ❌ No. It is not.
> **The correct claim, which IS true and IS defensible, is: "no *cloning* vector — we never accept reference audio." Say that, and stop there.**

There is **no documented public incident** of description-only voice design producing a named recognisable person — an honest null result. But the feature class is ~2 years old, the failure mode is subjective, there is no detector for it, and no vendor has any incentive to publish it. **Absence of reporting is weak evidence.** And the *mechanism* is not merely plausible; it is published, benchmarked, and shipped in open training data.

| # | Hole | Primary evidence | Conf. |
|---|---|---|---|
| **1** | **Description→identity is a TRAINED OBJECTIVE, not a bug** | **ParaSpeechCaps** (arXiv:2503.04713) §3.1: *"**We identify celebrities**… (b) a **ChatGPT-generated list of celebrities with distinctive voices**… resulting in a total of **594 celebrities**"*; *"we use **GPT-4** to obtain a list of celebrities that are likely to have them"*; accents obtained *"**by prompting GPT-4 with the celebrity's name and ask it to output their accent**."* | **HIGH** |
| **2** | Its scaling premise **is** the claim that descriptions identify voices | Same paper §3.2: *"**two speakers with high perceptual similarity usually share most intrinsic tags**"*; tags are copied between speakers at *"a **cosine similarity of at least 0.8** (corresponding to a similarity rating of 5 out of 6 in VoxSim)."* | **HIGH** |
| **3** | Novel-voice generation is collision-**neutral**, not collision-**avoiding** | **Speaker Generation** (Stanton et al., ICASSP 2022): *"For an ideal system… **s2s, g2s and g2g will be equal**."* TacoSpawn 128-dim: **s2s = g2s = g2g = 0.20**. The success criterion is that a generated voice sits **as close to its nearest real training speaker as two real speakers sit to each other** — measured against corpora of 1 100–1 468 speakers, against a real world of ~8×10⁹. | **HIGH** |
| **4** | Voice identity is low-dimensional → collisions are dense | ~**9 acoustic parameters ≈ 7 principal components**, > 50 % of variance across a 10 000-speaker corpus (arXiv:2510.16489); 256-dim embeddings project to ~20 dims retaining discriminability (arXiv:2110.03380). | **MEDIUM** |
| **5** | 🔴 **GENERATED (not cloned) voices already defeat speaker verification at scale** | **Dictionary Attacks on Speaker Verification** (Marras et al., IEEE, arXiv:2204.11304): *"a novel attack vector that aims to **match a large fraction of speaker population by chance**… Adversarial waveforms obtained with our approach can **match on average 69 % of females and 38 % of males** enrolled in the target system at a strict decision threshold calibrated to yield **false alarm rate of 1 %**. **By using the attack with a black-box voice cloning system**, we obtain master voices… **transferable between speaker encoders**."* And: *"the goal is to match a non-trivial fraction of the user population **by pure chance, without any knowledge of the victim's identity or voice**."* | **HIGH** |
| **6** | Periphrasis defeats name gates | AIR, arXiv:2410.03857 — **> 90 % ASR**, inverse scaling, three published defences fail. | **HIGH** |
| **7** | 🔴 **Imitation without copying is actionable — copyright is irrelevant** | ***Midler v. Ford Motor Co.***, 849 F.2d 460 (9th Cir. 1988): *"**A voice is not copyrightable. The sounds are not 'fixed.'**"* … *"A voice is as distinctive and personal as a face… **To impersonate her voice is to pirate her identity.**"* Holding: *"when a distinctive voice of a professional singer is widely known and is **deliberately imitated** in order to sell a product, the sellers have appropriated what is not theirs and have committed a tort in California."* Ford held a licence to the song and did no sampling. ***Waits v. Frito-Lay***, 978 F.2d 1093 (9th Cir. 1992): **$375 000 compensatory + $2 000 000 punitive** affirmed. | **HIGH** / Waits MEDIUM |
| **8** | A non-cloned voice still forced a product withdrawal | **OpenAI "Sky"** (May 2024): cast from a separate professional actress hired *before* any outreach to Scarlett Johansson; Altman: *"The voice of Sky is not Scarlett Johansson's, and it was never intended to resemble hers."* Washington Post reviewed records and reported no copying. **OpenAI pulled it anyway.** | **MEDIUM** (openai.com 403s to automated fetch; via NPR / WaPo / Variety) |
| **9** | Human-in-the-loop iteration = a black-box optimiser with a human scoring function | Vendors return multiple candidates per generation and invite iterate-audition-refine. Formally this is black-box optimisation; the user supplies the gradient. **Nothing about "no audio upload" constrains it.** | **No evidence found — unstudied** |

#### 🔴 The finding that matters most: coincidental collision, not deliberate attack

Holes 3, 4, 5 and 7 combine into a threat that **survives perfect description-layer moderation**:

> **We can refuse every named-person prompt flawlessly and still mint a voice that a jury believes is Tom Waits.**

*Midler* requires **deliberate imitation** — pure coincidence may fall outside it. But the moment a user types a description *aimed at* a person, deliberateness is supplied by the user, and Alaap is the instrumentality. **Our prompt logs become the plaintiff's evidence of intent.** And note what our architecture actually defends against: "no audio in" defeats a **copying** claim we were never going to face. **Right of publicity does not require copying. The architecture defends against the wrong tort.**

Two further pressures in the same direction:
- ***Asha Bhosle*** (§4.3) shows Indian courts **will enjoin a tool provider**, though on a ratio about converting *into* a celebrity's voice — which we do not do.
- **The research direction is dissolving our boundary.** *VoiceDesigner* (JHU + **Adobe Research**, arXiv:2608.13613, Aug 2026) fuses text-to-voice generation *and* **"cloning and attribute modification"** in one model, reports SIM-o 0.757 on Seed-TTS, and **contains no ethics, broader-impact or misuse statement at all.** Betting product safety on the design/clone boundary is betting against where the field is going. · **HIGH**

#### What to do about it

| Control | Why |
|---|---|
| 🔴 **Restate the external claim.** Drop "no impersonation vector"; say **"no voice cloning — we never accept reference audio."** | The stronger claim is false and will be falsified publicly at the worst moment. The weaker one is true, verifiable, and still a genuine differentiator. |
| 🔑 **Screen minted identities against an ASV index of high-risk public figures before release.** | **The only defence that survives periphrasis**, because it inspects the output. And it is the one place an audio-side check is possible *without accepting user audio* — the index is built from public figures, never from users. Block or re-mint on near-match; **log every check** as good-faith evidence. |
| **Never expose a speaker-similarity score to users.** | Exposing it hands the attacker the objective function from hole 5. |
| **Cap regenerations per description lineage; log the lineage.** | Breaks the human-in-the-loop optimiser (hole 9). |
| **Retain prompts.** | Cuts both ways — evidence of *our* good faith, evidence of *their* intent. |
| **Watermark everything, provenance-log everything.** | Post-incident attribution is the fallback when prevention fails, and it will. |
| ⚠️ **Note the tension with the ASV screen and the no-upload invariant.** | The screen needs an ASV model and a reference index of public figures. That index is *system-held*, built from public data, never user-supplied — the same distinction the brief already draws for the Tier-2 seed clip. **Enforce it identically at the API boundary** (§8). |

---

### 5.3 Dialogue-text moderation

A **different problem** from description gating: arbitrary text → speech in a minted voice. Threat classes: **vishing/fraud scripts**, threats and harassment naming real people, election disinformation, CSAM-adjacent, medical misinformation.

> **India makes this mandatory, not merely prudent** — IT Rules r.3(3)(a)(i) requires *"automated tools or other suitable mechanisms, to not allow any user to create"* CSAM/NCII/obscene content, false documents, arms/explosives content, or audio that *"falsely depicts or portrays a natural person… by misrepresenting… such person's identity, voice, conduct, action, statement."* (§4.3)

| Tool | Version (2026) | Licence | Commercial hosted OK? | Notes |
|---|---|---|---|---|
| **Detoxify** | unitaryai/detoxify | **Apache-2.0** | ✅ **Yes, clean** | Cheap always-on pre-filter. No vendor risk. |
| **Llama Guard 4** | 12B multimodal | **Llama 4 Community License** — ⚠️ **NOT OSI, NOT Apache** | **Conditionally** | Field-of-use restrictions via incorporated AUP; **"Built with Llama"** attribution; MAU threshold (commonly cited 700 M) above which a separate Meta licence is required. ⚠️ **The 700 M figure and exact attribution wording are UNVERIFIED — have counsel read the LICENSE file directly.** For us the MAU clause is almost certainly non-binding; **attribution and AUP flow-down are the real obligations.** |
| **ShieldGemma / 2** | ShieldGemma 2 | **Gemma Terms of Use** (custom; code samples Apache-2.0) | Conditionally | *"You must not use any of the Gemma Services: for the restricted uses set forth in the Gemma Prohibited Use Policy"*; distribution requires flow-down + notice. Google claims no rights in outputs. ⚠️ Search suggests **Gemma 4 may have moved to Apache-2.0 — MEDIUM, verify**; if true it becomes the better pick. |
| **OpenAI Moderation** | `omni-moderation-latest` | API ToS | Yes | **Free**, does not count toward usage limits. ⚠️ **Verify whether current terms restrict use on content not generated by OpenAI models** — historically such a clause existed. A free dependency with a field-of-use restriction is a continuity risk. |
| **Perspective API** | Jigsaw | Google APIs ToS | ❌ **Do not adopt** | **Sunsetting — out of service after 2026**; quota-increase requests stopped Feb 2026; **no migration path offered.** Also carries documented identity-term FP bias. |
| **NeMo Guardrails** | NVIDIA | Apache-2.0 | Yes | **Orchestration framework, not a classifier.** |

Llama Guard 4 hazard taxonomy: S1 Violent Crimes · S2 Non-Violent Crimes · S3 Sex-Related Crimes · S4 Child Sexual Exploitation · S5 Defamation · S6 Specialized Advice · S7 Privacy · S8 Intellectual Property · S9 Indiscriminate Weapons · S10 Hate · S11 Suicide & Self-Harm · S12 Sexual Content · S13 Elections · S14 Code Interpreter Abuse.

> 🔴 **The gap nobody fills: there is NO published, permissively-licensed classifier or dataset for scam/vishing script detection.** *No evidence found.* No off-the-shelf hazard category covers *"this is your bank calling about suspicious activity on your account."*
> **This is the largest tooling gap for a voice product specifically, because fraud is the highest-frequency real-world abuse of synthetic speech** — see the $6 M FCC forfeiture against Steve Kramer and the $1 M Lingo Telecom settlement over the Biden robocall (§7). We must build a bespoke layer: regex + small classifier over banking/authority/urgency framings, OTP and account-number patterns, and "do not hang up" scripts.

#### Minimum viable stack for the render endpoint

1. **Detoxify (Apache-2.0)** — always-on pre-filter. Clean licence, negligible cost.
2. **Llama Guard 4 or ShieldGemma self-hosted** for S1–S13 **triage** (not gate), with "Built with Llama" attribution and AUP flow-down in our ToS. ⚠️ **Its 43 % multilingual recall means it is close to useless on our Indic track — do not let it create false assurance there.**
3. **Bespoke fraud/vishing layer.** Nothing off-the-shelf exists. Highest-value custom work in this section.
4. **Named-real-person check on the DIALOGUE text too**, not just descriptions — harassment and disinformation arrive here, and S5/S8 are Llama Guard's self-declared weak categories.
5. **Human review queue + per-account rate limits + full prompt retention.**
6. **Avoid** Perspective API (sunset); **verify** OpenAI Moderation's terms before making it load-bearing.

---

## 6. Bias & fairness

### 6.1 Evidence on demographic skew effects

#### 🔴 First, a correction to the premise itself

> **The "76.1 % twenties" and "62.6 % female" figures are not measured demographics. They are an audio-LLM's estimates.**
> VoicePersona's gender, age and accent labels were generated by **Qwen2-Audio-7B-Instruct**, not human annotators. The 76.1 % figure is therefore *the distribution of an audio-LLM's age predictions*, and audio-LLM age estimation is known to regress toward the mode — which would **inflate** an apparent twenties concentration.
> — <https://github.com/PranavMishra17/VoicePersona-Dataset> (**dataset card**) · **HIGH**
>
> Slicing on machine labels while reporting them as demographics is also a documentation defect under the Datasheets composition question. **Audit before publishing the number anywhere** (experiment E3, §9). This does not make the skew unreal — it makes its *magnitude* unverified.

#### The "old man off-manifold" worry: half right, and mis-located

**Confirmed — age is a high-variance direction in speaker-embedding space.** Cross-age speaker verification on VoxCeleb-derived trials (Qin et al., Interspeech 2022, arXiv:2207.05929), same ECAPA-style baseline, nationality and gender controlled:

| Test set | EER |
|---|---|
| Vox-H (standard) | 1.939 % |
| Vox-CA5 (≥5 yr gap) | 3.407 % |
| Vox-CA10 (≥10 yr gap) | 4.974 % |
| **Vox-CA20 (≥20 yr gap)** | **10.419 %** |

**A 5.4× EER inflation at a 20-year age gap.** The paper models the embedding as `z = z_id + z_age` and gets a 10 % relative improvement by adversarially decoupling age. **Age is not a nuisance dimension; it is a major axis.** · **HIGH**

Attribute probing confirms embeddings carry demographics: x-vectors probed at `segment6` recover **gender at ~99 %** and speaking rate at ~100 % (Raj et al., SLT 2018, arXiv:1909.06351). ⚠️ **That paper does not probe age** — a widely-circulated "age MAE ≈ 4.9 years" figure could not be verified and **must not be cited**. · **HIGH** / claim **LOW**

**NOT confirmed — that sampling a sparse region catastrophically degrades output.** The closest published analogue to Alaap is **Speaker Generation / TacoSpawn** (Stanton et al., ICASSP 2022), which fits a GMM prior (K=10) over speaker embeddings and samples novel speakers:

| Dataset | s2s | g2s | g2g |
|---|---|---|---|
| libriclean (1 230 spk) | 0.41 | 0.41 | 0.40 |
| en1468 (1 468 spk) | 0.17 | 0.18 | 0.17 |
| enus1100 (1 100 spk) | 0.24 | 0.24 | 0.24 |

**Sampled speakers are as far from training speakers as training speakers are from each other, and as diverse among themselves.** Sampling a fitted density did not collapse to the mode, and F0 range was *"equally diverse"* between training and generated speakers.

**But the naturalness table carries the real signal:**

| Dataset | Locale | # training speakers | Training MOS | Generated MOS | Δ |
|---|---|---|---|---|---|
| libriclean | US | 200 | 3.37 ± 0.14 | 3.54 ± 0.14 | +0.17 |
| **en1468** | **AU** | **164** | 3.30 ± 0.15 | **3.03 ± 0.14** | **−0.27** |
| en1468 | US | 300 | 3.68 ± 0.11 | 3.62 ± 0.11 | −0.06 |
| en1468 | GB | 212 | 3.69 ± 0.12 | 3.51 ± 0.13 | −0.18 |

**The thinnest slice (164 speakers) is the only one where generated MOS falls materially below its own training MOS**, and the ordering by speaker count (164 < 200 < 212 < 300) roughly tracks the degradation. n=4 locales, confounded with locale identity, and the authors draw no such conclusion — **MEDIUM, suggestive not conclusive.** Notably, the paper contains **no analysis of sample quality versus embedding-space density**; that exact measurement is **UNVERIFIED anywhere in the literature.**

Weakly against a sharp cliff: optimal-transport GMM interpolation for *intermediate* attribute values (ICASSP 2023, arXiv:2210.09916) reports **no statistically significant naturalness degradation** under continuous attribute control.

#### 🔑 The distinction the brief misses: conditioning failure vs decoder failure

Alaap's TTS is **frozen**. That splits the worry into two failure modes which rebalancing treats *completely differently*:

| Mode | Mechanism | Does rebalancing VoicePersona help? |
|---|---|---|
| **(A) Conditioning failure** | The description→embedding map has seen few "old" descriptions, so "an old man's voice" maps to a poorly-estimated point. | ✅ Yes — this is what rebalancing is for. |
| **(B) Decoder failure** | The **frozen TTS's own speaker space** never contained 70-year-olds, so even a *perfect* aged embedding decodes to a young voice with lowered pitch. | ❌ **No. Rebalancing cannot touch this at all.** |

**Nothing in the literature distinguishes these for a frozen-decoder pipeline — UNVERIFIED.** But the distinction is decisive: **if (B) dominates, the brief's prescription is aimed at the wrong component entirely.** Experiment E2 (§9) separates them in a few hours with no training, and **should be run before any rebalancing effort is funded.**

#### Analogue evidence that skew produces measured degradation

- **Koenecke et al., PNAS 2020** — five commercial ASR systems, 42 white / 73 black speakers, matched on age and gender: **WER 0.35 for black vs 0.19 for white speakers**; > 40 errors per 100 words for black men. · **HIGH**
- **Hutiri & Ding, FAccT 2022** — VoxCeleb2 is **61 % male, 29 % US nationality**; subgroup detection-cost ratios **Indian females 2.58×** worse than system average, Norwegian males 2.57×, US females 0.92×. Conclusion: bias enters at data generation, model building *and* implementation — **balanced data alone is insufficient.** · **HIGH**
- **Dutch E2E ASR** (arXiv:2307.02009): norm speakers 9.6 % WER; native children **42.9 %**; non-native adults **59.0 %**. · **HIGH**
- **Child TTS**: adult-speech TTS WER 3.43 vs child TTS **17.61**. · **MEDIUM**
- **Atypical speech**: F5-TTS cloning of dysarthric speech (TORGO) shows *"a strong bias toward speech intelligibility over speaker and prosody preservation"*, evaluated with Disparate Impact and Parity Difference (Interspeech 2025). · **MEDIUM**

#### Can a model trained on 20-somethings reproduce presbyphonia?

Target acoustics are well characterised: pathologic presbyphonia vs age-matched controls shows **jitter 3.44 % vs 1.74 %, shimmer 7.82 vs 4.84** — roughly a doubling of both — and **F0 moves in opposite directions by sex** (down in women with laryngeal edema, up in men with vocal-fold atrophy). · **HIGH**

> **A naive "lower the pitch for old" heuristic is therefore wrong for half the population** — and pitch-lowering is exactly what an under-conditioned model will do.

The one study that synthesises elderly speech at scale (OpenVoice2 elderly augmentation, arXiv:2604.24770) reports large downstream ASR gains (**CV18 English 4.1 % → 2.2 % WER; VOTE400 Korean 11.6 % → 4.8 %**) but **performs no objective comparison of synthetic vs real elderly acoustics** — no F0, jitter, shimmer or rate validation. *"The augmentation helps a downstream classifier"* is established; ***"the synthetic voice is acoustically aged" is UNVERIFIED anywhere.*** · **HIGH**

#### 🔴 Three independent findings all disadvantage female voices

This is the cross-cutting result of this file, and it appears in three unrelated literatures:

| Finding | Source | Effect |
|---|---|---|
| **Watermark robustness** | AudioMarkBench (§2.1.3a) | Female-attributed audio has **statistically significantly higher FNR** under Gaussian-noise watermark removal, **for all evaluated methods**; two-tailed t-test **p ≈ 2.4 × 10⁻⁶**. Also under EnCodec, Opus, quantisation, and black/white-box attacks. |
| **Master-voice vulnerability** | Dictionary Attacks (§5.2, hole 5) | Generated master voices match **69 % of females vs 38 % of males** at FAR 1 %; the authors attribute it to *"an accidental intrinsic bias of speaker encoders"*. |
| **Speaker-verification subgroup cost** | Hutiri & Ding, FAccT 2022 | **Indian females 2.58×** the system-average detection cost. |

> **Our corpus is 62.6 % female (by machine estimate). The demographic we have the most of is the one that is (a) least protected by the watermark, (b) most vulnerable to impersonation-by-collision, and (c) worst served by speaker-verification tooling — including the ASV screen we propose in §5.2.**
> That conjunction is not in the brief and should be. **It means our per-slice watermark evaluation (§2.y step 5) and our ASV screen threshold (§5.2) must both be calibrated per gender, not globally.** · **HIGH** on each finding, **MEDIUM** on the conjunction being load-bearing for us.

### 6.2 Per-demographic eval slice reporting spec

#### Is there a standard? Essentially no — we are writing it

| Source | What it prescribes | Confidence |
|---|---|---|
| **Model Cards** (Mitchell et al., FAT* 2019) | Quantitative analyses *"should be disaggregated, that is, broken down by the chosen factors"*, reporting **unitary** (per factor) **and intersectional** results, e.g. "age and race". | **HIGH** |
| **Datasheets for Datasets** (Gebru et al., CACM 2021) | Verbatim: *"Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset."* **Our 62.6 %/76.1 % figures are precisely this answer — they belong in the datasheet, with the Qwen2-Audio provenance stated.** | **HIGH** |
| **NIST AI 600-1** (GenAI Profile) | *"Harmful Bias and Homogenization"* is one of 12 named risk categories, covering performance disparities between sub-groups or languages from non-representative training data. ⚠️ Quote NIST directly before citing — the operationalisation language seen came from a vendor summary. | **MEDIUM** |
| **TTS-specific convention** | **None exists.** The closest position paper recommends benchmarks *"covering diverse accents, genders, and languages"* and ITU-T P.808 for MOS, but gives **no disaggregation requirements, no sample-size guidance, no reporting checklist.** | **HIGH** |
| **EU AI Act Art. 55** | ⚠️ **Explicitly out of scope for us.** Art. 55 binds only **GPAI models with systemic risk**, presumed at **> 10²⁵ cumulative training FLOP** (Art. 51). We are orders of magnitude below. **Do not scope bias reporting to Art. 55.** (Art. 50 transparency is a *different* provision with a different trigger — see §4.1.) | **HIGH** |

> **So publishing a disaggregated TTS fairness report is a legitimate research contribution to claim, not merely a compliance chore.** [`06-evaluation-harness.md`](06-evaluation-harness.md) should own the harness; this section owns the slice definition.

#### The slice spec

**Factors.** Report unitary marginals for all four. Report intersections only for the starred cells — full crossing is 4×4×8×4 = 512 cells and is unaffordable.

- **Gender**: female · male · androgynous-or-unspecified
- **Age band**: child · 18–30 · 31–50 · **51+ ★** *(the band the corpus starves)*
- **Accent / language**: General American · British · **Indian English by L1 ★** (Hindi, Tamil, Bengali, Telugu, Marathi) · native Hindi / Tamil / Bengali
- **Voice-quality style**: modal · breathy · creaky-rough · tremulous *(this doubles as the §2.y phonation stratification — build one corpus, not two)*

**Starred intersections (~12 cells):** {51+} × {female, male} × {GA English, Indian English (Hindi L1), Hindi}, plus {51+} × {tremulous, creaky-rough} × {female, male}.

**Metrics per slice, each grounded, each with its mandatory control:**

| Metric | Grounding | Required control |
|---|---|---|
| **ASR-WER on synthesised output** (intelligibility) | Standard TTS intelligibility proxy | 🔑 **Mandatory: report the same ASR's WER on *real* speech from the same slice.** Koenecke (0.35 vs 0.19) and Svarah (§6.3) prove **the judge is itself demographically biased**; without this control an "old-voice WER gap" is unattributable. Also: below ~1.6 % WER further gains are perceptually negligible. |
| **Speaker-similarity cosine** | IndicVoices-R uses S-SIM from Wav2Vec2 fine-tuned via IndicSUPERB | **Use an Indic-appropriate encoder for Indic slices, not English ECAPA.** Similarity is *"sensitive to channel variations, background noise, and even phonetic content"* — **fix text content across slices.** |
| **UTMOS + NISQA + DNSMOS** (MOS proxy) | NISQA: Mittag et al., Interspeech 2021 · DNSMOS P.835: Reddy et al., ICASSP 2022 | 🔑 **Report all three, not one.** UTMOS is documented as sensitive to domain mismatch and OOD conditions, and predicted-MOS models *"provide only point estimates without associated confidence intervals."* **Disagreement between the three IS the OOD signal** — which is exactly what we are hunting. |
| **Description adherence** | Attribute-classification accuracy on sampled embeddings (ProPS protocol) | Train attribute classifiers on *real* held-out speech; apply to synthesised audio. |
| **Aged-voice acoustic target check** *(51+ slices only)* | Presbyphonia reference values: jitter 3.44 %, shimmer 7.82 | 🔑 **Falsifiable and cheap:** does "old man" output actually raise jitter/shimmer toward those values, or merely lower F0? **Remember the sex-opposite F0 direction.** |
| **Watermark TPR @ FPR = 10⁻³** | AudioMarkBench female/male gap, p ≈ 2.4×10⁻⁶ | **This slice is shared with §2.y.** One corpus, two reports. |

**Minimum n per slice — derivation, since no citation provides one:**

- Variance in TTS evaluation is **speaker-dominated**, so the binding constraint is *distinct voices*, not utterances. Require **≥ 20 distinct voices per slice, ≥ 10 utterances each (≥ 200 utterances).**
- **Predicted-MOS:** with per-utterance SD ≈ 0.6, a 95 % CI half-width of ±0.1 needs n = (1.96·0.6/0.1)² ≈ **138**. To detect a **0.3-MOS** gap between slices at 80 % power: n ≈ 2(1.96+0.84)²(0.6/0.3)² ≈ **63/slice**. *TacoSpawn's thin-slice degradation was −0.27 MOS, so 0.3 is exactly the right effect size to power for.*
- **WER:** ~200 utterances ≈ 2 000 words; SE ≈ √(0.1·0.9/2000) ≈ 0.67 pp, inflated ~2× for within-utterance correlation → ~1.3 pp. **Sufficient to resolve a 3 pp gap, not a 1 pp gap.** Report **bootstrap CIs over speakers, not words.**
- **Total: 200 utterances × 20 voices per slice × ~24 reported cells ≈ 4 800 utterances.** Tractable.

### 6.3 Accent/dialect fairness for Indian languages

#### "Indian English" is not one accent — quantified

**Svarah** (Javed et al., Interspeech 2023): 9.6 h, **117 speakers (54 M / 63 F), 65 districts, 19 states, 19 of 22 scheduled languages, 4 language families**, roughly balanced across 18–30 / 30–45 / 45–60 / 60+.

Aggregate gap vs LibriSpeech: **Whisper-large 7.2 vs 2.7** (2.7×); **Wav2Vec2-large 24.9 vs 1.8 (13.8×)**; WavLM-large 33.7; Google IN 20.7; Azure US 20.9.

**Within-"Indian English" spread, Whisper-large WER by speaker L1:**
Maithili 4.5 · Hindi 5.3 · Tamil 5.3 · Gujarati 6.0 · Odia 6.2 · Urdu 6.2 · Konkani 6.4 · Kannada 6.6 · Punjabi 6.7 · Marathi 6.8 · Kashmiri 6.9 · Sindhi 7.3 · Telugu 7.5 · Bengali 7.6 · Dogri 8.0 · Malayalam 8.1 · Nepali 9.8 · **Assamese 10.1 · Bodo 11.6**

> **A 2.6× spread (4.5 → 11.6) inside a single "Indian English" label.** The monolith problem, quantified. By style: read 6.2, extempore 7.4, **use-cases 11.2**. · **HIGH**

#### 🔑 WER will not surface our accent failures

**PSP** (arXiv:2604.25476), benchmarking ElevenLabs v3, Cartesia Sonic-3, Sarvam Bulbul, Indic Parler-TTS and Praxy Voice on Hindi/Telugu/Tamil, reports **retroflex collapse rate rising monotonically with phonological difficulty: Hindi ~1 %, Telugu ~40 %, Tamil ~68 %** — and finds that **"PSP ordering diverges from WER ordering."**

> **This is the single most directly transferable Indic finding for Alaap: add a phoneme-level accent metric alongside WER, or Tamil retroflex collapse at 68 % will pass our eval harness undetected.** · **HIGH** · Cross-ref [`04-indic-track.md`](04-indic-track.md) and [`06-evaluation-harness.md`](06-evaluation-harness.md).

#### Dataset demographics as actually published

| Dataset | Published statistics | Confidence |
|---|---|---|
| **IndicVoices** (ACL 2024) | 7 348 h total, **1 639 h transcribed**, 16 237 speakers, 22 languages, **145 of 742 districts**; read 9 % / extempore 74 % / conversational 17 %; **median 73 h transcribed per language.** Collection **quotas**: age 18–30/30–45/45–60/**60+ min 15 % each**; education 4 bands min 15 % each; occupation blue/white-collar/unemployed min 10 % each; gender "balanced". | **HIGH for quotas** — ⚠️ **the paper states targets, not achieved distributions.** Realised percentages are not published. A later version circulates as 23.7 K h / 51 K speakers / 400+ districts; **version-check before citing.** |
| **IndicVoices-R** (NeurIPS 2024 D&B) | **1 700+ h, 10 496 speakers, 22 languages, 93.25 % extempore**; first open TTS data for 9 languages (Dogri, Kashmiri, Konkani, Maithili, Nepali, Sanskrit, Santali, Sindhi, Urdu). Benchmark = zero/few/many-shot with **NORESQA-MOS + S-SIM**. | **HIGH** |
| **Vaani** (IISc/ARTPARK + Google) | **~31 255 h, 156 K speakers, 165 districts, 106 languages; 2 043 h transcribed.** Target 1 M speakers across all 773 districts. | **MEDIUM-HIGH** |
| **Rasa** | 10 h neutral + 1–3 h per Ekman emotion (Assamese, Bengali, Tamil); extended to min 20 h/speaker, **44 speaker-language pairs across 22 languages** — i.e. **~2 speakers per language.** | **MEDIUM** |
| **SYSPIN / LIMMITS** | 40 h TTS per language targeted for 9 languages; released for 6 + Indian English. | **MEDIUM** |

> **Nominal coverage of all 22 scheduled languages exists; usable quality does not.** Rasa gives ~2 speakers per language; the median IndicVoices language has 73 transcribed hours. Set against PSP's Tamil 68 % retroflex collapse, **nominal coverage ≠ usable quality.** A published **per-language TTS quality-gap table across all 22 languages is UNVERIFIED** — IndicVoices-R's benchmark is the vehicle for it, but the per-language N-MOS/S-SIM values were not extracted. **Extracting them is cheap and would directly serve [`04-indic-track.md`](04-indic-track.md).**

#### Caste, socioeconomic status, code-switching

- **Caste / socioeconomic representation in speech corpora: UNVERIFIED.** India-centric bias literature is text/LLM-focused (Indian-BhED; IndiCASA). **No speech-corpus paper documenting caste representation surfaced.** IndicVoices' occupation and education quotas are the closest available **proxy**.
  ➡️ **Recommendation: do not collect caste labels** — ethically fraught and likely to cause harm. Instead report the **education × occupation × rural/urban** marginals we can actually observe, and **state the proxy limitation explicitly** in the model card.
- **Code-switching (Hinglish):** the commonly-cited "30–50 % relative WER increase" traces to a **vendor blog — LOW, do not cite.** Primary sources exist (IITG-HingCoS; the multilingual & code-switching ASR challenge) but figures were not extracted. **Treat Hinglish as a required eval slice regardless** — a description-conditioned model *will* receive Hinglish prompts.

### 6.4 Does rebalancing actually work? — the brief's prescription is under-powered

**The brief prescribes "rebalance by sampling or reweighting" as *the* mitigation. The evidence does not support that framing.**

| Evidence | Finding |
|---|---|
| **Speaker verification across accents** (arXiv:2204.12649) | Unbalanced system: **USA false-alarm 0.91 % vs India 8.63 % (~9.5×).** A balanced backend (DCAPLDA) drives **FDR to 0.989 / 0.990** with near-zero calibration loss and no majority-group degradation — **balancing genuinely worked.** ⚠️ **But the same paper reports poor cross-accent generalisation when India was EXCLUDED from training.** Reweighting redistributes attention across data you have; **it cannot substitute for data you lack.** |
| **Hutiri & Ding, FAccT 2022** | Bias originates across data generation, model building *and* implementation; **not addressable through representation alone.** |
| **Dutch E2E ASR** (arXiv:2307.02009) | Speed perturbation + SpecAugment + VTLN reduced overall bias **29.12 % → 25.20 %** (13 % relative) — yet *"bias was and remained highest against non-native speakers."* **Partial closure at best.** |
| **General ML** | Resampling outperforms reweighting under SGD (arXiv:2009.13447). For speaker recognition, fairness gains from balancing are **operating-point dependent**, while adversarial/multi-task training improves fairness *across* operating points. |

**Augmentation for aged voices specifically: UNVERIFIED, and mechanistically doubtful.** VTLP is validated for ASR generally and as pseudo-speaker augmentation, but **no paper validates that pitch/formant-shifted young speech is acoustically aged.** The mechanism argues against it: presbyphonia's signature is **perturbation** (jitter, shimmer, tremor), not a formant warp — and **VTLP/pitch-shift manipulate exactly the parameters that do not carry the age signal**, while F0 direction is sex-opposite.

> ### Verdict on the brief's prescription
> **"Rebalance by sampling or reweighting" is not wrong, but it is under-powered and probably aimed at the wrong component. Present it as one of three, not as the mitigation.**
>
> 1. **Rebalancing is cheap and worth doing — for the conditioning map only.** It measurably fixes calibration-type disparities (FDR 0.989) and costs nothing. **Claim exactly that and no more.**
> 2. **Targeted data acquisition is what the literature actually supports for capability gaps.** In every case where a group was *absent* rather than merely rare, balancing failed. If the 51+ region is genuinely thin, adding real 51+ speakers is the intervention with evidence behind it.
> 3. **Honest per-slice reporting is mandatory regardless**, and is the one deliverable no ablation can invalidate.
>
> **And run experiment E2 (§9) before funding any rebalancing work** — if the frozen decoder is the binding constraint, rebalancing VoicePersona cannot help at all.

---

## 7. Abuse vectors the brief missed

The brief's §15.2 lists four obligations and §16 lists one risk ("voice-cloning misuse"). That framing assumes the only adversary is someone trying to impersonate a person. **A public description→voice service with a free-form render endpoint enables considerably more than that.** Ordered by expected harm × likelihood.

### A. Vectors with published evidence behind them

| # | Vector | Why it is real | Control |
|---|---|---|---|
| **A1** | 🔴 **Voice-biometric defeat (master voices).** Our service is a black-box generator of novel voices with an iterable input. That is exactly the primitive the dictionary attack needs: adversarial master voices match **69 % of females / 38 % of males** at FAR 1 %, **transfer between speaker encoders**, and require **no victim audio and no victim identity**. Banks and telcos deploy voice biometrics. | **HIGH** — arXiv:2204.11304 (IEEE). The attack was *demonstrated using a black-box voice cloning system*; ours is a black-box voice *generation* system with a text handle. | Never expose similarity scores; cap regenerations per lineage; rate-limit; watermark; log. **Add "defeating voice authentication" to the prohibited-use policy explicitly** — no vendor policy we found names it. |
| **A2** | 🔴 **Fraud and vishing scripts at industrial scale.** The render endpoint turns arbitrary text into convincing speech in an arbitrary voice. "This is your bank's fraud department" in 22 languages, at fractions of a cent per line. | **HIGH** — the FCC issued a **$6 M forfeiture** against Steve Kramer and a **$1 M settlement** with Lingo Telecom over one AI-voice robocall; 13 felony + 13 misdemeanour counts followed in NH. And **no permissively-licensed vishing-script classifier exists** (§5.3) — *no evidence found.* | The bespoke fraud layer in §5.3. **This is the highest-frequency real-world abuse of synthetic speech and the brief does not mention it once.** |
| **A3** | 🔴 **Coincidental collision → right-of-publicity liability with no impersonation intent.** A novel voice that sounds like a real person is actionable under *Midler*/*Waits* without any copying. | **HIGH** — §5.2 holes 3, 4, 7. *Waits*: **$2.375 M** affirmed. | ASV screen against a public-figure index before release; log it (§5.2). |
| **A4** | 🔴 **Watermark forgery — being blamed for audio we did not generate.** AudioSeal's detector is public and unkeyed. AudioMarkBench: *"All watermarking methods have high FPRs under white-box perturbations that preserve audio quality"* — an attacker can **stamp a Alaap-detectable mark onto third-party audio.** | **HIGH** | The **provenance log is the rebuttal**: a mark with no matching render record is a forgery. **This is a second, independent reason the log must exist and must be authoritative over the watermark.** Publish the rebuttal procedure. |
| **A5** | 🔴 **Child voices.** "A frightened eight-year-old girl" is a legitimate game-character description and an obvious CSAM-adjacent and grooming-script vector. **The brief never mentions minors' voices**, and our own §6.2 slice spec has a `child` age band. | **MEDIUM-HIGH** — India IT Rules r.3(3)(a)(i)(I) makes blocking CSAM-adjacent generation a **legal duty**, and DPDP imposes children's-data obligations. Llama Guard's S4 exists but at 69 % English / 43 % multilingual recall. | Treat child-voice descriptions as an elevated-review class: allow, but with stricter dialogue moderation, mandatory logging, and no bulk/API access without review. **Decide this deliberately rather than discovering it.** |
| **A6** | **Non-consensual intimate audio.** Sexual dialogue rendered in a voice described to match someone the user knows. Description-layer gazetteers do not fire on "my colleague Priya, 28, soft Bengali accent". | **MEDIUM-HIGH** — India r.3(3)(a)(i)(I) names NCII explicitly; Steam refuses Adult-Only sexual content via live-generated AI (§3.7.1). | Dialogue moderation + the private-individual problem below. |
| **A7** | **Harassment of private individuals.** The gazetteer protects celebrities. **Nobody is protecting the user's ex-partner, manager, or classmate** — an ordinary name is exactly the false-positive case we deliberately tuned the gazetteer *not* to catch. | **MEDIUM** — the FP/FN trade in §5.1 makes this structural, not incidental. | Dialogue-text named-person check (§5.3 item 4); reporting channel; retention for investigation. |
| **A8** | **Election disinformation in Indian languages.** Llama Guard 4 multilingual recall is **43 %**. India's IT Rules name the Representation of the People Act 1951 in the mandatory warning. | **HIGH** on the recall figure. | Do not rely on an English-centric classifier for the Indic track. Elevated review during Indian and EU election periods. |

### B. Vectors specific to *our* architecture

| # | Vector | Why it matters here |
|---|---|---|
| **B1** | 🔴 **Open-sourcing the model voids the watermark.** [`PHASE-10`](phases/PHASE-10-public-release.md) plans a public release of the dataset, model card and writeup. **If the mapper weights ship, anyone can run mint→render with the watermarking step deleted.** Every §4 obligation we discharge in our service is discharged for our service only — and under EU AI Act Art. 2, whoever then deploys the weights becomes the provider with their own Art. 50(2) duty. | **This is a strategic decision the brief has not framed as one.** Options: (a) release dataset + card + writeup but not weights; (b) release weights with watermarking baked into the released inference code and a licence term requiring it; (c) release everything and say so plainly in the model card. **(b) is weak — the code is trivially editable. Decide explicitly at S10, not by default.** |
| **B2** | 🔴 **Model extraction via the mint endpoint.** Every `mint(description) → identity` call returns a (text, embedding-derived audio) pair. **That is precisely the training data for a distillation clone of our mapper** — the one artefact that is actually ours. | Rate-limit mint (the brief already makes mint free and render metered — §12.2 — which is **exactly backwards for this threat**). Consider returning identities without exposing embeddings; cap mints per account per day; watermark seed clips too. |
| **B3** | 🔴 **Seed-waveform boundary erosion.** The brief's own tension: Tier-2 identity mints a *system-generated* seed and clones from it. **Any code path that lets a user influence, upload, or substitute that seed collapses the entire safety story**, including the NO FAKES safe harbour and the *Asha Bhosle* defence. | §8 makes this an enforced API invariant, not a convention. Also note the §5.2 ASV screen introduces a *second* audio input path (the public-figure index) that must be enforced identically. |
| **B4** | **Render-cache side channel.** The brief's §12.3 render cache is keyed by (identity, text). A timing difference between hit and miss lets a user **probe whether some other user has rendered a given line** — leaking unreleased game dialogue. | Cheap fix: **partition the cache per account**, or add constant-time response padding. Costs almost nothing at design time. |
| **B5** | **Free-mint cost attack.** Free mint + metered render (§12.2) means the unmetered operation is the GPU-expensive one. | Quota mint by account and IP; require an account before mint. |
| **B6** | **Descriptions are personal data.** *"Sounds like my grandmother — Bengali, 78, slight tremor"* is personal data about a third party under DPDP and GDPR, submitted without that person's knowledge. **We retain prompts for safety (§5.2) — which is itself a processing purpose we must declare.** | Declare prompt retention in the DPDP/GDPR notice with its safety purpose and retention period. Do not let the safety log become an undeclared dossier. |
| **B7** | **Liability transfer to our users.** A developer ships our audio, gets **delisted from itch.io** for failing to tag, or fails a Steam review on the "not illegal or infringing" promise, and looks to us. | The §3.8 artefacts — especially the rights attestation and the runtime-vs-build-time warning — are risk controls, not marketing collateral. |
| **B8** | **Prompt injection via the description field into our own moderation LLM.** If an LLM classifier reads user descriptions, the description is untrusted input to that classifier. | Standard: never let classifier output be interpreted as instructions; structured output only; the classifier is advisory (§5.3). |
| **B9** | **Anti-spoofing dataset contamination.** Our watermarked output will end up in ASVspoof-style corpora and in other people's training data. A systematically watermarked signal is a **shortcut feature** a spoof detector can learn instead of learning spoofing. | Not our liability, but worth a line in the model card. Also an argument for making the watermark *detectable* rather than *secret*. |
| **B10** | **Voice-identity squatting and resale.** A user mints a voice, publishes it in our library, and licenses it — including a voice that resembles a real person, or a competitor's established game character. | Library-publication is a distinct moderation surface from generation. Gate it separately; it is the point where §5.2's ASV screen matters most. |

### C. The honest summary

> Two of these — **A1 (voice-biometric defeat)** and **A2 (fraud scripts)** — are, in expected-harm terms, **larger than the impersonation risk the brief spends all its safety budget on**, and neither appears anywhere in the scope document. A2 has a documented **$7 M** enforcement precedent in a single incident. A1 has a peer-reviewed attack that our product is a ready-made instrument for.
>
> **B1 (open-sourcing voids the watermark)** is the one that could quietly undo the entire §4 compliance stack, and it is currently scheduled as a *goal* rather than evaluated as a *decision*.

---

## 8. What this means for the build

### 8.1 What to install, and when

| Stage | Component | Licence | Why |
|---|---|---|---|
| **S0 (now)** | `pip install audioseal` (0.2.0) | MIT code + MIT weights | Zero-cost to add now. Getting it into the render path early means the §2.y calibration runs on real output, not a synthetic proxy. |
| **S0 (now)** | Fork `github.com/SonyResearch/raw_bench` | *verify before vendoring* | The robustness harness. Do **not** write our own. Add Opus, polarity inversion, and our duration/phonation stratification. |
| **S0 (now)** | Provenance table in the DB | — | Already required by §12.1 for the model-upgrade problem. Safety and compliance get it free. Add `output_sha256`, `wm_message`, `wm_detect_score`. |
| **S9** | `Detoxify` (Apache-2.0) | Apache-2.0 | Always-on dialogue pre-filter. |
| **S9** | Llama Guard 4 **or** ShieldGemma, self-hosted | ⚠️ Llama 4 Community / Gemma ToU — **not** Apache | Triage only. Attribution + AUP flow-down in ToS. **Verify the Gemma-4-may-be-Apache-2.0 rumour first** — if true, prefer it. |
| **S9** | Bespoke fraud/vishing classifier | ours | **Nothing off-the-shelf exists.** Highest-value custom safety work. |
| **S9** | Person gazetteer over **Pantheon (~48 k)**, not Wikidata | CC-BY-SA — check | High-precision, tuned to accept "a sailor named Morgan". |
| **S9** | ASV screen index of high-risk public figures | — | The only defence that survives periphrasis (§5.2). |
| **S9** | Signed provenance metadata sidecar | — | EU CoP layer 1 of 2. Sign it before C2PA conformance; upgrade later. |
| **S9** | `POST /detect` public endpoint | — | EU CoP Commitment 2; CA SB 942 at scale. |
| **+1Q** | C2PA conformance application (Level 1) | — | Unlocks SSL.com's free certificate tier. |
| **+2Q** | `c2pa-python` (Apache-2.0) manifests on **web downloads only** | Apache-2.0 | Gets YouTube auto-labelling for free. Not for game-asset export — it does not survive the pipeline (§3.2). |
| **Watch** | c2pa-rs PR #2073 (Ogg/Opus), #2399 (soft-binding resolution); SyncGuard code release; Gemma 4 licence | — | Each is a trigger to revisit a decision here. |

### 8.2 What the API boundary must enforce — in code, not in a README

The brief's own §12.5 principle applies: *"A licence rule that lives only in a document will eventually be violated by a config change."* The same is true of every invariant below.

```
INVARIANT 1 — NO USER AUDIO, EVER
  The public API surface exposes NO parameter of audio type.
  Not a file, not a URL, not base64, not a data URI, not a
  "reference_id" that resolves to user-uploaded bytes.
  Enforced by: request-schema validation that rejects any
  audio-typed field, plus a contract test asserting the
  public OpenAPI schema contains zero audio inputs.
  This is the NO FAKES safe harbour, the TN "primary purpose"
  defence, and the Asha Bhosle defence. It is load-bearing.

INVARIANT 2 — SYSTEM-GENERATED AUDIO IS INTERNAL-ONLY
  Tier-2 seed waveforms and the ASV public-figure index are
  audio, but they are SYSTEM-held. They live behind an
  internal interface that the public router cannot reach.
  Enforced by: separate module, no public route, a test that
  fails if a public handler imports the seed/index module.

INVARIANT 3 — NO UNWATERMARKED BYTES LEAVE THE PROCESS
  Watermarking happens inside render(), before the buffer is
  returned — not in a middleware, not in an export step, not
  behind a flag.
  Enforced by: render() returns a WatermarkedAudio type that
  is the ONLY type the serialiser accepts. Make it
  structurally impossible to emit raw audio.
  Cost: 7.4 ms. There is no efficiency argument for a flag.

INVARIANT 4 — NO RENDER WITHOUT A PROVENANCE ROW
  The provenance write and the audio return are one
  transaction. If the write fails, the render fails.
  The log is authoritative over the watermark (§7 A4).

INVARIANT 5 — METADATA AND LABEL ARE NOT USER-REMOVABLE
  No export option strips provenance metadata or the label.
  India IT Rules r.3(3)(b) forbids ENABLING removal, not just
  removing.

INVARIANT 6 — NO SIMILARITY SCORE CROSSES THE BOUNDARY
  ASV screen results are internal. The API never returns a
  speaker-similarity number, rank, or "closest match".
  Exposing it hands the attacker the objective function
  (§5.2 hole 5, §7 A1).

INVARIANT 7 — DETECTION TESTS x AND -x
  The detector evaluates the signal and its polarity
  inversion, plus a small circular-shift grid.
  Polarity inversion drives AudioSeal bitwise accuracy to
  0.18 and message accuracy to 0.00 (§2.1.3b).
```

### 8.3 Architectural consequences worth stating plainly

1. **The watermark carries a presence bit. The provenance log carries identity.** Attribution accuracy of 0.39 under edits and 0.69 clean (§2.1.6) makes any other design unsound. This also means **the log must be durable, queryable by content hash, and authoritative** — it is the answer to "who made this", to forgery accusations, and to storefront and regulator queries alike.

2. **Two marking layers, because the EU Code of Practice requires two.** Signed metadata + watermark. *"Relying on fingerprinting or logging alone is not considered sufficient."* Our provenance log alone does not comply; neither does the watermark alone (§4.1).

3. **A public detection endpoint is a compliance deliverable, not a nice-to-have.** EU CoP Commitment 2 requires it, free to regulators, media, fact-checkers, researchers and civil society. California SB 942 requires uploads + URLs + an API above 1 M monthly users. Build it once at S9.

4. **The India audible-prefix toggle must exist before launch even if it stays off.** Per-jurisdiction, per-delivery-channel. Trivial at design time, expensive to retrofit (§4.3).

5. **The eval corpus is shared.** §2.y (watermark robustness by duration and phonation) and §6.2 (per-demographic quality slices) are **the same 4 800-utterance corpus with two report views.** Build it once; it serves E2, compliance evidence under CoP Measure 4.2, the model card, and [`06-evaluation-harness.md`](06-evaluation-harness.md).

6. **Marketing is a compliance surface.** The NO FAKES prong (iii) and TN §47-25-1105(a)(3) turn on how we describe the product. **Add "no real-person references" to the marketing review checklist**, not just the engineering one (§4.4 item 9).

7. **Restate the external safety claim** from "no impersonation vector" to **"no voice cloning — we never accept reference audio."** The first is false and will be falsified publicly; the second is true, verifiable, and still a genuine differentiator (§5.2).

8. **Decide the open-source question deliberately at S10.** Releasing mapper weights voids the watermark for every downstream user and transfers Art. 50(2) provider duties to them (§7 B1). This is currently scheduled as a goal, not evaluated as a decision.

---

## 9. Open — must be settled by experiment or counsel

| # | Question | Cheapest experiment | Est. cost / time | What it blocks |
|---|---|---|---|---|
| **X1** | 🔴 **Does AudioSeal detect reliably on OUR 1–3 s output?** No published number exists for a fully-watermarked short clip (§2.y). | The duration × phonation × attack sweep of §2.y, on 2 000 real pipeline renders + matched controls. Fork `raw_bench`; add Opus, polarity inversion, our stratification. Report **TPR @ FPR = 10⁻³ per duration bin**. | **~2 GPU-hours + 1 day** to build the harness | **S9 public launch.** Also produces the CoP Measure 4.2 self-testing evidence. |
| **X2** | 🔴 **Does testing `−x` recover the polarity-inversion failure?** Published: bitwise 0.18, message 0.00 (RAW-Bench). Untested: whether the trivial mitigation works. | Watermark 200 clips; detect on `x` and `−x`; report max-score TPR. | **30 minutes** | Invariant 7 in §8.2. Do this first — it is the cheapest item in the file. |
| **X3** | 🔴 **Is the watermark audible on whispered / breathy / growled voices?** UNVERIFIED everywhere; mechanistically plausible because the TF-loudness masker is absent in whisper (§2.1.4). | 40 pairs per phonation class; PESQ/ViSQOL vs unwatermarked + forced-choice ABX with ~15 listeners. | **~1 day + a small listener panel** | Whether stylized voices need a reduced watermark strength — or whether we ship a known audible artefact on our most distinctive product feature. |
| **X4** | 🔴 **Frozen-decoder ceiling: is "old man" failure mode (A) or (B)?** (§6.1) | **E2:** feed embeddings of **real 51+ speakers** (IndicVoices 60+ quota, Svarah 60+ band) **directly** to the frozen TTS, bypassing the description encoder. Measure UTMOS/NISQA/DNSMOS + jitter/shimmer. | **A few hours, no training** | **Any rebalancing spend.** If the frozen decoder degrades on *real* aged embeddings, rebalancing VoicePersona cannot help and the brief's prescription is refuted for our architecture. **Run before funding S4.** |
| **X5** | **Is generation quality actually a function of embedding-space density?** UNVERIFIED in the published literature — nobody has measured it. | **E1:** fit a GMM/KDE (K=10, per TacoSpawn) to VoicePersona embeddings; generate ~500 voices across age descriptions; regress UTMOS/NISQA/DNSMOS/S-SIM on **log-density under the prior**. | **~1 GPU-day, no training** | Nothing — but **either outcome is publishable**, and a slope of ≈0 kills the off-manifold worry with our own numbers. |
| **X6** | **Are the 62.6 % / 76.1 % skew figures real?** They are Qwen2-Audio-7B estimates, not measurements (§6.1). | **E3:** hand-label 200 random VoicePersona clips for age band and gender; publish the confusion matrix against the machine labels. | **~2 hours of human time** | Publishing the number anywhere. Required regardless to answer the Datasheets subpopulation question honestly. |
| **X7** | ⚖️ **Does India's Rule 3(3) audible-prefix requirement bind a NOVEL synthetic voice?** Turns on whether "depicts or portrays any individual… indistinguishable from a natural person" is a **realism** test or an **identity** test. Genuinely arguable both ways (§4.3). | **Counsel — Indian technology-law practice.** Not an experiment. | Legal spend | Indian launch, and the default state of the §8.2 toggle. **Highest-value legal question in this file.** |
| **X8** | ⚖️ **Are we an "intermediary" under IT Act s.2(1)(w) at all?** And does s.75 extraterritoriality reach a foreign-hosted service serving Indian users? | Counsel, same engagement as X7. | — | Scope of the whole §4.3 obligation set. |
| **X9** | ⚖️ **Llama 4 Community Licence:** exact attribution wording and the MAU clause. The 700 M figure is **UNVERIFIED** — nobody on this pass opened the LICENSE text. | Counsel or a careful read of `huggingface.co/meta-llama/Llama-Guard-4-12B/blob/main/LICENSE`. | ~1 hour | Whether Llama Guard 4 can ship, and what our ToS must flow down. |
| **X10** | **Does OpenAI's Moderation endpoint restrict use on non-OpenAI-generated content?** Historically such a clause existed. A free dependency with a field-of-use restriction is a continuity risk. | Read the current API terms. | ~30 min | Whether it can be load-bearing in §5.3. |
| **X11** | **Is Gemma 4 genuinely Apache-2.0?** MEDIUM rumour. If true, ShieldGemma becomes materially more attractive than Llama Guard. | Fetch `ai.google.dev/gemma/terms` and the Gemma 4 model card. | ~30 min | The §5.3 classifier choice. |
| **X12** | **SilentCipher's real robustness.** Its own paper (MP3 96–100 %, OGG 100 %) and RAW-Bench's independent full-message figures disagree materially, and a gain-adjustment failure is suspected (§2.4). | Re-read RAW-Bench Table 5 directly; then run gain, MP3 and OGG on the released checkpoint ourselves. | ~half a day | Whether SilentCipher is a viable fallback if X1 fails. |
| **X13** | **What is the per-language TTS quality gap across all 22 Indian languages?** UNVERIFIED — IndicVoices-R's benchmark is the vehicle but the per-language N-MOS / S-SIM values were never extracted. | Extract the IV-R benchmark tables. | ~2 hours | Honest per-language claims in [`04-indic-track.md`](04-indic-track.md) and the model card. |
| **X14** | **What are IndicVoices' ACHIEVED demographic distributions?** The paper publishes **quotas (targets), not realised percentages** (§6.3). | Ask the authors, or compute from the released metadata. | ~2 hours | Any claim that our Indic data is demographically balanced. |
| **X15** | ⚖️ **Do we release mapper weights at S10?** Doing so voids the watermark downstream and transfers Art. 50(2) duties (§7 B1). | Not an experiment — a **product decision** with legal input. | — | S10 scope. **Frame it as a decision now, not a default later.** |
| **X16** | **Does the render cache leak across accounts?** Timing oracle on (identity, text) (§7 B4). | Measure hit vs miss latency; if separable, partition the cache per account. | ~1 hour | Nothing yet — but it is far cheaper to partition at design time. |
| **X17** | **`raw_bench` and Pantheon licences** before vendoring either. | Fetch both LICENSE files. | ~15 min | The §8.1 install list. |

> **Run X2, X6 and X17 this week — together they are under a day and each removes a real unknown.**
> **Run X4 before any rebalancing budget is committed.**
> **X1 and X3 gate S9. X7 gates Indian launch.**

---

## 10. Sources

**Source-type key:** `LIC` = licence file fetched directly · `PAPER` = peer-reviewed / arXiv preprint · `REPO` = official repository or package index · `DOC` = official vendor/product documentation · `SPEC` = standards specification · `REG` = regulator / legislature / government primary · `COURT` = court opinion or order · `POLICY` = official platform policy page · `DATA` = dataset card / datasheet

### Watermarking (E2)

| # | URL | Type | Used for | Conf. |
|---|---|---|---|---|
| 1 | https://raw.githubusercontent.com/facebookresearch/audioseal/main/LICENSE | **LIC** | AudioSeal code = MIT, verbatim | HIGH |
| 2 | https://raw.githubusercontent.com/facebookresearch/audioseal/main/README.md | **REPO** | *"full MIT license (including the license for the model weights)"* — the decisive quote | HIGH |
| 3 | https://facebookresearch-audioseal.mintlify.app/ | **DOC** | *"available under the MIT license and can be used in both research and commercial applications"* | HIGH |
| 4 | https://huggingface.co/facebook/audioseal | **DOC** | Weights model card, `license: MIT` | HIGH |
| 5 | https://pypi.org/pypi/audioseal/json | **REPO** | v0.2.0 (2025-12-17); deps `numpy, omegaconf, torch, einops` — **no audiocraft/encodec**; MIT classifier; maintenance currency | HIGH |
| 6 | https://arxiv.org/pdf/2401.17264 | **PAPER** (ICML 2024) | Robustness Table 3 + App D.2 params; quality Table 1; attribution Table 4; localisation Fig 5 / Table 6 (IoU 0.802 vs prose 0.99); mixing Table 7; OOD Tables 8–10; runtime Table 5; training = 4.5 K h VoxPopuli; adversarial §6 | HIGH |
| 7 | https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf | **PAPER** (NeurIPS 2024 D&B) | AudioMarkBench: 5 s clips, τ=0.15, **Opus/EnCodec/SoundStream remove watermarks at ViSQOL ≥ 3**, white-box FNR→1.0, forgery, **female/male FNR gap p ≈ 2.4×10⁻⁶**, no age gap (teens–forties only), BN/TA in the language set | HIGH |
| 8 | https://www.isca-archive.org/interspeech_2025/ozer25_interspeech.pdf · https://arxiv.org/pdf/2505.19663 | **PAPER** (Interspeech 2025) | RAW-Bench: 20 distortions on raw 44.1 kHz; **polarity inversion 0.18/0.00**, phase shift 0.62/0.06, reverb 0.87/0.22, **OGG 0.95**, MP3 0.92, **DAC 0.00**; capacity 5.33 bps; MOS-LQO 4.93; *"audio watermarking models fail under neural compression"* | HIGH |
| 9 | https://arxiv.org/abs/2502.04230 | **PAPER** (ICML 2025) | XAttnMark: **the only duration ablation — detection 98.6–99.3 % at 1–10 s, attribution 81.2 % @1 s → 93.0 % @5 s**; independent AudioSeal numbers (**attribution avg 0.39, identity 0.69**); quality Table 4; generative-edit Table 2; HSJA Table 3; **no code released** (Dolby) | HIGH |
| 10 | https://arxiv.org/abs/2503.19176 | **PAPER** (SoK) | 9 schemes × 22 attacks; **pitch shift defeats every scheme**; Timbre ~65 % on singing; WavMark ≥1 s floor | HIGH |
| 11 | https://arxiv.org/abs/2504.10782 | **PAPER** | *"Deep Audio Watermarks are Shallow"*: vocoder/denoiser laundering, TPR@1%FPR ≈ 0.00–0.04; clean-speech-only scope | HIGH |
| 12 | https://arxiv.org/pdf/2509.05835 · https://ojs.aaai.org/index.php/AAAI/article/view/39997 | **PAPER** (AAAI 2026) | Overwriting attack: **~100 % ASR vs AudioSeal**; post-attack BER µ = 0.504–0.506 | HIGH |
| 13 | https://arxiv.org/html/2606.22310 | **PAPER** (Jun 2026) | *Learning to Evade*: removal 100 % / 0 % detected, replacement 3–17 %, at SNR 23.9–26.6 dB, ViSQOL 4.26–4.70 | HIGH |
| 14 | https://arxiv.org/abs/2511.21577 | **PAPER** | **HarmonicAttack: ~100 % ASR vs AudioSeal, WavMark, SilentCipher; 0.03–0.06 s/sample; cross-domain transfer** | HIGH |
| 15 | https://raw.githubusercontent.com/resemble-ai/perth/master/LICENSE · https://pypi.org/project/resemble-perth | **LIC** / **REPO** | Perth MIT (Resemble AI 2025); 34.4 MB wheel ⇒ weights bundled; deps ISC/BSD | HIGH |
| 16 | https://www.resemble.ai/benchmarks | **DOC** (vendor) | OSS Perth pitch-shift **10 %**, clipped Gaussian **45 %**; PerTh V2 (94 %/100 %) is **commercial and closed** | MEDIUM |
| 17 | https://raw.githubusercontent.com/wavmark/wavmark/main/LICENSE · https://huggingface.co/M4869/WavMark | **LIC** / **DOC** | WavMark MIT code + MIT weights; ⚠️ unfilled `[year]`/`[fullname]` placeholders | HIGH / MEDIUM |
| 18 | https://arxiv.org/abs/2308.12770 | **PAPER** | WavMark: 1 s units, BFD ~20 attempts/s, 10+22 bits, 1/1024 pattern FPR, segment BER table, PESQ 4.32, 0.38× RT CPU detection | HIGH |
| 19 | https://raw.githubusercontent.com/sony/silentcipher/master/LICENSE · https://huggingface.co/Sony/SilentCipher | **LIC** / **DOC** | SilentCipher MIT code + MIT weights (Sony Research 2024) | HIGH |
| 20 | https://arxiv.org/abs/2406.03822 | **PAPER** | SilentCipher: 32/40 bits, 6/12/24 s eval, MP3/OGG/AAC table, SDR 47.24, 1302× RT | HIGH |
| 21 | https://raw.githubusercontent.com/TimbreWatermarking/TimbreWatermarking/main/LICENSE | **LIC** | ❌ **Timbre = GPL-3.0**, verbatim | HIGH |
| 22 | https://ai.google.dev/responsible/docs/safeguards/synthid | **DOC** | *"SynthID **Text** has been open sourced"* — no audio embedding for developers | HIGH |
| 23 | Vertex AI / Lyria docs (docs.cloud.google.com) · deepmind.google/science/synthid · blog.google SynthID Detector | **DOC** | 🔑 *"all media generated by Imagen, Veo, and Lyria (**but not Chirp**) are watermarked"*; audio confined to Lyria/NotebookLM; Detector is waitlisted and detect-only | HIGH |
| 24 | https://github.com/PecholaL/IDEAW | **REPO** | Apache-2.0 code; **no weights**; authors' own "may be some errors" disclaimer | HIGH |
| 25 | https://arxiv.org/abs/2507.21150 · https://arxiv.org/abs/2508.17121 · https://arxiv.org/abs/2506.05891 · https://arxiv.org/abs/2601.22556 · https://arxiv.org/abs/2606.15187 | **PAPER** | WaveVerify (MIT, weights lic unverified) · **SyncGuard 0.5 s → 99.63 %, no code** · WAKE (8-bit key) · VocBulwark (in-model) · VoxWatermark | HIGH / MEDIUM |
| 26 | https://arxiv.org/abs/2511.02278 | **PAPER** | Multiplexing: AudioSeal + Perth complementary, 0.974 AUC / 0.905 TPR@0.05 over 11 attacks; *"peer-reviewed evaluation remains scarce"* for Perth | MEDIUM |
| 27 | https://github.com/SonyResearch/raw_bench · https://github.com/moyangkuo/AudioMarkBench | **REPO** | Robustness harnesses; AudioMarkBench = MPL-2.0 and pulls GPL-3.0 Timbre as a submodule | MEDIUM |
| 28 | https://arxiv.org/pdf/2505.23814 | **PAPER** | *Watermarking Without Standards Is Not AI Governance* — no harmonised standards, no verification infrastructure, no detection access | MEDIUM |

### Provenance, C2PA and platform rules

| # | URL | Type | Used for | Conf. |
|---|---|---|---|---|
| 29 | https://spec.c2pa.org/specifications/specifications/2.4/specs/_attachments/C2PA_Specification.pdf | **SPEC** | v2.4 (2026-04-01); A.1/A.3.4/A.3.5/A.3.7 audio embedding; §9.3.1 soft bindings; §2.4.1 durable CC; §18.28–29 `c2pa.ai-disclosure`; §15.7 untrusted signer | HIGH |
| 30 | https://github.com/c2pa-org/specifications (`softbinding-algorithm-list.json`) | **SPEC** | 🔑 **53 algorithms, 21 audio; `com.aiwatermark.audioseal.1` registered 2026-03-08**; published URL `spec.c2pa.org/softbinding-alg-list` is broken | HIGH |
| 31 | https://github.com/contentauth/c2pa-rs · c2pa-python | **REPO** | Dual MIT/Apache-2.0; audio = wav/mp3/flac/m4a only; **no Ogg/Opus (PR #2073 open)**; soft binding unimplemented (PRs #1299, #2399 open); `ai-disclosure` absent | HIGH |
| 32 | https://c2pa.org/conformance/ · opensource.contentauthenticity.org/docs/{conformance,trust-lists,durable-cr,signing} · ssl.com C2PA product page · c2pa.org/faqs | **SPEC/DOC** | Conformance programme; ITL frozen 2026-01-01; CAs Apr 2026; SSL.com free tier requires a conformance ID; watermark/fingerprint complementarity | HIGH |
| 33 | https://www.iso.org/standard/90726.html | **SPEC** | ISO/CD 22144 at stage 30.99 — **under development, not published** | MEDIUM |
| 34 | https://partner.steamgames.com/doc/gettingstarted/contentsurvey | **POLICY** | 🔑 Gen-AI section verbatim; *"content such as artwork, **sound**, narrative, localization"*; Pre- vs Live-Generated; Distribution Agreement promise; guardrails question; edit-lock | HIGH |
| 35 | store.steampowered.com (live product page) | **POLICY** | The public "AI Generated Content Disclosure" block as it actually renders | HIGH |
| 36 | https://itch.io/docs/creators/quality-guidelines · itch.io/t/4309690 | **POLICY** | *"Failure to tag your asset page may result in delisting"*; `AI Generated Sound` sub-tag | HIGH |
| 37 | learn.microsoft.com store-policies (v7.19) + certification-requirements (XR v16.3) | **POLICY** | §11.16 Live Generative AI verbatim; "Store" includes Xbox; **no AI XR in Xbox certification** | HIGH |
| 38 | partners.playstation.net · developer.nintendo.com | **POLICY** (gated) | ⚠️ **No public Sony or Nintendo AI-disclosure rule exists — UNVERIFIED, NDA-gated** | UNVERIFIED |
| 39 | https://legal.epicgames.com/epicgames/content-guidelines | **POLICY** | No AI mention in EGS content guidelines | MEDIUM |
| 40 | support.google.com/googleplay/android-developer/answer/{14094294,17262077,9888077} · developer.apple.com/app-store/review/guidelines/ | **POLICY** | Play GenAI-app scope + exclusions; store-asset-only AI declaration; Apple 5.1.2(i) third-party AI | HIGH |
| 41 | support.google.com/youtube/answer/14328491 · transparency.meta.com misinformation · tiktok.com/community-guidelines integrity-authenticity · newsroom.spotify.com 2025-09-25 | **POLICY** | YouTube triggers + C2PA auto-label + face-only likeness; Meta *"realistic-sounding audio"* duty + images-only auto-detection; **TikTok's generic-TTS exemption**; Spotify DDEX/ERN 4.3.1 + consent-gated impersonation | HIGH |
| 42 | sagaftra.org (2025 Interactive Media Agreement) | **POLICY** | 95.04 % ratification; Vocal Digital Replica covers all material; 7.5× Real Time Generation; **CBA, not law — binds signatories only** | MEDIUM (DataDome-blocked; via search index) |

### Regulation

| # | URL | Type | Used for | Conf. |
|---|---|---|---|---|
| 43 | https://ec.europa.eu/newsroom/dae/redirection/document/129555 | **REG** | **Code of Practice on Transparency of AI-Generated Content** (final 10 Jun 2026): Measure 1.1 two-layer marking verbatim; 1.1.1 signed metadata; 1.1.2 watermark; 1.1.3 logging insufficient; Commitment 2 detection; 1 M-user fee allowance; SME recital | HIGH |
| 44 | https://ec.europa.eu/newsroom/dae/redirection/document/131215 | **REG** | **Commission Art. 50 Guidelines** (20 Jul 2026): scope, audio, *"can plausibly exist"* deep-fake test, ¶2 *"as from 2 August 2026"*, provider/deployer split | HIGH |
| 45 | https://digital-strategy.ec.europa.eu/en/news/ai-omnibus-enters-force · /policies/{code-practice-ai-generated-content,guidelines-transparency-ai-generated-content,regulatory-framework-ai} | **REG** | Reg. (EU) 2026/1744 in force 27 Jul 2026; **Art. 50 NOT postponed**; ~190 CoP signatories; alternative-means requirement | HIGH |
| 46 | https://artificialintelligenceact.eu/article/{2,50,55,99,111,113}/ · /recital/133/ | **REG** (secondary reproduction) | Art. 50 full text, Art. 2 territorial scope, Art. 55 10²⁵ FLOP threshold (**we are out of scope**), Art. 99 fines + SME "whichever is lower", Art. 111(4), Recital 133 | MEDIUM-HIGH (50(2) corroborated HIGH by #43–44) |
| 47 | https://www.cencenelec.eu/news-events/news/2025/brief-news/2025-10-23-ai-standardization/ | **REG** | JTC 21 acceleration, Q4 2026 target; **no published harmonised standard yet** | MEDIUM |
| 48 | https://www.govinfo.gov/bulkdata/BILLSTATUS/119/s/BILLSTATUS-119s4591.xml · /content/pkg/BILLS-119s4591rs/html/BILLS-119s4591rs.htm | **REG** | NO FAKES **S.4591**: Senate Calendar No. 446, **not law**; §2(c)(2)(B)(i)–(iii) tool-provider liability; §2(d)(1)(A) safe harbour; digital-replica definition | HIGH |
| 49 | https://www.govinfo.gov/content/pkg/PLAW-119publ12/html/PLAW-119publ12.htm | **REG** | TAKE IT DOWN is **visual-only** — does not reach audio | HIGH |
| 50 | https://www.capitol.tn.gov/Bills/113/Bill/HB2091.pdf | **REG** | ELVIS Act eff. 1 Jul 2024; **§47-25-1105(a)(3) tool clause verbatim**; "voice" includes simulation; §47-25-1107(c) scienter | HIGH |
| 51 | https://docs.fcc.gov/public/attachments/FCC-24-17A1.pdf · FCC-24-84A1.pdf · federalregister.gov API | **REG** | CG 23-362 Declaratory Ruling (AI voice = "artificial"); NPRM Aug 2024; **no final AI-call rule through Aug 2026**; ⚠️ *"FCC withdrew"* claim is **UNVERIFIED** | HIGH |
| 52 | https://docs.fcc.gov/public/attachments/DOC-402762A1.pdf | **REG** | Kramer $6 M forfeiture (Biden robocall) — the fraud-vector precedent | HIGH |
| 53 | leginfo.legislature.ca.gov AB 853 / SB 942 (BPC ch. 25) | **REG** | Operative 2 Aug 2026; **1 M monthly users** threshold; latent disclosure content; detection tool with uploads + URLs + API | HIGH |
| 54 | https://www.meity.gov.in/static/uploads/2026/02/550681ab908f8afb135b0ad42816a1c9.pdf | **REG** | 🔑 IT Rules as on 10.02.2026: r.2(1)(wa), **r.3(3) verbatim** (not SSMI-limited), r.3(1)(ca), r.4(1A); **the "10 %" requirement was DROPPED** | HIGH |
| 55 | https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf | **REG** | DPDP Rules notified 14 Nov 2025; 18-month phasing; ₹250 cr / ₹200 cr / ₹50 cr penalties; children's data | HIGH |
| 56 | Asha Bhosle v. Mayk Inc., Bombay HC, IA(L) 30382/2025, order 29 Sep 2025 | **COURT** | 🔑 ¶15 tool-provider holding verbatim — Indian courts will enjoin a **tool provider** | HIGH |
| 57 | Anil Kapoor (Delhi HC CS(COMM) 652/2023) · Arijit Singh (Bombay HC, 26 Jul 2024) · Aishwarya Rai (Delhi HC CS(COMM) 956/2025) | **COURT** | Indian personality-rights line; all interim, no final adjudication | MEDIUM |

### Description-layer safety, impersonation and moderation

| # | URL | Type | Used for | Conf. |
|---|---|---|---|---|
| 58 | https://cyber.harvard.edu/people/tfisher/1988%20Midler.pdf | **COURT** | *Midler v. Ford*, 849 F.2d 460 (9th Cir. 1988): *"A voice is not copyrightable"*; *"To impersonate her voice is to pirate her identity"*; "deliberately imitated" holding | HIGH |
| 59 | https://law.justia.com/cases/federal/appellate-courts/F2/978/1093/183202/ | **COURT** | *Waits v. Frito-Lay*, 978 F.2d 1093: $375 k compensatory + $2 M punitive affirmed | MEDIUM |
| 60 | https://arxiv.org/abs/2503.04713 | **PAPER** | 🔑 **ParaSpeechCaps §3.1–3.2**: 594 named celebrities; GPT-4 as name→voice-attribute oracle; tag-equality ⟺ perceptual identity at cosine ≥ 0.8 | HIGH |
| 61 | https://arxiv.org/abs/2111.05095 | **PAPER** (ICASSP 2022) | 🔑 **Speaker Generation**: *"s2s, g2s and g2g will be equal"*; TacoSpawn 128-dim all = 0.20; MOS-by-locale table (**AU 164 speakers, −0.27**) | HIGH |
| 62 | https://arxiv.org/abs/2204.11304 | **PAPER** (IEEE) | 🔑 **Dictionary Attacks**: master voices match **69 % F / 38 % M at FAR 1 %**, black-box, no victim audio, transferable; *"accidental intrinsic bias… female speakers remarkably more vulnerable"* | HIGH |
| 63 | https://arxiv.org/abs/2410.03857 | **PAPER** | 🔑 **AIR implicit-reference jailbreak**: **> 90 % ASR**, inverse scaling, three published defences fail | HIGH |
| 64 | https://arxiv.org/abs/2511.10913 | **PAPER** (Nov 2025) | *Synthetic Voices, Real Threats*: red-teams 5 commercial TTS; obfuscation reduces refusals; moderation catches 57–93 %; explicitly **not** about impersonation | HIGH |
| 65 | https://arxiv.org/html/2608.13613 | **PAPER** (Aug 2026) | **VoiceDesigner** (JHU + Adobe): fuses text-to-voice **and cloning** in one model; SIM-o 0.757; **no ethics or misuse statement** | HIGH |
| 66 | https://arxiv.org/abs/2510.16489 · https://arxiv.org/abs/2110.03380 | **PAPER** | Voice identity ≈ 9 acoustic params ≈ 7 PCs, > 50 % variance over 10 000 speakers; 256-dim → ~20 dims | MEDIUM |
| 67 | https://elevenlabs.io/use-policy · /safety · /docs/eleven-creative/voices/voice-design | **DOC** | *"replicate the voice of another person"* (modality-neutral); safeguards are cloning-scoped; **Voice Design docs silent on real people** | HIGH |
| 68 | https://www.hume.ai/blog/how-to-clone-your-voice-with-ai · /designing-custom-voices-with-ai · /introducing-voice-control | **DOC** | 🔑 **Hume reversed** — OCTAVE clones from 5 s; the Dec 2024 framing was product-quality, not safety | HIGH |
| 69 | https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/speech-service/text-to-speech/limited-access · docs.cloud.google.com chirp3-instant-custom-voice · resemble.ai/our-commitment-to-consent | **DOC** | Azure CNV Limited Access; Google allow-list; Resemble consent workflow — **all cloning-scoped, none gate description** | HIGH / MEDIUM |
| 70 | https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/ (403 to automated fetch) | **DOC** | "Sky": independently cast, pulled anyway | MEDIUM (via NPR / WaPo / Variety) |
| 71 | https://huggingface.co/meta-llama/Llama-Guard-4-12B | **DOC** | **Llama 4 Community License** (not OSI); S1–S14 taxonomy; **English recall 69 % / FPR 11 %; multilingual recall 43 %**; Meta's own S5/S8/S13 knowledge caveat | HIGH |
| 72 | https://ai.google.dev/gemma/terms · gemma/docs/shieldgemma/model_card_2 · github.com/unitaryai/detoxify LICENSE | **DOC** / **LIC** | Gemma ToU flow-down verbatim; Detoxify Apache-2.0 | HIGH |
| 73 | https://arxiv.org/html/2604.25580v1 (+ Jigsaw sunset notice) | **PAPER** | **Perspective API out of service after 2026** — do not build on it | MEDIUM-HIGH |
| 74 | https://ceur-ws.org/Vol-2960/paper8.pdf · https://arxiv.org/html/2401.10825v3 | **PAPER** | Gazetteer *"pervasive spurious entity matching"*; +3.70 % F1 from spurious-match filtering | MEDIUM |
| 75 | https://arxiv.org/abs/2504.11168 | **PAPER** | Bypassing LLM guardrails: up to 100 % evasion against six production guardrails | MEDIUM |

### Bias and fairness

| # | URL | Type | Used for | Conf. |
|---|---|---|---|---|
| 76 | https://arxiv.org/html/2207.05929 | **PAPER** (Interspeech 2022) | Cross-age EER **1.939 % → 10.419 %** at ≥20 yr gap; `z = z_id + z_age` | HIGH |
| 77 | https://ar5iv.labs.arxiv.org/html/1909.06351 | **PAPER** (SLT 2018) | x-vector probing: gender ~99 %, rate ~100 %; ⚠️ **does not probe age** | HIGH |
| 78 | https://www.pnas.org/doi/10.1073/pnas.1915768117 | **PAPER** (PNAS 2020) | Koenecke: **WER 0.35 black vs 0.19 white**, five commercial systems | HIGH |
| 79 | https://ar5iv.labs.arxiv.org/html/2201.09486 | **PAPER** (FAccT 2022) | VoxCeleb2 61 % male / 29 % US; **Indian females 2.58×**; balance alone insufficient | HIGH |
| 80 | https://arxiv.org/html/2204.12649 | **PAPER** | USA 0.91 % vs India 8.63 % FA; DCAPLDA FDR 0.989/0.990; **fails when a group is absent** | HIGH |
| 81 | https://arxiv.org/html/2307.02009 | **PAPER** | Dutch ASR 9.6 / 42.9 / 59.0 % WER; augmentation 29.12 → 25.20 bias, non-native gap persists | HIGH |
| 82 | https://arxiv.org/html/2305.15760v1 | **PAPER** (Interspeech 2023) | **Svarah**: composition + per-L1 WER **4.5 (Maithili) → 11.6 (Bodo)**; per-style 6.2/7.4/11.2 | HIGH |
| 83 | https://arxiv.org/abs/2604.25476 | **PAPER** (2026) | 🔑 **PSP**: retroflex collapse Hindi ~1 % / Telugu ~40 % / **Tamil ~68 %**; *"PSP ordering diverges from WER ordering"* | HIGH |
| 84 | https://arxiv.org/html/2403.01926v1 · https://arxiv.org/html/2409.05356v1 · https://arxiv.org/html/2603.28714v1 · https://arxiv.org/html/2407.14056 | **PAPER/DATA** | IndicVoices (quotas, not realised); IndicVoices-R (1 700 h / 10 496 spk); Vaani (31 255 h / 165 districts); Rasa (~2 spk/language) | HIGH / MEDIUM |
| 85 | https://ar5iv.labs.arxiv.org/html/1810.03993 · https://www.arxiv.org/pdf/1803.09010v4 | **PAPER** | Model Cards (unitary + intersectional disaggregation); Datasheets subpopulation question verbatim | HIGH |
| 86 | https://pmc.ncbi.nlm.nih.gov/articles/PMC6583846/ | **PAPER** (J. Voice) | Presbyphonia **jitter 3.44 % vs 1.74 %, shimmer 7.82 vs 4.84**; sex-opposite F0 direction | HIGH |
| 87 | https://arxiv.org/html/2604.24770v1 | **PAPER** (2026) | Elderly TTS augmentation: 4.1 → 2.2 % / 11.6 → 4.8 % WER; ⚠️ **no acoustic validation of "agedness"** | HIGH |
| 88 | https://arxiv.org/html/2510.06927v1 · https://arxiv.org/html/2605.01597v1 | **PAPER** | TTS metric critiques; ITU-T P.808; **no sample-size or slice guidance exists in any TTS convention** | HIGH / MEDIUM |
| 89 | https://arxiv.org/abs/2607.05276 | **PAPER** (2026) | **ProPS** — description→MDN→GMM over x-vectors; essentially our architecture, published three months ago. ⚠️ **Highest-value unread source; results tables warrant a dedicated extraction pass** (see also [`02`](02-identity-representation.md)) | HIGH (method) |
| 90 | https://github.com/PranavMishra17/VoicePersona-Dataset | **DATA** | 🔑 Labels generated by **Qwen2-Audio-7B-Instruct** ⇒ the 62.6 % / 76.1 % skew figures are **model estimates, not measurements** | HIGH |

### Access failures — recorded honestly

| Resource | Failure | Consequence |
|---|---|---|
| **EUR-Lex** (Reg. (EU) 2026/1744, and the AI Act consolidated text) | AWS WAF bot challenge, HTTP 202 | EU statutory text is quoted from **official Commission PDFs** (HIGH) and a **secondary reproduction** (MEDIUM-HIGH). **Verify Art. 111(4) against the OJ before relying on the 2 Dec 2026 date.** |
| **openai.com** | HTTP 403 | Voice Engine, "Sky" and the GPT-4o system-card voice section are **MEDIUM via named outlets**; the system-card *"unintentional voice generation"* section is **UNVERIFIED — do not cite until opened manually.** |
| **sagaftra.org** | DataDome | 2025 IMA terms are **MEDIUM**; re-verify in a browser before customer-facing use. |
| **iso.org** | Refused fetch | ISO/CD 22144 stage is **MEDIUM**. |
| **partners.playstation.net · developer.nintendo.com** | NDA-gated | Sony and Nintendo AI-disclosure requirements are **UNVERIFIED. Do not represent their position from public sources.** |
| **congress.gov · indiankanoon.org** | HTTP 403 | Worked around via govinfo (primary) and court-uploaded PDFs. |
| **arXiv PDFs via WebFetch** | Binary, unparseable by the fetch tool | Worked around by downloading and extracting locally (pypdf) or reading page images. Numbers in this file were read from the actual tables, not from search snippets, except where tagged otherwise. |
