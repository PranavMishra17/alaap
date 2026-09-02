# PHASE 00 — Foundations

> **Goal:** prove the diagram end-to-end with zero training, build the measurement harness, settle the corpus, run the ten de-risking experiments, and be compliant from the first render.
> **GPU:** local (8–12 GB) · **Rented:** < $30 total · **Track:** ML, with Track B's schema contract
> **Supersedes:** scope §11 "S0 — Baseline & eval harness"
> **Evidence:** [`00`](../00-EXECUTIVE-VERDICT.md) · [`03`](../03-tts-backends-english.md) · [`05`](../05-datasets-and-annotation.md) · [`06`](../06-evaluation-harness.md) · [`08`](../08-licensing-propagation.md) · [`09`](../09-safety-and-watermarking.md) · [`12`](../12-speaker-manifold-navigability.md) · [`13`](../13-description-to-embedding-prior-art.md)

---

## Why this phase is bigger than the scope document assumed

Three things moved into S0 that scope §11 placed later:

1. **Corpus selection and licence clearance** (was S4) — ParaSpeechCaps is CC-BY-NC-SA, so the training corpus is now an open question that blocks S2.
2. **Compliance** (was S9) — EU AI Act Art. 50(2) applied 2026-08-02. Every render from now on is in scope.
3. **The experiment block** (didn't exist) — ten cheap experiments that de-risk the whole architecture before any code depends on it.

---

## 0. Exit criteria — all must be measured

| # | Criterion | Measured how |
|---|---|---|
| **X0.1** | **E1 returns `g2s ≈ s2s`** on the Qwen3-TTS space — or the Tier-2 pivot is formally taken and recorded | TacoSpawn §6.2 protocol; `g2s/s2s` ratio < 1.5 |
| **X0.2** | Baseline scorecard populated for English across all four eval axes | The table in §6 below, fully filled, no blanks |
| **X0.3** | Every corpus in the training set has a **traced upstream chain** and a written GO verdict | The GO-list table in §4, with licence URLs |
| **X0.4** | **Every rendered file is watermarked and provenance-logged** | Pick 10 renders at random; AudioSeal detects all 10; each traces to `(identity_id, backend_version, description, timestamp)` |
| **X0.5** | Identity store schema exists and represents **both tiers** | `INSERT` a Tier-1 and a Tier-2 identity; both round-trip |
| **X0.6** | All ten experiments have run and their results are recorded, including failures | `experiments/RESULTS.md` |

---

## 1. Environment — the real constraint is Python, not VRAM

**All candidate models fit in 12 GB simultaneously. What does not coexist is their dependency trees.** `parler-tts` hard-pins `transformers>=4.46.1,<=4.46.1` (exact). PromptTTS++ needs Python 3.8 / `torch==1.11.0+cu113`. Plan for **isolated venvs running as separate services from day one** — retrofitting this is painful. ([`04`](../04-indic-track.md))

```
envs/
  qwen3/        # primary: Qwen3-TTS Base + VoiceDesign
  parler/       # transformers==4.46.1 exactly — Indic designer
  prompttts/    # py3.8 + torch 1.11 — reference implementation, read-only
  eval/         # harness: whisper, wespeaker, ttsds, penn, brouhaha
  moss/         # MOSS-VoiceGenerator — Tier-2 English baseline
```

**Checkpoints to pin** (record exact revision hashes, not just names):

| Role | Repo | Licence |
|---|---|---|
| Tier-1 renderer | `Qwen3-TTS-12Hz-1.7B-Base` | Apache-2.0 |
| Voice designer | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Apache-2.0 |
| Tier-2 EN baseline | `OpenMOSS-Team/MOSS-VoiceGenerator` | Apache-2.0 |
| Cheap Tier-1 alt | `hexgrad/Kokoro-82M` | Apache-2.0 |
| Indic designer | `ai4bharat/indic-parler-tts` | Apache-2.0 |
| Reference impl | `line/promptttspp` + HF Space weights (1.32 GB + 660 MB) | Apache-2.0 |
| Watermark | `facebook/audioseal` | **MIT (code + weights)** |
| SV (eval) | `pyannote/wespeaker-voxceleb-resnet34-LM` | Apache-2.0 / CC-BY-4.0 |
| ASR EN | `openai/whisper-large-v3` | Apache-2.0 |
| ASR Indic | `ai4bharat/indic-conformer-600m-multilingual` | MIT |

**Do NOT install into any environment that touches training:** IndexTTS-2 (§3.4(c) bars using it *or its outputs* to improve any AI model, and §1.6 defines "Use" to include *running*), IndicF5, SPRING_F5, Indic-Mio, MioCodec, DhVaani, XTTS-v2, F5-TTS, VibeVoice, VoiceSculptor, Llasa-3B, xcodec2. ([`08`](../08-licensing-propagation.md), [`10`](../10-performance-control.md))

---

## 2. The experiment block — run these before building anything

Order matters. E3 and E0 are afternoon-scale and unblock the most.

### E3 — Are the embedding dimensions heterogeneously scaled? `~20 min, CPU`
Dump 1,000 real speaker embeddings from `extract_speaker_embedding()`. Plot per-dimension min / max / σ.
**Result to look for:** if ranges span > 1 order of magnitude, per-dimension rescaling is **mandatory** in the mapper's output layer (I1).
**Precedent:** a 704-d ECAPA+x-vector concat had its narrowest dim at `[−0.97, −0.11]` and its widest at `[−72.37, 81.59]`.

### E0 — Training-free emotion direction vectors `1 afternoon`
Implement `x_new = x(target, neutral) + α·τ_emotion` per [arXiv:2606.05367](https://arxiv.org/abs/2606.05367).
**Critical detail: `τ` must be averaged over ≥4 speakers.** Single-speaker `τ` leaks source timbre (SECS_W 0.810 vs 0.912).
**Target:** ΔEECS ≈ +0.29 at SECS_W ≥ 0.90. Artefact is ~64 KB.
**Also test:** steer the speech-LM (published cost ≈ 0.000 S-SIM) vs the flow decoder (−0.064). Prefer the LM.

### E1 — Is the `(2048,)` space navigable? `~4 GPU-h` ⭐ **most important**
Extract embeddings for ~500 LibriTTS-R speakers. Fit a 10-component GMM. Sample 200. Compute **s2s / g2s / g2g** exactly per TacoSpawn §6.2.
**Target:** `g2s ≈ g2g ≈ s2s`. TacoSpawn's learned space hit 0.20/0.20/0.20; its d-vector space failed at g2s 0.35 vs s2s 0.20.
**Pivot condition:** `g2s/s2s > 1.5` → Tier-2 wholesale (see [`GRAND-PLAN`](../GRAND-PLAN.md) §6.1).

### E2 — Vocoder drift `~2 GPU-h`
Render 100 identities, re-extract the embedding from the rendered audio, measure `cosine(input, output)`. Report mean / σ / worst.
**Why:** scope §4.3 claims Tier 1 reproduces "exactly, by construction." That is true at the input and false at the output.
**Decision:** if drift exceeds the `C_same` margin from E4, Tier 1 needs a closed loop (synthesize → re-extract → correct), which costs a GPU render per mint and erases Tier 1's economic advantage.

### E4 — Calibrate `C_same` and `C_diff` `~1 GPU-h`
50 speakers × 20 utterances. Compute intra-speaker and inter-speaker cosine distributions on `pyannote/wespeaker-voxceleb-resnet34-LM`.
**Why:** **no defensible universal threshold exists.** Published values are encoder-specific and non-comparable — SpeechBrain default 0.25, real same-speaker SIM-o 0.69–0.76, *different* real speakers 0.67 on another encoder. Calibrate in-run or every consistency number is meaningless.

### E5 — Does AudioSeal survive 1–3 s clips? `~2 GPU-h`
Fork Sony's `raw_bench`; sweep clip duration 1–10 s.
**Why:** all published evaluations are 10 s / 5 s / ~3 s. The AudioSeal paper's prose ("IoU 0.99 at 1 s") **contradicts its own Table 6 (0.802)**. Best analogue (XAttnMark): detection 98.6–99.3% at 1–10 s but **attribution collapses to 81.2% at 1 s**.
**Also test the attacks that actually matter:** polarity inversion (published 0.18/0.00 — an inaudible one-liner), Opus (removes it), OGG/Vorbis (0.95 — the Unity/FMOD/Wwise default), real reverb (0.22).

### E6 — Does PromptTTS++'s MDN give diverse voices? `~1 GPU-h`
Sample 20 style embeddings from one prompt. Render. Compute Vendi + GVD.
**Why:** this is the core A3 question answered on real, downloadable weights rather than by argument.

### E7 — Speaker count × session diversity `<10 GPU-h`
4 speaker arms × 2 session conditions × 3 seeds, on cached `z`, LibriTTS-P/R.
**Why:** no published ablation exists for a description→embedding mapper. Nearest evidence: at fixed 100 h, 60× speakers halves ECAPA EER — **but one session each is worse than 100 speakers with many sessions.**

### E8 — Real $/min on short lines `<$5, one afternoon`
Benchmark sweep per [`07`](../07-serving-and-cost.md) §9. Realistic game-dialogue line lengths (3–15 words), concurrency 1/8/16/32.
**Note:** published numbers disagree 3.2× and short prompts are *faster* in aggregate than long ones, contrary to scope §14's worry.

### E9 — Identity drift under heavy stylisation `~3 GPU-h`
Sweep aged / raspy / whispered / breathy / menacing intensity; measure SECS at each step.
**Why: nobody has measured this.** All published emotion-drift numbers cover the six canonical emotions, not the stylisation range scope §5 actually promises. Caveat every result: emotion2vec cosine drops **below chance** under speaker distractors ([arXiv:2604.26347](https://arxiv.org/abs/2604.26347)).

---

## 3. The eval harness — build before training anything

Four axes. Full method in [`06`](../06-evaluation-harness.md).

| Axis | Metric | Tool | Licence | Gate |
|---|---|---|---|---|
| **1. Adherence (objective)** | Per-attribute error vs target bins | `penn` (F0), Brouhaha (SNR/C50/VAD), `g2p-en`+`silero-vad` (rate) | MIT | **Primary gate** |
| **1b. Adherence (subjective)** | InstructTTSEval 3-tier | `gemini-2.5-pro` judge, binary | MIT dataset | Tiebreaker only |
| **2. Identity consistency** | Pairwise cosine vs calibrated `C_same` | `pyannote/wespeaker-resnet34-LM` | Apache-2.0 | Hard pass/fail |
| **3. Diversity** | **Normalized Vendi (target ≥0.35)** + **GVD (target ≥ −1 dB)** | `vendi-score` | MIT | Anti-collapse tripwire |
| **4. Intelligibility & naturalness** | WER (whisper-large-v3) + **TTSDS2** | `ttsds` | MIT | Regression guard |

### 3.1 Three traps baked into the harness

1. **Never use UTMOS/DNSMOS as a quality gate.** DNSMOS r = −0.788 with pitch; humans −0.059. It would systematically penalise high-pitched voices and fight axis 3. Use TTSDS2.
2. **Never report InstructTTSEval without the ceiling.** Real human reference audio scores **84.3 avg / 67.2 Role-Play**. Anything above ~85 is judge noise. Role-Play is our actual use case and is where everyone scores worst. ~$12.8/session EN.
3. **Aggregate to description level before significance testing** or you pseudo-replicate 20×. At N=50 descriptions you detect only **d ≈ 0.40**; the binary judge resolves **±10pp**. Use a paired bootstrap.

### 3.2 The fixed evaluation set

~50 character descriptions × a fixed dialogue script, **versioned, never changed without bumping the version.**

- Span age, gender, register, emotion, and stylisation — **and deliberately include low-density regions** (elderly, raspy, very low-pitched), because those are where quality cliffs live (+60% relative WER in sparse regions).
- **Line-length distribution must include short game-dialogue lines** (3–15 words), not just paragraphs.
- Tag every description with demographic slices so I10 (per-slice reporting) is free.

---

## 4. Corpus — settle the GO list

**ParaSpeechCaps is out** (CC-BY-NC-SA; its authors released their own fine-tune as NC-SA). Also out: CapSpeech, EARS, Expresso, Emilia, TextrolSpeech, SpeechCraft, GigaSpeech, VoxCeleb2.

### 4.1 The GO list

| Corpus | Hours | Speakers | Descriptions? | Licence | Role |
|---|---|---|---|---|---|
| **LibriTTS-P** | LibriTTS-R subset | **2,443** | ✅ human intrinsic adjectives | CC-BY-4.0 | ⭐ **The S2 training set** |
| LibriTTS-R | 585 | 2,456 | ❌ | CC-BY-4.0 (verified standalone LICENSE.txt) | Audio + targets |
| **GLOBE / GLOBE_V2** | — | **23,519** | ❌ | **CC0** | ⭐ Speaker diversity |
| MLS | large | many | ❌ | CC-BY-4.0 | Scale |
| Common Voice (CC0 subsets) | large | many | ❌ | CC0 | Scale |
| VCTK | 44 | 110 | ❌ | CC-BY-4.0 | Clean multi-session |
| IndicVoices-R | 1,704 | 10,496 | ❌ | CC-BY-4.0 *(explicitly "allowing commercial usage")* | Indic |
| Rasa | ~400 | 20 | ❌ | CC-BY-4.0 | Indic expressive |

**S2 trains on LibriTTS-P** — it has native human-written descriptions, so the measure-first annotation pipeline is **not** a blocker for S2. That pipeline (S4) is for *expanding* to GLOBE and IndicVoices-R.

### 4.2 Hold / verify

- **Google Crowdsourced Indic (SLR63–66/78/79)** is **CC-BY-SA** — the only ShareAlike in the mix. Do not mix into a corpus whose derivative you intend to release permissively.
- **IITM IndicTTS**: V2 EULA grants perpetual, sub-licensable, royalty-free rights with no NC clause — **but §2.2 forbids your downstream recipients from onward sale**, which is incompatible with an Apache-2.0 weight release. **Archive the V2 PDF now** — the URL every SPRINGLab README points at went 404 in Feb 2026.

---

## 5. Compliance from render #1

Not a launch task. EU Art. 50(2) has been live since 2026-08-02, and Art. 2(12) excludes Art. 50 from the open-source exemption.

| Requirement | Implementation | Stage |
|---|---|---|
| Watermark every render | **AudioSeal** (MIT code + weights; registered C2PA soft-binding algorithm `com.aiwatermark.audioseal.1`) | **S0** |
| Presence bit only — **no render ID in the payload** | 16-bit message averages 0.39 attribution, 0.69 clean | **S0** |
| Provenance log | `(identity_id, backend_id, backend_version, description, generation_params, user, timestamp)` per render | **S0** |
| Signed metadata layer | The Code of Practice requires **two** marking layers for audio | S6 |
| Free public detector | Required by the Code of Practice | S9 |
| Synthetic-audio disclosure in UI + export manifest | | S8 |
| **India "prominently prefixed audio disclosure"** | ⚠️ **COUNSEL QUESTION #1** — potentially product-destroying for a 1.5 s line, no technical workaround | before any Indic public serving |

**Skip C2PA manifests for v1** — the soft-binding registration gives the standards-endorsement benefit without the manifest plumbing.

---

## 6. The baseline scorecard — fill every cell

Populate with **zero training**. Every later improvement claim is measured against this.

| | MOSS-VoiceGen (Tier 2) | Qwen3-VoiceDesign (Tier 2) | Qwen3-Base + real spk vec (Tier 1 ceiling) | Indic Parler (Hindi) |
|---|---|---|---|---|
| Adherence — per-attribute error | | | | |
| Adherence — InstructTTSEval APS/DSD/**RP** *(ceiling 96.2/89.4/**67.2**)* | | | | |
| Identity consistency — mean pairwise cosine vs `C_same` | | | | |
| Identity — across process restart | | | | |
| Diversity — nVS *(target ≥0.35)* | | | | |
| Diversity — GVD *(target ≥ −1 dB)* | | | | |
| WER | | | | |
| TTSDS2 | | | | |
| **Per-slice: male / female / elderly / raspy / whispered** | | | | |
| RTF @ c=1 / c=32, short lines | | | | |
| VRAM peak | | | | |

**The Tier-1 ceiling column matters most.** Conditioning Qwen3-Base on a *real extracted* speaker vector tells you the best the two-tower design can possibly do. If the mapper later approaches it, the mapper is done.

---

## 7. Identity store schema

Must represent both tiers, both from day one (I2). Track B's contract — freeze it here.

```sql
CREATE TABLE identity (
  identity_id      uuid PRIMARY KEY,
  character_id     uuid NOT NULL,              -- one character, many languages (§4.5)
  language         text NOT NULL,
  description      text NOT NULL,

  -- Tier 1
  embedding        vector(2048),               -- pgvector; NULL if tier 2 only
  embedding_norm   jsonb,                      -- per-dim scale stats used (I1)

  -- Tier 2  (ALWAYS populated — I2)
  seed_audio_ref   text NOT NULL,              -- object-storage key
  seed_audio_sha   text NOT NULL,

  -- Reproduction recipe (Hume's precedent: store speech AND prompt AND params)
  generation_params jsonb NOT NULL,            -- seed, guidance, model_id, ...

  -- Provenance / migration safety
  backend_id       text NOT NULL,
  backend_version  text NOT NULL,
  identity_tier    smallint NOT NULL,
  minted_at        timestamptz NOT NULL,

  -- Search
  desc_embedding   vector(1024),               -- keep <2000 dims or HNSW won't index
  tags             text[]
);
CREATE INDEX ON identity USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON identity USING hnsw (desc_embedding vector_cosine_ops);
```

**pgvector note:** no gotcha at 2048-d (HNSW cap is 2,000 — **2048 exceeds it**, so index a reduced projection or use IVFFlat for the speaker vector). The trap flagged in [`07`](../07-serving-and-cost.md) was a 3072-d *text* embedding; keep `desc_embedding` ≤ 1024.

---

## 8. Build order

1. Environments + pinned checkpoints (§1)
2. **E3** (20 min) → decides the mapper output layer
3. Identity store schema (§7) → unblocks Track B
4. AudioSeal + provenance logging (§5) → every subsequent render is compliant
5. Eval harness (§3) + fixed eval set (§3.2)
6. **E4** → calibrates every threshold the harness needs
7. Baseline scorecard (§6) — zero training
8. **E0, E1, E2** → the architecture-deciding block
9. **E5, E6, E7, E8, E9** → the rest
10. Corpus download + licence-clearance table (§4), embeddings cached to disk
11. Write `experiments/RESULTS.md`, including failures

---

## 9. What "done" looks like

A table of baseline numbers nobody can argue with, a harness that will still be trustworthy in six months, an identity store that survives a model upgrade, ten experiments whose answers are written down, and a compliance posture that was correct from the first render rather than retrofitted.

**Then, and only then, S1.**

---

*Phase spec v2 · 2026-09-02 · [`GRAND-PLAN.md`](../GRAND-PLAN.md) · next: [`PHASE-01`](PHASE-01-identity-layer.md)*
