# S6 — minting Indic voices from a description

**Run:** 2026-09-05 · `SPRINGLab/Indic-Mio` + `MioCodec-25Hz-44.1kHz-v2` · Hindi · 250 corpus clips / 141 speakers
**Question:** can a description produce a voice that has never existed, rendering arbitrary Hindi?

---

## Why this is the step that matters

S5 carried an **existing** speaker's vector onto new text. That is voice cloning — useful, but it needs a recording of every character. The product claim is different:

> *"a very deep voice, very rough and gravelly, speaking very slowly"* → a voice nobody has recorded, saying whatever you write, in Hindi.

Everything needed already existed and had been validated separately:

| | |
|---|---|
| `S4` | measure Indic audio → percentile bins → captions (3 languages, cross-checked against an independent toolchain, 100% round-trip) |
| `S5` | content tokens + an arbitrary 128-d vector → correct Indic speech |
| `E15` | retrieve on **weighted bins** rather than sentence embeddings |

What was new is the **join**: `(caption, MioCodec embedding)` pairs. Nothing had built those, because no Indic corpus ships voice descriptions and MioCodec embeddings had never been measured against captions.

## The alignment check, which has caught a real failure before

S4's attributes and MioCodec's embeddings must describe the *same clips*. `stream_clips` is deterministic, so re-streaming gives the same order — but **E14b caught that exact assumption being false once**, at r = 0.011, when the duration bounds differed. So it is verified, not assumed, and the run aborts otherwise:

```
f0 fresh vs S4-cached: r = 1.0000
aligned.
```

## Result — it works, and it beats the English pipeline

Four voices minted from descriptions the mapper was **not** fitted on, each rendering three Hindi lines:

| voice | uniqueness | drift | consistency |
|---|---|---|---|
| guru | 1.000 | 0.808 | 0.717 |
| student | 1.162 | 0.676 | 0.710 |
| narrator | 0.770 | 0.745 | 0.716 |
| elder | 0.659 | 0.828 | 0.710 |
| **floors** | **0.30** | **0.40** | **0.43** |

Every voice clears every floor with room to spare. Against the English path measured in `E11` on the same three metrics:

| | Indic (S6) | English (E11) |
|---|---|---|
| drift, mean | **0.764** | 0.457 |
| consistency, mean | **0.713** | 0.564 |
| uniqueness, mean | **0.898** | 0.616 |

### The consistency number is the interesting one

Scored by **ECAPA**, which never saw MioCodec, against its real-speech calibration (`C_same` 0.6988, `C_diff` 0.2011):

| | raw | normalised |
|---|---|---|
| within one minted voice, across 3 different sentences | **0.7129** | **+1.03** |
| between different minted voices | 0.3004 | +0.20 |

**A minted voice is *more* self-consistent than a real person is across recordings.** That is not a flaw and not a surprise: a fixed 128-d vector has no bad-throat day, no change of microphone, no mood. Real speakers vary; this does not.

And **different minted voices measure as different people** (+0.20, much nearer "different" than "same"), while each sits **0.19–0.21 away from the nearest real speaker in the corpus** — so these are new voices, not retrievals of corpus speakers wearing a caption.

### The axis weights replicate on a third representation

The mapper measures its own axis weights at fit time. On MioCodec's 128-d space:

```
f0_mean 3.85 | spectral_tilt 1.44 | f0_cv 0.49 | hnr_db 0.47 | speaking_rate 0.16
```

Same ordering as GLOBE_V2/Qwen3 (`f0` 2.71) and LibriTTS-R/Qwen3-0.6B (`f0` 3.49). **Pitch dominates, speaking rate is nearly worthless for identity — on three corpora, three encoders, two languages.** `E15`'s finding is not an artefact of any one representation.

## Not established

- **4 voices, 3 lines each, one seed, Hindi only.** Small. `E11` needed 40 voices before the English catalog showed saturation; nothing here says where the Indic one saturates.
- **Nobody has listened to these renders.** All machine-scored. Given that a listener confirmation on S5's *wrong-codec* audio established naturalness but not correctness, this matters.
- **Intelligibility not measured for S6 specifically.** S5b measured it for the same LM + codec path (CER 0.000 on Tamil/English, 0.114 on Hindi, entirely `whisper-small` orthography) — but that was with *donor* embeddings, not *minted* ones. A minted vector is further from the training distribution and could plausibly cost intelligibility. **This is the next thing to measure.**
- `vtl_cm` is dropped on this corpus, mirroring S4: the formant estimate does not separate gender on IndicVoices-R (d = +0.11), so it is noise here. Indic captions run on five axes where English runs on six.
- `anchor_similarity` reads > 1 (e.g. 2.02) under `retrieval="hybrid"`, because the hybrid path returns a blended **z-score**, not a cosine. The field name is misleading in that mode and should be renamed or normalised.
- Licence: research-only per `ADR-009`. Indic-Mio's chain includes NC data; nothing here may ship as weights.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S6-indic-mint/run_indic_mint.py
envs/qwen3/Scripts/python.exe experiments/S6-indic-mint/run_indic_mint.py --skip-render   # mapper only
```

Needs `HF_TOKEN`, and S4's cache for the corpus (`experiments/S4-indic/out/measured_*.npz`). MioCodec embeddings are cached after the first run; delete `out/mio_emb_*.npz` to rebuild — **required if the codec changes**, since embeddings are codec-specific.
