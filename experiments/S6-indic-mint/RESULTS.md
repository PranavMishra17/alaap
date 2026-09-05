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
f0_mean 3.69 | spectral_tilt 1.29 | f0_cv 0.66 | jitter 0.55
hnr_db 0.50 | shimmer 0.19 | speaking_rate 0.10
```

> **Corrected.** The first version of this file printed `f0_mean 3.85 … speaking_rate 0.16`
> over five axes. Those were the **wrong-codec** run's weights: when the codec was fixed
> the render-stage numbers were refreshed and this line was not. The values above
> reproduce exactly across repeated runs. The finding is unchanged and slightly stronger —
> `speaking_rate` is now last of seven.

Same ordering as GLOBE_V2/Qwen3 (`f0` 2.71) and LibriTTS-R/Qwen3-0.6B (`f0` 3.49). **Pitch dominates, speaking rate is nearly worthless for identity — on three corpora, three encoders, two languages.** `E15`'s finding is not an artefact of any one representation.

## S6b — minting costs no intelligibility

The open question S6 left was whether a *minted* vector renders a distinctive voice
saying slightly **wrong words**. A minted point is one the corpus never contained —
that is the whole point of minting, and also the reason MioCodec's decoder might not
handle it as cleanly as a real speaker's embedding.

**The design is a controlled difference.** Content tokens are generated **once per
line** and decoded through six embeddings: the four minted voices and **two real
corpus speakers**. The words are identical by construction, so the LM, the sampling
seed, the codec and `whisper-small`'s Devanagari spelling habits are common to both
arms and cancel:

| | mean CER | n |
|---|---|---|
| minted voices | **0.069** | 12 |
| real donors (control) | 0.081 | 6 |
| **cost of minting** | **−0.012** | |

**Negative — minting is not worse than using a real speaker's vector.** And the
difference is far smaller than the noise it sits in: on line 1 the two *real*
donors differ from each other by **0.077**, six times the minted-vs-donor gap.
Which real person speaks matters more than whether the voice was minted at all.

Every residual error is the ASR's orthography, not a wrong word:

```
ref     : नमस्ते आप कैसे हैं आज मौसम बहुत अच्छा है
donor0  : नमस्ते आप कैसे है  आज मुसम बहुत अच्छा है
guru    : नमस्ते आप कैसे है  आज मुसम बहुत अच्छा है
student : नमस्ती आप कैसी है  आज मोसम बहुत अच्छा है
```

### The control that had to pass first

One token stream per line means every arm must decode to the **same number of
samples**. That is asserted before any CER is read, and the run aborts otherwise —
a length difference would mean the arms are not saying the same thing, and the
difference would be measuring something else entirely.

It also caught its own defect. The first run picked donors as clips `[0, 1]`, which
under `--per-speaker 2` are the **same person** — so the "two-donor control" was one
speaker compared with herself, and because both arms carried the same label one of
them silently overwrote the other (`n=3`, not 6). Donors are now selected as the
first clip of two *distinct* speakers, and duplicate arm labels abort the run.

## Not established

- **4 voices, 3 lines each, one seed, Hindi only.** Small. `E11` needed 40 voices before the English catalog showed saturation; nothing here says where the Indic one saturates.
- **S6b is 12 minted scores against 6 control scores, on 3 Hindi lines, one ASR.** Enough to rule out a large cost, not enough to resolve a small one; the measured −0.012 should be read as "no detectable difference", not as minting being *better*. `ai4bharat/indic-conformer-600m-multilingual` remains the better instrument and remains gated.
- **Nobody has listened to these renders.** All machine-scored. Given that a listener confirmation on S5's *wrong-codec* audio established naturalness but not correctness, this matters.
- **Intelligibility is now measured for minted vectors** — see the section below. There is no detectable cost.
- `vtl_cm` is dropped on this corpus, mirroring S4: the formant estimate does not separate gender on IndicVoices-R (d = +0.11), so it is noise here. Indic captions run on seven axes here.
- `anchor_similarity` reads > 1 (e.g. 2.02) under `retrieval="hybrid"`, because the hybrid path returns a blended **z-score**, not a cosine. The field name is misleading in that mode and should be renamed or normalised.
- Licence: research-only per `ADR-009`. Indic-Mio's chain includes NC data; nothing here may ship as weights.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S6-indic-mint/run_indic_mint.py
envs/qwen3/Scripts/python.exe experiments/S6-indic-mint/run_indic_mint.py --skip-render   # mapper only
envs/qwen3/Scripts/python.exe experiments/S6-indic-mint/run_mint_intelligibility.py       # S6b, needs minted.npz
```

Needs `HF_TOKEN`, and S4's cache for the corpus (`experiments/S4-indic/out/measured_*.npz`). MioCodec embeddings are cached after the first run; delete `out/mio_emb_*.npz` to rebuild — **required if the codec changes**, since embeddings are codec-specific.
