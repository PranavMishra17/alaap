# S4 — the Indic caption pipeline

**Run:** 2026-09-05 · `SPRINGLab/IndicVoices-R_Hindi` · 250 clips / 141 speakers / Devanagari
**Question:** can we manufacture (caption, voice) pairs for Hindi, and are the measurements underneath them real?

---

## Why this had to be tested before producing anything

No Indic corpus carries natural-language voice descriptions. `PHASE-05` §8 requires them to be manufactured by the measure-first route, independent of the S5 gate. The whole route rests on the measurements, and for Hindi the measurement stack had just grown a **hand-written grapheme phone counter** (`count_phones_indic`) that nothing had validated.

That counter exists because the old one failed *silently*. `speaking_rate` tried `g2p-en`, then an `[aeiouy]+` vowel-group fallback. On Devanagari g2p-en does not raise — it returns junk — and the regex matches nothing, so `max(..., 1)` turned an entire Hindi sentence into 2.5 phones. **Every Indic clip would have read ≈0.5 phones/s and the speaking-rate axis would have collapsed into a single bin, with no error raised anywhere.**

So this experiment opens by trying to *falsify* the measurements rather than by writing captions.

## The control

The SPRINGLab mirrors ship Data-Speech annotations beside the audio — `utterance_pitch_mean`, `snr`, `speaking_rate` — produced by a **different toolchain** (their phonemizer and pitch tracker vs `librosa.pyin` and our grapheme rules). Two independent instruments measuring the same physical quantity must agree.

## Result 1 — the measurements are real

| comparison | Pearson | **Spearman** | floor | verdict |
|---|---|---|---|---|
| our `f0_mean` vs their `utterance_pitch_mean` | 0.987 | **0.986** | 0.90 | ok |
| our `snr_db` vs their `snr` | 0.348 | **0.360** | 0.30 | ok |
| our `speaking_rate` vs their `speaking_rate` | 0.970 | **0.966** | 0.70 | **ok — this is the one that mattered** |

Spearman is the statistic that matters, because percentile binning depends on rank, not value.

**`count_phones_indic` correlates at ρ = 0.966 with an independently-built phonemizer.** The abugida rules — a consonant carries an inherent schwa unless a matra or virama follows, with Indo-Aryan word-final schwa deletion — reproduce a real phonemizer's ranking on real Hindi.

`snr_db` at 0.360 is expected and was floored accordingly: our SNR is an energy-percentile estimate over a VAD, theirs is a different definition. It is a sanity check, not an equivalence.

**Scale offset.** Our phones/s over theirs: **median 0.855, IQR 0.825–0.883.** A tight, stable ratio below 1 is the predicted signature of **medial schwa deletion**, which their lexicon models and our grapheme rules deliberately do not. Percentile bins are invariant to a scale factor, so this costs nothing here — but it is a known systematic bias and it is why the test is correlation, not equality.

## Result 2 — Hindi cannot reuse English bins

Where Hindi clips land in the **English** (GLOBE_V2, n=2,500) bin edges. Even occupancy would be 0.20 across the row:

| attribute | Hindi median | English median | shift (English σ) | occupancy in the 5 English bins |
|---|---|---|---|---|
| `f0_mean` | 198.84 | 127.34 | **+1.59** | 0.03 0.08 0.10 0.24 **0.54** |
| `f0_cv` | 0.11 | 0.15 | −0.70 | **0.56** 0.18 0.10 0.07 0.09 |
| `speaking_rate` | 17.91 | 21.58 | −0.93 | **0.56** 0.24 0.13 0.05 0.02 |
| `hnr_db` | 2.25 | 0.08 | +0.76 | 0.02 0.03 0.06 0.22 **0.67** |
| `spectral_tilt` | −6.81 | −6.05 | −0.49 | 0.43 0.15 0.13 0.15 0.14 |

**Over half of all Hindi clips fall into the single top English pitch bin.** A shared binner would describe most Hindi speakers as "very high-pitched" and most as "monotone" and "very slow" — three axes collapsed at once. **Per-language binners are mandatory**, and `Binner.fit` is already per-corpus, so the pipeline was already right; this measures how wrong the alternative would have been.

> **Confound, stated plainly.** GLOBE_V2 and IndicVoices-R differ in more than language: elicitation (217 of 250 Hindi clips are *extempore*, GLOBE is read speech), recording setup, and demographics. This table therefore measures a **corpus** shift, of which language is one component and not necessarily the largest. It is sufficient to justify per-language bins — that decision only needs the distributions to differ — but it is **not** evidence that Hindi speakers are higher-pitched than English speakers, and must not be cited as such.
>
> Gender is balanced within the Hindi sample (130 F / 120 M), so the pitch shift is at least not an artefact of *this* corpus's gender skew.

## Result 3 — captions round-trip perfectly

**250/250 captions (100.0%) parse back to exactly the bins that generated them**, across all five axes the caption expresses:

| axis | round-trip |
|---|---|
| `f0_mean`, `hnr_db`, `spectral_tilt`, `f0_cv`, `speaking_rate` | **100.0%** each |

`snr_db`, `shimmer` and `jitter` are **not scorable**: `caption_from_bins` renders `ORDER[:max_attrs]`, five of the eight binned axes, so those three never appear in the prose and cannot be parsed back.

> ### A correction — the first version of this number was wrong
>
> The initial scorer required *every binned axis* to round-trip, including the three the caption never writes, and duly reported **0.0% exact**. That measured the scorer, not the captions. The per-axis breakdown gave it away immediately: five axes at 100.0% and three at 0.0% is the signature of a coverage bug, not a parsing failure.
>
> Recorded because the discipline that catches this is the same one that caught the HNR/pitch artefact in S2 run 3 and the GLOBE_V2 speaker-label failure in E4: **a number that is suspiciously bad from a component that should work is a bug report about your setup.** It applies to 0.0% exactly as much as to 100%.

## What this establishes

1. **The Indic caption pipeline works end to end** — stream → measure → bin → caption → verify — on real Hindi, validated against an independent toolchain.
2. **`count_phones_indic` is sound** (ρ = 0.966) with a known, quantified 0.855 scale bias from unmodelled medial schwa deletion.
3. **Per-language binners are required**, and the code already does this.
4. Route **C** in `ADR-006` (train our own Indic tower on IndicVoices-R) now has a working annotation pipeline in front of it. This is the piece that was missing.

## Caveats and what is not established

- **250 clips, 141 speakers, one language, one corpus.** Bengali and Tamil are wired (`indicvoices_r_bn`, `indicvoices_r_ta`) and not yet run — Tamil matters most, because it exercises the *Dravidian* branch where final-schwa deletion must **not** apply.
- **Licence: development only.** The mirror declares no licence of its own (CC-BY-4.0 upstream). `stream_clips` refuses it without `dev_only=True`. Shipping needs the gated original — `NEEDS-FROM-YOU` §1.
- **Nothing here was rendered.** Qwen3-TTS has no Indic language at all (`ADR-006`), so no Hindi audio was synthesised and no drift/consistency number exists for Indic. This validates the **annotation** half only.
- Medial schwa deletion is lexically conditioned and unmodelled; the 0.855 ratio is its size.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S4-indic/run_indic_captions.py --n 250 --per-speaker 2
```

Measurements cache to `out/measured_*.npz`; re-runs skip streaming. Add `--refit` to re-measure, `--corpus indicvoices_r_ta` for Tamil.

> **Operational note.** A first attempt at `--n 400` **hung silently** during streaming at ~1,821 rows — 0.5 s CPU and 0 bytes read per 20 s, no exception, no timeout. Cause unknown (it is *not* a shard boundary; shards are ~2,632 rows). The 250-clip run completes at 1,117 rows scanned, i.e. it stays below the failure point rather than fixing it. See `HANDOFF.md` §8.
