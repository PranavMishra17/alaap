---
name: speech-feature-pitfalls
description: Use when computing, interpreting, or debugging any acoustic measurement — F0, formants, VTL, HNR, jitter/shimmer, speaking rate, voicing, SNR — or when an axis "does not separate" and you need to know whether that is the world or the estimator.
---

# Speech feature estimation pitfalls

**An axis that fails to separate is a bug report about the estimator until proven
otherwise.** In this project two axes written off as noise were both broken measurements,
and no *new* axis has ever helped while two *repaired* ones have.

## The first check for any axis: within vs between

```python
within  = mean over speakers of that speaker's own variance (>= 2 clips each)
between = variance of the speakers' means
ratio   = within / between
```

- `ratio << 1` — identity axis (this project: `f0_mean` 0.08, `spectral_tilt` 0.19,
  `vtl_cm` 0.21)
- `ratio > 1` — one person varies on it more than people differ; **not** an identity axis
  (`speaking_rate` 1.32, `f0_cv` 1.04)

**The screen has a threshold, not a line.** `jitter` (0.73) and `shimmer` (0.62) are under
1.0 and still *hurt* when added — a weakly-separating axis competes with strong ones.

**Run it on a corpus of REAL speakers as a control for the control.** If an axis fails on
real people, the axis is wrong; if it fails only on your synthetic set, your set is
homogeneous.

## Formants and vocal-tract length

- **F2 is vowel-dependent**, swinging ~800–2200 Hz between front and back vowels. Its
  within-speaker spread swamps its between-speaker spread.
- **Never average *through* F2.** Fitch's formant dispersion, `c / (2 * mean spacing)`,
  averages F2−F1 and F3−F2 and therefore cancels what F1 and F3 know. In this project it
  gave gender separation of +0.11, −0.17, +0.06, +0.11 across four corpora — chance.
- **Use single-formant uniform-tube estimates instead**: the k-th resonance sits at
  `(2k-1)c / 4L`, so `L = ½(c/4F₁ + 5c/4F₃)`. Same corpora: **+0.70, +1.03, +0.44, +0.68**.
- **Run LPC at ~2× max formant, not the native rate.** At 24 kHz an order-26 filter spends
  its poles on the 5–12 kHz region, which carries no formant information, and locks onto
  harmonics for low-F0 voices. That inverted the gender effect entirely.
- Absolute VTL from a uniform tube **over-reads** (~19 cm where a phonetician says 15).
  Fine if you percentile-bin it; never quote it as anatomy.

## Voicing, HNR and the difference between them

- **`voiced_mask` is an energy+ZCR VAD, not a periodicity detector.** With an energy
  percentile of 40 it is structurally capped near 0.60, and it is **bimodal**: ~0.60 for
  anything majority-harmonic, 0.000 for majority-noise. A mild breathy whisper reads 0.60
  and looks untouched.
- **`hnr_db` is the graded probe** for breathiness/whisper — on synthetics it spans +7.8 dB
  (harmonic) to −10.1 dB (noise). Use it, and use voicing only as corroboration.

## Speaking rate

- Needs a **phone count**, which needs text. `measure(wav, "", sr)` silently yields NaN and
  the axis vanishes from your results table looking like it had nothing to say.
- **g2p-en does not raise on Devanagari** — the `[aeiouy]+` fallback matches nothing and a
  whole Hindi sentence becomes ~2.5 phones. Script-aware counting is required; see
  `count_phones_indic`.
- It is **phones per second of voiced audio**, so trailing silence depresses it. If you
  time-stretch and keep the padding, you move the axis you are trying to measure.

## Time-stretching and pitch

- **A phase vocoder preserves pitch; resampling does not.** Naive resampling shifted f0 by
  −54.7/+91.3 Hz and collapsed speaker identity to ECAPA 0.10 — a different person.
- `librosa.effects.time_stretch` raises `t_out values must be in the range [0, D.shape[-1])`
  on the final frame for some signal lengths. Pad before, truncate to the exact expected
  length after.

## Choosing a representation

- **ECAPA is trained to be INVARIANT to channel and quality** so it can identify a speaker
  through them. Never use it to measure quality.
- **WavLM layers differ in what they encode.** Early layers carry pitch — layer 0 correlates
  with f0 at −0.578, which is DNSMOS territory. Middle layers do not (−0.054 at layer 5).
  Sweep and check the pitch correlation, do not pick on general grounds.

## MOS predictors

**Do not use DNSMOS/UTMOS as a quality gate for anything spanning the pitch range.**
Measured against humans: humans correlate with mean F0 at **r = −0.059**, DNSMOS at
**−0.788**, UTMOSv2 at **−0.722**. They also move <0.1 on prosodic corruption that costs
humans 1.84 MOS points. A distance-to-real-speech-manifold score avoids both.

## Codecs

**Matching vocabulary size does not imply a shared codebook.** Two FSQ codecs both with
12800 entries accept each other's indices without raising, and produce fluent speech saying
completely different words. Verify by encoding one clip with both and comparing indices —
chance agreement for 12800 entries is 0.0078%.

## Checklist when an axis "does not work"

1. Is it NaN for a boring reason (missing text, wrong sample rate)?
2. Does the within/between ratio fail on **real** speakers, or only on your set?
3. Is the estimator averaging through a component that is unreliable?
4. Is the analysis rate right for what is being estimated?
5. Is what you are measuring what the name says (VAD vs periodicity, quality vs identity)?
