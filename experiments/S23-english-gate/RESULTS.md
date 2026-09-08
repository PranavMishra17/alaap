# S23 — the naturalness gate's layer does not transfer, and the check that said it would was wrong

**Run:** 2026-09-08 · GLOBE_V2 150 real / 56 English renders · plus `S20b`'s cached Hindi features
**Question:** `S20b` chose WavLM base+ **layer 5** on Hindi. `S20` says explicitly the layer
choice may not transfer. Does it?

---

## Verdict

**No — and fixing two flaws in `S20b`'s own checks changes its Hindi conclusion too.**

| | |
|---|---|
| Layer 5 on **English** | **FAILS** the pitch check — r = **+0.323** within real speech |
| Layer 5 on **Hindi**, re-scored | passes by **0.011** (−0.154 against a 0.165 bar) |
| **Layer 7** | passes everything in **both** languages, r = +0.043 / −0.022 |
| What that costs | layer 7 is **blind to the codec** (t = 0.28 vs layer 5's 2.59) |

`NATURALNESS_LAYER = 7` is now the default; `NATURALNESS_LAYER_CODEC = 5` is kept and
labelled Hindi-only.

---

## Two flaws in the inherited checks

Both were copied from `S20b` unexamined. Neither shows on Hindi alone, which is why a
second language found them.

### 1. The pitch correlation was pooled across groups

`S20b` correlated F0 with isolation over **real and synthetic clips together**. Synthetic
clips score ~50 points higher *and* have a different F0 distribution, so a pooled correlation
partly measures the group split rather than the metric.

The question "is this a pitch detector" is a **within-group** question, and within *real*
speech is the confound-free probe: naturalness is constant there, so any correlation is the
metric responding to pitch alone. It is also the comparison `RESEARCH/06` made — humans at
r = −0.059 were rating real speech.

> **The decisive statistic was changed after seeing the numbers, and that is worth flagging.**
> The pre-committed check was the pooled r, inherited from `S20b`. It read +0.342 on English,
> sitting next to the 0.35 threshold, which is what prompted looking at it at all. The
> argument for within-real is *a priori*, not fitted — but a reader should weigh that it was
> made second. All three correlations are reported so the choice can be disagreed with.

Within-synthetic is deliberately **not** used: `E11`'s catalog was sampled *across* the pitch
axis by construction, so if the TTS renders pitch extremes worse, a correlation there is real
rather than a defect of the gate.

### 2. The real-vs-real control was a single split

`S20b` split the held-out real set in half once and compared. Averaged over **200 random
splits**:

| corpus | single-split range | 200-split mean |
|---|---|---|
| English | 0.08 → 0.75 | **0.18 – 0.20**, every layer |
| Hindi | — | **0.11 – 0.13**, every layer |

**The single-split values were noise.** Six of twelve English layers were being failed on it,
and the layer that "passed" best (4, at d = 0.08) differed from the one that "failed" worst
(0, at 0.75) only by which half a clip fell in.

### 3. The threshold, once corrected, had stopped checking

Re-running with within-real r against `S20b`'s inherited 0.35 bar passed **all twelve
layers**. A check that rejects nothing is not a check. The bar is now the one `S20b` itself
earned when it called three layers winners on `gap >= 5` with a standard error of 6:
**is the correlation distinguishable from zero at this n** (|r| ≥ 0.236 at n = 75, ≥ 0.165 at
n = 150).

## The sweep, English

150 GLOBE_V2 clips (75 reference / 75 held-out), 56 English renders from `E11-catalog` and
`S15-english-listening`.

> The synthetic arm **had to be rebuilt.** `run_layer_sweep.py` hardcodes `S6-indic-mint` and
> `S7-indic-catalog` — Hindi. An English reference against Hindi renders separates them on
> *language* and reports a triumphant gate.

| layer | real | synth | t | r pooled | **r within real** | ctrl d | |
|---|---|---|---|---|---|---|---|
| 4 | 44.5% | 97.5% | 14.89 | +0.325 | +0.297 | 0.19 | pitch trap |
| **5** | 44.4% | 96.7% | 14.70 | +0.342 | **+0.323** | 0.19 | **pitch trap** |
| 6 | 44.6% | 98.4% | 15.17 | +0.292 | +0.219 | 0.19 | usable, marginal |
| **7** | 46.3% | **99.0%** | 14.36 | +0.213 | **+0.043** | 0.19 | **usable** |
| 8 | 47.3% | 99.0% | 15.24 | +0.228 | +0.073 | 0.19 | usable |

Seven of twelve pass all four checks: 0, 3, 6, 7, 8, 9, 10. **Layer 5 is not among them.**

## The same statistics, applied back to Hindi

`S20b`'s own cached features, identical code path.

| layer | real | synth | t | **r within real** | codec t (`S20b`) |
|---|---|---|---|---|---|
| 4 | 45.9% | 95.8% | 21.02 | **−0.370** pitch trap | **2.72** |
| **5** | 47.0% | 97.0% | 21.51 | −0.154 *(bar 0.165)* | **2.59** |
| 6 | 46.0% | 98.5% | 22.43 | **+0.004** | 1.98 |
| **7** | 46.8% | 97.2% | 19.80 | −0.022 | 0.28 |

Two things `S20b` did not say:

- **Layer 4 is a pitch trap on Hindi** at −0.370. `S20b` scored it −0.211 pooled and passed
  it, then chose layer 5 over it on that margin. The corrected statistic disqualifies it.
- **Layer 6 is cleaner than layer 5 on Hindi** (+0.004 vs −0.154) *and* separates better
  (t = 22.43 vs 21.51). It lost to layer 5 only on the codec question, at t = 1.98 against 2.

## The real finding: the two jobs are in tension

**The layers that detect the codec are the ones that carry pitch.**

| | sees the codec | clean on pitch, both languages |
|---|---|---|
| layer 4 | ✅ t = 2.72 | ❌ trap in both |
| layer 5 | ✅ t = 2.59 | ❌ fails English |
| layer 6 | ✗ t = 1.98 | marginal both |
| layer 7 | ❌ t = 0.28 | ✅ +0.043 / −0.022 |

**No layer does both,** and that is not a tuning problem — early-middle layers encode
source characteristics, which is simultaneously what makes codec artefacts visible and what
makes pitch leak in.

So the choice follows the job, and both constants are shipped named:

```python
NATURALNESS_LAYER       = 7   # flagging generated speech; clean in BOTH languages
NATURALNESS_LAYER_CODEC = 5   # tracking codec degradation; Hindi only, marginal
```

## Calibration, so "pitch trap" is not overstated

Layer 5's English r of +0.323 is far closer to humans' −0.059 than to DNSMOS's −0.788 in
absolute terms. It is called a trap on **significance**, not size: at n = 75 that correlation
is distinguishable from zero and the human value is not, so the gate responds to pitch in a
way humans do not. Layer 7's +0.043 sits *below* the human figure.

## Not established

- **The codec arm was not run for English.** It would need a Qwen3-TTS roundtrip, not
  MioCodec — the Indic codec is not on the English path — and the codec-carrying model does
  not fit beside WavLM at ~2 GB free (`HANDOFF` §8b). So "does the English gate see its
  codec" is untouched.
- **The English synthetic arm is 56 clips from two experiments**, one of which (`S15`) is a
  listening set built for a different purpose. Not an independent sample of English renders.
- **Still never rejected anything.** `NATURALNESS_FLOOR = 75` remains chosen, not measured.
- **Only two languages.** Layer 7 works for both; that is not evidence it works for a third.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S23-english-gate/run_english_sweep.py --phase audio
envs/qwen3/Scripts/python.exe experiments/S23-english-gate/run_english_sweep.py --phase feats
envs/qwen3/Scripts/python.exe experiments/S23-english-gate/run_english_sweep.py --phase sweep
# the same statistics on S20b's Hindi features:
envs/qwen3/Scripts/python.exe experiments/S23-english-gate/run_english_sweep.py --phase sweep \
  --feats experiments/S20-naturalness-gate/out/sweep_feats_indicvoices_r_hi_300.npz \
  --label "Hindi / IndicVoices-R (S20b cached)"
```
