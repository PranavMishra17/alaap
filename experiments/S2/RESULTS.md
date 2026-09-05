# S2 — The whole loop, end to end (baseline scorecard)

> **Status:** ✅ RUN AND COMPLETE · **Date:** 2026-09-05
> **This is the S0 baseline scorecard** (`PHASE-00` exit criterion X0.2) and the first working `description → voice` pipeline.
> **Corpus:** LibriTTS-R `train.clean.100`, 600 clips → **206 speakers** · **Mapper:** 206 grounded (caption, vector) pairs
> **Artefacts:** `out/results.json`, `out/audio/*.wav` (18 renders), `out/binner.json`, `out/mapper.npz`, `out/speaker_space.npz`

---

## 0. The loop runs

```
corpus → measure → bin → grounded caption
                              ↓
        speaker vectors → SpeakerSpace → retrieval mapper
                              ↓
     novel description → mint → render → RE-MEASURE → adherence
```

Every stage works. 18 voices minted from 6 descriptions × 3 novelty settings, rendered, and scored **by re-measurement rather than by an LLM judge** — which is the entire payoff of measure-first captioning.

---

## 1. Scorecard

| Metric | Result | Target | |
|---|---|---|---|
| **Diversity** — normalised Vendi, novelty 0.0 | 0.333 | ≥ 0.35 | ⚠️ ALARM |
| novelty 0.5 | **0.465** | ≥ 0.35 | ✅ |
| novelty 1.0 | **0.610** | ≥ 0.35 | ✅ |
| **Separability** — silhouette across descriptions | **0.584** | ≥ 0.15 | ✅ comfortably |
| **Adherence** — exact bin match, novelty 0.5 | **0.319** | chance 0.20 | ✅ above chance |
| Adherence — mean bin distance, novelty 0.5 | **1.292** | chance ~1.60 | ✅ |
| Speaker space | eff_rank 46.4, identity_frac 0.163 | — | consistent with E3 |

### Adherence across the dial

| novelty | exact-match | mean bin distance |
|---|---|---|
| 0.0 (pure retrieval/SLERP) | 0.194 | 1.333 |
| **0.5 (blend)** | **0.319** | **1.292** |
| 1.0 (pure GMM) | 0.264 | 1.458 |
| *chance (5 bins)* | *0.20* | *~1.60* |

**The blend beats both ends.** Pure retrieval sits at chance; pure generation is better; the 50/50 mix is best on both adherence measures. That was not the expected shape — retrieval was supposed to be the high-adherence end — and it is the most interesting result here.

**Diversity moves monotonically with novelty exactly as E1 predicted** (0.333 → 0.465 → 0.610), and pure retrieval lands in the ALARM band, which is E1's `nn_ratio = 0.20×` finding showing up as a product-level metric.

> **So novelty 0.5 is the current default**: best adherence, comfortably-passing diversity. The dial is real, calibrated, and its middle is genuinely the best operating point rather than a compromise.

---

## 2. Honest assessment: the mapper is weak

**Exact-match 0.319 against a chance baseline of 0.20 is a 60% relative improvement, and that is all it is.** The loop works; the mapper is not yet good. Reasons, in order of likely size:

1. **206 speakers is a tiny fit set.** LibriTTS-R `train.clean.100` has ~247 total.
2. **Template captions have limited vocabulary.** Every caption is generated from the same phrase bank, so the text encoder sees little variety, and retrieval keys off a narrow signal.
3. **Vocoder drift (E2: 0.5519 in working space).** The rendered voice is not exactly the minted vector, so some adherence loss is baked in before the mapper is even blamed.
4. **The corpus is clean audiobook read speech.** No genuinely gravelly, aged, or shouted voices exist to retrieve — percentile binning *labels* the roughest 20% "very rough", but LibriTTS-R's roughest is not a character voice.

Point 4 deserves emphasis: **percentile binning guarantees every bin is occupied, which hides the fact that the corpus range is narrow.** A speaker labelled "very low-pitched" here is only low *relative to other audiobook readers*. This is exactly the low-density-region problem RESEARCH/12 warned about, arriving through a side door.

---

## 3. A measurement bug was found and fixed mid-run

The first scoring pass produced adherence of **0.042 / 0.125 / 0.208** — at or *below* chance. Following the rule earned in E1 and E4 (a suspiciously bad number is a bug report about your setup), I checked the instrument before believing it.

**`target_bins_from_text` was matching substrings without word boundaries.** So:

```
"a low, smooth voice, unhurried and gently inflected"
   -> speaking_rate = "rapid"        # "hurried" matched inside "unhurried"
```

The exact opposite of what the description says. It also missed multi-word forms — "very clear" fell through to nothing because only "clean" was in the synonym table.

**Fixed** with regex word-boundary matching and longest-match-first synonym ordering. Re-scoring the *same renders* with the corrected parser:

| novelty | buggy parser | fixed parser |
|---|---|---|
| 0.0 | 0.042 | **0.194** |
| 0.5 | 0.125 | **0.319** |
| 1.0 | 0.208 | **0.264** |

**No re-rendering was needed** — the measured attributes were already stored, so only the target-bin parse changed. That is a direct benefit of saving measurements rather than just scores.

> Three experiments in a row have now been saved by distrusting a bad number: E4 (leakage, then a bad corpus), E1 (a saturated metric), S2 (a parser bug). The pattern is consistent enough to be a standing habit.

---

## 4. What to fix next, in priority order

1. **Scale the fit set.** `train.clean.360` gives ~900 speakers, 4× more. Cheap — it is the same code path.
2. **Diversify caption phrasing.** Either more templates, or an LLM writing *from the bins* (still grounded, still reversible) so the text encoder sees natural variety.
3. **Add a corpus with real vocal range.** LibriTTS-R cannot supply gravelly or aged voices. This is the argument for VoicePersona v2 — and, per RESEARCH/05, precisely the coverage lost when the anime sources are dropped for licence reasons.
4. **Close the drift loop** (E2's recommendation): re-extract after minting, re-mint if working-space drift < 0.40.
5. **Report per-demographic slices** (I10). Not yet done for adherence.

---

## 5. Caveats

- **18 renders is a small sample.** Per-description adherence varies from 0.00 to 0.50 and should not be over-read.
- **One line per voice.** Adherence is measured on a single fixed sentence; attribute estimates from ~4 s of audio are noisy.
- **No listening test.** All machine-scored. `out/audio/` has all 18; they need ears on them.
- **`f0_range` and `voiced_frac` are measured but not binned or captioned** — unused signal currently on the floor.

---

## 6. Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S2/run_s2.py --n 600 --per-speaker 3
```

~13 min (CPU measurement dominates), plus ~4 min of rendering. Delete `out/corpus.npz` to force re-extraction.
