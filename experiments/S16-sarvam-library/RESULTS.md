# S16 — measuring a real commercial voice library

**Run:** 2026-09-06 · Sarvam `bulbul:v3` · 37 studio voices · Hindi
**Question:** does the retrieval result survive contact with a real library, and does the measurement pipeline even work on commercial TTS output?

---

## Why this run mattered

`S8` showed that answering a description by *retrieving* the nearest library voice beats
*minting* a new one. But its "library" was 141 corpus speakers standing in for a studio
library, because there was no API key. Every conclusion carried the caveat *"nothing here
was run against Sarvam"*.

`S8` was written so swapping the library changes no logic. This is the swap.

## The headline: retrieval over a real library is weaker

| | S8 — 141 corpus speakers | **S16 — 37 Bulbul voices** |
|---|---|---|
| exact bin match, text retrieval | 39.0% | **30.7%** |
| random control | 20.3% | 22.2% |
| **lift over chance** | **+18.7 pp** | **+8.5 pp** |
| voices reached / total | 95 / 141 | 34 / 37 |

**The lift halves.** Retrieval still beats chance, but by much less. Two reasons, both
measured below: the library is a quarter the size, and it is *homogeneous* on an axis real
speakers vary on.

This is the direction predicted before the run — fewer voices should retrieve worse — but
the size of the drop was not.

## Coverage: 37 curated voices span 64% of the real-speaker range

Binned against the binner fitted on **IndicVoices-R**, so this measures where Bulbul's
voices land inside the range real Hindi speakers occupy:

| axis | bins occupied | which |
|---|---|---|
| f0_mean | 4/5 | very low → high |
| speaking_rate | 4/5 | slow → quick |
| spectral_tilt | 3/5 | balanced, bright, very bright |
| hnr_db | 3/5 | very rough, rough, slightly rough |
| **f0_cv** | **2/5** | expressive, highly animated |

**64% of the real-speaker range.** The gaps are informative: no *dark*-timbred voices, no
*clear*-toned ones at the top of the HNR range, and nothing below "expressive" on
variation — **no monotone voices at all.** A curated library is chosen to sound good, and
sounding good excludes regions a character brief might legitimately ask for.

> **The first version of this measurement was meaningless and I nearly shipped it.** It
> binned the 37 voices with a binner *fitted on those same 37 voices*, so percentile edges
> partitioned them by construction and every axis read 5/5 regardless of how homogeneous
> the library was. Coverage of a set measured against itself is always total.

## The control fired, and a control for the control settled it

Each voice was synthesised on **two different sentences**. If the pipeline measures the
voice rather than the sentence, a voice's two takes must sit closer to each other than to
other voices'. Three axes failed:

| axis | within / between |
|---|---|
| f0_mean | 0.09 ✓ |
| hnr_db | 0.72 ✓ |
| spectral_tilt | 1.09 ✗ |
| f0_cv | 1.05 ✗ |
| speaking_rate | 1.54 ✗ |

Two readings, pointing opposite ways: either the measurement is tracking the sentence — in
which case S16 is unusable and so is every S4/S6/S8 number built the same way — or these
voices genuinely do not differ on those axes.

**Running the identical statistic on real speakers separated them.** IndicVoices-R, two
clips each, same code:

| axis | corpus | bulbul | |
|---|---|---|---|
| f0_mean | 0.05 | 0.09 | both separate |
| spectral_tilt | **0.14** | **1.09** | measurement fine — **Bulbul is homogeneous** |
| hnr_db | 0.28 | 0.72 | both separate |
| f0_cv | **1.04** | 1.05 | **fails on real speakers too** |
| speaking_rate | **1.32** | 1.54 | **fails on real speakers too** |

### That splits three ways, and the third is the important one

1. **`f0_mean` and `hnr_db` work everywhere.** The measurement is sound on studio-clean
   synthetic speech, which was not obvious — the pipeline was built for corpus recordings.
2. **`spectral_tilt` separates real speakers (0.14) and not Bulbul's (1.09).** Bulbul's
   voices share a timbre. That is a fact about the library, not the instrument.

   > **`S24` re-measured this against the cross-corpus channel confound `S21` warned about
   > (Bulbul is 22050 Hz, IndicVoices-R 24000 Hz) and the claim survives, sharpened.** Under
   > a ⅓-octave estimator — which is 63% `f0_mean` — Bulbul separates at 0.11. Under the
   > shipped, pitch-independent one it does not. Both together say it better than either:
   > **Bulbul's voices differ in PITCH but share a TIMBRE.** The bandwidth difference is real
   > and moves nothing: the bright-side skew *grows* under a common band, +0.51 → +1.23 sd.
3. **`f0_cv` and `speaking_rate` fail on *real speakers*.** Two ordinary clips of the same
   person differ on them more than two people do. **They are not identity axes at all.**

## Point 3 independently replicates S12

`S12` measured exactly this on CREMA-D — 91 actors performing six emotions — and found
`speaking_rate` at 2.01 and `f0_cv` at 1.57 on the within/between ratio, concluding they
are *delivery* axes rather than identity axes.

Here the same two axes fail the same way on **plain read speech with no emotion involved,
a different corpus, and a different measurement path.** S12's central claim did not need
acted emotion to show up; it is a property of the axes.

It also explains, from a third direction, why `E15` kept measuring `speaking_rate` at an
identity weight of 0.10–0.31 across three corpora. It was never a weak identity axis. It
was not an identity axis.

### Which raises a question about the described space

Captions are written over five axes. **Two of them carry no identity information.** `S9`
found the described space holds only ~38 effective voices; a description that specifies
rate and expressiveness is spending two of its five words on things that do not
distinguish anybody.

That is a concrete, testable explanation for the capacity ceiling, and it was not on
`S9`'s list of three candidate causes. **It is the most promising thing to test next:**
re-fit the mapper on identity axes only and re-measure the bound.

## Not established

- **Hindi only, `bulbul:v3`, two sentences per voice.** Two takes give a noisy
  within-voice estimate; the corpus comparison used the same design deliberately, but more
  takes would tighten both.
- **Nobody has listened to a single Bulbul render.** They are in `out/audio/`. Everything
  above is geometry and bin agreement.
- **37 of 39 documented voices.** The card lists ~39; 37 names were tried and all 37
  worked. No voice was rejected by the API.
- **Retrieval here is `text` mode only.** `S8`'s hybrid path nearly doubled adherence
  (39.0% → 65.6%) and was not run here, so 30.7% is a floor for this library, not a
  ceiling.
- **This says nothing about Bulbul's quality.** It measures how well 37 voices cover a
  described space, not how good they sound — and by reputation they sound considerably
  better than anything in this repo.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S16-sarvam-library/run_measure_bulbul.py
```

Needs `SARVAM_API_KEY` in `.env` — the **API subscription key** from
`dashboard.sarvam.ai/key-management`, not a Voice Agents key. Both start `sk_`; a product
key carries the product name in its second segment and is rejected with the same message
as a garbage string. ~6 minutes and a few rupees on the first run; audio is cached
afterwards.
