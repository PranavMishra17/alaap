# S13 — the direction channel is real, safe, and weaker than I predicted

**Run:** 2026-09-05 · `SPRINGLab/Indic-Mio` + `MioCodec-25Hz-44.1kHz-v2` · 78 renders · 3 voices × 2 Hindi lines
**Question:** can a text tag move delivery without moving who is speaking?

---

## What was being tested

`S12` measured which acoustic axes carry delivery rather than identity, on a corpus of
real actors. That says the axes exist. It does not say a TTS can be **driven** along them,
and a direction channel that cannot be driven is not a channel.

`RESEARCH/10` states the claim, marked **UNVERIFIED on reliability**:

> *"Indic-Mio + MioCodec put identity in a decoder-side `global_embedding` and performance
> in text tags — genuinely separate channels."*

The tag goes in the **text**, so it changes content tokens; identity enters at decode as a
128-d vector the tag never touches. Structurally separate. This tests whether that
separation does anything.

**The noise floor is the whole experiment.** The LM samples at temperature 0.9, so two
renders of the same line with the same identity already differ. 30 neutral renders (5
repeats × 3 voices × 2 lines) establish that spread; every tag effect is quoted against it.

## Result 1 — identity is untouched. This is the clean pass.

ECAPA-TDNN, against the same voice rendered neutral:

| | raw | normalised |
|---|---|---|
| `<neutral>` vs itself (the ceiling) | 0.9014 | +1.407 |
| `<happy>` | 0.8846 | +1.373 |
| `<sad>` | 0.8840 | +1.372 |
| `<angry>` | 0.8801 | +1.364 |
| `<surprise>` | 0.8812 | +1.366 |

**A tag costs about 0.02 of ECAPA similarity against a self-similarity ceiling of 0.90.**
That is within the render-to-render spread. Tagging does not change who is speaking.

And the axis `S12` warned about most:

| | `<happy>` | `<sad>` | `<angry>` | `<surprise>` |
|---|---|---|---|---|
| **`f0_mean`** effect | 0.05 | 0.21 | 0.04 | 0.21 |

**Essentially zero on every tag.** I wrote a prediction down before running this: that
`<angry>` and `<happy>` would drag `f0_mean`, because emotion tags in the training data
were almost certainly performed with pitch changes, and `S12` put that axis at a contested
1.04. **That did not happen.** The structural separation holds precisely where it mattered
most.

## Result 2 — delivery does move, but two measures disagree and both are needed

| axis | S12 | `<happy>` | `<sad>` | `<angry>` | `<surprise>` |
|---|---|---|---|---|---|
| speaking_rate | 2.01 | 1.27 | 0.89 | 0.55 | 0.82 |
| f0_cv | 1.57 | 0.90 | **3.46** | 0.11 | **3.60** |
| jitter | 1.64 | 0.24 | 0.24 | 0.01 | 0.17 |
| shimmer | 3.12 | **1.72** | 0.95 | 0.92 | **1.40** |

*(effect in noise-floor units: |mean(tagged) − mean(neutral)| / SD(neutral))*

Read alone, `f0_cv` looks like the big win. It is not. The same effects as **effect / standard
error** — is the shift consistent, rather than large:

| axis | `<happy>` | `<sad>` | `<angry>` | `<surprise>` |
|---|---|---|---|---|
| speaking_rate | **6.69** ✓ | **3.80** ✓ | 1.88 | **2.58** ✓ |
| f0_cv | 1.15 | 1.76 | 0.34 | 1.75 |
| jitter | 0.77 | 0.57 | 0.02 | 0.46 |
| shimmer | **6.66** ✓ | **3.34** ✓ | **2.94** ✓ | **5.84** ✓ |
| f0_mean | 0.14 | 0.66 | 0.12 | 0.58 |

**The two measures disagree, and the disagreement is the finding.**

- **`f0_cv`'s huge 3.46 / 3.60 does not survive.** Those shifts are driven by a few outlier
  renders; the tagged variance is enormous. Large but unreliable.
- **`shimmer` and `speaking_rate` do survive**, at z up to 6.7 — small shifts, but
  *consistent* ones. `shimmer` is both the most reliable and the largest at 1.72 SD.
- **`jitter` does nothing at all**, on any tag, by either measure.
- **`<angry>` is the weakest tag**, barely moving anything except shimmer.

So the channel is **reliable but small**. The mean delivery effect is 1.08 noise-floor
units — roughly the size of re-rolling the sampling seed.

### The ratio in the script's output is flattering, and should not be quoted alone

```
delivery axes, mean effect  1.08
identity-risk axes, mean    0.25
ratio                       4.38
```

4.38 looks excellent. It is high because the **denominator is near zero**, not because the
numerator is large. The honest summary is two separate statements:

> **Identity protection: excellent.** **Delivery movement: real but marginal.**

## Against the prediction, written down before the run

| predicted | actual |
|---|---|
| `speaking_rate` and `f0_cv` move 2–4 noise units | `speaking_rate` peaks at **1.27**; `f0_cv` reaches 3.46 but is not reliable — **wrong** |
| ECAPA identity holds above +0.8 normalised | **+1.36 to +1.37** — right, and by a wide margin |
| likely failure: tags drag `f0_mean` | did not happen, ≤0.21 — **right that it mattered, wrong that it would fail** |

Half right. The safety properties came in better than expected; the magnitude came in
worse.

## What this means for the direction channel

**It is buildable and it is safe, but tags alone are not enough of a lever.** A 1.08-unit
mean effect will not reliably render "authoritative versus asking for forgiveness" — the
distinctions in the original product question are finer than a discrete six-tag vocabulary
delivering roughly one noise-unit of movement.

Three routes, in order of what the evidence supports:

1. **Stack the tag with explicit prosody control.** `speaking_rate` responds *consistently*
   (z=6.7) but *weakly*. It is also the axis a caller can set directly rather than
   requesting — resample or re-time the render to a target rate, and the tag supplies the
   rest. This is the only route the measurements actively endorse.
2. **Word-level emphasis (`*word*`).** Documented by Indic-Mio and **not tested here** —
   `RESEARCH/10` calls it the only backend with word-level emphasis by default. A local
   lever may do what a sentence-level tag cannot.
3. **Larger intensity via repetition or tag stacking.** Untested and the most likely to
   break intelligibility; would need `S6b`'s CER check run alongside.

## Not established

- **n is small: 12 renders per tag against 30 neutral.** Effects below ~0.8 SD cannot be
  distinguished from zero here. The reliable findings (shimmer, speaking_rate) would
  survive a bigger run; the unreliable ones (`f0_cv`) might resolve either way.
- **Two Hindi lines, one language, one seed family.** Tag behaviour on Tamil or English,
  or on longer/emotional text, is untested.
- **Nobody has listened.** `<happy>`, `<sad>`, `<angry>` and `<surprise>` renders are saved
  in `out/v0_*.wav` and **no one has confirmed they sound like the emotion requested.** An
  acoustic shift on `shimmer` is not evidence that a listener hears "happy". Given `S11`
  found a borrowed threshold wrong the moment a person listened, this matters.
- **Intelligibility was not re-checked under tags.** `S5b` measured CER for untagged text.
  A tag changes the content tokens, so it could cost intelligibility, and that has not
  been measured.
- **The delivery axes come from acted American English** (`S12`, CREMA-D) and are being
  applied to Hindi synthesis. The axis set may not transfer.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S13-direction-channel/run_direction.py
```

~20 minutes on a 6 GB card for 78 renders. Needs `S4`'s measurement cache and `S6`'s
MioCodec embedding cache.
