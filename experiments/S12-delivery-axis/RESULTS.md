# S12 — delivery and identity use disjoint axes

**Run:** 2026-09-05 · CREMA-D · 1200 clips, 91 actors, 6 emotions, 12 fixed sentences
**Question:** is *how a line is delivered* a different measurement from *who is speaking*?

---

## Why this is the gate on the whole direction channel

A description says who a character **is** — "a deep, gravelly voice". A direction says how
a line is **delivered** — "say this authoritatively", "say this asking for forgiveness".
Users write them differently, and the product needs both. Nothing in this project had
checked they are different *measurements*.

They might not have been. If "authoritative" mostly lowers pitch, it collides head-on with
`f0_mean` — the axis `E15`, `S6` and `S8` all independently measured as the **dominant
identity axis** (weight 2.7–3.9 against a mean of 1.0). A direction channel built on an
axis the identity channel owns makes every angry character sound like a different person.
That is the failure `E0`'s τ vectors exist to avoid, and `E0` only ever checked it in
embedding space, never acoustically.

CREMA-D is the only design that can answer it: the **same 91 actors** performing the
**same 12 sentences** across **6 emotions**. Content and identity are both held fixed, so
what moves is delivery.

## Result

```
delivery_ratio = within-speaker variance across emotions / between-speaker variance
```

| axis | within-speaker | between-speaker | **ratio** | verdict |
|---|---|---|---|---|
| shimmer | 0.002 | 0.001 | **3.12** | **DELIVERY** |
| **speaking_rate** | 3.996 | 1.992 | **2.01** | **DELIVERY** |
| jitter | 0.000 | 0.000 | **1.64** | **DELIVERY** |
| f0_cv | 0.003 | 0.002 | **1.57** | **DELIVERY** |
| spectral_tilt | 0.627 | 0.492 | 1.28 | contested |
| f0_mean | 4069.1 | 3905.2 | 1.04 | contested |
| hnr_db | 0.889 | 0.855 | 1.04 | contested |
| vtl_cm | 0.618 | 0.790 | 0.78 | contested |

## The two channels are near-perfect complements

Put S12's delivery ratios next to the identity weights this project measured
independently on three corpora, three encoders and two languages:

| axis | identity weight (`E15`/`S6`/`S8`) | delivery ratio (`S12`) |
|---|---|---|
| `f0_mean` | **2.72 – 3.85** ← dominant | 1.04 |
| `spectral_tilt` | 1.11 – 1.44 | 1.28 |
| `hnr_db` | 0.43 – 0.60 | 1.04 |
| `f0_cv` | 0.23 – 0.66 | **1.57** |
| **`speaking_rate`** | **0.10 – 0.31** ← worthless | **2.01** |

**The axes identity throws away are exactly the axes delivery uses.**

`speaking_rate` is the clearest case. This project has measured it as nearly useless for
identity five separate times and treated that as a mild curiosity. It is not a curiosity —
it is the signature of an axis that belongs to the *other* channel. A speaker varies their
rate at will, which is precisely why it cannot identify them and precisely why it can
direct them.

That is a designed-in result, not a lucky one: **description and direction can run on
disjoint axis sets**, and a direction that only moves rate, expressiveness and voice
quality cannot move who is speaking.

## The warning in the same table

`f0_mean` sits at **1.04**. Emotion moves a speaker's pitch about as much as different
speakers differ in pitch.

So the naive implementation of "make this angry" — raise the pitch — is the single most
damaging thing a direction channel could do, because pitch is what identity is mostly
made of here. **Any direction channel must be explicitly forbidden from moving `f0_mean`,
and that constraint has to be checked, not assumed.** `E0`'s τ vectors are averaged over
many speakers precisely to avoid dragging timbre along; this is the acoustic statement of
the same risk, and it says the danger is concentrated in one axis.

`hnr_db` (1.04) and `spectral_tilt` (1.28) are contested in the same way and should be
used for direction only with the identity cost measured.

## What was nearly missed

The first run of this experiment produced the table **without `speaking_rate`** — the axis
that turned out to matter most. `measure()` counts phones from text, the CREMA-D loader
passed `""`, and the axis came back NaN and silently dropped out of the results.

Nothing raised. The table looked complete and plausible: shimmer, jitter and `f0_cv` as
delivery axes is a perfectly reasonable finding, and it would have been published without
the one result that makes the complementarity argument.

The script now maps the 12 sentence codes to their texts and **asserts that
`speaking_rate` is measurable on at least half the clips before any ratio is printed**
(1200/1200). Same lesson as the wrong-codec run and the dead `f0_std` axis: *a silently
dropped axis looks exactly like an axis that had nothing to say.*

## Not established

- **Six acted emotions are not the delivery space.** "Authoritative", "soft", "asking for
  forgiveness" are not CREMA-D categories. This shows delivery *as acted emotion* lives on
  particular axes; whether finer-grained direction lives on the same ones is untested.
- **Acted, not spontaneous.** CREMA-D actors perform emotions on command, which is known
  to exaggerate. Real dialogue delivery is subtler and the ratios would likely shrink.
- **American English, 91 actors.** Nothing here has been checked on Indic delivery, where
  the whole prosodic system differs.
- **Variance ratios are not effect sizes.** A high ratio says an axis moves with delivery;
  it does not say a listener can hear the difference, nor that a TTS can render it.
- **Nothing has been synthesised.** This measures a corpus. Whether the frozen tower can
  be *made* to change `speaking_rate` without changing identity is the next experiment,
  and it is the one that decides whether any of this ships.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S12-delivery-axis/run_delivery_split.py
```

~6 minutes to stream and measure 1200 clips; cached after the first run. CPU only.
