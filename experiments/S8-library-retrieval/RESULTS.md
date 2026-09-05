# S8 — description → nearest voice in a fixed library

**Run:** 2026-09-05 · 141-voice library from IndicVoices-R Hindi · 200 descriptions
**Question:** can you skip minting entirely and just *retrieve* the closest voice a TTS already has?

---

## Why this was worth testing before anything else

Every commercial Indic TTS ships a curated voice library rather than per-request speaker
synthesis. Sarvam's Bulbul has **39 studio voices** across 11 Indian languages. If a
description can reliably pick the right one, that path has no mapper, no drift floor, no
consistency floor, no NC-licence problem, and studio quality by construction.

It is testable today without an API key, because a "library voice" is just a fixed
identity with measurable acoustics and a caption. The 141 real Hindi speakers already
measured in `S4` are exactly that — and a *harder* library than Sarvam's, since they
were recorded, not curated for coverage. **Swap in 39 Bulbul voices and not one line of
the script changes.**

## Result — it works, and against the control it works clearly

| | retrieved | random control | lift |
|---|---|---|---|
| exact bin match | **39.0%** | 20.3% | **+18.7 pp** |
| within one bin | **66.5%** | 53.0% | **+13.5 pp** |

The control is not optional. Five axes × five bins gives a 20% per-axis hit rate for
free, so 39% looks like a real number only once 20.3% is sitting next to it. Retrieval
roughly **doubles** the exact-match rate over chance, and two thirds of retrieved
voices land within one bin of what was asked for.

## The number that actually decides library-vs-minting

Not adherence — **reach**. A library of 141 that answers every description with the same
12 voices is a library of 12.

| | |
|---|---|
| library voices ever reached, over 200 queries | **95 / 141 (67%)** |
| most-returned single voice | 8 / 200 queries (4%) |

No collapse onto a handful of favourites. And measured **in the same MioCodec space S7
used**, so the two are like for like rather than two different metrics compared as one:

| | normalised Vendi | effective voices |
|---|---|---|
| **S8, library retrieval** | 0.361 | **~34** |
| **S7, minted catalog** | 0.362 | **~14** |

**Retrieval reaches 2.4× the effective diversity of minting, on the same corpus, in the
same space, scored by the same metric.**

### And the Vendi scores are identical

0.361 versus 0.362. That is the finding under the finding.

Normalised Vendi measures how much of the *maximum possible* diversity a set achieves.
Both methods land on the same fraction — so **the ceiling is a property of the space,
not of the method**. Minting does not compress the space more than retrieval does; it
simply reaches fewer of its points, because a uniqueness floor rejects a new voice that
lands near an existing one, whereas retrieval just returns the neighbour it found.

That reframes S7's result. The Indic catalog is not small because minting is weak. It is
small because 128-d MioCodec identity space, described through five acoustic axes, holds
about a third of its nominal diversity — and minting then reaches less of that third.

## What retrieval cannot do

Set against the above, honestly:

- **It cannot produce a voice the library does not contain.** 39% exact adherence means
  three axes in five are wrong on average. Ask for a voice at a corner of bin space with
  no library speaker near it and you get the nearest neighbour, not the voice you asked
  for. Minting *interpolates*; retrieval only *selects*.
- **It gives no uniqueness guarantee for a new character.** Two different descriptions
  can and do return the same voice. Minting rejects that; retrieval cannot.
- **The library is someone else's.** No control over what is in it, and every character
  in every project drawn from the same 39 voices.

## Not established

- **The library here is corpus speakers, not TTS voices.** Real recordings carry channel
  and microphone variation a studio library would not. That likely *helps* diversity and
  *hurts* adherence, so the true Bulbul numbers could move either way.
- **200 queries, one corpus, one seed, `TextEncoder` cosine only.** The hybrid
  weighted-bin retrieval from `E15` was not used, and it beat plain text retrieval there
   — these adherence numbers are a floor, not a ceiling.
- **No audio was rendered or listened to.** Adherence is scored by re-binning already
  measured attributes, which is honest for "does the voice match the description" and
  says nothing about how it sounds in Bulbul's mouth.
- **Nothing here was run against Sarvam.** No API key. Every number is a stand-in
  measured on a corpus, and the claim is that the *method* transfers, not that these
  exact values will.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S8-library-retrieval/run_library_retrieval.py
```

CPU only, ~40 seconds. Reads `S4`'s measurement cache and (optionally, for the Vendi
comparison) `S6`'s MioCodec embedding cache.
