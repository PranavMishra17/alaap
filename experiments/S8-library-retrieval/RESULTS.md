# S8 — description → nearest voice in a fixed library

**Run:** 2026-09-05 · 141-voice Hindi and 138-voice Bengali libraries from IndicVoices-R · 200 descriptions each
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

## Result — it works, and E15's hybrid retrieval nearly doubles it

Three arms, same 200 queries, same library:

| | random control | text cosine | **hybrid (weighted bins)** | hybrid − text |
|---|---|---|---|---|
| exact bin match | 20.3% | 39.0% | **65.6%** | **+26.6 pp** |
| within one bin | 53.0% | 66.5% | **87.3%** | **+20.8 pp** |

The control is not optional. Five axes × five bins gives a 20% per-axis hit rate for
free, so 39% reads as a real number only once 20.3% sits beside it.

**Text cosine roughly doubles chance. Hybrid roughly doubles text.** Two thirds of
queries retrieve a voice that matches the description **exactly on every axis**, and
87% land within one bin — which is inside the width of the bins themselves.

### Why the hybrid gap is this large

`E15` found the same thing on English and it is worth restating, because it is the
single highest-leverage line in the retrieval path. A sentence embedding treats
"speaks quickly" and "very deep" as comparably informative English. They are not:

```
hybrid axis weights, measured on this library
f0_mean 2.72 | spectral_tilt 1.11 | hnr_db 0.60 | f0_cv 0.31 | speaking_rate 0.26
```

Pitch carries **ten times** the identity information of speaking rate. The hybrid path
blends the text cosine with a bin distance weighted by those measured values, and
weights the blend by the *share of identity information the description actually
pinned down* — so a description naming no acoustic words falls back to pure text and
can never be worse off.

Both arms go through `RetrievalMapper.retrieve`, the same ranking `mint` uses. A
retrieval experiment that reimplements the ranking measures its own reimplementation.

### But those queries are generated captions, not what a user types

Every query above is a `caption_from_bins` output naming all five axes. Real users do
not write like that — `E15b` measured generated captions yielding 5 of 6 axes and
realistic user text yielding **1.29 (22%)**. Reporting only the caption number would
present the ceiling as the operating point.

So: a second query set naming just **two** axes in the bin's own words
(*"a balanced, slightly rough voice"*), scored **only on the axes it named** — asking
whether a voice has an attribute the user never mentioned is not a fair question.

| | random | text | **hybrid** |
|---|---|---|---|
| exact match on the named axes | 21.2% | 38.2% | **53.8%** |

**Adherence falls from 65.6% to 53.8% when the query goes sparse, and hybrid still
beats chance by 2.5× and text by 15.6 pp.** The degradation is real and should be
quoted as the operating number; the 65.6% describes a user who writes like the system's
own captions.

## The number that actually decides library-vs-minting

Not adherence — **reach**. A library of 141 that answers every description with the same
12 voices is a library of 12.

| | |
|---|---|
| library voices ever reached, hybrid | **82 / 141 (58%)** |
| library voices ever reached, text | 95 / 141 (67%) |
| most-returned single voice | 9 / 200 queries (4%) |

No collapse onto a handful of favourites. And measured **in the same MioCodec space S7
used**, so the two are like for like rather than two different metrics compared as one:

| | normalised Vendi | effective voices |
|---|---|---|
| **S8, hybrid retrieval** | 0.405 | **~33** |
| **S8, text retrieval** | 0.361 | ~34 |
| **S7, minted catalog** | 0.362 | **~14** |

**Retrieval reaches ~2.4× the effective diversity of minting, on the same corpus, in the
same space, scored by the same metric.**

Note that hybrid reaches *fewer* voices (82 vs 95) but scores *higher* Vendi (0.405 vs
0.361) and lands on the same effective count. It is not trading diversity for accuracy —
it is discarding voices that were being returned for the wrong reasons.

### And the text and minting Vendi scores are identical

0.361 versus 0.362. That is the finding under the finding.

Normalised Vendi measures how much of the *maximum possible* diversity a set achieves.
Both methods land on the same fraction — so **the ceiling is a property of the space,
not of the method**. Minting does not compress the space more than retrieval does; it
simply reaches fewer of its points, because a uniqueness floor rejects a new voice that
lands near an existing one, whereas retrieval just returns the neighbour it found.

That reframes S7's result. The Indic catalog is not small because minting is weak. It is
small because 128-d MioCodec identity space, described through five acoustic axes, holds
about a third of its nominal diversity — and minting then reaches less of that third.

## It replicates on a second language and a disjoint set of speakers

Bengali: 138 library voices, none of them the same people, a different binner fitted on
different audio, the same 200 stratified descriptions.

| | Hindi (141 voices) | Bengali (138 voices) |
|---|---|---|
| exact bin match, hybrid | 65.6% | **66.5%** |
| within one bin, hybrid | 87.3% | **89.8%** |
| sparse two-axis query, hybrid | 53.8% | **53.8%** |
| random control | 20.3% | 22.1% |
| voices reached | 82 / 141 | 91 / 138 |
| normalised Vendi | 0.405 | 0.379 |
| **effective voices** | **~33** | **~34** |

Every number lands within a point or two, and the sparse figure is identical to three
significant figures on 400 independent decisions each. **This is a property of the
method, not of the Hindi corpus.**

That also makes the ~14 from `S7` look less like an Indic-Mio quirk and more like a
real gap between selecting and synthesising.

## What retrieval cannot do

Set against the above, honestly:

- **It cannot produce a voice the library does not contain.** Even at 65.6% exact,
  a third of queries get a voice wrong on at least one axis. Ask for a voice at a corner
  of bin space with no library speaker near it and you get the nearest neighbour, not the
  voice you asked for. Minting *interpolates*; retrieval only *selects*.
- **It gives no uniqueness guarantee for a new character.** Two different descriptions
  can and do return the same voice. Minting rejects that; retrieval cannot.
- **The library is someone else's.** No control over what is in it, and every character
  in every project drawn from the same 39 voices.

## Not established

- **The library here is corpus speakers, not TTS voices.** Real recordings carry channel
  and microphone variation a studio library would not. That likely *helps* diversity and
  *hurts* adherence, so the true Bulbul numbers could move either way.
- **200 queries, one corpus, one seed.** Both retrieval arms were run; nothing else
  was varied. `top_k=1` throughout — an interface that offered a shortlist of three
  would score far higher and is arguably the honest product design.
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
