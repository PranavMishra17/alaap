# S11 — a listener says the uniqueness floor was too low

**Run:** 2026-09-05 · 8 blind pairs · one fluent Hindi listener
**Question:** do the geometry numbers correspond to anything a person hears?

---

## Why this outranks every other measurement here

`S7`–`S10` are geometry. "22 effective voices" is a Vendi artefact, and the `uniqueness`
floor of **0.30** was borrowed — VoicePrivacy B3's threshold, applied in a working space
nobody had validated it for.

If two voices 0.30 apart are the same person to a listener, the floor is too low, the
catalog is an overcount, and every diversity number in this project measures something
that does not exist. That is one question and it needed a human.

## The controls passed, so the answers are readable

| pair | distance | truth | heard | | kind |
|---|---|---|---|---|---|
| 7 | 0.000 | same | **same** | ✓ | positive control — one voice, two sentences |
| 8 | 0.000 | same | **same** | ✓ | positive control |
| 5 | 0.623 | different | **diff** | ✓ | negative control — two real speakers |
| 4 | 0.899 | different | **diff** | ✓ | negative control |
| | | | | | |
| 1 | 0.323 | different | *same* | ✗ | test — just above the floor |
| 6 | 0.323 | different | **diff** | ✓ | test — just above the floor |
| 3 | 0.506 | different | **diff** | ✓ | test — comfortably accepted |
| 2 | 0.760 | different | **diff** | ✓ | test — far apart |

**Controls 4 / 4.** The listener recognised one voice across two different sentences as
the same person, and two real speakers as different people. Without that the test-pair
answers would mean nothing — this is the same discipline `S8` applied with its random
retrieval baseline.

**Test pairs 3 / 4, and the miss is exactly where it matters.**

## The result: 0.30 is a coin flip, 0.506 is not

| distance | heard correctly |
|---|---|
| 0.323 | **1 / 2** |
| ≥ 0.506 | **4 / 4** (incl. controls at 0.623, 0.899) |

Voices sitting just above the old floor are heard as the same person about half the time.
Everything at 0.506 and beyond was heard correctly. **The floor was admitting duplicates.**

## And raising it is nearly free

The reason this is good news rather than bad. Re-simulating acceptance over the same 80
minted vectors at higher floors:

| floor | accepted | Vendi | **effective voices** |
|---|---|---|---|
| **0.30** (old) | 50 | 0.431 | **21.5** |
| 0.35 | 46 | 0.461 | 21.2 |
| 0.40 | 41 | 0.511 | 20.9 |
| **0.45** (new) | 40 | 0.520 | **20.8** |
| 0.50 | 37 | 0.554 | 20.5 |
| 0.55 | 30 | 0.608 | 18.2 |

**0.30 → 0.45 drops 10 of 50 accepted voices and costs 3% of effective diversity.**

Those ten were duplicates. The catalog was overcounting; the metric was not
undercounting. A catalog of **40 voices where all 40 are distinct people** is strictly
better than 50 where ten are repeats — the second is a support problem, and it is the
exact failure `S7` warned about while using a floor that caused it.

## What changed in the code

`UNIQUENESS_MIN` stays **0.30** and `UNIQUENESS_MIN_MIOCODEC = 0.45` is added, with
`service.mint(uniqueness_min=...)` to select. `S7` uses the validated value.

**The 0.45 is deliberately not applied to English.** This listening test was Indic and
MioCodec only. Qwen3's 2048-d space is a different geometry and a cosine distance in it
does not mean the same thing — transferring a floor across spaces without measuring is
precisely the class of error this project keeps catching in itself. **The English path
needs its own listening test, and until it has one its floor is unvalidated in both
directions.**

## Not established — and this list is longer than the result

- **n = 8 pairs, and only 2 at the floor.** One listener. This is a pilot that detects a
  problem; it does not locate the correct floor. 0.45 is chosen because it is cheap, sits
  above the region that failed and below the region that passed — not because it was
  measured to be right.
- **The true threshold is somewhere in (0.323, 0.506) and this cannot narrow it.** A
  proper version needs ~10 pairs per distance band across several bands and more than one
  listener.
- **One listener who knows the project.** Not blind to the purpose, only to the answers.
- **Hindi only, minted voices only.** Whether the same floor holds for retrieved library
  voices, or in Bengali or Tamil, is untested.
- **The positive controls were easy.** Both scored correctly, but a control that is too
  easy inflates confidence in the instrument; a harder positive (same voice, very
  different prosody) would test it better.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S11-listening/run_listening_set.py
```

Writes 8 shuffled pairs, a scoresheet, and `out/ANSWER-KEY.json`. ~2 minutes on a 6 GB
card. Do not open the key before scoring.
