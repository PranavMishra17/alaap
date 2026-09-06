# S15 — the English uniqueness floor survives, on one pair

**Run:** 2026-09-06 · `Qwen3-TTS-12Hz-1.7B-Base` · GLOBE_V2 · 7 blind pairs · one listener
**Question:** `S11` found the borrowed 0.30 floor was too low for Indic. Is it too low for English?

---

## Why it had to be asked separately

`S11` put the floor in front of a listener on the Indic path and it failed — voices 0.323
apart were heard as the same person half the time, which moved `UNIQUENESS_MIN_MIOCODEC` to
0.45. That fix was deliberately **not** applied to English, because `ADR-011` says a floor
is a claim about perception and may not be transferred between spaces without being
re-measured. Qwen3's space is a different geometry and a cosine distance in it does not mean
what one means in MioCodec's.

So the English floor was unvalidated in **both** directions.

## Result — 7/7, controls included

| pair | distance | truth | heard | | kind |
|---|---|---|---|---|---|
| 6 | 0.000 | same | **same** | ✓ | positive control |
| 7 | 0.000 | same | **same** | ✓ | positive control |
| 2 | 1.031 | different | **diff** | ✓ | negative control — two real speakers |
| 3 | 1.083 | different | **diff** | ✓ | negative control |
| **1** | **0.300** | different | **diff** | ✓ | **test — exactly at the floor** |
| 5 | 0.505 | different | **diff** | ✓ | test — at the Indic floor |
| 4 | 0.785 | different | **diff** | ✓ | test — far apart |

**Controls 4/4, test pairs 3/3.** The listener recognised one voice across two different
sentences, two real speakers as different people, and — the point of the exercise — **two
minted voices exactly at the 0.30 floor as different people.**

## What this does and does not establish

**Does:** the English floor is not obviously broken the way the Indic one was. `S11`'s
0.323 pair came back "same person" half the time; English's 0.300 pair did not.

**Does not:** vindicate 0.30. **There is exactly one near-floor pair here.** A single
correct answer cannot distinguish "the floor is right" from "that pair happened to be
easy", and `S11` needed two pairs at the floor to find the problem it found.

The reason there is only one is itself informative, and was noted before the listening: a
60-voice minted pool produced **one** pair in [0.30, 0.35]. The English space is far more
spread out — its negative controls sit at **1.03 and 1.08** against Indic's 0.62 and 0.90 —
so near-floor collisions are genuinely rarer there. `E11`'s catalogue never accepted
anything below 0.359.

That is consistent with the floor mattering less in English. It is not the same as having
measured it.

## Not established

- **One near-floor pair, one listener, seven pairs total.** Directional, not settled.
- **Finding more near-floor pairs needs a much larger pool** — ~200 mints rather than 60,
  about 15 minutes of compute, which is what a real answer would take.
- **The listener is not blind to the purpose**, only to the answers.
- **English, GLOBE_V2, minted voices only.** Whether the same floor holds for retrieved
  library voices is untested.
- **`UNIQUENESS_MIN` is unchanged at 0.30** and stays that way. This is weak evidence for
  leaving it alone, not evidence for endorsing it.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S15-english-listening/run_english_listening.py
```

Writes 7 shuffled pairs, a scoresheet, and `out/ANSWER-KEY.json`. Do not open the key first.
