# S14c — discarded, and the discard condition was the flawed part

**Run:** 2026-09-09 · 12 pairs · one listener · comparative 2AFC
**Status:** **DISCARDED** on the pre-committed condition D2. The rate table was not printed.

---

## Controls first, because they gate everything

| pair | control | expected | got | |
|---|---|---|---|---|
| 04 | **negative** (rate 0.50) | B | B | **PASS** |
| 07 | **negative** (rate 0.50) | B | B | **PASS** |
| 05 | **positive** (nothing re-timed) | tie | B | confident |
| 08 | **positive** (nothing re-timed) | tie | B | confident |

**D1 passed 2/2.** Gross re-timing *is* audible to this listener — the instrument works at the
extreme, which `S14b` never established.

**D2 fired.** Both positive controls got a confident answer on pairs where **neither clip had
been touched**. Per the condition written down before the set was sent, the set is discarded
and the rate table is not printed.

## The obvious explanation is wrong, and I checked before writing it

The natural story: I varied the **voice** between the two sides of every pair (to decorrelate
duration), so each side carries a *different* baseline, and `S7`'s voices differ in render
quality. That would break the baseline-cancellation the whole comparative design rests on.

Scoring all eight `S7` voices with the naturalness gate (`NATURALNESS_LAYER = 7`, S23):

| | A | B | listener said | gate says more synthetic |
|---|---|---|---|---|
| pair05 | ind0003 **98.2** | ind0002 **94.6** | B | **A** |
| pair08 | ind0002 **94.6** | ind0000 **98.2** | B | **B** |

**It points opposite ways on the two pairs.** The voice-quality explanation is not supported.
*(And the gate spans only 94.6–98.2 across all eight voices — saturated near its ceiling, so
it has little resolving power here either. This is "not supported", not "ruled out".)*

Nor is it duration or sentence: in pair05 the chosen clip was the **short** line, in pair08
the **long** one.

## So the flaw is D2 itself

This listener answered "can't tell" on 3 of 12 pairs — about 25% of the time. D2 fired on
*both positive controls being confident*, which for a listener with a 75% non-tie rate
happens by chance **~56% of the time**:

```
P(D2 fires | listener guesses) = 0.75² ≈ 0.56
```

**A discard condition that fires on a coin flip is not a discard condition.** With two
positive controls it cannot distinguish "answering on something other than re-timing" from
"guessed twice rather than saying can't tell".

This is the same error the project's own `listening-tests` skill warns about, applied to the
wrong half of the set:

> *"One pair cannot distinguish 'the threshold is right' from 'that pair was easy.'"*

I sized the **test** pairs with that in mind — two per rate — and did not apply it to the
**controls**. The set is still discarded, because the condition was written down first and
honouring it is the only thing that makes any of these tests worth running. But the discard
is evidence about my design, not about the listener.

## What is *not* a finding, and is recorded only as a hypothesis

The discarded test answers show a clean pattern, stated here so the redesign can target it
and **not** as a result:

| rate | heard as more manipulated |
|---|---|
| 0.67 | 2 / 2 |
| 0.80 | 2 / 2 |
| 1.25 | 0 / 2 |
| 1.43 | 1 / 2 |

**Slowing may be audible where speeding is not.** That would make the clean range asymmetric,
and it is consistent with the listener's own earlier report of a rate-0.70 render:
*"60% slowed down, 40% slow speech effect."* If true, `listener_clean_range = (0.8, 1.25)` is
wrong on the slow side — 0.80 was heard 2/2.

**Also worth recording:** on pairs 7, 9 and 12 the listener volunteered "**both**" — hearing
manipulation in the untouched clip too. That is `S14b`'s finding reappearing inside a design
built to cancel it.

## The design tension this exposes

For a rate test you cannot have both at once:

| | duration cue hidden | baseline shared |
|---|---|---|
| same sentence + same voice | ❌ length identifies the re-timed clip | ✅ |
| different sentence + voice (`S14c`) | ✅ | ❌ each side carries its own baseline |

`S14b` chose the first and its controls failed. `S14c` chose the second and its controls
failed differently. **The way out is to stop asking "which one was manipulated" at all.**

`S14d` compares **r against 1/r on the same voice and the same line** — both sides re-timed by
the same magnitude in opposite directions. Neither is "the original", so length answers
nothing, and the baseline is not merely shared but *identical*. The question becomes "which
sounds more processed", which is the asymmetry the table above hypothesises.

Positive controls become **0.98 vs 1.02** on the same voice and line: identical content,
identical voice, both through the phase vocoder, a magnitude far too small to hear. And there
are **three** of them, so D2 needs 0.75³ ≈ 0.42 — still not ideal, and stated rather than
hidden.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S14-rate-control/score_clean_range_ab.py \
  --answers "tie,tie,a,b,b,a,b,b,tie,a,a,b"
```
