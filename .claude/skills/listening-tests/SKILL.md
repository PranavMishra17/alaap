---
name: listening-tests
description: Use when designing any test where a person judges audio — naturalness, "is this the same speaker", "does this sound processed", emotion recognition, or A/B quality. Covers design choice, mandatory controls, blinding, and what to do when a control fails.
---

# Listening test design

Every listening test in this project changed something the metrics had got wrong. They are
the highest-value measurements available and the easiest to design badly.

## 1. Pick the design from what the listener can actually discriminate

| design | question | use when | fails when |
|---|---|---|---|
| **2AFC / paired** | "which of these two is X?" | almost always | the two differ in something irrelevant (length, loudness) |
| **same/different** | "one person or two?" | identity questions | — |
| **single-stimulus** | "is this natural?" | rarely | the baseline is itself degraded |

> **S14b** used single-stimulus naturalness to find where a rate knob starts sounding
> processed. The listener called an **untouched** control "processed", because the synthesis
> is *already* audibly synthetic. The design could not separate the effect from the
> baseline. A comparative pair — "which sounds MORE processed?" — cancels a shared baseline
> because both sides carry it.

**If the baseline is imperfect, comparative designs are the only ones that work.**

## 2. Controls are not optional and their failure is not a nuisance

Every set needs both:

- **Positive control** — two stimuli that ARE the same thing (one voice, two sentences).
  Expect "same".
- **Negative control** — two that are definitely different (two real speakers). Expect
  "different".

For naturalness, the control is **real-vs-real**: a listener who can reliably pick a "more
real" one out of two real recordings is answering on room tone or accent, not naturalness.

**State the discard condition in writing before sending the set.** Then honour it.

> **S14b**'s controls failed — an untouched clip was called processed — and the set was
> discarded as pre-committed. It still produced the session's most consequential finding:
> the baseline sounds synthetic, which nothing had measured.

**A failed control is a result about your instrument, and often more valuable than the
result you wanted.**

## 3. Blind properly

- Shuffle pair order **and** A/B order within each pair.
- File names carry nothing: `pair03_A.wav`, never `minted_guru_line0.wav`.
- Write the answer key to a **separate file** and say "do not open it first".
- Level-match every stimulus. Loudness is the easiest accidental cue.

```python
def level(w, target_dbfs=-23.0):
    r = float(np.sqrt(np.mean(np.asarray(w, np.float32) ** 2)))
    return np.clip(w * (10 ** (target_dbfs / 20.0) / max(r, 1e-8)), -1, 1)
```

## 4. Ask the question you mean

> A fluent Hindi speaker confirmed renders "sounded good and were not gibberish". They were
> saying **entirely different words** than the prompt — the experiment had used the wrong
> codec. The listener was never told what the sentence was supposed to say.

**Ask "does this say X?", never "does this sound right?"** Supply X.

Same principle for naturalness: *"judge the voice, not the recording"* when one arm has room
tone and the other does not, and say why so the listener can deliberately ignore it.

## 5. Hold content and identity fixed unless they are the variable

If A and B differ in text *and* in what you are testing, the listener can answer on text.
Use the same sentence, the same speaker, and vary one thing.

Exception: a positive control must use **two different sentences**, or it is the same
waveform twice and identifiable by structure rather than by voice.

## 6. Size the set to the question

- Detecting a **large** effect (real vs synthetic): 6–10 pairs is enough.
- Locating a **threshold** (where does it start to fail): you need several pairs *per
  distance band*, not one.

> **S11** found a uniqueness floor was too low using **two** pairs at the floor — one came
> back wrong, which was enough to act on. **S15** had only **one** near-floor pair and could
> therefore only say "not obviously broken", not "correct".

**One pair cannot distinguish "the threshold is right" from "that pair was easy."**

## 7. Score against the key mechanically, and report controls first

Report `controls N/N` before any test-pair number. If controls failed, report that and stop
— do not present the rest as findings.

## Checklist

1. Comparative, or is the baseline clean enough for single-stimulus?
2. Positive and negative controls present?
3. Discard condition written down before sending?
4. Pair order, A/B order, filenames, levels — all neutral?
5. Is the question the one you mean, with X supplied?
6. Same text and speaker except for the variable under test?
7. Enough pairs *per band* if locating a threshold?
