# S13 — identity is safe from the text channel, and the text channel does nothing

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

## Result 2 — the small movement is sampling noise, not steering

*(Read this section knowing how it ends: a listener heard no emotion at all, and
the tokenizer shows why. The analysis below is kept because it is what the
measurements said before the listener was asked, and because the two statistics
disagreeing is a lesson worth keeping.)*

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

## ⚠️ A listener says the tags do nothing at all, and the tokenizer says why

The measured 1.08 noise-floor units said "weak". A fluent Hindi speaker, given the four
tagged renders of one sentence in one voice, said something stronger:

> *"NO — they are all the same speaker, all warm but no distinct feeling at all, in any
> of the 4. No emotion — just slight variations of the same voice."*

That is the drive half of `ADR-012` failing outright, and it confirms the identity half
by the same sentence: *all the same speaker* is exactly what the ECAPA numbers claimed.

### The cause: the tags are not tokens

None of the nine documented tags is a single token, and none is in the added vocabulary —
in **Indic-Mio or in its base `Aratako/MioTTS-0.6B`** (both vocab 164469):

```
<happy>      3 tokens  ['<h', 'appy', '>']
<sad>        3 tokens  ['<s', 'ad', '>']
<angry>      4 tokens  ['<', 'ang', 'ry', '>']
<whisper>    4 tokens  ['<', 'wh', 'isper', '>']
...
single-token tags: 0/9      in added_vocab: 0/9
added tokens that look like tags: </think>, <tool_call>, ... (Qwen's chat vocabulary)
```

The tags reach the model as ordinary subword text. No formatting variant changes this:
`<happy>`, ` <happy>`, `<happy>.`, `[happy]` and `(happy)` all tokenize to 3 pieces.

### `<whisper>` is the probe that settles it

Whispering is unmistakable acoustically — voicing collapses. English line, same speaker
embedding, 3 seeds averaged:

| variant | tokens | hnr_db | f0_mean | **voiced_frac** | speaking_rate |
|---|---|---|---|---|---|
| neutral | 84 | 4.82 | 215.5 | 0.583 | 24.32 |
| `<whisper>` | 119 | 3.97 | 216.7 | **0.555** | 17.97 |
| `<angry>` | 124 | 5.08 | 215.0 | 0.573 | 16.68 |
| `<enunciated>` | 158 | 5.83 | 215.3 | 0.576 | 13.08 |

**It is not whispering.** `f0_mean` is unchanged to within 1.5 Hz on every tag.

At a fixed seed, adding a tag changes the generated token stream almost completely
(prefix agreement 0–2.9%) — consistent with the tag text perturbing the sample rather
than conditioning anything.

### A hypothesis of mine, refuted by its own check

Token counts grow with tag length (84 → 119 → 124 → 158), so I proposed the model was
**speaking the tag aloud**. `whisper-small` transcribes all four tagged English renders
as exactly the reference sentence, no tag words present. **Refuted.** It would have gone
into an upstream report as a confident claim; the check cost two minutes.

This is filed upstream as Report 4 in `UPSTREAM-REPORTS.md`, with what could *not* be
ruled out stated: that the tags need a prompt format `MioTTS-Inference` uses and I did
not replicate.

## S13b — word-level emphasis fails too, and `*` is not the reason

`*` **is** a single token (id 9), unlike all nine emotion tags — so emphasis was
structurally a live candidate in a way `<happy>` never was. It is still an ordinary text
token rather than a special one, and wrapping fragments the word it marks:

```
'The mountains remember'    3 tokens  ['The', ' mountains', ' remember']
'The *mountains* remember'  6 tokens  ['The', ' *', 'mount', 'ains', '*', ' remember']
```

**The test needs no forced alignment.** A global acoustic difference cannot separate
"emphasis worked" from "the sample was re-rolled" — the trap this experiment already fell
into once. But emphasis has a property re-rolling does not: it is **local**. Emphasising an
early word should push energy earlier; emphasising a late word, later.

```
energy centroid = Σ t·rms(t) / Σ rms(t),   t normalised to [0, 1]
predicted:  centroid(*early*) < plain < centroid(*late*)
```

| condition | mean centroid | vs plain |
|---|---|---|
| plain | 0.4489 | — (seed-to-seed SD **0.0208**) |
| `The *mountains* remember…` | 0.4312 | −0.0177 = **0.85** noise units |
| `…you *regret*.` | 0.4345 | −0.0144 = **0.69** noise units |

**The ordering does not hold.** Both conditions move the centroid in the *same* direction,
downward, by less than one noise unit each. Emphasising the last word of the sentence does
not push energy later — it does the same thing as emphasising the second word.

That pattern is what an inert marker produces: the asterisks perturb sampling, and where
they sit makes no difference. A directional ordering cannot be manufactured by re-rolling
a seed, which is why the ordering rather than the magnitude was the test.

**So the text channel on this backend carries no direction at all** — not the documented
emotion tags, not the documented word stress.

### Caveats on this one specifically

- **4 seeds, one sentence, one voice, English.** Small.
- **The energy centroid is a crude proxy.** Emphasis can be realised as pitch accent with
  little energy change, and that would not show here. A pitch-contour version of the same
  directional test would be a stronger instrument and was not run.
- `*` being a plain text token means the finetune *could* still have learned it from data
  that used it — this measures the outcome, not the training.

## What this means for the direction channel

**The safe half is proven; the drive half does not exist on this backend.** Tags are not
a weak lever, they are not a lever — a listener hears nothing, and the tokenizer explains
why. Any plan that stacks something *onto* the tag is building on nothing.

What survives is the more valuable half: **identity is provably robust to whatever is
appended to the text.** That is the hard property, and it means a direction channel can be
built here as soon as there is anything that actually drives.

Three routes, now re-ordered by what the evidence supports:

1. ~~**Stack the tag with explicit prosody control.**~~ Dead as stated — there is no tag
   effect to stack onto. What survives is the second half of it: **explicit prosody
   control alone.** `speaking_rate` responds *consistently*
   (z=6.7) but *weakly*. It is also the axis a caller can set directly rather than
   requesting — resample or re-time the render to a target rate, and the tag supplies the
   rest. This is the only route the measurements actively endorse.
2. ~~**Word-level emphasis (`*word*`).**~~ **Tested — see S13b above. It does not work
   either.** `*` survives the tokenizer as a real token, and it still produces no
   positional effect.
3. **Direct signal-level control of `speaking_rate`.** Not a model capability at all —
   re-time the render. `S12` says rate is a real delivery axis and `S13` says identity
   survives text changes; nothing says the *model* has to be the one moving it.
4. ~~Larger intensity via tag stacking.~~ Dead for the same reason as route 1.

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
