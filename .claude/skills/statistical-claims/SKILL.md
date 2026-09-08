---
name: statistical-claims
description: Use whenever reporting, comparing, or acting on a measured number — before writing a result into a RESULTS.md, a commit message, a docstring, or a recommendation. Covers significance vs effect size, sample size, normalised metrics, and claims of absence.
---

# Statistical claims discipline

Every rule here was earned by a specific error in this project. The error is named so the
rule is checkable rather than moralistic.

## 1. A difference is not a finding until it is tested

**Compute the standard error and the test statistic before writing the number down.**

```python
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
t = (b.mean() - a.mean()) / se
```

`|t| >= 2` before the word "separates", "beats", "improves", "sees" appears anywhere.

> **S20b** declared three WavLM layers winners on a rule of `gap >= 5` — an effect-size cut
> with nothing behind it. The standard error was ~6 points. All three were `t < 1.3`. The
> script printed *"the limitation is removable: switch to layer 5"* on a difference a test
> rejects.

**Effect size and significance answer different questions.** Report both. A large
non-significant effect means "underpowered", not "real but noisy".

## 2. State n beside every mean, and be suspicious of small ones

The worst case of four samples **is one sample**.

> **E9** reported worst-case consistency as its headline finding at n=4 per group. A re-run
> flipped **both** worst-case verdicts. The document's own caveats section already said the
> worst-case gap needed more samples — and the recommendations section acted on it anyway.

> **S20** concluded a gate was blind to codec degradation from n=3. Resampling 3 of a later
> 40 reproduced that exact result in **41% of draws**.

**When a caveat says a number is under-powered, no later section may rely on that number.**
Grep your own document for the claim before shipping it.

## 3. Absence of evidence needs a power calculation

"No effect" is a claim about your instrument, not about the world, until you compute what
effect you could have detected.

```python
# n needed to reach |t| = 2 given the observed effect
n_needed = int(np.ceil(n_current * (2.0 / abs(t_observed)) ** 2))
```

Say it as *"not detected, and this design could not have resolved anything below X"* —
never as *"there is no difference"*.

## 4. A normalised metric is not a count

> **S9** had to retract a published claim because normalised Vendi 0.361 (over 95 items)
> and 0.362 (over 38) were read as equivalent. They are ~34 effective voices and ~14. The
> fraction is *of the maximum for a set of that size*.

Before comparing two normalised numbers, check they share a denominator. If they do not,
convert to absolute units first.

## 4b. A correlation pooled across groups measures the groups

If two groups differ on BOTH the thing you correlate and the thing you correlate it with,
the pooled `r` is partly the group split. Compute it **within** each group.

> **S20b** screened WavLM layers for "is this a pitch detector" using `r(F0, score)` pooled
> over real and synthetic clips. Synthetic clips score ~50 points higher AND have a
> different F0 distribution. Within real speech alone -- where naturalness is constant, so
> the correlation is unambiguous -- the chosen layer moved from -0.054 to **-0.154** on
> Hindi and to **+0.323** on English, which it fails. The flaw was invisible on one corpus.

Pick the within-group that is **confound-free**, not just any. Within *real* speech works
because quality is constant there; within *synthetic* does not, because quality genuinely
varies and may covary with pitch by construction.

**If you change which statistic decides after seeing the numbers, say so in the write-up**
and report all of them. An a priori argument made second is still worth less than one made
first, and the reader should get to weigh that.

## 4c. A control from one random split is one sample

> **S20b**'s real-vs-real control split a held-out set in half ONCE. Values ranged 0.08 to
> 0.75 across layers and six were rejected on it. Averaged over 200 random splits every
> layer landed at 0.11-0.13. The spread was entirely which half a clip fell in.

Repeat any split-based control over many splits and report the mean. It costs nothing when
the features are already computed.

## 4d. A threshold that rejects nothing has stopped checking

After fixing a statistic, re-check that its threshold still discriminates. Inheriting a bar
calibrated for the OLD statistic passed all 12 layers in one S23 pass -- technically a
"pass" for everything, actually a check that had stopped running. Prefer a bar derived from
the data (`|r|` distinguishable from zero at this n) over an inherited constant.

## 5. Compare like with like, or the comparison is of your scoring

> **S17** scored each arm on *its own* axis set, so one arm was graded on three axes and
> another on four. Adding a harder axis to one side lowers its mean whether or not anything
> got worse. A 95.8% and an 81.9% landed in the same column meaning different things.

Score every arm on the **intersection** of what all arms produce. Report anything outside
that intersection in its own column.

## 6. Name the confound before reading the result

If one arm differs from another in a way unrelated to the hypothesis, say so *before*
looking at the numbers, and say which comparisons are free of it.

> **S19** measured that the codec preserves room noise exactly (38.2 → 38.2 dB SNR) while
> the LM generates clean audio (50.5). So "which is real" could be answered by "which has
> room tone" on two of three comparisons. The third was clean, and it carried the result.

## 7. A metric is validated on a corpus, not in the abstract

Re-validate when the corpus, model, or codec changes. A threshold measured in one embedding
space may not transfer to another — `ADR-011` in this repo exists because a uniqueness floor
borrowed from a paper was wrong for one space and right for another.

## Checklist before writing a number into a document

1. Is there a test statistic, or only a difference?
2. Is n stated? Is it above ~30 per group for a mean, above ~100 for a small effect?
3. If claiming absence — what could this design have detected?
4. Are the two numbers on the same denominator?
5. Are all arms scored on the same axes?
6. Is any confound named *before* the result?
7. Does any caveat elsewhere in the document undercut this claim?
