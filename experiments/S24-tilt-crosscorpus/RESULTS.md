# S24 — the cross-corpus tilt confound is real, changes nothing, and led somewhere else

**Run:** 2026-09-08 · 37 Sarvam Bulbul voices @ 22050 Hz · 67 IndicVoices-R speakers @ 24000 Hz
**Question:** `S21`'s assertion A4 — the shipped `spectral_tilt` moves 0.70 dB/oct on a
16 kHz round-trip, so cross-corpus comparisons are partly comparing microphones. `S16` makes
one. Does its "no dark-timbred voices" survive?

---

## Verdict

| | |
|---|---|
| Is there a bandwidth difference? | **Yes.** Bulbul rolls off at 5943 Hz, the corpus at 3732 Hz |
| Does it move `S16`'s conclusion? | **No.** The bright-side skew *grows* under a common band, +0.51 → +1.23 sd |
| Did the experiment find something? | **Yes, and not what it went looking for** |

**`S16` stands as written.** The finding is elsewhere: a gap in `S21`, and a redundancy trap
that would have looked like a large win.

---

## P2 failed, and the diagnosis is the result

Three properties were asserted before measuring. **P2 failed**: Bulbul's within/between ratio
moved 1.09 → 0.23 under the band restriction, when a *constant* channel offset must cancel in
a within/between ratio. By the pre-commitment that meant the model of the confound was wrong,
so nothing was written up until it was diagnosed.

Decomposing the variants one at a time shows the band edges are not responsible at all:

| variant | bulbul within | bulbul between | ratio |
|---|---|---|---|
| shipped | 0.731 | 0.669 | 1.09 |
| **band only** (300–6000) | 1.785 | 0.852 | **2.09 — worse** |
| voiced only | 1.055 | 0.798 | 1.32 |
| **⅓-octave only** | 0.351 | **3.182** | **0.11** |
| all four (`S21` repaired) | 0.235 | 1.031 | 0.23 |

**⅓-octave banding raises between-voice variance 4.8×.** It is not removing noise from the
band edges; it is measuring something else.

## What it is measuring

On 67 real speakers with two takes each:

| variant | within/between | gender d | **\|r\| with `f0_mean`** | |
|---|---|---|---|---|
| **shipped** | 0.15 | −0.80 | **0.361** | independent of f0 |
| band only | 0.14 | −0.58 | 0.285 | independent of f0 |
| voiced only | 0.13 | −0.69 | 0.315 | independent of f0 |
| **+⅓-octave** | **0.06** | **−2.04** | **0.791** | **mostly `f0_mean` again** |
| `S21` repaired | 0.13 | −0.95 | 0.418 | independent of f0 |

Paired bootstrap over speakers, 2000 resamples:

| | Δ (+⅓oct − shipped) | 95% CI | |
|---|---|---|---|
| within/between | −0.098 | [−0.181, −0.050] | **excludes 0** |
| \|gender d\| | +1.287 | [+0.727, +1.929] | **excludes 0** |

**Both improvements are large and significant, and both are a trap.** ⅓-octave banding takes
`spectral_tilt`'s correlation with `f0_mean` from 0.36 to **0.791** — 63% shared variance.
`S18`'s redundancy screen rejects a candidate at r ≥ 0.80. This lands at 0.791, just inside,
while being the second-strongest gender separator in the project.

A gender d of −2.04 on an axis that is 63% pitch is `f0_mean`'s −3.08 leaking through. Adding
it to a set that already contains `f0_mean` would be adding a near-duplicate that *scores
beautifully on every screen the project runs*.

## This corrects the reason `S21` gave, not its decision

`S21` concluded "keep the shipped estimator — a repaired version changes nothing measurable"
on within/between −0.037, CI [−0.130, +0.046].

**That was measured on the bundled `repaired` variant only**, where the 300–6000 Hz band
restriction cancels the banding's effect (0.13, back near shipped's 0.15). `S21` never tested
⅓-octave banding alone on real speakers, so it could not see that repairing the leverage
changes a great deal.

The decision is unchanged and now better supported:

> Repairing `spectral_tilt`'s leverage does not fail to help. It helps enormously, **by
> turning the axis into `f0_mean`.** The shipped estimator is kept because its 0.36
> correlation is the most independent reading available, not because the alternatives are
> equivalent.

## And it sharpens `S16` rather than refuting it

`S16` reported Bulbul at 1.09 against the corpus's 0.14 and read it as *"Bulbul's voices
share a timbre."* Under the pitch-loaded ⅓-octave estimator Bulbul separates at 0.11.

Both are true and together they say more than either:

> **Bulbul's 37 voices differ in pitch but share a timbre.** They separate under an estimator
> that is 63% F0 and not under the one that is not.

That is exactly `S16`'s claim, measured twice with instruments of known and different
contamination. The coverage finding holds too — the bright-side skew *grows* under a common
band (+0.51 → +1.23 corpus sd), so "no dark-timbred voices" is a fact about the library.

## Not established

- **`S16`'s 3/5 was not exactly reproduced.** This run reads 4/5 with the shipped estimator,
  using quintile edges from a 93-speaker sample rather than `S16`'s fitted `Binner` with its
  HNR decorrelation. The *direction* is reproduced, not the bin count.
- **A4 itself is untouched.** The 0.70 dB/oct round-trip sensitivity `S21` measured is still
  real; this shows only that it does not bind on *this* pair of corpora, whose rolloffs
  (5943 / 3732 Hz) both sit below the 8 kHz band edge.
- **One corpus pair, Hindi only.** A comparison against a genuinely wideband corpus could
  behave differently, and nothing here tests that.
- **63% is not 100%.** The ⅓-octave variant is mostly but not entirely `f0_mean`; whether the
  remaining 37% carries anything useful is untested.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S24-tilt-crosscorpus/run_channel_confound.py
```

Needs `S21`'s cached corpus audio and `S16`'s Bulbul renders. No models, ~4 min.
