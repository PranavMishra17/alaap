# S21 — `spectral_tilt` reads pitch, and it does not matter

**Run:** 2026-09-08 · synthetic source-filter stimuli · 300 held-out Hindi clips · 67 Hindi
speakers with two takes each
**Question:** `S18` audited all 17 attributes for *whether* they separate speakers. It never
asked, of the ones that do, whether they separate them **for the reason the name claims**.
`spectral_tilt` was the outstanding case — ratio 0.19, redundancy 0.54, gender d −0.65, and
never scrutinised.

---

## Verdict

**The shipped estimator fails 3 of 4 pre-committed assertions on synthetics — and it is kept
anyway.** *(Headline revised after `S24`; the original read "repairing it changes nothing
measurable", which was true of the variant tested here and false in general. See below.)*

| | |
|---|---|
| Is the estimator defective? | **Yes.** r(F0) = +0.918 at brightness held constant by construction; +0.70 dB/oct shift on a 16 kHz round-trip |
| Does repairing it help? | **It helps enormously — by turning the axis into `f0_mean`.** ⅓-octave banding alone takes within/between 0.15 → 0.06 and gender d −0.80 → −2.04, both CIs excluding zero, while taking \|r\| with `f0_mean` from 0.36 to **0.791** (`S24`) |
| Action | **Keep the shipped estimator** — its 0.36 correlation is the most independent reading available |

**The streak of estimator audits that mattered breaks at 3** — `speaking_rate`, `f0_cv` and
`vtl_cm` were all defects worth fixing. Here the defect is real and the fix is worse than the
defect, which is a different and more useful kind of answer than "no effect".

---

## The suspicion, stated before any measurement

`spectral_tilt` fits an unweighted line to (log₂ f, dB) over **linear** FFT bins. OLS weights
a point by its leverage, `(x − x̄)²`, so in log-frequency both **ends** of the band dominate.
At this project's settings — sr 24000, n_fft 2048, band 80–8000 Hz:

| region | bins | share of leverage |
|---|---|---|
| below 500 Hz | **27** | **43.2%** |
| 500 Hz – 4 kHz | 308 | 27.4% |
| above 4 kHz | 341 | 29.4% |

**Twenty-seven bins carry 43% of the fit,** and they are not a smooth spectrum — they are the
first few F0 harmonics. A 100 Hz voice puts four under 500 Hz; a 220 Hz voice puts two and
leaves 80–220 Hz empty.

## A — the four assertions

Each is a property the axis must have **if its name is true**.

| | assertion | tolerance | shipped | |
|---|---|---|---|---|
| **A1** | recovers a known slope from shaped noise | ±0.5 dB/oct | 0.50 | **FAIL** (marginal) |
| **A2** | invariant to F0 at fixed brightness | — | **r = +0.918** | **FAIL** |
| **A3** | unmoved by 1 s of appended silence | <0.3 dB/oct | 0.15 | PASS |
| **A4** | unmoved by a 16 kHz resample round-trip | <0.5 dB/oct | 0.70 | **FAIL** |

**A3 was my hypothesis and it was wrong.** `spectral_tilt` is the only axis here that does not
use `voiced_mask`, so silence contamination looked like the obvious defect. It is not one.
Recorded because a rejected hypothesis is the cheap half of a result.

### A2 took two attempts

The first stimulus added **flat** noise to a harmonic series. Every variant then failed at
r(f0) ≈ **+0.97** — a suspiciously uniform number, which in this project has always meant the
setup rather than the world.

It did. A 250 Hz voice has 2.8× fewer harmonics across the band than a 90 Hz one, so 2.8×
more bins land in the flat inter-harmonic floor and drag the fit toward flat. **Real
aspiration is filtered by the same vocal tract**, so the valleys between harmonics carry the
envelope slope too. With a proper source-filter stimulus the variants separate:

| variant | spread | noise sd | **r(f0)** | |
|---|---|---|---|---|
| shipped | 0.92 | 0.39 | **+0.918** | reads F0 |
| +voiced frames | 0.91 | 0.37 | +0.922 | reads F0 |
| +power average | 0.76 | 0.67 | +0.812 | reads F0 |
| +⅓-octave bands | 3.48 | 0.32 | +0.922 | reads F0 |
| **+band floor 300 Hz** | 0.45 | 0.42 | +0.693 | unresolved |
| **repaired** (all four) | 1.32 | 0.62 | +0.600 | unresolved |

With n = 6 points, `|r| ≥ 0.811` is p < 0.05. The one change that matters is **raising the
band floor above F0**. Voiced-frame masking and power averaging do nothing for it.

> **The 0.5 dB/oct spread threshold in A2 was mis-specified.** It was pre-committed without
> knowing the estimator's repeatability, and the noise column shows ~0.4 dB/oct of it — the
> target was never resolvable. Said plainly rather than quietly relaxed. What survives is the
> **correlation**, which noise cannot manufacture.

## B — real speech, after a sample-rate bug of my own

300 held-out IndicVoices-R Hindi clips cached by `S20b`, F0 93–315 Hz.

> **Corrected.** The first pass measured these at 24 kHz. They are at **16 kHz** — `S20b`
> resampled them for WavLM (`run_layer_sweep.py:77`) while computing F0 on the 24 kHz
> original. Every frequency landed 1.5× wrong, putting the "band floor above F0" at 200 Hz
> instead of 300. It produced a clean-looking result (shipped +0.531, repaired +0.208) that
> **reversed** once fixed. This is assertion A4 — bandwidth sensitivity — biting the audit
> that was testing for it.

| variant | r(f0) | mean | sd |
|---|---|---|---|
| **shipped** | **+0.297** | −8.37 | 2.22 |
| +voiced | +0.308 | −8.03 | 2.06 |
| +power | +0.294 | −6.77 | 2.23 |
| +⅓-oct | +0.775 | −3.84 | 1.97 |
| +f>300 | **+0.149** | −8.09 | 2.76 |
| repaired | +0.423 | −8.87 | 2.04 |

**r(shipped) − r(repaired) = −0.126**, bootstrap 95% CI **[−0.214, −0.036]**, excludes zero.
The repaired estimator is *more* correlated with F0 on real speech, not less.

## C — 67 speakers with two takes each

The deciding phase. Pre-committed: C1 both estimators finite on ≥95% of clips; C2 the sample
must reproduce `S18`'s shipped ratio to ±0.10.

| | shipped | repaired |
|---|---|---|
| finite (C1) | 100% | 100% |
| within/between | **0.15** | 0.13 |
| \|r\| with `f0_mean` | 0.361 | 0.418 |
| gender d | −0.66 | −0.79 |

C2 passed: shipped 0.15 against `S18`'s 0.19.

**Paired bootstrap over speakers, 2000 resamples** — resampling speakers rather than clips,
so both takes stay with their owner:

| | Δ (repaired − shipped) | 95% CI | |
|---|---|---|---|
| within/between | −0.037 | [−0.130, +0.046] | **includes 0** |
| \|gender d\| | +0.173 | [−0.277, +0.669] | **includes 0** |

**Neither is a finding.** And note what this design could *not* have resolved: the ratio CI
is ±0.088 wide, so a real improvement smaller than ~0.09 would have been invisible here. This
is "not detected at n = 67 speakers", not "there is no difference".

## ⚠️ S24 SHARPENED THIS — the reason above is incomplete

Phase C tested only the bundled `repaired` variant (voiced + power + ⅓-octave + 300–6000 Hz),
where the **band restriction cancels the banding's effect**. `S24` tested ⅓-octave banding
ALONE on the same 67 speakers:

| | within/between | gender d | \|r\| with `f0_mean` |
|---|---|---|---|
| shipped | 0.15 | −0.80 | 0.361 |
| **+⅓-octave alone** | **0.06** | **−2.04** | **0.791** |
| `repaired` (tested here) | 0.13 | −0.95 | 0.418 |

Bootstrap: within/between −0.098 [−0.181, −0.050] and \|gender d\| +1.287 [+0.727, +1.929],
**both excluding zero**.

So "repairing it changes nothing measurable" is wrong as stated. **Repairing the leverage
changes a great deal — it turns the axis into `f0_mean`**, taking the correlation from 0.36
to 0.791 (63% shared variance, against `S18`'s 0.80 rejection line).

**The decision to keep the shipped estimator is unchanged and better supported**: its 0.36
correlation is the most independent reading available, not one of several equivalent ones.

## Why the synthetic defect does not show up on real speakers

On real speech pitch and brightness are **genuinely** correlated — the same larynx that
lowers F0 also changes the source spectrum. The shipped estimator's artefact happens to run
against that real correlation, so removing the artefact moves `spectral_tilt` *closer* to
`f0_mean`, not further away.

That is a story, not a measurement, and it is offered as one. What is measured is the two
bracketed rows above.

## What this means for the axis set

**`HANDOFF.md` §6 item 1 is unblocked, with a caveat to carry.** The `S18` set — `f0_mean`,
`spectral_tilt`, `vtl_cm` — can be adopted using the estimator as it ships. But
`spectral_tilt` is **not a clean brightness axis**: it carries a measured F0 artefact, and its
0.36 correlation with `f0_mean` on this corpus is part real and part estimator. Treat the set
as **~2.6 independent axes, not 3**, and do not read `spectral_tilt` as anatomy the way
`vtl_cm` must not be read as anatomy.

**One consequence is actionable now.** A4 says a 16 kHz round-trip moves the axis 0.70 dB/oct
against a between-speaker sd of ~2.0. **Cross-corpus `spectral_tilt` comparisons are partly
comparing microphones** — which matters directly for `S16` (Sarvam voices vs corpus binner)
and for anything scoring English against Hindi.

## Not established

- **Only Hindi.** One corpus, 67 speakers with two takes. `S17` checked `vtl_cm` on four.
- **No capacity measurement.** Whether either estimator changes effective-voice count is
  untested; `S18` says describable-space size does not bind, so the expected effect is small,
  but that is an inference.
- **The repaired candidate is not tuned.** `flo=300`/`fhi=6000` were chosen on stated grounds
  (above F0, below any anti-alias rolloff), not fitted. A tuned version might do better and
  nothing here rules that out.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S21-spectral-tilt/run_tilt_audit.py   # A + B, ~60 s
envs/qwen3/Scripts/python.exe experiments/S21-spectral-tilt/run_real_axis.py    # C, streams once then cached
```
