# E13 — geometry does not predict drift well enough to replace rendering

**Run:** 2026-09-05 · 50 rendered voices from E11's two arms · **no GPU, reads finished artefacts only**
**Questions:** (1) can drift be estimated from where a vector sits, so tuning stops costing 2.5 minutes per voice? (2) do extreme descriptions drift more?

---

## Why it was worth asking

E12 swept mapper settings on geometry alone because rendering is expensive, concluded that raising `novelty` helps, and **E11 then rendered it and found the opposite** — novelty buys uniqueness and pays for it in drift. Geometry-only tuning had produced a confidently wrong answer.

If drift were predictable from geometry, that failure mode would close: sweeps would carry a drift estimate and stay cheap. The hypothesis is straightforward — a vector far from the speakers the backend was trained on is one it has less reason to render faithfully.

## The confound, and why the pooled number is not the answer

E11's two arms differ in `novelty` **and** in drift. So *any* quantity that tracks novelty correlates with drift across the pooled data while predicting nothing. A proxy is only useful if it predicts drift **within** an arm, where novelty is held constant.

| feature | pooled ρ vs drift | **within `novelty=0.0` (n=40)** | within `novelty=0.75` (n=10) |
|---|---|---|---|
| distance to nearest corpus speaker | −0.398 | **−0.348** | −0.139 |
| 5-NN radius to corpus | −0.316 | **−0.332** | −0.176 |
| vector norm (corpus σ) | +0.393 | +0.135 | +0.261 |
| novelty setting | −0.344 | *(constant)* | *(constant)* |

## Result — the sign is right, the strength is not

**The hypothesis survives directionally.** Within the 40-voice arm, distance to the nearest corpus speaker correlates with drift at **ρ = −0.348** — further from the corpus, worse round-trip. That is real (p ≈ 0.03) and it is the predicted sign.

**It is nowhere near strong enough to be a proxy.** ρ = −0.35 accounts for roughly 12% of the rank variance in drift. Nearly nine tenths of what determines whether a vector renders faithfully is *not* captured by where it sits relative to the corpus. A tuning decision made on this would be a coin flip with a slight lean.

The `novelty=0.75` arm is too small (n=10) to add anything; its weaker coefficients should not be read as a trend.

The arm means show the same story at the group level and are worth recording:

| novelty | n | drift | distance to corpus | 5-NN radius |
|---|---|---|---|---|
| 0.00 | 40 | 0.457 | 0.486 | 0.568 |
| 0.75 | 10 | 0.390 | 0.540 | 0.594 |

Moving 0.054 further from the corpus cost 0.067 of drift — consistent, but a two-point line is not a model.

## Second question — do EXTREME descriptions drift more?

`RESEARCH/12` warned that averaging in sparse regions of speaker space costs **+60% relative WER**, and named "elderly", "raspy" and "very low-pitched" as exactly the low-density places where quality cliffs live. E9 asked the same question on 0.6B with entangled axes. E11's data answers it on 1.7B with the decorrelated binner and real rendered drift.

Extremity = how many of a description's five axes sit in an **outermost** bin. The sampler produces three natural groups (a centre cell, one-axis-off cells, and far corners):

| axes at an extreme | n | mean drift | below the 0.40 floor |
|---|---|---|---|
| 0 | 16 | 0.453 | 12% |
| 1 | 15 | 0.423 | 27% |
| **5** | **19** | **0.451** | **16%** |

| | ρ vs drift |
|---|---|
| pooled | −0.073 |
| within `novelty=0.0` (n=40) | −0.181 |

**No relationship.** A description with *all five* axes pushed to an extreme — "an extremely low voice, very rough and gravelly, very dark, monotone, very slow" — renders with the same fidelity as a fully central one (0.451 vs 0.453). The mid group's slightly worse showing is not monotonic and is noise at these sample sizes.

**This is good news for the product**, and it is the first direct evidence on it: the scope's promised range of "human + heavy stylization — aged, raspy, whispered, breathy, menacing, theatrical" does **not** carry an inherent identity cost on this backend. E9's concern, and `RESEARCH/12`'s extrapolation from WER to identity, do not reproduce here.

Two caveats that keep this from being settled. `RESEARCH/12`'s finding was about **WER** — intelligibility — and this measures **drift** — identity round-trip. They are different failure modes and this experiment cannot speak to the first. And extremity here is measured in *bin* space over five axes; a description can be extreme in ways the binner does not represent.

## What this means for how the project tunes

**Geometry sweeps must not be trusted without a rendering arm.** E12's reversal was not bad luck; there is no cheap geometric quantity here that stands in for the decoder's behaviour. Distance from the training distribution explains a minority of drift, so the remaining majority lives somewhere this experiment cannot see — plausibly in the decoder's own conditioning rather than in the vector's position at all.

Practical consequence: **budget the GPU hours.** A geometry sweep is a way to *narrow* the candidates, never to pick among them.

## Not established

- 50 voices, one backend, one corpus, English, two novelty settings.
- Only three geometric features tested. A learned predictor over more features, or over the working-space coordinates directly, was not tried and could do better — this rules out the *simple* proxies, not the idea.
- The `novelty=0.75` arm is n=10 and its coefficients are not meaningful on their own.
- Nothing here explains *what* does drive drift. Neither position relative to the corpus nor description extremity accounts for much of it. That question is open.
- The extremity groups are 0, 1 and 5 extreme axes — the sampler produces no 2–4 cells, so the middle of that range is untested.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E13-drift-proxy/run_drift_proxy.py
```

Seconds, on CPU. Reads `experiments/E11-catalog/*/rows.json` and `catalog.db` (read-only); pass `--arms <dir>:<novelty> ...` to include more arms as they finish.
