# Sanity check — where Alaap actually is, 2026-09-09

> Measured against [`RESEARCH/GRAND-PLAN.md`](../RESEARCH/GRAND-PLAN.md), not against memory.
> **Seven days. 143 commits. 35 experiments. 13 ADRs. 135 invariant tests.**

---

## The one-line answer

**The direction vector is right. The magnitude has all gone into one component, and the plan
named that failure mode in writing before it happened.**

Nothing here is wrong work. Almost none of it is *product* work.

---

## 1. Stage-by-stage, against the plan's own exit criteria

| # | Stage | Exit criterion (plan's words) | Status |
|---|---|---|---|
| **S0** | Foundations + 10 experiments | E1 returns `g2s ≈ s2s`, scorecard populated, corpus licence-cleared, every render watermarked | **7/10** — E1–E5, E9 ✅ (+ E10–E15 extra). **E6, E7, E8 never run** |
| **S1** | Voice identity persists | Same character across 20 lines + restart + backend bump | ✅ **done**, verified on hardware |
| **S2** | Retrieval mapper | Beats S0 on adherence at equal-or-better diversity | ✅ **done** |
| **S3** | Generative mapper (MDN → diffusion) | Beats S2 on adherence × diversity jointly | ❌ **not started** |
| **S4** | Dataset v2 | Attributes verifiable by re-measurement | 🟡 partial — Indic captions only |
| **S5** | **Indic gate** | Explicit go/no-go | ✅ **passed** — then eight more experiments past it |
| **S6** | Inference service | `mint` + `render` **over HTTP**, ≥2 backends, no client change | ❌ **not started** |
| **S7** | GPU hosting | p95 latency + $/min measured under load | ❌ not started |
| **S8** | Web app | A stranger mints a voice unaided | ❌ not started |
| **S9/S10** | Multi-tenant, public release | — | ❌ not started |

**`alaap/service.py` exists and works — but `VoiceService` is an in-process Python class.**
S6's criterion is *over HTTP*. That one word is the entire gap between here and a product.

### The minimum shippable thing is blocked, at its first step

The plan defines **MST = S0 + S1 + S6 + S7 + cut-down S8**, and says of it:

> *"Protect it. The most likely failure mode of this project is that S2/S3 research swallows
> the calendar and nothing ever ships."*

S0 ✅ and S1 ✅ are done. **S6, S7, S8 have zero commits.**

---

## 2. Where the effort actually went

```
  experiments/  E0 E1 E2 E3 E4 E5 E9 E10 E11 E12 E13 E14 E15
                S2 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15
                S16 S17 S18 S19 S20 S21 S22 S23 S24
                └────────────── 35 ──────────────┘
```

**A naming collision hides the gap.** `experiments/S6`…`S10` are *Indic* experiments. The
plan's **S6–S10 are the platform track**. Day to day the directories look like progress
through the stages; they are not the same S6.

### The last eight experiments are measurement-of-measurement

| | what it produced |
|---|---|
| S17 | dropped 2 dead axes, repaired `vtl_cm` |
| S18 | audited all 17 attributes — **no new axis helps** |
| S19 | naturalness deficit splits evenly, codec vs model |
| S20/S20b | built the first naturalness gate; then corrected its own layer |
| S21 | `spectral_tilt` reads pitch — **kept anyway** |
| S22 | adopted the S17/S18 axis set (the one shipped change) |
| S23 | the gate's layer doesn't transfer; **two of S20b's checks were wrong** |
| S24 | repairing tilt makes it redundant with `f0_mean` |

Six of those eight outputs are **caveats**, not capabilities. They are honest, rigorous, and
four of them found genuine bugs. They also do not move a user any closer to a voice.

---

## 3. The steelman — why this was not wasted

Stated fairly, because the case is real:

1. **Every one was justified when started.** The catalogue saturates at ~22 effective voices;
   that is the MST's core number; finding out why is legitimate product work.
2. **Four audits found real bugs.** `vtl_cm` was broken. `speaking_rate`/`f0_cv` were never
   identity axes. The gate's layer was wrong for English. S20b's pitch check was confounded.
   A pipeline shipping those would have been quietly wrong.
3. **Invariant I8 demanded it**: *"Do not start stage N+1 until stage N's exit criterion is
   measured."* The measurement discipline is the plan's, not a deviation from it.
4. **The tests and skills are durable.** 135 invariant tests and three project skills encode
   what was learned; the next person does not re-make these mistakes.

**The rebuttal is timing, not value.** The question "can description-driven minting exceed
~22 effective voices?" was answered **no** several experiments ago, and the answer has been
re-confirmed by increasingly indirect means since.

---

## 4. The finding that should change the plan

Buried in the results is a much better route to the MST than the one being pursued.

| approach | adherence | source |
|---|---|---|
| **minting** a voice from a description | 37% → 58% | S6, S17, S22 |
| **retrieval** from a curated library | **87%** | S8-library-retrieval |
| retrieval from a 37-voice commercial library | +8.5pp lift, spans 64% of the real range | S16 |

**The MST says "a browsable catalog of a few hundred curated voices." It does not say
"minted".** Retrieval over a curated library already measures at 87% — and the whole
saturation problem (~22 effective voices) is a *minting* problem.

> **The catalogue product does not need the mapper to get better. It needs a library and an
> HTTP endpoint.**

That reframing is drawn entirely from the project's own numbers, and it is the single
highest-value thing in this document.

---

## 5. Course correction

Ordered by value per unit of effort. This is a proposal; nothing here has been started.

| # | Action | Cost | Why now |
|---|---|---|---|
| **1** | **S6 — wrap `VoiceService` in an HTTP API.** FastAPI, `POST /mint`, `POST /render`, `GET /voices`. Two backends behind one interface; `public_servable` refused in a test. | **~1 day** | The single blocker to the MST. Everything else in it is built. |
| **2** | **Re-scope the catalogue to retrieval, not minting.** Curate 100–200 library voices; serve by retrieval at 87% rather than minting at 58%. | ~2 days | Turns a measured research limit into a non-issue by changing the product, not the model. |
| **3** | **E8 — real $/min on short lines.** The plan budgets **under $5** for this. | **<$5, ~2 h** | The last unrun S0 experiment that blocks a *stage* (S7 economics). |
| **4** | **Cut-down S8 — a page that lists voices and renders a line.** | ~3 days | "A stranger mints a voice unaided" is the exit criterion; a chooser + render box meets it. |
| **5** | **Freeze estimator auditing.** | — | Two consecutive audits (S21, S24) concluded "keep what ships". The lever is closed. |

### Deliberately NOT next

- **S3 / E6 — the generative mapper.** If the catalogue ships on retrieval, minting is a v2
  feature. E6 (does PromptTTS++'s MDN give diverse voices?) stops being a blocker.
- **E7 — speaker count × session diversity.** Corpus sizing only matters if the corpus is the
  constraint. S7 tested 141 → 432 speakers for **zero change** in effective voices.
- **More listening tests on rate control.** `S14d` is built and waiting; after it, stop.

### What would change this recommendation

- If **S14d** shows the rate artefact is badly asymmetric, `Direction.rate` needs rework
  before any service ships it.
- If a **library of 100–200 curated voices turns out to be hard to source under a clean
  licence**, the retrieval reframing collapses and minting is back on the critical path.
  `I4` says assume nothing here: five licence chains were mis-declared out of five checked.
- If **VPC 2026 results (2026-09-26)** show synthetic-speaker generation solved the diversity
  problem, S3 gets cheap and interesting again.

---

## 6. Honest scorecard

| | |
|---|---|
| **Research quality** | Genuinely high. Pre-committed properties, controls for controls, corrections written into the file that was wrong. |
| **Engineering quality** | 135 invariant tests, 16 modules, no orphaned code. |
| **Scientific honesty** | Unusually good — several published claims were retracted by later experiments **in the same repo**. |
| **Product progress** | **Stalled at the S1/S6 boundary for the entire project.** |
| **Plan adherence** | High on Track A, **zero on Track B**, which the plan says starts at S1. |

**Verdict: not a tangent — an over-investment.** The work is on the right axis; it has been
pushed far past the point where the next unit of measurement changes a decision. The plan's
own hedge (ship the catalogue) is untouched, and the evidence now says that hedge is easier
than it looked, because retrieval already works.
