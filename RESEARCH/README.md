# Alaap — Research Corpus

> **Purpose:** ground every decision from Stage 0 through public deployment in verified, primary-source evidence — so the build can proceed for months without re-deriving anything and without a major architectural deviation.
>
> **Input:** [`_source/VOICEFORGE-SCOPE-v1.md`](_source/VOICEFORGE-SCOPE-v1.md) — the scoping document this pass verifies.
> **Research date:** 2026-09-02 · **Pass 1** · 12 parallel domain researchers · ~9,500 lines of findings

---

## Start here

| Order | File | Read it when |
|-------|------|--------------|
| **1** | **[`00-EXECUTIVE-VERDICT.md`](00-EXECUTIVE-VERDICT.md)** | **Always first.** Does the architecture survive? What changed? Full corrections table. |
| 2 | [`GRAND-PLAN.md`](GRAND-PLAN.md) | The whole route from empty folder to public site, plus the ten invariants. |
| 3 | [`phases/`](phases/) | You are about to start a stage and need its exact spec. |
| 4 | `01`–`13` | You need the evidence behind a decision, or you are revisiting one. |
| 5 | [`SOURCES.md`](SOURCES.md) | You need the citation for a specific claim. |

**The one-line answer:** the two-tower design survives — but **every component named in the scope document has changed.**

---

## The research files

| # | File | Domain | Answers |
|---|------|--------|---------|
| 00 | [`00-EXECUTIVE-VERDICT.md`](00-EXECUTIVE-VERDICT.md) | Verdict · master corrections table · recommended stack | — |
| 01 | [`01-ttv-landscape.md`](01-ttv-landscape.md) | The TTV field; the one-to-many problem; generative heads | **A3**, B3 |
| 02 | [`02-identity-representation.md`](02-identity-representation.md) | Tier 1 vs Tier 2; the Zonos contract; ZONOS2 | **A1**, **A2**, C2 |
| 03 | [`03-tts-backends-english.md`](03-tts-backends-english.md) | Component C: conditioning mechanisms and licence ground truth | B1, B2, B6 |
| 04 | [`04-indic-track.md`](04-indic-track.md) | Indic backends, quality ceiling, per-language shippability | B4 |
| 05 | [`05-datasets-and-annotation.md`](05-datasets-and-annotation.md) | Corpora; the measure-first recipe; VoicePersona v2 | C1, C3, C4 |
| 06 | [`06-evaluation-harness.md`](06-evaluation-harness.md) | Adherence · identity · diversity · intelligibility | §10 |
| 07 | [`07-serving-and-cost.md`](07-serving-and-cost.md) | GPU hosting, cold starts, $/min, platform stack | D1, D2 |
| 08 | [`08-licensing-propagation.md`](08-licensing-propagation.md) | The three licence questions; upstream chains; the servability gate | E1, §15.3 |
| 09 | [`09-safety-and-watermarking.md`](09-safety-and-watermarking.md) | Watermarking, provenance, regulation, gating, bias | E2, §15 |
| 10 | [`10-performance-control.md`](10-performance-control.md) | The `Direction` channel; emotion without identity drift | B5, §4.4 |
| 11 | [`11-production-api-landscape.md`](11-production-api-landscape.md) | What ElevenLabs, Cartesia, Resemble, Hume actually do | **A1** (empirical) |
| 12 | [`12-speaker-manifold-navigability.md`](12-speaker-manifold-navigability.md) | **Is the manifold navigable by synthesis?** | **A2** (decisive) |
| 13 | [`13-description-to-embedding-prior-art.md`](13-description-to-embedding-prior-art.md) | Every published description→embedding system; the reference impl | **A3** |

## The plan

| File | What it fixes |
|------|---------------|
| [`GRAND-PLAN.md`](GRAND-PLAN.md) | The route, the ten invariants, the sequencing rule, the minimum shippable thing |
| [`phases/PHASE-00-foundations.md`](phases/PHASE-00-foundations.md) | Prove the diagram · build the harness · settle the corpus · **ten de-risking experiments** · compliant from render #1 |
| [`phases/PHASE-01-identity-layer.md`](phases/PHASE-01-identity-layer.md) | Identity survives 20 lines, a restart, and a version bump — in both tiers |
| [`phases/PHASE-02-retrieval-mapper.md`](phases/PHASE-02-retrieval-mapper.md) | First "our model" — contrastive retrieval; the catalog *is* the index |
| [`phases/PHASE-03-generative-mapper.md`](phases/PHASE-03-generative-mapper.md) | MDN → two-stage diffusion; the adherence↔diversity dial |
| [`phases/PHASE-04-dataset-v2.md`](phases/PHASE-04-dataset-v2.md) | Measure-first captions; VoicePersona v2 under a licence that holds |
| [`phases/PHASE-05-indic-gate.md`](phases/PHASE-05-indic-gate.md) | **A decision, not a build** — three options, explicit go/no-go |
| [`phases/PHASE-06-inference-service.md`](phases/PHASE-06-inference-service.md) | `mint` + `render` over HTTP; the licence gate in code |
| [`phases/PHASE-07-gpu-hosting.md`](phases/PHASE-07-gpu-hosting.md) | Real GPU infra; measured latency and unit cost |
| [`phases/PHASE-08-web-app.md`](phases/PHASE-08-web-app.md) | The voice library — where the minimum shippable thing lands |
| [`phases/PHASE-09-multitenant.md`](phases/PHASE-09-multitenant.md) | Accounts, quotas, abuse controls, compliance green |
| [`phases/PHASE-10-public-release.md`](phases/PHASE-10-public-release.md) | Public site, open dataset, honest model card, writeup |

---

## The ten invariants

Restated from [`GRAND-PLAN.md`](GRAND-PLAN.md) §1. If a decision contradicts one of these, the decision is wrong.

1. **Never treat the embedding space as isotropic.** Per-dimension rescale everything.
2. **Every identity stores both a vector and a seed clip**, plus `backend_version` and the full generation params.
3. **`public_servable` is enforced in code**, not documentation.
4. **A licence declaration on a derived artefact is a claim, not evidence.** Trace upstream. *(Five traps found in one pass.)*
5. **Identity stores timbre only.** Performance is per-line and never enters the identity record.
6. **No user audio upload in the public product.** Ever.
7. **Watermark and provenance-log from the first render**, not before launch.
8. **Do not start stage N+1 until stage N's exit criterion is *measured*.**
9. **Never report an InstructTTSEval score without the human ceiling**, and never use a MOS predictor as a quality gate.
10. **Report per-demographic slices**, never a single aggregate.

---

## Evidence standards this corpus enforces

- **Primary sources only.** Every factual claim cites a LICENSE file, model card, repo, official doc, spec, or paper — and names the source type. Listicles are not evidence.
- **Confidence is explicit.** HIGH / MEDIUM / LOW / UNVERIFIED on every claim. `UNVERIFIED` is a valid and valuable answer; invented plausibility is not.
- **Corrections beat confirmations.** Every file carries a *Corrections to VOICEFORGE-SCOPE.md* table.
- **Every open question ships with its cheapest experiment.** Nothing is left as "we should look into that."
- **Code licence ≠ weights licence ≠ data licence.** Tracked separately, upstream traced, everywhere.

---

## Dated items — re-check these

| When | What |
|---|---|
| **2026-09-16** | VoiceMOS 2026 results (released to participants 2026-08-31) |
| **2026-09-26** | **VPC 2026 results** (Sydney, SPSC/Interspeech) — likely supersedes several speaker-generation choices |
| ASAP | **Counsel question #1:** does India's "prominently prefixed audio disclosure" apply to a 1.5 s API-delivered game line? Gates all public Indic serving |
| ASAP | **Archive the IITM IndicTTS V2 licence PDF** — the URL every SPRINGLab README points at went 404 in Feb 2026 |
| Ongoing | All licence readings are a **2026-09-02 snapshot**. IndicF5's tag changed *24 seconds* after a maintainer's comment |

---

*Pass 1 complete · 2026-09-02 · 12 parallel domain researchers · primary sources only · corrections scored above confirmations.*
