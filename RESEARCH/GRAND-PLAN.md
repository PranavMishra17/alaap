# Alaap — The Grand Plan

> **From an empty folder to a public website, grounded in [`00-EXECUTIVE-VERDICT.md`](00-EXECUTIVE-VERDICT.md) and the 13 research files behind it.**
> **Date:** 2026-09-02 · Plan v2 (supersedes scope §11)
> **Per-stage detail:** [`phases/`](phases/)

---

## 0. The one-paragraph plan

Build the English track on **Qwen3-TTS** (Apache-2.0 end to end), because it is the only open model whose speaker vector is both externally addressable and the *sole* identity input. Copy **PromptTTS++**'s MDN mapper — the working open reference implementation of exactly the contract we need. Store **both** a vector and a seed clip for every identity, because an encoder change orphaned every stored vector once already. Train on **LibriTTS-P** and friends, not ParaSpeechCaps, which is non-commercial. Watermark with **AudioSeal** from the first render, because EU Art. 50(2) has been live since 2 August 2026. Ship a **curated catalog** first — it is a real product, and it de-risks the platform while the mapper is still research. Treat **Indic as a separate project with its own go/no-go gate**, because the only servable Indic backend has no speaker vector at all.

---

## 0.5 Locked decisions

Recorded in [`DECISIONS.md`](../DECISIONS.md). Summarised here because they shape every stage below.

| ADR | Decision |
|---|---|
| **000** | The project is **Alaap** (आलाप) — the opening improvisation where a voice explores its full range |
| **001** | Two-tower architecture kept; every component from the original scope replaced |
| **002** | **Indic ships as a curated catalog in v1** (Indic Parler-TTS, Tier-2 identity, no custom Indic voice design). The Tier-1 fine-tune is **deferred to the S5 gate, not dropped** |
| **003** | **Build the machinery on English, port to Indic early** |
| **004** | **8 languages in v1:** Hindi, Telugu, Bengali, Marathi, Kannada, Malayalam, Odia, Assamese |

### The Indic guards (ADR-003)

Building on English risks producing an English-shaped architecture that does not fit Indic — which would be fatal, because Indic is the point of the project. Three hard rules prevent it:

1. **No stage exits without its Indic column filled** in the scorecard, even if the number is bad. **A blank Indic column is a failed exit criterion.**
2. **The renderer adapter carries an Indic implementation from S6**, not later. If the interface cannot express Indic, the interface is wrong.
3. **The identity store schema is validated against a Tier-2 Indic identity at S1**, not assumed.

And one scheduling rule: **captioning IndicVoices-R starts during S2/S4**, independent of the S5 gate — otherwise "decide later" becomes "decide later, then wait three months." See [`PHASE-05`](phases/PHASE-05-indic-gate.md) §8.

---

## 1. The invariants

These do not change between stages. If a decision contradicts one of these, the decision is wrong.

| # | Invariant | Why | Source |
|---|---|---|---|
| **I1** | **Never ship a plain-MSE mapper, and never treat the embedding space as isotropic.** Per-dimension rescale every synthesized vector. | Per-dim scales span 3 orders of magnitude. Averaging/unweighted-MSE collapses every identity onto one voice (GVD −6.5 to −11.6 dB). Rescaling recovers it to −0.14. | [`12`](12-speaker-manifold-navigability.md) |
| **I2** | **Every identity record stores `backend_id` + `backend_version` + BOTH a vector and a seed clip + the full generation parameters.** | The Zonos v0.1→ZONOS2 encoder change orphaned every Tier-1 vector in 16 months. A model upgrade silently changing every shipped character's voice is unrecoverable. | [`02`](02-identity-representation.md) [`11`](11-production-api-landscape.md) |
| **I3** | **`public_servable` is enforced in code, not documentation.** The public deployment refuses to load any backend where it is false. | Five upstream licence traps found in one research pass. A rule living only in a README will be violated by a config change. | [`08`](08-licensing-propagation.md) |
| **I4** | **A licence declaration on a derived artefact is a claim, not evidence. Trace the upstream chain before adopting anything.** | VoiceSculptor→Llasa-3B, IndicF5→F5-TTS, SPRING_F5→F5-TTS, Zonos→VoxBlink2, Indic-Mio→Emilia/Expresso. Five for five. | [`08`](08-licensing-propagation.md) |
| **I5** | **Identity stores timbre only. Performance is per-line and never enters the identity record.** Mint from neutral, store neutral, add `α·τ` at render. | Emotion baked into identity makes dramatic content impossible and breaks consistency across a character's 400 lines. | [`10`](10-performance-control.md) |
| **I6** | **No user audio upload in the public product.** Ever. Enforced at the API boundary — including the distinction that Tier-2 seed clips are *system*-generated. | The only structural safety property we have. It is what keeps us outside the ELVIS Act core, NO FAKES §2(c)(2)(B), and *Arijit Singh* ¶18. | [`09`](09-safety-and-watermarking.md) [`08`](08-licensing-propagation.md) |
| **I7** | **Every rendered file is watermarked and provenance-logged, from the first render in S0.** | EU Art. 50(2) live since 2026-08-02; Art. 2(12) excludes Art. 50 from the open-source exemption. Not a launch task. | [`09`](09-safety-and-watermarking.md) |
| **I8** | **Do not start stage N+1 until stage N's exit criterion is *measured*.** Every exit criterion below is a number or a demonstrable behaviour. | This is the difference between this attempt and the last one. | scope §11 — kept |
| **I9** | **Never report an InstructTTSEval score without the human ceiling next to it** (84.3 avg / 67.2 Role-Play), and never report a MOS-predictor score as a quality gate. | Real human audio scores 84.3; >85 is judge noise. DNSMOS/UTMOS are anti-correlated with pitch (r ≈ −0.79 vs human −0.06) and would fight the diversity axis. | [`06`](06-evaluation-harness.md) [`13`](13-description-to-embedding-prior-art.md) |
| **I10** | **Report per-demographic eval slices, never a single aggregate.** | Three independent literatures disadvantage female voices; low-density regions of speaker space cost +60% relative WER — and "elderly", "raspy" are low-density regions. It is a quality instrument, not a courtesy. | [`09`](09-safety-and-watermarking.md) [`12`](12-speaker-manifold-navigability.md) |

---

## 2. What changed from scope §11

| Change | Was | Now | Why |
|---|---|---|---|
| **Dataset selection moves earlier** | S4 | **S0** | ParaSpeechCaps is CC-BY-NC-SA. The corpus is an open question and S2 cannot start until it is settled. |
| **Compliance moves earlier** | S9 | **S0** | EU Art. 50(2) has been live since 2026-08-02. |
| **New experiment block** | — | **S0-E**, ten experiments, all ≤1 day | Two of them (E0, E3) are afternoon-scale and unblock everything downstream. |
| **Indic gets a gate** | assumed parallel track | **S5 with an explicit go/no-go** | The only servable Indic backend has no speaker vector. |
| **Identity is both tiers** | Tier 1 target, Tier 2 fallback | **Both, always** | An encoder change already orphaned stored vectors once. |
| **Mapper head order** | flow matching first | **MDN first**, then two-stage diffusion | The one open working implementation uses MDN, and the flow-vs-regression trade is bidirectional. |

---

## 3. The route

```
 ── TRACK A · ML (local-first, 8–12 GB) ────────────────────────────────────────────

  S0 ──────────► S1 ──────► S2 ──────────► S3 ──────────► S4 ──────► S5
  Foundations    Identity   Retrieval      Generative     Dataset     INDIC
  + experiments  layer      mapper         mapper         v2          ⟨GATE⟩
  + eval harness
  + corpus       │          │                                         │
  + compliance   │          │                                         │
        │        │ identity │ catalog                                 │
        │        │ schema   │ exists                                  │
        ▼        ▼          ▼                                         ▼
 ── TRACK B · PLATFORM ─────────────────────────────────────────────────────────────

  S6 ──────────► S7 ──────────► S8 ──────────► S9 ──────────► S10
  Inference      GPU hosting    Web app        Multi-tenant   Public
  service                       (library)                     release

  ▲ MINIMUM SHIPPABLE THING = S0 + S1 + S6 + S7 + cut-down S8 ▲
```

**Track A leads on quality. Track B starts at S1**, because the identity-store schema is the contract between them and changing it late is the expensive mistake.

---

## 4. Stage summary

| # | Track | Goal | Exit criterion — a number or a demonstrable behaviour | GPU | Detail |
|---|---|---|---|---|---|
| **S0** | ML | Foundations: prove the diagram, build the harness, settle the corpus, run the ten experiments, watermark from render #1 | **E1 returns `g2s ≈ s2s`** (or the Tier-2 pivot is taken), baseline scorecard populated for EN, corpus licence-cleared, every render watermarked | local | [`PHASE-00`](phases/PHASE-00-foundations.md) |
| **S1** | ML | Voice identity persists | Same character across **20 lines + a process restart + a backend version bump**, both tiers, measured against a calibrated `C_same` | local | [`PHASE-01`](phases/PHASE-01-identity-layer.md) |
| **S2** | ML | First "our model" — contrastive retrieval mapper | Beats S0 on adherence at **equal or better diversity**, both measured, significance-tested at description level | local | [`PHASE-02`](phases/PHASE-02-retrieval-mapper.md) |
| **S3** | ML | Generative mapper (MDN → two-stage diffusion) | Beats S2 on **adherence × diversity jointly** — not adherence alone, because the trade is bidirectional | local + rented sweeps | [`PHASE-03`](phases/PHASE-03-generative-mapper.md) |
| **S4** | ML | VoicePersona v2, measure-first captions, corpus expansion | A generated voice's attributes are **verifiable by re-measurement**; v2 published under a licence that actually holds | local | [`PHASE-04`](phases/PHASE-04-dataset-v2.md) |
| **S5** | ML | **INDIC — GATE** | Explicit go/no-go against three options; if go, Indic identity consistency matches EN | rented | [`PHASE-05`](phases/PHASE-05-indic-gate.md) |
| **S6** | Platform | Inference service behind a stable API | `mint` + `render` over HTTP against **≥2 backends with no client change**; `public_servable` refuses a non-servable backend in a test | local | [`PHASE-06`](phases/PHASE-06-inference-service.md) |
| **S7** | Platform | Real GPU infra | **p95 render latency and $/min-of-audio measured on short lines** under realistic load | cloud | [`PHASE-07`](phases/PHASE-07-gpu-hosting.md) |
| **S8** | Platform | The website | A stranger mints a voice and downloads usable audio **unaided** | cloud | [`PHASE-08`](phases/PHASE-08-web-app.md) |
| **S9** | Platform | Multi-tenant | Accounts, quotas, moderation, provenance, abuse controls live; **compliance checklist fully green** | cloud | [`PHASE-09`](phases/PHASE-09-multitenant.md) |
| **S10** | Both | Public release | Public site + open dataset + model card with honest limitations + writeup | cloud | [`PHASE-10`](phases/PHASE-10-public-release.md) |

---

## 5. The minimum shippable thing — name it now, protect it

**S0 + S1 + S6 + S7 + a cut-down S8** is already a genuinely useful public product:

> **A browsable catalog of a few hundred curated voices, in English (and optionally 8 Indic languages), with batch dialogue rendering, per-line performance direction, and a game-engine-ready export manifest.**

No custom mapper required. It ships value on day one, de-risks the entire platform track against real users while Track A is still research, and gives the mapper an **honest baseline to beat in production** rather than in a notebook.

**Protect it.** The most likely failure mode of this project is that S2/S3 research swallows the calendar and nothing ever ships. The catalog is the hedge.

---

## 6. The ten S0 experiments

All ≤1 day. Together they de-risk the architecture. Run E3 and E0 first — they are afternoon-scale and unblock the most.

| # | Question | Cost | Blocks | Source |
|---|---|---|---|---|
| **E3** | Are the embedding dimensions heterogeneously scaled? | **~20 min CPU** | The mapper's output layer. **Do this first.** | [`12`](12-speaker-manifold-navigability.md) |
| **E0** | Do training-free emotion direction vectors work on Qwen3-TTS Base? | **1 afternoon** | The entire `Direction` channel | [`10`](10-performance-control.md) |
| **E1** | Is Qwen3-TTS's `(2048,)` a *learned* space or a *d-vector* space? | ~4 GPU-h | **Everything.** The most important experiment | [`12`](12-speaker-manifold-navigability.md) |
| **E2** | How large is vocoder drift? | ~2 GPU-h | Whether Tier 1 needs a closed loop; mint cost | [`12`](12-speaker-manifold-navigability.md) |
| **E4** | What are `C_same` / `C_diff` for the eval encoder? | ~1 GPU-h | Every identity-consistency threshold | [`06`](06-evaluation-harness.md) |
| **E5** | Does AudioSeal survive 1–3 s clips? | ~2 GPU-h | Whether watermarking works for game dialogue at all | [`09`](09-safety-and-watermarking.md) |
| **E6** | Does PromptTTS++'s MDN give diverse voices from one prompt? | ~1 GPU-h | The core A3 question, on real weights | [`13`](13-description-to-embedding-prior-art.md) |
| **E7** | Speaker count × session diversity | **<10 GPU-h** | Corpus sizing | [`05`](05-datasets-and-annotation.md) |
| **E8** | Real $/min on short lines | **<$5** | The whole cost model | [`07`](07-serving-and-cost.md) |
| **E9** | Identity drift under *heavy stylisation* (aged/raspy/whispered) | ~3 GPU-h | Whether the §5 voice range is reachable. **Nobody has measured this** | [`10`](10-performance-control.md) |

### 6.1 The pivot conditions

Write these down before running, so the result is not rationalised after the fact.

| If | Then |
|---|---|
| **E1** returns `g2s` ≫ `s2s` (say >1.5×) | The Qwen3 space resists generative priors. **Pivot to Tier 2 wholesale**: mapper outputs a *description + generation params*, identity is the seed clip. The two-tower split survives; only the representation changes. |
| **E2** shows drift > `C_same` margin | Tier 1 needs a closed loop → every mint costs a GPU render → **rework the free-mint economics** ([`07`](07-serving-and-cost.md), [`11`](11-production-api-landscape.md) §9.3) |
| **E5** shows AudioSeal fails below 3 s | Watermarking cannot be the sole compliance mechanism for game dialogue. Escalate to counsel; consider render-time metadata + a longer-clip watermark on the export bundle rather than per-line |
| **E9** shows heavy stylisation breaks identity | Scope §5's voice range is not reachable on this backend. Narrow the promise, or add a stylisation-specific workstream |
| **E0** fails to produce a usable `Direction` | Qwen3-TTS Base has no per-line control at all. Fall back to **CosyVoice2/3** (Apache-2.0, best-shaped control surface: separate `instruct_text`, `speed`, `<strong>`, `[laughter]`) as the renderer, accepting Tier 2 |

---

## 7. Budget

| Phase | Cost | Note |
|---|---|---|
| S0–S4 local work | **$0** | The whole point of caching embeddings once |
| S0 experiment block | **< $30** | All ten, at RunPod Community rates |
| S3 rented sweeps | **$400–900** | 3 runs plus failures. Anchor: Parler-TTS Mini pretrain was ~1,152 H100-hrs ≈ $2.3–4.6k — we are not doing that |
| S7 first warm GPU | **$248/mo** (RunPod Community 4090, no SLA) · **$122/mo** at 12h/day | Cheapest *with* an SLA is RunPod Secure L4 at $358/mo |
| Postgres | $0 → ~$25/mo | Supabase or Neon |
| Object storage | ~$0.015/GB-mo, **$0 egress** on R2 | Serve Opus (0.24 MB/min), not WAV (5.76 MB/min) |

**The economics fact that matters most:** marginal render cost recovers nothing. The bill is a **fixed warm-GPU floor**. Break-even for a $248/mo warm 4090 is **55 h/mo of generated audio vs ElevenLabs, 368 h/mo vs OpenAI `tts-1`**. **Below ~55 h/mo, self-hosting is more expensive per minute than just calling ElevenLabs.** Meter renders to shape demand, not to cover the GPU. ([`07`](07-serving-and-cost.md))

---

## 8. Risks — revised

| Risk | Severity | Mitigation | Changed? |
|---|---|---|---|
| **Identity collapse via isotropic treatment of the embedding space** | **High** — silent | I1: per-dim rescaling. GVD as the detector. | **New mechanism** — the brief had the wrong one |
| **Licence contamination via an upstream chain** | **High** | I3 + I4. Five instances already found. | **Escalated** from Medium-High |
| **Compliance gap already open** | **High** | I7. EU Art. 50(2) live since 2026-08-02. | **New** |
| **India's prefixed-audio-disclosure rule is product-destroying** | **High** | Counsel question #1. No technical workaround. | **New** |
| Model upgrade silently changes every existing voice | High | I2: both tiers + `backend_version` + treat upgrades as migrations with re-render and diff | unchanged |
| Description adherence unmeasurable | High | Harness before training. But **not** with MOS predictors (I9) | **Method changed** |
| **The Indic track has no viable Tier-1 path** | **High** | S5 gate, three explicit options | **New** |
| **Research swallows the calendar; nothing ships** | **High** | Protect the minimum shippable thing (§5) | **New** |
| Ungrounded captions poison the mapper | Medium-High | Measure-first pipeline — but Data-Speech computes only 9 columns, so 6 attributes must be built | **Scope larger than assumed** |
| Dataset licence propagates to the released model | Medium-High | Settled: train on the GO list only ([`08`](08-licensing-propagation.md)) | **Resolved** |
| Voice-biometric defeat / vishing | Medium-High | S9 abuse controls. We are a ready-made master-voice generator | **New** |
| 8–12 GB blocks progress | Medium | Cache embeddings once. **But ZONOS2 bf16 is 15.34 GB — does not fit.** Qwen3-1.7B does | **Partially realised** |
| GPU cost runaway | Medium | Warm-pool sizing; hard quotas at S9 | unchanged |
| Cold start ruins interactive UX | Medium | Floor is ~10–30 s and snapshots don't help. **Interactive requires a warm worker** — not optional | **Sharpened** |

---

## 9. How to use this plan

1. **Read [`00-EXECUTIVE-VERDICT.md`](00-EXECUTIVE-VERDICT.md)** if you have not.
2. **Open the phase file** for the stage you are starting. Each contains: goal, inputs, the exact build steps, the exit criterion as a number, the experiments it depends on, and its cross-references into the evidence.
3. **Do not skip the exit criterion.** I8 is the rule that separates this attempt from the last one.
4. **When something contradicts the research, update the research file and note it.** These documents are the project's memory; a stale one is worse than none.
5. **Re-check the dated items.** VPC 2026 results land **2026-09-26**. The India IT Rules question needs counsel. Licences change — [`08`](08-licensing-propagation.md) is a snapshot of 2026-09-02.

---

*Plan v2 · 2026-09-02 · grounded in 13 primary-source research files · supersedes scope §11.*
