# Alaap — Decision Log

> Locked decisions, with the reasoning and the evidence. **Do not relitigate these** without new evidence — the point of this file is that months of building can proceed without re-deriving them.
>
> Format: one ADR per decision. Status is `LOCKED`, `PROVISIONAL`, or `SUPERSEDED`.
> Evidence lives in [`RESEARCH/`](RESEARCH/README.md).

---

## ADR-000 — The name

**Date:** 2026-09-02 · **Status:** LOCKED

**Decision:** the project is **Alaap** (आलाप).

**Why:** in Hindustani classical music, the *alaap* is the opening improvisation in which a voice explores its full range before the composition begins — a voice mapping out its own possibility space. That is exactly what description-to-voice design does. It is pronounceable and spellable internationally, and it signals the Indic-first commitment in the name itself.

**Follow-up:** trademark and domain check not yet done.

---

## ADR-001 — The architecture survives; the parts list does not

**Date:** 2026-09-02 · **Status:** LOCKED · **Evidence:** [`00-EXECUTIVE-VERDICT.md`](RESEARCH/00-EXECUTIVE-VERDICT.md)

**Decision:** keep the two-tower design. Replace every component the original scope named.

```
description ──[frozen text encoder + TRAINED mapper]──► voice identity ──[FROZEN TTS]──► audio
```

**Why:** question A2 — "is the speaker-embedding manifold navigable by synthesis?" — is answered **yes**, and has been since 2018. Random unit-hypersphere vectors give naturalness MOS 3.65; WGAN-sampled embeddings are an official VoicePrivacy 2026 baseline; Zyphra ship SLERP speaker blending as a product feature. **PromptTTS++ already implements the exact contract, in Apache-2.0, with downloadable weights.**

| Role | Was | Now |
|---|---|---|
| Tier-1 renderer | Zonos-v0.1 | **Qwen3-TTS-12Hz-1.7B-Base** |
| Voice designer | VoxCPM2 | **Qwen3-VoiceDesign** / **MOSS-VoiceGenerator** |
| Mapper head | flow matching | **MDN first**, then two-stage diffusion |
| Identity | Tier 1, Tier 2 fallback | **Both, always.** Tier 3 dead |
| English corpus | ParaSpeechCaps | **LibriTTS-P + GLOBE** |
| Watermark | undecided | **AudioSeal** (MIT, code + weights) |

---

## ADR-002 — Indic ships as a curated catalog first; the Tier-1 decision is deferred, not dropped

**Date:** 2026-09-02 · **Status:** LOCKED · **Evidence:** [`04`](RESEARCH/04-indic-track.md), [`08`](RESEARCH/08-licensing-propagation.md), [`10`](RESEARCH/10-performance-control.md)

**Decision:** v1 ships **Indic Parler-TTS as a curated catalog** — its named voices across the 8 research-confirmed languages, Tier-2 seed-clip identity, **no custom Indic voice design**. The question of fine-tuning a vector-accepting backbone is **deferred to the S5 gate**, not answered now.

### Why this is not a retreat

The Indic problem is **three independent problems**, and only one is licensing:

| # | Problem | Status |
|---|---|---|
| 1 | **Licensing** — Indic-Mio, DhVaani, IndicF5, SPRING_F5 all carry upstream CC-BY-NC chains (Emilia, Expresso, F5-TTS) | Only bites for public/commercial serving. A *choice*, not a wall |
| 2 | **Architecture** — Indic Parler-TTS has **zero speaker-embedding code**; identity is a name token inside the description string | **The real blocker.** No licence fixes it. This is the CosyVoice failure (scope §2) repeating |
| 3 | **Data** | **Not a problem. It is the strongest asset in the project.** |

**On (3):** IndicVoices-R is **1,704 hours · 10,496 speakers · 22 languages · CC-BY-4.0**, with the parent paper explicitly stating the licence was chosen "allowing commercial usage," and 93.25% extempore speech. By Alaap's own criterion — speaker count over raw hours — **that is a better corpus than anything available in English** (10,496 speakers vs LibriTTS-P's 2,443). It simply has no captions, and RASMALAI's captioning recipe is published with every input CC-BY-4.0/MIT.

> **The real situation: excellent, commercially-clean Indic data, and no Indic renderer that accepts a speaker vector.**

### What the deferral costs, and the mitigation

Deferring means v1 has no description→novel-Indic-voice. Accepted, because it ships a real product now and keeps the harder question honest.

**But "decide later" must not become "decide later, then wait three months."** The long-lead item is **captioning IndicVoices-R** — so that work starts during S2/S4 as a background task, independent of the gate. When S5 arrives, the fine-tune must be a *choice*, not the start of a data project.

**See [`PHASE-05`](RESEARCH/phases/PHASE-05-indic-gate.md) for the three options the gate chooses between.**

---

## ADR-003 — Build the machinery on English, port to Indic early

**Date:** 2026-09-02 · **Status:** LOCKED

**Decision:** S0–S3 build the pipeline on English. Indic is ported in as early as each stage allows, and **every design decision is validated against Indic constraints as it is made**.

**Why:** English has LibriTTS-P (native human-written captions), mature eval tooling, and a working open reference implementation. Indic has none of those yet — the annotation pipeline would have to be built before anything could be trained. Building on English gets a working system in weeks rather than months.

**The risk this creates, and the guard:** an English-shaped architecture that does not fit Indic. Guard — three hard rules:

1. **No stage exits without its Indic column filled** in the scorecard, even if the number is bad. A blank Indic column is a failed exit criterion.
2. **The renderer adapter must carry an Indic implementation from S6**, not later. If the interface cannot express Indic, the interface is wrong.
3. **The identity store schema is validated against a Tier-2 Indic identity at S1**, not assumed.

**Explicitly not chosen:** "Indic leads" (slower to a working system), "strict parallel" (roughly double the work, high risk of neither finishing).

---

## ADR-004 — Eight languages in v1

**Date:** 2026-09-02 · **Status:** LOCKED · **Evidence:** [`04`](RESEARCH/04-indic-track.md) §7

**Decision:** **Hindi, Telugu, Bengali, Marathi, Kannada, Malayalam, Odia, Assamese.**

**Why:** these are the eight with confirmed quality on Indic Parler-TTS. Broad enough to prove the recipe generalises across script families; every one is backed by a number rather than a hope.

**Deliberately excluded from v1:**

| Language | Status | Reason |
|---|---|---|
| Tamil | conditional | NSS 75.48, only one recommended voice — verify against the S0 scorecard before promising it |
| Gujarati | conditional | 75.36, only 21 h of training data |
| **Punjabi** | **hold** | Officially "unofficial" in the model card, **no published quality number at all**, and the least training data of any language (11 h) |

**Text handling is not optional:** mixed-*script* code-switching works, but **Romanised Hindi degrades output** — Sarvam's own docs say so. An **IndicXlit transliteration pre-pass is required** for Hinglish input.

---

## ADR-005 — The ten invariants

**Date:** 2026-09-02 · **Status:** LOCKED · **Full text:** [`GRAND-PLAN.md`](RESEARCH/GRAND-PLAN.md) §1

1. Never treat the embedding space as isotropic. Per-dimension rescale everything.
2. Every identity stores both a vector and a seed clip, plus `backend_version` and generation params.
3. `public_servable` is enforced in code, not documentation.
4. **A licence declaration on a derived artefact is a claim, not evidence.** Trace upstream. *(Five traps found in one research pass.)*
5. Identity stores timbre only. Performance is per-line.
6. No user audio upload in the public product. Ever.
7. Watermark and provenance-log from the first render.
8. Do not start stage N+1 until stage N's exit criterion is *measured*.
9. Never report an InstructTTSEval score without the human ceiling; never use a MOS predictor as a quality gate.
10. Report per-demographic slices, never a single aggregate.

---

## Open questions — deliberately not decided yet

| # | Question | Decided at | Blocked on |
|---|---|---|---|
| Q1 | Does the "never train a TTS backbone" constraint survive? | **S5 gate** | Whether Indic Tier-1 is wanted enough to spend ~$200–800 and accept script/phonology extension risk |
| Q2 | Does India's "prominently prefixed audio disclosure" apply to a 1.5 s API-delivered game line? | **counsel** | Gates all public Indic serving. No technical workaround |
| Q3 | MDN vs two-stage diffusion for the mapper head | **S3** | E6, and S2's measured baseline |
| Q4 | Is Tier 1 viable at all on Qwen3-TTS? | **S0** | **E1** — `g2s/s2s > 1.5` triggers the Tier-2 pivot |
| Q5 | Open-source the mapper at S10, knowing it voids the watermark downstream? | **S10** | — |

---

## Change log

| Date | Change |
|---|---|
| 2026-09-02 | Research pass 1 complete. ADR-000 through ADR-005 locked. Project renamed VoiceForge → Alaap. |

---

*Decisions are cheap to record and expensive to re-derive. Add an ADR whenever a choice would otherwise get relitigated.*
