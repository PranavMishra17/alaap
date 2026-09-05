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

## ADR-006 — Indic has no publicly-servable path today; build on the mirrors, decide the route at S5

**Date:** 2026-09-05 · **Status:** accepted (routes deliberately left open) · **Supersedes:** the stack table in `RESEARCH/04` §9

### Context

Indic is the stated reason the project exists. Three facts, all verified in code or primary sources on 2026-09-05, jointly close every off-the-shelf route:

1. **`Qwen3-TTS` cannot speak any Indian language.** `LANG_ALIAS` in `alaap/renderer.py` is `{en, zh, fr, de, it, ja, ko, pt, ru, es}`. Every experiment E0–E10, the entire working two-tower loop, runs on a backend with no Indic phoneme inventory. No amount of speaker-vector work changes this — the TTS tower is frozen by design (ADR-001).
2. **`ai4bharat/indic-parler-tts` is CONDITIONAL** (`RESEARCH/08` §4.7): IITM IndicTTS EULA §2.2, plus unread gate terms.
3. **`SPRINGLab/Indic-Mio` is BLOCKED** (`RESEARCH/08` §4.4, HIGH confidence): Expresso (CC-BY-NC-4.0) is a declared direct training input; Emilia (NC) enters twice transitively; MioCodec — unavoidable at inference — derives from unlicensed code.

(2) and (3) are exactly the two halves of the stack `RESEARCH/04` §9 recommended. That document read declared licences; `08` traced the chains. **`08` wins.**

Separately, and independently of licence: **Indic Parler exposes no speaker vector at all** — its identity is a closed set of 69 names. So even if (2) cleared tomorrow, Parler alone gives Tier-3, which ADR-002 killed.

### Decision

**Keep building the Indic pipeline on corpora we can legally develop against, and do not pick a backend route until S5.**

Concretely, now:
- Corpus via the ungated `SPRINGLab/IndicVoices-R_*` mirrors, behind `data.DEV_ONLY` + the `dev_only=True` acknowledgement (CC-BY-4.0 upstream, undeclared on the mirror — develop yes, ship no).
- Captions via measure-first, which is backend-independent: `count_phones_indic` + `Binner.fit` on the Indic distribution + `caption_from_bins`.
- Everything above the renderer — mapper, geometry, identity store, eval — is language-agnostic already and needs no Indic decision.

### The four routes, and what each costs

| # | Route | Unblocks | Cost | Kills it |
|---|---|---|---|---|
| **A** | **Clear Parler's two conditions**, then use it as a Tier-2 *designer* only: description → seed waveform → freeze, and accept that identity lives in the waveform | Indic at Tier 2 | One click (read the gate terms) + one email to IITM | Either condition failing; also caps Indic at Tier 2 forever |
| **B** | **Lawyer-clear Indic-Mio** and get its `global_embedding` as the Indic speaker vector — the only clean route to Indic **Tier 1** | Indic at Tier 1 | Lawyer hour; `08` §4.4 says "NEEDS LAWYER before any reinstatement" | Expresso being a declared direct input is "the least deniable of the three" |
| **C** | **Train our own Indic tower** on IndicVoices-R (CC-BY-4.0, 1,704 h, 10,496 speakers — more speakers than any English corpus we have) | Indic at Tier 1, chain fully controlled | ~\$200–800 + script/phonology risk; breaks the "never train a backbone" constraint (Q1) | Nothing legal. Only budget and scope |
| **D** | **Ship Indic research-only / non-commercial**, treating the NC chains as acceptable for a non-commercial release | Indic immediately | Forecloses commercial use of that lineage | The product being commercial |

**Route C is the only one no third party can veto**, and its input corpus is already the project's strongest asset. That is worth weighing against its cost rather than treating "never train a backbone" as settled — which is precisely what Q1 exists to reopen at S5.

### Consequences

- `SERVABLE["indic-parler-tts"]` set to `False`; it had shipped `True`, contradicting the audit outright. `TestServableMatchesTheAudit` now pins code to audit.
- Indic **rendering** experiments cannot run until an HF token exists (all AI4Bharat repos are gated). Indic **measurement** experiments run now — see S4.
- `RESEARCH/04` §9 carries a SUPERSEDED banner pointing here.
- Q1 ("does the never-train-a-backbone constraint survive?") is no longer hypothetical. Route C is its concrete form.

---

## ADR-007 — catalog capacity is a DESCRIPTION problem; stop tuning the sampler

**Date:** 2026-09-05 · **Status:** accepted · **Supersedes:** the framing in E11/E12, not their measurements

### Context

E11 built the catalog and found it saturating at **40 voices** — mean uniqueness halving (0.803 → 0.457), three pairs breaching the uniqueness floor, and a Vendi score of 0.482, i.e. **20 effectively-distinct voices out of 42 minted.**

The obvious reading was that the sampler was set wrong, so `novelty` and `gmm_components` were swept. That produced two reversals and one withdrawn metric before the actual answer arrived, and the sequence is worth keeping because the failure mode recurs:

1. **A defect, not a curve.** `GaussianMixture.sample()` re-seeds from a fixed `random_state`, so the generative branch had **at most 64 possible outputs** for the life of a mapper. E1's "novelty 1.0 collapsed" was reading that bug.
2. **Spread is not a target.** After the fix, every diversity metric preferred the *crudest* prior available (a single Gaussian). "Far from the data" and "novel" produce identical spread numbers.
3. **The plausibility metric was broken too.** A GMM log-likelihood test scored **real held-out speakers the same as Gaussian noise**. Everything it justified was withdrawn, including a `typicality` mode added to `mint()`.
4. **Geometry does not predict rendering.** E13: distance-to-corpus correlates with drift at ρ = −0.35, ~12% of rank variance. E11 then rendered E12's recommendation and found the opposite.

### Decision

**The catalog's capacity is limited by the description, not by the mapper or the sampler. Work on the description.**

E14 measured it against the corpus's own real (caption, voice) pairs: five acoustic axes predict voice distance at **ρ = 0.260**, and the mapper already transports **90%** of that (`echoes source` 0.0%, so it is not retrieval in disguise). **No sampling knob can manufacture distinctions the description never made.**

### Consequences, in order of evidence

| # | Action | Evidence |
|---|---|---|
| 1 | **Re-weight the axes.** `f0` alone beats all five equally-weighted (0.386 vs 0.278 on GLOBE; 0.616 vs 0.374 on LibriTTS-R) | E14b, replicated E14c |
| 2 | **Retrieve on bins, not sentence embeddings.** The text path is barely better than chance (rank 129.7/350, chance 175); weighted bin retrieval is +66% / +55% | E15, replicated E15d |
| 3 | **Change the anchor corpus.** GLOBE_V2's speaker labels are unreliable (E4: EER 20.0% vs 2.46%), which depresses everything measured against them | E14c |
| 4 | **Add axes.** `vtl_cm` is built and validated, worth +15% relative | E14b |
| 5 | *Only then* revisit sampling | E12, E13 |

**And a standing rule that follows from (4) above:** *a geometry sweep narrows candidates; it never picks among them.* Anything that changes the sampler needs a rendering arm before its default moves.

### What was NOT decided

- `novelty` stays adaptive rather than fixed (see below), not raised.
- `retrieval="hybrid"` is implemented but **default off** — its evidence is all embedding-space, and rule (4) applies to it too.
- `ρ = 0.260` is **not** the limit of a five-axis description. It is that limit *on GLOBE_V2*; on LibriTTS-R the same axes reach 0.374.

---

## ADR-008 — novelty is adaptive, not a fixed setting

**Date:** 2026-09-05 · **Status:** accepted

**Context.** `novelty`'s cost is immediate and its benefit is deferred. Measured at matched catalog size, raising it degrades the vocoder round-trip from the very first voice (drift below floor 5% → 30% → 40% for 0.0 / 0.45 / 0.75), while its benefit — avoiding collisions — does not matter until the catalog is dense enough to collide, which for the control did not happen until voice 30. `novelty=0.45` bought **no** mean-uniqueness gain at all while costing six times the drift failures.

**Decision.** Mint at the caller's `novelty` and escalate by `NOVELTY_STEP` **only when a mint actually collides**. Drift and consistency failures never escalate — they are precisely the failures more novelty makes worse.

**Measured (E11 arm 4, 30 voices):** uniqueness floor breaches **3% → 0%**, closest pair 0.292 → 0.320, **drift-below-floor identical at 7%**. One escalation across thirty voices.

**Consequence.** `service.mint` owns this; callers keep passing `novelty=0.0`. Do not reintroduce a raised fixed default — three independent measurements (E11 drift, E14 transport 0.90×→0.35×, E1's off-manifold warning) say it costs description fidelity.

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
| 2026-09-05 | ADR-007 and ADR-008: catalog capacity is a description problem, not a sampler one — re-weight the axes, retrieve on bins, change the anchor corpus; and novelty becomes adaptive rather than fixed. |
| 2026-09-05 | ADR-006: no publicly-servable Indic path exists today. Both halves of `RESEARCH/04` §9's stack fail the licence audit; `Qwen3-TTS` has no Indic language at all. Four routes recorded, decision deferred to S5. |

---

*Decisions are cheap to record and expensive to re-derive. Add an ADR whenever a choice would otherwise get relitigated.*
