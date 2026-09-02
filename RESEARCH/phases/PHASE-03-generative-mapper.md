# PHASE 03 — The Generative Mapper

> **Goal:** model `p(speaker_vector | description)` rather than `E[·]`, and expose the adherence↔diversity trade as a user-facing dial.
> **GPU:** local for training, rented for sweeps ($400–900 for 3 runs + failures) · **Depends on:** S2
> **Evidence:** [`01`](../01-ttv-landscape.md) · [`12`](../12-speaker-manifold-navigability.md) · [`13`](../13-description-to-embedding-prior-art.md)

---

## 0. Exit criteria

| # | Criterion | Threshold |
|---|---|---|
| **X3.1** | Beats S2 on **adherence × diversity jointly** | Both measured; a Pareto improvement, or an explicit documented trade |
| **X3.2** | The CFG dial works and is characterised | Sweep α; produce the adherence-vs-diversity curve |
| **X3.3** | No mode collapse at any dial setting | nVS ≥ 0.35 and GVD ≥ −1 dB across the whole α range |

> **Note the exit criterion changed from the scope document.** Scope §11 says S3 must beat S2 "on adherence *without losing diversity*." The published evidence says that may not be achievable as literally written — flow matching wins fidelity but **loses** adherence. **Judge S3 on the joint frontier, not on adherence alone.**

---

## 1. Head selection — the order is now evidence-based

Scope §17-A3 lists flow matching / diffusion / normalizing flow. **It omits the family the one open working system actually uses.**

| Order | Head | Why | Cost |
|---|---|---|---|
| **1** | **MDN** (per-dim k-component GMM) | PromptTTS++ ships working Apache-2.0 code and a 1.32 GB checkpoint doing exactly this: BERT `[CLS]` → MLP 768→512→512→256 → `MDNLayer(256, 256, num_gaussians=10, dim_wise=True)`, `norm_style_emb: true` | Lowest |
| **2** | **Two-stage conditional diffusion** (HiStyle) | Stage 1 speaker embedding, stage 2 style conditioned on stage 1 **plus a residual from it**. 12 layers, hidden 512, ~30M/stage. Best in HiStyle's own comparison | Medium |
| **3** | Flow matching / OT-CFM | Deep Dubbing publishes a ~10M text→speaker-embedding module targeting Cam++ 192-d | Medium |
| **4** | Normalizing flow (Glow) | PromptSpeaker's approach — **no released weights**, reimplementation required | Highest |

**Start at 1.** It is cheapest, it has a reference implementation, and E6 will already have told you whether its samples are diverse.

**Steal HiStyle's residual connection regardless of which head wins.** It is the concrete mechanism for scope §4.4's timbre/performance separation.

---

## 2. The non-negotiable constraints

All three from [`12`](../12-speaker-manifold-navigability.md), all from published failures.

1. **Fit the prior to the real embedding distribution.** TacoSpawn explicitly reports standard-normal priors "performed worse." A free vector regressor will fail. Never sample raw Gaussian noise in an unnormalised space.
2. **Per-dimension rescale every output.** Non-negotiable (I1). This is what keeps GVD near 0 instead of −6.5.
3. **Generator architecture matters.** Meyer et al.: "the use of MLP … prevents the model to converge, leading to generated embeddings that can be **easily distinguished from original ones**." A **ResNet generator was required.**

---

## 3. The CFG dial — a genuinely novel product surface

**Zero of eight surveyed papers expose classifier-free guidance as an adherence↔diversity dial.** Neither does any commercial API beyond a bare `guidance_scale`. Given that the trade is real and bidirectional, shipping it as a user-facing control is both novel and correct.

**Published precedent for the mechanism:** VoicePrivacy 2024 entry T10 uses `s_anon = α·s̄ + (1−α)·ŝ` (pool average vs random identity). The organizers: *"A higher α puts more weight on the averaged speaker identity, typically resulting in **less anonymity but better utility** preservation, while a lower α increases randomness."*

**Translated to our product:** the mean-voice direction is the **high-adherence, low-diversity** end. Label the dial honestly — *"how literally should I take your description"* — and default it mid-range.

---

## 4. Diversity is where the contribution is

**HiStyle, VoiceDesigner, VoiceSculptor, MOSS-VoiceGenerator, Qwen3-TTS and Parler-TTS report no diversity metric at all.** Scope §10 axis 3 is not hygiene — it is a defensible research contribution, and the reason to publish at S10.

Measure with **normalized Vendi (target ≥0.35)** *and* **GVD (target ≥ −1 dB)**. GVD has a published scale with baselines; Vendi is the finer instrument. Use both.

---

## 5. Budget

| Item | Cost |
|---|---|
| Local training on cached vectors | $0 |
| Rented sweeps (A100-class, RunPod Community) | **$400–900** for 3 runs + failures |
| Anchor for scale | Parler-TTS Mini *pretrain* was ~1,152 H100-hrs ≈ $2.3–4.6k. **We are not doing that** — we train a 10–30M mapper on frozen embeddings |

---

## 6. If S3 does not beat S2

**That is a legitimate result, and the scope document does not allow for it.** Record it, ship S2, and say so in the model card. A retrieval mapper that never leaves the real-embedding manifold is a defensible product; a generative head that trades adherence for fidelity may not be an improvement for character work, where matching the description *is* the product.

---

*Phase spec v2 · 2026-09-02 · prev: [`PHASE-02`](PHASE-02-retrieval-mapper.md) · next: [`PHASE-04`](PHASE-04-dataset-v2.md)*
