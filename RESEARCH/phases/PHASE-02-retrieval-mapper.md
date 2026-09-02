# PHASE 02 — The Retrieval Mapper (first "our model")

> **Goal:** a CLIP-style contrastive joint space between description text and speaker embeddings; retrieve top-k, perturb, interpolate, mint.
> **GPU:** local — trains on cached vectors, never audio · **Depends on:** S0 (E1, E3, E6, E7), S1
> **Evidence:** [`01`](../01-ttv-landscape.md) · [`05`](../05-datasets-and-annotation.md) · [`12`](../12-speaker-manifold-navigability.md) · [`13`](../13-description-to-embedding-prior-art.md)

---

## 0. Exit criteria

| # | Criterion | Threshold |
|---|---|---|
| **X2.1** | Beats the S0 baseline on **description adherence** | per-attribute error, paired bootstrap at **description level** (N=50), p < 0.05 |
| **X2.2** | At **equal or better diversity** | nVS ≥ 0.35 **and** GVD ≥ −1 dB — not merely "not worse" |
| **X2.3** | Catalog populated and searchable | ≥ 200 minted, human-audited identities; semantic search returns sensible results |
| **X2.4** | Per-slice results reported, no slice regressed | I10 |

**X2.2 is the one that matters.** A collapsed mapper scores *well* on adherence and fails only here. This is the test that makes the silent failure visible.

---

## 1. Why retrieval first, not the generative head

Scope §4.2 ranks the generative head as "the correct answer" and retrieval as merely the "best first implementation." **The evidence says the trade is bidirectional.**

The only independent A/B on an identical two-tower substrate ([arXiv:2406.08812](https://arxiv.org/abs/2406.08812), Interspeech 2024):

| | FAD ↓ | attribute SRCC ↑ | speaker sim ↑ |
|---|---|---|---|
| Regression head | 5.244 | **0.74** | **0.41** |
| Flow matching | **3.559** | 0.60 | 0.36 |
| Hybrid | **3.126** | — | — |

**Flow matching wins fidelity but loses adherence.** So S3 is not an unambiguous upgrade over S2 — it is a trade you must measure. Retrieval-first is therefore not a compromise; it may be the better product.

**Best prior art: Unispeaker** ([arXiv:2501.06394](https://arxiv.org/abs/2501.06394)) — KV-Former + soft contrastive. Scope §17-A3 miscategorises it as a generative-head rival; it is actually the reference for *this* phase.

---

## 2. Build

### 2.1 Cache everything once

```
for each corpus in GO-list:
    for each utterance:
        z = extract_speaker_embedding(audio)      # frozen
        z = per_dimension_rescale(z)              # E3 — mandatory
        cache z to disk
    for each description:
        t = text_encoder(description)             # frozen
        cache t to disk
```

**After this, mapper training reads vectors and never audio** — a GPU-bound problem becomes laptop-friendly. Make this caching layer a first-class module, not a script (scope §9 is right about this).

### 2.2 Train the joint space

- **Corpus: LibriTTS-P** — it has native human-written intrinsic adjectives, 2,443 speakers, CC-BY-4.0. **The measure-first annotation pipeline is not a blocker for this phase.**
- Contrastive loss, in-batch negatives. HiStyle uses MSE + cosine-similarity contrastive; that combination is worth copying.
- **Budget 10–30M params**, not 10–50M. Published working mappers: ~10M (Deep Dubbing), ~30M/stage (HiStyle).

### 2.3 Mint by retrieval

```
description → t
  → top-k nearest speaker embeddings in the joint space
  → SLERP between nearest-neighbour, SAME-GENDER anchors     (never LERP; never extrapolate)
  → per-dimension rescale
  → enforce min cosine 0.3 from every existing identity
  → mint (Tier 1 + Tier 2, per S1)
```

**Three geometric constraints, all from published failures:**
- **SLERP, not LERP** — preserves unit norm on a hyperspherical space.
- **Nearest-neighbour, same-gender pairing** — random pairing gives "uneven coverage… leaving peripheral areas underrepresented."
- **Never extrapolate** — "extrapolation is not particularly meaningful" outside the real-embedding hull.

### 2.4 The catalog is the index

The retrieval index *is* the product catalog (scope §4.2 option 2, §12.3). Populating it with 200+ audited voices ships value at S8 and doubles as the mapper's substrate. Store `desc_embedding` ≤ 1024-d so pgvector HNSW indexes it.

---

## 3. The fallback if E1 failed

If S0's E1 showed the Qwen3 space resists generative priors (`g2s/s2s > 1.5`), **retrieval still works** — and this is the phase's hidden strength. Retrieval never leaves the real-embedding manifold, because every output is a weighted combination of real anchors. **The architecture-preserving fallback is simplex/SLERP weights over real anchors**, which is dry-runnable free on Kokoro.

That makes S2 the low-risk path to a working product regardless of E1's outcome.

---

## 4. Measurement traps

| Trap | Guard |
|---|---|
| Pseudo-replication | Aggregate 20 samples → 50 description-level units **before** testing |
| Underpowered claims | At N=50 you detect only **d ≈ 0.40**. Do not claim smaller wins |
| Adherence up, diversity down | X2.2 is a hard gate, not a nice-to-have |
| MOS predictors as a gate | I9 — they are anti-correlated with pitch |
| Judging with the conditioning encoder | Independent SV model only |

---

## 5. What ships

- The joint space + retrieval mapper (10–30M params)
- ≥200 audited catalog identities, both tiers
- Semantic search over the catalog ("gravelly old mentor")
- **An honest number for how much better than zero-training baseline this is**

---

*Phase spec v2 · 2026-09-02 · prev: [`PHASE-01`](PHASE-01-identity-layer.md) · next: [`PHASE-03`](PHASE-03-generative-mapper.md)*
