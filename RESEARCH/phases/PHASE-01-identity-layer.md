# PHASE 01 — The Identity Layer

> **Goal:** a voice identity that survives 20 lines, a process restart, and a backend version bump — in both tiers.
> **GPU:** local · **Depends on:** S0 experiments E1, E2, E4 · **Track:** ML, and the contract Track B builds against
> **Evidence:** [`02`](../02-identity-representation.md) · [`11`](../11-production-api-landscape.md) · [`12`](../12-speaker-manifold-navigability.md) · [`06`](../06-evaluation-harness.md)

---

## 0. Exit criteria

| # | Criterion | Threshold |
|---|---|---|
| **X1.1** | A character sounds the same across **20 different sentences** | mean pairwise cosine ≥ `C_same` (from E4), no outlier below `C_same − 2σ` |
| **X1.2** | Identity survives a **process restart** | re-load from DB, render, cosine vs pre-restart ≥ `C_same` |
| **X1.3** | Identity survives a **backend version bump** — or fails loudly | migration produces a diff report; no silent change |
| **X1.4** | **Both tiers** demonstrated on the same character | Tier-1 vector and Tier-2 seed clip both reproduce, and are compared |
| **X1.5** | Vocoder drift is characterised and handled | E2's number is recorded; if drift > margin, the closed loop is implemented |
| **X1.6** | Per-slice consistency reported | male / female / elderly / raspy / whispered — no slice below `C_same` |

---

## 1. The core insight this phase encodes

Scope §4.3 says a Tier-1 vector reproduces the voice **"exactly, by construction."** That is **true of the conditioning input and false of the rendered output** — vocoder drift means the x-vector extracted from rendered audio "often differs substantially from the x-vector at the vocoder input." Drift is **systematic and correctable**, not noise.

And the historical argument decides the design: **the Zonos v0.1 → ZONOS2 encoder change (256→2048, LDA 128→1024) orphaned every stored Tier-1 vector in 16 months.** The waveform is what survives a version bump.

> **Therefore: store both, always. Not "Tier 1 with Tier 2 as fallback" — both, on every identity, from the first mint.** (Invariant I2)

Every commercial product agrees: 100% of shipping prompt-to-voice APIs are Tier 2. Cartesia shipped Tier 1 with 192-d vectors and weighted mixing, then **withdrew it entirely on 2026-06-01 with no replacement for mixing.**

---

## 2. Build

### 2.1 Mint (Tier 1 + Tier 2 together)

```
description
  → designer (Qwen3-VoiceDesign or MOSS-VoiceGenerator)
  → seed waveform  ──────────────────────────► store as Tier 2  (seed_audio_ref)
  → extract_speaker_embedding(seed)
  → per-dimension rescale (E3)  ─────────────► store as Tier 1  (embedding)
  → store generation_params + backend_version
  → render a verification line
  → re-extract, compare (E2 closed loop)
  → accept or re-mint
```

**Three rules:**
1. **Mint from neutral.** No emotion in the description used for minting. Performance is added at render (I5).
2. **Per-dimension rescale before storing** (I1). Store the scale stats used in `embedding_norm` so a later encoder change is diagnosable.
3. **Enforce a minimum cosine distance from every existing identity** — VoicePrivacy B3 uses **0.3**. This is the catalog-uniqueness guarantee, and it is free.

### 2.2 The closed loop (only if E2 says you need it)

If drift exceeds the `C_same` margin: render → re-extract → compare to intended → correct or reject. **This costs one GPU render per mint**, which erases Tier 1's free-CPU-mint advantage from scope §12.2. Record that in the cost model.

### 2.3 Render

```
render(identity, text, direction | None):
    if identity.tier == 1:  condition on identity.embedding
    else:                   condition on identity.seed_audio
    apply direction (E0's α·τ, added at render — never stored on the identity)
    watermark (AudioSeal, presence bit only)
    provenance-log
```

---

## 3. The version-bump migration — build it now, not after it hurts

This is the failure scope §12.1 correctly calls "catastrophic and unrecoverable for anyone who already shipped a game."

**A backend upgrade is a migration, never an in-place swap.**

1. Pin `backend_version` on every identity (already in the S0 schema).
2. On upgrade, re-render a fixed probe script for every affected identity under both old and new versions.
3. Produce a **diff report**: cosine(old, new) per identity, sorted worst-first.
4. Identities below `C_same` are **quarantined**, not silently updated. The owner chooses: keep pinned to the old version, or accept the new voice.
5. Tier-2 identities re-clone from the stored seed clip — usually more stable across versions than a vector in a changed space.

**Test this at S1 with a synthetic version bump.** If you cannot demonstrate X1.3 now, you will discover you cannot do it when it matters.

---

## 4. Measurement

Use the harness from S0. Identity consistency is axis 2.

- **Score with an encoder independent of the conditioning encoder.** `pyannote/wespeaker-voxceleb-resnet34-LM` is fbank→ResNet, so it has zero lineage overlap with a WavLM- or ECAPA-conditioned space. Otherwise the system marks its own homework.
- **Use the calibrated `C_same` from E4.** No published universal threshold exists; values are encoder-specific and non-comparable.
- **Report per-slice** (I10). Low-density regions of speaker space cost **+60% relative WER** — and "elderly", "raspy" are low-density regions in every corpus we have.

---

## 5. Traps

| Trap | Why it bites | Guard |
|---|---|---|
| Storing only the vector | Encoder change orphans everything (16-month precedent) | I2 — always both |
| Trusting `seed` for reproduction | VoxCPM2's `retry_badcase` (default `True`) silently does `current_seed += 1` | Tier 3 is dead; never rely on it |
| Isotropic vector ops | Averaging/LERP collapses all identities onto one voice (GVD −6.5 to −11.6 dB) | Per-dim rescale; SLERP not LERP; never extrapolate |
| Interpolating across gender or language | Documented destabiliser; opposite-gender averaging costs WER 6.83→7.19% | Nearest-neighbour, same-gender pairing only |
| Emotion in the identity record | Breaks per-line direction and cross-line consistency | I5 — mint neutral, store neutral |
| Marking own homework | Same encoder for conditioning and scoring | Independent SV model |

---

## 6. Track B unblocks here

The identity-store schema is the contract between tracks. Once X1.5 passes, **freeze it and treat it as an interface.** Adapters absorb backend churn; the schema does not change. S6 can begin in parallel.

---

*Phase spec v2 · 2026-09-02 · prev: [`PHASE-00`](PHASE-00-foundations.md) · next: [`PHASE-02`](PHASE-02-retrieval-mapper.md)*
