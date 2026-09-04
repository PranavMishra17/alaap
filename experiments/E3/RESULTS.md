# E3 — Speaker-embedding geometry of Qwen3-TTS

> **Status:** ✅ RUN AND COMPLETE · **Date:** 2026-09-04
> **Model:** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (bf16, CUDA)
> **Data:** 1,000 clips from **1,025 distinct speakers**, `MushanW/GLOBE_V2` (CC0), 2–15 s, resampled to 24 kHz
> **Hardware:** RTX 3060 Laptop, 6 GB — **2.02 GB VRAM used**, extraction at **48.9 clips/s**
> **Wall clock:** ~7 min end to end (61 s model load, 184 s streaming, 20 s extraction)
> **Artefacts:** `out/embeddings.npy` (1000×1024), `out/meta.json`, `out/results.json`, `out/e3_geometry.png`

---

## 0. Verdict

**Invariant I1 is CONFIRMED — but the dominant problem is not the one the research predicted.**

| | Predicted (from Meyer et al. on ECAPA+x-vector) | **Measured on Qwen3-TTS** |
|---|---|---|
| Per-dim range spread | ~3 orders of magnitude | **8.0× (0.90 orders)** — far milder |
| Failure mechanism | averaging shrinks vectors toward the origin | **not origin-shrinkage — a huge shared mean offset** |
| Fix | per-dimension rescaling | **mean-centring first, then per-dim rescaling** |

> ### The headline finding, which appears in no research file
> **79.5% of every Qwen3-TTS speaker vector is a constant offset shared by all speakers. Only 20.5% of its length carries speaker identity.**
>
> `‖mean‖ = 10.62` vs `mean ‖z − mean‖ = 2.23`
>
> Consequence: raw cosine similarity between two **different** speakers is **0.9575**. The entire 1,025-speaker population is crammed into a cone spanning cosine 0.90–0.998. Raw cosine is effectively useless for discriminating speakers in this space.

---

## 1. Measured geometry

| Property | Value |
|---|---|
| Dimension | **1024** (`enc_dim == talker hidden_size`, confirmed in `config.json`) |
| dtype | bf16 native → float32 |
| Per-dim range | min 0.239 · median 0.365 · **max 1.902** |
| **Range ratio** | **8.0×** (0.90 orders of magnitude) |
| Per-dim std | min 0.0387 · median 0.0552 · max 0.4351 |
| **Std ratio** | **11.3×** → **variance ratio 126.6×** |
| Narrowest dim | #625 `[−0.1006, +0.1387]` |
| Widest dim | #706 `[−0.9023, +1.0000]` |
| **L2-normalised?** | **NO** — `‖z‖` mean 10.859, std 0.336, **CV 0.0309** |
| Zero-centred? | **NO** |
| `‖z‖` range | 9.953 – 11.943 (p1 10.14, p99 11.72) |

**On the norm:** CV 0.031 vs **0.022** for an isotropic Gaussian of the same dimension — so the norm concentrates only slightly *more* than random high-dimensional concentration already forces. The space is shell-like at radius ≈ 10.86, but that is largely a dimensionality artefact, not a designed normalisation.

---

## 2. Effective dimensionality — the space is ~50-D, not 1024-D

| Variance captured | Dims needed |
|---|---|
| 50% | **6** / 1024 |
| 80% | 55 |
| 90% | 129 |
| 95% | 219 |
| 99% | 427 |

**Participation ratio (effective rank): 50.7 / 1024.** Top PC alone explains **27.4%**; top-2 explain 38.2%.

> **This materially de-risks E1 and the whole generative-mapper plan.** Fitting a prior over a ~50-effective-dimensional manifold is a far smaller problem than fitting one over 1024 dimensions. TacoSpawn's failure case (a GMM sampling off-manifold in a d-vector space) becomes much more tractable at this effective rank. It also means the mapper's output head can be a low-rank projection rather than a full 1024-d regressor.

---

## 3. Separability under different treatments

All pairs below are **different** speakers, so lower mean cosine = better separated.

| Treatment | mean cos | std | p1 → p99 | spread |
|---|---|---|---|---|
| **raw** | **+0.9575** | 0.0184 | +0.901 → +0.982 | 0.081 |
| mean-centred | +0.0031 | 0.2741 | −0.494 → +0.598 | **1.092** |
| mean-centred + per-dim rescaled | −0.0005 | 0.1489 | −0.292 → +0.384 | 0.676 |
| PCA-whitened (k=50) | −0.0000 | 0.1385 | −0.304 → +0.349 | 0.653 |
| PCA-whitened (k=129) | −0.0008 | 0.0826 | −0.185 → +0.202 | 0.388 |

**Mean-centring alone moves the population from a 0.081-wide cone to a 1.092-wide distribution centred on zero — a ~13× improvement in usable dynamic range.** Per-dim rescaling on top of that trades some spread for isotropy, which is what a loss function needs.

---

## 4. The catalog-uniqueness rule has to move

The plan (from VoicePrivacy baseline B3) is to enforce a **minimum cosine distance of 0.3** between a newly minted identity and every existing one.

| Space | Nearest-neighbour cosine distance |
|---|---|
| **raw** | mean **0.0158**, max **0.0293** |
| centred + per-dim scaled | mean **0.5502**, max 0.7768 |

> **A 0.3 threshold is unreachable on raw vectors — the entire 1,025-speaker population has a maximum nearest-neighbour distance of 0.029, an order of magnitude below the threshold.** It is comfortably feasible in centred+scaled space.
>
> **Action:** the uniqueness check in `PHASE-01` §2.1 must operate in centred+scaled space, and the transform must be versioned alongside the identity (it already has a home: `identity.embedding_norm`).

---

## 5. Baseline for E4

`C_diff` (different speakers) is now measured:

- **raw: 0.9575 ± 0.0184**
- **centred+scaled: −0.0005 ± 0.1489**

E4 must measure `C_same` (same speaker, different utterances) **in the same space** and report the separation. This run used 1 utterance per speaker by design, so `C_same` is not yet available — that is exactly what E4 adds.

**Prediction to test in E4:** in raw space, `C_same` will be ~0.97–0.99 against a `C_diff` of 0.958 — a separation of a couple of percent, which would make raw-space identity scoring nearly worthless. In centred space the separation should be large. If that holds, **every identity-consistency threshold in the project must be defined in centred space.**

---

## 6. Corrections to the research corpus

| # | Research claim | Measured | Action |
|---|---|---|---|
| E3-1 | Per-dim ranges likely span ~3 orders of magnitude (Meyer precedent) | **8.0×, 0.90 orders** | Soften the claim; the Qwen3 space is far better conditioned than ECAPA+x-vector concats |
| E3-2 | "Averaging shrinks vectors toward the origin" is the collapse mechanism | **Not here.** Mean vector is 97.8% of typical length — no origin shrinkage | The collapse risk in *this* space comes from the shared-mean offset, not shrinkage |
| E3-3 | Per-dim rescaling is the highest-leverage fix | **Mean-centring is higher-leverage** (13× vs the rescale's marginal gain) | **I1 should read: mean-centre, then per-dim rescale.** Update `GRAND-PLAN` §1 |
| E3-4 | Enforce min cosine distance 0.3 from existing identities | **Impossible on raw vectors** (max NN distance 0.029) | Move the check into centred+scaled space |
| E3-5 | Zonos's vector is "not L2-normalised, so rescaling is a live risk" | Qwen3's is **also not L2-normalised**, `‖z‖ ≈ 10.86 ± 0.34` | Generated vectors must be **scaled onto the ‖z‖ ≈ 10.86 shell**, not the unit sphere |
| E3-6 | (not raised anywhere) | **Effective rank 50.7 / 1024** | Model the prior in ~50–130 dims. Low-rank mapper head. Materially de-risks E1 |

---

## 7. What this changes downstream

1. **`PHASE-02` / `PHASE-03` — the mapper operates in centred+scaled space.** Predict a centred vector, rescale, add the mean back, then project onto the `‖z‖ ≈ 10.86` shell. Store the transform with the identity.
2. **`PHASE-01` — uniqueness check moves to centred space** with a re-derived threshold.
3. **`PHASE-00` E1 gets easier and should be re-scoped**: fit the GMM in the top ~50–129 PCA dims, not raw 1024-d. TacoSpawn's `s2s/g2s/g2g` comparison should be run in that space.
4. **All identity-consistency thresholds (E4, `PHASE-01` X1.1) must be defined in centred space** or they measure almost nothing.
5. **Invariant I1 is amended** to "mean-centre, then per-dimension rescale."

---

## 8. Practical notes for the next run

- **6 GB VRAM is enough.** 0.6B-Base used **2.02 GB**. The 1.7B-Base (3.59 GB weights) should also fit; worth confirming, since 1.7B is the planned Tier-1 backend and its `enc_dim` is 2048.
- **Env isolation is not optional** — the first attempt used `--system-site-packages` and broke on the NumPy 1.x/2.x ABI split leaking in from a polluted global Python (tensorflow, spacy, numpy<2 pins). A clean venv fixed it. This confirms `PHASE-00` §1.
- `extract_speaker_embedding(audio, sr)` **asserts `sr == 24000`** and takes a **mono float32 numpy array**. It internally computes a 128-mel spectrogram (n_fft 1024, hop 256, win 1024, fmin 0, fmax 12000).
- GLOBE_V2 streams well **without `torchcodec`** if you `cast_column("audio", Audio(decode=False))` and decode bytes with `soundfile`. It carries `speaker_id`, `accent`, `age`, `gender` — free per-demographic slicing for invariant I10.
- Speaker density in the stream is ~1 new speaker per 20 rows; 1,025 speakers took 20,438 rows / 184 s.

## 9. Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E3/run_e3.py --n 1000 --per-speaker 1
```
