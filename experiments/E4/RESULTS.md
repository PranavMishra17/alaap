# E4 — Calibrating C_same / C_diff, and benchmarking the conditioning encoder

> **Status:** ✅ RUN AND COMPLETE (after two discarded runs — see §5) · **Date:** 2026-09-05
> **Conditioning encoder:** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (1024-d)
> **Independent ASV:** `speechbrain/spkrec-ecapa-voxceleb` (192-d) — eval-axis-2 scorer
> **Data:** 1,600 clips / **85 speakers** from LibriTTS-R `train.clean.100` (CC-BY-4.0). Zero speaker leakage.
> **Artefacts:** `out/results.json`, `out/embeddings.npz`, `out/speaker_space.npz`, `out/e4_calibration.png`

---

## 0. Verdict

**Three results, in descending order of importance.**

### ⭐ 1. The SpeakerSpace transform must be fit IN-DOMAIN. It halves the EER.

| Transform fit on | C_same | C_diff | d′ | **EER** |
|---|---|---|---|---|
| *(none — raw vectors)* | +0.9903 | +0.9679 | 2.10 | 9.57% |
| GLOBE-V2, 1,000 speakers (**cross-domain**) | +0.8133 | +0.5545 | 2.83 | 8.64% |
| **LibriTTS-R, disjoint speaker half (in-domain)** | **+0.5766** | **−0.0017** | **3.60** | **4.35%** |

A cross-domain transform buys almost nothing (9.57 → 8.64%). An in-domain one **more than halves the error** (9.57 → 4.35%) and is the difference between a barely-usable identity signal and a good one.

The tell is `C_diff`: cross-domain it sits at **+0.5545** — a large residual similarity floor, because the GLOBE mean does not cancel the LibriTTS mean. In-domain it lands at **−0.0017**, i.e. properly centred.

> **Invariant I1 needs a third clause:** *mean-centre, then per-dimension rescale — **and fit the transform on the domain you will actually operate in.***

### 2. Qwen3's conditioning encoder is a mediocre speaker discriminator — but far better than it first looks

| Encoder | Best space | **EER** |
|---|---|---|
| Qwen3-TTS conditioning encoder | pca50 (cross-domain fit) | 7.54% |
| Qwen3-TTS, **in-domain fit** | centred+scaled | **4.35%** |
| **ECAPA-TDNN** (independent ASV) | centred+scaled | **2.84%** |

At first read the gap was 4.70 pp and looked damning. With an in-domain transform it closes to **~1.5 pp**. Qwen3's encoder carries substantially more speaker information than raw cosine suggests — it is just packaged badly.

This is RESEARCH/12's *"discriminative ≠ generative"* finding observed from the other side: an encoder trained for **generative conditioning** is a weaker **discriminator**, but not by nearly as much as raw cosine implies.

### 3. Dimensionality reduction helps the conditioning encoder

`pca50` beat both raw and centred+scaled under the cross-domain fit (7.54% vs 9.57% / 8.64%) — consistent with E3's effective rank of ~50. The tail dimensions carry noise, not identity.

---

## 1. Full results (LibriTTS-R, in-domain-clean, zero leakage)

| variant | C_same | C_diff | sep | d′ | EER |
|---|---|---|---|---|---|
| qwen3 raw | +0.9903 | +0.9679 | +0.0224 | 2.10 | 9.57% |
| qwen3 centred | +0.8546 | +0.5263 | +0.3283 | 2.23 | 10.74% |
| qwen3 centred+scaled | +0.8133 | +0.5545 | +0.2588 | 2.83 | 8.64% |
| **qwen3 pca50** | +0.9011 | +0.6404 | +0.2608 | 2.76 | **7.54%** |
| ECAPA raw | +0.6766 | +0.2092 | +0.4674 | 3.97 | 4.17% |
| **ECAPA centred+scaled** | +0.5817 | −0.0074 | +0.5891 | **4.36** | **2.84%** |

*(All qwen3 rows above use the cross-domain GLOBE fit. The in-domain figure is in §0.1.)*

---

## 2. OPERATING POINTS — use these

**Eval axis 2 (scoring identity consistency of generated voices):**

```
encoder    speechbrain/spkrec-ecapa-voxceleb   (192-d, INDEPENDENT of conditioning)
space      centred + per-dim scaled, transform fit in-domain
threshold  cosine >= 0.2685
expected   C_same +0.5817 / C_diff -0.0074 / EER 2.84%
```

**Conditioning encoder (uniqueness checks, mapper target space):**

```
encoder    Qwen3-TTS speaker encoder (1024-d)
space      centred + per-dim scaled, transform fit IN-DOMAIN
expected   C_same +0.5766 / C_diff -0.0017 / EER 4.35%
```

> **Never score identity with the Qwen3 encoder.** It is the conditioning encoder; using it to grade its own output is marking its own homework (RESEARCH/06). ECAPA is the scorer.

---

## 3. GLOBE_V2 is unsuitable for speaker-verification work

Discovered by accident, worth recording loudly.

| Corpus | ECAPA EER | d′ |
|---|---|---|
| **GLOBE_V2** | **20.02%** | 2.04 |
| **LibriTTS-R** | **2.46%** | 4.70 |

**Identical code path, identical encoder, 8× difference.** That isolates the corpus, not the pipeline. GLOBE_V2's speaker labels and/or its enhancement processing do not preserve speaker identity reliably enough to calibrate C_same.

**What this does and does not invalidate:**

- ❌ **Invalidated:** any C_same / verification result measured on GLOBE_V2. The first two E4 runs were discarded for this.
- ✅ **Still valid:** all of E3. E3 measured *population geometry* — per-dimension scale, norms, effective rank, shared-mean fraction — using **one utterance per speaker**. None of those depend on speaker labels being correct. E3's `C_diff = 0.9575` also survives, because with one utterance each, different rows are different speakers regardless of label quality.

`alaap/data.py` now carries this warning at module level, and `libritts_r_train` is the default for anything label-dependent.

---

## 4. Corrections to E3's write-up

| # | E3 claimed | E4 measures | Correction |
|---|---|---|---|
| E4-1 | "Mean-centring gives ~13× usable dynamic range" | True for the *range*, but only **+0.9 pp EER** cross-domain | Centring's value depends almost entirely on being fit in-domain. Stated as a bare 13× it oversells |
| E4-2 | Effective rank 50.7 / 1024 | `SpeakerSpace.fit` reports **124.7** | Not a contradiction: E3 computed it on the **centred** spectrum; `SpeakerSpace` computes it after **per-dim scaling**, which flattens the spectrum and raises effective rank. Both are correct; report which |
| E4-3 | (implicit) one global transform suffices | **In-domain fit halves EER** | The transform is a per-domain artefact, not a per-encoder constant |

---

## 5. Two runs were discarded before this one

Recorded so the mistakes are not repeated.

**Run 1 — speaker leakage.** Fit the transform on E3's embeddings and tested on a fresh stream, both starting at row 0 of GLOBE_V2. GLOBE_V2 streams in a stable order, so **169/169 test speakers were also in the fit set**. Fixed by adding `skip=` to `stream_clips`, which is now threaded through the cache key.

**Run 2 — untrustworthy corpus.** With leakage fixed, ECAPA scored **EER 20%**, which is ~8× worse than ECAPA's published VoxCeleb performance. Rather than accept it, I validated the pipeline against LibriTTS-R and got **2.46%** — proving the pipeline correct and the corpus wrong. See §3.

> The general lesson: **a suspiciously bad number from a well-established baseline is a bug report about your setup, not a finding.** ECAPA's published EER is ~1%; measuring 20% should never have been written up as a result.

---

## 6. Open / next

| # | Question | Where |
|---|---|---|
| E4-O1 | Does the in-domain advantage survive when "domain" is *generated* speech rather than a different corpus? Generated audio is its own domain | E2 (vocoder drift) |
| E4-O2 | Does 1.7B-Base (2048-d) discriminate better than 0.6B (1024-d)? | re-run this with `--model` |
| E4-O3 | Should the mapper predict in PCA-50 space, given pca50 beat full-dim here? | E1 / PHASE-02 |
| E4-O4 | ECAPA is VoxCeleb-trained — RESEARCH/08 flags VoxCeleb as never properly licensed. Fine as a **measurement instrument**, must never be trained on or served | NEEDS-FROM-YOU.md |

## 7. Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E4/run_e4.py \
    --n 1600 --per-speaker 20 --corpus libritts_r_train
```
