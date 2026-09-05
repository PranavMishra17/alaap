# E10 — 0.6B vs 1.7B: does the bigger speaker space help?

> **Status:** ✅ RUN AND COMPLETE · **Date:** 2026-09-05
> **Data:** LibriTTS-R `train.clean.100`, 800 clips / ~80 speakers, **identical audio for both models**
> **Space:** in-domain `SpeakerSpace` fit on a disjoint speaker half (E4's finding)
> **Scorer:** ECAPA — independent of both conditioning encoders, and the stricter of the two available
> **Artefacts:** `out/results.json`

Answers three open questions at once: **E4-O2** (does 1.7B discriminate better?), **E0's biggest caveat** (the paper used 1.7B; is our extra identity loss a model-size artefact?), and **PHASE-00's** VRAM question.

---

## 0. Results

| | **0.6B-Base** | **1.7B-Base** | |
|---|---|---|---|
| dim | 1024 | **2048** | |
| **VRAM peak** | 2.19 GB | **4.09 GB** | ✅ **fits the 6 GB card** |
| embed clips/s | 41.3 | 49.4 | |
| **RTF** | 4.18 | **4.27** | ~identical |
| EER raw | 5.50% | 4.61% | |
| **EER in-domain** | 2.17% | **1.84%** | **−15% relative** |
| **d′** | 4.07 | **4.47** | |
| **effective rank** | 62.6 | **82.7** | richer manifold |
| identity fraction | 0.177 | 0.165 | |
| shell radius | 10.46 | 17.47 | |
| **drift (working space)** | 0.5169 | **0.5012** | ~unchanged |

---

## 1. Three conclusions

### ⭐ 1. 1.7B fits, and is strictly better at discrimination — switch to it

Peak VRAM **4.09 GB on a 6 GB card**, with headroom. It beats 0.6B on every identity metric: EER 1.84% vs 2.17%, d′ 4.47 vs 4.07, effective rank 82.7 vs 62.6.

**And it costs essentially nothing in speed** — RTF 4.27 vs 4.18, and embedding throughput was actually *higher* (49.4 vs 41.3 clips/s, though that gap is small enough to be thermal noise between runs).

`DEFAULT_MODEL` is now `Qwen3-TTS-12Hz-1.7B-Base`.

### 2. 1.7B does NOT fix vocoder drift — so E0's identity loss is not a model-size artefact

Drift is **0.5012 vs 0.5169**, i.e. unchanged within noise.

This matters because it removes the most attractive explanation for E0's result. The task-vector steering lost more identity than the paper reported, and the leading hypothesis was "they used 1.7B, we used 0.6B." **That hypothesis is now much weaker.** The remaining candidate explanations are the emotional corpus (CREMA-D vs theirs) and — the one E0 already established — **that the paper's WavLM metric is simply more forgiving than ECAPA.**

### 3. The E4 EER number was pessimistic

E4 measured 4.35% in-domain for 0.6B; here the same model measures **2.17%**. The configurations differ (800 clips / ~80 speakers with 10 utterances each here, vs 1600 / 85 with 20 there), and the space is fit on a disjoint half of the *same* session's speakers, which is the strictest in-domain condition.

Both are in the same range and the *direction* of every E4 conclusion is unaffected — but **2.17% is the better estimate of what this encoder can do**, and the operating thresholds should be re-derived at whichever configuration production actually uses.

---

## 2. What changes

1. **Default model → `Qwen3-TTS-12Hz-1.7B-Base`.** Better discrimination, fits the card, negligible speed cost.
2. **Experiments E0–E5 and S2 were all run on 0.6B.** Their conclusions are qualitative and hold, but any *number* in them is a 0.6B number. Re-running E2 (drift) and S2 (adherence) on 1.7B is cheap and would tighten the baseline.
3. **E0's remaining caveat narrows** to corpus and metric, not model size.
4. **The shell radius differs a lot** (10.46 vs 17.47), which is a reminder that `SpeakerSpace` artefacts are **per-model as well as per-domain**. A stored identity carries `space_ref` for exactly this reason (invariant I2).

---

## 3. Caveats

- **80 speakers, 6 rendered identities.** Small; the EER difference (2.17 vs 1.84%) is one number each, not a distribution.
- **RTF was measured on a laptop GPU that thermally throttles** (see E0 §7). Both models were measured in the same session at 61–68 °C, so the *comparison* is fair even if the absolute figures are not representative of serving hardware.
- **Only the Base checkpoints.** `VoiceDesign` and `CustomVoice` are separate models and untested here.

---

## 4. Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E10/run_model_compare.py \
    --n 800 --per-speaker 10 --n-drift 6
```
