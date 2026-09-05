# E2 — Vocoder drift, and whether a synthesized voice holds together

> **Status:** ✅ RUN AND COMPLETE · **Date:** 2026-09-05
> **Model:** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` · **30 identities × 4 lines = 120 renders**
> 15 identities from **real** extracted vectors, 15 **synthesized** from E1's winning GMM (k=5), shell-projected
> Scored with **ECAPA-TDNN**, independent of the conditioning encoder
> **Artefacts:** `out/results.json`, `out/audio/*.wav` (8 listenable samples)

---

## 0. The three results

### ⭐ 1. Synthesized identities are indistinguishable from real ones

| | **real vectors** | **synthetic vectors** |
|---|---|---|
| drift, working space | 0.5519 ± 0.0701 | **0.5392 ± 0.0771** |
| worst-case drift | 0.4046 | 0.3546 |
| consistency (ECAPA) | 0.6740 ± 0.0524 | **0.6489 ± 0.0787** |
| worst-case consistency | 0.5580 | 0.4051 |
| RTF | 4.43 | 4.60 |

**A voice sampled from a GMM over the speaker manifold behaves like a voice extracted from a real human**, on drift and on self-consistency, to within noise. The synthetic arm is very slightly worse on the worst case (0.4051 vs 0.5580 minimum consistency), which is worth watching but not disqualifying.

This is the single strongest evidence yet that the two-tower design is sound. E1 showed synthetic vectors are *statistically* in-distribution; E2 shows they *render* like real ones.

### ⭐ 2. Generated voices are MORE separable than real human speakers

Treating each of the 30 generated identities as a "speaker" and scoring with ECAPA:

| | C_same | C_diff | d′ | EER |
|---|---|---|---|---|
| **generated voices** (this run) | +0.6615 | +0.2297 | **4.72** | **2.78%** |
| real human speech (E4, same scorer) | +0.6766 | +0.2092 | 3.97 | 4.17% |

Generated identities are **easier to tell apart than real speakers are** (d′ 4.72 vs 3.97; EER 2.78% vs 4.17%).

That is not a paradox. Real recordings vary in microphone, room, health, and mood across sessions; a TTS backend renders the same identity under identical conditions every time. **The product consequence is good: identity consistency is achievable, and comfortably so.**

### 3. Drift is large in working space — but lands inside same-speaker range

Raw cosine drift reads **0.9887**, which looks excellent and means almost nothing: E4 established that raw `C_diff` between *different* speakers is already 0.9679, so raw cosine has almost no dynamic range here.

In working space, drift is **0.5519**. Compare against E4's in-domain calibration on real speech:

```
  C_diff (different speakers)   -0.0017
  DRIFT (intended -> rendered)  +0.5519    <-- here
  C_same (same speaker, 2 utts) +0.5766
```

> **The voice you get is about as similar to the vector you asked for as two recordings of one person are to each other.**

So scope §4.3's claim that a Tier-1 vector reproduces a voice *"exactly, by construction"* is **false** — RESEARCH/12 was right. But the drift lands squarely in same-speaker territory rather than drifting toward a different person. For a product that promises "this character sounds consistent", that is the standard that matters.

**Worst case is the thing to watch:** 0.4046 (real) and 0.3546 (synthetic), both below E4's `C_same`. A minority of renders drift enough to be borderline. The closed-loop correction RESEARCH/12 proposed (render → re-extract → compare → re-mint if below threshold) is therefore **worth building**, and now has a defensible threshold to trigger on.

---

## 1. What this settles

| Question | Answer |
|---|---|
| Do synthetic vectors render at all? | **Yes** — 120/120 renders succeeded, zero failures |
| Do they render as *well* as real ones? | **Yes**, to within noise on drift and consistency |
| Is the "exactly, by construction" claim true? | **No.** Drift is real and measurable |
| Is drift bad enough to break identity? | **No.** It sits at same-speaker level (0.5519 vs C_same 0.5766) |
| Can generated identities be told apart? | **Yes, better than real speakers** (EER 2.78%) |
| Does PHASE-01's X1.1 exit criterion pass? | **Yes** — consistency 0.674 mean / 0.558 worst, above E4's operating threshold of 0.2685 |

---

## 2. Consequences for the build

1. **PHASE-01 X1.1 is met, ahead of schedule.** "A character sounds the same across N lines" is measured: ECAPA consistency 0.674 mean, 0.558 worst case, against an operating threshold of 0.2685. What is *not* yet tested is a process restart and a backend version bump.
2. **Build the closed loop.** Re-extract after minting; if drift < ~0.40 in working space, re-mint. Costs one GPU render per mint — which, as RESEARCH/11 warned, erases Tier 1's free-CPU-mint economic advantage. That trade is now quantified rather than assumed.
3. **Never report drift in raw cosine.** 0.9887 is a meaningless number. Working space or nothing.
4. **RTF ~4.5 on a 6 GB laptop GPU without flash-attn.** Slow, but this is the worst-case hardware. Feeds E8's cost model; not representative of a 24 GB serving card.

---

## 3. Caveats

- **15 identities per arm is small.** The real-vs-synthetic difference (0.5519 vs 0.5392) is well within the ±0.07 spread and should not be read as synthetic being genuinely worse.
- **One backend, one model size.** All of this is 0.6B-Base. 1.7B may drift differently.
- **English only, clean read speech.** LibriTTS-R is studio audiobook audio. Drift under heavy stylisation (aged, raspy, whispered) is **still unmeasured** — that is E9, and RESEARCH/10 notes nobody has published it for anyone's model.
- **No listening test.** Every number here is machine-scored. `out/audio/` has 8 samples; they should be listened to before any of this is trusted as *perceptual* quality rather than embedding-space agreement.

---

## 4. Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E2/run_e2.py --n-real 15 --n-synth 15 --lines 4
```

~35 min on an RTX 3060 6 GB.
