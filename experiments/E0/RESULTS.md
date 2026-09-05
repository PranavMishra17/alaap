# E0 — A training-free Direction channel, and what "SECS ≥ 0.88" actually means

> **Status:** ✅ RUN AND COMPLETE (two runs, plus a calibration study) · **Date:** 2026-09-05
> **Method:** task-vector arithmetic, [arXiv:2606.05367](https://arxiv.org/abs/2606.05367) (de Brito & Candido Junior, June 2026)
> **Model:** `Qwen3-TTS-12Hz-0.6B-Base` *(the paper used **1.7B**)* · **Data:** CREMA-D, 1,200 clips, 91 actors, speaker-disjoint split
> **Artefacts:** `out/results.json`, `out/e0b_results.json`, `out/audio/`, `out/audio_b/`

---

## 0. Why this experiment exists

Qwen3-TTS-Base — the backend chosen for Tier-1 identity — has **no per-line control in its API**. Source inspection confirmed `generate_voice_clone` has no `instruct` parameter and never passes `instruct_ids`. Without a fix, the identity backend can say *who* speaks but nothing about *how*, which is disqualifying for dramatic content.

The proposed fix:

```
τ_emo = E_i[ x(s_i, emo) ] − E_i[ x(s_i, neutral) ]
x_new = x(target, neutral) + α · τ_emo
```

---

## 1. VERDICT

> **The method works, and I reproduce the paper's numbers on the paper's own instrument.**
> **But a more discriminative encoder shows roughly twice the identity loss that the paper's metric reveals.**

| | identity retained (normalised) |
|---|---|
| **Paper**, WavLM, reported 0.907/0.902/0.926 | **0.822 – 0.904** |
| **This run**, WavLM, τ over 32 speakers | **0.853** ✅ reproduces |
| **This run**, ECAPA, same audio | **0.591** ⚠️ |

Normalisation: `(SECS − C_diff) / (C_same − C_diff)`, where 1.0 = indistinguishable from the target and 0.0 = indistinguishable from a *different* speaker. Calibrated on **real human speech** (LibriTTS-R, 400 clips / 21 speakers):

```
  WavLM   C_same 0.9538   C_diff 0.6632   range 0.2906   EER 5.34%
  ECAPA   C_same 0.6988   C_diff 0.2011   range 0.4977   EER 2.58%
```

**ECAPA is the better discriminator** — EER 2.58% against WavLM's 5.34% on identical audio — **and it is the one reporting more damage.** At α = 1.0 the steered voice sits roughly 40% of the way from the target speaker toward a different person.

---

## 2. The methodological finding, which generalises beyond this experiment

**Raw SECS numbers are uninterpretable without the encoder's `C_diff` floor.**

WavLM x-vectors are extremely concentrated: two *completely different* real speakers score **0.6632**, and two clips of unrelated noise score ~0.98. So the paper's headline "SECS ≥ 0.88" occupies only the top **0.29** of the scale, not the top 0.12 that "0.88 out of 1.0" implies.

RESEARCH/06 stated this in the abstract — *"no defensible universal cosine threshold exists; published values are encoder-specific and non-comparable"* — and this is a concrete instance. Two encoders, the same audio, opposite conclusions:

| | WavLM raw | ECAPA raw |
|---|---|---|
| τ over 32 speakers, sad | **0.921** → "identity preserved" | **0.560** → "identity badly degraded" |

**Neither is wrong.** They have different dynamic ranges. But if you report only the more forgiving one, you overstate your result — and WavLM is the more forgiving one *and* the weaker discriminator.

> **Standing rule for this project: always report `C_same` and `C_diff` alongside any similarity number, measured on the same encoder, on real speech.** A bare SECS is not a claim.

---

## 3. `avg4spk` is not optimal — more speakers is better

The paper specifies τ averaged over **exactly 4 speakers**. Measured here, that is the *worst* setting tested:

| τ speakers | WavLM norm | **ECAPA norm** |
|---|---|---|
| 4 (the paper's `avg4spk`) | 0.700 | **0.328** |
| 16 | 0.811 | **0.399** |
| **32** | **0.853** | **0.591** |

Monotone improvement on both encoders. This supports the paper's own *rationale* — a τ from few speakers carries those speakers' timbre, which drags the target's identity toward them — while contradicting its chosen setting. With 32 speakers the leak is substantially reduced.

*(Run 1 used 37 speakers and a single-speaker control: single-speaker τ gave ECAPA 0.4226 vs multi-speaker 0.4025 — indistinguishable at that α, which is why the speaker-count sweep in run 2 was needed to see the effect.)*

---

## 4. Does it actually change the delivery?

Yes, measurably, from run 1's α sweep (τ over 37 speakers):

| emotion | α | ΔF0 (Hz) | ΔF0 std | Δrate |
|---|---|---|---|---|
| anger | 0.5 → 1.5 | +27.5 → **+111.6** | +13.1 → +21.0 | −0.89 → −1.74 |
| happy | 0.5 → 1.5 | +24.9 → **+114.8** | −3.0 → +17.6 | +1.76 → −0.10 |
| sad | 0.5 → 1.5 | +4.8 → **−26.5** | +6.9 → +2.2 | −0.85 → −0.71 |

The directions are **acoustically sensible**: anger raises pitch and pitch variance and slows delivery; sad lowers pitch. So τ genuinely encodes emotion rather than noise.

**But identity degrades monotonically with α** (ECAPA 0.41 at α=0.5 → 0.31 at α=1.5), so there is a real trade, and α is a dial on it rather than a free parameter.

---

## 5. Honest limitations

- **Wrong model size.** The paper used **1.7B**; this is **0.6B** (enc_dim 1024 vs 2048). A larger, better-separated embedding space might absorb the perturbation more gracefully. **This is the single most likely explanation for any remaining gap** and is worth testing — 1.7B is 3.6 GB and should fit the 6 GB card.
- **Wrong emotional corpus.** The paper's corpus is not CREMA-D. CREMA-D is acted, 16 kHz upsampled to 24 kHz, ~3 s, and its emotional intensity is theatrical rather than conversational.
- **No listening test.** Everything here is machine-scored. `out/audio_b/` holds the α=1.0 renders; whether they *sound* like the intended emotion, and whether the identity loss is audible, is unanswered.
- **No emotion classifier.** Emotion change is inferred from acoustic proxies (F0, rate), not from `emotion2vec` as the paper uses. The direction of change is sensible but the magnitude is not directly comparable.
- **4 target speakers, 3 emotions.** Small.

---

## 6. What this means for the build

1. **Qwen3-TTS-Base now has a Direction channel** — a real one, with a measured cost. `Direction.emotion` can move from `REJECT` toward `APPROXIMATE`, with the published bound being *"~40% of the way toward a different speaker at α=1.0, by the stricter encoder"*.
2. **Use τ over ≥32 speakers, not 4.** Contradicts the paper; measured here.
3. **Default α should be conservative.** RESEARCH/10 already argued for `intensity=0.6` rather than 1.0 on independent grounds; the identity-vs-emotion trade measured here supports that.
4. **Test on 1.7B before concluding.** If identity holds better there, the whole picture improves and 1.7B becomes the stronger Tier-1 candidate for a second reason.
5. **The τ artefact is ~64 KB per emotion** and is trivially shippable — but see the licence note: CREMA-D is ODbL upstream and the mirror declares nothing, so these τ vectors are **research-lane** until that is settled (`NEEDS-FROM-YOU.md`).

---

## 7. A misdiagnosis worth recording

Run 1 appeared to **hang** — one render sat for over 8 minutes with the GPU busy. The first hypothesis was unbounded generation, so a `max_new_tokens` cap was added to the renderer.

**That was the wrong diagnosis**, and a controlled test proved it: RTF was ~70 with the cap, without it, and at 500. The cap changed nothing.

The real cause was environmental. **Five orphaned Python processes** from earlier background runs were holding **5,550 of 6,144 MiB of VRAM**, and the laptop GPU was at **87 °C with `SW Thermal Slowdown` active** (1740 vs 2100 MHz). After reaping the orphans and cooling to 67 °C, RTF returned to **4.28–5.51** against E2's measured 4.43.

The cap stays — an unbounded generation loop is a genuine serving hazard — but the lesson is operational: **reap background processes, and check `nvidia-smi` before believing a performance number.**

---

## 8. Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/E0/run_e0.py  --n 1200 --targets 6
envs/qwen3/Scripts/python.exe experiments/E0/run_e0b.py --tau-speakers 4 16 32 --alphas 0.0 1.0
```
