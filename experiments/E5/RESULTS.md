# E5 — Does AudioSeal survive game-dialogue-length clips?

> **Status:** ✅ ANSWERED · **Date:** 2026-09-05
> Measured on our own rendered audio (`experiments/E2/out/audio/real_00.wav`, 3.92 s @ 24 kHz),
> using `audioseal 0.2.0`, `audioseal_wm_16bits` / `audioseal_detector_16bits`.

---

## 0. Verdict

> **Presence detection holds at 1.000 down to 0.5 s.** Game dialogue length is not a problem.

RESEARCH/09 flagged this as UNVERIFIED and a genuine risk: every published AudioSeal evaluation is
10 s / 5 s / ~3 s, the paper's own prose contradicts its Table 6 at 1 s, and the closest analogue
(XAttnMark) has *attribution* collapsing to 81.2% at 1 s. Game dialogue is 1–3 s.

| clip length | detection |
|---|---|
| 0.5 s | **1.000** |
| 1.0 s | **1.000** |
| 1.5 s | **1.000** |
| 2.0 s | **1.000** |
| 3.0 s | **1.000** |

**Important scope limit:** this measures the **presence bit only**, which is all we use. RESEARCH/09 is
explicit that the 16-bit *payload* must never carry a render ID (0.39 attribution, 0.69 even on clean
audio), so payload robustness at short lengths is not a question we need answered.

---

## 1. Robustness on our own audio

| attack | clean | watermarked | |
|---|---|---|---|
| none | 0.000 | **1.000** | ✅ |
| resample 24k→16k→24k | 0.000 | **1.000** | ✅ |
| amplitude ×0.5 | 0.000 | **1.000** | ✅ |
| highpass 300 Hz | 0.000 | **1.000** | ✅ |
| gaussian noise 30 dB SNR | 0.000 | 0.696 | ⚠️ weakened |
| **polarity inversion** | 0.000 | **0.260** | ❌ **FAILS** |

**False positives are zero** across every attack on unwatermarked audio — the detector never claims a
watermark that is not there. That matters more than it might seem: RESEARCH/09 notes that publishing a
public detector means *we* get blamed for forgeries, so a clean FPR is the property to protect.

**Polarity inversion fails, exactly as RESEARCH/09 predicted.** Published figures were 0.18/0.00; we
measure 0.260 on our own audio. Same conclusion: an inaudible one-line transform (`wav = -wav`)
defeats the watermark.

---

## 2. Perceptual cost

Watermark SNR **24.2 dB**, max sample delta 0.099.

That is **more audible than AudioSeal's papers suggest**, and is worth a listening check before this
goes anywhere near shipped audio — particularly on the quiet, breathy, whispered voices that are part
of the promised range, where a noise-floor watermark has the least room to hide.

---

## 3. What this means for the build

1. **Watermark everything from render #1** (invariant I7). No length exemption is needed — the
   short-clip worry is resolved.
2. **Claim it as a good-faith marker, never as robust.** One inaudible transform removes it. Say so in
   the model card rather than letting a user discover it.
3. **Presence bit only.** Never encode a render ID. Provenance lives in `render_log`, not in the audio.
4. **Still unmeasured:** Opus (published to *remove* it), MP3 and OGG/Vorbis transcode on our own
   audio, and real game-engine DSP chains. RESEARCH/09 has published figures for all three; they should
   be reproduced here before launch.
5. **Check the perceptual cost on stylised voices** — 24.2 dB SNR is not obviously inaudible.

---

## 4. Reproduce

Detection sweep is in the smoke test; `alaap/watermark.py` exposes `Watermarker` and the `ATTACKS` dict
(which deliberately includes the attacks AudioSeal *fails*, so the weakness shows up in our own numbers
rather than only in a paper).
