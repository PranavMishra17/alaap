# NEEDS FROM YOU — read this first when you wake up

> Everything I could not do alone. Nothing here blocks me *right now* — I route around all of it — but each one unlocks something.
> **Last updated:** 2026-09-05 (autonomous session)

---

## 🔴 BLOCKING — a decision or credential only you can give

*(nothing yet — I'll add here if I hit a real wall)*

---

## 🟡 UNLOCKS WORK — I've built around it, but it caps what I can verify

| # | What | Why it matters | What I did instead |
|---|---|---|---|
| — | *(none yet)* | | |

---

## 🟢 FYI — decisions I made on your behalf

I had to pick to keep moving. All reversible; say the word and I'll change any of them.

| # | Decision | Reasoning |
|---|---|---|
| 1 | **Experiments run on `Qwen3-TTS-12Hz-0.6B-Base`, not 1.7B** | Your GPU is 6 GB. 0.6B uses 2.02 GB and has the same `enc_dim == hidden_size` property, so all geometry findings transfer. I'll re-run key experiments on 1.7B once the pipeline is proven |
| 2 | **GLOBE_V2 (CC0) is the working corpus for geometry experiments** | 23,519 speakers, streams without download, CC0 so no licence risk. LibriTTS-P comes in when we need *captions*, which geometry work doesn't |

---

## Standing context

- **No AI attribution in git.** Ever. Enforced.
- **Git identity:** `Pranav Mishra <pranavmishra.fc17@gmail.com>`
- Committing locally, not pushing, per your instruction.
- Progress log: **`HANDOFF.md`** §1 and `experiments/*/RESULTS.md`.

---

## Appended 2026-09-05

### 🟢 FYI — more decisions made on your behalf

| # | Decision | Reasoning |
|---|---|---|
| 3 | **LibriTTS-R replaces GLOBE_V2 for anything speaker-label-dependent** | ECAPA scores EER 20.0% on GLOBE_V2 vs 2.46% on LibriTTS-R via the identical code path. GLOBE_V2's labels/enhancement don't preserve identity. It's still fine for population *geometry* (E3 stands) |
| 4 | **ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) is the eval-axis-2 scorer** | RESEARCH/06 wants an encoder independent of the conditioning one. ⚠️ It is VoxCeleb-trained, and RESEARCH/08 flags VoxCeleb as never properly licensed — **acceptable as a measurement instrument only. It must never be trained on, served, or shipped.** If you want it swapped for something cleaner, say so; `pyannote/wespeaker-voxceleb-resnet34-LM` has the same issue, so a genuinely clean alternative may need hunting |
