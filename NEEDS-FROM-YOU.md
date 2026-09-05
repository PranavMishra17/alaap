# NEEDS FROM YOU — read this first when you wake up

> Everything I could not do alone. Nothing here blocks me *right now* — I route around all of it — but each one unlocks something.
> **Last updated:** 2026-09-05 (autonomous session)

---

## 🔴 BLOCKING — a decision or credential only you can give

**Nothing. Both blockers cleared 2026-09-05.**

| was blocking | how it closed |
|---|---|
| **HF token + gated corpora** | You provided a token. `ai4bharat/indicvoices_r`, `Rasa`, `IndicVoices` and `indic-parler-tts` are all readable. Stored in `.hf_token`, gitignored, never printed and never sent anywhere but huggingface.co |
| **Unread Parler gate terms** | Answered by evidence, not by reading: `indic-parler-tts` declares **no `extra_gated_*` fields at all** — `gated: auto` with nothing but `license: apache-2.0`. There are no terms to read. `preflight.py` now checks this against the Hub rather than waiting for a paste |
| **The whole licence blockade** | You set the posture: research/personal, no commercial intent. `ADR-009`. The audit in `RESEARCH/08` stays exactly as written as the record of what may be *shipped* — it just is not a gate on *use* any more |

> **One residual, and it is not blocking:** the IITM IndicTTS EULA question (`RESEARCH/08` §4.7(a)) is still open. It bears on **redistribution**, not use. If you ever want to publish weights or an MIT release with audio, that question returns — and `provenance.assert_trainable` already refuses an unclean training mix.

**Token hygiene, worth one line:** the token is now in this conversation's history. Nothing here leaks it, but if that bothers you, rotate it at <https://huggingface.co/settings/tokens> and drop the new one into `.hf_token` — nothing else needs changing.

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

---

## Appended 2026-09-05 (later)

### 🟡 UNLOCKS WORK — licence questions I routed around

| # | What | Status | What I did |
|---|---|---|---|
| 5 | **Emotional speech corpus for the Direction channel (E0)** | CREMA-D is **ODbL** upstream; `confit/cremad-parquet` re-hosts it with **no declared licence** | Used it as a **research instrument only**. The τ (emotion direction) vectors derived from it are **research-lane until the ODbL derived-database question is answered**. If we ship a Direction channel, either settle ODbL or find a permissive emotional corpus |
| 6 | **Two HF mirrors declare licences their upstream does not support** | `NoahMartinezXiang/2018_RAVDESS` claims apache-2.0 over **CC-BY-NC-SA** RAVDESS; `NoahMartinezXiang/2014_CREMA-D` claims apache-2.0 over **ODbL** CREMA-D | **Avoided both.** This is invariant I4's exact trap and it is now 7 instances found. Do not use either |

### 🟢 FYI — more decisions

| # | Decision | Reasoning |
|---|---|---|
| 5 | **GLOBE_V2 at per-clip pairing is the mapper corpus** | The mapper needs (caption, vector) pairs, never speaker identity — so per-clip sidesteps GLOBE_V2's unreliable labels entirely and unlocks its vocal RANGE. LibriTTS-R is clean audiobook read speech with no gravelly or aged voices, which S2 identified as the binding limit on adherence |
| 6 | **Abandoned LibriTTS-R `train.clean.360`** | Streaming a 24 GB split stalled on shard fetch. GLOBE_V2 streams fine and gives more vocal range anyway |

### ⚙️ OPERATIONAL NOTE — your GPU runs hot

During the overnight run the RTX 3060 hit **87 °C with `SW Thermal Slowdown` active**
(clocks 1740 vs 2100 MHz), and orphaned Python processes from background experiments
accumulated until they held **5,550 of 6,144 MiB of VRAM**. Together those made
rendering **~10× slower** (RTF 70 vs 4.4) and looked exactly like a model bug.

Fixed by reaping the orphans; RTF returned to 4.3–5.5 and temperature to 67 °C.

**If you see renders crawl, check `nvidia-smi` before debugging the code.**
Worth considering a laptop cooling pad if this box is going to do long training runs.

---

## Appended 2026-09-05 (end of overnight run)

### 🟢 Where the numbers actually stand

**Quote `exact match 0.306` (chance 0.200) as the adherence figure — nothing higher.**
Earlier runs reported up to 0.403, but those were measured on axes entangled with pitch
(`f0_mean` vs `hnr_db` r = +0.62; 39.9% of the HNR estimate was pitch, not voice quality).
Decorrelating cost ~25% of the score, which is the correct direction — an inflated metric
would have been optimised against.

### 🟡 Decisions worth your input, none blocking

| # | Question | My call, and why |
|---|---|---|
| 7 | **Default model is now 1.7B** | E10: better on every identity metric, fits your 6 GB card at 4.09 GB peak, RTF unchanged. Reversible — pass `--model`. |
| 8 | **The τ emotion vectors are research-lane** | Derived from CREMA-D (ODbL upstream, mirror declares nothing). Fine to develop against; **needs settling before shipping a Direction channel.** |
| 9 | **A generative mapper (S3) is now motivated by evidence, not theory** | Retrieval can only offer attribute combinations that co-occur in the corpus. That is a real ceiling, and the first measured reason to want a generative head. |
