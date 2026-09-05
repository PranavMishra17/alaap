# NEEDS FROM YOU — read this first when you wake up

> Everything I could not do alone. Nothing here blocks me *right now* — I route around all of it — but each one unlocks something.
> **Last updated:** 2026-09-05 (autonomous session)

---

## 🔴 BLOCKING — a decision or credential only you can give

### 1. Hugging Face token + accept the IndicVoices-R terms  ⏱️ ~3 minutes

**This is the one you flagged in advance** — "if you need API keys for indic languages, implement the code and keep everything you need from me in a file". Here it is.

Every serious Indic speech corpus with real speaker diversity is **gated** on the Hub. I hit this wall on all three:

```
ai4bharat/indicvoices_r   DatasetNotFoundError: gated dataset. You must be authenticated.
ai4bharat/IndicVoices     DatasetNotFoundError: gated dataset. You must be authenticated.
ai4bharat/Rasa            DatasetNotFoundError: gated dataset. You must be authenticated.
```

Gating is not a licence problem — IndicVoices-R is **CC-BY-4.0**, which passes our audit (invariant I4). AI4Bharat just wants to know who is downloading it. So this is purely an access click.

**What to do:**

1. Sign in at <https://huggingface.co> (make an account if you have none — free).
2. Open <https://huggingface.co/datasets/ai4bharat/indicvoices_r> and click **Agree and access repository**. Do the same for <https://huggingface.co/datasets/ai4bharat/Rasa> while you are there — it is the emotion corpus we want for Indic τ vectors later.
3. Create a **read** token at <https://huggingface.co/settings/tokens>.
4. Give it to me either way:

```bash
envs/qwen3/Scripts/huggingface-cli.exe login
```

or, if you would rather not run anything, paste the token into a file I will read and never commit:

```bash
echo "hf_xxxxxxxxxxxxxxxxxxxx" > .hf_token
```

*(`.hf_token` is already in `.gitignore`. I will not print it, commit it, or send it anywhere except huggingface.co.)*

**What it unlocks:** IndicVoices-R is 1,899 speakers across 22 languages, 1,704 hours, studio-quality — it is the only corpus that can give Indic voices real speaker diversity. Without it the Indic catalog is built on ~2 speakers per language.

**While you are logged in, do one more thing —** it unblocks a *second*, bigger problem (below):

5. Open <https://huggingface.co/ai4bharat/indic-parler-tts>, accept the gate, and **copy the gate agreement text into `GATE-TERMS-indic-parler.txt`**. Screenshot is fine. I need to read what you actually agreed to.

**What I did instead:** built the entire Indic pipeline against ungated corpora so it runs the moment the token lands. Nothing is waiting on me.

---

### 2. Indic Parler-TTS is licence-blocked, and it is the backend the Indic plan runs on

I found a real defect in our own code tonight and fixed it, but the fix has a strategic cost you should see.

**The architecture problem first, since you asked it directly** ("is it just license or a backend two tower issue?"). It is **both, and the backend half is worse**:

> `Qwen3-TTS` — our entire working backend, every experiment E0–E10, the whole two-tower loop — **supports 10 languages and not one of them is Indian.** en, zh, fr, de, it, ja, ko, pt, ru, es. There is no Hindi. There is no amount of speaker-vector work that fixes this; the frozen TTS tower physically cannot produce Hindi phonemes.

So Indic needs a **different backend**, and the only clean-chain candidate is `ai4bharat/indic-parler-tts` — which is what ADR "ship Indic Parler catalog now" committed us to.

**The defect:** our code had `indic-parler-tts: True` in the licence gate, while our own audit (`RESEARCH/08` §4.7, §8.4) says **False — CONDITIONAL**. The gate would have allowed it into a public deployment. I set it to `False` and added a test pinning code to audit, because two copies of one fact drift and this pair already had.

**Why the audit blocks it** — two unsettled issues:

| # | Issue | Who can settle it |
|---|---|---|
| a | 382 of its 1,806 training hours are IITM **IndicTTS**. AI4Bharat relabels that CC-BY-4.0; the actual IITM EULA §2.2 forbids onward sublicensing. If §2.2 binds, Parler's own Apache-2.0 weight release is non-compliant and we inherit that. | IIT Madras — an email asking them to confirm the CC-BY-4.0 re-designation |
| b | The repo is **gated**, and a gate is a click-through whose terms are not in public metadata. We have not read them. | **You**, in 30 seconds — step 5 above |

**(b) is free and you can do it tonight.** (a) is an email, and `RESEARCH/08` calls it "the one worth spending a lawyer hour on".

**Until both clear:** Indic renders cannot be publicly served. Everything else Indic — corpus, captions, measurement, the mapper — is unaffected and I am building all of it. This blocks *shipping*, not *building*.

**Decision I need from you eventually (not tonight):** if IITM never answers, do we (i) ship Indic anyway on our own read of the risk, (ii) ship Indic as non-commercial/research only, or (iii) train our own Indic tower on the corpora whose chain we control? I will keep building toward all three.

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
