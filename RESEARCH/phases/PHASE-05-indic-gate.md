# PHASE 05 — The Indic Gate

> **Goal:** make an explicit, evidence-based decision about what Alaap ships in Indian languages — rather than assuming parity with English.
> **GPU:** rented for option C · **Depends on:** S0 (baseline scorecard, Indic column), S1
> **Evidence:** [`04`](../04-indic-track.md) · [`08`](../08-licensing-propagation.md) · [`10`](../10-performance-control.md) · [`09`](../09-safety-and-watermarking.md)

---

## 0. This phase is a decision, not a build

Scope §11 treats Indic as a parallel track reaching parity at S5. **After research, that is not available.** This phase exists to choose between three real options and record why.

---

## 1. What survived the licence audit

| Model | Verdict | Reason |
|---|---|---|
| **Indic Parler-TTS** | ✅ **servable — the only one** | Apache-2.0, clean chain, 21 languages |
| Indic-Mio + MioCodec | ❌ | Trained on **Expresso (CC-BY-NC-4.0)**; base `MioTTS-0.6B` and mandatory `MioCodec` both declare **Emilia** (gate says CC-BY-NC, tag says CC-BY-4.0) |
| DhVaani-0.5 | ❌ | Base `ZipVoice` has **no HF licence field at all**; also Emilia |
| IndicF5 / SPRING_F5 | ❌ | Fine-tunes of **CC-BY-NC-4.0 F5-TTS**; IndicF5's GitHub code additionally unlicensed |
| SPRING-INX (data) | ❌ | No licence field, no README terms; `Speech-Lab-IITM/SPRING-INX` returns **404** |
| MMS-TTS / SeamlessM4T | ❌ | **CC-BY-NC-4.0**, and single-voice-per-language |
| VoxCPM2 | ⚠️ Hindi only | Best-in-class Hindi CER 0.79%, no other Indic language |

**Indic-Mio was briefly the best hope** — MIT codec exposing a `global_embedding`, all 22 scheduled languages, 44 kHz, RTF < 0.1. The upstream trace killed it. That is invariant I4 earning its place.

## 2. And the survivor is structurally weak

**Indic Parler-TTS has zero speaker-embedding code** (grepped). Identity is a **name token in the same description string, through the same cross-attention, as the emotion word.**

| Consequence | Detail |
|---|---|
| **No Tier-1 identity is possible** | There is no vector |
| **Identity and performance share one channel** | Changing the emotion word can change the voice — breaks scope §4.4 outright |
| **The voice space is a closed set of 68 unique names** | ("Riya" appears twice in the 69.) No published experiment mints a novel voice; the card's own term for an unnamed description is *"random voice"* |
| **Quality is last place** | Worst of every system on InstructTTSEval-EN: APS 63.4 / DSD 48.7 / **RP 28.6** |
| 5 official languages have no named voice | Punjabi is "unofficial" yet has two |

---

## 3. The three options

> **DECIDED 2026-09-02 — [`ADR-002`](../../DECISIONS.md): Option A ships in v1. Option C is deferred to this gate, not dropped.**
>
> **The deferral has one condition attached.** "Decide later" must not become "decide later, then wait three months." The long-lead item for option C is **captioning IndicVoices-R** — so that work starts during **S2/S4** as a background task, independent of this gate. When S5 arrives, the fine-tune must be a *choice*, not the start of a data project. See §8.

### Option A — Curated Indic catalog ✅ **CHOSEN for v1**

Ship the 68 named voices × the confirmed languages, with **Tier-2 seed-clip identity**, and **no custom Indic voice design**.

- **Honest.** Does not promise description→novel-voice in Indic, which we cannot deliver.
- **Shippable now.** Fits the minimum-shippable-thing.
- **Useful.** A game studio needs consistent named character voices more than it needs infinite variety.
- **Mitigation for the one-channel problem:** freeze the voice-name token per identity; expose per-line direction only through the fields that measurably do not move identity. Measure drift per E9's protocol before exposing any of them.

### Option B — Research lane only

Use IndicF5 / Indic-Mio **privately** for quality benchmarking; serve nothing Indic publicly until a clean Tier-1 option exists.

- Zero legal risk, zero Indic product.
- Reasonable if counsel's answer on India's audio-disclosure rule is unfavourable (§5).

### Option C — Build the missing piece

**Reconstruct RASMALAI** from its CC-BY-4.0/MIT inputs (recipe published; IndicVoices-R, Rasa, IndicTrans2 all clean), then train an Indic description→voice model.

- This is a research project, not integration. Rented GPU, weeks.
- **Do not let it block the English track.** Fund it as a separate workstream after English ships.

---

## 4. Per-language shippability (option A)

| Verdict | Languages | Note |
|---|---|---|
| **Ship** | Hindi, Telugu, Bengali, Marathi, Kannada, Malayalam, Odia, Assamese | 8 |
| **Conditional** | Tamil (NSS 75.48, one recommended voice), Gujarati (75.36, 21 h data) | verify against the S0 Indic scorecard |
| **Hold** | **Punjabi** | officially "unofficial", **no published quality number at all**, least training data of any language (11 h) |

**Text handling:** mixed-*script* code-switching works; **Romanised Hindi degrades output** — Sarvam's own docs say so. An **IndicXlit transliteration pre-pass** is required, not optional, for Hinglish input.

---

## 5. The blocking compliance question

**India's IT Rules (notified 10 Feb 2026, and not limited to significant social-media intermediaries)** require a **"prominently prefixed audio disclosure."**

For a 1.5-second game dialogue line delivered via API to a developer, a prefixed spoken disclosure is **product-destroying, and there is no technical workaround.**

> **This is counsel question #1, and it gates all public Indic serving.** Get the answer before building option A's public surface, not after.
>
> *(The earlier "percentage of display area" labelling rule was in the October 2025 draft and did not survive.)*

---

## 6. Benchmark against the right ceiling

Scope §6 names **Sarvam Bulbul v3** as the Indic quality ceiling. **It is the wrong ceiling.** Sarvam's own blog states *"ElevenLabs v3 alpha leads on audio quality"* at full-band; Bulbul tops only **8 kHz telephony** — the wrong condition for games.

Benchmark full-band Indic quality against **ElevenLabs v3**, and use Bulbul only as the telephony-condition reference.

---

## 7. Exit criteria

| # | Criterion |
|---|---|
| **X5.1** | An option (A / B / C) is **chosen and recorded**, with the reasoning |
| **X5.2** | If A: Indic identity consistency measured against the same calibrated `C_same` as English, per language |
| **X5.3** | If A: the one-channel drift is quantified — sweep direction fields, measure identity movement |
| **X5.4** | Counsel's answer on the India audio-disclosure rule is on file |
| **X5.5** | The full-band quality gap to ElevenLabs v3 is **a number**, per language |
| **X5.6** | **The IndicVoices-R caption set exists** (§8), so that option C is a decision rather than a project start |

---

## 8. The long-lead item — start this during S2/S4, not at this gate

**Captioning IndicVoices-R is the prerequisite for option C, and it takes far longer than the decision does.** Starting it here would make the gate meaningless: the "choice" would be between shipping nothing new for months, or not doing it.

**So: begin during S2/S4, in parallel, regardless of which option this gate later picks.** The work is useful under every outcome — even under option A it produces a publishable Indic dataset, and it is the highest-leverage dataset action available to the project.

### What it involves

| Step | Detail |
|---|---|
| **Corpus** | IndicVoices-R — **1,704 h · 10,496 speakers · 22 languages · CC-BY-4.0**, licence chosen by AI4Bharat "allowing commercial usage." 93.25% extempore. **More speakers than any English corpus available to us** (LibriTTS-P has 2,443) |
| **Recipe** | Reconstruct **RASMALAI**'s pipeline. The corpus itself was never released — zero HF results — but **the recipe is fully published and every input is CC-BY-4.0/MIT** (IndicVoices-R, Rasa, IndicTrans2) |
| **Method** | The measure-first pipeline from [`PHASE-04`](PHASE-04-dataset-v2.md) — measure, bin, then have an LLM write prose *from the bins*. Grounded and **reversible**, so adherence stays objectively checkable |
| **Extra work vs English** | Data-Speech's `g2p` speaking-rate step is English-oriented; Indic needs its own G2P or a script-aware character-rate proxy. Pitch binning is per-speaker-per-gender, so a gender classifier that works on Indic speech is needed — verify the chosen one transfers |
| **Output** | A published Indic style-caption dataset. **Nothing like it exists** — no Indic corpus currently carries natural-language voice descriptions |

### Why it is worth doing under option A too

- It is the only thing standing between the project and a **genuinely novel contribution** — an Indic description→voice dataset that does not currently exist in any form.
- It makes the S5 gate a real choice.
- It is licence-clean end to end, which after five upstream traps is rare enough to be worth exploiting.

---

*Phase spec v2 · 2026-09-02 · **Option A locked 2026-09-02** ([`ADR-002`](../../DECISIONS.md)) · prev: [`PHASE-04`](PHASE-04-dataset-v2.md) · next: [`PHASE-06`](PHASE-06-inference-service.md)*
