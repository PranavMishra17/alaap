# PHASE 10 — Public Release

> **Goal:** public site, published dataset, an honest model card, and a writeup that makes the contribution legible.
> **Depends on:** everything
> **Evidence:** [`08`](../08-licensing-propagation.md) · [`09`](../09-safety-and-watermarking.md) · [`01`](../01-ttv-landscape.md) · [`06`](../06-evaluation-harness.md)

---

## 0. What actually ships

| Artefact | Licence | Condition |
|---|---|---|
| Public site | — | S9 checklist green |
| **VoicePersona v2** | **CC-BY-4.0** (not CC0) | Upstream chain traced; anime sources dropped; NOTICE file crediting every corpus |
| **The mapper weights** | Apache-2.0 | ⚠️ see §3 — this is a real trade |
| Model card | — | §2 |
| Eval harness + fixed eval set | MIT | The most reusable thing we build |
| Writeup / paper | — | §4 |

---

## 1. Licence hygiene at release

Every corpus in the training set is CC-BY-4.0 or CC0, so the release is clean — **provided the NOTICE file actually ships**. CC-BY attribution is a condition, not a suggestion.

**Do not ship:** anything derived from ParaSpeechCaps, Emilia, VoxCeleb2, Expresso, EARS, TextrolSpeech, SpeechCraft, GigaSpeech, SPRING-INX, or any F5-TTS descendant.

**Do not re-declare a permissive licence over content you did not clear.** That is the mistake VoicePersona v1 made, and **HuggingFace disabled `ESpeech/ESpeech-igm` in May 2026 on a voice actor's complaint about exactly that pattern.**

**Google Crowdsourced Indic (SLR63–66/78/79) is CC-BY-SA** — if any of it reached the corpus, the ShareAlike obligation propagates. Check before release.

---

## 2. The model card — lead with the limitations

An honest card is the single most valuable artefact for the project's credibility, and most of its content already exists in this research corpus.

**Must include:**

- **Per-demographic eval slices**, never a single aggregate (I10). Report male/female, age bands, and the stylisation range (aged, raspy, whispered) separately.
- **The diversity numbers** — nVS and GVD. **HiStyle, VoiceDesigner, VoiceSculptor, MOSS-VoiceGenerator, Qwen3-TTS and Parler-TTS report no diversity metric at all.** Publishing ours is a genuine contribution.
- **InstructTTSEval scores with the human ceiling stated** (84.3 avg / **67.2 Role-Play**). Never a bare number (I9).
- **Where the model fails**: low-density regions of speaker space (+60% relative WER precedent), heavy stylisation, cross-lingual identity.
- **What the watermark does and does not survive**: MP3 ✅, OGG/Vorbis ✅ (0.95), **Opus ✗**, **polarity inversion ✗ (0.18/0.00)**, real reverb ✗ (0.22).
- **The safety claim, stated correctly**: *no cloning vector — we never accept reference audio.* Not "no impersonation vector."
- Known unverified items and the experiments that would settle them.

---

## 3. The open-sourcing trade — decide it deliberately

**Releasing the mapper weights voids the watermark downstream.** Anyone can run the mapper, render through an unwatermarked backend, and the compliance property evaporates.

This is a real trade with no clean answer:

| Release | Don't release |
|---|---|
| Credibility, reproducibility, community | Watermark holds for all output that exists |
| Matches the project's stated open posture | Contradicts the open posture |
| Art. 2(12) means the open-source exemption doesn't apply to Art. 50 anyway | Harder to publish a credible paper |

**Recommended:** release the **mapper** (it produces vectors, not audio) and the **eval harness**, and be explicit in the card that watermarking is a property of *our service*, not of the weights. Do not release a turnkey unwatermarked pipeline.

---

## 4. The writeup — what is genuinely novel

Three contributions, each backed by a gap found in this research:

1. **The adherence↔diversity dial.** Zero of eight surveyed papers expose CFG this way; no commercial API does either. And the trade it exposes is real and bidirectional — flow matching wins fidelity but loses adherence.
2. **Diversity measurement for description→voice.** Almost nobody reports it. A mode-collapsed mapper scores *well* on adherence and fails only here.
3. **The measure-first, reversible caption pipeline**, evaluated by re-measurement. Grounded captions make adherence objectively checkable rather than LLM-judged.

**Worth stating plainly in the writeup:** the field's open models score ~38–50 on InstructTTSEval-EN against commercial systems at 68–89. Being honest about that gap is more useful than another leaderboard claim.

---

## 5. Re-check before publishing

| Item | Why |
|---|---|
| **VPC 2026 results** (workshop 2026-09-26) | Likely supersedes several speaker-generation choices |
| **VoiceMOS 2026 results** | Released to participants 2026-08-31; re-check after 2026-09-16 |
| All licence readings in [`08`](../08-licensing-propagation.md) | A snapshot of 2026-09-02. Licences change — IndicF5's tag changed *24 seconds* after a maintainer's comment |
| The IITM IndicTTS V2 PDF | Its URL 404'd in Feb 2026. Ship the archived copy as evidence of what was agreed |
| Counsel's answer on India's prefixed-audio-disclosure rule | Gates Indic serving entirely |

---

## 6. And then

The research corpus in `RESEARCH/` is the project's memory. **Keep it current.** When a finding turns out wrong, update the file and note it — a stale research document is worse than none, and this project's whole premise is that the last attempt failed for want of grounded, measured decisions.

---

*Phase spec v2 · 2026-09-02 · prev: [`PHASE-09`](PHASE-09-multitenant.md) · [`GRAND-PLAN.md`](../GRAND-PLAN.md)*
