# Alaap · आलाप

**Natural-language character description → a persistent, reusable voice identity → arbitrary dialogue rendered in that voice.** English and Indian languages, at quality usable in games and dramatic content.

> *Alaap* — the opening improvisation in Hindustani classical music, where a voice explores its full range before the composition begins.

---

## ⚠️ Status: research phase. There is no code yet.

**This repository currently contains research and planning only.** Nothing here has been built, run, or benchmarked by us.

| | |
|---|---|
| **What exists** | ~11,500 lines of primary-source technical research, a phased build plan, and a decision log |
| **What does not exist** | Any code. Any model. Any dataset. Any running system. Any measured result of our own |
| **Where we are** | About to start **Phase 00** — the first experiment is a 20-minute CPU script |
| **Confidence** | Every claim is tagged HIGH / MEDIUM / LOW / **UNVERIFIED**. Ten open questions are explicitly unresolved and carry the experiment that would settle each |

If you found this looking for a working text-to-voice system: **it is not built yet.** If you found it looking for a survey of the open TTV/voice-design landscape as of September 2026, with licences traced to primary sources, that part is real and may be useful to you.

---

## What the research found

The project verifies a prior scoping document against primary sources. The headline result:

> **The two-tower architecture survives — but every component the original scope named turned out to be the wrong choice.**

```
description ──[frozen text encoder + TRAINED mapper]──► voice identity ──[FROZEN TTS]──► audio
      A                    ~10-30M params                      B              C, swappable
```

Only the mapper is trained. The renderer is frozen and sits behind an adapter, so a better TTS is a swap rather than a rewrite.

A few findings that may be useful independently of this project:

- **Synthesizing a speaker vector works, and has since 2018.** Random unit-hypersphere vectors give naturalness MOS 3.65; WGAN-sampled embeddings are an official VoicePrivacy 2026 baseline. The commonly-assumed "off-manifold" risk is not borne out.
- **The classic mode-collapse failure is misdescribed almost everywhere.** Averaging speaker embeddings does not produce a "bland average voice" — it collapses *every* identity onto **one** voice, because zero-centred dimensions shrink toward the origin. The fix is per-dimension rescaling, and it is nearly free.
- **MOS predictors are strongly anti-correlated with pitch** (DNSMOS r ≈ −0.79) where humans are ≈ −0.06. Using them as a quality gate systematically penalises high-pitched voices.
- **Five open models declare a permissive licence that their upstream chain does not support.** Details, with quotes, in [`RESEARCH/08`](RESEARCH/08-licensing-propagation.md).
- **No commercial voice-design API is "Tier 1."** Cartesia shipped caller-manipulable 192-d voice embeddings with weighted mixing, then withdrew the capability entirely on 2026-06-01.

## Read it in this order

| | |
|---|---|
| 1 | **[`RESEARCH/00-EXECUTIVE-VERDICT.md`](RESEARCH/00-EXECUTIVE-VERDICT.md)** — the verdict and the full corrections table |
| 2 | [`RESEARCH/GRAND-PLAN.md`](RESEARCH/GRAND-PLAN.md) — the route and the ten invariants |
| 3 | [`RESEARCH/phases/`](RESEARCH/phases/) — eleven per-stage specs with measured exit criteria |
| 4 | [`RESEARCH/`](RESEARCH/README.md) `01`–`13` — the evidence, one file per domain |
| 5 | [`DECISIONS.md`](DECISIONS.md) — locked decisions with reasoning |

## The planned stack

Everything below is a *plan*, not a running system.

| Layer | Choice | Licence |
|---|---|---|
| Tier-1 renderer | `Qwen3-TTS-12Hz-1.7B-Base` | Apache-2.0 |
| Voice designer | `Qwen3-TTS-1.7B-VoiceDesign` · `MOSS-VoiceGenerator` | Apache-2.0 |
| Mapper reference | `line/promptttspp` (MDN head, 256-d) | Apache-2.0 |
| Indic (v1) | `ai4bharat/indic-parler-tts` | Apache-2.0 |
| Watermark | `facebook/audioseal` | MIT |
| Corpora | LibriTTS-P · GLOBE · IndicVoices-R | CC-BY-4.0 / CC0 |

**Languages planned for v1:** English + Hindi, Telugu, Bengali, Marathi, Kannada, Malayalam, Odia, Assamese.

## The rules that matter

1. Never treat the embedding space as isotropic — per-dimension rescale everything.
2. Every identity stores **both** a vector and a seed clip, plus `backend_version`.
3. `public_servable` is enforced in code, not documentation.
4. **A licence declaration on a derived artefact is a claim, not evidence.** Trace upstream.
5. Identity stores timbre only; performance is per-line.
6. No user audio upload in the public product. Ever.
7. Watermark and provenance-log from the first render.
8. **Do not start stage N+1 until stage N's exit criterion is measured.**
9. No InstructTTSEval score without its human ceiling; no MOS predictor as a quality gate.
10. Per-demographic slices, never a single aggregate.

Full text and evidence: [`RESEARCH/GRAND-PLAN.md`](RESEARCH/GRAND-PLAN.md) §1.

---

## Caveats, stated plainly

- **Not legal advice.** The licence analysis in [`RESEARCH/08`](RESEARCH/08-licensing-propagation.md) and [`RESEARCH/09`](RESEARCH/09-safety-and-watermarking.md) is engineering research against primary licence text. Rows needing counsel are marked. Do not rely on it for your own project without your own review.
- **A snapshot, not a maintained index.** All licence and pricing readings are from **2026-09-02** and go stale fast — one model's licence tag changed 24 seconds after a maintainer's comment.
- **Corrections welcome.** If a claim here is wrong, an issue with a primary source is more useful than almost anything else. Finding errors was scored above confirming the original scope, and that standard applies to the research itself.

## Licence

Research and documentation: **[CC-BY-4.0](LICENSE)**. Code added later will be MIT, noted at the time.

---

*Research pass 1 · 2026-09-02 · primary sources only.*
