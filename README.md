# Alaap · आलाप

**Natural-language character description → a persistent, reusable voice identity → arbitrary dialogue rendered in that voice.** English and Indian languages, at quality usable in games and dramatic content.

> *Alaap* — the opening improvisation in Hindustani classical music, where a voice explores its full range before the composition begins.

---

## Status

**Research pass 1 complete. No code yet.** The next step is Phase 00.

- **[`DECISIONS.md`](DECISIONS.md)** — locked decisions, with reasoning. Read before changing direction.
- **[`RESEARCH/`](RESEARCH/README.md)** — ~11.5k lines of primary-source verification across 13 domains.
  - **[`RESEARCH/00-EXECUTIVE-VERDICT.md`](RESEARCH/00-EXECUTIVE-VERDICT.md)** — start here.
  - **[`RESEARCH/GRAND-PLAN.md`](RESEARCH/GRAND-PLAN.md)** — the route, the ten invariants.
  - **[`RESEARCH/phases/`](RESEARCH/phases/)** — per-stage specs with measured exit criteria.

## The architecture

```
description ──[frozen text encoder + TRAINED mapper]──► voice identity ──[FROZEN TTS]──► audio
      A                    ~10-30M params                      B              C, swappable
```

Only the mapper is trained. The renderer is frozen and hides behind an adapter, so a better TTS is a swap rather than a rewrite.

## The stack

| Layer | Choice | Licence |
|---|---|---|
| Tier-1 renderer | `Qwen3-TTS-12Hz-1.7B-Base` | Apache-2.0 |
| Voice designer | `Qwen3-TTS-1.7B-VoiceDesign` · `MOSS-VoiceGenerator` | Apache-2.0 |
| Mapper reference | `line/promptttspp` (MDN head, 256-d) | Apache-2.0 |
| Indic (v1) | `ai4bharat/indic-parler-tts` | Apache-2.0 |
| Watermark | `facebook/audioseal` | MIT |
| Corpora | LibriTTS-P · GLOBE · IndicVoices-R | CC-BY-4.0 / CC0 |

**Languages in v1:** English + Hindi, Telugu, Bengali, Marathi, Kannada, Malayalam, Odia, Assamese.

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
