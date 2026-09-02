# PHASE 08 — The Web App: A Voice Library

> **Goal:** a stranger mints a voice and downloads usable audio without help.
> **GPU:** cloud · **Depends on:** S6, S7 · **This is where the minimum shippable thing lands.**
> **Evidence:** [`11`](../11-production-api-landscape.md) · [`09`](../09-safety-and-watermarking.md) · [`07`](../07-serving-and-cost.md) · [`06`](../06-evaluation-harness.md)

---

## 0. Exit criterion

**X8.1 — A person who has never seen the project mints a voice and downloads usable audio without help.** Test with three real strangers, not colleagues. Record where they stall.

---

## 1. Catalog-first, because you have a cold start and no users

The four surfaces, in build order.

### 1.1 Voice Library — the landing experience

Browse the curated catalog. Filter by language, gender, age, register, style tags. Play samples inline. **Semantic search** ("gravelly old mentor") over the pgvector index.

This is the product on day one, before any mapper is good. It is also the retrieval index from S2 — the catalog and the model are the same asset.

### 1.2 Voice Studio

`type a description → mint → preview on a sample line → adjust → save`

**Show several candidates per description.** This is not a nicety — it is **how every commercial system resolves the one-to-many problem.** ElevenLabs, Resemble, and Hume all do `prompt → N previews → user selects → persist`. A human picking from samples *is* the industry's answer to a modelling problem.

**Three things to copy directly:**

| From | What | Why |
|---|---|---|
| ElevenLabs | Log **`played_not_selected`** previews | Free preference data for the S2/S3 mapper. They label it "used for RLHF." **Cheap now, expensive to retrofit** |
| Resemble | **4-hour preview TTL** | Bounds preview storage; forces the persist step to be deliberate |
| Hume | Store **speech AND prompt AND params** on save | *"stores both the speech and the prompt that shaped it"* — matches invariant I2 |

**The CFG dial** from S3 goes here, labelled honestly: *"how literally should I take your description."* No commercial product ships this; it is a genuine differentiator, and it exists because the adherence↔diversity trade is real.

**Cost warning:** every preview is a GPU render (Tier 2 is the default). Cap free previews per user per day, and keep preview clips short (2–3 s).

### 1.3 Character sheet

A character is **one identity per language**, linked by `character_id`. Make the per-language decision **visible and intentional in the UI** rather than a hidden limitation. Show which languages a character has, and which are missing.

### 1.4 Script renderer — the surface that makes this real

Paste or upload a dialogue script → assign voices per speaker → set per-line `Direction` → batch render → download a **zip plus a manifest JSON** (`line_id → file → character → language → direction → backend_version`).

**This is what makes VoiceForge usable in a game pipeline rather than a toy.** It is also where the batch economics work (cold start amortises to nothing over 200 lines).

---

## 2. Export

| Format | Use |
|---|---|
| **Opus** | default streaming/preview — 0.24 MB/min |
| **WAV 48 kHz** | explicit download, where the backend supports it |
| **OGG/Vorbis** | **the watermark survives this at 0.95** — and it is the Unity/FMOD/Wwise default. Recommend it for game use |
| ❌ **Opus for final delivery** | **Opus removes the AudioSeal watermark.** Use it for previews only, never for delivered assets |

**Manifest JSON is mandatory**, and must carry the synthetic-audio disclosure (§3) plus `backend_version` per line so a re-render is reproducible.

Unity/Unreal-friendly file naming — a small detail that decides whether anyone uses this in practice.

---

## 3. Disclosure, in the product

Required, not optional. EU Art. 50(2) has been live since 2026-08-02.

- Synthetic audio labelled **in the UI** and **in the export manifest**
- Watermark on every delivered file (AudioSeal, presence bit)
- **Warn on lossy re-encode**: if a user exports Opus, tell them the watermark will not survive
- The free public detector (Code of Practice requirement) can land at S9, but link to it from here

---

## 4. What not to build

- **Do not accept user audio upload.** Invariant I6. It is the only structural safety property the product has, and it is what keeps VoiceForge outside the ELVIS Act core and *Arijit Singh* ¶18.
- **Do not benchmark the UX against ElevenLabs' architecture** — it is the same Tier-2 pattern we already have. Benchmark *quality*, not sophistication.
- **Do not expose free numeric vector entry.** Voice design must be **constrained navigation within the fitted distribution**; arbitrary vectors outside the real-embedding hull produce artefacts.

---

## 5. Latency budget

"Type a description, hear a sample" is the core interaction, and it **requires a warm worker** — cold-start floor is 10–30 s and snapshotting does not help. Set the p95 target against S7's measured number, and if it cannot be met, say so in the UI with a progress state rather than letting the user wonder.

---

*Phase spec v2 · 2026-09-02 · prev: [`PHASE-07`](PHASE-07-gpu-hosting.md) · next: [`PHASE-09`](PHASE-09-multitenant.md)*
