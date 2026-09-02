# 11 — Production Voice-Design APIs: What Commercial Systems Actually Do

> **Domain:** the commercial state of the art in prompt→voice design; Tier-1 vs Tier-2 in shipping products
> **Answers:** A1 (decisive — supplies the missing empirical evidence), and the product-shape question behind scope §13
> **Date:** 2026-09-02 · Pass 1 · **All API evidence from official OpenAPI specs / API references**
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [02-identity-representation.md](02-identity-representation.md) · [03-tts-backends-english.md](03-tts-backends-english.md) · [07-serving-and-cost.md](07-serving-and-cost.md) · [09-safety-and-watermarking.md](09-safety-and-watermarking.md)

---

## 0. Bottom line

- **No mainstream commercial voice-design API is Tier 1 today.** Every shipping prompt-to-voice product verified — ElevenLabs, Resemble, Hume — implements the *identical* Tier-2 pattern: `prompt → generate N preview audio samples → user selects one → persist a voice from that preview`. [HIGH]
- **The one genuine Tier-1 production system that ever existed was deliberately withdrawn.** Cartesia shipped caller-manipulable **192-dimensional voice embeddings**, weighted **vector mixing**, and direct vector→speech synthesis. All of it was **sunset on 2026-06-01**, and `/voices/mix` was removed **with no replacement**. Current docs state verbatim: *"Embeddings are not accepted in this API version."* [HIGH]
- **This is the single strongest piece of external evidence bearing on A1.** It does not by itself settle Tier 1 vs Tier 2 — a vendor may withdraw a feature for abuse, support-burden, or quality reasons rather than technical ones — but "the only company that shipped it removed it, and nobody else built it" is a materially different evidence position than the scope document assumed. [HIGH on the fact, MEDIUM on the inference]
- **The caller's manipulable representation of a voice, industry-wide, is a prompt string plus scalar knobs** (`guidance_scale`, `loudness`, `seed`, `prompt_strength`) — never a vector. The returned handle always references *generated preview audio*. [HIGH]
- **`seed`-based reproducibility is the commercial norm** (ElevenLabs: *"Same seed with same inputs produces same voice"*), i.e. what the scope doc calls **Tier 3** — reproducing identity by replaying the generation recipe. Industry treats this as sufficient. [HIGH]
- **Azure's `speakerProfileId` is a false positive for Tier 1.** Microsoft's docs say voice characteristics are *"encoded in"* it and the SSML element is literally `<mstts:ttsembedding>` — but the value is a **GUID**, an opaque server handle. Anyone grepping for "embedding" would be misled. [HIGH]
- **Two vendors' documentation is offline**, which is itself a due-diligence finding: `docs.play.ai` serves an **expired TLS certificate in front of a deleted Vercel deployment** (`DEPLOYMENT_NOT_FOUND`); Lovo's docs are an unloadable JS shell. [HIGH on the failure; both vendors UNVERIFIED]
- **Nobody exposes a diversity or adherence dial as a product surface** beyond `guidance_scale`. Alaap's planned CFG adherence↔diversity slider (scope §4.2 option 3) has no commercial precedent either. See [`01-ttv-landscape.md`](01-ttv-landscape.md), which found the same gap in the research literature. [MEDIUM-HIGH]

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| 11-1 | §4.3 treats Tier 1 (fixed embedding vector) as the target and Tier 2 as a "fallback tier" | **Inverted vs industry** | Tier 2 is what 100% of shipping voice-design products use. Tier 1 exists in zero current commercial products. Tier 2 should be the *default* design, Tier 1 the upside case. | Cartesia + ElevenLabs + Resemble + Hume API specs | HIGH |
| 11-2 | §4.3 "Tier 2 is a genuinely acceptable engineering answer" | **Understated** | It is not merely acceptable — it is the industry standard, chosen by every vendor with a shipping product. | as above | HIGH |
| 11-3 | §4.3 Tier 3 (description + RNG seed) rated "weak" | **Understated** | ElevenLabs ships seed-determinism as a documented product guarantee. It is weak *across model versions* — which remains true and important — but it is not weak in practice within a pinned version. | [ElevenLabs OpenAPI](https://api.elevenlabs.io/openapi.json) | HIGH |
| 11-4 | §6 benchmark row: "ElevenLabs Voice Design (product ceiling)" — implied to be architecturally distinct/advanced | **Correct as a quality ceiling, wrong as an architecture ceiling** | ElevenLabs Voice Design is architecturally the *same* Tier-2 pattern Alaap plans as its fallback. There is no hidden sophistication to catch up to. | [ElevenLabs design endpoint](https://elevenlabs.io/docs/api-reference/text-to-voice/design) | HIGH |
| 11-5 | §12.2 "make minting free and instant; meter only rendering", with Tier-2 minting flagged as a mere caveat | **The caveat is the main case** | If Tier 2 is the default (11-1), then *every* mint costs a GPU render, and the free-mint economic model needs rework. Multiple previews per description (scope §13.2 wants several candidates) multiplies this. | derived; see [`07-serving-and-cost.md`](07-serving-and-cost.md) | HIGH |

---

## 2. The industry-standard Tier-2 pattern

Every shipping prompt-to-voice-design API implements the same two-step shape:

```
STEP 1  design/preview          STEP 2  persist
────────────────────────        ──────────────────────────
prompt + knobs                  chosen preview id + name
      │                                │
      ▼                                ▼
N preview audio samples  ───────►  permanent voice_id
(base64 mp3 or signed URL)         (opaque handle)
+ opaque generated_voice_id
```

The user auditions and selects. **The selection step is load-bearing**: it is how the one-to-many problem (scope §4.2) is solved commercially — not by a generative head that models `p(z | description)`, but by *generating several samples and letting a human pick*. That is a product answer to a modelling problem, and it is worth taking seriously as prior art for scope §13.2's "show several candidates per description."

---

## 3. ElevenLabs Voice Design — TIER 2

**Source:** [`https://api.elevenlabs.io/openapi.json`](https://api.elevenlabs.io/openapi.json) — official OpenAPI 3.1.0, 300 paths. Confidence HIGH.

> ⚠️ **Verification trap:** `elevenlabs.io/docs/llms-full.txt` is misleadingly named — it is a 206 KB *index*, not the full corpus, and contains **zero** occurrences of `generated_voice_id`. Verification against it yields false negatives. The authoritative source is the OpenAPI JSON.

### 3.1 Step 1 — `POST /v1/text-to-voice/design`

Official description, verbatim:

> "Design a voice via a prompt. This method returns a list of voice previews. Each preview has a generated_voice_id and a sample of the voice as base64 encoded mp3 audio. To create a voice use the generated_voice_id of the preferred preview with the /v1/text-to-voice endpoint."

`VoiceDesignRequestModel`:

| Field | Type | Notes |
|---|---|---|
| `voice_description` | string, **required** | min 20, max 1000 chars |
| `model_id` | enum | `eleven_multilingual_ttv_v2` (default), `eleven_ttv_v3` |
| `text` | string, nullable | 100–1000 chars |
| `auto_generate_text` | bool | default false |
| `loudness` | number −1.0…1.0 | default 0.5; "0 corresponds to roughly -24 LUFS" |
| `seed` | int 0…2147483647 | *"Same seed with same inputs produces same voice."* |
| `guidance_scale` | number 0…100 | default 5 |
| `stream_previews` | bool | default false |
| `should_enhance` | bool | default false |
| `remixing_session_id` | string, nullable | |
| `remixing_session_iteration_id` | string, nullable | |
| `quality` | number −1.0…1.0, nullable | |
| `reference_audio_base64` | string, nullable | `eleven_ttv_v3` only |
| `prompt_strength` | number 0…1, nullable | prompt vs reference balance; v3 only |

`VoicePreviewResponseModel` — all five required: `audio_base_64` *(note the underscore before 64)*, `generated_voice_id`, `media_type`, `duration_secs`, `language`.

### 3.2 Step 2 — `POST /v1/text-to-voice`

There is **no** `/v1/text-to-voice/create` path. Body: `voice_name`, `voice_description`, `generated_voice_id` (required); optional `labels`, `played_not_selected_voice_ids` — *"List of voice ids that the user has played but not selected. **Used for RLHF.**"*

> **Design note worth stealing:** ElevenLabs harvests the *rejected* previews as a preference signal. Alaap's Voice Studio (scope §13.2) shows several candidates per description; logging which candidate the user picks — and which they auditioned and rejected — is free training data for the mapper, at zero extra product cost. Cheap to build in at S8, expensive to retrofit.

### 3.3 Is `generated_voice_id` a vector? No — three independent proofs

1. **Whole-spec scan for a numeric voice vector: zero hits.** The only float/int arrays anywhere in 300 paths are alignment timings (`character_start_times_seconds`, `word_start_times_ms`), `visual_waveform`, `waveform_visual`, `thumbnail_size`, and usage time series.
2. **All 40 "embedding" / 32 "vector" hits are RAG text embeddings** for Conversational AI agents (`EmbeddingModelEnum`, `RAGIndexRequestModel`, `vector_distance`) — unrelated to voice identity.
3. **`GET /v1/text-to-voice/{generated_voice_id}/stream`** returns `audio/mpeg`. The ID addresses **server-stored audio** — the defining Tier-2 signature.

Corroborating official statement ([help centre](https://elevenlabs.io/docs/help-center/product/voices/voice-cloning/can-i-export-my-voice-clones.md)):

> "You cannot download or export your voice clones as standalone files. Voice clones stay in your ElevenLabs account."

…and: *"Each clone will sound slightly different, even when you use the same audio."*

### 3.4 Voice Remixing — confirms the architecture

`POST /v1/text-to-voice/{voice_id}/remix` returns the **same** `VoicePreviewsResponseModel`. Remixing is preview-audio-based, not vector arithmetic. `voice_description` here has minLength **5** and describes *changes* to make; `guidance_scale` defaults to **2** (vs 5 for design).

*Interpretation (flagged as inference):* remix taking a `voice_id` → returning *preview audio* → requiring a second call to persist is the strongest single proof this is not a latent-space system. A vector-space tool would return a modified vector, not a fresh batch of MP3s to audition.

### 3.5 Preview TTL — **UNVERIFIED**

No TTL, expiry, or retention period appears in the OpenAPI spec, API reference, quickstart, product guide, or Voice Design FAQ — all five checked. **Do not assert a TTL.** The `/stream` endpoint proves retention, not an expiry window.

---

## 4. Cartesia — the rare Tier 1, now removed ⭐ **the key finding**

**Sources:** [`docs.cartesia.ai/2024-11-13/api-reference/voices/create.md`](https://docs.cartesia.ai/2024-11-13/api-reference/voices/create.md), `/voices/mix.md`, current `/api-reference/voices/*`, and [`/build-with-cartesia/tts-models/api-changes.md`](https://docs.cartesia.ai/build-with-cartesia/tts-models/api-changes.md). All embed literal OpenAPI YAML. Confidence HIGH.

### 4.1 What existed (API versions `2024-06-10`, `2024-11-13`)

The `Embedding` schema, verbatim from the official spec:

```yaml
Embedding:
  title: Embedding
  type: array
  items:
    type: number
    format: double
  description: >-
    A 192-dimensional vector (i.e. a list of 192 numbers) that represents
    the voice.
```

- **`POST /voices`** — "Create voice from an embedding". `CreateVoiceRequest`: `name`, `description`, **`embedding`** required. The `Voice` response carried `embedding`: *"The vector embedding of the voice."*
- **`POST /voices/mix`** — genuine vector arithmetic. `voices` is an array of `MixVoiceSpecifier`, a `oneOf` over `IdSpecifier {id, weight}` and `EmbeddingSpecifier {embedding, weight}`; returns `EmbeddingResponse {embedding}`. `Weight`: *"If weights do not sum to 1, they will be normalized."*
- **`POST /tts/bytes`** accepted `TTSRequestEmbeddingSpecifier`: `{mode: "embedding", embedding: [...192 floats], __experimental_controls}`.

That is a **complete Tier-1 loop**: obtain vector → blend vectors arithmetically → synthesize directly from a vector → persist a vector as a voice. Exactly the architecture Alaap's §7 diagram proposes.

### 4.2 What killed it

From the official "Breaking API changes" table:

| Breaking | Replacement | Sunset |
|---|---|---|
| Voice Embedding: `POST /voices/clone/clip` | `POST /voices/clone` | **2026-06-01** |
| **Mix Voices: `POST /voices/mix`** | **— (none)** | **2026-06-01** |
| Create Voice: `POST /voices` | `POST /voices/clone` | **2026-06-01** |

Plus: *"These endpoints stopped accepting voice embeddings on June 1, 2026"* — for `/tts/bytes`, `/tts/sse`, and the TTS WebSocket. The prior spec had already carried the inline warning: *"Voice embedding mode will no longer be supported after June 1, 2026. Use voice IDs instead."*

### 4.3 Current state (default version `2026-08-14`)

All six current `voices/*` reference pages fetched; `embedding` occurrences: **0 in every one**. `POST /voices` and `POST /voices/mix` no longer exist. Current `TTSRequestVoiceSpecifier`, verbatim:

> "Pass either a voice ID string or an object with a required `id` … **Embeddings are not accepted in this API version.**"

`POST /voices/clone` is now `multipart/form-data`: `clip` (binary audio), `name`, `language` required. Cartesia has **no prompt-based voice-design endpoint** — full docs index (`llms.txt`, 33,883 bytes) searched, zero results [MEDIUM-HIGH, absence-of-evidence from a complete official index].

### 4.4 Why this matters for Alaap

A production vendor **shipped caller-manipulable voice vectors — including the interpolation/mixing capability scope §4.3 lists as Tier 1's key advantage — and then deliberately withdrew them, providing no replacement for mixing.**

Read carefully, this cuts both ways and neither reading is free:

- **Against Tier 1:** the capability was built, exposed, and retired. Vendors do not usually remove features customers depend on unless the support, abuse, or quality cost exceeds the value. That is weak-but-real evidence that caller-facing voice vectors are more trouble than they are worth at product scale.
- **For Tier 1:** its removal says nothing about whether the *space was navigable* — the very question A2 asks. A vendor consolidating onto voice IDs for abuse-control, billing, or catalogue-governance reasons is fully consistent with the manifold working fine. The `mix` endpoint's existence is in fact **positive evidence that weighted combinations of voice vectors produced usable voices in production**.

**The honest conclusion:** Cartesia is *existence proof that Tier 1 is technically buildable* and *market evidence that it is not commercially necessary*. It should raise Alaap's confidence that vector interpolation works, and lower its confidence that Tier 1 is required for a good product. See [`02-identity-representation.md`](02-identity-representation.md) for the A1/A2 verdict this feeds.

---

## 5. Resemble AI — TIER 2, and says so explicitly

**Sources:** [`docs.resemble.ai/voice-creation/voice-design/*`](https://docs.resemble.ai/voice-creation/voice-design/design-overview). Confidence HIGH.

**Step 1 — `POST https://app.resemble.ai/api/v2/voice-design`**: body `user_prompt` (required), `is_voice_design_trial` (default true).

```json
{ "voice_candidates": {
    "voice_design_model_uuid": "VOICE_DESIGN_UUID",
    "samples": [ { "audio_url": "https://...", "sample_index": 0 } ] } }
```

**Documented TTL — the expiry evidence ElevenLabs lacks:** *"Processing is synchronous. Returns multiple voice samples to choose from. **Audio URLs expire in 4 hours.**"* [HIGH]

**Step 2 — `POST /api/v2/voice-design/{uuid}/{sample_index}/create_rapid_voice`**, body `voice_name` → `{"voice_uuid": ...}`, HTTP 202, training async. The page's own title: **"Create a voice clone from a selected voice design candidate."** Resemble names the Tier-2 pattern outright.

No embeddings anywhere: `GET /api/v2/voices/{uuid}` returns no vector; grep of the full 26,335-byte docs index for `embed` → **zero hits**.

> ⚠️ **Docs defect:** the overview page and API reference **disagree** on the response shape — flat array `voice_candidates: [{audio_url, voice_sample_index, uuid}]` vs nested `voice_candidates: {voice_design_model_uuid, samples: [{audio_url, sample_index}]}`. Field names differ too. Any integration must be validated against the live API. [HIGH]

**Chatterbox note:** no mention of Chatterbox anywhere in Resemble's hosted API docs. The open-source model's capabilities do **not** extend to the hosted API — relevant because scope §6 lists Chatterbox as a fallback renderer. [MEDIUM-HIGH]

---

## 6. Hume AI (Octave) — TIER 2, with the most explicit admission

**Source:** [`dev.hume.ai/docs/voice/voice-design`](https://dev.hume.ai/docs/voice/voice-design). Confidence HIGH.

Hume reuses the *TTS* endpoint as the design step: `POST /v0/tts` with `utterances[]` carrying both `description` (the voice prompt) and `text`, plus `num_generations`. Response `generations[]` carries `generation_id` and base64 `audio`. Persist via `POST /v0/tts/voices` with `generation_id` + `name`.

Hume's own architectural description, verbatim — an unusually candid statement of what a Tier-2 identity *is*:

> "Once you've created a generation that captures the voice you want, you can save it as a **custom voice**. This **stores both the speech and the prompt that shaped it**, so the model can reliably reproduce the same vocal identity in future requests."

**Stored artefact = audio + prompt.** This is a useful precedent for Alaap's identity record: store the seed clip *and* the description *and* the generation parameters, not just one of the three.

---

## 7. The rest — no voice design at all

| Vendor | Design? | Persist mechanism | Vector? | Notes |
|---|---|---|---|---|
| **OpenAI** | ❌ | `POST /audio/voices` (audio + `consent` id, ≤10 MiB) → `{object, id, name, created_at}` | ❌ | `instructions` (max 4096) is **delivery control on an existing voice**, not identity creation. Fixed voices: alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse, marin, cedar. Source: [official OpenAPI v2.3.0](https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml) |
| **Azure Personal Voice** | ❌ | `POST/PUT .../personalvoices/{id}` (audio + consent) | ❌ **GUID** | ⚠️ Docs say characteristics are *"encoded in"* `speakerProfileId` and the SSML element is `<mstts:ttsembedding>` — both are **false positives**. Value is `3059912f-a3dc-49e3-bdd0-02e449df1fe3`. *"the speaker profile ID is generated by the service."* |
| **Speechify** | ❌ | `POST /v1/voices` (10–30 s sample + consent challenge; speaker reads a returned phrase aloud, service transcribes and speaker-matches) | ❌ | Zero `embed` occurrences in the Create Voice reference |
| **PlayHT** | ❌ | `POST /api/v2/cloned-voices/instant` → id is literally `s3://voice-cloning-zero-shot/.../manifest.json` | ❌ | The voice ID **is** a pointer to a stored artefact bundle — the opposite of an inline vector. Endpoints tagged `"Hidden"` and absent from public `llms.txt` [MEDIUM that the index is merely incomplete] |
| **PlayAI** | — | — | — | **UNVERIFIED.** `docs.play.ai`: DNS resolves (CNAME → `cname.vercel-dns.com`), HTTPS fails `certificate has expired`; bypassing TLS purely to diagnose → HTTP 404 `DEPLOYMENT_NOT_FOUND`. **An expired cert in front of a deleted deployment** — a material vendor-viability signal. |
| **Lovo** | — | — | — | **UNVERIFIED.** ReadMe.io JS shell, no `.md` convention, no `llms.txt`; `docs.lovo.ai/llms.txt` → 404 |

---

## 8. Summary table

| System | Prompt→design? | Design endpoint | Design returns | Persist step | Vector exposed? | **Tier** | Conf. |
|---|---|---|---|---|---|---|---|
| **ElevenLabs** | ✅ | `POST /v1/text-to-voice/design` | `previews[]`: `audio_base_64`, `generated_voice_id`, `media_type`, `duration_secs`, `language` | `POST /v1/text-to-voice` → `voice_id` | ❌ none in 300-path spec | **TIER 2** | HIGH |
| **Cartesia ≤2024-11-13** | ❌ | — | — | `POST /voices` from `embedding` | ✅ **192-dim float array** | **TIER 1** | HIGH |
| **Cartesia current** | ❌ | — | — | `POST /voices/clone` (audio) | ❌ *"not accepted in this API version"* | **TIER 1 REMOVED** | HIGH |
| **Resemble** | ✅ | `POST /api/v2/voice-design` | `samples[]`: `audio_url` (**4 h TTL**), `sample_index` | `.../create_rapid_voice` → `voice_uuid` | ❌ | **TIER 2** | HIGH |
| **Hume (Octave)** | ✅ | `POST /v0/tts` w/ `description` | `generations[]`: `generation_id`, b64 `audio` | `POST /v0/tts/voices` → `id` | ❌ stores *"speech and the prompt"* | **TIER 2** | HIGH |
| **OpenAI** | ❌ | — | — | `POST /audio/voices` | ❌ | Tier 2 (clone) | HIGH |
| **Azure Personal Voice** | ❌ | — | — | `.../personalvoices/{id}` | ❌ GUID | Tier 2 (clone) | HIGH |
| **Speechify** | ❌ | — | — | `POST /v1/voices` | ❌ | Tier 2 (clone) | HIGH |
| **PlayHT** | ❌ | — | — | `POST /api/v2/cloned-voices/instant` | ❌ `s3://…/manifest.json` | Tier 2 (clone) | HIGH |
| **PlayAI** | — | — | — | — | — | **UNVERIFIED** | — |
| **Lovo** | — | — | — | — | — | **UNVERIFIED** | — |

---

## 9. What this means for the build

1. **Design the identity store Tier-2-first, Tier-1-capable** — the inverse of scope §4.3's emphasis. Both must remain representable (scope §4.3's instinct was right), but Tier 2 should be the path that is complete, tested, and shipped at S1; Tier 1 becomes an optimisation gated on the A2 experiment in [`02-identity-representation.md`](02-identity-representation.md).
2. **Store all three artefacts per identity, following Hume's precedent:** the seed clip, the description, and the full generation parameter set (`seed`, `guidance_scale`, `model_id`, `backend_version`). Any one alone is insufficient for reproduction across a version bump. This extends scope §7's identity-store field list.
3. **Rework the free-mint economics.** Scope §12.2 assumes minting is CPU-cheap because Tier 1 is the default. If Tier 2 is the default, every mint is a GPU render — and scope §13.2 wants *several candidates per description*, so it is N GPU renders. Either meter minting, cap free previews per user per day, or make preview generation deliberately short (2–3 s of audio) and cheap. Feed this to [`07-serving-and-cost.md`](07-serving-and-cost.md).
4. **Adopt the preview-and-select UX as the answer to the one-to-many problem**, not merely as a nicety. It is how every commercial system resolves it. This validates scope §13.2's "show several candidates."
5. **Log rejected previews as preference data from day one** (ElevenLabs' `played_not_selected_voice_ids`). Free supervision for the S2/S3 mapper; expensive to retrofit after launch.
6. **Copy Resemble's explicit preview TTL** (4 h). It bounds preview-audio storage cost and forces the persist step to be deliberate. Scope §12.3's object-storage budget should assume previews are ephemeral.
7. **Treat `seed` determinism as pinned-version-only.** Every vendor that offers it scopes it to a model version. This confirms scope §12.1's `backend_version`-on-every-identity rule is not paranoia — it is industry practice made explicit.
8. **Do not benchmark architecture against ElevenLabs.** Benchmark *quality* against it (scope §6 is right to), but there is no architectural sophistication to reverse-engineer: it is the same Tier-2 pattern.

---

## 10. Open — must be settled by experiment or vendor contact

| # | Question | Cheapest resolution | Est. cost | What it blocks |
|---|---|---|---|---|
| 11-O1 | Does ElevenLabs expire preview audio, and after how long? | Design a voice, record `generated_voice_id`, poll `/stream` at 1 h / 24 h / 7 d | ~$5 API credit, 1 week elapsed, unattended | Preview-storage cost model in [`07`](07-serving-and-cost.md) |
| 11-O2 | *Why* did Cartesia remove embeddings — abuse, support, quality, or manifold problems? | Ask Cartesia support/devrel directly; check their Discord changelog announcements | 1 email, ~1 week | Confidence weighting on A1; a "quality/manifold" answer would strongly favour Tier 2 |
| 11-O3 | Is PlayAI a going concern? | Retry `docs.play.ai` monthly; check company status | free | Nothing — informational only |
| 11-O4 | Does Resemble's live API match its overview page or its reference page? | One live call against a trial account | free tier | Only matters if Resemble is used as a benchmark |
| 11-O5 | How many previews does ElevenLabs return per design call, and at what audio length? | One live call, inspect `previews[]` and `duration_secs` | ~$1 | Sizing the equivalent Alaap mint cost |

---

## 11. Sources

| # | URL | Type | Used for | Conf. |
|---|---|---|---|---|
| 1 | https://api.elevenlabs.io/openapi.json | **Official OpenAPI 3.1.0 spec** (authoritative) | ElevenLabs full API surface; absence of voice vectors | HIGH |
| 2 | https://elevenlabs.io/docs/api-reference/text-to-voice/design | Official API reference | Design endpoint semantics | HIGH |
| 3 | https://elevenlabs.io/docs/eleven-api/guides/how-to/voices/voice-design | Official guide | Two-step flow | HIGH |
| 4 | https://elevenlabs.io/docs/help-center/product/voices/voice-cloning/can-i-export-my-voice-clones.md | Official help centre | No export of voice clones | HIGH |
| 5 | https://elevenlabs.io/docs/overview/capabilities/voice-remixing.md | Official capability page | Remix returns previews, not vectors | HIGH |
| 6 | https://elevenlabs.io/docs/eleven-api/concepts/voice-cloning.md | Official concepts page | IVC = inference-time conditioning; PVC = weight fine-tune | HIGH |
| 7 | https://docs.cartesia.ai/2024-11-13/api-reference/voices/create.md | Official versioned spec (embedded OpenAPI YAML) | **192-dim `Embedding` schema verbatim** | HIGH |
| 8 | https://docs.cartesia.ai/2024-11-13/api-reference/voices/mix.md | Official versioned spec | Weighted vector mixing | HIGH |
| 9 | https://docs.cartesia.ai/build-with-cartesia/tts-models/api-changes.md | **Official breaking-changes table** | **2026-06-01 sunset; no replacement for `mix`** | HIGH |
| 10 | https://docs.cartesia.ai/api-reference/tts/bytes.md | Official current spec | *"Embeddings are not accepted in this API version"* | HIGH |
| 11 | https://docs.cartesia.ai/api-reference/voices/clone.md | Official current spec | Current clone-from-audio flow | HIGH |
| 12 | https://docs.resemble.ai/voice-creation/voice-design/generate | Official API reference | `user_prompt`, candidates, **4 h TTL** | HIGH |
| 13 | https://docs.resemble.ai/voice-creation/voice-design/create-from-candidate | Official API reference | *"Create a voice clone from a selected voice design candidate"* | HIGH |
| 14 | https://docs.resemble.ai/voice-creation/voices/get | Official API reference | No embedding in voice object | HIGH |
| 15 | https://dev.hume.ai/docs/voice/voice-design | Official docs | *"stores both the speech and the prompt that shaped it"* | HIGH |
| 16 | https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml | **Official OpenAPI v2.3.0** | OpenAI voice paths; `instructions` semantics | HIGH |
| 17 | https://learn.microsoft.com/en-us/azure/ai-services/speech-service/personal-voice-create-voice | Official Microsoft Learn | `speakerProfileId` is a GUID | HIGH |
| 18 | https://learn.microsoft.com/en-us/azure/ai-services/speech-service/personal-voice-how-to-use | Official Microsoft Learn | `<mstts:ttsembedding>` naming trap | HIGH |
| 19 | https://docs.speechify.ai/build/api-reference/v1/voices/post | Official API reference | Consent-challenge clone flow | HIGH |
| 20 | https://docs.play.ht/reference/api-create-instant-voice-clone | Official API reference | `s3://…/manifest.json` voice IDs | HIGH |
| 21 | https://docs.play.ai/ | — | **Fetch failure: expired TLS cert → `DEPLOYMENT_NOT_FOUND`** | HIGH (on the failure) |
| 22 | https://docs.lovo.ai/llms.txt | — | **Fetch failure: HTTP 404** | HIGH (on the failure) |

---

*Pass 1 · 2026-09-02 · All API claims verified against official OpenAPI specifications or official API reference pages. Absence-of-feature findings rest on complete official documentation indexes and are marked MEDIUM-HIGH rather than HIGH.*
