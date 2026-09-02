# PHASE 06 — The Inference Service

> **Goal:** `mint` and `render` behind a stable HTTP API, backend-swappable, with the licence gate enforced in code.
> **Can start during S1**, once the identity schema is frozen. **GPU:** local
> **Evidence:** [`07`](../07-serving-and-cost.md) · [`10`](../10-performance-control.md) · [`08`](../08-licensing-propagation.md) · [`09`](../09-safety-and-watermarking.md) · [`11`](../11-production-api-landscape.md)

---

## 0. Exit criteria

| # | Criterion | Threshold |
|---|---|---|
| **X6.1** | `mint` + `render` work over HTTP against **≥2 different backends with no client change** | integration test, two adapters |
| **X6.2** | **`public_servable=False` refuses to load in the public config** | a test asserts the refusal — not a README claim |
| **X6.3** | Every render is watermarked and provenance-logged through the API path | sample 10, all detected and traceable |
| **X6.4** | `Direction` degrades **explicitly**, never silently | unsupported fields are rejected or reported in `Audio.degradations` |

---

## 1. The renderer adapter — the seam that prevents §2 happening again

```python
class Renderer(Protocol):
    backend_id: str
    backend_version: str          # pinned; recorded on every minted identity
    identity_tier: int            # 1 = vector, 2 = seed clip
    languages: list[str]
    public_servable: bool         # licence gate — enforced, not documented

    # NEW, from research:
    direction_support: Mapping[str, Honouring]   # HONOURED | APPROXIMATE | REJECT
    direction_bounds: Mapping[str, tuple]        # published ranges only

    def mint_identity(description: str, lang: str, seed: int) -> Identity: ...
    def render(identity: Identity, text: str, direction: Direction | None) -> Audio: ...
```

`Audio` carries a **`degradations`** field. **Rule: "approximate" without a published bound is `REJECT`.** A backend that cannot honour a Direction field must say so, not silently ignore it.

---

## 2. The `Direction` schema

Timbre lives on the identity; performance lives here (I5). Never merge them.

```python
@dataclass
class Direction:
    emotion: dict[Emotion, float] | None = None   # 8 categories, mixtures allowed
    intensity: float = 0.6                        # NOT 1.0 — three sources converge on this ceiling
    style: str | None = None                      # free text, behind an identity-lexicon gate
    rate: float | None = None
    pitch_var: float | None = None
    loudness: float | None = None
    target_seconds: float | None = None           # lip-sync / fixed slots
    timing_tolerance: float | None = None
    pauses: list[tuple[int, float]] | None = None
    emphasis: list[int] | None = None             # word indices
    inserts: list[tuple[int, str]] | None = None
    seed: int | None = None
    strict: bool = False                          # fail rather than degrade
```

### 2.1 Per-backend capability, from source reads

| Backend | Control surface | Notes |
|---|---|---|
| **Qwen3-TTS Base** | **nothing in the API** | `generate_voice_clone` has no `instruct`, never passes `instruct_ids`. Direction comes from **E0's training-free `x + α·τ`** |
| **CosyVoice 2/3** | **best-shaped** | separate `instruct_text`, `speed`, `<strong></strong>`, `[laughter]`, phoneme inpainting. Apache-2.0 |
| Zonos | 8-vector `[Happy,Sad,Disgust,Fear,Surprise,Anger,Other,Neutral]` L1-normalised, `pitch_std` 0–400, `speaking_rate` 0–40 | source comment **admits entanglement** |
| Chatterbox | `exaggeration` 0.25–2.0 (neutral 0.5) | documented rate coupling; `emotion_adv` reaches only T3, not S3Gen |
| **Kokoro** | **`ref_s[:,:128]`→decoder, `[128:]`→prosody predictor** | an **undocumented free timbre/prosody split**; returns `pred_dur` |
| VoxCPM2 | parenthetical control — the **official API**, not a demo hack | style collides with *content*; PilotTTS measures **−32.5% speaker similarity** with emotion on |
| Indic Parler-TTS | one channel for voice **and** style | see [`PHASE-05`](PHASE-05-indic-gate.md) §2 |
| F5-TTS `fix_duration`, MOSS-TTS `tokens=N` + `[pause X.Ys]` | the **only absolute duration controls** | matters for lip-sync |

**Do not build on punctuation-based control** — measured to fail (Interspeech 2025).

**Never install IndexTTS-2 in an environment that touches training** — §3.4(c) bars using it *or its outputs* to improve any AI model, and §1.6 defines "Use" to include running.

---

## 3. Topology

```
Browser ──► API (FastAPI, CPU, always-on)
              ├── mint   ──► designer + mapper ──► identity ──► Postgres
              ├── search ──► pgvector HNSW over identities ──► catalog
              └── render ──► job queue ──► GPU worker pool ──► R2 ──► signed URL
                                            │
                              warm pool; weights baked or volume-mounted
```

**Three corrections from research:**
- **Minting is not free if Tier 2 is the default** — and it is. Every mint is a GPU render, and the Studio wants *several candidates per description*. Cap free previews, or make previews deliberately short (2–3 s).
- **Interactive requires a warm worker.** Cold-start floor is ~10–30 s and snapshotting does not help when init is weight-loading-bound. "Type a description, hear a sample" is incompatible with scale-to-zero.
- **The render cache is a latency lever, not a cost lever** at this scale. Still build it — key on `(identity_id, backend_version, text, direction)` — but do not budget on it.

**Preview TTL:** copy Resemble's **4 hours**. It bounds preview storage and forces the persist step to be deliberate.

---

## 4. The licence gate

```python
def load_backend(adapter, deployment: Deployment):
    if deployment.is_public and not adapter.public_servable:
        raise LicenceGateError(
            f"{adapter.backend_id} is not publicly servable "
            f"(see RESEARCH/08-licensing-propagation.md)"
        )
```

**Test it.** A rule that lives only in a document will eventually be violated by a config change — and five upstream licence traps in one research pass is the evidence for that.

Servable: VoxCPM2, Qwen3-TTS (whole family), Chatterbox, Parler-TTS, CosyVoice 2/3, Kokoro, MOSS-VoiceGenerator, Indic Parler-TTS.
Not servable: IndicF5, SPRING_F5, Indic-Mio, DhVaani, VoiceSculptor, Llasa-3B, xcodec2, F5-TTS, XTTS-v2, IndexTTS-2, VibeVoice, Zonos-v0.1.

---

## 5. Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI, CPU, always-on | mint + search are CPU work |
| DB | Postgres + pgvector, HNSW | one store for identities *and* their index. **Keep any indexed vector ≤ 2000 dims** |
| Queue | Postgres-backed (pgmq / river) | one fewer moving part for a solo dev; Redis only if throughput demands it |
| Object storage | Cloudflare R2 | **$0 egress** |
| Audio | serve **Opus** (0.24 MB/min), archive WAV | 24× smaller than WAV (5.76 MB/min) |

---

*Phase spec v2 · 2026-09-02 · prev: [`PHASE-05`](PHASE-05-indic-gate.md) · next: [`PHASE-07`](PHASE-07-gpu-hosting.md)*
