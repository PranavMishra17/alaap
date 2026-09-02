# PHASE 07 — GPU Hosting & Real Unit Economics

> **Goal:** replace every estimate in scope §14 with a measured number, on realistic short game-dialogue lines.
> **GPU:** cloud · **Depends on:** S6, and S0's E8
> **Evidence:** [`07`](../07-serving-and-cost.md) · [`11`](../11-production-api-landscape.md)

---

## 0. Exit criteria

| # | Criterion |
|---|---|
| **X7.1** | **p95 render latency** measured under realistic load, on **short lines** (3–15 words) |
| **X7.2** | **$/minute-of-generated-audio** measured, with the arithmetic shown and every assumption flagged |
| **X7.3** | Break-even volume computed against the commercial alternatives |
| **X7.4** | Cold-start behaviour characterised for the chosen platform |
| **X7.5** | A monthly cost figure for the chosen topology |

---

## 1. The economic correction that changes the business

**The margin story holds. The cost model in scope §14 does not.**

$/min-of-audio lands between **$0.00005 and $0.00017** — against ElevenLabs $0.075, Cartesia Scale $0.028, OpenAI `tts-1` / Deepgram Aura-1 $0.01125. Two to three orders of magnitude.

**But marginal render cost recovers nothing. The bill is a fixed warm-GPU floor.**

| Break-even for a $248/mo warm RTX 4090 | Volume |
|---|---|
| vs ElevenLabs ($0.075/min) | **55 h/mo of generated audio** |
| vs OpenAI `tts-1` ($0.01125/min) | **368 h/mo** |

> **Below ~55 h/mo, self-hosting is *more expensive per minute* than simply calling ElevenLabs.**

**Consequence:** meter renders to **shape demand**, not to cover the GPU. And seriously consider proxying to a commercial API until volume crosses break-even — the architecture already supports it, because the renderer adapter makes backends swappable. A `Renderer` implementation that calls ElevenLabs is a legitimate S7 adapter.

---

## 2. Throughput — scope §14's worry is backwards

Scope §14 warns that "throughput on a batch of short game-dialogue lines is typically far worse than on one long paragraph."

**Short prompts show *higher* aggregate throughput at every concurrency ≥ 8** — 112.6 vs 96.0 audio-s/GPU-s at c=32. TTFB costs ~12% at 3 s.

**The dangerous number is the single-stream RTF.** VoxCPM2's README claims RTF ~0.13 on a 4090; the concurrent benchmark implies the card is being understated ~8×. Two published sources disagree **3.2×**, so E8 must measure our own stack rather than trusting either.

Best available evidence is genuinely short-utterance: the vLLM-Omni VoxCPM2 benchmark averages **3.05 s of audio per request** (33.07 audio-s/s ÷ 10.83 req/s).

---

## 3. Cold start

**Realistic floor for a 2B TTS model is ~10–30 s. Nothing published beats it without pre-warming.**

| Platform | Technique | Reality |
|---|---|---|
| Modal | GPU memory snapshots | Their own docs: *"will generally not improve your cold start times — and may even worsen them"* when init is weight-loading-bound — **exactly our case**. Their best win (45s→5s) was a **0.5B** model |
| RunPod | FlashBoot | p90 < 2 s / p95 < 2.3 s, **but requires consistent traffic**, still shows a 42 s max, and **bills the cold start** |
| vLLM-Omni VoxCPM2 | — | states **~60 s** cold init |

**The decisive arithmetic:** a 30 s cold start costs $0.00575. Amortised over a 200-line batch that is **+$0.0006/min** (fine). Over one 3-second line it is **$0.115/min — worse than ElevenLabs.**

> **Cold starts are irrelevant for batch and fatal for interactive.** "Type a description, hear a sample" requires a warm worker. Budget it; do not try to engineer around it.

---

## 4. Platform choice

| Option | Cost | When |
|---|---|---|
| **RunPod Community RTX 4090, $0.34/hr → $248.20/mo warm** | cheapest | ⭐ **Start here.** Peer capacity, no SLA |
| Same, warm 12h/day | **$122.40/mo** | If traffic is predictable and diurnal |
| Beam RTX 4090 | $306.60/mo | Serverless 4090 at $0.000191667/s, **image load not billed** — the right S7 serverless choice |
| RunPod Secure L4 | $357.70/mo | Cheapest option **with an SLA** |

**Note:** an 8–12 GB dev GPU will not run VoxCPM2 under vLLM-Omni (needs ≥24 GB). Use VoxCPM-0.6B locally; the serving path is a different machine.

---

## 5. Storage and egress

- **R2 egress is $0** — confirmed. Scope §14's "sleeper cost" worry is defused by its own choice of R2.
- **The residual lever is format:** serve **Opus (0.24 MB/min)**, not WAV (5.76 MB/min). **24×.** Archive WAV, serve Opus, offer WAV on explicit download.
- Preview audio: 4-hour TTL (Resemble's precedent) keeps preview storage bounded.

---

## 6. Fine-tune budget sanity check

Scope §14 estimates $70–250/run. **Defensible, not wildly off** — $57–143 on a single RunPod Community A100, but 2–3× that on Lambda or Modal.

Anchor for scale: **Parler-TTS Mini pretrain was 4 nodes × 8 H100 × ~1.5 days = ~1,152 H100-hrs ≈ $2.3–4.6k.** We are not doing that; we train a 10–30M mapper on frozen embeddings.

**Budget $400–900 for 3 runs plus failures.**

---

## 7. The benchmark to run

Full script spec in [`07`](../07-serving-and-cost.md) §9. **The whole sweep costs under $5 and one afternoon.**

Measure: RTF and $/min at concurrency 1 / 8 / 16 / 32, on line lengths 3 / 8 / 15 / 40 words, per backend, with and without the watermark step. Report p50 / p95 / p99, not means.

**Unverified and needing our own numbers:** Indic Parler-TTS and Parler-TTS speed (**none published anywhere**), Zonos batching support, pgmq throughput, and TTS cache hit rates (**no published data exists for any product**).

---

*Phase spec v2 · 2026-09-02 · prev: [`PHASE-06`](PHASE-06-inference-service.md) · next: [`PHASE-08`](PHASE-08-web-app.md)*
