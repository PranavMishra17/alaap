# 07 — Serving, Throughput & Unit Economics

> **Domain:** GPU hosting, cold starts, $/min-of-audio, platform stack
> **Answers:** D1, D2; replaces scope section 14's estimates
> **Date:** 2026-09-02 · Pass 1 · **All prices fetched 2026-09-02**
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [03-tts-backends-english.md](03-tts-backends-english.md) · [04-indic-track.md](04-indic-track.md) · [08-licensing-propagation.md](08-licensing-propagation.md)

---

## 0. Bottom line

- **The margin story holds by two to three orders of magnitude, and it is not close.** Marginal cost of a rendered minute on a self-hosted RTX 4090 is **$0.00005–0.00037/min** depending on which published throughput number you trust. The cheapest credible commercial API (OpenAI `tts-1`, Deepgram Aura-1) is **$0.01125/min**. Premium (ElevenLabs v2/v3) is **$0.075/min**. **[HIGH]**
- **But marginal cost is the wrong number to plan with.** At VoiceForge's likely scale the bill is a *fixed* warm-GPU floor (~$248/mo for a 24/7 RTX 4090 on RunPod Community), not a per-render charge. The card can theoretically emit **1,984–6,756 minutes of audio per GPU-hour**; nobody will ask it to. **Plan against duty cycle, not throughput.** **[HIGH]**
- **The brief's short-utterance worry is inverted.** On the only published 4090 concurrency benchmark, *short* prompts sustain **higher** aggregate throughput than long prompts at every concurrency ≥ 8 (112.6 vs 96.0 audio-s/s at c=32). The genuinely misleading number is the single-stream `RTF 0.13` in the VoxCPM2 README, which understates batched capacity by ~8×. **[HIGH]**
- **VoxCPM2's headline RTF ~0.13 on RTX 4090 via Nano-vLLM is VERIFIED** — and the conditions are worse-documented than the README implies: it is a single-stream figure with no stated utterance length. A separate vLLM-Omni measurement on H20 works out to **~3.05 seconds of audio per request** — i.e. it *is* a short-utterance benchmark, and it is the single most workload-relevant number found. **[HIGH]**
- **Cold start is the thing that actually breaks the product, not cost.** vLLM-Omni's own VoxCPM2 recipe states **~60 s** cold init. Modal explicitly warns that GPU memory snapshots **"will generally not improve your cold start times — and may even worsen them"** when initialization is weight-loading-bound, which is exactly our case. The realistic 2026 floor for a 2B TTS model from true zero is **~10–30 s**; sub-2 s only via warm-pool retention (RunPod FlashBoot), which requires consistent traffic. **[HIGH]**
- **Therefore: "type a description, hear a sample" is incompatible with scale-to-zero serverless.** You need ≥1 always-warm worker during active hours. Warm 12 h/day on RunPod Community = **$122/mo**; 24/7 = **$248/mo**. That is the real floor of the whole business. **[HIGH]**
- **The render cache saves latency, not money, at small scale.** With marginal render cost ~$0.0002/min, cache hits are *not* "the dominant cost lever" — warm-GPU duty cycle is. Build the cache for p95 and queue depth; don't justify it on GPU spend. **[MEDIUM]**
- **Egress is a solved problem the moment you pick R2** (zero egress, confirmed on Cloudflare's own docs). The brief's own choice defuses the brief's own worry. Residual lever: serve Opus, not WAV — **24× fewer bytes per minute**. **[HIGH]**
- **pgvector at 128-d is entirely unremarkable — no gotcha.** HNSW's 2,000-dim ceiling is 15× our speaker-embedding width. The trap is on the *other* column: a 3072-d text embedding **exceeds** the HNSW limit for `vector`. **[HIGH]**
- **The fine-tune budget in the scope doc is defensible**, not wildly off — but only on a single GPU at community-cloud prices. It is 20–50× below a from-scratch Parler-TTS Mini pretrain (32× H100 for ~1.5 days). **[HIGH]**

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| 1 | "Rendering = the entire GPU bill" | **Misleading** | At <100 h of audio/month the bill is the *fixed* warm-GPU floor (~$248/mo), not marginal renders (~$0.0002/min). Metering renders is demand-shaping, not cost recovery, until ~100–200 h/mo. | RunPod pricing; derived from nanovllm-voxcpm bench | HIGH |
| 2 | Short utterances "behave VERY differently… per-request overhead dominates" | **Partly wrong** | Directionally real but small. Short prompts show *higher* aggregate throughput than long at c≥8 (112.6 vs 96.0 audio-s/s at c=32). TTFB penalty on a 3 s line costs ~8–13% of throughput, not an order of magnitude. | nanovllm-voxcpm README benchmark tables | HIGH |
| 3 | "Cache hits are the dominant cost lever" | **Wrong at planned scale** | Cache is a *latency* and *queue-depth* lever. GPU spend is dominated by warm duty cycle. Still build it — for p95, not for dollars. | Derived (§2.3, §6.4) | MEDIUM |
| 4 | "Audio egress is the sleeper cost of any TTS product" | **True in general, false for this stack** | Cloudflare R2 charges **zero** egress. On S3 at $0.09/GB it would be real (~$52/mo per 100k min served as WAV). The brief's own R2 choice removes the risk. | developers.cloudflare.com/r2/pricing | HIGH |
| 5 | Tier-2 identities "DO need one GPU render at mint time" (cost caveat) | **True but mis-framed** | Economically trivial: ~$0.00007 per mint render. The real Tier-2 cost is **cold-start latency** on the mint path, not dollars. | Derived from RunPod serverless $/s + nanovllm TTFB | HIGH |
| 6 | Fine-tune "2–5 days A100-class ≈ $70–250/run", 3 runs | **Defensible** | Holds for **one** A100 at RunPod Community ($1.19/hr → $57–143). 2–3× higher on Lambda ($2.79/hr → $134–335). Budget **$400–900** for 3 runs incl. failures + checkpoint storage. | runpod.io/pricing; lambda.ai/pricing | HIGH |
| 7 | Implicit: 8–12 GB dev GPU can run the 2B backend | **Needs qualification** | VoxCPM2 weights fit in ~8 GB, but vLLM-Omni's own recipe requires **≥24 GB** (4.9 GB weights + 15.2 GB KV + 2 GB talker buffers ≈ 22 GB). On 8–12 GB locally you run VoxCPM-0.6B or plain PyTorch at reduced `gpu_memory_utilization`. | recipes.vllm.ai/openbmb/VoxCPM2 | HIGH |
| 8 | pgvector for 128-d speaker + text embedding | **Confirmed, with one trap** | 128-d is trivially fine. But `vector` HNSW caps at **2,000 dims** — a 3072-d text embedding will not index. Choose a ≤2000-d text embedding. | github.com/pgvector/pgvector | HIGH |
| 9 | Public-serving licence gate | **Confirmed clean** | VoxCPM2, Zonos-v0.1, Parler-TTS, Indic Parler-TTS all **Apache-2.0** (code + weights), no acceptable-use rider. Serving engines: nanovllm-voxcpm **MIT**, vLLM-Omni **Apache-2.0**. All `public_servable = True`. | Verified LICENSE file + model cards | HIGH |

---

## 2. D1 — throughput and $/min-of-audio

### 2.1 Published RTF / throughput per model

**Headline single-stream figures (as published by the model authors):**

| Model | Params | RTF | Hardware | Utterance length | Batched? | Source | Confidence |
|---|---|---|---|---|---|---|---|
| VoxCPM2 (PyTorch) | 2B | **~0.30** | RTX 4090 | **not stated** | no | OpenBMB/VoxCPM README | HIGH |
| VoxCPM2 (Nano-vLLM) | 2B | **~0.13** | RTX 4090 | **not stated** | yes (engine) | OpenBMB/VoxCPM README | HIGH |
| VoxCPM2 (vLLM-Omni) | 2B | **~0.12** steady-state | RTX 4090 | not stated | yes, `max_num_seqs: 4` default | recipes.vllm.ai | HIGH |
| VoxCPM1.5 (PyTorch / Nano-vLLM) | 0.8B | 0.15 / 0.08 | RTX 4090 | not stated | yes | OpenBMB/VoxCPM README | HIGH |
| VoxCPM-0.5B (PyTorch / Nano-vLLM) | 0.6B | 0.17 / 0.10 | RTX 4090 | not stated | yes | OpenBMB/VoxCPM README | HIGH |
| VoxCPM2 (llama.cpp-omni Q8_0) | 2B | 1.76 | Apple M4 Pro / Metal | not stated | no | OpenBMB/VoxCPM README | HIGH |
| Zonos-v0.1 (transformer & hybrid) | 1.6B | **~0.5** (stated as "real-time factor ~2×") | RTX 4090 | not stated | **UNVERIFIED** | Zyphra/Zonos README | HIGH |
| Indic Parler-TTS | 0.9B | **UNPUBLISHED** | — | — | supports SDPA/FA2/`torch.compile`/batching per Parler-TTS docs, no numbers | ai4bharat/indic-parler-tts card | **UNVERIFIED** |
| Parler-TTS Mini v1 | 0.88B | **UNPUBLISHED** | — | — | batching documented, no numbers | parler-tts/parler-tts-mini-v1 | **UNVERIFIED** |

> ⚠️ **Zonos naming trap.** Zyphra reports "real-time factor ~2×" meaning *2 s of audio per 1 s of compute*. Under the conventional RTF definition (compute ÷ audio) that is **RTF ≈ 0.5** — roughly **4× slower than VoxCPM2 under Nano-vLLM**. Do not compare the raw numbers across the two repos. Zyphra also state a **200–300 ms** latency target on a 4090, and that producing one second of audio requires **774 tokens** (86 frames × 9 codebooks).

**The number that actually matters — batched, short-utterance, on real hardware:**

*Source: `a710128/nanovllm-voxcpm` README, model `openbmb/VoxCPM2`, **NVIDIA GeForce RTX 4090**. Published `RTF_per_req` is defined by the author as `mean((request_wall_time − TTFB) / request_audio_duration)`.*

| Concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF/req (published) | **Aggregate audio-s/s (DERIVED = c ÷ RTF)** |
|---:|---:|---:|---:|---:|
| **Short prompt, no LoRA** ||||
| 1 | 0.0672 | 0.0672 | 0.1027 | **9.7** |
| 8 | 0.0789 | 0.0790 | 0.1307 | **61.2** |
| 16 | 0.0860 | 0.0864 | 0.1764 | **90.7** |
| 32 | 0.1142 | 0.1148 | 0.2842 | **112.6** ← peak |
| 64 | 0.1885 | 0.1907 | 0.6054 | **105.7** |
| **Long prompt, no LoRA** ||||
| 1 | 0.0768 | 0.0768 | 0.1163 | **8.6** |
| 8 | 0.0865 | 0.0867 | 0.1492 | **53.6** |
| 16 | 0.1346 | 0.1349 | 0.2017 | **79.3** |
| 32 | 0.2677 | 0.2684 | 0.3334 | **96.0** |
| 64 | 0.5510 | 0.5544 | 0.6724 | **95.2** |
| **Short prompt, LoRA (32 slots)** ||||
| 1 | 0.1375 | 0.1375 | 0.1284 | **7.8** |
| 32 | 0.2358 | 0.2366 | 0.3419 | **93.6** |
| 64 | 0.3287 | 0.3312 | 0.6400 | **100.0** |
| 128 † | 0.4712 | 0.4749 | 1.3215 | **96.9** |

† measured at `gpu_memory_utilization=0.7`. The aggregate column is **my derivation**, not published — flagged DERIVED throughout.

**Independent cross-check (different engine, different GPU):**

*Source: vLLM Blog, "Engineering TTS Inference in vLLM-Omni", 2026-06-23 — vendor engineering blog reporting their own platform numbers.*

| Model | Hardware | Concurrency | Req throughput | **Audio throughput** | **DERIVED s of audio/request** |
|---|---|---:|---:|---:|---:|
| VoxCPM2 | H20 × 1 | 64 | 4.19 → **10.83 req/s** | 12.16 → **33.07 audio-s/s** | **3.05 s** |
| Fish Speech S2 Pro | H20 × 1 | 64 | 5.95 req/s | 23.72 audio-s/s | 3.99 s |
| Higgs Audio V3 | H20 × 1 | 16 | 5.18 req/s | 35.26 audio-s/s | 6.81 s |
| Qwen3-TTS | H20 × 2 | 64 | — | 26.55 → **42.88 audio-s/s** | — |

**This is the single most important line in the document:** dividing VoxCPM2's published audio throughput by its published request throughput gives **33.07 ÷ 10.83 = 3.05 seconds of audio per request**. The vLLM-Omni VoxCPM2 benchmark **is a short-utterance benchmark**, squarely in game-dialogue territory. We are not extrapolating from paragraph numbers. **[HIGH — arithmetic on two published figures from the same table]**

vLLM-Omni also states directly that for VoxCPM2, after their VAE-decoder fix, **"long-text RTF no longer grows with text length; all lengths stay around RTF 0.132–0.138."** Length-independence is claimed by the engine authors. **[HIGH]**

**Reconciling the two sources.** 105.7 audio-s/s (4090, Nano-vLLM, DERIVED) vs 33.07 audio-s/s (H20, vLLM-Omni, published) is a **3.2× gap**. Three confounds, none resolvable from published data: different GPU (H20 is bandwidth-rich, FLOP-poor relative to a 4090), different engine and batching strategy, and different measurement definitions (`(wall − TTFB)/duration` vs end-to-end system throughput). **I do not claim to know which is right.** Below I compute costs under *both*, treating 33.07 as the conservative floor.

**Batching support summary:**

| Model | Batching | Continuous batching | vLLM path | Claimed speedup | Claimed by |
|---|---|---|---|---|---|
| VoxCPM2 | **Yes** | Yes — "concurrent requests via an internal scheduler" (Nano-vLLM); "CFM/LocDiT decode-tail batching" (vLLM-Omni) | **Nano-vLLM (MIT) and vLLM-Omni (Apache-2.0)**, both first-class | **+172.0% audio throughput**, +158.8% req throughput (vLLM-Omni vs baseline, H20, c=64) | vLLM project (own platform) |
| Zonos-v0.1 | **UNVERIFIED** — README does not state batching support; Zyphra's hosted API advertises "no restrictions on concurrent generations" but publishes no throughput | UNVERIFIED | None found | — | — |
| Indic Parler-TTS / Parler-TTS | Documented (SDPA, FA2, `torch.compile`, batching, streaming) but **no numbers published** | UNVERIFIED | None found | — | — |

> **Architectural consequence of the LoRA rows.** If a VoiceForge identity is realised as a **per-voice LoRA**, the Nano-vLLM benchmark caps you at **32 resident LoRA slots**, and every batch mixing more than 32 distinct voices thrashes. If identity is a **conditioning vector** (the brief's Tier-1 design), batching across arbitrarily many distinct voices is free. **The Tier-1 vector design is worth roughly 30% throughput and removes a hard batch-diversity ceiling.** This is a strong independent argument for the mapper-to-vector architecture. **[MEDIUM — inferred from the published LoRA-vs-no-LoRA delta]**

### 2.2 The short-utterance problem

The brief asked me to flag published numbers that are long-paragraph numbers and therefore optimistic. Here is the honest accounting:

**Numbers that ARE long-paragraph-or-unknown and therefore suspect:**
- VoxCPM2 README's `RTF ~0.13` — **utterance length not stated anywhere** in the README or the technical report. Treat as unknown-length, single-stream.
- Zonos' `~2× real-time` — length not stated.
- Every Parler-TTS number — there are none.

**Numbers that are genuinely short-utterance:**
- The vLLM-Omni VoxCPM2 row, at a derived **3.05 s/request**.
- The nanovllm-voxcpm "short prompt" tables (length not numerically defined — **UNVERIFIED**, but explicitly contrasted against a "long prompt" arm, so the *relative* comparison is sound).

**What the data actually shows.** Per-request overhead is real but modest. TTFB is **67 ms at c=1** rising to **189 ms at c=64** for short prompts. Model that as a full end-to-end throughput, including TTFB, for a *D*-second utterance:

```
aggregate_audio_s_per_s = c × D / (TTFB + RTF_per_req × D)
```

At c=32, short prompt (TTFB 0.1142 s, RTF 0.2842):

| Utterance D | Aggregate audio-s/s | Loss vs. TTFB-excluded 112.6 |
|---:|---:|---:|
| 2 s | 93.8 | −16.7% |
| 3 s (typical dialogue line) | **99.3** | **−11.8%** |
| 5 s | 104.0 | −7.6% |
| 10 s (paragraph) | 108.2 | −3.9% |

**Verdict: the short-utterance penalty is ~12% at 3 s, not an order of magnitude.** And it is more than offset by the fact that short prompts sustain higher raw throughput than long ones (112.6 vs 96.0 at c=32) — because long prompts inflate TTFB 2.3× (0.268 s vs 0.114 s) as prefill grows. **The brief's stated fear is directionally inverted for this model.** **[HIGH for the published inputs, MEDIUM for the model, since D for "short prompt" is undefined]**

The number that *is* dangerously optimistic is the opposite one: quoting `RTF 0.13` as capacity implies 7.7× real-time, when the batched 4090 figure is ~100–113× real-time. **The README understates the card by ~8×** — a planning error in the safe direction, but an error.

### 2.3 $/minute-of-audio (show the arithmetic)

**Stated assumptions, all flagged:**
- **A1.** Utterance = 3 s of audio (game dialogue line). Grounded in the derived 3.05 s/req from vLLM-Omni.
- **A2.** RTX 4090 at **RunPod Community Cloud $0.34/hr** (fetched 2026-09-02). Community Cloud is peer-supplied capacity — availability and reliability are weaker than Secure Cloud. Secure Cloud shown alongside at $0.74/hr.
- **A3.** Two throughput scenarios, since the sources disagree 3.2× (§2.1).
- **A4.** Ignores storage, egress, CPU/API host — priced separately in §6 and §7.
- **A5.** Assumes the GPU is *fully* loaded at the stated concurrency. §2.3d corrects for duty cycle, which is where reality lives.

**(a) OPTIMISTIC — nanovllm-voxcpm on RTX 4090, c=32, short prompt**

```
Throughput (DERIVED)   = 32 ÷ 0.2842                = 112.6 audio-s per GPU-second
Per GPU-hour           = 112.6 × 3600               = 405,360 audio-seconds
                       = 405,360 ÷ 60               = 6,756 minutes of audio / GPU-hour

RunPod Community $0.34 : $0.34 ÷ 6,756              = $0.0000503 per minute of audio
RunPod Secure    $0.74 : $0.74 ÷ 6,756              = $0.0001095 per minute of audio
```

**(b) CONSERVATIVE — vLLM-Omni measured 33.07 audio-s/s, applied to a 4090 price**

*Deliberately pessimistic: it assumes an RTX 4090 performs no better than the H20 the number was measured on.*

```
Throughput (published) = 33.07 audio-s per GPU-second
Per GPU-hour           = 33.07 × 3600 = 119,052 audio-s = 1,984 minutes / GPU-hour

RunPod Community $0.34 : $0.34 ÷ 1,984 = $0.0001714 per minute of audio
RunPod Secure    $0.74 : $0.74 ÷ 1,984 = $0.0003730 per minute of audio
```

**(c) SERVERLESS, single request, no batching — the S7 case**

```
Wall time for one 3 s line at c=1 = TTFB + RTF × D
                                  = 0.0672 + (0.1027 × 3) = 0.3753 s

RunPod Serverless 24 GB @ $0.69/hr = $0.00019167 / second
Cost per request  = 0.3753 × 0.00019167 = $0.00007193
Cost per minute   = $0.00007193 × 20 requests = $0.0014386 / minute of audio

(Beam RTX 4090 serverless @ $0.000191667/s gives an essentially identical $0.00144/min.)
```

**Cold-start surcharge (RunPod bills container start — see §3):**
```
30 s cold start × $0.00019167/s = $0.00575 per cold start

  Amortised over a 200-line batch (10 min of audio):  +$0.000575/min   → tolerable
  Amortised over ONE 3-second line:                    $0.1150/min     → CATASTROPHIC
```
**This single line is the architectural conclusion of the whole document: cold starts are irrelevant for batch and fatal for interactive.**

**(d) DUTY-CYCLE REALITY — the number to actually plan with**

A warm GPU costs the same whether it is saturated or idle. The honest unit cost is `monthly_cost ÷ minutes_actually_rendered`:

```
RunPod Community RTX 4090, 24/7 = $0.34 × 730 = $248.20 / month
Theoretical capacity (conservative case) = 1,984 min/hr × 730 = 1,448,320 min/mo

At   1,000 min/mo rendered:  $248.20 ÷ 1,000     = $0.248 / min   ← WORSE than ElevenLabs
At  10,000 min/mo rendered:  $248.20 ÷ 10,000    = $0.0248 / min  ← ~ Cartesia
At 100,000 min/mo rendered:  $248.20 ÷ 100,000   = $0.00248 / min ← 4.5× cheaper than the cheapest API
At full saturation:                                $0.00017 / min
```

**Break-even volume against each commercial provider, for a $248.20/mo warm 4090:**

| Beaten provider | Their $/min | Break-even min/mo | = hours of audio/mo |
|---|---:|---:|---:|
| ElevenLabs v2/v3 Multilingual | $0.0750 | 3,309 | **55 h** |
| ElevenLabs Flash/Turbo | $0.0375 | 6,619 | **110 h** |
| Cartesia Scale | $0.0280 | 8,864 | **148 h** |
| OpenAI `tts-1-hd` / Deepgram Aura-2 | $0.0225 | 11,031 | **184 h** |
| OpenAI `tts-1` / Deepgram Aura-1 | $0.01125 | 22,062 | **368 h** |

**Serverless-vs-warm crossover:** serverless at $0.69/hr billed only when active equals the $248.20/mo warm card at `248.20 ÷ 0.69 = 360 active hours/month = 49% duty cycle`. Below ~49% duty cycle, serverless wins on paper — and cold-start billing pushes the real crossover *lower* still, but latency pushes the product decision the other way.

### 2.4 Commercial price comparison

**Conversion assumption (stated explicitly, as required):** **750 characters per minute of synthesised speech.** This is *not* my invention — it is derived from Cartesia's own published plan table, where 1.25 M credits = 1,667 min and 8 M credits = 10,667 min, both exactly **750 credits/min**. (Whether 1 Cartesia credit = 1 character is **UNVERIFIED** — the pricing page does not say.) The conventional book figure of 150 wpm × ~5.8 chars/word gives ~870 chars/min; using 750 is therefore the **conservative** choice — it makes the commercial APIs look *cheaper* per minute than a higher figure would, i.e. it argues *against* my own conclusion. All rows fetched 2026-09-02.

| Provider | Model / tier | List price | Unit | **$/min equivalent** (arithmetic) | Source | Confidence |
|---|---|---|---|---|---|---|
| **ElevenLabs** | v3 & v2 Multilingual | $0.10 | per 1,000 chars | 0.10 × 0.750 = **$0.0750** | elevenlabs.io/pricing/api | HIGH |
| ElevenLabs | v3 Conversational, Flash, Turbo | $0.05 | per 1,000 chars | 0.05 × 0.750 = **$0.0375** | same | HIGH |
| ElevenLabs | Creator $22 / 220k chars | — | plan-implied | 220,000÷750=293 min; 22÷293 = **$0.0750** | same | HIGH |
| ElevenLabs | Scale $299 / 2.99M chars | — | plan-implied | 3,987 min; 299÷3,987 = **$0.0750** | same | HIGH |
| ElevenLabs | Business $990 / 9.9M chars | — | plan-implied | 13,200 min; 990÷13,200 = **$0.0750** | same | HIGH |
| **Cartesia** | Pro $5 / 100k credits (~133 min) | — | plan-implied | 5÷133 = **$0.0376** | cartesia.ai/pricing | HIGH |
| Cartesia | Startup $49 / 1.25M (~1,667 min) | — | plan-implied | 49÷1,667 = **$0.0294** | same | HIGH |
| Cartesia | Scale $299 / 8M (~10,667 min) | — | plan-implied | 299÷10,667 = **$0.0280** | same | HIGH |
| **OpenAI** | `tts-1` | $15.00 | per 1M chars | 15 × 750/1e6 = **$0.01125** | developers.openai.com/api/docs/pricing | HIGH |
| OpenAI | `tts-1-hd` | $30.00 | per 1M chars | 30 × 750/1e6 = **$0.0225** | same | HIGH |
| OpenAI | `gpt-4o-mini-tts` | $12.00 | per 1M **audio output tokens** | **UNVERIFIED** — audio-tokens-per-second not published on the pricing page; cannot convert | same | **UNVERIFIED** |
| OpenAI | `gpt-realtime` | $64.00 out | per 1M audio tokens | **UNVERIFIED** — same reason | same | **UNVERIFIED** |
| **Deepgram** | Aura-1 (PAYG) | $0.0150 | per 1,000 chars | 0.015 × 0.750 = **$0.01125** | deepgram.com/pricing | HIGH |
| Deepgram | Aura-1 (Growth) | $0.0135 | per 1,000 chars | **$0.010125** | same | HIGH |
| Deepgram | Aura-2 (PAYG) | $0.030 | per 1,000 chars | **$0.0225** | same | HIGH |
| Deepgram | Aura-2 (Growth) | $0.027 | per 1,000 chars | **$0.02025** | same | HIGH |
| **Sarvam** | Bulbul v3 | ₹30 | per 10,000 chars | ₹3/1k × 0.750 = **₹2.25/min**; at an *assumed* ₹85–90/USD ≈ **$0.025–0.026** | docs.sarvam.ai pricing | HIGH (INR) / **ASSUMPTION** (USD) |
| **Google Cloud TTS** | Standard / WaveNet / Neural2 / Chirp 3 HD / Studio | — | per 1M chars | **UNVERIFIED — see note** | cloud.google.com/text-to-speech/pricing | **UNVERIFIED** |
| **Azure AI Speech** | Neural, Neural HD | — | per 1M chars | **UNVERIFIED — see note** | azure.microsoft.com/…/speech-services/ | **UNVERIFIED** |
| **PlayHT / PlayAI** | — | — | — | **UNVERIFIED — see note** | play.ht / play.ai | **UNVERIFIED** |

**UNVERIFIED notes — I did not substitute a remembered or third-party number:**
- **Google Cloud TTS.** The official pricing page (`cloud.google.com/text-to-speech/pricing`) is client-rendered; four separate fetches returned only the page title with the pricing table truncated. Search surfaced third-party sites quoting $4/1M (Standard & WaveNet), $16/1M (Neural2), $30/1M (Chirp 3 HD), $160/1M (Studio), with free tiers of 4M chars (Standard) and 1M chars (WaveNet/Neural2/Chirp 3/Studio). **These are third-party listicles and are inadmissible under this project's evidence standard — I record them only so a human can verify them against the live page, and they must not be used in any model until confirmed.** If roughly correct, Chirp 3 HD lands at ~$0.0225/min, i.e. level with OpenAI `tts-1-hd`, which does not change any conclusion here.
- **Azure AI Speech.** The pricing page renders literal `$-` placeholders where the numbers belong; only the free tier (**0.5 M characters/month**) resolved. Requires the Azure pricing calculator or a signed-in portal session.
- **PlayHT/PlayAI.** Neither `play.ht` nor `play.ai` resolved via DNS from this environment on 2026-09-02. Company status unconfirmed.

### 2.5 VERDICT on the margin story

**The margin story holds, decisively — but the brief has the mechanism wrong.**

| | Self-hosted VoxCPM2 (conservative, saturated) | Self-hosted (10% duty cycle) | Cheapest API | Premium API |
|---|---:|---:|---:|---:|
| $/min of audio | $0.00017 | $0.0017 | $0.01125 | $0.0750 |
| Ratio vs cheapest API | **66× cheaper** | **6.6× cheaper** | 1× | — |
| Ratio vs ElevenLabs | **441× cheaper** | **44× cheaper** | — | 1× |

Even under the pessimistic branch — H20-grade throughput, on a rented consumer card, at 10% utilisation — self-hosting beats the cheapest commercial API by 6.6× and ElevenLabs by 44×. There is no plausible assumption set where a description-conditioned TTS product built on Apache-2.0 weights loses to reselling ElevenLabs.

**Three caveats that matter more than the ratio:**

1. **The margin is a step function, not a slope.** Below ~55 h of rendered audio per month, a warm GPU is *more expensive per minute* than ElevenLabs, because you are paying $248/mo for an idle card. The product must either (a) start on serverless and accept batch-only latency, or (b) treat the warm-GPU line as a fixed cost of *existing*, funded by subscription rather than by render metering.
2. **Metering renders is not how you cover the GPU bill.** At $0.0002/min marginal, a render meter recovers nothing. Meter to shape demand and cap abuse; price the *subscription* against the $248/mo floor. This is a real correction to the brief's economic model, not a quibble.
3. **The competitive moat is not price.** At these ratios, price is not the differentiator — no customer distinguishes $0.0002 from $0.002. The moat is the description→voice mapper and the catalog. Do not build the business case on undercutting ElevenLabs; build it on doing something ElevenLabs does not do, and note that the *cost* of doing so is negligible.

---

## 3. D2 — cold starts

| Platform | Technique | Published cold start | Conditions | Source | Confidence |
|---|---|---|---|---|---|
| **Modal** | GPU memory snapshot (alpha) | **20 s → 2 s** (Parakeet/NeMo, P0); **45 s → 5 s** (vLLM Qwen2.5-0.5B, P0); **8.5 s → 2.25 s** (ViT + `torch.compile`, P0) | "3–10× faster" for init-heavy functions. **Alpha.** Incompatible with multi-GPU and non-CUDA code. **Requires code rewriting.** | modal.com/blog/gpu-mem-snapshots | HIGH |
| **Modal** | ⚠️ **Snapshot anti-pattern** | — | *"If the majority of your initialization latency is spent loading weights, GPU Memory Snapshots will generally not improve your cold start times — and may even worsen them."* | modal.com/docs/guide/memory-snapshot | HIGH |
| **Modal** | Container boot | **"~one second"** | Base container only, before any model load | modal.com/docs/guide/cold-start | HIGH |
| **Modal** | Weights baked into image / Volume | *"For models in the tens of gigabytes, this can reduce boot times from minutes to seconds."* | No specific seconds given | modal.com/docs/guide/cold-start | HIGH |
| **Modal** | Keep-warm | `min_containers`, `buffer_containers`; `scaledown_window` default **60 s**, settable **2 s – 20 min** | Billed while warm | modal.com/docs/guide/cold-start | HIGH |
| **RunPod** | FlashBoot (warm-state retention) | **p90 < 2 s**, **p95 < 2.3 s**, **min 563 ms**, **max 42 s**; "reduced cold-start costs for Whisper endpoint by more than 70%" | ⚠️ *"The more popular an endpoint is, the more likely FlashBoot will help."* **Requires consistent traffic.** Model = Whisper (size not stated). On by default for new endpoints. | runpod.io/blog/introducing-flashboot-serverless-cold-start | HIGH |
| **RunPod** | Billing during cold start | — | ⚠️ *"You're billed from when a worker starts until it fully stops, rounded up to the nearest second"* — **cold start is billed**. Default idle timeout 5 s. | docs.runpod.io/serverless/pricing | HIGH |
| **Baseten** | Baseten Delivery Network (BDN) | **2–3× faster** cold starts; **>2 GB/s** weight download to H100 nodes; **9 s** zero-to-inference for SDXL on A100 | Checkpoints "~10 GB to 100s of GB". *"Weight transfer doesn't consume billable GPU time."* | baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast | HIGH |
| **Cerebrium** | Container scale + snapshotting | *"scales containers up and down in 1–3 seconds"*; "memory and GPU snapshotting can restore workloads even faster" | **No model-loading numbers published** | cerebrium.ai/pricing | MEDIUM |
| **Beam** | Container/image load not billed | **No cold-start numbers published** | *"We don't charge for the time to spin up a server or load your container image"* — only app-code load is billed. Cost mitigation, not latency mitigation. | beam.cloud/pricing | MEDIUM |
| **Replicate** | — | **No cold-start numbers published** | Confirms private models are billed for *"the time they spend setting up"* | replicate.com/pricing | MEDIUM |
| **Fal** | — | **UNVERIFIED** — pricing page mentions neither cold starts nor scale-to-zero | — | fal.ai/pricing | **UNVERIFIED** |
| **Together** | — | **UNVERIFIED** — GPU clusters are reserved/on-demand instances, not serverless; cold start not applicable | — | together.ai/pricing | **UNVERIFIED** |
| **VoxCPM2 specifically** | vLLM-Omni init | **~60 s** | *"Cold-start initialization requires approximately ~60 s due to subprocess initialization, model loading, and CUDA graph capture."* | recipes.vllm.ai/openbmb/VoxCPM2 | HIGH |

### Which technique actually works — verdict

**Ranked, for a 2B TTS model specifically:**

1. **Keep-warm (`min_containers` / active workers / a rented pod).** The only technique that reliably delivers sub-second p95. Everything else is a *mitigation*. **This is the answer.**
2. **Weights local to the node** — baked into the image, on a network volume, or via a delivery network (Baseten BDN at >2 GB/s). Modal: "minutes to seconds" for tens-of-GB models. VoxCPM2's ~4.9 GB of weights at 2 GB/s is ~2.5 s of pure transfer, so this converts a download-bound cold start into a compute-bound one.
3. **Warm-state retention (RunPod FlashBoot).** Genuinely effective — p90 < 2 s — but **conditional on consistent traffic**, and RunPod says so plainly. A low-traffic public site in month one will not benefit. It also does not eliminate the 42 s tail.
4. **Memory / GPU snapshotting.** ⚠️ **Explicitly the wrong tool here.** Modal's own docs state it does not help — and can hurt — when initialization is weight-loading-bound. Their best published win (45 s → 5 s) was on vLLM with a **0.5B** model, where CUDA graph capture and Python import dominate. VoxCPM2's ~60 s init *is* substantially weight-load and CUDA-graph capture; the graph-capture portion would benefit, the weight-load portion would not. **Worth an experiment, not worth a plan.**

### The realistic floor for a 2B TTS model cold start in 2026

| Path | Realistic cold start | Confidence |
|---|---|---|
| True zero — pull image, download weights from HF, load, capture CUDA graphs | **60 s+** (vLLM-Omni's own published figure for VoxCPM2) | HIGH |
| Weights already node-local, full engine init | **~10–30 s** — CUDA graph capture and engine setup dominate | **MEDIUM (INTERPOLATED** — no one publishes this for VoxCPM2 specifically; bounded above by the 60 s figure and below by Baseten's 9 s SDXL-on-A100 datapoint for a comparable-sized model) |
| Warm-state retention (FlashBoot) on a busy endpoint | **~0.6–2.3 s** (p90 < 2 s, p95 < 2.3 s), with a 42 s tail | HIGH (for Whisper; **UNVERIFIED for a 2B TTS model**) |
| Genuinely warm container | **67–190 ms TTFB** | HIGH |

**Answer to D2: the floor from true zero is ~10 s, and nothing published gets a 2B TTS model below ~10 s without pre-warming.** Every sub-2 s number in this table is a warm-pool number wearing a cold-start costume.

---

## 4. GPU pricing table (fetched 2026-09-02)

`$/mo always-warm = $/hr × 730`.

| Provider | GPU | $/hr on-demand | **$/mo always-warm** | Notes | Source | Confidence |
|---|---|---:|---:|---|---|---|
| **RunPod** Community | **RTX 4090 (24 GB)** | **$0.34** | **$248.20** | ⭐ cheapest credible 24 GB. Peer-supplied — weaker SLA | runpod.io/pricing | HIGH |
| RunPod Community | A40 (48 GB) | $0.35 | $255.50 | best $/GB-VRAM on the board | same | HIGH |
| RunPod Community | L4 (24 GB) | $0.44 | $321.20 | datacenter part, low TDP | same | HIGH |
| RunPod Community | RTX 5090 (32 GB) | $0.69 | $503.70 | | same | HIGH |
| RunPod Community | L40S (48 GB) | $0.79 | $576.70 | | same | HIGH |
| RunPod Community | A100 PCIe (80 GB) | $1.19 | $868.70 | ⭐ fine-tune target | same | HIGH |
| RunPod Community | H100 PCIe (80 GB) | $1.99 | $1,452.70 | | same | HIGH |
| RunPod Secure | RTX 4090 | $0.74 | $540.20 | | same | HIGH |
| RunPod Secure | L4 | $0.49 | $357.70 | ⭐ cheapest *secure* 24 GB | same | HIGH |
| RunPod Secure | A40 / L40S / RTX 5090 | $0.44 / $0.99 / $0.99 | $321 / $723 / $723 | | same | HIGH |
| RunPod Secure | A100 80 GB / H100 PCIe | $1.39 / $2.89 | $1,015 / $2,110 | | same | HIGH |
| **Beam** | **RTX 4090** | **$0.42** | **$306.60** | + $30/mo free credits on Developer | beam.cloud/pricing | HIGH |
| Beam | RTX 5090 / A6000 / L40S | $0.68 / $0.51 / $0.72 | $496 / $372 / $526 | | same | HIGH |
| Beam | A100 80 GB SXM / H100 PCIe | $1.30 / $1.74 | $949 / $1,270 | | same | HIGH |
| **Modal** | L4 (24 GB) | $0.7992 (=$0.000222/s) | $583.42 | + $30/mo credits (Starter); Team $250/mo | modal.com/pricing | HIGH |
| Modal | A10 (24 GB) | $1.1016 | $804.17 | | same | HIGH |
| Modal | L40S / A100 40 GB / A100 80 GB | $1.9512 / $2.0988 / $2.4984 | $1,424 / $1,532 / $1,824 | | same | HIGH |
| Modal | H100 SXM5 / H200 / B200 | $3.9492 / $4.5396 / $6.2496 | $2,883 / $3,314 / $4,562 | | same | HIGH |
| **Cerebrium** | L4 / A10 | $0.7992 / $1.1016 | $583 / $804 | identical to Modal's card rates | cerebrium.ai/pricing | HIGH |
| Cerebrium | A100 40 / 80 GB, H100 | $1.998 / $2.0988 / $3.3984 | $1,459 / $1,532 / $2,481 | Standard plan $100/mo; Hobby free | same | HIGH |
| **Baseten** | L4 (24 GB) | $0.8484 (=$0.01414/min) | $619.33 | | baseten.co/pricing | HIGH |
| Baseten | A10G (24 GB) / A100 80 GB / H100 | $1.2072 / $4.0002 / $6.4998 | $881 / $2,920 / $4,745 | | same | HIGH |
| **Lambda** | Quadro RTX 6000 (24 GB) | $0.69 | $503.70 | cheapest Lambda 24 GB | lambda.ai/pricing | HIGH |
| Lambda | A10 (24 GB) / A6000 (48 GB) | $1.29 / $1.09 | $942 / $796 | | same | HIGH |
| Lambda | A100 40 GB PCIe/SXM · 80 GB SXM | $1.99 · $2.79 | $1,453 · $2,037 | ⭐ fine-tune reference | same | HIGH |
| Lambda | H100 SXM / GH200 / B200 | $3.99–4.29 / $2.29 / $6.69–6.99 | $2,913+ / $1,672 / $4,884+ | | same | HIGH |
| **Replicate** | T4 / L40S / A100 80 GB / H100 | $0.81 / $3.51 / $5.04 / $5.49 | $591 / $2,562 / $3,679 / $4,008 | **No 24 GB tier offered** | replicate.com/pricing | HIGH |
| **Fal** | H100 / H200 / RTX PRO 6000 | $4.50 / $4.50 / $2.99 list (disc. $1.89 / $2.10 / $1.10) | $3,285 / $3,285 / $2,183 | **No 24 GB or A100 tier listed** | fal.ai/pricing | HIGH |
| **Together** | H100 / H200 / B200 | $3.99 / $5.99 / $8.19 | $2,913 / $4,373 / $5,979 | reserved H100 $3.69→$3.19 (7d→181d+). **No 24 GB or A100 listed** | together.ai/pricing | HIGH |
| **Nebius** | L40S (AMD / Intel) | from $1.55 / $1.82 | $1,132 / $1,329 | **No 24 GB or A100 listed** | nebius.com/prices | HIGH |
| Nebius | HGX H100 / H200 | $3.85 / $4.50 | $2,811 / $3,285 | "up to 35% less" reserving clusters for months | same | HIGH |
| **CoreWeave** | L40S 8×48 GB / A100 8×80 GB / H100 8×80 GB | $18.00 / $21.60 / $49.24 per node → **$2.25 / $2.70 / $6.155 per GPU-hr** | $1,643 / $1,971 / $4,493 per GPU | **No 24 GB class**; 8-GPU nodes only; reserved "up to 60%" | coreweave.com/pricing | HIGH |
| **AWS** | g5.xlarge — A10G 24 GB | $1.006 | $734.38 | 1yr RI $0.654 → $477/mo; **3yr RI $0.435 → $317.55/mo** | instances.vantage.sh (AWS price-list mirror) | **MEDIUM** (third-party mirror) |
| AWS | g6e.xlarge — L40S 48 GB | $1.861 | $1,358.53 | 1yr $1.172 → $856; 3yr $0.804 → $587 | same | **MEDIUM** |
| **Vast.ai** | marketplace | **UNVERIFIED** — live rates render client-side | — | Publishes: per-second billing, no minimum; **interruptible "50%+ cheaper"**, **reserved "up to 50%" off** | vast.ai/pricing | **UNVERIFIED** |
| **GCP** | L4 / A100 / H100 | **UNVERIFIED** — pricing page truncated on fetch | — | — | cloud.google.com/compute/gpus-pricing | **UNVERIFIED** |
| **Azure** | NC/ND/NV | **UNVERIFIED** — page renders `$-` placeholders | — | — | azure.microsoft.com | **UNVERIFIED** |
| **Hyperbolic** | — | **UNVERIFIED** — pricing URL 404 on both `.xyz` and `.ai` | — | — | hyperbolic.ai | **UNVERIFIED** |

### Cheapest always-warm 24 GB-class card — ranked

| # | Option | $/mo | Reliability caveat |
|---|---|---:|---|
| 1 | **RunPod Community RTX 4090** | **$248.20** | Peer-supplied capacity. No SLA. Host can disappear. **Mitigate: checkpoint nothing to local disk; treat the pod as cattle.** |
| 2 | **Beam RTX 4090** | **$306.60** | Managed; $30/mo credits offset ~10% |
| 3 | AWS g5.xlarge, 3-yr reserved | $317.55 | Enterprise SLA — but a **3-year lock** is indefensible for a pre-revenue solo project |
| 4 | RunPod Community L4 | $321.20 | Datacenter part, more stable supply than 4090 |
| 5 | RunPod Secure L4 | $357.70 | ⭐ **Cheapest option with a real SLA.** Recommended once there are paying users. |

**Recommendation:** start on **RunPod Community RTX 4090 ($248/mo)** or, better, run it **warm only during active hours** — `$0.34 × 12 × 30 = $122.40/mo` — and move to **RunPod Secure L4 ($357.70/mo)** the moment a paying customer exists. The $109/mo premium buys an SLA and is trivially covered by two subscribers.

### Serverless per-second GPU rates

| Platform | 24 GB class | 48 GB class | 80 GB class | Billing note |
|---|---|---|---|---|
| **Beam** | RTX 4090 **$0.000191667/s** ($0.69/hr) | L40S $0.000486/s | A100 80 GB $0.000625/s | Container-image load **not billed** |
| **RunPod Serverless** | $0.69/hr (L4, A5000, 3090, MIG-24) | $1.75/hr | — | **Cold start IS billed**; 5 s default idle timeout |
| **Modal** | L4 $0.000222/s · A10 $0.000306/s | L40S $0.000542/s | A100 80 GB $0.000694/s | $30/mo free credits (Starter) |
| **Cerebrium** | L4 $0.000222/s · A10 $0.000306/s | L40s $0.000542/s | A100 80 GB $0.000583/s | Hobby tier free, 5 GPU concurrency |
| **Replicate** | *(none)* | L40S $0.000975/s | A100 80 GB $0.0014/s | Billed for setup + idle |
| **Baseten** | L4 $0.01414/min · A10G $0.02012/min | — | A100 $0.06667/min | "You do not pay for idle time" |

**Beam's RTX 4090 at $0.000191667/s is the cheapest serverless path to a 24 GB card found anywhere**, and it does not bill image load. It is the correct S7 choice.

---

## 5. Fine-tune event budget sanity check

**The scope doc estimates: description-conditioned fine-tune, 2–5 days on A100-class ≈ $70–250/run, ~3 runs.**

**The anchor (primary source — Parler-TTS `training/README.md`):**

> **"4 nodes of 8 H100 80GB to train Parler-TTS Mini"** — **"around 1.5 days"** — 880M params, 45,000 hours of annotated audio, 4 epochs, per-device batch 4, grad accum 6, 20,000 warmup steps.

**What that pretrain actually costs, at prices fetched today:**

```
Compute  = 4 nodes × 8 GPUs × 36 hours = 1,152 H100-hours

  RunPod Community H100 PCIe @ $1.99/hr : 1,152 × 1.99 = $2,292
  Together reserved H100      @ $3.19/hr : 1,152 × 3.19 = $3,675
  Lambda H100 SXM             @ $3.99/hr : 1,152 × 3.99 = $4,596
  Modal H100 SXM5             @ $3.9492/hr: 1,152 × 3.9492 = $4,550
```

**A from-scratch Parler-TTS Mini pretrain is a $2,300–4,600 event.** That is 20–50× the scope doc's per-run figure — but the scope doc is budgeting a **fine-tune**, not a pretrain, so this is a bound, not a contradiction.

**Now check the fine-tune figure directly.** The scope says 2–5 days on A100-class:

```
Single A100, 2 days  =  48 A100-hours
Single A100, 5 days  = 120 A100-hours

  RunPod Community A100 80GB @ $1.19/hr : $57  – $143   ← scope's $70–250 lands here
  RunPod Secure    A100 80GB @ $1.39/hr : $67  – $167
  Beam             A100 80GB @ $1.30/hr : $62  – $156
  Lambda           A100 80GB @ $2.79/hr : $134 – $335   ← scope's range breaks
  Modal            A100 80GB @ $2.4984/hr: $120 – $300  ← scope's range breaks
```

**Cross-check by data volume.** The pretrain processed 45,000 h × 4 epochs = 180,000 audio-hours in 1,152 H100-hours → **~156 audio-hours processed per H100-hour** (DERIVED). At 48–120 A100-hours (an A100 being materially slower than an H100 for this workload), you can plausibly process on the order of **3,000–8,000 audio-hours** — enough for a 1,000–2,500 h corpus at 3 epochs. For a description-conditioning fine-tune that is a sensible corpus size. **The scope's time estimate is internally consistent with its cost estimate.**

### Verdict: **NOT wildly off. Directionally correct with two conditions.**

| Condition | Status |
|---|---|
| **One** GPU, not a node of 8 | Required. The estimate silently assumes single-GPU. |
| **Community-cloud** pricing (RunPod/Beam/Vast) | Required. On Lambda/Modal/AWS the range breaks by 2–3×. |
| Fine-tune, not pretrain | Required. A pretrain is 20–50× more. |

**Recommended budget correction:** replace "$70–250/run × 3 runs = $210–750" with **$400–900 for the fine-tune programme**, covering: 3 successful runs at $60–170 each on RunPod Community A100, plus ~2 failed/aborted runs (realistic for a first-time training pipeline), plus checkpoint storage (a 0.9B model checkpoint set across runs is tens of GB — at RunPod network storage $0.05–0.07/GB/mo, negligible; on R2 at $0.015/GB/mo, also negligible).

**Also flag:** rent **A100 80 GB, not 40 GB**. The $0.20/hr difference at RunPod Community ($1.19 vs — 40 GB not listed) is irrelevant next to the risk of an OOM at hour 30 of a 5-day run.

---

## 6. Platform stack recommendations

### 6.1 Database & pgvector

**Current state (github.com/pgvector/pgvector, fetched 2026-09-02):**

| Property | Value |
|---|---|
| Version | **0.8.6** |
| `vector` type max dims | 16,000 |
| **HNSW max dims (`vector`)** | **2,000** |
| IVFFlat max dims | 2,000 |
| Binary quantization indexable | up to 64,000 dims |
| HNSW defaults | `m = 16`, `ef_construction = 64`, `ef_search = 40` (query-time) |
| Operators | L2 `<->` · inner product `<#>` · **cosine `<=>`** · L1 `<+>` · Hamming `<~>` · Jaccard `<%>` |
| `halfvec` | 2 bytes/dim + 8 bytes — halves index size |
| Build parallelism | `max_parallel_maintenance_workers`, default 2 |

**Is HNSW appropriate at our scale (thousands → low millions)? Yes, unambiguously.**

- **128-d is 15× under the 2,000-dim HNSW ceiling.** No dimensional gotcha whatsoever on the speaker column.
- Supabase's own guidance: **"HNSW should be your default choice when creating a vector index"**, and critically — *"you are safe to build an HNSW index immediately after the table is created"*, unlike IVFFlat which needs representative data present before building. For a catalog that grows from zero, **IVFFlat is the wrong index and HNSW is the right one**, for a structural reason rather than a performance one.
- Raw storage at 1M identities × 128-d: `1e6 × (128 × 4 + 8) = 520 MB`. Comfortable.
- pgvector notes indexes *"build significantly faster when the graph fits into `maintenance_work_mem`"* — set `maintenance_work_mem` generously before the build.

**⚠️ The real trap is the *other* column.** A 3072-d text embedding (e.g. OpenAI `text-embedding-3-large` at full width) **exceeds the 2,000-dim HNSW limit for `vector` and will not index.** Options: (a) request reduced dimensions (1536 or 1024) from the embedding provider; (b) use `halfvec`; (c) pick a natively ≤1024-d model. **Recommendation: use a 1024-d or 1536-d text embedding and stop thinking about it.**

**Cosine vs L2 for speaker embeddings.** Speaker-verification embeddings (x-vector/ECAPA lineage) are conventionally compared by **cosine similarity** — magnitude carries recording-condition information, not identity. Use `vector_cosine_ops` / `<=>`.

> **Engineering note (not a cited fact):** if you L2-normalise the speaker vectors on write, cosine distance and negative inner product induce **identical rankings**, and `<#>` is cheaper to compute. Normalising on write and indexing with `vector_ip_ops` is a free optimisation — but only if you enforce normalisation as an invariant (a `CHECK` constraint or a trigger), because a single unnormalised row silently corrupts ranking. **Recommendation: normalise on write, but index with `vector_cosine_ops` anyway** — the safety is worth more than the cycles at our scale.

**Index DDL:**
```sql
CREATE INDEX idx_identity_speaker ON identities
  USING hnsw (speaker_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_identity_desc ON identities
  USING hnsw (description_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

SET hnsw.ef_search = 100;  -- raise from default 40 for better recall on "voices like this"
```
Defaults `m=16 / ef_construction=64` are correct up to ~1M rows. Only raise `m` to 24–32 if measured recall is inadequate; it costs build time and index size.

**Supabase vs Neon — which better supports pgvector + HNSW today**

| | **Supabase** | **Neon** |
|---|---|---|
| Free | $0 · **500 MB DB** · 5 GB egress · 1 GB storage · ⚠️ **paused after 1 week inactivity** · max 2 projects | $0 · **0.5 GB storage/project** · 100 CU-hours · 5 GB egress · **scale-to-zero after 5 min** · autoscale to 2 CU |
| Paid entry | **Pro $25/mo** · 8 GB DB (+$0.125/GB) · 250 GB egress (+$0.09/GB) · 100 GB storage (+$0.0213/GB) · $10 compute credit | **Launch** PAYG · $0.106/CU-hr · $0.35/GB-mo storage · 500 GB egress (+$0.10/GB) · autoscale to 16 CU · scale-to-zero **disableable** |
| Next tier | Team $599/mo | Scale · $0.222/CU-hr · scale-to-zero configurable 1 min → always-on |
| pgvector + HNSW | ✅ Supported, with explicit HNSW guidance in their docs | ✅ Supported |
| **Managed queue** | ✅ **Supabase Queues — managed pgmq**, exactly-once within visibility window, RLS-controlled | ❌ none managed |
| Also bundles | Auth, object storage, edge functions | DB branching, snapshots |

**Recommendation: Supabase Pro ($25/mo) for production; Neon Free for throwaway dev branches.**

Three reasons, in order:
1. **Supabase Queues is managed pgmq** — it collapses §6.3 into a line of SQL and removes an entire service from the architecture. Neon has no equivalent.
2. Supabase bundles auth and object storage, which a public "anyone can create voices" site needs on day one.
3. **Both free tiers are disqualifying for production, for different reasons** — Supabase pauses after a week of inactivity (fatal for a public catalog with sparse early traffic); Neon's 5-minute scale-to-zero adds a DB cold start to the *first search of the session*, on top of the GPU cold start. Neon's Launch tier can disable scale-to-zero; that is the fix if you prefer Neon.

Neon's genuine edge is **branching** — a per-PR database branch is excellent for developing the mint/search path. Use it for dev, not prod.

### 6.2 Object storage & egress math

**Prices fetched 2026-09-02:**

| Provider | Storage | Egress | Operations | Free tier |
|---|---:|---:|---|---|
| **Cloudflare R2** | **$0.015/GB-mo** (IA $0.01) | **$0 — "R2 doesn't charge for egress"** | Class A $4.50/M · Class B $0.36/M | 10 GB-mo · 1M Class A · 10M Class B |
| **Backblaze B2** | $6.95/TB-mo = **$0.00695/GB-mo** | $0.01/GB beyond **free 3× stored data**; **unlimited free via Cloudflare/bunny/Fastly CDN** | Class A/B/C **free**; Class D $0.004/10k (2,500/day free) | No minimum retention |
| **Bunny Edge Storage** | HDD $0.01/GB (1 region) · SSD $0.02/GB/region | Free to Bunny CDN (CDN bandwidth billed separately) | **No API fees** | $1/mo minimum |
| **Hetzner Object Storage** | **UNVERIFIED** — page renders placeholders; structure shows a monthly base including **1 TB storage + 1 TB egress**, then per-TB overage | UNVERIFIED | — | — |

**Bytes per minute of audio (computed from codec definitions — arithmetic, not a fetched claim):**

| Format | Bitrate | **MB per minute** | Ratio vs 48k WAV |
|---|---|---:|---:|
| WAV 48 kHz / 16-bit / mono | 768 kbps | **5.76** | 1.0× |
| WAV 44.1 kHz / 16-bit / mono | 705.6 kbps | 5.29 | 0.92× |
| WAV 24 kHz / 16-bit / mono | 384 kbps | 2.88 | 0.50× |
| FLAC (≈50–60% of WAV) | — | ~2.9–3.5 | ~0.55× *(approximate)* |
| MP3 128 kbps | 128 kbps | 0.96 | 0.167× |
| MP3 64 kbps mono | 64 kbps | 0.48 | 0.083× |
| **Opus 32 kbps** | 32 kbps | **0.24** | **0.042×** |
| Opus 24 kbps | 24 kbps | 0.18 | 0.031× |

*(48,000 samples/s × 2 bytes × 60 s = 5,760,000 B = 5.76 MB, using GB = 10⁹ B to match cloud billing.)*

**Quantifying the "sleeper cost".** For **100,000 minutes served per month**:

| Format | Egress volume | **On R2** | On S3 @ $0.09/GB | On B2 @ $0.01/GB |
|---|---:|---:|---:|---:|
| WAV 48k mono | 576 GB | **$0.00** | $51.84 | $5.76 (likely $0 under 3× free) |
| MP3 128k | 96 GB | **$0.00** | $8.64 | $0.96 |
| Opus 32k | 24 GB | **$0.00** | $2.16 | $0.24 |

**Storage** for a 100,000-minute library:

| Format | Volume | R2 @ $0.015/GB-mo | B2 @ $0.00695/GB-mo |
|---|---:|---:|---:|
| WAV 48k masters | 576 GB | **$8.64/mo** | $4.00/mo |
| Opus 32k derivatives | 24 GB | **$0.36/mo** | $0.17/mo |

**Verdict on the brief's egress claim.** "Audio egress is the sleeper cost of any TTS product" is **true on S3/GCS/Azure and false on R2** — and the brief already chose R2. Egress is a **$0 line item**. What remains is a 24× storage-and-bandwidth lever from format choice, which costs single-digit dollars either way at our scale.

**Recommendation — store one thing, serve another:**
- **Store masters as 48 kHz FLAC** (lossless, ~45% smaller than WAV). VoxCPM2 emits 48 kHz natively; keep it. Masters exist to re-derive any future format without re-paying GPU.
- **Serve Opus 32 kbps** for previews and web playback — 24× smaller than WAV, universally supported in browsers via WebM/Ogg.
- **Serve 48 kHz WAV/FLAC only on explicit download**, for users pulling assets into a game engine.
- R2 Class A (writes) at $4.50/M: 1M renders/month = $4.50. Class B (reads) at $0.36/M: 10M plays = $3.60. Both negligible.

### 6.3 Job queue

**Recommendation: Supabase Queues (managed pgmq) + a plain Python worker loop on the GPU box.**

| Option | Verdict for a solo dev on a small budget |
|---|---|
| **Supabase Queues / pgmq** | ✅ **Recommended.** Already in this dev environment. Zero new infrastructure, zero new bill. PostgreSQL licence. Guaranteed delivery, **exactly-once within a visibility window**, archival, RLS-controlled access. Crucially: **enqueue the render job in the same transaction that writes the render row** — no dual-write, no orphaned jobs, no reconciliation cron. That property alone is worth more than any throughput advantage Redis offers at our scale. |
| Celery + Redis | ❌ Adds a Redis to run, secure, monitor and pay for. Celery's operational surface (result backends, visibility timeouts, prefetch semantics) is famously the source of subtle bugs. Not justified for one queue and one worker pool. |
| RQ / Dramatiq | ❌ Simpler than Celery, but still requires Redis. Same objection, less payoff. |
| Temporal | ❌ Correct answer to a problem we don't have. Render is a single-step job with a retry, not a durable multi-step workflow. Self-hosting Temporal is a bigger job than the render pipeline. |
| **River** | ❌ **Go-only.** Confirmed: River workers are Go; Python can *insert* jobs but not work them. Wrong language for a FastAPI/PyTorch stack. |
| SQS / managed cloud queue | ⚠️ Works, costs almost nothing, but reintroduces the dual-write problem and a second vendor for no benefit over pgmq. |

**Caveat, stated honestly:** **pgmq publishes no throughput benchmark** — I looked, and the repository contains none. **UNVERIFIED.** At VoiceForge's scale (tens to low thousands of jobs/minute) Postgres will not be the bottleneck — the GPU will be, by three orders of magnitude. But do not repeat an unmeasured claim; §9 gives the experiment.

**Shape:**
```
FastAPI (CPU) --[same txn]--> renders row + pgmq.send()
GPU worker: pgmq.read(vt=300) -> render -> upload to R2 -> update row -> pgmq.delete()
Crash/timeout -> visibility timeout expires -> message redelivered -> retried
N failures -> pgmq.archive() -> dead-letter table -> alert
```
Set the visibility timeout to **~5× the p99 render time**, not to a round number. Batch renders should `pgmq.read_batch()` so one worker pull covers many lines — this is what turns the cold-start amortisation in §2.3c from catastrophic to tolerable.

### 6.4 Render cache design

**Key.** The brief proposes `(identity_id, backend_version, text, direction)`. That is **almost right** — it is missing the output-format dimension, and it needs an explicit normalisation rule:

```
cache_key = sha256(
    identity_id      ‖ 0x1F ‖
    backend_version  ‖ 0x1F ‖   -- MUST include engine version, not just weights
    normalize(text)  ‖ 0x1F ‖
    direction        ‖ 0x1F ‖
    sample_rate      ‖ 0x1F ‖
    format           ‖ 0x1F ‖
    seed                        -- omit only if generation is deterministic
)
```

- **`backend_version` must cover the serving engine, not just the checkpoint.** Nano-vLLM and vLLM-Omni produce different audio from identical weights (different batching, different diffusion scheduling). Version the tuple `(weights_sha, engine, engine_version, precision)`.
- **`normalize(text)`**: Unicode NFC, collapse runs of whitespace, strip leading/trailing. **Do NOT lowercase** — casing is prosodically meaningful to TTS models and lowercasing would silently merge two distinct renders.
- **`seed`**: VoxCPM2 is diffusion-based and therefore stochastic. Either pin the seed per `(identity, text)` so renders are reproducible and cacheable, or include the seed in the key and accept a lower hit rate. **Pin it** — reproducibility is worth more to a game-dialogue workflow than variety, and users who want variety can ask for a re-roll explicitly.

**Storage.** Two-level, no third:
1. **Postgres** `render_cache(cache_key bytea PRIMARY KEY, identity_id, r2_key, duration_ms, bytes, format, created_at, last_hit_at, hit_count)` — the lookup, joinable and analysable.
2. **R2** holds the bytes, at `renders/{cache_key_hex[:2]}/{cache_key_hex}.{ext}` — the two-character prefix shard keeps listings sane.

Do **not** add Redis in front. A Postgres primary-key lookup is sub-millisecond and you already pay for Postgres; the GPU is 400,000× slower than either.

**Realistic hit rate: UNVERIFIED — no published TTS cache hit-rate data exists.** I searched and found none. Rather than invent one, here is the reasoning a design should rest on: in iterative game-dialogue work the same *line* is re-rendered whenever a *neighbouring* line, the voice, or the delivery direction changes, so the hit rate is governed by **edit locality**, not by text repetition. Two structural consequences:
- A **voice change invalidates every line for that character at once** — hit rate collapses to ~0 on exactly the operation users perform most while auditioning voices. This is the dominant failure mode and it argues for making *voice preview* cheap and *bulk re-render* explicit.
- A **text edit to one line invalidates only that line** — hit rate should be very high during script polish.

**Instrument `hit_count` and `last_hit_at` from day one and measure it.** Do not size infrastructure against an assumed number. And per §2.5, remember the cache is buying **p95 latency and queue depth**, not GPU dollars.

---

## 7. The recommended deployment for each stage

| Stage | Deployment | Monthly $ |
|---|---|---:|
| **S6 — local dev** | One 8–12 GB consumer GPU. ⚠️ vLLM-Omni's VoxCPM2 recipe wants **≥24 GB** (4.9 GB weights + 15.2 GB KV + 2 GB talker ≈ 22 GB). On 8–12 GB run **VoxCPM-0.6B** (~5 GB VRAM, RTF 0.10 w/ Nano-vLLM) or VoxCPM2 in plain PyTorch (~8 GB) with reduced `gpu_memory_utilization`. Postgres+pgvector in Docker. Local filesystem for audio. **Do all mapper/CPU work here — it is free.** | **$0** |
| **S7 — first paid GPU** | **Beam RTX 4090 serverless** ($0.000191667/s, image load not billed, $30/mo credits) or RunPod Serverless 24 GB ($0.69/hr). Scale-to-zero. **Batch-only — accept 30–60 s cold start; do not promise interactive preview yet.** Supabase Free. R2 free tier. | **$5–30** |
| **S8 — public launch** | **1× RunPod Community RTX 4090 warm 12 h/day** = $122.40 (or 24/7 = $248.20). **Supabase Pro** $25. **R2** ~$1–5. FastAPI on a $5–10 CPU VPS or Supabase Edge Functions. Serverless burst overflow for out-of-hours. | **$155–290** |
| **S9 — scale** | 2–4× RTX 4090 (Community $496–993) **or** migrate to **RunPod Secure L4** for SLA ($357.70 each). Supabase Pro + compute add-on. R2 $10–30. Second region if latency demands it. | **$500–1,200** |

**The single most important budget line:** the step from S7 to S8 is where a fixed $122–248/mo GPU floor appears, and it appears **because of latency, not throughput**. If VoiceForge can live with batch-only rendering, S7 economics extend a long way — well past the point where the product has revenue.

---

## 8. What this means for the build

1. **Build the adapter interface around batched, multi-voice requests from day one.** The throughput numbers only materialise at concurrency 16–32 (90–113 audio-s/s vs 9.7 at c=1 — an **11× difference**). An adapter that renders one line per call leaves 90% of the card on the floor. `render(lines: list[Line]) -> list[Audio]` is the signature; single-line is the degenerate case.

2. **Choose the vector identity (Tier-1) over per-voice LoRA wherever possible.** The published LoRA benchmark caps at **32 resident slots** and costs ~30% throughput at low concurrency. A conditioning vector batches across unlimited distinct voices for free. This is a serving argument for the brief's core architecture, independent of the quality argument.

3. **Ship VoxCPM2 first and treat Zonos as a fallback.** VoxCPM2 is Apache-2.0, has two independent production serving engines (Nano-vLLM MIT, vLLM-Omni Apache-2.0), publishes real concurrency benchmarks, natively supports **description-conditioned voice creation** — the actual product — and emits 48 kHz. Zonos is Apache-2.0 and ~4× slower under the honest RTF conversion, with **no published batching story**. Indic Parler-TTS publishes **no speed numbers at all** and must be benchmarked before it is committed to.

4. **Enforce `public_servable` in the adapter, as the brief demands.** All four backends surveyed are Apache-2.0 with no acceptable-use rider (VoxCPM2 LICENSE verified directly), so all four pass — but the gate must exist in code before the first non-Apache backend is tempting. See [08-licensing-propagation.md](08-licensing-propagation.md).

5. **Design two render paths with different SLAs, and say so in the UI.** *Interactive preview* (warm worker, p95 ≤ 1.5 s) and *batch render* (serverless, minutes, cheap). Do not try to serve both from one pool — the cold-start arithmetic in §2.3c makes a single pool either expensive or slow.

6. **Do not build render metering as the revenue mechanism.** At $0.0002/min marginal it recovers nothing. Price a subscription against the ~$250/mo warm-GPU floor; meter to cap abuse.

7. **Keep the mint path entirely off the GPU for Tier-1.** The brief is right that this is nearly free — and §2.3c shows Tier-2's one GPU render costs **$0.00007**. Neither is a cost problem. The Tier-2 problem is that a mint on a cold serverless worker takes 30–60 s, which is a *product* failure. Route Tier-2 mints to the warm pool.

8. **Set `hnsw.ef_search` explicitly** (100, not the default 40) for "find voices like this one". Recall matters more than latency on a search that returns a dozen rows.

9. **Store 48 kHz FLAC masters, serve 32 kbps Opus.** 24× on bytes, and it costs one line of ffmpeg.

10. **Instrument three counters before launch:** GPU duty cycle, cache hit rate, and p95 render latency split by warm/cold. Every open question in §9 is answered by these three numbers, and none of them can be answered by more reading.

---

## 9. Open — must be settled by experiment

| Question | Cheapest experiment | Est. cost / time | What it blocks |
|---|---|---|---|
| **Real audio-s/s for VoxCPM2 on a 4090 at short-utterance lengths** — sources disagree 3.2× (105.7 derived vs 33.07 published) | Run the benchmark script below on a rented RunPod Community 4090 | **$0.34 × 2 h ≈ $0.70**, 2 h | All of §2.3; GPU sizing for S8/S9 |
| **What is "short prompt" in the nanovllm benchmark?** (undefined — the whole §2.2 model rests on it) | Same run; log actual audio durations | included above | Confidence in §2.2 |
| **VoxCPM2 cold start with node-local weights** (interpolated 10–30 s, unpublished) | Time engine init 10× on a fresh pod with weights pre-staged on a network volume | $0.70, 1 h | Whether serverless interactive is viable at all |
| **Does Modal GPU snapshotting help VoxCPM2?** Modal warns it may not for weight-bound init | Deploy VoxCPM2 to Modal with `enable_gpu_snapshot=True`, measure P0 both ways | **$0 (within $30 Starter credits)**, 3 h | S7 platform choice |
| **RunPod FlashBoot p95 for a 2B TTS model** (published figure is Whisper only) | Deploy to RunPod Serverless, drive 200 spaced requests, read the cold-start histogram | ~$2, 2 h | Whether S7 can serve interactive |
| **Indic Parler-TTS throughput** — no published numbers anywhere | Same benchmark script, `--model ai4bharat/indic-parler-tts` | $0.70, 2 h | The entire Indic track's cost model — see [04-indic-track.md](04-indic-track.md) |
| **Zonos batching** — unverified whether it batches at all | Same script, `--model Zyphra/Zonos-v0.1-transformer`, sweep concurrency | $0.70, 2 h | Whether Zonos is a viable fallback |
| **Render cache hit rate** — no published data for any TTS product | Instrument `hit_count` / `last_hit_at` from launch; report weekly | $0, ongoing | §6.4 sizing; whether cache is worth its complexity |
| **pgmq throughput on Supabase Pro** — unpublished | `pgbench` script: 10k enqueue + 10k read/delete, measure ops/s | $0, 1 h | Whether the queue ever needs replacing |
| **HNSW index size and build time at 1M × 128-d** — unpublished | Generate 1M random 128-d vectors, build, `SELECT pg_relation_size()` | $0, 1 h | Supabase tier sizing |
| **Google Cloud TTS and Azure TTS list prices** — pages would not render | Open both pricing pages in a browser and read the tables | $0, 10 min | Completing §2.4 |

### Benchmark script spec — measuring $/min on short lines

The one experiment that settles §2.3. Write it as `bench/tts_throughput.py`.

**Inputs**
- `--model` HF id · `--engine {nanovllm,vllm-omni,pytorch}` · `--gpu` label for the report
- `--concurrency 1,4,8,16,32,64` (sweep)
- `--corpus` a JSONL of **real game dialogue lines**, not Lorem Ipsum, bucketed by word count: **3–5, 6–10, 11–15, 16–30 words**, ≥200 lines per bucket
- `--voices N` distinct identities per batch (sweep **1, 8, 32, 64** — this is what exposes the LoRA-slot ceiling)
- `--duration 300` seconds of steady-state load per cell, after a **60 s discarded warmup**
- `--price-per-hour` the actual rented $/hr

**Procedure**
1. Warm up: 60 s at the target concurrency; **discard**.
2. For each `(bucket, concurrency, voices)` cell: drive a **closed-loop** load generator holding exactly `concurrency` requests in flight for 300 s.
3. Per request record: `t_submit`, `t_first_chunk` (TTFB), `t_complete`, `audio_duration_s` (decode the output — do **not** trust a predicted length), `text_word_count`, `identity_id`.
4. Sample `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1` throughout.

**Required outputs — per cell**
```
audio_seconds_per_gpu_second = Σ audio_duration_s ÷ wall_clock_s      ← THE number
minutes_of_audio_per_gpu_hour = audio_seconds_per_gpu_second × 60
cost_per_minute_of_audio      = price_per_hour ÷ minutes_of_audio_per_gpu_hour
rtf_per_request               = mean((t_complete − t_first_chunk) ÷ audio_duration_s)
ttfb_p50, ttfb_p95, ttfb_p99
e2e_p50, e2e_p95, e2e_p99
gpu_util_mean, vram_peak_gb
requests_per_second, error_rate
```

**Report format:** one CSV row per cell, plus a plot of `cost_per_minute_of_audio` vs concurrency, one line per word-count bucket.

**Acceptance criteria — the run has failed if it does not answer these:**
- Does `audio_seconds_per_gpu_second` for the **3–5 word** bucket land nearer 33 or nearer 105 on a 4090?
- At what concurrency does `ttfb_p95` exceed **500 ms**? *(That concurrency is the ceiling for the interactive pool.)*
- Does throughput degrade when `--voices` rises from 1 → 64? *(If yes, the identity mechanism is batch-hostile and §8.2 becomes mandatory rather than advisory.)*
- What is `cost_per_minute_of_audio` at the knee of the curve?

**Total cost to run the full sweep across all four models: under $5 and one afternoon.** There is no excuse for shipping the cost model unmeasured.

---

## 10. Sources

| # | URL | Type | Used for | Confidence in source |
|---|---|---|---|---|
| 1 | https://github.com/OpenBMB/VoxCPM (README) | Model repo (official) | VoxCPM2/1.5/0.5B RTF, VRAM, params, Apache-2.0 | HIGH |
| 2 | https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE | Licence file (official) | Apache-2.0 confirmed, no rider | HIGH |
| 3 | https://arxiv.org/html/2606.06928v1 | Paper (VoxCPM2 technical report) | RTF 0.30/0.13, params, serving engines; confirms utterance length **not stated** | HIGH |
| 4 | https://github.com/a710128/nanovllm-voxcpm (README) | Serving-engine repo (MIT) | **The concurrency benchmark tables** — TTFB + RTF/req, short vs long, ±LoRA, RTX 4090 | HIGH |
| 5 | https://recipes.vllm.ai/openbmb/VoxCPM2 | Official vLLM recipe | ≥24 GB requirement, VRAM breakdown, RTF ~0.12, **~60 s cold start**, `max_num_seqs: 4` | HIGH |
| 6 | https://vllm.ai/blog/2026-06-23-vllm-omni-tts | Vendor engineering blog (own platform) | **VoxCPM2 33.07 audio-s/s @ 10.83 req/s on H20 → 3.05 s/req**; length-independent RTF 0.132–0.138 | HIGH |
| 7 | https://github.com/vllm-project/vllm-omni | Framework repo | Apache-2.0, streaming, OpenAI-compatible API | HIGH |
| 8 | https://voxcpm.readthedocs.io/en/latest/deployment/nanovllm_voxcpm.html | Official docs | Scheduler config, `max_num_seqs`, "not production-ready" warning | HIGH |
| 9 | https://github.com/Zyphra/Zonos (README) | Model repo (official) | "RTF ~2×", 6 GB+ VRAM, Apache-2.0 | HIGH |
| 10 | https://www.zyphra.com/our-work/beta-release-of-zonos-v0-1 | Vendor blog (own model) | 200–300 ms latency, 1.6B×2, 774 tokens/s-of-audio | HIGH |
| 11 | https://huggingface.co/ai4bharat/indic-parler-tts | Model card (official) | 0.9B, Apache-2.0, 21 languages, 69 voices, **no speed numbers** | HIGH |
| 12 | https://huggingface.co/parler-tts/parler-tts-mini-v1 | Model card (official) | 0.9B, Apache-2.0, 45k h | HIGH |
| 13 | https://github.com/huggingface/parler-tts/blob/main/training/README.md | Training docs (official) | **4 nodes × 8 H100 80GB, ~1.5 days** — the fine-tune sanity anchor | HIGH |
| 14 | https://www.runpod.io/pricing | Official pricing | 4090 $0.34/$0.74, L4, A40, A100, H100, serverless, storage | HIGH |
| 15 | https://docs.runpod.io/serverless/pricing | Official docs | **Cold start is billed**; 5 s idle timeout | HIGH |
| 16 | https://www.runpod.io/blog/introducing-flashboot-serverless-cold-start | Vendor blog (own platform) | FlashBoot p90 <2 s, p95 <2.3 s, min 563 ms, max 42 s, >70% cost cut | HIGH |
| 17 | https://modal.com/pricing | Official pricing | Per-second GPU rates, storage, plan tiers, $30 credits | HIGH |
| 18 | https://modal.com/docs/guide/memory-snapshot | Official docs | **"will generally not improve… may even worsen"** for weight-bound init; alpha status | HIGH |
| 19 | https://modal.com/blog/gpu-mem-snapshots | Vendor blog (own platform) | 20→2 s Parakeet, 45→5 s vLLM Qwen2.5-0.5B, 8.5→2.25 s ViT | HIGH |
| 20 | https://modal.com/docs/guide/cold-start | Official docs | ~1 s container boot; "minutes to seconds" for tens-of-GB; `min_containers` | HIGH |
| 21 | https://www.baseten.co/pricing/ | Official pricing | Per-minute GPU rates | HIGH |
| 22 | https://www.baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast/ | Vendor blog (own platform) | 2–3× faster, >2 GB/s, 9 s SDXL on A100 | HIGH |
| 23 | https://www.beam.cloud/pricing | Official pricing | **RTX 4090 $0.42/hr, $0.000191667/s**; image load not billed | HIGH |
| 24 | https://www.cerebrium.ai/pricing | Official pricing | Per-second rates; "1–3 s" container scaling | HIGH |
| 25 | https://replicate.com/pricing | Official pricing | Hardware rates; no 24 GB tier; setup billed | HIGH |
| 26 | https://lambda.ai/pricing | Official pricing | A100 $1.99/$2.79, H100 $3.99–4.29, RTX 6000 $0.69 | HIGH |
| 27 | https://www.together.ai/pricing | Official pricing | H100 $3.99 on-demand, reserved $3.69→$3.19 | HIGH |
| 28 | https://nebius.com/prices | Official pricing | H100 $3.85, L40S from $1.55; "up to 35%" reserved | HIGH |
| 29 | https://www.coreweave.com/pricing | Official pricing | Node rates → per-GPU derivation; "up to 60%" reserved | HIGH |
| 30 | https://fal.ai/pricing | Official pricing | H100/H200/B200 rates; no 24 GB tier | HIGH |
| 31 | https://vast.ai/pricing | Official pricing | Per-second billing, interruptible "50%+", reserved "up to 50%" — **rates UNVERIFIED** | MEDIUM |
| 32 | https://instances.vantage.sh/aws/ec2/g5.xlarge · /g6e.xlarge | **Third-party mirror of the AWS public price list** | g5.xlarge $1.006 / RI $0.654 / $0.435; g6e.xlarge $1.861 | **MEDIUM — not AWS's own page** |
| 33 | https://elevenlabs.io/pricing/api | Official pricing | $0.10 & $0.05 per 1k chars; plan tiers | HIGH |
| 34 | https://cartesia.ai/pricing | Official pricing | Plan tiers + minute equivalents → **750 credits/min** | HIGH |
| 35 | https://developers.openai.com/api/docs/pricing | Official pricing | tts-1 $15/1M, tts-1-hd $30/1M, gpt-4o-mini-tts $12/1M tokens | HIGH |
| 36 | https://www.deepgram.com/pricing | Official pricing | Aura-1 $0.0150/1k, Aura-2 $0.030/1k, Growth rates | HIGH |
| 37 | https://docs.sarvam.ai/api-reference-docs/pricing | Official docs | Bulbul v3 ₹30/10k chars | HIGH |
| 38 | https://cloud.google.com/text-to-speech/pricing | Official pricing | **Would not render — UNVERIFIED** | — |
| 39 | https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/ | Official pricing | Free tier 0.5M chars; **prices are `$-` placeholders — UNVERIFIED** | — |
| 40 | https://developers.cloudflare.com/r2/pricing/ | Official docs | $0.015/GB-mo, Class A $4.50/M, Class B $0.36/M, **zero egress**, free tier | HIGH |
| 41 | https://www.cloudflare.com/developer-platform/products/r2/ | Official product page | "R2 doesn't charge for egress" | HIGH |
| 42 | https://www.backblaze.com/cloud-storage/pricing | Official pricing | $6.95/TB-mo, 3× free egress, free API calls | HIGH |
| 43 | https://bunny.net/pricing/storage/ | Official pricing | $0.01–0.02/GB, no API fees, $1 minimum | HIGH |
| 44 | https://www.hetzner.com/storage/object-storage/ | Official product page | Structure only — **prices UNVERIFIED (placeholders)** | — |
| 45 | https://github.com/pgvector/pgvector | Extension repo (official) | v0.8.6, **HNSW 2,000-dim cap**, m/ef defaults, operators, halfvec | HIGH |
| 46 | https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes | Official docs | "HNSW should be your default"; safe to build on an empty table | HIGH |
| 47 | https://supabase.com/pricing | Official pricing | Free/Pro/Team; **1-week inactivity pause**; egress and storage overages | HIGH |
| 48 | https://neon.com/pricing | Official pricing | Free/Launch/Scale; 5-min scale-to-zero; $0.106/$0.222 per CU-hr | HIGH |
| 49 | https://supabase.com/docs/guides/queues | Official docs | Supabase Queues **is managed pgmq**; exactly-once within visibility window | HIGH |
| 50 | https://github.com/tembo-io/pgmq | Extension repo (official) | SQS-parity API, visibility-timeout semantics, PostgreSQL licence; **no published throughput** | HIGH |
| 51 | https://github.com/riverqueue/river | Job-queue repo (official) | Go + Postgres, MPL-2.0, **Python can insert only** | HIGH |

**Source-type ledger:** 43 of 51 are official vendor pricing pages, official docs, model repos, licence files, or the models' own papers. 5 are vendor engineering blogs reporting numbers for their own platform (#6, #10, #16, #19, #22) — admissible under the stated standard. 1 is a third-party mirror of a public price list, flagged MEDIUM (#32). 3 are official pages that failed to render and are recorded as UNVERIFIED (#38, #39, #44). **No third-party listicle, comparison blog, or "best GPU cloud" article was used for any number in this document.**
