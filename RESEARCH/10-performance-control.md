# 10 — Per-Line Performance Control & Timbre/Style Separation

> **Domain:** the `Direction` channel; emotion without identity drift; duration & emphasis control
> **Answers:** B5; scope section 4.4
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [02-identity-representation.md](02-identity-representation.md) · [03-tts-backends-english.md](03-tts-backends-english.md) · [04-indic-track.md](04-indic-track.md) · [06-evaluation-harness.md](06-evaluation-harness.md) · [08-licensing-propagation.md](08-licensing-propagation.md)

## 0. Bottom line

- **IndexTTS2's disentanglement is NOT transferable.** It is enforced at *training time* by a Gradient Reversal Layer plus a speaker classifier inside Stage 2 of a three-stage T2S training run, with the speaker perceiver conditioner frozen and the emotion perceiver conditioner trainable, on a curated 135-hour emotional subset. There is no inference-time artefact to lift out. Porting it means training an acoustic model, which is out of scope. **HIGH** — [arXiv 2506.21619v2 §Proposed Method](https://arxiv.org/html/2506.21619v2).
- **But the goal it achieves is reachable on a frozen backend by a different route: activation steering.** Two 2025–2026 papers (EmoSteer-TTS, CoCoEmo) inject learned direction vectors into intermediate activations via forward hooks, no retraining. CoCoEmo on CosyVoice2 raises emotion similarity from 0.743 → 0.779 while speaker similarity moves 0.871 → 0.870 (WavLM cosine) — **essentially zero identity drift**, versus −0.018 for the model's own text-instruction channel. **HIGH** — [arXiv 2602.03420v2 Table 2](https://arxiv.org/html/2602.03420v2).
- **The brief's licence citation is wrong in form, right in substance.** The "improve any AI model" clause is **§3.4(c)**, not §2(c). §2.2 is the 100M-MAU / RMB-1bn threshold. §3.4(c) forbids using IndexTTS2 *or its outputs* to improve any AI model except IndexTTS2 itself, its derivatives, or **non-commercial** models. VoiceForge's mapper is commercial. **Verdict: IndexTTS2 must never touch mapper training, data generation, distillation, or any feedback loop.** **HIGH** — LICENSE fetched from repo root.
- **Our primary backend has no Direction channel *in its API*.** `Qwen3TTSModel.generate_voice_clone()` takes no `instruct` parameter and never passes `instruct_ids` to the underlying model; only `generate_voice_design()` and `generate_custom_voice()` do. The speaker-embedding path and the instruction path live in **different checkpoints** — `speaker_encoder_config` is present only in the Base config and absent from VoiceDesign and CustomVoice. Qwen's own InstructTTSEval table has no Base row. **HIGH** — source read.
- **But a training-free Direction channel exists *inside the x-vector itself*, and someone has already published it on our exact checkpoint.** arXiv 2606.05367 runs a four-operand elimination study on **Qwen3-TTS-12Hz-1.7B-Base** and localises emotion to the `(2048,)` ECAPA x-vector. Centroid arithmetic `x_new = x(target, neutral) + α·τ_emo` gives **ΔEECS +0.29** at **SECS_W 0.912** — provided τ is averaged over ≥4 source speakers (a single-speaker τ scores 0.810, leaking the source's timbre). **This is the answer to Qwen3-TTS Base's missing Direction channel, and it costs an afternoon of numpy.** **HIGH** — paper read in full.
- **The brief's "emotion must NOT be baked into the identity vector" is the right *goal* but a false *description* of this backend.** On Qwen3-TTS Base emotion **is** in the x-vector and is its **dominant** carrier: swapping all codec tokens of an angry utterance onto a neutral x-vector yields speech "indistinguishable from the neutral baseline". The saving grace is geometric — `‖τ_emo‖` is 15% of the x-vector norm and its projection onto the identity axis is **<1% of ‖τ‖**. Emotion is separable by *subtraction*, not absent. **HIGH**.
- **Kokoro-82M has an undocumented, free timbre/prosody split.** `KModel.forward_with_tokens` sends `ref_s[:, 128:]` to the duration/F0/energy predictor and `ref_s[:, :128]` to the decoder. The 256-dim voice vector is literally `[timbre(128) | prosody(128)]`. Holding the first half fixed and swapping the second half is a real, zero-cost Direction channel. **HIGH** on the code split; **UNVERIFIED** that it is perceptually clean.
- **The one-channel problem is fatal for Indic Parler-TTS and structural for VoxCPM2.** Parler-TTS has *zero* speaker-embedding code (`grep` for `speaker_embed|spk_emb|x_vector|d_vector` → 0 hits); identity is a name token inside the same description string as the emotion word, entering by the same cross-attention. VoxCPM2's style control is a parenthesised prefix inside `text` — the *same syntax* as its voice design. **HIGH** — source read.
- **The Indic track has a better option than Parler.** Indic-Mio + MioCodec put identity in a decoder-side `global_embedding` and performance in text tags — genuinely separate channels — and it is the **only backend in this audit with documented word-level emphasis by default** (`*word*`), plus `<happy>`/`<whisper>`/`<angry>` tags. Apache-2.0 / MIT. **HIGH** on documentation; **UNVERIFIED** on reliability.
- **Precise duration control exists on frozen backends, but not on the ones we favour.** F5-TTS `fix_duration` (seconds → frames) and MOSS-TTS v1.5 `tokens=N` are the two exact mechanisms. IndexTTS-2's celebrated <0.07% token-error duration control is **explicitly not enabled in the released code**; IndexTTS-2.5 replaced it with a `duration_factor` ratio (0.5–2.0). **HIGH**.
- **Word-level emphasis is nearly absent.** Only CosyVoice2/3 (`<strong></strong>` special tokens) and Indic-Mio (`*word*`) expose it. Neither publishes an evaluation of it. Everything else: no. **HIGH** on availability, **UNVERIFIED** on reliability.
- **Where you steer decides whether identity survives — this is now measured, not theorised.** On CosyVoice2, steering the flow-matching decoder costs **−0.064 S-SIM** at α=2; steering the speech-LM costs **≈0.000** at α=5 and gives *better* control. Mechanism: the SLM is not speaker-conditioned, the flow decoder is. Joint steering is worse than either alone. **HIGH** — [arXiv 2607.00946 Table 2](https://arxiv.org/abs/2607.00946).
- **Budget ~10% speaker-similarity loss for expressive delivery on any current system, and up to 45% on a bad one.** PilotTTS's head-to-head: turning emotion control on costs IndexTTS-v1 −44.9%, VoxCPM −32.5%, CosyVoice3 −12.9%, best-in-class −9.5%. Only Fish-Speech S2 is flat, and that is because its emotion control barely does anything. **HIGH** — [arXiv 2605.27258 Table 3](https://arxiv.org/abs/2605.27258).
- **There is a successor.** IndexTTS-2.5 shipped 2026-08-10 (arXiv 2601.03888, v5 2026-08-11) under the same bilibili licence. No IndexTTS-3. **HIGH**.

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| 1 | IndexTTS-2's `§2(c)` bars using it "to improve any AI model" | **Wrong citation, right substance** | The clause is **§3.4(c)**, under "Additional Obligations for You and Downstream Recipients". §2 is "Scope of License and Restrictions" and has only §2.1/2.2/2.3; §2.2 is the 100M-MAU / RMB-1bn threshold. | [LICENSE](https://raw.githubusercontent.com/index-tts/index-tts/main/INDEX_MODEL_LICENSE) → 404; actual file at `/LICENSE` | HIGH |
| 2 | Voice design lives in `Qwen3-TTS-1.7B-VoiceDesign` | **Wrong repo id** | Correct id is **`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`**. The un-prefixed name 401s. | [HF model index](https://huggingface.co/api/models?search=Qwen3-TTS) | HIGH |
| 3 | VoiceDesign has `speaker_encoder_config: null` | **Nearly right** | The key is **absent entirely** from the VoiceDesign and CustomVoice `config.json`. Only Base has it (`{"enc_dim": 2048, "sample_rate": 24000}`). Same conclusion, cleaner evidence. | [Base config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/config.json), [VD config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/raw/main/config.json) | HIGH |
| 4 | (implied) IndexTTS-2 is the current bilibili model | **Superseded** | **IndexTTS-2.5** released 2026-08-10: Chinese/English/Japanese/Spanish/Arabic, 25 Hz semantic codec, Zipformer S2M, GRPO post-training, `duration_factor` 0.5–2.0, 2.28× RTF. Same licence. arXiv 2601.03888. | [repo README News](https://raw.githubusercontent.com/index-tts/index-tts/main/README.md), [arXiv](https://arxiv.org/abs/2601.03888) | HIGH |
| 5 | IndexTTS-2 gives precise duration control | **Not in the shipped code** | README, 2025/09/08 entry: *"The first autoregressive TTS model with precise synthesis duration control … <i>This functionality is not yet enabled in this release.</i>"* `infer_v2.py` has no duration parameter. IndexTTS-2.5 ships only a *ratio* (`duration_factor`), not an absolute target. | repo README + `indextts/infer_v2.py` | HIGH |
| 6 | VoxCPM2's voice design is `f"({control}){text}"`, a demo-app hack | **Confirmed and worse** | It is not a demo hack — it is the **documented official API** on the model card, and the *same* parenthetical syntax is used for both voice design (identity) and style control. Identity-by-description and style-by-description occupy literally one field. | [VoxCPM2 card](https://huggingface.co/openbmb/VoxCPM2/resolve/main/README.md) | HIGH |
| 7 | Zonos exposes a `speaking_rate` dial | **True; unit comment is wrong** | Source comment: *"Speaking rate in phonemes per minute (0 to 40). 30 is very fast, 10 is slow."* 30 phonemes/**minute** is physically absurd; the value is phonemes per **second**. The parameter is real; the docstring is a defect. | [zonos/conditioning.py L353](https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/conditioning.py) | HIGH (that the comment is wrong) |
| 8 | `Renderer.render(identity, text, direction) -> Audio` | **Under-specified** | The protocol has nowhere to report *partial* honouring. A backend that silently approximates `rate` while rejecting `emphasis` currently has no way to say so. Needs a `degradations` field on `Audio` and a `direction_support` capability map on the Renderer. See §7. | design | — |
| 9 | Backends that cannot honour a `Direction` "must declare so and degrade EXPLICITLY" | **Correct and load-bearing** | Reinforced: three of the audited backends (Qwen3-TTS Base, Kokoro, Chatterbox) can honour almost nothing *through their APIs*, and two (Parler, VoxCPM2 design mode) will silently *corrupt identity* rather than fail. Explicit rejection is not a nicety here. | this doc | HIGH |
| 10 | "**Emotion must NOT be baked into the identity vector**" | **Right goal, wrong premise** | On Qwen3-TTS Base emotion **is** in the x-vector and is its **dominant** carrier — proven by controlled dissociation (`full_swap`: all codec tokens from an angry utterance + a neutral x-vector → speech "indistinguishable from the neutral baseline"). This is a *consequence of the architecture*: the ECAPA-TDNN encoder is co-trained with the backbone under a synthesis objective, not a frozen verifier, so reconstruction pressure forces it to retain prosodic-emotional cues. The principle survives as an **engineering rule** — mint from neutral audio, store the neutral x-vector, add `α·τ_emo` at render time — but it cannot be asserted as a property of the representation. | [arXiv 2606.05367 §4.1, §5.2](https://arxiv.org/abs/2606.05367) | HIGH |
| 11 | (implied) per-line direction requires a per-line API parameter | **Too narrow** | Three published training-free channels bypass the API entirely: x-vector centroid arithmetic (Qwen3-TTS Base), SLM activation steering (CosyVoice2, IndexTTS2), and SAE feature steering. All operate on a frozen backend. The `Renderer` abstraction should not assume Direction is a call argument. | §5 | HIGH |
| 12 | (implied) transcript punctuation is a usable fallback control | **Measured to fail** | Interspeech 2025 (Shim et al., SFU) tested Parler-TTS, Coqui-XTTS, VoiceCraft, ToucanTTS and Matcha-TTS out of the box on minimal comma/no-comma pairs: the systems *"failed to produce acoustic signals that accurately convey distinct prosodic boundaries."* A 100-sentence fine-tune fixed list-commas (pause 11.26 ms → 101.55 ms, *p*<.01) but **not** vocative commas (*p*=0.89). Do not plan on punctuation. | [ISCA archive](https://www.isca-archive.org/interspeech_2025/shim25_interspeech.pdf) | HIGH |

---

## 2. B5 — IndexTTS2's disentanglement

Paper verified to exist: **IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech**, Zhou, Zhou, He, Zhou, Wang, Deng, Shu. arXiv:2506.21619, v1 2025-06-23, v2 2025-09-03. Repo `index-tts/index-tts` verified live, README last updated for IndexTTS-2.5.

### 2.1 Mechanism

**Three modules:** Text-to-Semantic (T2S, autoregressive transformer) → Semantic-to-Mel (S2M, non-autoregressive flow matching) → BigVGANv2 vocoder.

**The two conditioning paths (both enter the T2S input sequence):**

| Path | Representation | Extractor | Where it enters |
|---|---|---|---|
| **Timbre** | `c`, speaker attributes | **pre-trained speaker perceiver conditioner** (frozen from Stage 2 onward) | first slot of the T2S prefix |
| **Emotion/rhythm** | `e`, emotion embedding | **Conformer-based emotion perceiver conditioner** (trainable in Stage 2) | *added to* `c` — the sequence becomes `[c+e, p, e⟨BT⟩, E_text, e⟨BA⟩, E_sem]` |

Note the shape of this: emotion is **summed into the speaker slot**, not concatenated as a separate token. Disentanglement is therefore not architectural separation — it is a *learned property of the two encoders*.

**How disentanglement is ENFORCED — the answer to the sub-question:**

> "To minimize the content overlap between `e` and `c` while enhancing feature disentanglement, we employ a **GRL** during training. This adversarial mechanism forces `e` to exclusively capture emotional and rhythmic attributes, remaining invariant to speaker-specific timbre characteristics."

Concretely, **Stage 2** of a **three-stage** T2S training regime:

- Stage 1 — full dataset, sequence `[c, p, e⟨BT⟩, E_text, e⟨BA⟩, E_sem]`; duration embedding `p` randomly zeroed with p=30% so the model learns both controlled and free-running modes.
- **Stage 2** — sequence becomes `[c+e, …]`. Speaker perceiver **frozen**; emotion perceiver **trainable**. A **Gradient Reversal Layer (Ganin et al. 2016) plus a speaker classifier** is attached. Trained on a curated **135 hours** of high-quality emotional speech. Joint loss:
  `L_AR = −(1/(T+1)) Σ log q(y_t) − α log q(e)` where `q(e)` is the posterior that `e` originates from the target speaker and α is the loss coefficient. The negative sign in front of the classifier term is the adversarial pressure: `e` is pushed to be *uninformative* about speaker.
- Stage 3 — all feature conditioners frozen; fine-tune on the full dataset for robustness.

Ablation (Table 2, emotional test set): removing the three-stage strategy drops SS 0.836 → 0.773 and ES 0.887 → 0.689. The disentanglement is entirely a product of that training schedule.

**No gradient reversal at inference. No orthogonality loss. No separate encoder at inference beyond the two perceivers, which are themselves trained weights.**

**Duration control:** `p = W_num · h(T)` where `h(T)` is a one-hot over the target semantic-token count `T` and `W_num ∈ ℝ^{L_speech × D}` is an embedding table. The trick is the tied constraint **`W_sem = W_num`** — the duration embedding table is the same table as the semantic positional embeddings, which lets the AR decoder align position with remaining duration. Free-running mode is `p = 0`. Reported token-number error rate: **<0.02%** at original duration, **<0.03%** at 0.875×/1.125×, **0.067%** worst case (SeedTTS test-zh at 0.75×). *Not enabled in the released code.*

**Soft instruction mechanism (T2E):** seven emotions {Anger, Happiness, Fear, Disgust, Sadness, Surprise, Neutral}. DeepSeek-R1 is the teacher, producing a 7-simplex distribution from text; **Qwen-3-1.7b is the student, fine-tuned by LoRA** on 1,000 text→distribution pairs with cross-entropy against the soft teacher distribution. At inference the emotion vector is a weighted average over a **fixed, precomputed emotion-embedding set 𝒱** (embeddings extracted from a handful of emotional audio samples per emotion using the T2S emotion perceiver): `e_input = Σ_e p_e · mean(𝒱_e)`.

The shipped code (`indextts/infer_v2.py`) is broader than the paper: **eight** emotions in the released ordering `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`, with a `QwenEmotion` class carrying a Chinese-key prompt (`文本情感分类`), score clamping to `[0.0, 1.2]`, an `emo_bias` renormalisation `[0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]`, a global rescale if the vector sums above 0.8, and a candid TODO admitting QwenEmotion cannot distinguish 悲伤 (sad) from 低落 (melancholic), worked around by a keyword swap list. Emotion vectors are mixed with the audio-derived emotion vector as `emovec = emovec_mat + (1 − Σ w) · emovec`.

**GPT latent enhancement:** the last-transformer-layer hidden states `H_GPT` of the T2S module are fused (MLP, 50% random fusion probability) with the semantic tokens `Q_sem` before entering the S2M flow-matching module. Purpose: stop slurring under strong emotion. Ablation shows it *lowers* objective SS (0.869 → 0.836) but *raises* subjective SMOS (4.15 → 4.24) and cuts WER (2.766 → 1.883).

**Released API** (`IndexTTS2.infer`, verified in source):
```
infer(spk_audio_prompt, text, output_path,
      emo_audio_prompt=None, emo_alpha=1.0,
      emo_vector=None,
      use_emo_text=False, emo_text=None, use_random=False,
      interval_silence=200, ...)
```
Five mutually-composable emotion routes: separate emotional reference wav; `emo_alpha ∈ [0,1]`; explicit 8-float `emo_vector`; `use_emo_text=True` (derive from the line itself); `emo_text` (a *separate* emotion description string). Note `emo_alpha` semantics differ by route — it scales the audio-prompt merge weight in `merge_emovec(..., alpha=emo_alpha)`, but for `emo_vector` it is clamped to `[0,1]` and applied as a scalar multiplier truncated to 4 decimals.

### 2.2 Transferable to a frozen backend? (the decisive answer)

**No. The mechanism is architecture-bound and training-bound. Verdict: out of scope as stated in the brief.**

Three independent reasons, each sufficient:

1. **The disentanglement lives in weights, not in an algorithm.** GRL is a training-time operator; at inference it is a no-op identity function. There is nothing to copy across. To reproduce the property on another backend you would have to attach a speaker classifier to that backend's style encoder and run adversarial training — i.e. train the acoustic model. Excluded by the LOCKED CONSTRAINT.
2. **The conditioning shape is unique.** `[c+e, p, …]` requires the target backend to have (a) a speaker slot in a prefix sequence and (b) a same-dimensional emotion encoder trained to be additive into it. Qwen3-TTS Base injects a `(2048,)` x-vector as `speaker_embed.view(1,1,-1)` with no second additive slot; Kokoro takes a 256-dim `ref_s` split by index; Parler has no speaker vector at all. There is no common interface.
3. **Even the parts that look portable are not.** The T2E LoRA-on-Qwen3 is portable *in form* — but it emits a probability distribution over **IndexTTS2's own precomputed emotion-embedding set 𝒱**, which is extracted with **IndexTTS2's emotion perceiver**. Without that perceiver the distribution indexes nothing. And building an equivalent 𝒱 for another backend requires that backend to *have* an emotion encoder, which is the thing we were trying to obtain.

**What we need instead — and it exists.** The *goal* (per-line emotion at fixed identity on a frozen backend) is achieved without retraining by **activation steering**, covered in §5. The headline: CoCoEmo gets stronger emotion control than IndexTTS2's own `emo_vector` at *better* speaker preservation, on a frozen backend, with forward hooks and ~4k utterances per emotion. That, not IndexTTS2's architecture, is the technique to build on.

**One genuinely portable idea from the paper.** Not the disentanglement — the *emotion-embedding set* construction: pick N reference clips per emotion, embed them with whatever encoder the backend already has, average, and use the mean vectors as a fixed basis. If a backend has *any* style-vector input, this gives categorical emotion control for free. It also inverts usefully: EmoSteer-TTS's "emotion erasure" operator `x̂ = f_r(x − β(ŝ·x)ŝ)` — projecting the emotion component *out* of an activation — is exactly the operation VoiceForge wants at **identity-minting** time, to strip residual emotion out of a reference clip before extracting the timbre vector. See §8.

### 2.3 Licence and the §3.4(c) hazard

Repo root `/LICENSE` is the **bilibili Model Use License Agreement**. `INDEX_MODEL_LICENSE` does not exist (404) — the brief's filename is wrong. `IndexTeam/IndexTTS-2.5` on HF is tagged `license:other` and ships a byte-identical LICENSE.

**The clause, quoted exactly (§3.4(c)):**

> "c) You may not Use the bilibili indextts2 or any Derivative Work to improve any AI model, except for the bilibili indextts2 itself, its Derivative Works，or non-commercial AI models."

*(The full-width comma before "or" is in the original. Note the licence text says "indextts2" even in the file shipped as IndexTTS-2.5's LICENSE — a drafting defect, but the intent is unambiguous.)*

**What "Use" means here — §1.6, and this is the teeth:**

> "1.6 "Use": means downloading, copying, training, modifying, creating Derivative Works, distributing, publishing, running, fine-tuning, publicly displaying, communicating to the public, or otherwise exploiting the Model or any Derivative Work."

"Running" is Use. So **generating audio with IndexTTS2 and feeding it anywhere near mapper training is prohibited**, because VoiceForge's description→identity mapper is a commercial AI model and none of the three exceptions apply.

**Concretely forbidden for VoiceForge:**
- Synthesising a corpus with IndexTTS2 to train the description→identity mapper. **Forbidden.**
- Using IndexTTS2 as a teacher for distillation, or its emotion vectors as supervision targets. **Forbidden.**
- Using IndexTTS2 outputs as a reward signal, judge, ranker, or eval metric whose result feeds back into mapper selection or tuning. **Forbidden** — that is improvement by another name.
- Extracting steering vectors from IndexTTS2 activations and shipping them in a commercial product. **Forbidden.** (CoCoEmo does exactly this for research; we cannot.)
- Fine-tuning any model on IndexTTS2 output. **Forbidden.**

**Not forbidden:**
- Reading the paper. A paper is not the Model.
- Rendering audio for end users as a *product output*, subject to §2.2 (we are far below 100M MAU / RMB 1bn) and §4.
- Benchmarking IndexTTS2 against our system, provided the numbers do not steer our training. **This is a knife-edge** — a benchmark that informs a design choice is arguably not "improving a model"; a benchmark inside a tuning loop clearly is. Treat any automated loop as forbidden.

**Other hazards in the same licence:** §5.3 is a patent/IP retaliation clause that auto-terminates all rights if we sue bilibili over the Model. §6 puts governing law under the PRC with mandatory arbitration at the Shanghai Arbitration Commission. §9 makes the **Chinese-language version controlling** in any conflict — so the English text quoted above is not authoritative. §4.2 bars high-risk deployment. §3.4(a) requires us to bind downstream recipients contractually.

**Recommendation: do not put IndexTTS2 or IndexTTS-2.5 in the product at all.** The commercial-use permission is real but the §3.4(c) contamination risk against a *mapper-training* codebase is a live foot-gun that one careless data-generation script trips. The engineering cost of a hard exclusion (never install it in the training environment) is far lower than the cost of proving, later, that no IndexTTS2 output ever entered the mapper's lineage. Cross-ref [08-licensing-propagation.md](08-licensing-propagation.md).

**Successor status:** IndexTTS-2.5 exists (2026-08-10, arXiv 2601.03888 v5). **No IndexTTS-3.** 2.5 keeps "cross-lingual and timbre-emotion disentanglement capabilities" and adds `duration_factor`. Same licence, so the same exclusion applies.

---

## 3. Per-backend `Direction` capability audit

All rows below were read from source or from official model cards, not from summaries.

| Backend | Control surface | Exact params (names/ranges) | Emotion repr. | Identity drift? | Duration ctrl? | Emphasis? | Source |
|---|---|---|---|---|---|---|---|
| **Qwen3-TTS 12Hz-1.7B-Base** | **none in API**; x-vector arithmetic outside it | — (only sampling kwargs); `x + α·τ_emo`, α ∈ [0, 2.5] | 8 × ℝ²⁰⁴⁸ τ vectors | **SECS_W 0.912** (multi-spk τ) | no | no | `qwen3_tts_model.py`; [arXiv 2606.05367](https://arxiv.org/abs/2606.05367) |
| **Qwen3-TTS 12Hz-1.7B-CustomVoice** | free-text `instruct` | `instruct: str \| list[str]`, unbounded NL | NL text | UNVERIFIED | no | no | same file, L732+ |
| **Qwen3-TTS 12Hz-1.7B-VoiceDesign** | free-text `instruct` | `instruct: str \| list[str]` (required) | NL text | **channel-shared with identity** | no | no | same file, L637+ |
| **Zonos-v0.1** | 8-vector + 3 scalars | see §3.2 | 8-float simplex, renormalised | **documented as entangled** | `speaking_rate` only | no | `zonos/conditioning.py` |
| **CosyVoice 2** | `instruct_text` + `speed` + tags | `inference_instruct2(tts_text, instruct_text, prompt_wav, …, speed=1.0)` | NL text | **−0.018 to −0.020 S-SIM** | `speed: float` | **`<strong></strong>`** | `cosyvoice/cli/cosyvoice.py`, `cosyvoice/tokenizer/tokenizer.py` |
| **CosyVoice 3** | same + phoneme inpainting | same, `<\|endofprompt\|>`-terminated instruct | NL text | UNVERIFIED | `speed: float` | **`<strong></strong>`** | same |
| **Chatterbox / Multilingual V3** | 1 scalar | `exaggeration: 0.25–2.0` (neutral 0.5), `cfg_weight: 0.0–1.0` | scalar `emotion_adv` (1,1,1) | **documented rate coupling** | no (indirect) | no | `src/chatterbox/tts.py`, `gradio_tts_app.py` |
| **Kokoro-82M** | `speed` + a latent split | `speed: float \| Callable[[int],float]` | none | none (no emotion input) | **`pred_dur` returned** | no | `kokoro/model.py` |
| **Indic-Mio** | text tags + asterisks | `<happy> <sad> <angry> <disgust> <fear> <surprise>`; EN adds `<enunciated> <confused> <whisper>` | discrete tag in text | **structurally separated** | no | **`*word*`** | [model card](https://huggingface.co/SPRINGLab/Indic-Mio) |
| **MioCodec-25Hz** | identity only | `decode(content_token_indices, global_embedding)` | — | — | no | — | [model card](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz) |
| **Indic Parler-TTS** | **one description string** | `generate(input_ids=<description>, prompt_input_ids=<line>)` | NL text, **fused with identity** | **structurally guaranteed** | no | no | `parler_tts/modeling_parler_tts.py` |
| **VoxCPM2** | parenthetical inside `text` | `text="(style)content"` | NL text, **shares text channel** | separate (`reference_wav_path`) | no | no | [model card](https://huggingface.co/openbmb/VoxCPM2) |
| **F5-TTS** | `speed` + `fix_duration` | `speed: float=1.0`, `fix_duration: float\|None` (seconds) | none | n/a | **exact, frame-quantised** | no | `src/f5_tts/infer/utils_infer.py` |
| **MOSS-TTS v1.5** | `tokens` + `[pause X.Ys]` | `build_user_message(text, reference=[...], language=…, tokens=N)` | none documented | n/a | **exact token count** | no | [model card](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) |
| **Orpheus 3B** | paralinguistic tags | `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>` | none (inserts only) | **name prefix shares text channel** | no | no | [repo README](https://github.com/canopyai/Orpheus-TTS) |
| **Sesame CSM-1b** | context only | `generate(text, speaker:int, context:List[Segment], max_audio_length_ms=90000, temperature=0.9, topk=50)` | none | n/a | cap only, not target | no | `generator.py` |
| **Higgs Audio 2** | `scene` role / system prompt | chat-role separated | NL text | UNVERIFIED | no | no | [model card](https://huggingface.co/bosonai/higgs-tts-2-3b-base) |
| **XTTS-v2** | — | — | — | — | — | — | **DISQUALIFIED: CPML non-commercial** |
| **IndexTTS-2 / 2.5** | richest of all | see §2.1 | 8-float vector + audio + NL | −0.008 S-SIM @ 0.6 scale | 2: not enabled; 2.5: `duration_factor` 0.5–2.0 | no | `indextts/infer_v2.py` |

### 3.1 Qwen3-TTS (Base / CustomVoice / VoiceDesign) — an empty API over a steerable vector

**The decisive read.** `generate_voice_clone()` (the *only* method that accepts a speaker embedding) has this signature and this call:

```python
def generate_voice_clone(self, text, language=None, ref_audio=None, ref_text=None,
                         x_vector_only_mode=False, voice_clone_prompt=None,
                         non_streaming_mode=False, **kwargs):
    ...
    talker_codes_list, _ = self.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=languages,
        non_streaming_mode=non_streaming_mode,
        **gen_kwargs,
    )
```

No `instruct` parameter. **No `instruct_ids` passed.** By contrast both `generate_voice_design()` and `generate_custom_voice()` build `instruct_ids` via `self._build_instruct_text(ins)` — which is just `f"<|im_start|>user\n{instruct}<|im_end|>\n"` — and pass `instruct_ids=instruct_ids` into the same `self.model.generate`. **The underlying model method therefore accepts an `instruct_ids` argument that the voice-clone path deliberately does not use.**

**Confirmation from three independent directions:**
1. `config.json` — `speaker_encoder_config` present only in Base (`enc_dim: 2048`), absent from VoiceDesign (`tts_model_type: "voice_design"`) and CustomVoice (`tts_model_type: "custom_voice"`).
2. `generate_voice_clone` raises if `self.model.tts_model_type != "base"`; `generate_voice_design` and `generate_custom_voice` are gated the other way. The three modes are mutually exclusive checkpoints.
3. Qwen's own **InstructTTSEval** table in the Base model card lists `Qwen3TTS-25Hz/12Hz-1.7B-CustomVoice` and `Qwen3TTS-12Hz-1.7B-VD` — and **no Base row at all**. They did not evaluate Base on instruction following, because it does not do it.

**The question the brief asks — "if you hold `speaker_embed` fixed and vary the style instruction, does the perceived speaker change?" — is unanswerable as posed on Base**, because you cannot vary a style instruction on Base through the public API. It is a live question for **CustomVoice**, where timbre is a fixed table entry (9 speakers) and `instruct` varies per line — that is architecturally the exact shape VoiceForge wants, minus arbitrary identity minting. **UNVERIFIED**; no published SS-under-instruct number exists for any Qwen3-TTS variant. Experiment E1/E2 in §9.

**CustomVoice's 9 speakers** (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee) score APS 83.0 / DSD 77.8 / RP 61.2 on InstructTTSEval-ZH — respectable but below Gemini-flash (88.2/90.9/77.3). VoiceDesign scores *higher* (85.2/81.1/65.1) but has no reusable identity.

**The workflow Qwen themselves recommend** is the "Voice Design then Clone" pattern in the README: use VoiceDesign to synthesise a short reference clip matching the persona, feed it to `create_voice_clone_prompt`, then `generate_voice_clone` for every line. That is precisely VoiceForge's `mint_identity` → `render` split — **and it confirms that once you are in clone mode, the per-line Direction channel is gone from the API.**

Licence: Apache-2.0 (`SPDX-License-Identifier: Apache-2.0` in every source header, copyright 2026 Alibaba Qwen team).

### 3.1a The x-vector IS the Direction channel — arXiv 2606.05367, on our exact checkpoint

**This is the most consequential finding in this document for the English track, and it was published specifically on `Qwen3-TTS-12Hz-1.7B-Base`.**

*"Task-Vector Arithmetic for Emotional Expressivity Control in Language-Model-Based Text-to-Speech"* (arXiv 2606.05367, 2026-06-03) runs a **four-operand elimination study** over the Base model's inference pipeline, asking where emotion lives:

| Step | Operand | Result |
|---|---|---|
| 1 | **Backbone weights** (full FT at lr ∈ {2e-6, 2e-5}; LoRA r=64 α=128 on q/k/v/o_proj ≈29M params; + `codec_head` and 15 `lm_head` ≈60M; lr sweep 1e-6…1e-4; epochs 4–39) | **FAILS.** No operational window of controlled emotional speech. Weight-space task arithmetic does not transfer to LM-TTS. |
| 2 | **Continuous codec embeddings** | **FAILS.** α sweeps produce abrupt "no effect → degenerate noise" transitions. Adding a continuous vector to discrete-token embeddings goes out-of-distribution. (Codebook 1 carries the largest τ norm, ‖τ₁‖₂ = 0.137 vs ‖τ_summed‖₂ = 0.293.) |
| 3 | **Discrete codec tokens** | **FAILS — and this is the decisive experiment.** `full_swap`: take *all* codec tokens of an angry utterance, pair them with a *neutral* x-vector → *"calm and coherent speech, indistinguishable from the neutral baseline, even though the `ref_code` carries the entire acoustic signal of angry."* The inverse sounds angry. **The LM ignores the emotional colouring of the tokens in favour of the x-vector.** |
| 4 | **x-vector (ECAPA-TDNN, ℝ²⁰⁴⁸)** | **WORKS.** |

**Architecture, confirmed by this paper against our own config read:** the speaker encoder is an **ECAPA-TDNN (SE-Res2Net)** producing `x ∈ ℝ^2048`, injected as **global conditioning into the codec-embedding sequence, without intermediate projection**. Crucially it is **not a frozen speaker-verification encoder** — it is *co-trained with the backbone under the synthesis objective*. That is exactly why emotion is in there: *"since the Base model depends exclusively on the x-vector for zero-shot cloning, the reconstruction objective pressures the embedding to retain the prosodic-emotional cues necessary for expressive speech, not only identity."*

**The method — training-free, and trivial to implement:**
```
τ_emo = E_i[ x(s_i, emo) ] − E_i[ x(s_i, neutral) ]        # centroids over source speakers
x_new = x(target, neutral) + α · τ_emo                     # applied to an UNSEEN target
```
- Centroids over **50 utterances per speaker per emotion**, from ESD.
- **`avg4spk`**: S = {0011, 0014, 0017, 0020}, gender-balanced. **`single0017`**: S = {0017}.
- α grid: **{0, 0.5, 1, 1.5, 2, 2.5}** (EN→EN); α = 0 is the unmanipulated ICL baseline.
- All audio resampled to 24 kHz (matching `speaker_encoder_config.sample_rate: 24000`).

**Results (EN held-out, unseen ESD speakers {0013, 0019}, n=30 sentences per combination):**

| τ variant | ΔEECS (emotion gain) | **SECS_W** (WavLM, independent) | UTMOS | mean ‖τ‖ |
|---|---|---|---|---|
| `single0017` | **+0.291** | **0.810** | 3.03 | 2.94 |
| **`avg4spk`** | **+0.288** | **0.912** | 3.23 | 1.60 |

**The single most actionable number in this research: averaging τ over four speakers instead of one buys +0.102 SECS_W for free, at identical emotion gain.** The paper's explanation is exact and generalises: *"The `single0017` variant pays a toll in identity because it transports the source speaker's timbre residual to the target via τ; averaging over four speakers cancels this idiosyncratic residual, preserving the shared emotional axis."* Gains are positive across all six target × emotion combinations, Δ ∈ [+0.20, +0.39]; WER stays at 5–7%, the human ceiling reachable by Whisper.

**The geometry that makes it work:**
- `‖τ_emo‖` is only **15% of the x-vector norm**.
- The projection of τ onto the identity axis is **<1% of ‖τ‖** — *"the emotional direction is thus practically orthogonal to the identity axis."*
- Consequently `cos( x(0017, neutral), x(0017, angry) ) = 0.988` — neutral and angry x-vectors of the same speaker are near-collinear.
- *"The x-vector remains a dominant identity bottleneck, but it carries, along this reduced-norm direction, an effective emotional axis."*

**Cross-lingual validation:** τ extracted from English ESD, applied to unseen **Brazilian Portuguese** speakers (emoUERJ) with parallel ground truth. ΔEECS +0.092 (smaller only because the PT-BR baseline already starts at EECS ∈ [0.70, 0.91]; absolute best-α EECS ∈ [0.76, 0.97] matches EN→EN). WER ≈ 0. On the language-agnostic `xvec_cos_GT` metric, **`avg4spk` stays on the natural within-speaker reference line across the entire α grid while `single0017` falls below it at high α** — direct evidence of source-timbre leakage in the single-speaker variant. **τ is language-agnostic**, which matters for the "one voice profile per language per character" constraint: one τ library may serve all languages.

**Why this is exactly right for VoiceForge:** α is a **continuous, post-hoc-selectable intensity knob** — *"Decreasing α monotonically recovers identity at the cost of emotional fidelity; the operating point is selectable post-hoc over the α sweep, without re-synthesis."* That is `Direction.intensity` with a published trade-off curve, on our primary backend, with no training.

**Caveats the paper states itself, and which we must respect:**
1. Validated **only** on Qwen3-TTS-12Hz-1.7B. The architectural argument (any LM-TTS with a *learnable* speaker encoder co-trained for synthesis) is plausible but unvalidated elsewhere.
2. *"Encoders trained exclusively for speaker verification, or with explicit identity–emotion disentanglement (e.g., GRL), may compress the operational window of α."* **This is a direct trade-off against §2.1:** the GRL that makes IndexTTS2's emotion path clean would *destroy* this method. Disentanglement and steerability are in tension.
3. The Step-1 negative (weights) rests on a low-variability regime (~30 min, one speaker, one emotion), where loss-quality divergence is independently reported. Large multi-speaker fine-tuning is not ruled out — but the x-vector localisation rests on Step 3, which is independent of that.
4. EECS (emotion2vec cosine) saturates at ~0.85–0.97; gains in that region may be over-projection. See §6.6 on why this metric is unreliable in general.
5. "Code and artifacts for reproduction are publicly available" — **UNVERIFIED**, no repo URL captured; footnote reference not resolved.

**Answer to the brief's question, restated.** "If you hold `speaker_embed` fixed and vary the style instruction, does the perceived speaker change?" On Base there is no style instruction — but the deeper answer is more useful: **you do not hold `speaker_embed` fixed. You move it, deliberately, along a direction that is <1% aligned with identity.** Direction on Qwen3-TTS Base is x-vector arithmetic, not a parameter.

### 3.2 Zonos-v0.1 — the brief is right, and the source is unusually honest

`make_cond_dict` verified in full. The relevant parameters:

```python
emotion: list[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
#                       Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
fmax: float = 22050.0            # 0–24000; "For voice cloning use 22050"
pitch_std: float = 20.0          # 0–400; 20–45 normal, 60–150 expressive, higher => "crazier"
speaking_rate: float = 15.0      # 0–40 (see correction #7)
vqscore_8: list[float] = [0.78]*8    # 0.5–0.8, per 1/8th of audio; hybrid model only
ctc_loss: float = 0.0                # hybrid only
dnsmos_ovrl: float = 4.0             # hybrid only
speaker_noised: bool = False
unconditional_keys = {"vqscore_8", "dnsmos_ovrl"}
```

The emotion vector is **8 floats, order `[Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral]`, L1-normalised at the end of the function** (`cond_dict[k] /= cond_dict[k].sum(dim=-1)`). It is a probability-like simplex, not categorical, and not calibrated intensities — raising anger necessarily lowers everything else.

**Do the emotion dials perturb identity?** The source itself says yes, in a comment above the parameter:

> `# Emotion vector from 0.0 to 1.0`
> `#   Is entangled with pitch_std because more emotion => more pitch variation`
> `#                     VQScore and DNSMOS because they favor neutral speech`

**Architecturally, though, the channels are separate.** The `PrefixConditioner` concatenates independent per-attribute conditioners into a prefix; `speaker` goes through a `PassthroughConditioner`, `emotion` through its own, scalars through `FourierConditioner` (random-Fourier features over a min/max-normalised scalar) and `language_id` through an `IntegerConditioner`. So `speaker` is a distinct prefix token, not summed into emotion. The entanglement is statistical (training-data correlation), not structural. **This makes Zonos the best-instrumented dial set in the audit** — and the one whose drift is most likely measurable and compensable, because `pitch_std` and `speaking_rate` are separately addressable and can be pinned while emotion varies.

Apache-2.0. **Caveat: `Zyphra/Zonos-v0.1-transformer` last modified 2025-06-03** — 15 months stale. Treat as unmaintained.

### 3.3 VoxCPM2 — style and content share the text channel

Verified against the official model card (not the demo app):

```python
# Voice Design — identity in the parenthetical
wav = model.generate(text="(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!")

# Controllable cloning — STYLE in the same parenthetical
wav = model.generate(text="(slightly faster, cheerful tone)This is a cloned voice with style control.",
                     reference_wav_path="speaker.wav", cfg_value=2.0, inference_timesteps=10)
```

`VoxCPM._generate` (source, `src/voxcpm/core.py`) has parameters `text, prompt_wav_path, prompt_text, cfg_value=2.0, inference_timesteps=10, normalize=False, denoise=False, retry_badcase=True, retry_badcase_max_times=3, retry_badcase_ratio_threshold=6.0`. **There is no style, instruct, emotion, or control parameter anywhere in the signature.** The sibling finding is confirmed at the API level and is not a demo-app artefact — it is the documented interface.

**The structural problem, stated precisely:** for VoxCPM2, *identity and style do not collide* (identity can come from `reference_wav_path`, an audio channel). What collides is **style and content**. Both live in the `text` string, distinguished only by a leading parenthesis. Consequences:
- Style directives are subject to text normalisation (`normalize=True` runs a `TextNormalizer` over the whole string including the parenthetical).
- A line of dialogue that legitimately begins with a parenthetical — stage directions, an aside, "(whispering) I told you" written literally in a script — is ambiguous.
- The style prefix consumes context length and may be spoken aloud on failure. The card's own Notes admit: *"Voice Design and Style Control results may vary between runs; generating 1–3 times is recommended."*
- `retry_badcase=True` by default silently increments the seed (sibling finding), so a failed style render is retried under a different seed — style, seed, and reproducibility are all coupled.

Apache-2.0, 30 languages including Hindi, 48 kHz output. The family is `VoxCPM-0.5B` → `VoxCPM1.5` → `VoxCPM2`; all three exist on HF.

### 3.4 Indic Parler-TTS — one channel, and it is the worst case

**Architecture, read from `parler_tts/modeling_parler_tts.py`:**
- The **description** is encoded by a T5 text encoder → `encoder_hidden_states` → consumed by the decoder's **cross-attention**.
- The **line to be spoken** is encoded separately → `prompt_hidden_states` → **prepended to `inputs_embeds`** as a self-attention prefix: `inputs_embeds = torch.cat([prompt_hidden_states, inputs_embeds], dim=1)`.
- `grep -c "speaker_embed|spk_emb|x_vector|d_vector"` over the 187 KB modeling file: **0**.

**There is no speaker-embedding path in Parler-TTS. None.** Identity is a *name token* inside the description string, learned during training over 69 named speakers ("Divya's voice is monotone yet slightly fast in delivery…").

**Answer to the brief's question — can voice and style vary independently?** **No, not in any architecturally guaranteed way.** Voice and style are the same T5-encoded sequence hitting the same cross-attention. There is no separate pathway to hold fixed. Whether the model has *learned* to treat the name token as sufficient to pin timbre regardless of surrounding adjectives is an empirical question — and the empirical evidence is discouraging: on InstructTTSEval-EN, `Parler-tts-mini` scores **APS 63.4 / DSD 48.7 / RP 28.6** and `Parler-tts-large` **60.0 / 45.9 / —**, the lowest of every system Qwen benchmarked, well below VoxInstruct (54.9/57.0/39.3) and far below Qwen VoiceDesign (82.9/82.4/68.4). Parler is a weak instruction follower even before we ask it to hold identity constant.

Indic Parler-TTS specifics (Apache-2.0, 21 languages, 69 voices): emotion is officially supported in only **10 languages** (Assamese, Bengali, Bodo, Dogri, Kannada, Malayalam, Marathi, Sanskrit, Nepali, Tamil); the emotion set is {Command, Anger, Narration, Conversation, Disgust, Fear, Happy, Neutral, Proper Noun, News, Sad, Surprise}; other controls are pitch, speaking rate, background noise, reverberation, expressivity, voice quality — **all as adjectives in the same string**.

**Honest verdict for the Indic track: Indic Parler-TTS cannot serve as a per-line-directed renderer with identity guarantees.** Mitigations in §4.

### 3.5 Indic-Mio + MioCodec — the Indic track's real answer

**This is a genuinely two-channel design, and the audit's only backend with documented word-level emphasis.**

MioCodec (MIT, 132M params, 25 Hz, 12,800 vocab, 341 bps) explicitly decomposes speech into:
1. **Content tokens** — "linguistic information and phonetic content ('what' is being said)"
2. **Global embeddings** — "a continuous vector representing broad acoustic characteristics ('how') — including speaker identity, recording environment, and microphone traits"

```python
resynth = model.decode(content_token_indices=features.content_token_indices,
                       global_embedding=features.global_embedding)
vc_wave = model.voice_conversion(source, reference)   # swap identity, keep content tokens
```

Indic-Mio (Apache-2.0, 22 scheduled Indian languages + English, 44.1 kHz via `MioCodec-25Hz-44.1kHz`, RTF < 0.1) is a Qwen3-0.6B-class LM fine-tuned from `Aratako/MioTTS-0.6B` that emits those content tokens from text.

**Per-line Direction surface, from the card:**
> "For emotion and style control, place the tags **at the end** of the sentence."
> Indian languages: `<happy>`, `<sad>`, `<angry>`, `<disgust>`, `<fear>`, `<surprise>`
> English: `<happy>`, `<sad>`, `<enunciated>`, `<confused>`, `<angry>`, `<whisper>`
> "A word can be stressed by using asterisks(\*) around it. For example: `No! I could *never* do it!`"

**Why this is structurally better than Parler.** Emotion tags steer the **LM**, which produces content tokens carrying prosody. Identity is applied at **decode** time via `global_embedding`. The two never share a tensor. An emotion tag cannot, by construction, change which global embedding the decoder receives. This matches exactly the architectural principle CoCoEmo established empirically (§5): steer prosody at the sequence model, apply timbre at the acoustic renderer.

**Caveats, stated honestly:** (a) the card's Transformers example does not show how to inject a chosen `global_embedding` into the LM path — the LM emits tokens and MioCodec decodes them; a reference-conditioned path exists via `MioTTS-Inference` but with only 4 documented preset voices (`jp_female`, `jp_male`, `en_female`, `en_male`) — **UNVERIFIED** how arbitrary Indic identities are supplied. (b) Trained in <6 hours on one A6000 ADA; no emphasis or emotion evaluation is published. (c) `<whisper>` is English-only. Cross-ref [04-indic-track.md](04-indic-track.md).

### 3.6 Kokoro-82M — one dial, and a hidden second one

Documented control is `speed` alone:
```python
def __call__(self, text, voice=None, speed: float|Callable[[int],float] = 1, ...)
```
`speed` divides the predicted per-token duration: `duration = torch.sigmoid(duration).sum(axis=-1) / speed`.

**The undocumented split.** In `KModel.forward_with_tokens`:

```python
s = ref_s[:, 128:]                                   # -> prosody predictor
d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
duration = self.predictor.duration_proj(x)
F0_pred, N_pred = self.predictor.F0Ntrain(en, s)     # F0 and energy from the SAME half
audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])   # -> acoustic decoder
```

The 256-dim voice vector is **`[decoder_style(0:128) | prosody_style(128:256)]`**. Duration, F0 and energy are all predicted from the second half; timbre rendering uses only the first half. **This is a free, frozen-backend timbre/prosody split** — assemble `ref_s = cat([identity[:128], direction_prosody[128:]])` and you have a Direction channel Kokoro's docs never mention. Whether the halves are perceptually independent is **UNVERIFIED**; experiment E3 in §9.

Second free capability: `KModel.Output` carries **`pred_dur: torch.LongTensor`** — the per-token frame counts. Total duration is therefore known *before* vocoding, and per-word timing is derivable via the phoneme→word mapping. Combined with the cheap forward pass this gives closed-loop duration targeting by binary search on `speed` (2–4 iterations to hit a slot). See §3.13.

Apache-2.0. `load_voice()` accepts a raw `torch.FloatTensor`; multi-voice averaging via `torch.mean(torch.stack(packs), dim=0)` is upstream. Style packs are length-indexed (`pack[len(ps)-1]`), so the style vector already varies with phoneme count — any blending must respect that indexing.

### 3.7 Chatterbox — one scalar, with documented rate coupling

`exaggeration` enters as a `T3Cond` field alongside `speaker_emb`:
```python
t3_cond = T3Cond(speaker_emb=ve_embed,
                 cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                 emotion_adv=exaggeration * torch.ones(1, 1, 1))
```
Range from the reference Gradio app: **`gr.Slider(0.25, 2, step=.05, value=.5)`**, labelled "Exaggeration (Neutral = 0.5, extreme values can be unstable)". Companion `cfg_weight`: `gr.Slider(0.0, 1, step=.05, value=0.5)`, labelled **"CFG/Pace"**.

**Does it disturb identity?** The README documents cross-coupling to *rate*, not directly to timbre:
> "Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing."
> "For Expressive or Dramatic Speech: try lower `cfg_weight` (~0.3) and increase `exaggeration` to ~0.7 or higher."

**Structurally favourable:** `emotion_adv` conditions only **T3** (the token LM). The acoustic path **S3Gen** is conditioned by `s3gen_ref_dict` (prompt mel + prompt tokens + embedding) derived from the reference wav and *not* touched by exaggeration. This is the same SLM/decoder split CoCoEmo recommends, so identity should be comparatively robust — **UNVERIFIED**, no published number. Chatterbox Multilingual V3 (500M, 23+ languages, `language_id` param) claims "more consistent speaker similarity". MIT licence.

Note Chatterbox's identity is a reference **wav**, not a mintable vector — Tier 2/3 for VoiceForge — though `Conditionals.save()/load()` serialises the whole conditioning bundle to a `.pt`, which is a usable identity record.

### 3.8 CosyVoice 2 / 3 — the best-shaped Direction API in the audit

```python
inference_zero_shot(tts_text, prompt_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)
inference_instruct2(tts_text, instruct_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)
add_zero_shot_spk(prompt_text, prompt_wav, zero_shot_spk_id)     # persist an identity
```

**Three orthogonal inputs: identity (`prompt_wav` / persisted `zero_shot_spk_id`), performance (`instruct_text`), rate (`speed`).** This is exactly VoiceForge's `Direction` shape. CosyVoice3 terminates instructions with `<|endofprompt|>` and prefixes them conversationally (`'You are a helpful assistant. 请用广东话表达。<|endofprompt|>'`).

**Emphasis and paralinguistics** — from `CosyVoice2Tokenizer` / `CosyVoice3Tokenizer` `additional_special_tokens`:
`<strong>`, `</strong>`, `<laughter>`, `</laughter>`, `[breath]`, `[quick_breath]`, `[laughter]`, `[cough]`, `[sigh]`, `[clucking]`, `[accent]`, `[hissing]`, `[vocalized-noise]`, `[lipsmack]`, `[mn]`, `[noise]` — **plus the full CMU ARPAbet set (`[AA0]`…`[ZH]`) and a full Pinyin-with-tone set** for pronunciation inpainting. CosyVoice3 adds `<|endofsystem|>`.

`<strong>...</strong>` is the **only paired emphasis markup in any Apache-2.0 backend audited**. Caveat: the source comment says *"NOTE: non-chat model, all these special tokens keep randomly initialized"* — that refers to the tokenizer's `add_special_tokens` call, not necessarily to the trained embeddings, but it is a warning sign. No published emphasis evaluation exists. **MEDIUM** that `<strong>` works reliably; experiment E4.

**Identity drift under instruct — measured.** See §6: CosyVoice2 S-SIM 0.871 (no instruct) → 0.853 / 0.851 (two instruct phrasings) on CREMA-D; 0.896 → 0.887 on IEMOCAP. Consistently **≈ −0.02 / −0.01**. Small but real, and it is *the* number to beat.

Both Apache-2.0 (`FunAudioLLM/CosyVoice2-0.5B`, `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`).

### 3.9 F5-TTS — no style channel, but the best duration control

No emotion, style, or instruct parameter exists. What it has:
```python
infer_process(..., speed=1.0, fix_duration=None, nfe_step=32, cfg_strength=2.0,
              sway_sampling_coef=-1.0, cross_fade_duration=0.15)
```
`fix_duration` is in **seconds** and is converted to mel frames exactly:
```python
duration = int(fix_dur * target_sample_rate / hop_length)
```
At 24 kHz / hop 256 that is **10.67 ms granularity**. Note the semantics trap: `fix_duration` is the **total** including the reference audio prefix — the multi-chunk allocator computes `target_total = fix_duration - ref_sec` then re-adds `ref_sec` per chunk. Wrapping code must subtract the reference length.

Also note an undocumented override: `if len(gen_text.encode("utf-8")) < 10: local_speed = 0.3` — short lines (under 10 bytes, i.e. most interjections: "No!", "Wait.") are silently slowed to 0.3× regardless of the caller's `speed`. That will bite on game barks.

MIT-licensed code; check weight licence separately.

### 3.10 MOSS-TTS v1.5 — exact token-count duration and inline pauses

Apache-2.0. API:
```python
processor.build_user_message(text=..., reference=[ref_audio], language="French", tokens=325)
```
- **`tokens=N`** — explicit semantic-token count, the same paradigm as IndexTTS2's `p`, and unlike IndexTTS2 it **is** enabled.
- **`[pause X.Ys]`** inline, e.g. `"…它的名字是[pause 3.2s]静夜思！"` — explicit, second-precision silence insertion. **The only backend audited with a documented inline pause primitive.**
- Pinyin and IPA pronunciation control in the text stream (`/həloʊ, meɪ aɪ æsk .../`).

No emotion or style channel is documented. **UNVERIFIED** whether one exists. Duration precision is **UNVERIFIED** (no published error rate).

### 3.11 Orpheus, Sesame CSM, Higgs Audio 2

- **Orpheus 3B** (Apache-2.0): emotive **inserts** only — `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>` — not prosody control. Worse, **identity is a name prefix in the same text channel**: `"{name}: I went to the …"` with name ∈ {tara, leah, jess, leo, dan, mia, zac, zoe}. One-channel, same class as Parler.
- **Sesame CSM-1b** (Apache-2.0): `generate(text, speaker:int, context:List[Segment], max_audio_length_ms=90_000, temperature=0.9, topk=50)`. Identity is an **integer index** plus conversational `context` segments. `max_audio_length_ms` is a *cap* (`max_generation_len = int(max_audio_length_ms / 80)`), not a duration target. **No Direction channel whatsoever.**
- **Higgs Audio 2**: control via a **`scene` role in a chat template** and a system prompt — genuinely a separate channel from the user text, which is architecturally good, and it wins 75.7% vs `gpt-4o-mini-tts` on EmergentTTS-Eval "Emotions". **But the licence is the "BOSON HIGGS AUDIO 2 COMMUNITY LICENSE AGREEMENT", explicitly "based upon the Meta Llama 3 Community License Agreement"** — not Apache-2.0, not MIT. **Fails the LOCKED CONSTRAINT.** Exclude.

### 3.12 XTTS-v2 — disqualified

`LICENSE.txt` fetched from the HF repo: **"Coqui Public Model License 1.0.0 … This license allows only non-commercial use of a machine learning model and its outputs."** Public hosting is commercial use. **Exclude, no further analysis.**

### 3.13 Duration & timing control — consolidated

| Backend | Mechanism | Target type | Granularity | Published accuracy |
|---|---|---|---|---|
| **F5-TTS** | `fix_duration` (sec) | **absolute** | 10.67 ms (hop/sr) | none published |
| **MOSS-TTS v1.5** | `tokens=N` | **absolute** | 1 semantic token | none published |
| **MOSS-TTS v1.5** | `[pause X.Ys]` | **absolute silence** | 0.1 s | none published |
| **IndexTTS2 (paper)** | `p = W_num·h(T)` | **absolute** | 1 semantic token | **<0.02%–0.067% token error** — *not enabled in released code* |
| **IndexTTS-2.5** | `duration_factor` | ratio 0.5–2.0 | — | none published |
| **Kokoro** | `speed` + returned `pred_dur` | ratio → **closed-loop absolute** | per-token frames | derivable by search |
| **CosyVoice 2/3** | `speed: float` | ratio | — | none |
| **Zonos** | `speaking_rate` 0–40 | rate | — | none |
| **Chatterbox** | none (`exaggeration` couples to rate) | — | — | — |
| **Sesame CSM** | `max_audio_length_ms` | **cap only** | — | — |
| Qwen3-TTS, VoxCPM2, Parler, Indic-Mio, Orpheus, Higgs | **none** | — | — | — |

For a lip-sync window, only F5-TTS and MOSS-TTS v1.5 hit an absolute target open-loop. Kokoro hits one in 2–4 cheap iterations. **Everything we want as a primary backend (Qwen3-TTS) has nothing** — see §8.

### 3.14 Word-level emphasis — consolidated

| Backend | Markup | Evidence | Confidence it works |
|---|---|---|---|
| **CosyVoice 2 / 3** | `<strong>word</strong>` | special token in `CosyVoice2Tokenizer`/`CosyVoice3Tokenizer` | **MEDIUM** — token exists, no eval published |
| **Indic-Mio** | `*word*` | documented on model card with example | **MEDIUM** — documented, no eval |
| **CosyVoice 2 / 3** | `[breath] [laughter] [sigh] [cough] …` | special tokens | MEDIUM |
| **Orpheus** | `<laugh> <sigh> <gasp> …` | README | MEDIUM — *inserts*, not emphasis |
| Everything else | — | — | **None.** |

Honest answer to the brief's question: **no frozen open backend has a *demonstrated* reliable word-level emphasis capability.** Two have plausible markup. Nobody has published a measurement. Fall-backs: capitalisation and punctuation in the transcript (unmeasured folklore), or post-hoc prosody editing on the rendered word (§5).

---

## 4. The one-channel problem (Indic Parler-TTS, VoxCPM2)

**The problem, stated generally.** When identity and performance are expressed in the same input field, per-line direction and identity consistency are in *direct competition*, and no amount of care at the call site can separate them — the model's own attention has already mixed them. Three distinct severities appeared in this audit:

| Severity | Backends | What collides | Consequence |
|---|---|---|---|
| **Severe — identity ⊗ style** | Indic Parler-TTS, Orpheus (name prefix), Qwen3-TTS VoiceDesign | speaker identity and performance share one text sequence and one attention path | Every per-line direction is a fresh roll of the identity dice. Consistency across a character's 200 lines is not achievable, only approximated. |
| **Moderate — style ⊗ content** | VoxCPM2 (both design and clone modes) | style directive and the line to speak share the `text` string | Directive can be spoken aloud, normalised, or truncated; scripts with literal parentheses are ambiguous; interacts with `retry_badcase` seed drift. Identity itself is safe (`reference_wav_path`). |
| **None** | CosyVoice 2/3, Indic-Mio, Chatterbox, Zonos, Higgs | separate fields / separate modules | Direction is addressable. |

**Honest assessment for the Indic track.** The brief's proposed pairing is `indic-parler-tts` (designer) + `Indic-Mio` (renderer) + `MioCodec` (speaker vector). This research supports that split for a reason the brief may not have intended: **Parler should be used only as a designer, never as a per-line renderer.** As a designer it is fine — you invoke it once per character to realise a description into a reference clip, and the identity/style entanglement is harmless because you are choosing both together, deliberately, one time. As a renderer it is disqualified: every `Direction` you apply reaches through the same channel that holds the character's name.

**Mitigation options for a one-channel backend, ranked:**

1. **Demote it to a designer.** Use it once, capture a reference clip, extract a `global_embedding` (MioCodec) or a speaker vector, and render every line on a two-channel backend. *This is the recommendation.* Cost: near zero. Loses nothing, because the design step is where the one-channel behaviour is actually wanted.
2. **Freeze the identity clause, vary only a trailing style clause.** Keep the description prefix byte-identical across every line of a character (`"Divya's voice is …"`) and append direction adjectives after it. Reduces variance but does **not** eliminate it — cross-attention is order-sensitive but global. Verify with E1; expect residual drift.
3. **Render, then re-impose identity.** Generate the line on the one-channel backend, then run **MioCodec voice conversion** (`model.voice_conversion(source, reference)`) to force the character's `global_embedding` back onto it. This is a real, cheap fix that exploits MioCodec's content/global split: prosody survives, timbre is reset. **This is the strongest mitigation and the one most worth prototyping.** Costs one extra codec round-trip (132M params, RTF ≪ 0.1). Risk: the conversion also flattens some prosody.
4. **Post-hoc prosody transfer.** Render neutral on the identity-safe backend, render directed on the one-channel backend, transplant F0 + duration from the second onto the first. Highest fidelity to identity, most fragile.
5. **Accept and gate.** Declare `emotion: "approx"` with a documented drift bound, and reject `Direction` entirely on lines flagged identity-critical.

**For VoxCPM2 specifically,** the moderate severity is manageable: always pass identity via `reference_wav_path` (never via a design parenthetical at render time), keep the style parenthetical to a short fixed vocabulary, set `retry_badcase=False` for reproducibility, and escape any literal leading parenthesis in the script. Under those rules VoxCPM2 is usable as a Tier-2 renderer with an approximate Direction.

---

## 5. Frozen-backend control techniques, ranked

Only techniques compatible with a **frozen backend** are listed. Ranked by (control quality × identity preservation) ÷ solo-dev cost.

| # | Technique | Control quality | Identity preserved? | Solo-dev cost | Frozen-compatible? |
|---|---|---|---|---|---|
| **1** | **x-vector centroid arithmetic** (2606.05367) — *validated on our exact checkpoint* | High + continuous (α) | **SECS_W 0.912** with multi-speaker τ | **Lowest of all** — numpy on a 2048-vector | **Yes, fully** |
| **2** | **SLM-side activation steering** (CoCoEmo) | **High + continuous + composable/mixed** | **Best measured: ±0.001 S-SIM at α=5** | Medium — hooks + ~4k utt/emotion + a layer sweep | **Yes, fully** |
| **3** | **Native instruct channel** (CosyVoice2/3, Higgs `scene`, Qwen CustomVoice) | High, categorical/NL | −0.01 to −0.02 S-SIM | **Lowest** — a string | Yes |
| **4** | **Emotion reference audio** (IndexTTS2 `emo_audio_prompt`) | High | −0.008 to −0.014 S-SIM | Low — curate a clip library | Yes, if the backend has the input |
| **5** | **Per-segment emotion + duration** (TED-TTS, 2601.03170) | **Highest granularity published** — sub-utterance | SSIM 0.485 vs 0.457 baseline (*improves* it) | Medium — public code, but IndexTTS2-bound | Yes — **but §3.4(c)-blocked for us** |
| **6** | **Fixed emotion-embedding basis** (IndexTTS2 T2E trick, generalised) | Medium, categorical + interpolable | Backend-dependent | Low — N clips/emotion + mean | Yes, if the backend has *any* style vector |
| **7** | **Native scalar dials** (Zonos `emotion`/`pitch_std`/`speaking_rate`, Chatterbox `exaggeration`) | Medium, coarse | Documented entanglement | Lowest | Yes |
| **8** | **Latent-half swap** (Kokoro `ref_s[:,128:]`) | **UNVERIFIED** | **UNVERIFIED**, structurally promising | Very low — tensor slicing | Yes |
| **9** | **Text tags / markup** (`<happy>`, `<strong>`, `*word*`, `[pause 3.2s]`) | Low–medium, discrete | Usually fine (separate module) | Lowest | Yes |
| **10** | **Codec-level identity re-imposition** (MioCodec `voice_conversion`) | n/a (fixes identity, not style) | **Excellent** | Low | Yes (post-process) |
| **11** | **SAE feature steering** (2606.01479) | High + *interpretable* + bidirectional | ≈ instruct-mode | High — train a 10.5M SAE (~40 MB) offline | Backbone frozen; SAE is trained |
| **12** | **Trainable steering layer** (EmoShift, 2601.22873) | High (α extendable past training) | **SpkSIM 82.23 → 82.41** — unchanged | Medium — 10M params, 5 epochs on ESD | Backbone frozen; adapter trained |
| **13** | **Post-hoc prosody transfer** (F0 + duration transplant onto a neutral render) | Medium–high on rate/pitch, none on voice quality | Good by construction | High — alignment + resynthesis stack | Yes (post-process) |
| **14** | **CFG manipulation** (`cfg_weight`, `cfg_value`, `cfg_strength`) | Low→high but couples to text fidelity | **Spk-sv 0.90 → 0.79** across the useful range | Lowest | Yes |
| **15** | **CFM/decoder-side activation steering** (EmoSteer-TTS as published) | High | **−0.064 S-SIM — the wrong site** | Medium | Yes, but **do not use as primary** |
| **16** | **Speech editing as a post-pass** (VoiceCraft-class) | High, local | Risky | Very high | Technically yes |
| **17** | **Emotional voice conversion post-pass** (TRACE-EVC, TargetSEC) | High, and TRACE-EVC takes *relative* NL instructions | **SECS 0.55–0.68** — steep cost | High | Yes (post-process) |
| **18** | **Transcript punctuation engineering** | **Measured to fail** on stock open models | Fine (it does nothing) | Lowest | Yes, but pointless |
| **19** | **GST / style tokens** | — | — | — | **No.** The token bank is jointly trained; no post-hoc extraction method exists. |
| **20** | SSML | — | — | — | **No open backend audited implements SSML.** Do not plan on it. |

### 5.0 Why #1 and #2 are the whole answer

Both are training-free, both operate on a frozen backend, and between them they cover every backend shape we care about:
- **If the backend conditions on a learnable speaker vector co-trained for synthesis → use x-vector arithmetic (#1).** Qwen3-TTS Base is exactly this. Cost: one afternoon.
- **If the backend is a speech-LM + decoder stack → steer the LM (#2).** CosyVoice2, IndexTTS2, Chatterbox T3, Indic-Mio are this shape.
- The two are **complementary, not alternatives** — CoCoEmo shows steering stacks on top of native conditioning for additional control (at some identity cost).

### 5.0a The steering-site rule, and the mechanism behind it

**arXiv 2607.00946** (Univ. Melbourne, 2026-07-01) is a controlled comparison of *where* to steer on CosyVoice2, and it is the paper that settles the design question:

| Property | SLM (speech LM) | CFM (flow-matching decoder) |
|---|---|---|
| Hidden dim | 896 | 256 |
| Emotion probe acc. (within / cross-speaker) | 0.80 / **0.71** | 0.89 / **0.62** |
| **Within–cross gap** (speaker leakage) | **0.08** | **0.32** |
| Local intrinsic dim | ~28 | ~13 |
| ΔLID under steering | +0.84 | −1.48 |
| Discriminability peak | mid-to-late layers (10–17) | uniform, no peak |

The mechanism, quoted: *"the SLM is not conditioned on speaker embeddings, whereas the flow-matching module is explicitly conditioned on speaker embeddings and reference speech, so perturbing CFM activations directly interferes with speaker-dependent representations."*

That single sentence generalises into VoiceForge's architectural rule: **intervene in the module that does not see the identity.** It explains every result in §6 — why CFM steering costs 0.064 and SLM steering costs 0.000; why Kokoro's prosody predictor (which sees only `ref_s[128:]`) is a promising intervention point; why Indic-Mio's text tags are safe (the LM never sees `global_embedding`); and why Parler is hopeless (there is only one module and it sees everything).

Corroborating independent evidence: **DUET** (arXiv 2606.00066) measures emotion as only **8.5% of hidden-state variance** with speaker identity dominating, and finds emotion and speaker directions **near-orthogonal on F5-TTS (|cos θ| = 0.029** at the shared probe layer, separability peaking at layer 16). The accent-steering paper (arXiv 2603.05977) runs the same recipe on **Qwen3-TTS** for accent and reports middle layers (15, 20) as the best trade-off, with α=2.0 dropping Spk-Sim 0.84 → 0.76 for the 1.7B model — a useful reminder that steering the *backbone* of Qwen3-TTS is far costlier than steering its x-vector (§3.1a).

⚠️ **Caveat on DUET:** it argues identity preservation *geometrically* and **reports no SECS/SIM numbers at all**. Confidence in its identity claim: **LOW**.

### 5.1 Why activation steering is #1

**EmoSteer-TTS** (arXiv 2508.03543, v2 2025-08-06; HKUST-GZ + Tencent AI Lab). *"the first method that achieves training-free and continuous fine-grained emotion control in TTS."* Mechanism:
1. Collect activations from M neutral + N emotional generations at selected layers; interpolate token sequences to a fixed length; take the difference `u^l`.
2. Probe each token position by steering with it alone and scoring the output with **emotion2vec**; keep the top-k positions; zero the rest → `s^l`.
3. Softmax-weight the survivors → `ŝ^l`.
4. At inference: `x̂^l = f_r(x^l + α·ŝ^l)`, with `f_r = ‖x^l‖₂ / ‖x^l + û^l‖₂` renormalising to preserve the original L2 norm.
- `α > 0` steers toward, `α < 0` steers away, `α = 0` is a no-op → **continuous intensity, for free**.
- **Emotion erasure:** `x̂^l = f_r(x^l − β(ŝ^l·x^l)ŝ^l)` — project the emotion component out. Directly useful at identity-mint time.
- Implementation: *"steering operations are implemented as hook functions … registered either before or after the forward pass of the first residual stream in each DiT block."* Demonstrated on **F5-TTS (22 DiT layers; steer 1, 6, 11, 16, 21), E2-TTS, CosyVoice2**. Applied across all flow-matching steps (32 for F5/E2, 10 for CosyVoice2).

**CoCoEmo** (arXiv 2602.03420, code at `github.com/wsssy/CoCoEmo`) improves on it and answers the *where* question definitively:
- A cross-conditioning diagnostic shows **"SLM is the primary driver of emotional prosody… emotion steering should be applied at the SLM."** Conditioning only the flow-matching decoder leaves energy contours overlapping across emotions — *"the flow-matching module does not alter the prosody but mainly performs acoustic rendering."*
- Layer/operation probing: **CosyVoice2 layers 10–17, `attn_output`**; **IndexTTS2 layers 5–10**. Mid-to-late layers and attention outputs generalise.
- Supports **mixed emotions** with proportional control (steering vector = consensus-distribution-weighted mix), which is what drama actually needs — "glad you came, but I wish it weren't so late".
- *"CoCoEmo achieves comparable and stronger mixed-emotion control while better preserving speaker similarity"* than EmoSteer-TTS, and *"simultaneous steering in both SLM and flow-matching modules degrades performance."*
- Steering vectors extracted from **ESD + RAVDESS + CREMA-D**, 20,691 utterances (~4,000/emotion), speaker-independent 0.5/0.2/0.3 split. All three are standard, obtainable emotional corpora.

**The architectural lesson VoiceForge should adopt wholesale:** in a two-stage backend (sequence model → acoustic decoder), **prosody belongs to the sequence model and timbre belongs to the decoder.** Every backend that scores well on identity-under-emotion in this audit has that shape and conditions them separately: Chatterbox (`emotion_adv` → T3 only, identity → S3Gen), Indic-Mio (tags → LM, `global_embedding` → MioCodec), CosyVoice2 (instruct → LM, `prompt_wav` → flow decoder). Every backend that fails has one fused path.

**Related 2026 work, verified to exist:** "A Geometric Perspective on Composable Emotion Steering in TTS" (arXiv 2607.00946); "DUET: Unified Dual-Space Emotion Control for Diffusion and Flow-Matching Driven TTS" (arXiv 2606.00066); "Controllable Affective Generation via Latent Vector Steering" (arXiv 2608.25569). The field is moving fast and in our direction.

### 5.2 Techniques that require training — noted and excluded

Verified real, but all require training an acoustic model, so **out of scope**:
- **FC-TTS** (arXiv 2605.24618, 2026-05-23) — dual-reference disentangled style/timbre control. The closest published system to VoiceForge's ideal, and it needs full training.
- **Voice Impression Control in Zero-Shot TTS** (arXiv 2506.05688 v3, 2026-02-18) — a low-dimensional vector over impression pairs (dark–bright), GRL for disentanglement, LLM-generated target vectors from a natural-language description. Architecturally the closest thing to VoiceForge's `description → identity` mapper in the literature. Worth reading for the mapper design (cross-ref [02-identity-representation.md](02-identity-representation.md)); not usable as a frozen technique.
- **EmoSphere++** (VAD-sphere emotion control), **HED-TTS**, **EmoDubber**, **EmoVoice**, **FleSpeech**, **ControlSpeech**, **SelfTTS** (arXiv 2603.22252), **Joycent** (arXiv 2606.16417).

---

## 6. Identity drift under emotion — published numbers

**Two directly usable tables exist.** Both were found by reading the papers, not summaries.

### 6.1 CoCoEmo Table 2 — CREMA-D, in-distribution, S-SIM = cosine over **WavLM-base speaker embeddings**

| Backbone | Method | E-SIM ↑ | TEP ↑ | ρ ↑ | H-Rate ↑ | **S-SIM ↑** | WER ↓ | N-MOS ↑ |
|---|---|---|---|---|---|---|---|---|
| CosyVoice2 | **No-steer (baseline)** | 0.743 | 0.065 | — | — | **0.871** | 1.07 | 4.11 |
| CosyVoice2 | Instruction1 (qualitative NL) | 0.761 | 0.200 | 0.111 | 0.694 | **0.853** *(−0.018)* | 0.22 | 4.11 |
| CosyVoice2 | Instruction2 (quantitative NL) | 0.762 | 0.169 | 0.104 | 0.688 | **0.851** *(−0.020)* | 0.06 | 3.36 |
| CosyVoice2 | **CoCoEmo α=3.0** | 0.762 | 0.100 | 0.166 | 0.709 | **0.872** *(+0.001)* | 1.01 | 4.25 |
| CosyVoice2 | **CoCoEmo α=5.0** | 0.779 | 0.149 | 0.209 | 0.724 | **0.870** *(−0.001)* | 0.78 | 3.96 |
| CosyVoice2 | CoCoEmo + Ins1, α=5.0 | 0.790 | 0.335 | 0.319 | 0.760 | 0.846 *(−0.025)* | 0.20 | 3.00 |

**Read this carefully. It is the single most important table in this document.**
- The **native text-instruction channel costs ≈ 0.02 S-SIM.** That is the price of the easiest technique.
- **Activation steering costs ≈ 0.001 S-SIM** — statistical zero — while delivering *better* rank-correlation and hit-rate control than instruction.
- Stacking them gives the best emotion control and the **worst** identity preservation (−0.025). The drift is additive and attributable to the *text* channel, not the steering.
- The intensity parameter α behaves monotonically on control (E-SIM 0.762 → 0.779 → 0.790) with **no corresponding monotonic identity cost** (0.872 → 0.870). This is the empirical refutation of the assumption that stronger emotion must mean more drift — *it only does when the emotion is delivered through a channel that also carries identity information.*

### 6.2 CoCoEmo Table 3 — IEMOCAP, out-of-distribution, high text–emotion mismatch

| Backbone | Method | E-SIM ↑ | TEP ↑ | **S-SIM ↑** | WER ↓ | N-MOS ↑ |
|---|---|---|---|---|---|---|
| CosyVoice2 | No-steer | 0.802 | 0.197 | **0.896** | 9.18 | 4.22 |
| CosyVoice2 | Instruction | 0.843 | 0.436 | **0.887** *(−0.009)* | 4.01 | 4.23 |
| CosyVoice2 | CoCoEmo α=6.0 | 0.862 | 0.504 | **0.892** *(−0.004)* | 8.74 | 4.40 |
| **IndexTTS2** | No-steer | 0.825 | 0.318 | **0.885** | 5.13 | 4.17 |
| **IndexTTS2** | **`emo_vector` (scale 0.6)** | 0.872 | 0.667 | **0.877** *(−0.008)* | 5.75 | 4.05 |
| **IndexTTS2** | CoCoEmo α=6.0 | 0.874 | 0.681 | **0.886** *(+0.001)* | 6.58 | 4.21 |

**IndexTTS2's GRL disentanglement demonstrably works:** its own emotion vector more than doubles target-emotion probability (0.318 → 0.667) for **−0.008 S-SIM**. That is the best emotion-per-unit-drift of any *native* channel measured anywhere in this audit. It is exactly what the three-stage GRL training bought — and exactly what we cannot have without training.

**A hard operating bound, stated by CoCoEmo's own experimental setup:**
> "Emotion-vector control (IndexTTS2): use the built-in emotion-weight vector for control, with **scaling set to 0.6, the maximum that preserves speech intelligibility and speaker characteristics**."

An independent group, choosing an operating point for a fair comparison, found **0.6** to be the ceiling. This corroborates IndexTTS2's own code, which rescales any emotion vector summing above **0.8** (`if emo_sum > 0.8: scale_factor = 0.8/emo_sum`) and recommends `emo_alpha ≈ 0.6` for text-derived emotion. **Three independent signals converge on ~0.6–0.8 as the intensity ceiling before identity degrades.** That is a concrete design constant for VoiceForge's `Direction.intensity`.

### 6.3 EmoSteer-TTS Table 1 — cross-method, S-SIM from an SER-model embedding (Bredin et al. 2020), **not comparable in absolute terms** to §6.1/§6.2

| Method | Type | WER ↓ | **S-SIM ↑** | E-SIM ↑ | N-MOS ↑ |
|---|---|---|---|---|---|
| EmoSphere++ | label, trained | 16.25 | 0.44 | 0.25 | 3.23 |
| EmoDubber | label, trained | 18.61 | 0.41 | 0.25 | 2.47 |
| HED-TTS | label, trained | 13.27 | 0.52 | 0.22 | 3.31 |
| EmoVoice | description, trained | 2.91 | 0.58 | 0.27 | 3.81 |
| **CosyVoice2** | description, trained | 2.53 | **0.73** | 0.24 | 3.69 |
| FleSpeech | description, trained | 9.34 | 0.54 | 0.29 | 3.07 |
| **F5-TTS + EmoSteer-TTS** | **training-free** | 2.79 | 0.64 | **0.29** | 3.29 |
| **E2-TTS + EmoSteer-TTS** | **training-free** | 3.28 | 0.59 | 0.28 | 3.31 |
| **CosyVoice2 + EmoSteer-TTS** | **training-free** | 2.83 | 0.65 | 0.26 | 3.65 |

The relative story: **purpose-built emotional TTS systems sacrifice a great deal of speaker similarity** (0.41–0.58) to get emotion, whereas a general zero-shot model with a light control channel keeps 0.73. Steering the flow-matching module (EmoSteer-TTS) costs CosyVoice2 0.73 → 0.65 — **an 11% relative drop, and materially worse than CoCoEmo's SLM-level steering.** This is the empirical basis for CoCoEmo's claim that steering belongs at the SLM, and a direct warning: *where* you steer determines whether identity survives.

### 6.4 IndexTTS2's own numbers (arXiv 2506.21619v2, Table 2, emotional test set)

| Model | SS ↑ | WER ↓ | ES ↑ | SMOS ↑ | EMOS ↑ |
|---|---|---|---|---|---|
| MaskGCT | 0.810 | 4.059 | 0.841 | 3.42 | 3.37 |
| F5-TTS | 0.773 | 3.053 | 0.757 | 3.37 | 3.16 |
| CosyVoice2 | 0.803 | 1.831 | 0.802 | 3.13 | 3.09 |
| SparkTTS | 0.673 | 2.299 | 0.832 | 3.01 | 3.16 |
| IndexTTS (v1) | 0.649 | 1.136 | 0.660 | 3.17 | 2.74 |
| **IndexTTS2** | **0.836** | 1.883 | **0.887** | **4.24** | **4.22** |
| — GPT latent | 0.869 | 2.766 | 0.888 | 4.15 | 4.15 |
| — Training strategy | 0.773 | 1.362 | 0.689 | 3.44 | 2.82 |

Removing the three-stage GRL training costs **SS −0.063 and ES −0.198 simultaneously** — the disentanglement buys both, which is the point.

### 6.5 The steering-site table — arXiv 2607.00946 Table 2 (CosyVoice2, WavLM S-SIM)

**The cleanest controlled experiment in the literature: identical backbone, identical vectors, only the injection site and α vary.**

| Data | Config | E-SIM ↑ | TEP ↑ | ρ ↑ | H-Rate ↑ | **S-SIM ↑** | WER ↓ |
|---|---|---|---|---|---|---|---|
| CREMA-D | **No-steer** | .743 | .065 | – | – | **.871** | 1.07 |
| | CFM α=1.0 | .767 | .097 | .098 | .691 | **.858** *(−.013)* | 0.76 |
| | **CFM α=2.0** | .786 | .160 | .193 | .717 | **.807** *(−.064)* | 0.79 |
| | SLM α=3.0 | .762 | .100 | .166 | .709 | **.872** *(+.001)* | 1.01 |
| | **SLM α=5.0** | .779 | .149 | .209 | .724 | **.870** *(−.001)* | 0.78 |
| | Joint α=2.0 | .787 | .163 | .176 | .711 | **.808** *(−.063)* | 1.06 |
| IEMOCAP | **No-steer** | .903 | .197 | – | – | **.888** | 6.70 |
| | CFM α=2.0 | .909 | .272 | .117 | .721 | **.844** *(−.044)* | 6.15 |
| | **SLM α=5.0** | .915 | .253 | .215 | .755 | **.890** *(+.002)* | 6.27 |
| | Joint α=2.0 | .911 | .274 | .170 | .737 | **.845** *(−.043)* | 6.29 |

Three conclusions, all directly actionable:
1. **Decoder steering drift scales with α** (−.013 at α=1 → −.064 at α=2). **LM steering does not** (+.001 at α=3, −.001 at α=5).
2. **LM steering delivers *better* proportional control** at zero identity cost (ρ .209 vs .193).
3. **Joint steering pays the decoder's full identity cost and gets worse control than the LM alone.** Do not do it.

Corroborating sweep (CoCoEmo appendices): IndexTTS2's native `Emo-Vector` scaling 0.1 → 0.8 moves S-SIM **0.865 → 0.830** (CREMA-D) and **0.882 → 0.833** (IEMOCAP), *"particularly beyond 0.6"* — an independent third confirmation of the 0.6 ceiling. Joint steering past α=2.0 hits **WER 20.93, S-SIM 0.770–0.801** — a cliff, not a gradient.

### 6.6 PilotTTS Table 3 — the head-to-head neutral-vs-emotional comparison

**Source: arXiv 2605.27258, 51 speaker prompts (15 expressive anime/film voices, 36 ordinary).** This is the only table that turns emotion control on and off on multiple *whole systems* and reports the delta.

| Condition | VoxCPM | Fish-Speech S2 | IndexTTS (v1) | CosyVoice 3 | PilotTTS |
|---|---|---|---|---|---|
| **Without emotion control** | 0.4982 | 0.5727 | 0.7680 | 0.7963 | **0.8101** |
| **With emotion control** | 0.3361 | 0.5731 | 0.4233 | 0.6940 | **0.7329** |
| **Δ absolute** | −0.162 | +0.000 | **−0.345** | −0.102 | **−0.077** |
| **Δ relative** | **−32.5%** | 0.0% | **−44.9%** | **−12.9%** | **−9.5%** |

**This is the number to plan the product around: expect to pay ~10% speaker similarity for expressive delivery even on a well-decoupled system, and 30–45% on a badly-decoupled one.** Fish-Speech S2's 0.0% is not a win — it indicates its emotion control barely moves the output. **VoxCPM's −32.5% is a direct warning against its text-prefix style channel** (§3.3, §4) and is the only quantitative evidence we have for that family.

### 6.7 EmoSphere++ — the zero-shot penalty is the one that matters for us

**Source: arXiv 2411.02625 (TAFFC 2025), Tables II & III.** SECS_R = Resemblyzer, SECS_W = WavLM-base-sv, ECA = emotion classification accuracy.

| | **Seen speakers** SECS_AVG | ECA | | **Unseen (zero-shot)** SECS_AVG | ECA |
|---|---|---|---|---|---|
| Ground Truth | 0.8311 | 95.53 | | 0.8504 | 100.00 |
| EmoSphere++ | **0.8181** *(−1.6%)* | 93.53 | | **0.7592** *(−10.7%)* | 94.61 |

**The penalty is 6.7× larger in the zero-shot regime — which is the only regime VoiceForge operates in.** SECS_R alone falls 0.7725 → 0.6543 (−15.3%). Emotion accuracy barely moves (94.61): the model *keeps the emotion and pays in identity*. The paper's own reading: *"adapting to unseen speakers is more complex than adapting to unseen emotions."* Every number in §6.1–6.5 comes from seen-or-near-seen conditions; **assume our drift will be worse.**

Also from Table IV: the naive "multiply the emotion embedding by a scalar" approach preserves identity (SECS_AVG 0.8112) but **barely conveys emotion (ECA 40.63%)** and is "unstable when adjusted on labels such as sad". Scalar intensity on an emotion embedding is not a substitute for a proper direction vector.

### 6.8 CFG strength vs identity — ReStyle-TTS Table 4 (WavLM `Spk-sv`, mean over 10 attributes)

| Setting | Attr Δ (control effect) ↑ | WER% ↓ | **Spk-sv ↑** |
|---|---|---|---|
| λ_cfg = 2 (strong reference guidance) | 2.1% | 1.83 | **0.90** |
| λ_cfg = 0.5 | 7.6% | 2.67 | **0.85** |
| **DCFG** (λ_text=2, λ_audio=0.5) | **51.2%** | 2.31 | **0.79** |
| DCFG without timbre-consistency reward | 51.0% | 2.32 | **0.71** |
| λ_cfg = −0.5 | — | **unusable (WER > 1.0)** | — |

The paper's own verdict: *"under CFG, improving controllability inevitably degrades text fidelity, and there exists no suitable value that can simultaneously achieve both controllability and text faithfulness."* Note that their timbre-consistency reward buys back **0.08 Spk-sv (0.71 → 0.79) at zero controllability cost** — reward-shaping on speaker similarity works. **CFG is a poor primary Direction knob:** ~0.11 Spk-sv for 51% controllability, versus ~0.001 for the same effect via LM steering.

### 6.9 ⚠️ A methodological warning that invalidates naive use of every number above

**"The False Resonance: A Critical Examination of Emotion Embedding Similarity for Speech Generation Evaluation" (arXiv 2604.26347, 2026-04-29).**

- The near-universal **E-SIM / EECS metric (emotion2vec cosine)** is **contaminated by speaker identity and linguistic content**.
- **Even in the ideal speaker-and-text-matched condition, emotion2vec and emotion2vec+ reach only 60–70% triplet accuracy.**
- **Under speaker or linguistic distractors they frequently fall *below chance*.**
- Deep layers *actively degrade* alignment with human judgement.
- The authors' warning: *"If an inaccurate metric rewards sound copying over real emotional expression, models will simply learn to duplicate speaker and linguistic details."*

**Consequences for VoiceForge, and they are not optional:**
1. Every E-SIM/EECS number in §6 is partly measuring speaker copying. Treat them as *weak* evidence of emotion control. The S-SIM numbers are unaffected and remain the reliable half.
2. Our harness must **fix speaker and text across conditions** (E1 already specifies this) and pair emotion2vec cosine with an **independent SER classifier** plus human listening. Cross-ref [06-evaluation-harness.md](06-evaluation-harness.md).
3. This is also why 2606.05367's own caveat about EECS saturating at 0.85–0.97 should be taken seriously rather than treated as a formality.

### 6.10 Cross-paper comparison is invalid — read this before quoting any number

S-SIM absolute values in this section come from **five different encoders**: WavLM-base (§6.1, 6.2, 6.5), WavLM-base-plus-sv (§3.1a, §6.8), pyannote/SER (§6.3), FunASR (§6.4), Resemblyzer (§6.7), ERes2Net (SAE work). §6.1 and §6.3 differ by ~0.2 absolute on *the same model* purely from encoder choice. **Only within-table deltas are meaningful.** Report drift as a ratio against a between-identity floor (E1 step 6), never as a bare cosine.

A related confound, flagged by the accent-steering paper: *"speaker embeddings could be distant even for the same speaker with different emotions"* — so an S-SIM drop after prosody manipulation may be **metric artifact, not real timbre change**. Human listening is the tiebreaker.

### 6.11 What is missing

- **No published SS-under-instruct number for Qwen3-TTS CustomVoice or VoiceDesign.** The Base x-vector arithmetic is measured (§3.1a); the instruct variants are not. UNVERIFIED.
- **No published identity-drift number for Zonos, Chatterbox, Indic-Mio, or Indic Parler-TTS.** VoxCPM now has one (§6.6, −32.5%) and it is bad.
- **No measured F0-shift-vs-timbre-breakdown threshold for classical DSP prosody transplant** (WORLD/PSOLA). The formant-coupling concern is well known but no 2024–2026 primary source quantifies "how much pitch shift before timbre breaks". Treat any such figure as UNVERIFIED. Genuine gap in the literature.
- **EmoVoice publishes no speaker-similarity metric at all** — only WER, Emo_Sim, recall, UTMOS. Notable absence.
- No study measures drift as a function of *stylisation extremity* (aged, raspy, whispered) rather than of the six canonical emotions. Our use case is exactly the extreme end, and nobody has measured it (E9).

---

## 7. PROPOSED `Direction` SCHEMA

Design constraints: expressive enough for drama; degradable across a 10× capability range; every field must be either honoured, approximated with a recorded bound, or **rejected loudly**; nothing may silently no-op.

### 7.1 The schema

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Emotion = Literal["neutral", "happy", "sad", "angry", "afraid",
                  "disgusted", "surprised", "melancholic"]
# 8 categories: the intersection of IndexTTS2's shipped ordering and Zonos's
# vector, minus Zonos's "Other". Every audited categorical backend maps into
# this set without inventing a category.

@dataclass(frozen=True, slots=True)
class Direction:
    # ---- performance: WHAT emotion, HOW MUCH -------------------------------
    emotion: dict[Emotion, float] = field(default_factory=dict)
    #   Sparse map, each value in [0.0, 1.0]. Empty == neutral.
    #   Sum is NOT required to be 1 (mixed emotion is the point), but sum is
    #   clamped to <= 1.0 at validation. Rationale: CoCoEmo shows mixed-emotion
    #   steering is both achievable and dramatically necessary.

    intensity: float = 0.6
    #   [0.0, 1.0]. Global scale over `emotion`. DEFAULT 0.6, NOT 1.0.
    #   Justification: three independent sources converge on ~0.6-0.8 as the
    #   ceiling before identity degrades -- IndexTTS2's own emo_alpha guidance,
    #   its 0.8 sum-rescale in code, and CoCoEmo's independently chosen 0.6.

    style: str | None = None
    #   Free natural-language performance note: "through gritted teeth",
    #   "a theatrical aside", "exhausted, barely audible".
    #   MUST NOT contain identity terms. Validated against a banned-lexicon
    #   (gender, age, accent, "voice of", speaker names) -- see 7.4.

    # ---- prosody: measurable dials, all RELATIVE to this identity's neutral -
    rate: float = 1.0            # [0.25, 4.0] duration multiplier; >1 = slower
    pitch_var: float | None = None   # [0.0, 1.0] normalised F0 expressiveness
    loudness: float = 0.0        # [-1.0, +1.0]  -1 = whisper, +1 = shout

    # ---- timing ------------------------------------------------------------
    target_seconds: float | None = None      # absolute slot length
    timing_tolerance: float = 0.05           # acceptable |error| as a fraction
    pauses: tuple[tuple[int, float], ...] = ()   # (after word index, seconds)

    # ---- word level --------------------------------------------------------
    emphasis: tuple[int, ...] = ()           # word indices in `text`
    inserts: tuple[tuple[int, InsertKind], ...] = ()   # (word index, kind)

    # ---- determinism -------------------------------------------------------
    seed: int | None = None
    strict: bool = False
    #   strict=True: raise DirectionNotSupported rather than approximate or
    #   drop ANY requested field. Use for golden/regression lines.
```

```python
class InsertKind(str, Enum):
    BREATH = "breath"; SIGH = "sigh"; LAUGH = "laugh"; CHUCKLE = "chuckle"
    GASP = "gasp"; COUGH = "cough"; SNIFFLE = "sniffle"; YAWN = "yawn"

class Honouring(str, Enum):
    NATIVE   = "native"     # backend has a first-class parameter for it
    APPROX   = "approx"     # emulated; a bound must be published
    REJECT   = "reject"     # cannot be done; must raise or be dropped loudly
```

### 7.2 The capability contract — amending the `Renderer` protocol

The brief's protocol has no way to report partial honouring. Minimum amendment:

```python
class Renderer(Protocol):
    backend_id: str; backend_version: str; identity_tier: int
    languages: list[str]; public_servable: bool

    direction_support: Mapping[str, Honouring]   # NEW: field name -> Honouring
    direction_bounds: Mapping[str, str]          # NEW: field -> documented bound,
                                                 # e.g. {"rate": "0.7-1.4 before WER doubles"}

    def mint_identity(description: str, lang: str, seed: int) -> Identity: ...
    def render(identity: Identity, text: str,
               direction: Direction | None) -> Audio: ...

@dataclass
class Degradation:
    field: str
    honouring: Honouring
    requested: object
    applied: object | None
    note: str

@dataclass
class Audio:
    samples: np.ndarray; sample_rate: int
    duration_s: float
    degradations: tuple[Degradation, ...]   # NEW: empty == fully honoured
    identity_drift: float | None = None     # NEW: measured S-SIM vs the
                                            # identity's neutral anchor, when
                                            # the harness computes it
```

**Rules, non-negotiable:**
1. A backend **must** declare `Honouring` for **every** field of `Direction`. A missing key is a configuration error, not a default.
2. `REJECT` + `strict=True` → raise `DirectionNotSupported(field, backend_id)`. Never render.
3. `REJECT` + `strict=False` → render without it and append a `Degradation`. **`Audio.degradations` must never be silently discarded** by the calling layer; the pipeline logs it per line.
4. `APPROX` **requires** a published bound in `direction_bounds`. "Approximate" without a number is `REJECT`.
5. `intensity` is clamped to the backend's own ceiling; the clamp is a `Degradation` if it bites.

### 7.3 Per-backend capability matrix

`N` = native · `A` = approximated (bound required) · `R` = must reject

Two columns per Qwen3 Base: **API** = what the shipped `generate_voice_clone` exposes; **+τ** = with x-vector centroid arithmetic (§3.1a) in front of it. The `+τ` column is what we should actually build.

| Field | Qwen3 Base (API) | **Qwen3 Base (+τ)** | Qwen3 CustomVoice | CosyVoice 2/3 | Zonos | Chatterbox | Kokoro | Indic-Mio | VoxCPM2 | Indic Parler | F5-TTS | MOSS v1.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `emotion` (categorical) | **R** | **N** (τ per emotion) | N (NL) | **N** (NL) | **N** (8-vec) | A (1 scalar) | **R** | **N** (tags) | A (text) | A (**identity-unsafe**) | **R** | **R** |
| `emotion` (mixed) | **R** | **A** (Στ; untested) | A | A | **N** (simplex) | **R** | **R** | **R** | A | A | **R** | **R** |
| `intensity` | **R** | **N** (α, continuous) | A | A | **N** (vec mag) | **N** (`exaggeration`) | **R** | **R** | A | A | **R** | **R** |
| `style` (free NL) | **R** | **R** (τ is categorical) | **N** | **N** | **R** | **R** | **R** | **R** | **N** (paren) | **N** (**unsafe**) | **R** | **R** |
| `rate` | **R** | **R** | A (via NL) | **N** (`speed`) | **N** (`speaking_rate`) | A (`cfg_weight`) | **N** (`speed`) | **R** | A (via text) | A (via NL) | **N** (`speed`) | **N** (`tokens`) |
| `pitch_var` | **R** | **A** (τ side-effect) | A | A | **N** (`pitch_std`) | **R** | **R** | **R** | A | A | **R** | **R** |
| `loudness` | **R** | **R** | A | A | **R** | **R** | **R** | A (`<whisper>`, EN) | A | A | **R** | **R** |
| `target_seconds` | **R** | **R** | **R** | A (search on `speed`) | A (search) | **R** | **A→N** (`pred_dur` loop) | **R** | **R** | **R** | **N** (`fix_duration`) | **N** (`tokens`) |
| `pauses` | **R** | **R** | **R** | A (`[breath]`) | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **N** (`[pause X.Ys]`) |
| `emphasis` | **R** | **R** | **R** | **N**? (`<strong>`) | **R** | **R** | **R** | **N**? (`*w*`) | **R** | **R** | **R** | **R** |
| `inserts` | **R** | **R** | **R** | **N** (`[laughter]`…) | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** |
| `seed` | N | N | N | N | N | N | n/a (det.) | N | **A** (`retry_badcase`!) | N | N | N |

`N?` = markup exists in source/docs, reliability unmeasured — ship as `APPROX` with bound `"unvalidated"` until E4 lands.

**What the matrix says at a glance:**
- **Qwen3-TTS Base's API rejects every Direction field — but with a τ library in front of it, it gains native categorical emotion and a continuous intensity dial.** That single addition converts our primary backend from a pure timbre renderer into a usable dramatic one. It stays blind to rate, timing, emphasis and free-text style.
- **CosyVoice 2/3 is the only backend that is `N` or `A` on all eleven substantive fields.**
- **Kokoro and F5-TTS are complementary specialists**: Kokoro for cheap closed-loop timing, F5-TTS for exact absolute duration; neither does emotion.
- **MOSS-TTS v1.5 owns pauses** and is one of two backends with absolute duration.
- Indic-Mio is the only Indic option with a safe emotion channel *and* emphasis.
- **Nothing in the matrix does everything.** A drama pipeline needs 2–3 backends behind one `Direction` — which is exactly what the `Renderer` protocol plus honest degradation is for.

### 7.4 Explicit degradation rules

1. **Identity-safety veto.** Any field whose delivery channel also carries identity (Parler `emotion`/`style`, VoxCPM2 design-mode parenthetical, Orpheus name prefix) is `APPROX` **at best**, and the `Degradation.note` must say `"delivered on the identity channel; timbre may shift"`. It may never be silently `NATIVE`.
2. **`style` lexicon gate.** Before a free-text `style` reaches any backend, reject it if it contains identity terms (gender, age, accent, "voice", "sounds like", a known speaker name). This is what keeps performance direction from becoming identity redefinition — the discipline the brief demands, enforced at the boundary rather than by convention.
3. **Intensity ceiling.** Clamp `intensity` to `min(requested, backend_ceiling)`, default ceiling **0.6** (§6.2). Record a `Degradation` when it bites. Raise a backend's ceiling only on E1 evidence.
4. **Timing.** `target_seconds` on a backend with `A` is a bounded search: adjust `rate`, re-render, stop at `timing_tolerance` or 4 iterations. If unmet, emit a `Degradation` with the achieved duration — **never** time-stretch the waveform silently (it moves formants and therefore identity).
5. **`emphasis` and `inserts`** carry word indices into `text`. If the backend's frontend re-tokenises (F5-TTS pinyin conversion, Parler's separate tokenizer), indices are unmappable → `REJECT`, not a guess.
6. **Composition order.** `emotion` → `intensity` → `style` → prosodic scalars → timing → word-level. A later field never overrides an earlier one; conflicts are `Degradation`s. Specifically: if `emotion` implies a rate change and `rate` is also set, `rate` wins and the conflict is recorded.
7. **`seed` on VoxCPM2** must set `retry_badcase=False`, otherwise the seed is a lie (silent `current_seed += 1`). If the caller wants both retries and determinism → `REJECT`.
8. **Neutral anchor.** Every identity stores a **neutral render** of a fixed calibration sentence, produced with `Direction=None`. All drift measurement is against that anchor. Without it §6-style numbers cannot be computed at runtime.

---

## 8. What this means for the build

1. **Build the τ library. It is the highest value-per-hour item in this entire research pass.** Qwen3-TTS Base has no Direction channel in its API, but arXiv 2606.05367 demonstrates one *inside the x-vector*, on our exact checkpoint, training-free. Concretely: download ESD, pick ≥4 gender-balanced source speakers, extract 50 x-vectors per speaker per emotion with `model.extract_speaker_embedding()`, store 8 mean-difference vectors `τ_emo ∈ ℝ^2048`, and render with `x = identity + α·τ`. **Total artefact: 8 × 2048 floats = 64 KB.** Expected result: ΔEECS ≈ +0.29 at SECS_W ≈ 0.912. Use the **multi-speaker** average — the single-speaker variant costs 0.102 SECS_W for no gain. This converts our primary backend from a flat renderer into a directed one for the price of an afternoon.
2. **Restate the identity/emotion principle as an engineering rule, not a claim about the representation.** The brief asserts "emotion must NOT be baked into the identity vector." On Qwen3-TTS Base emotion demonstrably *is* in the x-vector and dominates the codec tokens. The principle is still achievable, but by **discipline rather than by architecture**: (a) mint identities only from neutral reference audio; (b) store the neutral x-vector as the identity record; (c) express Direction as an additive `α·τ` applied at render time and never persisted. Under those three rules the identity record contains "timbre + neutral prosody", the Direction is separable and reversible, and the brief's API-level separation holds. Write this into [02-identity-representation.md](02-identity-representation.md) — it changes what `mint_identity` must guarantee.
3. **Qwen3-TTS Base still cannot be the whole story.** Even with τ it is blind to rate, absolute timing, emphasis, pauses and free-text style. It is an excellent identity renderer (Apache-2.0, real `(2048,)` vector, `x_vector_only_mode`, best-in-class WER 0.77/1.24 on Seed-TTS) with a *categorical emotion dial bolted on*. Drama needs more. Plan for a second renderer.
4. **CosyVoice 2/3 is the strongest Direction backend and should be evaluated as a co-primary.** Apache-2.0, persistent identities (`add_zero_shot_spk`), separate `instruct_text`, `speed`, `<strong>` emphasis, paralinguistic tokens, phoneme inpainting, and the **only published identity-drift number for its own instruct channel (−0.02 S-SIM)**. Its weakness is that identity is a reference wav, not a mintable vector (cross-ref [02-identity-representation.md](02-identity-representation.md)) — but Qwen's own "Voice Design then Clone" pattern shows the bridge: mint with a designer, capture a clip, clone from it.
5. **Adopt the SLM/decoder principle as an architectural filter.** Prefer backends where a sequence model produces prosody-bearing tokens and a separate acoustic module applies timbre. CoCoEmo proved emotion lives in the SLM and the decoder only renders; every identity-safe backend in this audit has that shape. Make it a selection criterion, not an afterthought.
6. **Plan for activation steering as the Pass-2 Direction mechanism.** It is the only technique that delivers strong control at ~0.001 S-SIM cost on a frozen backend. Prerequisites: a hookable backend (CosyVoice2 and F5-TTS both demonstrated), ESD + RAVDESS + CREMA-D (all publicly obtainable), and a layer sweep. Budget it as a real work item, not a stretch goal. **Extract the vectors from a permissively-licensed backend only** — never from IndexTTS2 (§2.3).
7. **Default `Direction.intensity` to 0.6, not 1.0.** Three independent sources converge there. A schema whose default is the failure point is a bad schema.
8. **Use emotion *erasure* at mint time — the τ library gives it to you for free.** If a reference clip carries incidental emotion, that emotion enters the x-vector (§3.1a) and is baked into the identity record. Two published operators remove it: subtract the matching `τ_emo` (once the library exists, this is one line), or use EmoSteer-TTS's projection operator `x̂ = f_r(x − β(ŝ·x)ŝ)`. This is the concrete way to honour the brief's principle at the *representation* level rather than by convention alone. Cross-ref [02-identity-representation.md](02-identity-representation.md). Test as E7.
9. **Indic track: demote Parler to designer-only; make Indic-Mio the renderer.** Parler has no speaker embedding and the worst instruction-following scores in Qwen's benchmark. Indic-Mio has separate channels, emotion tags, `<whisper>`, and `*word*` emphasis. Add **MioCodec `voice_conversion` as the identity-repair post-pass** for anything rendered on a one-channel backend. Cross-ref [04-indic-track.md](04-indic-track.md).
10. **Timing is a two-backend problem.** No backend we favour hits an absolute duration. If lip-sync slots are a real requirement, either F5-TTS or MOSS-TTS v1.5 must be in the stack for those lines, or the pipeline accepts closed-loop search (2–4 renders) on `rate`. Kokoro's returned `pred_dur` makes it the cheapest place to prototype the search loop. Decide this before the API freezes — `target_seconds` is either in the schema honestly or not at all.
11. **Exclude IndexTTS2/2.5, Higgs Audio 2, and XTTS-v2 from the codebase.** IndexTTS2 for §3.4(c) contamination risk against mapper training (§2.3); Higgs for a Llama-derived community licence; XTTS-v2 for CPML non-commercial. Cross-ref [08-licensing-propagation.md](08-licensing-propagation.md).
12. **The evaluation harness needs a drift metric now.** Every §6 number is a `S-SIM(directed, neutral_anchor)` cosine over an **independent** speaker encoder (WavLM-base or ECAPA — never the backend's own encoder, which is circular). Add it to [06-evaluation-harness.md](06-evaluation-harness.md) alongside a between-identity floor so drift is reported as a *ratio*, not a bare cosine. Bare cosines are not comparable across encoders — §6.1 and §6.3 differ by 0.2 absolute purely from encoder choice.
13. **Amend the `Renderer` protocol** with `direction_support`, `direction_bounds`, and `Audio.degradations` (§7.2). Without these the brief's "degrade EXPLICITLY, never silently" is a comment, not a contract.
14. **Do not trust emotion2vec cosine as the primary control metric.** arXiv 2604.26347 shows it scores 60–70% triplet accuracy even under ideal matched conditions and drops *below chance* with speaker or linguistic distractors — it partly measures speaker copying. Pair it with an independent SER classifier and human listening, and hold speaker and text fixed across conditions (§6.9). This affects [06-evaluation-harness.md](06-evaluation-harness.md) more than it affects this document.
15. **Budget the identity cost honestly in the product spec.** PilotTTS's head-to-head (§6.6) says expressive delivery costs ~10% speaker similarity even on a well-decoupled system. EmoSphere++ (§6.7) says the penalty is ~6.7× worse in the zero-shot regime, which is the only regime we operate in. A promise of "the same character voice across every line, at full dramatic range" is not deliverable at current state of the art. Decide now what the product claims.
16. **Note what we are giving up by excluding IndexTTS2.** TED-TTS (arXiv 2601.03170, public code) is the only published method offering *intra-utterance* per-segment emotion **and** duration on a frozen backend, and it reports SSIM 0.485 vs 0.457 baseline — it *improves* speaker similarity. It is built on IndexTTS2 and is therefore §3.4(c)-blocked for us. The idea (segment-level direction with a shared speaker embedding, EOS-logit modulation for a token budget) is worth reimplementing on a permissively-licensed backend; the implementation is not usable.

---

## 9. Open — must be settled by experiment

| # | Question | Cheapest experiment | Est. cost/time | What it blocks |
|---|---|---|---|---|
| **E0** | **Does x-vector centroid arithmetic reproduce on our install?** | §9.0 below — **run this first** | **~4 h** | Whether Qwen3-TTS Base gets a Direction channel at all; the whole English track's expressiveness |
| **E1** | Does style conditioning move the speaker? (per backend) | §9.1 below | ~2 GPU-h/backend + 1 day to build | The entire `Direction` capability matrix; whether identity records are stable |
| **E2** | Can `instruct_ids` be smuggled into Qwen3-TTS **Base**? | §9.2 below | 2–3 h | Whether our primary backend can have Direction at all |
| **E3** | Is Kokoro's `ref_s[:,:128] / [128:]` split perceptually clean? | §9.3 below | 3 h | A free Direction channel on a Tier-1 identity backend |
| **E4** | Does `<strong>` (CosyVoice2/3) / `*word*` (Indic-Mio) actually emphasise? | 100 sentences × {marked, unmarked}; force-align with MFA or WhisperX; measure Δduration, Δpeak-F0, Δenergy on the target word vs its unmarked control. "Reliable" = significant consistent shift in ≥80% of cases with no spillover to neighbours. | 1 day | Whether `emphasis` is `NATIVE` or `REJECT` in the matrix |
| **E5** | How precise is absolute duration control? | F5-TTS `fix_duration` and MOSS `tokens=N` at {0.75, 0.875, 1.0, 1.125, 1.25}× natural on 200 lines; measure \|actual − target\| and WER. Mirrors IndexTTS2's Table 4 so results are comparable. | 4 h | `target_seconds` honouring; the lip-sync story |
| **E6** | Does activation steering port to our chosen backend? | Reproduce CoCoEmo on CosyVoice2 (their code, their datasets), then repeat the layer/operation probe on our backend. Success = TEP up, S-SIM within 0.005. | 3–5 days | The Pass-2 Direction mechanism |
| **E7** | Does emotion erasure improve minted identity stability? | Extract speaker vectors from N clips with and without EmoSteer-style erasure; measure within-identity variance across emotional source clips. | 1 day | Whether the identity vector is emotion-contaminated (cross-ref 02) |
| **E8** | Does MioCodec `voice_conversion` repair identity without killing prosody? | Render directed lines on Parler; convert to the target `global_embedding`; measure S-SIM (should rise) and F0-contour correlation to the pre-conversion render (should stay high). | 1 day | The Indic one-channel mitigation |
| **E9** | Does drift behave differently for *stylisation* (aged, raspy, whispered) than for the 6 canonical emotions? | Extend E1's direction set with the brief's heavy-stylisation vocabulary. Nobody has published this. | +1 h on E1 | Whether §6's numbers generalise to our actual use case |

### 9.0 E0 in full — reproduce x-vector task arithmetic on Qwen3-TTS Base

**Run this before anything else. It is four hours and it decides whether our primary backend can act.**

*Materials.* ESD (Emotional Speech Database, public); `Qwen/Qwen3-TTS-12Hz-1.7B-Base`; `microsoft/wavlm-base-plus-sv` for the independent metric.

*Procedure.*
1. Pick four gender-balanced ESD source speakers. The paper used **{0011, 0014, 0017, 0020}**; reuse them so results are directly comparable.
2. For each speaker × each emotion (incl. neutral), take **50 utterances**, resample to **24 kHz**, and extract `x = model.extract_speaker_embedding(audio, ...)` → `(2048,)`.
3. Compute centroids and `τ_emo = mean_over_speakers(centroid_emo) − mean_over_speakers(centroid_neutral)`. Save 8 vectors. **Artefact: 64 KB.**
4. Sanity-check the geometry before rendering anything: `‖τ‖ / ‖x‖` should be ≈ 0.15, and `|proj(τ, identity_axis)| / ‖τ‖` should be < 0.01. **If these do not hold, stop — the method will not work and something is wrong with the extraction.**
5. Take an unseen target speaker; build `x_new = x(target, neutral) + α·τ` for **α ∈ {0, 0.5, 1, 1.5, 2, 2.5}**.
6. Render via `create_voice_clone_prompt(..., x_vector_only_mode=True)` with the x-vector overridden, 30 held-out sentences per combination.
7. Measure SECS_W (WavLM, independent), an SER classifier score (not emotion2vec alone — §6.9), WER, UTMOS.
8. **Run the avg4spk vs single-speaker contrast.** This is the cheapest part and the highest-value result: expect ≈ +0.10 SECS_W for the multi-speaker τ at equal emotion gain. Confirming it locally validates the whole extraction.

*Pass criteria.* SECS_W ≥ 0.88 at the α that maximises SER score, with WER within 2pp of the α=0 baseline. Then set `Direction.intensity → α` mapping from the sweep and fill in `direction_bounds` for Qwen3-TTS Base.

*Extensions, nearly free once the harness exists.* (a) Does τ compose? `x + α₁τ_angry + α₂τ_sad` for mixed emotion — untested in the literature, and mixed emotion is what drama needs. (b) Does τ extend to the brief's heavy stylisations (whispered, raspy, aged) if we build τ from a stylised corpus rather than ESD? Also untested. (c) Does τ transfer across languages for our Indic track, as it did EN→PT-BR?

### 9.1 E1 in full — "does style conditioning move the speaker embedding"

**The decisive cross-backend experiment. Run it immediately after E0 — E0's harness is most of E1's harness.**

*Hypothesis.* For a backend with separate identity and style channels, `S-SIM(render(id, text, d), render(id, text, None))` stays near 1.0 for all directions `d`. For a one-channel backend it degrades toward the between-identity floor as direction strength rises.

*Materials.*
- **12 identities** per backend, spanning the space we actually ship: 3 male / 3 female / 2 child / 2 aged / 2 heavily stylised (raspy, breathy). Minted via that backend's normal path.
- **5 texts**, 8–15 words, emotionally neutral in content (so text sentiment cannot confound), fixed across all conditions.
- **9 directions**: `None` (the anchor) plus 8 = {happy, sad, angry, afraid, surprised, whispered, shouted, aged/theatrical}, each at `intensity ∈ {0.2, 0.4, 0.6, 0.8, 1.0}` for the sweep arm.
- **Independent speaker encoder.** WavLM-base-plus-sv or SpeechBrain ECAPA-TDNN. **Never the backend's own encoder** — that measures the encoder's invariance, not the render's.

*Procedure.*
1. Render the anchor `A0(id, text)` with `Direction=None`, fixed seed.
2. Render `A1(id, text, d, i)` for each direction and intensity, same seed.
3. Embed everything with the independent encoder.
4. **Within-identity similarity** `W = cos(emb(A0), emb(A1))`.
5. **Between-identity floor** `F = mean over i≠j of cos(emb(A0_i), emb(A0_j))` — the similarity of *genuinely different* speakers on this backend and this encoder. Without `F` a raw cosine is meaningless.
6. **Identity Drift Ratio** `IDR = (1 − W) / (1 − F)`. Scale-free, comparable across backends and encoders.
7. Fit `IDR ~ intensity` per direction; report the slope. A backend with a **flat** slope has genuine disentanglement; a rising slope quantifies the trade-off and sets the intensity ceiling.
8. Secondary: WER (Whisper-large-v3) to catch intelligibility collapse; E-SIM (emotion2vec) to confirm the direction actually *did* something — a backend that ignores Direction trivially scores IDR ≈ 0 and must not be credited for it.

*Pass criteria.* `IDR < 0.15` at `intensity = 0.6` **and** a measurable E-SIM shift → the field is `NATIVE`. `0.15 ≤ IDR < 0.35` → `APPROX`, and the intensity ceiling is set where IDR crosses 0.15. `IDR ≥ 0.35` → `REJECT`, or route to the MioCodec repair post-pass (E8).

*Cost.* 12 × 5 × 9 = **540 renders** for the main arm, plus 12 × 5 × 8 × 4 = 1,920 for the intensity sweep. At 1–3 s per render on one 8–12 GB GPU that is **1–2 GPU-hours per backend**. Embedding extraction is minutes. Build cost: one day for the harness, then it is reusable for E3, E4, E9 and for regression testing every backend upgrade.

*What it blocks.* Everything. The capability matrix in §7.3 is currently populated from architecture and documentation; §6's numbers come from other people's backends on other people's test sets. Until E1 runs, **every `NATIVE`/`APPROX` call in the matrix is a hypothesis**, and `direction_bounds` cannot be filled in — which means rule 4 of §7.4 cannot be satisfied and no backend can honestly declare `APPROX`.

### 9.2 E2 in full — smuggling `instruct_ids` into Qwen3-TTS Base

`generate_voice_clone` never passes `instruct_ids`, but `self.model.generate` clearly accepts it (both other paths do). Two things must be true for Base to gain a Direction channel: the argument must be *accepted*, and the Base *weights* must respond to it.

```python
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", ...)
prompt = m.create_voice_clone_prompt(ref_audio=REF, ref_text=None, x_vector_only_mode=True)
d = m._prompt_items_to_voice_clone_prompt(prompt)
ids = m._tokenize_texts([m._build_assistant_text("I told you not to come back.")])
instr = m._tokenize_texts([m._build_instruct_text("Say this in a furious whisper.")])

a = m.model.generate(input_ids=ids, ref_ids=None, voice_clone_prompt=d,
                     languages=["English"], non_streaming_mode=False)          # control
b = m.model.generate(input_ids=ids, ref_ids=None, voice_clone_prompt=d,
                     languages=["English"], instruct_ids=instr, non_streaming_mode=False)
```

Outcomes: **(i)** `TypeError` → the shared `generate` is not actually shared; Base is closed, full stop. **(ii)** Accepted, `a == b` bit-for-bit → the argument is ignored on a `base` checkpoint; Base is closed. **(iii)** Accepted and `a != b` → run a 20-line E1 to check whether the difference is *the requested* direction or just noise. Only outcome (iii) with a real E-SIM shift changes the architecture. **2–3 hours.** Do this before committing to a second English backend.

### 9.3 E3 in full — Kokoro's latent split

Load `af_heart.pt` (voice A) and a contrasting voice B. For a fixed phoneme string of length L, construct:
- baseline `[A[L-1,:, :128] | A[L-1,:,128:]]`
- **prosody swap** `[A[L-1,:, :128] | B[L-1,:,128:]]`
- **timbre swap** `[B[L-1,:, :128] | A[L-1,:,128:]]`

Render all three. Measure S-SIM against baseline (independent encoder) and prosodic distance (F0 mean/std, `pred_dur` total, energy). **Prediction if the split is clean:** prosody swap → S-SIM high, prosody distance high; timbre swap → S-SIM low, prosody distance low. If confirmed, interpolate `(1−λ)A[128:] + λB[128:]` for `λ ∈ [0,1]` to get a **continuous, identity-preserving prosody dial on a frozen Apache-2.0 backend**, at the cost of a tensor slice. Respect the length indexing (`pack[len(ps)-1]`) throughout. **3 hours.**

---

## 10. Sources

| # | URL | Type | Used for | Confidence in source |
|---|---|---|---|---|
| 1 | https://arxiv.org/abs/2506.21619 · https://arxiv.org/html/2506.21619v2 | Paper (arXiv, v2 2025-09-03) | IndexTTS2 architecture, GRL, three-stage training, T2E, duration control, Tables 2 & 4 | HIGH |
| 2 | https://raw.githubusercontent.com/index-tts/index-tts/main/LICENSE | LICENSE file, fetched directly | bilibili licence §1.6, §2.2, §3.4(c), §5.3, §6, §9 | HIGH |
| 3 | https://raw.githubusercontent.com/index-tts/index-tts/main/indextts/infer_v2.py | Source code | `infer()` signature, `emo_vector`, `emo_alpha`, `QwenEmotion`, emo_bias, 0.8 rescale | HIGH |
| 4 | https://raw.githubusercontent.com/index-tts/index-tts/main/README.md | Official README | IndexTTS-2.5 release, `duration_factor`, "not yet enabled in this release", model zoo | HIGH |
| 5 | https://arxiv.org/abs/2601.03888 | Paper (arXiv, v5 2026-08-11) | IndexTTS-2.5 existence, 25 Hz codec, Zipformer, GRPO, RTF | HIGH |
| 6 | https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/qwen_tts/inference/qwen3_tts_model.py | Source code | `generate_voice_clone` has no `instruct`; `_build_instruct_text`; `ref_code=None` under `x_vector_only_mode` | HIGH |
| 7 | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/README.md | Model card | Usage, CustomVoice speakers, "Voice Design then Clone", InstructTTSEval & Seed-TTS tables | HIGH |
| 8 | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/config.json · .../VoiceDesign/... · .../CustomVoice/... | Config files | `speaker_encoder_config` present only in Base; `tts_model_type` values | HIGH |
| 9 | https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/conditioning.py | Source code | `make_cond_dict` full signature, emotion order & renormalisation, entanglement comment, conditioner classes | HIGH |
| 10 | https://raw.githubusercontent.com/Zyphra/Zonos/main/LICENSE · README.md | LICENSE + README | Apache-2.0; control claims | HIGH |
| 11 | https://raw.githubusercontent.com/resemble-ai/chatterbox/master/src/chatterbox/tts.py · gradio_tts_app.py · README.md · LICENSE | Source + README + LICENSE | `emotion_adv`, exaggeration 0.25–2.0, cfg_weight 0–1, rate coupling, MIT, Multilingual V3 | HIGH |
| 12 | https://raw.githubusercontent.com/hexgrad/kokoro/main/kokoro/model.py · kokoro/pipeline.py | Source code | `ref_s[:, :128]` / `[128:]` split, `speed`, `pred_dur`, `load_voice` averaging | HIGH |
| 13 | https://raw.githubusercontent.com/huggingface/parler-tts/main/parler_tts/modeling_parler_tts.py | Source code | description → cross-attention; prompt → prepended embeds; **zero** speaker-embedding code | HIGH |
| 14 | https://huggingface.co/ai4bharat/indic-parler-tts/resolve/main/README.md | Model card | one description string, 69 speakers, 10 emotion languages, control attributes | HIGH |
| 15 | https://huggingface.co/SPRINGLab/Indic-Mio/resolve/main/README.md | Model card | `<happy>` etc. tags, `*word*` emphasis, MioCodec pipeline, Apache-2.0 | HIGH |
| 16 | https://huggingface.co/Aratako/MioCodec-25Hz-24kHz/resolve/main/README.md | Model card | content tokens vs `global_embedding`, `decode()`, `voice_conversion()`, MIT | HIGH |
| 17 | https://huggingface.co/openbmb/VoxCPM2/resolve/main/README.md | Model card | `text="(style)content"`, controllable cloning, "results may vary", Apache-2.0, 30 languages | HIGH |
| 18 | https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/src/voxcpm/core.py | Source code | `_generate` signature: no style/instruct parameter; `retry_badcase` defaults | HIGH |
| 19 | https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/cli/cosyvoice.py | Source code | `inference_instruct2`, `speed`, `add_zero_shot_spk` | HIGH |
| 20 | https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/tokenizer/tokenizer.py | Source code | `<strong></strong>`, `[laughter]`, `[breath]`, CMU/Pinyin tokens in CosyVoice2 & 3 tokenizers | HIGH |
| 21 | https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/resolve/main/README.md | Model card | CosyVoice3 existence, instruct usage, `<\|endofprompt\|>`, Apache-2.0 | HIGH |
| 22 | https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/utils_infer.py | Source code | `fix_duration` → frames, `speed`, the `<10 byte → speed 0.3` override | HIGH |
| 23 | https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5/resolve/main/README.md | Model card | `tokens=N`, `[pause X.Ys]`, Pinyin/IPA, Apache-2.0 | HIGH |
| 24 | https://raw.githubusercontent.com/canopyai/Orpheus-TTS/main/README.md | Official README | emotive tags, `{name}: text` prefix | HIGH |
| 25 | https://raw.githubusercontent.com/SesameAILabs/csm/main/generator.py | Source code | `generate()` signature, `max_audio_length_ms` is a cap | HIGH |
| 26 | https://huggingface.co/bosonai/higgs-tts-2-3b-base/resolve/main/LICENSE · README.md | LICENSE + card | Llama-3-derived community licence; `scene` role; EmergentTTS-Eval win rates | HIGH |
| 27 | https://huggingface.co/coqui/XTTS-v2/resolve/main/LICENSE.txt | LICENSE file | Coqui Public Model License 1.0.0 — non-commercial | HIGH |
| 28 | https://arxiv.org/abs/2508.03543 · https://arxiv.org/html/2508.03543v2 | Paper (arXiv, v2 2025-08-06) | EmoSteer-TTS: training-free activation steering, hook implementation, erasure operator, Table 1 | HIGH |
| 29 | https://arxiv.org/abs/2602.03420 · https://arxiv.org/html/2602.03420v2 | Paper (arXiv) | CoCoEmo: SLM-vs-flow analysis, layer probing, Tables 2 & 3 S-SIM, IndexTTS2 0.6 ceiling | HIGH |
| 30 | https://arxiv.org/abs/2605.24618 | Paper (arXiv, 2026-05-23) | FC-TTS — dual-reference disentangled control (training-based, excluded) | HIGH (exists) |
| 31 | https://arxiv.org/abs/2506.05688 | Paper (arXiv, v3 2026-02-18) | Voice Impression Control — GRL + LLM-generated impression vector (training-based) | HIGH (exists) |
| 32 | https://arxiv.org/abs/2505.23009 | Paper (arXiv) | EmergentTTS-Eval — benchmark for expressiveness, model-as-judge | HIGH |
| 33 | **https://arxiv.org/abs/2606.05367** · https://arxiv.org/html/2606.05367v1 | **Paper (arXiv, 2026-06-03) — read in full** | **Task-vector arithmetic on Qwen3-TTS-12Hz-1.7B-Base**: four-operand elimination study, `full_swap` dissociation, τ construction, avg4spk vs single (SECS_W 0.912 vs 0.810), τ geometry (15% norm, <1% identity projection), EN→PT-BR transfer | **HIGH** |
| 34 | **https://arxiv.org/abs/2607.00946** | **Paper (arXiv, 2026-07-01) — Table 2 read** | **Steering-site comparison on CosyVoice2**: SLM vs CFM probe accuracy, within–cross gap, LID, and the S-SIM cost of each site (−0.064 CFM vs ±0.001 SLM); the speaker-conditioning mechanism | **HIGH** |
| 35 | **https://arxiv.org/abs/2605.27258** | **Paper (arXiv) — Table 3 read** | **PilotTTS head-to-head neutral-vs-emotional SS** across VoxCPM, Fish-Speech S2, IndexTTS, CosyVoice 3, PilotTTS (−9.5% to −44.9%) | **HIGH** |
| 36 | **https://arxiv.org/abs/2604.26347** | **Paper (arXiv, 2026-04-29)** | **"The False Resonance"** — emotion2vec cosine is contaminated by speaker and content; 60–70% triplet accuracy at best, below chance under distractors | **HIGH** |
| 37 | https://arxiv.org/abs/2411.02625 | Paper (TAFFC 2025), Tables II–IV | EmoSphere++ seen vs unseen SECS (−1.6% vs −10.7%); scalar-intensity failure mode (ECA 40.63) | HIGH |
| 38 | https://arxiv.org/abs/2601.03632 | Paper (arXiv, 2026-01-07), Table 4 | ReStyle-TTS — CFG↔identity trade-off (Spk-sv 0.90→0.79), DCFG, orthogonal LoRA fusion, timbre-consistency reward | HIGH |
| 39 | https://arxiv.org/abs/2601.03170 · https://github.com/Simon-leong/TED-TTS | Paper + code (2026-01-06) | TED-TTS — training-free intra-utterance per-segment emotion + duration on IndexTTS2; SSIM 0.485 vs 0.457 | HIGH |
| 40 | https://arxiv.org/abs/2606.01479 | Paper (arXiv, 2026-05-31) | SAE emotion control on IndexTTS2 layer-16 residual; 10.5M-param SAE (~40 MB); bidirectional induction/suppression | HIGH |
| 41 | https://arxiv.org/abs/2601.22873 | Paper (arXiv, 2026-01-30) | EmoShift — 10M-param steering layer on frozen CosyVoice-300M-Instruct; SpkSIM 82.23→82.41 | MEDIUM |
| 42 | https://arxiv.org/abs/2606.00066 | Paper (arXiv, 2026-05-20) | DUET — emotion = 8.5% of hidden-state variance; emotion⊥speaker \|cos θ\|=0.029 on F5-TTS. **Reports no SECS — identity claim is geometric only** | MEDIUM (existence HIGH, identity claim LOW) |
| 43 | https://arxiv.org/abs/2603.05977 | Paper (arXiv, 2026-03-06) | Activation steering for accent on **Qwen3-TTS**; layer sweep (15, 20 best); Spk-Sim 0.84→0.76 at α=2.0 | MEDIUM |
| 44 | https://www.isca-archive.org/interspeech_2025/shim25_interspeech.pdf | Paper (Interspeech 2025, ISCA archive) | Measured failure of punctuation control on 5 open systems; 100-sentence fine-tune fixes list-commas (11.26→101.55 ms, p<.01) not vocatives (p=0.89) | HIGH |
| 45 | https://arxiv.org/abs/2410.00316 · https://github.com/tonychenxyz/emoknob | Paper (EMNLP 2024) + code | EmoKnob — training-free emotion direction in MetaVoice speaker space; the 2024 precursor to #33 | MEDIUM |
| 46 | https://arxiv.org/abs/2607.03666 · https://arxiv.org/abs/2606.07293 | Papers (arXiv, 2026) | TRACE-EVC (relative NL instructions, SECS 0.68) and TargetSEC (in-the-wild EVC, SECS 0.29 vs 0.58 GT ceiling) — EVC post-pass identity cost | MEDIUM |
| 47 | https://arxiv.org/abs/2608.25569 · https://arxiv.org/abs/2603.22252 · https://arxiv.org/abs/2606.16417 | Papers (arXiv, 2026) | Latent vector steering; SelfTTS (GRL + contrastive, CKA≈0.014); Joycent — training-based, excluded | MEDIUM (existence HIGH) |
| 48 | https://arxiv.org/abs/1803.09017 · https://arxiv.org/abs/1803.09047 | Papers (2018) | GST and Tacotron prosody transfer — the origin of style tokens; jointly trained, hence unusable on a frozen backbone | HIGH |
| 49 | https://arxiv.org/abs/2403.16973 · https://github.com/jasonppy/VoiceCraft | Paper (ACL 2024) + code | VoiceCraft — speech editing as a post-pass; noted as least stable of five systems in source #44 | MEDIUM |
| 50 | https://huggingface.co/api/models?search=… | HF API | Repo existence/licence tags for VoxCPM2/1.5, CosyVoice3, IndexTTS-2.5, MOSS, Dia2, Zonos last-modified | HIGH |
