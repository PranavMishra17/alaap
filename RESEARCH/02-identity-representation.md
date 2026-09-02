# 02 — Voice Identity Representation (Tier 1 vs Tier 2) & the Zonos Contract

> **Domain:** identity persistence; speaker-embedding manifolds; Zonos speaker-embedding contract
> **Answers:** A1 (decisive), A2 (MOST decisive), C2
> **Date:** 2026-09-02 · Pass 1
> **Cross-refs:** [00-EXECUTIVE-VERDICT.md](00-EXECUTIVE-VERDICT.md) · [01-ttv-landscape.md](01-ttv-landscape.md) · [03-tts-backends-english.md](03-tts-backends-english.md) · [06-evaluation-harness.md](06-evaluation-harness.md) · [08-licensing-propagation.md](08-licensing-propagation.md)

## 0. Bottom line

- **The two-tower design survives — and it is no longer speculative. It has been published twice.** **PromptTTS 2** (arXiv 2309.02285, Microsoft) trains a diffusion "variation network" that generates a voice representation *from a text prompt with no reference audio at inference*, and **TacoSpawn** (arXiv 2111.05095, Google) samples novel speaker embeddings from a learned prior at near-parity MOS (3.62 generated vs 3.68 real). Description → vector → frozen renderer is a demonstrated architecture. **[HIGH]**
- **But both papers agree on the failure mode the brief does not mention: you must sample from a *learned prior fitted to the real speaker distribution*, never from a standard normal or a uniform hypersphere.** TacoSpawn explicitly reports that a standard-normal prior "performed worse". This is the single most important design constraint in this document. **[HIGH]**
- **The brief targets the wrong model.** Zonos-v0.1's contract is real and essentially as described (128-d, 256→128 LDA, verified from the checkpoint itself), but v0.1's last commit was **2025-03-05**, it has **zero releases**, and it was superseded in June 2026 by **ZONOS2** — which the brief does not mention. **[HIGH]**
- **🚩 Zonos-v0.1's speaker encoder is licence-poisoned, which breaks the brief's core premise.** Zyphra tag it `apache-2.0`, but their own model card says the weights are the **VoxBlink2 "pretrain" models**, and **VoxBlink2 is CC-BY-NC-SA-4.0 (non-commercial)**. The upstream checkpoints carry *no* licence statement, and the ecosystem convention (WeSpeaker: "the pretrained model follows the license of its corresponding dataset") propagates the dataset licence. Re-hosting under Apache does not cure this. **Public hosting = commercial use, so Tier 1 on Zonos-v0.1 is not viable as-is.** **[HIGH on facts, MEDIUM on the legal conclusion — needs counsel]**
- **ZONOS2 fixes exactly that.** MIT code + Apache-2.0 weights, and its speaker encoder traces cleanly: `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` (Apache-2.0, **12M params / 24 MB**, ECAPA-TDNN, `enc_dim 2048`) ← `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (Apache-2.0). It also **drops the GPL-3.0 eSpeak/phonemizer dependency** that v0.1 carries. **[HIGH]**
- **The decisive navigability sub-question is ANSWERED affirmatively for ZONOS2 — by Zyphra themselves.** The official server ships **SLERP interpolation between speaker embeddings** (`_slerp_embeddings`, exposed as `speaker_blend_t` and a UI slider) and **additive emotion "direction" vectors** built as `mean(emotion) − mean(neutral)`. Vector arithmetic in this space is a shipped, perceptually-calibrated product feature. **[HIGH]**
- **Nobody — including Zyphra — samples random speaker vectors.** Every published path starts from real extracted embeddings and blends or perturbs them. Description → *de novo* vector on Zonos specifically is unprecedented; we would be first. **[HIGH on the absence]**
- **A1 verdict: Tier 1 canonical, Tier 2 as a mandatory durability anchor — not an either/or.** Zyphra's own production dataclass carries `source_type: Literal["audio","embedding_file"]` and stores the embedding *and* the audio; their shipped default voices are MP3s. Meanwhile the 2026 open-source SOTA, **VoiceSculptor** (arXiv 2601.10629 — **it exists**), deliberately chose pure Tier 2. Store both. **[HIGH]**
- **XTTS-v2 is disqualified and the brief should stop counting on it.** Code is MPL-2.0 but the **weights are CPML — explicitly non-commercial** ("This license allows only non-commercial use"), Coqui is defunct (`coqui.ai/cpml` 404s — there is no licensor left to grant an exception), and the maintained `idiap` fork ships the same non-commercial registry. YourTTS is worse: **CC BY-NC-ND** (non-commercial *and* no-derivatives). **[HIGH]**
- **VRAM is an unsolved architectural risk.** v0.1 fits 12GB trivially (4,686 MiB measured). **ZONOS2 in bf16 is 15.34 GB and does NOT fit.** The GGUF quants that do fit (q6_k 6.79 GB) run only on `zonos2.cpp` — **which has no LICENSE file at all**. **[HIGH]**
- **Indian languages: v0.1 is not viable** (trained on 6 languages; Zyphra: performance elsewhere "is not robust"). ZONOS2 lists Hindi/Tamil/Telugu/Bengali at **Tier 3**, its weakest, with no published numbers. Clean-licensed alternatives with real Indian coverage exist: **IndicF5** (MIT, 11 Indian languages), **Indic Parler-TTS** (Apache-2.0), **Kokoro** (Apache-2.0, 4 Hindi voices). **[HIGH]**

---

## 1. Corrections to VOICEFORGE-SCOPE.md

| # | Scope claim | Verdict | Correction | Primary source | Confidence |
|---|---|---|---|---|---|
| 1 | Zonos-v0.1 is Apache-2.0 (unverified) | **CONFIRMED for the TTS — but NOT for the speaker encoder** | Code LICENSE is verbatim Apache-2.0 and all three HF repos tag `apache-2.0`. **However** the speaker-embedding weights derive from **VoxBlink2, whose dataset is CC-BY-NC-SA-4.0**, and the upstream checkpoints carry no licence at all. See §2.1. | [LICENSE](https://raw.githubusercontent.com/Zyphra/Zonos/main/LICENSE); [VoxBlink2 asv README](https://raw.githubusercontent.com/VoxBlink2/ScriptsForVoxBlink2/main/asv/README.md) | HIGH / MEDIUM legal |
| 2 | 1.6B params | **CONFIRMED** | safetensors header: exactly **1,624,411,136**, BF16, 3.25 GB. | [HF API](https://huggingface.co/api/models/Zyphra/Zonos-v0.1-transformer) | HIGH |
| 3 | `make_speaker_embedding(wav, sr)` returns a 128-d vector | **CONFIRMED, two corrections** | Returns shape **(1, 128)**, dtype **bfloat16**, and is **NOT L2-normalised** — raw LDA output. Both matter for storage and synthesis. | [`model.py:90-95`](https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/model.py) | HIGH |
| 4 | ResNet293-SimAM-ASP from VoxBlink2, **256→128** via LDA | **CONFIRMED** | `ResNet293_based(embd_dim=256)`. I downloaded the LDA checkpoint and read its pickle metadata: `weight` = **DoubleStorage, shape (128, 256)**, `bias` (128,). Exactly 256→128, float64. *(A `ECAPA_TDNN` class with a 192-d head also sits in that file — it is dead code, not the Zonos path.)* | [`speaker_cloning.py`](https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/speaker_cloning.py); `..._LDA-128.pt` metadata | HIGH |
| 5 | Shipped as a SEPARATE checkpoint | **CONFIRMED** | `Zyphra/Zonos-v0.1-speaker-embedding`: `ResNet293_SimAM_ASP_base.pt` (396 MB) + `..._LDA-128.pt` (264 KB), pulled via `hf_hub_download`. | source | HIGH |
| 6 | Injected via `make_cond_dict` with pitch/rate/quality/emotion dials | **CONFIRMED** | Full signature verified. Note `vqscore_8`, `ctc_loss`, `dnsmos_ovrl`, `speaker_noised` are **hybrid-only**. | [`conditioning.py`](https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/conditioning.py); [CONDITIONING_README](https://raw.githubusercontent.com/Zyphra/Zonos/main/CONDITIONING_README.md) | HIGH |
| 7 | "The ONE model where description → vector → frozen renderer is directly implementable" | **WRONG — both over- and under-stated** | Under-stated: **ZONOS2** takes a `.npy` speaker vector as a first-class server input and already ships blending. **StyleTTS2** exposes a cleaner 256-d `ref_s` and natively samples style *without reference audio*. **Kokoro** ships 54 voice tensors. Over-stated: Zonos-v0.1 has **no vector-input path in its reference UI at all**. | [ZONOS2 `api_server.py`](https://raw.githubusercontent.com/Zyphra/ZONOS2/main/python/zonos2/server/api_server.py); StyleTTS2 `models.py` | HIGH |
| 8 | XTTS-v2 exposes `gpt_cond_latent` + `speaker_embedding` | **CONFIRMED mechanically, but DISQUALIFIED on licence** | Interface is real: `gpt_cond_latent` **(1,32,1024)** — a *sequence*, not a vector — plus `speaker_embedding` **(1,512,1)**. But weights are **CPML: "This license allows only non-commercial use."** Coqui is defunct; `coqui.ai/cpml` 404s; the `idiap` fork ships the same registry. | [`xtts.py`](https://raw.githubusercontent.com/coqui-ai/TTS/dev/TTS/tts/models/xtts.py); [CPML in the weights repo](https://huggingface.co/coqui/XTTS-v2/raw/main/LICENSE.txt) | HIGH |
| 9 | StyleTTS2 is a vector-addressable option | **PARTLY — better interface, worse licence** | `ref_s` is a clean **256-d** vector (128 style + 128 prosody), and interpolation is first-class in the reference implementation. But **code is MIT while the checkpoints carry no licence at all**, and inference depends on GPL espeak/phonemizer. | [StyleTTS2 LICENSE](https://raw.githubusercontent.com/yl4579/StyleTTS2/main/LICENSE); HF tree API (no LICENSE file) | HIGH |
| 10 | YourTTS is an option | **DISQUALIFIED** | **CC BY-NC-ND 4.0** — non-commercial *and* no-derivatives. Strictly worse than XTTS-v2. | [`.models.json`](https://raw.githubusercontent.com/coqui-ai/TTS/dev/TTS/.models.json) | HIGH |
| 11 | VoiceSculptor (2026) uses "design a waveform, then clone" | **CONFIRMED — it exists, and it does** | arXiv **2601.10629**, `ASLP-lab/VoiceSculptor`. Verbatim: "The designed voice is then **rendered into a prompt waveform** and fed into a **cloning model**." Current open-source SOTA on InstructTTSEval-Zh (67.6 vs ElevenLabs 50.9). **But its Apache-2.0 tag is contradicted by its own base (`Llasa-3B`, CC-BY-NC-4.0) and codec (`xcodec2`, CC-BY-NC-4.0).** zh/en only. | [arXiv 2601.10629](https://arxiv.org/abs/2601.10629); [HF API](https://huggingface.co/api/models/HKUSTAudio/Llasa-3B) | HIGH |
| 12 | (implicit) Zonos-v0.1 is a live, maintained target | **WRONG** | Last commit **2025-03-05**; 0 releases; 0 tags; 164 open issues; inactive ~18 months. There is no "Zonos v0.2" — the line went v0.1 → ZONOS2. | [GitHub API](https://api.github.com/repos/Zyphra/Zonos/commits) | HIGH |
| 13 | (implicit) Zonos covers Indian languages | **WRONG for v0.1** | The 109 `supported_language_codes` (incl. `hi, bn, ta, te, mr, gu, kn, ml, pa, ur`) are **eSpeak phonemizer** codes, not trained languages. Zyphra: "the model's performance on these languages is not robust." | [`conditioning.py:316`](https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/conditioning.py); [Zyphra blog](https://www.zyphra.com/post/beta-release-of-zonos-v0-1) | HIGH |
| 14 | (implicit) the stack is cleanly permissive | **INCOMPLETE** | Zonos-v0.1 requires `phonemizer` (**GPLv3+**) and eSpeak-NG (**GPL-3.0**) at runtime; Apache-2.0 on Zonos does not cure the chain. **ZONOS2 drops both.** Also `descript/dac_44khz` weights carry **no licence tag**. | [PyPI phonemizer](https://pypi.org/pypi/phonemizer/json); [espeak-ng COPYING](https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/COPYING) | HIGH |
| 15 | `make_cond_dict` docstring: "By default, it will generate a random speaker embedding" | **WRONG (upstream doc bug)** | `speaker=None` yields the **learned unconditional vector**, not a random draw. Variety comes from AR sampling. Do not use as a "random voice" baseline. | [`conditioning.py:43-50`](https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/conditioning.py) | HIGH |

---

## 2. A2 — Zonos: licence, and the speaker-embedding contract

### 2.1 Licence (code vs weights)

Every licence artefact fetched directly rather than trusting model-card prose.

| Artefact | Licence | Evidence | Confidence |
|---|---|---|---|
| Zonos-v0.1 **code** | **Apache-2.0** | [`LICENSE`](https://raw.githubusercontent.com/Zyphra/Zonos/main/LICENSE), 11,357 bytes verbatim | HIGH |
| `Zonos-v0.1-transformer` / `-hybrid` **weights** | **Apache-2.0** | HF API `cardData.license` | HIGH |
| `Zonos-v0.1-speaker-embedding` **weights** | tagged **Apache-2.0** — **but see the provenance failure below** | HF API | tag HIGH / validity **LOW** |
| Zyphra's statement | "released… under an Apache 2.0 license" | [official blog](https://www.zyphra.com/post/beta-release-of-zonos-v0-1) | HIGH |
| **ZONOS2 code** | **MIT** | [`LICENSE`](https://raw.githubusercontent.com/Zyphra/ZONOS2/main/LICENSE) — "MIT License / Copyright (c) 2026 Zyphra, Inc." | HIGH |
| **ZONOS2 weights** | **Apache-2.0** | HF API; paper: "We release our model weights and example inference code under an Apache 2.0 license" | HIGH |

For both generations of the *TTS itself*, code licence == weights licence. That is the good case. The problems are in the dependency chains.

#### 🚩 The VoxBlink2 provenance failure (Zonos-v0.1 only)

Traced end to end:

1. `Zyphra/Zonos-v0.1-speaker-embedding` is tagged **`apache-2.0`**.
2. Its own README says, verbatim: *"The speaker embedding models are based on the ResNet293-SimAM-ASP models from VoxBlink2. **We use the pretrain models** as we found the finetunes performed worse."* The **pretrain** models are the **VoxBlink2-trained** ones, not the VoxCeleb2-finetuned ones.
3. Upstream [`VoxBlink2/ScriptsForVoxBlink2`](https://github.com/VoxBlink2/ScriptsForVoxBlink2): *"The dataset is licensed under the **CC BY-NC-SA 4.0** license. This means that you can share and adapt the dataset for **non-commercial purposes**."* The `asv/` directory lists ResNet293-SimAM-ASP but carries **no licence or terms statement for the checkpoints at all**.
4. The governing convention in this ecosystem (WeSpeaker, whose pipeline VoxBlink2 used): *"The pretrained model in WeNet follows the license of it's corresponding dataset."*
5. Corroborating signal: WeSpeaker's own VoxBlink2-derived repo (`wespeaker-voxlink2-samresnet34_ft`) carries **no licence field**, while every VoxCeleb-derived sibling does.

**Assessment [HIGH on facts, MEDIUM on the legal conclusion]:** Zyphra's Apache tag is a **unilateral assertion** with no upstream permissive grant anywhere in the chain, over weights trained on an explicitly non-commercial dataset. Since public hosting is commercial use, **the Zonos-v0.1 128-d space should be treated as unusable for this project pending counsel.** ZONOS2 does **not** inherit this problem — its encoder traces cleanly to Apache-2.0 Qwen3-TTS.

#### Dependency chain, per component

| Gen | Component | Licence | Verdict |
|---|---|---|---|
| v0.1 | `phonemizer` | **GPLv3+** | ⚠️ copyleft |
| v0.1 | eSpeak-NG | **GPL-3.0** | ⚠️ copyleft |
| v0.1 | ResNet293 speaker weights | **CC-BY-NC-SA-4.0 provenance** | 🚫 **blocker** |
| v0.1 | `descript/dac_44khz` weights | **no licence tag** | ⚠️ UNVERIFIED |
| ZONOS2 | Mini-SGLang | MIT | ✅ |
| ZONOS2 | NeMo-text-processing (vendored) | Apache-2.0 | ✅ |
| ZONOS2 | `Qwen3-Voice-Embedding-12Hz-1.7B` | **Apache-2.0** | ✅ |
| ZONOS2 | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | **Apache-2.0** | ✅ |
| ZONOS2 | **`zonos2.cpp`** (only quantized runtime) | **NO LICENSE FILE — 404** | 🚫 **blocker** |

Two items for [08-licensing-propagation.md](08-licensing-propagation.md):

1. **Zonos-v0.1 cannot ship as a clean permissive stack** — GPL front-end *plus* NC-provenance speaker weights. ZONOS2 eliminates both: its `pyproject.toml` contains no `phonemizer`/`espeak`, and the README describes "nemo TN normalized UTF-8 bytes" as the text front-end.
2. **`zonos2.cpp` has no LICENSE file** (raw fetch → 404; no LICENSE in the root listing). Under default copyright that is all-rights-reserved — and it is the only published runtime for the GGUF weights that fit a 12GB card. The weights are Apache-2.0; the gap is the code.

### 2.2 The embedding contract (from source code, with the actual call signatures)

#### Zonos-v0.1 — verified line by line

```python
# zonos/model.py:90-95
def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
    if self.spk_clone_model is None:
        self.spk_clone_model = SpeakerEmbeddingLDA()
    _, spk_embedding = self.spk_clone_model(wav.to(self.spk_clone_model.device), sr)
    return spk_embedding.unsqueeze(0).bfloat16()          # -> (1, 128) bfloat16

# zonos/speaker_cloning.py:405-412
def forward(self, wav, sample_rate):
    emb = self.model(wav, sample_rate).to(torch.float32)   # (1, 256)
    return emb, self.lda(emb)                              # (1, 256), (1, 128)
```

| Property | Value |
|---|---|
| Encoder | `ResNet293_based(embd_dim=256)` → ASP pooling (out_dim 10240) → `Linear(10240, 256)` |
| Front-end | log-Mel Fbank @ **16 kHz**, n_fft 512, win 25 ms, hop 10 ms, 80 mels, per-utterance mean subtraction over time |
| LDA | `nn.Linear(256, 128, bias=True)`; checkpoint weight **float64, (128, 256)** |
| Returned | **dim 128, shape (1,128), dtype bfloat16** |
| **Normalisation** | **NONE** — raw LDA output. No `F.normalize` anywhere in the path |
| Determinism | Fully deterministic (eval mode, `dropout=0`, BN running stats, deterministic FFT). The bfloat16 cast additionally rounds away small cross-device differences |
| Injection | `make_cond_dict(speaker=t)` → `t.view(1,1,-1)` → `PassthroughConditioner(cond_dim=128, projection="linear", uncond_type="learned")` → `nn.Linear(128, 2048)` |
| Validation | **`assert x.shape[-1] == self.cond_dim` — and nothing else.** No range, norm, or manifold check |

**Geometry of the 128-d space — my own analysis of the shipped LDA matrix, not previously published:**

| Property | Value | Implication |
|---|---|---|
| `rank(W)` | **128** (full) | no dead output dimensions |
| singular values | max **346.9**, min **5.64**, **cond ≈ 61.5** | strongly **anisotropic** — some directions carry ~60× more scale |
| row norms | mean **66.5**, std 18.5 | outputs are large-magnitude, nowhere near unit norm |
| row Gram off-diagonal | max abs **10,924** | rows **not orthogonal** — a general affine map, not a rotation |
| bias norm | 12.6 | the space is not centred at the origin |

Consequences that constrain the mapper: `torch.randn(1,128)` (‖·‖≈11.3) is off by ~6× in scale and ignores the anisotropy — naive Gaussian sampling is wrong, which is exactly what TacoSpawn found empirically (§2.3). A plain-MSE mapper will be dominated by the top few directions, so **train in a whitened space**. *(Implications are my inference from the artefact, flagged as such; the numbers are computed.)*

#### ZONOS2 — the contract has changed

From `params.json` (authoritative): `"speaker_embedding_dim": 2048`, `"speaker_lda_dim": 1024`, `"dim": 2048`, `"n_layers": 28`, `"moe_n_experts": 16`, `"moe_router_topk": 1`.

From the technical report §II-C, verbatim:

> "we condition on ECAPA-TDNN speaker embeddings… to extract a 2048-dimensional embedding e_x… the embedding is projected through an Linear Discriminant Analysis (LDA) transform to a **1024-dimensional** vector ê_x. Because the LDA is estimated from speaker-labeled data to maximize between-speaker variance relative to within-speaker variance, it retains the directions that separate one speaker from another while attenuating the factors that vary across different recordings of the same speaker…"
>
> "h_spk = W_spk ê_x + b_spk, where W_spk ∈ R^(d_model × 1024)"

| Property | Zonos-v0.1 | ZONOS2 |
|---|---|---|
| Encoder | ResNet293-SimAM-ASP (VoxBlink2, **NC provenance**) | **ECAPA-TDNN** — `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`, `enc_dim 2048`, **12,001,088 params / 24 MB**, **Apache-2.0**, 24 kHz, 128 mels |
| Raw dim → LDA dim | 256 → 128 | **2048 → 1024** |
| Injection | `Linear(128, 2048)` | `W_spk ∈ R^(2048×1024)`, single affine map at sequence position 0 |
| Vector input | none in reference UI | **first-class** `.npy`/`.npz` + `speaker_embedding=` in the Python API |
| Text front-end | eSpeak phonemes (**GPL**) | NeMo TN + UTF-8 bytes (Apache-2.0) |

The `.npy` loader is the Tier 1 injection point, and its validation is strikingly permissive:

```python
arr = np.load(io.BytesIO(embedding_bytes), allow_pickle=False)
arr = np.asarray(arr, dtype=np.float32); arr = np.squeeze(arr)
if   arr.ndim == 1: vector = arr
elif arr.ndim == 2: vector = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)  # ← averaging sanctioned
if vector.shape[-1] != expected_dim: raise ValueError(...)                       # ← the ONLY check
```

The **only** validation is dimension, and **multi-row files are mean-averaged** — linear averaging of speaker embeddings is an operation the official implementation performs by design. The **12M-param, 24 MB, Apache-2.0, CPU-trivial** encoder makes Tier 1's "free/CPU to mint" claim hold comfortably.

> ⚠️ **One unexecuted assumption.** `Qwen3SpeakerEmbedding.forward` returns `self.model(input_values=mel).last_hidden_state`. For this ECAPA architecture with attentive statistics pooling that should be the pooled `(B, 2048)` vector, consistent with `enc_dim: 2048` and `speaker_embedding_dim: 2048` — but **I have not run it** to rule out `(B, T, 2048)`. One line of code settles it; see §8.

### 2.3 IS THE MANIFOLD NAVIGABLE BY SYNTHESIS? (the decisive sub-question)

**Answer: YES with a specific, non-negotiable condition — sample from a prior fitted to the real speaker distribution, never from an uninformed one.** The evidence now comes from three independent directions.

#### (i) First-party: Zyphra ship navigation of this exact space

**Interpolation — SHIPPED.** `_slerp_embeddings(v0, v1, t)` performs true spherical interpolation between two cached speaker embeddings, exposed over HTTP as `speaker_blend_embedding_id_a/_b/_t` and in the bundled web UI as *"Pick two cached speakers to SLERP between them."* Also in the C++ port. A vendor does not ship a blend slider that produces artefacts. **[HIGH]**

**Vector arithmetic — SHIPPED AND CALIBRATED.** Emotion control is `emb' = base + strength · Σ slider_e · dir_e`, where each `dir_e = mean(emotion) − mean(neutral)` over an emotion-labelled corpus. Six 2048-d directions ship with a `manifest.json` and a `calibration.json` tuned against the **emotion2vec** recogniser. README: it "preserves timbre while prosody shifts". **[HIGH]**

Two reusable implementation details:
- **`preserve_norm=True`** rescales any perturbed vector back to the base embedding's L2 norm — documented as *"so the injected vector keeps the magnitude the model was trained on."* Magnitude is load-bearing, consistent with the unnormalised anisotropic space measured above.
- For LDA-space directions they **project through the LDA, add the delta there, then map back via the LDA pseudo-inverse** so the unchanged model re-applies LDA and recovers the injected vector. A ready-made technique for targeting the 1024-d space through a 2048-d API.

#### (ii) Academic: generating speaker embeddings works, with a caveat

| Work | What it shows | Numbers | Conf |
|---|---|---|---|
| **PromptTTS 2** ([2309.02285](https://arxiv.org/abs/2309.02285), Microsoft) | Diffusion "variation network" generates the voice representation **from a text prompt, with no reference audio at inference** — literally our architecture. Fixed-dimension rep (hidden 512), injected by cross-attention into any TTS backbone; 44K hrs | Attribute accuracy **93.33%**; MOS **3.88** (GT 4.38, codec ceiling 4.30). **Resampling the variation network gives similarity 0.355** vs 0.914 for resampling the backbone → genuinely diverse voices, no collapse | HIGH |
| **TacoSpawn / "Speaker Generation"** ([2111.05095](https://arxiv.org/abs/2111.05095), Google) | Fits a **mixture-of-Gaussians prior** over a learned speaker table and samples novel speakers | MOS generated **3.62±0.11** vs real 3.68±0.11 (gap inside CIs). Diversity g2g ≈ s2s | HIGH |
| **VoicePrivacy B1** ([2404.02677](https://arxiv.org/abs/2404.02677)) | Pseudo-speaker x-vectors made by **averaging 100 PLDA-far pool vectors** — never-observed embeddings — then resynthesizing | **WER 2.91%** — fully intelligible | HIGH |
| **VoicePrivacy B3** (same) | **Wasserstein GAN** generating artificial embeddings in a 128-d space | WER **4.35%** | HIGH |
| **WGAN speaker embeddings** ([2210.07002](https://arxiv.org/abs/2210.07002)) | GAN-generated embeddings give "intelligible and content-preserving" speech, confirmed by human evaluation | — | HIGH |
| **INSIDE** ([2508.19210](https://arxiv.org/abs/2508.19210)) | **SLERP between real embeddings in a pretrained space**, decoded by YourTTS, used to train SV models | +3.06–5.24% relative. Chose SLERP because it "better fits the hyperspherical geometry… and preserves unit norm" | HIGH |
| **GMM-OT interpolation** ([2210.09916](https://arxiv.org/abs/2210.09916)) | Continuous attribute control "without statistically significant degradation of speech naturalness" | — | HIGH |
| VoxGenesis / VoiceLens / SpeakerVAE ([2403.00529](https://arxiv.org/abs/2403.00529), [2309.14094](https://arxiv.org/abs/2309.14094), [2511.07135](https://arxiv.org/abs/2511.07135)) | Latent speaker manifolds, flows and VAEs generating "novel, unseen speakers with quality comparable to training speakers" | — | HIGH |

#### (iii) The three caveats that shape the design

1. **🚩 Uninformed sampling fails — every source agrees.** TacoSpawn reports a standard-normal prior "performed worse", and criticises prior work that sampled "uniformly random points on the unit hypersphere" with "no attempt to ensure these speakers have a distribution similar to the training speakers". **The mapper must target a fitted density (Gaussian/MoG/flow) over real embeddings, or stay inside the hull of real anchors.** [HIGH]
2. **⚠️ Frozen SV encoders are a *harder* target than purpose-built ones.** TacoSpawn A/B-tested exactly this: learned 128-d embeddings gave g2s=0.20/g2g=0.20; pretrained 256-d d-vectors gave 0.35/0.27, concluding *"d-vectors appear to be much less amenable to modeling with a simple parametric prior."* [Ulgen et al. 2407.04291](https://arxiv.org/abs/2407.04291) explains why: SV training "suppresses intra-speaker variability… leading to overly compact representations that may discard variations crucial for generation." **We are forced to use the renderer's frozen encoder, so this risk is live.** *One mitigating hypothesis, flagged as inference and testable: TacoSpawn's complaint is that d-vectors carry too much utterance-specific nuisance to fit a simple prior — and ZONOS2's LDA is explicitly designed to strip exactly that nuisance (duration, noise, lexical content, pauses). The LDA-1024 space may therefore be materially more prior-friendly than a raw d-vector space. §8 tests it.*
3. **⚠️ Synthetic identities are under-varied.** INSIDE reports that "cosine similarity scores between different utterances from the same synthetic identity are significantly higher than those of real identities" — synthesized speakers are *too* internally consistent. For games and dramatic content that is a mild positive (consistency is what we want), but it will show up in any realism metric.

#### (iv) The first-party counter-evidence — take it seriously

ZONOS2 technical report §VII-C, verbatim:

> "we observed causal leakage of information regarding the length, lexical content, and pause distribution of the source clone audio which caused **inference instability characterized by either silent output or 'glossolalia' babble output**."

A synthesized vector is by construction a mismatch case — it encodes no real utterance. **The failure mode is silence or babble, not graceful degradation.** The LDA was their most effective mitigation, which is mildly reassuring for targeting the LDA space, but it means the mapper must land *on* the populated region, not merely somewhere dimensionally valid.

*(Note: v0.1's `speaker_noised` flag is **not** evidence of embedding-space noise tolerance — the docs describe it as a flag for acoustic noise in the reference clip. Do not over-read it.)*

#### (v) What nobody has published

| Operation | Zonos-v0.1 | ZONOS2 |
|---|---|---|
| interpolation / SLERP | **UNVERIFIED** | **HIGH — official feature** |
| vector arithmetic | **UNVERIFIED** | **HIGH — emotion directions** |
| Gaussian noise on the vector | **UNVERIFIED** | not found |
| random / prior-sampled vectors | code exists, **no perceptual report** | not found |
| trained text→speaker-embedding generator | **none found** | **none found** |
| artefact / off-manifold failure analysis | **none** | **none** |

The one public instance of random-vector injection — [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR/blob/main/tools/reference_backends/zonos_tts_reference.py) feeding `torch.randn(1,1,128)` into the speaker slot — is a **numerical-parity harness for a C++ port** and never reports what the audio sounds like.

**Venues exhaustively searched and empty for v0.1:** all 250 `Zyphra/Zonos` issues+PRs and all 858 repo comments; GitHub Discussions (disabled); all 51 HF discussions across the three v0.1 repos; 24 HF Spaces; 6 GitHub code-search variants; GitHub issue search; the major integrations (ComfyUI-Zonos, AudioLab, achatbot, the 500★ `sdbds/Zonos-for-windows` fork); arXiv `all:"Zonos"` (40 hits — **no Zonos v0.1 paper exists**); Semantic Scholar citations of ZONOS2 (**zero**); both Zyphra blogs.

> **Coverage gap, stated honestly:** Reddit could not be searched directly (blocked; `search.json` 403). **Treat r/LocalLLaMA coverage as LOW confidence / effectively unsearched**, not a confirmed negative.
>
> **False lead discounted:** Zonos PR #101's "latent space interpolation" is a cosine crossfade between *acoustic* latents for long-form stitching — nothing to do with speaker embeddings.

#### Verdict on A2

**The two-tower architecture survives.** It is published (PromptTTS 2, TacoSpawn), the specific space is vendor-navigated (ZONOS2 SLERP + direction vectors), and synthesized embeddings are known to decode intelligibly (VoicePrivacy). The amendment is that the mapper must **regress or sample into a fitted model of the real embedding distribution**, not emit free vectors. §8 settles the residual uncertainty for our exact renderer in about an hour.

### 2.4 Maintenance status & VRAM

| | Zonos-v0.1 | ZONOS2 |
|---|---|---|
| Released | Feb 2025 | **2026-06-12** |
| Last commit | **2025-03-05** | 2026-07-06 |
| Releases / tags | **0 / 0** | — |
| Stars | 7,242 | 308 |
| Open issues | 164 | — |
| Paper | **none exists** | [arXiv 2606.24320](https://arxiv.org/abs/2606.24320) — 8B total / 900M active, 6M hrs |
| HF downloads (30d) | 147,120 | 677 + 3,285 (GGUF) |

Ecosystem: `Zyphra/zonos2.cpp` (ggml, **unlicensed**), `Zyphra/ZONOS2_finetuning` (Apache-2.0 trainer), `Zyphra/ZTTS1-Eval` (Apache-2.0 benchmark, 9 read + 17 in-the-wild languages).

| Configuration | Size | Fits 12GB? |
|---|---|---|
| Zonos-v0.1 bf16 | 3.25 GB weights; **4,686 MiB measured on RTX 3090** | ✅ comfortably |
| `ZONOS1-GGUF` q8_0 / q4_k | **1.77 GB** / **0.94 GB** | ✅ |
| **ZONOS2 bf16** | **15.34 GB** | ❌ **does not fit** |
| ZONOS2 GGUF q8_0 / q6_k / q4_k | 8.54 / **6.79** / 4.92 GB | ✅ **but unlicensed runtime** |

Two useful notes: the ZONOS1-GGUF card states *"All quants keep the prefix conditioner (including the speaker projection) at full precision"* — **quantization does not degrade the speaker path**, exactly the path we care about. It also warns *"Unlike its MoE sibling ZONOS2, this dense backbone tolerates 4-bit quantization gracefully"* — so prefer q6_k/q8_0 for ZONOS2.

ZONOS2's PyTorch stack is also heavier and more CUDA-specific than v0.1 (`sgl_kernel`, `flashinfer-cubin`, `nvidia-cutlass-dsl`), Linux x86_64 + NVIDIA only — relevant given the dev box is Windows. And one report to A/B before committing: [ZONOS2 issue #9](https://github.com/Zyphra/ZONOS2/issues/9) — *"very underwhelming cloning quality… Zonos v1 did pretty well"*, unanswered. Single report, MEDIUM.

---

## 3. A1 — Tier 1 vs Tier 2, on evidence

**Verdict: Tier 1 canonical, Tier 2 as a mandatory durability anchor. Store both. Reject Tier 3.**

### Evidence for Tier 1

1. **The model authors state the fixed-length vector loses little identity.** ZONOS2 report §II-C: *"the embedding is high-bandwidth; a single 2048-dimensional vector can capture nearly all of the desired speaker characteristics, so **little identity information is lost relative to conditioning on the full utterance**."* This is the direct answer to "how much of voice identity survives in a vector vs a waveform", from people who trained both. **[HIGH]**
2. **They also give the positive case:** the vector "occupies a single position at the start of the sequence… does not grow the context the decoder must attend over, regardless of how long the reference recording is", and avoids "reliance on a transcription of the speech in the target clone utterance."
3. **The LDA removes exactly the nuisance we do not want persisted** — duration, noise, lexical content, pause structure. A stored waveform carries all of it; the stored vector is a cleaner identity carrier. **[HIGH]**
4. **Reproducibility is by construction.** Extraction is deterministic and generation is seeded (`TTSSamplingParams(seed=42)`), so the same vector + seed + text + revision reproduces the same audio. **No published cross-*version* stability numbers exist** — see the Tier 2 case. **[HIGH within-version; UNVERIFIED across versions]**
5. **Minting is free and CPU-only** — 12M params, 24 MB.
6. **Interpolation is real and shipped** — a capability Tier 2 structurally cannot offer.
7. **PromptTTS 2 proves the whole Tier 1 pipeline end to end** — text prompt → generated voice representation → speech, no reference audio at inference, MOS 3.88, and resampling gives similarity 0.355 (diverse, not collapsed). **[HIGH]**

### Evidence for Tier 2

1. **Zyphra's own product ships audio, not vectors, as the canonical default-voice artefact.** `default_voices/` contains `AmericanFemale.mp3`, `AmericanMale.mp3`, `BritishFemale.mp3`, embedded at startup and cached. If a vector were strictly superior as a *persistence* format, the vendor would ship `.npy`. **[HIGH]**
2. **The decisive Tier 1 weakness — encoder-version coupling — is real and already happened.** A stored vector is meaningful only relative to `(encoder, encoder revision, LDA, model revision)`. v0.1 → ZONOS2 changed the encoder (ResNet293 → ECAPA-TDNN), the raw dim (256 → 2048) and the LDA dim (128 → 1024). **Every v0.1 vector is worthless against ZONOS2.** A stored *waveform* survives that transition — you re-embed and carry on. This is not hypothetical; it happened inside one model family in 16 months. **[HIGH]**
3. **Tier 2 is backend-agnostic** — a 10 s seed clip works against any zero-shot TTS, including the many with no vector interface (§5). Real optionality under a "frozen and swappable C" constraint.
4. **🔑 The 2026 open-source SOTA deliberately chose Tier 2.** **VoiceSculptor** (arXiv 2601.10629, `ASLP-lab/VoiceSculptor`) — verbatim: *"The designed voice is then **rendered into a prompt waveform** and fed into a **cloning model** for high-fidelity timbre transfer."* It scores **67.6 AVG on InstructTTSEval-Zh vs ElevenLabs' 50.9**. That is the strongest available signal about what currently wins in practice. **[HIGH]** *(Licence caution: its Apache-2.0 tag is contradicted by its own base `Llasa-3B` and codec `xcodec2`, both **CC-BY-NC-4.0**. zh/en only. Treat as commercially unusable pending counsel — but its **architecture** is valid evidence regardless of its licence.)*

### The synthesis — and the strongest single artefact

Zyphra's production code models both tiers in one record:

```python
@dataclass
class CachedSpeakerReference:
    speaker_id: str
    label: str
    source_type: Literal["audio", "embedding_file"]   # ← the tier discriminator
    embedding: torch.Tensor                            # ← Tier 1
    created_at: float
    original_name: str | None = None
    audio_bytes: bytes | None = None                   # ← Tier 2, retained alongside
    audio_media_type: str | None = None
```

A shipping implementation of this exact problem carries a `source_type` discriminator and keeps **both**. Treating the tiers as competing options is the wrong frame.

| | Decision |
|---|---|
| Canonical identity | **Tier 1 vector**, per `(character, language, render_target)` |
| Durability anchor | **Tier 2 seed waveform**, minted once at design time, retained forever |
| On model/encoder version bump | Re-derive Tier 1 from the retained waveform. Cost: one CPU embed. Identity survives |
| Blending / emotion steering | Operate on the Tier 1 vector (SLERP + direction vectors, both proven) |
| **Tier 3** (description + RNG seed) | **Reject**, as the brief proposes — breaks on any version bump, no anchor to re-derive from |

Cost: ~160 KB of audio per voice profile. Given "one voice profile per language per character", negligible.

### Is a fixed vector expressive enough for heavy stylization (raspy, whispered, aged, menacing)?

**UNVERIFIED, and this is the second-most important open question.** Three concrete cautions:

- Zonos-v0.1's own README says **audio prefixes, not speaker embeddings**, are the mechanism for "behaviours such as whispering which can otherwise be challenging to replicate when cloning from speaker embeddings." A first-party admission that the embedding does **not** capture whisper. **[HIGH]**
- The LDA is explicitly designed to *discard* within-speaker variation — which is where much stylization lives. Ulgen et al. make the same point generally: SV-trained embeddings "discard variations crucial for generation."
- ZONOS2's own answer to expressivity is **not** the speaker vector: it is `accurate_mode=false` + emotion directions + `emotion_cfg_scale`. The architecture separates identity from style on purpose.

**Implication:** do not expect the identity vector alone to carry raspy/whispered/aged. Architect **identity vector + a separate style/emotion vector** from the start. This is consistent with the brief's "human + heavy stylization" goal but changes *where* stylization is implemented — and it is why §6 gives style its own table.

### Production systems

**NOT VERIFIED in this pass.** The brief's claimed ElevenLabs flow (`/v1/text-to-voice/design` → `generated_voice_id` + audio previews → `/v1/text-to-voice` to persist) was **not confirmed from official docs**; PlayHT and Resemble likewise. **Do not treat the brief's description as established.** If that flow is accurate it is strong Tier 2 evidence — previews are audio, and the persisted voice is created *from* a preview — but it must be verified. See §8. One indirect datapoint in hand: VoiceSculptor benchmarks **ElevenLabs at 50.9 AVG on InstructTTSEval-Zh** (vs Gemini 2.5-Flash 85.4) — evidence about instruction-following quality, not about internal representation.

---

## 4. C2 — which speaker encoder defines the target manifold

| Encoder | Dim | Licence (code / weights) | Navigable? | Consumable by which backend? | Eval-independent? | Verdict |
|---|---|---|---|---|---|---|
| **ZONOS2 ECAPA-TDNN → LDA** (`Qwen3-Voice-Embedding`) | 2048 → **1024** | **Apache-2.0 / Apache-2.0** (base Qwen3-TTS also Apache-2.0) | **YES, demonstrably** — SLERP + mean-difference directions shipped by the vendor *in this exact space* | **ZONOS2 natively** (`.npy`, dim-checked only) | Yes | ✅ **RECOMMENDED target** |
| **SpeechBrain ECAPA** `spkrec-ecapa-voxceleb` | **192** (raw, **not** L2-normed) | **Apache-2.0 / Apache-2.0** | Strong indirect (VoicePrivacy, INSIDE) | not natively by Zonos | Yes | ✅ best *general-purpose* clean encoder |
| WeSpeaker CAM++-LM | 512 | Apache-2.0 / **apache-2.0** | indirect | no | Yes | ✅ clean |
| **WeSpeaker ReDimNet2-B6-LM** | **192** | Apache-2.0 / **apache-2.0** | — | no | **Yes** | ✅ **clean eval judge** |
| `funasr/campplus` | 192 | Apache-2.0 / **apache-2.0** | — | no | Yes | ✅ clean eval judge |
| 3D-Speaker ERes2Net | 192 | Apache-2.0 / Apache-2.0 | — | no | Yes | ✅ clean |
| WeSpeaker ResNet221-LM | 256 | Apache-2.0 / **apache-2.0** | — | no | Yes | ✅ clean |
| WeSpeaker ResNet293/34-LM | 256 | Apache-2.0 / **cc-by-4.0** | — | no | Yes | ⚠️ attribution attaches |
| **UniSpeech WavLM-large SV** (the field's standard judge) | **256** | **CC-BY-SA-3.0** / no statement | — | no | **Yes — de-facto standard** | ⚠️ eval-only; flag to counsel |
| `microsoft/wavlm-base-plus-sv` | 512 | unilm MIT / **no licence field** | — | no | Yes | ⚠️ unclear |
| NeMo TitaNet-Large | 192 | Apache-2.0 / **cc-by-4.0** | — | no | Yes | ⚠️ + very heavy NeMo dep |
| Resemblyzer / GE2E | 256 (**L2-normed**) | Apache-2.0 | widely interpolated | no | Yes | dated, but the only bounded manifold (unit sphere) |
| `pyannote/embedding` | — | MIT / **gated (README 401s)** | — | no | Yes | 🚫 gating disqualifies |
| **Zonos v0.1 ResNet293 → LDA-128** | 256 → **128** (raw) | Apache tag / **CC-BY-NC-SA-4.0 provenance** | **UNVERIFIED** | Zonos-v0.1 only | Yes | 🚫 **licence-disqualified** |
| IDRnD ReDimNet (upstream) | 192 | **MIT** / no explicit weights licence; `vb2` variants VoxBlink2-derived | — | no | Yes | ⚠️ avoid `vb2` |

**Recommendation.**

- **Target space: the encoder the renderer already uses** — for ZONOS2 that is ECAPA-TDNN 2048 → LDA 1024. Any other choice requires learning a translation into the renderer's space, adding a failure mode for no benefit. **Regress in the 1024-d LDA space** (what the model actually consumes, and where Zyphra's own emotion machinery renormalises), then map back to 2048-d via the LDA pseudo-inverse using the technique already in `emotion.py`.
- **Eval judge: `Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM` (192-d, Apache-2.0)** or `funasr/campplus` (192-d, Apache-2.0) for zero licence ambiguity. Use **UniSpeech WavLM-large SV (256-d)** additionally when comparability with published numbers matters — but note it is **CC-BY-SA-3.0**, not MIT/Apache, and the checkpoints carry no licence statement. Eval-only, never shipped, so exposure is far lower than for a training target.

**Eval independence is cleanly achievable.** The judge encoders in current use, per the ZONOS2 report's benchmark table: **WavLM** (Seed-TTS-Eval, MiniMax-ML), **ERes2Net** (CV3-Eval), **ReDimNet** (ZTTS1-Eval). All are architecturally distinct from an ECAPA-TDNN training target, so the harness does not mark its own homework. Confirmed by a closed chain: seed-tts-eval names "WavLM-large fine-tuned on the speaker verification task" with a specific Drive file ID, UniSpeech's README lists that same ID, and F5-TTS's eval script loads `wavlm_large_finetune.pth`. **[HIGH]**

**Two licence traps worth internalising:** WeSpeaker's own docs say *"The pretrained model in WeNet follows the license of it's corresponding dataset"* — which **contradicts** the Apache tags on several of their HF repos and is the mechanism that poisons anything VoxBlink2-derived. And VoxCeleb1 metadata is **CC-BY-SA-4.0** (commercial use fine, ShareAlike attaches), not CC-BY-4.0.

**Two gaps I will not paper over:**
- **Anisotropy / hubness in *speaker* embedding spaces: UNVERIFIED.** Well-established for text embeddings; **no primary paper found** measuring it for x-vector/ECAPA spaces. *(My own measurement of Zonos's LDA matrix in §2.2 is a partial, artefact-level substitute — it shows the projection is strongly anisotropic, but says nothing about how real embeddings populate the result.)*
- **LDA-projected spaces as a synthesis target: UNVERIFIED.** No paper compares LDA-projected vs raw embeddings for *generation*. LDA is universal in SV *scoring*, which is a different result. This is precisely the hypothesis in §2.3(iii) and it is testable in §8.

---

## 5. Other vector-addressable backends

| Model | Code licence | Weights licence | Speaker vector exposed? | Indian langs | Conf |
|---|---|---|---|---|---|
| **ZONOS2** | **MIT** | **Apache-2.0** | **YES** — `.npy`/`.npz` **2048-d**, dim-checked only; + SLERP blend + emotion directions | hi, ta, te, bn (**Tier 3**, no published numbers) | HIGH |
| **Zonos-v0.1** | Apache-2.0 | Apache-2.0 (**but NC speaker-encoder provenance**) | **YES** — 128-d via `make_cond_dict`; no UI path | 109 eSpeak codes, **not trained** | HIGH |
| **Kokoro-82M** | **Apache-2.0** | **Apache-2.0** | **YES** — 54 voice tensors, each **(510, 1, 256)** float32 (downloaded and verified) | **4 Hindi voices** | HIGH |
| **StyleTTS2** | **MIT** (GPL runtime dep) | **NO LICENCE AT ALL** on the checkpoints | **YES, cleanest** — `ref_s` = **256-d** = concat(style 128, prosody 128); interpolation first-class; **native diffusion style sampling with no reference audio** | English (needs per-lang PL-BERT) | HIGH |
| **XTTS-v2** | MPL-2.0 | 🚫 **CPML — NON-COMMERCIAL** | YES — `gpt_cond_latent` **(1,32,1024)** *sequence* + `speaker_embedding` **(1,512,1)** | `hi` (17 langs) | HIGH |
| **YourTTS** | MPL-2.0 | 🚫 **CC BY-NC-ND 4.0** | d-vector, `d_vector_dim=512` | no | HIGH |
| **VoiceSculptor** | Apache tag | 🚫 **NC via `Llasa-3B` + `xcodec2` (both CC-BY-NC-4.0)** | Tier 2 by design (renders waveform, then clones) | zh/en only | HIGH |
| **Chatterbox** (Resemble) | — | **MIT** | **UNVERIFIED** | `hi` listed (23 langs) | licence HIGH / interface UNVERIFIED |
| **IndicF5** (AI4Bharat) | — | **MIT** | **UNVERIFIED** | **11 Indian languages** (as,bn,gu,mr,hi,kn,ml,or,pa,ta,te) | licence HIGH / interface UNVERIFIED |
| **Indic Parler-TTS** | — | **Apache-2.0** | **UNVERIFIED** (expected description-string conditioning) | **12+** incl. hi, ks, ne | licence HIGH / interface UNVERIFIED |
| **CosyVoice2-0.5B** | — | **Apache-2.0** | **UNVERIFIED** | no | licence HIGH |
| F5-TTS, Fish-Speech, Orpheus, Dia, IndexTTS-2, Spark-TTS, Higgs v2, VibeVoice, Sesame CSM | — | — | **NOT REACHED** | — | — |

**Three conclusions.**

1. **The brief's "Zonos is the ONE model" premise is wrong in both directions.** ZONOS2 is a materially better instance; StyleTTS2 has a cleaner interface and can already sample style without reference audio; Kokoro is Apache-2.0 on both sides with 54 interpolatable voice tensors.
2. **Kokoro is currently the only backend verified Apache-2.0 on code *and* weights with a confirmed addressable vector *and* Hindi.** Its hard limit: the model card says *"Decoder only: no diffusion, no encoder release"* — so it **cannot clone from audio** and **cannot sample new styles**. You get a closed vocabulary of 54 seed vectors plus whatever arithmetic you do on them. Pure Tier 1 with no escape hatch — but that makes it an excellent, cheap **testbed for the navigability experiment**.
3. **For Indian languages the clean-licence field is `IndicF5` (MIT, 11 languages), `Indic Parler-TTS` (Apache-2.0) and Kokoro's 4 Hindi voices** — not Zonos. This may end up driving backend selection more than the vector interface does. Belongs in [03-tts-backends-english.md](03-tts-backends-english.md) and a companion Indic doc.

---

## 6. Proposed identity store schema (both tiers representable)

Constraints encoded: one profile per language per character; a vector is meaningless without the exact `(encoder, LDA, renderer)` triple that minted it; Tier 2 audio is the anchor that survives version bumps; every render reproducible; style separate from identity.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── The creative identity. Language-independent. ──────────────────
CREATE TABLE character (
    character_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slug               text UNIQUE NOT NULL,
    display_name       text NOT NULL,
    description        text NOT NULL,        -- the natural-language prompt (tower A input)
    description_sha256 bytea NOT NULL,       -- normalised-prompt hash: cache hits + dedup
    created_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX character_desc_hash_idx ON character (description_sha256);

-- ── A frozen (encoder + LDA + renderer) triple. A Tier-1 vector is
--    valid against EXACTLY ONE of these rows. This table is what makes
--    the v0.1 -> ZONOS2 class of migration survivable instead of
--    silently rendering the wrong voice. ────────────────────────────
CREATE TABLE render_target (
    render_target_id   text PRIMARY KEY,      -- 'zonos2@2026-06-22'
    tts_family         text NOT NULL,         -- 'zonos-v0.1' | 'zonos2' | 'kokoro'
    tts_repo           text NOT NULL,
    tts_revision       text NOT NULL,         -- HF commit sha; ANY change invalidates vectors
    encoder_id         text NOT NULL,         -- 'qwen3-voice-ecapa-2048'
    encoder_revision   text NOT NULL,
    raw_dim            int  NOT NULL,         -- 256 (v0.1) | 2048 (ZONOS2)
    lda_dim            int  NOT NULL,         -- 128 (v0.1) | 1024 (ZONOS2)
    is_l2_normalised   boolean NOT NULL DEFAULT false,   -- both Zonos generations: FALSE
    storage_dtype      text NOT NULL DEFAULT 'float32',  -- store f32; cast at inference
    licence_class      text NOT NULL,         -- 'permissive' | 'noncommercial' | 'unverified'
    is_active          boolean NOT NULL DEFAULT true,
    UNIQUE (tts_family, tts_revision, encoder_id, encoder_revision)
);

-- ── One voice profile per character per language (LOCKED CONSTRAINT).
--    Carries BOTH tiers: the vector to render with, the audio to survive on.
CREATE TABLE voice_profile (
    voice_profile_id  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    character_id      uuid NOT NULL REFERENCES character(character_id) ON DELETE CASCADE,
    language          text NOT NULL,          -- BCP-47: 'en-IN', 'hi-IN', 'ta-IN'
    render_target_id  text NOT NULL REFERENCES render_target(render_target_id),
    source_type       text NOT NULL CHECK (source_type IN ('embedding_file','audio','mapper')),

    -- Tier 1 -------------------------------------------------------
    -- raw encoder-space vector, exactly as the backend's .npy loader expects.
    -- NOTE: pgvector INDEXES cap at 2000 dims, so a 2048-d vector is storable
    -- but NOT indexable -> we index the LDA vector instead.
    embedding_raw     vector(2048),
    -- post-LDA vector: the space the model actually consumes, and the correct
    -- space for similarity, dedup and interpolation.
    embedding_lda     vector(1024),
    embedding_norm    real,                   -- L2 norm; magnitude is load-bearing
    embedding_sha256  bytea,

    -- Tier 2 (durability anchor - ALWAYS populated) ----------------
    seed_audio_uri    text,                   -- object-store key for the ~10s mint clip
    seed_audio_sha256 bytea,
    seed_audio_ms     int,
    seed_audio_mime   text,                   -- 'audio/flac' preferred (lossless)

    -- provenance / reproducibility ---------------------------------
    mapper_version    text,
    mint_seed         bigint,
    mint_params       jsonb NOT NULL DEFAULT '{}'::jsonb,
    derived_from      uuid REFERENCES voice_profile(voice_profile_id),  -- blends/edits
    created_at        timestamptz NOT NULL DEFAULT now(),

    UNIQUE (character_id, language),
    CONSTRAINT tier1_complete CHECK (
        source_type <> 'embedding_file'
        OR (embedding_raw IS NOT NULL AND embedding_norm IS NOT NULL)),
    CONSTRAINT tier2_anchor CHECK (seed_audio_uri IS NOT NULL)
);

CREATE INDEX voice_profile_lda_hnsw
    ON voice_profile USING hnsw (embedding_lda vector_cosine_ops);
CREATE INDEX voice_profile_target_idx ON voice_profile (render_target_id);

-- ── Style/emotion is NOT identity. Separate and composable.
--    Mirrors ZONOS2's shipped emotion_directions/ design. ───────────
CREATE TABLE style_direction (
    style_direction_id text NOT NULL,         -- 'happy' | 'raspy' | 'aged' | 'menacing'
    render_target_id   text NOT NULL REFERENCES render_target(render_target_id),
    space              text NOT NULL CHECK (space IN ('raw','lda','proj')),
    direction          vector(2048) NOT NULL, -- mean(style) - mean(neutral)
    default_strength   real NOT NULL DEFAULT 1.0,
    PRIMARY KEY (style_direction_id, render_target_id)
);

-- ── The fitted prior over REAL embeddings. This is what the mapper
--    samples from - never a standard normal (TacoSpawn). ────────────
CREATE TABLE speaker_prior (
    speaker_prior_id  text PRIMARY KEY,
    render_target_id  text NOT NULL REFERENCES render_target(render_target_id),
    kind              text NOT NULL CHECK (kind IN ('gaussian','mog','flow')),
    n_components      int,
    fitted_on_n       int NOT NULL,           -- how many real embeddings it was fit to
    mean              vector(1024),
    params_uri        text NOT NULL,          -- covariance / MoG / flow weights blob
    median_norm       real NOT NULL,          -- target magnitude for generated vectors
    created_at        timestamptz NOT NULL DEFAULT now()
);

-- ── Every render, reproducible. Proves identity stability over time. ──
CREATE TABLE render (
    render_id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    voice_profile_id  uuid NOT NULL REFERENCES voice_profile(voice_profile_id),
    render_target_id  text NOT NULL REFERENCES render_target(render_target_id),
    text_sha256       bytea NOT NULL,
    seed              bigint NOT NULL,
    sampler_params    jsonb NOT NULL,
    style_applied     jsonb NOT NULL DEFAULT '{}'::jsonb,  -- {"happy":1.0,"arousal":0.3}
    audio_uri         text NOT NULL,
    audio_sha256      bytea NOT NULL,
    sim_to_seed       real,                   -- scored by an INDEPENDENT encoder
    sim_encoder_id    text,                   -- 'redimnet2-b6-lm' | 'wavlm-large-sv'
    created_at        timestamptz NOT NULL DEFAULT now()
);
```

**Six design decisions worth calling out:**

1. **`render_target` is a first-class table, not a column.** The v0.1 → ZONOS2 transition proved encoder and LDA dims change under you. Vectors are foreign-keyed to the exact triple that minted them, so a version bump orphans them loudly rather than silently rendering the wrong voice. `licence_class` is on this table because licence viability is a property of the *backend choice*, and this project has already found two blockers there.
2. **`seed_audio_uri` is `NOT NULL` by constraint.** Tier 2 is not optional — it is the only artefact that survives an encoder change.
3. **Two vector columns.** pgvector stores up to 16,000 dims but **indexes only up to 2,000**, so a 2048-d ZONOS2 vector cannot be HNSW-indexed. Keeping the 1024-d LDA vector alongside solves that *and* is the correct space for similarity and interpolation anyway.
4. **`embedding_norm` is stored explicitly** because ZONOS2's own code treats magnitude as load-bearing (`preserve_norm=True`, "the magnitude the model was trained on"). Any blend or edit must restore it.
5. **`speaker_prior` is a table.** This is the schema-level encoding of the single most important finding in §2.3: generated vectors must come from a density fitted to real embeddings. Making it a first-class, versioned artefact keeps that discipline enforceable rather than buried in training code.
6. **Style is separate from identity**, mirroring `emotion_directions/`. This makes "same character, different emotion" cheap and keeps identity stable across a performance — and it is where raspy/whispered/aged must live, given §3.

---

## 7. What this means for the build

1. **Drop Zonos-v0.1 as a production target.** Two independent blockers: the speaker weights have **CC-BY-NC-SA-4.0 provenance**, and the runtime pulls **GPL-3.0** eSpeak/phonemizer. It remains useful as a *cheap local prototyping rig* (fits 12GB trivially, official GGUF quants) as long as nothing minted on it reaches production.
2. **Make ZONOS2 the primary target.** MIT + Apache throughout, clean Apache-2.0 encoder provenance, no GPL, first-class `.npy` vector input, shipped SLERP and vector arithmetic, Tier-3 Hindi/Tamil/Telugu/Bengali, active repo with a paper.
3. **Resolve the 12GB problem before committing.** ZONOS2 bf16 is 15.34 GB. Options: (a) ask Zyphra to licence `zonos2.cpp`; (b) test bf16 with CPU offload of inactive experts (900M active of 8B makes this plausible); (c) quantize ourselves from the MIT PyTorch repo. **This is a live risk to the locked "one 8-12GB consumer GPU" constraint and it is not yet solved.**
4. **Build the mapper as a conditional density model over a fitted prior — never a free vector regressor, never a standard-normal sampler.** Concretely: extract a large corpus of real embeddings, fit mean/covariance (and a MoG) in the **1024-d LDA space**, whiten, train `description → whitened LDA vector`, un-whiten, **restore the target norm**, map to 2048-d via the LDA pseudo-inverse. TacoSpawn is unambiguous that the prior is what makes this work.
5. **Have a fallback that cannot fail: constrain outputs to the hull of real anchors.** If free generation proves fragile, predict **simplex weights over a curated basis of real embeddings** (or SLERP coefficients, per INSIDE). This preserves the whole architecture — description still maps to a vector — while guaranteeing on-manifold outputs. Design for this from day one so the pivot is a loss-function change, not a rewrite.
6. **Steal four techniques verbatim from ZONOS2:** SLERP not LERP; `preserve_norm` after any perturbation; mean-difference directions from real populations for style; and the LDA-pseudo-inverse round-trip for targeting the model's true input space.
7. **Architect identity and style as two composable vectors.** Zyphra route whisper through audio prefixes and emotion through separate directions. Expecting a single identity vector to carry "raspy/whispered/aged" contradicts both the LDA's purpose and the vendor's own design.
8. **Evaluate with an encoder we never train against** — ReDimNet2-B6-LM or CAM++ (both Apache-2.0), adding WavLM-large SV for literature comparability with a licence flag. `Zyphra/ZTTS1-Eval` (Apache-2.0, 9 read + 17 in-the-wild languages) is a ready-made harness for [06-evaluation-harness.md](06-evaluation-harness.md).
9. **Prototype the navigability experiment on Kokoro first** — Apache-2.0 both sides, 54 real voice tensors, 256-d, runs on anything. It is the cheapest possible rig for testing interpolation and prior-sampling before spending GPU time on ZONOS2.
10. **Treat Indian-language quality as unproven and test it early.** ZONOS2 lists them at its weakest tier with no numbers. IndicF5 (MIT, 11 languages) and Indic Parler-TTS (Apache-2.0) are the clean-licence alternatives. This may be the constraint that ultimately picks the backend.

---

## 8. Open — must be settled by experiment

| Question | Cheapest experiment | Est. cost/time | What it blocks |
|---|---|---|---|
| **Is the speaker space navigable by synthesis?** | The scripted experiment below | **~1 hr GPU** | The entire two-tower design |
| Is the VoxBlink2 → Zonos licence chain fatal? | Legal review of §2.1; email Zyphra for written provenance | 1 wk (external) | Whether v0.1 can be used at all |
| Does ZONOS2 fit 12GB at acceptable quality *and* licence? | bf16 + expert offload, measure peak VRAM/RTF; open an issue asking Zyphra to licence `zonos2.cpp` | 2 hrs + an issue | The locked hardware constraint |
| Is `Qwen3SpeakerEmbedding.forward` pooled `(B,2048)` or a sequence? | `print(enc(wav, sr).shape)` | **5 min, CPU** | Whole mapper output shape |
| Is the LDA space more prior-friendly than raw embeddings? *(the §2.3(iii) hypothesis)* | Fit a Gaussian + MoG to both; compare log-likelihood and decoded quality | 3 hrs | Which space the mapper targets |
| Is Indian-language output usable? | 10 Hindi + 10 Tamil lines on ZONOS2 and IndicF5; native-speaker MOS | 1 day | Backend choice; possibly the product |
| Can 1024-d carry raspy/whispered/aged? | Embed labelled stylized speech; test linear separability; render mean-difference directions | 1 day | Whether style needs a second channel |
| **What does ElevenLabs Voice Design actually do?** | Fetch the official API reference directly | 30 min | Nothing structural — but the brief's Tier 2 claim is currently **UNVERIFIED** |
| Cross-version vector stability | Re-embed one seed clip under two encoder revisions; cosine | 30 min | Tier 1 migration policy |
| Do Kokoro voice-packs interpolate? | Average/SLERP two `.pt` voice tensors, render | 1 hr | A cheap clean-licence fallback |
| Chatterbox / IndicF5 / Indic-Parler vector interfaces | Read `inference()` signatures | 2 hrs | §5 completeness |

### The decisive experiment — speaker-space navigability (~1 hour)

**Hypothesis.** Vectors produced by *geodesic combination of real embeddings* and by *sampling a density fitted to the real embedding population* decode to coherent, natural, speaker-distinct voices; vectors from an uninformed prior do not.

**Setup.** ZONOS2 (`D_raw=2048`, `D_lda=1024`), or Kokoro (`D=256`) for a free dry run. One fixed text, one fixed seed, `accurate_mode=true`, emotion off. Judge = **ReDimNet2-B6-LM** (Apache-2.0), *not* the ECAPA target.

**Step 0 — characterise the real distribution (the step everyone skips, and the one TacoSpawn says decides the outcome).**
Extract `E = {e_1 … e_N}`, N ≈ 500, from diverse real clips (Common Voice / LibriTTS / an Indic corpus). Record `μ`, `Σ`, the distribution of `‖e‖`, and fit **both** a single Gaussian and a **mixture of Gaussians** (TacoSpawn's choice). Everything below is defined relative to these — **not** to `N(0, I)`.

**Step 1 — six arms, ~20 renders each.**

| Arm | Construction | Purpose |
|---|---|---|
| **A. Real** (control) | `e_i` unmodified | upper bound |
| **B. SLERP** | `slerp(e_i, e_j, t)`, `t ∈ {0.25, 0.5, 0.75}` | is the *between* region navigable? (expect YES — vendor ships it) |
| **C. Perturbed** | `e_i + σ·Lz`, `L=chol(Σ)`, `σ ∈ {0.1, 0.25, 0.5}`, rescaled to `‖e_i‖` | how far off a real point can we move? |
| **D. Gaussian-fit sample** | `e ~ N(μ, Σ)`, rescaled to median real norm | what a *free* mapper's outputs look like |
| **E. MoG sample** | `e ~ MoG(E)`, rescaled | **the arm that decides it** — TacoSpawn's actual method |
| **F. Naive random** | `torch.randn(D)` unscaled | negative control; expect failure |

**Step 2 — measure, per render.**
1. **Intelligibility:** WER (Whisper-L / Qwen3-ASR). *Primary artefact detector — "glossolalia babble" shows here first.*
2. **Naturalness:** UTMOS / DNSMOS.
3. **Distinctness:** pairwise judge-cosine *between* arm members — a space that collapses to one average voice fails here even at perfect WER. Compare against Arm A's spread. *(PromptTTS 2's 0.355 is a useful reference for "healthy diversity".)*
4. **Intra-identity consistency:** render the same vector twice with different seeds; judge-cosine. Low = the vector does not determine identity. *(INSIDE warns synthetic identities score* too *high here — record it, do not optimise it.)*
5. **Blind human listen:** 20 clips, arms shuffled, "is this a plausible single human voice?" 1–5.

**Step 3 — decision rule.**

| Result | Reading | Action |
|---|---|---|
| Arm E (and/or D) within ~10% of Arm A on WER and UTMOS, with distinctness spread comparable to A | **Space is navigable by synthesis** | Build the mapper as a conditional density over the fitted prior. Proceed as planned |
| E/D fail but B and C (σ ≤ 0.25) pass | **Locally navigable only** | **Pivot the mapper to simplex/SLERP weights over a curated basis of real embeddings.** Architecture preserved, outputs guaranteed on-manifold. *This is the outcome to plan for* |
| B, C, D, E all fail; only A works | **Not navigable — Tier 1 synthesis is dead** | **Pivot to Tier 2**: mapper emits a retrieval/selection over a voice bank, or design-a-waveform-then-clone (VoiceSculptor's route) |
| Arm F passes | Something is wrong with the test | Re-check the injection path |

**Why this design.** Arms D and E are the crux because a trained mapper's outputs are, distributionally, draws from a fitted model of the embedding distribution — not real points and not white noise. Arm E specifically replicates the one method with published near-parity MOS. Arms B/C bracket them, and the B-passes-E-fails outcome has a **specific, architecture-preserving fallback** that is worth far more than a binary yes/no. Total cost: one GPU-hour plus ~30 minutes of listening — and it can be dry-run for free on Kokoro first.

---

## 9. Sources

| # | URL | Type | Used for | Confidence |
|---|---|---|---|---|
| 1 | https://raw.githubusercontent.com/Zyphra/Zonos/main/LICENSE | Primary — LICENSE | v0.1 code Apache-2.0 | HIGH |
| 2 | https://huggingface.co/api/models/Zyphra/Zonos-v0.1-transformer | Primary — HF API | weights Apache-2.0; 1,624,411,136 params | HIGH |
| 3 | https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding (+ README raw) | Primary — model card | "based on ResNet293-SimAM-ASP from VoxBlink2… **we use the pretrain models**"; 256→128 | HIGH |
| 4 | https://github.com/VoxBlink2/ScriptsForVoxBlink2 + /asv/README.md | Primary — upstream repo | **CC BY-NC-SA 4.0** dataset; checkpoints carry no licence | HIGH |
| 5 | https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md | Primary — official docs | "pretrained model… follows the license of it's corresponding dataset" | HIGH |
| 6 | https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/speaker_cloning.py | Primary — source | `embd_dim=256`; no L2 anywhere; dead `ECAPA_TDNN` class | HIGH |
| 7 | https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/model.py | Primary — source | `make_speaker_embedding` → (1,128) bfloat16 | HIGH |
| 8 | https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/conditioning.py | Primary — source | `make_cond_dict`; dim-only assert; 109 lang codes; stale docstring | HIGH |
| 9 | `ResNet293_SimAM_ASP_base_LDA-128.pt` (HF resolve) | Primary — checkpoint | pickle metadata: weight DoubleStorage (128,256) — settles 256→128 | HIGH |
| 10 | *(own numpy analysis of #9)* | Derived — computation | rank 128, cond 61.5, row norms ~66, non-orthogonal | HIGH (numbers) / inference (implications) |
| 11 | https://raw.githubusercontent.com/Zyphra/Zonos/main/CONDITIONING_README.md | Primary — official docs | speaker conditioner spec; `speaker_noised` = acoustic noise | HIGH |
| 12 | https://www.zyphra.com/post/beta-release-of-zonos-v0-1 | Primary — official blog | Apache-2.0; 200k hrs; other languages "not robust" | HIGH |
| 13 | https://raw.githubusercontent.com/Zyphra/ZONOS2/main/LICENSE | Primary — LICENSE | ZONOS2 code **MIT** | HIGH |
| 14 | https://huggingface.co/api/models/Zyphra/ZONOS2 | Primary — HF API | weights Apache-2.0; `model.pth` 15.34 GB | HIGH |
| 15 | https://huggingface.co/Zyphra/ZONOS2/raw/main/params.json | Primary — checkpoint config | `speaker_embedding_dim 2048`, `speaker_lda_dim 1024`, MoE 16/top-1 | HIGH |
| 16 | https://arxiv.org/abs/2606.24320 (PDF extracted, 15 pp) | Primary — paper | §II-C speaker embeddings + `h_spk` equation; §VII-C "silent output or glossolalia babble"; eval scorer table | HIGH |
| 17 | https://raw.githubusercontent.com/Zyphra/ZONOS2/main/python/zonos2/server/api_server.py | Primary — official source | `_slerp_embeddings`; `speaker_blend_t`; `_load_embedding_vector`; `CachedSpeakerReference.source_type` | HIGH |
| 18 | https://raw.githubusercontent.com/Zyphra/ZONOS2/main/python/zonos2/tts/emotion.py | Primary — official source | mean-difference directions; `preserve_norm`; LDA-pinv round-trip | HIGH |
| 19 | https://raw.githubusercontent.com/Zyphra/ZONOS2/main/README.md + NOTICE + pyproject.toml | Primary — official docs | language tiers; `.npy` voices; **no phonemizer/eSpeak**; third-party notices | HIGH |
| 20 | https://raw.githubusercontent.com/Zyphra/ZONOS2/main/python/zonos2/models/speaker_cloning.py | Primary — source | `Qwen3SpeakerEmbedding` → `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` | HIGH |
| 21 | https://huggingface.co/marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B (API + config.json) | Primary — model card/config | Apache-2.0; `EcapaTdnnSpeakerEncoder`; `enc_dim 2048`; **12,001,088 params** | HIGH |
| 22 | https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base | Primary — HF API | Apache-2.0 — closes ZONOS2's encoder provenance chain | HIGH |
| 23 | https://huggingface.co/api/models/Zyphra/ZONOS2-GGUF | Primary — HF API | q4_k 4.92 / q6_k 6.79 / q8_0 8.54 / f16 15.34 GB | HIGH |
| 24 | https://api.github.com/repos/Zyphra/zonos2.cpp + raw LICENSE (**404**) | Primary — API + fetch | **no LICENSE file** — unlicensed runtime | HIGH |
| 25 | https://huggingface.co/Zyphra/ZONOS1-GGUF | Primary — model card | v0.1 quants; "prefix conditioner kept at full precision" | HIGH |
| 26 | https://api.github.com/repos/Zyphra/Zonos/commits | Primary — API | last commit 2025-03-05; 0 releases | HIGH |
| 27 | https://github.com/Zyphra/Zonos/pull/187#issuecomment-2727898782 | Primary — user measurement | 4,686 MiB on RTX 3090 | HIGH |
| 28 | https://github.com/Zyphra/ZONOS2/issues/9 | Primary — user report | ZONOS2 cloning possibly worse than v0.1 (single, unanswered) | MEDIUM |
| 29 | https://github.com/CrispStrobe/CrispASR/.../zonos_tts_reference.py | Primary — third-party code | only public `torch.randn(1,1,128)` injection; **no quality report** | HIGH (code) |
| 30 | https://arxiv.org/abs/2309.02285 (ar5iv full text) | Primary — paper | **PromptTTS 2**: diffusion variation net, no ref at inference; 93.33%; MOS 3.88; **sim 0.355** | HIGH |
| 31 | https://arxiv.org/abs/2111.05095 (ar5iv full text) | Primary — paper | **TacoSpawn**: MoG prior; MOS 3.62 vs 3.68; **d-vectors less amenable to a parametric prior**; normal prior worse | HIGH |
| 32 | https://arxiv.org/abs/2404.02677 | Primary — paper | VoicePrivacy 2024 B1 (averaged x-vectors, **WER 2.91%**), B3 (WGAN, 4.35%) | HIGH |
| 33 | https://arxiv.org/abs/2210.07002 | Primary — paper | WGAN speaker embeddings → intelligible speech, human-evaluated | HIGH |
| 34 | https://arxiv.org/abs/2508.19210 | Primary — paper | **INSIDE**: SLERP in pretrained speaker space; synthetic identities under-varied | HIGH |
| 35 | https://arxiv.org/abs/2210.09916 | Primary — paper | GMM-OT interpolation, no significant naturalness loss | HIGH |
| 36 | https://arxiv.org/abs/2407.04291 | Primary — paper | SV embeddings "suppress intra-speaker variability… discard variations crucial for generation" | HIGH |
| 37 | https://arxiv.org/abs/{2403.00529, 2309.14094, 2511.07135} | Primary — papers | VoxGenesis, VoiceLens, SpeakerVAE — generative speaker manifolds | HIGH |
| 38 | https://arxiv.org/abs/2601.10629 + github.com/ASLP-lab/VoiceSculptor | Primary — paper + repo | **VoiceSculptor exists**; "rendered into a prompt waveform… fed into a cloning model" = Tier 2; SOTA 67.6 vs ElevenLabs 50.9 | HIGH |
| 39 | https://huggingface.co/api/models/HKUSTAudio/{Llasa-3B,xcodec2} | Primary — HF API | both **cc-by-nc-4.0** — contradicts VoiceSculptor's Apache tag | HIGH |
| 40 | https://huggingface.co/coqui/XTTS-v2/raw/main/LICENSE.txt | Primary — LICENSE in weights repo | **CPML: "allows only non-commercial use"** | HIGH |
| 41 | https://raw.githubusercontent.com/coqui-ai/TTS/dev/TTS/{.models.json,tts/models/xtts.py} | Primary — registry + source | `xtts_v2: CPML`, `your_tts: CC BY-NC-ND 4.0`; `(1,32,1024)` + `(1,512,1)` | HIGH |
| 42 | coqui.ai and coqui.ai/cpml → **404** | Live HTTP probe | licensor defunct; canonical licence URL dead | HIGH |
| 43 | https://raw.githubusercontent.com/yl4579/StyleTTS2/main/LICENSE + HF tree API | Primary — LICENSE + API | MIT code; **checkpoints carry no licence**; GPL runtime dep; `ref_s` 256-d | HIGH |
| 44 | https://huggingface.co/hexgrad/Kokoro-82M (voices/*.pt downloaded) | Primary — model card + artefact | Apache-2.0 both sides; **(510,1,256) f32**; 54 voices, 4 Hindi; "no diffusion, no encoder" | HIGH |
| 45 | https://huggingface.co/api/models/{ResembleAI/chatterbox,ai4bharat/IndicF5,ai4bharat/indic-parler-tts} | Primary — HF API | MIT / MIT / Apache-2.0; Indian-language coverage | HIGH (licence only) |
| 46 | https://huggingface.co/api/models/speechbrain/spkrec-ecapa-voxceleb + hyperparams.yaml + SpeechBrain LICENSE | Primary — API + config | 192-d, Apache-2.0 both sides, **raw (not L2-normed)** | HIGH |
| 47 | https://huggingface.co/api/models?author=Wespeaker + per-model config.yaml | Primary — API + configs | dims 256/512/192; apache-2.0 vs cc-by-4.0 split | HIGH |
| 48 | https://raw.githubusercontent.com/microsoft/UniSpeech/main/{LICENSE,.../ecapa_tdnn.py} | Primary — LICENSE + source | **CC-BY-SA-3.0**; judge embedding 256-d | HIGH |
| 49 | https://raw.githubusercontent.com/BytedanceSpeech/seed-tts-eval/main/README.md + F5-TTS eval script | Primary — repos | de-facto judge = WavLM-large SV | HIGH |
| 50 | https://huggingface.co/api/models/nvidia/speakerverification_en_titanet_large + titanet-large.yaml | Primary — API + config | 192-d; **cc-by-4.0** weights | HIGH |
| 51 | https://raw.githubusercontent.com/resemble-ai/Resemblyzer/master/{LICENSE,voice_encoder.py} | Primary — repo | Apache-2.0; 256-d; **L2-normed** | HIGH |
| 52 | https://pypi.org/pypi/phonemizer/json; espeak-ng COPYING | Primary — metadata + LICENSE | GPLv3+ / GPL-3.0 | HIGH |
| 53 | https://huggingface.co/api/models/descript/dac_44khz | Primary — HF API | **no licence tag** on the weights | HIGH (that it is absent) |
| 54 | Zonos issues/PRs (250) + comments (858); 51 HF discussions; 24 Spaces; arXiv `all:"Zonos"`; S2 citations | Primary — exhaustive sweep | the v0.1 navigability negative | HIGH |
| 55 | Reddit (indirect web search only; direct access blocked) | — | no relevant threads surfaced | **LOW — coverage gap** |

---

*Pass 1 complete. **A2 fully settled** from primary sources, source code and the checkpoint artefacts themselves. **A1 settled** on first-party evidence plus PromptTTS 2 / TacoSpawn / VoiceSculptor. **C2 settled** with a clean recommendation and two named literature gaps. Still open and enumerated in §8: the ElevenLabs/PlayHT/Resemble API flows (the brief's claim is **UNVERIFIED**), the Chatterbox/IndicF5/Indic-Parler vector interfaces, and the wider voice-design paper survey — those belong to [03-tts-backends-english.md](03-tts-backends-english.md) and [08-licensing-propagation.md](08-licensing-propagation.md).*
