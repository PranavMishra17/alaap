# Study guide — from theoretical ML to reading Alaap's papers unaided

> **Target:** ~5 h/week, ~13 weeks, ~65 hours.
> **Starting point assumed:** you know ML theory and algorithms. You do **not** know PyTorch,
> and you have not worked with audio.
> **End state:** you can read every paper this project rests on, argue with its backend
> choices, and change an experiment and trust the result.

**Paper citations:** arXiv IDs are given **only** where [`RESEARCH/SOURCES.md`](../RESEARCH/SOURCES.md)
verifies them. Everything else is title + author + year — search the title. This is
deliberate: a wrong arXiv digit sends you to a different paper.

---

## How to use this

Each unit has the same four parts:

| | |
|---|---|
| **⟶ After this you can** | the capability, stated as something you can do |
| **Read** | the minimum, not the maximum |
| **Do** | a concrete task **in this repo** — this is where it sticks |
| **Check** | a question. If you cannot answer it, do not move on |

**The `Do` steps matter more than the reading.** You know theory already; what you lack is
the muscle memory of tensors and the ear for audio. Both come from touching things.

### Where tinytorch fits

You are taking Stanford's tinytorch. It covers Part 1 below far better than any reading list
could, because you build the thing rather than reading about it. So:

- **Run tinytorch as your Part 1**, weeks 1–4, and use my Part 1 only as a *checklist of what
  you should be able to do when it ends*.
- Adjust the week numbers here to tinytorch's actual pacing. If it runs 6 weeks, push
  everything after it back two weeks — the ordering is what matters, not the calendar.
- **Do not wait for it to finish before starting Part 2.** Audio/DSP has no PyTorch
  dependency and is the other half of this project. Run it in parallel from week 3.

```
  wk  1   2   3   4   5   6   7   8   9  10  11  12  13
      ├───tinytorch / Part 1───┤
              ├──Part 2 audio──┤
                          ├─P3 speaker─┤
                                  ├──P4 TTS──┤
                                              ├P5┤
                                                  ├P6┤
                                                      ├P7┤
```

---

# Part 0 · Orientation — 2 hours, do this first

**⟶ After this you can** describe what Alaap does, and run it.

**Read** — in this order, and nothing else yet:
1. [`learning/architecture.html`](architecture.html) — open it in a browser. Diagrams 1–5.
2. [`README.md`](../README.md)
3. [`RESEARCH/00-EXECUTIVE-VERDICT.md`](../RESEARCH/00-EXECUTIVE-VERDICT.md)
4. [`learning/00-SANITY-CHECK.md`](00-SANITY-CHECK.md) — where the project actually is

**Do**
```bash
envs/qwen3/Scripts/python.exe scripts/demo_script_render.py
```
Four minutes. Then **listen to `demo_out/audio/`**. No number in this repo substitutes for
that.

**Check** — Why is the TTS model never trained? What would break if it were?

---

# Part 1 · PyTorch and tensors — weeks 1–4, 20 h

> **This is tinytorch's job.** Below is the exit checklist, not a substitute syllabus.

**⟶ After this you can** read any model file in this repo without guessing what a line does.

### 1.1 The tensor
Shapes, dtypes, devices, broadcasting, `view`/`reshape`/`permute`, indexing.

**Intuition:** a tensor is a numpy array that remembers where it came from. That memory *is*
autograd.

### 1.2 Autograd
`requires_grad`, the graph, `backward()`, `torch.no_grad()`, why `.detach()` exists.

**Intuition:** the graph is built forward as a side effect of computing, then walked
backward. Nothing is symbolic; it is a recorded tape.

### 1.3 `nn.Module`
`__init__` vs `forward`, parameters vs buffers, `train()`/`eval()`, `state_dict`.

**Why it matters here:** `model.eval()` and `torch.no_grad()` appear in every experiment in
this repo. Know exactly what each turns off — they are not the same thing.

### 1.4 The training loop
Dataset/DataLoader, loss, optimiser, `zero_grad`, LR schedules.

### 1.5 The things that bite
- CPU vs CUDA, and `.to(device)` on **both** model and data
- memory: activations dominate, not weights
- `float32` vs `bfloat16`
- **why a model that "fits in VRAM" still OOMs**

> **This repo's own version of that lesson:** loading two models in one process segfaults on
> a 15.7 GB box while the 6 GB GPU sits at 614 MiB. It is **system RAM**, not VRAM. See
> `HANDOFF.md` §8b.

**Do** — in this repo, no training required:
```python
import torch, torchaudio
m = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model().eval()
x = torch.randn(1, 16000)          # 1 second at 16 kHz
with torch.no_grad():
    hs, _ = m.extract_features(x, num_layers=12)
print(len(hs), hs[0].shape)
```
Then explain, out loud: why 12 tensors? What are the three dimensions of each? Why
`no_grad()`?

**Check** — Why does `hs[0].shape[1]` change when you change the input length, and what is the
ratio? *(This is the frame rate, and it is the first properly "audio" idea.)*

---

# Part 2 · Sound as data — weeks 3–5, 12 h

**⟶ After this you can** read `alaap/acoustics.py` and say what every function measures and
how it could be wrong.

### 2.1 Sampling and the time domain — 2 h
Sample rate, Nyquist, bit depth, why 16 kHz / 22.05 kHz / 24 kHz / 44.1 kHz all appear here.

**Intuition:** a waveform is a very long list of air-pressure readings. Nyquist says you can
represent frequencies up to half your sample rate and nothing above.

**Why it matters:** this project's audio lives at **four** sample rates — corpus 24 kHz,
WavLM 16 kHz, MioCodec 44.1 kHz, Sarvam 22.05 kHz — and analysing one at another's rate
puts every frequency off by a ratio. That happened in S21 phase B and *reversed* a result.

> **Do:** load a clip at 24 kHz, resample to 16 kHz, plot both spectra. Find the cliff.

### 2.2 The frequency domain — 3 h
DFT/FFT, window functions, the **STFT**, the spectrogram, mel scale, MFCCs.

**Intuition:** the STFT is "run an FFT on a sliding window". Every parameter is a trade:
a long window gives fine frequency resolution and blurry time; a short one, the reverse.

**Read** — any good DSP intro; Smith's *Mathematics of the DFT* if you want rigour.

**Do** — in this repo:
```python
import librosa, numpy as np
S = np.abs(librosa.stft(wav, n_fft=2048, hop_length=512))
```
`n_fft=2048` at 24 kHz gives what frequency resolution? What time resolution? Both numbers
appear in `spectral_tilt` and both matter.

### 2.3 The source–filter model — 3 h
Glottal source (pitch, F0) vs vocal-tract filter (formants). Why they are separable and why
that separation is the whole basis of describing a voice.

**Intuition:** the vocal folds make a buzz at F0. The mouth and throat are a tube that
resonates at certain frequencies — **formants**. Change the buzz, same person. Change the
tube, different person.

**This is the single most important idea in Part 2.** `f0_mean` measures the source;
`vtl_cm` and the formants measure the filter; `spectral_tilt` is supposed to measure the
source's spectral shape and — as S21 found — partly measures the buzz rate instead.

### 2.4 The estimators, and how each one lies — 4 h
Read `alaap/acoustics.py` top to bottom. Every docstring is a lesson someone paid for.

| function | the trap |
|---|---|
| `f0_track` | pYIN; octave errors on creaky voice |
| `voiced_mask` | an energy+ZCR VAD, **not** a periodicity detector; structurally capped near 0.60 and bimodal |
| `formants` | LPC order and analysis rate; run at ~2× max formant, not native |
| `vocal_tract_length` | **never average through F2** — it is vowel-dependent |
| `spectral_tilt` | OLS over linear FFT bins puts 43% of leverage on 27 bins |
| `speaking_rate` | needs text; silently NaN without it; g2p-en does not raise on Devanagari |

**Read** the project skill `.claude/skills/speech-feature-pitfalls/SKILL.md` — it is the
compressed version of everything above, written from real failures here.

**Check** — Why does a 250 Hz voice read a *flatter* spectral tilt than a 90 Hz voice with an
identical spectral envelope? *(Answer in `experiments/S21-spectral-tilt/RESULTS.md`.)*

---

# Part 3 · Speaker identity — weeks 6–7, 10 h

**⟶ After this you can** read the three papers the whole architecture rests on, and explain
why a speaker embedding is not just "an embedding".

### 3.1 From i-vectors to ECAPA — 3 h
**Read**
- Snyder et al., *"X-Vectors: Robust DNN Embeddings for Speaker Recognition"*, ICASSP 2018
- Desplanques et al., *"ECAPA-TDNN: Emphasized Channel Attention, Propagation and
  Aggregation in TDNN Based Speaker Verification"*, Interspeech 2020

**Intuition:** train a network to tell speakers apart; throw away the classifier; the
penultimate layer is now a *speaker embedding*. Everything downstream inherits whatever that
training made it invariant to.

**Why it matters here — and it is a rule, not a detail:**
> **ECAPA is trained to be INVARIANT to channel and quality**, so it can recognise you down a
> bad phone line. That is exactly why it must never be used to measure quality. It is the
> project's *independent judge* for identity, and nothing else.

### 3.2 Cosine, EER, and calibration — 2 h
Cosine similarity, thresholds, **EER**, and why a threshold from a paper does not transfer.

**Do** — read `alaap/metrics.py`'s `CALIBRATION` and E4's results. `C_same = 0.6988`,
`C_diff = 0.2011`. Then answer: why must these be fit **in-domain**? *(E4 measured that doing
so halves EER.)*

### 3.3 Is the space navigable? — 3 h
**This is the intellectual core of the project.** Three papers, all in `SOURCES.md`:

| | |
|---|---|
| [arXiv:1806.04558](https://arxiv.org/abs/1806.04558) §3.6 — Jia et al., NeurIPS 2018 | random unit-hypersphere vectors → MOS 3.65, "as natural as real speakers". **The manifold is navigable.** |
| [arXiv:2111.05095](https://arxiv.org/abs/2111.05095) §6.2 — TacoSpawn | **discriminative d-vector spaces resist parametric priors** (g2s 0.35 vs s2s 0.20). Learned spaces do not. |
| [arXiv:2207.04834](https://arxiv.org/abs/2207.04834) §5.1.1 — Meyer et al. | **the collapse and its fix.** Per-dim rescaling: GVD −6.50 → −0.14. Dimension ranges span three orders of magnitude. |

**Intuition for the third one:** if one dimension ranges over ±80 and another over ±0.5,
then an unweighted MSE loss is *entirely* about the first dimension. Every voice collapses
onto the same one. Rescale per dimension or the model learns nothing.

### 3.4 Moving through the space — 2 h
Interpolation vs extrapolation; SLERP vs LERP; sparse regions.

| | |
|---|---|
| [arXiv:2106.05762](https://arxiv.org/abs/2106.05762) | 1,225 pairwise interpolations, WER 6.42–7.35%. **Interpolation works.** |
| [arXiv:2310.03538](https://arxiv.org/abs/2310.03538) | *"extrapolation is not particularly meaningful"* |
| [arXiv:2508.19210](https://arxiv.org/abs/2508.19210) | **SLERP not LERP** — preserves unit norm |
| [arXiv:2005.08601](https://arxiv.org/abs/2005.08601) | sparse-region averaging costs **+60% relative WER** |

**Check** — Why does `mint(novelty=0.7)` produce voices *closer* to the centre of the space,
not further from it? *(S9. It is not obvious, and the experiment retracted a claim over it.)*

---

# Part 4 · How a TTS model actually works — weeks 8–9, 10 h

**⟶ After this you can** read the model card of any open TTS system and predict whether it
can serve this architecture.

### 4.1 The shape of the field — 2 h
Autoregressive vs non-autoregressive; two-stage (text→tokens→waveform) vs end-to-end.

**Read** — Kim et al., *"VITS: Conditional Variational Autoencoder with Adversarial Learning
for End-to-End Text-to-Speech"*, 2021. It is the clean end-to-end baseline everything else
is a reaction to.

### 4.2 Neural audio codecs — 3 h · **the key unlock**
**Read**
- Zeghidour et al., *"SoundStream: An End-to-End Neural Audio Codec"*, 2021
- Défossez et al., *"High Fidelity Neural Audio Compression"* (EnCodec), 2022
- Mentzer et al., *"Finite Scalar Quantization: VQ-VAE Made Simple"*, 2023 — **FSQ**, which
  is what MioCodec uses

**Intuition:** a codec learns to turn a waveform into a short sequence of integers and back.
Once audio is integers, a language model can generate it. That is why modern TTS is a
language model.

**Why FSQ specifically:** it replaces a learned codebook with a fixed grid — quantise each
of 5 dimensions to `[8,8,8,5,5]` levels, giving 12 800 combinations, no codebook collapse,
no auxiliary losses.

> **And the trap, which cost this project a week:** two FSQ codecs with the *same* levels
> share **no codebook**. Every index is valid in both. Nothing raises. You get fluent speech
> saying different words. Diagram 10 in `architecture.html`.

### 4.3 Self-supervised speech representations — 2 h
**Read** — Chen et al., *"WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack
Speech Processing"*, 2022. HuBERT if you want the predecessor.

**Intuition:** train on unlabelled audio by masking and predicting. Different layers end up
encoding different things — early layers pitch and speaker, middle layers phonetic content.

**Why it matters here:** the naturalness gate is *distance to the real-speech manifold in
WavLM features*, and **which layer you pool changes what you measure**. Layer 0 correlates
with F0 at −0.578, which is the DNSMOS trap in disguise. See S20b and S23.

### 4.4 Flow matching and diffusion, enough to argue — 3 h
**Read** — Lipman et al., *"Flow Matching for Generative Modeling"*, 2023. Skim the maths;
get the picture.

**Intuition:** diffusion learns to reverse noise step by step. Flow matching learns a
velocity field that transports noise to data along a straight-ish path — fewer steps, same
destination.

**The one A/B that matters here**, from `SOURCES.md`:
> [arXiv:2406.08812](https://arxiv.org/abs/2406.08812), Interspeech 2024 — flow matching wins
> **fidelity** (FAD 3.559 vs 5.244) and **loses adherence** (SRCC 0.60 vs 0.74). A hybrid is
> best. **The trade is bidirectional** — which is why this project's S3 plans MDN *first*.

---

# Part 5 · Description → voice — week 10, 5 h

**⟶ After this you can** read the mapper code and the paper it copies, and explain the design
choice in `top_k=2`.

### 5.1 Mixture Density Networks — 2 h
**Read** — Bishop, *"Mixture Density Networks"*, 1994 (tech report, ~20 pages, very readable).

**Intuition:** a normal regressor predicts one output per input. But one description
("a warm low voice") corresponds to *many* valid voices. An MDN predicts a **mixture of
Gaussians** instead of a point, so the model can say "any of these".

**Why it matters:** it is the difference between one voice per description and a family. The
reference implementation is `MDNLayer(256, 256, num_gaussians=10, dim_wise=True)`.

### 5.2 PromptTTS++ — 2 h · **the reference implementation**
**Read** — Shimizu et al., *"PromptTTS++: Controlling Speaker Identity in Prompt-Based
Text-to-Speech Using Natural Language Descriptions"*, 2024. Then the code:
[`line/promptttspp`](https://github.com/line/promptttspp) — Apache-2.0, weights downloadable.

This is the **only** open working implementation of exactly the contract Alaap needs:
natural-language description → speaker embedding → frozen TTS.

### 5.3 Data-Speech and measure-first captions — 1 h
**Read** — Lyth & King, *"Natural language guidance of high-fidelity text-to-speech with
synthetic annotations"*, 2024 (the Parler-TTS / Data-Speech line).

**Do** — compare against `alaap/acoustics.py`: Data-Speech computes **9 columns from 5
tools**, equal-width bins. Alaap uses **percentile** bins and adds six attributes Data-Speech
has not got. Read `Binner`'s docstring for why percentile — it is a real finding about
skewed corpora.

**Check** — Why does `Binner` regress HNR on log-F0 before binning it? *(39.9% of HNR
variance was pitch, a measurement artefact — and it made "a high, harsh voice" impossible to
satisfy.)*

---

# Part 6 · Evaluation, and why most of it is wrong — week 11, 5 h

**⟶ After this you can** design a measurement that will not fool you. This is the part of the
project with the highest ratio of hard-won knowledge to published literature.

### 6.1 The MOS-predictor trap — 1.5 h
**Read** — Reddy et al. *DNSMOS* (2021) and Saeki et al. *UTMOS* (2022), then
`RESEARCH/06-evaluation-harness.md` §5.2.

| correlation with mean F0 | |
|---|---|
| humans | **−0.059** |
| DNSMOS | **−0.788** |
| UTMOSv2 | −0.722 |

**They also move less than 0.1 on prosodic corruption that costs humans 1.84 MOS points.**
For a project whose point is spanning the pitch range, a MOS gate systematically rejects
high-pitched voices for a reason humans do not share. Invariant `I9`.

### 6.2 Diversity: the Vendi score — 1 h
**Read** — Friedman & Dieng, *"The Vendi Score: A Diversity Evaluation Metric for Machine
Learning"*, 2023.

**Intuition:** the exponential of the entropy of the eigenvalues of a similarity matrix. If
you have 40 items that are really 20 distinct things, Vendi × n ≈ 20 — "effective voices".

> **The trap this repo hit:** normalised Vendi is a fraction **of the maximum for a set of
> that size**. 0.361 over 95 items and 0.362 over 38 are *not* the same thing — they are ~34
> and ~14 effective voices. S9 retracted a published claim over this.

### 6.3 Intelligibility — 0.5 h
WER/CER, ASR as a judge, and why CER not WER for Indic scripts.

### 6.4 Listening tests — 2 h · **read the skill, it is better than the literature**
**Read** — `.claude/skills/listening-tests/SKILL.md`, then
`experiments/S14-rate-control/RESULTS-S14c.md` as a worked failure.

The compressed lessons:
- **Comparative, not single-stimulus**, when the baseline is itself imperfect
- Positive **and** negative controls, always
- **Write the discard condition down before sending the set** — then honour it
- **Compute the false-positive rate of that condition.** S14c's fired on a coin flip
- Ask **"does this say X?"**, never "does this sound right?"

**Check** — Design a test for "does the rate knob sound processed?" without letting the
listener answer on clip length. *(There is a real tension here; S14b and S14c both failed it
in opposite ways.)*

---

# Part 7 · The roads not taken — weeks 12–13, 6 h

**⟶ After this you can** argue about backend choice rather than inherit it.

For each: what it is, why it was rejected, and what would change the answer.

| system | why not | what would change it |
|---|---|---|
| **CosyVoice 2/3** | speaker vector exists but `use_spk_embedding: False`, no `spk2info.pt` shipped. Best-shaped *control* surface though — separate `instruct_text`, `speed`, `<strong>`, `[laughter]` | it is the documented fallback if the Direction channel had failed |
| **F5-TTS** | flow-matching, strong; but IndicF5 is a fine-tune of it under **CC-BY-NC-4.0** despite an MIT card | a clean-licence Indic fine-tune |
| **Zonos / ZONOS2** | v0.1 encoder derives from VoxBlink2 (CC-BY-NC-SA); ZONOS2 is **15.34 GB bf16, does not fit 12 GB**. Ships SLERP blending + additive emotion vectors — the same ideas as here | more VRAM, and a licence trace |
| **IndexTTS-2** | licence bars using it **or its outputs** to improve any AI model; "Use" includes *running* | nothing — it is structurally incompatible with a training project |
| **XTTS-v2** | code MPL-2.0, **weights CPML**; Coqui defunct, the fork cannot relicense | nothing |
| **Indic Parler-TTS** | **zero speaker-embedding code.** Identity is a *name token* in the same string as style | a fork that exposes a vector |
| **Kokoro-82M** | tiny and good; `ref_s[:,:128]`→decoder, `[128:]`→prosody is an undocumented free timbre/prosody split | worth revisiting for a small/fast tier |

**Read** — `RESEARCH/03-tts-backends-english.md` and `RESEARCH/04-indic-track.md` in full.
They are the most reusable thing in the repo: primary-source verification of a dozen systems.

**Do** — pick one rejected system and write a one-page case for reopening it. Then check
whether `RESEARCH/08-licensing-propagation.md` already kills your case.

**Check** — Why does invariant `I4` say a licence declaration is "a claim, not evidence"?
How many chains were mis-declared out of how many checked?

---

## The five ideas that took this project longest to learn

If you remember nothing else:

1. **A suspiciously good or suspiciously bad number is a bug report about your setup.**
   Recorded ~17 times here. Every single time.
2. **State the property before you read the result.** Otherwise you will rationalise.
3. **An axis that fails to separate is a bug in the estimator until proven otherwise.**
   Two "noise" axes turned out to be broken measurements.
4. **Effect size is not significance.** A gap of 8 points with a standard error of 6 is
   nothing, and it will look like a finding.
5. **A metric is validated on a corpus, not in the abstract.** Re-run the validation when the
   corpus, model, or codec changes.

These are the three project skills in `.claude/skills/` — `statistical-claims`,
`speech-feature-pitfalls`, `listening-tests`. Read them at week 1 and again at week 13; they
will mean different things.

---

## If you only have 10 hours, not 65

The ruthless subset, in order:

1. Part 0 (2 h) — orientation and the demo
2. §2.3 source–filter (1 h) — nothing else in audio makes sense without it
3. §3.3 the three navigability papers (3 h) — the intellectual core
4. §4.2 neural codecs (2 h) — why modern TTS is a language model
5. §6.1 + §6.4 the evaluation traps (2 h) — what stops you fooling yourself

That gets you able to follow any conversation about this project. It does not get you able to
change an experiment — that needs Part 1.
