# S5 — the two-tower split works in Indic

**Run:** 2026-09-05 · `SPRINGLab/Indic-Mio` (0.61B) + `Aratako/MioCodec-25Hz-44.1kHz` · Hindi, Tamil, English
**Question:** can a stored speaker vector carry a voice onto new Indic text, with the TTS frozen?

> This is `RESEARCH/04`'s **E1** — the experiment that document called *"the highest-value experiment in this file"* and *"the entire Tier-1 claim for Indic"*. It became runnable when the licence question was settled as research-only and the HF gate opened.

---

## Why this backend and not the one everything else runs on

`Qwen3-TTS` cannot speak any Indian language — `LANG_ALIAS` is ten languages, none Indian (`ADR-006`). That is a frozen-tower capability limit; no speaker-vector work touches it.

MioCodec is built around the split **as a documented API**. Its own card:

> MioCodec decomposes speech into two distinct components: **Content Tokens** … "what" is being said … and **Global Embeddings**, a continuous vector representing … speaker identity, recording environment, and microphone traits.

```
text ──> Indic-Mio LM ──> content tokens  ┐
                                          ├──> MioCodec.decode() ──> audio
identity ────────────> global embedding   ┘
```

`decode(global_embedding=…, content_token_indices=…)` accepts an **arbitrary 128-d tensor**. Unlike Qwen3's `x_vector_only_mode` inference flag, this is a public method — the split is the design, not a side effect.

## Result — it works

Four lines generated (Hindi, Hindi with a `<happy>` tag, Tamil, English), each decoded twice: the **same content tokens** through **two different speaker embeddings** taken from two real Hindi speakers.

**Scored by ECAPA-TDNN, which has never seen MioCodec** — the project's independent verifier, calibrated at C_same 0.6988 / C_diff 0.2011 on real speech:

| | raw cosine | normalised (0 = different speaker, 1 = same) |
|---|---|---|
| rendered vs **the donor whose embedding it got** | **0.5074** | **+0.615** |
| rendered vs the **other** donor | 0.1484 | −0.106 |
| margin | **+0.3590** | |
| correct direction | **8 / 8** | |

Every line, both donors, right direction. The rendered voice is **61.5% of the way from "a different person" to "the same person"**, and measures as *not* the other donor at all.

Both donors are **female** — this is the harder within-gender case, and the two real donors sit only 0.2151 apart under ECAPA themselves.

### Content and identity really are separable

The same content tokens produce two audibly different speakers. And the codec-level swap test is unambiguous — re-encoding a swapped render moves it **toward the donor** (0.9956) and **away from the original** (0.9875).

## ⚠️ The first run of this experiment used the WRONG CODEC

Recorded in full because it is the most instructive failure in the project so far,
and because an adversarial review caught it, not I.

The first run decoded Indic-Mio's tokens through **`MioCodec-25Hz-44.1kHz`** (the
legacy variant). The output was fluent, well-articulated Hindi **saying different
words than the prompt**:

```
PROMPT   : नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।
legacy   : अज़्ट उद आयार मुशिलो के लब शिबगो जो लिख चिए आया
corrected: नमस्ती आप कैसे है, आज मोसम बहुत अच्छा है
```

**Nothing raised.** Both codecs are FSQ with `levels [8,8,8,5,5]` = 12800 entries,
so every index is *valid* in either — the shared vocabulary size is a coincidence
of architecture, not a shared codebook. Measured on identical audio:

| | |
|---|---|
| exact token agreement, legacy vs 24 kHz | **0.0000%** (chance 0.0078%) |
| content-tokenizer weights, 24 kHz vs 44.1kHz-**v2** | **bit-identical**, 7/7 probe tensors |
| content-tokenizer weights, 24 kHz vs 44.1kHz-**legacy** | 0/7 identical |
| log-mel corr, legacy decoding 24 kHz's tokens | +0.418 (vs +0.94 for either codec on its own) |

**Every identity metric in this document was unaffected** — donor embeddings and
decoder came from the same codec, so that path was internally consistent. What was
false was the *content* premise: "the same content tokens … while the words stay".
The words were never the prompt's words.

**Why a listener did not catch it.** A fluent Hindi speaker was sent the audio and
reported it sounded good and was not gibberish. That was an honest answer to a
badly-formed question: they were never told what the sentence was supposed to say.
Fluent-sounding wrong words are indistinguishable from correct words without a
reference. **Ask "does this say X?", never "does this sound right?"**

### What was actually wrong with the model card

My original three claims went out for adversarial review before filing. One was
refuted, and it was the one I was least sure of:

1. **"The card names the wrong codec" — REFUTED.** The card is right: the base model
   `MioTTS-0.6B` states twice that it uses `MioCodec-25Hz-24kHz`, and Indic-Mio is a
   ~6-hour finetune of it. **The real card bug is the sample rate** — it loads a
   24 000 Hz codec and writes the wav at `44100`, a 1.84× speed-up.
2. **`MioCodec` vs `MioCodecModel` — my error, not a library bug.** The 24 kHz and v2
   variants have an integrated iSTFT head and no external vocoder; `MioCodecModel` is
   their documented loader. `MioCodec`'s `No vocoder weights found with prefix
   'vocoder.'` is a *correct refusal*, and the class I should have used is the first
   thing MioCodec's README documents.
3. **Tensor shapes — CONFIRMED.** 1 of 9 shape combinations works; both arguments must
   be 1-D. `decode_batch` is the intended batched path.
4. **Positional argument — CONFIRMED, and understated.** `codec.decode(codes_tensor)`
   binds codes to `global_embedding`. The deeper problem is that the card's example has
   **no speaker input at all** — it cannot be repaired by reordering arguments.

This experiment now uses **`MioCodec-25Hz-44.1kHz-v2`**, whose tokenizer is bit-identical
to the 24 kHz model's and which is natively 44.1 kHz — matching what the authors'
own `MioTTS-Inference` defaults to.

## A measurement trap this nearly walked into## A measurement trap this nearly walked into

The first smoke test compared MioCodec global embeddings by **raw cosine** and printed *"NOT separable — red flag"*: two different speakers sat at 0.9882, the same speaker at 0.9970, a separation of 0.0088.

**That verdict was wrong, and it was the metric, not the codec.** These embeddings carry a large common-mode component, which is exactly why this project puts every Qwen3 vector through `SpeakerSpace` (centre → rescale → fit in-domain) before measuring anything — invariant I1 — and why `metrics.CALIBRATION` exists at all. A raw cosine on an uncentred embedding space is not a verification score.

The independent ECAPA numbers above are the honest measurement, and they are unambiguous. **Two speakers cannot calibrate anything**; S5b should fit `SpeakerSpace` over many MioCodec embeddings and report a proper EER, exactly as E4 did for Qwen3.

## Not established

- **Intelligibility is now measured** (S5b, `run_intelligibility.py`), and it was the check that exposed the codec error. With the corrected codec: **CER 0.000 on Tamil and English**, 0.114 mean on Hindi — and the Hindi residual is `whisper-small`'s Devanagari orthography (नमस्ते→नमस्ती, मौसम→मोसम), not synthesis error. Critically, **CER spread between the two donors is 0.000 on every line**: who speaks has no effect on what is said, which is the two towers being genuinely independent. `ai4bharat/indic-conformer-600m-multilingual` is the better instrument and is gated pending one accept click.
- **Nobody has listened yet.** Everything here is machine-scored.
- 2 donors, 4 lines, 8 comparisons, one seed. Small.
- No drift/consistency floors, no minting from a *description* — this carries an **existing** speaker's vector, it does not yet mint a new one from text. That is the next step and the one that makes it a two-tower *product* rather than a voice cloner.
- The global embedding mixes speaker with **recording environment and microphone** by the card's own admission, so some of the transfer may be channel rather than voice (`RESEARCH/04`'s E3).
- Licence: Indic-Mio's chain includes NC data (`RESEARCH/08` §4.4). Fine under the research-only posture recorded in `ADR-009`; it cannot ship weights or, arguably, generated audio.

## Reproduce

```bash
envs/qwen3/Scripts/python.exe experiments/S5-indic-mio/run_two_tower.py
```

~1 minute on a 6 GB card. Needs `HF_TOKEN` (gated corpus) and `miocodec`, installed with `--no-deps` so it cannot pull a torch that breaks `qwen-tts`'s pinned stack.
