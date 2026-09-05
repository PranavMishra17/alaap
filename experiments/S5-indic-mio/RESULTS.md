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

## Three things the model card gets wrong

Worth recording, because each cost a debugging cycle:

1. **The codec pairing.** The card's example loads `MioCodec-25Hz-24kHz` and then writes the output at 44100 Hz. The 24 kHz variant has an integrated iSTFTHead and **no vocoder weights**, so `from_pretrained` raises `No vocoder weights found with prefix 'vocoder.'`. Indic-Mio outputs 44 kHz and pairs with **`MioCodec-25Hz-44.1kHz`**.
2. **Argument shapes.** The card builds `[1, 1, T]`. Every batched shape raises `too many values to unpack (expected 3)`. **Both arguments must be 1-D**: content `[T]`, global `[128]`. Probed exhaustively — 1 of 9 shape combinations works.
3. **Argument order.** The card calls `codec.decode(codes_tensor)` positionally, which binds the codes to `global_embedding`. The signature is `decode(global_embedding=…, content_token_indices=…)`.

## A measurement trap this nearly walked into

The first smoke test compared MioCodec global embeddings by **raw cosine** and printed *"NOT separable — red flag"*: two different speakers sat at 0.9882, the same speaker at 0.9970, a separation of 0.0088.

**That verdict was wrong, and it was the metric, not the codec.** These embeddings carry a large common-mode component, which is exactly why this project puts every Qwen3 vector through `SpeakerSpace` (centre → rescale → fit in-domain) before measuring anything — invariant I1 — and why `metrics.CALIBRATION` exists at all. A raw cosine on an uncentred embedding space is not a verification score.

The independent ECAPA numbers above are the honest measurement, and they are unambiguous. **Two speakers cannot calibrate anything**; S5b should fit `SpeakerSpace` over many MioCodec embeddings and report a proper EER, exactly as E4 did for Qwen3.

## Not established

- **Intelligibility is completely unmeasured.** Identity transfer says nothing about whether the Hindi and Tamil are *correct* — a voice can carry perfectly and say gibberish. `RESEARCH/04` names `ai4bharat/indic-conformer-600m-multilingual` as the eval ASR; a WER/CER pass is the next thing that matters, and this result should not be leaned on until it exists.
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
