# Upstream reports — ready to post

Three issues for **`SPRINGLab/Indic-Mio`**, one optional nit for **`Aratako/MioCodec`**.

Everything below was **adversarially re-verified** before being written. My first draft
had three claims; one was refuted and one was my own mistake, and both are excluded.
What remains is only what reproduces.

**Where to post**

| | link |
|---|---|
| Indic-Mio discussions (5 existing threads, maintainer active) | <https://huggingface.co/SPRINGLab/Indic-Mio/discussions> |
| New Indic-Mio discussion | <https://huggingface.co/SPRINGLab/Indic-Mio/discussions/new> |
| MioCodec issues (MIT, 28★) | <https://github.com/Aratako/MioCodec/issues> |
| The card being reported | <https://huggingface.co/SPRINGLab/Indic-Mio> |

**Versions to cite.** `miocodec` self-reports `0.1.0` but the repo has **no tags or
releases**, so cite the commit: **`77473544375d57e96cbdfd5d7d257e8f280fa8e3`**
(current `main`). Models: `SPRINGLab/Indic-Mio` @ `25feace0`,
`MioCodec-25Hz-24kHz` @ `3a737f0d`, `MioCodec-25Hz-44.1kHz-v2` @ `67faba34`.
torch 2.5.1+cu121, transformers 4.57.3, Windows.

---

## Report 1 — the Transformers example cannot run as written

**Post to:** Indic-Mio discussions · **Suggested title:** `Transformers example in the model card is not runnable (shapes, argument binding, missing speaker input, sample rate)`

> Thanks for Indic-Mio — the 22-language coverage and the emotion tags are genuinely
> useful, and the model itself works well once driven correctly.
>
> The **Approach 2: Directly with Transformers** snippet on the card cannot run as
> written. Four separate problems, smallest first. All reproduced against `miocodec`
> commit `7747354437`, transformers 4.51.3, torch 2.5.1.
>
> **1. The codes are passed positionally into the wrong parameter.**
>
> The card has `wav = codec.decode(codes_tensor)`. The signature is:
>
> ```python
> MioCodec.decode(self, global_embedding=None, content_token_indices=None,
>                 content_embedding=None, target_audio_length=None, features=None)
> ```
>
> so `codes_tensor` binds to `global_embedding` and the content is never supplied:
>
> ```
> ValueError: Either content_token_indices or content_embedding must be provided.
> ```
>
> **2. There is no speaker input anywhere in the example.** `global_embedding` is
> required, and the snippet has no reference audio, no preset and no
> `synthesize_from_tokens` call. This one cannot be fixed by reordering arguments —
> a required input is missing entirely.
>
> **3. The tensor shape is wrong.** The card builds `[1, 1, T]` via
> `torch.tensor([audio_codes]).unsqueeze(0)`. `decode` unconditionally does
> `content_embedding.unsqueeze(0)` internally, so anything pre-batched becomes 4-D
> and fails in `module/transformer.py:594` at `bsz, seqlen, _dim = x.shape`:
>
> ```
> ValueError: too many values to unpack (expected 3)
> ```
>
> I tested all 9 combinations of `content ∈ {[T], [1,T], [1,1,T]}` ×
> `global ∈ {[128], [1,128], [1,1,128]}`. **Exactly one works: both 1-D.**
> (`decode_batch` is the batched entry point and works as documented with
> `[B, dim]` / `[B, max_seq_len]` — so this is a card issue, not a library one.)
>
> **4. The sample rate is wrong by 1.84×.** The example loads
> `Aratako/MioCodec-25Hz-24kHz`, whose `config.sample_rate == 24000`, then writes
> `sf.write("output.wav", ..., 44100)`. The result plays 1.84× too fast.
>
> **Minimal reproduction** (needs only the codec, no LM download):
>
> ```python
> import torch, inspect
> from miocodec import MioCodec, MioCodecModel   # commit 7747354437
>
> # the card's loader class refuses the card's codec -- these variants have an
> # integrated iSTFT head and no external vocoder, so MioCodecModel is the loader
> try:
>     MioCodec.from_pretrained("Aratako/MioCodec-25Hz-24kHz")
> except ValueError as e:
>     print("card's class ->", e)   # No vocoder weights found with prefix 'vocoder.'
>
> m = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-24kHz").eval()
> print(m.config.sample_rate)       # 24000  -- but the card writes at 44100
>
> codes = torch.randint(0, 12800, (50,), dtype=torch.long)
> g = torch.randn(128)
>
> try:                                            # problem 3
>     m.decode(global_embedding=g, content_token_indices=codes[None, None, :])
> except ValueError as e:
>     print("card's shape ->", e)   # too many values to unpack (expected 3)
>
> try:                                            # problems 1 + 2
>     m.decode(torch.tensor([codes.tolist()]).unsqueeze(0))
> except ValueError as e:
>     print("card's call  ->", e)   # Either content_token_indices or ... must be provided
>
> m.decode(global_embedding=g, content_token_indices=codes)   # the only working form
> ```
>
> **Suggested replacement for the card's Approach 2:**
>
> ```python
> import torch, soundfile as sf
> from transformers import AutoTokenizer, AutoModelForCausalLM
> from miocodec import MioCodecModel, load_audio
>
> tokenizer = AutoTokenizer.from_pretrained("SPRINGLab/Indic-Mio", trust_remote_code=True)
> model = AutoModelForCausalLM.from_pretrained(
>     "SPRINGLab/Indic-Mio", dtype=torch.bfloat16, device_map="cuda")
>
> prompt = tokenizer.apply_chat_template(
>     [{"role": "user", "content": "नमस्ते, आप कैसे हैं?"}],
>     tokenize=False, add_generation_prompt=True)
> inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
> output = model.generate(**inputs, max_new_tokens=1024, do_sample=True,
>                         temperature=0.9, top_p=0.9)
>
> generated = output[0][inputs["input_ids"].shape[1]:]
> SPEECH_OFFSET = tokenizer.convert_tokens_to_ids("<|s_0|>")   # 151669, derived
> audio_codes = [t.item() - SPEECH_OFFSET for t in generated
>                if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + 12800]
>
> # MioCodecModel, not MioCodec: integrated iSTFT head, no external vocoder
> codec = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz-v2").eval().cuda()
> reference = load_audio("reference_speaker.wav",
>                        sample_rate=codec.config.sample_rate).cuda()
> wav = codec.synthesize_from_tokens(audio_codes, reference)
> sf.write("output.wav", wav.cpu().numpy(), codec.config.sample_rate)  # never hardcoded
> ```
>
> Deriving `SPEECH_OFFSET` from the tokenizer rather than hardcoding `151669` also
> makes the example robust to a future vocab change. I verified the card's value is
> correct today: `<|s_0|>` → 151669, `<|s_12799|>` → 164468, contiguous, exactly
> 12800 tokens.
>
> Minor: the snippet is fenced as ` ```bash ` but is Python.

---

## Report 2 — please document which codec pairs with which loader

**Post to:** Indic-Mio discussions · **Suggested title:** `Which MioCodec variant should Indic-Mio use? The three are not interchangeable`

> This one cost me the most time and I think it will catch others, so it may be worth
> a line on the card.
>
> There are three MioCodec variants, and **their content-token spaces are not
> interchangeable**, even though all three are FSQ with `levels [8,8,8,5,5]` = 12800
> entries. Because the vocabulary *size* matches, every index is accepted by every
> codec and **nothing raises** — you just get fluent, well-articulated speech saying
> completely different words.
>
> Comparing the content-tokenizer weights (`conv_downsample`, `local_encoder`,
> `local_quantizer`) across variants:
>
> | pair | probe tensors identical |
> |---|---|
> | `24kHz` vs `44.1kHz-v2` | **7 / 7, bit-identical** |
> | `24kHz` vs `44.1kHz` (legacy) | 0 / 7 |
> | `44.1kHz-v2` vs `44.1kHz` (legacy) | 0 / 7 |
>
> And empirically, encoding one real Hindi clip with both and comparing indices:
>
> ```
> exact index agreement, legacy vs 24kHz : 0.0000%   (chance = 0.0078%)
> log-mel corr vs original:
>    24kHz codec   <- its OWN tokens                     +0.938
>    44.1k-legacy  <- its OWN tokens                     +0.940
>    44.1k-legacy  <- 24kHz codec's tokens               +0.418     <- silent failure
> ```
>
> What that looks like end to end (`openai/whisper-small` transcribing):
>
> ```
> PROMPT   : नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।
> legacy   : अज़्ट उद आयार मुशिलो के लब शिबगो जो लिख चिए आया     <- wrong codec
> 24kHz    : नमस्ते आप कैसे है, आज मोशम बहुत अच्छा है            <- correct
> v2       : नमस्ते आप कैसे है, आज मोसम बहुत अच्छा है            <- correct
>
> PROMPT   : The mountains remember every footstep, even the ones you regret.
> legacy   : "That's it for today. Thank you for watching."
> 24kHz    : "The mountains remember every footstep, even the ones you regret."
> ```
>
> So `24kHz` and `44.1kHz-v2` both work and are token-compatible; the legacy
> `44.1kHz` is a different token space entirely.
>
> Two suggestions:
> 1. **State on the card which variant to use.** The card links `MioCodec-25Hz-24kHz`
>    but claims 44 kHz output and writes at 44100; `MioTTS-Inference` defaults to
>    `MioCodec-25Hz-44.1kHz-v2`. Naming `-v2` explicitly, alongside the note that its
>    tokenizer matches the 24 kHz model's, would remove the ambiguity.
> 2. **Note the loader split**: `MioCodecModel` for `24kHz` / `44.1kHz-v2` (integrated
>    iSTFT head), `MioCodec` for the legacy external-vocoder build. Loading the wrong
>    one gives `No vocoder weights found with prefix 'vocoder.'`, which is a correct
>    refusal but reads as a broken download.

---

## Report 3 *(optional, low priority)* — FSQ accepts out-of-range indices silently

**Post to:** <https://github.com/Aratako/MioCodec/issues> · **Suggested title:** `FiniteScalarQuantizer.decode silently accepts out-of-range indices`

> Small robustness note, not something that bit me in practice.
>
> `decode` accepts content token indices outside `[0, 12800)` without complaint —
> `12800`, `20000` and `-1` all produce finite audio, because
> `(indices // basis) % levels` wraps modularly. A range check in
> `FiniteScalarQuantizer.decode` would turn a silent corruption into an error.
>
> Related and probably more useful: `decode` also accepts *any* 1-D 128-vector as
> `global_embedding` with no validation that it came from an encoder. I passed
> Gaussian noise and got a clean waveform with no warning. That flexibility is
> genuinely useful — it is what makes the codec usable for a two-tower TTS split,
> which is what I am using it for — so I would not remove it, only mention it.
>
> ```python
> import torch
> from miocodec import MioCodecModel
> m = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-24kHz").eval()
> m.decode(global_embedding=torch.randn(128),
>          content_token_indices=torch.tensor([12800, 20000, -1] * 20))  # no error
> ```

---

## What I checked and am NOT reporting

Recorded so the same ground is not re-covered:

| checked | finding |
|---|---|
| `MioCodec` fails on the 24 kHz repo | **Not a bug.** Wrong loader class — `MioCodecModel` is documented in MioCodec's README. My error. |
| "the card names the wrong codec" | **Refuted.** `MioTTS-0.6B` states `MioCodec-25Hz-24kHz` twice, and Indic-Mio is a finetune of it. The real defect is the 44100 write. |
| `SPEECH_OFFSET = 151669` | **Correct**, and derivable from the tokenizer. |
| tokens dropped by the offset filter | Only a trailing `<|im_end|>`. No content lost. |
| `int32` / `float32` code tensors | Bit-identical output to `long`. `float64` raises cleanly. No silent degradation. |
| `config.vocab_size` (164480) vs `len(tokenizer)` (164469) | 11 untrained padding rows, samplable in principle, decode to nothing. Worth knowing, not worth reporting. |
