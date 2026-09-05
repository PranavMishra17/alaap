# Upstream reports — ready to post

Four issues for **`SPRINGLab/Indic-Mio`**, one optional nit for **`Aratako/MioCodec`**.

**Report 4 is the important one** and was found after the others: the documented emotion
tags appear not to function at all.

Everything below was **adversarially re-verified** before being written. My first draft
had three claims; one was refuted and one was my own mistake, and both are excluded.
What remains is only what reproduces.

## How to use this file

Each report below has a **title line** to copy into the subject field, and a body wrapped
in a four-backtick fence. **Select everything between the ```` ```` ```` markers and paste
it** — it is raw markdown, so the inner ``` code blocks, tables and bold survive the paste
intact. Nothing needs reformatting.

**Where to post**

| | link |
|---|---|
| Indic-Mio discussions (5 existing threads, maintainer active) | <https://huggingface.co/SPRINGLab/Indic-Mio/discussions> |
| New Indic-Mio discussion | <https://huggingface.co/SPRINGLab/Indic-Mio/discussions/new> |
| MioCodec issues (MIT, 28★) | <https://github.com/Aratako/MioCodec/issues> |
| New MioCodec issue | <https://github.com/Aratako/MioCodec/issues/new> |
| The card being reported | <https://huggingface.co/SPRINGLab/Indic-Mio> |

**Versions to cite.** `miocodec` self-reports `0.1.0` but the repo has **no tags or
releases**, so cite the commit: **`77473544375d57e96cbdfd5d7d257e8f280fa8e3`**
(current `main`). Models: `SPRINGLab/Indic-Mio` @ `25feace0`,
`MioCodec-25Hz-24kHz` @ `3a737f0d`, `MioCodec-25Hz-44.1kHz-v2` @ `67faba34`.
torch 2.5.1+cu121, transformers 4.57.3, Windows.

**Suggested order.** **Report 4 first** — it says a documented feature does not work, and
it is the one a maintainer most needs to see. Then Report 2 (cost the most time, most
likely to catch someone else), then Report 1 (biggest, but a documentation fix). Report 3
is optional.

---

## Report 1 — the Transformers example cannot run as written

**Post to:** <https://huggingface.co/SPRINGLab/Indic-Mio/discussions/new>

**Title** — copy this line:

```text
Transformers example in the model card is not runnable (shapes, argument binding, missing speaker input, sample rate)
```

**Body** — copy everything between the four-backtick fences:

````markdown
Thanks for Indic-Mio — the 22-language coverage is genuinely useful and the model
works well once driven correctly.

The **Approach 2: Directly with Transformers** snippet on the card cannot run as
written. Four separate problems, smallest first. All reproduced against `miocodec`
commit `7747354437`, transformers 4.51.3, torch 2.5.1.

### 1. The codes are passed positionally into the wrong parameter

The card has `wav = codec.decode(codes_tensor)`. The signature is:

```python
MioCodec.decode(self, global_embedding=None, content_token_indices=None,
                content_embedding=None, target_audio_length=None, features=None)
```

so `codes_tensor` binds to `global_embedding` and the content is never supplied:

```
ValueError: Either content_token_indices or content_embedding must be provided.
```

### 2. There is no speaker input anywhere in the example

`global_embedding` is required, and the snippet has no reference audio, no preset and
no `synthesize_from_tokens` call. This one cannot be fixed by reordering arguments — a
required input is missing entirely.

### 3. The tensor shape is wrong

The card builds `[1, 1, T]` via `torch.tensor([audio_codes]).unsqueeze(0)`. `decode`
unconditionally does `content_embedding.unsqueeze(0)` internally, so anything
pre-batched becomes 4-D and fails in `module/transformer.py:594` at
`bsz, seqlen, _dim = x.shape`:

```
ValueError: too many values to unpack (expected 3)
```

I tested all 9 combinations of `content ∈ {[T], [1,T], [1,1,T]}` ×
`global ∈ {[128], [1,128], [1,1,128]}`. **Exactly one works: both 1-D.**
(`decode_batch` is the batched entry point and works as documented with
`[B, dim]` / `[B, max_seq_len]` — so this is a card issue, not a library one.)

### 4. The sample rate is wrong by 1.84×

The example loads `Aratako/MioCodec-25Hz-24kHz`, whose `config.sample_rate == 24000`,
then writes `sf.write("output.wav", ..., 44100)`. The result plays 1.84× too fast.

### Minimal reproduction

Needs only the codec, no LM download:

```python
import torch, inspect
from miocodec import MioCodec, MioCodecModel   # commit 7747354437

# the card's loader class refuses the card's codec -- these variants have an
# integrated iSTFT head and no external vocoder, so MioCodecModel is the loader
try:
    MioCodec.from_pretrained("Aratako/MioCodec-25Hz-24kHz")
except ValueError as e:
    print("card's class ->", e)   # No vocoder weights found with prefix 'vocoder.'

m = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-24kHz").eval()
print(m.config.sample_rate)       # 24000  -- but the card writes at 44100

codes = torch.randint(0, 12800, (50,), dtype=torch.long)
g = torch.randn(128)

try:                                            # problem 3
    m.decode(global_embedding=g, content_token_indices=codes[None, None, :])
except ValueError as e:
    print("card's shape ->", e)   # too many values to unpack (expected 3)

try:                                            # problems 1 + 2
    m.decode(torch.tensor([codes.tolist()]).unsqueeze(0))
except ValueError as e:
    print("card's call  ->", e)   # Either content_token_indices or ... must be provided

m.decode(global_embedding=g, content_token_indices=codes)   # the only working form
```

### Suggested replacement for the card's Approach 2

```python
import torch, soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from miocodec import MioCodecModel, load_audio

tokenizer = AutoTokenizer.from_pretrained("SPRINGLab/Indic-Mio", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "SPRINGLab/Indic-Mio", dtype=torch.bfloat16, device_map="cuda")

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "नमस्ते, आप कैसे हैं?"}],
    tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024, do_sample=True,
                        temperature=0.9, top_p=0.9)

generated = output[0][inputs["input_ids"].shape[1]:]
SPEECH_OFFSET = tokenizer.convert_tokens_to_ids("<|s_0|>")   # 151669, derived
audio_codes = [t.item() - SPEECH_OFFSET for t in generated
               if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + 12800]

# MioCodecModel, not MioCodec: integrated iSTFT head, no external vocoder
codec = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz-v2").eval().cuda()
reference = load_audio("reference_speaker.wav",
                       sample_rate=codec.config.sample_rate).cuda()
wav = codec.synthesize_from_tokens(audio_codes, reference)
sf.write("output.wav", wav.cpu().numpy(), codec.config.sample_rate)  # never hardcoded
```

Deriving `SPEECH_OFFSET` from the tokenizer rather than hardcoding `151669` also
makes the example robust to a future vocab change. I verified the card's value is
correct today: `<|s_0|>` → 151669, `<|s_12799|>` → 164468, contiguous, exactly
12800 tokens.

Minor: the snippet is fenced as ```` ```bash ```` but is Python.
````

---

## Report 2 — please document which codec pairs with which loader

**Post to:** <https://huggingface.co/SPRINGLab/Indic-Mio/discussions/new>

**Title** — copy this line:

```text
Which MioCodec variant should Indic-Mio use? The three are not interchangeable
```

**Body** — copy everything between the four-backtick fences:

````markdown
This one cost me the most time and I think it will catch others, so it may be worth a
line on the card.

There are three MioCodec variants, and **their content-token spaces are not
interchangeable**, even though all three are FSQ with `levels [8,8,8,5,5]` = 12800
entries. Because the vocabulary *size* matches, every index is accepted by every codec
and **nothing raises** — you just get fluent, well-articulated speech saying completely
different words.

Comparing the content-tokenizer weights (`conv_downsample`, `local_encoder`,
`local_quantizer`) across variants:

| pair | probe tensors identical |
|---|---|
| `24kHz` vs `44.1kHz-v2` | **7 / 7, bit-identical** |
| `24kHz` vs `44.1kHz` (legacy) | 0 / 7 |
| `44.1kHz-v2` vs `44.1kHz` (legacy) | 0 / 7 |

And empirically, encoding one real Hindi clip with both and comparing indices:

```
exact index agreement, legacy vs 24kHz : 0.0000%   (chance = 0.0078%)

log-mel corr vs original:
   24kHz codec   <- its OWN tokens                     +0.938
   44.1k-legacy  <- its OWN tokens                     +0.940
   44.1k-legacy  <- 24kHz codec's tokens               +0.418     <- silent failure
```

What that looks like end to end (`openai/whisper-small` transcribing):

```
PROMPT   : नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।
legacy   : अज़्ट उद आयार मुशिलो के लब शिबगो जो लिख चिए आया     <- wrong codec
24kHz    : नमस्ते आप कैसे है, आज मोशम बहुत अच्छा है            <- correct
v2       : नमस्ते आप कैसे है, आज मोसम बहुत अच्छा है            <- correct

PROMPT   : The mountains remember every footstep, even the ones you regret.
legacy   : "That's it for today. Thank you for watching."
24kHz    : "The mountains remember every footstep, even the ones you regret."
```

So `24kHz` and `44.1kHz-v2` both work and are token-compatible; the legacy `44.1kHz`
is a different token space entirely.

Two suggestions:

1. **State on the card which variant to use.** The card links `MioCodec-25Hz-24kHz` but
   claims 44 kHz output and writes at 44100; `MioTTS-Inference` defaults to
   `MioCodec-25Hz-44.1kHz-v2`. Naming `-v2` explicitly, alongside the note that its
   tokenizer matches the 24 kHz model's, would remove the ambiguity.
2. **Note the loader split**: `MioCodecModel` for `24kHz` / `44.1kHz-v2` (integrated
   iSTFT head), `MioCodec` for the legacy external-vocoder build. Loading the wrong one
   gives `No vocoder weights found with prefix 'vocoder.'`, which is a correct refusal
   but reads as a broken download.
````

---

## Report 3 *(optional, low priority)* — FSQ accepts out-of-range indices silently

**Post to:** <https://github.com/Aratako/MioCodec/issues/new>

**Title** — copy this line:

```text
FiniteScalarQuantizer.decode silently accepts out-of-range indices
```

**Body** — copy everything between the four-backtick fences:

````markdown
Small robustness note, not something that bit me in practice.

`decode` accepts content token indices outside `[0, 12800)` without complaint —
`12800`, `20000` and `-1` all produce finite audio, because
`(indices // basis) % levels` wraps modularly. A range check in
`FiniteScalarQuantizer.decode` would turn a silent corruption into an error.

Related and probably more useful: `decode` also accepts *any* 1-D 128-vector as
`global_embedding` with no validation that it came from an encoder. I passed Gaussian
noise and got a clean waveform with no warning. That flexibility is genuinely useful —
it is what makes the codec usable for a two-tower TTS split, which is what I am using
it for — so I would not remove it, only mention it.

```python
import torch
from miocodec import MioCodecModel
m = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-24kHz").eval()
m.decode(global_embedding=torch.randn(128),
         content_token_indices=torch.tensor([12800, 20000, -1] * 20))  # no error
```
````

---

## Report 4 — the emotion/style tags and word stress appear to have no effect

**Post to:** <https://huggingface.co/SPRINGLab/Indic-Mio/discussions/new>

**Title** — copy this line:

```text
Emotion tags and word stress appear to have no effect (tags are not tokens in the tokenizer)
```

**Body** — copy everything between the four-backtick fences:

````markdown
Thank you for Indic-Mio — the two-tower behaviour is excellent and I have been using it
successfully for Hindi, Bengali and Tamil.

I cannot get the documented emotion and style tags to do anything, and I think the
tokenizer explains why. Reporting it with what I measured, in case I am driving it wrong.

### What the card says

> Tags for Indian languages: `<happy>`, `<sad>`, `<angry>`, `<disgust>`, `<fear>`, `<surprise>`
> Tags for English: `<happy>`, `<sad>`, `<enunciated>`, `<confused>`, `<angry>`, `<whisper>`

placed at the end of the sentence. That is exactly how I used them.

### The tags are not tokens

None of the nine documented tags is a single token, and none is in the added vocabulary —
in **Indic-Mio or in the base `Aratako/MioTTS-0.6B`** (both vocab size 164469):

```
tag             n tokens  pieces
<happy>                3  ['<h', 'appy', '>']
<sad>                  3  ['<s', 'ad', '>']
<angry>                4  ['<', 'ang', 'ry', '>']
<disgust>              5  ['<', 'dis', 'g', 'ust', '>']
<fear>                 3  ['<f', 'ear', '>']
<surprise>             4  ['<', 'sur', 'prise', '>']
<enunciated>           5  ['<', 'en', 'unc', 'iated', '>']
<confused>             4  ['<', 'conf', 'used', '>']
<whisper>              4  ['<', 'wh', 'isper', '>']

single-token tags:  0/9
in added_vocab:     0/9
added tokens that look like tags: ['</think>', '</tool_call>', '</tool_response>',
                                   '<think>', '<tool_call>', '<tool_response>']
```

The added vocabulary is Qwen's chat/tool tokens; no emotion tag was ever added.

### And behaviourally they do nothing

`<whisper>` is the clearest probe, because whispering is unmistakable acoustically —
voicing should collapse. English line, same speaker embedding, 3 seeds averaged:

```
variant         tokens   hnr_db   f0_mean   voiced_frac   speaking_rate
neutral             84     4.82     215.5         0.583           24.32
<whisper>          119     3.97     216.7         0.555           17.97
<angry>            124     5.08     215.0         0.573           16.68
<enunciated>       158     5.83     215.3         0.576           13.08
```

`voiced_frac` barely moves — it is not whispering. `f0_mean` is unchanged to within
1.5 Hz across every tag.

On Hindi, across 78 renders (3 voices x 2 lines, with 30 untagged renders establishing
the sampling noise floor at temperature 0.9), the mean effect of a tag on the acoustic
axes that carry delivery was **1.08 noise-floor units** — i.e. about the same as
re-rolling the sampling seed. A fluent Hindi speaker listened to `<happy>`, `<sad>`,
`<angry>` and `<surprise>` renders of one sentence in one voice and reported them as the
same voice with no distinguishable emotion.

At a fixed seed, adding a tag changes the generated token stream almost completely
(prefix agreement with the untagged generation: 0–2.9%), which is consistent with the
tag text perturbing sampling rather than conditioning anything.

### Two things I checked so they are not confounders

- **The tag is not being spoken aloud.** `whisper-small` transcribes all four tagged
  English renders as exactly the reference sentence, with no "whisper"/"enunciated"
  appearing. Intelligibility is unaffected.
- **Formatting does not change the tokenization**: `<happy>`, ` <happy>`, `<happy>.`,
  `[happy]` and `(happy)` all tokenize to 3 pieces.

### Word stress (`*word*`) also appears inert

The card documents this alongside the tags:

> A word can be stressed by using asterisks(*) around it.

`*` **is** a single token here (id 9), unlike the emotion tags, so this one is not a
tokenizer problem. But it produces no positional effect. Emphasising an early word versus
a late word in the same English sentence, 4 seeds each, measuring the energy centroid in
normalised time:

```
condition                              mean centroid
plain                                        0.4489   (seed-to-seed SD 0.0208)
The *mountains* remember every ...           0.4312   -0.85 noise units
... even the ones you *regret*.              0.4345   -0.69 noise units
```

If the marker were obeyed, emphasising the *last* word would push energy later than
emphasising the *second* word. Both move the same direction, by less than one
seed-to-seed SD. (Caveat: the energy centroid would miss emphasis realised purely as
pitch accent.)

### What I could not rule out

- That the tags require a prompt format other than plain inline text — something
  `MioTTS-Inference` does that I have not replicated. If so, a card example showing the
  correct call would fix this entirely.
- That they work in a language or on a checkpoint I did not test. I used Hindi and
  English on `SPRINGLab/Indic-Mio` @ `25feace0` with `MioCodec-25Hz-44.1kHz-v2`.

If the tags are meant to work as plain text, adding them as special tokens and
fine-tuning briefly on tagged data would likely be needed. If they were never trained in
the Indic finetune, saying so on the card would save people the search — the Rasa corpus
is expressive, so it is a reasonable thing for a reader to assume works.

Happy to run any check that would help.
````

---

## What I checked and am NOT reporting

Not for posting — recorded so the same ground is not re-covered.

| checked | finding |
|---|---|
| `MioCodec` fails on the 24 kHz repo | **Not a bug.** Wrong loader class — `MioCodecModel` is documented in MioCodec's README. My error. |
| "the card names the wrong codec" | **Refuted.** `MioTTS-0.6B` states `MioCodec-25Hz-24kHz` twice, and Indic-Mio is a finetune of it. The real defect is the 44100 write. |
| `SPEECH_OFFSET = 151669` | **Correct**, and derivable from the tokenizer. |
| tokens dropped by the offset filter | Only a trailing `<\|im_end\|>`. No content lost. |
| `int32` / `float32` code tensors | Bit-identical output to `long`. `float64` raises cleanly. No silent degradation. |
| "the model speaks the tag text aloud" | **Refuted by my own check.** Token counts grow with tag length (84 → 158), which looked like it. `whisper-small` transcribes every tagged render as exactly the reference sentence. |
| `config.vocab_size` (164480) vs `len(tokenizer)` (164469) | 11 untrained padding rows, samplable in principle, decode to nothing. Worth knowing, not worth reporting. |
