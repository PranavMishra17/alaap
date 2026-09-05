"""Why did the emotion tags do nothing? Three checks, cheapest first.

A listener reported all four tagged renders as the same voice with no distinct
emotion -- which matches S13's measured 1.08 noise-floor units, but stronger:
not "weak", "absent". Before concluding the channel does not exist, check
whether the tag is even reaching the model as a CONTROL rather than as text.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LM = "SPRINGLab/Indic-Mio"
TAGS = ["<happy>", "<sad>", "<angry>", "<disgust>", "<fear>", "<surprise>",
        "<enunciated>", "<confused>", "<whisper>"]
tok = AutoTokenizer.from_pretrained(LM, trust_remote_code=True)

print("[1] is each documented tag a SINGLE token, or is it being read as text?")
print(f"  {'tag':<14} {'n tokens':>9}  pieces")
single = 0
for t in TAGS:
    ids = tok.encode(t, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]
    single += (len(ids) == 1)
    print(f"  {t:<14} {len(ids):>9}  {pieces}")
print(f"\n  {single}/{len(TAGS)} tags are single tokens")
print("  A tag split into '<', 'happy', '>' is not a control signal -- it is")
print("  three ordinary text tokens the model was never trained to obey.")

print("\n[2] are they in the tokenizer's added/special vocabulary?")
added = set(getattr(tok, "get_added_vocab", dict)().keys())
for t in TAGS:
    print(f"  {t:<14} {'IN added_vocab' if t in added else 'not in added_vocab'}")
spec = [s for s in getattr(tok, "all_special_tokens", []) if s]
print(f"  special tokens: {spec[:12]}")
sp_like = [k for k in added if k.startswith("<") and not k.startswith("<|")]
print(f"  added tokens that look like tags: {sorted(sp_like)[:20] or 'NONE'}")

print("\n[3] does the tag change what the LM generates, at a fixed seed?")
dev = "cuda" if torch.cuda.is_available() else "cpu"
lm = AutoModelForCausalLM.from_pretrained(LM, torch_dtype=torch.bfloat16,
                                          device_map=dev, trust_remote_code=True).eval()
SPEECH_OFFSET, CODEBOOK = 151669, 12800
LINE = "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।"


def gen(text, seed):
    torch.manual_seed(seed)
    p = tok.apply_chat_template([{"role": "user", "content": text}],
                                tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        o = lm.generate(**inp, max_new_tokens=1024, do_sample=True, temperature=0.9,
                        top_p=0.9, pad_token_id=tok.eos_token_id)
    n = o[0][inp["input_ids"].shape[1]:]
    return [t.item() - SPEECH_OFFSET for t in n
            if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + CODEBOOK]


base = gen(LINE, 1000)
print(f"  neutral: {len(base)} tokens")
for t in ["<happy>", "<sad>", "<angry>"]:
    g = gen(f"{LINE} {t}", 1000)
    m = min(len(base), len(g))
    agree = float(np.mean([a == b for a, b in zip(base[:m], g[:m])])) if m else float("nan")
    print(f"  {t:<11} {len(g):>4} tokens | prefix agreement with neutral {agree:.1%}")
print("\n  A tag that changes nothing about the token stream cannot change the audio.")
print("  A tag that changes it completely may just be re-rolling the sample.")
