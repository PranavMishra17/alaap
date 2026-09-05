"""
S13b — does `*word*` emphasis do anything, where the emotion tags did not?

The tags failed for a mechanical reason: none of them is a token. `*` IS a
token (id 9, single), so emphasis is structurally a live candidate in a way
`<happy>` never was. It is still an ordinary text token, not a special one --
so this measures whether the finetune learned to obey it.

THE DESIGN, which needs no forced alignment. A global acoustic difference
cannot distinguish "emphasis worked" from "the sample was re-rolled" -- that
is exactly the trap S13 fell into. But emphasis has a property re-rolling does
not: it is LOCAL. Emphasising an EARLY word should push acoustic energy
earlier in the utterance; emphasising a LATE word should push it later.

    energy centroid = sum(t * rms(t)) / sum(rms(t)),  t normalised to [0, 1]

    if emphasis works:   centroid(*early*) < centroid(plain) < centroid(*late*)

That ordering is a directional prediction stated before the run, and re-rolling
the sampling seed cannot produce it. The plain condition, rendered at several
seeds, gives the noise floor the difference must clear.

    envs/qwen3/Scripts/python.exe experiments/S13-direction-channel/run_emphasis.py
"""
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import librosa
import soundfile as sf
from miocodec import MioCodecModel
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SR, SPEECH_OFFSET, CODEBOOK = 44100, 151669, 12800
SEEDS = [1000, 1097, 1194, 1291]

PLAIN = "The mountains remember every footstep, even the ones you regret."
EARLY = "The *mountains* remember every footstep, even the ones you regret."
LATE = "The mountains remember every footstep, even the ones you *regret*."

dev = "cuda" if torch.cuda.is_available() else "cpu"
codec = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz-v2").to(dev).eval()
tok = AutoTokenizer.from_pretrained("SPRINGLab/Indic-Mio", trust_remote_code=True)
lm = AutoModelForCausalLM.from_pretrained("SPRINGLab/Indic-Mio", torch_dtype=torch.bfloat16,
                                          device_map=dev, trust_remote_code=True).eval()
vec = np.load("experiments/S6-indic-mint/out/mio_emb_indicvoices_r_hi_250_2.npz"
              )["Z"][0].astype(np.float32)


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


def render(cc):
    ct = torch.as_tensor(cc, dtype=torch.long, device=dev).reshape(-1)
    ge = torch.as_tensor(vec, device=dev).reshape(-1)
    with torch.no_grad():
        return codec.decode(global_embedding=ge,
                            content_token_indices=ct).squeeze().float().cpu().numpy()


def centroid(w):
    """Energy centroid in normalised time. 0.5 = energy evenly spread."""
    r = librosa.feature.rms(y=w, frame_length=2048, hop_length=512)[0]
    if r.sum() <= 0:
        return float("nan")
    t = np.linspace(0, 1, len(r))
    return float((t * r).sum() / r.sum())


print(f"{'condition':<12} " + " ".join(f"{'s' + str(i):>8}" for i in range(len(SEEDS)))
      + f" {'mean':>8} {'sd':>7}")
res = {}
for label, text in [("plain", PLAIN), ("*early*", EARLY), ("*late*", LATE)]:
    cs = []
    for k, sd in enumerate(SEEDS):
        cc = gen(text, sd)
        if not cc:
            cs.append(np.nan)
            continue
        w = render(cc)
        cs.append(centroid(w))
        if k == 0:
            sf.write(os.path.join(OUT, f"emph_{label.strip('*')}.wav"), w, SR)
    cs = np.array(cs, float)
    res[label] = cs
    print(f"  {label:<10} " + " ".join(f"{c:>8.4f}" for c in cs)
          + f" {np.nanmean(cs):>8.4f} {np.nanstd(cs):>7.4f}")

p, e, l = (np.nanmean(res[k]) for k in ("plain", "*early*", "*late*"))
noise = np.nanstd(res["plain"])
print()
print("=" * 72)
print("S13b — does *word* emphasis move energy toward the emphasised word?")
print("=" * 72)
print(f"  predicted ordering:  centroid(*early*) < plain < centroid(*late*)")
print(f"  observed:            {e:.4f}  {'<' if e < p else '>'}  {p:.4f}  "
      f"{'<' if p < l else '>'}  {l:.4f}")
print(f"  ordering holds:      {'YES' if e < p < l else 'NO'}")
print()
print(f"  plain seed-to-seed SD (the noise floor)  {noise:.4f}")
print(f"  |early - plain|  {abs(e-p):.4f}  = {abs(e-p)/noise if noise else float('nan'):.2f} noise units")
print(f"  |late  - plain|  {abs(l-p):.4f}  = {abs(l-p)/noise if noise else float('nan'):.2f} noise units")
print()
print("  A directional ordering cannot be produced by re-rolling the seed, so it")
print("  is the ordering -- not the magnitude -- that decides whether * is obeyed.")
print("=" * 72)
