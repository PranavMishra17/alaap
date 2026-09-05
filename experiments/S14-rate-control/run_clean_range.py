"""
S14b — where does re-timing start to SOUND processed?

`S14` measured the range over which identity and words survive re-timing:
0.67x to 1.43x. A listener then heard the slow extreme as *"60% slowed down,
40% slow speech effect"* — so that bound is not the range over which the output
sounds clean, and `RATE_BOUND["listener_clean_range"]` currently records a
guess of (0.8, 1.25) with nothing behind it.

This measures the real one.

THE DESIGN IS SINGLE-STIMULUS, NOT A/B. An A/B against the original is
worthless here: the two clips have different durations, so the listener can
always tell which was processed and would be answering "which is longer" rather
than "does this sound processed". Instead each clip is heard alone and rated on
one question:

    "Does this sound like a natural recording, or like audio that has been
     sped up or slowed down?"

THE CONTROLS ARE UNPROCESSED CLIPS at rate 1.00, mixed in blind. If those come
back "processed", the listener is hearing the synthesis itself rather than the
re-timing, and no rating in the set is readable — the same structure S11 used,
where the controls decide whether the answers mean anything.

Rates span the measured bound and bracket the guessed clean range:

    0.67  the slow edge of the bound   -- already reported as 60% artefact
    0.80  the guessed clean edge
    1.00  CONTROL, untouched
    1.25  the guessed clean edge
    1.43  the fast edge of the bound

    envs/qwen3/Scripts/python.exe experiments/S14-rate-control/run_clean_range.py
"""
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.acoustics import Attributes, Binner
from alaap.captions import caption_from_bins
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder
from alaap.timing import retime

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_clean")
os.makedirs(OUT, exist_ok=True)
S6_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "S6-indic-mint", "out")
SR, SPEECH_OFFSET, CODEBOOK = 44100, 151669, 12800
RATES = [0.67, 0.80, 1.00, 1.25, 1.43]
LINES = ["नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।",
         "हम कल सुबह जल्दी निकलेंगे।"]
SEED = 11

import librosa
import soundfile as sf

print("[1/3] rebuilding a voice")
d4 = np.load("experiments/S4-indic/out/measured_indicvoices_r_hi_250_2.npz", allow_pickle=True)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d4["attrs"]))]
Z = np.load(os.path.join(S6_OUT, "mio_emb_indicvoices_r_hi_250_2.npz"),
            allow_pickle=True)["Z"].astype(np.float64)
n = min(len(Z), len(attrs))
attrs, Z = attrs[:n], Z[:n]
binner = Binner.fit(attrs)
binner.edges.pop("vtl_cm", None)
bins = [binner.bin_one(a) for a in attrs]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
space = SpeakerSpace.fit(Z, n_components=min(64, n - 1))
mapper = RetrievalMapper(space, TextEncoder(), pca_dims=space.components.shape[0],
                         retrieval="hybrid").fit(caps, Z, anchor_bins=bins)
vec = mapper.mint("a mid-range voice, warm-toned, at a steady pace",
                  novelty=0.0, seed=0, top_k=2).vector

print("[2/3] rendering")
from miocodec import MioCodecModel
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda" if torch.cuda.is_available() else "cpu"
codec = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz-v2").to(dev).eval()
tok = AutoTokenizer.from_pretrained("SPRINGLab/Indic-Mio", trust_remote_code=True)
lm = AutoModelForCausalLM.from_pretrained("SPRINGLab/Indic-Mio", torch_dtype=torch.bfloat16,
                                          device_map=dev, trust_remote_code=True).eval()


def render(text, seed):
    torch.manual_seed(seed)
    p = tok.apply_chat_template([{"role": "user", "content": text}],
                                tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        o = lm.generate(**inp, max_new_tokens=1024, do_sample=True, temperature=0.9,
                        top_p=0.9, pad_token_id=tok.eos_token_id)
    nn = o[0][inp["input_ids"].shape[1]:]
    cc = [t.item() - SPEECH_OFFSET for t in nn
          if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + CODEBOOK]
    ct = torch.as_tensor(cc, dtype=torch.long, device=dev).reshape(-1)
    ge = torch.as_tensor(np.asarray(vec, np.float32), device=dev).reshape(-1)
    with torch.no_grad():
        return codec.decode(global_embedding=ge,
                            content_token_indices=ct).squeeze().float().cpu().numpy()


bases = [render(t, 3000 + i) for i, t in enumerate(LINES)]

print("[3/3] writing a blind, shuffled set")
items = [(li, r) for li in range(len(LINES)) for r in RATES]
rng = np.random.default_rng(SEED)
order = rng.permutation(len(items))
key = []
for slot, ix in enumerate(order, 1):
    li, r = items[int(ix)]
    w = retime(bases[li], r)
    sf.write(os.path.join(OUT, f"clip{slot:02d}.wav"), w, SR)
    key.append({"clip": slot, "rate": r, "line": li,
                "is_control": r == 1.00,
                "duration_s": round(len(w) / SR, 2)})
    print(f"      clip{slot:02d}  rate {r:.2f}{'  <- CONTROL' if r == 1.0 else ''}")

json.dump(key, io.open(os.path.join(OUT, "ANSWER-KEY.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
sheet = [
    "# S14b — how far can speaking rate be pushed before it sounds processed?", "",
    "Listen to each clip **on its own**. One question:", "",
    "> **Does this sound like a natural recording, or like audio that has been",
    "> sped up or slowed down?**", "",
    "Not *is it fast or slow* — some clips genuinely are. The question is whether",
    "it sounds like a **processing artefact**.",
    "",
    "Answer `natural` or `processed` for each. Some clips are untouched and are",
    "there to check the test itself — if those come back `processed`, you are",
    "hearing the synthesis rather than the re-timing and none of the ratings",
    "can be read.", "",
    "| clip | natural or processed? |", "|---|---|"]
sheet += [f"| {i:02d} | |" for i in range(1, len(key) + 1)]
sheet += ["", "Answer key in `out_clean/ANSWER-KEY.json` — **do not open it first.**"]
io.open(os.path.join(OUT, "SCORESHEET.md"), "w", encoding="utf-8").write("\n".join(sheet))
print(f"\n  {len(key)} clips -> {OUT}")
