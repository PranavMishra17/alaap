"""
S13 — can the frozen tower be DIRECTED without changing who is speaking?

S12 measured which acoustic axes carry delivery rather than identity, on a
corpus of real actors. That says the axes exist. It does not say a TTS can be
made to move them on command, and a direction channel that cannot be driven is
not a channel.

`RESEARCH/10` states the claim this tests, and marks it UNVERIFIED:

    "Indic-Mio + MioCodec put identity in a decoder-side global_embedding and
     performance in text tags -- genuinely separate channels."
                                    -- HIGH on documentation, UNVERIFIED on reliability

The architecture makes it plausible: the tag goes in the TEXT, so it changes
the content tokens, while identity enters at decode as a 128-d vector that the
tag never touches. If that separation is real then a tag should move S12's
delivery axes and leave identity alone.

WHAT WOULD MAKE THIS MEANINGLESS, and is therefore measured first. The LM
samples at temperature 0.9, so two renders of the SAME text with the SAME
identity already differ. Every tag effect must be reported against that noise
floor or sampling variance will be read as steering:

    effect = |mean(tagged) - mean(neutral)| / SD(neutral repeats)

An effect of 1.0 is one noise-floor unit -- indistinguishable from re-rolling
the seed. This is the same discipline as S8's random-retrieval control and
S12's within/between ratio: a number is not readable until the thing it must
beat is sitting next to it.

THE TWO OUTCOMES THAT MATTER
    delivery axes move      speaking_rate, f0_cv, jitter, shimmer  (S12: ratio >= 1.5)
    identity does NOT       ECAPA against the neutral render, and f0_mean, which
                            S12 flagged as contested at 1.04 -- the axis a naive
                            "make it angry" would move and must not

    envs/qwen3/Scripts/python.exe experiments/S13-direction-channel/run_direction.py
"""
import argparse
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
from alaap.acoustics import Attributes, Binner, measure
from alaap.captions import caption_from_bins
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder
from alaap.metrics import CALIBRATION

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
S6_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "S6-indic-mint", "out")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--clips", type=int, default=250)
ap.add_argument("--per-speaker", type=int, default=2)
ap.add_argument("--codec", default="Aratako/MioCodec-25Hz-44.1kHz-v2")
ap.add_argument("--lm", default="SPRINGLab/Indic-Mio")
ap.add_argument("--voices", type=int, default=3,
                help="identities; a tag effect that only holds for one voice is not a channel")
ap.add_argument("--reps", type=int, default=5,
                help="neutral repeats per voice/line -- this IS the noise floor")
args = ap.parse_args()

CODEC_SR = 44100
SPEECH_OFFSET = 151669
CODEBOOK = 12800

# Indic-Mio's documented tags, placed at end of sentence (RESEARCH/04).
TAGS = ["<happy>", "<sad>", "<angry>", "<surprise>"]
LINES = [
    "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।",
    "हम कल सुबह जल्दी निकलेंगे।",
]
# S12's verdicts, so the table can be read against them rather than eyeballed.
DELIVERY = {"speaking_rate": 2.01, "f0_cv": 1.57, "jitter": 1.64, "shimmer": 3.12}
IDENTITY_RISK = {"f0_mean": 1.04, "hnr_db": 1.04, "spectral_tilt": 1.28}

import librosa
import soundfile as sf

# --------------------------------------------------------------- 1. voices
print(f"[1/4] minting {args.voices} identities")
S4 = f"experiments/S4-indic/out/measured_{args.corpus}_{args.clips}_{args.per_speaker}.npz"
EMB = os.path.join(S6_OUT, f"mio_emb_{args.corpus}_{args.clips}_{args.per_speaker}.npz")
for f in (S4, EMB):
    if not os.path.exists(f):
        sys.exit(f"missing {f}")
d4 = np.load(S4, allow_pickle=True)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d4["attrs"]))]
Z = np.load(EMB, allow_pickle=True)["Z"].astype(np.float64)
n = min(len(Z), len(attrs))
attrs, Z = attrs[:n], Z[:n]
binner = Binner.fit(attrs)
binner.edges.pop("vtl_cm", None)
bins = [binner.bin_one(a) for a in attrs]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
space = SpeakerSpace.fit(Z, n_components=min(64, n - 1))
mapper = RetrievalMapper(space, TextEncoder(), pca_dims=space.components.shape[0],
                         retrieval="hybrid").fit(caps, Z, anchor_bins=bins)
CAST = ["a low voice, warm-toned, at a steady pace",
        "a high voice, crisp-toned, speaking quickly",
        "a mid-range voice, slightly rough, measured"][:args.voices]
voices = [{"desc": d, "v": mapper.mint(d, novelty=0.0, seed=i, top_k=2).vector}
          for i, d in enumerate(CAST)]

# ------------------------------------------------------------- 2. renderer
print(f"[2/4] loading {args.lm} + codec")
from miocodec import MioCodecModel
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda" if torch.cuda.is_available() else "cpu"
codec = MioCodecModel.from_pretrained(args.codec).to(dev).eval()
tok = AutoTokenizer.from_pretrained(args.lm, trust_remote_code=True)
lm = AutoModelForCausalLM.from_pretrained(args.lm, torch_dtype=torch.bfloat16,
                                          device_map=dev, trust_remote_code=True).eval()
from alaap.encoder import IndependentSV
sv = IndependentSV()


def gen_codes(text, seed):
    torch.manual_seed(seed)
    p = tok.apply_chat_template([{"role": "user", "content": text}],
                                tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        o = lm.generate(**inp, max_new_tokens=1024, do_sample=True, temperature=0.9,
                        top_p=0.9, pad_token_id=tok.eos_token_id)
    new = o[0][inp["input_ids"].shape[1]:]
    return [t.item() - SPEECH_OFFSET for t in new
            if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + CODEBOOK]


def render(cc, vec):
    ct = torch.as_tensor(cc, dtype=torch.long, device=dev).reshape(-1)
    ge = torch.as_tensor(np.asarray(vec, np.float32), device=dev).reshape(-1)
    with torch.no_grad():
        return codec.decode(global_embedding=ge,
                            content_token_indices=ct).squeeze().float().cpu().numpy()


def analyse(w, text):
    a = measure(librosa.resample(w, orig_sr=CODEC_SR, target_sr=24000), text, 24000)
    e = sv.embed(librosa.resample(w, orig_sr=CODEC_SR, target_sr=16000), sr=16000)
    return a.to_dict(), e


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


# ------------------------------------------- 3. the noise floor, then the tags
print(f"[3/4] {args.reps} neutral repeats per voice/line -- the noise floor")
rows = []
for vi, vo in enumerate(voices):
    for li, line in enumerate(LINES):
        for rep in range(args.reps):
            cc = gen_codes(line, 1000 + 97 * rep + li)
            if not cc:
                continue
            w = render(cc, vo["v"])
            a, e = analyse(w, line)
            rows.append({"voice": vi, "line": li, "tag": "<neutral>", "rep": rep,
                         "attrs": a, "ecapa": e.tolist()})
    print(f"      voice {vi}: {sum(1 for r in rows if r['voice'] == vi)} neutral renders",
          flush=True)

print(f"[4/4] the same lines with {len(TAGS)} tags")
for vi, vo in enumerate(voices):
    for li, line in enumerate(LINES):
        for tag in TAGS:
            for rep in range(2):
                cc = gen_codes(f"{line} {tag}", 1000 + 97 * rep + li)
                if not cc:
                    continue
                w = render(cc, vo["v"])
                a, e = analyse(w, line)
                rows.append({"voice": vi, "line": li, "tag": tag, "rep": rep,
                             "attrs": a, "ecapa": e.tolist()})
                if vi == 0 and li == 0 and rep == 0:
                    sf.write(os.path.join(OUT, f"v0_{tag.strip('<>')}.wav"), w, CODEC_SR)
    print(f"      voice {vi} tagged", flush=True)

json.dump([{k: v for k, v in r.items() if k != "ecapa"} for r in rows],
          io.open(os.path.join(OUT, "rows.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# ----------------------------------------------------------------- report
AXES = list(DELIVERY) + list(IDENTITY_RISK)
neutral = [r for r in rows if r["tag"] == "<neutral>"]


def vals(rs, ax):
    return np.array([r["attrs"].get(ax, np.nan) for r in rs], float)


print()
print("=" * 78)
print("S13 — does a text tag move delivery without moving identity?")
print("=" * 78)
print(f"  {len(rows)} renders | {args.voices} voices x {len(LINES)} lines")
print()
print("  EFFECT SIZE in noise-floor units: |mean(tagged) - mean(neutral)| / SD(neutral)")
print("  1.0 = indistinguishable from re-rolling the sampling seed")
print()
hdr = f"  {'axis':<16} {'S12':>5} " + " ".join(f"{t.strip('<>')[:7]:>8}" for t in TAGS)
print(hdr)
print("  " + "-" * (len(hdr) - 2))
summary = {}
for ax in AXES:
    nv = vals(neutral, ax)
    nv = nv[np.isfinite(nv)]
    if len(nv) < 3 or nv.std() < 1e-9:
        continue
    s12 = DELIVERY.get(ax, IDENTITY_RISK.get(ax, float("nan")))
    cells, eff = [], []
    for t in TAGS:
        tv = vals([r for r in rows if r["tag"] == t], ax)
        tv = tv[np.isfinite(tv)]
        e = abs(tv.mean() - nv.mean()) / nv.std() if len(tv) else np.nan
        eff.append(e)
        cells.append(f"{e:>8.2f}")
    summary[ax] = float(np.nanmean(eff))
    print(f"  {ax:<16} {s12:>5.2f} " + " ".join(cells))

print()
print("  IDENTITY COST — ECAPA against the same voice rendered neutral")
c_same, c_diff, _ = CALIBRATION["ecapa"]
for t in ["<neutral>"] + TAGS:
    sims = []
    for vi in range(args.voices):
        base = [np.array(r["ecapa"]) for r in rows
                if r["voice"] == vi and r["tag"] == "<neutral>"]
        got = [np.array(r["ecapa"]) for r in rows
               if r["voice"] == vi and r["tag"] == t]
        if not base or not got:
            continue
        b = np.mean(base, 0)
        sims += [cos(b, g) for g in got if not (t == "<neutral>" and np.allclose(g, b))]
    if sims:
        m = float(np.mean(sims))
        print(f"    {t:<12} {m:.4f}   normalised {(m-c_diff)/(c_same-c_diff):+.3f}"
              f"   {'(self, the ceiling)' if t == '<neutral>' else ''}")

print()
d = np.nanmean([summary.get(a, np.nan) for a in DELIVERY])
i = np.nanmean([summary.get(a, np.nan) for a in IDENTITY_RISK])
print(f"  delivery axes, mean effect  {d:.2f}")
print(f"  identity-risk axes, mean    {i:.2f}")
print(f"  ratio                       {d/i if i else float('nan'):.2f}   "
      f"(>1 means the tag moves delivery more than it moves identity)")
print("=" * 78)
