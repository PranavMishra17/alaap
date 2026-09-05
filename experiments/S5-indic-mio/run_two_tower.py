"""
S5 — does the two-tower split actually work in Indic?

This is RESEARCH/04's E1, the experiment that document called "the highest-value
experiment in this file", and it is now runnable because the licence question
was settled as research-only and the HF gate is open.

THE ARCHITECTURE, and why Indic-Mio is different from Qwen3-TTS.

Qwen3-TTS exposes a speaker vector through an inference-mode flag
(`x_vector_only_mode=True`) and cannot speak any Indian language. MioCodec is
built around the split as a documented API -- its own card says it decomposes
speech into:

    content tokens     linguistic/phonetic content, "what" is said, 25 Hz
    global embedding   a continuous 128-d vector for "how" -- speaker identity,
                       recording environment, microphone

so the two towers are:

    text --> Indic-Mio LM --> content tokens  ]
                                              ]--> MioCodec.decode() --> audio
    identity ------------> global embedding   ]

`decode(global_embedding=..., content_token_indices=...)` takes an ARBITRARY
128-d tensor. That is the whole Tier-1 claim, and unlike Qwen3's flag it is a
public method rather than an inference mode.

WHAT THIS RUN ANSWERS
  1. does the LM produce decodable content tokens for Indic text at all?
  2. does swapping the global embedding change WHO speaks while the words stay?
  3. does a voice survive being carried to NEW text -- the thing that makes a
     stored identity worth anything?

A NOTE ON THE MEASUREMENT. Raw cosine between MioCodec global embeddings
saturates: two DIFFERENT speakers sit at 0.988 and the same speaker at 0.997.
That is a property of the embedding's common-mode component, not evidence that
identity is absent -- the same reason this project puts every Qwen3 vector
through SpeakerSpace before measuring anything (invariant I1). So similarity
here is reported RAW and clearly labelled; S5b calibrates it properly across
many speakers. Do not read a raw cosine as a verification score.

    envs/qwen3/Scripts/python.exe experiments/S5-indic-mio/run_two_tower.py
"""
import argparse
import io
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.data import stream_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--lm", default="SPRINGLab/Indic-Mio")
# CODEC CHOICE IS LOAD-BEARING -- see S5's RESULTS.md.
# `MioCodec-25Hz-44.1kHz` (legacy) has a content-token space UNRELATED to the
# one Indic-Mio was trained on: 0.0000% index agreement against the 24 kHz
# model on identical audio, where chance is 0.0078%. Both are FSQ with 12800
# entries, so every index is accepted and NOTHING RAISES -- the output is
# fluent-sounding speech saying different words.
# `-v2` shares the 24 kHz model's tokenizer bit-for-bit and is natively
# 44.1 kHz. Use MioCodecModel (integrated iSTFT head, no external vocoder);
# MioCodec is the loader for the legacy external-vocoder variant only.
ap.add_argument("--codec", default="Aratako/MioCodec-25Hz-44.1kHz-v2")
ap.add_argument("--max-new-tokens", type=int, default=1024)
ap.add_argument("--temperature", type=float, default=0.9)
ap.add_argument("--top-p", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

CODEC_SR = 44100
SPEECH_OFFSET = 151669      # from the Indic-Mio model card
CODEBOOK = 12800

LINES = [
    ("hi", "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।"),
    ("hi", "मुझे यह फिल्म बहुत पसंद आई! <happy>"),
    ("ta", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"),
    ("en", "The mountains remember every footstep, even the ones you regret."),
]

import librosa
import soundfile as sf


def to_codec_sr(wav, src=24000):
    return librosa.resample(np.asarray(wav, np.float32), orig_sr=src,
                            target_sr=CODEC_SR)


print(f"[1/5] loading codec {args.codec}")
from miocodec import MioCodecModel
codec = MioCodecModel.from_pretrained(args.codec)
dev = "cuda" if torch.cuda.is_available() else "cpu"
codec = codec.to(dev).eval()

print(f"[2/5] loading LM {args.lm}")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.lm, trust_remote_code=True)
lm = AutoModelForCausalLM.from_pretrained(
    args.lm, torch_dtype=torch.bfloat16, device_map=dev, trust_remote_code=True).eval()
print(f"      {sum(p.numel() for p in lm.parameters())/1e9:.2f}B params on {dev}")


def encode(wav24):
    t = torch.as_tensor(to_codec_sr(wav24)).reshape(1, -1).to(dev)
    with torch.no_grad():
        return codec.encode(t, return_content=True, return_global=True)


def gen_content(text, seed):
    """Text -> content tokens. The LM never sees a speaker."""
    torch.manual_seed(seed)
    msgs = [{"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        out = lm.generate(**inp, max_new_tokens=args.max_new_tokens,
                          do_sample=True, temperature=args.temperature,
                          top_p=args.top_p, pad_token_id=tok.eos_token_id)
    new = out[0][inp["input_ids"].shape[1]:]
    codes = [t.item() - SPEECH_OFFSET for t in new
             if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + CODEBOOK]
    return codes, len(new)


def decode(codes, gemb):
    # BOTH must be 1-D: content [T], global [128]. Probed empirically -- every
    # batched shape raises "too many values to unpack". The model card shows
    # [1, 1, T] and also passes the codes positionally into global_embedding,
    # so it is wrong twice for this version of miocodec.
    ct = torch.as_tensor(codes, dtype=torch.long, device=dev).reshape(-1)
    ge = torch.as_tensor(gemb, device=dev).reshape(-1)
    with torch.no_grad():
        w = codec.decode(global_embedding=ge, content_token_indices=ct)
    return w.squeeze().float().cpu().numpy()


def cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


print("[3/5] two real Hindi speakers as identity donors")
clips = stream_clips("indicvoices_r_hi", n=8, per_speaker=1, dev_only=True,
                     min_dur=4.0, max_dur=10.0, progress_every=0)
donors = []
for c in clips[:2]:
    f = encode(c.wav)
    donors.append({"spk": c.speaker_id, "g": f.global_embedding,
                   "gv": f.global_embedding.reshape(-1).float().cpu().numpy(),
                   "wav": c.wav, "gender": c.gender})
    sf.write(os.path.join(OUT, f"donor_{c.speaker_id[:10]}.wav"),
             to_codec_sr(c.wav), CODEC_SR)
for d in donors:
    print(f"      {d['spk'][:14]}  {d['gender']}  g-dim {d['gv'].shape[0]}")

print("[4/5] generating; the SAME content tokens decoded with EACH identity")
rows, t0 = [], time.time()
for li, (lang, text) in enumerate(LINES):
    codes, n_new = gen_content(text, args.seed + li)
    if not codes:
        print(f"      [{li}] {lang}: NO speech tokens in {n_new} generated -- skipped")
        rows.append({"lang": lang, "text": text, "ok": False})
        continue
    row = {"lang": lang, "text": text, "ok": True, "n_codes": len(codes),
           "sec": len(codes) / 25.0, "renders": []}
    for d in donors:
        wav = decode(codes, d["g"])
        back = encode(librosa.resample(wav, orig_sr=CODEC_SR, target_sr=24000))
        bv = back.global_embedding.reshape(-1).float().cpu().numpy()
        row["renders"].append({
            "spk": d["spk"], "carried": cos(bv, d["gv"]),
            "other": cos(bv, [x["gv"] for x in donors if x is not d][0]),
        })
        fn = f"line{li}_{lang}_{d['spk'][:10]}.wav"
        sf.write(os.path.join(OUT, fn), wav, CODEC_SR)
    rows.append(row)
    r = row["renders"]
    print(f"      [{li}] {lang} {len(codes)} codes = {row['sec']:.1f}s | "
          f"identity carried {r[0]['carried']:.4f}/{r[1]['carried']:.4f} "
          f"vs other {r[0]['other']:.4f}/{r[1]['other']:.4f} "
          f"| {time.time()-t0:.0f}s", flush=True)

print("[5/5] summary")
ok = [r for r in rows if r.get("ok")]
print()
print("=" * 78)
print("S5 — the two-tower split on Indic-Mio")
print("=" * 78)
print(f"  lines generated      {len(ok)}/{len(LINES)}")
if ok:
    carried = np.mean([x["carried"] for r in ok for x in r["renders"]])
    other = np.mean([x["other"] for r in ok for x in r["renders"]])
    print(f"  identity carried     {carried:.4f}   (RAW cosine -- saturates, see header)")
    print(f"  vs the OTHER donor   {other:.4f}")
    print(f"  margin               {carried - other:+.4f}  "
          f"{'the decoded voice tracks the embedding it was given'
             if carried > other else 'IT DOES NOT -- identity is not controlled'}")
print(f"  audio                {OUT}")
print()
print("  LISTEN to line0_hi_*.wav for the two donors: same words, and the")
print("  question is whether they sound like two different people.")
print("=" * 78)
