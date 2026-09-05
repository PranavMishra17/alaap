"""
S6b — does MINTING a voice cost intelligibility?

S5b answered this for DONOR embeddings: real vectors, taken from real clips,
carried onto new text. CER 0.000 on Tamil and English, 0.114 on Hindi, all of
the Hindi residual `whisper-small`'s Devanagari orthography.

A minted vector is a different object. It is produced by a mapper from a text
description, and it sits further from the distribution MioCodec's decoder was
trained on -- it is a point the corpus never contained, which is the entire
point of minting and also the reason it might not decode cleanly. Nothing in
S5b or S6 rules out the possibility that a minted vector renders a distinctive
voice saying slightly wrong words.

THE DESIGN. Content tokens are generated ONCE per line and decoded through
six different global embeddings: the four minted voices and two REAL corpus
speakers. The words are identical by construction, and the ASR is the same
instrument on the same lines, so:

    CER(minted) - CER(donor)

isolates the cost of the vector being minted. Everything else -- the LM, the
sampling seed, the codec, the ASR's Devanagari spelling habits -- is common to
both sides and cancels in the difference. The donor arm is the control; its
absolute CER should land in S5b's Hindi band, and if it does not, the
instrument moved and no number here should be read.

WHAT WOULD FALSIFY THE PIPELINE: a positive difference of the same order as
the CER itself. What would confirm it: a difference near zero, meaning the
decoder treats a minted point in embedding space no differently from a real
one.

    envs/qwen3/Scripts/python.exe experiments/S6-indic-mint/run_indic_mint.py --skip-render
    envs/qwen3/Scripts/python.exe experiments/S6-indic-mint/run_mint_intelligibility.py
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
from alaap.metrics import cer, normalise_transcript

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

ap = argparse.ArgumentParser()
ap.add_argument("--asr", default="openai/whisper-small",
                help="ungated fallback; IndicConformer is the better instrument "
                     "and is gated. Bias cancels in the minted-vs-donor difference.")
ap.add_argument("--codec", default="Aratako/MioCodec-25Hz-44.1kHz-v2")
ap.add_argument("--lm", default="SPRINGLab/Indic-Mio")
ap.add_argument("--device", default="cpu", help="device for the ASR")
args = ap.parse_args()

CODEC_SR = 44100
SPEECH_OFFSET = 151669
CODEBOOK = 12800
MINTED = os.path.join(OUT, "minted.npz")
NAN = float("nan")

if not os.path.exists(MINTED):
    sys.exit(f"missing {MINTED} -- run run_indic_mint.py (--skip-render is enough) first")
d = np.load(MINTED, allow_pickle=True)
names = [str(x) for x in d["names"]]
V, donor_V = d["V"].astype(np.float32), d["donor_V"].astype(np.float32)
donor_ids = [str(x) for x in d["donor_ids"]]
SCRIPT = [str(x) for x in d["script"]]

# arm label -> (kind, vector). Donors carry the speaker id so the control is
# traceable back to a real clip.
ARMS = ([(n, "minted", V[i]) for i, n in enumerate(names)] +
        [(f"donor{i}:{s[:8]}", "donor", donor_V[i]) for i, s in enumerate(donor_ids)])
# Renders are keyed by label. A duplicate label would silently overwrite an arm
# and shrink the control -- which it did once, when both donors were the same
# speaker under --per-speaker 2.
if len({l for l, _, _ in ARMS}) != len(ARMS):
    sys.exit(f"ABORT: duplicate arm labels {[l for l, _, _ in ARMS]}")
if len(set(donor_ids)) != len(donor_ids):
    sys.exit(f"ABORT: donors are the same speaker {donor_ids} -- the control "
             "would compare one person with herself")
print(f"[1/4] {len(ARMS)} arms ({len(names)} minted, {len(donor_ids)} real) "
      f"x {len(SCRIPT)} Hindi lines")

# ------------------------------------------------------------ 1. synthesise
import librosa
import soundfile as sf
from miocodec import MioCodecModel
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[2/4] loading {args.lm} + codec on {dev}")
codec = MioCodecModel.from_pretrained(args.codec).to(dev).eval()
tok = AutoTokenizer.from_pretrained(args.lm, trust_remote_code=True)
lm = AutoModelForCausalLM.from_pretrained(args.lm, torch_dtype=torch.bfloat16,
                                          device_map=dev, trust_remote_code=True).eval()


def gen_codes(text, seed):
    torch.manual_seed(seed)
    p = tok.apply_chat_template([{"role": "user", "content": text}],
                                tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        o = lm.generate(**inp, max_new_tokens=1024, do_sample=True,
                        temperature=0.9, top_p=0.9, pad_token_id=tok.eos_token_id)
    new = o[0][inp["input_ids"].shape[1]:]
    return [t.item() - SPEECH_OFFSET for t in new
            if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + CODEBOOK]


def render(codes, vec):
    ct = torch.as_tensor(codes, dtype=torch.long, device=dev).reshape(-1)
    ge = torch.as_tensor(np.asarray(vec, np.float32), device=dev).reshape(-1)
    with torch.no_grad():
        return codec.decode(global_embedding=ge,
                            content_token_indices=ct).squeeze().float().cpu().numpy()


# seed 100+i matches run_indic_mint.py, so these are the same utterances S6
# audited for drift and consistency, not a fresh sample.
codes_per_line = [gen_codes(t, 100 + i) for i, t in enumerate(SCRIPT)]
print("      content tokens per line: " +
      ", ".join(str(len(c)) for c in codes_per_line))

wavs = {}
for li, codes in enumerate(codes_per_line):
    if not codes:
        print(f"      line{li}: LM emitted no speech tokens, skipping")
        continue
    for label, kind, vec in ARMS:
        w = render(codes, vec)
        wavs[(li, label)] = w
        if kind == "donor":
            sf.write(os.path.join(OUT, f"ctrl_{label.replace(':', '_')}_line{li}.wav"),
                     w, CODEC_SR)

# THE CONTROL THAT MAKES THE COMPARISON VALID, asserted before any CER is read:
# one token stream per line means every arm must produce the same number of
# samples. A length difference would mean the arms are not saying the same
# thing, and the difference below would be measuring something else.
for li in sorted({k[0] for k in wavs}):
    lens = {len(wavs[(li, l)]) for l, _, _ in ARMS}
    if len(lens) != 1:
        sys.exit(f"ABORT: line{li} rendered to {sorted(lens)} samples across arms. "
                 "The arms are not decoding identical content.")
print("      identical render length across all arms on every line — arms comparable")
del lm, codec
torch.cuda.empty_cache()

# ------------------------------------------------------------------ 2. ASR
print(f"[3/4] transcribing {len(wavs)} renders with {args.asr}")
tok_env = os.environ.get("HF_TOKEN") or (
    io.open(".hf_token", encoding="utf-8").read().strip()
    if os.path.exists(".hf_token") else None)
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model=args.asr,
                device=0 if args.device == "cuda" else -1, token=tok_env)


def transcribe(w):
    w16 = librosa.resample(np.asarray(w, np.float32), orig_sr=CODEC_SR, target_sr=16000)
    return pipe(w16, generate_kwargs={"language": "hi", "task": "transcribe"})["text"]


rows = []
for (li, label), w in sorted(wavs.items()):
    kind = next(k for l, k, _ in ARMS if l == label)
    hyp = transcribe(w)
    c = cer(SCRIPT[li], hyp)
    rows.append({"line": li, "arm": label, "kind": kind, "cer": c,
                 "ref": normalise_transcript(SCRIPT[li]),
                 "hyp": normalise_transcript(hyp)})
    print(f"      line{li} {label:<18} CER {c:.3f}")

json.dump(rows, io.open(os.path.join(OUT, "mint_intelligibility.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

# --------------------------------------------------------------- 3. report
mint_c = [r["cer"] for r in rows if r["kind"] == "minted"]
don_c = [r["cer"] for r in rows if r["kind"] == "donor"]
print()
print("=" * 78)
print("S6b — does minting a voice cost intelligibility?")
print("=" * 78)
print(f"  {'line':<7} " + " ".join(f"{l[:11]:>12}" for l, _, _ in ARMS))
for li in sorted({r["line"] for r in rows}):
    cs = {r["arm"]: r["cer"] for r in rows if r["line"] == li}
    print(f"  line{li:<3} " + " ".join(f"{cs.get(l, NAN):>12.3f}" for l, _, _ in ARMS))
print()
print(f"  minted  mean CER {np.nanmean(mint_c):.3f}  (n={len(mint_c)})")
print(f"  donor   mean CER {np.nanmean(don_c):.3f}  (n={len(don_c)})   <- control")
print(f"  cost of minting  {np.nanmean(mint_c) - np.nanmean(don_c):+.3f}")
print()
print("  Same content tokens through both arms, so the LM, the sampling seed,")
print("  the codec and the ASR's spelling habits are common to both and cancel")
print("  in the difference. Read the difference; the absolute CER is an upper")
print("  bound carrying whisper's Devanagari orthography.")
for li in sorted({r["line"] for r in rows})[:1]:
    r0 = next(r for r in rows if r["line"] == li)
    print()
    print(f"  ref: {r0['ref']}")
    for r in rows:
        if r["line"] == li:
            print(f"  {r['arm'][:16]:<17} {r['hyp'][:56]}")
print("=" * 78)
