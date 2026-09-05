"""
S14 — direction by re-timing the signal, since the text channel does not work

`S13`/`S13b` established that Indic-Mio's text channel carries no direction:
the emotion tags are not tokens, and word stress is inert. What survived was
the harder property — identity is provably robust to whatever the text does —
so a direction channel is buildable here as soon as something actually drives.

`S12` says `speaking_rate` is a genuine delivery axis (within/between ratio
2.01) and near-worthless for identity (weight 0.10-0.31). So drive it directly,
in the signal, and stop asking the model to cooperate. A phase vocoder changes
duration while preserving pitch, which is exactly the shape this needs: move
the delivery axis, leave the identity axis alone.

THREE PROPERTIES, ALL STATABLE BEFORE THE RUN

  1. rate tracks the request     measured speaking_rate should scale as 1/factor.
                                 If it does not, the instrument is broken and no
                                 other number here is readable.
  2. pitch does not move         a phase vocoder preserves f0 by construction.
                                 f0_mean must stay put, because S12 flagged it as
                                 the contested axis and ADR-012 puts it off-limits
                                 to direction.
  3. the words survive           CER against the prompt, at every factor.

NAIVE RESAMPLING IS THE NEGATIVE CONTROL. Simply playing the audio faster
changes duration AND pitch together -- it is what the Indic-Mio card's own
44100-vs-24000 bug does accidentally. Including it shows the measurement can
detect the failure it is claiming to avoid; if naive resampling scores as well
as the vocoder on identity, the identity metric is not sensitive enough to be
trusted here.

THE DELIVERABLE is not "it works" -- it is the measured RANGE over which it
works, so the Direction channel can have a bound like DRIFT_FLOOR does.

    envs/qwen3/Scripts/python.exe experiments/S14-rate-control/run_rate_control.py
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
from alaap.metrics import CALIBRATION, cer, normalise_transcript
from alaap.timing import retime

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
S6_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "S6-indic-mint", "out")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--clips", type=int, default=250)
ap.add_argument("--per-speaker", type=int, default=2)
ap.add_argument("--asr", default="openai/whisper-small")
args = ap.parse_args()

SR, SPEECH_OFFSET, CODEBOOK = 44100, 151669, 12800
# stretch factor: <1 compresses (faster speech), >1 expands (slower)
FACTORS = [0.60, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00]
LINES = [("hi", "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।"),
         ("en", "The mountains remember every footstep, even the ones you regret.")]

import librosa
import soundfile as sf

# ------------------------------------------------------------- 1. base renders
print("[1/4] minting 2 voices and rendering the base lines")
S4 = f"experiments/S4-indic/out/measured_{args.corpus}_{args.clips}_{args.per_speaker}.npz"
EMB = os.path.join(S6_OUT, f"mio_emb_{args.corpus}_{args.clips}_{args.per_speaker}.npz")
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
        "a high voice, crisp-toned, speaking quickly"]
voices = [mapper.mint(d, novelty=0.0, seed=i, top_k=2).vector for i, d in enumerate(CAST)]

from miocodec import MioCodecModel
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda" if torch.cuda.is_available() else "cpu"
codec = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz-v2").to(dev).eval()
tok = AutoTokenizer.from_pretrained("SPRINGLab/Indic-Mio", trust_remote_code=True)
lm = AutoModelForCausalLM.from_pretrained("SPRINGLab/Indic-Mio", torch_dtype=torch.bfloat16,
                                          device_map=dev, trust_remote_code=True).eval()
from alaap.encoder import IndependentSV
sv = IndependentSV()


def gen(text, seed):
    torch.manual_seed(seed)
    p = tok.apply_chat_template([{"role": "user", "content": text}],
                                tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        o = lm.generate(**inp, max_new_tokens=1024, do_sample=True, temperature=0.9,
                        top_p=0.9, pad_token_id=tok.eos_token_id)
    nn = o[0][inp["input_ids"].shape[1]:]
    return [t.item() - SPEECH_OFFSET for t in nn
            if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + CODEBOOK]


def render(cc, vec):
    ct = torch.as_tensor(cc, dtype=torch.long, device=dev).reshape(-1)
    ge = torch.as_tensor(np.asarray(vec, np.float32), device=dev).reshape(-1)
    with torch.no_grad():
        return codec.decode(global_embedding=ge,
                            content_token_indices=ct).squeeze().float().cpu().numpy()


base = {}
for li, (lang, text) in enumerate(LINES):
    cc = gen(text, 2000 + li)
    for vi, v in enumerate(voices):
        base[(vi, li)] = render(cc, v)
    print(f"      line{li} ({lang}) rendered for {len(voices)} voices: "
          + ", ".join(f"{len(base[(vi, li)])/SR:.2f}s" for vi in range(len(voices))),
          flush=True)
del lm
torch.cuda.empty_cache()

# ------------------------------------------------------------- 2. re-time
print(f"[2/4] re-timing at {len(FACTORS)} factors, two methods")


# The canonical implementation now lives in alaap.timing, so the experiment
# and the shipped Direction channel cannot drift apart. `retime` takes a SPEED
# multiplier; this experiment sweeps STRETCH factors, hence the reciprocal.
def vocoder(w, f):
    """Phase vocoder: duration changes, pitch does not."""
    return retime(w, 1.0 / f)


def naive(w, f):
    """Resample-and-relabel: duration AND pitch change. The negative control."""
    return librosa.resample(np.asarray(w, np.float32),
                            orig_sr=int(SR * f), target_sr=SR)


def embed(w):
    return sv.embed(librosa.resample(w, orig_sr=SR, target_sr=16000), sr=16000)


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


rows = []
for (vi, li), w0 in base.items():
    lang, text = LINES[li]
    e0 = embed(w0)
    a0 = measure(librosa.resample(w0, orig_sr=SR, target_sr=24000), text, 24000)
    for f in FACTORS:
        for meth, fn in (("vocoder", vocoder), ("naive", naive)):
            w = fn(w0, f)
            a = measure(librosa.resample(w, orig_sr=SR, target_sr=24000), text, 24000)
            rows.append({"voice": vi, "line": li, "lang": lang, "factor": f,
                         "method": meth, "rate": a.speaking_rate, "f0": a.f0_mean,
                         "rate0": a0.speaking_rate, "f00": a0.f0_mean,
                         "ecapa": cos(e0, embed(w)), "wav": None})
            if vi == 0 and meth == "vocoder" and f in (0.7, 1.0, 1.5):
                p = os.path.join(OUT, f"{lang}_x{f:.2f}.wav")
                sf.write(p, w, SR)
                rows[-1]["wav"] = p
    print(f"      voice{vi} line{li} done", flush=True)

# THE FIRST PROPERTY, asserted before anything else is read.
voc = [r for r in rows if r["method"] == "vocoder" and np.isfinite(r["rate"])]
req = np.array([1.0 / r["factor"] for r in voc])
got = np.array([r["rate"] / r["rate0"] for r in voc])
rr = float(np.corrcoef(req, got)[0, 1])
print(f"      requested vs achieved rate ratio: r = {rr:.4f}")
if rr < 0.95:
    sys.exit("ABORT: measured speaking_rate does not track the requested factor. "
             "The instrument is broken and no number below is readable.")

# ------------------------------------------------------------------ 3. ASR
print(f"[3/4] CER at every factor with {args.asr}")
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model=args.asr, device=-1)
for r in rows:
    if r["voice"] != 0:
        continue
    lang, text = LINES[r["line"]]
    w = (vocoder if r["method"] == "vocoder" else naive)(base[(0, r["line"])], r["factor"])
    w16 = librosa.resample(w, orig_sr=SR, target_sr=16000)
    hyp = pipe(w16, generate_kwargs={"language": lang, "task": "transcribe"})["text"]
    r["cer"] = cer(text, hyp)
    r["hyp"] = normalise_transcript(hyp)

json.dump([{k: v for k, v in r.items() if k != "wav"} for r in rows],
          io.open(os.path.join(OUT, "rows.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- 4. report
c_same, c_diff, _ = CALIBRATION["ecapa"]
print()
print("=" * 78)
print("S14 — driving speaking_rate in the signal instead of through the model")
print("=" * 78)
print(f"  requested vs achieved rate: r = {rr:.4f}  (property 1: the instrument works)")
print()
print(f"  {'factor':>7} {'rate x':>8} {'f0 shift':>9} {'ECAPA':>8} {'normed':>8} "
      f"{'CER hi':>7} {'CER en':>7}")
print("  " + "-" * 66)
for f in FACTORS:
    v = [r for r in rows if r["method"] == "vocoder" and r["factor"] == f]
    if not v:
        continue
    rx = np.nanmean([r["rate"] / r["rate0"] for r in v])
    df = np.nanmean([r["f0"] - r["f00"] for r in v])
    ec = np.nanmean([r["ecapa"] for r in v])
    ch = np.nanmean([r.get("cer", np.nan) for r in v if r["lang"] == "hi"])
    ce = np.nanmean([r.get("cer", np.nan) for r in v if r["lang"] == "en"])
    print(f"  {f:>7.2f} {rx:>8.2f} {df:>+9.1f} {ec:>8.4f} "
          f"{(ec-c_diff)/(c_same-c_diff):>+8.3f} {ch:>7.3f} {ce:>7.3f}")

print()
print("  NEGATIVE CONTROL — naive resampling, which moves pitch too")
print(f"  {'factor':>7} {'f0 shift':>9} {'ECAPA':>8} {'CER en':>7}")
print("  " + "-" * 36)
for f in (0.70, 1.00, 1.50):
    v = [r for r in rows if r["method"] == "naive" and r["factor"] == f]
    if not v:
        continue
    print(f"  {f:>7.2f} {np.nanmean([r['f0']-r['f00'] for r in v]):>+9.1f} "
          f"{np.nanmean([r['ecapa'] for r in v]):>8.4f} "
          f"{np.nanmean([r.get('cer', np.nan) for r in v if r['lang']=='en']):>7.3f}")

# the deliverable: the usable range
ok = [f for f in FACTORS
      if np.nanmean([r["ecapa"] for r in rows
                     if r["method"] == "vocoder" and r["factor"] == f]) >= 0.80
      and np.nanmean([r.get("cer", np.nan) for r in rows
                      if r["method"] == "vocoder" and r["factor"] == f
                      and r["lang"] == "en"]) <= 0.10]
print()
if ok:
    print(f"  USABLE RANGE (ECAPA >= 0.80 and English CER <= 0.10): "
          f"x{min(ok):.2f} to x{max(ok):.2f}")
    print(f"    = speech from {1/max(ok):.2f}x to {1/min(ok):.2f}x normal rate")
else:
    print("  USABLE RANGE: none of the tested factors clears both floors")
print("=" * 78)
