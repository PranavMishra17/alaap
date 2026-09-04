"""
E3 — Are Qwen3-TTS speaker-embedding dimensions heterogeneously scaled?

Decides whether per-dimension rescaling is mandatory in the mapper's output
layer (invariant I1). If dimension ranges span more than ~1 order of magnitude,
any isotropic operation (averaging, plain Gaussian noise, unweighted MSE, naive
LERP) will silently collapse every generated identity onto one voice.

Precedent: Meyer et al. Interspeech 2022 found a 704-d ECAPA+x-vector concat
whose narrowest dim spanned [-0.97, -0.11] and widest [-72.37, 81.59].

Outputs -> experiments/E3/out/
"""
import argparse, io, json, os, sys, time, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import librosa
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
ap.add_argument("--n", type=int, default=1000, help="total embeddings")
ap.add_argument("--per-speaker", type=int, default=1, help="max utterances per speaker")
ap.add_argument("--min-dur", type=float, default=2.0)
ap.add_argument("--max-dur", type=float, default=15.0)
args = ap.parse_args()

SR = 24000  # extract_speaker_embedding asserts sr == 24000

# ---------------------------------------------------------------- 1. model
print(f"[1/4] loading {args.model} ...", flush=True)
t0 = time.time()
from qwen_tts import Qwen3TTSModel

wrapper = Qwen3TTSModel.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="cuda",
)
model = wrapper.model
model.eval()
print(f"      loaded in {time.time()-t0:.1f}s | device={model.device} dtype={model.dtype}", flush=True)
print(f"      speaker_encoder_sample_rate = {model.speaker_encoder_sample_rate}", flush=True)
if torch.cuda.is_available():
    print(f"      VRAM after load: {torch.cuda.memory_allocated()/2**30:.2f} GB "
          f"(reserved {torch.cuda.memory_reserved()/2**30:.2f} GB)", flush=True)

# ---------------------------------------------------------------- 2. audio
print(f"[2/4] streaming GLOBE_V2 for {args.n} clips "
      f"(<= {args.per_speaker}/speaker) ...", flush=True)
from datasets import load_dataset, Audio

ds = load_dataset("MushanW/GLOBE_V2", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

clips, meta, per_spk, seen, skipped = [], [], defaultdict(int), 0, 0
t0 = time.time()
for r in ds:
    seen += 1
    spk = r["speaker_id"]
    if per_spk[spk] >= args.per_speaker:
        continue
    try:
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    except Exception:
        skipped += 1
        continue
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    dur = len(wav) / sr
    if not (args.min_dur <= dur <= args.max_dur):
        continue
    if sr != SR:
        wav = librosa.resample(y=wav.astype(np.float32), orig_sr=int(sr), target_sr=SR)
    clips.append(wav.astype(np.float32))
    meta.append({"speaker_id": spk, "accent": r.get("accent"), "age": r.get("age"),
                 "gender": r.get("gender"), "dur": round(dur, 2)})
    per_spk[spk] += 1
    if len(clips) % 100 == 0:
        print(f"      {len(clips)}/{args.n} clips | {len(per_spk)} speakers "
              f"| {seen} scanned | {time.time()-t0:.0f}s", flush=True)
    if len(clips) >= args.n:
        break

print(f"      collected {len(clips)} clips from {len(per_spk)} distinct speakers "
      f"({seen} scanned, {skipped} undecodable) in {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- 3. extract
print(f"[3/4] extracting speaker embeddings ...", flush=True)
t0 = time.time()
embs = []
with torch.inference_mode():
    for i, wav in enumerate(clips):
        e = model.extract_speaker_embedding(audio=wav, sr=SR)
        embs.append(e.detach().float().cpu().numpy().reshape(-1))
        if (i + 1) % 200 == 0:
            print(f"      {i+1}/{len(clips)} | {time.time()-t0:.0f}s", flush=True)

Z = np.stack(embs).astype(np.float64)   # (N, D)
print(f"      done: Z.shape={Z.shape} in {time.time()-t0:.0f}s "
      f"({len(clips)/(time.time()-t0):.1f} clips/s)", flush=True)

np.save(os.path.join(OUT, "embeddings.npy"), Z.astype(np.float32))
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f)

# ---------------------------------------------------------------- 4. analyse
print(f"[4/4] analysing ...", flush=True)
N, D = Z.shape
dmin, dmax = Z.min(0), Z.max(0)
dmean, dstd = Z.mean(0), Z.std(0)
drange = dmax - dmin
norms = np.linalg.norm(Z, axis=1)

order = np.argsort(drange)
ratio = drange.max() / max(drange.min(), 1e-12)
decades = np.log10(ratio)

# is it L2-normalised?
norm_cv = norms.std() / norms.mean()
l2_normalised = bool(norm_cv < 0.01)

res = {
  "model": args.model,
  "n_embeddings": int(N), "dim": int(D),
  "n_distinct_speakers": len(per_spk),
  "dtype_extracted": "bfloat16 -> float32",
  "per_dim_range": {
      "min": float(drange.min()), "max": float(drange.max()),
      "median": float(np.median(drange)), "mean": float(drange.mean()),
      "max_over_min_ratio": float(ratio),
      "orders_of_magnitude": float(decades),
  },
  "narrowest_dim": {"idx": int(order[0]), "min": float(dmin[order[0]]),
                    "max": float(dmax[order[0]]), "range": float(drange[order[0]])},
  "widest_dim":     {"idx": int(order[-1]), "min": float(dmin[order[-1]]),
                    "max": float(dmax[order[-1]]), "range": float(drange[order[-1]])},
  "per_dim_std": {"min": float(dstd.min()), "max": float(dstd.max()),
                  "median": float(np.median(dstd)),
                  "max_over_min_ratio": float(dstd.max()/max(dstd.min(),1e-12))},
  "mean_abs": {"min": float(np.abs(dmean).min()), "max": float(np.abs(dmean).max()),
               "median": float(np.median(np.abs(dmean)))},
  "l2_norm": {"mean": float(norms.mean()), "std": float(norms.std()),
              "min": float(norms.min()), "max": float(norms.max()),
              "coeff_of_variation": float(norm_cv)},
  "is_l2_normalised": l2_normalised,
  "zero_centred": bool(np.abs(dmean).mean() < 0.1 * dstd.mean()),
  "VERDICT_rescaling_mandatory": bool(decades >= 1.0),
}
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(res, f, indent=2)

# ---------------------------------------------------------------- plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"E3 — Qwen3-TTS speaker-embedding geometry\n"
             f"{args.model}  |  N={N} clips from {len(per_spk)} speakers  |  D={D}",
             fontsize=12)

ax[0,0].plot(drange[order], lw=1.4)
ax[0,0].set_yscale("log"); ax[0,0].set_title("Per-dimension range (sorted, log y)")
ax[0,0].set_xlabel("dimension (sorted)"); ax[0,0].set_ylabel("max - min")
ax[0,0].grid(alpha=.3)

ax[0,1].hist(np.log10(np.maximum(drange, 1e-12)), bins=60, color="tab:orange")
ax[0,1].set_title("log10(range) histogram")
ax[0,1].set_xlabel("log10(max - min)"); ax[0,1].set_ylabel("count"); ax[0,1].grid(alpha=.3)

idx = np.arange(D)
ax[1,0].fill_between(idx, dmean-dstd, dmean+dstd, alpha=.35, label="mean +/- std")
ax[1,0].plot(idx, dmean, lw=.6, label="mean")
ax[1,0].axhline(0, color="k", lw=.6, ls="--")
ax[1,0].set_title("Per-dimension mean +/- std (unsorted)")
ax[1,0].set_xlabel("dimension"); ax[1,0].legend(); ax[1,0].grid(alpha=.3)

ax[1,1].hist(norms, bins=60, color="tab:green")
ax[1,1].set_title(f"||z|| distribution  (CV={norm_cv:.4f}, "
                  f"{'L2-NORMALISED' if l2_normalised else 'NOT normalised'})")
ax[1,1].set_xlabel("L2 norm"); ax[1,1].set_ylabel("count"); ax[1,1].grid(alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "e3_geometry.png"), dpi=140)

# ---------------------------------------------------------------- report
print()
print("=" * 74)
print("E3 RESULT")
print("=" * 74)
print(f"  model                 {args.model}")
print(f"  embeddings            {N} from {len(per_spk)} distinct speakers")
print(f"  dimension             {D}")
print()
print(f"  per-dim range  min    {drange.min():.6g}")
print(f"                 median {np.median(drange):.6g}")
print(f"                 max    {drange.max():.6g}")
print(f"  max/min ratio         {ratio:.1f}x   ({decades:.2f} orders of magnitude)")
print()
print(f"  narrowest dim  #{order[0]:<5} [{dmin[order[0]]:+.4f}, {dmax[order[0]]:+.4f}]")
print(f"  widest dim     #{order[-1]:<5} [{dmin[order[-1]]:+.4f}, {dmax[order[-1]]:+.4f}]")
print()
print(f"  per-dim std    ratio  {dstd.max()/max(dstd.min(),1e-12):.1f}x")
print(f"  ||z||                 mean={norms.mean():.4f} std={norms.std():.4f} CV={norm_cv:.5f}")
print(f"  L2-normalised?        {'YES' if l2_normalised else 'NO'}")
print(f"  zero-centred?         {'YES' if res['zero_centred'] else 'NO'}")
print()
print("-" * 74)
if res["VERDICT_rescaling_mandatory"]:
    print("  VERDICT: PER-DIMENSION RESCALING IS MANDATORY (invariant I1 confirmed)")
    print("           Ranges span >= 1 order of magnitude. Any isotropic operation")
    print("           will collapse identities. Mapper output layer must rescale.")
else:
    print("  VERDICT: ranges are comparatively uniform (< 1 order of magnitude).")
    print("           Rescaling is cheap insurance but not strictly forced.")
print("-" * 74)
print(f"  artefacts -> {OUT}")
print("=" * 74)
