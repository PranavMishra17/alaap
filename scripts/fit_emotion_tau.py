"""
Fit the emotion direction vectors (tau) for a given backend, and write the asset.

WHY THIS EXISTS. `assets/emotion_tau_qwen3_0.6B.npz` was produced ad hoc during
E0 and nothing in the repo regenerated it. That mattered the moment E10 made
1.7B the default: **tau is per-model** — vectors fitted on 0.6B are 1024-d and
meaningless against 1.7B's 2048-d — so `Qwen3BaseRenderer._load_tau` correctly
dropped the only asset there was, and emotion direction silently stopped
working on the default backend. (It no longer fails silently: `steer()` now
returns a degradation, and `Direction(strict=True)` raises. But the capability
was still gone, and there was no way to rebuild it.)

THE METHOD is E0's, unchanged — task-vector arithmetic (arXiv:2606.05367):

    tau_e = E_i[ x(s_i, e) ] - E_i[ x(s_i, neutral) ]

averaging **within each speaker first**, then across speakers. Averaging over
clips directly would let a speaker with many clips dominate the direction, so
the emotion vector would carry that speaker's identity.

Fitted on a **speaker-disjoint half** of CREMA-D, matching E0, so the identity
costs E0 measured on the held-out half still describe this asset. Same seed,
same split.

LICENCE, and why the asset says so. CREMA-D is ODbL upstream and the HF mirror
declares nothing, so anything derived from it is **research-lane only** and must
not ship. That line is written into the asset's JSON, not just into this
docstring, so it travels with the file.

    envs/qwen3/Scripts/python.exe scripts/fit_emotion_tau.py \\
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base
"""
import argparse
import io
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alaap.encoder import SpeakerEncoder, DEFAULT_MODEL

SR = 24000
CACHE_DIR = os.path.join("experiments", "E0", "out")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--n", type=int, default=1200, help="CREMA-D clips to embed")
ap.add_argument("--out", default=None, help="defaults to assets/emotion_tau_qwen3_<tag>.npz")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

tag = "1.7B" if "1.7B" in args.model else ("0.6B" if "0.6B" in args.model
                                           else args.model.split("/")[-1])
out_npz = args.out or f"assets/emotion_tau_qwen3_{tag}.npz"
out_json = os.path.splitext(out_npz)[0] + ".json"
mtag = args.model.split("/")[-1].replace(".", "")
cache = os.path.join(CACHE_DIR, f"crema_{args.n}_{mtag}.npz")

# ------------------------------------------------------------------ 1. data
print(f"[1/3] CREMA-D embeddings for {args.model}")
if os.path.exists(cache):
    d = np.load(cache, allow_pickle=True)
    Zq = d["Zq"].astype(np.float64)
    emo, actor = np.asarray(d["emo"]), np.asarray(d["actor"])
    print(f"      cached {Zq.shape} from {os.path.basename(cache)}")
else:
    import librosa
    import soundfile as sf
    from datasets import Audio, load_dataset
    os.makedirs(CACHE_DIR, exist_ok=True)
    ds = load_dataset("confit/cremad-parquet", split="train",
                      streaming=True).cast_column("audio", Audio(decode=False))
    wavs, emo, actor, t0 = [], [], [], time.time()
    for r in ds:
        try:
            w, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        except Exception:
            continue
        if w.ndim > 1:
            w = w.mean(axis=1)
        if len(w) / sr < 1.0:
            continue
        if sr != SR:
            w = librosa.resample(y=w, orig_sr=sr, target_sr=SR)
        pk = float(np.abs(w).max())
        if pk > 1.0:
            w = w / pk
        wavs.append(w.astype(np.float32))
        emo.append(str(r["emotion"]))
        actor.append(os.path.basename(str(r["file"])).split("_")[0])
        if len(wavs) >= args.n:
            break
    print(f"      {len(wavs)} clips, {len(set(actor))} actors | "
          f"{time.time()-t0:.0f}s")
    enc = SpeakerEncoder(args.model)
    Zq = enc.embed_many(wavs, progress_every=200).astype(np.float64)
    del enc
    import torch
    torch.cuda.empty_cache()
    emo, actor = np.asarray(emo), np.asarray(actor)
    np.savez_compressed(cache, Zq=Zq.astype(np.float32), emo=emo, actor=actor)
    print(f"      cached -> {os.path.basename(cache)}")

dim = int(Zq.shape[1])
actors = sorted(set(actor.tolist()))
print(f"      Zq{Zq.shape} | {len(actors)} actors | "
      f"emotions {sorted(set(emo.tolist()))}")

# ------------------------------------------- 2. speaker-disjoint fit, as E0
rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(actors))
fit_actors = {actors[i] for i in perm[:len(actors) // 2]}
fit_m = np.array([a in fit_actors for a in actor])
print(f"[2/3] fitting on {len(fit_actors)} of {len(actors)} actors "
      f"(speaker-disjoint, seed {args.seed}) -- the other half stays held out "
      f"so E0's measured identity costs still apply")


def per_speaker_mean(mask):
    """E_i[x(s_i, .)]: average WITHIN each speaker first, then across."""
    out = []
    for a in sorted({s for s, k in zip(actor, mask) if k}):
        sel = mask & (actor == a)
        if sel.sum():
            out.append(Zq[sel].mean(0))
    return np.stack(out) if out else None


neutral_fit = per_speaker_mean(fit_m & (emo == "neutral"))
if neutral_fit is None:
    sys.exit("no neutral clips in the fit half -- cannot form a task vector")

taus, meta = {}, {}
for e in sorted(set(emo.tolist())):
    if e == "neutral":
        continue
    emo_fit = per_speaker_mean(fit_m & (emo == e))
    if emo_fit is None:
        continue
    n = min(len(emo_fit), len(neutral_fit))
    t = emo_fit[:n].mean(0) - neutral_fit[:n].mean(0)
    taus[e] = t
    meta[e] = {"n_speakers": int(n), "norm": float(np.linalg.norm(t))}
    print(f"      tau[{e:<8}] {n} speakers  ||tau|| = {meta[e]['norm']:.4f}")

if not taus:
    sys.exit("no emotion directions could be formed")

mean_norm = float(np.linalg.norm(Zq, axis=1).mean())
share = 100 * np.mean([m["norm"] for m in meta.values()]) / mean_norm
print(f"      mean ||x|| = {mean_norm:.3f} -> tau is {share:.1f}% of a vector")

# ----------------------------------------------------------------- 3. write
print(f"[3/3] writing {out_npz}")
os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
np.savez_compressed(
    out_npz, model=np.array(args.model), dim=np.array(dim),
    n_speakers=np.array(int(np.median([m["n_speakers"] for m in meta.values()]))),
    source=np.array("CREMA-D via confit/cremad-parquet"),
    **{f"tau_{e}": v.astype(np.float32) for e, v in taus.items()})

json.dump({
    "model": args.model,
    "dim": dim,
    "n_speakers": int(np.median([m["n_speakers"] for m in meta.values()])),
    "emotions": meta,
    "method": "arXiv:2606.05367 task-vector arithmetic",
    "fit": f"speaker-disjoint half of CREMA-D, seed {args.seed}; per-speaker "
           f"mean before cross-speaker mean",
    "tau_share_of_vector_norm_pct": round(share, 2),
    "licence": "derived from CREMA-D (ODbL upstream, mirror declares nothing) "
               "-- RESEARCH-LANE ONLY, do not ship",
}, open(out_json, "w"), indent=2)

print()
print("=" * 74)
print(f"  wrote {len(taus)} directions ({', '.join(sorted(taus))})")
print(f"  dim {dim} -- matches {args.model}")
print(f"  {out_npz}")
print(f"  {out_json}")
print()
print("  RESEARCH-LANE ONLY: derived from CREMA-D (ODbL upstream, mirror")
print("  declares nothing). Recorded in the JSON so it travels with the file.")
print("=" * 74)
