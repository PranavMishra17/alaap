"""
S20b — is there a WavLM layer that sees the codec, and does it stay honest?

`S20` built a working naturalness gate with one stated weakness: the codec
roundtrip scores 53.3 against real speech's 53.3, identical, while a listener
told those apart 2 times in 3 (`S19`). So the gate sees GENERATED speech and is
blind to the codec's own contribution -- half of what `S19` measured.

Layer 6 was chosen on general grounds (middle layers carry phonetic and quality
detail, top layers drift toward the pretext task) and never swept. If some other
layer separates the roundtrip, the limitation disappears.

TWO THINGS THIS MUST NOT DO, and they are the reason it is a sweep with checks
rather than a search for the biggest number:

  1. A layer that separates roundtrip from real by tracking PITCH is the
     DNSMOS trap again (RESEARCH/06: humans r = -0.059, DNSMOS -0.788). Every
     layer is scored on that correlation, and one that separates while failing
     it is rejected, not celebrated.
  2. A layer that separates two REAL clips from each other is not measuring
     naturalness at all. `S19`'s human control was a real-vs-real tie; a layer
     that fails the same control is measuring channel or speaker.

n MATTERS HERE AND S20 DID NOT HAVE IT. Its roundtrip figure rested on 3 clips.
This builds 40, because the question is whether a small effect exists and 3
samples cannot answer that either way.

    envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_layer_sweep.py --phase audio
    envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_layer_sweep.py --phase feats
    envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_layer_sweep.py --phase sweep
"""
import argparse
import glob
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
from alaap.acoustics import f0_track
from alaap.metrics import isolation_pct

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--phase", default="sweep", choices=["audio", "feats", "sweep"])
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--n", type=int, default=100, help="real clips; 40 also become roundtrips")
ap.add_argument("--n-round", type=int, default=40)
ap.add_argument("--codec", default="Aratako/MioCodec-25Hz-44.1kHz-v2")
args = ap.parse_args()

SR_CORPUS, SR_CODEC, SR_W = 24000, 44100, 16000
AUDIO = os.path.join(OUT, f"sweep_audio_{args.corpus}_{args.n}.npz")
FEATS = os.path.join(OUT, f"sweep_feats_{args.corpus}_{args.n}.npz")

import librosa

# ------------------------------------------------------------- phase audio
if args.phase == "audio":
    from alaap.data import stream_clips
    from miocodec import MioCodecModel
    print(f"[audio] {args.n} real clips, {args.n_round} of them also round-tripped")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = MioCodecModel.from_pretrained(args.codec).to(dev).eval()
    payload, f0s, k = {}, [], 0
    for c in stream_clips(args.corpus, n=args.n, per_speaker=1, dev_only=True,
                          min_dur=2.0, max_dur=8.0, progress_every=25):
        w = np.asarray(c.wav, np.float32)
        payload[f"R{k}"] = librosa.resample(w, orig_sr=SR_CORPUS, target_sr=SR_W)
        f0s.append(float(np.nanmean(f0_track(w, SR_CORPUS))))
        if k < args.n_round:
            t = torch.as_tensor(librosa.resample(w, orig_sr=SR_CORPUS, target_sr=SR_CODEC)
                                ).reshape(1, -1).to(dev)
            with torch.no_grad():
                f = codec.encode(t, return_content=True, return_global=True)
                y = codec.decode(
                    global_embedding=f.global_embedding.reshape(-1),
                    content_token_indices=f.content_token_indices.reshape(-1)
                ).squeeze().float().cpu().numpy()
            payload[f"T{k}"] = librosa.resample(np.asarray(y, np.float32),
                                                orig_sr=SR_CODEC, target_sr=SR_W)
        k += 1
    payload["n"] = k
    payload["n_round"] = min(args.n_round, k)
    payload["f0"] = np.array(f0s, float)
    np.savez_compressed(AUDIO, **payload)
    print(f"        {k} real, {payload['n_round']} roundtrip -> {AUDIO}")
    sys.exit(0)

# ------------------------------------------------------------- phase feats
if args.phase == "feats":
    import soundfile as sf
    import torchaudio
    if not os.path.exists(AUDIO):
        sys.exit(f"run --phase audio first (missing {AUDIO})")
    z = np.load(AUDIO, allow_pickle=True)
    m = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model().eval()
    NL = 12

    def all_layers(x):
        """One forward pass gives every layer -- sweeping 12 layers costs one."""
        if len(x) < SR_W // 2:
            return None
        with torch.no_grad():
            hs, _ = m.extract_features(torch.from_numpy(np.asarray(x, np.float32)
                                                        ).unsqueeze(0), num_layers=NL)
        return np.stack([h.squeeze(0).mean(0).numpy() for h in hs])   # (NL, D)

    def collect(keys):
        out = []
        for k in keys:
            v = all_layers(z[k])
            if v is not None:
                out.append(v)
        return np.stack(out) if out else None

    R = collect([f"R{i}" for i in range(int(z["n"]))])
    T = collect([f"T{i}" for i in range(int(z["n_round"]))])
    print(f"[feats] real {R.shape}  roundtrip {T.shape}")

    # synthetic renders already on disk
    pats = ["experiments/S6-indic-mint/out/*_line*.wav",
            "experiments/S7-indic-catalog/out/*_line*.wav"]
    S, sf0 = [], []
    for pat in pats:
        for f in sorted(glob.glob(pat))[:24]:
            w, sr = sf.read(f, dtype="float32")
            if w.ndim > 1:
                w = w.mean(1)
            v = all_layers(librosa.resample(w, orig_sr=sr, target_sr=SR_W))
            if v is None:
                continue
            S.append(v)
            sf0.append(float(np.nanmean(f0_track(
                librosa.resample(w, orig_sr=sr, target_sr=SR_CORPUS), SR_CORPUS))))
    S = np.stack(S)
    print(f"        synthetic {S.shape}")
    np.savez_compressed(FEATS, R=R, T=T, S=S, f0=z["f0"], sf0=np.array(sf0, float))
    print(f"        -> {FEATS}")
    sys.exit(0)

# ------------------------------------------------------------- phase sweep
if not os.path.exists(FEATS):
    sys.exit(f"run --phase audio then --phase feats (missing {FEATS})")
z = np.load(FEATS, allow_pickle=True)
R, T, S, f0, sf0 = z["R"], z["T"], z["S"], z["f0"], z["sf0"]
NL = R.shape[1]
rng = np.random.default_rng(0)
idx = rng.permutation(len(R))
half = len(R) // 2
ref_i, held_i = idx[:half], idx[half:]
print(f"[sweep] reference {len(ref_i)} real | held-out {len(held_i)} | "
      f"roundtrip {len(T)} | synthetic {len(S)} | {NL} layers")

print()
print(f"  {'layer':>5} {'real':>7} {'round':>7} {'synth':>7} {'gap':>7} {'t':>6} "
      f"{'r(f0)':>7} {'ctrl d':>8}  verdict")
print("  " + "-" * 80)
rows = []
for L in range(NL):
    ref = R[ref_i, L, :]
    s_held = np.array([isolation_pct(R[i, L, :][None, :], ref) for i in held_i])
    s_rt = np.array([isolation_pct(t[None, :], ref) for t in T[:, L, :]])
    s_syn = np.array([isolation_pct(s[None, :], ref) for s in S[:, L, :]])
    gap = float(s_rt.mean() - s_held.mean())
    allf = np.concatenate([f0[held_i], sf0])
    alls = np.concatenate([s_held, s_syn])
    m = np.isfinite(allf) & np.isfinite(alls)
    rp = float(np.corrcoef(allf[m], alls[m])[0, 1]) if m.sum() > 10 else np.nan
    h1, h2 = s_held[: len(s_held) // 2], s_held[len(s_held) // 2:]
    d = float(abs(h1.mean() - h2.mean()) /
              max(np.sqrt((h1.var(ddof=1) + h2.var(ddof=1)) / 2), 1e-9))
    # A GAP IS NOT A FINDING WITHOUT A TEST. The first version of this script
    # called any gap >= 5 points "sees the codec" -- an effect-size cut with
    # nothing behind it. At this n the standard error on the difference is
    # about 6 points, so a +7.9 gap sits at t = 1.19 and is not distinguishable
    # from zero. Two layers were declared winners on that basis.
    se = float(np.sqrt(s_rt.var(ddof=1) / len(s_rt) + s_held.var(ddof=1) / len(s_held)))
    t = gap / se if se > 0 else 0.0
    ok_pitch, ok_ctrl = abs(rp) < 0.35, d < 0.35
    ok_syn = s_syn.mean() > s_held.mean() + 5
    v = ("SEES THE CODEC" if abs(t) >= 2 and gap > 0 and ok_pitch and ok_ctrl and ok_syn
         else "pitch trap" if abs(t) >= 2 and gap > 0 and not ok_pitch
         else "fails control" if abs(t) >= 2 and gap > 0 and not ok_ctrl
         else "gap not significant" if ok_pitch and ok_ctrl and ok_syn
         else "unusable")
    rows.append({"layer": L, "real": s_held.mean(), "round": s_rt.mean(),
                 "synth": s_syn.mean(), "gap": gap, "se": se, "t": t,
                 "r_f0": rp, "ctrl_d": d, "verdict": v})
    print(f"  {L:>5} {s_held.mean():>6.1f}% {s_rt.mean():>6.1f}% {s_syn.mean():>6.1f}% "
          f"{gap:>+7.1f} {t:>6.2f} {rp:>+7.3f} {d:>8.2f}  {v}")

json.dump(rows, io.open(os.path.join(OUT, "layer_sweep.json"), "w", encoding="utf-8"),
          indent=2)
good = [r for r in rows if r["verdict"] == "SEES THE CODEC"]
print()
print("=" * 78)
if good:
    b = max(good, key=lambda r: r["gap"])
    print(f"  Layer {b['layer']} separates the codec roundtrip by {b['gap']:+.1f} points")
    print(f"  while keeping r(f0) = {b['r_f0']:+.3f} and a real-vs-real tie "
          f"(d = {b['ctrl_d']:.2f}).")
    print(f"  S20's limitation is removable: switch the gate to layer {b['layer']}.")
else:
    best = max(rows, key=lambda r: r["gap"])
    need = int(np.ceil(len(T) * (2.0 / max(abs(best["t"]), 1e-6)) ** 2))
    print("  NO layer separates the codec roundtrip SIGNIFICANTLY.")
    print(f"  The largest gap is layer {best['layer']} at {best['gap']:+.1f} points,")
    print(f"  t = {best['t']:.2f}, which is not distinguishable from zero.")
    print()
    print("  So S20's stated limitation is NOT CONFIRMED either -- its n=3")
    print("  roundtrip sample could not have detected a gap this size. Both the")
    print("  claim and its refutation are under-powered.")
    print(f"  Detecting a gap of this magnitude at t=2 needs ~{need} roundtrip")
    print(f"  clips against ~{need} real, against the {len(T)} used here.")
print("=" * 78)
