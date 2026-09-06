"""
S20 — a naturalness gate, and the validation it has to survive first

`S19` established two things. Synthesis is distinguishable from real speech by a
listener (7 of 7 decisive judgements), and **no metric in this project can see
it** — the codec roundtrip differs from the real recording by 0.0 dB of SNR and
0.18 dB of HNR, and a listener told them apart anyway. Drift, consistency, CER
and uniqueness are all identity or intelligibility gates. There is no
naturalness gate.

THE OBVIOUS FIX IS THE ONE THIS PROJECT ALREADY RULED OUT. `RESEARCH/06` §5.2
records the trap in detail: Takagi et al. perturbed F0 and measured six MOS
predictors against humans. Humans showed **r = -0.059** with mean F0. **DNSMOS
showed -0.788, UTMOSv2 -0.722.** For a project whose point is spanning the pitch
range, a MOS gate systematically rejects high-pitched voices for a reason humans
do not share -- it would fight the diversity axis directly. The same paper found
all six blind to prosody: pitch-accent corruption cost humans 1.84 MOS points
and moved every model less than 0.1.

SO THIS IS NOT A MOS PREDICTOR. It is a distance to the real-speech manifold, in
a self-supervised representation, using machinery the project already validated:
`metrics.isolation_pct` scores how typical a sample is against a reference set,
and `E12` validated it by checking that held-out REAL speakers score ~50%.

    score = isolation of a clip's WavLM features against real corpus clips
            high = sits where real speech does not

WavLM rather than ECAPA, deliberately. ECAPA is trained to be INVARIANT to
channel and quality so that it can identify a speaker through them -- exactly
the wrong property here.

THE VALIDATION BATTERY, all of which must pass before any number is used:

    1. held-out REAL clips score as typical         (or the gate is arbitrary)
    2. SYNTHETIC clips score higher than real       (or it measures nothing)
    3. |correlation with f0_mean| is small          (or it is DNSMOS again)
    4. it does not separate two REAL clips          (matching S19's human control)

Check 3 is the one that matters. A gate that passes 1, 2 and 4 and fails 3 is
the trap RESEARCH/06 warns about, wearing a different implementation.

    envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_gate.py --phase real
    envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_gate.py --phase synth
    envs/qwen3/Scripts/python.exe experiments/S20-naturalness-gate/run_gate.py --phase validate
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
from alaap.metrics import isolation_pct, knn_radius

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--phase", default="validate", choices=["real", "synth", "validate"])
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--n-real", type=int, default=140)
ap.add_argument("--layer", type=int, default=6,
                help="WavLM layer; middle layers carry phonetic/quality detail, "
                     "the top ones drift toward the pretext task")
args = ap.parse_args()

SR = 16000
REAL_CACHE = os.path.join(OUT, f"real_{args.corpus}_{args.n_real}_L{args.layer}.npz")
SYNTH_CACHE = os.path.join(OUT, f"synth_L{args.layer}.npz")

import librosa


def wavlm():
    import torchaudio
    b = torchaudio.pipelines.WAVLM_BASE_PLUS
    return b.get_model().eval()


def feats(model, w, sr):
    """Mean-pooled hidden states from one layer. Fixed-length regardless of duration."""
    x = librosa.resample(np.asarray(w, np.float32), orig_sr=sr, target_sr=SR)
    if len(x) < SR // 2:
        return None
    t = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        hs, _ = model.extract_features(t, num_layers=args.layer + 1)
    return hs[args.layer].squeeze(0).mean(0).numpy().astype(np.float64)


# ------------------------------------------------------------------ phase real
if args.phase == "real":
    from alaap.data import stream_clips
    print(f"[real] {args.n_real} clips from {args.corpus}")
    m = wavlm()
    F, P = [], []
    for i, c in enumerate(stream_clips(args.corpus, n=args.n_real, per_speaker=1,
                                       dev_only=True, min_dur=2.0, max_dur=10.0,
                                       progress_every=40)):
        v = feats(m, c.wav, 24000)
        if v is None:
            continue
        F.append(v)
        P.append(float(np.nanmean(f0_track(c.wav, 24000))))
    np.savez_compressed(REAL_CACHE, F=np.vstack(F), f0=np.array(P, float))
    print(f"       {len(F)} clips -> {REAL_CACHE}")
    sys.exit(0)

# ----------------------------------------------------------------- phase synth
if args.phase == "synth":
    import soundfile as sf
    # Every synthetic render this project has produced and kept, plus S19's
    # codec-roundtrip arm, which is the interesting middle case: real speech
    # that has only been through the codec.
    pats = [("S6 minted", "experiments/S6-indic-mint/out/*_line*.wav"),
            ("S7 catalog", "experiments/S7-indic-catalog/out/*_line*.wav"),
            ("S13 tagged", "experiments/S13-direction-channel/out/v0_*.wav"),
            ("S14 retimed", "experiments/S14-rate-control/out/hi_x1.00.wav")]
    m = wavlm()
    groups, F, G, P = {}, [], [], []
    for label, pat in pats:
        files = sorted(glob.glob(pat))[:24]
        for f in files:
            w, sr = sf.read(f, dtype="float32")
            if w.ndim > 1:
                w = w.mean(1)
            v = feats(m, w, sr)
            if v is None:
                continue
            F.append(v)
            G.append(label)
            P.append(float(np.nanmean(f0_track(
                librosa.resample(w, orig_sr=sr, target_sr=24000), 24000))))
        groups[label] = len(files)
        print(f"       {label:<14} {len(files)} files")
    # S19's A (real held-out) and B (codec roundtrip), from its cache
    ab = "experiments/S19-naturalness/out/ab.npz"
    if os.path.exists(ab):
        z = np.load(ab, allow_pickle=True)
        for i in range(int(z["n"])):
            for k, lab in (("A", "S19 real (held-out)"), ("B", "S19 codec roundtrip")):
                v = feats(m, z[f"{k}{i}"], 44100)
                if v is None:
                    continue
                F.append(v)
                G.append(lab)
                P.append(float(np.nanmean(f0_track(
                    librosa.resample(z[f"{k}{i}"], orig_sr=44100, target_sr=24000), 24000))))
        print(f"       S19 A/B        {int(z['n'])*2} files")
    np.savez_compressed(SYNTH_CACHE, F=np.vstack(F),
                        G=np.array(G, dtype=object), f0=np.array(P, float))
    print(f"       -> {SYNTH_CACHE}")
    sys.exit(0)

# -------------------------------------------------------------- phase validate
for f in (REAL_CACHE, SYNTH_CACHE):
    if not os.path.exists(f):
        sys.exit(f"missing {f}; run --phase real and --phase synth first")
zr, zs = np.load(REAL_CACHE, allow_pickle=True), np.load(SYNTH_CACHE, allow_pickle=True)
R, r_f0 = zr["F"], zr["f0"]
S, G, s_f0 = zs["F"], [str(x) for x in zs["G"]], zs["f0"]

# split the real set: half is the reference manifold, half is held-out
rng = np.random.default_rng(0)
idx = rng.permutation(len(R))
ref, held = R[idx[: len(R) // 2]], R[idx[len(R) // 2:]]
held_f0 = r_f0[idx[len(R) // 2:]]
print(f"[validate] reference {len(ref)} real clips | held-out {len(held)}")


def score(X):
    """One isolation score per clip, against the reference manifold."""
    return np.array([isolation_pct(x[None, :], ref) for x in X], float)


s_held = score(held)
s_syn = score(S)

print()
print("=" * 78)
print("S20 — is this a usable naturalness gate?")
print("=" * 78)
print(f"  {'group':<26} {'n':>4} {'isolation':>10}")
print(f"  {'held-out REAL':<26} {len(held):>4} {np.mean(s_held):>9.1f}%")
for lab in dict.fromkeys(G):
    m = [i for i, g in enumerate(G) if g == lab]
    print(f"  {lab:<26} {len(m):>4} {np.mean(s_syn[m]):>9.1f}%")

# ---- the four checks
print()
ok = {}
ok["1 held-out real is typical"] = 35.0 <= np.mean(s_held) <= 70.0
syn_only = [i for i, g in enumerate(G) if "real" not in g.lower()]
ok["2 synthetic scores higher"] = np.mean(s_syn[syn_only]) > np.mean(s_held) + 5
allf0 = np.concatenate([held_f0, s_f0])
alls = np.concatenate([s_held, s_syn])
m = np.isfinite(allf0) & np.isfinite(alls)
r_pitch = float(np.corrcoef(allf0[m], alls[m])[0, 1]) if m.sum() > 10 else np.nan
ok["3 not a pitch detector"] = abs(r_pitch) < 0.35
# 4: split held-out in two and check the gate does not separate them
h1, h2 = s_held[: len(s_held) // 2], s_held[len(s_held) // 2:]
sep = abs(h1.mean() - h2.mean()) / max(np.sqrt((h1.var(ddof=1) + h2.var(ddof=1)) / 2), 1e-9)
ok["4 real vs real is a tie"] = abs(sep) < 0.35

print(f"  {'check':<32} {'result':>10}")
for k, v in ok.items():
    print(f"  {k:<32} {'PASS' if v else 'FAIL':>10}")
print()
print(f"  correlation with f0_mean : r = {r_pitch:+.3f}   "
      f"(humans -0.06, DNSMOS -0.79, UTMOSv2 -0.72)")
print(f"  real-vs-real separation  : d = {sep:+.3f}")
print()
if all(ok.values()):
    print("  USABLE. Every check passed, including the pitch trap RESEARCH/06 names.")
else:
    print("  NOT USABLE as it stands. Failing: " +
          ", ".join(k for k, v in ok.items() if not v))
print("=" * 78)
json.dump({"held_out_real": float(np.mean(s_held)),
           "by_group": {lab: float(np.mean(s_syn[[i for i, g in enumerate(G) if g == lab]]))
                        for lab in dict.fromkeys(G)},
           "r_pitch": r_pitch, "real_vs_real_d": float(sep),
           "checks": {k: bool(v) for k, v in ok.items()}},
          io.open(os.path.join(OUT, "validation.json"), "w", encoding="utf-8"), indent=2)
