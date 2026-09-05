"""
E14b — does adding vocal-tract length actually raise the ceiling?

E14 measured the reference: among REAL speakers, distance in five-axis bin
space predicts distance in voice-embedding space at **rho = 0.260**, and the
mapper already transports 90% of that. So the catalog saturates because the
DESCRIPTION cannot distinguish more voices, not because the mapper is losing
information, and the only way to raise the ceiling is to describe more.

Vocal-tract length was then built and validated (females 15.53 cm, males
16.37 cm, all formant signs correct, r = -0.415 against F0 so it is not a
restatement of pitch). This is the test that decides whether it was worth it:

    reference with 5 axes   f0_mean, spectral_tilt, hnr_db, f0_cv, speaking_rate
    reference with 6 axes   the same plus vtl_cm

on the SAME speakers and the SAME embeddings. If rho does not move, VTL is a
correct measurement that carries no additional identity information, and the
representation problem needs a different answer.

HOW THE PAIRING WORKS, and how it is checked. Measuring formants needs the
audio, which the cached corpus does not store -- it stores embeddings and
attributes. But `stream_clips` is deterministic: same corpus, same filters,
same order, no sampling. So the first N clips of a fresh stream are the same
clips, in the same order, as the first N rows of the cache, and the fresh
measurements can be paired with the cached embeddings.

That assumption is load-bearing, so it is VERIFIED rather than assumed: the
re-measured f0_mean is correlated against the cached f0_mean for the same
indices, and the run aborts if they do not match almost exactly. A silent
misalignment here would produce a plausible-looking and completely meaningless
number.

    envs/qwen3/Scripts/python.exe experiments/E14-transport/run_vtl_gain.py
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.acoustics import Attributes, Binner, BIN_LABELS, measure
from alaap.data import stream_clips
from alaap.geometry import SpeakerSpace

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus-cache",
                default="experiments/S2/out/"
                        "corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--n-pairs", type=int, default=80000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

CACHE = os.path.join(OUT, f"vtl_attrs_{args.n}.npz")


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def cos_d(A):
    A = np.asarray(A, float)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return 1.0 - A @ A.T


print("[1/4] cached corpus")
d = np.load(args.corpus_cache, allow_pickle=True)
Z = d["Z"].astype(np.float64)
cached = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
print(f"      {len(Z)} embeddings, {len(cached)} cached attributes")

if os.path.exists(CACHE):
    print(f"[2/4] loading re-measured attributes from {os.path.basename(CACHE)}")
    fresh = [Attributes.from_dict(a)
             for a in json.loads(str(np.load(CACHE, allow_pickle=True)["attrs"]))]
else:
    print(f"[2/4] re-streaming the first {args.n} clips and measuring with formants")
    # S2 built the cached corpus with these bounds. They are NOT the
    # stream_clips defaults (2.0/15.0), and using the defaults selects a
    # different set of clips -- the alignment check below caught exactly that,
    # at r = 0.011 instead of ~1.0.
    clips = stream_clips("globe_v2", n=args.n, per_speaker=1,
                         min_dur=3.0, max_dur=12.0, progress_every=100)
    fresh, t0 = [], time.time()
    for i, c in enumerate(clips):
        fresh.append(measure(c.wav, c.text or "", 24000))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"      {i+1}/{len(clips)} | {el/60:.1f} min | "
                  f"eta {el/(i+1)*(len(clips)-i-1)/60:.1f} min", flush=True)
    np.savez_compressed(CACHE, attrs=json.dumps([a.to_dict() for a in fresh]))

n = min(len(fresh), len(Z), len(cached))
fresh, Zc, cachedc = fresh[:n], Z[:n], cached[:n]

# ---------------------------------------------------- the alignment check
print("[3/4] verifying the fresh clips are the cached clips")
a_new = np.array([a.f0_mean for a in fresh])
a_old = np.array([a.f0_mean for a in cachedc])
r = float(np.corrcoef(a_new, a_old)[0, 1])
rel = float(np.median(np.abs(a_new - a_old) / np.maximum(a_old, 1e-9)))
print(f"      f0_mean fresh vs cached: r = {r:.4f}, median relative diff {rel:.4%}")
if r < 0.99 or rel > 0.02:
    print()
    print("!" * 78)
    print("ABORT: the re-streamed clips are NOT the cached clips.")
    print("Pairing fresh measurements with cached embeddings would produce a")
    print("plausible-looking and meaningless number. Fix the alignment first.")
    print("!" * 78)
    sys.exit(1)
print("      aligned.")

# ----------------------------------------------------------- the question
print("[4/4] does a sixth axis raise the reference?")
vtl_ok = np.isfinite([a.vtl_cm for a in fresh]).mean()
print(f"      vtl_cm estimated for {vtl_ok:.1%} of clips")

space = SpeakerSpace.fit(Zc, n_components=150)
E = space.encode(Zc)
binner = Binner.fit(fresh)

FIVE = ["f0_mean", "spectral_tilt", "hnr_db", "f0_cv", "speaking_rate"]
SIX = FIVE + ["vtl_cm"]
BIDX = {a: {lbl: k for k, lbl in enumerate(BIN_LABELS[a])} for a in SIX}
bins = [binner.bin_one(a) for a in fresh]


def bin_matrix(axes):
    return np.array([[BIDX[a][b[a]] for a in axes] for b in bins], float)


rng = np.random.default_rng(args.seed)
iu, ju = np.triu_indices(n, k=1)
if len(iu) > args.n_pairs:
    sel = rng.choice(len(iu), args.n_pairs, replace=False)
    iu, ju = iu[sel], ju[sel]
d_voice = cos_d(E)[iu, ju]

res = {"n": n, "n_pairs": int(len(iu)), "vtl_coverage": float(vtl_ok)}
for name, axes in (("5 axes", FIVE), ("6 axes (+vtl)", SIX)):
    B = bin_matrix(axes)
    res[name] = spearman(np.abs(B[iu] - B[ju]).sum(1), d_voice)

# and VTL on its own, to see how much it carries alone
Bv = bin_matrix(["vtl_cm"])
res["vtl alone"] = spearman(np.abs(Bv[iu] - Bv[ju]).sum(1), d_voice)
Bf = bin_matrix(["f0_mean"])
res["f0 alone"] = spearman(np.abs(Bf[iu] - Bf[ju]).sum(1), d_voice)

json.dump(res, open(os.path.join(OUT, "results_vtl_gain.json"), "w"), indent=2)

gain = res["6 axes (+vtl)"] - res["5 axes"]
print()
print("=" * 78)
print("E14b — does vocal-tract length raise the description's ceiling?")
print("=" * 78)
print(f"  {n} real speakers | {len(iu):,} pairs | VTL estimated for {vtl_ok:.0%}")
print()
print(f"  {'description':<20} {'rho vs voice distance':>24}")
print(f"  {'-'*20} {'-'*24}")
for k in ("f0 alone", "vtl alone", "5 axes", "6 axes (+vtl)"):
    print(f"  {k:<20} {res[k]:>24.3f}")
print()
print(f"  gain from adding VTL: {gain:+.3f} "
      f"({gain/max(res['5 axes'],1e-9):+.1%} relative)")
print()
if gain > 0.02:
    print("  VTL adds real information. Put it in captions and re-run E11.")
else:
    print("  VTL is a correct measurement that adds little identity information")
    print("  here. Widening the description needs a different axis -- do NOT")
    print("  add it to captions on this evidence.")
print("=" * 78)
