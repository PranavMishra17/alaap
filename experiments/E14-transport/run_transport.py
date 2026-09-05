"""
E14 — how much of a description survives into the voice?

E11 found the catalog saturating at forty voices. E12 found the knobs cannot
fix it without breaking drift. Neither asked the question underneath both:

    when two descriptions differ, do the voices they mint differ CORRESPONDINGLY?

That is the premise of the two-tower design. If the mapper compresses -- a
large change in the description producing a small change in the voice -- then
saturation is the mapper discarding information, and no sampling knob recovers
it.

MEASURING THE DESCRIPTION SIDE. The obvious independent variable, distance
between sentence-embeddings of the two captions, is a bad one: it measures the
text encoder as much as the description, and two captions differing in one bin
can sit arbitrarily close in MiniLM space. So difference is measured in BIN
space instead -- the L1 distance between the two captions' bin indices, which
is exactly the acoustic difference the caption was built to express and is
independent of any encoder.

THE REFERENCE, and what it is not. The corpus is 2,500 real (caption, voice)
pairs, captions generated from the audio itself. So the same correlation is
computable on ground truth: among REAL speakers, how well does a bin-space
difference track a voice-embedding difference?

That number is a REFERENCE, not a ceiling. A mapper can exceed it, and does,
because it constructs voices deterministically from captions while reality
does not -- two real speakers with identical bins are still different people.
The reference says how much of voice identity these five axes explain AT ALL;
the mapper's number says how much structure the mapper imposes. Reading the
reference as an upper bound would be wrong.

    reference   bin distance vs REAL voice distance    (held-out corpus pairs)
    actual      bin distance vs MINTED voice distance  (same captions, minted)

A third column guards a specific failure: if the mapper mostly returns its
nearest anchor, "actual" looks good while the mapper is doing retrieval rather
than mapping.

CPU only, no rendering.

    envs/qwen3/Scripts/python.exe experiments/E14-transport/run_transport.py
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.acoustics import Attributes, Binner
from alaap.captions import caption_from_bins
from alaap.geometry import SpeakerSpace
from alaap.mapper import TextEncoder, RetrievalMapper

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus-cache",
                default="experiments/S2/out/"
                        "corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz")
ap.add_argument("--binner", default="experiments/S2/out/"
                                    "binner_globe_v2_Qwen3-TTS-12Hz-17B-Base.json")
ap.add_argument("--n-test", type=int, default=400, help="held-out captions to mint")
ap.add_argument("--n-pairs", type=int, default=60000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def cos_d(A, B=None):
    A = np.asarray(A, float)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = A if B is None else (np.asarray(B, float) /
                             (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12))
    return 1.0 - A @ B.T


print("[1/4] corpus and captions")
d = np.load(args.corpus_cache, allow_pickle=True)
Z = d["Z"].astype(np.float64)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
binner = Binner.load(args.binner)
caps = [caption_from_bins(binner.bin_one(a), seed=i) for i, a in enumerate(attrs)]

# Disjoint split: the mapper is fitted on one half and tested on captions from
# the other. Minting from a caption the mapper was fitted on measures memory.
rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(Z))
half = len(Z) // 2
i_fit, i_test = perm[:half], perm[half:half + args.n_test]

space = SpeakerSpace.fit(Z[i_fit], n_components=150)
text = TextEncoder()
# S9b/S10: pca_dims=50 of 150 zero-pads the discarded components, so every
# minted voice is near-identical there -- it cost 11%-vs-20% of the available
# diversity in E11. Use the full basis.
mapper = RetrievalMapper(space, text, pca_dims=space.components.shape[0]).fit(
    [caps[i] for i in i_fit], Z[i_fit])
print(f"      fitted on {len(i_fit)} pairs | testing {len(i_test)} held-out captions")

print("[2/4] the reference: bin distance vs REAL voice distance")
E_true = space.encode(Z[i_test])

# bin-space coordinates: index of each caption's bin on each axis
from alaap.acoustics import BIN_LABELS
AXES = [a for a in BIN_LABELS]
BIDX = {a: {lbl: k for k, lbl in enumerate(BIN_LABELS[a])} for a in AXES}
B = np.array([[BIDX[a][binner.bin_one(attrs[i])[a]] for a in AXES]
              for i in i_test], dtype=np.float64)

n = len(i_test)
iu, ju = np.triu_indices(n, k=1)
if len(iu) > args.n_pairs:
    sel = rng.choice(len(iu), args.n_pairs, replace=False)
    iu, ju = iu[sel], ju[sel]
d_bin = np.abs(B[iu] - B[ju]).sum(1)          # L1 in bin space
d_true = cos_d(E_true)[iu, ju]
ceiling = spearman(d_bin, d_true)
print(f"      {len(iu):,} pairs | reference rho = {ceiling:.3f} "
      f"| bin distance range {d_bin.min():.0f}-{d_bin.max():.0f}")

print("[3/4] minting the same captions")
res = {"reference": ceiling, "n_test": int(n), "n_pairs": int(len(iu)),
       "settings": []}
for nov in (0.0, 0.45, 0.75, 1.0):
    V = np.vstack([mapper.mint(caps[i], novelty=nov, seed=k).vector
                   for k, i in enumerate(i_test)])
    E_mint = space.encode(V)
    d_mint = cos_d(E_mint)[iu, ju]
    actual = spearman(d_bin, d_mint)
    # is it just returning its nearest anchor?
    D = cos_d(E_mint, space.encode(Z[i_fit]))
    nearest_is_anchor = float(np.mean(
        [caps[i_fit[j]] == caps[i] for j, i in zip(D.argmin(1), i_test)]))
    # and how much does the minted voice resemble the TRUE voice for its caption?
    per_item = float(np.mean([1.0 - cos_d(E_mint[[k]], E_true[[k]])[0, 0]
                              for k in range(n)]))
    res["settings"].append({
        "novelty": nov, "actual": actual, "ratio": actual / max(ceiling, 1e-9),
        "nearest_is_source_caption": nearest_is_anchor,
        "mean_cos_to_true_voice": per_item})
    print(f"      novelty {nov:.2f} | rho {actual:.3f} | "
          f"ratio {actual/max(ceiling,1e-9):.2f} | "
          f"cos-to-true {per_item:.3f}", flush=True)

print("[4/4] writing")
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)

print()
print("=" * 84)
print("E14 — how much description structure survives the mapper?")
print("=" * 84)
print(f"  REFERENCE  bin distance vs REAL voice distance, {len(iu):,} pairs")
print(f"             rho = {ceiling:.3f}   <- how much of voice identity these")
print(f"                                  five axes explain at all. NOT a ceiling:")
print(f"                                  a mapper can exceed it by being more")
print(f"                                  deterministic than reality is.")
print()
print(f"  {'novelty':>7} {'rho':>7} {'vs reference':>13} {'cos to true voice':>18} "
      f"{'echoes source':>14}")
print(f"  {'-'*7} {'-'*7} {'-'*13} {'-'*18} {'-'*14}")
for r in res["settings"]:
    print(f"  {r['novelty']:>7.2f} {r['actual']:>7.3f} {r['ratio']:>12.2f}x "
          f"{r['mean_cos_to_true_voice']:>18.3f} "
          f"{r['nearest_is_source_caption']:>13.1%}")
print()
print("  'vs reference' compares the structure the mapper imposes against the")
print("  structure that actually exists between real speakers. Above 1.00x means")
print("  the mapper is MORE deterministic than reality -- which it must be, since")
print("  it is a function -- not that it is doing better than possible.")
print("  'echoes source' is how often a minted voice's nearest real speaker is")
print("  the very speaker its caption came from -- high means retrieval, not")
print("  mapping.")
print("=" * 84)
