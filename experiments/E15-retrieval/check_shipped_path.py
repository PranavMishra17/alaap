"""Does the SHIPPED hybrid path reproduce what E15 measured inline?

E15 scored bin retrieval with its own inline code. alaap.mapper implements it
separately. Two implementations of one idea drift, so this runs the real
mapper end to end on the same held-out protocol and checks the gain survives.
"""
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from alaap.acoustics import Attributes, Binner
from alaap.captions import caption_from_bins
from alaap.geometry import SpeakerSpace
from alaap.mapper import TextEncoder, RetrievalMapper

CC = "experiments/S2/out/corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz"
VT = "experiments/E14-transport/out/vtl_attrs_500.npz"

d = np.load(CC, allow_pickle=True)
Z = d["Z"].astype(np.float64)
attrs = [Attributes.from_dict(a)
         for a in json.loads(str(np.load(VT, allow_pickle=True)["attrs"]))]
n = min(len(attrs), len(Z))
attrs, Z = attrs[:n], Z[:n]

rng = np.random.default_rng(0)
perm = rng.permutation(n)
n_test = 150
i_test, i_fit = perm[:n_test], perm[n_test:]

space = SpeakerSpace.fit(Z[i_fit], n_components=150)
binner = Binner.fit([attrs[i] for i in i_fit])
bins = [binner.bin_one(a) for a in attrs]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
text = TextEncoder()

E_test = space.encode(Z[i_test])


def unit(A):
    A = np.asarray(A, float)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


print(f"  {len(i_fit)} anchors, {n_test} held-out speakers, real mapper end to end")
print()
print(f"  {'retrieval':<12} {'cos to true voice':>18} {'mean rank of true':>18}")
print(f"  {'-'*12} {'-'*18} {'-'*18}")

E_fit = space.encode(Z[i_fit])
for mode in ("text", "hybrid"):
    m = RetrievalMapper(space, text, pca_dims=50, retrieval=mode).fit(
        [caps[i] for i in i_fit], Z[i_fit],
        anchor_bins=[bins[i] for i in i_fit] if mode == "hybrid" else None)
    V = np.vstack([m.mint(caps[i], novelty=0.0, seed=k).vector
                   for k, i in enumerate(i_test)])
    E_m = space.encode(V)
    cos_true = float(np.mean(np.sum(unit(E_m) * unit(E_test), axis=1)))
    D = 1.0 - unit(E_m) @ unit(E_fit).T
    tgt = (1.0 - unit(E_test) @ unit(E_fit).T).argmin(1)
    rank = float(np.mean([1 + np.where(np.argsort(D[r]) == tgt[r])[0][0]
                          for r in range(n_test)]))
    print(f"  {mode:<12} {cos_true:>18.4f} {rank:>18.1f}")
    if mode == "hybrid":
        print()
        print("  weights the mapper measured for itself:")
        for a, w in sorted(zip(m.axes, m.axis_weights), key=lambda t: -t[1]):
            print(f"      {a:<16} {w:.2f}")
print()
print(f"  E15 measured, inline: text 0.0958 / rank 129.7, "
      f"bin-weighted 0.1587 / rank 96.3 (chance {len(i_fit)//2})")
