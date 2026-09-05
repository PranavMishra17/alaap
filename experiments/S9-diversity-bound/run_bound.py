"""The ceiling nobody measured: how much diversity do the REAL speakers hold?

Every method in S7/S8 draws from a space fitted on corpus speakers. Whatever
normalised Vendi those speakers achieve is an UPPER BOUND on any catalog built
from them -- minting cannot invent diversity the space does not contain, and
retrieval obviously cannot. Stating the bound before reading a method's score
is the difference between "minting is weak" and "the space is small".
"""
import io, json, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.geometry import SpeakerSpace
from alaap.metrics import vendi_score

CS = ["indicvoices_r_hi", "indicvoices_r_bn", "indicvoices_r_ta"]
allZ, allS = [], []
for c in CS:
    d4 = np.load(f"experiments/S4-indic/out/measured_{c}_250_2.npz", allow_pickle=True)
    metas = json.loads(str(d4["metas"]))
    Z = np.load(f"experiments/S6-indic-mint/out/mio_emb_{c}_250_2.npz",
                allow_pickle=True)["Z"].astype(np.float64)
    k = min(len(Z), len(metas))
    allZ.append(Z[:k]); allS += [f"{c}:{m['speaker_id']}" for m in metas[:k]]
Z = np.vstack(allZ)
print(f"pooled {len(Z)} clips, {len(set(allS))} speakers")

space = SpeakerSpace.fit(Z, n_components=64)
# one point per speaker, so a talkative speaker is not counted twice
byspk = {}
for z, s in zip(space.encode(Z), allS):
    byspk.setdefault(s, []).append(z)
E = np.vstack([np.mean(v, 0) for v in byspk.values()])

print()
print(f"{'set':<34} {'n':>5} {'Vendi':>8} {'effective':>10}")
print("-" * 60)
v = vendi_score(E)
print(f"{'REAL speakers, pooled hi+bn+ta':<34} {len(E):>5} {v:>8.3f} {v*len(E):>10.0f}")
for c in CS:
    idx = [i for i, s in enumerate(byspk) if s.startswith(c)]
    Ec = E[idx]
    vc = vendi_score(Ec)
    print(f"{'REAL speakers, ' + c[14:]:<34} {len(Ec):>5} {vc:>8.3f} {vc*len(Ec):>10.0f}")
print("-" * 60)
print(f"{'S7 minted, pooled':<34} {55:>5} {0.257:>8.3f} {14:>10}")
print(f"{'S7 minted, hi only':<34} {38:>5} {0.362:>8.3f} {14:>10}")
print(f"{'S8 retrieved, hi':<34} {82:>5} {0.405:>8.3f} {33:>10}")


# ---------------------------------------------------------------------------
# WHERE DO MINTED VECTORS ACTUALLY SIT?
#
# Corpus size does not move minting off ~14 (S7: 141 -> 432 speakers, no
# change) and neither does novelty (0.0 -> 0.35 -> 0.70 gives 14, 12, 14).
# Retrieval reaches 33 from the SAME anchors in the SAME space. So the loss is
# in what minting RETURNS, not in what it draws from.
#
# The hypothesis with a shape you can state in advance: minting blends anchors,
# and any blend of points on a shell lands INSIDE it. If that is the mechanism,
# minted vectors will sit measurably nearer the centroid than real speakers do,
# and their pairwise spread will be narrower. Both are one line to check.
print()
print("=" * 66)
print("where minted vectors sit, against the speakers they were built from")
print("=" * 66)
from alaap.acoustics import Attributes, Binner
from alaap.captions import caption_from_bins
from alaap.catalog import sample_cells
from alaap.mapper import RetrievalMapper, TextEncoder

d4 = np.load("experiments/S4-indic/out/measured_indicvoices_r_hi_250_2.npz",
             allow_pickle=True)
A = [Attributes.from_dict(a) for a in json.loads(str(d4["attrs"]))]
Zh = np.load("experiments/S6-indic-mint/out/mio_emb_indicvoices_r_hi_250_2.npz",
             allow_pickle=True)["Z"].astype(np.float64)
k = min(len(A), len(Zh))
A, Zh = A[:k], Zh[:k]
bn = Binner.fit(A)
bn.edges.pop("vtl_cm", None)
bins = [bn.bin_one(a) for a in A]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
sp = SpeakerSpace.fit(Zh, n_components=64)
mp = RetrievalMapper(sp, TextEncoder(), pca_dims=32, retrieval="hybrid").fit(
    caps, Zh, anchor_bins=bins)

E_real = sp.encode(Zh)
cells = sample_cells(80, 0)
for nov in (0.0, 0.7):
    M = np.vstack([sp.encode(mp.mint(caption_from_bins(c, seed=i), novelty=nov,
                                     seed=i).vector)[0]
                   for i, c in enumerate(cells)])
    cen = E_real.mean(0)
    rr = np.linalg.norm(E_real - cen, axis=1)
    rm = np.linalg.norm(M - cen, axis=1)
    print(f"  novelty {nov:.2f}  radius from corpus centroid: "
          f"minted {rm.mean():.3f} vs real {rr.mean():.3f}  "
          f"({rm.mean()/rr.mean():.0%} of real)")
    print(f"              per-dim std ratio minted/real: "
          f"{(M.std(0).mean() / E_real.std(0).mean()):.0%}")
