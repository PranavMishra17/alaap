"""
E12 — novelty and prior size, measured against held-out real speakers.

Two questions the catalog's capacity depends on:

    novelty          how far to move from the retrieved anchor
    gmm_components   how finely to model the speaker prior we sample from

and one methodological requirement that cost three attempts to get right:
**a diversity number means nothing without a plausibility number beside it.**
Points far from all real speakers are also far from each other, so "novel" and
"implausible" produce identical spread. Spread alone will always select the
most degenerate sampler available.

THE PROTOCOL, and why it is this shape.

The corpus is split in half. The mapper, its prior, and the PCA space are fitted
on one half; the OTHER half is the reference that plausibility is measured
against. Nothing under test has ever seen a reference speaker.

That split exists because the first two versions of this experiment were wrong:

  1. A likelihood metric -- fit a full-covariance GMM to the reference, compare
     log-likelihoods -- scored REAL held-out speakers at 0-1%, the same as
     Gaussian noise. With ~1,250 points in 50 dimensions it measures proximity
     to its own training set. Every conclusion drawn from it was withdrawn.
  2. Fitting that reference on the SAME speakers the mapper's anchors came from
     made retrieval-heavy settings look good for the wrong reason.

So the metric is now non-parametric (`metrics.isolation_pct`, a k-NN radius),
and it is validated by a control printed at the top of every run: **held-out
real speakers must land near 50%.** They land at 54%. Gaussian noise and
dimension-shuffled speakers both land at 100%.

READING THE OUTPUT
    nn median / vs real   spread between minted voices, against real speaker
                          spacing in the same space
    <floor                share colliding under UNIQUENESS_MIN
    isolation             50 = as typical as a median real speaker.
                          ABOVE 50 = out in the tails, further from the data
                          than real people are.
                          BELOW 50 = crowded into denser regions than real
                          speakers occupy, which is its own failure -- it means
                          the sampler avoids the edges of the population.

STILL GEOMETRY ONLY. Nothing here is rendered, so drift, consistency and
adherence are unmeasured. A setting that wins here can still render badly.

    envs/qwen3/Scripts/python.exe experiments/E12-novelty/run_sweeps.py
"""
import io, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from alaap.acoustics import Attributes, Binner, BIN_LABELS
from alaap.captions import caption_from_bins
from alaap.geometry import SpeakerSpace
from alaap.mapper import TextEncoder, RetrievalMapper
from alaap.metrics import nn_distances, isolation_pct, vendi_score
from alaap.service import UNIQUENESS_MIN

d = np.load("experiments/S2/out/corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz",
            allow_pickle=True)
Z = d["Z"].astype(np.float64)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
binner = Binner.load("experiments/S2/out/binner_globe_v2_Qwen3-TTS-12Hz-17B-Base.json")
caps = [caption_from_bins(binner.bin_one(a), seed=i) for i, a in enumerate(attrs)]

rng = np.random.default_rng(0)
perm = rng.permutation(len(Z)); half = len(Z)//2
i_fit, i_eval = perm[:half], perm[half:]
space = SpeakerSpace.fit(Z[i_fit], n_components=150)
R = space.encode(Z[i_eval])                       # held-out reference speakers
F = space.encode(Z[i_fit])
base_real = isolation_pct(F, R)
nn_real = float(np.median(nn_distances(R[rng.choice(len(R), 300, replace=False)])))
text = TextEncoder()
capsf = [caps[i] for i in i_fit]

AX = ["f0_mean","spectral_tilt","hnr_db","f0_cv","speaking_rate"]
r2 = np.random.default_rng(0); cells, seen = [], set()
def add(i):
    k=tuple(i)
    if k in seen: return
    seen.add(k); cells.append({a: BIN_LABELS[a][j] for a,j in zip(AX,i)})
mid=[2]*5; add(mid)
for ai in range(5):
    for v in (0,1,3,4):
        x=list(mid); x[ai]=v; add(x)
for _ in range(64): add([int(r2.choice([0,4])) for _ in range(5)])
while len(cells)<300: add([int(r2.integers(0,5)) for _ in range(5)])
descs=[caption_from_bins(c, seed=i) for i,c in enumerate(cells[:300])]

print(f"  CONTROL  held-out REAL speakers isolation {base_real:.0f}%  "
      f"(50 = typical)   nn median {nn_real:.3f}")
print()
out = {"control_real_isolation": base_real, "nn_real": nn_real, "novelty": [],
       "components": []}
print("  NOVELTY  (gmm_components=5)")
print(f"  {'novelty':>7} {'nn med':>7} {'vs real':>8} {'<floor':>7} {'isolation':>10}")
print(f"  {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*10}")
m5 = RetrievalMapper(space, text, pca_dims=50, gmm_components=5).fit(capsf, Z[i_fit])
for nov in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
    V = np.vstack([m5.mint(x, novelty=nov, seed=i).vector for i, x in enumerate(descs)])
    E = space.encode(V); nn = nn_distances(E); iso = isolation_pct(E, R)
    out["novelty"].append({"novelty": nov, "nn_median": float(np.median(nn)),
                           "nn_vs_real": float(np.median(nn)/nn_real),
                           "frac_below_floor": float((nn<UNIQUENESS_MIN).mean()),
                           "isolation_pct": iso})
    print(f"  {nov:>7.2f} {np.median(nn):>7.3f} {np.median(nn)/nn_real:>7.2f}x "
          f"{(nn<UNIQUENESS_MIN).mean():>6.1%} {iso:>9.0f}%")

print()
print("  COMPONENTS  (novelty=1.0)")
print(f"  {'k':>7} {'nn med':>7} {'vs real':>8} {'<floor':>7} {'isolation':>10}")
print(f"  {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*10}")
for k in (1, 3, 5, 12, 24, 48):
    mk = RetrievalMapper(space, text, pca_dims=50, gmm_components=k).fit(capsf, Z[i_fit])
    V = np.vstack([mk.mint(x, novelty=1.0, seed=i).vector for i, x in enumerate(descs)])
    E = space.encode(V); nn = nn_distances(E); iso = isolation_pct(E, R)
    out["components"].append({"k": k, "actual": int(mk.gmm.n_components),
                              "nn_median": float(np.median(nn)),
                              "nn_vs_real": float(np.median(nn)/nn_real),
                              "frac_below_floor": float((nn<UNIQUENESS_MIN).mean()),
                              "isolation_pct": iso})
    print(f"  {k:>7} {np.median(nn):>7.3f} {np.median(nn)/nn_real:>7.2f}x "
          f"{(nn<UNIQUENESS_MIN).mean():>6.1%} {iso:>9.0f}%")
json.dump(out, open("experiments/E12-novelty/out/results_heldout.json","w"), indent=2)
