"""
E12b — how many GMM components does the speaker prior actually want?

E12 fixed the generative branch and found novelty now helps monotonically, but
even at its best the minted voices sit at **0.75x real speaker spacing**. That
ceiling is set by the prior the generative branch samples from, and that prior
has never been sized: `gmm_components` has been 5 since the mapper was written,
capped further by `min(gmm_components, len(P) // 10)`.

Five full-covariance Gaussians over a 50-dimensional PCA space is a very coarse
description of 2,500 speakers. If the speaker manifold has more structure than
that, the sampler is drawing from a blurred version of it, and every generated
voice inherits the blur.

This sweeps the component count at fixed novelty. CPU only -- nothing rendered.

WHAT WOULD BE A TRAP. More components always fits the training data better, and
past some point the mixture starts modelling individual speakers rather than
the distribution they came from. A sampler that reproduces its anchors is not
generating novel voices, it is retrieving them with extra steps -- and it would
score WELL on diversity while doing it. So this reports two things alongside
the diversity numbers:

    bic                per-sample BIC, the standard fit criterion
    nn_to_corpus       distance from each minted voice to the nearest REAL
                       speaker in the corpus

`nn_to_corpus` is the memorisation detector. If it falls as components rise,
the mixture is memorising: the "new" voices are landing on top of real ones.
Real speakers sit some distance from each OTHER, so a healthy sampler should
keep its outputs about that far from the corpus too, not closer.

THE OPPOSITE TRAP, which nn_to_corpus cannot see. A prior that is too COARSE
fails the other way. One Gaussian fitted to a bimodal speaker distribution puts
its mass in the middle -- between the modes, where few real speakers live. Such
samples are far from every real speaker and far from each other, so they score
WELL on both spread and nn_to_corpus while sitting in exactly the sparse region
RESEARCH/12 measured at +60% relative WER. "Far from the data" reads identically
to "novel" and to "off-manifold".

    on_manifold_pct    where minted voices' log-likelihood falls within the
                       distribution of REAL speakers' log-likelihood

scored under an independent REFERENCE density (a separate mixture fit on the
corpus, never the one being swept -- scoring samples under the model that
generated them is circular). 50 means the median minted voice is as typical as
the median real speaker. A low number means the sampler is generating in the
tails, whatever its spread says.

    envs/qwen3/Scripts/python.exe experiments/E12-novelty/run_components_sweep.py
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
from alaap.acoustics import Attributes, Binner, BIN_LABELS
from alaap.captions import caption_from_bins
from alaap.geometry import SpeakerSpace
from alaap.mapper import TextEncoder, RetrievalMapper
from alaap.metrics import vendi_score, nn_distances
from alaap.service import UNIQUENESS_MIN

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus-cache",
                default="experiments/S2/out/"
                        "corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz")
ap.add_argument("--binner", default="experiments/S2/out/"
                                    "binner_globe_v2_Qwen3-TTS-12Hz-17B-Base.json")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--novelty", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

COMPONENTS = [1, 3, 5, 8, 12, 20, 32, 48]
AXES = ["f0_mean", "spectral_tilt", "hnr_db", "f0_cv", "speaking_rate"]


def sample_cells(n, seed):
    rng = np.random.default_rng(seed)
    cells, seen = [], set()

    def add(idx):
        k = tuple(idx)
        if k in seen:
            return
        seen.add(k)
        cells.append({a: BIN_LABELS[a][i] for a, i in zip(AXES, idx)})

    mid = [2] * len(AXES)
    add(mid)
    for ai in range(len(AXES)):
        for v in (0, 1, 3, 4):
            idx = list(mid); idx[ai] = v; add(idx)
    for _ in range(min(64, n)):
        add([int(rng.choice([0, 4])) for _ in AXES])
    g = 0
    while len(cells) < n and g < n * 200:
        g += 1
        add([int(rng.integers(0, 5)) for _ in AXES])
    return cells[:n]


print(f"[1/2] loading {os.path.basename(args.corpus_cache)}")
d = np.load(args.corpus_cache, allow_pickle=True)
Z = d["Z"].astype(np.float64)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
binner = Binner.load(args.binner)
caps = [caption_from_bins(binner.bin_one(a), seed=i) for i, a in enumerate(attrs)]
space = SpeakerSpace.fit(Z, n_components=150)
E_corpus = space.encode(Z)

# Independent reference density for the on-manifold test. Fit ONCE, on the
# corpus, and never swept -- scoring samples under the mixture that produced
# them would be circular.
from sklearn.mixture import GaussianMixture
_P_ref = space.to_pca(space.encode(Z))[:, :50]
REF = GaussianMixture(24, covariance_type="full", reg_covar=1e-4,
                      random_state=1, max_iter=500).fit(_P_ref)
_ll_real = REF.score_samples(_P_ref)

rng0 = np.random.default_rng(args.seed)
idx = rng0.choice(len(E_corpus), size=min(args.n, len(E_corpus)), replace=False)
nn_real = nn_distances(E_corpus[idx])
vendi_real = float(vendi_score(E_corpus[idx]))
print(f"      REAL speakers (n={len(idx)}): vendi {vendi_real:.3f} | "
      f"nn median {np.median(nn_real):.3f}")

text = TextEncoder()          # built once; refitting it per arm would be waste
cells = sample_cells(args.n, args.seed)
descs = [caption_from_bins(c, seed=i) for i, c in enumerate(cells)]

print(f"[2/2] sweeping gmm_components at novelty={args.novelty} "
      f"({len(COMPONENTS)} settings x {args.n} mints)")
rows, t0 = [], time.time()
for nc in COMPONENTS:
    m = RetrievalMapper(space, text, pca_dims=50, gmm_components=nc).fit(caps, Z)
    actual = m.gmm.n_components          # capped by min(nc, len(P)//10)
    V = np.vstack([m.mint(desc, novelty=args.novelty, seed=i).vector
                   for i, desc in enumerate(descs)])
    E = space.encode(V)
    nn = nn_distances(E)
    # distance from each minted voice to the nearest REAL speaker
    C = E @ E_corpus.T
    C /= (np.linalg.norm(E, axis=1)[:, None] *
          np.linalg.norm(E_corpus, axis=1)[None, :] + 1e-12)
    nn_corpus = 1.0 - C.max(1)
    rows.append({
        "requested_components": nc, "actual_components": int(actual),
        "bic_per_sample": float(m.gmm.bic(m.P) / len(m.P)),
        "vendi_normalised": float(vendi_score(E)),
        "vendi_vs_real": float(vendi_score(E)) / max(vendi_real, 1e-9),
        "nn_median": float(np.median(nn)),
        "nn_vs_real": float(np.median(nn) / max(np.median(nn_real), 1e-9)),
        "frac_below_uniqueness_floor": float((nn < UNIQUENESS_MIN).mean()),
        "nn_to_corpus_median": float(np.median(nn_corpus)),
        "on_manifold_pct": float(
            (_ll_real < np.median(REF.score_samples(
                space.to_pca(space.encode(V))[:, :50]))).mean() * 100.0),
    })
    print(f"      k={nc:>2} (actual {actual:>2}) | vendi {rows[-1]['vendi_normalised']:.3f} "
          f"| nn {rows[-1]['nn_median']:.3f} | to-corpus "
          f"{rows[-1]['nn_to_corpus_median']:.3f} | on-manifold "
          f"{rows[-1]['on_manifold_pct']:.0f}% | {time.time()-t0:.0f}s", flush=True)

json.dump({"n": args.n, "novelty": args.novelty,
           "real_baseline": {"vendi_normalised": vendi_real,
                             "nn_median": float(np.median(nn_real))},
           "sweep": rows},
          open(os.path.join(OUT, "results_components.json"), "w"), indent=2)

print("\n" + "=" * 92)
print(f"E12b — GMM components at novelty={args.novelty}   "
      f"(GEOMETRY ONLY: nothing rendered)")
print("=" * 92)
print(f"  real speakers: vendi {vendi_real:.3f}  nn median {np.median(nn_real):.3f}"
      f"   <- healthy 'to-corpus' should sit near this too")
print()
print(f"  {'k':>3} {'used':>5} {'BIC/n':>10} {'vendi':>7} {'nn med':>7} "
      f"{'vs real':>8} {'<floor':>7} {'to-corpus':>10} {'on-manif':>9}")
print(f"  {'-'*3} {'-'*5} {'-'*10} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*10} {'-'*9}")
best = max(rows, key=lambda r: r["nn_median"])
for r in rows:
    star = "  <-- most spread" if r is best else ""
    print(f"  {r['requested_components']:>3} {r['actual_components']:>5} "
          f"{r['bic_per_sample']:>10.1f} {r['vendi_normalised']:>7.3f} "
          f"{r['nn_median']:>7.3f} {r['nn_vs_real']:>7.2f}x "
          f"{r['frac_below_uniqueness_floor']:>6.1%} "
          f"{r['nn_to_corpus_median']:>10.3f} {r['on_manifold_pct']:>8.0f}%{star}")
print()
print("  'to-corpus' is distance to the nearest REAL speaker. If it FALLS as k")
print("  rises, the mixture is MEMORISING -- 'new' voices landing on real people.")
print()
print("  'on-manif' is where the median minted voice sits in the distribution of")
print("  REAL speaker likelihoods, under an independent reference mixture. 50%")
print("  means as typical as the median real speaker. A LOW number means the")
print("  sampler is generating in the tails -- far from the data reads the same")
print("  as 'novel' and as 'off-manifold', and only this column tells them apart.")
print("=" * 92)
