"""
E12 — what novelty setting gives the biggest catalog?

E11 is measuring how fast a catalog saturates at `novelty=0.0`, and saturation
showed up almost immediately: mean uniqueness fell 0.803 -> 0.592 inside the
first twenty voices. `novelty` is the one knob that plausibly changes that, and
nothing has swept it against a catalog-sized sample.

E1 measured its two endpoints on the OLD (entangled) axes:

    novelty 0.0   pure retrieval / SLERP     0.20x natural speaker spacing
    novelty 1.0   pure GMM sample            0.89x natural speaker spacing

but S2 run 4 then decorrelated the acoustic axes, and E1's note records that
novelty 1.0 "collapsed to 0.097" once that happened -- so those endpoints
describe a geometry the project no longer uses. The knob is currently set by a
number that has been invalidated.

WHY THIS RUNS ON CPU. Minting a vector is pure linear algebra over cached
corpus embeddings; only RENDERING needs the GPU. So this sweep can run while
E11 occupies the GPU, and it costs nothing but CPU time.

WHAT IT CANNOT TELL YOU -- stated up front, because it is the whole caveat.
Without rendering there is no drift and no consistency, so this measures
GEOMETRY ONLY: how spread out the minted vectors are, not whether they still
render into speech that sounds like the description. Higher novelty pushes
samples further from the retrieval anchors, and far enough out the vectors stop
being renderable -- which is exactly what E1's off-manifold warning is about.

So the output is a CANDIDATE setting, not a decision. The winner has to be
confirmed by an E11-style run that actually renders and measures drift.

SPREAD IS NOT THE TARGET. Every diversity metric below can be maximised by
sampling OFF the manifold -- points far from all real speakers are far from
each other too, so "novel" and "implausible" produce identical numbers. That is
the sparse region RESEARCH/12 measured at +60% relative WER. So the deciding
column is a likelihood comparison, not a distance:

    on_manifold_pct    where the median minted voice's log-likelihood falls
                       within the distribution of REAL speaker log-likelihoods,
                       under an INDEPENDENT reference mixture. 50 = as typical
                       as the median real speaker; 0 = less likely than every
                       real speaker in the corpus.

Also reported:
    vendi (normalised)   fraction of maximum diversity
    nn median / p05      nearest-neighbour cosine distance between minted voices
    frac below floor     share that would trip UNIQUENESS_MIN against each other

`anchor_sim_mean` is recorded but is CONSTANT across novelty by construction --
anchor_similarity is the retrieval similarity of the top anchor, computed
before novelty is applied. It does not measure what novelty costs.

Outputs -> experiments/E12-novelty/out/

    envs/qwen3/Scripts/python.exe experiments/E12-novelty/run_novelty_sweep.py
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
ap.add_argument("--n", type=int, default=300, help="voices per novelty setting")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

NOVELTIES = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
AXES = ["f0_mean", "spectral_tilt", "hnr_db", "f0_cv", "speaking_rate"]


def sample_cells(n: int, seed: int) -> list[dict[str, str]]:
    """Same stratified cover E11 mints from, so the two are comparable."""
    rng = np.random.default_rng(seed)
    cells, seen = [], set()

    def add(idx):
        key = tuple(idx)
        if key in seen:
            return
        seen.add(key)
        cells.append({a: BIN_LABELS[a][i] for a, i in zip(AXES, idx)})

    mid = [2] * len(AXES)
    add(mid)
    for ai in range(len(AXES)):
        for v in (0, 1, 3, 4):
            idx = list(mid); idx[ai] = v; add(idx)
    for _ in range(min(64, n)):
        add([int(rng.choice([0, 4])) for _ in AXES])
    guard = 0
    while len(cells) < n and guard < n * 200:
        guard += 1
        add([int(rng.integers(0, 5)) for _ in AXES])
    return cells[:n]


# ----------------------------------------------------------------- mapper
print(f"[1/3] rebuilding the mapper from {os.path.basename(args.corpus_cache)}")
d = np.load(args.corpus_cache, allow_pickle=True)
Z = d["Z"].astype(np.float64)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
binner = Binner.load(args.binner)
caps = [caption_from_bins(binner.bin_one(a), seed=i) for i, a in enumerate(attrs)]
space = SpeakerSpace.fit(Z, n_components=150)
mapper = RetrievalMapper(space, TextEncoder(), pca_dims=50).fit(caps, Z)
print(f"      {len(caps)} anchors | {space}")

# --------------------------------------------------- the real-speaker baseline
# Every number below is meaningless without something to compare it to. The
# corpus itself is the reference: this is what REAL speaker spacing looks like
# in the same space, on the same metric.
# Independent reference density for the on-manifold test. E12b found that
# every diversity metric here can be maximised by sampling OFF the manifold:
# points far from all real speakers are far from each other too, so "novel"
# and "off-manifold" produce identical spread numbers. Only a likelihood
# comparison separates them. Fit once on the corpus, never on the samples.
from sklearn.mixture import GaussianMixture
E_real = space.encode(Z)
_P_ref = space.to_pca(E_real)[:, :50]
REF = GaussianMixture(24, covariance_type="full", reg_covar=1e-4,
                      random_state=1, max_iter=500).fit(_P_ref)
_ll_real = REF.score_samples(_P_ref)
idx = np.random.default_rng(args.seed).choice(len(E_real),
                                              size=min(args.n, len(E_real)),
                                              replace=False)
nn_real = nn_distances(E_real[idx])
vendi_real = float(vendi_score(E_real[idx]))
print(f"      REAL speakers (n={len(idx)}): vendi {vendi_real:.3f} | "
      f"nn median {np.median(nn_real):.3f}")

cells = sample_cells(args.n, args.seed)
descs = [caption_from_bins(c, seed=i) for i, c in enumerate(cells)]
print(f"[2/3] minting {len(descs)} vectors at each of "
      f"{len(NOVELTIES)} novelty settings (CPU only, no rendering)")

rows, t0 = [], time.time()
for nov in NOVELTIES:
    V, anchors = [], []
    for i, desc in enumerate(descs):
        m = mapper.mint(desc, novelty=nov, seed=i)
        V.append(m.vector)
        anchors.append(m.anchor_similarity)
    E = space.encode(np.vstack(V))
    nn = nn_distances(E)
    vs = float(vendi_score(E))
    rows.append({
        "novelty": nov,
        "vendi_normalised": vs,
        "effective_voices": vs * len(E),
        "vendi_vs_real": vs / max(vendi_real, 1e-9),
        "nn_median": float(np.median(nn)),
        "nn_p05": float(np.percentile(nn, 5)),
        "nn_vs_real": float(np.median(nn) / max(np.median(nn_real), 1e-9)),
        "frac_below_uniqueness_floor": float((nn < UNIQUENESS_MIN).mean()),
        "anchor_sim_mean": float(np.mean(anchors)),
        "on_manifold_pct": float(
            (_ll_real < np.median(REF.score_samples(
                space.to_pca(E)[:, :50]))).mean() * 100.0),
    })
    print(f"      novelty {nov:.2f} | vendi {vs:.3f} | nn {np.median(nn):.3f} "
          f"| on-manifold {rows[-1]['on_manifold_pct']:.0f}% "
          f"| {time.time()-t0:.0f}s", flush=True)

json.dump({"n": args.n, "real_baseline":
           {"vendi_normalised": vendi_real,
            "nn_median": float(np.median(nn_real))},
           "sweep": rows},
          open(os.path.join(OUT, "results.json"), "w"), indent=2)

# -------------------------------------------------------------------- report
print("\n" + "=" * 88)
print("E12 — novelty vs catalog capacity   (GEOMETRY ONLY: nothing was rendered)")
print("=" * 88)
print(f"  real speakers (n={len(idx)}):  vendi {vendi_real:.3f}   "
      f"nn median {np.median(nn_real):.3f}   <- the target to match")
print()
print(f"  {'novelty':>7} {'vendi':>7} {'nn med':>7} {'vs real':>8} "
      f"{'<floor':>7} {'on-manif':>9}")
print(f"  {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*9}")
best = max(rows, key=lambda r: r["vendi_normalised"])
for r in rows:
    star = "  <-- most diverse" if r is best else ""
    print(f"  {r['novelty']:>7.2f} {r['vendi_normalised']:>7.3f} "
          f"{r['nn_median']:>7.3f} {r['nn_vs_real']:>7.2f}x "
          f"{r['frac_below_uniqueness_floor']:>6.1%} "
          f"{r['on_manifold_pct']:>8.0f}%{star}")
print()
print("  'vs real' is the ratio to real speaker spacing in the same space -- "
      "1.00x")
print("  means synthesised voices are as far apart as real people are.")
print("  'on-manif' is where the median minted voice falls in the distribution")
print("  of REAL speaker likelihoods. 50% = as typical as the median real")
print("  speaker; LOW = generating in the tails. Spread alone cannot tell")
print("  'novel' from 'off-manifold' -- this column is what does.")
print()
print("  NOT MEASURED HERE: drift and consistency. A setting that wins on")
print("  geometry can still render badly, which is what E1's off-manifold")
print("  warning was about. Confirm the winner with a rendering run.")
print("=" * 88)
