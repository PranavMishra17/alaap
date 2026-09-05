"""
E15 — should retrieval go through a sentence encoder at all?

E14b found that `f0_mean` ALONE predicts voice distance better than all five
axes equally weighted (rho 0.394 vs 0.281). The description is not only too
narrow, it is badly WEIGHTED -- and weighting costs no new measurement, which
makes it the cheapest gain available.

But the mapper never sees the axes. It retrieves by cosine similarity between
MiniLM sentence embeddings of the captions, so the weighting it applies is
whatever MiniLM happens to assign to the words "low-pitched" against "rough"
against "quick". Nothing chose that, and there is no reason it should match how
much each axis actually carries about a voice.

Since captions are GENERATED from bins, and free text can be parsed back to
target bins (`captions.target_bins_from_text`), there is a direct alternative:
retrieve in bin space with weights measured from the corpus.

    text        cosine over MiniLM sentence embeddings          (current)
    bin-equal   L1 in bin space, every axis weighted the same
    bin-weighted L1 in bin space, each axis weighted by how much it predicts
                voice distance -- the weights are MEASURED on the fit half,
                never on the test half

MEASURED HOW. For held-out speakers we know the true voice, so retrieval can be
scored directly, with no rendering and no GPU:

    cos to true voice   does the minted vector resemble the voice whose caption
                        was used? this is the thing retrieval is FOR
    transport rho       does bin distance still predict minted-voice distance
    nn to true rank     where the true speaker ranks among all fit speakers by
                        distance to the minted vector (1 = perfect)

A caveat worth stating before the numbers: bin-space retrieval is being scored
partly on bin-space transport, which flatters it. `cos to true voice` and the
rank are the honest columns -- they are about the voice, not the description.

    envs/qwen3/Scripts/python.exe experiments/E15-retrieval/run_retrieval.py
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
from alaap.acoustics import Attributes, Binner, BIN_LABELS
from alaap.captions import caption_from_bins
from alaap.geometry import SpeakerSpace
from alaap.mapper import TextEncoder, RetrievalMapper

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus-cache",
                default="experiments/S2/out/"
                        "corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz")
ap.add_argument("--vtl-attrs", default="experiments/E14-transport/out/vtl_attrs_500.npz",
                help="re-measured attributes carrying vtl_cm (from E14b); if it "
                     "does not exist the run falls back to the cache's own "
                     "attributes and five axes")
ap.add_argument("--dedup-ids", action="store_true",
                help="keep one clip per speaker (needed for per_speaker>1 caches)")
ap.add_argument("--n-test", type=int, default=150)
ap.add_argument("--top-k", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

AXES6 = ["f0_mean", "spectral_tilt", "hnr_db", "f0_cv", "speaking_rate", "vtl_cm"]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def unit(A):
    A = np.asarray(A, float)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


print("[1/4] corpus")
d = np.load(args.corpus_cache, allow_pickle=True)
Z_all = d["Z"].astype(np.float64)
if os.path.exists(args.vtl_attrs):
    attrs = [Attributes.from_dict(a) for a in
             json.loads(str(np.load(args.vtl_attrs, allow_pickle=True)["attrs"]))]
    print(f"      using E14b's re-measured attributes (six axes)")
else:
    attrs = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
    print(f"      {args.vtl_attrs} absent -- falling back to the cache's own "
          f"attributes (five axes, no vtl_cm)")
n_all = min(len(attrs), len(Z_all))
attrs, Z = attrs[:n_all], Z_all[:n_all]

if args.dedup_ids and "ids" in d.files:
    # a per_speaker>1 cache repeats speakers, and two clips of one speaker sit
    # close in BOTH bin space and voice space -- leaving them in inflates every
    # correlation here.
    ids = [str(x) for x in d["ids"]][:n_all]
    seen, keep = set(), []
    for i, sp in enumerate(ids):
        if sp not in seen:
            seen.add(sp); keep.append(i)
    Z, attrs = Z[keep], [attrs[i] for i in keep]
    n_all = len(keep)
    print(f"      deduplicated to {n_all} distinct speakers")

# vtl_cm is only usable if it was actually measured
AXES6[:] = [a for a in AXES6
            if a != "vtl_cm" or np.isfinite([x.vtl_cm for x in attrs]).mean() > 0.8]
print(f"      {n_all} speakers | axes: {', '.join(AXES6)}")

rng = np.random.default_rng(args.seed)
perm = rng.permutation(n_all)
n_test = min(args.n_test, n_all // 3)
i_test, i_fit = perm[:n_test], perm[n_test:]

space = SpeakerSpace.fit(Z[i_fit], n_components=min(150, len(i_fit) - 1))
binner = Binner.fit([attrs[i] for i in i_fit])
caps = [caption_from_bins(binner.bin_one(a), seed=i) for i, a in enumerate(attrs)]
BIDX = {a: {lbl: k for k, lbl in enumerate(BIN_LABELS[a])} for a in AXES6}
B = np.array([[BIDX[a][binner.bin_one(x)[a]] for a in AXES6] for x in attrs], float)

E_fit, E_test = space.encode(Z[i_fit]), space.encode(Z[i_test])
P_fit = space.to_pca(E_fit)[:, :50]

print("[2/4] measuring per-axis weights on the FIT half only")
iu, ju = np.triu_indices(len(i_fit), k=1)
if len(iu) > 60000:
    sel = rng.choice(len(iu), 60000, replace=False)
    iu, ju = iu[sel], ju[sel]
dv = (1.0 - unit(E_fit) @ unit(E_fit).T)[iu, ju]
W = []
for k, a in enumerate(AXES6):
    b = B[i_fit][:, k]
    W.append(max(spearman(np.abs(b[iu] - b[ju]), dv), 0.0))
W = np.array(W)
W = W / W.sum() * len(W)          # mean weight 1, so scales stay comparable
for a, w in zip(AXES6, W):
    print(f"      {a:<15} weight {w:.2f}")

print("[3/4] retrieving held-out captions three ways")
text = TextEncoder()
T_fit = text.encode([caps[i] for i in i_fit])
T_test = text.encode([caps[i] for i in i_test])


def slerp(a, b, t):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    ua, ub = a / max(na, 1e-12), b / max(nb, 1e-12)
    om = np.arccos(np.clip(ua @ ub, -1, 1))
    if om < 1e-6:
        return (1 - t) * a + t * b
    return (ua * np.sin((1 - t) * om) + ub * np.sin(t * om)) / np.sin(om) * \
        ((1 - t) * na + t * nb)


def retrieve(scores_desc):
    """scores_desc: (n_test, n_fit) similarity, higher = closer."""
    out = np.zeros((len(i_test), P_fit.shape[1]))
    for r in range(len(i_test)):
        order = np.argsort(-scores_desc[r])[:args.top_k]
        w = scores_desc[r][order]
        w = np.exp((w - w.max()) * 8.0); w = w / w.sum()
        cur = P_fit[order[0]].copy()
        for j in range(1, len(order)):
            cur = slerp(cur, P_fit[order[j]], float(w[j] / w[:j + 1].sum()))
        out[r] = cur
    return out


Bf, Bt = B[i_fit], B[i_test]
methods = {
    "text (current)": unit(T_test) @ unit(T_fit).T,
    "bin-equal": -np.abs(Bt[:, None, :] - Bf[None, :, :]).sum(2).astype(float),
    "bin-weighted": -(np.abs(Bt[:, None, :] - Bf[None, :, :]) * W).sum(2),
}

print("[4/4] scoring against the true held-out voices")
res = {"weights": {a: float(w) for a, w in zip(AXES6, W)},
       "n_test": int(n_test), "methods": {}}
iu2, ju2 = np.triu_indices(n_test, k=1)
d_bin_test = (np.abs(Bt[iu2] - Bt[ju2]) * W).sum(1)

for name, sc in methods.items():
    P_out = retrieve(sc)
    Wv = space.from_pca(np.pad(P_out, ((0, 0),
                        (0, space.components.shape[0] - P_out.shape[1]))))
    V = space.decode(Wv, project_to_shell=True)
    E_mint = space.encode(V)
    cos_true = float(np.mean(np.sum(unit(E_mint) * unit(E_test), axis=1)))
    D = 1.0 - unit(E_mint) @ unit(E_fit).T
    # where does the TRUE speaker's nearest fit-neighbour rank? use the true
    # voice's own nearest fit speaker as the target index
    tgt = (1.0 - unit(E_test) @ unit(E_fit).T).argmin(1)
    rank = float(np.mean([1 + np.where(np.argsort(D[r]) == tgt[r])[0][0]
                          for r in range(n_test)]))
    d_mint = (1.0 - unit(E_mint) @ unit(E_mint).T)[iu2, ju2]
    res["methods"][name] = {"cos_to_true": cos_true,
                            "mean_rank_of_true_neighbour": rank,
                            "transport_rho": spearman(d_bin_test, d_mint)}
    print(f"      {name:<16} cos-to-true {cos_true:.4f} | rank {rank:.1f} "
          f"| transport {res['methods'][name]['transport_rho']:.3f}", flush=True)

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)

print()
print("=" * 82)
print("E15 — text-embedding retrieval vs weighted bin retrieval")
print("=" * 82)
print(f"  {n_test} held-out speakers | {len(i_fit)} anchors | top_k={args.top_k}")
print(f"  measured axis weights: " +
      ", ".join(f"{a.split('_')[0]} {w:.2f}" for a, w in zip(AXES6, W)))
print()
print(f"  {'method':<16} {'cos to true voice':>18} {'rank of true':>13} "
      f"{'transport':>10}")
print(f"  {'-'*16} {'-'*18} {'-'*13} {'-'*10}")
for k, v in res["methods"].items():
    print(f"  {k:<16} {v['cos_to_true']:>18.4f} "
          f"{v['mean_rank_of_true_neighbour']:>13.1f} {v['transport_rho']:>10.3f}")
print()
print("  'cos to true voice' and 'rank of true' are the honest columns: they")
print("  score the VOICE. 'transport' is measured in bin space and therefore")
print("  flatters the bin methods -- read it as a consistency check, not a win.")
print(f"  rank 1.0 would be perfect; chance is {len(i_fit)/2:.0f}.")
print("=" * 82)
