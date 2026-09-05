"""
E1 — Is the Qwen3-TTS speaker space navigable by synthesis?

THE decisive experiment (RESEARCH/12, scope question A2). If a generative
prior fitted to real embeddings samples vectors that land off-manifold, the
two-tower design cannot work and the project pivots to Tier 2 wholesale.

--------------------------------------------------------------------------
METRIC DESIGN NOTE — read this before trusting any number below.

Run 1 of this experiment used TacoSpawn's s2s/g2s protocol literally:
MEDIAN PAIRWISE cosine distance among real vs generated. It reported every
generator at g2s/s2s = 0.97-1.00 and declared a PASS -- including a naive
per-dimension Gaussian, which is exactly the generator TacoSpawn showed to
be off-manifold.

That was the metric failing, not the generators succeeding. After centring
and PCA whitening, every pair of vectors is near-orthogonal, so median
pairwise cosine distance saturates at ~1.0 for ANY generator. The metric
had no discriminative power in this space.

The discriminating quantity is NEAREST-NEIGHBOUR distance to the real
manifold, benchmarked against how far a HELD-OUT REAL speaker sits from the
fit set. That gives a natural unit:

    nn_ratio = median nn_dist(generated -> fit)
             / median nn_dist(held-out real -> fit)

    nn_ratio ~ 1.0   indistinguishable from a genuinely new real speaker
    nn_ratio >> 1.0  off-manifold -- the TacoSpawn failure
    nn_ratio ~ 0.0   memorising the fit set; no novelty

Backed by two independent checks the original protocol lacked:
  * a BALANCED, STRATIFIED real-vs-generated discriminator (run 1 omitted
    both and produced sub-chance AUCs, which is a bug signature)
  * RBF-kernel MMD, a distribution-level two-sample statistic
--------------------------------------------------------------------------

PIVOT CONDITION (GRAND-PLAN section 6.1): nn_ratio > 1.5 -> Tier 2 wholesale.

Outputs -> experiments/E1/out/
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.data import stream_clips
from alaap.encoder import SpeakerEncoder, DEFAULT_MODEL
from alaap.geometry import SpeakerSpace

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--per-speaker", type=int, default=4)
ap.add_argument("--corpus", default="libritts_r_train")
ap.add_argument("--n-gen", type=int, default=500)
ap.add_argument("--pca", type=int, default=50)
args = ap.parse_args()
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ 1. data
print(f"[1/5] {args.n} clips from {args.corpus} (<= {args.per_speaker}/speaker)")
cache_f = os.path.join(OUT, "real_embeddings.npz")
if os.path.exists(cache_f):
    d = np.load(cache_f); Z = d["Z"].astype(np.float64); ids = list(d["ids"])
    print(f"      cached: Z{Z.shape}, {len(set(ids))} speakers")
else:
    clips = stream_clips(corpus=args.corpus, n=args.n, per_speaker=args.per_speaker)
    ids = [c.speaker_id for c in clips]
    enc = SpeakerEncoder(args.model)
    Z = enc.embed_many([c.wav for c in clips]).astype(np.float64)
    np.savez_compressed(cache_f, Z=Z.astype(np.float32), ids=np.array(ids))
    del enc
    import torch; torch.cuda.empty_cache()
    print(f"      Z{Z.shape}, {len(set(ids))} speakers")

uniq = sorted(set(ids)); idsa = np.asarray(ids)
S = np.stack([Z[idsa == s].mean(0) for s in uniq])   # one vector per speaker
print(f"      speaker centroids: {S.shape}")

# ---------------------------------------------------- 2. split + fit space
n_fit = len(S) // 2
perm = rng.permutation(len(S))
S_fit, S_held = S[perm[:n_fit]], S[perm[n_fit:]]
print(f"[2/5] fit on {len(S_fit)} speakers, hold out {len(S_held)} (disjoint)")

space = SpeakerSpace.fit(S_fit, n_components=min(args.pca * 3, len(S_fit) - 1))
print(f"      {space}")
k = min(args.pca, space.components.shape[0])
P_fit = space.to_pca(space.encode(S_fit))[:, :k]
P_held = space.to_pca(space.encode(S_held))[:, :k]
print(f"      PCA k={k} | {space.n_components_for(0.90)} dims for 90% var")

# ------------------------------------------------------------- 3. generators
print(f"[3/5] fitting generators, {args.n_gen} samples each")


def gen_independent(n):
    """Naive per-dim independent Gaussian. TacoSpawn's documented failure."""
    return rng.normal(P_fit.mean(0), P_fit.std(0), size=(n, P_fit.shape[1]))


def gen_full_gauss(n):
    mu = P_fit.mean(0); C = np.cov(P_fit.T) + 1e-6 * np.eye(P_fit.shape[1])
    return rng.multivariate_normal(mu, C, size=n)


def make_gmm(ncomp):
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(ncomp, covariance_type="full", reg_covar=1e-4,
                        random_state=0, max_iter=500).fit(P_fit)
    return lambda n: g.sample(n)[0]


def gen_slerp(n):
    """SLERP between nearest-neighbour anchors -- the PHASE-02 retrieval plan."""
    Wn = P_fit / np.maximum(np.linalg.norm(P_fit, axis=1, keepdims=True), 1e-12)
    C = Wn @ Wn.T; np.fill_diagonal(C, -9)
    out = []
    for _ in range(n):
        i = rng.integers(len(P_fit))
        nbrs = np.argsort(-C[i])[:5]
        j = nbrs[rng.integers(len(nbrs))]
        t = rng.uniform(0.25, 0.75)
        a, b = P_fit[i], P_fit[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        ua, ub = a / na, b / nb
        om = np.arccos(np.clip(ua @ ub, -1, 1))
        v = ((ua * np.sin((1 - t) * om) + ub * np.sin(t * om)) / np.sin(om)
             if om > 1e-6 else ua)
        out.append(v * ((1 - t) * na + t * nb))
    return np.stack(out)


def gen_resample(n):
    """Bootstrap the fit set. Sanity floor: nn_ratio must come out ~0."""
    return P_fit[rng.integers(0, len(P_fit), n)]


generators = {
    "independent gauss": gen_independent,
    "full-cov gauss":    gen_full_gauss,
    "GMM k=5":           make_gmm(5),
    "GMM k=10":          make_gmm(10),
    "GMM k=20":          make_gmm(20),
    "SLERP (retrieval)": gen_slerp,
    "resample real":     gen_resample,
}

# -------------------------------------------------------------- 4. evaluate
def cosdist(A, B):
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return 1.0 - An @ Bn.T


def nn_to(A, B):
    return float(np.median(cosdist(A, B).min(1)))


def median_pairwise(A):
    D = cosdist(A, A); iu = np.triu_indices(len(A), 1)
    return float(np.median(D[iu]))


def mmd_rbf(X, Y):
    Z = np.vstack([X, Y])[:400]
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    gamma = 1.0 / max(np.median(d2[d2 > 0]), 1e-9)

    def kk(A, B):
        d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d, 0))
    return float(kk(X, X).mean() + kk(Y, Y).mean() - 2 * kk(X, Y).mean())


def discriminator_auc(real, gen, seed=0):
    """Balanced classes, stratified folds, scaled features. 0.5 = indistinguishable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    r = np.random.default_rng(seed)
    m = min(len(real), len(gen))
    real = real[r.choice(len(real), m, replace=False)]
    gen = gen[r.choice(len(gen), m, replace=False)]
    X = np.vstack([real, gen]); y = np.r_[np.zeros(m), np.ones(m)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


real_nn = nn_to(P_held, P_fit)
print("[4/5] evaluating")
print(f"      REFERENCE: a new REAL speaker sits {real_nn:.4f} from the fit set")
print(f"      (a generator matching that is indistinguishable from a new real speaker)")

results = {}
for name, fn in generators.items():
    G = np.asarray(fn(args.n_gen), dtype=np.float64)
    nn = nn_to(G, P_fit)
    ratio = nn / max(real_nn, 1e-9)
    auc = discriminator_auc(P_held, G)
    results[name] = {
        "real_nn_reference": real_nn, "nn_dist_to_real": nn, "nn_ratio": ratio,
        "g2g_median": median_pairwise(G), "discriminator_auc": auc,
        "mmd": mmd_rbf(P_held, G),
        "norm_ratio": float(np.linalg.norm(G, axis=1).mean() /
                            np.linalg.norm(P_held, axis=1).mean()),
        "n_gen": int(len(G))}
    ok = (0.5 <= ratio <= 1.5) and auc < 0.75
    print(f"      {name:<20} nn={nn:.4f} ({ratio:4.2f}x)  AUC={auc:.3f}  "
          f"MMD={results[name]['mmd']:+.4f}  |z|={results[name]['norm_ratio']:.2f}  "
          f"[{'PASS' if ok else 'FAIL'}]")

# ------------------------------------------------------------- 5. report
def _score(kk):
    if kk == "resample real":
        return 9e9                      # memorises the fit set; not a generator
    r = results[kk]
    return abs(r["nn_ratio"] - 1.0) + abs(r["discriminator_auc"] - 0.5)


best = min(results, key=_score)
verdict = (0.5 <= results[best]["nn_ratio"] <= 1.5) and \
          results[best]["discriminator_auc"] < 0.75

summary = {"model": args.model, "corpus": args.corpus, "n_speakers": len(uniq),
           "n_fit": len(S_fit), "n_held": len(S_held), "pca_dims": int(k),
           "n_gen": args.n_gen, "real_nn_reference": real_nn,
           "space_stats": space.stats.__dict__, "by_generator": results,
           "best_generator": best, "PIVOT_THRESHOLD_nn_ratio": 1.5,
           "VERDICT_two_tower_survives": bool(verdict)}
json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
names = list(results); x = np.arange(len(names))
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(x, [results[n]["nn_ratio"] for n in names], .55, color="tab:blue")
ax[0].axhline(1.0, color="g", ls="--", label="= a new real speaker")
ax[0].axhline(1.5, color="r", ls="--", label="pivot threshold")
ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax[0].set_ylabel("nn(gen->fit) / nn(new real->fit)")
ax[0].set_title("On-manifold? 1.0 = as close to real as a new real speaker")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].bar(x, [results[n]["discriminator_auc"] for n in names], .55, color="tab:purple")
ax[1].axhline(0.5, color="g", ls="--", label="indistinguishable")
ax[1].axhline(0.75, color="r", ls="--", label="detectably synthetic")
ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax[1].set_ylim(0.35, 1.02); ax[1].set_ylabel("real-vs-generated AUC")
ax[1].set_title("Detectably synthetic?"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.suptitle(f"E1 — is the Qwen3-TTS speaker space navigable by synthesis?  "
             f"PCA-{k}, {len(S_fit)} fit / {len(S_held)} held-out speakers")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "e1_navigability.png"), dpi=140)

print()
print("=" * 92)
print("E1 RESULT  —  A2, the decisive question")
print("=" * 92)
print(f"  {len(uniq)} speakers | PCA-{k} | {len(S_fit)} fit / {len(S_held)} held out")
print(f"  REFERENCE: a new REAL speaker sits {real_nn:.4f} from the fit set")
print()
print(f"  {'generator':<20} {'nn_dist':>9} {'nn_ratio':>9} {'AUC':>7} {'MMD':>9} {'|z|':>6}")
print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*7} {'-'*9} {'-'*6}")
for n, r in results.items():
    print(f"  {n:<20} {r['nn_dist_to_real']:>9.4f} {r['nn_ratio']:>9.2f} "
          f"{r['discriminator_auc']:>7.3f} {r['mmd']:>+9.4f} {r['norm_ratio']:>6.2f}")
print()
print("-" * 92)
if verdict:
    print("  VERDICT: the space IS navigable by synthesis. Two-tower design SURVIVES.")
    print(f"           Best generator '{best}': nn_ratio={results[best]['nn_ratio']:.2f} "
          f"(target 1.0, pivot >1.5), AUC={results[best]['discriminator_auc']:.3f}")
else:
    print("  VERDICT: generated vectors land OFF-MANIFOLD. PIVOT TO TIER 2.")
print("-" * 92)
print(f"  artefacts -> {OUT}")
print("=" * 92)
