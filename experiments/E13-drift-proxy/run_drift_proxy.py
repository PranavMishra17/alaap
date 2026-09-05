"""
E13 — can geometry predict drift, so that tuning stops needing the GPU?

E11 rendered two novelty settings and found the trade the geometry sweeps could
not see: raising novelty improves uniqueness and wrecks **drift** -- the
vocoder round-trip, intended vector vs the vector re-extracted from the audio
it produced. A falling drift means the backend is not reproducing the identity
it was handed.

That measurement costs ~2.5 minutes per voice on this hardware, which is why
E12 swept geometry instead and reached a conclusion E11 then overturned. If
drift were predictable from where a vector SITS, tuning would cost seconds
instead of hours, and geometry sweeps could be trusted again because they would
carry a drift estimate with them.

THE HYPOTHESIS. A vector far from the speakers the backend was trained on is a
vector the backend has less reason to render faithfully. So drift should fall
with distance from the corpus.

    extremity    how many of the five described axes sit in an OUTERMOST bin
                 -- E9 asked this on 0.6B; this is the same question with
                 rendered drift and a decorrelated binner
    d_corpus     cosine distance to the nearest real speaker
    knn5         distance to the 5th nearest real speaker (local density)
    norm_z       vector norm, in corpus standard deviations
    novelty      the setting it was minted at -- the known confound

WHAT WOULD MAKE THIS WORTHLESS. The two E11 arms differ in novelty AND in
drift, so ANY quantity that tracks novelty will correlate with drift across the
pooled data without predicting anything. A proxy is only useful if it predicts
drift **within** an arm, where novelty is constant. Both are reported, and the
within-arm number is the one that counts.

Reads the finished artefacts only -- no GPU, no rendering.

    envs/qwen3/Scripts/python.exe experiments/E13-drift-proxy/run_drift_proxy.py
"""
import argparse
import io
import json
import os
import sqlite3
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.geometry import SpeakerSpace
from alaap.metrics import knn_radius

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=[
    "experiments/E11-catalog/out:0.00",
    "experiments/E11-catalog/out_nov0.75:0.75"])
ap.add_argument("--corpus-cache",
                default="experiments/S2/out/"
                        "corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz")
args = ap.parse_args()


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan"), len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


print("[1/3] corpus reference")
d = np.load(args.corpus_cache, allow_pickle=True)
Z = d["Z"].astype(np.float64)
space = SpeakerSpace.fit(Z, n_components=150)
E_corpus = space.encode(Z)
corpus_sd = float(np.linalg.norm(E_corpus, axis=1).std())
corpus_norm = float(np.linalg.norm(E_corpus, axis=1).mean())
print(f"      {len(Z)} speakers | mean norm {corpus_norm:.2f} (sd {corpus_sd:.2f})")

print("[2/3] loading rendered arms")
rows = []
for spec in args.arms:
    path, nov = spec.rsplit(":", 1)
    rj = os.path.join(path, "rows.json")
    db = os.path.join(path, "catalog.db")
    if not (os.path.exists(rj) and os.path.exists(db)):
        print(f"      skip {path} (missing artefacts)")
        continue
    recs = json.load(open(rj))
    uri = "file:" + db.replace("\\", "/").replace(" ", "%20") + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    emb = {r["identity_id"]: np.frombuffer(r["embedding"], dtype=np.float32)
           .astype(np.float64)
           for r in con.execute(
               "SELECT identity_id, embedding FROM identity "
               "WHERE embedding IS NOT NULL")}
    con.close()
    n0 = len(rows)
    for r in recs:
        iid = r.get("identity_id")
        if "error" in r or iid not in emb or r.get("drift") is None:
            continue
        rows.append({"novelty": float(nov), "drift": float(r["drift"]),
                     "consistency": r.get("consistency"),
                     "uniqueness": r.get("uniqueness"),
                     "attempts": r.get("attempts", 1),
                     "cell": r.get("cell") or {},
                     "vec": emb[iid]})
    print(f"      {path}  novelty={nov}  {len(rows)-n0} voices")

if len(rows) < 10:
    print("not enough rendered voices yet"); sys.exit(1)

print("[3/3] measuring")
V = np.vstack([r["vec"] for r in rows])
E = space.encode(V)
C = E @ E_corpus.T
C /= (np.linalg.norm(E, axis=1)[:, None] *
      np.linalg.norm(E_corpus, axis=1)[None, :] + 1e-12)
d_corpus = 1.0 - C.max(1)
knn5 = knn_radius(E, E_corpus, k=5)
norm_z = (np.linalg.norm(E, axis=1) - corpus_norm) / max(corpus_sd, 1e-9)

drift = np.array([r["drift"] for r in rows])
nov = np.array([r["novelty"] for r in rows])
feats = {"d_corpus": d_corpus, "knn5": knn5, "norm_z": norm_z, "novelty": nov}

report = {"n": len(rows), "pooled": {}, "within_arm": {}}
print()
print("=" * 80)
print("E13 — does geometry predict drift?")
print("=" * 80)
print(f"  {len(rows)} rendered voices across {len(set(nov))} novelty settings")
print()
print(f"  POOLED (confounded by novelty -- see the header)")
print(f"    {'feature':<12} {'spearman vs drift':>20}")
for k, v in feats.items():
    r, n = spearman(v, drift)
    report["pooled"][k] = r
    print(f"    {k:<12} {r:>20.3f}")

print()
print(f"  WITHIN ARM (novelty constant -- this is the one that counts)")
print(f"    {'novelty':>8} {'n':>4}  " +
      "  ".join(f"{k:>10}" for k in ("d_corpus", "knn5", "norm_z")))
for u in sorted(set(nov)):
    m = nov == u
    if m.sum() < 8:
        print(f"    {u:>8.2f} {int(m.sum()):>4}  (too few to correlate)")
        continue
    vals = []
    report["within_arm"][str(u)] = {}
    for k in ("d_corpus", "knn5", "norm_z"):
        r, _ = spearman(feats[k][m], drift[m])
        report["within_arm"][str(u)][k] = r
        vals.append(f"{r:>10.3f}")
    print(f"    {u:>8.2f} {int(m.sum()):>4}  " + "  ".join(vals))

print()
print(f"  arm means:")
print(f"    {'novelty':>8} {'n':>4} {'drift':>8} {'d_corpus':>10} {'knn5':>8}")
for u in sorted(set(nov)):
    m = nov == u
    print(f"    {u:>8.2f} {int(m.sum()):>4} {drift[m].mean():>8.3f} "
          f"{d_corpus[m].mean():>10.3f} {knn5[m].mean():>8.3f}")

# ---------------------------------------------------------------- extremity
# The other candidate for what drives drift, and the one E9 asked about on
# 0.6B: how EXTREME the description is. Each caption is a cell in bin space;
# extremity counts how many of its five axes sit in an outermost bin.
from alaap.acoustics import BIN_LABELS
ext = []
for r in rows:
    c = r["cell"]
    ext.append(sum(1 for a, lbl in c.items()
                   if a in BIN_LABELS and lbl in (BIN_LABELS[a][0],
                                                  BIN_LABELS[a][-1])))
ext = np.array(ext, dtype=float)
report["extremity"] = {}
if ext.std() > 0:
    print()
    print("  EXTREMITY -- does asking for an extreme voice cost drift?")
    print("  (how many of the 5 described axes sit in an outermost bin)")
    r_all, _ = spearman(ext, drift)
    report["extremity"]["pooled"] = r_all
    print(f"    pooled            spearman vs drift {r_all:>7.3f}")
    for u in sorted(set(nov)):
        m = nov == u
        if m.sum() >= 8:
            rr, _ = spearman(ext[m], drift[m])
            report["extremity"][str(u)] = rr
            print(f"    novelty {u:<5.2f} n={int(m.sum()):<3} "
                  f"spearman vs drift {rr:>7.3f}")
    print()
    print(f"    {'axes at an extreme':>20} {'n':>4} {'mean drift':>11} "
          f"{'below floor':>12}")
    for e in sorted(set(ext.astype(int))):
        m = ext == e
        if m.sum() >= 3:
            print(f"    {e:>20} {int(m.sum()):>4} {drift[m].mean():>11.3f} "
                  f"{(drift[m] < 0.40).mean():>11.0%}")

json.dump(report, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print()
print("  A pooled correlation alone proves nothing here: the arms differ in")
print("  novelty AND drift, so anything tracking novelty correlates for free.")
print("  Only a within-arm correlation would make a usable GPU-free proxy.")
print("=" * 80)
