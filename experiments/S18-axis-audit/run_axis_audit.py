"""
S18 — audit every measured axis for whether it identifies anybody

The same one-line check has now found three things:

    speaking_rate   within/between 1.32 on real speakers -> not an identity axis (S16)
    f0_cv           1.04                                 -> not an identity axis (S16)
    vtl_cm          at chance, and the ESTIMATOR was broken -- fixing it gave
                    +0.70/+1.03/+0.44/+0.68 gender separation and a fourth
                    real axis (S17)

Three for three, on axes that were picked one at a time because something else
drew attention to them. This runs it over EVERY measured attribute at once,
because the base rate says there are probably more.

THE CHECK. For each axis, using two clips of the same speaker:

    within  = mean over speakers of the variance between that speaker's takes
    between = variance of the speakers' own means
    ratio   = within / between

    ratio < 1   the axis separates people      -> identity candidate
    ratio > 1   one person varies on it more than people differ  -> not identity

PASSING THAT IS NOT ENOUGH. An axis can separate speakers and still add
nothing, by restating one that is already in the set -- `f0_std` is largely
`f0_mean` in disguise, since a higher-pitched voice has more absolute room to
move. So the second screen is REDUNDANCY: the maximum absolute correlation with
the axes already used for identity. A candidate must clear both.

    envs/qwen3/Scripts/python.exe experiments/S18-axis-audit/run_axis_audit.py
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
from alaap.acoustics import BIN_LABELS, RECORDING_AXES, Attributes
from alaap.data import norm_gender

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--corpora", default="indicvoices_r_hi,indicvoices_r_bn,indicvoices_r_ta")
args = ap.parse_args()

CURRENT_IDENTITY = ["f0_mean", "spectral_tilt", "hnr_db", "vtl_cm"]
KNOWN_DEAD = ["f0_cv", "speaking_rate"]
# every numeric field a measurement produces, not just the ones captions use
CANDIDATES = ["f0_mean", "f0_std", "f0_range", "f0_cv", "speaking_rate", "hnr_db",
              "spectral_tilt", "jitter", "shimmer", "snr_db", "voiced_frac",
              "duration_s", "f1", "f2", "f3", "formant_dispersion", "vtl_cm"]

data = {}
for c in [x.strip() for x in args.corpora.split(",") if x.strip()]:
    p = f"experiments/S4-indic/out/measured_{c}_250_2.npz"
    if not os.path.exists(p):
        print(f"  missing {p}, skipping {c}")
        continue
    d = np.load(p, allow_pickle=True)
    attrs = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
    metas = json.loads(str(d["metas"]))
    by = {}
    for a, m in zip(attrs, metas):
        by.setdefault(m["speaker_id"], []).append((a, norm_gender(m.get("gender"))))
    data[c] = {k: v for k, v in by.items() if len(v) >= 2}
    print(f"[{c}] {len(data[c])} speakers with >=2 clips")

if not data:
    sys.exit("no corpora available")


def ratio(corpus, axis):
    w, mu = [], []
    for takes in data[corpus].values():
        x = np.array([getattr(t[0], axis, np.nan) for t in takes[:2]], float)
        if np.isfinite(x).all():
            w.append(x.var(ddof=1))
            mu.append(x.mean())
    if len(mu) < 20:
        return np.nan
    b = np.var(mu, ddof=1)
    return float(np.mean(w) / b) if b > 0 else np.nan


def gender_d(corpus, axis):
    F, M = [], []
    for takes in data[corpus].values():
        v = np.nanmean([getattr(t[0], axis, np.nan) for t in takes])
        g = takes[0][1]
        if not np.isfinite(v) or g is None:
            continue
        (F if g == "Female" else M).append(v)
    if min(len(F), len(M)) < 10:
        return np.nan
    F, M = np.array(F), np.array(M)
    return float((M.mean() - F.mean()) /
                 max(np.sqrt((F.var(ddof=1) + M.var(ddof=1)) / 2), 1e-9))


# speaker-mean matrix per corpus, for the redundancy screen
def spk_means(corpus, axes):
    rows = []
    for takes in data[corpus].values():
        rows.append([np.nanmean([getattr(t[0], a, np.nan) for t in takes]) for a in axes])
    return np.array(rows, float)


print()
print(f"  {'axis':<20} " +
      " ".join(f"{c.replace('indicvoices_r_','')[:4]:>7}" for c in data) +
      f" {'mean':>7} {'redund':>7}  verdict")
print("  " + "-" * (22 + 8 * len(data) + 26))

rows = []
for ax in CANDIDATES:
    rs = [ratio(c, ax) for c in data]
    if not np.isfinite(rs).any():
        continue
    m = float(np.nanmean(rs))
    # redundancy: max |corr| against the CURRENT identity axes, excluding itself
    others = [a for a in CURRENT_IDENTITY if a != ax]
    reds = []
    for c in data:
        X = spk_means(c, [ax] + others)
        ok = np.isfinite(X).all(1)
        if ok.sum() < 20:
            continue
        X = X[ok]
        for j in range(1, X.shape[1]):
            if X[:, 0].std() > 1e-9 and X[:, j].std() > 1e-9:
                reds.append(abs(np.corrcoef(X[:, 0], X[:, j])[0, 1]))
    red = float(np.max(reds)) if reds else np.nan

    if ax in RECORDING_AXES:
        v = "recording axis, excluded by design"
    elif ax in CURRENT_IDENTITY:
        v = "IN USE"
    elif ax in KNOWN_DEAD:
        v = "known dead (S16)"
    elif m >= 1.0:
        v = "not identity — within > between"
    elif np.isfinite(red) and red >= 0.80:
        v = f"separates, but redundant (r={red:.2f})"
    else:
        v = "*** CANDIDATE ***"
    rows.append({"axis": ax, "ratios": rs, "mean_ratio": m, "redundancy": red,
                 "verdict": v, "gender_d": float(np.nanmean([gender_d(c, ax) for c in data]))})
    print(f"  {ax:<20} " + " ".join(f"{r:>7.2f}" for r in rs) +
          f" {m:>7.2f} {red:>7.2f}  {v}")

json.dump(rows, io.open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=2)

cands = [r for r in rows if "CANDIDATE" in r["verdict"]]
print()
print("=" * 78)
print("S18 — which measured axes could carry identity?")
print("=" * 78)
print(f"  ratio  = within-speaker variance / between-speaker variance, over 2 takes")
print(f"  redund = max |correlation| with an axis already in the identity set")
print()
if cands:
    print(f"  {len(cands)} candidate(s) beyond the four in use:")
    for r in cands:
        cap = "has caption labels" if r["axis"] in BIN_LABELS else "NO caption labels yet"
        print(f"    {r['axis']:<20} ratio {r['mean_ratio']:.2f}  "
              f"redundancy {r['redundancy']:.2f}  gender d {r['gender_d']:+.2f}  ({cap})")
else:
    print("  none. The four axes in use are the ones this corpus supports,")
    print("  and widening the description needs a NEW measurement rather than")
    print("  a better reading of an existing one.")
print("=" * 78)
