"""
Analyse a catalog — including one that is still being built.

`run_catalog.py` only writes its summary after the last mint, and a 300-voice
run takes on the order of 12 hours on this hardware. That makes the headline
number (Vendi score: how many DISTINCT voices the catalog actually holds)
unavailable until the very end, and lost entirely if the run is interrupted.

This reads the artefacts instead of the process:

    out/catalog.db   the identities, each carrying its minted vector
    out/rows.json    the per-mint audit trail, flushed every 10 mints

so it works on a partial run and cannot disturb one — it opens the SQLite file
read-only.

    envs/qwen3/Scripts/python.exe experiments/E11-catalog/analyse.py
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
from alaap.geometry import SpeakerSpace
from alaap.metrics import vendi_score, nn_distances
from alaap.service import UNIQUENESS_MIN, DRIFT_FLOOR, CONSISTENCY_FLOOR

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "out"))
ap.add_argument("--corpus-cache",
                default="experiments/S2/out/"
                        "corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz")
ap.add_argument("--write", action="store_true",
                help="write results.json (default: print only)")
args = ap.parse_args()

rows_p = os.path.join(args.out, "rows.json")
db_p = os.path.join(args.out, "catalog.db")
rows = json.load(open(rows_p)) if os.path.exists(rows_p) else []
ok = [r for r in rows if "error" not in r]
clean = [r for r in ok if r.get("clean")]


def arr(key, src):
    v = [r[key] for r in src if r.get(key) is not None]
    return np.array(v, dtype=np.float64) if v else np.array([])


# ------------------------------------------------------------- the vectors
# Open read-only (immutable=1) so a run in progress is never blocked or
# disturbed by this. sqlite3 URIs need forward slashes even on Windows.
V, tiers = None, {}
if os.path.exists(db_p):
    import sqlite3
    uri = "file:" + db_p.replace("\\", "/").replace(" ", "%20") + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    vecs = []
    for r in con.execute("SELECT embedding, identity_tier FROM identity"):
        tiers[r["identity_tier"]] = tiers.get(r["identity_tier"], 0) + 1
        if r["embedding"] is not None:
            vecs.append(np.frombuffer(r["embedding"], dtype=np.float32)
                        .astype(np.float64))
    con.close()
    if vecs:
        V = np.vstack(vecs)

summary = {
    "n_attempted": len(rows), "n_minted": len(ok),
    "n_errors": len(rows) - len(ok), "n_clean": len(clean),
    "clean_rate": len(clean) / max(len(ok), 1),
    "tiers": tiers,
}
for k, floor in (("drift", DRIFT_FLOOR), ("consistency", CONSISTENCY_FLOOR),
                 ("uniqueness", UNIQUENESS_MIN)):
    a = arr(k, ok)
    if len(a):
        summary[k] = {"mean": float(a.mean()), "min": float(a.min()),
                      "p05": float(np.percentile(a, 5)),
                      "below_floor": int((a < floor).sum()), "floor": floor}

# ------------------------------------------------------- saturation curve
BLOCK = max(10, len(ok) // 8)
curve = []
for s in range(0, len(ok), BLOCK):
    blk = ok[s:s + BLOCK]
    u = arr("uniqueness", blk)
    curve.append({"from": s, "to": s + len(blk),
                  "clean_rate": sum(1 for r in blk if r.get("clean")) / len(blk),
                  "mean_uniqueness": float(u.mean()) if len(u) else None,
                  "mean_attempts": float(np.mean([r.get("attempts", 1) for r in blk]))})
summary["saturation_curve"] = curve

# ----------------------------------------------------------- diversity
if V is not None and len(V) >= 10:
    d = np.load(args.corpus_cache, allow_pickle=True)
    space = SpeakerSpace.fit(d["Z"].astype(np.float64), n_components=150)
    E = space.encode(V)
    # vendi_score(normalise=True) already returns a FRACTION of the maximum,
    # not a count. Multiplying by n gives the effective number of voices;
    # dividing by n again (as an earlier version did) is meaningless.
    # RESEARCH/06 targets normalised VS >= 0.35, alarm < 0.20, collapse < 0.10.
    vs_norm = float(vendi_score(E))
    nn = nn_distances(E)
    summary["diversity"] = {
        "n_vectors": int(len(V)),
        "vendi_normalised": vs_norm,
        "effective_voices": vs_norm * len(V),
        "verdict": ("collapse" if vs_norm < 0.10 else
                    "alarm" if vs_norm < 0.20 else
                    "below target" if vs_norm < 0.35 else "ok"),
        "nn_median": float(np.median(nn)), "nn_p05": float(np.percentile(nn, 5)),
        "nn_min": float(nn.min()),
    }

if args.write:
    json.dump(summary, open(os.path.join(args.out, "results.json"), "w"), indent=2)

# ------------------------------------------------------------------ report
print()
print("=" * 78)
print("E11 — catalog saturation" + ("" if len(ok) else "  (nothing minted yet)"))
print("=" * 78)
print(f"  minted             {len(ok)}   ({summary['n_errors']} errors, "
      f"{len(clean)} clean = {summary['clean_rate']:.1%})")
print(f"  tiers              {tiers or '-'}")
for k in ("drift", "consistency", "uniqueness"):
    if k in summary:
        m = summary[k]
        print(f"  {k:<18} mean {m['mean']:.3f}  min {m['min']:.3f}  "
              f"p05 {m['p05']:.3f}  |  {m['below_floor']} below floor "
              f"{m['floor']}")
if "diversity" in summary:
    d = summary["diversity"]
    print()
    print(f"  VENDI (normalised) {d['vendi_normalised']:.3f}  [{d['verdict']}]  "
          f"target >=0.35, alarm <0.20")
    print(f"  effective voices   {d['effective_voices']:.1f} of "
          f"{d['n_vectors']} minted")
    print(f"  nn distance        median {d['nn_median']:.3f}  "
          f"p05 {d['nn_p05']:.3f}  min {d['nn_min']:.3f}")
print()
print("  saturation — does the catalog get harder to add to as it fills?")
print(f"    {'block':<12} {'clean':>7} {'uniq':>7} {'tries':>6}")
for c in curve:
    u = f"{c['mean_uniqueness']:.3f}" if c["mean_uniqueness"] is not None else "  -  "
    bar = "#" * int(round(c["clean_rate"] * 24))
    print(f"    {c['from']:>4}-{c['to']:<7} {c['clean_rate']:>6.0%} {u:>7} "
          f"{c['mean_attempts']:>6.2f}  {bar}")
print("=" * 78)
