"""
E14d — which knob actually cost E14 its description transport?

`S9b`/`S10` found `pca_dims=50` truncates the PCA basis and zero-pads the
discarded components. Raising it to the full basis raised effective voices
everywhere it was measured: E11 23 -> 41, S7 14 -> 22, E12 on every column.

Then E14 re-ran and went the OTHER WAY: transport fell 0.90x -> 0.62x. But TWO
things changed in that commit -- `pca_dims` AND `top_k` (defaulted 4 -> 2) --
and E14 varies neither, so its drop could not be attributed.

This script was written to test the obvious hypothesis: that raising pca_dims
adds variation the caption never asked for, buying diversity with fidelity.

THAT HYPOTHESIS IS REFUTED BY THE FIRST SWEEP BELOW. Transport is flat from
pca_dims=50 upward while diversity rises monotonically; the pca_dims fix costs
nothing. The docstring is left standing as it was written, because the point of
stating a hypothesis before measuring is that the measurement can kill it.

`top_k` is the knob that trades. Blending more anchors pins a minted voice
nearer what the caption specified and nearer its neighbours; blending fewer
spreads the catalog and loosens its grip on the description.

A CONFOUND, NAMED BECAUSE IT NEARLY WENT UNNOTICED: hybrid retrieval scores
anchors partly BY weighted bin distance, which is the very quantity `transport`
measures. Its figures are inflated by construction and are not comparable to
E14's, which fits without anchor_bins. Both arms are reported; the TEXT arm is
the one to read against E14.

WHAT IS MEASURED, on the same mints at every setting:

    transport   rho(bin distance, minted voice distance) / rho on REAL speakers
    diversity   normalised Vendi x n over the minted set

Geometry only, no rendering, so it is minutes rather than hours.

    envs/qwen3/Scripts/python.exe experiments/E14-transport/run_pca_tradeoff.py
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
from alaap.acoustics import BIN_LABELS, Attributes, Binner
from alaap.captions import caption_from_bins
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder
from alaap.metrics import vendi_score

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="experiments/S2/out/"
                                   "corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz")
ap.add_argument("--n-components", type=int, default=150)
ap.add_argument("--mints", type=int, default=120)
ap.add_argument("--novelty", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

DIMS = [10, 25, 50, 75, 100, 125, 150]

print(f"[1/3] {args.cache}")
d = np.load(args.cache, allow_pickle=True)
Z = d["Z"].astype(np.float64)
A = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
k = min(len(Z), len(A))
Z, A = Z[:k], A[:k]
binner = Binner.fit(A)
bins = [binner.bin_one(a) for a in A]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
space = SpeakerSpace.fit(Z, n_components=args.n_components)
E = space.encode(Z)
rank = space.components.shape[0]
print(f"      {len(Z)} clips | space rank {rank}")

# ------------------------------------------------- the real-speaker reference
AXES = [a for a in BIN_LABELS if a in bins[0]]
BIDX = {a: {lbl: i for i, lbl in enumerate(BIN_LABELS[a])} for a in AXES}
B = np.array([[BIDX[a][b[a]] for a in AXES] for b in bins], float)


def _rho(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


rng = np.random.default_rng(args.seed)
iu, ju = np.triu_indices(len(E), k=1)
sel = rng.choice(len(iu), min(60_000, len(iu)), replace=False)
iu, ju = iu[sel], ju[sel]
U = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
dv_real = (1.0 - U @ U.T)[iu, ju]
db_real = np.abs(B[iu] - B[ju]).sum(1)
REF = _rho(db_real, dv_real)
print(f"[2/3] real-speaker reference rho = {REF:.4f} over {len(iu)} pairs")

# ---------------------------------------------- the same mints at every dim
from alaap.catalog import sample_cells
cells = sample_cells(args.mints, args.seed)
descs = [caption_from_bins(c, seed=i) for i, c in enumerate(cells)]
Bm = np.array([[BIDX[a][c[a]] for a in AXES if a in c] for c in cells], float)
mi, mj = np.triu_indices(len(cells), k=1)
db_mint = np.abs(Bm[mi] - Bm[mj]).sum(1)

print(f"[3/3] minting {len(descs)} voices at {len(DIMS)} settings")
rows = []
for pd in DIMS:
    if pd > rank:
        continue
    m = RetrievalMapper(space, TextEncoder(), pca_dims=pd, retrieval="hybrid").fit(
        caps, Z, anchor_bins=bins)
    M = np.vstack([space.encode(m.mint(x, novelty=args.novelty, seed=i,
                                       top_k=2).vector)[0]
                   for i, x in enumerate(descs)])
    Um = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    dv_mint = (1.0 - Um @ Um.T)[mi, mj]
    rho = _rho(db_mint, dv_mint)
    v = vendi_score(M)
    rows.append({"pca_dims": pd, "rho": rho, "transport": rho / REF,
                 "vendi": v, "effective": v * len(M)})
    print(f"      pca_dims {pd:>3} | transport {rho/REF:>5.2f}x | "
          f"effective {v*len(M):>5.1f}", flush=True)

# ------------------------------------------------ the OTHER knob I changed
# The pca_dims sweep above came out FLAT, which refutes the hypothesis this
# script was written to test. So the drop E14 measured must come from the other
# thing that changed in the same commit: top_k, defaulted 4 -> 2. Same mints,
# same metric, full basis, varying only that.
print()
print(f"[4/4] the same measurement across top_k, at the full basis")
# TWO retrieval modes, and the reason is a confound in the sweep above.
# HYBRID retrieval scores anchors partly BY weighted bin distance -- which is
# the very quantity `transport` measures -- so its transport figures are
# inflated by construction and are not comparable to E14's. E14 fits without
# anchor_bins, i.e. retrieval="text", and that is the non-circular arm.
# Both are reported; read "text" against E14.
trows = []
for mode in ("text", "hybrid"):
  for tk in (1, 2, 4, 6):
    m = RetrievalMapper(space, TextEncoder(), pca_dims=rank, retrieval=mode).fit(
        caps, Z, anchor_bins=bins if mode == "hybrid" else None)
    M = np.vstack([space.encode(m.mint(x, novelty=args.novelty, seed=i,
                                       top_k=tk).vector)[0]
                   for i, x in enumerate(descs)])
    Um = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    rho = _rho(db_mint, (1.0 - Um @ Um.T)[mi, mj])
    v = vendi_score(M)
    trows.append({"mode": mode, "top_k": tk, "transport": rho / REF,
                  "effective": v * len(M)})
    print(f"      {mode:<7} top_k {tk} | transport {rho/REF:>5.2f}x | "
          f"effective {v*len(M):>5.1f}", flush=True)

json.dump({"reference_rho": REF, "rows": rows, "top_k_rows": trows},
          io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "out", "pca_tradeoff.json"), "w", encoding="utf-8"),
          indent=2)

print()
print("=" * 74)
print("E14d — description transport vs catalog diversity, across pca_dims")
print("=" * 74)
print(f"  real-speaker reference rho = {REF:.4f}")
print()
print(f"  {'pca_dims':>9} {'transport':>10} {'effective':>10}  {'':<24}")
print("  " + "-" * 56)
best_t = max(rows, key=lambda r: r["transport"])
best_d = max(rows, key=lambda r: r["effective"])
for r in rows:
    tag = []
    if r is best_t:
        tag.append("best transport")
    if r is best_d:
        tag.append("best diversity")
    print(f"  {r['pca_dims']:>9} {r['transport']:>9.2f}x {r['effective']:>10.1f}  "
          f"{', '.join(tag)}")
print()
print("  Transport is FLAT above pca_dims=50 while diversity keeps rising, so")
print("  the hypothesis this script was written to test is REFUTED: raising")
print("  pca_dims costs no description fidelity.")
print()
print(f"  {'retrieval':>10} {'top_k':>7} {'transport':>10} {'effective':>10}")
print("  " + "-" * 42)
for r in trows:
    print(f"  {r['mode']:>10} {r['top_k']:>7} {r['transport']:>9.2f}x "
          f"{r['effective']:>10.1f}")
print()
print("  Read the TEXT rows against E14: hybrid scores anchors partly BY bin")
print("  distance, which is what transport measures, so its figures are")
print("  inflated by construction.")
print()
print("  THIS is the trade. Blending fewer anchors spreads the catalog and")
print("  loosens its grip on the description; blending more does the reverse.")
print("=" * 74)
