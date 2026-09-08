"""
S22 -- confirm the adopted axis set reproduces what S17/S18 measured

`CATALOG_AXES` changed on 2026-09-08 from the five shipped caption axes to the
S17/S18 identity set:

    OUT  f0_cv, speaking_rate   within-speaker spread EXCEEDS between-speaker
                                spread on real people (1.04, 1.32 -- S16,
                                reproducing S12 and E15)
    OUT  hnr_db                 weakest at 0.59, fails outright on Tamil at
                                1.32; buys 0.3 effective voices for 15.6 points
                                of adherence (S18)
    IN   vtl_cm                 never noise -- the ESTIMATOR was broken (S17)

    ->   f0_mean, spectral_tilt, vtl_cm

S17/S18 measured that swap in a THREE-ARM COMPARISON inside one script. This
runs the ADOPTED CONSTANT end to end instead, which is a different thing: it
checks that what the library now does by default is what the experiment
recommended, rather than what a local variable in that experiment did.

PROPERTIES ASSERTED BEFORE THE NUMBERS ARE READ

  P1  `alaap.catalog.CATALOG_AXES` is exactly the S18 set. If this run is not
      exercising the adopted constant it confirms nothing.
  P2  Every sampled cell mentions all three axes and nothing else, and every
      one renders to a caption. S17 arm 3 is why: caption and sampler drifting
      apart cost 1.2 effective voices and a third of the minting budget.
  P3  Effective voices land within +-2.0 of S17's 22.3 for this arm. Wider than
      that and the adoption is not the thing S17 measured.
  P4  Adherence is at least 90%. S18 measured 98.1%; the point of dropping
      hnr_db was that it cost 15.6 points, so anything near the old 82.5% means
      the drop did not take effect.

WHAT THIS DOES *NOT* DO. It is geometry, like S17 -- no codec, no LM, nothing
rendered or heard. `HANDOFF` item 1 also asks for S7's render pass "under the
audited pipeline rather than in geometry"; that needs codec + LM resident
together and is blocked below ~3 GB free RAM (HANDOFF 8b). Run
`experiments/S7-indic-catalog/run_indic_catalog.py` when the box has room.

    envs/qwen3/Scripts/python.exe experiments/S22-adopt-axes/run_confirm.py
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
from alaap.captions import ORDER, PHRASES, caption_from_bins
from alaap.catalog import CATALOG_AXES, CATALOG_AXES_V1, sample_cells
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder
from alaap.metrics import nn_distances, vendi_score
from alaap.service import UNIQUENESS_MIN_MIOCODEC

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
S6_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "S6-indic-mint", "out")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--clips", type=int, default=250)
ap.add_argument("--per-speaker", type=int, default=2)
ap.add_argument("--mints", type=int, default=80)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

S17_EFFECTIVE, S17_TOL = 22.3, 2.0
S18_ADHERENCE, ADH_FLOOR = 0.981, 0.90
EXPECTED = ["f0_mean", "spectral_tilt", "vtl_cm"]

# ------------------------------------------------------------------ P1 and P2
print("P1  the adopted constant is the S18 set")
print(f"    CATALOG_AXES    = {CATALOG_AXES}")
print(f"    CATALOG_AXES_V1 = {CATALOG_AXES_V1}")
assert list(CATALOG_AXES) == EXPECTED, (
    f"this run is not exercising the adopted set: {CATALOG_AXES}")
print("    PASS")

print()
print("P2  sampler and caption renderer agree")
probe = sample_cells(min(args.mints, 5 ** len(CATALOG_AXES)), args.seed)
assert all(set(c) == set(CATALOG_AXES) for c in probe), "cells carry stray axes"
for a in CATALOG_AXES:
    assert a in ORDER and a in PHRASES, f"{a} is sampled but cannot be said"
assert all(caption_from_bins(c, seed=i).strip().endswith(".")
           for i, c in enumerate(probe[:10]))
print(f"    {len(probe)} cells, all 3-axis, all renderable    PASS")

# ---------------------------------------------------------------- the corpus
print()
print(f"[1/3] corpus and the bound ({args.corpus})")
d4 = np.load(f"experiments/S4-indic/out/"
             f"measured_{args.corpus}_{args.clips}_{args.per_speaker}.npz",
             allow_pickle=True)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d4["attrs"]))]
metas = json.loads(str(d4["metas"]))
Z = np.load(os.path.join(S6_OUT, f"mio_emb_{args.corpus}_{args.clips}_"
                                 f"{args.per_speaker}.npz"),
            allow_pickle=True)["Z"].astype(np.float64)
n = min(len(Z), len(attrs))
attrs, metas, Z = attrs[:n], metas[:n], Z[:n]

space = SpeakerSpace.fit(Z, n_components=min(64, n - 1))
E = space.encode(Z)
byspk = {}
for e, m in zip(E, metas):
    byspk.setdefault(m["speaker_id"], []).append(e)
E_spk = np.vstack([np.mean(v, 0) for v in byspk.values()])
BOUND = vendi_score(E_spk) * len(E_spk)
print(f"      {len(E_spk)} real speakers | bound ~{BOUND:.0f} effective voices")

binner = Binner().fit(attrs)
full_bins = [binner.bin_one(a) for a in attrs]
order = {a: {lab: i for i, lab in enumerate(BIN_LABELS[a])} for a in CATALOG_AXES}

# ------------------------------------------------------------------ the arm
print(f"[2/3] minting {args.mints} voices through the adopted set")
caps = [caption_from_bins({k: v for k, v in b.items() if k in CATALOG_AXES}, seed=i)
        for i, b in enumerate(full_bins)]
anchor_bins = [{k: v for k, v in b.items() if k in CATALOG_AXES} for b in full_bins]
m = RetrievalMapper(space, TextEncoder(), pca_dims=space.components.shape[0],
                    retrieval="hybrid").fit(caps, Z, anchor_bins=anchor_bins)

cells = sample_cells(args.mints, args.seed)
descs = [caption_from_bins(c, seed=i) for i, c in enumerate(cells)]
En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
acc, adh, per_axis = [], [], {a: [] for a in CATALOG_AXES}
for i, (c, txt) in enumerate(zip(cells, descs)):
    r = m.mint(txt, novelty=0.0, seed=i, top_k=2)
    e = space.encode(r.vector)[0]
    if acc:
        P = np.vstack(acc)
        u = float(np.min(1.0 - (P @ e) / np.maximum(
            np.linalg.norm(P, axis=1) * np.linalg.norm(e), 1e-12)))
    else:
        u = 1.0
    if u >= UNIQUENESS_MIN_MIOCODEC:
        acc.append(e)
    nb = full_bins[int(np.argmax(En @ (e / max(np.linalg.norm(e), 1e-12))))]
    hits = []
    for a in CATALOG_AXES:
        if a in c and a in nb:
            h = float(order[a][c[a]] == order[a][nb[a]])
            per_axis[a].append(h)
            hits.append(h)
    adh.append(float(np.mean(hits)) if hits else np.nan)
    if (i + 1) % 20 == 0:
        print(f"      {i + 1}/{len(cells)} minted, {len(acc)} accepted")

A = np.vstack(acc)
v = vendi_score(A) if len(A) > 1 else float("nan")
eff = v * len(A)
adherence = float(np.nanmean(adh))
space_size = 5 ** len(CATALOG_AXES)

print()
print(f"[3/3] results")
print(f"      describable cells   {space_size}")
print(f"      sampled             {len(cells)}  ({len(cells) / space_size:.0%} of the space)")
print(f"      accepted            {len(A)} / {len(cells)}")
print(f"      normalised Vendi    {v:.3f}")
print(f"      EFFECTIVE VOICES    {eff:.1f}   ({eff / BOUND:.0%} of the ~{BOUND:.0f} bound)")
print(f"      adherence           {adherence:.1%}")
for a in CATALOG_AXES:
    print(f"        {a:<16} {np.mean(per_axis[a]):.1%}")
print(f"      median NN distance  {np.median(nn_distances(A)):.3f}")

# ---------------------------------------------------------------- P3 and P4
print()
ok = True
d = abs(eff - S17_EFFECTIVE)
print(f"P3  effective voices within +-{S17_TOL} of S17's {S17_EFFECTIVE}")
print(f"    {eff:.1f}  (off by {d:.1f})   {'PASS' if d <= S17_TOL else 'FAIL'}")
ok &= d <= S17_TOL
print(f"P4  adherence at least {ADH_FLOOR:.0%} (S18 measured {S18_ADHERENCE:.1%})")
print(f"    {adherence:.1%}   {'PASS' if adherence >= ADH_FLOOR else 'FAIL'}")
ok &= adherence >= ADH_FLOOR

print()
print("=" * 78)
if ok:
    print("  The adopted default reproduces what S17/S18 recommended.")
else:
    print("  THE ADOPTION DOES NOT REPRODUCE THE EXPERIMENT. Do not ship it.")
print("=" * 78)
print()
print(f"  CAVEAT, from S17: this arm samples {len(cells) / space_size:.0%} of its own")
print("  describable space, so its acceptance rate is flattered by nearly running")
print("  out of distinct things to ask for. The saturation CURVE from a run this")
print("  close to the ceiling measures the sampler, not the catalog.")
print("  CAVEAT, from S21: `spectral_tilt` reads F0 at r = +0.918 where brightness")
print("  is fixed by construction. Count this set as ~2.6 independent axes, not 3.")

json.dump({"axes": list(CATALOG_AXES), "corpus": args.corpus, "mints": args.mints,
           "cells": space_size, "sampled": len(cells), "accepted": len(A),
           "vendi": float(v), "effective": float(eff), "bound": float(BOUND),
           "adherence": adherence,
           "per_axis": {a: float(np.mean(per_axis[a])) for a in CATALOG_AXES},
           "P3_pass": bool(d <= S17_TOL), "P4_pass": bool(adherence >= ADH_FLOOR)},
          io.open(os.path.join(OUT, "confirm.json"), "w", encoding="utf-8"), indent=2)
print(f"\n  -> {os.path.join(OUT, 'confirm.json')}")
