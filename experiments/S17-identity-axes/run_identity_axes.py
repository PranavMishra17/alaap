"""
S17 — does dropping the two dead caption axes raise the ceiling?

`S16` found that `f0_cv` and `speaking_rate` are not identity axes at all: on
real speakers, two ordinary clips of one person differ on them MORE than two
people do (within/between 1.04 and 1.32). `S12` found the same two axes on
acted emotion. `E15` measured `speaking_rate` at an identity weight of 0.10-0.31
across three corpora and called it "nearly worthless", which was the same fact
seen a third time.

Captions are written over five axes. Two of them distinguish nobody. `S9` found
the described space holds only ~38 effective voices and listed three candidate
causes; this was not among them, and it is more mechanical than any of them.

BUT THE OBVIOUS FIX MIGHT MAKE IT WORSE, which is why this is measured rather
than applied. Dropping two axes shrinks the DESCRIBABLE space from 5^5 = 3,125
cells to 5^3 = 125. Fewer distinct descriptions can mean fewer distinct voices,
regardless of whether the dropped axes carried identity. The two effects push
opposite ways:

    removing dead axes    -> less dilution, each word does more work
    fewer axes            -> fewer distinguishable descriptions to ask with

THREE ARMS, same corpus, same n, same seeds:

    5-axis          what ships today
    3-axis          identity axes only (f0_mean, spectral_tilt, hnr_db)
    5-axis, gated   all five written, but the mapper's hybrid weights forced to
                    zero on the two dead ones -- keeps the description rich for
                    the reader while ignoring what cannot identify anyone

MEASURED: effective voices (normalised Vendi x n) against S9's ~38 bound, and
adherence scored ON THE IDENTITY AXES ONLY, because an arm that never writes
`speaking_rate` cannot be marked down for missing it.

    envs/qwen3/Scripts/python.exe experiments/S17-identity-axes/run_identity_axes.py
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
# PINNED TO THE V1 SET. This experiment's committed RESULTS.md numbers were
# measured on the five shipped caption axes, before S17/S18's identity set was
# adopted (2026-09-08). Following the new default here would silently change
# what every quoted acceptance count and adherence percentage means. Re-running
# it on the current set is a DIFFERENT question and deserves its own arm.
# Arm 1 IS "the five shipped axes"; pointing it at the new default would
# have made it compare the identity set against itself.
from alaap.catalog import CATALOG_AXES_V1 as CATALOG_AXES, sample_cells
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder
from alaap.metrics import nn_distances, vendi_score
from alaap.service import UNIQUENESS_MIN_MIOCODEC

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
S6_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "S6-indic-mint", "out")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--clips", type=int, default=250)
ap.add_argument("--per-speaker", type=int, default=2)
ap.add_argument("--mints", type=int, default=80)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

# vtl_cm joins the identity set: the VTL estimator was fixed in this same
# experiment (dispersion averaged through F2, the vowel-dependent formant,
# cancelling F1 and F3). Gender separation went +0.11 -> +0.70 on hi,
# -0.17 -> +1.03 on bn, +0.06 -> +0.44 on ta, +0.11 -> +0.68 on en, so it
# now clears the d >= 0.30 bar S4/S6 drop axes below.
IDENTITY = ["f0_mean", "spectral_tilt", "hnr_db", "vtl_cm"]
DEAD = ["f0_cv", "speaking_rate"]

print(f"[1/4] corpus and the bound ({args.corpus})")
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
binner = Binner.fit(attrs)
full_bins = [binner.bin_one(a) for a in attrs]
space = SpeakerSpace.fit(Z, n_components=min(64, n - 1))
E = space.encode(Z)

# the bound, one point per speaker, as S9 defines it
byspk = {}
for e, m in zip(E, metas):
    byspk.setdefault(m["speaker_id"], []).append(e)
E_spk = np.vstack([np.mean(v, 0) for v in byspk.values()])
vb = vendi_score(E_spk)
BOUND = vb * len(E_spk)
print(f"      {len(E_spk)} real speakers | bound ~{BOUND:.0f} effective voices")

# ------------------------------------------------------------------- arms
# over the union: vtl_cm is an identity axis but not one of the five
# CATALOG_AXES the shipped captions use.
order = {a: {lab: i for i, lab in enumerate(BIN_LABELS[a])}
         for a in dict.fromkeys(list(CATALOG_AXES) + IDENTITY)}


# Scored on the axes COMMON TO EVERY ARM, not on each arm's own axis set.
# Scoring each arm over its own axes is not a comparison: the 5-axis arm's
# cells contain no vtl_cm, so it would be graded on three axes while the
# 4-axis arm was graded on four, and adding a harder axis to one side lowers
# its mean whether or not anything got worse. That is how a 95.8% and an 81.9%
# ended up in the same column on the first run.
SCORED = ["f0_mean", "spectral_tilt", "hnr_db"]


def adherence_identity(cell, bins_of_voice):
    hits = [order[a][cell[a]] == order[a][bins_of_voice[a]]
            for a in SCORED if a in cell and a in bins_of_voice]
    return float(np.mean(hits)) if hits else np.nan


def adherence_vtl(cell, bins_of_voice):
    """Reported separately, since only the arm that writes it can be scored."""
    a = "vtl_cm"
    if a in cell and a in bins_of_voice:
        return float(order[a][cell[a]] == order[a][bins_of_voice[a]])
    return np.nan


def run(label, axes, gate_dead):
    caps = [caption_from_bins({k: v for k, v in b.items() if k in axes}, seed=i)
            for i, b in enumerate(full_bins)]
    anchor_bins = [{k: v for k, v in b.items() if k in axes} for b in full_bins]
    m = RetrievalMapper(space, TextEncoder(), pca_dims=space.components.shape[0],
                        retrieval="hybrid").fit(caps, Z, anchor_bins=anchor_bins)
    if gate_dead and m.axis_weights is not None:
        w = m.axis_weights.copy()
        for i, a in enumerate(m.axes):
            if a in DEAD:
                w[i] = 0.0
        # renormalise so alpha, which is a WEIGHT SHARE, stays comparable
        m.axis_weights = w / max(w.sum(), 1e-12) * len(w)

    cells = sample_cells(args.mints, args.seed, axes=axes)
    descs = [caption_from_bins(c, seed=i) for i, c in enumerate(cells)]
    V, acc, adh, vadh = [], [], [], []
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
        V.append(e)
        # where did the minted voice land, in bin terms? nearest real clip's bins
        En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        en = e / max(np.linalg.norm(e), 1e-12)
        nb = full_bins[int(np.argmax(En @ en))]
        adh.append(adherence_identity(c, nb))
        vadh.append(adherence_vtl(c, nb))
    A = np.vstack(acc)
    v = vendi_score(A) if len(A) > 1 else float("nan")
    cells_available = 5 ** len(axes)
    return {"label": label, "axes": len(axes), "cells": cells_available,
            "accepted": len(A), "vendi": v, "effective": v * len(A),
            "share_of_bound": v * len(A) / BOUND,
            "adherence": float(np.nanmean(adh)),
            "vtl_adherence": float(np.nanmean(vadh)) if np.isfinite(vadh).any() else np.nan,
            "nn_median": float(np.median(nn_distances(A))) if len(A) > 2 else np.nan}


print(f"[2/5] arm 1 — the 5 shipped caption axes")
r1 = run("5-axis (shipped)", CATALOG_AXES, False)
print(f"[3/5] arm 2 — identity axes only ({len(IDENTITY)}, incl. vtl_cm)")
r2 = run(f"{len(IDENTITY)}-axis (identity only)", IDENTITY, False)
print(f"[4/5] arm 3 — 5 axes written, dead ones gated to zero weight")
r3 = run("5-axis, dead gated", CATALOG_AXES, True)

# S18 audited every measured axis for the within/between property and screened
# survivors for redundancy against the set already in use. Most survivors are
# derivatives of axes already here -- f0_std and f0_range restate f0_mean, f3
# and formant_dispersion restate vtl_cm. Only jitter (redundancy 0.36) and
# shimmer (0.50) are both separating AND independent, so only they are worth
# the cost of a wider description.
print(f"[5/5] arms 4-5 — adding S18's independent candidates")
r4 = run("5-axis (+shimmer)", IDENTITY + ["shimmer"], False)
r5 = run("6-axis (+jitter+shimmer)", IDENTITY + ["jitter", "shimmer"], False)
# hnr_db is the weakest axis IN the set (0.59 mean) and it fails outright on
# Tamil (1.32). If the threshold is what matters rather than the < 1 line, a
# set without it should do better, not worse.
r6 = run("3-axis (-hnr_db)", [a for a in IDENTITY if a != "hnr_db"], False)

rows = [r1, r2, r3, r4, r5, r6]
json.dump({"bound": BOUND, "rows": rows},
          io.open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=2)

print()
print("=" * 82)
print(f"S17 — do the two dead caption axes cost capacity? (bound ~{BOUND:.0f})")
print("=" * 82)
print(f"  {'arm':<24} {'cells':>7} {'accept':>7} {'effective':>10} "
      f"{'of bound':>9} {'adherence':>10} {'vtl adh':>9}")
print("  " + "-" * 72)
for r in rows:
    va = r.get("vtl_adherence", float("nan"))
    print(f"  {r['label']:<24} {r['cells']:>7} {r['accepted']:>7} "
          f"{r['effective']:>10.1f} {r['share_of_bound']:>8.0%} "
          f"{r['adherence']:>10.1%} "
          f"{('%.1f%%' % (va*100)) if np.isfinite(va) else '-':>9}")
print()
print("  'cells' is how many distinct descriptions the axis set can express.")
print("  'adherence' is scored on f0_mean / spectral_tilt / hnr_db -- the axes")
print("  EVERY arm writes -- so the column compares like with like. vtl_cm is")
print("  reported separately because only one arm asks for it.")
print("=" * 82)
