"""
S10 — does the English pipeline have the same defect, and how does catalog
      SEARCH compare between English and Indic?

Two questions, one corpus, because they share every input.

(1) THE SAME DEFECT? S9b found minting reaching 37% of the diversity its own
    speakers hold, caused by blending anchors in a truncated basis. Every
    English experiment (E9, E11, E12, E14, S2) ran with the same defaults --
    `pca_dims` well below the space's rank and `top_k=4` -- so if the mechanism
    is real it is in those numbers too, and E11's "~20 effective voices" is an
    understatement of what the English path can do rather than a measurement of
    what it can.

    The bound was never measured for English either. E11 reported Vendi 0.482
    against `RESEARCH/06`'s 0.35 floor, which says "not mode-collapsed"; it does
    not say how much of the available diversity was reached.

(2) SEARCH vs MINT IN ENGLISH. S8 showed that on Indic, answering a description
    by RETRIEVING the nearest library voice beats minting a new one. Whether
    that holds on Qwen3's 2048-d space over GLOBE_V2 is a different question:
    the space is 16x wider, the corpus is 10x larger, and the encoder is a
    different family. If retrieval wins there too, it is a property of the
    approach; if it does not, S8's result is about MioCodec.

Every number is reported against the bound, which is the discipline S9 added.

    envs/qwen3/Scripts/python.exe experiments/S10-english-bound/run_english_bound.py
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
# PINNED TO THE V1 SET. This experiment's committed RESULTS.md numbers were
# measured on the five shipped caption axes, before S17/S18's identity set was
# adopted (2026-09-08). Following the new default here would silently change
# what every quoted acceptance count and adherence percentage means. Re-running
# it on the current set is a DIFFERENT question and deserves its own arm.
from alaap.catalog import CATALOG_AXES_V1 as CATALOG_AXES, sample_cells
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder
from alaap.metrics import nn_distances, vendi_score

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="experiments/S2/out/"
                                   "corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz")
ap.add_argument("--n-components", type=int, default=150)
ap.add_argument("--mints", type=int, default=80)
ap.add_argument("--queries", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

print(f"[1/4] {args.cache}")
d = np.load(args.cache, allow_pickle=True)
Z = d["Z"].astype(np.float64)
A = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
ids = [str(x) for x in d["ids"]]
k = min(len(Z), len(A), len(ids))
Z, A, ids = Z[:k], A[:k], ids[:k]
print(f"      {len(Z)} clips, {len(set(ids))} speakers, {Z.shape[1]}-d")

binner = Binner.fit(A)
bins_all = [binner.bin_one(a) for a in A]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins_all)]
space = SpeakerSpace.fit(Z, n_components=args.n_components)
E = space.encode(Z)
print(f"      {space}")

# ---------------------------------------------------------------- the bound
# One point per speaker: a talkative speaker must not be counted as several.
byspk = {}
for e, s in zip(E, ids):
    byspk.setdefault(s, []).append(e)
E_spk = np.vstack([np.mean(v, 0) for v in byspk.values()])
v_bound = vendi_score(E_spk)
BOUND = v_bound * len(E_spk)
print(f"[2/4] the bound: {len(E_spk)} real speakers, Vendi {v_bound:.3f} "
      f"-> ~{BOUND:.0f} effective voices")

# ------------------------------------------------------- minting, old vs new
print(f"[3/4] minting {args.mints} voices at the old and new settings")
cells = sample_cells(args.mints, args.seed)
descs = [caption_from_bins(c, seed=i) for i, c in enumerate(cells)]
rank = space.components.shape[0]
cen = E.mean(0)
r_real = np.linalg.norm(E - cen, axis=1).mean()

rows = []
for label, pdims, tk in [("E11's settings", min(50, rank), 4),
                         ("full basis", rank, 4),
                         ("full basis + top_k 2", rank, 2),
                         ("pure retrieval (top_k 1)", rank, 1)]:
    m = RetrievalMapper(space, TextEncoder(), pca_dims=pdims,
                        retrieval="hybrid").fit(caps, Z, anchor_bins=bins_all)
    M = np.vstack([space.encode(m.mint(x, novelty=0.0, seed=i, top_k=tk).vector)[0]
                   for i, x in enumerate(descs)])
    v = vendi_score(M)
    rows.append({"label": label, "pca_dims": pdims, "top_k": tk, "vendi": v,
                 "eff": v * len(M),
                 "radius": float(np.linalg.norm(M - cen, axis=1).mean() / r_real),
                 "nn": float(np.median(nn_distances(M)))})
    print(f"      {label:<26} pca {pdims:>3} top_k {tk} -> "
          f"~{v*len(M):.0f} effective", flush=True)

# ------------------------------------------------ retrieval over a library
# Same construction as S8: one library entry per SPEAKER, retrieved by
# description, scored by re-binning the retrieved speaker's own measured
# attributes against the queried cell.
print(f"[4/4] library retrieval, {args.queries} queries")
FIELDS = list(CATALOG_AXES)
by = {}
for a, s in zip(A, ids):
    by.setdefault(s, []).append(a)
library = []
for sid, aa in sorted(by.items()):
    vals = {f: float(np.nanmean([getattr(x, f) for x in aa])) for f in FIELDS}
    proto = Attributes(**{**aa[0].to_dict(), **vals})
    b = {kk: vv for kk, vv in binner.bin_one(proto).items() if kk in FIELDS}
    library.append({"sid": sid, "bins": b, "caption": caption_from_bins(b, seed=len(library))})
LIB_Z = np.vstack([np.mean([Z[i] for i, s in enumerate(ids) if s == v["sid"]], 0)
                   for v in library])
lib_space = SpeakerSpace.fit(LIB_Z, n_components=min(args.n_components, len(library) - 1))

qcells = sample_cells(args.queries, args.seed)
queries = [caption_from_bins(c, seed=10_000 + i) for i, c in enumerate(qcells)]
enc = TextEncoder()
order_lbl = {a: {lab: i for i, lab in enumerate(BIN_LABELS[a])} for a in FIELDS}


def adh(cell, voice):
    ex = sum(order_lbl[a][cell[a]] == order_lbl[a][voice["bins"][a]]
             for a in FIELDS if a in voice["bins"])
    return ex / len(FIELDS)


arms = {}
for mode in ("text", "hybrid"):
    mm = RetrievalMapper(lib_space, enc,
                         pca_dims=lib_space.components.shape[0], retrieval=mode).fit(
        [v["caption"] for v in library], LIB_Z,
        anchor_bins=[v["bins"] for v in library] if mode == "hybrid" else None)
    arms[mode] = [int(mm.retrieve(q, top_k=1)[0][0]) for q in queries]
rng = np.random.default_rng(args.seed)
rand = rng.integers(0, len(library), size=len(queries))

scores = {m: float(np.mean([adh(qcells[i], library[arms[m][i]])
                            for i in range(len(queries))])) for m in arms}
scores["random"] = float(np.mean([adh(qcells[i], library[int(rand[i])])
                                  for i in range(len(queries))]))
reached = sorted(set(arms["hybrid"]))
E_lib = lib_space.encode(LIB_Z)
v_ret = vendi_score(E_lib[reached])

print()
print("=" * 78)
print(f"S10 — English: the bound, the defect, and search vs mint")
print("=" * 78)
print(f"  corpus: {len(set(ids))} speakers, {Z.shape[1]}-d Qwen3 space")
print()
print(f"  {'':30} {'Vendi':>7} {'effective':>10} {'of bound':>9}")
print(f"  {'REAL speakers (the bound)':30} {v_bound:>7.3f} {BOUND:>10.0f} {'100%':>9}")
for r in rows:
    print(f"  {r['label']:30} {r['vendi']:>7.3f} {r['eff']:>10.0f} "
          f"{r['eff']/BOUND:>9.0%}")
print(f"  {'library retrieval (reached)':30} {v_ret:>7.3f} "
      f"{v_ret*len(reached):>10.0f} {v_ret*len(reached)/BOUND:>9.0%}")
print()
print(f"  minting fix: {rows[0]['eff']:.0f} -> {rows[2]['eff']:.0f} effective "
      f"({rows[2]['eff']/rows[0]['eff']-1:+.0%})")
print(f"  radial contraction at E11's settings: {rows[0]['radius']:.0%} of real")
print()
print("  SEARCH: description -> nearest library voice, exact bin match")
print(f"    {'random':<10} {scores['random']:.1%}")
print(f"    {'text':<10} {scores['text']:.1%}")
print(f"    {'hybrid':<10} {scores['hybrid']:.1%}")
print(f"    reached {len(reached)}/{len(library)} library voices")
print("=" * 78)
