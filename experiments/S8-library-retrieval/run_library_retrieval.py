"""
S8 — can a description just RETRIEVE the right voice from a fixed library?

The idea under test, in its cheapest form: forget minting a new speaker vector.
Take a TTS that already has a curated voice library -- Sarvam's Bulbul ships 39
studio voices across 11 Indian languages -- measure every voice with the same
pipeline that measures corpus audio, caption it, and answer a description by
RETRIEVING the nearest library voice instead of synthesising a new one.

If that works it is enormously cheaper than everything else in this repo: no
mapper, no drift, no consistency floor, no licence problem, and the voices are
studio quality by construction.

WHY IT IS TESTABLE TODAY, WITHOUT AN API KEY. A "library voice" is a fixed
identity with measurable acoustics and a caption. The 141 real Hindi speakers
already measured in S4 are exactly that, and they are a HARDER library than
Sarvam's: they were recorded, not curated for coverage. Swap the library for
39 Bulbul voices and not one line below changes.

THE TWO NUMBERS THAT DECIDE IT

  adherence  does the retrieved voice actually MATCH the description? Scored
             by re-binning the retrieved speaker's measured attributes against
             the queried bin cell. This must beat the RANDOM control, which is
             what "retrieval" degenerates to if the captions carry no signal.

  coverage   how many DISTINCT library voices does the whole query set ever
             reach? A library of 141 that answers every description with the
             same 12 voices is a library of 12. This is the number that
             decides library-vs-minting, because it is directly comparable to
             S7's effective-voice count (Vendi x accepted).

The control is not optional. Retrieval scores look plausible in isolation --
some axis always matches by chance, since five axes with five bins gives a 20%
per-axis hit rate for free.

    envs/qwen3/Scripts/python.exe experiments/S8-library-retrieval/run_library_retrieval.py
"""
import argparse
import hashlib
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
from alaap.catalog import CATALOG_AXES, sample_cells
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder
from alaap.metrics import nn_distances, vendi_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--clips", type=int, default=250)
ap.add_argument("--per-speaker", type=int, default=2)
ap.add_argument("--queries", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

S4 = f"experiments/S4-indic/out/measured_{args.corpus}_{args.clips}_{args.per_speaker}.npz"
if not os.path.exists(S4):
    sys.exit(f"missing {S4}")

# ------------------------------------------------------- 1. build the library
print(f"[1/4] building a voice library from {args.corpus}")
d = np.load(S4, allow_pickle=True)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
metas = json.loads(str(d["metas"]))
binner = Binner.fit(attrs)
binner.edges.pop("vtl_cm", None)          # dropped on this corpus, as in S4/S6

# One library entry per SPEAKER, not per clip -- a TTS voice is an identity,
# and averaging its clips is the closest analogue to a studio voice's fixed
# character. Attributes are averaged in their own units before binning.
by_spk: dict[str, list[Attributes]] = {}
for a, m in zip(attrs, metas):
    by_spk.setdefault(m["speaker_id"], []).append(a)

def _stable_seed(s: str) -> int:
    """Python's hash() is salted per process, so it would make the captions --
    and therefore every number below -- differ between runs."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 10_000


FIELDS = [f for f in CATALOG_AXES]
library = []
for sid, aa in sorted(by_spk.items()):
    vals = {f: float(np.nanmean([getattr(x, f) for x in aa])) for f in FIELDS}
    proto = Attributes(**{**aa[0].to_dict(), **vals})
    b = {k: v for k, v in binner.bin_one(proto).items() if k in FIELDS}
    library.append({"speaker_id": sid, "bins": b, "clips": [],
                    "caption": caption_from_bins(b, seed=_stable_seed(sid))})
print(f"      {len(library)} library voices from {len(attrs)} clips")
print(f"      axes: {FIELDS}")

# MioCodec embeddings for the same speakers, so diversity is measured in the
# SAME space S7 measured its minted catalog in. Comparing a distinct-voice
# COUNT against S7's Vendi-effective count would be comparing two different
# metrics as though they were one -- the exact error this project keeps
# catching in its own numbers.
EMB = ("experiments/S6-indic-mint/out/"
       f"mio_emb_{args.corpus}_{args.clips}_{args.per_speaker}.npz")
LIB_Z = None
if os.path.exists(EMB):
    Zc = np.load(EMB, allow_pickle=True)["Z"].astype(np.float64)
    nz = min(len(Zc), len(metas))
    idx = {v["speaker_id"]: k for k, v in enumerate(library)}
    acc = [[] for _ in library]
    for k in range(nz):
        acc[idx[metas[k]["speaker_id"]]].append(Zc[k])
    LIB_Z = np.vstack([np.mean(a, 0) if a else np.zeros(Zc.shape[1]) for a in acc])
    from alaap.geometry import SpeakerSpace
    LIB_E = SpeakerSpace.fit(Zc[:nz], n_components=min(64, nz - 1)).encode(LIB_Z)
    print(f"      + MioCodec embeddings, so diversity is comparable to S7")
else:
    LIB_E = None
    print(f"      no embedding cache at {EMB} -- Vendi will be skipped")

# ---------------------------------------------------------- 2. the queries
cells = sample_cells(args.queries, args.seed)
queries = [caption_from_bins(c, seed=10_000 + i) for i, c in enumerate(cells)]
print(f"[2/4] {len(queries)} descriptions over a stratified cover of bin space")

# -------------------------------------------------------- 3. retrieve
# Both arms go through RetrievalMapper.retrieve, the SAME ranking mint uses.
# A retrieval experiment that reimplements the ranking measures its own
# reimplementation, and E15's hybrid weights only exist inside the mapper.
print("[3/4] retrieving: text cosine vs E15 hybrid (weighted bins)")
if LIB_Z is None:
    sys.exit("hybrid retrieval needs the MioCodec embedding cache -- the axis "
             "weights are MEASURED against speaker vectors, not assumed.")

lib_caps = [v["caption"] for v in library]
lib_bins = [v["bins"] for v in library]
enc = TextEncoder()
lib_space = SpeakerSpace.fit(LIB_Z, n_components=min(64, len(library) - 1))

arms = {}
for mode in ("text", "hybrid"):
    m = RetrievalMapper(lib_space, enc, pca_dims=min(32, lib_space.components.shape[0]),
                        retrieval=mode).fit(
        lib_caps, LIB_Z, anchor_bins=lib_bins if mode == "hybrid" else None)
    arms[mode] = np.array([int(m.retrieve(q, top_k=1)[0][0]) for q in queries])
    if mode == "hybrid" and m.axis_weights is not None:
        print("      hybrid axis weights: " +
              ", ".join(f"{a} {w:.2f}" for a, w in
                        sorted(zip(m.axes, m.axis_weights), key=lambda t: -t[1])))

picks = arms["hybrid"]
rng = np.random.default_rng(args.seed)
rand_picks = rng.integers(0, len(library), size=len(queries))


def adherence(cell, voice):
    """Exact-bin and within-one-bin agreement over the five caption axes."""
    order = {a: {lab: i for i, lab in enumerate(BIN_LABELS[a])} for a in FIELDS}
    exact = near = 0
    for a in FIELDS:
        if a not in voice["bins"]:
            continue
        gi, pi = order[a][cell[a]], order[a][voice["bins"][a]]
        exact += (gi == pi)
        near += (abs(gi - pi) <= 1)
    return exact / len(FIELDS), near / len(FIELDS)


# ---------------------------------------- 3b. what a real user actually types
# Every query above is a caption_from_bins output naming all five axes. Real
# users do not write like that: E15b measured generated captions yielding 5 of
# 6 axes and realistic user text yielding 1.29 (22%). Scoring only on generated
# captions would report the ceiling as if it were the operating point.
#
# So: a second query set naming just TWO axes, in the bin's own words, and
# scored ONLY on the axes it actually named -- asking whether a voice has an
# attribute the user never mentioned is not a fair question.
print("[3b/4] sparse queries: two named axes, as a user would write them")
srng = np.random.default_rng(args.seed + 1)
sparse = []
for i, c in enumerate(cells):
    picked = sorted(srng.choice(len(FIELDS), 2, replace=False))
    ax = [FIELDS[k] for k in picked]
    sparse.append((f"a {c[ax[0]]}, {c[ax[1]]} voice", ax))

sparse_arms = {}
for mode in ("text", "hybrid"):
    m = RetrievalMapper(lib_space, enc,
                        pca_dims=min(32, lib_space.components.shape[0]),
                        retrieval=mode).fit(
        lib_caps, LIB_Z, anchor_bins=lib_bins if mode == "hybrid" else None)
    sparse_arms[mode] = [int(m.retrieve(q, top_k=1)[0][0]) for q, _ in sparse]


def named_adherence(cell, voice, named):
    """Exact agreement over ONLY the axes the query actually named."""
    order = {a: {lab: i for i, lab in enumerate(BIN_LABELS[a])} for a in FIELDS}
    hits = sum(order[a][cell[a]] == order[a][voice["bins"][a]]
               for a in named if a in voice["bins"])
    return hits / len(named)


sparse_scores = {
    mode: float(np.mean([named_adherence(cells[i], library[sparse_arms[mode][i]], ax)
                         for i, (_, ax) in enumerate(sparse)]))
    for mode in ("text", "hybrid")}
sparse_scores["random"] = float(np.mean(
    [named_adherence(cells[i], library[int(rand_picks[i])], ax)
     for i, (_, ax) in enumerate(sparse)]))

rows = []
for i, (cell, qi) in enumerate(zip(cells, queries)):
    tex, tnr = adherence(cell, library[arms["text"][i]])
    ex, nr = adherence(cell, library[arms["hybrid"][i]])
    rex, rnr = adherence(cell, library[rand_picks[i]])
    rows.append({"query": qi, "cell": cell,
                 "retrieved": library[picks[i]]["speaker_id"],
                 "retrieved_caption": library[picks[i]]["caption"],
                 "exact": ex, "near": nr,
                 "text_exact": tex, "text_near": tnr,
                 "random_exact": rex, "random_near": rnr})

json.dump(rows, io.open(os.path.join(OUT, "rows.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# ------------------------------------------------------------- 4. report
ex = np.mean([r["exact"] for r in rows])
nr = np.mean([r["near"] for r in rows])
tex = np.mean([r["text_exact"] for r in rows])
tnr = np.mean([r["text_near"] for r in rows])
rex = np.mean([r["random_exact"] for r in rows])
rnr = np.mean([r["random_near"] for r in rows])
reached = len(set(int(p) for p in picks))
counts = np.bincount(picks, minlength=len(library))
top = np.argsort(-counts)[:5]

print()
print("=" * 78)
print(f"S8 — description -> nearest library voice ({len(library)} voices, "
      f"{len(queries)} queries)")
print("=" * 78)
print(f"  {'':20} {'random':>9} {'text':>9} {'hybrid':>9} {'hyb-text':>10}")
print(f"  {'exact bin match':20} {rex:>9.1%} {tex:>9.1%} {ex:>9.1%} {ex-tex:>+10.1%}")
print(f"  {'within one bin':20} {rnr:>9.1%} {tnr:>9.1%} {nr:>9.1%} {nr-tnr:>+10.1%}")
print()
print(f"  text arm reached   {len(set(int(p) for p in arms['text']))}/{len(library)} voices")
print()
print(f"  SPARSE QUERIES -- two named axes, scored only on those two:")
print(f"    {'random':<10} {sparse_scores['random']:.1%}")
print(f"    {'text':<10} {sparse_scores['text']:.1%}")
print(f"    {'hybrid':<10} {sparse_scores['hybrid']:.1%}")
print(f'    e.g. "{sparse[0][0]}"')
print()
print("  THE CONTROL: random is what retrieval degenerates to if the captions")
print("  carry no signal. Five axes x five bins gives 20% per-axis for free.")
print()
print(f"  library voices ever reached : {reached}/{len(library)} "
      f"({reached/len(library):.0%})")
print(f"  most-returned voice answers : {counts.max()}/{len(queries)} queries "
      f"({counts.max()/len(queries):.0%})")
print("  top 5 voices by share:")
for t in top:
    print(f"    {library[t]['speaker_id'][:14]:<16} {counts[t]:>4} queries  "
          f"| {library[t]['caption'][:52]}")
print()
if LIB_E is not None:
    R = LIB_E[sorted(set(int(p) for p in picks))]
    v = vendi_score(R)
    nnl = nn_distances(R)
    print()
    print(f"  Vendi over the REACHED voices, in the same MioCodec space S7 used:")
    print(f"    normalised Vendi {v:.3f} -> ~{v*len(R):.0f} effective voices "
          f"from {len(R)} reached")
    print(f"    nn distance median {np.median(nnl):.3f} min {np.min(nnl):.3f}")
    print()
    print(f"  EFFECTIVE SIZE, like for like")
    print(f"    S8 library retrieval : ~{v*len(R):.0f} effective voices")
    print(f"    S7 minted catalog    : ~14 effective voices (38 accepted of 80)")
else:
    print(f"  EFFECTIVE LIBRARY SIZE = {reached} distinct voices reachable")
print("=" * 78)
