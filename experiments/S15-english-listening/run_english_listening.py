"""
S15 — is the ENGLISH uniqueness floor right? (S11, on the other backend)

`S11` put the borrowed 0.30 floor in front of a listener on the Indic path and
it failed: voices 0.323 apart were heard as the same person half the time. The
fix — 0.45 — was deliberately NOT applied to English, because `ADR-011` says a
floor is a claim about perception and may not be transferred between spaces
without being re-measured. Qwen3's working space is 2048-d before projection,
a different geometry entirely, and a cosine distance in it does not mean what
one means in MioCodec's 128-d space.

So the English floor is currently unvalidated in BOTH directions: nothing says
0.30 is too low there, and nothing says it is high enough.

This is S11's design, unchanged, on Qwen3:

    positive control   the SAME voice, two different sentences  -> same person
    negative control   two different REAL corpus speakers       -> different
    test pairs         minted voices bracketing the 0.30 floor

Every pair is two DIFFERENT sentences, controls included, so the positive
control is not the same waveform twice. Pairs and A/B order are shuffled; file
names carry nothing; the key is written separately.

One difference from S11 worth noting: E11's catalog had a minimum observed
uniqueness of 0.359, so nothing it accepted ever sat as close as the 0.323
pairs that failed in Indic. This deliberately MINTS pairs down at the floor
anyway -- the question is what the floor permits, not what one run happened to
produce.

    envs/qwen3/Scripts/python.exe experiments/S15-english-listening/run_english_listening.py
"""
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.acoustics import Attributes, Binner
from alaap.captions import caption_from_bins
from alaap.catalog import sample_cells
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
CACHE = "experiments/S2/out/corpus_globe_v2_2500_1_Qwen3-TTS-12Hz-17B-Base.npz"
UNIQ_FLOOR = 0.30
SEED = 13
SENTENCES = [
    "The mountains remember every footstep, even the ones you regret.",
    "I told them the gate would not hold, and nobody listened.",
    "There is a light on in the kitchen, which means she waited up.",
    "We should leave before the tide turns against us.",
]

print("[1/4] rebuilding the English mapper")
d = np.load(CACHE, allow_pickle=True)
Z = d["Z"].astype(np.float64)
A = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
ids = [str(x) for x in d["ids"]]
k = min(len(Z), len(A), len(ids))
Z, A, ids = Z[:k], A[:k], ids[:k]
binner = Binner.fit(A)
bins = [binner.bin_one(a) for a in A]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
space = SpeakerSpace.fit(Z, n_components=150)
mapper = RetrievalMapper(space, TextEncoder(), pca_dims=space.components.shape[0],
                         retrieval="hybrid").fit(caps, Z, anchor_bins=bins)
E_corpus = space.encode(Z)
print(f"      {len(Z)} clips, {len(set(ids))} speakers | {space}")

print("[2/4] minting a pool and choosing pairs by distance")
cells = sample_cells(60, 0)
pool = []
for i, c in enumerate(cells):
    m = mapper.mint(caption_from_bins(c, seed=i), novelty=0.0, seed=i, top_k=2)
    pool.append({"v": m.vector, "e": space.encode(m.vector)[0]})
P = np.vstack([p["e"] for p in pool])
Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
D = 1.0 - Pn @ Pn.T
np.fill_diagonal(D, np.inf)


def pick(lo, hi, used):
    best, bd, mid = None, np.inf, (lo + hi) / 2
    for a in range(len(pool)):
        for b in range(a + 1, len(pool)):
            if a in used or b in used or not (lo <= D[a, b] <= hi):
                continue
            if abs(D[a, b] - mid) < bd:
                best, bd = (a, b), abs(D[a, b] - mid)
    return best


rng = np.random.default_rng(SEED)
used, pairs = set(), []

En = E_corpus / np.maximum(np.linalg.norm(E_corpus, axis=1, keepdims=True), 1e-12)
first = {}
for i, s in enumerate(ids):
    first.setdefault(s, i)
sidx = list(first.values())
for _ in range(2):
    a, b = rng.choice(len(sidx), 2, replace=False)
    ia, ib = sidx[int(a)], sidx[int(b)]
    pairs.append({"kind": "negative control (two real speakers)", "truth": "different",
                  "dist": float(1 - En[ia] @ En[ib]), "va": Z[ia], "vb": Z[ib]})
for kk in (0, 1):
    pairs.append({"kind": "positive control (one voice, two sentences)", "truth": "same",
                  "dist": 0.0, "va": pool[kk]["v"], "vb": pool[kk]["v"]})
    used.add(kk)
for lo, hi, label in [(UNIQ_FLOOR, 0.35, "just above the 0.30 floor"),
                      (UNIQ_FLOOR, 0.35, "just above the 0.30 floor"),
                      (0.45, 0.55, "at the Indic floor (0.45)"),
                      (0.62, 0.95, "far apart")]:
    pr = pick(lo, hi, used)
    if pr is None:
        print(f"      no pair in [{lo}, {hi}] -- skipping '{label}'")
        continue
    a, b = pr
    used.update((a, b))
    pairs.append({"kind": f"test — {label}", "truth": "different (per the metric)",
                  "dist": float(D[a, b]), "va": pool[a]["v"], "vb": pool[b]["v"]})
print("      distances: " + ", ".join(f"{p['dist']:.2f}" for p in pairs))

print("[3/4] loading the renderer")
from alaap.renderer import Qwen3BaseRenderer, load_backend
# load_backend takes an INSTANCE and enforces the licence gate on it (I3/I4),
# it does not construct one from a name.
r = load_backend(Qwen3BaseRenderer("Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
print(f"      {r.backend_id}@{r.backend_version}")

print("[4/4] rendering")
import soundfile as sf
order = rng.permutation(len(pairs))
key = []
for slot, pi in enumerate(order, 1):
    p = pairs[int(pi)]
    s1, s2 = rng.choice(len(SENTENCES), 2, replace=False)
    va, vb = (p["va"], p["vb"]) if rng.random() < 0.5 else (p["vb"], p["va"])
    for tag, vec, si in (("A", va, s1), ("B", vb, s2)):
        au = r.render_from_vector(np.asarray(vec, np.float32), SENTENCES[int(si)], "en")
        sf.write(os.path.join(OUT, f"pair{slot:02d}_{tag}.wav"), au.wav, au.sample_rate)
    key.append({"pair": slot, "kind": p["kind"], "truth": p["truth"],
                "working_distance": round(p["dist"], 4)})
    print(f"      pair{slot:02d}  d={p['dist']:.3f}  {p['kind']}", flush=True)

json.dump(key, io.open(os.path.join(OUT, "ANSWER-KEY.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
sheet = ["# S15 — English listening set", "",
         "Same question as before, for each pair:", "",
         "> **Are these two clips the same person, or two different people?**", "",
         "Not *do they sound different* — two recordings of one person always do.",
         "Every pair is two different English sentences, so you must judge the voice.",
         "Some pairs have known answers and check the test itself.", "",
         "| pair | same person? |", "|---|---|"]
sheet += [f"| {i:02d} | |" for i in range(1, len(key) + 1)]
sheet += ["", "Answer key in `out/ANSWER-KEY.json` — **do not open it first.**"]
io.open(os.path.join(OUT, "SCORESHEET.md"), "w", encoding="utf-8").write("\n".join(sheet))
print(f"\n  {len(key)} pairs -> {OUT}")
