"""
E9 — Do EXTREME voices hold together as well as ordinary ones?

RESEARCH/12 found that averaging inside sparse regions of speaker space costs
**+60% relative WER** (10.94% vs 6.83%), and warned that "elderly", "raspy"
and "very low-pitched" are exactly the low-density regions where quality
cliffs live. RESEARCH/10 adds that nobody has published identity-drift numbers
for heavy stylisation at all -- every published figure covers the six
canonical emotions instead.

That matters because the project's promised voice range (scope section 5) is
"human + heavy stylization -- aged, raspy, whispered, breathy, menacing,
theatrical". If identity collapses precisely at the extremes, the promise is
undeliverable and we should know now rather than at S8.

DESIGN. Rather than trying to *induce* stylisation -- which this backend has
no channel for -- we mint from descriptions that sit at the EXTREME end of
each measured attribute and compare them against descriptions at the CENTRE:

    EXTREME   "a very deep voice, very rough and gravelly, ..."
    CENTRAL   "a mid-range voice, clear-toned, at a steady pace, ..."

and measure, for each minted identity:

    consistency  pairwise ECAPA similarity across N different sentences
                 (the property that makes this a voice LIBRARY at all)
    drift        intended vector vs the vector re-extracted from the render
    adherence    re-measured attributes vs the description's target bins

If the extreme group is materially worse on any of these, the voice range is
narrower than promised.

Runs on 0.6B deliberately: the cached GLOBE corpus was embedded with 0.6B, and
mixing encoders would invalidate the comparison. E10 switched the default to
1.7B; this passes --model explicitly.

Outputs -> experiments/E9/out/
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.acoustics import Attributes, Binner, measure, adherence_error
from alaap.captions import caption_from_bins, target_bins_from_text
from alaap.geometry import SpeakerSpace
from alaap.encoder import IndependentSV
from alaap.mapper import TextEncoder, RetrievalMapper
from alaap.metrics import normalised_similarity

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
AUD = os.path.join(OUT, "audio")
os.makedirs(AUD, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
ap.add_argument("--corpus-cache",
                default="experiments/S2/out/corpus_globe_v2_2500_1.npz")
ap.add_argument("--binner", default="experiments/S2/out/binner_globe_v2.json")
ap.add_argument("--novelty", type=float, default=0.0)
ap.add_argument("--lines", type=int, default=3)
args = ap.parse_args()

SCRIPT = ["The mountains remember every footstep.",
          "It's cold today, colder than anyone promised.",
          "Bring me the lantern and say nothing more."][:args.lines]

# Extremes deliberately stack the OUTER bin of several attributes at once --
# that is what a character description actually looks like.
EXTREME = [
    ("deep+gravelly",  "a very deep voice, very rough and gravelly, speaking very slowly"),
    ("high+piercing",  "a very high voice, very bright and edgy, speaking very rapidly"),
    ("dark+monotone",  "an extremely low voice, muffled and warm, almost monotone"),
    ("harsh+animated", "a piercingly high voice, harsh and rasping, highly animated"),
]
CENTRAL = [
    ("mid+clear",   "a mid-range voice, clear-toned, at a steady pace"),
    ("mid+even",    "a moderately pitched voice, balanced in timbre, moderately expressive"),
    ("mid+natural", "a middle-register voice, with a clean tone, at a natural, even tempo"),
    ("mid+plain",   "a mid-range voice, smooth and clear, with natural intonation"),
]

# ------------------------------------------------------------- 1. mapper
print(f"[1/3] rebuilding the mapper from {args.corpus_cache}")
d = np.load(args.corpus_cache, allow_pickle=True)
Z = d["Z"].astype(np.float64)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
binner = Binner.load(args.binner)
caps = [caption_from_bins(binner.bin_one(a), seed=i) for i, a in enumerate(attrs)]
space = SpeakerSpace.fit(Z, n_components=150)
mapper = RetrievalMapper(space, TextEncoder(), pca_dims=50).fit(caps, Z)
print(f"      {len(caps)} pairs | {space}")

# --------------------------------------------------------------- 2. render
from alaap.renderer import Qwen3BaseRenderer, load_backend
import soundfile as sf
r = Qwen3BaseRenderer(args.model); load_backend(r, is_public_deployment=True)
sv = IndependentSV()

groups = [("extreme", EXTREME), ("central", CENTRAL)]
total = sum(len(g[1]) for g in groups) * len(SCRIPT)
print(f"[2/3] {total} renders ({len(EXTREME)} extreme + {len(CENTRAL)} central "
      f"x {len(SCRIPT)} lines)")

rows, t0, n = [], time.time(), 0
for gname, items in groups:
    for tag, desc in items:
        m = mapper.mint(desc, novelty=args.novelty, seed=0)
        v = m.vector
        per_line = []
        for li, line in enumerate(SCRIPT):
            try:
                au = r.render_from_vector(v.astype(np.float32), line, "en")
            except Exception as e:
                print(f"      !! {tag} line {li}: {e}"); continue
            back = r.extract_vector(au.wav, au.sample_rate).astype(np.float64)
            a_, b_ = space.encode(v)[0], space.encode(back)[0]
            per_line.append({
                "ecapa": sv.embed(au.wav, sr=au.sample_rate),
                "drift": float(a_ @ b_ / max(np.linalg.norm(a_)*np.linalg.norm(b_), 1e-12)),
                "attrs": measure(au.wav, line, au.sample_rate),
                "dur": len(au.wav) / au.sample_rate})
            if li == 0:
                sf.write(os.path.join(AUD, f"{gname}_{tag}.wav"), au.wav, au.sample_rate)
            n += 1
            if n % 6 == 0:
                el = time.time() - t0
                print(f"      {n}/{total} | {el/60:.1f} min | "
                      f"eta {el/n*(total-n)/60:.1f} min", flush=True)
        if per_line:
            rows.append({"group": gname, "tag": tag, "desc": desc,
                         "anchor_sim": m.anchor_similarity, "renders": per_line})

# -------------------------------------------------------------- 3. analyse
print("[3/3] measuring")


def cos(x, y):
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


out = {}
for gname, _ in groups:
    sub = [x for x in rows if x["group"] == gname]
    cons, drifts, adh, durs = [], [], [], []
    for x in sub:
        E = [p["ecapa"] for p in x["renders"]]
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                cons.append(cos(E[i], E[j]))
        drifts += [p["drift"] for p in x["renders"]]
        durs += [p["dur"] for p in x["renders"]]
        tgt = target_bins_from_text(x["desc"])
        for p in x["renders"]:
            adh.append(adherence_error(tgt, p["attrs"], binner))
    out[gname] = {
        "n_identities": len(sub), "n_renders": len(drifts),
        "consistency_mean": float(np.mean(cons)) if cons else None,
        "consistency_min": float(np.min(cons)) if cons else None,
        "consistency_norm": normalised_similarity(float(np.mean(cons)), "ecapa")
        if cons else None,
        "drift_mean": float(np.mean(drifts)), "drift_min": float(np.min(drifts)),
        "exact_match": float(np.mean([a["exact_match_rate"] for a in adh])),
        "bin_distance": float(np.mean([a["mean_bin_distance"] for a in adh])),
        "anchor_sim": float(np.mean([x["anchor_sim"] for x in sub])),
        "duration_mean": float(np.mean(durs)),
    }

json.dump({"model": args.model, "novelty": args.novelty, "groups": out},
          open(os.path.join(OUT, "results.json"), "w"), indent=2)

print()
print("=" * 84)
print("E9 RESULT — do EXTREME voices hold together as well as ordinary ones?")
print("=" * 84)
e, c = out["extreme"], out["central"]
print(f"  {'':<28} {'EXTREME':>14} {'CENTRAL':>14} {'delta':>10}")
print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*10}")
def row(lbl, k, fmt="{:.4f}", better_high=True):
    a, b = e.get(k), c.get(k)
    if a is None or b is None:
        return
    d = a - b
    mark = ""
    if abs(d) > (0.05 if abs(b) < 2 else 0.3):
        worse = (d < 0) if better_high else (d > 0)
        mark = "  <-- WORSE" if worse else "  (better)"
    print(f"  {lbl:<28} {fmt.format(a):>14} {fmt.format(b):>14} "
          f"{fmt.format(d):>10}{mark}")
row("identity consistency", "consistency_mean")
row("  normalised (0=diff spk)", "consistency_norm", "{:.3f}")
row("  worst case", "consistency_min")
row("vocoder drift", "drift_mean")
row("  worst case", "drift_min")
row("adherence exact-match", "exact_match", "{:.3f}")
row("adherence bin-distance", "bin_distance", "{:.3f}", better_high=False)
row("retrieval anchor similarity", "anchor_sim", "{:.4f}")
row("render duration s", "duration_mean", "{:.2f}")
print()
print(f"  reference: ECAPA C_same 0.6988 / C_diff 0.2011 on real speech")
print(f"  audio -> {AUD}")
print("=" * 84)
