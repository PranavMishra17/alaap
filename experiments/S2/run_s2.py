"""
S2 — the whole loop, end to end.

    corpus -> measure -> bin -> grounded caption
                                      |
                                      v
              speaker vectors -> SpeakerSpace -> retrieval mapper
                                      |
              novel description ------+--> mint -> render -> RE-MEASURE
                                                              |
                                                     adherence score

This is the first time every piece runs together, and the first honest
answer to "does a description actually produce the voice it describes?".

It also produces the S0 BASELINE SCORECARD that PHASE-00 exit criterion
X0.2 demands, so every later improvement has a number to beat.

Because the captions are GROUNDED (generated from measured bins, never from
an audio-LM's impression), adherence is checkable by arithmetic rather than
by an LLM judge. That is the whole point of measure-first.

Outputs -> experiments/S2/out/
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.data import stream_clips, SR
from alaap.acoustics import measure, Binner, adherence_error, BIN_LABELS
from alaap.captions import caption_from_bins, target_bins_from_text
from alaap.geometry import SpeakerSpace
from alaap.metrics import verification_stats, vendi_score, nn_distances
from alaap.mapper import TextEncoder, RetrievalMapper

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
AUD = os.path.join(OUT, "audio")
os.makedirs(AUD, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=600, help="corpus clips")
ap.add_argument("--per-speaker", type=int, default=3)
ap.add_argument("--corpus", default="libritts_r_train")
ap.add_argument("--model", default=None, help="defaults to alaap.encoder.DEFAULT_MODEL")
ap.add_argument("--candidates", type=int, default=3, help="voices minted per description")
ap.add_argument("--line", default="The mountains remember every footstep.")
ap.add_argument("--skip-render", action="store_true")
args = ap.parse_args()
from alaap.encoder import DEFAULT_MODEL as _DM
if args.model is None:
    args.model = _DM

# The fixed evaluation set (PHASE-00 section 3.2). Deliberately spans the
# range, INCLUDING low-density regions -- E3/RESEARCH/12 warn that "elderly",
# "raspy", "very low-pitched" are exactly where quality cliffs live.
EVAL_DESCRIPTIONS = [
    "a very deep, gravelly voice, speaking slowly and almost monotone",
    "a bright, high-pitched voice, highly animated and speaking quickly",
    "a warm, dark-timbred voice with a faint rasp, at a steady pace",
    "a very clear, crystalline voice, expressive and measured",
    "a low, smooth voice, unhurried and gently inflected",
    "a high, harsh voice racing through the words",
]

STAGE = time.time()
def stage(msg):
    print(f"\n[{time.time()-STAGE:6.0f}s] {msg}", flush=True)

# ------------------------------------------------------- 1. corpus + measure
stage(f"1. corpus: {args.n} clips from {args.corpus}")
# The cache key MUST include the model. It did not, and a 1.7B run silently
# loaded 0.6B embeddings; every render then failed with a 2048-vs-1024
# mismatch. That failed LOUDLY only by luck -- two models with the same
# enc_dim would have produced quietly wrong numbers. alaap.encoder's
# EmbeddingCache already hashes model_id for exactly this reason; this
# script's ad-hoc cache did not.
_mtag = args.model.split("/")[-1].replace(".", "")
cache = os.path.join(
    OUT, f"corpus_{args.corpus}_{args.n}_{args.per_speaker}_{_mtag}.npz")
if os.path.exists(cache):
    d = np.load(cache, allow_pickle=True)
    Z = d["Z"].astype(np.float64); ids = list(d["ids"])
    attrs_d = json.loads(str(d["attrs"])); texts = list(d["texts"])
    print(f"      cached: {Z.shape}, {len(set(ids))} speakers")
else:
    clips = stream_clips(corpus=args.corpus, n=args.n,
                         per_speaker=args.per_speaker, min_dur=3.0, max_dur=12.0)
    print(f"      measuring acoustics ({len(clips)} clips, ~0.9s each)...")
    t0 = time.time()
    attrs_d, texts, ids = [], [], []
    for i, c in enumerate(clips):
        attrs_d.append(measure(c.wav, c.text or "").to_dict())
        texts.append(c.text or ""); ids.append(c.speaker_id)
        if (i + 1) % 100 == 0:
            print(f"        {i+1}/{len(clips)} | {time.time()-t0:.0f}s", flush=True)
    from alaap.encoder import SpeakerEncoder
    enc = SpeakerEncoder(args.model)
    Z = enc.embed_many([c.wav for c in clips], progress_every=200).astype(np.float64)
    del enc
    import torch; torch.cuda.empty_cache()
    np.savez_compressed(cache, Z=Z.astype(np.float32), ids=np.array(ids),
                        attrs=json.dumps(attrs_d), texts=np.array(texts, dtype=object))

from alaap.acoustics import Attributes
attrs = [Attributes(**a) for a in attrs_d]
print(f"      {len(attrs)} measured, {len(set(ids))} speakers")

# -------------------------------------------------- 2. bin + caption + space
stage("2. binning, captioning, fitting the speaker space")
binner = Binner.fit(attrs)
binner.save(os.path.join(OUT, f"binner_{args.corpus}_{_mtag}.json"))

# Build (caption, vector) pairs.
#
# Two modes, and the choice matters:
#
#   per-speaker averaging  needs TRUSTWORTHY speaker labels. Good on
#                          LibriTTS-R; E4 showed GLOBE_V2's labels are not
#                          reliable enough (ECAPA EER 20% vs 2.46%).
#   per-CLIP               needs no labels at all. Each clip is its own
#                          (caption, vector) pair.
#
# The mapper only ever needs pairs, never speaker identity -- so per-clip
# sidesteps the label problem entirely and unlocks GLOBE_V2's 23,519 voices
# for VOCAL RANGE, which is what LibriTTS-R actually lacks (S2 RESULTS
# section 2, point 4: clean audiobook read speech has no gravelly or aged
# voices to retrieve, and percentile binning HIDES that).
uniq = sorted(set(ids)); idsa = np.asarray(ids)
PER_CLIP = args.per_speaker == 1 or len(uniq) > 0.8 * len(ids)
spk_vec, spk_cap, spk_bins = [], [], []
if PER_CLIP:
    print(f"      per-CLIP pairs ({len(uniq)} distinct ids / {len(ids)} clips) "
          f"-- no speaker labels needed")
    for i in range(len(ids)):
        b = binner.bin_one(attrs[i])
        spk_vec.append(Z[i]); spk_bins.append(b)
        spk_cap.append(caption_from_bins(b, seed=i))
else:
    print(f"      per-SPEAKER pairs ({len(uniq)} speakers)")
    for s in uniq:
        m = idsa == s
        v = Z[m].mean(0)
        sub = [attrs[i] for i in np.where(m)[0]]
        avg = Attributes(**{f: float(np.mean([getattr(a, f) for a in sub]))
                            for f in attrs[0].to_dict()})
        b = binner.bin_one(avg)
        spk_vec.append(v); spk_bins.append(b)
        spk_cap.append(caption_from_bins(b, seed=abs(hash(s)) % 10000))
spk_vec = np.stack(spk_vec)
print(f"      {len(spk_vec)} speaker identities with grounded captions")
print(f"      e.g. {spk_cap[0]}")

space = SpeakerSpace.fit(spk_vec, n_components=min(150, len(spk_vec) - 1))
space.save(os.path.join(OUT, f"speaker_space_{args.corpus}_{_mtag}.npz"))
print(f"      {space}")

# ------------------------------------------------------------- 3. mapper
stage("3. fitting the retrieval mapper")
te = TextEncoder()
mapper = RetrievalMapper(space, te, pca_dims=50).fit(spk_cap, spk_vec)
mapper.save(os.path.join(OUT, f"mapper_{args.corpus}_{_mtag}.npz"))
print(f"      fitted on {len(spk_cap)} (caption, vector) pairs, "
      f"text dim {te.dim}, PCA {mapper.k}")

# ------------------------------------------------ 4. mint across the dial
stage("4. minting from the fixed evaluation set")
NOVELTIES = [0.0, 0.5, 1.0]
minted = []
for di, desc in enumerate(EVAL_DESCRIPTIONS):
    for nv in NOVELTIES:
        cands = mapper.mint_many(desc, n=args.candidates, novelty=nv)
        for ci, m in enumerate(cands):
            minted.append({"desc_i": di, "desc": desc, "novelty": nv,
                           "cand": ci, "res": m})
print(f"      {len(minted)} voices minted "
      f"({len(EVAL_DESCRIPTIONS)} descriptions x {len(NOVELTIES)} novelty "
      f"x {args.candidates} candidates)")

# diversity per (description, novelty) -- the anti-mode-collapse check
div = {}
for di in range(len(EVAL_DESCRIPTIONS)):
    for nv in NOVELTIES:
        V = np.stack([m["res"].working for m in minted
                      if m["desc_i"] == di and m["novelty"] == nv])
        if len(V) >= 2:
            div[f"d{di}_n{nv}"] = {"vendi": vendi_score(V),
                                   "nn_spread": float(np.mean(nn_distances(V)))}
by_nov = {nv: float(np.mean([v["vendi"] for kk, v in div.items()
                             if kk.endswith(f"n{nv}")])) for nv in NOVELTIES}
print(f"      normalised Vendi by novelty: "
      + "  ".join(f"{nv}->{by_nov[nv]:.3f}" for nv in NOVELTIES))

# separability between descriptions (do different descriptions -> different voices?)
allV = np.stack([m["res"].working for m in minted if m["novelty"] == 0.5])
allL = [str(m["desc_i"]) for m in minted if m["novelty"] == 0.5]
from alaap.metrics import cluster_separability
sil = cluster_separability(allV, allL)
print(f"      cluster separability across descriptions (silhouette): {sil:.3f}")

# ------------------------------------------------------------- 5. render
results_render = None
if not args.skip_render:
    stage("5. rendering + re-measuring (the adherence loop)")
    from alaap.renderer import Qwen3BaseRenderer, load_backend
    import soundfile as sf
    r = Qwen3BaseRenderer(args.model); load_backend(r, is_public_deployment=True)
    rows = []
    t0 = time.time()
    todo = [m for m in minted if m["cand"] == 0]     # one candidate per cell
    for i, m in enumerate(todo):
        try:
            a = r.render_from_vector(m["res"].vector.astype(np.float32),
                                     args.line, "en")
        except Exception as e:
            print(f"      !! render failed: {e}"); continue
        at = measure(a.wav, args.line, a.sample_rate)
        tgt = target_bins_from_text(m["desc"])
        adh = adherence_error(tgt, at, binner)
        fn = f"{args.corpus}_{_mtag}_d{m['desc_i']}_n{m['novelty']}.wav"
        sf.write(os.path.join(AUD, fn), a.wav, a.sample_rate)
        rows.append({"desc_i": m["desc_i"], "desc": m["desc"],
                     "novelty": m["novelty"], "file": fn,
                     "target_bins": tgt, "got_bins": binner.bin_one(at),
                     "measured": at.to_dict(),
                     "exact_match_rate": adh["exact_match_rate"],
                     "mean_bin_distance": adh["mean_bin_distance"],
                     "anchor_similarity": m["res"].anchor_similarity})
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            print(f"      {i+1}/{len(todo)} | {el/60:.1f} min | "
                  f"eta {el/(i+1)*(len(todo)-i-1)/60:.1f} min", flush=True)
    results_render = rows
    print(f"      rendered {len(rows)} in {(time.time()-t0)/60:.1f} min")

# ------------------------------------------------------------- 6. report
stage("6. report")
summary = {
    "corpus": args.corpus, "n_clips": len(attrs), "n_speakers": len(uniq),
    "space_stats": space.stats.__dict__,
    "mapper": {"n_pairs": len(spk_cap), "text_dim": te.dim, "pca": mapper.k},
    "diversity_by_novelty": by_nov,
    "cluster_separability_silhouette": sil,
    "eval_descriptions": EVAL_DESCRIPTIONS,
    "renders": results_render,
}
json.dump(summary, open(os.path.join(OUT, f"results_{args.corpus}_{_mtag}.json"), "w"),
          indent=2, default=float)

print()
print("=" * 90)
print("S2 BASELINE SCORECARD")
print("=" * 90)
print(f"  corpus            {args.corpus}, {len(attrs)} clips, {len(uniq)} speakers")
print(f"  speaker space     dim {space.stats.dim}, eff_rank "
      f"{space.stats.effective_rank:.1f}, identity_frac "
      f"{space.stats.identity_fraction:.3f}")
print(f"  mapper            {len(spk_cap)} grounded (caption, vector) pairs")
print()
print("  DIVERSITY (normalised Vendi; RESEARCH/06 target >= 0.35, collapse < 0.10)")
for nv in NOVELTIES:
    v = by_nov[nv]
    tag = "OK" if v >= 0.35 else ("ALARM" if v >= 0.20 else "COLLAPSE")
    print(f"    novelty {nv:<4} {v:.3f}   [{tag}]")
print(f"  SEPARABILITY      silhouette {sil:.3f}   "
      f"[{'OK' if sil >= 0.15 else 'WEAK'}]  (target >= 0.15)")
if results_render:
    print()
    print("  ADHERENCE (re-measured; the reversibility payoff)")
    print(f"    {'novelty':<9} {'exact-match':>12} {'mean bin dist':>15}")
    for nv in NOVELTIES:
        sub = [r for r in results_render if r["novelty"] == nv]
        if sub:
            print(f"    {nv:<9} {np.mean([s['exact_match_rate'] for s in sub]):>12.3f} "
                  f"{np.mean([s['mean_bin_distance'] for s in sub]):>15.3f}")
    print()
    print("  PER-DESCRIPTION (novelty 0.5)")
    for r in [x for x in results_render if x["novelty"] == 0.5]:
        print(f"    \"{r['desc'][:52]}\"")
        print(f"      match {r['exact_match_rate']:.2f}  bin-dist "
              f"{r['mean_bin_distance']:.2f}  -> {r['file']}")
print()
print(f"  audio -> {AUD}")
print("=" * 90)
