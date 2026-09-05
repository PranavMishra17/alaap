"""
E4 — Calibrate C_same / C_diff, and benchmark the conditioning encoder
     against a real ASV model on identical audio.

RESEARCH/06 is emphatic that no defensible universal cosine threshold exists;
published values are encoder-specific and non-comparable. Every identity
threshold in the project (PHASE-01 X1.1, the catalog uniqueness rule, the S1
exit criterion) is meaningless until these distributions are measured.

Three things this run establishes:

  1. C_same and C_diff for the Qwen3-TTS conditioning encoder, in raw space
     and in the working space E3's geometry findings imply.
  2. The same, for an INDEPENDENT ECAPA-TDNN ASV encoder on the SAME clips --
     which is both the eval-axis-2 scorer (never mark your own homework) and
     a reference for how much speaker information the TTS encoder actually
     carries.
  3. A disjoint fit/test split. The first E4 run had 169/169 speakers leak
     because GLOBE_V2 streams in a stable order and both calls started at
     row 0. `skip` fixes it.

Outputs -> experiments/E4/out/
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.data import stream_clips, SR
from alaap.encoder import SpeakerEncoder, IndependentSV, DEFAULT_MODEL
from alaap.geometry import SpeakerSpace
from alaap.metrics import verification_stats, nn_distances, _cos

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--n", type=int, default=1200)
ap.add_argument("--per-speaker", type=int, default=20)
ap.add_argument("--skip", type=int, default=0, help="rows to skip")
ap.add_argument("--corpus", default="libritts_r_train")
ap.add_argument("--fit-from", default="experiments/E3/out/embeddings.npy")
args = ap.parse_args()

# ------------------------------------------------------------------ 1. data
print(f"[1/5] streaming {args.n} clips from {args.corpus} (<= {args.per_speaker}/speaker, skip={args.skip})")
t0 = time.time()
clips = stream_clips(corpus=args.corpus, n=args.n, per_speaker=args.per_speaker,
                     skip=args.skip)
ids = [c.speaker_id for c in clips]
uniq = sorted(set(ids))
multi = sum(1 for s in uniq if ids.count(s) >= 2)
print(f"      {len(clips)} clips | {len(uniq)} speakers | {multi} with >=2 utts "
      f"| {time.time()-t0:.0f}s")

# leakage check against the fit set
e3_meta = "experiments/E3/out/meta.json"
leak = 0
if os.path.exists(e3_meta):
    fit_ids = {m["speaker_id"] for m in json.load(open(e3_meta))}
    leak = len(fit_ids & set(ids))
print(f"      speaker overlap with fit set: {leak} "
      f"({'CLEAN' if leak == 0 else 'LEAKAGE'})")

# --------------------------------------------------------------- 2. encoders
print(f"[2/5] embedding with BOTH encoders")
t0 = time.time()
enc = SpeakerEncoder(args.model)
Zq = enc.embed_many([c.wav for c in clips])
print(f"      Qwen3  {Zq.shape} dim={enc.dim} vram={enc.vram_gb():.2f}GB "
      f"| {time.time()-t0:.0f}s")
del enc
import torch; torch.cuda.empty_cache()

t0 = time.time()
sv = IndependentSV()
Ze = sv.embed_many([c.wav for c in clips], sr=SR)
print(f"      ECAPA  {Ze.shape} | {time.time()-t0:.0f}s")
del sv; torch.cuda.empty_cache()

np.savez_compressed(os.path.join(OUT, "embeddings.npz"), Zq=Zq, Ze=Ze,
                    ids=np.array(ids))

# ------------------------------------------------------------- 3. fit space
print(f"[3/5] fitting SpeakerSpace on {args.fit_from}")
Zfit = np.load(args.fit_from).astype(np.float64)
space = SpeakerSpace.fit(Zfit, n_components=256)
print(f"      {space}")
space.save(os.path.join(OUT, "speaker_space.npz"))
space_e = SpeakerSpace.fit(Ze.astype(np.float64), n_components=128)  # ECAPA, self-fit
print(f"      ECAPA space: {space_e}")

# ------------------------------------------------------------- 4. calibrate
print("[4/5] calibrating")
variants = {
    "qwen3 raw":            Zq.astype(np.float64),
    "qwen3 centred":        Zq.astype(np.float64) - space.mean,
    "qwen3 centred+scaled": space.encode(Zq),
    "qwen3 pca50":          space.to_pca(space.encode(Zq))[:, :50],
    "ECAPA raw":            Ze.astype(np.float64),
    "ECAPA centred+scaled": space_e.encode(Ze),
}
results = {}
for name, X in variants.items():
    vs = verification_stats(X, ids)
    nn = nn_distances(X)
    results[name] = {**vs.to_dict(), "nn_dist_mean": float(nn.mean()),
                     "nn_dist_max": float(nn.max())}
    print(f"      {name:<22} C_same={vs.same_mean:+.4f}  C_diff={vs.diff_mean:+.4f}  "
          f"d'={vs.d_prime:5.2f}  EER={vs.eer*100:5.2f}%")

# ------------------------------------------------------------- 5. report
qbest = min([k for k in results if k.startswith("qwen3")], key=lambda k: results[k]["eer"])
ebest = min([k for k in results if k.startswith("ECAPA")], key=lambda k: results[k]["eer"])
summary = {
    "model": args.model, "n_clips": len(clips), "n_speakers": len(uniq),
    "n_speakers_multi": multi, "skip": args.skip, "leakage_speakers": leak,
    "qwen3_dim": int(Zq.shape[1]), "ecapa_dim": int(Ze.shape[1]),
    "space_stats": space.stats.__dict__,
    "by_variant": results,
    "best_qwen3": qbest, "best_ecapa": ebest,
    "eer_gap_qwen3_vs_ecapa_pp": (results[qbest]["eer"] - results[ebest]["eer"]) * 100,
    "OPERATING_POINT_conditioning": {
        "space": qbest, "cosine_threshold": results[qbest]["eer_threshold"],
        "C_same": results[qbest]["same_mean"], "C_diff": results[qbest]["diff_mean"],
        "eer_pct": results[qbest]["eer"] * 100},
    "OPERATING_POINT_eval_axis2": {
        "space": ebest, "cosine_threshold": results[ebest]["eer_threshold"],
        "C_same": results[ebest]["same_mean"], "C_diff": results[ebest]["diff_mean"],
        "eer_pct": results[ebest]["eer"] * 100},
}
json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
idsa = np.asarray(ids)
n = len(variants)
fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.4 * ((n + 1) // 2), 7.2))
for ax, (name, X) in zip(axes.ravel(), variants.items()):
    C = _cos(np.asarray(X, float)); iu = np.triu_indices(len(X), 1)
    m = idsa[iu[0]] == idsa[iu[1]]; s = C[iu]
    ax.hist(s[~m], bins=70, alpha=.6, density=True, label="different", color="tab:red")
    ax.hist(s[m], bins=70, alpha=.6, density=True, label="same", color="tab:green")
    ax.axvline(results[name]["eer_threshold"], color="k", ls="--", lw=1)
    ax.set_title(f"{name}\nd'={results[name]['d_prime']:.2f}  "
                 f"EER={results[name]['eer']*100:.1f}%", fontsize=10)
    ax.set_xlabel("cosine"); ax.legend(fontsize=7); ax.grid(alpha=.3)
for ax in axes.ravel()[n:]:
    ax.axis("off")
plt.suptitle(f"E4 — genuine vs impostor cosine | {len(uniq)} speakers, "
             f"{len(clips)} clips | conditioning encoder vs independent ASV", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "e4_calibration.png"), dpi=140)

print()
print("=" * 84)
print("E4 RESULT")
print("=" * 84)
print(f"  {len(clips)} clips | {len(uniq)} speakers ({multi} multi-utt) | leakage={leak}")
print()
_dp = "d-prime"
print(f"  {'variant':<22} {'C_same':>9} {'C_diff':>9} {'sep':>8} {_dp:>7} {'EER':>8}")
print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*8}")
for k, r in results.items():
    print(f"  {k:<22} {r['same_mean']:>+9.4f} {r['diff_mean']:>+9.4f} "
          f"{r['separation']:>+8.4f} {r['d_prime']:>7.2f} {r['eer']*100:>7.2f}%")
print()
print(f"  conditioning encoder best : {qbest}  EER {results[qbest]['eer']*100:.2f}%")
print(f"  independent ASV best      : {ebest}  EER {results[ebest]['eer']*100:.2f}%")
print(f"  GAP                       : {summary['eer_gap_qwen3_vs_ecapa_pp']:+.2f} pp")
print()
print(f"  eval-axis-2 operating point: {ebest}, cosine >= "
      f"{results[ebest]['eer_threshold']:.4f}")
print(f"  artefacts -> {OUT}")
print("=" * 84)
