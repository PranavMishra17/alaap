"""
E2 — Vocoder drift, and does a SYNTHESIZED voice hold together?

Scope section 4.3 claims a Tier-1 vector reproduces a voice "exactly, by
construction". RESEARCH/12 says that is false at the output: the embedding
extracted from rendered audio "often differs substantially from the x-vector
at the vocoder input" (Panariello et al., Interspeech 2023). Nobody has
measured it for Qwen3-TTS.

This run measures three things at once:

  1. DRIFT      cos(intended vector, vector re-extracted from the render).
                How far is the voice you get from the voice you asked for?

  2. CONSISTENCY  Render the SAME identity across N different sentences and
                measure pairwise similarity with an INDEPENDENT encoder
                (ECAPA -- never mark your own homework). This is PHASE-01's
                exit criterion X1.1 measured for the first time.

  3. REAL vs SYNTHETIC  Do vectors sampled from the E1 GMM hold together as
                well as vectors extracted from real humans? If synthetic
                identities drift more or are less self-consistent, that is a
                cost of the two-tower design and has to be known.

All similarity is reported in BOTH raw and working space, because E4 showed
raw cosine barely discriminates (C_same 0.9903 vs C_diff 0.9679).

Outputs -> experiments/E2/out/   (includes .wav files you can listen to)
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.renderer import Qwen3BaseRenderer, load_backend
from alaap.encoder import IndependentSV
from alaap.geometry import SpeakerSpace
from alaap.metrics import verification_stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
AUD = os.path.join(OUT, "audio")
os.makedirs(AUD, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--n-real", type=int, default=15)
ap.add_argument("--n-synth", type=int, default=15)
ap.add_argument("--lines", type=int, default=4)
ap.add_argument("--pca", type=int, default=50)
ap.add_argument("--save-audio", type=int, default=8)
args = ap.parse_args()
rng = np.random.default_rng(0)

SCRIPT = [
    "The mountains remember every footstep, even the ones you regret.",
    "It's cold today, colder than the almanac promised.",
    "Bring me the lantern and say nothing more about it.",
    "I have counted the stars twice and both times came up short.",
    "You should leave before the tide turns against us.",
    "Nobody warned me the door would open from the inside.",
][:max(args.lines, 1)]

# ------------------------------------------------------- 1. source vectors
print("[1/4] preparing identity vectors")
d = np.load("experiments/E1/out/real_embeddings.npz")
Z, ids = d["Z"].astype(np.float64), list(d["ids"])
uniq = sorted(set(ids)); idsa = np.asarray(ids)
S = np.stack([Z[idsa == s].mean(0) for s in uniq])
print(f"      {len(S)} real speaker centroids, dim {S.shape[1]}")

space = SpeakerSpace.fit(S, n_components=min(args.pca * 3, len(S) - 1))
k = min(args.pca, space.components.shape[0])
P = space.to_pca(space.encode(S))[:, :k]
print(f"      {space}")

# real identities
real_idx = rng.choice(len(S), args.n_real, replace=False)
V_real = S[real_idx]

# synthetic identities from the E1 winner: GMM k=5 in PCA space
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(5, covariance_type="full", reg_covar=1e-4,
                      random_state=0, max_iter=500).fit(P)
P_syn = gmm.sample(args.n_synth)[0]
W_syn = space.from_pca(np.pad(P_syn, ((0, 0), (0, space.components.shape[0] - k))))
V_syn = space.decode(W_syn, project_to_shell=True)
print(f"      {len(V_real)} real + {len(V_syn)} synthetic (GMM k=5, shell-projected)")
print(f"      |v| real {np.linalg.norm(V_real,axis=1).mean():.3f} | "
      f"synth {np.linalg.norm(V_syn,axis=1).mean():.3f}")

identities = ([("real", i, V_real[i]) for i in range(len(V_real))] +
              [("synth", i, V_syn[i]) for i in range(len(V_syn))])

# ------------------------------------------------------------- 2. render
print(f"[2/4] rendering {len(identities)} identities x {len(SCRIPT)} lines "
      f"= {len(identities)*len(SCRIPT)} clips")
r = Qwen3BaseRenderer(); load_backend(r, is_public_deployment=True)
sv = IndependentSV()

rows, saved = [], 0
t_start = time.time()
for kind, i, v in identities:
    v32 = v.astype(np.float32)
    per_line = []
    for li, line in enumerate(SCRIPT):
        t0 = time.time()
        try:
            a = r.render_from_vector(v32, line, "en")
        except Exception as e:
            print(f"      !! {kind}-{i} line {li} failed: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        dur = len(a.wav) / a.sample_rate
        back = r.extract_vector(a.wav, a.sample_rate).astype(np.float64)
        ecapa = sv.embed(a.wav, sr=a.sample_rate)
        per_line.append({"line": li, "dur": dur, "rtf": dt / max(dur, 1e-6),
                         "back": back, "ecapa": ecapa})
        if saved < args.save_audio and li == 0:
            import soundfile as sf
            sf.write(os.path.join(AUD, f"{kind}_{i:02d}.wav"), a.wav, a.sample_rate)
            saved += 1
    if per_line:
        rows.append({"kind": kind, "idx": i, "v": v, "renders": per_line})
    done = len(rows)
    if done % 5 == 0:
        el = time.time() - t_start
        print(f"      {done}/{len(identities)} identities | {el/60:.1f} min "
              f"| eta {el/done*(len(identities)-done)/60:.1f} min", flush=True)

print(f"      rendered in {(time.time()-t_start)/60:.1f} min")

# ------------------------------------------------------------- 3. analyse
print("[3/4] measuring drift and consistency")


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


out = {"real": {}, "synth": {}}
for kind in ("real", "synth"):
    sub = [x for x in rows if x["kind"] == kind]
    drift_raw, drift_work, cons_ecapa, cons_qwen, rtfs = [], [], [], [], []
    for x in sub:
        v = x["v"]
        vw = space.encode(v)[0]
        for rr in x["renders"]:
            drift_raw.append(cos(v, rr["back"]))
            drift_work.append(cos(vw, space.encode(rr["back"])[0]))
            rtfs.append(rr["rtf"])
        # within-identity consistency across different sentences
        E = np.stack([rr["ecapa"] for rr in x["renders"]])
        Q = np.stack([rr["back"] for rr in x["renders"]])
        if len(E) >= 2:
            for a_ in range(len(E)):
                for b_ in range(a_ + 1, len(E)):
                    cons_ecapa.append(cos(E[a_], E[b_]))
                    cons_qwen.append(cos(space.encode(Q[a_])[0], space.encode(Q[b_])[0]))
    out[kind] = {
        "n_identities": len(sub), "n_renders": len(drift_raw),
        "drift_raw_mean": float(np.mean(drift_raw)), "drift_raw_std": float(np.std(drift_raw)),
        "drift_work_mean": float(np.mean(drift_work)), "drift_work_std": float(np.std(drift_work)),
        "drift_work_min": float(np.min(drift_work)),
        "consistency_ecapa_mean": float(np.mean(cons_ecapa)) if cons_ecapa else None,
        "consistency_ecapa_std": float(np.std(cons_ecapa)) if cons_ecapa else None,
        "consistency_ecapa_min": float(np.min(cons_ecapa)) if cons_ecapa else None,
        "consistency_qwen_work_mean": float(np.mean(cons_qwen)) if cons_qwen else None,
        "rtf_mean": float(np.mean(rtfs)),
    }

# cross-identity ECAPA floor: are different identities actually different?
allE, allLab = [], []
for x in rows:
    for rr in x["renders"]:
        allE.append(rr["ecapa"]); allLab.append(f"{x['kind']}-{x['idx']}")
vs_gen = verification_stats(np.stack(allE).astype(np.float64), allLab)
out["generated_voices_verification"] = vs_gen.to_dict()

# ------------------------------------------------------------- 4. report
summary = {"n_identities": len(rows), "lines_per_identity": len(SCRIPT),
           "pca_dims": int(k), "space_stats": space.stats.__dict__, **out}
json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)

print()
print("=" * 88)
print("E2 RESULT — vocoder drift & synthesized-identity coherence")
print("=" * 88)
print(f"  {len(rows)} identities x {len(SCRIPT)} lines | PCA-{k}")
print()
print(f"  {'':<28} {'REAL vectors':>16} {'SYNTHETIC':>16}")
print(f"  {'-'*28} {'-'*16} {'-'*16}")
rw, sw = out["real"], out["synth"]
def row(lbl, a, b, fmt="{:.4f}"):
    fa = fmt.format(a) if a is not None else "-"
    fb = fmt.format(b) if b is not None else "-"
    print(f"  {lbl:<28} {fa:>16} {fb:>16}")
row("drift, raw cosine", rw["drift_raw_mean"], sw["drift_raw_mean"])
row("drift, WORKING space", rw["drift_work_mean"], sw["drift_work_mean"])
row("  worst-case drift", rw["drift_work_min"], sw["drift_work_min"])
row("consistency (ECAPA)", rw["consistency_ecapa_mean"], sw["consistency_ecapa_mean"])
row("  worst-case", rw["consistency_ecapa_min"], sw["consistency_ecapa_min"])
row("RTF", rw["rtf_mean"], sw["rtf_mean"], "{:.2f}")
print()
print("  Generated voices, treated as speakers (independent ECAPA scorer):")
print(f"    C_same {vs_gen.same_mean:+.4f}   C_diff {vs_gen.diff_mean:+.4f}   "
      f"d-prime {vs_gen.d_prime:.2f}   EER {vs_gen.eer*100:.2f}%")
print()
print("  Benchmarks from E4 on real human speech:")
print("    ECAPA raw       C_same +0.6766  C_diff +0.2092  d-prime 3.97  EER 4.17%")
print(f"  audio samples -> {AUD}")
print("=" * 88)
