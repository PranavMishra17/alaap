"""
E0 — A training-free Direction channel via emotion task vectors.

Qwen3-TTS-Base has NO per-line control in its API. Source inspection
(RESEARCH/10, confirmed here in alaap/renderer.py) found `generate_voice_clone`
has no `instruct` parameter and never passes `instruct_ids`. So the Tier-1
backend we chose for identity has, out of the box, nothing to say HOW a line
should be delivered -- which would make it useless for dramatic content.

The proposed fix is task-vector arithmetic in speaker-embedding space, from
"Task-Vector Arithmetic for Emotional Expressivity Control in
Language-Model-Based Text-to-Speech" (arXiv:2606.05367, de Brito & Candido
Junior, June 2026), which evaluated on Qwen3-TTS-12Hz-1.7B:

    tau_emo = E_i[ x(s_i, emo) ] - E_i[ x(s_i, neutral) ]
    x_new   = x(target, neutral) + alpha * tau_emo

Reported: emotion2vec cosine +0.29 (English), WavLM SECS >= 0.88.

WHY tau MUST BE AVERAGED OVER MANY SPEAKERS: a single-speaker difference
carries that speaker's timbre as well as the emotion, so applying it drags
the target's identity toward the donor. CREMA-D is fully balanced (91 actors
x 12 sentences x 6 emotions), so averaging is well-conditioned.

MEASUREMENT INDEPENDENCE: the steering happens in Qwen3 embedding space, so
it must NOT be scored there. Two independent instruments are used:
  * ECAPA-TDNN  -- identity preservation (did the speaker survive?)
  * acoustics   -- did the delivery actually change? (F0, rate, dynamics)
  * a linear emotion probe fit on ECAPA embeddings, never on Qwen3's

SPEAKER-DISJOINT: tau is fitted on one half of the actors and applied to the
other half. Actors come from the filename prefix (1068_TIE_ANG_XX.wav).

LICENCE NOTE: CREMA-D is ODbL upstream. `confit/cremad-parquet` re-hosts it
with no declared licence. Fine as a RESEARCH instrument; the tau vectors
derived from it are research-lane until the ODbL derived-database question
is answered. Flagged in NEEDS-FROM-YOU.md. (Two other mirrors declare
apache-2.0 over RAVDESS/CREMA-D, which their upstreams do not support --
invariant I4's exact trap, avoided.)

Outputs -> experiments/E0/out/
"""
import argparse, io as _io, json, os, sys, time, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.encoder import SpeakerEncoder, IndependentSV, DEFAULT_MODEL
from alaap.geometry import SpeakerSpace
from alaap.acoustics import measure

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
AUD = os.path.join(OUT, "audio")
os.makedirs(AUD, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--n", type=int, default=1200, help="CREMA-D clips to load")
ap.add_argument("--targets", type=int, default=6)
ap.add_argument("--emotions", nargs="*", default=["anger", "sad", "happy"])
ap.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.5, 1.0, 1.5])
ap.add_argument("--line", default="I told you exactly what would happen.")
args = ap.parse_args()
SR = 24000

# ------------------------------------------------------------------ 1. data
print(f"[1/5] loading CREMA-D ({args.n} clips)")
# The cache key MUST carry the model. Zq holds that model's speaker
# embeddings, and the same bug was already caught once in S2, where a key
# without model_id silently served 0.6B embeddings to a 1.7B run. It failed
# loudly there only because the dims differed.
_mtag = args.model.split("/")[-1].replace(".", "")
cache = os.path.join(OUT, f"crema_{args.n}_{_mtag}.npz")
if os.path.exists(cache):
    d = np.load(cache, allow_pickle=True)
    Zq, Ze = d["Zq"].astype(np.float64), d["Ze"].astype(np.float64)
    emo, actor = list(d["emo"]), list(d["actor"])
    print(f"      cached: {Zq.shape}")
else:
    import soundfile as sf, librosa
    from datasets import load_dataset, Audio
    ds = load_dataset("confit/cremad-parquet", split="train",
                      streaming=True).cast_column("audio", Audio(decode=False))
    wavs, emo, actor = [], [], []
    t0 = time.time()
    for r in ds:
        try:
            w, sr = sf.read(_io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        except Exception:
            continue
        if w.ndim > 1:
            w = w.mean(axis=1)
        if len(w) / sr < 1.0:
            continue
        if sr != SR:
            w = librosa.resample(y=w, orig_sr=sr, target_sr=SR)
        pk = float(np.abs(w).max())
        if pk > 1.0:
            w = w / pk
        wavs.append(w.astype(np.float32))
        emo.append(str(r["emotion"]))
        actor.append(os.path.basename(str(r["file"])).split("_")[0])
        if len(wavs) >= args.n:
            break
    print(f"      {len(wavs)} clips, {len(set(actor))} actors, "
          f"{len(set(emo))} emotions | {time.time()-t0:.0f}s")
    enc = SpeakerEncoder(args.model)
    Zq = enc.embed_many(wavs, progress_every=300).astype(np.float64)
    del enc
    import torch; torch.cuda.empty_cache()
    sv = IndependentSV()
    Ze = sv.embed_many(wavs, sr=SR, progress_every=300).astype(np.float64)
    del sv; torch.cuda.empty_cache()
    np.savez_compressed(cache, Zq=Zq.astype(np.float32), Ze=Ze.astype(np.float32),
                        emo=np.array(emo), actor=np.array(actor))

emo = np.asarray(emo); actor = np.asarray(actor)
actors = sorted(set(actor.tolist()))
print(f"      Zq{Zq.shape} Ze{Ze.shape} | {len(actors)} actors | "
      f"emotions {sorted(set(emo.tolist()))}")

# --------------------------------------------------- 2. speaker-disjoint tau
rng = np.random.default_rng(0)
perm = rng.permutation(len(actors))
fit_actors = {actors[i] for i in perm[:len(actors) // 2]}
test_actors = {actors[i] for i in perm[len(actors) // 2:]}
print(f"[2/5] tau fitted on {len(fit_actors)} actors, applied to "
      f"{len(test_actors)} held-out actors (disjoint)")

fit_m = np.array([a in fit_actors for a in actor])


def per_speaker_mean(mask):
    """E_i[x(s_i, .)] -- average WITHIN each speaker first, then across."""
    out = []
    for a in sorted({s for s, k in zip(actor, mask) if k}):
        sel = mask & (actor == a)
        if sel.sum():
            out.append(Zq[sel].mean(0))
    return np.stack(out) if out else None


neutral_fit = per_speaker_mean(fit_m & (emo == "neutral"))
taus, tau_meta = {}, {}
for e in sorted(set(emo.tolist())):
    if e == "neutral":
        continue
    emo_fit = per_speaker_mean(fit_m & (emo == e))
    if emo_fit is None or neutral_fit is None:
        continue
    n = min(len(emo_fit), len(neutral_fit))
    taus[e] = emo_fit[:n].mean(0) - neutral_fit[:n].mean(0)
    tau_meta[e] = {"n_speakers": int(n), "norm": float(np.linalg.norm(taus[e]))}
    print(f"      tau[{e:<8}] over {n} speakers, ||tau|| = {tau_meta[e]['norm']:.4f}")

# how big is tau relative to a speaker vector? (a sanity scale check)
mean_norm = float(np.linalg.norm(Zq, axis=1).mean())
print(f"      mean ||x|| = {mean_norm:.3f}  ->  tau is "
      f"{100*np.mean([t['norm'] for t in tau_meta.values()])/mean_norm:.1f}% of a vector")

# single-speaker tau, for the leakage comparison the paper warns about
one_actor = sorted(fit_actors)[0]
tau_single = {}
for e in taus:
    a = Zq[(actor == one_actor) & (emo == e)]
    nn = Zq[(actor == one_actor) & (emo == "neutral")]
    if len(a) and len(nn):
        tau_single[e] = a.mean(0) - nn.mean(0)

# ------------------------------------------------------------ 3. targets
space = SpeakerSpace.fit(Zq[fit_m], n_components=128)
tgt_actors = sorted(test_actors)[:args.targets]
targets = {}
for a in tgt_actors:
    sel = (actor == a) & (emo == "neutral")
    if sel.sum():
        targets[a] = Zq[sel].mean(0)
print(f"[3/5] {len(targets)} held-out target speakers (neutral vectors)")

# -------------------------------------------------------------- 4. render
print(f"[4/5] rendering {len(targets)}x{len(args.emotions)}x{len(args.alphas)} "
      f"= {len(targets)*len(args.emotions)*len(args.alphas)} clips")
from alaap.renderer import Qwen3BaseRenderer, load_backend
import soundfile as sf
r = Qwen3BaseRenderer(args.model); load_backend(r, is_public_deployment=True)
sv = IndependentSV()

rows = []
t0 = time.time()
n_done = 0
total = len(targets) * len(args.emotions) * len(args.alphas)
for a, base in targets.items():
    for e in args.emotions:
        if e not in taus:
            continue
        for al in args.alphas:
            v = (base + al * taus[e]).astype(np.float32)
            try:
                au = r.render_from_vector(v, args.line, "en")
            except Exception as ex:
                print(f"      !! {a}/{e}/a{al}: {ex}"); continue
            ac = measure(au.wav, args.line, au.sample_rate)
            ec = sv.embed(au.wav, sr=au.sample_rate)
            fn = f"{a}_{e}_a{al}.wav"
            if al in (0.0, 1.0):
                sf.write(os.path.join(AUD, fn), au.wav, au.sample_rate)
            rows.append({"actor": a, "emotion": e, "alpha": al, "file": fn,
                         "ecapa": ec, "acoustics": ac.to_dict()})
            n_done += 1
            if n_done % 10 == 0:
                el = time.time() - t0
                print(f"      {n_done}/{total} | {el/60:.1f} min | "
                      f"eta {el/n_done*(total-n_done)/60:.1f} min", flush=True)

# single-speaker tau comparison, alpha=1.0 only -- the leakage check
leak_rows = []
for a, base in list(targets.items())[:3]:
    for e in args.emotions:
        if e not in tau_single:
            continue
        v = (base + 1.0 * tau_single[e]).astype(np.float32)
        try:
            au = r.render_from_vector(v, args.line, "en")
        except Exception:
            continue
        leak_rows.append({"actor": a, "emotion": e,
                          "ecapa": sv.embed(au.wav, sr=au.sample_rate)})
print(f"      rendered {len(rows)} (+{len(leak_rows)} single-speaker-tau) "
      f"in {(time.time()-t0)/60:.1f} min")

# ------------------------------------------------------------- 5. analyse
print("[5/5] measuring")


def cos(x, y):
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


base_ec = {(x["actor"], x["emotion"]): x["ecapa"]
           for x in rows if x["alpha"] == 0.0}
res = defaultdict(lambda: defaultdict(list))
for x in rows:
    b = base_ec.get((x["actor"], x["emotion"]))
    if b is None:
        continue
    res[x["emotion"]][x["alpha"]].append({
        "secs": cos(b, x["ecapa"]),
        "f0_mean": x["acoustics"]["f0_mean"],
        "f0_std": x["acoustics"]["f0_std"],
        "rate": x["acoustics"]["speaking_rate"],
        "tilt": x["acoustics"]["spectral_tilt"]})

summary = {"model": args.model, "n_clips": int(len(Zq)),
           "n_actors": len(actors), "tau": tau_meta,
           "tau_pct_of_vector": float(100 * np.mean([t["norm"] for t in tau_meta.values()]) / mean_norm),
           "line": args.line, "by_emotion": {}}
for e, byal in res.items():
    a0 = byal.get(0.0, [])
    summary["by_emotion"][e] = {}
    for al, lst in sorted(byal.items()):
        summary["by_emotion"][e][str(al)] = {
            "n": len(lst),
            "secs_mean": float(np.mean([x["secs"] for x in lst])),
            "secs_min": float(np.min([x["secs"] for x in lst])),
            "f0_mean": float(np.mean([x["f0_mean"] for x in lst])),
            "f0_std": float(np.mean([x["f0_std"] for x in lst])),
            "rate": float(np.mean([x["rate"] for x in lst])),
            "d_f0": float(np.mean([x["f0_mean"] for x in lst]) -
                          np.mean([x["f0_mean"] for x in a0])) if a0 else 0.0,
            "d_f0_std": float(np.mean([x["f0_std"] for x in lst]) -
                              np.mean([x["f0_std"] for x in a0])) if a0 else 0.0,
            "d_rate": float(np.mean([x["rate"] for x in lst]) -
                            np.mean([x["rate"] for x in a0])) if a0 else 0.0}

if leak_rows:
    ls = [cos(base_ec[(x["actor"], x["emotion"])], x["ecapa"])
          for x in leak_rows if (x["actor"], x["emotion"]) in base_ec]
    summary["single_speaker_tau_secs"] = float(np.mean(ls)) if ls else None

json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2)

print()
print("=" * 88)
print("E0 RESULT — training-free Direction on Qwen3-TTS-Base")
print("=" * 88)
print(f"  tau over {list(tau_meta.values())[0]['n_speakers']} speakers | "
      f"tau is {summary['tau_pct_of_vector']:.1f}% of a typical vector's length")
print(f"  line: \"{args.line}\"")
print()
for e in sorted(summary["by_emotion"]):
    print(f"  {e.upper()}")
    print(f"    {'alpha':>6} {'SECS':>8} {'SECS min':>9} {'dF0 Hz':>9} "
          f"{'dF0std':>8} {'d rate':>8}")
    for al in sorted(summary["by_emotion"][e], key=float):
        v = summary["by_emotion"][e][al]
        flag = "" if v["secs_mean"] >= 0.88 else "   <-- identity loss"
        print(f"    {al:>6} {v['secs_mean']:>8.4f} {v['secs_min']:>9.4f} "
              f"{v['d_f0']:>+9.1f} {v['d_f0_std']:>+8.1f} {v['d_rate']:>+8.2f}{flag}")
    print()
if summary.get("single_speaker_tau_secs") is not None:
    print(f"  LEAKAGE CHECK: single-speaker tau SECS = "
          f"{summary['single_speaker_tau_secs']:.4f}  "
          f"(multi-speaker at alpha=1.0 should be higher)")
print(f"  reference: the paper reports WavLM SECS >= 0.88 at useful alpha")
print(f"  audio -> {AUD}")
print("=" * 88)
