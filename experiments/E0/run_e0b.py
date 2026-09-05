"""
E0b — Faithful reproduction of arXiv:2606.05367's task-vector steering.

E0 run 1 produced a strong NEGATIVE result: identity collapsed (ECAPA SECS
0.31-0.59) at every useful alpha, against the paper's reported SECS >= 0.88.
Before accepting that as a contradiction, two implementation differences were
found by re-reading the paper rather than a summary of it:

  1. The paper uses **avg4spk** -- tau averaged over exactly FOUR speakers.
     Run 1 averaged over 37. More averaging is not obviously better: a tau
     over 37 speakers may converge on a generic "emotional speech" direction
     (and a CREMA-D-vs-target domain component) rather than a clean emotion
     delta.
  2. The paper measures SECS with **microsoft/wavlm-base-plus-sv**. Run 1
     used ECAPA. These do not share an absolute scale, so 0.88 quoted for one
     is not a threshold for the other.

This run varies (1) and measures with BOTH encoders, plus -- crucially -- the
C_diff baseline for each, so that "0.88" can be interpreted rather than
merely compared. WavLM x-vectors are highly concentrated (cos between two
noise clips is ~0.98), so a raw 0.88 may be a low bar in that space.

Outputs -> experiments/E0/out/e0b_*
"""
import argparse, json, os, sys, time, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.encoder import IndependentSV, WavLMSV, DEFAULT_MODEL
from alaap.acoustics import measure

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
AUD = os.path.join(OUT, "audio_b")
os.makedirs(AUD, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--tau-speakers", nargs="*", type=int, default=[4, 37])
ap.add_argument("--alphas", nargs="*", type=float, default=[0.0, 1.0])
ap.add_argument("--targets", type=int, default=4)
ap.add_argument("--emotions", nargs="*", default=["anger", "happy", "sad"])
ap.add_argument("--line", default="I told you exactly what would happen.")
args = ap.parse_args()
SR = 24000

# --------------------------------------------------------------- 1. cached
d = np.load(os.path.join(OUT, "crema_1200.npz"), allow_pickle=True)
Zq = d["Zq"].astype(np.float64)
emo = np.asarray(d["emo"]); actor = np.asarray(d["actor"])
actors = sorted(set(actor.tolist()))
print(f"[1/4] cached CREMA-D: Zq{Zq.shape}, {len(actors)} actors")

rng = np.random.default_rng(0)
perm = rng.permutation(len(actors))
fit_actors = [actors[i] for i in perm[:len(actors) // 2]]
test_actors = [actors[i] for i in perm[len(actors) // 2:]]


def spk_means(actor_list, mask_emo):
    out, used = [], []
    for a in actor_list:
        sel = (actor == a) & mask_emo
        if sel.sum():
            out.append(Zq[sel].mean(0)); used.append(a)
    return (np.stack(out) if out else None), used


def build_tau(n_spk, emotion, seed=0):
    """tau over EXACTLY n_spk speakers, as avg4spk does."""
    r = np.random.default_rng(seed)
    cand = [a for a in fit_actors
            if ((actor == a) & (emo == emotion)).sum()
            and ((actor == a) & (emo == "neutral")).sum()]
    if len(cand) < n_spk:
        return None
    pick = [cand[i] for i in r.permutation(len(cand))[:n_spk]]
    e, _ = spk_means(pick, emo == emotion)
    n, _ = spk_means(pick, emo == "neutral")
    return e.mean(0) - n.mean(0)


# ------------------------------------------------------------- 2. targets
targets = {}
for a in test_actors[:args.targets]:
    sel = (actor == a) & (emo == "neutral")
    if sel.sum():
        targets[a] = Zq[sel].mean(0)
print(f"[2/4] {len(targets)} held-out targets | tau speaker counts "
      f"{args.tau_speakers}")

# how many fit-actors actually have BOTH the emotion and a neutral clip?
feasible = {}
for e in args.emotions:
    feasible[e] = sum(1 for a in fit_actors
                      if ((actor == a) & (emo == e)).sum()
                      and ((actor == a) & (emo == "neutral")).sum())
print(f"      usable fit-actors per emotion: {feasible}")

taus = {}
for n_spk in args.tau_speakers:
    for e in args.emotions:
        if n_spk > feasible[e]:
            print(f"      tau[n={n_spk:>2}, {e:<6}] SKIPPED "
                  f"(only {feasible[e]} usable actors)")
            continue
        t = build_tau(n_spk, e)
        if t is not None:
            taus[(n_spk, e)] = t
            print(f"      tau[n={n_spk:>2}, {e:<6}] ||tau|| = {np.linalg.norm(t):.4f}")
if not taus:
    sys.exit("no tau could be built -- increase --n so more actors have both "
             "the emotion and a neutral clip")

# ------------------------------------------------------------- 3. render
from alaap.renderer import Qwen3BaseRenderer, load_backend
import soundfile as sf
r = Qwen3BaseRenderer(args.model); load_backend(r, is_public_deployment=True)
sv = IndependentSV(); wl = WavLMSV()

jobs = []
for a, base in targets.items():
    for e in args.emotions:
        jobs.append((a, base, e, 0, 0.0))                      # shared baseline
        for n_spk in sorted({k[0] for k in taus if k[1] == e}):
            for al in args.alphas:
                if al == 0.0:
                    continue
                jobs.append((a, base, e, n_spk, al))
print(f"[3/4] rendering {len(jobs)} clips")

rows, t0 = [], time.time()
for i, (a, base, e, n_spk, al) in enumerate(jobs):
    v = base if al == 0.0 else base + al * taus[(n_spk, e)]
    try:
        au = r.render_from_vector(np.asarray(v, dtype=np.float32), args.line, "en")
    except Exception as ex:
        print(f"      !! {a}/{e}/n{n_spk}/a{al}: {ex}"); continue
    ac = measure(au.wav, args.line, au.sample_rate)
    rows.append({"actor": a, "emotion": e, "n_spk": n_spk, "alpha": al,
                 "ecapa": sv.embed(au.wav, sr=au.sample_rate),
                 "wavlm": wl.embed(au.wav, sr=au.sample_rate),
                 "f0": ac.f0_mean, "f0_std": ac.f0_std, "rate": ac.speaking_rate})
    if al == 1.0 and n_spk == args.tau_speakers[0]:
        sf.write(os.path.join(AUD, f"{a}_{e}_n{n_spk}_a{al}.wav"), au.wav, au.sample_rate)
    if (i + 1) % 8 == 0:
        el = time.time() - t0
        print(f"      {i+1}/{len(jobs)} | {el/60:.1f} min | "
              f"eta {el/(i+1)*(len(jobs)-i-1)/60:.1f} min", flush=True)
print(f"      done in {(time.time()-t0)/60:.1f} min")

# ------------------------------------------------------------- 4. analyse
def cos(x, y):
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


# C_diff baselines: how similar are DIFFERENT speakers in each space?
# Without this, "SECS >= 0.88" cannot be interpreted.
base_rows = [x for x in rows if x["alpha"] == 0.0]
def cdiff(key):
    v = [x[key] for x in base_rows]
    s = [cos(v[i], v[j]) for i in range(len(v)) for j in range(i + 1, len(v))
         if base_rows[i]["actor"] != base_rows[j]["actor"]]
    return float(np.mean(s)) if s else float("nan")


c_ecapa, c_wavlm = cdiff("ecapa"), cdiff("wavlm")
base = {(x["actor"], x["emotion"]): x for x in base_rows}
agg = defaultdict(lambda: defaultdict(list))
for x in rows:
    if x["alpha"] == 0.0:
        continue
    b = base.get((x["actor"], x["emotion"]))
    if b is None:
        continue
    agg[(x["n_spk"], x["emotion"])][x["alpha"]].append({
        "secs_ecapa": cos(b["ecapa"], x["ecapa"]),
        "secs_wavlm": cos(b["wavlm"], x["wavlm"]),
        "d_f0": x["f0"] - b["f0"], "d_f0_std": x["f0_std"] - b["f0_std"],
        "d_rate": x["rate"] - b["rate"]})

summary = {"model": args.model, "c_diff_ecapa": c_ecapa, "c_diff_wavlm": c_wavlm,
           "tau_norms": {f"{k[0]}_{k[1]}": float(np.linalg.norm(v))
                         for k, v in taus.items()},
           "cells": {}}
for (n_spk, e), byal in agg.items():
    for al, lst in byal.items():
        summary["cells"][f"n{n_spk}_{e}_a{al}"] = {
            "n": len(lst),
            "secs_ecapa": float(np.mean([x["secs_ecapa"] for x in lst])),
            "secs_wavlm": float(np.mean([x["secs_wavlm"] for x in lst])),
            "d_f0": float(np.mean([x["d_f0"] for x in lst])),
            "d_f0_std": float(np.mean([x["d_f0_std"] for x in lst])),
            "d_rate": float(np.mean([x["d_rate"] for x in lst]))}
json.dump(summary, open(os.path.join(OUT, "e0b_results.json"), "w"), indent=2)

print()
print("=" * 92)
print("E0b RESULT — does avg4spk + WavLM reproduce the paper?")
print("=" * 92)
print(f"  BASELINES (different speakers, so the floor a SECS number sits above):")
print(f"    ECAPA C_diff = {c_ecapa:.4f}      WavLM C_diff = {c_wavlm:.4f}")
print(f"  paper reports SECS_wavlm >= 0.88 (0.907/0.902/0.926 angry/happy/sad)")
print()
print(f"  {'tau spk':>8} {'emotion':<8} {'alpha':>6} {'SECS wavlm':>11} "
      f"{'SECS ecapa':>11} {'dF0':>8} {'d rate':>8}")
print(f"  {'-'*8} {'-'*8} {'-'*6} {'-'*11} {'-'*11} {'-'*8} {'-'*8}")
for k in sorted(summary["cells"]):
    v = summary["cells"][k]
    n_spk, e, al = k.split("_")
    flag = "" if v["secs_wavlm"] >= 0.88 else "  <-- below paper"
    print(f"  {n_spk[1:]:>8} {e:<8} {al[1:]:>6} {v['secs_wavlm']:>11.4f} "
          f"{v['secs_ecapa']:>11.4f} {v['d_f0']:>+8.1f} {v['d_rate']:>+8.2f}{flag}")
print()
print(f"  audio -> {AUD}")
print("=" * 92)
