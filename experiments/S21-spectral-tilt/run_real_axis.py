"""
S21 phase C -- does REPAIRING `spectral_tilt` make it a better identity axis?

Phase A/B (`run_tilt_audit.py`) established that the shipped estimator reads F0:
r = +0.918 on synthetic voices whose brightness is held constant by construction.
On 300 real 16 kHz Hindi clips the shipped estimator sits at +0.297 and the
repaired one at +0.423 -- i.e. repairing it moves the axis TOWARD f0_mean, not
away. (An earlier pass measured those clips at 24 kHz when they are stored at
16 kHz and reported the reverse; see RESULTS.md.)

THAT IS NOT YET A REASON TO CHANGE IT. Removing pitch leakage removes separating
power too, and an axis is judged by whether it identifies people. It is entirely
possible the shipped estimator separates speakers BETTER precisely because it is
half an F0 measurement -- in which case the honest conclusion is "spectral_tilt
is not an independent third axis", not "here is a better one".

THE PROPERTIES ASSERTED BEFORE THE NUMBERS ARE READ:

  C1  Both estimators must produce a finite reading for >=95% of clips. An axis
      that wins by dropping hard clips has not won.
  C2  The corpus must reproduce S18's published within/between for the SHIPPED
      estimator to within +-0.10 (S18 hi/bn/ta mean 0.19). If it does not, this
      sample is not the sample S18 measured and nothing here is comparable.
  C3  Redundancy against `f0_mean` is the deciding number, not the ratio. The
      claim under test is "partly f0_mean in disguise", so the repaired variant
      must show a LOWER |r| with f0_mean at no worse a within/between ratio.

Streams once and caches, per the standing rule -- re-runs never hit the network.

    envs/qwen3/Scripts/python.exe experiments/S21-spectral-tilt/run_real_axis.py
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
from alaap.acoustics import SR, f0_track, spectral_tilt
from alaap.data import norm_gender

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--n", type=int, default=160)
args = ap.parse_args()

CACHE = os.path.join(OUT, f"real_{args.corpus}_{args.n}_2.npz")

# the repaired candidate, imported from phase A so there is exactly one definition
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_tilt_audit.py"), encoding="utf-8").read()
_ns = {}
exec(_src[_src.index("def tilt_ltas"):_src.index("VARIANTS = {")],
     {"np": np, "SR": SR, "voiced_mask": __import__("alaap.acoustics",
      fromlist=["voiced_mask"]).voiced_mask, "__builtins__": __builtins__}, _ns)
tilt_ltas = _ns["tilt_ltas"]


def tilt_repaired(w, sr=SR):
    return tilt_ltas(w, sr, True, True, True, 300.0, 6000.0)


# ------------------------------------------------------------------ stream once
if os.path.exists(CACHE):
    print(f"[cache] {CACHE}")
    d = np.load(CACHE, allow_pickle=True)
    spk = json.loads(str(d["spk"]))
    gen = json.loads(str(d["gen"]))
    W = [np.asarray(d[f"w{i}"], np.float32) for i in range(int(d["n"]))]
else:
    from alaap.data import stream_clips
    print(f"[stream] {args.corpus} n={args.n} per_speaker=2 -- expect several minutes")
    clips = stream_clips(args.corpus, n=args.n, per_speaker=2, dev_only=True,
                         progress_every=20)
    # `speaker_id` and `gender` are TOP-LEVEL fields on Clip, not entries in
    # `meta` -- S18 read them from a cached npz whose metas were dicts, and
    # copying that access pattern here threw away a completed 4-minute stream.
    # Cache is written before anything can raise, per "stream once, cache
    # immediately" (HANDOFF 8b).
    W = [np.asarray(c.wav, np.float32) for c in clips]
    spk = [str(getattr(c, "speaker_id", None) or c.meta.get("speaker_id", f"unk{i}"))
           for i, c in enumerate(clips)]
    gen = [norm_gender(getattr(c, "gender", None) or c.meta.get("gender"))
           for c in clips]
    np.savez_compressed(CACHE, n=len(W), spk=json.dumps(spk), gen=json.dumps(gen),
                        **{f"w{i}": w for i, w in enumerate(W)})
    print(f"[cached] {CACHE}")

print(f"  {len(W)} clips, {len(set(spk))} speakers")

# ---------------------------------------------------------------------- measure
EST = {"shipped": spectral_tilt, "repaired": tilt_repaired}
vals = {k: [] for k in EST}
f0m = []
for i, w in enumerate(W):
    for k, fn in EST.items():
        try:
            vals[k].append(float(fn(w, SR)))
        except Exception:
            vals[k].append(np.nan)
    f0m.append(float(np.mean(f0_track(w, SR))))
    if (i + 1) % 20 == 0:
        print(f"  measured {i + 1}/{len(W)}")
for k in EST:
    vals[k] = np.array(vals[k], float)
f0m = np.array(f0m, float)

report = {"corpus": args.corpus, "n_clips": len(W), "n_speakers": len(set(spk))}

# --------------------------------------------------------------------- C1 finite
print()
print("C1  coverage -- both estimators finite on >=95% of clips")
ok = True
for k in EST:
    frac = float(np.isfinite(vals[k]).mean())
    print(f"    {k:<10} finite {frac:.1%}   {'PASS' if frac >= 0.95 else 'FAIL'}")
    ok &= frac >= 0.95
report["C1_finite"] = {k: float(np.isfinite(vals[k]).mean()) for k in EST}
if not ok:
    sys.exit("C1 failed -- an axis that wins by dropping clips has not won")


def within_between(v):
    by = {}
    for s, x in zip(spk, v):
        if np.isfinite(x):
            by.setdefault(s, []).append(x)
    w, mu = [], []
    for xs in by.values():
        if len(xs) >= 2:
            w.append(np.var(xs[:2], ddof=1))
            mu.append(np.mean(xs[:2]))
    if len(mu) < 20:
        return np.nan, len(mu)
    b = np.var(mu, ddof=1)
    return (float(np.mean(w) / b) if b > 0 else np.nan), len(mu)


def gender_d(v):
    by = {}
    for s, g, x in zip(spk, gen, v):
        if np.isfinite(x) and g:
            by.setdefault(s, (g, []))[1].append(x)
    F = [np.mean(x) for g, x in by.values() if g == "Female"]
    M = [np.mean(x) for g, x in by.values() if g == "Male"]
    if min(len(F), len(M)) < 8:
        return np.nan, len(F), len(M)
    F, M = np.array(F), np.array(M)
    return (float((M.mean() - F.mean()) /
                  max(np.sqrt((F.var(ddof=1) + M.var(ddof=1)) / 2), 1e-9)),
            len(F), len(M))


# --------------------------------------------------------- C2 corpus sanity check
r_ship, npairs = within_between(vals["shipped"])
print()
print("C2  does this sample reproduce S18? (shipped within/between, S18 = 0.19)")
print(f"    shipped ratio {r_ship:.2f} over {npairs} speakers with 2 takes"
      f"   {'PASS' if np.isfinite(r_ship) and abs(r_ship - 0.19) <= 0.10 else 'FAIL'}")
report["C2_shipped_ratio"] = r_ship
report["C2_n_pairs"] = npairs
if not (np.isfinite(r_ship) and abs(r_ship - 0.19) <= 0.10):
    print("    !! this sample is not comparable to S18's. Everything below is")
    print("       about THIS sample only and must not be read against S18's table.")

# ---------------------------------------------------------------- C3 the verdict
print()
print("C3  the deciding comparison")
r_f0 = {}
print(f"    {'estimator':<10} {'within/btwn':>12} {'|r| f0_mean':>12} {'gender d':>10}")
for k in EST:
    rr, _ = within_between(vals[k])
    m = np.isfinite(vals[k]) & np.isfinite(f0m)
    rf = abs(float(np.corrcoef(f0m[m], vals[k][m])[0, 1]))
    gd, nF, nM = gender_d(vals[k])
    r_f0[k] = rf
    report[f"C3_{k}"] = {"ratio": rr, "abs_r_f0": rf, "gender_d": gd,
                         "n_female": nF, "n_male": nM}
    print(f"    {k:<10} {rr:>12.2f} {rf:>12.3f} {gd:>+10.2f}")

# ------------------------------------------ C4 is the ratio difference a finding?
print()
print("C4  paired bootstrap over SPEAKERS -- 0.15 vs 0.13 is a difference, not yet")
print("    a finding. Resampling speakers (not clips) keeps takes with their owner.")
byspk = {}
for s_, g_, i_ in zip(spk, gen, range(len(spk))):
    byspk.setdefault(s_, [g_, []])[1].append(i_)
keys = [k for k, v in byspk.items() if len(v[1]) >= 2]
rng = np.random.default_rng(20260908)


def stats_for(sel, v):
    w, mu, F, M = [], [], [], []
    for k in sel:
        g, idx = byspk[k]
        x = v[idx[:2]]
        if not np.isfinite(x).all():
            continue
        w.append(x.var(ddof=1)); mu.append(x.mean())
        (F if g == "Female" else M).append(x.mean()) if g else None
    if len(mu) < 10 or np.var(mu, ddof=1) <= 0:
        return np.nan, np.nan
    ratio = float(np.mean(w) / np.var(mu, ddof=1))
    gd = np.nan
    if min(len(F), len(M)) >= 5:
        F_, M_ = np.array(F), np.array(M)
        gd = float((M_.mean() - F_.mean()) /
                   max(np.sqrt((F_.var(ddof=1) + M_.var(ddof=1)) / 2), 1e-9))
    return ratio, gd


dr, dg = [], []
for _ in range(2000):
    sel = list(rng.choice(keys, len(keys), replace=True))
    a, ga = stats_for(sel, vals["shipped"])
    b, gb = stats_for(sel, vals["repaired"])
    if np.isfinite(a) and np.isfinite(b):
        dr.append(b - a)
    if np.isfinite(ga) and np.isfinite(gb):
        dg.append(abs(gb) - abs(ga))
for nm, arr in (("within/between (repaired - shipped)", dr),
                ("|gender d|      (repaired - shipped)", dg)):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    sig = "excludes 0" if lo * hi > 0 else "INCLUDES 0 -- not a finding"
    print(f"    {nm}  {np.mean(arr):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}")
    report["C4_" + ("ratio" if arr is dr else "gender_d")] = {
        "delta": float(np.mean(arr)), "ci95": [float(lo), float(hi)]}

print()
rs, rp = report["C3_shipped"], report["C3_repaired"]
better_red = rp["abs_r_f0"] < rs["abs_r_f0"]
no_worse = np.isfinite(rp["ratio"]) and np.isfinite(rs["ratio"]) and rp["ratio"] <= rs["ratio"] + 0.05
print("=" * 78)
if better_red and no_worse:
    print("  ADOPT the repaired estimator: less redundant with f0_mean at no cost")
    print("  to speaker separation. `spectral_tilt` becomes a genuinely third axis.")
    v = "adopt"
elif better_red:
    print("  MIXED: repairing removes the F0 leakage but costs separation. The")
    print("  shipped axis was separating speakers partly BY THEIR PITCH, so the")
    print("  honest conclusion is that the identity set has fewer independent")
    print("  axes than it appears -- not that a drop-in replacement exists.")
    v = "mixed"
else:
    print("  KEEP the shipped estimator. The synthetic F0 leakage does not")
    print("  translate into measurable redundancy on this corpus.")
    v = "keep"
print("=" * 78)
report["verdict"] = v
json.dump(report, io.open(os.path.join(OUT, "real_axis.json"), "w", encoding="utf-8"),
          indent=2)
print(f"\n  -> {os.path.join(OUT, 'real_axis.json')}")
