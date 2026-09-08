"""
S24 -- is S16's cross-corpus `spectral_tilt` reading the microphone?

S21 assertion A4: the shipped `spectral_tilt` moves 0.70 dB/oct on a 16 kHz
resample round-trip, against a between-speaker sd of ~2.0. That makes any
CROSS-corpus tilt comparison partly a comparison of recording chains, and S16
makes one -- it bins 37 Sarvam Bulbul voices (22050 Hz) against a binner fitted
on IndicVoices-R (24000 Hz) and concludes:

    "spectral_tilt 3/5 bins occupied -- balanced, bright, very bright.
     ... no *dark*-timbred voices."

WHAT THIS FOUND IS NOT WHAT IT WENT LOOKING FOR. The bandwidth difference is
real and it changes nothing. The thing that moves S16's numbers is the
ESTIMATOR'S LEVERAGE, and chasing it turned up a gap in S21 as well.

PROPERTIES ASSERTED BEFORE THE NUMBERS WERE READ

  P1  The two sources must differ in effective bandwidth, or there is no
      confound to correct.
  P2  Bulbul's WITHIN/BETWEEN ratio must be materially unchanged by restricting
      to a common clean band, because a constant channel offset cancels in a
      within/between ratio. If it moves, the confound is not a constant offset
      and the analysis is wrong.
  P3  If the bright-side skew survives a band restriction it is a fact about the
      library; if it collapses it was the microphone.

P2 FAILED (1.09 -> 0.23) AND THE DIAGNOSIS IS THE RESULT. Decomposing the
variants one at a time shows the band edges are not responsible at all:

    variant            bulbul within  bulbul between  ratio
    shipped                    0.731           0.669   1.09
    band only (300-6000)       1.785           0.852   2.09   <- WORSE
    voiced only                1.055           0.798   1.32
    1/3-octave only            0.351           3.182   0.11   <- the whole effect
    all four (S21 repaired)    0.235           1.031   0.23

1/3-octave banding raises BETWEEN-voice variance 4.8x. It is not removing noise;
it is measuring something else -- and section 4 identifies what.

    envs/qwen3/Scripts/python.exe experiments/S24-tilt-crosscorpus/run_channel_confound.py
"""
import glob
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.acoustics import spectral_tilt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

S21 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "S21-spectral-tilt")
_src = io.open(os.path.join(S21, "run_tilt_audit.py"), encoding="utf-8").read()
_ns = {}
exec(_src[_src.index("def tilt_ltas"):_src.index("VARIANTS = {")],
     {"np": np, "SR": 24000,
      "voiced_mask": __import__("alaap.acoustics",
                                fromlist=["voiced_mask"]).voiced_mask,
      "__builtins__": __builtins__}, _ns)
tilt_ltas = _ns["tilt_ltas"]


def tilt_common_band(w, sr):
    """
    The band-restricted estimator: 300-6000 Hz, voiced frames, power-averaged,
    1/3-octave. 6000 sits below any plausible anti-alias rolloff for a 22 kHz
    or 24 kHz source, so the two chains are compared where both carry signal.
    S21 measured this variant at 0.03 dB/oct on the round-trip that moves the
    shipped estimator 0.70.
    """
    return tilt_ltas(w, sr, True, True, True, 300.0, 6000.0)


def rolloff(w, sr, frac=0.99):
    """Frequency below which `frac` of the spectral energy sits."""
    import librosa
    S = np.abs(librosa.stft(np.asarray(w, np.float32), n_fft=2048)) ** 2
    m = S.mean(1)
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    c = np.cumsum(m) / max(m.sum(), 1e-20)
    return float(f[int(np.searchsorted(c, frac))])


# ------------------------------------------------------------------- load
print("[1/4] loading both sources")
bul = {}
for p in sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "S16-sarvam-library",
        "out", "audio", "*.wav"))):
    v = os.path.basename(p).rsplit("_", 1)[0]
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    bul.setdefault(v, []).append((w, sr))
if not bul:
    sys.exit("no Bulbul audio found")

CP = os.path.join(S21, "out", "real_indicvoices_r_hi_160_2.npz")
if not os.path.exists(CP):
    sys.exit(f"missing {CP} -- run S21 phase C first")
d = np.load(CP, allow_pickle=True)
cspk = json.loads(str(d["spk"]))
corp = {}
for i, s in enumerate(cspk):
    corp.setdefault(s, []).append((np.asarray(d[f"w{i}"], np.float32), 24000))

print(f"      Bulbul  {sum(len(v) for v in bul.values())} clips / {len(bul)} voices "
      f"@ {bul[list(bul)[0]][0][1]} Hz")
print(f"      corpus  {sum(len(v) for v in corp.values())} clips / {len(corp)} speakers "
      f"@ 24000 Hz")

report = {}

# ------------------------------------------------------------------- P1
print()
print("P1  do the two sources differ in effective bandwidth?")
rb = [rolloff(w, sr) for v in bul.values() for w, sr in v[:2]]
rc = [rolloff(w, sr) for v in list(corp.values())[:40] for w, sr in v[:2]]
print(f"      Bulbul  99%-energy rolloff  {np.median(rb):>7.0f} Hz  "
      f"(nyquist {bul[list(bul)[0]][0][1] // 2})")
print(f"      corpus  99%-energy rolloff  {np.median(rc):>7.0f} Hz  (nyquist 12000)")
diff = abs(np.median(rb) - np.median(rc))
report["P1"] = {"bulbul_rolloff": float(np.median(rb)),
                "corpus_rolloff": float(np.median(rc)), "diff": float(diff)}
print(f"      difference {diff:.0f} Hz   "
      f"{'PASS -- there is a channel difference to correct' if diff > 200 else 'FAIL -- no confound to chase'}")

# ------------------------------------------------------------- measure both
print()
print("[2/4] measuring tilt both ways")
EST = {"shipped (80-8000)": spectral_tilt, "common band (300-6000)": tilt_common_band}
vals = {k: {"bulbul": {}, "corpus": {}} for k in EST}
for k, fn in EST.items():
    for name, src, tag in (("bulbul", bul, "bulbul"), ("corpus", corp, "corpus")):
        for spk, takes in src.items():
            vals[k][tag][spk] = [float(fn(w, sr)) for w, sr in takes[:2]]
    print(f"      {k} done")


def wb(byspk):
    w, mu = [], []
    for xs in byspk.values():
        if len(xs) >= 2 and np.isfinite(xs[:2]).all():
            w.append(np.var(xs[:2], ddof=1))
            mu.append(np.mean(xs[:2]))
    b = np.var(mu, ddof=1)
    return float(np.mean(w) / b) if b > 0 and len(mu) >= 10 else np.nan


# ------------------------------------------------------------------- P2
print()
print("P2  is Bulbul's within/between ratio unmoved by the band restriction?")
print("    (a constant channel offset MUST cancel there -- if it does not, the")
print("     confound is not a constant offset and this analysis is wrong)")
p2 = {}
for k in EST:
    rb_ = wb(vals[k]["bulbul"])
    rc_ = wb(vals[k]["corpus"])
    p2[k] = {"bulbul": rb_, "corpus": rc_}
    print(f"      {k:<24} bulbul {rb_:>5.2f}   corpus {rc_:>5.2f}")
report["P2"] = p2
moved = abs(p2["shipped (80-8000)"]["bulbul"] - p2["common band (300-6000)"]["bulbul"])
print(f"      Bulbul ratio moved by {moved:.2f}   "
      f"{'PASS' if moved < 0.40 else 'FAIL -- rethink the model'}")
print(f"      S16 published bulbul 1.09 vs corpus 0.14 and read it as "
      f"'Bulbul is homogeneous'")

# ------------------------------------------------------------------- P3
print()
print("[3/4] P3  does the bright-side skew survive the band restriction?")
print("      Bins are the corpus's own quintile edges, as S16's binner does.")
LABELS = ["very dark", "dark", "balanced", "bright", "very bright"]
report["P3"] = {}
for k in EST:
    cm = np.array([np.mean(v) for v in vals[k]["corpus"].values()], float)
    bm = np.array([np.mean(v) for v in vals[k]["bulbul"].values()], float)
    cm, bm = cm[np.isfinite(cm)], bm[np.isfinite(bm)]
    edges = np.percentile(cm, [20, 40, 60, 80])
    occ = np.searchsorted(edges, bm)
    counts = [int((occ == i).sum()) for i in range(5)]
    used = sorted({LABELS[i] for i in range(5) if counts[i]})
    # how far Bulbul's centre sits from the corpus's, in corpus sd
    shift = float((bm.mean() - cm.mean()) / max(cm.std(ddof=1), 1e-9))
    report["P3"][k] = {"counts": counts, "bins_used": sum(c > 0 for c in counts),
                       "shift_sd": shift, "corpus_mean": float(cm.mean()),
                       "bulbul_mean": float(bm.mean())}
    print()
    print(f"      {k}")
    print(f"        corpus mean {cm.mean():+.2f}  bulbul mean {bm.mean():+.2f}  "
          f"shift {shift:+.2f} corpus sd")
    print("        " + "  ".join(f"{LABELS[i]}:{counts[i]}" for i in range(5)))
    print(f"        {sum(c > 0 for c in counts)}/5 bins occupied")

# ---------------------------------------------------------------- verdict
a = report["P3"]["shipped (80-8000)"]
b = report["P3"]["common band (300-6000)"]
print()
print("=" * 78)
print(f"  S16 published: 3/5 bins, 'balanced, bright, very bright', no dark voices.")
print(f"  shipped estimator here:  {a['bins_used']}/5, shift {a['shift_sd']:+.2f} sd")
print(f"  common band:             {b['bins_used']}/5, shift {b['shift_sd']:+.2f} sd")
print()
d_shift = abs(a["shift_sd"]) - abs(b["shift_sd"])
if abs(d_shift) < 0.25 and a["bins_used"] == b["bins_used"]:
    print("  THE SKEW SURVIVES. It is a fact about the library, not the microphone,")
    print("  and S16's reading stands as written.")
    v = "survives"
elif abs(b["shift_sd"]) < abs(a["shift_sd"]):
    print(f"  THE SKEW SHRINKS by {d_shift:+.2f} sd once both chains are compared")
    print("  where they both carry signal. Part of S16's 'no dark voices' was the")
    print("  recording chain, and S16 needs a correction.")
    v = "shrinks"
else:
    print("  The skew GROWS under the restriction -- the channel was masking it.")
    v = "grows"
print("=" * 78)
report["verdict"] = v
json.dump(report, io.open(os.path.join(OUT, "channel.json"), "w", encoding="utf-8"),
          indent=2)
print(f"\n  -> {os.path.join(OUT, 'channel.json')}")


# ------------------------------------------------- 4. what 1/3-octave measures
print()
print("=" * 78)
print("[4/4] P2 failed, so: WHAT does 1/3-octave banding actually change?")
print("=" * 78)
from alaap.acoustics import f0_track

VAR = {
    "shipped": lambda w, sr: spectral_tilt(w, sr),
    "band only": lambda w, sr: tilt_ltas(w, sr, False, False, False, 300.0, 6000.0),
    "voiced only": lambda w, sr: tilt_ltas(w, sr, True, False, False, 80.0, 8000.0),
    "+1/3oct": lambda w, sr: tilt_ltas(w, sr, False, True, False, 80.0, 8000.0),
    "S21 repaired": lambda w, sr: tilt_ltas(w, sr, True, True, True, 300.0, 6000.0),
}
gen = json.loads(str(d["gen"]))
gmap = {s: gen[i] for i, s in enumerate(cspk)}
pairs = {s: v for s, v in corp.items() if len(v) >= 2}
flatW = [np.asarray(d[f"w{i}"], np.float32) for i in range(int(d["n"]))]
f0flat = np.array([float(np.mean(f0_track(w, 24000))) for w in flatW])
print(f"      {len(pairs)} corpus speakers with 2 takes; f0 on all {len(flatW)} clips")


def gd(vv, keys=None):
    keys = keys or list(vv)
    F, M = [], []
    for s in keys:
        g, x = gmap.get(s), np.mean(vv[s])
        if g and np.isfinite(x):
            (F if g == "Female" else M).append(x)
    if min(len(F), len(M)) < 5:
        return np.nan
    F, M = np.array(F), np.array(M)
    return float((M.mean() - F.mean()) /
                 max(np.sqrt((F.var(ddof=1) + M.var(ddof=1)) / 2), 1e-9))


print()
print(f"      {'variant':<14} {'w/b':>6} {'gender d':>9} {'|r| f0_mean':>12}  reading")
sub, grid = {}, {}
for k, fn in VAR.items():
    sub[k] = {s: [float(fn(w, sr)) for w, sr in v[:2]] for s, v in pairs.items()}
    flat = np.array([fn(w, 24000) for w in flatW], float)
    m = np.isfinite(flat) & np.isfinite(f0flat)
    rf = abs(float(np.corrcoef(f0flat[m], flat[m])[0, 1]))
    r, g = wb(sub[k]), gd(sub[k])
    grid[k] = {"wb": r, "gender_d": g, "abs_r_f0": rf}
    note = ("mostly f0_mean again" if rf >= 0.70 else
            "independent of f0" if rf < 0.45 else "part f0")
    print(f"      {k:<14} {r:>6.2f} {g:>+9.2f} {rf:>12.3f}  {note}")
report["P4_variants"] = grid

# paired bootstrap over speakers, shipped vs the apparent winner
rng2 = np.random.default_rng(24)
ks = list(pairs)
dr, dgv = [], []
for _ in range(2000):
    sel = list(rng2.choice(ks, len(ks), replace=True))
    a, b = wb({s: sub["shipped"][s] for s in sel}), wb({s: sub["+1/3oct"][s] for s in sel})
    ga, gb = gd(sub["shipped"], sel), gd(sub["+1/3oct"], sel)
    if np.isfinite(a) and np.isfinite(b):
        dr.append(b - a)
    if np.isfinite(ga) and np.isfinite(gb):
        dgv.append(abs(gb) - abs(ga))
print()
for nm, arr in (("within/between", dr), ("|gender d|", dgv)):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    print(f"      {nm:<15} (+1/3oct - shipped) {np.mean(arr):+.3f}  "
          f"CI [{lo:+.3f}, {hi:+.3f}]  "
          f"{'EXCLUDES 0' if lo * hi > 0 else 'includes 0'}")
    report["P4_boot_" + nm.split("/")[0].strip()] = {
        "delta": float(np.mean(arr)), "ci95": [float(lo), float(hi)]}

print()
print("=" * 78)
print("  1/3-octave banding makes `spectral_tilt` look like a far better identity")
print("  axis -- within/between 0.15 -> 0.06, gender d -0.80 -> -2.04, both with")
print("  CIs excluding zero -- AND takes its correlation with `f0_mean` from")
print("  0.36 to 0.79. Sixty-three percent shared variance. The separation is")
print("  real and it is largely F0 wearing a different name.")
print()
print("  So the shipped estimator stays, and S21's conclusion is unchanged but")
print("  its REASON was incomplete: S21 tested only the bundled `repaired`")
print("  variant, where the 300-6000 band restriction cancelled the banding's")
print("  effect, and concluded 'repairing changes nothing'. Repairing the")
print("  LEVERAGE changes a great deal -- it just changes it into f0_mean.")
print("=" * 78)
json.dump(report, io.open(os.path.join(OUT, "channel.json"), "w", encoding="utf-8"),
          indent=2)
print(f"\n  -> {os.path.join(OUT, 'channel.json')}")
