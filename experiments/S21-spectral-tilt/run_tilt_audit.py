"""
S21 -- scrutinise `spectral_tilt`, the one strong identity axis nobody has checked

S17 found `vtl_cm` was never noise -- the ESTIMATOR was broken, averaging through
F2. S18 then audited all 17 attributes for *whether they separate speakers* but
never asked, of the ones that do, *whether they separate them for the reason the
name claims*. `spectral_tilt` is the outstanding case: ratio 0.19 (strong),
redundancy 0.54, gender d -0.65, and never scrutinised. Base rate for this check
finding something in this project: 3 for 3.

THE SUSPICION, stated before any measurement.

`spectral_tilt` fits an unweighted line to (log2 f, dB) over LINEAR FFT bins.
Ordinary least squares weights a point by its leverage, (x - xbar)^2, so in
log2-frequency the two ENDS of the band dominate. Measured on this project's
settings (sr 24000, n_fft 2048, band 80-8000 Hz):

      27 bins below 500 Hz carry 43.2% of the leverage
     341 bins above 4 kHz carry 29.4%
     the 308 bins in between carry 27.4%

The 27 low bins are not a smooth spectrum. They are the first few F0 harmonics,
and where they land depends on the speaker's pitch. A 100 Hz voice puts four
harmonics under 500 Hz; a 220 Hz voice puts two. If that moves the fit, then
`spectral_tilt` is partly a restatement of `f0_mean` wearing a different name,
and its gender d of -0.65 is pitch leaking through rather than brightness.

FOUR PRE-COMMITTED ASSERTIONS. Stated here, before the numbers, per the rule this
project earned the hard way. Each is a property the axis must have if its NAME is
true. Failing any one is a bug report about the estimator, not about the world.

  A1  RECOVERY.   On noise shaped to a known slope s, the reading must be within
                  +-0.5 dB/oct of s. (Is the formula even right?)
  A2  F0 INVARIANCE. On synthetic voices with an IDENTICAL envelope slope and F0
                  swept 90-250 Hz, the readings must span < 0.5 dB/oct. This is
                  the load-bearing one: "spectral tilt" that moves with pitch at
                  fixed brightness is measuring pitch.
  A3  SILENCE.    Appending 1 s of silence must move the reading < 0.3 dB/oct.
                  (`spectral_tilt` averages over ALL frames, unlike every other
                  axis here, which uses `voiced_mask`.)
  A4  BANDWIDTH.  A 16 kHz resample round-trip must move it < 0.5 dB/oct.
                  Otherwise the axis reads the channel, and cross-corpus
                  comparisons are comparing microphones.

    envs/qwen3/Scripts/python.exe experiments/S21-spectral-tilt/run_tilt_audit.py
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
from alaap.acoustics import SR, spectral_tilt, voiced_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260908)

A1_TOL, A2_TOL, A3_TOL, A4_TOL = 0.5, 0.5, 0.3, 0.5


# ----------------------------------------------------------------- candidates
def tilt_ltas(wav, sr=SR, voiced_only=True, third_octave=True, power=True, flo=80.0, fhi=8000.0):
    """
    The repaired candidate: a long-term average spectrum done the textbook way.

    Three changes from the shipped estimator, each testable in isolation:
      voiced_only   -- restrict to voiced frames, as every other axis here does
      power         -- average power, not magnitude (an LTAS is a power average)
      third_octave  -- collapse to 1/3-octave bands before fitting, so each
                       octave contributes equally instead of leverage piling up
                       on 27 low bins and 341 high ones
    """
    import librosa
    w = np.asarray(wav, dtype=np.float32)
    S = np.abs(librosa.stft(w, n_fft=2048, hop_length=512))
    if voiced_only:
        vm = voiced_mask(w, sr, frame_ms=25.0, hop_ms=512 * 1000.0 / sr)
        k = min(len(vm), S.shape[1])
        if vm[:k].sum() >= 8:
            S = S[:, :k][:, vm[:k]]
    P = (S ** 2) if power else S
    m = P.mean(1) + 1e-20
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hi = min(fhi, 0.9 * sr / 2)
    ok = (f > flo) & (f < hi)
    fo, mo = f[ok], m[ok]
    db = (10.0 if power else 20.0) * np.log10(mo)
    if third_octave:
        edges = flo * 2.0 ** (np.arange(0, int(np.ceil(3 * np.log2(hi / flo))) + 1) / 3.0)
        cx, cy = [], []
        for lo, up in zip(edges[:-1], edges[1:]):
            sel = (fo >= lo) & (fo < up)
            if sel.sum():
                cx.append(np.sqrt(lo * up))
                cy.append(db[sel].mean())
        if len(cx) < 6:
            return float("nan")
        x, y = np.log2(np.array(cx)), np.array(cy)
    else:
        x, y = np.log2(fo), db
    A = np.vstack([x, np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


VARIANTS = {
    "shipped": lambda w, sr: spectral_tilt(w, sr),
    "+voiced": lambda w, sr: tilt_ltas(w, sr, True, False, False),
    "+power": lambda w, sr: tilt_ltas(w, sr, False, False, True),
    "+1/3oct": lambda w, sr: tilt_ltas(w, sr, False, True, False),
    "+f>300": lambda w, sr: tilt_ltas(w, sr, False, False, False, flo=300.0),
    "repaired": lambda w, sr: tilt_ltas(w, sr, True, True, True, 300.0, 6000.0),
}


# ------------------------------------------------------------------- stimuli
def shaped_noise(slope_db_oct, dur=3.0, sr=SR):
    """White noise filtered so its magnitude spectrum has exactly this slope."""
    n = int(dur * sr)
    X = np.fft.rfft(RNG.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    g = np.ones_like(f)
    nz = f > 20
    g[nz] = 10 ** (slope_db_oct * np.log2(f[nz] / 1000.0) / 20.0)
    g[~nz] = g[nz][0]
    w = np.fft.irfft(X * g, n).astype(np.float32)
    return w / (np.abs(w).max() + 1e-9) * 0.5


def synth_voice(f0, slope_db_oct, dur=3.0, sr=SR, aspiration=0.05):
    """
    A source-filter voice: harmonics at f0 PLUS aspiration noise, both through
    the SAME envelope of known slope.

    BRIGHTNESS IS HELD CONSTANT ACROSS f0 BY CONSTRUCTION. Any reading that moves
    with f0 is the estimator, not the stimulus.

    The aspiration term is not decoration, it is what makes this a fair test. A
    first version of this stimulus added FLAT noise, and every variant then
    "failed" A2 at r(f0) ~ +0.97 -- a suspiciously uniform number, which in this
    project has always meant the setup. The mechanism: a 250 Hz voice has 2.8x
    fewer harmonics than a 90 Hz one across the band, so 2.8x more bins fall in
    the flat inter-harmonic floor, dragging the fit toward flat. That is an
    artefact of flat noise. Real aspiration is filtered by the same vocal tract,
    so the inter-harmonic valleys carry the envelope slope too.
    """
    t = np.arange(int(dur * sr)) / sr
    w = np.zeros_like(t)
    for k in range(1, int((0.45 * sr) / f0) + 1):
        fk = k * f0
        if fk > 0.45 * sr:
            break
        a = 10 ** (slope_db_oct * np.log2(fk / 1000.0) / 20.0)
        w += a * np.sin(2 * np.pi * fk * t + RNG.uniform(0, 2 * np.pi))
    w = w / (w.std() + 1e-9)
    w = w + aspiration * shaped_noise(slope_db_oct, dur, sr) / 0.5 * 1.0
    env = np.clip(np.sin(2 * np.pi * 3.0 * t) * 0.5 + 0.75, 0, None)  # syllabic
    w = w * env
    return (w / (np.abs(w).max() + 1e-9) * 0.5).astype(np.float32)


report, failures = {}, []


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# --------------------------------------------------- A1 recovery of a known slope
head("A1  RECOVERY -- shaped noise of known slope, tolerance +-%.1f dB/oct" % A1_TOL)
slopes = [0.0, -3.0, -6.0, -9.0, -12.0]
print(f"  {'true':>6} " + " ".join(f"{k:>9}" for k in VARIANTS))
a1 = {k: [] for k in VARIANTS}
for s in slopes:
    w = shaped_noise(s)
    row = {k: fn(w, SR) for k, fn in VARIANTS.items()}
    for k, v in row.items():
        a1[k].append(v - s)
    print(f"  {s:>6.1f} " + " ".join(f"{row[k]:>9.2f}" for k in VARIANTS))
print(f"  {'|err|':>6} " + " ".join(f"{np.max(np.abs(a1[k])):>9.2f}" for k in VARIANTS))
report["A1_max_abs_err"] = {k: float(np.max(np.abs(v))) for k, v in a1.items()}
for k, v in report["A1_max_abs_err"].items():
    print(f"    {k:<10} max error {v:5.2f}  {'PASS' if v < A1_TOL else 'FAIL'}")
if report["A1_max_abs_err"]["shipped"] >= A1_TOL:
    failures.append("A1 recovery")

# ------------------------------------------------------- A2 invariance to F0
head("A2  F0 INVARIANCE -- fixed envelope slope, F0 swept; tolerance <%.1f dB/oct spread" % A2_TOL)
f0s, NDRAW = [90, 110, 140, 175, 210, 250], 5
print("  envelope slope held at -9.0 dB/oct for every row; only F0 changes.")
print(f"  {NDRAW} independent draws per F0, so the estimator's OWN repeatability")
print("  can be separated from any trend with F0 -- a control for the control.")
print(f"  {'f0':>6} " + " ".join(f"{k:>9}" for k in VARIANTS))
a2 = {k: [] for k in VARIANTS}       # mean per f0
a2sd = {k: [] for k in VARIANTS}     # within-f0 sd == the noise floor
for f0 in f0s:
    draws = {k: [] for k in VARIANTS}
    for _ in range(NDRAW):
        w = synth_voice(f0, -9.0)
        for k, fn in VARIANTS.items():
            draws[k].append(fn(w, SR))
    for k in VARIANTS:
        a2[k].append(float(np.mean(draws[k])))
        a2sd[k].append(float(np.std(draws[k], ddof=1)))
    print(f"  {f0:>6d} " + " ".join(f"{a2[k][-1]:>9.2f}" for k in VARIANTS))
print(f"  {'spread':>6} " + " ".join(f"{np.ptp(a2[k]):>9.2f}" for k in VARIANTS))
print(f"  {'noise':>6} " + " ".join(f"{np.mean(a2sd[k]):>9.2f}" for k in VARIANTS))
report["A2_spread"] = {k: float(np.ptp(v)) for k, v in a2.items()}
report["A2_noise_sd"] = {k: float(np.mean(v)) for k, v in a2sd.items()}
report["A2_r_with_f0"] = {k: float(np.corrcoef(f0s, v)[0, 1]) for k, v in a2.items()}
# with n=6 points, |r| >= 0.811 is p < 0.05 two-tailed
R_CRIT = 0.811
for k in VARIANTS:
    sp, r, nz = (report["A2_spread"][k], report["A2_r_with_f0"][k],
                 report["A2_noise_sd"][k])
    tag = "reads F0" if abs(r) >= R_CRIT else ("clean" if abs(r) < 0.4 else "unresolved")
    print(f"    {k:<10} spread {sp:5.2f}  noise sd {nz:4.2f}  r(f0) {r:+.3f}  {tag}")
print()
print("  NOTE: the 0.5 dB/oct spread threshold was pre-committed WITHOUT knowing")
print("  the estimator's repeatability, and the noise row shows it was not a")
print("  resolvable target. The surviving evidence in A2 is the CORRELATION with")
print("  F0, which noise cannot manufacture. Reported as such rather than quietly")
print("  relaxing the number.")
if abs(report["A2_r_with_f0"]["shipped"]) >= 0.4:
    failures.append("A2 F0 invariance (by correlation)")

# ----------------------------------------------------------- A3 silence padding
head("A3  SILENCE -- append 1 s of silence, tolerance <%.1f dB/oct shift" % A3_TOL)
a3 = {k: [] for k in VARIANTS}
for f0 in (110, 175, 250):
    w = synth_voice(f0, -9.0)
    wp = np.concatenate([w, np.zeros(SR, np.float32)])
    for k, fn in VARIANTS.items():
        a3[k].append(fn(wp, SR) - fn(w, SR))
report["A3_max_shift"] = {k: float(np.max(np.abs(v))) for k, v in a3.items()}
for k in VARIANTS:
    v = report["A3_max_shift"][k]
    print(f"    {k:<10} max shift {v:5.2f} dB/oct   {'PASS' if v < A3_TOL else 'FAIL'}")
if report["A3_max_shift"]["shipped"] >= A3_TOL:
    failures.append("A3 silence")

# ----------------------------------------------------------------- A4 bandwidth
head("A4  BANDWIDTH -- 24k -> 16k -> 24k round-trip, tolerance <%.1f dB/oct" % A4_TOL)
import librosa
a4 = {k: [] for k in VARIANTS}
for f0 in (110, 175, 250):
    w = synth_voice(f0, -9.0)
    wb = librosa.resample(librosa.resample(w, orig_sr=SR, target_sr=16000),
                          orig_sr=16000, target_sr=SR).astype(np.float32)
    for k, fn in VARIANTS.items():
        a4[k].append(fn(wb, SR) - fn(w, SR))
report["A4_max_shift"] = {k: float(np.max(np.abs(v))) for k, v in a4.items()}
for k in VARIANTS:
    v = report["A4_max_shift"][k]
    print(f"    {k:<10} max shift {v:5.2f} dB/oct   {'PASS' if v < A4_TOL else 'FAIL'}")
if report["A4_max_shift"]["shipped"] >= A4_TOL:
    failures.append("A4 bandwidth")

head("VERDICT on the SHIPPED estimator")
if failures:
    print(f"  FAILS {len(failures)} of 4 pre-committed assertions: " + ", ".join(failures))
else:
    print("  passes all four. The axis measures what its name says.")
report["failures"] = failures
json.dump(report, io.open(os.path.join(OUT, "synthetic.json"), "w", encoding="utf-8"),
          indent=2)
print(f"\n  -> {os.path.join(OUT, 'synthetic.json')}")


# ------------------------------------------------- B  does it hold on REAL speech?
head("B  REAL SPEECH -- held-out IndicVoices-R Hindi clips (cached by S20b)")
RP = os.path.join("experiments", "S20-naturalness-gate", "out",
                  "sweep_audio_indicvoices_r_hi_300.npz")
if not os.path.exists(RP):
    print(f"  {RP} not present -- skipping (synthetic result stands on its own)")
else:
    # THESE CLIPS ARE AT 16 kHz, NOT 24 kHz. S20b resampled them for WavLM
    # (run_layer_sweep.py:77) while computing f0 on the 24 kHz original. A first
    # pass here measured them at SR=24000 and every frequency landed 1.5x wrong,
    # putting the "band floor above F0" at 200 Hz instead of 300 -- which is
    # assertion A4 (bandwidth) biting the audit itself. Measured at the true rate.
    SR_CACHE = 16000
    d = np.load(RP, allow_pickle=True)
    nreal = int(d["n"])
    f0 = np.asarray(d["f0"], float)
    W = [np.asarray(d[f"R{i}"], np.float32) for i in range(nreal)]
    print(f"  {nreal} clips at {SR_CACHE} Hz, F0 {np.nanmin(f0):.0f}-"
          f"{np.nanmax(f0):.0f} Hz (median {np.nanmedian(f0):.0f})")
    print()
    print("  On real speech, brightness and pitch are genuinely correlated in the")
    print("  world, so a nonzero r here is NOT by itself a fault. What the")
    print("  synthetic A2 licenses is the COMPARISON: an estimator that reads F0")
    print("  artefactually must correlate with it MORE than one that does not.")
    print()
    print(f"  {'variant':<10} {'r(f0)':>8} {'mean':>8} {'sd':>8}   n_finite")
    realr = {}
    tilts = {}
    for k, fn in VARIANTS.items():
        v = np.array([fn(w, SR_CACHE) for w in W], float)
        ok = np.isfinite(v) & np.isfinite(f0)
        r = float(np.corrcoef(f0[ok], v[ok])[0, 1])
        realr[k] = r
        tilts[k] = v
        print(f"  {k:<10} {r:>+8.3f} {np.nanmean(v):>8.2f} {np.nanstd(v):>8.2f}   {ok.sum():>4d}")
    report["B_real_r_with_f0"] = realr
    dr = realr["shipped"] - realr["repaired"]
    # Fisher z test on two correlations from the SAME sample (dependent) is not
    # valid here; report the difference and a bootstrap CI instead.
    bs = []
    idx = np.arange(nreal)
    for _ in range(2000):
        s = RNG.choice(idx, nreal, replace=True)
        a = np.corrcoef(f0[s], tilts["shipped"][s])[0, 1]
        b = np.corrcoef(f0[s], tilts["repaired"][s])[0, 1]
        bs.append(a - b)
    lo, hi95 = np.percentile(bs, [2.5, 97.5])
    report["B_delta_r"] = {"delta": dr, "ci95": [float(lo), float(hi95)]}
    print()
    print(f"  r(shipped) - r(repaired) = {dr:+.3f}   95% CI [{lo:+.3f}, {hi95:+.3f}]"
          f"   ({'excludes 0' if lo * hi95 > 0 else 'includes 0'})")
    rr = float(np.corrcoef(tilts['shipped'], tilts['repaired'])[0, 1])
    report["B_r_shipped_vs_repaired"] = rr
    print(f"  r(shipped, repaired)     = {rr:+.3f}  -- how much the axis MOVES if repaired")

json.dump(report, io.open(os.path.join(OUT, "synthetic.json"), "w", encoding="utf-8"),
          indent=2)
print(f"\n  -> {os.path.join(OUT, 'synthetic.json')}")
