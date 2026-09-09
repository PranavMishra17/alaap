"""
S14d -- stop asking "which one was manipulated".

S14b and S14c both had their controls fail, in opposite ways, and together they
map the tension:

                            duration cue hidden   baseline shared
    same sentence + voice          NO                  YES        (S14b)
    different sentence + voice     YES                 NO         (S14c)

S14b's listener called an untouched clip processed. S14c varied the voice to
hide the duration cue and its positive controls -- two untouched clips -- both
got a confident answer.

THE WAY OUT IS TO STOP ASKING WHICH CLIP WAS MANIPULATED. Every pair here is
the SAME VOICE saying the SAME LINE, re-timed by the same magnitude in OPPOSITE
directions:

    rate 1/m   vs   rate m        e.g. 0.80 vs 1.25

Neither is the original, so length identifies nothing. The baseline is not
merely shared, it is IDENTICAL -- the same underlying render, the same phase
vocoder, the same everything except the sign of the manipulation. What is left
to hear is the artefact.

WHAT THIS MEASURES, AND WHAT IT DOES NOT

  DOES:     asymmetry. Is slowing more audible than speeding, and from what
            magnitude? S14c's discarded answers hypothesise yes -- 0.67 and
            0.80 heard 2/2, 1.25 heard 0/2 -- and the listener's own report of
            a 0.70 render was "60% slowed down, 40% slow speech effect".
  DOES NOT: the absolute clean range. Nothing here compares against an
            untouched render, because that comparison is exactly what duration
            gives away. If both directions tie at every magnitude, the artefact
            is symmetric and this says nothing about whether either is audible.

CONTROLS

  POSITIVE (must be "can't tell"): 0.98 vs 1.02, same voice, same line. Both
      through the phase vocoder, a magnitude far too small to hear. THREE of
      them, because S14c's discard condition used two and fired on a coin
      flip -- a listener who says "can't tell" 25% of the time trips a
      both-confident rule 56% of the time. Three makes it 0.42. Still not
      good, and said out loud rather than hidden.
  NEGATIVE (must be answered): 0.50 vs 1.00, same voice, same line. Gross, and
      yes duration helps here -- its job is to prove engagement, not to be
      cue-free. S14c passed this 2/2.

DISCARD CONDITIONS, WRITTEN DOWN BEFORE THE SET IS SENT:

    D1  either negative control answered wrong        -> discard
    D2  ALL THREE positive controls answered with confidence -> discard

    envs/qwen3/Scripts/python.exe experiments/S14-rate-control/run_symmetry.py
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
from alaap.timing import retime

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_sym")
os.makedirs(OUT, exist_ok=True)
SR = 44100
MAGS = [1.10, 1.25, 1.43]          # each becomes (1/m, m)
rng = np.random.default_rng(90909)


def level(w, target_dbfs=-23.0):
    w = np.asarray(w, np.float32)
    r = float(np.sqrt(np.mean(w ** 2)))
    return np.clip(w * (10 ** (target_dbfs / 20.0) / max(r, 1e-8)), -1, 1)


bases = []
for p in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "S7-indic-catalog", "out",
                                       "hi_ind*_line*.wav"))):
    b = os.path.basename(p)
    bases.append((b.split("_")[1], int(b.split("line")[1].split(".")[0]), p))
if not bases:
    sys.exit("no S7 Hindi renders found")
# longer clips carry more of the artefact and are easier to judge
bases = [x for x in bases if sf.info(x[2]).duration >= 2.5]
rng.shuffle(bases)
print(f"[1/3] {len(bases)} base renders >= 2.5 s, "
      f"{len({v for v, _, _ in bases})} voices")

specs = [(1.0 / m, m, "test") for m in MAGS]
specs += [(1.0 / m, m, "test") for m in MAGS]        # each magnitude twice
specs += [(0.98, 1.02, "pos")] * 3
specs += [(1.00, 0.50, "neg")] * 2
rng.shuffle(specs)

# A/B side balanced by construction: exactly half the SLOWER clips on each side
n = len(specs)
sides = [True] * (n // 2) + [False] * (n - n // 2)
rng.shuffle(sides)
sides = iter(sides)

print(f"[2/3] writing {n} pairs to {OUT}")
key = []
for i, (r1, r2, kind) in enumerate(specs, 1):
    voice, line, path = bases[(i - 1) % len(bases)]
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    assert sr == SR, sr
    # r1 is the slower rate for TEST and POSITIVE pairs; for the NEGATIVE
    # control r1 = 1.00 and r2 = 0.50, so r2 is the slower one. Track the side
    # of r1 and derive the genuinely-slower side from the rates, or the key
    # mislabels every negative control.
    c1, c2 = level(retime(w, r1)), level(retime(w, r2))
    r1_is_a = next(sides)
    A, B = (c1, c2) if r1_is_a else (c2, c1)
    slower_side = ("A" if r1_is_a else "B") if r1 < r2 else ("B" if r1_is_a else "A")
    sf.write(os.path.join(OUT, f"pair{i:02d}_A.wav"), A, SR)
    sf.write(os.path.join(OUT, f"pair{i:02d}_B.wav"), B, SR)
    if kind == "pos":
        ans = "tie"
    elif kind == "neg":
        ans = slower_side                    # the 0.50 clip, whichever side
    else:
        ans = "?"                            # THE QUESTION -- no key exists
    key.append({"pair": i, "kind": kind, "slower_rate": round(r1, 3),
                "faster_rate": round(r2, 3), "answer": ans,
                "r1_side": "A" if r1_is_a else "B",
                "slower_side": slower_side,
                "voice": voice, "line": line,
                "dur_A": round(len(A) / SR, 2), "dur_B": round(len(B) / SR, 2)})
    tag = {"test": "", "pos": "  <- POSITIVE CONTROL",
           "neg": "  <- NEGATIVE CONTROL"}[kind]
    print(f"      pair{i:02d}  {r1:.2f} vs {r2:.2f}  slower={slower_side}  "
          f"{voice} line{line}{tag}")

nslowA = sum(1 for k in key if k["slower_side"] == "A")
print(f"      side balance: slower clip on A in {nslowA}/{n} pairs")
assert abs(nslowA - n / 2) <= 1
json.dump(key, io.open(os.path.join(OUT, "ANSWER-KEY.json"), "w", encoding="utf-8"),
          indent=2)

# --------------------------------------------------------- one playthrough
def tone(f, dur, amp_dbfs=-30.0):
    t = np.arange(int(SR * dur)) / SR
    return (np.sin(2 * np.pi * f * t) *
            (10 ** (amp_dbfs / 20.0)) *
            np.minimum(1.0, np.minimum(t, dur - t) / 0.02)).astype(np.float32)


parts = [np.zeros(SR, np.float32)]
for k in key:
    a, _ = sf.read(os.path.join(OUT, f"pair{k['pair']:02d}_A.wav"), dtype="float32")
    b, _ = sf.read(os.path.join(OUT, f"pair{k['pair']:02d}_B.wav"), dtype="float32")
    parts += [tone(440, 0.40), np.zeros(int(SR * 0.35), np.float32), a,
              np.zeros(int(SR * 0.45), np.float32), tone(880, 0.15),
              np.zeros(int(SR * 0.35), np.float32), b,
              np.zeros(int(SR * 1.5), np.float32)]
full = np.clip(np.concatenate(parts), -1, 1)
sf.write(os.path.join(OUT, "ALL-PAIRS.wav"), full, SR)

lines = [
    "# S14d — which sounds MORE processed?", "",
    "**Do not open `ANSWER-KEY.json` first.**", "",
    f"Play `ALL-PAIRS.wav` once — {len(full) / SR / 60:.1f} minutes, {n} pairs.", "",
    "> LOW tone = new pair, **A** next.  HIGH blip = **B** next.", "",
    "> **Every pair is the SAME VOICE saying the SAME SENTENCE.** Both clips have",
    "> been time-adjusted — one slower, one faster, by the same amount. Neither is",
    "> the original, so length tells you nothing about which was touched more.",
    "> **Which sounds MORE PROCESSED — A, B, or can't tell?**", "",
    "- \"Can't tell\" is the *correct* answer on several pairs. Please use it.",
    "- Judge the artefact, not the speed. One will obviously be slower; that is",
    "  not the question. The question is which one sounds more *damaged*.",
    "- At most two listens per pair.", "",
    "| pair | A / B / can't tell |", "|---|---|",
]
lines += [f"| {i:02d} |  |" for i in range(1, n + 1)]
lines += ["", "---", "",
          "## Declared before you listen", "",
          "- **2 negative controls** are grossly manipulated. Wrong on either and the",
          "  set is discarded.",
          "- **3 positive controls** are adjusted by ~2%, far too little to hear. If",
          "  ALL THREE get a confident answer the set is discarded. (`S14c` used two",
          "  and that fired on a coin flip — this is better, not good.)",
          "- The remaining 6 ask whether **slowing and speeding damage equally**.",
          "  They cannot tell you the absolute clean range: nothing here is compared",
          "  against an untouched render, because that is exactly the comparison",
          "  length gives away.", ""]
io.open(os.path.join(OUT, "SCORESHEET.md"), "w", encoding="utf-8").write("\n".join(lines))
print(f"[3/3] -> {OUT}  ({len(full) / SR / 60:.1f} min)")
