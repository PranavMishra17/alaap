"""
S14c -- where does re-timing start to sound processed? COMPARATIVE redo.

S14b asked "does this sound natural or processed?" one clip at a time. The
listener called an UNTOUCHED control processed, and 8 of 10 clips overall. The
set was discarded on a pre-committed condition, and it produced the more
important finding: the SYNTHESIS ITSELF is audibly processed, so single-stimulus
naturalness cannot separate the re-timing from the baseline it sits on.

A comparative design cancels a shared baseline, because both sides carry it.

THE OBJECTION S14b RAISED AGAINST A/B, AND HOW THIS ANSWERS IT

    "the two clips have different durations, so the listener can always tell
     which was processed and would be answering 'which is longer'."

True when both clips are the same sentence in the same voice. So they are not.
Each pair puts a RE-TIMED clip against an UNTOUCHED clip of a DIFFERENT line in
a DIFFERENT voice, and every rate is tested TWICE with the roles arranged so
that the duration cue points OPPOSITE WAYS:

    line0 is ~4.0 s, line1 is ~2.1 s -- nearly 2x apart naturally

    pair type L:  line0 @ rate 0.67 (6.0 s)  vs  line1 @ 1.00 (2.1 s)
                  -> the re-timed clip is LONGER
    pair type S:  line1 @ rate 0.67 (3.2 s)  vs  line0 @ 1.00 (4.0 s)
                  -> the re-timed clip is SHORTER

A listener answering on duration gets one of each counterbalanced pair right and
one wrong, landing at 1 of 2. A listener hearing the artefact gets both. THE
DESIGN DIAGNOSES ITS OWN FAILURE MODE, which is what S14b lacked.

CONTROLS -- one that must be easy, one that must be impossible

    NEGATIVE (must be answered correctly):  rate 0.50 vs 1.00, far outside the
        0.67-1.43 bound. A listener who cannot hear THIS cannot hear anything
        subtler, and no test answer in the set is readable.
    POSITIVE (must be answered "can't tell"): 1.00 vs 1.00, two untouched clips.
        A listener who confidently names one as more time-manipulated is
        answering on voice, content or room, not on re-timing.

DISCARD CONDITIONS, WRITTEN DOWN BEFORE THE SET IS SENT:

    D1  either negative control answered wrong  -> discard the whole set
    D2  both positive controls answered with confidence (not "can't tell")
        -> discard; the listener is picking on something other than re-timing

THE QUESTION ASKED, verbatim on the scoresheet, naming the baseline to ignore:

    "Both clips are synthetic speech and both will sound synthetic. Ignore that.
     One of them may have been sped up or slowed down. Which sounds MORE
     TIME-MANIPULATED -- A, B, or can't tell?"

Needs no model: every base clip is an existing rate-1.00 render, re-timed by a
phase vocoder.

    envs/qwen3/Scripts/python.exe experiments/S14-rate-control/run_clean_range_ab.py
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
from alaap.timing import RATE_MAX, RATE_MIN, retime

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_ab")
os.makedirs(OUT, exist_ok=True)
SR = 44100
RATES = [0.67, 0.80, 1.25, 1.43]     # the bound's edges and the guessed clean edges
NEG_RATE = 0.50                       # well outside the bound; must be audible
rng = np.random.default_rng(1408)


def level(w, target_dbfs=-23.0):
    """Loudness is the easiest accidental cue. Match every stimulus."""
    w = np.asarray(w, np.float32)
    r = float(np.sqrt(np.mean(w ** 2)))
    return np.clip(w * (10 ** (target_dbfs / 20.0) / max(r, 1e-8)), -1, 1)


# ------------------------------------------------------- gather base clips
pool = {}
for p in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "S7-indic-catalog", "out",
                                       "hi_ind*_line*.wav"))):
    b = os.path.basename(p)
    voice, line = b.split("_")[1], int(b.split("line")[1].split(".")[0])
    pool.setdefault(line, []).append((voice, p))
if not pool:
    sys.exit("no S7 Hindi renders found; run S7 first")

LONG_LINE = max(pool, key=lambda k: sf.info(pool[k][0][1]).duration)
SHORT_LINE = min(pool, key=lambda k: sf.info(pool[k][0][1]).duration)
d_long = sf.info(pool[LONG_LINE][0][1]).duration
d_short = sf.info(pool[SHORT_LINE][0][1]).duration
print(f"[1/3] base clips: line{LONG_LINE} ~{d_long:.2f}s, line{SHORT_LINE} ~{d_short:.2f}s, "
      f"{len(pool[LONG_LINE])} voices")

# The counterbalance only decorrelates duration if the two lines really differ.
assert d_long > 1.5 * d_short, (
    f"lines are {d_long:.2f}s and {d_short:.2f}s -- too close for the duration "
    f"cue to point opposite ways, which is the whole counterbalance")

voices = [v for v, _ in pool[LONG_LINE] if any(v == w for w, _ in pool[SHORT_LINE])]
rng.shuffle(voices)
assert len(voices) >= 4, f"need >=4 voices present in both lines, have {len(voices)}"


def clip(line, voice):
    w, sr = sf.read(dict((v, p) for v, p in pool[line])[voice], dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    assert sr == SR, sr
    return w


# ------------------------------------------------------------ build pairs
# Each entry: (rate, retimed_line, plain_line, kind). Voices differ within a
# pair so the listener cannot use "same voice, different length".
specs = []
for r in RATES:
    specs.append((r, LONG_LINE, SHORT_LINE, "test"))    # re-timed clip LONGER
    specs.append((r, SHORT_LINE, LONG_LINE, "test"))    # re-timed clip SHORTER
specs.append((NEG_RATE, LONG_LINE, SHORT_LINE, "neg"))
specs.append((NEG_RATE, SHORT_LINE, LONG_LINE, "neg"))
specs.append((1.00, LONG_LINE, SHORT_LINE, "pos"))
specs.append((1.00, SHORT_LINE, LONG_LINE, "pos"))
rng.shuffle(specs)

# A/B side assigned by CONSTRUCTION, not by a coin per pair. Drawing each side
# independently gave 7 "B" against 3 "A" on the first build -- a listener who
# always answered B would have scored 7 of 10 without hearing anything. Exactly
# half the re-timed clips go on each side.
n_sided = sum(1 for sp in specs if sp[3] != "pos")
sides = [True] * (n_sided // 2) + [False] * (n_sided - n_sided // 2)
rng.shuffle(sides)
sides = iter(sides)

print(f"[2/3] writing {len(specs)} pairs to {OUT}")
key = []
for i, (r, rl, pl, kind) in enumerate(specs, 1):
    v1, v2 = rng.choice(voices, 2, replace=False)
    treated = level(retime(clip(rl, v1), r))
    plain = level(clip(pl, v2))
    # A/B order randomised INDEPENDENTLY of everything else
    # positive controls have nothing re-timed, so their side is cosmetic
    a_is_treated = bool(rng.integers(0, 2)) if kind == "pos" else next(sides)
    A, B = (treated, plain) if a_is_treated else (plain, treated)
    sf.write(os.path.join(OUT, f"pair{i:02d}_A.wav"), A, SR)
    sf.write(os.path.join(OUT, f"pair{i:02d}_B.wav"), B, SR)
    ans = "A" if a_is_treated else "B"
    if kind == "pos":
        ans = "tie"                      # nothing was re-timed; both are 1.00
    key.append({
        "pair": i, "kind": kind, "rate": r, "answer": ans,
        "treated_line": rl, "plain_line": pl,
        "treated_voice": str(v1), "plain_voice": str(v2),
        "treated_dur": round(len(treated) / SR, 2),
        "plain_dur": round(len(plain) / SR, 2),
        # which side is longer, so a duration-following listener is detectable
        "longer_side": ("A" if len(A) > len(B) else "B"),
        "duration_cue_agrees": (("A" if len(A) > len(B) else "B") == ans),
    })
    tag = {"test": "", "neg": "  <- NEGATIVE CONTROL", "pos": "  <- POSITIVE CONTROL"}[kind]
    print(f"      pair{i:02d}  rate {r:.2f}  treated={ans:>3}  "
          f"{key[-1]['treated_dur']:.2f}s vs {key[-1]['plain_dur']:.2f}s{tag}")

nA = sum(1 for k in key if k["answer"] == "A")
nB = sum(1 for k in key if k["answer"] == "B")
print(f"      side balance: {nA} A, {nB} B  (a listener answering one side "
      f"always scores {max(nA, nB)}/{nA + nB})")
assert abs(nA - nB) <= 1, "side assignment is exploitable"
agree = [k for k in key if k["kind"] == "test" and k["duration_cue_agrees"]]
print(f"      duration cue agrees with the answer on {len(agree)}/8 test pairs "
      f"(8 would be a confound, 4 is the counterbalance working)")

json.dump(key, io.open(os.path.join(OUT, "ANSWER-KEY.json"), "w", encoding="utf-8"),
          indent=2)

# ------------------------------------------------------------- scoresheet
lines = [
    "# S14c — which sounds MORE time-manipulated?", "",
    "**Do not open `ANSWER-KEY.json` until you have written every answer down.**", "",
    "Play `pair01_A.wav` then `pair01_B.wav`, and so on. For each pair answer:", "",
    "> **Both clips are synthetic speech and both will sound synthetic.**",
    "> **Ignore that.** One of them may have been sped up or slowed down.",
    "> **Which sounds MORE TIME-MANIPULATED — A, B, or can't tell?**", "",
    "Notes that matter:", "",
    "- The two clips in a pair are **different sentences in different voices**, and",
    "  **one is naturally about twice as long as the other**. Length tells you nothing",
    "  about which was manipulated — that is deliberate, and half the pairs are built",
    "  so the longer clip is the untouched one.",
    "- \"Can't tell\" is a real answer and some pairs are built to earn it. Use it.",
    "- Listen to each pair at most twice. First impression is the measurement.", "",
    f"{len(specs)} pairs, about {len(specs) * 15 // 60} minutes.", "",
    "| pair | A / B / can't tell |", "|---|---|",
]
lines += [f"| {i:02d} |  |" for i in range(1, len(specs) + 1)]
lines += ["", "---", "",
          "## What this set can and cannot conclude", "",
          "Declared before you listen:", "",
          "- **2 pairs are negative controls** (a rate far outside the usable bound).",
          "  If either comes back wrong, the whole set is discarded — it would mean the",
          "  artefact is not audible at all, and no subtler answer could be read.",
          "- **2 pairs are positive controls** where NEITHER clip was touched. If both",
          "  get a confident answer, the set is discarded — you would be picking on",
          "  voice or content rather than on re-timing.",
          "- The remaining 8 locate the edge, 2 pairs per rate. **Two judgements per",
          "  rate is thin**: it can show an edge is obviously wrong, not confirm one",
          "  is exactly right.", ""]
io.open(os.path.join(OUT, "SCORESHEET.md"), "w", encoding="utf-8").write("\n".join(lines))
print(f"[3/3] -> {os.path.join(OUT, 'SCORESHEET.md')}")
print()
print(f"  Currently recorded as a GUESS: listener_clean_range (0.8, 1.25),")
print(f"  inside the measured bound {RATE_MIN}-{RATE_MAX}. This set tests it.")
