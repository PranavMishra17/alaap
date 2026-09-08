"""
S14c scorer -- controls first, then the rates, and never the other way round.

    envs/qwen3/Scripts/python.exe experiments/S14-rate-control/score_clean_range_ab.py \
        --answers "A,B,tie,B,tie,A,B,tie,B,A,A,B"

Twelve comma-separated answers in pair order. Each is A, B, or one of
tie / t / ? / - / cant / "can't tell".

THE DISCARD CONDITIONS ARE CHECKED BEFORE ANY RATE IS REPORTED, and if either
fires the rate table is not printed at all. S14b's whole value came from
honouring exactly this, so it is enforced in code rather than remembered.

    D1  either NEGATIVE control wrong  -> the artefact is not audible at all
    D2  both POSITIVE controls answered confidently -> something other than
        re-timing is being judged
"""
import argparse
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_ab")

ap = argparse.ArgumentParser()
ap.add_argument("--answers", required=True)
args = ap.parse_args()

key = json.load(io.open(os.path.join(OUT, "ANSWER-KEY.json"), encoding="utf-8"))


def norm(a):
    a = a.strip().lower()
    if a in ("a", "b"):
        return a.upper()
    if a in ("tie", "t", "?", "-", "", "cant", "can't", "can't tell", "cant tell"):
        return "tie"
    sys.exit(f"unparseable answer {a!r}; use A, B or tie")


given = [norm(x) for x in args.answers.split(",")]
if len(given) != len(key):
    sys.exit(f"{len(given)} answers for {len(key)} pairs")
for k, g in zip(key, given):
    k["given"] = g
    k["correct"] = (g == k["answer"])

neg = [k for k in key if k["kind"] == "neg"]
pos = [k for k in key if k["kind"] == "pos"]
test = [k for k in key if k["kind"] == "test"]

print("=" * 78)
print("CONTROLS  (reported before anything else, and they gate the rest)")
print("=" * 78)
for k in neg:
    print(f"  pair{k['pair']:02d}  NEGATIVE (rate {k['rate']:.2f})  "
          f"expected {k['answer']}, got {k['given']}   "
          f"{'PASS' if k['correct'] else 'FAIL'}")
for k in pos:
    conf = k["given"] != "tie"
    print(f"  pair{k['pair']:02d}  POSITIVE (nothing re-timed)  expected tie, "
          f"got {k['given']}   {'confident -> bad' if conf else 'PASS'}")

d1 = not all(k["correct"] for k in neg)
d2 = all(k["given"] != "tie" for k in pos)
print()
if d1 or d2:
    print("!" * 78)
    if d1:
        print("  D1 FIRED: a negative control was answered wrong. A rate far outside")
        print("  the usable bound was not heard as more time-manipulated, so nothing")
        print("  subtler in this set can be read.")
    if d2:
        print("  D2 FIRED: both positive controls got a confident answer, on pairs")
        print("  where NEITHER clip was re-timed. Something other than re-timing is")
        print("  being judged -- voice, content or length.")
    print()
    print("  THE SET IS DISCARDED. The rate table is deliberately not printed.")
    print("  This is a result about the instrument, not a nuisance: S14b's discard")
    print("  is what established that the synthesis itself sounds processed.")
    print("!" * 78)
    json.dump(key, io.open(os.path.join(OUT, "scored.json"), "w", encoding="utf-8"),
              indent=2)
    sys.exit(0)

print("  controls %d/%d -- the set is readable" % (
    sum(k["correct"] for k in neg) + sum(k["given"] == "tie" for k in pos),
    len(neg) + len(pos)))

# ------------------------------------------------ did they follow duration?
dur_follow = sum(1 for k in test if k["given"] == k["longer_side"])
print()
print(f"  duration check: {dur_follow}/{len(test)} answers picked the LONGER clip.")
if dur_follow >= len(test) - 1:
    print("  -> that is answering on length, not artefact. The counterbalance was")
    print("     built so length is right half the time; near-perfect agreement")
    print("     with it means the rate answers below are not about re-timing.")
elif dur_follow <= 1:
    print("  -> consistently picking the SHORTER clip, which is also a length")
    print("     strategy. Same caution applies.")
else:
    print("  -> not a length strategy. The rate answers stand.")

print()
print("=" * 78)
print("BY RATE  (2 judgements each -- enough to reject an edge, not to confirm one)")
print("=" * 78)
by = {}
for k in test:
    by.setdefault(k["rate"], []).append(k)
print(f"  {'rate':>6} {'heard':>7}  reading")
for r in sorted(by):
    ks = by[r]
    n = sum(k["correct"] for k in ks)
    if n == len(ks):
        rd = "AUDIBLE -- outside the clean range"
    elif n == 0:
        rd = "inaudible -- inside the clean range"
    else:
        rd = "split; at the edge, or under-powered at n=2"
    print(f"  {r:>6.2f} {n:>4}/{len(ks):<2}  {rd}")

aud = [r for r in sorted(by) if sum(k["correct"] for k in by[r]) == len(by[r])]
inaud = [r for r in sorted(by) if sum(k["correct"] for k in by[r]) == 0]
print()
lo = [r for r in inaud if r < 1]
hi = [r for r in inaud if r > 1]
print("  Clean range implied by what was NOT heard: "
      f"{min(lo) if lo else '?'} to {max(hi) if hi else '?'}")
print(f"  Currently recorded as a guess: (0.8, 1.25)")
print()
print("  Two judgements per rate cannot confirm an edge is exactly right. It can")
print("  show one is wrong, which is what this set is for.")
json.dump(key, io.open(os.path.join(OUT, "scored.json"), "w", encoding="utf-8"), indent=2)
print(f"\n  -> {os.path.join(OUT, 'scored.json')}")
