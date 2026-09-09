"""
S14d scorer -- controls first, and the test pairs have NO key.

    envs/qwen3/Scripts/python.exe experiments/S14-rate-control/score_symmetry.py \
        --answers "A,tie,A,B,B,tie,B,tie,A,tie,tie"

Eleven answers in pair order: A, B, or tie / t / ? / - / "can't tell".

The six test pairs are `0.5 * (1/m, m)` on one voice and one line, so there is
no correct answer -- the question is whether the listener consistently names
the SLOWER side. That is the asymmetry, and it is scored as a sign test rather
than as right-or-wrong.

    D1  either negative control wrong                -> discard
    D2  ALL THREE positive controls answered confidently -> discard
"""
import argparse
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_sym")
ap = argparse.ArgumentParser()
ap.add_argument("--answers", required=True)
args = ap.parse_args()

key = json.load(io.open(os.path.join(OUT, "ANSWER-KEY.json"), encoding="utf-8"))


def norm(a):
    a = a.strip().lower()
    if a in ("a", "b"):
        return a.upper()
    if a in ("tie", "t", "?", "-", "", "cant", "can't", "can't tell", "cant tell",
             "ct", "both"):
        return "tie"
    sys.exit(f"unparseable answer {a!r}; use A, B or tie")


given = [norm(x) for x in args.answers.split(",")]
if len(given) != len(key):
    sys.exit(f"{len(given)} answers for {len(key)} pairs")
for k, g in zip(key, given):
    k["given"] = g

neg = [k for k in key if k["kind"] == "neg"]
pos = [k for k in key if k["kind"] == "pos"]
test = [k for k in key if k["kind"] == "test"]

print("=" * 78)
print("CONTROLS")
print("=" * 78)
for k in neg:
    ok = k["given"] == k["answer"]
    print(f"  pair{k['pair']:02d}  NEGATIVE (1.00 vs 0.50)  expected {k['answer']}, "
          f"got {k['given']}   {'PASS' if ok else 'FAIL'}")
for k in pos:
    print(f"  pair{k['pair']:02d}  POSITIVE (0.98 vs 1.02)  expected tie, "
          f"got {k['given']}   {'confident' if k['given'] != 'tie' else 'PASS'}")

d1 = not all(k["given"] == k["answer"] for k in neg)
d2 = all(k["given"] != "tie" for k in pos)
print()
if d1 or d2:
    print("!" * 78)
    if d1:
        print("  D1 FIRED: a 2x speed change was not heard as more processed.")
    if d2:
        print("  D2 FIRED: all three ~2% controls got a confident answer.")
    print("  THE SET IS DISCARDED. No asymmetry table.")
    print("!" * 78)
    json.dump(key, io.open(os.path.join(OUT, "scored.json"), "w", encoding="utf-8"),
              indent=2)
    sys.exit(0)

nt = sum(1 for k in pos if k["given"] == "tie")
print(f"  controls PASS -- negatives {len(neg)}/{len(neg)}, positives {nt}/{len(pos)} tie")

print()
print("=" * 78)
print("ASYMMETRY  (no key exists here -- this is the question)")
print("=" * 78)
print(f"  {'pair':>5} {'rates':>14} {'slower':>7} {'said':>6}  reading")
by = {}
for k in sorted(test, key=lambda x: x["slower_rate"]):
    picked = ("SLOWER" if k["given"] == k["slower_side"] else
              "faster" if k["given"] in ("A", "B") else "tie")
    by.setdefault((k["slower_rate"], k["faster_rate"]), []).append(picked)
    print(f"  {k['pair']:>5} {k['slower_rate']:.2f} vs {k['faster_rate']:.2f}"
          f"  {k['slower_side']:>7} {k['given']:>6}  {picked}")

print()
print(f"  {'magnitude':>18} {'slower':>7} {'faster':>7} {'tie':>5}  reading")
tot = {"SLOWER": 0, "faster": 0, "tie": 0}
for (lo, hi), picks in sorted(by.items()):
    c = {p: picks.count(p) for p in ("SLOWER", "faster", "tie")}
    for p in c:
        tot[p] += c[p]
    rd = ("slowing sounds worse" if c["SLOWER"] == len(picks) else
          "speeding sounds worse" if c["faster"] == len(picks) else
          "symmetric / inaudible" if c["tie"] == len(picks) else "split")
    print(f"  {lo:.2f} vs {hi:.2f}      {c['SLOWER']:>7} {c['faster']:>7} "
          f"{c['tie']:>5}  {rd}")

print()
n_dir = tot["SLOWER"] + tot["faster"]
print(f"  overall: slower {tot['SLOWER']}, faster {tot['faster']}, tie {tot['tie']}")
if n_dir == 0:
    print("  Every directional pair came back a tie: the artefact is SYMMETRIC at")
    print("  these magnitudes, and the clean range needs no separate slow edge.")
elif n_dir <= 2:
    print(f"  Only {n_dir} of {len(test)} pairs got a direction. A sign test needs at")
    print("  least 5 consistent calls to reach p < 0.05; this cannot establish an")
    print("  asymmetry either way. Underpowered, not null.")
else:
    lead = max(("SLOWER", tot["SLOWER"]), ("faster", tot["faster"]), key=lambda x: x[1])
    # exact one-sided sign test, p = 0.5^n_dir * sum(C(n_dir, k) for k >= lead)
    from math import comb
    p = sum(comb(n_dir, k) for k in range(lead[1], n_dir + 1)) / 2 ** n_dir
    print(f"  {lead[0]} chosen {lead[1]}/{n_dir} directional pairs, sign test p = {p:.3f}"
          f"   {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    if p >= 0.05:
        print("  Six test pairs is a small sign test. Report the direction as a lean,")
        print("  not a finding.")

print()
print("  REMEMBER WHAT THIS CANNOT SAY: nothing here was compared against an")
print("  untouched render, so it does not give the absolute clean range. It says")
print("  whether the two directions damage equally.")
json.dump(key, io.open(os.path.join(OUT, "scored.json"), "w", encoding="utf-8"), indent=2)
print(f"\n  -> {os.path.join(OUT, 'scored.json')}")
