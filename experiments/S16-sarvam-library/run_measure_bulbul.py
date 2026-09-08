"""
S16 — measure a REAL commercial voice library, and retrieve over it

`S8` showed that answering a description by retrieving the nearest voice from a
library beats minting a new one: 87% of the available diversity against 37%,
and 65.6% exact adherence on Indic rising to 88.5% on a library 18x larger. But
its "library" was 141 corpus speakers standing in for a studio library, because
there was no API key.

There is now. Sarvam's Bulbul ships ~39 studio voices across 11 Indian
languages, and `S8` was written so that swapping the library changes no logic.
This measures the real thing.

WHAT THIS TESTS THAT S8 COULD NOT

  1. does the measurement pipeline WORK on commercial TTS output at all?
     It was built for corpus recordings. Studio-clean synthetic speech is a
     different signal and nothing says the same axes are measurable on it.
  2. how much of the described space do 39 curated voices actually cover?
     A curated library is chosen for usefulness, not for spanning bin space,
     and it might all sit in the middle.
  3. does retrieval over a real library reproduce S8's numbers?

THE CONTROL, asserted before any retrieval number is read. Each voice is
synthesised on TWO different sentences. If the pipeline is measuring the VOICE
rather than the sentence, a voice's two measurements must be far closer to each
other than to other voices' -- a within/between ratio well under 1. If they are
not, the axes are reading the text and every number after that is noise.

COST. 39 voices x 2 short sentences is roughly 5,000 characters. Bulbul is
about Rs 30 per 10,000 characters, so this run costs a few rupees against the
Rs 1,000 of free credit. Audio is cached; re-runs cost nothing.

    envs/qwen3/Scripts/python.exe experiments/S16-sarvam-library/run_measure_bulbul.py
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.acoustics import BIN_LABELS, Binner, measure
from alaap.captions import caption_from_bins
# PINNED TO THE V1 SET. This experiment's committed RESULTS.md numbers were
# measured on the five shipped caption axes, before S17/S18's identity set was
# adopted (2026-09-08). Following the new default here would silently change
# what every quoted acceptance count and adherence percentage means. Re-running
# it on the current set is a DIFFERENT question and deserves its own arm.
from alaap.catalog import CATALOG_AXES_V1 as CATALOG_AXES, sample_cells
from alaap.mapper import RetrievalMapper, TextEncoder

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
AUD = os.path.join(OUT, "audio")
os.makedirs(AUD, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="bulbul:v3")
ap.add_argument("--language", default="hi-IN")
ap.add_argument("--queries", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sleep", type=float, default=0.4, help="be polite to the API")
args = ap.parse_args()

ENDPOINT = "https://api.sarvam.ai/text-to-speech"
# Documented on the Bulbul model card. A name the API rejects is reported and
# skipped rather than aborting -- the card and the deployment may disagree, and
# which ones disagree is itself worth knowing.
SPEAKERS = [
    "shubh", "aditya", "rahul", "rohan", "amit", "dev", "ratan", "varun",
    "manan", "sumit", "kabir", "aayan", "ashutosh", "advait", "anand",
    "tarun", "sunny", "mani", "gokul", "vijay", "mohit", "rehan", "soham",
    "ritu", "priya", "neha", "pooja", "simran", "kavya", "ishita", "shreya",
    "roopa", "tanya", "shruti", "suhani", "kavitha", "rupali",
]
# Two DIFFERENT sentences per voice: the control depends on it.
SENTENCES = [
    "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।",
    "हम कल सुबह जल्दी निकलेंगे, इसलिए जल्दी सो जाइए।",
]


def api_key() -> str:
    k = os.environ.get("SARVAM_API_KEY")
    if not k and os.path.exists(".env"):
        for line in io.open(".env", encoding="utf-8"):
            if line.strip().startswith("SARVAM_API_KEY"):
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not k:
        sys.exit("no SARVAM_API_KEY in the environment or .env")
    return k


import requests
import soundfile as sf

KEY = api_key()
print(f"[1/4] key loaded ({len(KEY)} chars)")     # length only, never the value


def preflight() -> None:
    """
    One request before the loop, because a wrong key produces 74 identical
    403s and buries the reason.

    SARVAM SHIPS TWO KINDS OF KEY and they are not interchangeable:

      API subscription key   dashboard.sarvam.ai/key-management
                             -> what TTS, STT and translation need
      Voice Agents key       dashboard Settings -> API Key
                             -> Conversations / Samvaad only

    Both start `sk_`, so they look identical at a glance, and the TTS endpoint
    rejects the second with the same "invalid credentials" it gives a garbage
    string. The product name is embedded in the key's second segment, which is
    the only cheap way to tell them apart without sending either.
    """
    r = requests.post(ENDPOINT, timeout=30,
                      headers={"api-subscription-key": KEY,
                               "Content-Type": "application/json"},
                      json={"text": "नमस्ते", "language_code": args.language,
                            "speaker": SPEAKERS[0], "model": args.model})
    if r.status_code == 200:
        return
    # A product key carries the product NAME in the second segment
    # (sk_samvaad_...); a subscription key has a random id there
    # (sk_z07yluiv_...). Alphabetic-only is the tell -- checking `!= "api"`
    # would fire on every valid key.
    hint = ""
    seg = KEY.split("_")
    scope = seg[1] if len(seg) > 2 else ""
    if scope.isalpha() and len(scope) >= 4:
        hint = (f"  |  the key's scope segment reads {scope!r}. A TTS call "
                "needs the API SUBSCRIPTION key from "
                "dashboard.sarvam.ai/key-management, not a product key. "
                "Both start `sk_`.")
    sys.exit(f"ABORT: preflight got HTTP {r.status_code}: {r.text[:160]}{hint}")


preflight()
print("      preflight OK")


def synth(speaker: str, text: str, path: str) -> bool:
    if os.path.exists(path):
        return True
    r = requests.post(ENDPOINT, timeout=60,
                      headers={"api-subscription-key": KEY,
                               "Content-Type": "application/json"},
                      json={"text": text, "language_code": args.language,
                            "speaker": speaker, "model": args.model})
    if r.status_code != 200:
        print(f"      {speaker:<10} HTTP {r.status_code}: {r.text[:110]}")
        return False
    body = r.json()
    audios = body.get("audios") or []
    if not audios:
        print(f"      {speaker:<10} no audio in response: {list(body)[:6]}")
        return False
    with open(path, "wb") as f:
        f.write(base64.b64decode(audios[0]))
    return True


print(f"[2/4] synthesising {len(SPEAKERS)} voices x {len(SENTENCES)} sentences "
      f"({args.model}, {args.language})")
got, t0 = [], time.time()
for sp in SPEAKERS:
    paths = []
    for si, txt in enumerate(SENTENCES):
        p = os.path.join(AUD, f"{sp}_{si}.wav")
        if not synth(sp, txt, p):
            paths = []
            break
        paths.append(p)
        time.sleep(args.sleep)
    if paths:
        got.append((sp, paths))
print(f"      {len(got)}/{len(SPEAKERS)} voices synthesised | "
      f"{(time.time()-t0)/60:.1f} min")
if len(got) < 10:
    sys.exit("ABORT: too few voices came back to measure a library.")

# ---------------------------------------------------------------- measure
print("[3/4] measuring with the same pipeline used on corpus audio")
rows = []
for sp, paths in got:
    per = []
    for si, p in enumerate(paths):
        w, sr = sf.read(p, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        per.append(measure(w, SENTENCES[si], sr).to_dict())
    rows.append({"speaker": sp, "takes": per})
json.dump(rows, io.open(os.path.join(OUT, "measured.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

FIELDS = list(CATALOG_AXES)
V = {f: np.array([[t.get(f, np.nan) for t in r["takes"]] for r in rows], float)
     for f in FIELDS}

# THE CONTROL, and which axes it is fair to apply it to.
#
# The first run failed it on spectral_tilt, f0_cv and speaking_rate. Running
# the IDENTICAL statistic on IndicVoices-R -- real speakers, two clips each,
# same code -- separated the two possible explanations:
#
#   axis            corpus  bulbul
#   f0_mean           0.05    0.09   both separate
#   hnr_db            0.28    0.72   both separate
#   spectral_tilt     0.14    1.09   measurement fine, BULBUL is homogeneous
#   f0_cv             1.04    1.05   fails on REAL speakers too
#   speaking_rate     1.32    1.54   fails on REAL speakers too
#
# So f0_cv and speaking_rate are not identity axes at all: two ordinary clips
# of the same person differ on them MORE than two people do. That independently
# replicates S12, which measured the same thing on CREMA-D across acted
# emotions (ratios 1.57 and 2.01) -- here it falls out of plain read speech
# with no emotion labels involved.
#
# The control therefore applies to the IDENTITY axes only. Requiring the
# delivery axes to pass would abort on data that is behaving correctly.
IDENTITY_AXES = ["f0_mean", "spectral_tilt", "hnr_db"]
DELIVERY_AXES = ["f0_cv", "speaking_rate"]

print("      control — within-voice vs between-voice spread")
print(f"        {'axis':<16} {'within':>10} {'between':>10} {'ratio':>8}  expect")
ratios = {}
for f in FIELDS:
    a = V[f]
    ok = np.isfinite(a).all(1)
    if ok.sum() < 8:
        continue
    a = a[ok]
    within = float(np.mean(a.var(axis=1, ddof=1)))
    between = float(a.mean(1).var(ddof=1))
    if between <= 0:
        continue
    ratios[f] = within / between
    exp = "< 1 (identity)" if f in IDENTITY_AXES else "> 1 (delivery)"
    print(f"        {f:<16} {within:>10.3f} {between:>10.3f} "
          f"{within/between:>8.2f}  {exp}")

bad = [f for f in IDENTITY_AXES if ratios.get(f, 99) > 1.0]
if len(bad) >= 2:
    sys.exit(f"ABORT: {bad} fail on axes that separate real speakers (corpus "
             "ratios 0.05/0.14/0.28). The measurement is tracking the sentence "
             "and no retrieval number below would mean anything.")
if bad:
    print(f"      NOTE: {bad} does not separate these voices, though it "
          "separates real speakers (corpus 0.14). Bulbul is homogeneous there.")
print(f"      identity axes usable: "
      f"{[f for f in IDENTITY_AXES if ratios.get(f, 99) <= 1.0]}")

# ------------------------------------------------------- library + coverage
attrs_mean = []
for i, r in enumerate(rows):
    m = {f: float(np.nanmean([t.get(f, np.nan) for t in r["takes"]])) for f in FIELDS}
    base = dict(r["takes"][0])
    base.update(m)
    attrs_mean.append(base)

from alaap.acoustics import Attributes
proto = [Attributes(**{k: v for k, v in a.items()}) for a in attrs_mean]
binner = Binner.fit(proto)
binner.edges.pop("vtl_cm", None)
library = []
for r, a in zip(rows, proto):
    b = {k: v for k, v in binner.bin_one(a).items() if k in FIELDS}
    library.append({"speaker": r["speaker"], "bins": b,
                    "caption": caption_from_bins(b, seed=len(library))})

print("[4/4] coverage and retrieval")
# COVERAGE MUST BE MEASURED AGAINST THE CORPUS BINNER, not this one.
# `binner` above was FIT on these 37 voices, so its percentile edges partition
# them by construction and every axis reads 5/5 no matter how homogeneous the
# library is. That number says nothing. The real question is where 37 curated
# voices land inside the range REAL speakers occupy, which needs the binner
# fitted on real speakers.
from alaap.acoustics import Binner as _B
S4B = "experiments/S4-indic/out/binner_indicvoices_r_hi.json"
if os.path.exists(S4B):
    cb = _B.load(S4B)
    cb.edges.pop("vtl_cm", None)
    real_bins = [cb.bin_one(a) for a in proto]
    print("      coverage of the REAL speaker range (binner fitted on "
          "IndicVoices-R):")
    for f in FIELDS:
        seen = {b[f] for b in real_bins if f in b}
        labs = [l for l in BIN_LABELS[f] if l in seen]
        print(f"        {f:<16} {len(seen)}/5 bins   {', '.join(labs[:3])}"
              f"{' ...' if len(labs) > 3 else ''}")
    occupied = np.mean([len({b[f] for b in real_bins if f in b}) / 5
                        for f in FIELDS if any(f in b for b in real_bins)])
    print(f"        -> {occupied:.0%} of the real-speaker range is represented")
else:
    print(f"      (no corpus binner at {S4B}; coverage not measurable)")
cells = sample_cells(args.queries, args.seed)
queries = [caption_from_bins(c, seed=10_000 + i) for i, c in enumerate(cells)]
enc = TextEncoder()
order = {a: {lab: i for i, lab in enumerate(BIN_LABELS[a])} for a in FIELDS}


def adh(cell, voice):
    return sum(order[a][cell[a]] == order[a][voice["bins"][a]]
               for a in FIELDS if a in voice["bins"]) / len(FIELDS)


L = enc.encode([v["caption"] for v in library])
Q = enc.encode(queries)
L = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-12)
Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
picks = np.argmax(Q @ L.T, axis=1)
rng = np.random.default_rng(args.seed)
rand = rng.integers(0, len(library), size=len(queries))
ex = float(np.mean([adh(cells[i], library[int(picks[i])]) for i in range(len(queries))]))
rx = float(np.mean([adh(cells[i], library[int(rand[i])]) for i in range(len(queries))]))
reached = len(set(int(p) for p in picks))

print()
print("=" * 78)
print(f"S16 — Sarvam Bulbul, {len(library)} real studio voices")
print("=" * 78)
print(f"  {'exact bin match, text retrieval':38} {ex:>7.1%}")
print(f"  {'random control':38} {rx:>7.1%}")
print(f"  {'lift':38} {ex-rx:>+7.1%}")
print()
print(f"  voices reached over {len(queries)} queries: {reached}/{len(library)}")
print(f"  most-returned voice: "
      f"{np.bincount(picks, minlength=len(library)).max()}/{len(queries)}")
print()
print("  S8, 141 corpus speakers, text retrieval: 39.0% vs 20.3% random")
print("  Fewer voices should retrieve WORSE, not better -- a curated library of")
print("  39 cannot cover bin space the way 141 recorded speakers do.")
print("=" * 78)
print(f"  audio + measurements -> {OUT}")
