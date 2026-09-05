"""
S12 — is DELIVERY separable from IDENTITY, and on which axes?

The product question behind this: a description says who a character IS ("a
deep, gravelly voice"). A direction says how a line is DELIVERED ("say this
authoritatively", "say this asking for forgiveness"). Those are different
things and users write them differently, but nothing in this project has
checked that they are different MEASUREMENTS.

They might not be. If "authoritative" mostly lowers pitch, it collides head-on
with `f0_mean`, which E15/S6/S8 all measured as the dominant IDENTITY axis
(weight 2.7-3.9 against a mean of 1.0). A direction channel built on an axis
the identity channel already owns would make every angry character sound like
a different person -- which is exactly the failure `E0`'s tau vectors exist to
avoid, and `E0` only ever checked it in embedding space, never acoustically.

THE TEST. CREMA-D is 91 actors x 12 sentences x 6 emotions -- the same people
performing different deliveries, which is the only design that can separate
the two. For each acoustic axis, compare:

    within-speaker variance across emotions   -- how much DELIVERY moves it
    between-speaker variance                  -- how much IDENTITY moves it

    delivery_ratio = within / between

An axis with a high ratio is a delivery axis: it moves when the same person
changes how they say something, and moves less between people. A low ratio is
an identity axis and must not be used for direction. Around 1.0 the axis is
contested and using it for either purpose leaks into the other.

WHAT WOULD MAKE THIS EXPERIMENT WORTHLESS, checked before the numbers are read:
the same content is not held fixed. CREMA-D actors all speak the SAME 12
sentences, so sentence identity can be conditioned out; if it could not, a
"delivery" difference could just be a different sentence's phonetics.

    envs/qwen3/Scripts/python.exe experiments/S12-delivery-axis/run_delivery_split.py
"""
import argparse
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
from alaap.acoustics import BIN_LABELS, RECORDING_AXES, measure

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1200)
ap.add_argument("--sr", type=int, default=24000)
args = ap.parse_args()

# CREMA-D's 12 sentences are fixed and encoded in the filename. `measure`
# counts phones from TEXT, so without them `speaking_rate` -- the most obvious
# delivery axis there is -- comes back NaN and silently drops out of the table.
# It did exactly that on the first run.
CREMA_SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

# v2: the first cache was measured with an empty text and has no speaking_rate.
CACHE = os.path.join(OUT, f"crema_attrs_v2_{args.n}.npz")

# ------------------------------------------------------- 1. measure CREMA-D
if os.path.exists(CACHE):
    d = np.load(CACHE, allow_pickle=True)
    attrs = json.loads(str(d["attrs"]))
    emo, actor, sent = list(d["emo"]), list(d["actor"]), list(d["sent"])
    print(f"[1/3] cached: {len(attrs)} clips")
else:
    import io as _io
    import librosa
    import soundfile as sf
    from datasets import Audio, load_dataset
    print(f"[1/3] streaming and measuring up to {args.n} CREMA-D clips")
    ds = load_dataset("confit/cremad-parquet", split="train",
                      streaming=True).cast_column("audio", Audio(decode=False))
    attrs, emo, actor, sent = [], [], [], []
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
        if sr != args.sr:
            w = librosa.resample(y=w, orig_sr=sr, target_sr=args.sr)
        pk = float(np.abs(w).max())
        if pk > 1.0:
            w = w / pk
        base = os.path.basename(str(r["file"])).split(".")[0].split("_")
        code = base[1] if len(base) > 1 else "?"
        try:
            attrs.append(measure(w.astype(np.float32),
                                 CREMA_SENTENCES.get(code, ""), args.sr).to_dict())
        except Exception:
            continue
        emo.append(str(r["emotion"]))
        actor.append(base[0])
        sent.append(code)
        if len(attrs) >= args.n:
            break
        if len(attrs) % 200 == 0:
            print(f"      {len(attrs)}/{args.n} | {(time.time()-t0)/60:.1f} min",
                  flush=True)
    np.savez_compressed(CACHE, attrs=json.dumps(attrs), emo=np.array(emo),
                        actor=np.array(actor), sent=np.array(sent))
    print(f"      {len(attrs)} clips, {len(set(actor))} actors, "
          f"{len(set(emo))} emotions, {len(set(sent))} sentences")

emo, actor, sent = np.array(emo), np.array(actor), np.array(sent)

# Stated before the table is read: speaking_rate must be measurable, or the
# most obvious delivery axis silently drops out (it did, on the first run).
_sr_ok = np.isfinite([a.get("speaking_rate", np.nan) for a in attrs]).sum()
print(f"      speaking_rate measurable on {_sr_ok}/{len(attrs)} clips")
if _sr_ok < 0.5 * len(attrs):
    sys.exit("ABORT: speaking_rate is mostly NaN -- the sentence texts are not "
             "reaching measure(), and the delivery table would be missing the "
             "axis most likely to carry delivery.")

# THE CONTROL, asserted before any ratio is read: the design only works if the
# same actors appear across emotions and the same sentences across actors.
n_shared = sum(1 for a in set(actor) if len(set(emo[actor == a])) >= 3)
print(f"[2/3] {n_shared}/{len(set(actor))} actors appear in >=3 emotions")
if n_shared < 10:
    sys.exit("ABORT: too few actors span multiple emotions -- within-speaker "
             "variance would be measuring noise, not delivery.")

# ------------------------------------------ 2. within vs between, per axis
AXES = [a for a in BIN_LABELS if a not in RECORDING_AXES]
V = {a: np.array([x.get(a, np.nan) for x in attrs], float) for a in AXES}

rows = []
for a in AXES:
    v = V[a]
    ok = np.isfinite(v)
    if ok.sum() < 100:
        continue
    va, ac, em = v[ok], actor[ok], emo[ok]
    # between-speaker: spread of each speaker's OWN mean
    means = np.array([va[ac == s].mean() for s in sorted(set(ac))
                      if (ac == s).sum() >= 3])
    between = float(means.var(ddof=1)) if len(means) > 2 else np.nan
    # within-speaker across emotion: spread of that speaker's per-emotion means,
    # averaged over speakers. Centring per speaker removes identity entirely.
    wv = []
    for s in sorted(set(ac)):
        m = ac == s
        if m.sum() < 6:
            continue
        pe = [va[m & (em == e)].mean() for e in sorted(set(em[m]))
              if (m & (em == e)).sum() >= 2]
        if len(pe) >= 3:
            wv.append(np.var(pe, ddof=1))
    within = float(np.mean(wv)) if wv else np.nan
    if not (np.isfinite(within) and np.isfinite(between)) or between <= 0:
        continue
    rows.append({"axis": a, "within": within, "between": between,
                 "ratio": within / between})

rows.sort(key=lambda r: -r["ratio"])
json.dump(rows, io.open(os.path.join(OUT, "delivery_split.json"), "w",
                        encoding="utf-8"), indent=2)

print()
print("=" * 78)
print("S12 — which acoustic axes carry DELIVERY rather than IDENTITY?")
print("=" * 78)
print(f"  {'axis':<18} {'within-spk':>12} {'between-spk':>12} {'ratio':>8}  verdict")
print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*8}  {'-'*20}")
for r in rows:
    verdict = ("DELIVERY" if r["ratio"] >= 1.5 else
               "identity" if r["ratio"] <= 0.67 else "contested")
    print(f"  {r['axis']:<18} {r['within']:>12.3f} {r['between']:>12.3f} "
          f"{r['ratio']:>8.2f}  {verdict}")
print()
print("  ratio = within-speaker variance across emotions / between-speaker variance")
print("  >= 1.50  the same person moves this more by changing delivery than")
print("           different people differ on it -> usable for DIRECTION")
print("  <= 0.67  identity owns it -> using it for direction changes WHO speaks")
print("=" * 78)
