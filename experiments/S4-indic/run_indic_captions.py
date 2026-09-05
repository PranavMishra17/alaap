"""
S4 — the Indic caption pipeline, and a real test of whether it measures anything.

PHASE-05 section 8 requires this to start regardless of the S5 gate, because no
Indic corpus carries natural-language voice descriptions at all. Everything the
project does downstream of a caption -- the mapper, the retrieval index, minting
-- needs (caption, voice) pairs, and for Indic those pairs do not exist. They
have to be manufactured by the measure-first route:

    audio -> measure -> percentile-bin -> caption from bins
                   ^                          |
                   +----- verify by re-parsing +

The whole approach rests on the measurements being real. For English that was
checkable against a literature baseline (LibriTTS-R, 20.7 phones/s voiced). For
Hindi there is no such baseline in our notes, and worse, `speaking_rate` had to
grow a whole new phone counter (count_phones_indic) that nothing has validated.

So this experiment does NOT begin by producing captions. It begins by trying to
falsify the measurements, using a control the corpus hands us for free.

THE CONTROL. The SPRINGLab mirrors ship Data-Speech annotations alongside the
audio -- utterance_pitch_mean, snr, speaking_rate -- produced by a DIFFERENT
toolchain from alaap.acoustics (praat/penn vs librosa.pyin, and their phonemizer
vs our grapheme rules). Two independent instruments measuring the same physical
quantity must agree. Where they agree, our numbers are real. Where they do not,
the disagreement localises the bug:

    our f0_mean       vs their utterance_pitch_mean   -> is pitch tracking sane?
    our snr_db        vs their snr                    -> is the VAD sane?
    our speaking_rate vs their speaking_rate          -> IS count_phones_indic SANE?

The third is the one that matters. count_phones_indic is a hand-written
grapheme rule set; if it is wrong, it is wrong silently, and every Hindi caption
downstream inherits a bogus speaking-rate axis. A weak correlation here kills
the Indic caption pipeline until it is fixed.

Note the two rates are not the same statistic -- ours divides by VOICED seconds,
theirs by total duration -- so the test is CORRELATION, not equality. A constant
offset is expected and harmless (percentile binning is invariant to it); a low
correlation is fatal.

SECOND QUESTION. Can English bins be reused for Hindi? If the two languages'
attribute distributions differ, a binner fitted on English puts most Hindi
voices into the same bin and the captions stop discriminating. Percentile bins
are fitted per corpus for exactly this reason, but the SIZE of the difference
decides whether we need per-language binners forever or just once.

Outputs -> experiments/S4-indic/out/

    envs/qwen3/Scripts/python.exe experiments/S4-indic/run_indic_captions.py
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
from alaap.acoustics import (Attributes, Binner, measure, count_phones_indic,
                             detect_script, voiced_mask, BIN_LABELS)
from alaap.captions import (caption_from_bins, target_bins_from_text,
                            ORDER as CAPTION_ORDER)
from alaap.data import stream_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--per-speaker", type=int, default=2)
ap.add_argument("--refit", action="store_true", help="ignore the cache")
args = ap.parse_args()

CACHE = os.path.join(OUT, f"measured_{args.corpus}_{args.n}_{args.per_speaker}.npz")


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan"), len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def spearman(x, y):
    """Rank correlation -- what percentile binning actually depends on."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


# ------------------------------------------------------- 1. measure the corpus
if os.path.exists(CACHE) and not args.refit:
    print(f"[1/5] loading cached measurements from {os.path.basename(CACHE)}")
    d = np.load(CACHE, allow_pickle=True)
    attrs = [Attributes.from_dict(a) for a in json.loads(str(d["attrs"]))]
    refs = json.loads(str(d["refs"]))
    metas = json.loads(str(d["metas"]))
    texts = json.loads(str(d["texts"]))
else:
    print(f"[1/5] streaming {args.n} clips from {args.corpus}")
    clips = stream_clips(args.corpus, n=args.n, per_speaker=args.per_speaker,
                         dev_only=True, min_dur=2.0, max_dur=15.0,
                         progress_every=100)
    print(f"[1/5] measuring {len(clips)} clips (librosa.pyin, the slow part)")
    attrs, refs, metas, texts, t0 = [], [], [], [], time.time()
    for i, c in enumerate(clips):
        a = measure(c.wav, c.text or "", sr=24000)
        attrs.append(a.to_dict())
        # Their speaking_rate is per TOTAL second; ours is per VOICED second.
        # Record both plus the raw ingredients so the comparison is auditable.
        vs = float(voiced_mask(c.wav, 24000).sum() * 0.010)
        refs.append({
            "ref_pitch": c.meta.get("utterance_pitch_mean"),
            "ref_pitch_std": c.meta.get("utterance_pitch_std"),
            "ref_snr": c.meta.get("snr"),
            "ref_rate": c.meta.get("speaking_rate"),
            "ref_duration": c.meta.get("duration"),
            "our_phones": count_phones_indic(c.text or ""),
            "our_voiced_s": vs,
            "our_total_s": float(len(c.wav) / 24000),
        })
        metas.append({"speaker_id": c.speaker_id, "gender": c.gender,
                      "age": c.age, "area": c.meta.get("area"),
                      "scenario": c.meta.get("scenario"),
                      "state": c.meta.get("state")})
        texts.append(c.text or "")
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"      {i+1}/{len(clips)} | {el/60:.1f} min | "
                  f"eta {el/(i+1)*(len(clips)-i-1)/60:.1f} min", flush=True)
    np.savez_compressed(CACHE, attrs=json.dumps(attrs), refs=json.dumps(refs),
                        metas=json.dumps(metas), texts=json.dumps(texts))
    attrs = [Attributes.from_dict(a) for a in attrs]

n = len(attrs)
scripts = {detect_script(t) for t in texts if t}
print(f"      {n} clips | {len({m['speaker_id'] for m in metas})} speakers "
      f"| scripts {scripts}")

# ------------------------------------- 2. falsify the measurements (the point)
print("\n[2/5] cross-checking against the corpus's own Data-Speech annotations")

ours_pitch = [a.f0_mean for a in attrs]
ours_snr = [a.snr_db for a in attrs]
# convert our voiced-second rate back to a per-total-second rate to match theirs
ours_rate_total = [r["our_phones"] / max(r["our_total_s"], 1e-6) for r in refs]

checks = [
    ("f0_mean          vs utterance_pitch_mean", ours_pitch,
     [r["ref_pitch"] for r in refs], 0.90,
     "pitch tracking (librosa.pyin vs theirs)"),
    ("snr_db           vs snr", ours_snr,
     [r["ref_snr"] for r in refs], 0.30,
     "VAD / noise floor -- different definitions, expect weak"),
    ("speaking_rate    vs speaking_rate", ours_rate_total,
     [r["ref_rate"] for r in refs], 0.70,
     "COUNT_PHONES_INDIC -- this is the one that matters"),
]

print(f"  {'comparison':<42} {'pearson':>8} {'spearman':>9} {'floor':>6}  verdict")
print(f"  {'-'*42} {'-'*8} {'-'*9} {'-'*6}  {'-'*7}")
cross = {}
fatal = []
for label, a, b, floor, note in checks:
    pr, npair = pearson(a, b)
    sr_, _ = spearman(a, b)
    ok = np.isfinite(sr_) and sr_ >= floor
    if not ok and floor >= 0.70:
        fatal.append(label)
    print(f"  {label:<42} {pr:>8.3f} {sr_:>9.3f} {floor:>6.2f}  "
          f"{'ok' if ok else 'BELOW FLOOR'}")
    print(f"  {'':<42} {note}")
    cross[label.split()[0]] = {"pearson": pr, "spearman": sr_,
                               "floor": floor, "n": npair, "passed": bool(ok)}

# scale offset on the rate, for the record
ratio = np.array(ours_rate_total) / np.maximum(
    np.array([r["ref_rate"] for r in refs], dtype=float), 1e-6)
print(f"\n  our phones/s over theirs: median {np.median(ratio):.3f} "
      f"(IQR {np.percentile(ratio,25):.3f}-{np.percentile(ratio,75):.3f})")
print("  a stable ratio is fine -- percentile bins are invariant to scale.")
print("  a stable ratio BELOW 1 is the expected sign of medial schwa deletion,")
print("  which their phonemizer models and our grapheme rules do not.")

if fatal:
    print("\n" + "!" * 78)
    print("STOPPING. " + ", ".join(fatal) + " failed its floor.")
    print("Captions built on a broken axis are worse than no captions, because")
    print("they look fine. Fix the measurement before generating anything.")
    print("!" * 78)
    json.dump({"corpus": args.corpus, "n": n, "cross_check": cross,
               "captions_generated": False},
              open(os.path.join(OUT, f"results_{args.corpus}.json"), "w"),
              indent=2)
    sys.exit(1)

# ------------------------------------------ 3. does Hindi need its own binner?
LANG = (args.corpus.rsplit("_", 1)[-1] or "??").upper()
print(f"\n[3/5] {LANG} vs English attribute distributions")
EN_REF = os.path.join("experiments", "S2", "out", "corpus_globe_v2_2500_1.npz")
lang_cmp = {}
if os.path.exists(EN_REF):
    de = np.load(EN_REF, allow_pickle=True)
    en_attrs = [Attributes.from_dict(a) for a in json.loads(str(de["attrs"]))]
    print(f"      English reference: {len(en_attrs)} GLOBE_V2 clips")
    print(f"  {'attribute':<16} {LANG.lower()+' med':>10} {'eng med':>10} "
          f"{'shift':>8}  {LANG.lower()+' pcts in eng bins':>24}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*8}  {'-'*24}")
    en_binner = Binner.fit(en_attrs)
    for key in ("f0_mean", "f0_cv", "speaking_rate", "hnr_db", "spectral_tilt"):
        # via the ENGLISH binner's _value, so hnr_db carries the same pitch
        # correction the English edges were fitted under. Comparing raw hnr_db
        # against corrected edges would manufacture a difference.
        h = np.array([en_binner._value(a, key) for a in attrs], float)
        e = np.array([en_binner._value(a, key) for a in en_attrs], float)
        h, e = h[np.isfinite(h)], e[np.isfinite(e)]
        # where do Hindi values land in the ENGLISH bin edges?
        edges = en_binner.edges.get(key)
        occ = None
        if edges is not None:
            idx = np.digitize(h, np.asarray(edges, float))
            occ = np.bincount(np.clip(idx, 0, 4), minlength=5) / max(len(h), 1)
        shift = (np.median(h) - np.median(e)) / max(e.std(), 1e-9)
        lang_cmp[key] = {"target_median": float(np.median(h)),
                         "english_median": float(np.median(e)),
                         "shift_in_english_sd": float(shift),
                         "occupancy_in_english_bins":
                             [float(x) for x in occ] if occ is not None else None}
        occs = " ".join(f"{x:.2f}" for x in occ) if occ is not None else "-"
        print(f"  {key:<16} {np.median(h):>10.2f} {np.median(e):>10.2f} "
              f"{shift:>+8.2f}  {occs:>24}")
    print("\n  'shift' is in English standard deviations. Occupancy is the")
    print(f"  fraction of {LANG} clips in each of the five ENGLISH bins;")
    print(f"  0.20 x5 would mean English bins happen to fit {LANG} perfectly.")
else:
    print(f"      (no English reference at {EN_REF}; skipping)")

# --------------------------------------------------- 4. fit bins and caption
print(f"\n[4/5] fitting {LANG} bins and writing captions")
binner = Binner.fit(attrs)
binner.save(os.path.join(OUT, f"binner_{args.corpus}.json"))
bins = [binner.bin_one(a) for a in attrs]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]

# ------------------------------------------------- 5. reversibility of caption
print("[5/5] verifying captions round-trip back to the bins that made them")

# Score ONLY the axes the caption actually renders into prose. caption_from_bins
# writes ORDER[:max_attrs] -- five of the eight binned axes -- so jitter, snr_db
# and shimmer never appear in the text and can never be parsed back out. An
# earlier version of this check required every binned axis to round-trip and
# duly reported 0.0% exact, which measured the scorer, not the captions.
# ORDER is weight-ordered and caption_from_bins writes ORDER[:max_attrs].
# Hardcoding 5 silently stopped scoring an axis when max_attrs became 6.
from alaap.captions import caption_from_bins as _cfb
import inspect as _insp
_MAXA = _insp.signature(_cfb).parameters["max_attrs"].default
EXPRESSED = [f for f in CAPTION_ORDER[:_MAXA] if f in BIN_LABELS]
OMITTED = [f for f in BIN_LABELS if f not in EXPRESSED]

exact, per_axis_hits, per_axis_tot = 0, {}, {}
for b, c in zip(bins, caps):
    parsed = target_bins_from_text(c)
    all_ok = True
    for k in EXPRESSED:
        if k not in b:
            continue
        per_axis_tot[k] = per_axis_tot.get(k, 0) + 1
        if parsed.get(k) == b[k]:
            per_axis_hits[k] = per_axis_hits.get(k, 0) + 1
        else:
            all_ok = False
    exact += all_ok
rt = exact / max(len(caps), 1)
print(f"      scoring {len(EXPRESSED)} expressed axes: {', '.join(EXPRESSED)}")
print(f"      not written into prose, so not scorable: {', '.join(OMITTED)}")
print(f"      {exact}/{len(caps)} captions round-trip exactly ({rt:.1%})")
for k in sorted(per_axis_tot):
    print(f"        {k:<16} {per_axis_hits.get(k,0)/per_axis_tot[k]:.1%}")

with io.open(os.path.join(OUT, f"captions_{args.corpus}.txt"), "w", encoding="utf-8") as f:
    for i in range(min(25, len(caps))):
        f.write(f"[{i}] spk={metas[i]['speaker_id']} {metas[i]['gender']} "
                f"{metas[i]['age']} {metas[i]['area']}\n")
        f.write(f"     text:    {texts[i][:70]}\n")
        f.write(f"     caption: {caps[i]}\n\n")

json.dump({"corpus": args.corpus, "n": n,
           "n_speakers": len({m["speaker_id"] for m in metas}),
           "scripts": sorted(str(s) for s in scripts),
           "cross_check": cross,
           "rate_ratio_median": float(np.median(ratio)),
           "language_comparison": lang_cmp,
           "caption_roundtrip_exact": rt,
           "caption_roundtrip_per_axis":
               {k: per_axis_hits.get(k, 0) / per_axis_tot[k] for k in per_axis_tot},
           "caption_axes_expressed": EXPRESSED,
           "caption_axes_not_expressed": OMITTED,
           "captions_generated": True},
          open(os.path.join(OUT, f"results_{args.corpus}.json"), "w"), indent=2)

print("\n" + "=" * 78)
print("S4 — Indic caption pipeline")
print("=" * 78)
print(f"  corpus        {args.corpus}  ({n} clips, "
      f"{len({m['speaker_id'] for m in metas})} speakers)")
print(f"  measurements  cross-checked against an independent toolchain:")
for k, v in cross.items():
    print(f"                  {k:<16} spearman {v['spearman']:>6.3f} "
          f"(floor {v['floor']:.2f}) {'ok' if v['passed'] else 'FAIL'}")
print(f"  captions      {len(caps)} written, {rt:.1%} round-trip exactly")
print(f"  artefacts     {OUT}")
print("=" * 78)
