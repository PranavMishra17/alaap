"""
S7 — where does the INDIC catalog saturate?

S6 minted four Indic voices and every one cleared every floor. Four voices
prove the path works; they say nothing about how many distinct identities the
backend can actually hold. E11 needed 40 English voices before the catalog
started returning voices it already had. That number is the product: a catalog
of 500 voices where 300 are audibly the same voice is a catalog of 200 voices
and a support problem.

This is E11's question on MioCodec, with E11's sampler (`alaap.catalog`), so
the two curves are directly comparable.

WHY THIS IS CHEAP HERE, AND WHY THAT IS THE ARCHITECTURE SHOWING. E11 had to
run the LM once per voice per line -- N voices cost N x L generations. Here the
towers are separable, so the content tokens are generated ONCE and decoded
through every speaker vector:

    L LM generations + (N x L) codec decodes,  not  (N x L) LM generations

N voices cost the same LM time as one. That is not an optimisation trick; it
is the two-tower split being real, and it is the same property that made S5's
identity test valid (same tokens, different identity). It also makes the
comparison BETWEEN voices exact: every voice says the identical utterance, so
a consistency or drift difference cannot be a content difference.

WHAT IS MEASURED per voice
    uniqueness   working-space distance to the nearest ACCEPTED voice
    drift        intended vector vs the vector re-extracted from the render
    consistency  same identity across different sentences (ECAPA, independent)
    accepted     did it clear all three floors

and over the catalog as it grows
    rolling acceptance rate      <- the saturation curve
    nearest-neighbour distances
    normalised Vendi             <- effective voices / minted voices

    envs/qwen3/Scripts/python.exe experiments/S7-indic-catalog/run_indic_catalog.py --n 80
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
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.acoustics import Attributes, Binner
from alaap.captions import caption_from_bins
from alaap.data import norm_gender
from alaap.catalog import sample_cells, saturation_curve
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder
from alaap.service import (CONSISTENCY_FLOOR, DRIFT_FLOOR,
                           UNIQUENESS_MIN_MIOCODEC as UNIQUENESS_MIN)
from alaap.metrics import nn_distances, vendi_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
S6_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "S6-indic-mint", "out")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=80)
ap.add_argument("--corpus", default="indicvoices_r_hi",
                help="comma-separated; multiple corpora are POOLED into one "
                     "described space, which is what widening it means")
ap.add_argument("--clips", type=int, default=250)
ap.add_argument("--per-speaker", type=int, default=2)
ap.add_argument("--codec", default="Aratako/MioCodec-25Hz-44.1kHz-v2")
ap.add_argument("--lm", default="SPRINGLab/Indic-Mio")
ap.add_argument("--retrieval", default="hybrid", choices=["text", "hybrid"])
ap.add_argument("--novelty", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
# S9b: these two ARE the diversity knobs. Truncating the basis makes every
# minted voice identical on the discarded components; blending k anchors
# lands nearer the centroid the larger k is. Defaults are the measured best.
ap.add_argument("--pca-dims", type=int, default=0,
                help="0 = the full space, which is what S9b measured as best")
ap.add_argument("--top-k", type=int, default=2,
                help="anchors blended per mint; 1 is pure retrieval")
# S11 put this floor in front of a listener: pairs at d=0.323 were heard
# as the same person half the time, everything at d>=0.506 correctly.
# 0.45 costs 10 nominal voices of 50 and 3% of effective diversity --
# the ten were duplicates.
ap.add_argument("--uniqueness", type=float, default=UNIQUENESS_MIN)
ap.add_argument("--keep-audio", type=int, default=8,
                help="write wavs for the first K accepted voices only; "
                     "N x L 44.1 kHz wavs is a lot of disk for no extra evidence")
args = ap.parse_args()

CODEC_SR = 44100
SPEECH_OFFSET = 151669
CODEBOOK = 12800
CORPORA = [c.strip() for c in args.corpus.split(",") if c.strip()]
TAG = "+".join(c.replace("indicvoices_r_", "") for c in CORPORA)

# The axis set is part of the run's identity, so it is part of the filename.
# S17/S18's swap changed acceptance, adherence AND the describable space; a
# re-run that overwrote `catalog_E_hi.npz` would silently invalidate the numbers
# this experiment's own RESULTS.md quotes. Same lesson as a cache key needing the
# model id, which this project has learned twice (E0, S2).
from alaap.catalog import CATALOG_AXES, CATALOG_AXES_V1
AXTAG = "v1-5axis" if list(CATALOG_AXES) == CATALOG_AXES_V1 else         "id-" + "+".join(a.replace("_mean", "").replace("_cm", "")
                         .replace("spectral_", "") for a in CATALOG_AXES)
TAG = f"{TAG}_{AXTAG}"


def _s4(c):
    return (f"experiments/S4-indic/out/"
            f"measured_{c}_{args.clips}_{args.per_speaker}.npz")


def _emb(c):
    return os.path.join(S6_OUT, f"mio_emb_{c}_{args.clips}_{args.per_speaker}.npz")


SCRIPT_HI = [
    "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।",
    "मुझे यह फिल्म बहुत पसंद आई!",
    "हम कल सुबह जल्दी निकलेंगे।",
]

import librosa
import soundfile as sf

# --------------------------------------------------- 1. rebuild S6's mapper
print(f"[1/5] building the described space from {len(CORPORA)} corpus/corpora: {TAG}")
attrs, metas, Z = [], [], []
for c in CORPORA:
    if not os.path.exists(_s4(c)):
        sys.exit(f"missing {_s4(c)} -- run experiments/S4-indic/run_indic_captions.py")
    if not os.path.exists(_emb(c)):
        sys.exit(f"missing {_emb(c)} -- run:  experiments/S6-indic-mint/run_indic_mint.py "
                 f"--corpus {c} --skip-render   (builds the MioCodec embedding cache)")
    d4 = np.load(_s4(c), allow_pickle=True)
    aa = [Attributes.from_dict(a) for a in json.loads(str(d4["attrs"]))]
    mm = json.loads(str(d4["metas"]))
    dz = np.load(_emb(c), allow_pickle=True)
    zz, f0f = dz["Z"].astype(np.float64), dz["f0"].astype(np.float64)
    k = min(len(zz), len(aa))
    # The alignment check, PER CORPUS. E14b caught this assumption being false
    # once at r = 0.011; pooling three corpora is three chances to hit it.
    r = float(np.corrcoef(f0f[:k], np.array([a.f0_mean for a in aa[:k]]))[0, 1])
    print(f"      {c:<22} {k:>4} clips | f0 alignment r = {r:.4f}")
    if r < 0.99:
        sys.exit(f"ABORT: {c}'s embedding cache is not aligned with its S4 measurements.")
    # speaker ids are only unique WITHIN a corpus
    for m in mm[:k]:
        m["speaker_id"] = f"{c}:{m['speaker_id']}"
    attrs += aa[:k]
    metas += mm[:k]
    Z.append(zz[:k])
Z = np.vstack(Z)
n = len(attrs)
print(f"      pooled: {n} clips, {len({m['speaker_id'] for m in metas})} speakers")

# One binner over the POOLED attributes. Percentile bins computed per corpus
# would mean "high-pitched" denotes a different pitch in each language, and a
# description could not address the pooled space at all.
binner = Binner.fit(attrs)
gv = {"Female": [], "Male": []}
for a, m in zip(attrs, metas):
    g = norm_gender(m.get("gender"))
    if g in gv and np.isfinite(a.vtl_cm):
        gv[g].append(a.vtl_cm)
if len(gv["Female"]) >= 20 and len(gv["Male"]) >= 20:
    F, M = np.array(gv["Female"]), np.array(gv["Male"])
    dcoh = (M.mean() - F.mean()) / max(np.sqrt((F.var(ddof=1) + M.var(ddof=1)) / 2), 1e-9)
    if dcoh < 0.30:
        binner.edges.pop("vtl_cm", None)
        print(f"      vtl_cm dropped (gender d = {dcoh:+.2f})")
bins = [binner.bin_one(a) for a in attrs]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]

space = SpeakerSpace.fit(Z, n_components=min(64, n - 1))
PCA_DIMS = args.pca_dims or space.components.shape[0]
mapper = RetrievalMapper(space, TextEncoder(), pca_dims=PCA_DIMS,
                         retrieval=args.retrieval).fit(
    caps, Z, anchor_bins=bins if args.retrieval == "hybrid" else None)
print(f"      {space}")
print(f"      pca_dims {PCA_DIMS} | top_k {args.top_k}")

# --------------------------------------------------------- 2. descriptions
cells = sample_cells(args.n, args.seed)
descs = [(f"ind{i:04d}", caption_from_bins(c, seed=i), c) for i, c in enumerate(cells)]
print(f"[2/5] {len(descs)} descriptions over a stratified cover of bin space")
print(f"      floors: uniqueness {args.uniqueness} drift {DRIFT_FLOOR} "
      f"consistency {CONSISTENCY_FLOOR} -- rejections ARE the result")

# ------------------------------------------- 3. content tokens, generated ONCE
print(f"[3/5] loading {args.lm} + codec")
from miocodec import MioCodecModel
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda" if torch.cuda.is_available() else "cpu"
codec = MioCodecModel.from_pretrained(args.codec).to(dev).eval()
tok = AutoTokenizer.from_pretrained(args.lm, trust_remote_code=True)
lm = AutoModelForCausalLM.from_pretrained(args.lm, torch_dtype=torch.bfloat16,
                                          device_map=dev, trust_remote_code=True).eval()


def gen_codes(text, seed):
    torch.manual_seed(seed)
    p = tok.apply_chat_template([{"role": "user", "content": text}],
                                tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        o = lm.generate(**inp, max_new_tokens=1024, do_sample=True,
                        temperature=0.9, top_p=0.9, pad_token_id=tok.eos_token_id)
    new = o[0][inp["input_ids"].shape[1]:]
    return [t.item() - SPEECH_OFFSET for t in new
            if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + CODEBOOK]


codes_per_line = [gen_codes(t, 100 + i) for i, t in enumerate(SCRIPT_HI)]
codes_per_line = [c for c in codes_per_line if c]
print(f"      {len(codes_per_line)} token streams: " +
      ", ".join(str(len(c)) for c in codes_per_line) + " tokens")
if len(codes_per_line) < 2:
    sys.exit("ABORT: fewer than 2 usable token streams -- consistency needs two "
             "different sentences and would otherwise be measured against itself.")
del lm
torch.cuda.empty_cache()

# ------------------------------------------------------- 4. mint and audit
from alaap.encoder import IndependentSV
sv = IndependentSV()


def render(codes, vec):
    ct = torch.as_tensor(codes, dtype=torch.long, device=dev).reshape(-1)
    ge = torch.as_tensor(np.asarray(vec, np.float32), device=dev).reshape(-1)
    with torch.no_grad():
        return codec.decode(global_embedding=ge,
                            content_token_indices=ct).squeeze().float().cpu().numpy()


def enc_g(w):
    t = torch.as_tensor(np.asarray(w, np.float32)).reshape(1, -1).to(dev)
    with torch.no_grad():
        return codec.encode(t, return_content=False,
                            return_global=True).global_embedding.reshape(-1).float().cpu().numpy()


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


print(f"[4/5] minting and auditing {len(descs)} voices")
rows, accepted_E, t0 = [], [], time.time()
for i, (cid, desc, cell) in enumerate(descs):
    m = mapper.mint(desc, novelty=args.novelty, seed=i, top_k=args.top_k)
    e = space.encode(m.vector)[0]

    # uniqueness against ACCEPTED voices only -- a rejected voice is not in the
    # catalog, so it cannot be the thing a later voice collides with.
    if accepted_E:
        P = np.vstack(accepted_E)
        uniq = float(np.min(1.0 - (P @ e) / np.maximum(
            np.linalg.norm(P, axis=1) * np.linalg.norm(e), 1e-12)))
    else:
        uniq = 1.0

    drifts, embs = [], []
    for li, codes in enumerate(codes_per_line):
        w = render(codes, m.vector)
        drifts.append(cos(space.encode(enc_g(w))[0], e))
        embs.append(sv.embed(librosa.resample(w, orig_sr=CODEC_SR, target_sr=16000),
                             sr=16000))
        if len([r for r in rows if r["accepted"]]) < args.keep_audio:
            sf.write(os.path.join(OUT, f"{TAG}_{cid}_line{li}.wav"), w, CODEC_SR)
    drift = float(np.mean(drifts))
    cons = float(np.mean([cos(embs[a], embs[b])
                          for a in range(len(embs)) for b in range(a + 1, len(embs))]))

    ok = (uniq >= args.uniqueness and drift >= DRIFT_FLOOR
          and cons >= CONSISTENCY_FLOOR)
    why = ("" if ok else
           ",".join(x for x, bad in [("uniqueness", uniq < args.uniqueness),
                                     ("drift", drift < DRIFT_FLOOR),
                                     ("consistency", cons < CONSISTENCY_FLOOR)] if bad))
    if ok:
        accepted_E.append(e)
    rows.append({"character_id": cid, "description": desc, "cell": cell,
                 "uniqueness": uniq, "drift": drift, "consistency": cons,
                 "accepted": ok, "rejected_for": why,
                 "anchor_score": m.anchor_score, "score_kind": m.score_kind,
                 "e": e.tolist()})
    if (i + 1) % 10 == 0 or i == len(descs) - 1:
        el = time.time() - t0
        na = len(accepted_E)
        print(f"      {i+1}/{len(descs)} | accepted {na} ({na/(i+1):.0%}) | "
              f"{el/60:.1f} min | eta {el/(i+1)*(len(descs)-i-1)/60:.1f} min",
              flush=True)

json.dump([{k: v for k, v in r.items() if k != "e"} for r in rows],
          io.open(os.path.join(OUT, f"rows_{TAG}.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
np.savez_compressed(os.path.join(OUT, f"catalog_E_{TAG}.npz"),
                    E=np.vstack([r["e"] for r in rows]),
                    accepted=np.array([r["accepted"] for r in rows]))

# ------------------------------------------------------------- 5. report
acc = [r["accepted"] for r in rows]
curve = saturation_curve(acc, window=20)
A = np.vstack([r["e"] for r in rows if r["accepted"]])
nn = nn_distances(A) if len(A) > 1 else np.array([np.nan])
vendi = vendi_score(A) if len(A) > 1 else float("nan")

print()
print("=" * 78)
print(f"S7 — Indic catalog saturation ({len(rows)} minted, corpus {TAG})")
print("=" * 78)
print(f"  accepted            {sum(acc)}/{len(acc)}  ({sum(acc)/len(acc):.1%})")
reasons = {}
for r in rows:
    if not r["accepted"]:
        reasons[r["rejected_for"]] = reasons.get(r["rejected_for"], 0) + 1
for k, v in sorted(reasons.items(), key=lambda t: -t[1]):
    print(f"    rejected for {k:<28} {v}")
print()
print(f"  rolling acceptance (window 20), first -> last:")
step = max(1, len(curve) // 8)
print("    " + "  ".join(f"{curve[i]:.0%}" for i in range(0, len(curve), step)))
print()
print(f"  nn distance among accepted: median {np.median(nn):.3f} "
      f"min {np.min(nn):.3f} max {np.max(nn):.3f}")
print(f"  normalised Vendi          : {vendi:.3f}  "
      f"-> ~{vendi*len(A):.0f} effective voices from {len(A)} accepted")
print(f"  drift        mean {np.mean([r['drift'] for r in rows]):.3f} "
      f"(floor {DRIFT_FLOOR})")
print(f"  consistency  mean {np.mean([r['consistency'] for r in rows]):.3f} "
      f"(floor {CONSISTENCY_FLOOR})")
print()
print(f"  audio (first {args.keep_audio} accepted) + rows.json -> {OUT}")
print("=" * 78)
