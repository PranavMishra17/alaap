"""
S6 — mint an Indic voice from a DESCRIPTION, not from a reference clip.

S5 showed a stored speaker vector carries a voice onto new Indic text. But it
carried an EXISTING speaker's vector, which makes it a voice cloner. The
product claim is different and stronger:

    "a warm, low-pitched Hindi voice, speaking slowly"  -->  a voice that
    has never existed, rendering arbitrary Hindi dialogue

This is the English pipeline (S2 + E11 + E15) ported onto MioCodec's 128-d
space. Every piece already exists and is validated separately:

    S4   measure Indic audio -> percentile bins -> captions      (3 languages,
         cross-checked against an independent toolchain, 100% round-trip)
    S5   content tokens + an arbitrary 128-d vector -> Indic audio
    E15  retrieve on weighted bins rather than sentence embeddings

What is new here is the join: (caption, MioCodec embedding) pairs, which
nothing has built before, because no Indic corpus ships voice descriptions
and MioCodec embeddings had never been measured against captions.

THE ALIGNMENT PROBLEM, and how it is checked. S4's cached attributes came from
`stream_clips` with specific bounds. To pair them with MioCodec embeddings the
SAME clips must be re-streamed in the SAME order. `stream_clips` is
deterministic, so that holds -- but E14b already caught this exact assumption
being false once, when the duration bounds differed and the correlation between
fresh and cached f0 was 0.011 instead of 1.0. So it is VERIFIED, not assumed,
and the run aborts if the check fails.

WHAT IS MEASURED
    uniqueness   working-space distance to already-minted voices
    drift        intended vector vs the vector re-extracted from the render
    consistency  same identity across different sentences (ECAPA, independent)

the same three floors the English path audits against, so the numbers are
directly comparable to E11's.

    envs/qwen3/Scripts/python.exe experiments/S6-indic-mint/run_indic_mint.py
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
from alaap.data import norm_gender, stream_clips
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder
from alaap.metrics import nn_distances

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--lang", default="hi")
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--per-speaker", type=int, default=2)
# CODEC CHOICE IS LOAD-BEARING -- see S5's RESULTS.md.
# `MioCodec-25Hz-44.1kHz` (legacy) has a content-token space UNRELATED to the
# one Indic-Mio was trained on: 0.0000% index agreement against the 24 kHz
# model on identical audio, where chance is 0.0078%. Both are FSQ with 12800
# entries, so every index is accepted and NOTHING RAISES -- the output is
# fluent-sounding speech saying different words.
# `-v2` shares the 24 kHz model's tokenizer bit-for-bit and is natively
# 44.1 kHz. Use MioCodecModel (integrated iSTFT head, no external vocoder);
# MioCodec is the loader for the legacy external-vocoder variant only.
ap.add_argument("--codec", default="Aratako/MioCodec-25Hz-44.1kHz-v2")
ap.add_argument("--lm", default="SPRINGLab/Indic-Mio")
ap.add_argument("--retrieval", default="hybrid", choices=["text", "hybrid"])
ap.add_argument("--novelty", type=float, default=0.0)
ap.add_argument("--skip-render", action="store_true")
args = ap.parse_args()

CODEC_SR = 44100
SPEECH_OFFSET = 151669
CODEBOOK = 12800
S4_CACHE = (f"experiments/S4-indic/out/"
            f"measured_{args.corpus}_{args.n}_{args.per_speaker}.npz")
EMB_CACHE = os.path.join(OUT, f"mio_emb_{args.corpus}_{args.n}_{args.per_speaker}.npz")

# Neutral descriptions spanning the described space. Deliberately NOT the
# corpus's own captions -- minting from a caption the mapper was fitted on
# measures memory, not generalisation.
CAST = [
    ("guru",     "a very deep voice, very rough and gravelly, speaking very slowly"),
    ("student",  "a high voice, crisp-toned, speaking quickly and highly animated"),
    ("narrator", "a mid-range voice, warm-toned, at a steady pace"),
    ("elder",    "a low voice, slightly hoarse, with restrained intonation"),
]
SCRIPT_HI = [
    "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।",
    "मुझे यह फिल्म बहुत पसंद आई!",
    "हम कल सुबह जल्दी निकलेंगे।",
]

import librosa
import soundfile as sf


def to_codec_sr(w, src=24000):
    return librosa.resample(np.asarray(w, np.float32), orig_sr=src, target_sr=CODEC_SR)


# --------------------------------------------------- 1. captions from S4
print(f"[1/6] S4 measurements for {args.corpus}")
if not os.path.exists(S4_CACHE):
    sys.exit(f"missing {S4_CACHE} -- run experiments/S4-indic/run_indic_captions.py first")
d4 = np.load(S4_CACHE, allow_pickle=True)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d4["attrs"]))]
metas = json.loads(str(d4["metas"]))
print(f"      {len(attrs)} clips, {len({m['speaker_id'] for m in metas})} speakers")

binner = Binner.fit(attrs)
# S4 drops vtl_cm on these corpora: the formant estimate does not separate
# gender there, so it is noise. Mirror that decision rather than re-deriving it.
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
        print(f"      vtl_cm dropped (gender d = {dcoh:+.2f} < 0.30), as in S4")
bins = [binner.bin_one(a) for a in attrs]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
print(f"      captions over axes: {sorted(binner.edges)}")

# ------------------------------------ 2. MioCodec embeddings for those clips
print(f"[2/6] MioCodec embeddings for the SAME clips")
if os.path.exists(EMB_CACHE):
    dz = np.load(EMB_CACHE, allow_pickle=True)
    Z, f0_fresh = dz["Z"].astype(np.float64), dz["f0"].astype(np.float64)
    print(f"      cached {Z.shape}")
else:
    from miocodec import MioCodecModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = MioCodecModel.from_pretrained(args.codec).to(dev).eval()
    from alaap.acoustics import f0_track
    clips = stream_clips(args.corpus, n=args.n, per_speaker=args.per_speaker,
                         dev_only=True, min_dur=2.0, max_dur=15.0, progress_every=100)
    Zl, f0l, t0 = [], [], time.time()
    for i, c in enumerate(clips):
        t = torch.as_tensor(to_codec_sr(c.wav)).reshape(1, -1).to(dev)
        with torch.no_grad():
            f = codec.encode(t, return_content=False, return_global=True)
        Zl.append(f.global_embedding.reshape(-1).float().cpu().numpy())
        f0l.append(float(np.mean(f0_track(c.wav, 24000))))
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"      {i+1}/{len(clips)} | {el/60:.1f} min", flush=True)
    Z, f0_fresh = np.vstack(Zl), np.array(f0l)
    np.savez_compressed(EMB_CACHE, Z=Z.astype(np.float32), f0=f0_fresh)
    del codec
    torch.cuda.empty_cache()

# ------------------------------------------------- 3. verify the alignment
print("[3/6] verifying the embedded clips ARE the measured clips")
n = min(len(Z), len(attrs))
Z, attrs, caps, f0_fresh = Z[:n], attrs[:n], caps[:n], f0_fresh[:n]
f0_cached = np.array([a.f0_mean for a in attrs])
r = float(np.corrcoef(f0_fresh, f0_cached)[0, 1])
print(f"      f0 fresh vs S4-cached: r = {r:.4f}")
if r < 0.99:
    sys.exit("ABORT: re-streamed clips are NOT the S4 clips. Pairing captions "
             "with the wrong embeddings would produce a meaningless mapper. "
             "(E14b hit exactly this at r=0.011 when duration bounds differed.)")
print("      aligned.")

# --------------------------------------------------------- 4. the mapper
print(f"[4/6] fitting the mapper on {n} (caption, MioCodec vector) pairs")
space = SpeakerSpace.fit(Z, n_components=min(64, n - 1))
# full basis, not min(32, ...): S9b measured that truncating it zero-pads
# 14.2% of real speaker variance and costs ~4 effective voices in 80 mints.
mapper = RetrievalMapper(space, TextEncoder(),
                         pca_dims=space.components.shape[0],
                         retrieval=args.retrieval).fit(
    caps, Z, anchor_bins=bins[:n] if args.retrieval == "hybrid" else None)
print(f"      {space}")
if mapper.axis_weights is not None:
    print("      measured axis weights: " +
          ", ".join(f"{a.split('_')[0]} {w:.2f}"
                    for a, w in sorted(zip(mapper.axes, mapper.axis_weights),
                                       key=lambda t: -t[1])))

# ------------------------------------------------------------- 5. mint
print(f"[5/6] minting {len(CAST)} Indic voices from descriptions")
E_corpus = space.encode(Z)
minted = []
for name, desc in CAST:
    m = mapper.mint(desc, novelty=args.novelty, seed=0)
    e = space.encode(m.vector)[0]
    # uniqueness against everything already minted, as service.mint does
    if minted:
        prev = np.vstack([x["e"] for x in minted])
        u = float(np.min(1.0 - (prev @ e) /
                         np.maximum(np.linalg.norm(prev, axis=1) * np.linalg.norm(e), 1e-12)))
    else:
        u = 1.0
    # and how far it sits from any REAL speaker in the corpus
    dc = float(np.min(1.0 - (E_corpus @ e) /
                      np.maximum(np.linalg.norm(E_corpus, axis=1) * np.linalg.norm(e), 1e-12)))
    minted.append({"name": name, "desc": desc, "v": m.vector, "e": e,
                   "uniqueness": u, "to_nearest_real": dc,
                   "anchor_score": m.anchor_score,
                   "score_kind": m.score_kind})
    print(f"      {name:<9} uniqueness {u:.3f} | nearest real speaker {dc:.3f} "
          f"| anchor {m.score_kind} {m.anchor_score:.3f}")

nn = nn_distances(np.vstack([x["e"] for x in minted]))
print(f"      minted-to-minted nn: median {np.median(nn):.3f} min {nn.min():.3f}")

# Persist for S6b. It needs the minted vectors AND real corpus vectors from the
# same space, so it can decode identical content tokens through both and
# attribute any CER difference to minting rather than to the path.
# Two DISTINCT speakers. With --per-speaker 2 the first two clips are the same
# person, which would make the control a comparison of one speaker with herself.
_seen, DONOR_IDX = set(), []
for _i, _m in enumerate(metas[:n]):
    if _m["speaker_id"] not in _seen:
        _seen.add(_m["speaker_id"])
        DONOR_IDX.append(_i)
    if len(DONOR_IDX) == 2:
        break
np.savez_compressed(
    os.path.join(OUT, "minted.npz"),
    names=np.array([x["name"] for x in minted]),
    V=np.vstack([x["v"] for x in minted]).astype(np.float32),
    donor_V=Z[DONOR_IDX].astype(np.float32),
    donor_ids=np.array([metas[i]["speaker_id"] for i in DONOR_IDX]),
    script=np.array(SCRIPT_HI))
print(f"      minted vectors + {len(DONOR_IDX)} real donors -> out/minted.npz")

if args.skip_render:
    print("[6/6] --skip-render, stopping before synthesis")
    sys.exit(0)

# ------------------------------------------------------------ 6. render
print(f"[6/6] rendering {len(SCRIPT_HI)} Hindi lines per voice")
from miocodec import MioCodecModel
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda" if torch.cuda.is_available() else "cpu"
codec = MioCodecModel.from_pretrained(args.codec).to(dev).eval()
tok = AutoTokenizer.from_pretrained(args.lm, trust_remote_code=True)
lm = AutoModelForCausalLM.from_pretrained(args.lm, torch_dtype=torch.bfloat16,
                                          device_map=dev, trust_remote_code=True).eval()

from alaap.encoder import IndependentSV
sv = IndependentSV()


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


def render(codes, vec):
    ct = torch.as_tensor(codes, dtype=torch.long, device=dev).reshape(-1)
    ge = torch.as_tensor(np.asarray(vec, np.float32), device=dev).reshape(-1)
    with torch.no_grad():
        return codec.decode(global_embedding=ge, content_token_indices=ct
                            ).squeeze().float().cpu().numpy()


def enc_g(w44):
    t = torch.as_tensor(np.asarray(w44, np.float32)).reshape(1, -1).to(dev)
    with torch.no_grad():
        return codec.encode(t, return_content=False,
                            return_global=True).global_embedding.reshape(-1).float().cpu().numpy()


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


codes_per_line = [gen_codes(t, 100 + i) for i, t in enumerate(SCRIPT_HI)]
rows = []
for mi in minted:
    embs, drifts = [], []
    for li, codes in enumerate(codes_per_line):
        if not codes:
            continue
        w = render(codes, mi["v"])
        back = enc_g(w)
        drifts.append(cos(space.encode(back)[0], mi["e"]))
        w16 = librosa.resample(w, orig_sr=CODEC_SR, target_sr=16000)
        embs.append(sv.embed(w16, sr=16000))
        sf.write(os.path.join(OUT, f"{mi['name']}_line{li}.wav"), w, CODEC_SR)
    cons = [cos(embs[i], embs[j]) for i in range(len(embs)) for j in range(i + 1, len(embs))]
    rows.append({"name": mi["name"], "desc": mi["desc"],
                 "uniqueness": mi["uniqueness"], "to_nearest_real": mi["to_nearest_real"],
                 "drift": float(np.mean(drifts)) if drifts else None,
                 "consistency": float(np.mean(cons)) if cons else None,
                 "ecapa": [e.tolist() for e in embs]})
    print(f"      {mi['name']:<9} drift {rows[-1]['drift']:.3f} | "
          f"consistency {rows[-1]['consistency']:.3f}", flush=True)

# cross-voice ECAPA: are the minted voices different PEOPLE?
cross = []
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        a = np.mean(np.array(rows[i]["ecapa"]), 0)
        b = np.mean(np.array(rows[j]["ecapa"]), 0)
        cross.append(cos(a, b))

json.dump([{k: v for k, v in r.items() if k != "ecapa"} for r in rows],
          open(os.path.join(OUT, "results.json"), "w"), indent=2)

from alaap.metrics import CALIBRATION
c_same, c_diff, _ = CALIBRATION["ecapa"]
print()
print("=" * 78)
print("S6 — minting Indic voices from descriptions")
print("=" * 78)
print(f"  {'voice':<10} {'uniqueness':>11} {'drift':>8} {'consistency':>12}")
print(f"  {'-'*10} {'-'*11} {'-'*8} {'-'*12}")
for r in rows:
    print(f"  {r['name']:<10} {r['uniqueness']:>11.3f} {r['drift']:>8.3f} "
          f"{r['consistency']:>12.3f}")
print()
print(f"  ECAPA between DIFFERENT minted voices: mean {np.mean(cross):.4f}")
print(f"    normalised: {(np.mean(cross)-c_diff)/(c_same-c_diff):+.3f}   "
      f"(0 = different speakers, 1 = same speaker)")
print(f"  ECAPA within one voice (consistency): "
      f"mean {np.mean([r['consistency'] for r in rows]):.4f}")
print(f"  audio -> {OUT}")
print("=" * 78)
