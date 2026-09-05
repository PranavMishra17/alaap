"""
S11 — do the numbers correspond to anything a person hears?

Everything in S7-S10 is geometry. "22 effective voices" is a Vendi artefact,
and the `uniqueness` floor of 0.30 is VoicePrivacy B3's threshold borrowed and
applied in a working space nobody has validated it for. If two voices 0.30
apart are the same person to a listener, then the floor is too low, the
catalog is an overcount, and every diversity number in this project is
measuring something that does not exist.

That is one question and it needs a human. This builds the set to ask it with.

DESIGN, and why each part is not optional.

BLIND. Pairs are shuffled, A/B order within each pair is shuffled, and file
names carry no information. The answer key is written to a separate file.

CONTROLS. Two pair types have known answers:

    positive  the SAME minted vector, two different sentences  -> same person
    negative  two different REAL corpus speakers               -> different

If the listener fails the controls, the instrument is broken and no test-pair
answer means anything. This is the same discipline S8 applied with its random
retrieval baseline: a score is not readable until a control is sitting next
to it.

EVERY PAIR IS TWO DIFFERENT SENTENCES, controls included. Otherwise the
positive control is literally the same waveform twice and is identifiable by
structure rather than by voice -- and a listener comparing identical sentences
is doing an easier task than the product ever asks of them.

THE DISTANCES ARE THE POINT. Test pairs are selected to sit at working-space
distances that bracket the floor:

    ~0.30-0.35   just above the uniqueness floor -- the catalog says these are
                 different voices, and this is the claim being tested
    ~0.45-0.55   comfortably accepted
    ~0.65+       far apart, should be obvious

If the near-floor pairs come back "same person" and the far pairs come back
"different", the floor needs raising and the number that changes is the
catalog size, not the method.

    envs/qwen3/Scripts/python.exe experiments/S11-listening/run_listening_set.py
"""
import argparse
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from alaap.acoustics import Attributes, Binner
from alaap.captions import caption_from_bins
from alaap.catalog import sample_cells
from alaap.geometry import SpeakerSpace
from alaap.mapper import RetrievalMapper, TextEncoder

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
S6_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "S6-indic-mint", "out")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="indicvoices_r_hi")
ap.add_argument("--clips", type=int, default=250)
ap.add_argument("--per-speaker", type=int, default=2)
ap.add_argument("--pool", type=int, default=60, help="voices to mint and choose pairs from")
ap.add_argument("--codec", default="Aratako/MioCodec-25Hz-44.1kHz-v2")
ap.add_argument("--lm", default="SPRINGLab/Indic-Mio")
ap.add_argument("--top-k", type=int, default=2)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

CODEC_SR = 44100
SPEECH_OFFSET = 151669
CODEBOOK = 12800
UNIQ_FLOOR = 0.30

# Four sentences so a pair can always be given two DIFFERENT ones.
SCRIPT = [
    "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।",
    "मुझे यह फिल्म बहुत पसंद आई!",
    "हम कल सुबह जल्दी निकलेंगे।",
    "यह रास्ता सीधा बाज़ार तक जाता है।",
]

import librosa
import soundfile as sf

# ------------------------------------------------------------- 1. the mapper
print(f"[1/4] rebuilding the mapper ({args.corpus})")
S4 = f"experiments/S4-indic/out/measured_{args.corpus}_{args.clips}_{args.per_speaker}.npz"
EMB = os.path.join(S6_OUT, f"mio_emb_{args.corpus}_{args.clips}_{args.per_speaker}.npz")
for f in (S4, EMB):
    if not os.path.exists(f):
        sys.exit(f"missing {f}")
d4 = np.load(S4, allow_pickle=True)
attrs = [Attributes.from_dict(a) for a in json.loads(str(d4["attrs"]))]
metas = json.loads(str(d4["metas"]))
dz = np.load(EMB, allow_pickle=True)
Z, f0f = dz["Z"].astype(np.float64), dz["f0"].astype(np.float64)
n = min(len(Z), len(attrs))
Z, attrs, metas, f0f = Z[:n], attrs[:n], metas[:n], f0f[:n]
r = float(np.corrcoef(f0f, np.array([a.f0_mean for a in attrs]))[0, 1])
if r < 0.99:
    sys.exit(f"ABORT: embedding cache not aligned with S4 measurements (r={r:.4f})")

binner = Binner.fit(attrs)
binner.edges.pop("vtl_cm", None)
bins = [binner.bin_one(a) for a in attrs]
caps = [caption_from_bins(b, seed=i) for i, b in enumerate(bins)]
space = SpeakerSpace.fit(Z, n_components=min(64, n - 1))
mapper = RetrievalMapper(space, TextEncoder(), pca_dims=space.components.shape[0],
                         retrieval="hybrid").fit(caps, Z, anchor_bins=bins)
E_corpus = space.encode(Z)

# ------------------------------------------------------------ 2. the pairs
print(f"[2/4] minting {args.pool} voices and choosing pairs by distance")
cells = sample_cells(args.pool, 0)
pool = []
for i, c in enumerate(cells):
    m = mapper.mint(caption_from_bins(c, seed=i), novelty=0.0, seed=i, top_k=args.top_k)
    pool.append({"v": m.vector, "e": space.encode(m.vector)[0]})
P = np.vstack([p["e"] for p in pool])
Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
D = 1.0 - Pn @ Pn.T
np.fill_diagonal(D, np.inf)


def pick_pair(lo, hi, used):
    """Closest pair to the middle of [lo, hi] that reuses no voice."""
    best, bd = None, np.inf
    mid = (lo + hi) / 2
    for a in range(len(pool)):
        for b in range(a + 1, len(pool)):
            if a in used or b in used or not (lo <= D[a, b] <= hi):
                continue
            if abs(D[a, b] - mid) < bd:
                best, bd = (a, b), abs(D[a, b] - mid)
    return best


rng = np.random.default_rng(args.seed)
used, pairs = set(), []

# two REAL speakers, as far apart as two real people ordinarily are
spk_first = {}
for i, m in enumerate(metas):
    spk_first.setdefault(m["speaker_id"], i)
sidx = list(spk_first.values())
En = E_corpus / np.maximum(np.linalg.norm(E_corpus, axis=1, keepdims=True), 1e-12)
for _ in range(2):
    a, b = rng.choice(len(sidx), 2, replace=False)
    ia, ib = sidx[int(a)], sidx[int(b)]
    pairs.append({"kind": "negative control (two real speakers)",
                  "truth": "different", "dist": float(1 - En[ia] @ En[ib]),
                  "va": Z[ia], "vb": Z[ib]})

# the SAME voice twice -- different sentences, so it is not the same waveform
for k in (0, 1):
    pairs.append({"kind": "positive control (one voice, two sentences)",
                  "truth": "same", "dist": 0.0,
                  "va": pool[k]["v"], "vb": pool[k]["v"]})
    used.add(k)

for lo, hi, label in [(UNIQ_FLOOR, 0.35, "just above the uniqueness floor"),
                      (UNIQ_FLOOR, 0.35, "just above the uniqueness floor"),
                      (0.45, 0.55, "comfortably accepted"),
                      (0.62, 0.90, "far apart")]:
    pr = pick_pair(lo, hi, used)
    if pr is None:
        print(f"      no pair available in [{lo}, {hi}] -- skipping '{label}'")
        continue
    a, b = pr
    used.update((a, b))
    pairs.append({"kind": f"test — {label}", "truth": "different (per the metric)",
                  "dist": float(D[a, b]), "va": pool[a]["v"], "vb": pool[b]["v"]})

order = rng.permutation(len(pairs))
print(f"      {len(pairs)} pairs: " +
      ", ".join(f"{p['dist']:.2f}" for p in pairs))

# ----------------------------------------------------------- 3. synthesise
print(f"[3/4] loading {args.lm} + codec")
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
        o = lm.generate(**inp, max_new_tokens=1024, do_sample=True, temperature=0.9,
                        top_p=0.9, pad_token_id=tok.eos_token_id)
    new = o[0][inp["input_ids"].shape[1]:]
    return [t.item() - SPEECH_OFFSET for t in new
            if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + CODEBOOK]


codes = [gen_codes(t, 200 + i) for i, t in enumerate(SCRIPT)]
codes = [c for c in codes if c]
if len(codes) < 2:
    sys.exit("ABORT: need two usable token streams so a pair can use two sentences")
print(f"      {len(codes)} sentences: " + ", ".join(str(len(c)) for c in codes) + " tokens")


def render(cc, vec):
    ct = torch.as_tensor(cc, dtype=torch.long, device=dev).reshape(-1)
    ge = torch.as_tensor(np.asarray(vec, np.float32), device=dev).reshape(-1)
    with torch.no_grad():
        return codec.decode(global_embedding=ge,
                            content_token_indices=ct).squeeze().float().cpu().numpy()


print("[4/4] rendering")
key = []
for slot, pi in enumerate(order, 1):
    p = pairs[int(pi)]
    # two different sentences, and a coin flip for which vector is A
    s1, s2 = rng.choice(len(codes), 2, replace=False)
    va, vb = (p["va"], p["vb"]) if rng.random() < 0.5 else (p["vb"], p["va"])
    fa = os.path.join(OUT, f"pair{slot:02d}_A.wav")
    fb = os.path.join(OUT, f"pair{slot:02d}_B.wav")
    sf.write(fa, render(codes[int(s1)], va), CODEC_SR)
    sf.write(fb, render(codes[int(s2)], vb), CODEC_SR)
    key.append({"pair": slot, "kind": p["kind"], "truth": p["truth"],
                "working_distance": round(p["dist"], 4)})
    print(f"      pair{slot:02d}  d={p['dist']:.3f}  {p['kind']}")

json.dump(key, io.open(os.path.join(OUT, "ANSWER-KEY.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

sheet = ["# S11 — listening set", "",
         "Listen to each pair. One question only:", "",
         "> **Are these two clips the same person, or two different people?**", "",
         "Not *do they sound different* — two recordings of one person always sound",
         "different. The question is whether they are the same **person**.",
         "",
         "Every pair is two different Hindi sentences, so you cannot compare waveforms;",
         "you have to judge the voice. Some pairs have known answers and are there to",
         "check the test itself.", "",
         "| pair | same person? | confident? |",
         "|---|---|---|"]
sheet += [f"| {i:02d} | | |" for i in range(1, len(key) + 1)]
sheet += ["", f"Answer key is in `out/ANSWER-KEY.json` — **do not open it first.**",
          "", "Audio: `experiments/S11-listening/out/pairNN_A.wav` and `_B.wav`."]
io.open(os.path.join(OUT, "SCORESHEET.md"), "w", encoding="utf-8").write("\n".join(sheet))
print(f"\n  {len(key)} pairs -> {OUT}")
print(f"  scoresheet: experiments/S11-listening/out/SCORESHEET.md")
print(f"  answer key: experiments/S11-listening/out/ANSWER-KEY.json (do not open first)")
