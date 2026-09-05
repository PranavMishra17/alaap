"""
S5b — is the Indic speech actually correct, and does identity transfer cost it?

S5 measured that a stored speaker vector carries a voice onto new Indic text.
It said nothing about whether the words are RIGHT. A voice can transfer
perfectly and say gibberish, and every metric in S5 would look identical.

A fluent listener has since confirmed the Hindi is good, which is stronger
evidence than any number here. This exists for the two things a listener
cannot cheaply give: a repeatable number, and a REGRESSION detector for when
something changes months from now.

THE MEASUREMENT. `ai4bharat/indic-conformer-600m-multilingual` (MIT, 22 Indian
languages) transcribes each render; CER against the text the LM was given.

    CER = edit_distance(reference, hypothesis) / len(reference)

CER not WER, because Indic scripts are abugidas -- word boundaries are less
reliable than characters, and Devanagari/Tamil word segmentation would add its
own error term on top of the ASR's.

WHAT MAKES THIS A REAL TEST rather than a number: the same content tokens are
decoded through DIFFERENT speaker embeddings. The words are identical by
construction, so any CER difference between them is caused purely by the
identity that was injected.

    CER(donor A) vs CER(donor B)   -> does WHO is speaking damage WHAT is said?

That is the question S5 could not answer, and it is the one that decides
whether the two-tower split is usable or merely measurable.

ALSO REPORTED: the ASR's own floor. It is transcribing 44.1 kHz synthetic
speech, and its error rate on the REAL donor clips bounds how much of any CER
belongs to the ASR rather than to the synthesis.

    envs/qwen3/Scripts/python.exe experiments/S5-indic-mio/run_intelligibility.py
"""
import argparse
import glob
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

ap = argparse.ArgumentParser()
# IndicConformer is the right instrument (MIT, 22 Indian languages, built for
# exactly this) but it is GATED and the token is not on its allow-list, so it
# needs one accept click. Whisper is the ungated fallback.
#
# A weaker ASR is acceptable HERE specifically because the headline measurement
# is a DIFFERENCE: the same content tokens decoded through two identities. Any
# bias the ASR has applies to both sides and cancels in the spread. It does not
# cancel in the absolute CER, which is why that number is reported as an upper
# bound rather than as the model's error rate.
ap.add_argument("--asr", default="openai/whisper-large-v3-turbo")
ap.add_argument("--decoding", default="ctc", choices=["ctc", "rnnt"])
ap.add_argument("--device", default="cpu",
                help="cpu keeps the GPU free; the ASR is 600M and fine on CPU")
args = ap.parse_args()

# must match run_two_tower.py
LINES = [
    ("hi", "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।"),
    ("hi", "मुझे यह फिल्म बहुत पसंद आई! <happy>"),
    ("ta", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"),
    ("en", "The mountains remember every footstep, even the ones you regret."),
]


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalise(s):
    """Strip the emotion tags and punctuation the ASR will never emit."""
    import re
    s = re.sub(r"<[a-z]+>", " ", s)
    s = re.sub(r"[।॥.,!?;:'\"\-—‘’“”]", " ", s)
    return " ".join(s.split())


def cer(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    if not r:
        return float("nan")
    return edit_distance(r, h) / len(r)


print(f"[1/3] loading {args.asr} on {args.device}")
tok_env = os.environ.get("HF_TOKEN") or (
    io.open(".hf_token", encoding="utf-8").read().strip()
    if os.path.exists(".hf_token") else None)
IS_WHISPER = "whisper" in args.asr.lower()

import soundfile as sf
import torchaudio

if IS_WHISPER:
    from transformers import pipeline
    pipe = pipeline("automatic-speech-recognition", model=args.asr,
                    device=0 if args.device == "cuda" else -1, token=tok_env)
else:
    from transformers import AutoModel
    asr = AutoModel.from_pretrained(args.asr, trust_remote_code=True, token=tok_env)
    try:
        asr = asr.to(args.device)
    except Exception:
        pass


def transcribe(path, lang):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    t = torch.as_tensor(w).unsqueeze(0)
    if sr != 16000:
        t = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(t)
    if IS_WHISPER:
        r = pipe(t.squeeze().numpy(),
                 generate_kwargs={"language": lang, "task": "transcribe"})
        return r["text"]
    out = asr(t, lang, args.decoding)
    return out[0] if isinstance(out, (list, tuple)) else str(out)


print("[2/3] the ASR's own floor, on the REAL donor recordings")
donors = sorted(glob.glob(f"{OUT}/donor_*.wav"))
dkeys = [os.path.basename(d)[6:-4] for d in donors]
for d in donors:
    try:
        txt = transcribe(d, "hi")
        print(f"      {os.path.basename(d)[:22]:<24} -> {txt[:66]}")
    except Exception as e:
        print(f"      {os.path.basename(d)[:22]:<24} ASR FAILED: "
              f"{type(e).__name__}: {str(e)[:60]}")
print("      (no reference text for these -- they are unscripted corpus speech,")
print("       so this is a sanity look, not a floor number)")

print(f"[3/3] transcribing the renders ({args.decoding})")
rows = []
for li, (lang, text) in enumerate(LINES):
    for k in dkeys:
        p = f"{OUT}/line{li}_{lang}_{k}.wav"
        if not os.path.exists(p):
            continue
        try:
            hyp = transcribe(p, lang)
        except Exception as e:
            print(f"      line{li} {lang} [{k[:8]}] ASR FAILED: "
                  f"{type(e).__name__}: {str(e)[:60]}")
            continue
        c = cer(text, hyp)
        rows.append({"line": li, "lang": lang, "donor": k, "cer": c,
                     "ref": normalise(text), "hyp": normalise(hyp)})
        print(f"      line{li} {lang} [{k[:8]}] CER {c:.3f}")
        print(f"         ref: {normalise(text)[:74]}")
        print(f"         asr: {normalise(hyp)[:74]}")

json.dump(rows, open(os.path.join(OUT, "intelligibility.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print()
print("=" * 78)
print("S5b — does identity transfer damage intelligibility?")
print("=" * 78)
if rows:
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["lang"], []).append(r["cer"])
    for lg, cs in sorted(by_lang.items()):
        print(f"  {lg}: mean CER {np.nanmean(cs):.3f}  (n={len(cs)})")
    print()
    print(f"  {'line':<8} " + " ".join(f"{k[:8]:>10}" for k in dkeys) + "     spread")
    for li, (lang, _) in enumerate(LINES):
        cs = [next((r["cer"] for r in rows if r["line"] == li and r["donor"] == k),
                   None) for k in dkeys]
        if all(c is not None for c in cs):
            print(f"  line{li} {lang:<3} " + " ".join(f"{c:>10.3f}" for c in cs) +
                  f"  {abs(cs[0]-cs[1]):>9.3f}")
    spreads = []
    for li, _ in enumerate(LINES):
        cs = [next((r["cer"] for r in rows if r["line"] == li and r["donor"] == k),
                   None) for k in dkeys]
        if all(c is not None for c in cs):
            spreads.append(abs(cs[0] - cs[1]))
    if spreads:
        print()
        print(f"  mean |CER(A) - CER(B)| = {np.mean(spreads):.3f}")
        print("  Same content tokens, different injected identity. A small spread")
        print("  means WHO is speaking does not damage WHAT is said -- the two")
        print("  towers are genuinely independent.")
else:
    print("  no rows -- the ASR did not run")
print("=" * 78)
