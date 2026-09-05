"""Independent check on S5: does an OUTSIDE verifier agree the identities differ?

MioCodec's own global embedding saying "identity was carried" is circular -- it
is the same representation that was injected. ECAPA-TDNN never saw MioCodec and
is the project's standard independent verifier (E4 calibrated it: C_same 0.6988,
C_diff 0.2011 on real speech, EER 2.46% on LibriTTS-R).

Two questions:
  1. do the two rendered voices differ from EACH OTHER under ECAPA?
  2. does each rendered voice resemble the DONOR whose embedding it was given,
     more than the other donor?

(2) is the one that matters -- it is the difference between "the decoder makes
two different voices" and "the decoder makes the voice you asked for".
"""
import glob
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath("."))
from alaap.encoder import IndependentSV
from alaap.metrics import CALIBRATION

OUT = "experiments/S5-indic-mio/out"
SV_SR = 16000
sv = IndependentSV()


def emb(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SV_SR:
        w = librosa.resample(w, orig_sr=sr, target_sr=SV_SR)
    return sv.embed(w, sr=SV_SR)


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


donors = sorted(glob.glob(f"{OUT}/donor_*.wav"))
assert len(donors) == 2, donors
dkey = [os.path.basename(d)[6:-4] for d in donors]
demb = [emb(d) for d in donors]
print(f"  donors: {dkey}")
print(f"  donor A vs donor B (real speech, ECAPA): {cos(demb[0], demb[1]):+.4f}")
c_same, c_diff, _ = CALIBRATION["ecapa"]
print(f"  ECAPA calibration on real speech: same {c_same:.4f}  diff {c_diff:.4f}")
print()

lines = sorted({os.path.basename(f).rsplit("_", 1)[0]
                for f in glob.glob(f"{OUT}/line*.wav")})
print(f"  {'line':<14} {'to own donor':>13} {'to other':>10} {'margin':>8}  verdict")
print(f"  {'-'*14} {'-'*13} {'-'*10} {'-'*8}  {'-'*7}")
own_all, oth_all, wins = [], [], 0
for ln in lines:
    for i, k in enumerate(dkey):
        p = f"{OUT}/{ln}_{k}.wav"
        if not os.path.exists(p):
            continue
        e = emb(p)
        own, oth = cos(e, demb[i]), cos(e, demb[1 - i])
        own_all.append(own); oth_all.append(oth); wins += own > oth
        print(f"  {ln[:14]:<14} {own:>13.4f} {oth:>10.4f} {own-oth:>+8.4f}  "
              f"{'ok' if own > oth else 'WRONG WAY'}   [{k[:8]}]")

print()
print(f"  mean to own donor : {np.mean(own_all):+.4f}")
print(f"  mean to other     : {np.mean(oth_all):+.4f}")
print(f"  mean margin       : {np.mean(own_all)-np.mean(oth_all):+.4f}")
print(f"  correct direction : {wins}/{len(own_all)}")
print()
print("  Normalised against ECAPA's own calibration (0 = a different speaker,")
print("  1 = the same speaker on real audio):")
for nm, v in (("to own donor", np.mean(own_all)), ("to other", np.mean(oth_all))):
    print(f"    {nm:<14} {(v - c_diff) / (c_same - c_diff):+.3f}")
