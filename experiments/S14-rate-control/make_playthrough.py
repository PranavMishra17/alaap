"""
S14c -- stitch the 12 pairs into ONE file, so listening is one action.

Twenty-four separate file-opens is enough friction to stop a 3-minute task from
happening. This writes a single WAV with tone markers:

    LOW  tone (440 Hz, 0.40 s)  = a new pair starts; clip A comes next
    HIGH blip (880 Hz, 0.15 s)  = clip B comes next
    1.5 s of silence            = write your answer

The markers are IDENTICAL for every pair and every side, so they cue nothing
about the answer, and they are 7 dB below the speech so they do not startle.
The stimuli themselves are byte-identical to the individual pair files -- this
only concatenates what `run_clean_range_ab.py` already wrote.

    envs/qwen3/Scripts/python.exe experiments/S14-rate-control/make_playthrough.py
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
import numpy as np
import soundfile as sf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_ab")
SR = 44100


def tone(f, dur, amp_dbfs=-30.0):
    t = np.arange(int(SR * dur)) / SR
    w = np.sin(2 * np.pi * f * t) * (10 ** (amp_dbfs / 20.0))
    env = np.minimum(1.0, np.minimum(t, dur - t) / 0.02)   # de-click both ends
    return (w * env).astype(np.float32)


def sil(dur):
    return np.zeros(int(SR * dur), np.float32)


key = json.load(io.open(os.path.join(OUT, "ANSWER-KEY.json"), encoding="utf-8"))
parts = [sil(1.0)]
marks = []
for k in key:
    i = k["pair"]
    marks.append((len(np.concatenate(parts)) / SR, i))
    a, _ = sf.read(os.path.join(OUT, f"pair{i:02d}_A.wav"), dtype="float32")
    b, _ = sf.read(os.path.join(OUT, f"pair{i:02d}_B.wav"), dtype="float32")
    parts += [tone(440, 0.40), sil(0.35), a,
              sil(0.45), tone(880, 0.15), sil(0.35), b,
              sil(1.5)]

full = np.concatenate(parts)
p = os.path.join(OUT, "ALL-PAIRS.wav")
sf.write(p, np.clip(full, -1, 1), SR)
print(f"  {len(key)} pairs -> {p}")
print(f"  {len(full) / SR / 60:.1f} minutes total")
print()
print("  LOW tone  = new pair, A next")
print("  HIGH blip = B next")
print()
print(f"  {'pair':>4}  starts at")
for at, i in marks:
    print(f"  {i:>4}  {int(at // 60)}:{at % 60:04.1f}")
