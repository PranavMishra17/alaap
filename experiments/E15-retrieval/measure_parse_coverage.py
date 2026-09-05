import io, os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
sys.path.insert(0, os.path.abspath("."))
from alaap.captions import target_bins_from_text, caption_from_bins
from alaap.acoustics import BIN_LABELS
import numpy as np

AXES6 = ["f0_mean","spectral_tilt","hnr_db","f0_cv","speaking_rate","vtl_cm"]

# Grouped by how a real request would arrive. The first group is taken from
# the project's OWN existing examples (demo_script_render CAST, E9's cells,
# the scope document's promised range); the rest are written to look like what
# a game writer actually types. Both are mine, which is the bias to state.
GROUPS = {
 "project's own examples": [
  "a very deep voice, very rough and gravelly, speaking very slowly",
  "a high voice, crisp-toned, speaking quickly and highly animated",
  "a mid-range voice, warm-toned, at a steady pace",
  "an extremely low voice, muffled and warm, almost monotone",
  "a piercingly high voice, harsh and rasping, highly animated",
  "a mid-range voice, smooth and clear, with natural intonation",
 ],
 "character-sheet style": [
  "a gravelly old sailor, world-weary",
  "a nervous young clerk who talks too fast",
  "the queen: cold, precise, unhurried",
  "a giant of a man, voice like rocks grinding",
  "a scheming courtier, silky and quiet",
  "a battle-hardened sergeant barking orders",
  "a frightened child hiding in a cupboard",
  "an ancient tree spirit, slow and resonant",
 ],
 "plain description": [
  "deep male voice, slow",
  "high pitched, fast, cheerful",
  "raspy and low",
  "soft breathy whisper",
  "loud, harsh, aggressive",
  "calm and monotone",
 ],
 "no acoustic content at all": [
  "someone menacing",
  "a villain",
  "make it sound cool",
  "the protagonist's best friend",
 ],
}

print("  How much of a description does target_bins_from_text actually recover?")
print(f"  (6 axes possible: {', '.join(a.split('_')[0] for a in AXES6)})")
print()
allc = []
for g, items in GROUPS.items():
    cov = []
    print(f"  {g}")
    for t in items:
        b = target_bins_from_text(t)
        got = [a for a in AXES6 if a in b]
        cov.append(len(got))
        allc.append(len(got))
        print(f"    {len(got)}/6  {t[:52]:<52} {','.join(a.split('_')[0] for a in got)}")
    print(f"    -> mean {np.mean(cov):.2f}/6 axes\n")

# the reference point: a caption the pipeline generated itself
gen = [caption_from_bins({a: BIN_LABELS[a][i % 5] for a in AXES6}, seed=i)
       for i in range(6)]
gcov = [len([a for a in AXES6 if a in target_bins_from_text(t)]) for t in gen]
print(f"  generated captions (the E15 condition): mean {np.mean(gcov):.2f}/6")
print(f"  ALL user-style text:                    mean {np.mean(allc):.2f}/6 "
      f"({np.mean(allc)/6:.0%} of axes)")
