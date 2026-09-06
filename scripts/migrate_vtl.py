"""Recompute `vtl_cm` in cached measurements from the cached formants.

S17 replaced the VTL estimator: formant dispersion averaged the spacings
THROUGH F2, the vowel-dependent formant, which cancelled what F1 and F3 knew.
Gender effect size went from +0.11 / -0.17 / +0.06 / +0.11 to +0.70 / +1.03 /
+0.44 / +0.68 across hi, bn, ta and en.

The caches hold the OLD value -- but they also hold f1/f2/f3, and
`vocal_tract_length` is a pure function of those. So this is a recompute, not a
re-measure: no audio is streamed and nothing is re-estimated. A backup is
written beside each file before it is touched.

    envs/qwen3/Scripts/python.exe scripts/migrate_vtl.py [--apply]
"""
import argparse
import glob
import io
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alaap.acoustics import vocal_tract_length

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true", help="without this, dry run")
ap.add_argument("--glob", default="experiments/*/out/measured_*.npz")
args = ap.parse_args()

for path in sorted(glob.glob(args.glob)):
    d = np.load(path, allow_pickle=True)
    if "attrs" not in d:
        continue
    attrs = json.loads(str(d["attrs"]))
    if not attrs or "f1" not in attrs[0]:
        print(f"  {os.path.basename(path):<44} no formants cached, skipped")
        continue
    old = np.array([a.get("vtl_cm", np.nan) for a in attrs], float)
    for a in attrs:
        a["vtl_cm"] = vocal_tract_length(
            np.array([a.get("f1", np.nan), a.get("f2", np.nan),
                      a.get("f3", np.nan), a.get("f4", np.nan)], float))
    new = np.array([a["vtl_cm"] for a in attrs], float)
    print(f"  {os.path.basename(path):<44} "
          f"{np.nanmean(old):>6.2f} -> {np.nanmean(new):>6.2f} cm  "
          f"({np.isfinite(new).mean():.0%} finite)")
    if args.apply:
        shutil.copy2(path, path + ".pre-vtl-fix")
        payload = {k: d[k] for k in d.files}
        payload["attrs"] = json.dumps(attrs)
        np.savez_compressed(path, **payload)

print("\ndry run — pass --apply to write" if not args.apply else "\nwritten; "
      "originals kept as *.pre-vtl-fix")
