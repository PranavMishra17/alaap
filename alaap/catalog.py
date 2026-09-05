"""
Catalog construction: how a described space is covered by generated voices.

Split out of E11 so the Indic catalog (S7) is built with the SAME sampler the
English one was. A saturation curve measured on a different cover of bin space
is not comparable to E11's, and the whole point of running it on a second
backend is the comparison.
"""
from __future__ import annotations

import numpy as np

from .acoustics import BIN_LABELS

# The axes caption_from_bins actually renders into prose, in the order it
# renders them. snr/jitter/shimmer are RECORDING-quality axes: a catalog voice
# should not be described as "very noisy" on purpose, so they are excluded even
# where the mapper measures them as carrying identity.
CATALOG_AXES = ["f0_mean", "spectral_tilt", "hnr_db", "f0_cv", "speaking_rate"]


def sample_cells(n: int, seed: int = 0,
                 axes: list[str] | None = None) -> list[dict[str, str]]:
    """
    Stratified cover of bin space: all one-axis moves off centre first, then
    the far corners, then a space-filling random fill. Deterministic in `seed`.

    Not exhaustive on purpose -- five axes x five bins is 3,125 cells and most
    are uninteresting interiors. The corners come early because E9 found the
    quality cliffs there, and a catalog that never probes them would report a
    saturation number that only holds for ordinary voices.
    """
    axes = axes or CATALOG_AXES
    rng = np.random.default_rng(seed)
    cells: list[dict[str, str]] = []
    seen: set[tuple[int, ...]] = set()

    def add(idx):
        key = tuple(idx)
        if key in seen:
            return False
        seen.add(key)
        cells.append({a: BIN_LABELS[a][i] for a, i in zip(axes, idx)})
        return True

    mid = [2] * len(axes)
    add(mid)
    for ai in range(len(axes)):
        for v in (0, 1, 3, 4):
            idx = list(mid)
            idx[ai] = v
            add(idx)
    for _ in range(min(64, n)):
        add([int(rng.choice([0, 4])) for _ in axes])
    guard = 0
    while len(cells) < n and guard < n * 200:
        guard += 1
        add([int(rng.integers(0, 5)) for _ in axes])
    return cells[:n]


def saturation_curve(accepted: list[bool], window: int = 20) -> list[float]:
    """
    Rolling acceptance rate as the catalog grows. Falling means new
    descriptions are landing on voices that already exist -- which is what
    saturation IS, seen from the side that ships.
    """
    out = []
    for i in range(len(accepted)):
        lo = max(0, i - window + 1)
        w = accepted[lo:i + 1]
        out.append(sum(w) / len(w))
    return out
