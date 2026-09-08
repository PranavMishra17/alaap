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

# The axes caption_from_bins actually renders into prose. snr/jitter/shimmer are
# RECORDING-quality axes: a catalog voice should not be described as "very noisy"
# on purpose, so they are excluded even where the mapper measures them as
# carrying identity.
#
# THE V1 SET IS KEPT BECAUSE COMMITTED NUMBERS WERE MEASURED ON IT.
#
# `CATALOG_AXES` is read by eight experiments. Changing it in place would leave
# every RESULTS.md that quotes an acceptance count or an adherence percentage
# silently describing a run nobody can reproduce -- the same class of mistake as
# a cache key without a model id, which this project has now made twice (E0, S2).
# So the old set stays, named, and anything re-run against it says which it used.
CATALOG_AXES_V1 = ["f0_mean", "spectral_tilt", "hnr_db", "f0_cv", "speaking_rate"]

# The identity set, adopted 2026-09-08 from S17/S18 (and cleared by S21).
#
#   f0_cv, speaking_rate  DROPPED. Within-speaker spread EXCEEDS between-speaker
#       spread on real people -- 1.04 and 1.32 (S16), reproducing S12 on acted
#       emotion and E15 on three corpora. Two of every five words spent
#       describing a voice described nothing that separates voices.
#   hnr_db  DROPPED. Weakest of the four at 0.59, and fails outright on Tamil at
#       1.32. Measured cost of keeping it: 0.3 effective voices bought for 15.6
#       points of adherence (S18).
#   vtl_cm  ADDED. Never noise -- the ESTIMATOR was broken, averaging through
#       the vowel-dependent F2. Repaired in S17; gender separation went from
#       ~0 to +0.44..+1.03 on four corpora.
#
# S17 measured 19.6 -> 22.3 effective voices and 85.8% -> 98.1% adherence for
# this swap. BOTH CHANNELS MUST CHANGE TOGETHER: S17 arm 3 kept 5-axis cells and
# only zeroed the mapper weights, and got the adherence win with a diversity LOSS
# (18.4, below baseline) because descriptions differing only in a dead axis
# collide and are rejected as duplicates. Caption and sampler are both driven
# from here, so they cannot drift apart.
#
# CAVEAT TO CARRY (S21): `spectral_tilt` is a usable identity axis but NOT a
# clean brightness axis -- it reads F0 at r = +0.918 where brightness is fixed
# by construction. Count this set as ~2.6 independent axes, not 3.
CATALOG_AXES = ["f0_mean", "spectral_tilt", "vtl_cm"]


def sample_cells(n: int, seed: int = 0,
                 axes: list[str] | None = None) -> list[dict[str, str]]:
    """
    Stratified cover of bin space: all one-axis moves off centre first, then
    the far corners, then a space-filling random fill. Deterministic in `seed`.

    Not exhaustive on purpose -- five axes x five bins is 3,125 cells and most
    are uninteresting interiors. The corners come early because E9 found the
    quality cliffs there, and a catalog that never probes them would report a
    saturation number that only holds for ordinary voices.

    RAISES when `n` exceeds the describable space, rather than returning fewer
    cells than asked for. Adopting the 3-axis identity set (S17/S18) took that
    space from 3,125 cells to 125, so a caller asking for 300 used to get 125
    back in silence -- and a saturation curve built on that reads as the CATALOG
    saturating when really the SAMPLER ran out of things to ask for. S18 flagged
    that the cell count must bind eventually and left where untested; this makes
    the boundary say so instead of being absorbed into a result.

    The fix when this fires is more bins per axis or another real axis, NOT
    catching it -- S18 measured that neither buys capacity at n=80, so a caller
    hitting this is asking a question the axis set cannot currently answer.
    """
    axes = axes or CATALOG_AXES
    space = 5 ** len(axes)
    if n > space:
        raise ValueError(
            f"asked for {n} distinct cells but {len(axes)} axes x 5 bins is only "
            f"{space} describable descriptions ({', '.join(axes)}). Returning "
            f"{space} silently would make a saturation curve read as the catalog "
            f"saturating rather than the sampler exhausting. Add bins or an axis."
        )
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
