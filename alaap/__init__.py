"""
Alaap — description -> persistent voice identity -> rendered dialogue.

Layering (see RESEARCH/GRAND-PLAN.md):

    data.py      corpus streaming, licence-audited sources only   (I4)
    encoder.py   frozen speaker encoder + cache-embeddings-once
    geometry.py  SpeakerSpace: the ONLY place raw vectors are touched  (I1)
    metrics.py   identity consistency + diversity                 (I9, I10)

Nothing outside geometry.py should manipulate a raw speaker embedding.
E3 established why: 79.5% of a raw Qwen3-TTS vector is a shared constant
offset, so raw cosine cannot discriminate speakers at all.
"""
__version__ = "0.1.0"

from .geometry import SpeakerSpace, cosine_matrix, pairwise_cosines  # noqa: F401
from .metrics import (verification_stats, vendi_score, nn_distances,  # noqa: F401
                      cluster_separability)

__all__ = ["SpeakerSpace", "cosine_matrix", "pairwise_cosines",
           "verification_stats", "vendi_score", "nn_distances",
           "cluster_separability", "__version__"]
