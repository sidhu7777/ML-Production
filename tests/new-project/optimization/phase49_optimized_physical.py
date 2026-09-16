"""Phase 49 optimizer: current production physical scorer, array/parallel execution.

Delegates to the proven Phase 43 v2 execution kernel while retaining the
current production scorer and signature. No propagation, PAP, DEM, membership,
or calibration rule is changed here.
"""
from phase43_optimized_physical import score_candidates_phase43_v2


def score_candidates_phase49(*args, **kwargs):
    kwargs.setdefault("workers", 4)
    return score_candidates_phase43_v2(*args, **kwargs)
