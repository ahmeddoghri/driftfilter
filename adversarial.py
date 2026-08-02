"""Adversarial seeds for the faster-drift stress test.

TUNING_SEEDS: used to characterize driftfilter.py's degradation under
faster drift and tune driftfilter_v2.py's velocity-tracking fix.
HOLDOUT_SEEDS: disjoint, evaluated exactly once after the fix was finalized.

Both are evaluated at STRESS_SHIFT, roughly 4.4x the published drift rate
(total_shift=3.4 over 240 steps) -- still linear, still plausible, just
faster. driftfilter.py's plain-EMA filter degrades sharply at this rate
even though it hits a perfect, seed-invariant 1.0 at the published rate.
"""

STRESS_SHIFT = 15.0

TUNING_SEEDS = list(range(1, 41))
HOLDOUT_SEEDS = list(range(1000, 1030))
