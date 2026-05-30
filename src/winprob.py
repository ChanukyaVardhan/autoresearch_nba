"""Tiny fixed score+time -> win-probability baseline for the 'edge' feature.

A simple logistic in (margin / sqrt(time_remaining)). NOT fit to outcomes here
(that would be a separate calibration step); it is a fixed, deterministic prior so
the edge feature = model_winprob - market_implied_prob is well-defined. Coefficients
are intentionally frozen so the feature is stable across autoresearch iterations.
"""
from __future__ import annotations

import math

# Frozen coefficient: scales margin-per-sqrt-minute into logit space. Chosen so a
# 10-point lead with ~12 min left gives ~0.85 win prob (rough NBA prior).
_K = 0.28


def home_winprob(home_margin: int, game_secs_remaining: float) -> float:
    """P(home wins) from current margin and time left. Deterministic."""
    mins_left = max(game_secs_remaining, 1.0) / 60.0
    z = _K * home_margin / math.sqrt(mins_left)
    return 1.0 / (1.0 + math.exp(-z))
