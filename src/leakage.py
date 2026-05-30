"""Leakage tripwires (DESIGN s4 invariants). Run EVERY autoresearch iteration:
an edit that introduces lookahead must FAIL the harness, not score well.

Core test: prefix-invariance. feature_construction(game, t, pos) must be identical
whether the game's events after t exist or not — i.e. it depends only on data with
wall_clock <= t. We test by comparing the live game against a truncated copy.
"""
from __future__ import annotations

import copy

import numpy as np

from .feature_construction import FEATURE_DIM, feature_construction
from .game import Game
from .types import PositionState


def _truncate(game: Game, t: int) -> Game:
    """A shallow copy of the game whose candles/pbp/box are sliced to <= t, so any
    accidental peek past t would yield a DIFFERENT vector than the full game."""
    g = copy.copy(game)
    g._feat_cache = {}  # MUST NOT share the cache with the full game (leakage test)
    # slice candles
    keep_c = [c for c in game._candles if c.ts <= t]
    if len(keep_c) < 1:
        keep_c = game._candles[:1]
    g._candles = keep_c
    g._candle_ts = [c.ts for c in keep_c]
    # slice pbp
    keep_p = [r for r in game._pbp if r["_wc"] <= t]
    if len(keep_p) < 1:
        keep_p = game._pbp[:1]
    g._pbp = keep_p
    g._pbp_ts = [r["_wc"] for r in keep_p]
    # box snapshots only up to t
    g._box = {k: v for k, v in game._box.items() if k <= t} or {min(game._box): game._box[min(game._box)]}
    return g


def check_prefix_invariance(game: Game, n_probe: int = 8) -> tuple[bool, str]:
    """Returns (ok, message). Probes several t's across the game."""
    steps = game.steps_ts
    if len(steps) < 3:
        return True, "too few steps to probe"
    probes = np.linspace(1, len(steps) - 2, num=min(n_probe, len(steps) - 2), dtype=int)
    pos = PositionState()
    for pi in probes:
        t = steps[int(pi)]
        full = feature_construction(game, t, pos)
        trunc = feature_construction(_truncate(game, t), t, pos)
        if full.shape != trunc.shape:
            return False, f"dim mismatch at t={t}: {full.shape} vs {trunc.shape}"
        if not np.allclose(full, trunc, atol=1e-6, equal_nan=False):
            diff = np.where(~np.isclose(full, trunc, atol=1e-6))[0]
            return False, f"LEAKAGE at t={t}: features differ at idx {diff.tolist()[:5]}"
    return True, "prefix-invariant"


def check_finite_and_dim(game: Game) -> tuple[bool, str]:
    pos = PositionState()
    for t in game.steps_ts[::5]:
        x = feature_construction(game, t, pos)
        if x.shape[0] != FEATURE_DIM:
            return False, f"dim {x.shape[0]} != FEATURE_DIM {FEATURE_DIM} at t={t}"
        if not np.all(np.isfinite(x)):
            return False, f"non-finite feature at t={t}"
    return True, "finite + fixed dim"


def run_leakage_suite(games: list[Game]) -> tuple[bool, str]:
    """Run all tripwires on a sample of games. ALL must pass."""
    sample = games[: min(len(games), 6)]
    for g in sample:
        ok, msg = check_finite_and_dim(g)
        if not ok:
            return False, f"{g.event_ticker}: {msg}"
        ok, msg = check_prefix_invariance(g)
        if not ok:
            return False, f"{g.event_ticker}: {msg}"
    return True, "leakage suite passed"
