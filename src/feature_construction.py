"""feature_construction() — the state encoder. *** CODEX-EDITABLE SURFACE ***

This is the primary file the autoresearch loop rewrites (DESIGN s4). It is a PURE,
strictly-causal function: it may only read game data with wall_clock <= t (via the
Game accessors), must return a fixed-length float32 vector with no NaN/inf, and must
have no I/O or global state.

HARD RULES (enforced by tests, do not violate when editing):
  - Causality / prefix-invariance: output depends only on data with ts/wall_clock<=t.
  - NEVER read game-end player_stats or settlement (lookahead = cheating).
  - FEATURE_DIM is constant across all calls and all games.
"""
from __future__ import annotations

import math

import numpy as np

from .game import Game
from .types import PositionState
from .winprob import home_winprob

# Fixed input dimension (nets depend on it). Update FEATURE_NAMES if you change this.
TOP_K = 3  # top-K players per team encoded

FEATURE_NAMES = (
    # market microstructure
    "implied_prob", "mid", "spread",
    "vel_1", "vel_3", "vel_5", "accel",
    "log_volume_z", "volume_surge", "oi_delta",
    # game state
    "score_margin_norm", "period_norm", "period_secs_rem_norm", "game_secs_rem_norm",
    "run_60s", "run_180s", "home_possession", "last_is_timeout",
    # derived edge
    "model_winprob", "edge", "buy_edge", "sell_edge",
    "net_buy_edge", "net_hold_edge", "edge_delta_180", "model_wp_delta_180",
    "market_score_divergence",
    # position context
    "is_holding", "entry_price", "unrealized_pnl", "time_in_trade", "budget_frac_rem",
)
# player block: TOP_K players/team * (present, pts_rate, reb_rate, ast_rate, fouls) * 2 teams
_PLAYER_FEATS = ("present", "pts_rate_180", "reb_rate_180", "ast_rate_180", "fouls")
FEATURE_DIM = len(FEATURE_NAMES) + 2 * TOP_K * len(_PLAYER_FEATS)


def _safe(x: float) -> float:
    return float(x) if (x == x and abs(x) != math.inf) else 0.0


def _player_block(game: Game, t: int) -> list[float]:
    """Top-K-by-recent-activity players per team -> fixed-width feature block."""
    box = game.box_at(t)
    box_180 = game.box_at(t - 180)
    out: list[float] = []
    # We don't have team membership per parsed name without the resolution map;
    # split by whether the player's points contributed to home/away is not directly
    # known from the box. As a deterministic proxy, rank all players by total points
    # and emit TOP_K for "team A" and next TOP_K for "team B" placeholder slots.
    # (Team assignment refinement is a candidate edit; presence bit guards padding.)
    players = sorted(box.players.items(), key=lambda kv: -kv[1]["points"])
    for slot in range(2 * TOP_K):
        if slot < len(players):
            name, st = players[slot]
            st0 = box_180.players.get(name, {k: 0.0 for k in st})
            out.extend([
                1.0,
                (st["points"] - st0.get("points", 0.0)) / 180.0,
                (st["rebounds"] - st0.get("rebounds", 0.0)) / 180.0,
                (st["assists"] - st0.get("assists", 0.0)) / 180.0,
                min(st["fouls"], 6.0) / 6.0,
            ])
        else:
            out.extend([0.0] * len(_PLAYER_FEATS))
    return out


# Index range of the 5 position-context features within the vector. These depend
# on the agent's state and so are NOT cached.
_POS_START = FEATURE_NAMES.index("is_holding")
_POS_SLICE = slice(_POS_START, _POS_START + 5)


def _static_vector(game: Game, t: int) -> np.ndarray:
    """All features EXCEPT the 5 position-context ones (which depend on the agent).
    Cached per (game, t) since it's identical across rollouts/PPO epochs — this is
    the speed win: the expensive score/box reconstruction runs once per step, not
    once per rollout."""
    cache = game._feat_cache  # dict t -> np.ndarray (set up lazily on Game)
    cached = cache.get(t)
    if isinstance(cached, np.ndarray) and cached.shape == (FEATURE_DIM,):
        return cached
    c = game.candle_at(t)
    win = game.candles_window(t, 6)
    score = game.score_at(t)

    mid = c.mid
    mids = [w.mid for w in win]
    def back(n: int) -> float:
        return mids[-1 - n] if len(mids) > n else (mids[0] if mids else mid)
    vel_1 = mid - back(1); vel_3 = mid - back(3); vel_5 = mid - back(5)
    accel = vel_1 - (back(1) - back(2))

    vols = [w.volume for w in win if w.volume == w.volume]
    mean_vol = (sum(vols[:-1]) / max(1, len(vols) - 1)) if len(vols) > 1 else (vols[0] if vols else 0.0)
    volume_surge = (c.volume / mean_vol) if mean_vol > 0 else 1.0
    log_volume_z = math.log1p(max(0.0, c.volume)) / 15.0
    oi_delta = 0.0
    if len(win) >= 2 and win[-2].open_interest == win[-2].open_interest:
        oi_delta = (c.open_interest - win[-2].open_interest) / max(1.0, abs(c.open_interest))

    margin = score.home_points - score.away_points
    s60 = game.score_at(t - 60); s180 = game.score_at(t - 180)
    run_60 = (margin - (s60.home_points - s60.away_points)) / 12.0
    run_180 = (margin - (s180.home_points - s180.away_points)) / 24.0
    mwp = home_winprob(margin, score.game_secs_remaining)
    prev_margin = s180.home_points - s180.away_points
    prev_mwp = home_winprob(prev_margin, s180.game_secs_remaining)
    prev_c = game.candle_at(t - 180)
    prev_edge = prev_mwp - prev_c.implied_prob
    edge = mwp - c.implied_prob
    buy_edge = mwp - c.yes_ask_close
    sell_edge = mwp - c.yes_bid_close
    edge_delta_180 = edge - prev_edge
    model_wp_delta_180 = mwp - prev_mwp
    market_delta_180 = c.mid - prev_c.mid
    market_score_divergence = model_wp_delta_180 - market_delta_180

    base = [
        c.implied_prob, mid, c.spread,
        vel_1, vel_3, vel_5, accel,
        log_volume_z, min(volume_surge, 5.0) / 5.0, oi_delta,
        max(-1.0, min(1.0, margin / 25.0)),
        score.period / 6.0,
        score.period_secs_remaining / 720.0,
        score.game_secs_remaining / (48 * 60.0),
        max(-1.0, min(1.0, run_60)), max(-1.0, min(1.0, run_180)),
        1.0 if score.home_has_possession else 0.0,
        1.0 if score.last_event_is_timeout else 0.0,
        mwp, max(-1.0, min(1.0, edge)),
        max(-1.0, min(1.0, buy_edge)), max(-1.0, min(1.0, sell_edge)),
        max(-1.0, min(1.0, buy_edge - c.spread)),
        max(-1.0, min(1.0, sell_edge - c.spread)),
        max(-1.0, min(1.0, edge_delta_180)),
        max(-1.0, min(1.0, model_wp_delta_180)),
        max(-1.0, min(1.0, market_score_divergence)),
        0.0, 0.0, 0.0, 0.0, 0.0,  # position slots (filled per-call, not cached)
    ]
    arr = np.array([_safe(x) for x in base] + _player_block(game, t), dtype=np.float32)
    cache[t] = arr
    return arr


def feature_construction(game: Game, t: int, position: PositionState) -> np.ndarray:
    arr = _static_vector(game, t).copy()
    mid = game.candle_at(t).mid
    arr[_POS_SLICE] = [
        1.0 if position.is_holding else 0.0,
        _safe(position.avg_entry),
        _safe(max(-1.0, min(1.0, position.unrealized_pnl(mid)))),
        _safe(max(0.0, min(1.0, (t - (position.lots[0].t_entry if position.lots else t)) / (40 * 60.0)))),
        _safe(position.budget_remaining),
    ]
    assert arr.shape[0] == FEATURE_DIM, f"feature dim {arr.shape[0]} != {FEATURE_DIM}"
    return arr
