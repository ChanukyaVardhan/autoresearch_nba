"""Core types for the autoresearch NBA trading system.

See DESIGN_autoresearch_trading.md. We model a single side (HOME). Actions:
FLAT -> {SKIP, BUY}; HOLDING -> {HOLD, SELL}. Budget normalized to 1.0; multiple
lots allowed. Every action advances the clock by T minutes (wall-clock).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class Action(IntEnum):
    """Unified action head. Legality is state-dependent (see action_mask)."""

    SKIP_HOLD = 0  # SKIP when FLAT, HOLD when HOLDING (no fill either way)
    BUY = 1        # open/add a HOME lot at yes_ask  (legal only when budget remains)
    SELL = 2       # close lot(s) at yes_bid          (legal only when HOLDING)


N_ACTIONS = 3


@dataclass
class Lot:
    """One open long-HOME position."""

    entry_price: float   # yes_ask paid at entry (dollars, 0..1)
    size: float          # fraction of total budget deployed in this lot
    t_entry: int         # wall-clock epoch seconds of entry


@dataclass
class PositionState:
    """Aggregate position across open lots. budget is normalized to 1.0."""

    lots: list[Lot] = field(default_factory=list)
    budget_total: float = 1.0

    @property
    def is_holding(self) -> bool:
        return len(self.lots) > 0

    @property
    def deployed(self) -> float:
        return sum(l.size for l in self.lots)

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget_total - self.deployed)

    @property
    def total_size(self) -> float:
        return self.deployed

    @property
    def avg_entry(self) -> float:
        d = self.deployed
        if d <= 0:
            return 0.0
        return sum(l.entry_price * l.size for l in self.lots) / d

    def unrealized_pnl(self, mid: float) -> float:
        """Mark-to-mid P&L of open lots (in budget-fraction * price units)."""
        return sum((mid - l.entry_price) * l.size for l in self.lots)


@dataclass
class Candle:
    """One minute of trading on the HOME market side. Price is dollars 0..1."""

    ts: int            # ts_utc as epoch seconds
    price_open: float
    price_high: float
    price_low: float
    price_close: float
    price_mean: float
    price_previous: float
    yes_bid_close: float
    yes_ask_close: float
    volume: float
    open_interest: float

    @property
    def implied_prob(self) -> float:
        return self.price_close

    @property
    def mid(self) -> float:
        return 0.5 * (self.yes_bid_close + self.yes_ask_close)

    @property
    def spread(self) -> float:
        return self.yes_ask_close - self.yes_bid_close


@dataclass
class ScoreState:
    """Point-in-time game state reconstructed from PBP (causal as of t)."""

    home_points: int
    away_points: int
    period: int
    period_secs_remaining: float
    game_secs_remaining: float
    home_has_possession: bool
    last_event_is_timeout: bool


# Per-player point-in-time counting stats (the 12 player_stats fields).
PLAYER_STAT_FIELDS = (
    "points", "rebounds", "assists", "blocks", "steals", "turnovers",
    "field_goals_made", "field_goals_attempted",
    "three_points_made", "three_points_attempted",
    "free_throws_made", "fouls",
)


@dataclass
class PlayerLine:
    name: str
    is_home: bool
    on_court: bool
    stats: dict[str, float]  # keys subset of PLAYER_STAT_FIELDS


@dataclass
class Settlement:
    """Terminal reward info. END-STATE — reward path only, never a feature."""

    home_won: bool          # did HOME settle YES?
    home_final: int
    away_final: int
