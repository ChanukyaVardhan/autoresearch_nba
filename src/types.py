"""Core types for the autoresearch NBA trading system.

See DESIGN_autoresearch_trading.md. We model a single side (HOME). Actions:
FLAT -> {SKIP, BUY}; HOLDING -> {HOLD, SELL}. Default budget $10 (literal cash);
multiple lots allowed. Every action advances the clock by one step.

PositionState was rewritten 2026-06-07 to track REAL CASH:
  - `lots[].size` is now number of CONTRACTS (not budget fractions).
  - `cash` is current dollars; `budget_total` is initial dollars.
  - On BUY: cash -= price * contracts. On SELL/settle: cash += price * contracts.
  - PnL formulas now report literal dollars.
  - `budget_remaining` returns the FRACTION (0..1) of initial cash remaining,
    so feature_construction stays in [0..1] without modification.
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
    """One open long-HOME position. `size` is the number of contracts."""

    entry_price: float   # yes_ask paid at entry (dollars per contract, 0..1)
    size: float          # number of contracts held (real units)
    t_entry: int         # wall-clock epoch seconds of entry


@dataclass
class PositionState:
    """Aggregate position across open lots, in REAL CASH dollars.

    Default budget is $10 of literal capital. The agent's policy sees
    normalized features (budget_remaining as fraction, PnL as fraction of
    budget), so its training is invariant to the absolute budget size.
    """

    lots: list[Lot] = field(default_factory=list)
    budget_total: float = 100.0  # initial cash in $
    cash: float = 100.0          # current cash in $

    def __post_init__(self) -> None:
        # If the caller passed only budget_total, mirror it into cash so the
        # invariant `cash == budget_total - dollars_deployed` holds at t=0.
        if self.cash == 100.0 and self.budget_total != 100.0 and not self.lots:
            self.cash = self.budget_total

    @property
    def is_holding(self) -> bool:
        return len(self.lots) > 0

    @property
    def total_size(self) -> float:
        """Total contracts held across all open lots."""
        return sum(l.size for l in self.lots)

    @property
    def dollars_deployed(self) -> float:
        """$ tied up in open lots, valued at entry cost (price * contracts)."""
        return sum(l.entry_price * l.size for l in self.lots)

    @property
    def deployed(self) -> float:
        """Alias for dollars_deployed (kept for compatibility with old call sites)."""
        return self.dollars_deployed

    @property
    def budget_remaining(self) -> float:
        """FRACTION of initial cash remaining (0..1). Stable feature regardless
        of the dollar size of `budget_total`, so the policy network sees the
        same range whether budget is $1 or $10 or $1000."""
        if self.budget_total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.cash / self.budget_total))

    @property
    def cash_remaining(self) -> float:
        """Literal dollars currently available to spend."""
        return self.cash

    @property
    def avg_entry(self) -> float:
        contracts = self.total_size
        if contracts <= 0:
            return 0.0
        return sum(l.entry_price * l.size for l in self.lots) / contracts

    def unrealized_pnl(self, mid: float) -> float:
        """Mark-to-mid PnL on open lots in DOLLARS."""
        return sum((mid - l.entry_price) * l.size for l in self.lots)

    def unrealized_pnl_fraction(self, mid: float) -> float:
        """Mark-to-mid PnL normalized by initial budget (for features). In
        [-1, 1] since |PnL| <= dollars_deployed <= budget_total."""
        if self.budget_total <= 0:
            return 0.0
        return self.unrealized_pnl(mid) / self.budget_total

    def equity(self, mid: float) -> float:
        """Total equity in $ = cash + market value of open lots at `mid`."""
        return self.cash + sum(mid * l.size for l in self.lots)


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
