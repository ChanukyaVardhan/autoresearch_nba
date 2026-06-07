"""Backtest simulator — the FIXED, TRUSTED ground (DESIGN s8). Codex NEVER edits
this; editing the scorer is how an agent cheats its own benchmark.

Owns: per-state action masking, fills at yes_ask (BUY) / yes_bid (SELL), the +T
clock advance (every action consumes one step), normalized budget + multi-lot
accounting, terminal settlement, and the fee model. Produces per-step rewards so the
same simulator serves both RL rollouts and evaluation.

Single HOME side. fee_model: "none" (default) charges 0 fees but the bid/ask SPREAD
is ALWAYS applied (it's structural, not a fee).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .game import Game
from .types import Action, Lot, N_ACTIONS, PositionState


def fee_none(notional: float) -> float:
    return 0.0


def kalshi_taker_fee(notional: float) -> float:
    # Placeholder smooth taker fee; turn on later via fee_model.
    return 0.0


FEE_MODELS: dict[str, Callable[[float], float]] = {
    "none": fee_none,
    "kalshi": kalshi_taker_fee,
}


@dataclass
class Trade:
    t: int
    action: str
    price: float
    size: float


@dataclass
class EpisodeResult:
    realized_pnl: float
    trades: list[Trade]
    n_buys: int
    n_sells: int
    max_drawdown: float
    deployed_peak: float
    settled_open: bool


def legal_actions(pos: PositionState, buy_size: float, min_lot: float = 0.05) -> np.ndarray:
    """Mask (len N_ACTIONS) of legal actions for the current state."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[Action.SKIP_HOLD] = True  # always legal (SKIP when flat, HOLD when holding)
    if pos.budget_remaining >= min_lot:
        mask[Action.BUY] = True
    if pos.is_holding:
        mask[Action.SELL] = True
    return mask


@dataclass
class StepTransition:
    reward: float
    done: bool


class BacktestEnv:
    """Finite-horizon MDP over one game's HOME side. Deterministic transitions."""

    def __init__(self, game: Game, budget: float = 1.0, fee_model: str = "none",
                 min_lot: float = 0.05):
        # budget = 1.0 == $1 of capital PER GAME. The agent trades that $1 over the
        # game; PnL is profit on the $1 (e.g. +0.72 = +72c/game).
        self.game = game
        self.fee = FEE_MODELS[fee_model]
        self.min_lot = min_lot
        self.steps = game.steps_ts
        self.budget = budget
        self.reset()

    def reset(self) -> None:
        self.pos = PositionState(lots=[], budget_total=self.budget)
        self.i = 0
        self._prev_equity = 0.0
        self.trades: list[Trade] = []
        self.n_buys = 0
        self.n_sells = 0
        self._equity_curve: list[float] = []
        self._deployed_peak = 0.0

    @property
    def t(self) -> int:
        return self.steps[self.i]

    @property
    def done(self) -> bool:
        return self.i >= len(self.steps) - 1

    def _equity(self, t: int) -> float:
        """Realized-so-far is folded into rewards; equity here = unrealized mark."""
        mid = self.game.candle_at(t).mid
        return self.pos.unrealized_pnl(mid)

    def action_mask(self) -> np.ndarray:
        return legal_actions(self.pos, buy_size=self.min_lot, min_lot=self.min_lot)

    def step(self, action: int, buy_fraction: float) -> StepTransition:
        """Apply action at current t, then advance the clock by one step (T).
        Returns per-step reward = change in (realized+unrealized) equity, net fees.
        buy_fraction in [0,1] = fraction of REMAINING budget to deploy on BUY.
        """
        t = self.t
        c = self.game.candle_at(t)
        realized = 0.0

        mask = self.action_mask()
        if not mask[action]:
            action = Action.SKIP_HOLD  # illegal -> coerce to no-op (mask should prevent)

        if action == Action.BUY:
            size = max(self.min_lot, buy_fraction * self.pos.budget_remaining)
            size = min(size, self.pos.budget_remaining)
            if size >= self.min_lot:
                price = c.yes_ask_close
                self.pos.lots.append(Lot(entry_price=price, size=size, t_entry=t))
                realized -= self.fee(size)
                self.trades.append(Trade(t, "BUY", price, size))
                self.n_buys += 1
        elif action == Action.SELL and self.pos.is_holding:
            price = c.yes_bid_close
            # close all lots (oldest-first is moot when closing all)
            for lot in self.pos.lots:
                realized += (price - lot.entry_price) * lot.size
                realized -= self.fee(lot.size)
                self.trades.append(Trade(t, "SELL", price, lot.size))
            self.n_sells += 1
            self.pos.lots = []
        # SKIP_HOLD: nothing

        self._deployed_peak = max(self._deployed_peak, self.pos.deployed)

        # advance clock
        prev_unreal = self._equity(t)
        self.i += 1
        settled_open = False
        if self.done:
            # terminal: settle any open lots at settlement price
            sp = self.game.settlement_price()
            for lot in self.pos.lots:
                realized += (sp - lot.entry_price) * lot.size
                self.trades.append(Trade(self.t, "SETTLE", sp, lot.size))
            if self.pos.lots:
                settled_open = True
            self.pos.lots = []
            new_unreal = 0.0
        else:
            new_unreal = self._equity(self.t)

        # per-step reward: realized this step + change in unrealized mark
        reward = realized + (new_unreal - prev_unreal)
        self._equity_curve.append((self._equity_curve[-1] if self._equity_curve else 0.0) + reward)
        self._last_settled_open = settled_open
        return StepTransition(reward=reward, done=self.done)

    def result(self) -> EpisodeResult:
        eq = self._equity_curve or [0.0]
        peak = eq[0]
        max_dd = 0.0
        for v in eq:
            peak = max(peak, v)
            max_dd = max(max_dd, peak - v)
        return EpisodeResult(
            realized_pnl=eq[-1],
            trades=self.trades,
            n_buys=self.n_buys,
            n_sells=self.n_sells,
            max_drawdown=max_dd,
            deployed_peak=self._deployed_peak,
            settled_open=getattr(self, "_last_settled_open", False),
        )
