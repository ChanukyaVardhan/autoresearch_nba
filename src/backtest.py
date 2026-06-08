"""Backtest simulator — the FIXED, TRUSTED ground (DESIGN s8). Codex NEVER edits
this; editing the scorer is how an agent cheats its own benchmark.

Owns: per-state action masking, fills at yes_ask (BUY) / yes_bid (SELL), the +T
clock advance (every action consumes one step), REAL CASH accounting in literal
dollars, terminal settlement, the fee model, and order latency.

Single HOME side. Defaults:
  budget   = $10 (literal dollars of capital — see CHANGES below)
  latency  = 0  (fill at the same step the action was decided; pass latency>=1 to
                 fill at step (i + latency)'s prices, modeling order-routing lag)
  fee_model= "none" (charges 0 fees; bid/ask SPREAD is ALWAYS applied — it's
             structural, not a fee).

CHANGES on 2026-06-07
---------------------
Prior simulator treated `Lot.size` as "fraction of budget" and accounted for the
budget as "max position notional", never tracking literal cash. PnL was numerically
correct but capital-efficiency metrics (drawdown vs. real $ at risk, deployed peak)
were inflated. This rewrite:
  * `Lot.size` is now CONTRACTS.
  * BUY costs `price * contracts` in cash. SELL/settlement returns `price * contracts`.
  * `buy_fraction` is now the fraction of REMAINING CASH to deploy on a BUY.
  * `PositionState.budget_remaining` still returns a fraction (0..1) so the
    feature network sees stable inputs regardless of budget size.
  * `latency` parameter added: action at step i fills using prices from
    candle_at(steps[min(i + latency, last_step)]). Models order-routing delay
    where the market may move against you between decide and fill.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from .game import Game
from .types import Action, Lot, N_ACTIONS, PositionState


# A latency "spec" can be:
#   int N                              constant N steps of fill delay
#   ("uniform", lo, hi)                random integer in [lo, hi] inclusive each step
#   ("exponential", mean)              non-negative integer from Exp(1/mean), rounded
#   callable(rng) -> int               custom: any callable taking np.random.Generator
#                                       and returning a non-negative int
LatencySpec = Union[
    int,
    tuple,
    Callable[[np.random.Generator], int],
]


def _make_latency_sampler(
    spec: LatencySpec,
) -> Callable[[np.random.Generator], int]:
    """Translate a latency spec into a per-step sampling function. Raises on
    bad input — fail fast in __init__, not silently mid-trial."""
    if callable(spec) and not isinstance(spec, tuple):
        return spec  # user supplied their own sampler
    if isinstance(spec, bool):
        # bool is an int subclass but almost certainly a typo
        raise TypeError(f"latency must not be bool, got {spec}")
    if isinstance(spec, int):
        if spec < 0:
            raise ValueError(f"latency must be >= 0, got {spec}")
        n = spec
        return lambda rng, _n=n: _n
    if isinstance(spec, tuple) and len(spec) >= 2:
        kind = spec[0]
        if kind == "uniform" and len(spec) == 3:
            lo, hi = int(spec[1]), int(spec[2])
            if lo < 0 or hi < lo:
                raise ValueError(f"invalid uniform latency range ({lo}, {hi})")
            return lambda rng, _lo=lo, _hi=hi: int(rng.integers(_lo, _hi + 1))
        if kind == "exponential" and len(spec) == 2:
            mean = float(spec[1])
            if mean < 0:
                raise ValueError(f"exponential mean must be >= 0, got {mean}")
            # Round to integer steps; max with 0 in case of float underflow
            return lambda rng, _m=mean: max(0, int(round(rng.exponential(_m))))
    raise ValueError(
        f"unknown latency spec: {spec!r}. Expected int, "
        '("uniform", lo, hi), ("exponential", mean), or a callable.'
    )


# Fee functions take (price, contracts, rate_cents) and return dollars.
# Kept as module-level so tests can call them directly.

def fee_none(price: float, contracts: float, rate_cents: float = 0.0) -> float:
    return 0.0


def kalshi_taker_fee(price: float, contracts: float, rate_cents: float = 1.75) -> float:
    """Kalshi taker fee per trade, in dollars.

    Formula: fee_cents = ceil(rate_cents * contracts * P * (1 - P))
    where P is the fill price in [0, 1]. Kalshi always rounds UP to the
    nearest cent. The standard general-market rate is 1.75¢; some series
    (e.g. elections) charge more — pass `rate_cents` explicitly when needed.
    """
    if contracts <= 0 or price <= 0 or price >= 1:
        # Pure 0 or pure 1 contracts have no P*(1-P) component;
        # the ceil of 0 is 0. Zero-contract trades have no fee.
        # (Edge cases prevent floating-point ceil(0) → 0 surprise.)
        if contracts <= 0:
            return 0.0
        if price <= 0 or price >= 1:
            return 0.0
    fee_cents = math.ceil(rate_cents * contracts * price * (1.0 - price))
    return fee_cents / 100.0


FEE_MODELS: dict[str, Callable[..., float]] = {
    "none": fee_none,
    "kalshi": kalshi_taker_fee,
}


# Minimum trade size in CONTRACTS. 0.05 contracts ≈ 2.5–4¢ at typical prices.
DEFAULT_MIN_LOT_CONTRACTS = 0.05


@dataclass
class Trade:
    t: int
    action: str
    price: float
    size: float       # contracts


@dataclass
class EpisodeResult:
    realized_pnl: float            # final cash - initial cash (in $)
    trades: list[Trade]
    n_buys: int
    n_sells: int
    max_drawdown: float            # peak-to-trough drawdown of equity, in $
    deployed_peak: float           # max $ tied up in open lots at any point
    deployed_peak_fraction: float  # same, as fraction of initial budget
    settled_open: bool


def legal_actions(pos: PositionState, *, min_buy_cost: float = 0.0) -> np.ndarray:
    """Mask (len N_ACTIONS) of legal actions for the current state.

    BUY is legal only when there's enough cash to afford the minimum lot at the
    cheapest possible price ($0.01). `min_buy_cost` is `min_lot_contracts *
    cheapest_price`; default 0 means "can always try BUY if any cash remains".
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[Action.SKIP_HOLD] = True
    if pos.cash > min_buy_cost:
        mask[Action.BUY] = True
    if pos.is_holding:
        mask[Action.SELL] = True
    return mask


@dataclass
class StepTransition:
    reward: float
    done: bool


class BacktestEnv:
    """Finite-horizon MDP over one game's HOME side. Deterministic transitions
    (the underlying candle stream is fixed in backtest)."""

    def __init__(
        self,
        game: Game,
        budget: float = 100.0,
        fee_model: str = "none",
        fee_rate_cents: float = 1.75,
        min_lot: float = DEFAULT_MIN_LOT_CONTRACTS,
        latency: LatencySpec = 0,
        slippage_cents: LatencySpec = 0,
        seed: int | None = None,
    ):
        """
        budget         — initial cash in $ (default $100).
        fee_model      — "none" (free) or "kalshi" (per-trade taker fee).
        fee_rate_cents — multiplier for the Kalshi fee formula. Default 1.75¢
                         (general markets). Some series charge more (e.g. 3.5¢
                         for elections). Ignored when fee_model="none".
        min_lot        — minimum trade size in CONTRACTS (not dollars).
        latency        — order-routing delay measured in STEPS (1 step = 1
                         minute in this sim). Use this only for modeling rare
                         catastrophic delays (disconnect, reconnect). For
                         normal sub-second routing lag, use slippage_cents
                         instead. Forms: int N, ("uniform", lo, hi),
                         ("exponential", mean), or callable(rng) -> int.
        slippage_cents — within-bar price drift you pay on each fill. The
                         candle's close-of-bar bid/ask is not where the market
                         actually was at the moment your order hit the
                         exchange — typically you pay 0..3 ¢ worse. Same spec
                         format as latency; sampled per fill. Recommended:
                         ("uniform", 0, 1) for typical conditions, ("uniform",
                         1, 3) for stress testing.
        seed           — RNG seed for reproducible sampling. None = OS entropy.
        """
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        if fee_rate_cents < 0:
            raise ValueError(f"fee_rate_cents must be >= 0, got {fee_rate_cents}")
        self.game = game
        self.fee_rate_cents = float(fee_rate_cents)
        _fee_fn = FEE_MODELS[fee_model]
        # Closure that bakes in the rate so call sites can pass (price, contracts)
        self.fee: Callable[[float, float], float] = (
            lambda price, contracts, _f=_fee_fn, _r=self.fee_rate_cents:
                _f(price, contracts, _r)
        )
        self.min_lot = float(min_lot)
        self._sample_latency = _make_latency_sampler(latency)
        self._sample_slippage_cents = _make_latency_sampler(slippage_cents)
        self._rng = np.random.default_rng(seed)
        self._latency_spec = latency             # kept for repr/debug; do not mutate
        self._slippage_spec = slippage_cents     # kept for repr/debug; do not mutate
        self.steps = game.steps_ts
        self.budget = float(budget)
        self.reset()

    def reset(self) -> None:
        self.pos = PositionState(lots=[], budget_total=self.budget, cash=self.budget)
        self.i = 0
        self.trades: list[Trade] = []
        self.n_buys = 0
        self.n_sells = 0
        self._equity_curve: list[float] = []
        self._deployed_peak = 0.0
        self._last_settled_open = False

    @property
    def t(self) -> int:
        return self.steps[self.i]

    @property
    def done(self) -> bool:
        return self.i >= len(self.steps) - 1

    def _equity(self, t: int) -> float:
        """Mark-to-mid equity (cash + open-lot mark value) in $.

        Initial equity is `self.budget`; reward streams sum to (final equity -
        budget). For backwards-compat with the prior reward semantics (which
        used unrealized PnL, not cash + value), we still return
        `unrealized_pnl(mid)` — this equals the change in equity minus realized
        cash flows, which is what the rewards already capture.
        """
        mid = self.game.candle_at(t).mid
        return self.pos.unrealized_pnl(mid)

    def action_mask(self) -> np.ndarray:
        return legal_actions(self.pos, min_buy_cost=self.min_lot * 0.01)

    def _fill_price(self, side: str) -> float:
        """Latency + slippage aware fill price. Samples both per call so each
        fill is independent.

        Step 1: pick the target candle. If latency > 0, look that many steps
                ahead (clamped to the last available step) — models rare
                catastrophic routing delay.
        Step 2: apply within-bar slippage. Add sampled cents to the ask (BUY
                pays more) or subtract from the bid (SELL receives less) —
                models normal sub-step price drift inside the 1-minute bar.
        Step 3: clip to [0.0, 1.0] (Kalshi binary contract bounds)."""
        latency = self._sample_latency(self._rng)
        target_idx = min(self.i + max(0, latency), len(self.steps) - 1)
        c = self.game.candle_at(self.steps[target_idx])
        base = c.yes_ask_close if side == "buy" else c.yes_bid_close
        slippage = self._sample_slippage_cents(self._rng) / 100.0
        if side == "buy":
            return float(min(1.0, base + slippage))
        return float(max(0.0, base - slippage))

    def step(self, action: int, buy_fraction: float) -> StepTransition:
        """Apply action at current t, then advance the clock by one step.

        `buy_fraction` ∈ [0,1] = fraction of REMAINING CASH to deploy on a BUY.
        Returns per-step reward = realized PnL this step + change in unrealized
        mark, net any fees. PnL is in literal $.

        Bug fix 2026-06-07: `prev_unreal` is now captured BEFORE the action so
        the reward at the BUY step properly reflects the spread cost (the gap
        between ask paid and mid). Previously prev_unreal was captured after
        the action, causing the spread component to leak into a later step's
        accounting and the realized_pnl total to be off by the unrealized
        carrying loss when the position settled.
        """
        t = self.t
        c_now = self.game.candle_at(t)
        realized = 0.0

        # Capture unrealized BEFORE the action so the reward stream picks up
        # the spread cost (ask - mid for BUY, mid - bid for SELL) as a real
        # immediate loss on this turn.
        prev_unreal = self._equity(t)

        mask = self.action_mask()
        if not mask[action]:
            action = Action.SKIP_HOLD  # coerce illegal -> no-op

        if action == Action.BUY:
            price = self._fill_price("buy")
            if price > 0:
                # Compute desired contract size from cash + buy_fraction
                buy_fraction = float(np.clip(buy_fraction, 0.0, 1.0))
                cash_to_spend = buy_fraction * self.pos.cash
                contracts = cash_to_spend / price
                # Apply min_lot floor (in contracts), if affordable
                if contracts < self.min_lot:
                    contracts = self.min_lot
                # Cap by cash (can't spend more than we have)
                max_contracts = self.pos.cash / price
                contracts = min(contracts, max_contracts)
                # Only execute if we can afford the (possibly clamped) min lot
                if contracts >= self.min_lot - 1e-12 and contracts * price <= self.pos.cash + 1e-9:
                    cost = contracts * price
                    fee = self.fee(price, contracts)
                    # Fee comes out of cash (real money flow) AND out of the
                    # reward stream so the agent sees the friction.
                    self.pos.cash -= cost + fee
                    self.pos.lots.append(Lot(entry_price=price, size=contracts, t_entry=t))
                    realized -= fee
                    self.trades.append(Trade(t, "BUY", price, contracts))
                    self.n_buys += 1
                # else: not enough cash for even min_lot at this price -> no-op
        elif action == Action.SELL and self.pos.is_holding:
            price = self._fill_price("sell")
            # close all lots (oldest-first is moot when closing all)
            for lot in self.pos.lots:
                proceeds = price * lot.size
                fee = self.fee(price, lot.size)
                # Fee comes out of cash AND out of reward
                self.pos.cash += proceeds - fee
                realized += (price - lot.entry_price) * lot.size - fee
                self.trades.append(Trade(t, "SELL", price, lot.size))
            self.n_sells += 1
            self.pos.lots = []
        # SKIP_HOLD: nothing

        self._deployed_peak = max(self._deployed_peak, self.pos.dollars_deployed)

        # advance clock (prev_unreal was captured at the top, before the action)
        self.i += 1
        settled_open = False
        if self.done:
            # terminal: settle any open lots at the settlement price (0 or 1)
            sp = self.game.settlement_price()
            for lot in self.pos.lots:
                proceeds = sp * lot.size
                self.pos.cash += proceeds
                realized += (sp - lot.entry_price) * lot.size
                self.trades.append(Trade(self.t, "SETTLE", sp, lot.size))
            if self.pos.lots:
                settled_open = True
            self.pos.lots = []
            new_unreal = 0.0
        else:
            new_unreal = self._equity(self.t)

        # per-step reward: realized this step + change in unrealized mark.
        # Sum over an episode == final_cash - initial_budget.
        reward = realized + (new_unreal - prev_unreal)
        self._equity_curve.append(
            (self._equity_curve[-1] if self._equity_curve else 0.0) + reward
        )
        # Sticky: once an open lot has been settled this episode, stay True.
        # (Prior behavior reset to False on any subsequent terminal step.)
        self._last_settled_open = self._last_settled_open or settled_open
        return StepTransition(reward=reward, done=self.done)

    def result(self) -> EpisodeResult:
        eq = self._equity_curve or [0.0]
        peak = eq[0]
        max_dd = 0.0
        for v in eq:
            peak = max(peak, v)
            max_dd = max(max_dd, peak - v)
        deployed_peak_fraction = (
            self._deployed_peak / self.budget if self.budget > 0 else 0.0
        )
        return EpisodeResult(
            realized_pnl=eq[-1],
            trades=self.trades,
            n_buys=self.n_buys,
            n_sells=self.n_sells,
            max_drawdown=max_dd,
            deployed_peak=self._deployed_peak,
            deployed_peak_fraction=deployed_peak_fraction,
            settled_open=self._last_settled_open,
        )
