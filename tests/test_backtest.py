"""Unit tests for the rewritten cash-accounting BacktestEnv.

Pins the contracts that the autoresearch loop and evaluator rely on:
  - cash conservation (sum of trades + remaining cash == initial budget +/- PnL)
  - spread is paid on every round trip
  - latency uses future-step prices on the correct side of the spread
  - settlement pays out at the true settlement price
  - default budget is $10
  - PnL scales linearly with budget (policy quality is budget-invariant)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import BacktestEnv, _make_latency_sampler, kalshi_taker_fee
from src.types import Action, Candle


# ---------------------------------------------------------------------------
# Minimal stub Game for backtest tests — fixed candle stream + settlement
# ---------------------------------------------------------------------------

class StubGame:
    """A toy Game with a hand-specified candle stream and settlement."""

    def __init__(self, candles, settlement: float):
        self._candles = {c.ts: c for c in candles}
        self.steps_ts = [c.ts for c in candles]
        self._settle = settlement

    def candle_at(self, ts):
        return self._candles[ts]

    def settlement_price(self) -> float:
        return self._settle


def _candle(ts, bid, ask, *, mid=None):
    """Quick candle constructor — only the fields the env reads need real values."""
    if mid is None:
        mid = 0.5 * (bid + ask)
    return Candle(
        ts=ts,
        price_open=mid, price_high=mid, price_low=mid, price_close=mid,
        price_mean=mid, price_previous=mid,
        yes_bid_close=bid, yes_ask_close=ask,
        volume=0.0, open_interest=0.0,
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_default_budget_is_100_dollars():
    g = StubGame([_candle(0, 0.50, 0.50)], settlement=0.0)
    env = BacktestEnv(g)
    assert env.budget == 100.0
    assert env.pos.cash == 100.0
    assert env.pos.budget_total == 100.0
    # The fraction-normalized feature is 1.0 (full cash remaining)
    assert env.pos.budget_remaining == 1.0


def test_custom_budget_round_trips():
    g = StubGame([_candle(0, 0.50, 0.50)], settlement=0.0)
    env = BacktestEnv(g, budget=100.0)
    assert env.pos.cash == 100.0 and env.pos.budget_total == 100.0


def test_rejects_invalid_budget_and_latency():
    g = StubGame([_candle(0, 0.50, 0.50)], settlement=0.0)
    with pytest.raises(ValueError):
        BacktestEnv(g, budget=0.0)
    with pytest.raises(ValueError):
        BacktestEnv(g, latency=-1)


# ---------------------------------------------------------------------------
# Cash conservation — the load-bearing invariant
# ---------------------------------------------------------------------------

def test_buy_costs_real_cash():
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.55, 0.65), _candle(2, 0.55, 0.65)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01)
    env.step(Action.BUY, 0.5)  # spend half of $10 at ask=0.65
    # Spent ~$5, got ~5/0.65 = ~7.692 contracts
    assert env.pos.cash == pytest.approx(5.0, abs=1e-6)
    assert env.pos.total_size == pytest.approx(5.0 / 0.65, abs=1e-6)


def test_sell_returns_cash_minus_spread():
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.55, 0.65), _candle(2, 0.55, 0.65)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01)
    env.step(Action.BUY, 1.0)   # spend all $10 at ask=0.65 -> 15.384 contracts
    n_contracts = env.pos.total_size
    env.step(Action.SELL, 0.0)  # close all at bid=0.55
    # Final cash = contracts * 0.55 = 15.384 * 0.55 ≈ $8.461
    # Loss = $10 - $8.461 = $1.538 ≈ spread (0.10) * contracts (15.384)
    assert env.pos.cash == pytest.approx(n_contracts * 0.55, abs=1e-6)
    expected_loss = (0.65 - 0.55) * n_contracts
    assert env.pos.cash == pytest.approx(10.0 - expected_loss, abs=1e-6)


def test_cash_conservation_through_settlement():
    """Sum of all cash flows == final cash. With settlement at YES=1.0, profit
    equals (1.0 - entry) * contracts. Uses 3 candles so BUY doesn't terminate
    immediately (settlement only fires when self.i reaches len(steps)-1)."""
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.55, 0.65), _candle(2, 0.55, 0.65)],
        settlement=1.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01)
    env.step(Action.BUY, 1.0)         # spend $10 at 0.65
    contracts = env.pos.total_size    # 15.384...
    env.step(Action.SKIP_HOLD, 0.0)   # advance one step, still holding
    env.step(Action.SKIP_HOLD, 0.0)   # terminal -> settles at 1.0
    expected_profit = (1.0 - 0.65) * contracts
    expected_final_cash = 10.0 + expected_profit  # ~$15.38
    assert env.pos.cash == pytest.approx(expected_final_cash, abs=1e-6)
    res = env.result()
    assert res.realized_pnl == pytest.approx(expected_profit, abs=1e-6)


# ---------------------------------------------------------------------------
# Spread — round trip with no price change must lose money
# ---------------------------------------------------------------------------

def test_spread_eats_pnl_on_round_trip():
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.55, 0.65), _candle(2, 0.55, 0.65)],
        settlement=0.5,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01)
    env.step(Action.BUY, 0.5)
    env.step(Action.SELL, 0.0)
    # We bought at 0.65 and sold at 0.55 — loss is positive
    res = env.result()
    assert res.realized_pnl < 0


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def test_latency_zero_uses_current_candle():
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.70, 0.80), _candle(2, 0.55, 0.65)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, latency=0)
    env.step(Action.BUY, 1.0)  # fills at step 0's ask = 0.65
    assert env.pos.lots[0].entry_price == pytest.approx(0.65)


def test_latency_one_fills_at_next_candle_price():
    """latency=1 means the order routing eats the next tick — fill at i+1's
    prices instead of i's."""
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.70, 0.80), _candle(2, 0.55, 0.65)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, latency=1)
    env.step(Action.BUY, 1.0)  # decided at step 0, fills at step 1's ask = 0.80
    assert env.pos.lots[0].entry_price == pytest.approx(0.80)


def test_latency_clamps_at_last_step():
    """If latency would point past the last candle, fall back to the final
    available price (you don't 'see' beyond the horizon). Uses 3 candles so
    we can inspect the lot before terminal settlement clears it."""
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.70, 0.80), _candle(2, 0.85, 0.95)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, latency=5)
    env.step(Action.BUY, 1.0)  # latency clamps to last step -> ask=0.95
    assert env.pos.lots[0].entry_price == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

def test_settlement_yes_pays_one_dollar_per_contract():
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.55, 0.65), _candle(2, 0.55, 0.65)],
        settlement=1.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01)
    env.step(Action.BUY, 1.0)
    contracts = env.pos.total_size
    env.step(Action.SKIP_HOLD, 0.0)
    env.step(Action.SKIP_HOLD, 0.0)  # terminal -> settle at 1.0
    res = env.result()
    assert res.settled_open
    assert res.realized_pnl == pytest.approx((1.0 - 0.65) * contracts, abs=1e-6)


def test_settlement_no_loses_entry_cost():
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.55, 0.65), _candle(2, 0.55, 0.65)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01)
    env.step(Action.BUY, 1.0)  # spent $10 at 0.65
    contracts = env.pos.total_size
    env.step(Action.SKIP_HOLD, 0.0)
    env.step(Action.SKIP_HOLD, 0.0)  # settle at 0.0
    res = env.result()
    # Lost the entire entry value: -0.65 * contracts == -$10
    assert res.realized_pnl == pytest.approx(-0.65 * contracts, abs=1e-6)
    assert res.realized_pnl == pytest.approx(-10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Scaling — policy quality should be budget-invariant
# ---------------------------------------------------------------------------

def test_pnl_scales_linearly_with_budget():
    """Same actions on the same candle stream: PnL at budget=$100 should be
    exactly 10x the PnL at budget=$10."""
    def _run(budget):
        g = StubGame(
            [_candle(0, 0.55, 0.65), _candle(1, 0.70, 0.80), _candle(2, 0.65, 0.75)],
            settlement=1.0,
        )
        env = BacktestEnv(g, budget=budget, min_lot=0.01)
        env.step(Action.BUY, 1.0)
        env.step(Action.SKIP_HOLD, 0.0)  # settle at 1.0
        return env.result().realized_pnl

    p10 = _run(10.0)
    p100 = _run(100.0)
    assert p10 != 0
    assert p100 / p10 == pytest.approx(10.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Reward stream sums to total PnL
# ---------------------------------------------------------------------------

def test_reward_stream_sums_to_realized_pnl():
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.70, 0.80), _candle(2, 0.65, 0.75), _candle(3, 0.80, 0.90)],
        settlement=1.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01)
    rewards = []
    rewards.append(env.step(Action.BUY, 1.0).reward)
    rewards.append(env.step(Action.SKIP_HOLD, 0.0).reward)
    rewards.append(env.step(Action.SELL, 0.0).reward)
    rewards.append(env.step(Action.SKIP_HOLD, 0.0).reward)
    total = sum(rewards)
    assert total == pytest.approx(env.result().realized_pnl, abs=1e-6)


# ---------------------------------------------------------------------------
# Budget exhaustion — can't overspend
# ---------------------------------------------------------------------------

def test_cannot_buy_more_than_cash_allows():
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.55, 0.65), _candle(2, 0.55, 0.65)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=1.0, min_lot=0.01)
    env.step(Action.BUY, 1.0)  # spend everything
    cash_after_first = env.pos.cash
    env.step(Action.BUY, 1.0)  # try to spend more — should no-op or only spend what's left
    assert env.pos.cash >= 0
    # Total cash deployed must not exceed initial budget
    assert env.pos.dollars_deployed <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Variable latency — distributions + reproducibility
# ---------------------------------------------------------------------------

def _stairstep_candles(n=20, start_ask=0.20, step=0.05):
    """Each candle has a distinctly different ask — makes the latency-driven
    fill price unambiguous to assert on."""
    cs = []
    for i in range(n):
        ask = start_ask + step * i
        bid = ask - 0.02
        cs.append(_candle(i, bid, ask))
    return cs


def test_constant_int_latency_still_works():
    """Backwards compat: latency=0 → fill at current candle, latency=2 → +2."""
    g = StubGame(_stairstep_candles(10), settlement=0.0)
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, latency=0, seed=0)
    env.step(Action.BUY, 1.0)
    assert env.pos.lots[0].entry_price == pytest.approx(0.20)

    g = StubGame(_stairstep_candles(10), settlement=0.0)
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, latency=2, seed=0)
    env.step(Action.BUY, 1.0)
    # step 0 + latency 2 = step 2, ask = 0.20 + 0.05*2 = 0.30
    assert env.pos.lots[0].entry_price == pytest.approx(0.30)


def test_uniform_latency_samples_only_within_range():
    """100 draws of Uniform[1, 3] should cover exactly {1, 2, 3} — nothing more, nothing less."""
    sampler = _make_latency_sampler(("uniform", 1, 3))
    rng = np.random.default_rng(0)
    samples = {sampler(rng) for _ in range(200)}
    assert samples == {1, 2, 3}

    sampler0 = _make_latency_sampler(("uniform", 0, 0))
    rng = np.random.default_rng(0)
    assert all(sampler0(rng) == 0 for _ in range(20))


def test_exponential_latency_sample_mean_near_target():
    """1000 draws of Exp(mean=2) should average close to 2 (round-to-int adds a
    little bias but stays in a sane band)."""
    sampler = _make_latency_sampler(("exponential", 2.0))
    rng = np.random.default_rng(1234)
    samples = [sampler(rng) for _ in range(2000)]
    mean = sum(samples) / len(samples)
    # Rounded exponential mean is slightly above the rate (rounding 0.5 up),
    # but should land in [1.5, 2.5] easily for 2000 samples.
    assert 1.5 < mean < 2.5
    assert min(samples) >= 0
    assert max(samples) >= 5  # tail exists


def test_callable_latency_lets_user_define_distribution():
    """User can pass any callable taking np.random.Generator -> int."""
    g = StubGame(_stairstep_candles(10), settlement=0.0)
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, latency=lambda _rng: 1, seed=0)
    env.step(Action.BUY, 1.0)
    assert env.pos.lots[0].entry_price == pytest.approx(0.25)  # step 0 + 1 = ask 0.25


def test_same_seed_produces_same_latency_fills():
    """Reproducibility: same seed + same actions = identical fill prices."""
    g1 = StubGame(_stairstep_candles(20), settlement=0.0)
    g2 = StubGame(_stairstep_candles(20), settlement=0.0)
    env1 = BacktestEnv(g1, budget=10.0, min_lot=0.01, latency=("uniform", 0, 3), seed=7)
    env2 = BacktestEnv(g2, budget=10.0, min_lot=0.01, latency=("uniform", 0, 3), seed=7)
    for _ in range(5):
        env1.step(Action.BUY, 0.1)
        env2.step(Action.BUY, 0.1)
    prices1 = [l.entry_price for l in env1.pos.lots]
    prices2 = [l.entry_price for l in env2.pos.lots]
    assert prices1 == prices2


def test_different_seeds_produce_different_latency_fills():
    """Sanity: different seeds → different sampled latencies → different fills.
    (Could collide by chance; using enough steps makes that vanishingly unlikely.)"""
    g1 = StubGame(_stairstep_candles(30), settlement=0.0)
    g2 = StubGame(_stairstep_candles(30), settlement=0.0)
    env1 = BacktestEnv(g1, budget=10.0, min_lot=0.01, latency=("uniform", 0, 5), seed=1)
    env2 = BacktestEnv(g2, budget=10.0, min_lot=0.01, latency=("uniform", 0, 5), seed=2)
    for _ in range(15):
        env1.step(Action.BUY, 0.05)
        env2.step(Action.BUY, 0.05)
    prices1 = [l.entry_price for l in env1.pos.lots]
    prices2 = [l.entry_price for l in env2.pos.lots]
    assert prices1 != prices2


# ---------------------------------------------------------------------------
# Slippage in cents — the realistic intra-bar friction
# ---------------------------------------------------------------------------

def test_slippage_zero_is_a_no_op():
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.55, 0.65), _candle(2, 0.55, 0.65)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, slippage_cents=0, seed=0)
    env.step(Action.BUY, 1.0)
    assert env.pos.lots[0].entry_price == pytest.approx(0.65)


def test_slippage_constant_makes_buy_pay_more():
    """Constant slippage_cents=2 should add 2 cents to the ask on BUY."""
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.55, 0.65), _candle(2, 0.55, 0.65)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, slippage_cents=2, seed=0)
    env.step(Action.BUY, 1.0)
    assert env.pos.lots[0].entry_price == pytest.approx(0.65 + 0.02)


def test_slippage_constant_makes_sell_receive_less():
    """Constant slippage_cents=2 should subtract 2 cents from the bid on SELL."""
    g = StubGame(
        [_candle(0, 0.55, 0.65), _candle(1, 0.55, 0.65), _candle(2, 0.55, 0.65)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, slippage_cents=2, seed=0)
    env.step(Action.BUY, 1.0)
    # Sell at next step — bid is 0.55, slippage drops to 0.53
    env.step(Action.SELL, 0.0)
    # The SELL trade should be recorded at 0.53
    sell_trades = [t for t in env.trades if t.action == "SELL"]
    assert sell_trades and sell_trades[0].price == pytest.approx(0.53)


def test_slippage_uniform_stays_within_range():
    """slippage_cents=("uniform", 0, 3) — every BUY fill should sit between
    candle ask and ask + 0.03 (inclusive). Small ask increments keep us
    safely under the $1 contract ceiling so clipping doesn't fire."""
    g = StubGame(_stairstep_candles(50, start_ask=0.20, step=0.01), settlement=0.0)
    env = BacktestEnv(
        g, budget=10.0, min_lot=0.01,
        slippage_cents=("uniform", 0, 3), seed=42,
    )
    for _ in range(30):
        env.step(Action.BUY, 0.03)
    fills = env.pos.lots  # all lots opened, never sold
    for i, lot in enumerate(fills):
        candle_ask = round(0.20 + 0.01 * i, 6)
        # Fill must be within [ask, ask + 0.03]
        assert candle_ask - 1e-9 <= lot.entry_price <= candle_ask + 0.03 + 1e-9, (
            f"fill {i}: ask={candle_ask} fill={lot.entry_price}"
        )


def test_slippage_clamps_at_one():
    """Slippage cannot push a BUY fill above $1 (binary contract ceiling)."""
    g = StubGame(
        [_candle(0, 0.95, 0.99), _candle(1, 0.95, 0.99), _candle(2, 0.95, 0.99)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, slippage_cents=5, seed=0)
    env.step(Action.BUY, 1.0)
    # ask=0.99 + 5¢ slippage = 1.04 -> clipped to 1.00
    assert env.pos.lots[0].entry_price == pytest.approx(1.0)


def test_slippage_clamps_at_zero_on_sell():
    g = StubGame(
        [_candle(0, 0.02, 0.04), _candle(1, 0.02, 0.04), _candle(2, 0.02, 0.04)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=10.0, min_lot=0.01, slippage_cents=5, seed=0)
    env.step(Action.BUY, 1.0)  # ask=0.04 + 5¢ = 0.09 (no clip needed)
    env.step(Action.SELL, 0.0)  # bid=0.02 - 5¢ = -0.03 -> clipped to 0.00
    sell_trades = [t for t in env.trades if t.action == "SELL"]
    assert sell_trades and sell_trades[0].price == pytest.approx(0.0)


def test_slippage_seed_reproducibility():
    g1 = StubGame(_stairstep_candles(20), settlement=0.0)
    g2 = StubGame(_stairstep_candles(20), settlement=0.0)
    env1 = BacktestEnv(g1, budget=10.0, min_lot=0.01, slippage_cents=("uniform", 0, 3), seed=99)
    env2 = BacktestEnv(g2, budget=10.0, min_lot=0.01, slippage_cents=("uniform", 0, 3), seed=99)
    for _ in range(8):
        env1.step(Action.BUY, 0.05)
        env2.step(Action.BUY, 0.05)
    assert [l.entry_price for l in env1.pos.lots] == [l.entry_price for l in env2.pos.lots]


def test_bad_latency_spec_raises():
    g = StubGame([_candle(0, 0.5, 0.5)], settlement=0.0)
    with pytest.raises(ValueError):
        BacktestEnv(g, latency=("uniform", 3, 1))            # hi < lo
    with pytest.raises(ValueError):
        BacktestEnv(g, latency=("exponential", -1.0))        # negative mean
    with pytest.raises(ValueError):
        BacktestEnv(g, latency=("not-a-distribution", 1, 2))
    with pytest.raises(TypeError):
        BacktestEnv(g, latency=True)                          # bool is suspicious
    with pytest.raises((TypeError, ValueError)):
        BacktestEnv(g, latency="2")                           # str not allowed


# ---------------------------------------------------------------------------
# Kalshi fee model — the formula and its effects on PnL / cash
# ---------------------------------------------------------------------------

def test_kalshi_fee_formula_at_50_cents():
    """fee = ceil(1.75 × 10 × 0.5 × 0.5) = ceil(4.375) = 5¢ = $0.05"""
    assert kalshi_taker_fee(0.50, 10) == pytest.approx(0.05)


def test_kalshi_fee_formula_at_60_cents():
    """fee = ceil(1.75 × 20 × 0.6 × 0.4) = ceil(8.4) = 9¢ = $0.09"""
    assert kalshi_taker_fee(0.60, 20) == pytest.approx(0.09)


def test_kalshi_fee_extreme_price_rounds_up():
    """At extreme prices the raw fee is tiny but ceil pulls it up to 1¢ minimum."""
    # 1 contract at 0.95: 1.75 * 1 * 0.95 * 0.05 = 0.083125 → ceil = 1¢
    assert kalshi_taker_fee(0.95, 1) == pytest.approx(0.01)
    assert kalshi_taker_fee(0.05, 1) == pytest.approx(0.01)


def test_kalshi_fee_zero_at_pure_prices():
    """At price 0 or 1, P(1-P) = 0 so no fee."""
    assert kalshi_taker_fee(0.0, 10) == 0.0
    assert kalshi_taker_fee(1.0, 10) == 0.0


def test_kalshi_fee_zero_contracts_no_fee():
    assert kalshi_taker_fee(0.50, 0) == 0.0


def test_kalshi_fee_with_custom_rate():
    """rate_cents=3.5 (e.g. elections series): fee doubles."""
    standard = kalshi_taker_fee(0.50, 10, 1.75)
    election = kalshi_taker_fee(0.50, 10, 3.5)
    # standard = 5¢, election = ceil(8.75) = 9¢
    assert standard == pytest.approx(0.05)
    assert election == pytest.approx(0.09)


def test_fees_reduce_cash_on_buy():
    """The fee must come out of cash, not just the reward stream."""
    g = StubGame(
        [_candle(0, 0.45, 0.55), _candle(1, 0.45, 0.55), _candle(2, 0.45, 0.55)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=100.0, min_lot=0.01, fee_model="kalshi", seed=0)
    # 10 contracts at 0.55 (ask): cost = $5.50; fee = ceil(1.75 * 10 * 0.55 * 0.45) = ceil(4.3125) = 5¢ = $0.05
    # We'd like exactly 10 contracts — use buy_fraction so size lands on 10
    desired_cash_to_spend = 10 * 0.55  # $5.50 for 10 contracts at ask 0.55
    env.step(Action.BUY, desired_cash_to_spend / env.pos.cash)
    # Expect cash = 100 - 5.50 - 0.05 = 94.45
    contracts = env.pos.total_size
    expected_cash = 100.0 - contracts * 0.55 - kalshi_taker_fee(0.55, contracts)
    assert env.pos.cash == pytest.approx(expected_cash, abs=1e-6)


def test_round_trip_pnl_reduced_by_both_side_fees():
    """Round trip with fees should differ from round trip without fees by
    exactly (buy_fee + sell_fee)."""
    candles = [_candle(0, 0.45, 0.55), _candle(1, 0.45, 0.55), _candle(2, 0.45, 0.55)]
    g_free = StubGame(list(candles), settlement=0.0)
    env_free = BacktestEnv(g_free, budget=100.0, min_lot=0.01, fee_model="none", seed=0)
    env_free.step(Action.BUY, 0.5)
    env_free.step(Action.SELL, 0.0)
    env_free.step(Action.SKIP_HOLD, 0.0)
    pnl_free = env_free.result().realized_pnl

    g_paid = StubGame(list(candles), settlement=0.0)
    env_paid = BacktestEnv(g_paid, budget=100.0, min_lot=0.01, fee_model="kalshi", seed=0)
    env_paid.step(Action.BUY, 0.5)
    contracts_bought = env_paid.pos.total_size
    env_paid.step(Action.SELL, 0.0)
    env_paid.step(Action.SKIP_HOLD, 0.0)
    pnl_paid = env_paid.result().realized_pnl

    buy_fee = kalshi_taker_fee(0.55, contracts_bought)
    sell_fee = kalshi_taker_fee(0.45, contracts_bought)
    expected_diff = -(buy_fee + sell_fee)
    actual_diff = pnl_paid - pnl_free
    assert actual_diff == pytest.approx(expected_diff, abs=1e-6)


def test_fees_break_cash_invariant_when_not_deducted():
    """Regression guard: after the fix, sum(rewards) must equal final_cash -
    initial_budget, even with fees on."""
    g = StubGame(
        [_candle(0, 0.45, 0.55), _candle(1, 0.45, 0.55), _candle(2, 0.45, 0.55),
         _candle(3, 0.50, 0.60)],
        settlement=1.0,
    )
    env = BacktestEnv(g, budget=100.0, min_lot=0.01, fee_model="kalshi", seed=0)
    rewards = []
    rewards.append(env.step(Action.BUY, 0.3).reward)
    rewards.append(env.step(Action.SKIP_HOLD, 0.0).reward)
    rewards.append(env.step(Action.SELL, 0.0).reward)
    rewards.append(env.step(Action.SKIP_HOLD, 0.0).reward)  # terminal -> settle
    total_reward = sum(rewards)
    cash_pnl = env.pos.cash - 100.0
    assert total_reward == pytest.approx(cash_pnl, abs=1e-6)


def test_kalshi_fee_disabled_by_fee_model_none():
    g = StubGame(
        [_candle(0, 0.45, 0.55), _candle(1, 0.45, 0.55), _candle(2, 0.45, 0.55)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=100.0, min_lot=0.01, fee_model="none", seed=0)
    env.step(Action.BUY, 0.5)
    contracts = env.pos.total_size
    # cash should be EXACTLY 100 - contracts*0.55 (no fee)
    assert env.pos.cash == pytest.approx(100.0 - contracts * 0.55, abs=1e-6)


def test_custom_fee_rate_propagates_to_env():
    g = StubGame(
        [_candle(0, 0.45, 0.55), _candle(1, 0.45, 0.55), _candle(2, 0.45, 0.55)],
        settlement=0.0,
    )
    env = BacktestEnv(g, budget=100.0, min_lot=0.01,
                     fee_model="kalshi", fee_rate_cents=3.5, seed=0)
    env.step(Action.BUY, 0.5)
    contracts = env.pos.total_size
    expected_fee = kalshi_taker_fee(0.55, contracts, rate_cents=3.5)
    expected_cash = 100.0 - contracts * 0.55 - expected_fee
    assert env.pos.cash == pytest.approx(expected_cash, abs=1e-6)


def test_fee_rate_cents_must_be_non_negative():
    g = StubGame([_candle(0, 0.50, 0.50)], settlement=0.0)
    with pytest.raises(ValueError):
        BacktestEnv(g, fee_rate_cents=-0.1)


