"""Observability layer (DESIGN s5.1). Produces a rich, structured diagnostic report
the autoresearch loop feeds back to Codex so it can reason about HOW to improve
feature_construction / training — not just whether the headline moved.

All of this is FIXED/trusted (Codex does not edit it; like backtest/evaluate). It is
read-only over a trained policy + a split, and it never touches end-state data except
the settlement that the simulator already books.

The report answers questions like:
  - What is the policy actually doing? (action mix, entry/exit timing, sizing)
  - Where does P&L come from / leak? (by game phase, by score-margin regime,
    favorite vs underdog, blowout vs close)
  - Is the value net calibrated? (predicted value vs realized return-to-go)
  - Which features carry signal? (permutation-importance proxy on the policy)
  - Which games are the worst losers / best winners? (concrete examples to study)
  - Are there pathologies? (never trades, always trades, one-game domination)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .backtest import BacktestEnv
from .feature_construction import FEATURE_DIM, FEATURE_NAMES, feature_construction
from .game import Game
from .networks import N_SIZE, PolicyNet, CriticNet, SIZE_BUCKETS
from .types import Action


@dataclass
class GameDiag:
    event_ticker: str
    pnl: float
    n_buys: int
    n_sells: int
    settled_open: bool
    home_won: bool
    first_entry_frac: float   # fraction into game of first BUY (1.0 = never)
    avg_entry_price: float


@dataclass
class Report:
    # --- policy behaviour ---
    action_mix: dict           # SKIP_HOLD / BUY / SELL counts across all steps
    avg_first_entry_frac: float
    pct_games_no_trade: float
    sizing_mix: dict           # bucket -> count (on BUY steps)
    # --- pnl attribution ---
    pnl_by_margin_regime: dict # 'leading'/'trailing'/'close' -> mean step contribution
    pnl_favorite_vs_dog: dict  # by whether HOME was favorite at entry
    one_game_domination: float # share of total |pnl| from the single biggest game
    # --- value-net calibration ---
    value_calibration_mae: float
    value_corr: float
    # --- feature signal ---
    feature_importance: dict   # feature_name -> permutation drop in headline (proxy)
    # --- concrete examples ---
    worst_games: list          # list[GameDiag-as-dict]
    best_games: list
    # --- sanity ---
    notes: list = field(default_factory=list)


def _run_policy(game: Game, policy: PolicyNet, critic: CriticNet | None,
                collect: bool = True):
    """Greedy rollout collecting per-step diagnostics."""
    env = BacktestEnv(game)
    steps = []
    while True:
        t = env.t
        x = feature_construction(game, t, env.pos)[None, :]
        mask = env.action_mask()
        a_probs, s_probs, _ = policy.policy(x, mask[None, :])
        a = int(np.argmax(a_probs[0]))
        sz_idx = int(np.argmax(s_probs[0]))
        v = None
        if critic is not None:
            v = float(critic.value(x)[0][0])
        score = game.score_at(t)
        margin = score.home_points - score.away_points
        tr = env.step(a, float(SIZE_BUCKETS[sz_idx]))
        if collect:
            steps.append({
                "t": t, "a": a, "sz_idx": sz_idx, "reward": tr.reward,
                "margin": margin, "frac": (t - game.t_start) / max(1, game.t_end - game.t_start),
                "value": v, "x": x[0],
            })
        if tr.done:
            break
    return env.result(), steps


def _regime(margin: int) -> str:
    if margin >= 6:
        return "leading"
    if margin <= -6:
        return "trailing"
    return "close"


def build_report(games: list[Game], policy: PolicyNet, critic: CriticNet,
                 baseline_headline: float, score_fn) -> Report:
    from .evaluate import evaluate

    action_mix = {a.name: 0 for a in Action}
    sizing_mix = {f"{b:.2f}": 0 for b in SIZE_BUCKETS}
    regime_pnl = {"leading": [], "trailing": [], "close": []}
    fav_pnl = {"favorite": [], "underdog": []}
    diags: list[GameDiag] = []
    calib_pred, calib_real = [], []
    notes: list[str] = []

    for g in games:
        res, steps = _run_policy(g, policy, critic)
        # action + sizing + regime attribution
        first_entry = 1.0
        entry_price = 0.0
        for i, s in enumerate(steps):
            action_mix[Action(s["a"]).name] += 1
            if s["a"] == Action.BUY:
                sizing_mix[f"{SIZE_BUCKETS[s['sz_idx']]:.2f}"] += 1
                if first_entry == 1.0:
                    first_entry = s["frac"]
                    entry_price = g.candle_at(s["t"]).yes_ask_close
            regime_pnl[_regime(s["margin"])].append(s["reward"])
            # value calibration: predicted value vs realized return-to-go
            if s["value"] is not None:
                rtg = sum(z["reward"] for z in steps[i:])
                calib_pred.append(s["value"]); calib_real.append(rtg)
        # favorite/underdog by HOME implied prob at first entry (or tip)
        tip_prob = g.candle_at(g.t_start).implied_prob
        bucket = "favorite" if tip_prob >= 0.5 else "underdog"
        fav_pnl[bucket].append(res.realized_pnl)
        diags.append(GameDiag(
            event_ticker=g.event_ticker, pnl=res.realized_pnl,
            n_buys=res.n_buys, n_sells=res.n_sells, settled_open=res.settled_open,
            home_won=g.settlement.home_won, first_entry_frac=first_entry,
            avg_entry_price=entry_price,
        ))

    pnls = np.array([d.pnl for d in diags], float)
    no_trade = float(np.mean([1.0 if (d.n_buys == 0) else 0.0 for d in diags])) if diags else 0.0
    domination = float(np.max(np.abs(pnls)) / (np.sum(np.abs(pnls)) + 1e-9)) if len(pnls) else 0.0

    # value calibration
    cp, cr = np.array(calib_pred), np.array(calib_real)
    cal_mae = float(np.mean(np.abs(cp - cr))) if len(cp) else 0.0
    cal_corr = float(np.corrcoef(cp, cr)[0, 1]) if len(cp) > 2 and cp.std() > 0 and cr.std() > 0 else 0.0

    # feature-importance proxy: permute each feature across a sample, measure headline drop
    fi = _feature_importance(games[: min(len(games), 30)], policy, baseline_headline, score_fn)

    # notes / pathology flags
    total_actions = sum(action_mix.values()) or 1
    if action_mix["BUY"] / total_actions < 0.005:
        notes.append("policy almost never BUYs — likely collapsed to always-skip; "
                     "consider stronger entry signal features or higher entropy.")
    if no_trade > 0.5:
        notes.append(f"{no_trade:.0%} of games have zero trades — under-trading.")
    if domination > 0.4:
        notes.append(f"{domination:.0%} of |P&L| comes from one game — unstable; "
                     "metric is being driven by an outlier.")
    if cal_corr < 0.1 and len(cp):
        notes.append("value net poorly calibrated (corr<0.1) — critic not learning; "
                     "check value features / training.")

    diags.sort(key=lambda d: d.pnl)
    return Report(
        action_mix=action_mix,
        avg_first_entry_frac=float(np.mean([d.first_entry_frac for d in diags])) if diags else 1.0,
        pct_games_no_trade=no_trade,
        sizing_mix=sizing_mix,
        pnl_by_margin_regime={k: (float(np.mean(v)) if v else 0.0) for k, v in regime_pnl.items()},
        pnl_favorite_vs_dog={k: (float(np.mean(v)) if v else 0.0) for k, v in fav_pnl.items()},
        one_game_domination=domination,
        value_calibration_mae=cal_mae,
        value_corr=cal_corr,
        feature_importance=fi,
        worst_games=[asdict(d) for d in diags[:5]],
        best_games=[asdict(d) for d in diags[-5:]],
        notes=notes,
    )


def _feature_importance(games, policy, baseline_headline, score_fn) -> dict:
    """Permutation-importance proxy: zero each feature dimension and measure the drop
    in the headline. A dim whose zeroing hurts a lot is one the policy relies on."""
    from .evaluate import evaluate
    rng = np.random.default_rng(0)
    base = evaluate(games, policy).headline
    importance = {}
    # group-zeroing per named scalar feature (player block summarized as one group)
    n_named = len(FEATURE_NAMES)
    for j, name in enumerate(FEATURE_NAMES):
        drop = _headline_with_zeroed(games, policy, [j])
        importance[name] = round(base - drop, 5)
    # player block as one group
    importance["player_block"] = round(base - _headline_with_zeroed(
        games, policy, list(range(n_named, FEATURE_DIM))), 5)
    return importance


def _headline_with_zeroed(games, policy, zero_idx: list[int]) -> float:
    """Re-run greedy eval but zero out the given feature indices before the policy."""
    from .evaluate import Metrics, score
    rets = []
    zidx = np.array(zero_idx)
    for g in games:
        env = BacktestEnv(g)
        while True:
            x = feature_construction(g, env.t, env.pos)[None, :].copy()
            x[0, zidx] = 0.0
            mask = env.action_mask()
            a_probs, s_probs, _ = policy.policy(x, mask[None, :])
            a = int(np.argmax(a_probs[0])); sz = float(SIZE_BUCKETS[int(np.argmax(s_probs[0]))])
            if env.step(a, sz).done:
                break
        rets.append(env.result().realized_pnl)
    rets = np.array(rets, float)
    mean_r, std_r = (rets.mean(), rets.std()) if len(rets) else (0.0, 0.0)
    sharpe = mean_r / (std_r + 1e-9)
    return sharpe  # headline ~ sharpe (overtrade penalty omitted for the proxy)
