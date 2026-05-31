"""training() — PyTorch PPO trainer. *** CODEX-EDITABLE SURFACE ***

Sequential RL over the finite-horizon per-game MDP (DESIGN s5): clipped PPO with a
critic baseline + GAE, entropy bonus, normalized advantages. Deterministic given
`seed`. Proper autograd (torch), so the critic actually learns.

Codex may edit hyperparameters, network sizes, reward shaping, advantage handling.
It may NOT edit backtest.py (trusted simulator) or read end-state data.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))  # use available CPU cores

from .backtest import BacktestEnv
from .feature_construction import FEATURE_DIM, FEATURE_NAMES, feature_construction
from .game import Game
from .networks import CriticNet, N_SIZE, PolicyNet, SIZE_BUCKETS
from .types import Action


@dataclass
class PPOConfig:
    # Reward-first PPO: dense edge-potential shaping supplies the exploration signal.
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.997
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    entropy_coef: float = 0.045  # keep stochastic BUY/SELL samples alive early
    size_entropy_coef: float = 0.012
    value_coef: float = 0.8
    entry_prior_coef: float = 0.0
    entry_prior_warmup_iters: int = 0
    edge_potential_coef: float = 0.06
    iters: int = 50
    batch_games: int = 64
    trade_cost: float = 0.0005   # curb churn without suppressing first entries
    pyramid_buy_cost: float = 0.004  # discourage repeated add-ons after entry
    seed: int = 0


IMPLIED_PROB_IDX = FEATURE_NAMES.index("implied_prob")
BUY_EDGE_IDX = FEATURE_NAMES.index("buy_edge")
SELL_EDGE_IDX = FEATURE_NAMES.index("sell_edge")
IS_HOLDING_IDX = FEATURE_NAMES.index("is_holding")
BUDGET_FRAC_REM_IDX = FEATURE_NAMES.index("budget_frac_rem")


def _rollout(game: Game, policy: PolicyNet, rng: np.random.Generator):
    """One episode under the current (stochastic) policy. Returns arrays."""
    env = BacktestEnv(game)
    S, NEXT_S, A, SZ, LOGP, R, MASK = [], [], [], [], [], [], []
    while True:
        x = feature_construction(game, env.t, env.pos)[None, :]
        mask = env.action_mask()
        a_probs, s_probs, _ = policy.policy(x, mask[None, :])
        a = int(rng.choice(Action.__len__() if hasattr(Action, "__len__") else 3, p=a_probs[0]))
        sz = int(rng.choice(N_SIZE, p=s_probs[0]))
        logp = float(np.log(a_probs[0, a] + 1e-12))
        if a == Action.BUY:
            logp += float(np.log(s_probs[0, sz] + 1e-12))
        tr = env.step(a, float(SIZE_BUCKETS[sz]))
        if tr.done:
            next_x = np.zeros_like(x[0], dtype=np.float32)
        else:
            next_x = feature_construction(game, env.t, env.pos)
        S.append(x[0]); A.append(a); SZ.append(sz); LOGP.append(logp)
        NEXT_S.append(next_x); R.append(tr.reward); MASK.append(mask)
        if tr.done:
            break
    return (np.array(S, np.float32), np.array(NEXT_S, np.float32),
            np.array(A, np.int64), np.array(SZ, np.int64),
            np.array(LOGP, np.float32), np.array(R, np.float32), np.array(MASK, bool))


def _gae(rewards, values, gamma, lam):
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def _edge_potential_np(states: np.ndarray) -> np.ndarray:
    """Causal potential: deployed capital is valuable when exit-adjusted edge is positive."""
    if states.size == 0:
        return np.zeros(0, dtype=np.float32)
    holding = states[:, IS_HOLDING_IDX].astype(np.float32)
    deployed = np.clip(1.0 - states[:, BUDGET_FRAC_REM_IDX], 0.0, 1.0)
    sell_edge = np.clip(states[:, SELL_EDGE_IDX], -0.35, 0.35).astype(np.float32)
    return holding * deployed * sell_edge


def train(games: list[Game], cfg: PPOConfig | None = None,
          val_games: list | None = None, return_history: bool = False,
          eval_every: int = 5):
    """Train PPO. If return_history=True, also returns a learning-curve dict with
    per-iteration train avg reward, and (every `eval_every` iters) the greedy
    train/val mean realized PnL — so train and val curves can be compared for
    overfitting/undertraining. Default (return_history=False) keeps the old
    (policy, critic) return so existing callers are unaffected."""
    cfg = cfg or PPOConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    policy = PolicyNet(FEATURE_DIM, cfg.hidden, seed=cfg.seed)
    critic = CriticNet(FEATURE_DIM, cfg.hidden, seed=cfg.seed)
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=cfg.lr)

    # learning curves (your ask): per-iteration train avg reward, and periodic
    # train/val realized-PnL so we can see if BOTH rise (healthy) or diverge (overfit).
    history = {"iter": [], "train_reward": [], "train_pnl": [], "val_pnl": []}
    best_val_pnl = -float("inf")
    best_state: tuple[dict, dict] | None = None

    for it_i in range(cfg.iters):
        # ---- collect a batch of rollouts ----
        idx = rng.permutation(len(games))[: min(len(games), cfg.batch_games)]
        bS, bA, bSZ, bOLD, bADV, bRET, bMASK = ([] for _ in range(7))
        ep_rewards = []  # sum-of-rewards (= realized PnL) per game this iteration
        for gi in idx:
            S, NEXT_S, A, SZ, OLD, R, MASK = _rollout(games[gi], policy, rng)
            if len(R) == 0:
                continue
            ep_rewards.append(float(R.sum()))  # raw episode PnL (pre-shaping)
            # Dense causal shaping: credit entering positive edge and exiting negative
            # edge, while preserving raw PnL for reported learning curves.
            traded = np.isin(A, [int(Action.BUY), int(Action.SELL)]).astype(np.float32)
            pyramid_buy = (
                (A == int(Action.BUY))
                & (S[:, IS_HOLDING_IDX] > 0.5)
            ).astype(np.float32)
            deployed = np.clip(1.0 - S[:, BUDGET_FRAC_REM_IDX], 0.0, 1.0)
            pot_now = _edge_potential_np(S)
            pot_next = _edge_potential_np(NEXT_S)
            Rsh = (
                R
                - cfg.trade_cost * traded
                - cfg.pyramid_buy_cost * pyramid_buy * (0.5 + deployed)
                + cfg.edge_potential_coef * (cfg.gamma * pot_next - pot_now)
            )
            with torch.no_grad():
                V = critic.forward(torch.from_numpy(S)).numpy()
            adv, ret = _gae(Rsh, V, cfg.gamma, cfg.lam)
            bS.append(S); bA.append(A); bSZ.append(SZ); bOLD.append(OLD)
            bADV.append(adv); bRET.append(ret); bMASK.append(MASK)
        if not bS:
            continue
        # record the train learning curve (mean episode PnL under current policy)
        history["iter"].append(it_i)
        history["train_reward"].append(round(float(np.mean(ep_rewards)) if ep_rewards else 0.0, 5))
        S = torch.from_numpy(np.concatenate(bS))
        A = torch.from_numpy(np.concatenate(bA))
        SZ = torch.from_numpy(np.concatenate(bSZ))
        OLD = torch.from_numpy(np.concatenate(bOLD))
        ADV = torch.from_numpy(np.concatenate(bADV).astype(np.float32))
        RET = torch.from_numpy(np.concatenate(bRET).astype(np.float32))
        MASK = torch.from_numpy(np.concatenate(bMASK))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
        is_buy = (A == int(Action.BUY))

        # ---- PPO epochs ----
        for _ in range(cfg.epochs):
            a_logits, s_logits = policy.forward(S, MASK)
            a_logp_all = F.log_softmax(a_logits, dim=-1)
            logp = a_logp_all.gather(1, A.unsqueeze(1)).squeeze(1)
            s_logp_all = F.log_softmax(s_logits, dim=-1)
            logp = logp + torch.where(
                is_buy, s_logp_all.gather(1, SZ.unsqueeze(1)).squeeze(1), torch.zeros_like(logp)
            )
            ratio = torch.exp(logp - OLD)
            s1 = ratio * ADV
            s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ADV
            policy_loss = -torch.min(s1, s2).mean()
            # entropy (action head) for exploration
            a_probs = a_logp_all.exp()
            action_entropy = -(a_probs * a_logp_all).sum(-1).mean()
            s_probs = s_logp_all.exp()
            buy_legal = MASK[:, int(Action.BUY)]
            if torch.any(buy_legal):
                size_entropy = -(s_probs[buy_legal] * s_logp_all[buy_legal]).sum(-1).mean()
            else:
                size_entropy = torch.zeros((), dtype=S.dtype, device=S.device)
            value = critic.forward(S)
            value_loss = F.mse_loss(value, RET)
            # warmup_iters=0 -> no warmup, full entry-prior throughout (v0 behavior).
            # warmup_iters>0 -> anneal the crutch to 0 over that many iters.
            if cfg.entry_prior_warmup_iters <= 0:
                entry_prior_coef = cfg.entry_prior_coef
            else:
                warmup_left = max(0.0, 1.0 - (it_i / cfg.entry_prior_warmup_iters))
                entry_prior_coef = cfg.entry_prior_coef * warmup_left
            if entry_prior_coef > 0.0:
                positive_edge_entry = (
                    (S[:, IS_HOLDING_IDX] < 0.5)
                    & (S[:, IMPLIED_PROB_IDX] >= 0.35)
                    & (S[:, IMPLIED_PROB_IDX] <= 0.85)
                    & (S[:, BUY_EDGE_IDX] >= 0.02)
                    & MASK[:, int(Action.BUY)]
                )
            else:
                positive_edge_entry = torch.zeros_like(A, dtype=torch.bool)
            if torch.any(positive_edge_entry):
                entry_target = torch.full(
                    (int(positive_edge_entry.sum().item()),),
                    int(Action.BUY),
                    dtype=torch.long,
                    device=S.device,
                )
                entry_prior_loss = F.cross_entropy(a_logits[positive_edge_entry], entry_target)
            else:
                entry_prior_loss = torch.zeros((), dtype=S.dtype, device=S.device)
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                + entry_prior_coef * entry_prior_loss
                - cfg.entropy_coef * action_entropy
                - cfg.size_entropy_coef * size_entropy
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(critic.parameters()), 1.0)
            opt.step()

        # periodic greedy train/val PnL for the overfitting view (healthy = both rise)
        if return_history and (it_i % eval_every == 0 or it_i == cfg.iters - 1):
            tp = _greedy_mean_pnl(games, policy)
            vp = _greedy_mean_pnl(val_games, policy) if val_games else None
            history["train_pnl"].append((it_i, tp))
            if vp is not None:
                history["val_pnl"].append((it_i, vp))
                val_trades = _greedy_avg_trades(val_games, policy)
                if vp > best_val_pnl and (vp > 0.0 or val_trades > 0.05):
                    best_val_pnl = vp
                    best_state = (
                        copy.deepcopy(policy.state_dict()),
                        copy.deepcopy(critic.state_dict()),
                    )

    if best_state is not None:
        policy.load_state_dict(best_state[0])
        critic.load_state_dict(best_state[1])
    if return_history:
        return policy, critic, history
    return policy, critic


def _greedy_mean_pnl(games, policy) -> float:
    """Mean realized PnL of the greedy policy over `games` (cheap; no gradients)."""
    if not games:
        return 0.0
    tot = 0.0
    for g in games:
        env = BacktestEnv(g)
        while True:
            x = feature_construction(g, env.t, env.pos)[None, :]
            a_probs, s_probs, _ = policy.policy(x, env.action_mask()[None, :])
            a = int(a_probs[0].argmax()); sz = float(SIZE_BUCKETS[int(s_probs[0].argmax())])
            if env.step(a, sz).done:
                break
        tot += env.result().realized_pnl
    return round(tot / len(games), 5)


def _greedy_avg_trades(games, policy) -> float:
    """Mean greedy BUY/SELL count; used only to avoid checkpointing all-skip ties."""
    if not games:
        return 0.0
    tot = 0.0
    for g in games:
        env = BacktestEnv(g)
        while True:
            x = feature_construction(g, env.t, env.pos)[None, :]
            a_probs, s_probs, _ = policy.policy(x, env.action_mask()[None, :])
            a = int(a_probs[0].argmax()); sz = float(SIZE_BUCKETS[int(s_probs[0].argmax())])
            if env.step(a, sz).done:
                break
        res = env.result()
        tot += res.n_buys + res.n_sells
    return tot / len(games)
