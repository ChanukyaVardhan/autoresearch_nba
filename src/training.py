"""training() — PyTorch PPO trainer. *** CODEX-EDITABLE SURFACE ***

Sequential RL over the finite-horizon per-game MDP (DESIGN s5): clipped PPO with a
critic baseline + GAE, entropy bonus, normalized advantages. Deterministic given
`seed`. Proper autograd (torch), so the critic actually learns.

Codex may edit hyperparameters, network sizes, reward shaping, advantage handling.
It may NOT edit backtest.py (trusted simulator) or read end-state data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))  # use available CPU cores

from .backtest import BacktestEnv
from .feature_construction import FEATURE_DIM, feature_construction
from .game import Game
from .networks import CriticNet, N_SIZE, PolicyNet, SIZE_BUCKETS
from .types import Action


@dataclass
class PPOConfig:
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.997
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    entropy_coef: float = 0.05   # higher: avoid premature collapse to always-skip
    value_coef: float = 0.5
    entry_prior_coef: float = 0.08
    iters: int = 40
    batch_games: int = 64
    trade_cost: float = 0.0005   # tiny per-trade shaping penalty (curb churn, not kill trading)
    seed: int = 0


IMPLIED_PROB_IDX = 0
EDGE_IDX = 19
IS_HOLDING_IDX = 20


def _rollout(game: Game, policy: PolicyNet, rng: np.random.Generator):
    """One episode under the current (stochastic) policy. Returns arrays."""
    env = BacktestEnv(game)
    S, A, SZ, LOGP, R, MASK = [], [], [], [], [], []
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
        reward = tr.reward
        if a in (Action.BUY, Action.SELL):
            reward -= 0.0  # base reward; trade cost applied as shaping below via cfg
        S.append(x[0]); A.append(a); SZ.append(sz); LOGP.append(logp)
        R.append(tr.reward); MASK.append(mask)
        if tr.done:
            break
    return (np.array(S, np.float32), np.array(A, np.int64), np.array(SZ, np.int64),
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

    for it_i in range(cfg.iters):
        # ---- collect a batch of rollouts ----
        idx = rng.permutation(len(games))[: min(len(games), cfg.batch_games)]
        bS, bA, bSZ, bOLD, bADV, bRET, bMASK = ([] for _ in range(7))
        ep_rewards = []  # sum-of-rewards (= realized PnL) per game this iteration
        for gi in idx:
            S, A, SZ, OLD, R, MASK = _rollout(games[gi], policy, rng)
            if len(R) == 0:
                continue
            ep_rewards.append(float(R.sum()))  # raw episode PnL (pre-shaping)
            # trade-cost shaping: penalize BUY/SELL steps to curb churn
            traded = np.isin(A, [int(Action.BUY), int(Action.SELL)]).astype(np.float32)
            Rsh = R - cfg.trade_cost * traded
            Rn = (Rsh - Rsh.mean()) / (Rsh.std() + 1e-6)
            with torch.no_grad():
                V = critic.forward(torch.from_numpy(S)).numpy()
            adv, ret = _gae(Rn, V, cfg.gamma, cfg.lam)
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
            entropy = -(a_probs * a_logp_all).sum(-1).mean()
            value = critic.forward(S)
            value_loss = F.mse_loss(value, RET)
            favorite_entry = (
                (S[:, IS_HOLDING_IDX] < 0.5)
                & (S[:, IMPLIED_PROB_IDX] >= 0.52)
                & (S[:, EDGE_IDX] >= -0.03)
                & MASK[:, int(Action.BUY)]
            )
            if torch.any(favorite_entry):
                entry_target = torch.full(
                    (int(favorite_entry.sum().item()),),
                    int(Action.BUY),
                    dtype=torch.long,
                    device=S.device,
                )
                size_target = torch.full(
                    (int(favorite_entry.sum().item()),),
                    N_SIZE - 1,
                    dtype=torch.long,
                    device=S.device,
                )
                entry_prior_loss = (
                    F.cross_entropy(a_logits[favorite_entry], entry_target)
                    + 0.25 * F.cross_entropy(s_logits[favorite_entry], size_target)
                )
            else:
                entry_prior_loss = torch.zeros((), dtype=S.dtype, device=S.device)
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                + cfg.entry_prior_coef * entry_prior_loss
                - cfg.entropy_coef * entropy
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
