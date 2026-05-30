"""The Codex agent prompt for the autoresearch loop — kept here as a first-class,
editable artifact (tune the agent's behavior without touching optimizer logic).

`build_task(iteration, metrics, best, diagnostics)` returns the full prompt string
piped to `codex exec` each iteration.
"""
from __future__ import annotations

import json

TASK_TEMPLATE = """You are an autoresearch agent improving a sequential-RL NBA live-trading model.
This is iteration {iter} of an autonomous research loop. You have a bash shell and git.

WORKING DIRECTORY is the src/ package (a dedicated git repo rooted one level up at
.autoresearch_nba). You may edit ONLY these two files:
  - feature_construction.py  (the causal state encoder)
  - training.py              (the PyTorch PPO trainer)
Do NOT edit any other file (backtest.py, evaluate.py, networks.py, game.py, etc. are
the fixed/trusted harness; editing them is cheating the benchmark and will be reverted).

HARD CONSTRAINTS (a leakage/validation harness + an independent Claude verifier
enforce these — violations are REJECTED and reverted):
- feature_construction MUST stay pure and STRICTLY CAUSAL: read only game data with
  wall_clock <= t via the Game accessors. NEVER read game-end player_stats or the
  settlement (lookahead = cheating; a prefix-invariance test will reject it).
- FEATURE_DIM must remain a fixed constant; the returned vector length must equal it
  for every call and every game. If you add/remove a feature, update
  FEATURE_NAMES/FEATURE_DIM consistently.
- If you hardcode feature indices in training.py, derive them from FEATURE_NAMES
  (e.g. FEATURE_NAMES.index("edge")) — never hardcode an integer that silently breaks.
- Output float32, finite (no NaN/inf). Keep function/class signatures intact.

DATA ACCESS — STRICT (this is the most important rule):
- You may ONLY read game data under ../data_train/ (the TRAIN split). Inspecting it to
  design better features is encouraged.
- You must NEVER read, open, list, glob, or reference the validation or eval game data.
  It is NOT in data_train/. Do not touch ../data/ (the full set), and do not hardcode
  behavior keyed to specific game tickers, dates, or matchups. Doing so is data leakage
  and the verifier will reject your edit.
- The only validation feedback you get is the AGGREGATE train + validation METRIC
  CURVES shown below — never the underlying val/eval games. Optimize the mechanism,
  not those specific games.

DO NOT REWARD-HACK. The headline is a proxy for a genuinely better trading policy, not
the target to game. Specifically forbidden: peeking at end-state/outcome data;
overfitting to validation by hardcoding to its games; tampering with the simulator,
scorer, or leakage tests; inflating the metric without real edge. Prefer edits that
make the policy learn edge FROM REWARD over hardcoded priors that merely imitate the
baseline. An independent reviewer reads your diff for exactly these tricks.

GOAL: maximize the validation 'headline' (a Sharpe-like risk-adjusted return) BY
GENUINE EDGE. The honest baseline to beat: buy-favorite-hold ~ +0.0015/game on train.

Use the diagnostics below to decide WHAT to change (action_mix/pct_games_no_trade =
collapsing to always-skip?; feature_importance = which features matter vs are dead;
value_corr = is the critic learning?; pnl_by_margin_regime/favorite_vs_dog = where
P&L leaks). Make ONE focused, diagnostics-motivated change.

DO EXACTLY THIS, IN ORDER (you have bash + git):
1. Reason about the diagnostics and decide on ONE focused experiment.
2. Edit feature_construction.py and/or training.py to implement it.
3. Write the proposal doc to ../EXPERIMENTS/iter-{iter:02d}.md with these sections:
   ## Hypothesis  (one sentence: what you expect to improve and why)
   ## Rationale   (which diagnostic motivated this; the mechanism)
   ## Diff summary (bullet list of the concrete code changes you made)
   ## Features    (the FULL current list of feature names your feature_construction
                   produces this iteration, in order — copy FEATURE_NAMES + note any
                   you ADDED or REMOVED vs the previous iteration. This lets us track
                   feature changes experiment-to-experiment.)
   Leave a `## Result` heading at the end — the harness fills it in after grading.
4. Stage and commit from the repo root with:
   `cd .. && git add -A && git commit -m "iter {iter}: <short hypothesis>"`
   (the repo is .autoresearch_nba; committing is expected and isolated.)
Do NOT run training yourself — the harness grades your commit and decides keep/revert.
If the previous iteration's Result shows the metric dropped, propose a DIFFERENT
direction this time rather than repeating it.

=== latest validation metrics ===
{metrics}

=== best-so-far ===
{best}

=== diagnostics ===
{diagnostics}
"""


def build_task(iteration: int, metrics: dict, best: dict, diagnostics: dict | None) -> str:
    return TASK_TEMPLATE.format(
        iter=iteration,
        metrics=json.dumps(metrics, indent=1),
        best=json.dumps(best, indent=1),
        diagnostics=json.dumps(diagnostics or {}, indent=1),
    )
