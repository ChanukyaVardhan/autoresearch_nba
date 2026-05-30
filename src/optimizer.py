"""CodeOptimizer — the autoresearch agent that rewrites feature_construction.py and
training.py. Backend: the **OpenAI Codex CLI** (`codex exec`) as a REASONING AGENT
(DESIGN s7), not a one-shot completion.

Each iteration, the Codex agent runs non-interactively against the src/ directory:
it reasons over the diagnostics, edits the two editable files in place (sandboxed to
workspace-write), and writes a short hypothesis. The harness then leakage-checks,
trains, and grades the result — keeping or reverting the agent's edits.

Auth: the Codex CLI uses its own stored credentials (`codex login` / API key in
~/.codex). No key is passed from here.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

EDITABLE_FILES = ("feature_construction.py", "training.py")
SRC_DIR = Path(__file__).resolve().parent

# Resolve the codex binary (npm-global install isn't always on PATH in subprocesses).
_CODEX_CANDIDATES = [
    os.path.expanduser("~/.npm-global/bin/codex"),
    "/usr/local/bin/codex",
    "/opt/homebrew/bin/codex",
    "codex",
]


def _codex_bin() -> str:
    for c in _CODEX_CANDIDATES:
        if c == "codex" or os.path.exists(c):
            return c
    return "codex"


TASK_TEMPLATE = """You are an autoresearch agent improving a sequential-RL NBA live-trading model.

WORKING DIRECTORY: this is the src/ package. You may edit ONLY these two files:
  - feature_construction.py  (the causal state encoder)
  - training.py              (the PyTorch PPO trainer)
Do NOT edit any other file (backtest.py, evaluate.py, networks.py, game.py, etc. are
the fixed/trusted harness and editing them is cheating the benchmark).

HARD CONSTRAINTS (a leakage/validation harness enforces these — violations are rejected):
- feature_construction MUST stay pure and STRICTLY CAUSAL: read only game data with
  wall_clock <= t via the Game accessors. NEVER read game-end player_stats or the
  settlement (lookahead = cheating; a prefix-invariance test will reject it).
- FEATURE_DIM must remain a fixed constant; the returned vector length must equal it
  for every call and every game (the networks depend on it). If you add/remove a
  feature, update FEATURE_NAMES/FEATURE_DIM consistently.
- Output float32, finite (no NaN/inf). Keep function/class signatures intact.

GOAL: maximize the validation 'headline' (a Sharpe-like risk-adjusted return).
The honest baseline to beat: buy-favorite-hold ~ +0.0015/game on train.

Make ONE focused, diagnostics-motivated change this iteration. Use the diagnostics
below to decide WHAT to change (action_mix/pct_games_no_trade = is it collapsing to
always-skip?; feature_importance = which features matter vs are dead; value_corr =
is the critic learning?; pnl_by_margin_regime/favorite_vs_dog = where P&L leaks).

After editing, write a one-line hypothesis describing your change to the file
HYPOTHESIS.txt in the working directory (overwrite it).

=== latest validation metrics ===
{metrics}

=== best-so-far ===
{best}

=== diagnostics ===
{diagnostics}
"""


@dataclass
class Proposal:
    hypothesis: str
    files: dict[str, str] = field(default_factory=dict)  # changed file -> new content
    applied_in_place: bool = True  # the agent already wrote the edits to disk


def _hash_files() -> dict[str, str]:
    out = {}
    for f in EDITABLE_FILES:
        p = SRC_DIR / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""
    return out


class CodeOptimizer:
    """Codex-CLI-backed reasoning agent. Edits files in place per iteration."""

    def __init__(self, model: str = "gpt-5.5", reasoning_effort: str = "medium",
                 timeout_s: int = 1800):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self.codex = _codex_bin()

    def propose(self, current_files: dict[str, str], last_metrics: dict,
                best_metrics: dict, diagnostics: dict | None = None) -> Proposal:
        task = TASK_TEMPLATE.format(
            metrics=json.dumps(last_metrics, indent=1),
            best=json.dumps(best_metrics, indent=1),
            diagnostics=json.dumps(diagnostics or {}, indent=1),
        )
        hyp_file = SRC_DIR / "HYPOTHESIS.txt"
        if hyp_file.exists():
            hyp_file.unlink()

        before = _hash_files()
        env = dict(os.environ)
        env["PATH"] = os.path.expanduser("~/.npm-global/bin") + ":" + env.get("PATH", "")
        cmd = [
            self.codex, "exec",
            "-m", self.model,
            "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
            "-C", str(SRC_DIR),
            "-s", "workspace-write",
            "--skip-git-repo-check",
            "-",  # read prompt from stdin
        ]
        proc = subprocess.run(
            cmd, input=task, capture_output=True, text=True,
            env=env, timeout=self.timeout_s,
        )
        after = _hash_files()
        changed = {f: (SRC_DIR / f).read_text()
                   for f in EDITABLE_FILES if before.get(f) != after.get(f)}

        hypothesis = ""
        if hyp_file.exists():
            hypothesis = hyp_file.read_text().strip()[:500]
        if not hypothesis:
            # fall back to the agent's last stdout line
            tail = [l for l in proc.stdout.splitlines() if l.strip()]
            hypothesis = (tail[-1][:500] if tail else "(no hypothesis)")

        return Proposal(hypothesis=hypothesis, files=changed, applied_in_place=True)
