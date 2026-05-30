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


from .prompt import build_task  # the Codex agent prompt lives in src/prompt.py


REPO_ROOT = SRC_DIR.parent  # .autoresearch_nba (dedicated git repo)


@dataclass
class Proposal:
    hypothesis: str
    files: dict[str, str] = field(default_factory=dict)  # changed file -> new content
    applied_in_place: bool = True  # the agent already wrote the edits to disk
    commit_sha: str = ""           # the commit the agent made (if any)
    doc_path: str = ""             # EXPERIMENTS/iter-NN.md
    usage: dict = field(default_factory=dict)  # token usage from the codex call
    cost_usd: float = 0.0          # estimated $ cost of this codex call


# Per-1M-token USD pricing for the optimizer model (update if pricing changes).
# Cached input tokens are billed at a reduced rate; reasoning tokens bill as output.
MODEL_PRICING = {
    # model: (input, cached_input, output) per 1M tokens
    "gpt-5.5": (1.25, 0.125, 10.0),
    "gpt-5": (1.25, 0.125, 10.0),
    "gpt-4.1": (2.0, 0.5, 8.0),
}


def _estimate_cost(model: str, usage: dict) -> float:
    inp = usage.get("input_tokens", 0) or 0
    cached = usage.get("cached_input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    reasoning = usage.get("reasoning_output_tokens", 0) or 0
    p_in, p_cached, p_out = MODEL_PRICING.get(model, MODEL_PRICING["gpt-5.5"])
    uncached_in = max(0, inp - cached)
    return round(
        (uncached_in * p_in + cached * p_cached + (out + reasoning) * p_out) / 1e6, 6
    )


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

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True).stdout.strip()

    def propose(self, iteration: int, last_metrics: dict, best_metrics: dict,
                diagnostics: dict | None = None, prior_error: str | None = None) -> Proposal:
        task = build_task(iteration, last_metrics, best_metrics, diagnostics)
        if prior_error:
            # retry within the same iteration: show the agent its own failure so it
            # FIXES it rather than us wasting the iteration on a broken edit.
            task += (
                "\n\n=== YOUR PREVIOUS ATTEMPT THIS ITERATION FAILED THE HARNESS ===\n"
                f"{prior_error}\n"
                "Your current edits are STILL ON DISK. Diagnose and FIX the problem in "
                "feature_construction.py / training.py, run your self-check to confirm "
                "it works, then re-commit. Do not start a different experiment — repair "
                "this one so it passes."
            )
        head_before = self._git("rev-parse", "HEAD")
        before = _hash_files()
        env = dict(os.environ)
        env["PATH"] = os.path.expanduser("~/.npm-global/bin") + ":" + env.get("PATH", "")
        cmd = [
            self.codex, "exec",
            "-m", self.model,
            "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
            # DATA-LEAK GUARD: these games are real 2026 NBA games — any internet access
            # could reveal who won and let the agent hardcode outcomes. Disable ALL
            # web/browser/computer-use tools so the agent works offline from code+train data.
            "-c", "tools.web_search=false",
            "--disable", "standalone_web_search",
            "--disable", "web_search_cached",
            "--disable", "web_search_request",
            "--disable", "browser_use",
            "--disable", "browser_use_external",
            "--disable", "computer_use",
            "--disable", "network_proxy",
            "-C", str(SRC_DIR),
            "-s", "workspace-write",  # write files but the sandbox blocks network for shell cmds
            "--skip-git-repo-check",
            "--json",  # emit JSONL events incl. turn.completed usage
            "-",
        ]
        proc = subprocess.run(cmd, input=task, capture_output=True, text=True,
                              env=env, timeout=self.timeout_s)

        # Parse token usage from the last turn.completed JSON event -> estimate cost.
        usage: dict = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict):
                u = ev["usage"]
                # accumulate across turns (the agent may take several)
                for k in ("input_tokens", "cached_input_tokens", "output_tokens",
                          "reasoning_output_tokens"):
                    usage[k] = usage.get(k, 0) + int(u.get(k, 0) or 0)
        cost_usd = _estimate_cost(self.model, usage)

        after = _hash_files()
        changed = {f: (SRC_DIR / f).read_text()
                   for f in EDITABLE_FILES if before.get(f) != after.get(f)}
        head_after = self._git("rev-parse", "HEAD")
        commit_sha = head_after if head_after != head_before else ""

        doc = REPO_ROOT / "EXPERIMENTS" / f"iter-{iteration:02d}.md"
        hypothesis = ""
        if doc.exists():
            for line in doc.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    hypothesis = line.strip()[:500]
                    break
        if not hypothesis and commit_sha:
            hypothesis = self._git("log", "-1", "--format=%s")

        return Proposal(hypothesis=hypothesis or "(no hypothesis)", files=changed,
                        applied_in_place=True, commit_sha=commit_sha,
                        doc_path=str(doc) if doc.exists() else "",
                        usage=usage, cost_usd=cost_usd)
