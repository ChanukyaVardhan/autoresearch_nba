"""Autoresearch loop driver (DESIGN s7).

Per iteration:
  1. Codex proposes an edit to feature_construction.py / training.py (hypothesis + files).
  2. Apply edit to a sandbox copy; run leakage suite. If it fails -> revert, log.
  3. Build features (train) -> PPO train -> backtest on VALIDATION -> metrics.
  4. Keep if headline improved over best-so-far, else revert. Log either way.
  5. Repeat until iters/plateau. Best frozen.
  6. ONE-TIME eval on holdout (separate entrypoint: run_eval.py).

Deterministic harness: fixed seed; only the edited code changes between iterations.

This module ORCHESTRATES but does NOT auto-run on import. Use run_loop().
"""
from __future__ import annotations

import importlib
import shutil
import sys
import time
from pathlib import Path

from .experiment_log import ExperimentLog, LogEntry
from .game import load_split
from .leakage import run_leakage_suite
from .optimizer import EDITABLE_FILES, CodeOptimizer

import subprocess

SRC = Path(__file__).resolve().parent
WORK = SRC.parent              # .autoresearch_nba (the dedicated git repo)
ARTIFACTS = WORK / "artifacts"


def _read(name: str) -> str:
    return (SRC / name).read_text()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(WORK), *args],
                          capture_output=True, text=True).stdout.strip()


def _revert_to(sha: str) -> None:
    """Hard-restore the working tree to a known-good commit (drops the bad edit but
    keeps it in history as an experiment record)."""
    _git("reset", "--hard", sha)


def _append_result(doc_path: str, kept: bool, metrics: dict, note: str) -> None:
    if not doc_path:
        return
    p = Path(doc_path)
    if not p.exists():
        return
    res = [f"\n## Result", f"- verdict: **{'KEPT' if kept else 'REVERTED'}** — {note}"]
    for k in ("headline", "mean_return", "sharpe", "win_rate", "avg_trades", "total_pnl"):
        if k in metrics:
            res.append(f"- {k}: {metrics[k]}")
    p.write_text(p.read_text() + "\n".join(res) + "\n")


def _reload_editable():
    """Reimport the editable modules so the harness uses the latest code."""
    for mod in ("feature_construction", "training", "evaluate", "leakage"):
        m = f"src.{mod}"
        if m in sys.modules:
            importlib.reload(sys.modules[m])


def _train_and_validate(train_games, val_games, seed: int, with_report: bool = False):
    _reload_editable()
    from .training import PPOConfig, train  # reimported
    from .evaluate import evaluate, score
    policy, critic = train(train_games, PPOConfig(seed=seed))
    metrics = evaluate(val_games, policy)
    report = None
    if with_report:
        from .observability import build_report
        report = build_report(val_games, policy, critic, metrics.headline, score)
    return policy, critic, metrics, report


def run_loop(data_dir: Path, iters: int = 20, seed: int = 0,
             model: str = "gpt-5.5", reasoning: str = "medium",
             trace: bool = True) -> None:
    from .tracing import TraceLogger
    log = ExperimentLog(ARTIFACTS / "experiment_log.jsonl")
    tracer = TraceLogger(run_name=f"autoresearch-{model}-seed{seed}", enabled=trace)
    if tracer.active:
        print(f"tracing -> {tracer.endpoint} (open Raindrop Workshop / OTLP UI)")
    train_games = load_split(data_dir, "train")
    val_games = load_split(data_dir, "val")
    print(f"loaded train={len(train_games)} val={len(val_games)}")

    from dataclasses import asdict
    with tracer.span("run", {"iters": iters, "seed": seed, "model": model,
                             "n_train": len(train_games), "n_val": len(val_games)}):
        # baseline (iteration 0): current code, no edit. Full diagnostic report so the
        # optimizer can reason about HOW to improve, not just the headline.
        with tracer.span("iteration", {"iter": 0, "kind": "baseline"}) as sp:
            ok, msg = run_leakage_suite(train_games)
            if not ok:
                raise RuntimeError(f"baseline leakage failure: {msg}")
            _, _, base_m, base_report = _train_and_validate(train_games, val_games, seed, with_report=True)
            best_headline = base_m.headline
            best_files = {f: _read(f) for f in EDITABLE_FILES}
            cur_report = base_report
            sp.set_attrs(base_m.__dict__, prefix="metrics")
            sp.set(best_headline=best_headline, kept=True, verdict="baseline")
        log.append(LogEntry(0, "baseline", log.file_hashes(best_files),
                            base_m.__dict__, True, best_headline, 0.0, "baseline"))
        print(f"[iter 0 baseline] headline={best_headline:.4f}")

        opt = CodeOptimizer(model=model, reasoning_effort=reasoning)
        best_sha = _git("rev-parse", "HEAD")  # last KEPT commit (good tree)
        for it in range(1, iters + 1):
            t0 = time.time()
            with tracer.span("iteration", {"iter": it}) as sp:
                pre_sha = _git("rev-parse", "HEAD")
                with tracer.span("codex.propose", {"model": model}) as psp:
                    # The Codex agent edits files + writes EXPERIMENTS/iter-NN.md + commits.
                    prop = opt.propose(it, base_m.__dict__, {"headline": best_headline},
                                       diagnostics=asdict(cur_report) if cur_report else {})
                    psp.set(hypothesis=prop.hypothesis, commit=prop.commit_sha[:8],
                            files_changed=list(prop.files.keys()))
                sp.set(hypothesis=prop.hypothesis, commit=prop.commit_sha[:8])
                if not prop.files:
                    sp.set(verdict="no_change", kept=False)
                    log.append(LogEntry(it, prop.hypothesis, {}, {}, False, best_headline,
                                        time.time() - t0, "agent made no file changes"))
                    print(f"[iter {it}] agent made no file changes")
                    continue

                kept = False; note = ""; metrics = {}
                try:
                    with tracer.span("leakage_check") as lsp:
                        ok, msg = run_leakage_suite(train_games)
                        lsp.set(passed=ok, detail=msg)
                    if not ok:
                        note = f"REJECTED (leakage): {msg}"
                    else:
                        with tracer.span("train_and_validate") as tsp:
                            _, _, m, report = _train_and_validate(train_games, val_games, seed, with_report=True)
                            tsp.set_attrs(m.__dict__, prefix="metrics")
                            if report:
                                tsp.set_attrs(asdict(report), prefix="diag")
                        metrics = m.__dict__
                        if m.headline > best_headline:
                            best_headline = m.headline; best_sha = prop.commit_sha or best_sha
                            kept = True; note = f"KEPT (headline {m.headline:.4f})"
                            base_m = m; cur_report = report
                        else:
                            note = f"reverted (headline {m.headline:.4f} <= {best_headline:.4f})"
                except Exception as e:
                    note = f"REJECTED (error): {type(e).__name__}: {e}"

                if kept:
                    _append_result(prop.doc_path, kept, metrics, note)
                    _git("add", "-A"); _git("commit", "-q", "-m", f"iter {it}: result KEPT")
                else:
                    # Capture the proposal doc (so the experiment record survives the
                    # revert), restore the good tree, then re-write the doc with its
                    # Result and commit it. The bad code edit is dropped; the record stays.
                    from pathlib import Path as _P
                    doc_txt = _P(prop.doc_path).read_text() if prop.doc_path and _P(prop.doc_path).exists() else ""
                    _revert_to(pre_sha if pre_sha else best_sha)
                    if doc_txt and prop.doc_path:
                        _P(prop.doc_path).parent.mkdir(parents=True, exist_ok=True)
                        _P(prop.doc_path).write_text(doc_txt)
                        _append_result(prop.doc_path, kept, metrics, note)
                    _git("add", "-A"); _git("commit", "-q", "-m", f"iter {it}: REVERTED — {note}")

                sp.set(verdict="kept" if kept else "reverted", kept=kept,
                       best_headline=best_headline,
                       headline=metrics.get("headline", 0.0) if metrics else 0.0)
                log.append(LogEntry(it, prop.hypothesis, {"commit": prop.commit_sha},
                                    metrics, kept, best_headline, time.time() - t0, note))
                print(f"[iter {it}] {note} | {prop.commit_sha[:8]} | {prop.hypothesis[:55]}")
    tracer.shutdown()
    (ARTIFACTS / "best_headline.txt").write_text(str(best_headline))
    print(f"DONE. best validation headline={best_headline:.4f} @ {best_sha[:8]}")
