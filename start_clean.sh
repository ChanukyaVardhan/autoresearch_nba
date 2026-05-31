#!/usr/bin/env bash
# Clean-restart the autoresearch system: kill everything, wipe stale dashboard data,
# restore the pinned baseline code, start ONE dashboard + ONE loop.
# Usage:  ./start_clean.sh [iters] [baseline_ref]
#   iters         default 10
#   baseline_ref  git commit/tag to start experiments from (default BASELINE_REF below)
#
# BASELINE_REF is the STANDARD, KNOWN-GOOD commit every experiment starts from. Pinning
# it means runs are reproducible — not "whatever HEAD happens to be". To move the
# baseline forward, bump this after committing a new verified baseline.
BASELINE_REF_DEFAULT="d04304c"
set -uo pipefail
cd "$(dirname "$0")"
ITERS="${1:-10}"
BASELINE_REF="${2:-$BASELINE_REF_DEFAULT}"
echo "==> baseline ref: $BASELINE_REF (experiments start from this commit's code)"

echo "==> 1. killing any existing loops / codex / dashboard"
pkill -9 -f "run_loop.py"      2>/dev/null
pkill -9 -f "codex exec"       2>/dev/null
pkill -9 -f "run_dashboard.py" 2>/dev/null
sleep 2
LEFT=$(ps aux | grep -E "run_loop.py|codex exec|run_dashboard.py" | grep -v grep | wc -l | tr -d ' ')
if [ "$LEFT" != "0" ]; then
  echo "!! $LEFT processes survived — killing again"; pkill -9 -f "run_loop.py|codex exec|run_dashboard.py" 2>/dev/null; sleep 2
fi
echo "    survivors: $(ps aux | grep -E 'run_loop.py|codex exec|run_dashboard.py' | grep -v grep | wc -l | tr -d ' ')"

echo "==> 2. wiping stale dashboard feed + status"
rm -f artifacts/metrics.jsonl artifacts/status.json
mkdir -p artifacts

echo "==> 3. restoring editable files from baseline ref $BASELINE_REF (no Codex drift)"
git checkout "$BASELINE_REF" -- src/feature_construction.py src/training.py 2>/dev/null \
  || { echo "!! could not checkout $BASELINE_REF — aborting"; exit 1; }
echo "    feature_construction.py + training.py reset to $BASELINE_REF"
# make sure no leftover experiment docs pollute a fresh run
rm -f EXPERIMENTS/iter-*.md

echo "==> 4. starting dashboard (http://127.0.0.1:6060)"
nohup python3 run_dashboard.py 6060 > /tmp/claude/dashboard.log 2>&1 &
sleep 3
if curl -fsS -o /dev/null http://127.0.0.1:6060/ 2>/dev/null; then
  echo "    dashboard UP at http://127.0.0.1:6060"
else
  echo "    !! dashboard not responding — check /tmp/claude/dashboard.log"
fi

echo "==> 5. starting autoresearch loop ($ITERS iters, gpt-5.5 medium)"
nohup python3 run_loop.py --iters "$ITERS" --model gpt-5.5 --reasoning medium > /tmp/claude/loop.log 2>&1 &
echo "    loop PID $! — logs: /tmp/claude/loop.log"
echo ""
echo "DONE. One dashboard + one loop running. Watch http://127.0.0.1:6060"
echo "Tail the loop:  tail -f /tmp/claude/loop.log"
