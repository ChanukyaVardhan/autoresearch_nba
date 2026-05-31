#!/usr/bin/env bash
# Clean-restart the autoresearch system: kill everything, wipe stale dashboard data,
# restore the pristine v0 baseline code, start ONE dashboard + ONE loop.
# Usage:  ./start_clean.sh [iters]      (default 10 iterations)
set -uo pipefail
cd "$(dirname "$0")"
ITERS="${1:-10}"

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

echo "==> 3. restoring pristine v0 baseline editable files (no Codex drift)"
# feature_construction from the original baseline commit; training.py at HEAD (has the
# return_history harness contract) but PPOConfig is the v0 baseline (set in code).
git checkout 33786e2 -- src/feature_construction.py 2>/dev/null
git checkout HEAD     -- src/training.py 2>/dev/null
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
