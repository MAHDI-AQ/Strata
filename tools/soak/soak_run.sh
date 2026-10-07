#!/bin/bash
# Soak runner: starts the monitor, runs sustained N-agent traffic (agent_sim) for the window,
# stops the monitor, writes a soak summary.  Usage:
#   soak_run.sh <outdir> <minutes> [agents] [ctx_cap] [msg_tokens] [max_tokens]
set -u
OUT=${1:?outdir}
MIN=${2:?minutes}
AGENTS=${3:-3}
CTXCAP=${4:-131072}
MSGT=${5:-2000}
MAXT=${6:-128}
HERE=$(cd "$(dirname "$0")" && pwd)
REC=/srv/lab/recipes/q3-xxs-3x262k
SERVE_LOG=/home/mhd67/strata-serving-lab/serve-8096.log
mkdir -p "$OUT"
rm -f "$OUT/soak.stop"
setsid nohup python3 "$HERE/soak_monitor.py" "$OUT" 300 >"$OUT/monitor.out" 2>&1 &
MON=$!
echo "monitor pid $MON"
python3 "$HERE/agent_sim.py" \
  --url http://127.0.0.1:8096 --model q3-xxs-3x262k \
  --serve-log "$SERVE_LOG" --engine-log "$REC/logs/engine.log" \
  --agents "$AGENTS" --turns 100000 --ctx-cap "$CTXCAP" --msg-tokens "$MSGT" \
  --max-tokens "$MAXT" --minutes "$MIN" \
  --out "$OUT" >"$OUT/sim.out" 2>&1
SIMRC=$?
touch "$OUT/soak.stop"
sleep 5
kill $MON 2>/dev/null || true
{
  echo "soak window: ${MIN} min, agents ${AGENTS}, ctx-cap ${CTXCAP}"
  echo "sim rc=$SIMRC"
  echo "turns recorded: $(wc -l < "$OUT/turns.jsonl" 2>/dev/null || echo 0)"
  echo "health-fail events: $(grep -c HEALTH-FAIL "$OUT/soak-events.log" 2>/dev/null || echo 0)"
  echo "monitor rows: $(wc -l < "$OUT/soak-monitor.csv" 2>/dev/null || echo 0)"
  echo "--- last monitor rows ---"
  tail -5 "$OUT/soak-monitor.csv" 2>/dev/null
  echo "--- sim stop reason ---"
  grep -E "stop_reason" "$OUT/summary.md" 2>/dev/null | head -2
} > "$OUT/soak-summary.txt" 2>&1
cat "$OUT/soak-summary.txt"
