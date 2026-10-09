#!/usr/bin/env bash
# Recipe boot: stops any server on 8096, waits for a CLEAN teardown (processes gone,
# GPUs back to idle), then boots this recipe's config and waits for ready.
# Self-locating: run from anywhere as <recipe>/start_server.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="$(basename "$DIR")"
CFG="$DIR/config.json"
LOG="$DIR/logs/server.log"
mkdir -p "$DIR/logs"

echo "== $NAME :: stopping any current server =="
sudo -n fuser -k -TERM 8096/tcp 2>/dev/null || true
sudo -n pkill -TERM -f "[s]erve/server.py" 2>/dev/null || true
sudo -n pkill -TERM -f "[b]uild/strata" 2>/dev/null || true
sudo -n pkill -TERM -f "[b]in/strata" 2>/dev/null || true

# graceful window, then hard kill
for i in $(seq 1 15); do
  if ! pgrep -f "[b]uild/strata" >/dev/null 2>&1 && ! pgrep -f "[b]in/strata" >/dev/null 2>&1 && ! pgrep -f "[s]erve/server.py" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
sudo -n pkill -9 -f "[b]uild/strata" 2>/dev/null || true
sudo -n pkill -9 -f "[b]in/strata" 2>/dev/null || true
sudo -n pkill -9 -f "[s]erve/server.py" 2>/dev/null || true

# teardown gate: processes gone AND both GPUs back to idle before any boot
for i in $(seq 1 90); do
  PROCS=$( { pgrep -f "[b]uild/strata" || true; pgrep -f "[b]in/strata" || true; pgrep -f "[s]erve/server.py" || true; } 2>/dev/null | wc -l )
  MAXMEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -1)
  if [ "$PROCS" -eq 0 ] && [ -n "$MAXMEM" ] && [ "$MAXMEM" -lt 600 ]; then
    echo "teardown clean (gpu max used ${MAXMEM} MiB)"
    break
  fi
  if [ "$i" -eq 90 ]; then
    echo "TEARDOWN GATE FAILED: procs=$PROCS gpu_max=${MAXMEM:-?} MiB"
    exit 1
  fi
  sleep 2
done
sleep 3

# watchdog contract + bounded logs: publish the active config pointer, touch the restart marker
# (the stall watchdog honors both), and rotate the engine log - append-mode across boots - keeping
# exactly one previous generation.
WS_HOME=/home/mhd67/strata-serving-lab
if [ -d "$WS_HOME/scratch" ]; then
  echo "$CFG" > "$WS_HOME/scratch/.current-campaign-config"
  touch "$WS_HOME/.last-serving-restart"
fi
ELOG="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("log",""))' "$CFG" 2>/dev/null)" || true
[ -z "$ELOG" ] && ELOG="$DIR/logs/engine.log"
[ -f "$ELOG" ] && mv -f "$ELOG" "$ELOG.1"

: > "$LOG"
sudo -n bash -c "ulimit -l unlimited && cd /home/mhd67/strata-serving-lab/upstream && exec nohup taskset -c 0-7,16-23 python3 serve/server.py --engine strata --config $CFG --port 8096 > $LOG 2>&1 < /dev/null &"

for i in $(seq 1 240); do
  if curl -s --max-time 2 http://127.0.0.1:8096/health | grep -q '"loaded": true'; then
    echo "READY in ${i}s"
    curl -s --max-time 3 http://127.0.0.1:8096/health; echo
    exit 0
  fi
  sleep 1
done
echo "BOOT TIMEOUT"
curl -s --max-time 3 http://127.0.0.1:8096/health || true
exit 1
