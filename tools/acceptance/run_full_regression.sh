#!/usr/bin/env bash
# tools/acceptance/run_full_regression.sh - Strata AGX unified end-to-end regression runner.
#
#   Stage 1 (parity):   kernel parity + unit battery (GPU tests need the cards free).
#   Stage 2 (serving):  live HTTP serving on port 8096, then the 4-stream checks.
#
# The boot is NOT bundled: point BOOT_SCRIPT at your server boot script (it must be
# TERM-first and wait for both GPUs to go idle before launching - the recipe scripts in
# docs/SERVING_CONFIG.md section 6 are the reference form).
#
#   BOOT_SCRIPT=/path/to/start_server.sh bash tools/acceptance/run_full_regression.sh
#
# Exit: 0 all stages ran and passed; 1 a stage failed; 3 refusal (no boot script).
set -eu

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DIR"

if [ -z "${BOOT_SCRIPT:-}" ]; then
  echo "REFUSED: BOOT_SCRIPT is not set - stage 2 needs a server boot script."
  echo "  export the absolute path of your recipe's start script, e.g.:"
  echo "    BOOT_SCRIPT=/path/to/recipes/<name>/start_server.sh bash tools/acceptance/run_full_regression.sh"
  exit 3
fi
if [ ! -x "$BOOT_SCRIPT" ]; then
  echo "REFUSED: BOOT_SCRIPT '$BOOT_SCRIPT' is not an executable file"
  exit 3
fi

echo "=================================================================================="
echo "  STRATA AGX: UNIFIED END-TO-END REGRESSION & VALIDATION HARNESS"
echo "=================================================================================="

echo "[1/4] Stopping existing server instances (TERM-first; wait for idle GPUs)..."
sudo -n fuser -k -TERM 8096/tcp 2>/dev/null || true
sudo -n pkill -TERM -f '[b]uild/strata' 2>/dev/null || true
sudo -n pkill -TERM -f '[s]erve/server.py' 2>/dev/null || true
for i in $(seq 1 15); do
  { pgrep -f '[b]uild/strata' || pgrep -f '[s]erve/server.py'; } >/dev/null 2>&1 || break
  sleep 2
done
sudo -n pkill -9 -f '[b]uild/strata' 2>/dev/null || true
sudo -n pkill -9 -f '[s]erve/server.py' 2>/dev/null || true
for i in $(seq 1 60); do
  PROCS=$( { pgrep -f '[b]uild/strata' || true; pgrep -f '[s]erve/server.py' || true; } 2>/dev/null | wc -l )
  MAXMEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -1)
  if [ "$PROCS" -eq 0 ] && [ -n "$MAXMEM" ] && [ "$MAXMEM" -lt 600 ]; then
    echo "      teardown clean (gpu max used ${MAXMEM} MiB)"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "      TEARDOWN GATE FAILED: procs=$PROCS gpu_max=${MAXMEM:-?} MiB"
    exit 1
  fi
  sleep 2
done

echo "[2/4] Executing Stage 1: Kernel Parity & Unit Test Battery..."
python3 tools/acceptance/phase5_regression_suite.py --skip-serving

echo "[3/4] Booting the server via BOOT_SCRIPT=$BOOT_SCRIPT ..."
bash "$BOOT_SCRIPT"
if ! curl -s -m 5 http://127.0.0.1:8096/health | grep -q '"loaded": true'; then
  echo "      SERVER NOT LOADED after boot - aborting"
  curl -s -m 5 http://127.0.0.1:8096/health || true
  exit 1
fi

echo "[4/4] Executing Stage 2: Live HTTP Serving & 4-Stream Concurrency..."
python3 tools/acceptance/phase5_regression_suite.py --skip-parity

echo "=================================================================================="
echo "  STRATA AGX UNIFIED REGRESSION COMPLETE: ALL GATES VERIFIED AND PASSED!"
echo "=================================================================================="
