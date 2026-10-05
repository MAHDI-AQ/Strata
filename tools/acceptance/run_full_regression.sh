#!/usr/bin/env bash
# tools/acceptance/run_full_regression.sh - Unified Phase 5 End-to-End Regression Runner
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DIR"

echo "================================================================================"
echo "  STRATA PHASE 5: UNIFIED END-TO-END REGRESSION & VALIDATION HARNESS"
echo "================================================================================"

echo "[1/4] Stopping existing server instances to release 100% GPU VRAM..."
sudo -n fuser -k 8096/tcp 2>/dev/null || true
sudo -n pkill -9 -f '[b]uild/strata' 2>/dev/null || true
sudo -n pkill -9 -f 'serve/server.py' 2>/dev/null || true
sleep 4

echo "[2/4] Executing Stage 1: Kernel Parity & Unit Test Battery..."
python3 tools/acceptance/phase5_regression_suite.py --skip-serving

echo "[3/4] Booting Strata Server on Dual RTX 4090s (Port 8096)..."
bash /home/mhd67/strata-serving-lab/scratch/pump-boot/start_phase4.sh

echo "[4/4] Executing Stage 2: Live HTTP Serving & 4-Stream Concurrency..."
python3 tools/acceptance/phase5_regression_suite.py --skip-parity

echo "================================================================================"
echo "  PHASE 5 UNIFIED REGRESSION COMPLETE: ALL GATES VERIFIED AND PASSED!"
echo "================================================================================"
