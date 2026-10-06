#!/usr/bin/env bash
# recipe_lifecycle_test.sh - the recipe boot/stop discipline test (the fast-kill GPU-fault
# class as a regression gate). Cycles <recipe>/start_server.sh -> stop_server.sh -> start_server.sh
# and asserts: READY on both boots, health identity, teardown-gate line + procs==0 after stop,
# and ZERO new Xid entries across the transitions.
#
#   bash tools/acceptance/recipe_lifecycle_test.sh --recipe DIR [--expect-commit SHA7] [--health-port 8096]
#
# Exit: 0 pass (prints 'checks run: N'); 1 any failure; 3 refusal (preconditions unmet).
set -u
RECIPE=""
EXPECT_COMMIT=""
PORT=8096

while [ $# -gt 0 ]; do
  case "$1" in
    --recipe) RECIPE="$2"; shift ;;
    --expect-commit) EXPECT_COMMIT="$2"; shift ;;
    --health-port) PORT="$2"; shift ;;
    *) echo "recipe_lifecycle_test: unknown arg $1" >&2; exit 3 ;;
  esac
  shift
done

# ---------------------------------------------------------------- preconditions (refusal, never vacuous)
[ -n "$RECIPE" ] || { echo "REFUSED: --recipe DIR required (e.g. /path/to/recipes/<name>)"; exit 3; }
[ -d "$RECIPE" ] || { echo "REFUSED: recipe dir '$RECIPE' not found"; exit 3; }
for f in start_server.sh stop_server.sh config.json; do
  [ -f "$RECIPE/$f" ] || { echo "REFUSED: $RECIPE/$f missing"; exit 3; }
done

HOST="127.0.0.1:$PORT"
TOTAL=0
FAILS=0
want() { TOTAL=$((TOTAL + 1)); if [ "$2" = "0" ]; then echo "  [PASS] $1"; else echo "  [FAIL] $1"; FAILS=$((FAILS + 1)); fi; }
xid_now() { (journalctl -k -b --no-pager 2>/dev/null || sudo -n journalctl -k -b --no-pager 2>/dev/null) | grep -ci "xid" || true; }
health() { curl -s -m 4 "http://$HOST/health"; }

X0="$(xid_now)"
echo "xid entries before: $X0"
ELOG="$RECIPE/logs/engine.log"
E0=0; [ -f "$ELOG" ] && E0="$(wc -l < "$ELOG")"
echo "engine.log lines before: $E0"

echo "=== boot 1 ==="
OUT="$(bash "$RECIPE/start_server.sh" 2>&1)"; RC=$?
echo "$OUT" | tail -4
want "boot1 start_server.sh rc=0" $RC
echo "$OUT" | grep -q "READY in"; want "boot1 reached READY" $?
H1="$(health)"; echo "health1: $H1"
echo "$H1" | grep -q '"loaded": true'; want "boot1 /health loaded" $?
if [ -n "$EXPECT_COMMIT" ]; then
  echo "$H1" | grep -q "\"commit\": \"$EXPECT_COMMIT\""; want "boot1 health commit == $EXPECT_COMMIT" $?
fi

echo "=== stop ==="
OUT="$(bash "$RECIPE/stop_server.sh" 2>&1)"; RC=$?
echo "$OUT" | tail -4
want "stop_server.sh rc=0" $RC
echo "$OUT" | grep -q "teardown clean"; want "stop printed teardown clean" $?
P=$( { pgrep -f "[b]uild/strata" || true; pgrep -f "[b]in/strata" || true; pgrep -f "[s]erve/server.py" || true; } 2>/dev/null | wc -l )
[ "$P" -eq 0 ]; want "procs==0 after stop (got $P)" $?

echo "=== boot 2 ==="
OUT="$(bash "$RECIPE/start_server.sh" 2>&1)"; RC=$?
echo "$OUT" | tail -4
want "boot2 start_server.sh rc=0" $RC
echo "$OUT" | grep -q "READY in"; want "boot2 reached READY" $?
H2="$(health)"; echo "health2: $H2"
echo "$H2" | grep -q '"loaded": true'; want "boot2 /health loaded" $?

X1="$(xid_now)"
echo "xid entries after: $X1"
[ "$X1" -eq "$X0" ]; want "zero new Xid entries across the cycle ($X0 -> $X1)" $?

E1=0; [ -f "$ELOG" ] && E1="$(wc -l < "$ELOG")"
echo "engine.log lines after: $E1 (before $E0)"

echo "checks run: $TOTAL"
if [ "$FAILS" -gt 0 ]; then
  echo "recipe_lifecycle_test: FAIL ($FAILS of $TOTAL)"
  exit 1
fi
echo "recipe_lifecycle_test: PASS"
exit 0
