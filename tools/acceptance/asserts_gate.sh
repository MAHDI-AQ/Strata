#!/usr/bin/env bash
# asserts_gate.sh - the NDEBUG/asserts integrity gate (G11.1a).
#
# The Release build class carries -DNDEBUG, which makes every runtime assert() inert.
# This gate proves a dedicated asserts-enabled build class exists and actually keeps
# assert() alive:
#   1. -DSTRATA_ENABLE_ASSERTS=ON is honored: zero -DNDEBUG in the generated build rules
#   2. the sentinel target (tests/core/asserts_active_check.cpp) compiles - its
#      '#ifdef NDEBUG #error' makes the compile itself the falsifier - runs clean,
#      and ABORTS when the probe env trips it (assert() proven live at runtime)
#   3. the default class (option OFF) still carries -DNDEBUG (shipping parity preserved)
#
# Exit: 0 pass ('checks run: N'); 1 any failure; 3 refusal (toolchain/tree missing).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TMP="${TMPDIR:-/tmp}/strata-asserts-gate.$$"
TOTAL=0
FAILS=0
want() { TOTAL=$((TOTAL + 1)); if [ "$2" = "0" ]; then echo "  [PASS] $1"; else echo "  [FAIL] $1"; FAILS=$((FAILS + 1)); fi; }

[ -d "$ROOT/src" ] || { echo "REFUSED: $ROOT is not the engine tree"; exit 3; }
command -v cmake >/dev/null 2>&1 || { echo "REFUSED: cmake not on PATH"; exit 3; }
# ninja resolution: PATH, then the beside-the-repo tools dir, then the main build's recorded program
NINJA="$(command -v ninja || true)"
if [ -z "$NINJA" ] && [ -x "$ROOT/tools/ninja-bin/ninja" ]; then NINJA="$ROOT/tools/ninja-bin/ninja"; fi
if [ -z "$NINJA" ] && [ -x "$(dirname "$ROOT")/tools/ninja-bin/ninja" ]; then NINJA="$(dirname "$ROOT")/tools/ninja-bin/ninja"; fi
if [ -z "$NINJA" ] && [ -f "$ROOT/build/CMakeCache.txt" ]; then
  NINJA="$(grep -m1 '^CMAKE_MAKE_PROGRAM:' "$ROOT/build/CMakeCache.txt" | cut -d= -f2)"
fi
[ -n "$NINJA" ] && [ -x "$NINJA" ] || { echo "REFUSED: ninja not found (PATH, tools/ninja-bin, build cache)"; exit 3; }

OVERRIDE=()
LLAMA_SRC="$ROOT/build/_deps/strata_llamacpp-src"
[ -d "$LLAMA_SRC" ] && OVERRIDE=(-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP="$LLAMA_SRC")

mkdir -p "$TMP" || exit 3
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "=== asserts-enabled configure ==="
cmake -G Ninja -S "$ROOT" -B "$TMP/build-asserts" \
  -DSTRATA_ENABLE_ASSERTS=ON -DCMAKE_MAKE_PROGRAM="$NINJA" "${OVERRIDE[@]}" > "$TMP/configure-asserts.log" 2>&1
want "configure (asserts ON) rc=0" $?
TAILN="$(tail -2 "$TMP/configure-asserts.log" | tr '\n' ' ')"
echo "      $TAILN"

N1="$(grep -c -- "-DNDEBUG" "$TMP/build-asserts/build.ninja" 2>/dev/null || true)"
N1="${N1:-0}"
echo "      -DNDEBUG occurrences in the asserts build: $N1 (vendor C TUs keep their own Release flags)"

echo "=== sentinel compile + runtime probes (the falsifiers) ==="
cmake --build "$TMP/build-asserts" --target asserts_active_check > "$TMP/sentinel.log" 2>&1
want "sentinel asserts_active_check compiles with asserts live" $?

if [ -x "$TMP/build-asserts/asserts_active_check" ]; then
  NC="$("$NINJA" -C "$TMP/build-asserts" -t commands asserts_active_check | grep -c -- "-DNDEBUG" || true)"
  NC="${NC:-0}"
else
  NC=-1
fi
[ "$NC" = "0" ]; want "sentinel's own compile rules carry zero -DNDEBUG (found $NC)" $?

"$TMP/build-asserts/asserts_active_check" >/dev/null 2>&1
want "sentinel runs clean (rc=0)" $?

STRATA_ASSERTS_PROBE_TRIP=1 "$TMP/build-asserts/asserts_active_check" > "$TMP/probe.log" 2>&1
RC=$?
[ "$RC" -eq 134 ]; want "assert() actually aborts when tripped (SIGABRT rc=134; got $RC)" $?

echo "=== default-class parity (option OFF keeps NDEBUG) ==="
cmake -G Ninja -S "$ROOT" -B "$TMP/build-default" -DCMAKE_MAKE_PROGRAM="$NINJA" "${OVERRIDE[@]}" > "$TMP/configure-default.log" 2>&1
want "configure (default) rc=0" $?
N2="$(grep -c -- "-DNDEBUG" "$TMP/build-default/build.ninja" 2>/dev/null || true)"
N2="${N2:-0}"
[ "$N2" -gt "${N1:-0}" ]; want "asserts build carries fewer -DNDEBUG than default ($N1 < $N2)" $?

echo "checks run: $TOTAL"
if [ "$FAILS" -gt 0 ]; then
  echo "asserts_gate: FAIL ($FAILS of $TOTAL)"
  exit 1
fi
echo "asserts_gate: PASS"
exit 0
