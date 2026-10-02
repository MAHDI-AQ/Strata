#!/usr/bin/env bash
# selftest_acceptance.sh - the acceptance harness's own gate.  Runs the static checks, every
# component selftest, the synthetic fixture battery (positive / negative / mutation / falsify /
# zero-data), the runner dry-run + refusal paths, and (when receipts are present) the recorded
# calibration.  Asserts the suite's own CHECK COUNT - a stage that ran zero checks FAILS.
#
#   bash selftest_acceptance.sh [--recorded-root DIR] [--require-recorded] [--r0-dir DIR]
#
# Env: PY (python3), WS (lab root; default /home/mhd67/strata-serving-lab).
# Exit: 0 all executed checks PASS and total >= MIN_TOTAL / 1 any FAIL / 3 refusal (require-recorded).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="${WS:-/home/mhd67/strata-serving-lab}"
PY="${PY:-python3}"
RECORDED_ROOT=""
R0_DIR=""
REQUIRE_RECORDED=0
MIN_TOTAL=90      # hard floor: updated consciously when the battery grows (never silently lowered)

while [ $# -gt 0 ]; do
  case "$1" in
    --recorded-root) RECORDED_ROOT="$2"; shift ;;
    --r0-dir) R0_DIR="$2"; shift ;;
    --require-recorded) REQUIRE_RECORDED=1 ;;
    *) echo "selftest_acceptance: unknown arg $1" >&2; exit 4 ;;
  esac
  shift
done

TMP="$HERE/tmp/selftest-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$TMP" || exit 3
TOTAL=0
FAILS=0

want() { # $1=name $2=cond(0/1 via test result)
  TOTAL=$((TOTAL + 1))
  if [ "$2" = "0" ]; then
    echo "  [PASS] $1"
  else
    echo "  [FAIL] $1"
    FAILS=$((FAILS + 1))
  fi
}

count_from() { # $1=file with a 'checks run: N' line -> echoes N (last match)
  grep -o 'checks run: [0-9]*' "$1" | tail -1 | grep -o '[0-9]*'
}

jget() { # $1=json file $2=python expression over d
  "$PY" - "$1" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(eval(sys.argv[2], {"d": d}))  # noqa: S307 - fixed expressions authored in this file only
PY
}

step() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------- stage 1: static
step "stage 1: static"
bash -n "$HERE/run_envelope.sh"; want "run_envelope.sh bash -n" $?
bash -n "$HERE/selftest_acceptance.sh"; want "selftest bash -n" $?
for f in token_calib.py fairness.py retention_gate.py score_envelope.py calibrate_recorded.py; do
  "$PY" -m py_compile "$HERE/$f"; want "$f py_compile" $?
done
for f in token_calib.py fairness.py retention_gate.py score_envelope.py calibrate_recorded.py; do
  out="$("$PY" "$HERE/$f" --help 2>&1)"; rc=$?
  [ $rc -eq 0 ] && [ -n "$out" ]; want "$f --help exits 0 with text" $?
done
run="$(head -40 "$HERE/run_envelope.sh" | grep -c 'dry-run')"
[ "$run" -ge 1 ]; want "run_envelope.sh advertises dry-run" $?

# ---------------------------------------------------------------- stage 2: component selftests
step "stage 2: component selftests (check counts asserted)"
declare -A FLOORS=( [token_calib]=8 [fairness]=6 [retention_gate]=6 [score_envelope]=4 )
for comp in token_calib fairness retention_gate score_envelope; do
  log="$TMP/$comp.selftest.log"
  "$PY" "$HERE/$comp.py" --selftest > "$log" 2>&1; rc=$?
  want "$comp selftest rc=0" $rc
  n="$(count_from "$log")"; n="${n:-0}"
  [ "$n" -ge "${FLOORS[$comp]}" ]; want "$comp selftest checks>=${FLOORS[$comp]} (ran $n)" $?
  TOTAL=$((TOTAL + n - 1))   # count the component's own checks into the suite total (minus the wrapper's own)
done

# ---------------------------------------------------------------- stage 3: fixture battery
step "stage 3: synthetic fixture battery (scorer)"
FIX="$TMP/fixtures"
"$PY" - "$FIX" <<'PY'
import json, os, sys
from pathlib import Path
fix = Path(sys.argv[1])

def streak(dir_, n, pump_rate, ttft, dec, prompt=87500, gen=1024):
    d = fix / dir_; d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rows.append({"run": dir_, "agent": i, "ok": True, "prompt_tokens": prompt,
                     "ttft_client_s": ttft + 0.1 * i, "t_fire": 1000.0 + 0.1 * i,
                     "t_done": 1000.0 + ttft + 300, "decode_tps": dec[i], "gen_tokens": gen,
                     "engine_prompt_ms": prompt / pump_rate * 1000.0,
                     "engine_decode_tps": dec[i], "reused_tokens": 0, "finish": "length",
                     "cancelled": "False"})
    (d / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

def sustained(dir_, n=5, turns=6, slow_ttft=False, dead_reuse=False):
    d = fix / dir_; d.mkdir(parents=True, exist_ok=True)
    rows = []
    for ag in range(n):
        base = 87500 + ag * 7
        rows.append({"agent": ag, "turn": 1, "ok": True, "prompt_tokens": base, "ttft_client_s": 55.0,
                     "gen_tokens": 384, "decode_tps": 60.0, "stream_span_s": 6.4,
                     "engine_prompt_ms": base / 8000.0 * 1000.0})
        prev = base
        for turn in range(2, turns + 1):
            newp = prev + 42 + 384
            reused = 0 if dead_reuse else prev - 1
            rows.append({"agent": ag, "turn": turn, "ok": True, "prompt_tokens": newp,
                         "prev_prompt_tokens": prev, "reused_tokens": reused,
                         "resume_engine_log": prev - 1,
                         "ttft_client_s": 1.8 if slow_ttft else 0.6,
                         "gen_tokens": 384, "decode_tps": 58.0 + 0.5 * (ag % 3),
                         "stream_span_s": 6.4})
            prev = newp
    (d / "turns.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

def gates(path):
    Path(path).write_text(json.dumps({"g1": {"rc": 0, "nonexcluded_diff": False}, "faults": 0}), encoding="utf-8")

# positive arm: every class clears the 5x87.5K bars
streak("pos/streak", 5, 8000, 55.0, [60, 62, 64, 66, 68])
sustained("pos/sustained")
gates(fix / "pos" / "gates.json")
# negative: mutate the SAME fixture's streak decode below the bar
streak("neg_streak/streak", 5, 8000, 55.0, [30, 32, 40, 44, 48])
sustained("neg_streak/sustained")
gates(fix / "neg_streak" / "gates.json")
# mutation: dead prefix reuse in the sustained leg (single field family changed)
streak("mut_reuse/streak", 5, 8000, 55.0, [60, 62, 64, 66, 68])
sustained("mut_reuse/sustained", dead_reuse=True)
gates(fix / "mut_reuse" / "gates.json")
# slow retained TTFT: the recorded 1B class (1.8 s p50 > 1.0 s)
streak("slow_ttft/streak", 5, 8000, 55.0, [60, 62, 64, 66, 68])
sustained("slow_ttft/sustained", slow_ttft=True)
gates(fix / "slow_ttft" / "gates.json")
# falsify: N=8, aggregate pump 5,000 <= 5,400 (pre-registered falsifier)
streak("falsify/streak", 8, 5000, 100.0, [55, 56, 57, 58, 59, 60, 61, 62])
# zero-data refusal
(fix / "zero" / "streak").mkdir(parents=True, exist_ok=True)
(fix / "zero" / "sustained").mkdir(parents=True, exist_ok=True)
print("fixtures ->", fix)
PY
want "fixture generator rc=0" $?

score_fix() { # $1=tag $2=cell -> sets SCORE_RC, SCORE_DIR
  local tag="$1" cell="$2"
  SCORE_DIR="$TMP/score-$tag"
  "$PY" "$HERE/score_envelope.py" --cell "$cell" --streak-dir "$FIX/$tag/streak" \
      --sustained-dir "$FIX/$tag/sustained" $( [ -f "$FIX/$tag/gates.json" ] && echo --gates "$FIX/$tag/gates.json" ) \
      --write "$SCORE_DIR.json" > "$TMP/score-$tag.log" 2>&1
  SCORE_RC=$?
}

score_fix pos 5x87.5K
[ "$SCORE_RC" -eq 0 ]; want "positive fixture: exit 0 CONFIRM" $?
v="$(jget "$SCORE_DIR.json" "d['verdict']")"
[ "$v" = "CONFIRM" ]; want "positive fixture: verdict CONFIRM (got $v)" $?
n="$(jget "$SCORE_DIR.json" "d['checks']")"
[ "${n:-0}" -ge 50 ]; want "positive fixture: checks>=50 (ran ${n:-0})" $?
g="$(jget "$SCORE_DIR.json" "len(d['gaps'])")"
[ "${g:-0}" -ge 1 ]; want "positive fixture: named gaps surfaced (got ${g:-0})" $?

score_fix neg_streak 5x87.5K
[ "$SCORE_RC" -eq 1 ]; want "negative streak: exit 1 NOT_CONFIRM" $?
b="$(jget "$SCORE_DIR.json" "'decode_deep_streak' in [x.split(':')[0] for x in d['blockers']]")"
[ "$b" = "True" ]; want "negative streak: blocker names the streak channel" $?

score_fix mut_reuse 5x87.5K
[ "$SCORE_RC" -eq 1 ]; want "mutation (dead reuse): verdict flips to NOT_CONFIRM" $?
b="$(jget "$SCORE_DIR.json" "'retention' in [x.split(':')[0] for x in d['blockers']]")"
[ "$b" = "True" ]; want "mutation (dead reuse): retention blocker present" $?

score_fix slow_ttft 5x87.5K
[ "$SCORE_RC" -eq 1 ]; want "slow retained TTFT: NOT_CONFIRM" $?
b="$(jget "$SCORE_DIR.json" "'retention' in [x.split(':')[0] for x in d['blockers']]")"
[ "$b" = "True" ]; want "slow retained TTFT: retention blocker present" $?

score_fix falsify 8x87.5K
[ "$SCORE_RC" -eq 2 ]; want "NEGATIVE CONTROL - falsify fixture: exit 2 FALSIFY" $?
v="$(jget "$SCORE_DIR.json" "d['verdict']")"
[ "$v" = "FALSIFY" ]; want "NEGATIVE CONTROL - falsify verdict named (got $v)" $?

score_fix zero 5x87.5K
[ "$SCORE_RC" -eq 3 ]; want "zero-data fixture: exit 3 REFUSED (a green here would be vacuous)" $?
TOTAL=$((TOTAL + 5))   # each score_fix run executed the scorer's own check count; count them coarsely
log_n="$(count_from "$TMP/score-pos.log")"
[ "${log_n:-0}" -ge 50 ]; want "scorer printed its own check count (got ${log_n:-none})" $?

# ---------------------------------------------------------------- stage 4: runner dry-run + refusals
step "stage 4: runner dry-run + refusal paths (mock lab root)"
MOCK="$TMP/mockws"
mkdir -p "$MOCK/tools" "$MOCK/scratch/lane-acceptance-kits/fixed" "$MOCK/scratch/pump-boot"
printf '#!/usr/bin/env bash\necho mock\n' > "$MOCK/tools/restart-campaign.sh"
printf '#!/usr/bin/env bash\necho mock\n' > "$MOCK/scratch/lane-acceptance-kits/fixed/run_decode_streak.sh"
printf 'print("mock")\n' > "$MOCK/scratch/lane-acceptance-kits/fixed/agent_sim.py"
printf '{"exe":"upstream/build/strata"}' > "$MOCK/scratch/pump-boot/c5-pump.json"
OUTD="$TMP/dry-out"

WS="$MOCK" PY="$PY" bash "$HERE/run_envelope.sh" --dry-run --cell 5x87.5K --label d1 \
    --config "$MOCK/scratch/pump-boot/c5-pump.json" --out "$OUTD" > "$TMP/dry1.log" 2>&1
[ $? -eq 0 ]; want "runner dry-run single arm rc=0" $?
grep -q "est_streak=$( ( "$PY" "$HERE/token_calib.py" --engine-tokens 87500 | awk '{print $NF}' ) )" "$TMP/dry1.log"
want "runner dry-run uses calibrated est size" $?
[ ! -d "$OUTD" ]; want "runner dry-run writes nothing" $?

printf 'a1|5x87.5K|%s/scratch/pump-boot/c5-pump.json|streak,sustained\na2|8x87.5K|%s/scratch/pump-boot/c5-pump.json|streak\n' "$MOCK" "$MOCK" > "$TMP/plan.txt"
WS="$MOCK" PY="$PY" bash "$HERE/run_envelope.sh" --dry-run --plan "$TMP/plan.txt" --out "$TMP/dry-out2" > "$TMP/dry2.log" 2>&1
[ $? -eq 0 ]; want "runner dry-run plan (battery) rc=0" $?
[ "$(grep -c 'plan: cell=' "$TMP/dry2.log")" -ge 2 ]; want "runner dry-run plans both arms" $?
[ ! -d "$TMP/dry-out2" ]; want "runner plan dry-run writes nothing" $?

date -u +%Y-%m-%dT%H:%M:%SZ > "$MOCK/scratch/.primary-measuring"
WS="$MOCK" PY="$PY" bash "$HERE/run_envelope.sh" --dry-run --cell 5x87.5K --label d2 \
    --config "$MOCK/scratch/pump-boot/c5-pump.json" --out "$OUTD" > "$TMP/dry3.log" 2>&1
[ $? -eq 0 ] && grep -q "marker present" "$TMP/dry3.log"
want "runner dry-run reports (not enforces) the marker" $?
rm -f "$MOCK/scratch/.primary-measuring"

WS="$MOCK" PY="$PY" bash "$HERE/run_envelope.sh" --dry-run --cell 5x87.5K --label d3 \
    --config "$MOCK/scratch/pump-boot/nothere.json" --out "$OUTD" > /dev/null 2>&1
[ $? -eq 3 ]; want "runner refuses a missing config (rc=3)" $?
WS="$MOCK" PY="$PY" bash "$HERE/run_envelope.sh" --dry-run --cell 9x9000 --label d4 \
    --config "$MOCK/scratch/pump-boot/c5-pump.json" --out "$OUTD" > /dev/null 2>&1
[ $? -ne 0 ]; want "runner refuses an unknown cell" $?

# ---------------------------------------------------------------- stage 5: recorded calibration
step "stage 5: recorded-receipt calibration"
if [ -z "$RECORDED_ROOT" ] && [ -d "$WS/scratch/lane-stack-validate" ]; then RECORDED_ROOT="$WS/scratch"; fi
if [ -n "$RECORDED_ROOT" ]; then
  args=(--root "$RECORDED_ROOT")
  [ -n "$R0_DIR" ] && args+=(--r0-dir "$R0_DIR")
  "$PY" "$HERE/calibrate_recorded.py" "${args[@]}" > "$TMP/recorded.log" 2>&1; rc=$?
  want "recorded calibration rc=0" $rc
  n="$(grep -o 'checks run' "$TMP/recorded.log" | wc -l)"
  rec_n="$(grep -o '[0-9]* checks run, [0-9]* failed' "$TMP/recorded.log" | tail -1)"
  NUM="$(echo "$rec_n" | grep -o '^[0-9]*')"; NUM="${NUM:-0}"
  [ "$NUM" -ge 25 ]; want "recorded calibration checks>=25 (ran $NUM)" $?
  TOTAL=$((TOTAL + NUM))
else
  if [ "$REQUIRE_RECORDED" = "1" ]; then
    echo "  [FAIL] --require-recorded but no recorded receipts at $WS/scratch"; FAILS=$((FAILS + 1))
    TOTAL=$((TOTAL + 1))
  else
    echo "  [SKIP] recorded calibration (no receipts found; pass --recorded-root or --require-recorded)"
  fi
fi

# ---------------------------------------------------------------- final
step "final"
echo "total checks executed: $TOTAL (floor $MIN_TOTAL), FAILS=$FAILS"
[ "$TOTAL" -ge "$MIN_TOTAL" ]; want "suite check count >= $MIN_TOTAL" $?
if [ "$FAILS" -gt 0 ]; then
  echo "SELFTEST FAILED ($FAILS failures)"
  exit 1
fi
echo "SELFTEST PASS"
exit 0
