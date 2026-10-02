#!/usr/bin/env bash
# run_envelope.sh - N-scaling acceptance runner (harness extension E2/E4): boot one arm through
# tools/restart-campaign.sh, run the calibrated fresh/deep-streak leg + the sustained multi-turn
# leg with the existing load kits, then score the arm against envelope.json with score_envelope.py.
# One receipt dir per arm.  The fleet is restored on EVERY exit path unless --no-restore.
#
#   bash run_envelope.sh --cell 5x87.5K --label flip-a --config /abs/pump-boot/c5-pump.json
#   bash run_envelope.sh --dry-run --cell 5x87.5K --label flip-a --config /abs/c5-pump.json
#   bash run_envelope.sh --plan plan.txt            # battery: label|cell|config|legs lines
#
# Engine-token calibration: leg sizes are derived from the cell's ENGINE-token target via
# token_calib.py (est = engine / 0.729); a run never trusts legacy est-space shapes.
#
# Env: WS (lab root, default /home/mhd67/strata-serving-lab), URL, SERVE_LOG, ENGINE_LOG, PY.
# Refuses: marker scratch/.primary-measuring present; missing config/envelope/kits; relative config.
# Exit: 0 all arms CONFIRM / 1 any arm NOT_CONFIRM / 2 any arm FALSIFY / 3 REFUSED, harness input
#       or boot/restore failure / 4 usage.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="${WS:-/home/mhd67/strata-serving-lab}"
URL="${URL:-http://127.0.0.1:8096}"
SERVE_LOG="${SERVE_LOG:-$WS/serve-8096.log}"
ENGINE_LOG="${ENGINE_LOG:-$WS/serve-combined-8-engine.log}"
PY="${PY:-python3}"
MARKER="$WS/scratch/.primary-measuring"
ENVELOPE="$HERE/envelope.json"
STREAK_KIT="${STREAK_KIT:-$WS/scratch/lane-acceptance-kits/fixed}"
SUSTAINED_KIT="${SUSTAINED_KIT:-$WS/scratch/lane-acceptance-kits/fixed}"
RESTART="$WS/tools/restart-campaign.sh"
DEFAULT_RESTORE_CFG="$WS/scratch/pump-boot/c5-pump.json"

MODE="full"
CELL=""; LABEL=""; CONFIG=""; OUT=""; PLAN=""; GATES_IN=""; LEGS="streak,sustained"
NO_RESTORE=0; HOLD_MARKER=0
TURNS=6; SMAX=384; SEED=20261021; MIN_TPS=50
MINUTES_STREAK=12; MINUTES_SUSTAINED=40

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE="dry" ;;
    --cell) CELL="$2"; shift ;;
    --label) LABEL="$2"; shift ;;
    --config) CONFIG="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --plan) PLAN="$2"; shift ;;
    --legs) LEGS="$2"; shift ;;
    --gates-in) GATES_IN="$2"; shift ;;
    --restore-cfg) DEFAULT_RESTORE_CFG="$2"; shift ;;
    --no-restore) NO_RESTORE=1 ;;
    --hold-marker) HOLD_MARKER=1 ;;
    --min-tps) MIN_TPS="$2"; shift ;;
    --turns) TURNS="$2"; shift ;;
    --seed) SEED="$2"; shift ;;
    --minutes-streak) MINUTES_STREAK="$2"; shift ;;
    --minutes-sustained) MINUTES_SUSTAINED="$2"; shift ;;
    *) echo "run_envelope: unknown arg $1" >&2; exit 4 ;;
  esac
  shift
done
if [ -z "$OUT" ]; then OUT="$HERE/results/envelope-$(date -u +%Y%m%dT%H%M%SZ)"; fi
if [ "$MODE" = "dry" ]; then
  LOG="/dev/null"
else
  mkdir -p "$OUT" || exit 3
  LOG="$OUT/run.log"
fi

log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- preflight (shared)
preflight_common() {
  [ -f "$ENVELOPE" ] || { log "REFUSED: envelope missing at $ENVELOPE"; return 3; }
  [ -d "$STREAK_KIT" ] || { log "REFUSED: streak kit dir missing: $STREAK_KIT"; return 3; }
  [ -d "$SUSTAINED_KIT" ] || { log "REFUSED: sustained kit dir missing: $SUSTAINED_KIT"; return 3; }
  [ -f "$RESTART" ] || { log "REFUSED: restart recipe missing: $RESTART"; return 3; }
  if [ -f "$MARKER" ]; then
    if [ "$MODE" = "dry" ]; then log "marker present ($MARKER) - reported (dry-run only)";
    else log "REFUSED: marker present: $MARKER (a measurement holds the box)"; return 3; fi
  fi
  return 0
}

cell_field() { # $1=cell $2=python expr on the cell dict (via json)
  "$PY" - "$ENVELOPE" "$1" "$2" <<'PY'
import json, sys
env = json.load(open(sys.argv[1]))
cell = env["cells"].get(sys.argv[2]) or next((v for k, v in env["cells"].items() if k.lower() == sys.argv[2].lower()), None)
if cell is None:
    print(f"cell {sys.argv[2]!r} not in envelope", file=sys.stderr); sys.exit(4)
print(eval(sys.argv[3], {"cell": cell, "env": env}))
PY
}

# ---------------------------------------------------------------- one arm
run_one_arm() { # $1=label $2=cell $3=config
  local label="$1" cell="$2" cfg="$3"
  local arm="$OUT/arms/$label"
  local rc=0

  case "$cfg" in /*) : ;; *) log "[$label] REFUSED: config must be an ABSOLUTE path: $cfg"; return 3 ;; esac
  [ -f "$cfg" ] || { log "[$label] REFUSED: config not found: $cfg"; return 3; }
  local engine_tokens est_streak est_cap est_steady n
  engine_tokens="$(cell_field "$cell" 'cell["ctx_engine"]')" || return 3
  n="$(cell_field "$cell" 'cell["n"]')" || return 3
  est_streak="$("$PY" "$HERE/token_calib.py" --engine-tokens "$engine_tokens" | awk '{print $NF}')" || return 3
  est_cap="$(cell_field "$cell" 'cell["est_ctx_cap"]')" || return 3
  est_steady="$("$PY" - "$engine_tokens" <<'PY'
import sys
et = float(sys.argv[1])
print(int(round(0.9 * et / 0.729)))
PY
)" || return 3

  local C_STREAK=""
  case ",$LEGS," in *,streak,*) C_STREAK="OUT=$arm/streak AGENTS=$n PROMPT_TOKENS=$est_streak MAX_TOKENS=1024 MINUTES=$MINUTES_STREAK REQ_BOUND=900 SEED=$SEED MIN_TPS=$MIN_TPS RESTART_CFG= BUSY=1 bash $STREAK_KIT/run_decode_streak.sh" ;; esac
  local C_SUSTAIN=""
  case ",$LEGS," in *,sustained,*) C_SUSTAIN="$PY $SUSTAINED_KIT/agent_sim.py --url $URL --serve-log $SERVE_LOG --engine-log $ENGINE_LOG --out $arm/sustained --agents $n --turns $TURNS --minutes $MINUTES_SUSTAINED --ctx-cap $est_cap --steady-min-ctx $est_steady --max-tokens $SMAX --seed $SEED --target-agent-tps $MIN_TPS" ;; esac
  local C_BOOT="bash $RESTART $cfg"
  local C_SCORE="$PY $HERE/score_envelope.py --cell $cell --streak-dir $arm/streak --sustained-dir $arm/sustained --write $arm/score.json"
  [ -n "$GATES_IN" ] && C_SCORE="$C_SCORE --gates $arm/gates.json"

  log "[$label] plan: cell=$cell n=$n engine_target=$engine_tokens est_streak=$est_streak est_cap=$est_cap est_steady=$est_steady"
  log "[$label] boot:  $C_BOOT"
  log "[$label] streak: ${C_STREAK:-(skipped)}"
  log "[$label] sustain: ${C_SUSTAIN:-(skipped)}"
  log "[$label] score: $C_SCORE"

  if [ "$MODE" = "dry" ]; then
    return 0
  fi
  mkdir -p "$arm"

  # --- boot (teardown-aware recipe owns teardown + readiness)
  log "[$label] booting via restart-campaign..."
  if ! bash "$RESTART" "$cfg" >> "$LOG" 2>&1; then
    log "[$label] ABORT: boot rc!=0"; rc=3
  fi
  # /metrics must answer fast at idle after the boot (server-change wedge class)
  if [ $rc -eq 0 ]; then
    if ! "$PY" - "$URL" <<'PY' >> "$LOG" 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/metrics", timeout=10) as r:
    m = json.loads(r.read())
e = m.get("engine") or {}
print("post-boot /metrics ok:", {k: e.get(k) for k in ("model", "concurrency", "kv", "version", "batch_rows")})
PY
    then log "[$label] ABORT: /metrics did not answer after boot"; rc=3; fi
  fi

  # --- pins
  if [ $rc -eq 0 ]; then
    local head
    head="$(git -C "$WS/upstream" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    "$PY" - "$arm/arm.json" "$label" "$cell" "$cfg" "$head" "$engine_tokens" "$est_streak" "$est_cap" "$est_steady" "$WS" <<'PY'
import json, sys, hashlib, time, os
out, label, cell, cfg, head, et, es, ec, ess, ws = sys.argv[1:11]
def sha(p):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None
cfg_data = json.load(open(cfg))
exe = cfg_data.get("exe") or os.path.join(ws, "upstream", "build", "strata")
if not os.path.isabs(exe):
    exe = os.path.join(ws, exe)
json.dump({"label": label, "cell": cell, "config": cfg, "config_sha256": sha(cfg),
           "head": head, "binary": exe, "binary_sha256": sha(exe),
           "engine_target_tokens": float(et), "est_streak_tokens": int(es),
           "est_ctx_cap": int(ec), "est_steady_min_ctx": int(ess),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          open(out, "w"), indent=1)
print("pins ->", out)
PY
  fi

  # --- streak leg (kit's own gate rc is recorded, never fatal for the arm: the envelope scorer is the verdict)
  if [ $rc -eq 0 ] && [ -n "$C_STREAK" ]; then
    log "[$label] streak leg running..."
    ( cd "$STREAK_KIT" && env OUT="$arm/streak" AGENTS="$n" PROMPT_TOKENS="$est_streak" MAX_TOKENS=1024 \
        MINUTES="$MINUTES_STREAK" REQ_BOUND=900 SEED="$SEED" MIN_TPS="$MIN_TPS" RESTART_CFG= BUSY=1 \
        bash "$STREAK_KIT/run_decode_streak.sh" ) >> "$LOG" 2>&1
    local src=$?
    log "[$label] streak leg rc=$src (kit gate; scorer re-derives)"
  fi

  # --- sustained leg
  if [ $rc -eq 0 ] && [ -n "$C_SUSTAIN" ]; then
    log "[$label] sustained leg running..."
    $PY "$SUSTAINED_KIT/agent_sim.py" --url "$URL" --serve-log "$SERVE_LOG" --engine-log "$ENGINE_LOG" \
      --out "$arm/sustained" --agents "$n" --turns "$TURNS" --minutes "$MINUTES_SUSTAINED" \
      --ctx-cap "$est_cap" --steady-min-ctx "$est_steady" --max-tokens "$SMAX" --seed "$SEED" \
      --target-agent-tps "$MIN_TPS" >> "$LOG" 2>&1
    local src2=$?
    log "[$label] sustained leg rc=$src2"
    [ $src2 -eq 0 ] || log "[$label] sustained leg non-zero rc=$src2 (receipt kept; scorer will judge)"
  fi

  # --- gates copy (G1 spots + faults produced by the window's spot leg)
  [ -n "$GATES_IN" ] && [ -f "$GATES_IN" ] && cp "$GATES_IN" "$arm/gates.json"

  # --- score
  if [ $rc -eq 0 ]; then
    log "[$label] scoring..."
    $PY "$HERE/score_envelope.py" --cell "$cell" --streak-dir "$arm/streak" --sustained-dir "$arm/sustained" \
      $( [ -f "$arm/gates.json" ] && echo --gates "$arm/gates.json" ) --write "$arm/score.json" >> "$LOG" 2>&1
    rc=$?
    log "[$label] score rc=$rc ($( [ -f "$arm/score.json" ] && "$PY" -c "import json;print(json.load(open('$arm/score.json'))['verdict'])" || echo NO_SCORE ))"
  fi

  # arm receipt status
  local verdict="ABORT"
  [ -f "$arm/score.json" ] && verdict="$("$PY" -c "import json;print(json.load(open('$arm/score.json'))['verdict'])")"
  echo "$verdict" > "$arm/STATUS"
  log "[$label] STATUS=$verdict"
  return $rc
}

# ---------------------------------------------------------------- main
preflight_common || exit $?

if [ "$HOLD_MARKER" = "1" ] && [ "$MODE" != "dry" ]; then
  if [ -f "$MARKER" ]; then log "REFUSED: marker already present"; exit 3; fi
  ( umask 022; date -u +%Y-%m-%dT%H:%M:%SZ > "$MARKER" )
  MARKER_OWNED=1
  log "marker held: $MARKER"
else
  MARKER_OWNED=0
fi

restore_fleet() {
  local rc=$?
  if [ "$NO_RESTORE" = "1" ] || [ "$MODE" = "dry" ]; then
    log "restore skipped (no-restore/dry)"
  else
    log "restoring fleet: bash $RESTART $DEFAULT_RESTORE_CFG"
    bash "$RESTART" "$DEFAULT_RESTORE_CFG" >> "$LOG" 2>&1 && log "fleet restore OK" || log "FLEET RESTORE FAILED - operator attention"
  fi
  if [ "$MARKER_OWNED" = "1" ]; then rm -f "$MARKER"; log "marker released"; fi
  exit $rc
}
trap restore_fleet EXIT INT TERM

FINAL=0
if [ -n "$PLAN" ]; then
  [ -f "$PLAN" ] || { log "REFUSED: plan not found: $PLAN"; exit 3; }
  while IFS='|' read -r plabel pcell pcfg plegs; do
    [ -z "$plabel" ] && continue
    case "$plabel" in \#*) continue ;; esac
    LEGS="${plegs:-streak,sustained}"
    run_one_arm "$plabel" "$pcell" "$pcfg"; r=$?
    case "$r" in
      0) ;;
      2) [ "$FINAL" -lt 2 ] && FINAL=2 ;;
      3) FINAL=3 ;;
      *) [ "$FINAL" -lt 1 ] && FINAL=1 ;;
    esac
  done < "$PLAN"
else
  [ -n "$CELL" ] || { log "usage: --cell required (or --plan)"; exit 4; }
  [ -n "$LABEL" ] || LABEL="arm-$(date -u +%Y%m%dT%H%M%SZ)"
  [ -n "$CONFIG" ] || { log "usage: --config required"; exit 4; }
  run_one_arm "$LABEL" "$CELL" "$CONFIG"; FINAL=$?
fi

log "envelope run done: exit=$FINAL (0 CONFIRM / 1 NOT_CONFIRM / 2 FALSIFY / 3 refused-or-failed)"
exit $FINAL
