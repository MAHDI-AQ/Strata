# Soak kit — sustained agent traffic for the acceptance/hardening phase (P21)

The soak: continuous N-agent turn-taking against the live engine for hours-to-days, with a monitor
sampling health/RSS/VRAM/log sizes, ending on a marker. Purpose: prove the settled posture (all
kept knobs) survives sustained real-shaped traffic with bounded logs and zero wedges — the P21
gate before the seal call.

## Run

    bash soak_run.sh <outdir> <minutes> [agents=3] [ctx_cap=131072] [msg_tokens=2000] [max_tokens=128]

- `agent_sim.py` (vendored from the lane-acceptance kit, 2026-10-07; its log-prefix regexes accept
  both `[strata request N]` and `[strata-agx request N]`) drives growing conversations with
  per-turn JSONL metrics; sliding window at ctx-cap.
- `soak_monitor.py` samples every 300 s into `soak-monitor.csv` (+ `soak-events.log` on faults);
  stops when `<outdir>/soak.stop` appears (the runner writes it at window end).
- The runner writes `soak-summary.txt` (turns, rc, health-fail count, last monitor rows, sim stop
  reason).

## Acceptance reading (what a PASS looks like)

- sim `stop_reason` = time/window (never `fatal`); turns recorded > 0 with no failed-turn cluster.
- `soak-events.log` has zero HEALTH-FAIL events; monitor CSV continuous (no >2-interval gaps).
- server.log / engine.log growth bounded by the recipe's rotation (engine.log rotates per boot;
  watch server.log size in the CSV — sustained >200 MiB = rotate it).
- Spot-check the kept-knob ledger (`Research/2026-10-07-phase16-feature-ledger.md`) remains what the
  run exercised; any crash → engine.log autopsy + the two-failure reset discipline.

## Status

Harness landed 2026-10-07 (repo `tools/soak/`). The 48 h soak runs AFTER the remaining queue slices
(S27 arms, P28) so the posture is frozen; then P21 close → the seal package.
