# acceptance — the N-scaling acceptance harness (M0)

The program's finish-line instrument: it boots arms, runs the calibrated load legs, scores every
class of the master-plan §11.5 signature against `envelope.json`, and names the decode channel of
every claim. It gates the Strata serving lane program, so it lives in the repo (`tools/acceptance/`,
a sibling of `tools/hip/` and `tools/vision/`) — not in scratch.

## Files

| file | role |
|---|---|
| `envelope.json` | machine form of §11.5 + lane-sglang-target §3: cells, bars, channels, calibration factor, falsifier |
| `token_calib.py` | engine-token calibration: sizes prompts at `engine_target / 0.729`; asserts landed shapes ±2% (E5) |
| `run_envelope.sh` | runner: boots one arm via `tools/restart-campaign.sh`, runs the streak + sustained legs with the existing load kits, scores, restores the fleet on every exit path (E2/E4) |
| `score_envelope.py` | scorer: all classes vs envelope.json; CONFIRM / NOT_CONFIRM / FALSIFY / REFUSED (E3) |
| `fairness.py` | share band ±15 / zero <70, burst TTFT spread, ITL gap record (E6) |
| `retention_gate.py` | multi-agent sustained retention: coverage ≥90%, retained TTFT ≤1.0 s p50 / ≤2.0 s p95 (E7) |
| `calibrate_recorded.py` | replays the recorded 2026-10-02 receipts through the scorer and asserts the recorded numbers/verdicts |
| `selftest_acceptance.sh` | the suite gate: static + component selftests + fixture battery + dry-run + recorded calibration; asserts its own check count |

## Reading convention (declared once)

Per-agent = **WALL** (agent-observed: engine prompt tokens ÷ client TTFT under full-fleet load;
steady-window decode). Aggregate = same-run Σ. Shapes are declared in **ENGINE** tokens and
asserted ±2% every run (`token_calib.py`; est = engine ÷ 0.729). The busy reading is a diagnostic.
Every decode claim names its channel: **deep-streak** (1024-tok concurrent burst), **sustained-steady**
(per-agent p50 over turns ≥ 4), **all-retained** (per-agent p50 over retained turns, gen≥128 filter
with the recorded <20-row fallback). Judged statistic = cross-agent median (the recorded convention:
"974 per-agent wall", "median 65.4", "median 36.0"); min/max are reported.

## Usage

```bash
# one arm
bash tools/acceptance/run_envelope.sh --cell 5x87.5K --label a1 \
    --config ./scratch/pump-boot/c5-pump.json \
    --gates-in /path/to/gates.json

# battery (plan: label|cell|config|legs per line; '#' comments)
bash tools/acceptance/run_envelope.sh --plan plan.txt

# pre-validate while the box is busy
bash tools/acceptance/run_envelope.sh --dry-run --plan plan.txt

# score receipts directly (no boot)
python3 tools/acceptance/score_envelope.py --cell 5x87.5K \
    --streak-dir <prelim-decode-streak RESULTS_DIR> --sustained-dir <agent-sim RESULTS_DIR>

# the suite gate (recorded calibration auto-detects at $WS/scratch)
bash tools/acceptance/selftest_acceptance.sh
```

Exit codes: runner `0 CONFIRM / 1 NOT_CONFIRM / 2 FALSIFY / 3 refused-or-failed`; scorer
`0 CONFIRM / 1 NOT_CONFIRM / 2 FALSIFY / 3 REFUSED (zero data) / 4 input`.

## Vendor/owner boundaries

- Boots are owned by `tools/restart-campaign.sh` (lab root, single owner); this runner never boots
  directly and never edits fleet configs.
- Legs are driven by the existing kits (`scratch/lane-acceptance-kits/fixed/`): `run_decode_streak.sh`
  (streak leg) and `agent_sim.py` (sustained leg). Override with `STREAK_KIT` / `SUSTAINED_KIT`.
- G1 spots + fault scan arrive as a `--gates-in` receipt (produced by the window's spot leg with the
  existing md-pair comparator); absent gates = `integrity: NOT_EVALUATED` and the cell cannot CONFIRM.
- Refuses while `scratch/.primary-measuring` exists (reported, not enforced, in `--dry-run`).

## Known instrument gaps (named, never silently passed)

- **ITL p99/p50** needs per-token timestamps: today's receipts carry only per-turn spans, so the
  criterion is recorded as a gap (with a per-turn mean-gap diagnostic) until the sustained kit
  records token timestamps.
- **Per-slot round-coverage histogram** needs engine profile per-slot counters (M0 counter item in
  the server/engine; not in this harness's scope).
- Gaps appear in the score output `gaps[]`; they do not block a class verdict but stay visible in
  every receipt (a CONFIRM with gaps still shows them).

## Falsifiers built in

- Pre-registered: aggregate prefill ≤ 5,400 at N=8 → `FALSIFY` (master plan §11.5).
- Zero-data refusal: any run with no scoreable rows → exit 3, never a green.
- Fixture battery (`selftest_acceptance.sh`): positive CONFIRM, negative streak, dead-reuse
  mutation, slow-retained-TTFT, falsify, zero-data — and the recorded calibration (§ below).
- Recorded calibration (`calibrate_recorded.py`): 38 checks — deep-streak 65.4/65.6/43.3 + R1 pairs,
  all-retained 36.0/38.9, t4-6 53.5/51.2, post-ramp 48.8 (flipped; the reverted 48.4 is not
  reconstructible under one uniform rule — reported), R0 38.1 / 944-974-1174 / 5,229 pump, k24
  5,235 pump, retention 1A PASS / 1B/1B2 FAIL, and the composite finish-line snapshot NOT_CONFIRM
  with the streak clause PASS and the sustained clause MISS.
