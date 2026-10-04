import os
#!/usr/bin/env python3
"""calibrate_recorded.py - recorded-receipt calibration for the acceptance harness (M0 evidence).

Scores the RECORDED arm receipts of 2026-10-02 through exactly the scorer the N-sweep will use,
and asserts the harness reproduces the recorded channel numbers + verdicts.  Every assertion
names its source receipt; failures print the computed vs recorded values (no silent tolerance).

Calibrated set (recorded value -> source):
  deep-streak  65.4  stack-validate round2 flipped      (lane-stack-validate/report.md section 3)
  deep-streak  43.3  stack-validate round2 reverted
  deep-streak  65.6  lane-refix-r3 refix streak
  deep-streak  61.3 / 42.9 / 61.6 / 42.4            (lane-r1-r3 r1-summary.json)
  all-retained 36.0 / 38.9                          (stack-validate section 5)
  sustained-steady (t4-6) 53.5 / 51.2 ; post-ramp 48.8 (flipped; the reverted 48.4 is NOT
  reconstructible under one uniform rule - reported, not asserted)
  R0  decode 38.1 / per-agent wall 944-974-1174 / agg pump 5,229  (ship-confirm prelim)
  k24 agg pump 5,235                                 (lane-receipts wave-a report)
  retention  1A PASS (p50 0.54) ; 1B/1B2 FAIL on the 1.0 s p50 bound
  composite verdicts: streak clause crossed, sustained clause open -> NOT_CONFIRM

Usage: calibrate_recorded.py [--root SCRATCH_ROOT] [--r0-dir DIR] [--json] [--require]
Exit: 0 all asserts PASS / 1 any FAIL / 3 --require and receipts missing.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score_envelope as S      # noqa: E402
import retention_gate as RG     # noqa: E402

CHECKS: list[dict] = []


def record(name: str, computed, expected, tol, src: str, ok: bool) -> None:
    CHECKS.append({"check": name, "computed": computed, "expected": expected, "tol": tol,
                   "source": src, "ok": bool(ok)})


def near(computed, expected, tol) -> bool:
    return computed is not None and abs(float(computed) - float(expected)) <= tol


def first(pattern: str):
    hits = sorted(glob.glob(pattern))
    return Path(hits[-1]) if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.getenv("STRATA_SCRATCH", "./scratch"))
    ap.add_argument("--r0-dir", default=None, help="ship-confirm prelim receipt override")
    ap.add_argument("--require", action="store_true", help="fail (exit 3) when a receipt root is missing")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    env = S.load_envelope(None)
    cell5 = S.cell_lookup(env, "5x87.5K")
    cell8 = S.cell_lookup(env, "8x87.5K")
    missing = []

    def need(label, p):
        if p is None:
            missing.append(label)
        return p

    # ---- locate receipts (both the box layout and the local mirrors' layouts)
    sv = root / "lane-stack-validate"
    peel = {
        "sv_flip_streak": need("sv flip streak", first(str(sv / "round2" / "prelim-decode-streak-*"))),
        "sv_rev_streak": need("sv reverted streak", first(str(sv / "revert" / "round2" / "prelim-decode-streak-*"))),
        "sv_flip_sust": need("sv flip sustained", first(str(sv / "results" / "sustained-realB-*"))),
        "sv_rev_sust": need("sv reverted sustained", first(str(sv / "revert" / "results" / "sustained-realB-*"))),
        "sv_flip_fast": first(str(sv / "results" / "sustained-realA-*")),
        "refix_streak": need("refix streak (65.6)",
                             first(str(root / "lane-refix-r3" / "mirror" / "refix" / "streak" / "prelim-decode-streak-*"))
                             or first(str(root / "lane-refix-r3" / "refix" / "streak" / "prelim-decode-streak-*"))),
        "r1_on1": need("r1 on1", first(str(root / "lane-r1-r3" / "r1" / "on1" / "run" / "*"))),
        "r1_off1": need("r1 off1", first(str(root / "lane-r1-r3" / "r1" / "off1" / "run" / "*"))),
        "r1_on2": first(str(root / "lane-r1-r3" / "r1" / "on2" / "run" / "*")),
        "r1_off2": first(str(root / "lane-r1-r3" / "r1" / "off2" / "run" / "*")),
        "r0": need("R0 prelim (ship-confirm)",
                   Path(a.r0_dir) if a.r0_dir else first(str(root / "ship-confirm" / "results" / "prelim-5x128-*"))),
        "k24": first(str(root / "lane-receipts" / "a2-k" / "k24" / "run" / "prelim-decode-streak-*")),
    }
    if missing and a.require:
        print(f"calibrate_recorded: REFUSED - missing receipt roots: {', '.join(missing)}", file=sys.stderr)
        return 3
    if missing:
        print(f"calibrate_recorded: note - {len(missing)} receipt(s) absent ({', '.join(missing)}); "
              f"skipping those asserts", file=sys.stderr)

    # ---- deep-streak channel
    for tag, d, expected in (("sv_flip_streak", peel["sv_flip_streak"], 65.4),
                             ("sv_rev_streak", peel["sv_rev_streak"], 43.3),
                             ("refix_streak", peel["refix_streak"], 65.6),
                             ("r1_on1", peel["r1_on1"], 61.3),
                             ("r1_off1", peel["r1_off1"], 42.9),
                             ("r1_on2", peel["r1_on2"], 61.6),
                             ("r1_off2", peel["r1_off2"], 42.4)):
        if d is None:
            continue
        rows = S.burst_rows(str(d))
        ds = S.score_streak_decode(cell5, rows)
        record(f"deep-streak {tag}", ds.get("median"), expected, 0.05, str(d),
               near(ds.get("median"), expected, 0.05))
        # recorded verdicts: >=50 flipped class, <50 bounded class
        want_pass = expected >= 50
        got_pass = ds["verdict"] == "PASS"
        record(f"deep-streak verdict {tag}", ds["verdict"], "PASS" if want_pass else "MISS", 0,
               "master plan 11.5 decode >=50 at N<=8", got_pass == want_pass)

    # ---- sustained channels
    for tag, d, exp_all, exp_t46, exp_pr, assert_pr in (
            ("sv_flip_sust", peel["sv_flip_sust"], 36.0, 53.5, 48.8, True),
            ("sv_rev_sust", peel["sv_rev_sust"], 38.9, 51.2, 48.4, False)):
        if d is None:
            continue
        turns = S.sustained_rows(str(d))
        sd = S.score_sustained_decode(cell5, turns, env)
        chans = {c["channel"]: c for c in sd.get("channels", [])}
        record(f"all-retained {tag}", chans.get("all-retained", {}).get("median"), exp_all, 0.05,
               str(d), near(chans.get("all-retained", {}).get("median"), exp_all, 0.05))
        record(f"t4-6 {tag}", chans.get("sustained-steady", {}).get("median"), exp_t46, 0.05,
               "stack-validate section 5 table", near(chans.get("sustained-steady", {}).get("median"), exp_t46, 0.05))
        pr = chans.get("post-ramp", {}).get("median")
        if assert_pr:
            record(f"post-ramp {tag}", pr, exp_pr, 0.1, "stack-validate section 5 table",
                   near(pr, exp_pr, 0.1))
        else:
            # documented: the reverted 48.4 is not reconstructible under one uniform rule
            record(f"post-ramp {tag} (uniform-rule variant; recorded 48.4 unreconstructible)", pr, "<50", 0,
                   "note: uniform rule computes %.2f; both readings sit below the bar" % (pr or -1),
                   pr is not None and pr < 50)
        # verdicts: both postures' all-retained below the >=50 bar
        record(f"all-retained verdict {tag}", chans.get("all-retained", {}).get("verdict"), "MISS", 0,
               "recorded: below bar on both postures", chans.get("all-retained", {}).get("verdict") == "MISS")

    # ---- R0 numbers
    if peel["r0"] is not None:
        rows = S.burst_rows(str(peel["r0"]))
        pf = S.score_prefill(cell5, rows)
        ds = S.score_streak_decode(cell5, rows)
        tt = S.score_ttft(cell5, rows)
        record("R0 decode p50", ds.get("median"), 38.1, 0.05, "lane-sglang-target section 1.3", near(ds.get("median"), 38.1, 0.05))
        record("R0 per-agent wall min", pf.get("per_agent_min"), 944, 1.0, "same", near(pf.get("per_agent_min"), 944, 1.0))
        record("R0 per-agent wall median", pf.get("per_agent_median"), 974, 1.0, "same", near(pf.get("per_agent_median"), 974, 1.0))
        record("R0 per-agent wall max", pf.get("per_agent_max"), 1174, 1.0, "same", near(pf.get("per_agent_max"), 1174, 1.0))
        record("R0 agg prefill (pump)", pf.get("agg_pump_tps"), 5229, 6.0, "same (Sigma prompt / Sigma prompt_ms = 83,805 ms)", near(pf.get("agg_pump_tps"), 5229, 6.0))
        record("R0 below prefill bar", pf.get("agg_used_tps") is not None and pf["agg_used_tps"] < 7500, True, 0,
               "R0 vs R1 bar", bool(pf.get("agg_used_tps") is not None and pf["agg_used_tps"] < 7500))
        record("R0 shape PASS (est 120000 -> engine 87.5K +-2%)", S.score_shape(cell5, env, rows, []).get("verdict"), "PASS", 0,
               "E5 calibration", S.score_shape(cell5, env, rows, []).get("verdict") == "PASS")
    if peel["k24"] is not None:
        rows = S.burst_rows(str(peel["k24"]))
        pf = S.score_prefill(cell5, rows)
        record("k24 agg prefill (pump)", pf.get("agg_pump_tps"), 5235, 6.0, "lane-receipts wave-a (83,687.5 ms)", near(pf.get("agg_pump_tps"), 5235, 6.0))

    # ---- retention
    for tag, d, want in (("1A (stagger 20, short)", peel["sv_flip_fast"], "PASS"),
                         ("1B (stagger 6, long)", peel["sv_flip_sust"], "FAIL"),
                         ("1B2 reverted long", peel["sv_rev_sust"], "FAIL")):
        if d is None:
            continue
        r = RG.analyze(str(d), env)
        record(f"retention {tag}", r.get("verdict"), want, 0,
               "11.5 retained TTFT <=1.0 s p50 (long arms recorded 1.73/1.54)", r.get("verdict") == want)

    # ---- composite verdict-level reproduction (finish-line snapshot)
    if peel["refix_streak"] is not None and peel["sv_flip_sust"] is not None:
        res = S.score_cell("5x87.5K", env, str(peel["refix_streak"]), str(peel["sv_flip_sust"]), None)
        record("composite verdict (refix streak + flipped sustained)",
               res.get("verdict"), "NOT_CONFIRM", 0, "finish line not met at the recorded snapshot",
               res.get("verdict") == "NOT_CONFIRM")
        record("composite: streak clause PASS",
               res["classes"]["decode_deep_streak"]["verdict"], "PASS", 0, "recorded: streak clause crossed",
               res["classes"]["decode_deep_streak"]["verdict"] == "PASS")
        record("composite: sustained clause MISS",
               res["classes"]["decode_sustained"]["verdict"], "MISS", 0, "recorded: sustained clause OPEN",
               res["classes"]["decode_sustained"]["verdict"] == "MISS")
        record("composite: retention FAIL (long arm 1.73 s > 1.0 s)",
               res["classes"]["retention"]["verdict"], "FAIL", 0, "recorded long-arm retained TTFT",
               res["classes"]["retention"]["verdict"] == "FAIL")

    # ---- negative control on recorded data: the bounded streak arm must not pass the streak bar
    if peel["sv_rev_streak"] is not None:
        rows = S.burst_rows(str(peel["sv_rev_streak"]))
        ds = S.score_streak_decode(cell5, rows)
        record("NEGATIVE CONTROL: bounded 43.3 must NOT pass >=50",
               ds["verdict"], "MISS", 0, "pre-registered bar", ds["verdict"] == "MISS")

    checks = len(CHECKS)
    failed = [c for c in CHECKS if not c["ok"]]
    out = {"checks_run": checks, "failed": len(failed), "missing_receipts": missing, "results": CHECKS}
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        for c in CHECKS:
            print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}: computed={c['computed']} "
                  f"expected={c['expected']} (tol {c['tol']})  <- {c['source']}")
        print(f"recorded calibration: {checks} checks run, {len(failed)} failed"
              + (f", {len(missing)} receipt(s) missing: {', '.join(missing)}" if missing else ""))
    if checks == 0:
        print("calibrate_recorded: REFUSED - zero checks executed", file=sys.stderr)
        return 3
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
