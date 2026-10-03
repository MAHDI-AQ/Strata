#!/usr/bin/env python3
"""retention_gate.py - multi-agent sustained retention gate (harness extension E7).

Scores the master-plan 11.5 retention clause on a sustained multi-turn receipt (turns.jsonl
written by the sustained instrument / agent_sim): on re-sent conversation turns,
  coverage: >= 90 percent of re-sent turns resume (reused >= 0.95 x the agent's previous prompt)
  retained TTFT: <= 1.0 s p50 and <= 2.0 s p95 over the resumed turns.

Two verification tiers (kept separate on purpose - the recorded instrument's fast turns can
escape the /metrics poller, so a per-turn ratio is not always computable):
  tier-1 (ratio-checkable): prev prompt known AND reused_tokens present -> reused >= 0.95*prev
  tier-2 (engine-log evidence): the turn carries the engine's own `slot resumes N tokens` value
        (resume_engine_log) but no computable prev; counted as coverage EVIDENCE, flagged as
        ratio-unchecked.  (The recorded definitive offline check upgrades these from
        /metrics?requests=all chains; this gate reports the split rather than hiding it.)

Usage:
  retention_gate.py --sustained-dir DIR [--n 5] [--envelope PATH] [--json]
  retention_gate.py --selftest

Exit: 0 PASS / 1 FAIL / 3 REFUSED (no data) / 4 input.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

NUM_RE = re.compile(r"^\s*(\d+)")


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, max(0, int(math.ceil(q * len(v))) - 1))]


def med(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _num(v):
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        m = NUM_RE.match(v)
        if m:
            return int(m.group(1))
    return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass
    except OSError:
        pass
    return rows


def load_envelope(path: str | None) -> dict:
    p = Path(path) if path else Path(__file__).resolve().parent / "envelope.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def analyze(sustained_dir: str | None, envelope: dict) -> dict:
    env = envelope or {}
    ret = env.get("retention") or {}
    ratio_min = float(ret.get("reused_ratio_min", 0.95))
    coverage_min = float(ret.get("coverage_pct_min", 90))
    ttft_p50_max = float(ret.get("retained_ttft_p50_s_max", 1.0))
    ttft_p95_max = float(ret.get("retained_ttft_p95_s_max", 2.0))
    out: dict = {"checks": 0}

    if not sustained_dir:
        out["status"] = "REFUSED"
        out["reason"] = "no --sustained-dir given"
        return out
    turns = load_jsonl(Path(sustained_dir) / "turns.jsonl")
    ok = [t for t in turns if t.get("ok")]
    retained = [t for t in ok if int(t.get("turn") or 0) > 1]
    if not retained:
        out["status"] = "REFUSED"
        out["reason"] = "no ok retained turns (turn>=2) in receipt"
        return out

    per_turn = []
    verified = ratio_fail = evidenced = 0
    ratio_checkable = 0
    for t in retained:
        ag, turn = t.get("agent"), int(t.get("turn") or 0)
        prev = t.get("prev_prompt_tokens")
        if prev is None:
            for p in turns:
                if p.get("agent") == ag and int(p.get("turn") or 0) == turn - 1:
                    prev = p.get("prompt_tokens")
                    break
        reused = _num(t.get("reused_tokens"))
        resume = _num(t.get("resume_engine_log"))
        row = {"agent": ag, "turn": turn, "prev_prompt": prev, "reused": reused,
               "resume_engine_log": resume, "ttft_s": t.get("ttft_client_s")}
        if prev is not None and reused is not None:
            ratio_checkable += 1
            okrow = reused >= ratio_min * prev
            row["pass"] = okrow
            row["ratio"] = round(reused / prev, 5) if prev else None
            verified += 1 if okrow else 0
            ratio_fail += 0 if okrow else 1
        else:
            row["pass"] = None
            if resume is not None and resume > 0:
                evidenced += 1
                row["evidence"] = "engine-log resume (ratio unchecked)"
        per_turn.append(row)

    total = len(retained)
    cover_evidence = verified + evidenced
    evidence_pct = round(100.0 * cover_evidence / total, 1)
    ratio_pct = round(100.0 * verified / ratio_checkable, 1) if ratio_checkable else None
    coverage_pass = (ratio_fail == 0) and (evidence_pct >= coverage_min)
    ttfts = [float(t["ttft_client_s"]) for t in retained if t.get("ttft_client_s") is not None]
    ttft_p50 = med(ttfts)
    ttft_p95 = pct(ttfts, 0.95)
    ttft_pass = (ttft_p50 is not None and ttft_p50 <= ttft_p50_max
                 and ttft_p95 is not None and ttft_p95 <= ttft_p95_max)

    out.update({
        "status": "OK",
        "retained_turns": total,
        "ratio_checkable": ratio_checkable,
        "ratio_verified": verified,
        "ratio_failed": ratio_fail,
        "ratio_pass_pct": ratio_pct,
        "evidence_turns": evidenced,
        "coverage_evidence_pct": evidence_pct,
        "coverage_min_pct": coverage_min,
        "coverage_pass": coverage_pass,
        "ttft_p50_s": ttft_p50, "ttft_p95_s": ttft_p95,
        "ttft_p50_max_s": ttft_p50_max, "ttft_p95_max_s": ttft_p95_max,
        "ttft_pass": bool(ttft_pass),
        "verdict": "PASS" if (coverage_pass and ttft_pass) else "FAIL",
        "per_turn": per_turn,
        "checks": 2 + total,
    })
    return out


def selftest() -> int:
    checks = 0
    ok = True

    def want(name, cond):
        nonlocal checks, ok
        checks += 1
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("retention_gate selftest:")
    import tempfile
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="ret_selftest_"))

    def write(tag, rows):
        d = tmp / tag
        d.mkdir(exist_ok=True)
        (d / "turns.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return str(d)

    base = []
    for ag in range(5):
        base.append({"agent": ag, "turn": 1, "ok": True, "prompt_tokens": 86000 + ag, "ttft_client_s": 80.0})
        for turn in range(2, 7):
            base.append({"agent": ag, "turn": turn, "ok": True, "prompt_tokens": 86000 + ag + 42 * turn,
                         "prev_prompt_tokens": 86000 + ag + 42 * (turn - 1),
                         "reused_tokens": 86000 + ag + 42 * (turn - 1) - 1, "ttft_client_s": 0.55})
    a = analyze(write("pass", base), {})
    want("full resume + fast TTFT: PASS", a["verdict"] == "PASS" and a["coverage_evidence_pct"] == 100.0)
    # negative: a dead prefix (reused=0) on every retained turn -> coverage FAIL
    dead = [dict(r) for r in base]
    for r in dead:
        if r["turn"] >= 2:
            r["reused_tokens"] = 0
    b = analyze(write("dead", dead), {})
    want("reused=0 everywhere: coverage FAIL", b["coverage_pass"] is False and b["verdict"] == "FAIL")
    # slow retained TTFT: p50 1.7s -> TTFT FAIL (recorded stack-validate 1B class)
    slow = [dict(r) for r in base]
    for r in slow:
        if r["turn"] >= 2:
            r["ttft_client_s"] = 1.7
    c = analyze(write("slow", slow), {})
    want("retained TTFT p50 1.7s > 1.0s: FAIL", c["ttft_pass"] is False and c["verdict"] == "FAIL")
    # tier-2 only: fast turns with resume evidence and no computable prev (recorded 1A class)
    fast = []
    for ag in range(5):
        fast.append({"agent": ag, "turn": 1, "ok": True, "prompt_tokens": 86000 + ag, "ttft_client_s": 16.7})
        for turn in range(2, 7):
            row = {"agent": ag, "turn": turn, "ok": True, "ttft_client_s": 0.54, "resume_engine_log": 86000 + ag + 44 * turn}
            if turn == 2:
                row["prev_prompt_tokens"] = 86000 + ag
                row["reused_tokens"] = 86000 + ag + 1
            fast.append(row)
    d = analyze(write("fast", fast), {})
    want("tier-2 evidence counts toward >=90% coverage", d["coverage_evidence_pct"] >= 90.0 and d["verdict"] == "PASS")
    want("tier split reported (ratio_checkable < retained)", d["ratio_checkable"] < d["retained_turns"])
    # zero data -> REFUSED
    e = analyze(write("empty", []), {})
    want("empty receipt: REFUSED, not PASS", e["verdict"] if "verdict" in e else e["status"] == "REFUSED")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"checks run: {checks}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sustained-dir", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--envelope", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.sustained_dir:
        print("retention_gate: need --sustained-dir", file=sys.stderr)
        return 4
    res = analyze(a.sustained_dir, load_envelope(a.envelope))
    if res.get("status") == "REFUSED":
        print(f"retention_gate: REFUSED - {res.get('reason')}", file=sys.stderr)
        return 3
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"RETENTION: {res['verdict']}  (checks run: {res['checks']})")
        print(f"  coverage: evidence {res['coverage_evidence_pct']}% (>= {res['coverage_min_pct']}%) "
              f"[ratio-checked {res['ratio_verified']}/{res['ratio_checkable']} fail={res['ratio_failed']}; "
              f"tier-2 evidence {res['evidence_turns']}/{res['retained_turns']} turns] -> {'PASS' if res['coverage_pass'] else 'FAIL'}")
        print(f"  retained TTFT: p50 {res['ttft_p50_s']} s (<= {res['ttft_p50_max_s']}) p95 {res['ttft_p95_s']} s "
              f"(<= {res['ttft_p95_max_s']}) -> {'PASS' if res['ttft_pass'] else 'FAIL'}")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
