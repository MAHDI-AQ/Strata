#!/usr/bin/env python3
"""fairness.py - fairness/burst analyzer for the N-scaling acceptance harness (harness extension E6).

Scores the §3.4 fairness criteria from the load-kit receipts (nothing modelled):

  share band      -- steady multi-turn window (turns >= steady_min_turn): each active agent's
                     committed tokens within +-15 percent of the per-agent mean; zero agents
                     below 70 percent of the mean.  A window whose median generation is tiny
                     (< --thin-gen, default 32) is reported but marked NOT EVALUABLE - a
                     4-token-generation run cannot carry a share claim.
  burst spread    -- TTFT p90/p50 across the simultaneous (burst/streak) leg; <= 1.3 at N<=8,
                     <= 1.5 at N=16.
  ITL proxy       -- per-turn mean inter-token gap (stream_span_s / (gen_tokens-1)) pooled
                     p99/p50 <= 2.5.  This is a PROXY: the receipts carry no per-token
                     timestamps; the true ITL distribution is an instrument gap (named, not
                     silently passed).

Usage:
  fairness.py --sustained-dir DIR [--burst-dir DIR] [--n 5] [--envelope PATH] [--json]
  fairness.py --selftest

Exit: 0 PASS (all evaluable components pass) / 1 FAIL / 3 REFUSED (no data) / 4 input.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

DEFAULT_STEADY_MIN_TURN = 4
DEFAULT_THIN_GEN = 32


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, max(0, int(math.ceil(q * len(v))) - 1))]


def med(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


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


def analyze(sustained_dir: str | None, burst_dir: str | None, n: int | None, envelope: dict) -> dict:
    env = envelope or {}
    fair = env.get("fairness") or {}
    dw = env.get("decode_windows") or {}
    steady_min_turn = int(dw.get("steady_min_turn_default", DEFAULT_STEADY_MIN_TURN))
    band_pct = float(fair.get("share_band_pct", 15))
    floor_pct = float(fair.get("zero_floor_pct", 70))
    spread_max = float(fair.get("burst_spread_max_n16", 1.5)) if (n or 5) >= 16 else float(fair.get("burst_spread_max_n_le_8", 1.3))
    itl_max = float(fair.get("itl_p99_over_p50_max", 2.5))

    out: dict = {"checks": 0, "gaps": [], "components": {}}

    # ---- burst spread (simultaneous leg)
    if burst_dir:
        rows = [r for r in load_jsonl(Path(burst_dir) / "requests.jsonl") if r.get("ok")]
        ttfts = [float(r["ttft_client_s"]) for r in rows if r.get("ttft_client_s") is not None]
        if len(ttfts) >= 3:
            p50, p90 = med(ttfts), pct(ttfts, 0.9)
            ratio = round(p90 / p50, 3) if p50 else None
            ok = ratio is not None and ratio <= spread_max
            out["components"]["burst_spread"] = {"evaluable": True, "ttft_p50_s": p50, "ttft_p90_s": p90,
                                                 "ratio": ratio, "max": spread_max, "pass": ok,
                                                 "n": len(ttfts)}
            out["checks"] += 1
        else:
            out["components"]["burst_spread"] = {"evaluable": False, "n": len(ttfts),
                                                 "reason": "fewer than 3 ttft readings in the burst leg"}
            out["gaps"].append("burst_spread (no/too little burst-leg ttft data)")
    else:
        out["components"]["burst_spread"] = {"evaluable": False, "reason": "no --burst-dir given"}
        out["gaps"].append("burst_spread (no burst leg provided)")

    # ---- share band + ITL proxy (sustained leg)
    if sustained_dir:
        turns = load_jsonl(Path(sustained_dir) / "turns.jsonl")
        ok = [t for t in turns if t.get("ok")]
        steady = [t for t in ok if int(t.get("turn") or 0) >= steady_min_turn]
        agents = sorted({t.get("agent") for t in steady if t.get("agent") is not None})
        gens = [int(t.get("gen_tokens") or 0) for t in steady]
        thin = (med(gens) or 0) < DEFAULT_THIN_GEN or len(steady) < max(2, len(agents))
        if steady and agents and not thin:
            tok: dict = {a: 0 for a in agents}
            for t in steady:
                tok[t["agent"]] += int(t.get("gen_tokens") or 0)
            vals = [tok[a] for a in agents]
            mean = sum(vals) / len(vals) if vals else 0
            devs = {a: round(100.0 * (tok[a] - mean) / mean, 1) for a in agents} if mean else {}
            max_dev = max((abs(v) for v in devs.values()), default=None)
            min_share = round(100.0 * min(vals) / mean, 1) if mean else None
            ok_band = max_dev is not None and max_dev <= band_pct
            ok_floor = min_share is not None and min_share >= floor_pct
            out["components"]["share_band"] = {"evaluable": True, "window": f"turn>={steady_min_turn}",
                                               "per_agent_tokens": {f"a{a}": tok[a] for a in agents},
                                               "mean": round(mean, 1), "dev_pct": devs,
                                               "max_abs_dev_pct": max_dev, "band_pct": band_pct,
                                               "min_share_pct": min_share, "floor_pct": floor_pct,
                                               "pass": bool(ok_band and ok_floor), "n": len(steady)}
            out["checks"] += 2
        else:
            out["components"]["share_band"] = {"evaluable": False, "n": len(steady), "thin": thin,
                                               "reason": "steady window empty or generation too thin for a share claim"}
            out["gaps"].append("share_band (steady window empty or thin)")

        gaps_span = [(float(t["stream_span_s"]), int(t.get("gen_tokens") or 0))
                     for t in ok if int(t.get("turn") or 0) > 1 and t.get("stream_span_s") is not None
                     and (t.get("gen_tokens") or 0) > 1]
        gaps = [span / (g - 1) for span, g in gaps_span if span > 0]
        if gaps:
            p50g, p99g = med(gaps), pct(gaps, 0.99)
            out["components"]["itl_proxy"] = {
                "evaluable": False,
                "reason": "no per-token timestamps in the receipt: the true per-token ITL distribution "
                          "cannot be computed (instrument extension required)",
                "diagnostic": {"channel": "per-turn mean inter-token gap (retained turns)",
                               "p50_s": round(p50g, 4), "p99_s": round(p99g, 4) if p99g else None,
                               "ratio": round(p99g / p50g, 3) if p50g else None, "n": len(gaps),
                               "note": "phase-mixed across turns; informational only, never a jitter claim"}}
            out["gaps"].append("itl_p99_over_p50 (needs per-token timestamps; per-turn mean-gap diagnostic recorded)")
        else:
            out["components"]["itl_proxy"] = {"evaluable": False, "n": 0,
                                              "reason": "no stream_span_s/gen data for a gap distribution"}
            out["gaps"].append("itl_p99_over_p50 (no span data)")
    else:
        for name in ("share_band", "itl_proxy"):
            out["components"][name] = {"evaluable": False, "reason": "no --sustained-dir given"}
            out["gaps"].append(f"{name} (no sustained leg provided)")

    # ---- per-slot round coverage: engine-profile instrumentation, not in receipts
    out["components"]["slot_coverage"] = {"evaluable": False,
                                          "reason": "per-slot round coverage needs engine profile per-slot counters (M0 counter item)"}
    out["gaps"].append("slot_coverage (engine profile extension pending)")

    ev = [c for c in out["components"].values() if c.get("evaluable")]
    fails = [k for k, c in out["components"].items() if c.get("evaluable") and c.get("pass") is False]
    out["verdict"] = "FAIL" if fails else ("PASS" if ev else "REFUSED")
    out["failed_components"] = fails
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

    print("fairness selftest:")
    import tempfile
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="fair_selftest_"))
    # fixture: 5 agents x 6 turns, steady turns 4-6, balanced tokens -> PASS
    rows = []
    for turn in range(1, 7):
        for ag in range(5):
            rows.append({"agent": ag, "turn": turn, "ok": True, "gen_tokens": 384,
                         "stream_span_s": 4.0 + 0.05 * ag, "ttft_client_s": 80.0 + ag})
    (tmp / "turns.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    a = analyze(str(tmp), None, 5, {})
    want("balanced window: share band PASS", a["components"]["share_band"]["pass"] is True)
    want("ITL is a named gap, never judged without per-token timestamps",
         a["components"]["itl_proxy"]["evaluable"] is False and any("itl_p99" in g for g in a["gaps"]))
    # starve one agent in the steady window -> share band FAIL
    rows2 = [dict(r) for r in rows]
    for r in rows2:
        if r["turn"] >= 4 and r["agent"] == 3:
            r["gen_tokens"] = 60
    (tmp / "t2").mkdir(exist_ok=True)
    (tmp / "t2" / "turns.jsonl").write_text("\n".join(json.dumps(r) for r in rows2) + "\n", encoding="utf-8")
    b = analyze(str(tmp / "t2"), None, 5, {})
    want("starved agent: share band FAIL", b["components"]["share_band"]["pass"] is False)
    want("starved agent: fairness verdict FAIL", b["verdict"] == "FAIL")
    # thin window (4-token gens) -> NOT evaluable, not a pass
    rows3 = [dict(r, gen_tokens=4, stream_span_s=0.0) for r in rows]
    (tmp / "t3").mkdir(exist_ok=True)
    (tmp / "t3" / "turns.jsonl").write_text("\n".join(json.dumps(r) for r in rows3) + "\n", encoding="utf-8")
    c = analyze(str(tmp / "t3"), None, 5, {})
    want("thin window: share band NOT evaluable", c["components"]["share_band"]["evaluable"] is False)
    # burst spread: fixture of 5 simultaneous ttfts, one late -> FAIL vs 1.3
    (tmp / "burst").mkdir(exist_ok=True)
    burst = [{"agent": i, "ok": True, "ttft_client_s": v} for i, v in enumerate([80, 82, 84, 86, 120])]
    (tmp / "burst" / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in burst) + "\n", encoding="utf-8")
    d = analyze(None, str(tmp / "burst"), 5, {})
    want("burst spread 120/84 > 1.3: FAIL", d["components"]["burst_spread"]["pass"] is False)
    # zero data -> REFUSED
    e = analyze(None, None, 5, {})
    want("no inputs: REFUSED, not PASS", e["verdict"] == "REFUSED" and e["checks"] == 0)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"checks run: {checks}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sustained-dir", default=None)
    ap.add_argument("--burst-dir", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--envelope", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.sustained_dir and not a.burst_dir:
        print("fairness: need --sustained-dir and/or --burst-dir", file=sys.stderr)
        return 4
    env = load_envelope(a.envelope)
    res = analyze(a.sustained_dir, a.burst_dir, a.n, env)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"FAIRNESS: {res['verdict']}  (checks run: {res['checks']})")
        for name, c in res["components"].items():
            if c.get("evaluable"):
                print(f"  {name}: {'PASS' if c.get('pass') else 'FAIL'} {json.dumps({k: v for k, v in c.items() if k != 'reason'})}")
            else:
                print(f"  {name}: NOT EVALUABLE - {c.get('reason')}")
    if res["verdict"] == "REFUSED":
        return 3
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
