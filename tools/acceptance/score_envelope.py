#!/usr/bin/env python3
"""score_envelope.py - the N-scaling acceptance scorer (harness extension E3).

Scores ONE arm's receipts against envelope.json (the machine form of the master-plan 11.5
signature + lane-sglang-target section 3).  Nothing is modelled: every number is recomputed
from the raw receipts (requests.jsonl / turns.jsonl / metrics-series.jsonl / gates.json);
summary.md is never trusted.

Classes (each names its instrument + channel):
  shape            engine prompt tokens within +-2 percent of the declared cell shape
  prefill_wall     per-agent WALL prefill = engine prompt tokens / client TTFT, under fleet load
  prefill_agg      same-run aggregate (pump = sum prompt / sum engine prompt_ms;
                   cross-check = sum prompt / fleet wall)
  decode_deep_streak   channel deep-streak (1024-token concurrent burst)
  decode_sustained     channels all-retained + sustained-steady (multi-turn workload instrument)
  decode_agg       aggregate decode (sum of per-agent rates; live /metrics cross-check)
  ttft_fresh       fresh-turn TTFT p50 vs the cell bound
  fairness         share band +-15 / zero <70; burst spread; ITL proxy (see fairness.py)
  retention        coverage >=90 percent; retained TTFT p50 <=1.0 s / p95 <=2.0 s (see retention_gate.py)
  integrity        G1 spots + fault scan from gates.json (NOT EVALUATED when absent)

Verdicts: CONFIRM (every class PASS) | NOT_CONFIRM (blockers listed) | FALSIFY (pre-registered
master-plan 11.5 falsifier: aggregate prefill <= 5400 at N=8) | REFUSED (zero data).

Usage:
  score_envelope.py --cell 5x87.5K --streak-dir D [--sustained-dir D] [--gates G.json]
                    [--arm-dir DIR] [--envelope PATH] [--json] [--write OUT.json]
Exit: 0 CONFIRM / 1 NOT_CONFIRM / 2 FALSIFY / 3 REFUSED / 4 input.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fairness as fair_mod          # noqa: E402
import retention_gate as ret_mod     # noqa: E402
import token_calib as cal_mod        # noqa: E402

NEAR_FACTOR = 0.9


def med(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, max(0, int(math.ceil(q * len(v))) - 1))]


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
    p = Path(path) if path else HERE / "envelope.json"
    return json.loads(p.read_text(encoding="utf-8"))


def cell_lookup(env: dict, cell: str) -> dict:
    cells = env.get("cells") or {}
    if cell in cells:
        return cells[cell]
    # tolerate case/spacing variants: "5x87.5k", "5X87.5K"
    for k, v in cells.items():
        if k.lower() == str(cell).lower():
            return v
    raise KeyError(f"cell {cell!r} not in envelope (have: {', '.join(cells)})")


def burst_rows(d: str | None) -> list[dict]:
    if not d:
        return []
    rows = load_jsonl(Path(d) / "requests.jsonl")
    if not rows:
        # tolerate a nested single results subdir (the kits write RESULTS_DIR=<parent>/<stamp>)
        sub = sorted([p for p in Path(d).iterdir() if p.is_dir()]) if Path(d).is_dir() else []
        for p in sub:
            rows = load_jsonl(p / "requests.jsonl")
            if rows:
                break
    return rows


def sustained_rows(d: str | None) -> list[dict]:
    if not d:
        return []
    rows = load_jsonl(Path(d) / "turns.jsonl")
    if not rows:
        sub = sorted([p for p in Path(d).iterdir() if p.is_dir()]) if Path(d).is_dir() else []
        for p in sub:
            rows = load_jsonl(p / "turns.jsonl")
            if rows:
                break
    return rows


def cls(verdict, **kw):
    d = {"verdict": verdict}
    d.update(kw)
    return d


def verdict_of(value, bar):
    if value is None:
        return "NOT_EVALUATED"
    if value >= bar:
        return "PASS"
    if value >= NEAR_FACTOR * bar:
        return "NEAR"
    return "MISS"


# ---------------------------------------------------------------- classes

def score_shape(cell: dict, env: dict, streak: list[dict], turns: list[dict]) -> dict:
    declared = float(cell["ctx_engine"])
    tol = float((env.get("calibration") or {}).get("tolerance_pct", 2))
    obs = [int(r["prompt_tokens"]) for r in streak if r.get("prompt_tokens") is not None]
    if not obs:
        obs = [int(t["prompt_tokens"]) for t in turns if t.get("prompt_tokens") is not None]
    if not obs:
        obs = [int(t["ctx_engine_tokens"]) for t in turns if t.get("ctx_engine_tokens") is not None]
    res = cal_mod.check_values(declared, obs, tol)
    if res["ok"] is None:
        return cls("NOT_EVALUATED", reason=res["reason"], declared_engine_tokens=declared)
    return cls("PASS" if res["ok"] else "FAIL", declared_engine_tokens=declared, tol_pct=tol,
               observed_n=res["n"], observed_min=res["min"], observed_max=res["max"],
               window=res["window"], out_of_tol=res["out_of_tol"],
               checks=1 + res["n"])


def score_prefill(cell: dict, streak: list[dict]) -> dict:
    ok = [r for r in streak if r.get("ok")]
    pairs = [(int(r["prompt_tokens"]), float(r["ttft_client_s"])) for r in ok
             if r.get("prompt_tokens") is not None and r.get("ttft_client_s") is not None]
    if not pairs:
        return cls("NOT_EVALUATED", reason="no ok burst/streak rows with prompt tokens + client TTFT")
    wall = [p / t for p, t in pairs if t > 0]
    bar_agent = float(cell["prefill"]["per_agent_wall_min_tps"])
    bar_agg = float(cell["prefill"]["agg_min_tps"])
    pump_ms = sum(float(r["engine_prompt_ms"]) for r in ok if r.get("engine_prompt_ms") is not None)
    sum_pt = sum(p for p, _ in pairs)
    agg_pump = (sum_pt / (pump_ms / 1000.0)) if pump_ms > 0 else None
    fires = [float(r["t_fire"]) for r in ok if r.get("t_fire") is not None]
    firsts = [float(r["t_fire"]) + float(r["ttft_client_s"]) for r in ok
              if r.get("t_fire") is not None and r.get("ttft_client_s") is not None]
    agg_wall = (sum_pt / (max(firsts) - min(fires))) if fires and firsts and max(firsts) > min(fires) else None
    agg_used = agg_pump if agg_pump is not None else agg_wall
    cross_warn = None
    if agg_pump is not None and agg_wall is not None:
        if abs(agg_pump - agg_wall) / agg_pump > 0.05:
            cross_warn = f"pump {agg_pump:.0f} vs wall {agg_wall:.0f} differ >5%"
    # judged statistic = cross-agent median ("974 per-agent wall" convention); min/max reported
    v_agent = verdict_of(med(wall), bar_agent)
    v_agg = verdict_of(agg_used, bar_agg)
    overall = "PASS" if v_agent == "PASS" and v_agg == "PASS" else \
              ("NEAR" if "MISS" not in (v_agent, v_agg) else "MISS")
    return cls(overall, per_agent_wall_tps=[round(w, 1) for w in wall],
               per_agent_median=round(med(wall), 1), per_agent_min=round(min(wall), 1),
               per_agent_max=round(max(wall), 1), per_agent_bar_tps=bar_agent,
               agg_pump_tps=round(agg_pump, 1) if agg_pump is not None else None,
               agg_wall_tps=round(agg_wall, 1) if agg_wall is not None else None,
               agg_used_tps=round(agg_used, 1) if agg_used is not None else None,
               agg_bar_tps=bar_agg, cross_check_warn=cross_warn,
               checks=2 + len(wall))


def score_streak_decode(cell: dict, streak: list[dict]) -> dict:
    ok = [r for r in streak if r.get("ok")]
    rates = [float(r["decode_tps"]) for r in ok if r.get("decode_tps") is not None]
    if not rates:
        return cls("NOT_EVALUATED", channel="deep-streak",
                   reason="no ok streak rows with done-line decode rates")
    gens = [int(r.get("gen_tokens") or 0) for r in ok]
    gen_note = None
    if gens and max(gens) < 1024 * 0.9:
        gen_note = f"max gen {max(gens)} < 1024: not a full deep-streak emission"
    bar = float(cell["decode"]["per_agent_min_tps"])
    v = "NOT_EVALUATED" if gen_note else verdict_of(med(rates), bar)
    return cls(v, channel="deep-streak", per_agent_tps=[round(x, 1) for x in rates],
               min_agent=round(min(rates), 1), median=round(med(rates), 1), bar_tps=bar,
               gen_note=gen_note, checks=1 + len(rates))


def _agent_p50s(turns: list[dict], pred) -> dict:
    per: dict = {}
    agents = sorted({t.get("agent") for t in turns if t.get("agent") is not None})
    for ag in agents:
        vals = [float(t["decode_tps"]) for t in turns
                if t.get("agent") == ag and t.get("decode_tps") is not None and pred(t)]
        per[ag] = med(vals) if vals else None
    return per


def score_sustained_decode(cell: dict, turns: list[dict], env: dict) -> dict:
    ok = [t for t in turns if t.get("ok")]
    if not ok:
        return cls("NOT_EVALUATED", channels=[],
                   reason="no ok turns in the sustained receipt")
    dw = env.get("decode_windows") or {}
    steady_min = int(dw.get("steady_min_turn_default", 4))
    post_ramp = int(dw.get("post_ramp_min_turn", 3))
    gen_min = int(dw.get("gen_min_tokens", 128))
    fmin = int(dw.get("gen_min_fallback_rows", 20))
    bar = float(cell["decode"]["per_agent_min_tps"])

    ret = [t for t in ok if int(t.get("turn") or 0) > 1]
    ret_ge = [t for t in ret if int(t.get("gen_tokens") or 0) >= gen_min]
    use_all = len(ret_ge) < fmin
    pred_ret = (lambda t: int(t.get("gen_tokens") or 0) >= gen_min) if not use_all else (lambda t: True)
    pa_ret = _agent_p50s(ret, pred_ret)
    vals_ret = [v for v in pa_ret.values() if v is not None]

    pa_steady = _agent_p50s(ok, lambda t: int(t.get("turn") or 0) >= steady_min)
    vals_st = [v for v in pa_steady.values() if v is not None]
    pa_pr = _agent_p50s(ok, lambda t: int(t.get("turn") or 0) >= post_ramp)
    vals_pr = [v for v in pa_pr.values() if v is not None]

    channels = []
    fid = None
    if use_all and ret:
        mg = max(int(t.get("gen_tokens") or 0) for t in ret)
        if mg < gen_min:
            fid = f"low-gen window (max gen {mg} < {gen_min}); fallback rule in effect - informational"
    if vals_ret:
        channels.append({"channel": "all-retained", "per_agent_p50": {f"a{k}": round(v, 1) for k, v in pa_ret.items() if v is not None},
                         "min_agent": round(min(vals_ret), 1), "median": round(med(vals_ret), 1),
                         "bar_tps": bar, "verdict": verdict_of(med(vals_ret), bar),
                         "filter": "all" if use_all else f"gen>={gen_min}", "gen_fidelity": fid})
    if vals_st:
        channels.append({"channel": "sustained-steady", "window": f"turn>={steady_min}",
                         "per_agent_p50": {f"a{k}": round(v, 1) for k, v in pa_steady.items() if v is not None},
                         "min_agent": round(min(vals_st), 1), "median": round(med(vals_st), 1),
                         "bar_tps": bar, "verdict": verdict_of(med(vals_st), bar)})
    if vals_pr:
        channels.append({"channel": "post-ramp", "window": f"turn>={post_ramp}", "info_only": True,
                         "per_agent_p50": {f"a{k}": round(v, 1) for k, v in pa_pr.items() if v is not None},
                         "min_agent": round(min(vals_pr), 1), "median": round(med(vals_pr), 1)})
    if not channels:
        return cls("NOT_EVALUATED", channels=[], reason="no retained/steady window had decode readings")
    judged = [c for c in channels if not c.get("info_only")]
    order = {"PASS": 0, "NEAR": 1, "MISS": 2, "NOT_EVALUATED": 3}
    overall = sorted((c["verdict"] for c in judged), key=lambda v: order.get(v, 9), reverse=True)[0]
    return cls(overall, channels=channels, bars={"per_agent_min_tps": bar},
               checks=len(channels) * 1 + len(vals_ret) + len(vals_st))


def score_decode_agg(cell: dict, turns: list[dict], sustained_dir: str | None) -> dict:
    ok = [t for t in turns if t.get("ok")]
    agents = sorted({t.get("agent") for t in ok if t.get("agent") is not None})
    if not agents:
        return cls("NOT_EVALUATED", reason="no ok turns")
    pa = {}
    for ag in agents:
        vals = [float(t["decode_tps"]) for t in ok if t.get("agent") == ag and t.get("decode_tps") is not None]
        pa[ag] = med(vals)
    vals = [v for v in pa.values() if v is not None]
    if not vals:
        return cls("NOT_EVALUATED", reason="no decode rates")
    agg = sum(vals)
    bar = float(cell["decode"]["agg_min_tps"])
    live = None
    if sustained_dir:
        series = load_jsonl(Path(sustained_dir) / "metrics-series.jsonl")
        pts = [(float(s["t"]), float(s["tok_s_mean"])) for s in series if s.get("tok_s_mean") is not None and s.get("t") is not None]
        if pts:
            last_t = max(t for t, _ in pts)
            win = [v for t, v in pts if t >= last_t - 30.0]
            live = med(win if len(win) >= 3 else [v for _, v in pts])
    cross_warn = None
    if live is not None and live > 0 and abs(agg - live) / agg > 0.05:
        cross_warn = f"sum-of-per-agent {agg:.0f} vs live channel last-30s p50 {live:.0f} differ >5%"
    return cls(verdict_of(agg, bar), agg_sum_of_agents_tps=round(agg, 1),
               live_channel_p50_tps=round(live, 1) if live is not None else None,
               live_channel_read="last-30s p50 of /metrics tok_s_mean (recorded convention)",
               bar_tps=bar, cross_check_warn=cross_warn, checks=1 + len(vals))


def score_ttft(cell: dict, streak: list[dict]) -> dict:
    ok = [r for r in streak if r.get("ok")]
    ttfts = [float(r["ttft_client_s"]) for r in ok if r.get("ttft_client_s") is not None]
    if not ttfts:
        return cls("NOT_EVALUATED", reason="no client TTFT readings in the burst/streak leg")
    bound = float(cell["ttft_fresh_bound_s"])
    p50 = med(ttfts)
    return cls("PASS" if p50 <= bound else "MISS", p50_s=round(p50, 2),
               min_s=round(min(ttfts), 2), max_s=round(max(ttfts), 2), bound_s=bound,
               checks=1 + len(ttfts))


def score_integrity(gates_path: str | None) -> dict:
    if not gates_path:
        return cls("NOT_EVALUATED", reason="no gates.json provided (G1 spots + fault scan + duty)")
    try:
        g = json.loads(Path(gates_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return cls("NOT_EVALUATED", reason=f"gates file unreadable: {e}")
    g1 = g.get("g1") or g.get("G1") or {}
    rc = g1.get("rc")
    if rc is None:
        # recorded lane shape: {"G1": {"bootA-bootB": 0, ...}} - rc is max of arms' comparisons
        try:
            rc = max(int(v) for v in g1.values())
        except (TypeError, ValueError):
            rc = None
    diff = g1.get("nonexcluded_diff", g1.get("diff", None))
    faults = g.get("faults")
    if isinstance(faults, dict):
        f_total = sum(int(v or 0) for v in faults.values())
    elif isinstance(faults, list):
        f_total = len(faults)
    elif isinstance(faults, (int, float)):
        f_total = int(faults)
    else:
        f_total = None
    mov = g.get("mem") or {}
    checks = 2
    problems = []
    if rc is None:
        problems.append("no G1 rc readable")
    elif int(rc) != 0:
        problems.append(f"G1 compare rc={rc}")
    if diff is True:
        problems.append("non-excluded G1 DIFF present")
    if f_total is None:
        problems.append("no fault count readable")
    elif f_total != 0:
        problems.append(f"fault lines: {f_total}")
    for k in ("rss_span_mib", "gpu_span_mib"):
        v = mov.get(k)
        if v is not None:
            checks += 1
            lim = 512 if k == "rss_span_mib" else 256
            if float(v) > lim:
                problems.append(f"{k}={v} over {lim}")
    if problems:
        return cls("FAIL", problems=problems, g1_rc=rc, faults=f_total, checks=checks)
    return cls("PASS", g1_rc=rc, faults=f_total, checks=checks)


def score_cell(cell_name: str, env: dict, streak_dir: str | None, sustained_dir: str | None,
               gates_path: str | None) -> dict:
    cell = cell_lookup(env, cell_name)
    streak = burst_rows(streak_dir)
    turns = sustained_rows(sustained_dir)
    if not streak and not turns:
        return {"status": "REFUSED", "reason": "no requests.jsonl (streak/burst leg) and no turns.jsonl (sustained leg) found",
                "cell": cell_name, "checks": 0}

    classes = {
        "shape": score_shape(cell, env, streak, turns),
        "prefill": score_prefill(cell, streak),
        "decode_deep_streak": score_streak_decode(cell, streak),
        "decode_sustained": score_sustained_decode(cell, turns, env),
        "decode_agg": score_decode_agg(cell, turns, sustained_dir),
        "ttft_fresh": score_ttft(cell, streak),
        "fairness": (lambda r: cls(r["verdict"] if r["verdict"] != "REFUSED" else "NOT_EVALUATED",
                                   detail=r, gaps=r.get("gaps", []), checks=r.get("checks", 0),
                                   reason=r.get("reason")))(fair_mod.analyze(sustained_dir, streak_dir, int(cell["n"]), env)),
        "retention": (lambda r: cls(r.get("verdict") if r.get("status") == "OK" else "NOT_EVALUATED",
                                    detail=r, checks=r.get("checks", 0), reason=r.get("reason")))(ret_mod.analyze(sustained_dir, env)),
        "integrity": score_integrity(gates_path),
    }
    checks = sum(int(c.get("checks", 1)) for c in classes.values())
    if checks == 0:
        return {"status": "REFUSED", "reason": "zero checks executed", "cell": cell_name, "checks": 0}

    blockers = [f"{k}:{c['verdict']}" for k, c in classes.items()
                if c["verdict"] not in ("PASS", "INFO")]
    falsify = False
    fals = env.get("falsifiers") or {}
    agg = classes["prefill"].get("agg_used_tps")
    if int(cell["n"]) == 8 and agg is not None and agg <= float(fals.get("agg_flat_n8_prefill_max_tps", 5400)):
        falsify = True
    if falsify:
        verdict = "FALSIFY"
    elif not blockers:
        verdict = "CONFIRM"
    else:
        verdict = "NOT_CONFIRM"
    gaps = []
    for k, c in classes.items():
        if c["verdict"] == "NOT_EVALUATED":
            gaps.append(f"{k}: {c.get('reason', 'not evaluated')}")
        for g in (c.get("gaps") or []):
            if g not in gaps:
                gaps.append(g)
    return {
        "schema": "strata-acceptance-score/1",
        "cell": cell_name, "n": int(cell["n"]), "rung": cell.get("rung"),
        "shape_declared_engine_tokens": cell["ctx_engine"],
        "verdict": verdict, "falsify": falsify, "blockers": blockers, "gaps": gaps,
        "classes": classes, "checks": checks,
        "authority": "master-plan 11.5 + lane-sglang-target section 3 (envelope.json)",
    }


def selftest() -> int:
    """Tiny inline battery - the full fixture battery is selftest_acceptance.sh."""
    checks = 0
    ok = True

    def want(name, cond):
        nonlocal checks, ok
        checks += 1
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("score_envelope selftest:")
    env = load_envelope(None)
    want("envelope loads with all six cells",
         sorted(env["cells"]) == sorted(["5x87.5K", "5x128K", "8x87.5K", "8x128K", "16x87.5K", "16x128K"]))
    want("cell bars: 5x87.5K prefill 1500/7500 decode 50/250",
         env["cells"]["5x87.5K"]["prefill"]["per_agent_wall_min_tps"] == 1500
         and env["cells"]["5x87.5K"]["prefill"]["agg_min_tps"] == 7500
         and env["cells"]["5x87.5K"]["decode"]["per_agent_min_tps"] == 50
         and env["cells"]["5x87.5K"]["decode"]["agg_min_tps"] == 250)
    want("factor 0.729 and est caps consistent",
         abs(env["calibration"]["est_to_engine_factor"] - 0.729) < 1e-9
         and env["cells"]["5x87.5K"]["est_ctx_cap"] == 120000
         and env["cells"]["5x128K"]["est_ctx_cap"] == 175600)
    r = score_cell("5x87.5K", env, None, None, None)
    want("no data: REFUSED with zero checks", r["status"] == "REFUSED" and r["checks"] == 0)
    print(f"checks run: {checks}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell", default=None)
    ap.add_argument("--streak-dir", default=None, help="burst/deep-streak leg receipt dir (requests.jsonl)")
    ap.add_argument("--sustained-dir", default=None, help="sustained leg receipt dir (turns.jsonl)")
    ap.add_argument("--arm-dir", default=None, help="receipt root with streak/ sustained/ subdirs")
    ap.add_argument("--gates", default=None, help="gates.json (G1 spots + faults + mem spans)")
    ap.add_argument("--envelope", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", default=None, help="write the score JSON to this path")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.cell:
        print("score_envelope: --cell is required", file=sys.stderr)
        return 4
    streak_dir, sustained_dir = a.streak_dir, a.sustained_dir
    if a.arm_dir:
        streak_dir = streak_dir or str(Path(a.arm_dir) / "streak")
        sustained_dir = sustained_dir or str(Path(a.arm_dir) / "sustained")
    try:
        env = load_envelope(a.envelope)
        res = score_cell(a.cell, env, streak_dir, sustained_dir, a.gates)
    except KeyError as e:
        print(f"score_envelope: {e}", file=sys.stderr)
        return 4
    if res.get("status") == "REFUSED":
        print(f"score_envelope: REFUSED - {res['reason']}", file=sys.stderr)
        return 3
    if a.write:
        Path(a.write).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print("=" * 88)
        print(f"CELL {res['cell']} (N={res['n']}, rung {res['rung']})  ->  {res['verdict']}"
              + ("   [PRE-REGISTERED FALSIFIER FIRED]" if res["falsify"] else ""))
        print("=" * 88)
        for name, c in res["classes"].items():
            v = c["verdict"]
            extra = ""
            if name == "prefill":
                extra = f"per-agent min {c.get('per_agent_min')}/{c.get('per_agent_bar_tps')} | agg {c.get('agg_used_tps')}/{c.get('agg_bar_tps')}"
            elif name == "decode_deep_streak":
                extra = f"min {c.get('min_agent')} median {c.get('median')}/{c.get('bar_tps')}"
            elif name == "decode_sustained":
                extra = " | ".join(f"{ch['channel']}: min {ch.get('min_agent')} med {ch.get('median')}"
                                   for ch in c.get("channels", []) if not ch.get("info_only"))
            elif name == "decode_agg":
                extra = f"agg {c.get('agg_sum_of_agents_tps')}/{c.get('bar_tps')} (live {c.get('live_channel_p50_tps')})"
            elif name == "ttft_fresh":
                extra = f"p50 {c.get('p50_s')}s (max {c.get('max_s')}) bound {c.get('bound_s')}s"
            elif name == "shape":
                extra = f"observed {c.get('observed_min')}..{c.get('observed_max')} vs {c.get('declared_engine_tokens')}"
            elif name in ("fairness", "retention", "integrity"):
                extra = "; ".join(c.get("problems", [])) if c.get("problems") else (c.get("reason", "") or "")
            print(f"  {name:<20} {v:<14} {extra}")
        if res["blockers"]:
            print(f"blockers: {', '.join(res['blockers'])}")
        if res["gaps"]:
            print("gaps (not evaluated - block CONFIRM):")
            for g in res["gaps"]:
                print(f"  - {g}")
        print(f"checks run: {res['checks']}")
    return {"CONFIRM": 0, "NOT_CONFIRM": 1, "FALSIFY": 2}.get(res["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
