#!/usr/bin/env python3
"""token_calib.py - engine-token calibration for the N-scaling acceptance harness.

The kit instruments size prompts in EST tokens (client chars/4); this model's tokenizer lands
at est * ~0.729 ENGINE tokens (measured twice on the coder IQ1_M pack: est 120000 -> engine
87,473..87,852).  Declared shapes are ENGINE tokens (master plan 11.5) and every acceptance run
asserts its landed engine prompt counts within +-2 percent of the declared target.

Modes:
  size:   token_calib.py --engine-tokens 87500 [--factor 0.729] [--envelope PATH]
          -> prints the est-space size to hand the load kit (engine_target / factor), rounded.
  check:  token_calib.py --check --declared 87500 (--observed N | --receipt DIR) [--tol 2]
          -> PASS iff every observed engine prompt count is within tol percent of declared.
             --receipt reads requests.jsonl (streak/burst legs) or turns.jsonl (sustained leg).
  selftest: token_calib.py --selftest
          -> fixture battery with a printed check count; exit 0 PASS / 1 FAIL.

Exit codes: 0 PASS / 1 FAIL (check) / 3 REFUSED (no data to check) / 4 input error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_FACTOR = 0.729
DEFAULT_TOL = 2.0


def envelope_path_default() -> Path:
    return Path(__file__).resolve().parent / "envelope.json"


def load_factor(envelope: str | None) -> float:
    p = Path(envelope) if envelope else envelope_path_default()
    try:
        env = json.loads(p.read_text(encoding="utf-8"))
        f = float((env.get("calibration") or {}).get("est_to_engine_factor"))
        return f if f > 0 else DEFAULT_FACTOR
    except (OSError, ValueError, TypeError):
        return DEFAULT_FACTOR


def est_size(engine_tokens: float, factor: float) -> int:
    if factor <= 0:
        raise ValueError("factor must be > 0")
    return int(round(engine_tokens / factor))


def observed_from_receipt(receipt: Path) -> list[int]:
    """Engine prompt token counts from a leg receipt dir (requests.jsonl or turns.jsonl)."""
    out: list[int] = []
    rp = receipt / "requests.jsonl"
    if rp.exists():
        for ln in rp.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            v = r.get("prompt_tokens")
            if v is None:
                v = r.get("ctx_engine")
            if v is not None:
                out.append(int(v))
    tp = receipt / "turns.jsonl"
    if tp.exists():
        for ln in tp.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                t = json.loads(ln)
            except ValueError:
                continue
            v = t.get("prompt_tokens")
            if v is None:
                v = t.get("ctx_engine_tokens")
            if v is not None:
                out.append(int(v))
    return out


def check_values(declared: float, observed: list[int], tol_pct: float) -> dict:
    if not observed:
        return {"ok": None, "reason": "no observed engine prompt counts"}
    lo = declared * (1.0 - tol_pct / 100.0)
    hi = declared * (1.0 + tol_pct / 100.0)
    within = [v for v in observed if lo <= v <= hi]
    out_of = [v for v in observed if not (lo <= v <= hi)]
    return {"ok": not out_of, "declared": declared, "tol_pct": tol_pct,
            "window": [round(lo, 1), round(hi, 1)], "n": len(observed),
            "min": min(observed), "max": max(observed), "within": len(within),
            "out_of_tol": out_of[:10], "checks": 1 + len(observed)}


def selftest() -> int:
    checks = 0
    ok = True

    def want(name: str, cond: bool) -> None:
        nonlocal checks, ok
        checks += 1
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("token_calib selftest:")
    f = load_factor(None)
    want("default factor loads from envelope (0.729)", abs(f - 0.729) < 1e-9)
    want("size: 87500 engine -> 120027 est", est_size(87500, f) == 120027)
    want("size: 128000 engine -> 175583 est", est_size(128000, f) == 175583)
    try:
        est_size(100, 0.0)
        want("zero factor raises", False)
    except ValueError:
        want("zero factor raises", True)
    # positive fixture: est 120000 -> engine 87473..87852 lands inside +-2% of 87500
    r = check_values(87500, [87473, 87539, 87852, 87700], 2.0)
    want("check: recorded spread passes at 87500 +-2%", r["ok"] is True and r["out_of_tol"] == [])
    # negative fixture: a legacy est-space shape (~89600) must FAIL 87500 +-2%
    r2 = check_values(87500, [89600, 89500], 2.0)
    want("check: legacy 89.6K shape FAILS the 87.5K declare", r2["ok"] is False)
    # negative fixture: declared 128K, observed 89.6K (the known landmine) must FAIL
    r3 = check_values(128000, [89600], 2.0)
    want("check: landmine (est-sized 128K -> 89.6K engine) FAILS", r3["ok"] is False)
    # no data -> ok None (refusal signal), never a green
    r4 = check_values(87500, [], 2.0)
    want("check: zero observations = refusal signal, not PASS", r4["ok"] is None)
    print(f"checks run: {checks}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine-tokens", type=float, default=None, help="declared engine-token target (size mode)")
    ap.add_argument("--factor", type=float, default=None)
    ap.add_argument("--envelope", default=None, help="path to envelope.json (factor source)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--declared", type=float, default=None)
    ap.add_argument("--observed", type=int, default=None)
    ap.add_argument("--receipt", default=None, help="leg receipt dir to read engine prompt counts from")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    factor = a.factor if a.factor else load_factor(a.envelope)

    if a.check:
        if a.declared is None:
            print("token_calib: --check needs --declared", file=sys.stderr)
            return 4
        if a.observed is not None:
            observed = [a.observed]
        elif a.receipt:
            observed = observed_from_receipt(Path(a.receipt))
        else:
            print("token_calib: --check needs --observed or --receipt", file=sys.stderr)
            return 4
        res = check_values(a.declared, observed, a.tol)
        if res["ok"] is None:
            print(f"token_calib: REFUSED - {res['reason']}", file=sys.stderr)
            return 3
        if a.json:
            print(json.dumps(res, indent=1))
        else:
            verdict = "PASS" if res["ok"] else "FAIL"
            print(f"shape calibration: {verdict} (declared {a.declared:g} engine tokens +-{a.tol:g}%; "
                  f"observed n={res['n']} min={res['min']} max={res['max']}; window {res['window']})")
            if res["out_of_tol"]:
                print(f"  OUT OF TOL: {res['out_of_tol']}")
            print(f"checks run: {res['checks']}")
        return 0 if res["ok"] else 1

    if a.engine_tokens is None:
        ap.print_help()
        return 4
    est = est_size(a.engine_tokens, factor)
    if a.json:
        print(json.dumps({"engine_target": a.engine_tokens, "factor": factor, "est_tokens": est}))
    else:
        print(f"engine target {a.engine_tokens:g} / factor {factor:g} -> est tokens {est}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
