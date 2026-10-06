#!/usr/bin/env python3
"""flag_owner_gate.py - single-owner gate for serving values (no value twice).

A serving value must have exactly ONE owner: either a CLI flag or an
environment gate.  This gate enforces that for the values that historically
had BOTH (a flag and a STRATA_* env override for the same value), and scans
for new flag/env mirror pairs.

Checks (each counted; the gate refuses to pass vacuously):
  A. Cross-check: every value in KILLED_ENV must be owned by its CLI flag and
     must NOT still be read from the environment anywhere in engine source.
     Convention: each env probe must find its own file to be a check that ran.
  B. The CLI flag for each value is still parsed in src/program/generate.cpp.
  C. Name-mirror scan: a getenv("STRATA_X") whose normalized name (lowercase,
     '_'->'-', STRATA_ stripped) equals a parsed --x flag name is a second
     owner unless allowlisted with a written reason.

Usage: flag_owner_gate.py [--root REPO_ROOT]
Exit: 0 pass ('checks run: N'); 1 any failure (each named); 3 refusal (bad root).
"""
import argparse
import re
import sys
from pathlib import Path

# Historical dual-owner values: env name -> the flag that now owns the value.
KILLED_ENV = {
    "STRATA_NANOBATCH": "--nanobatch",
    "STRATA_SPEC_MIN_P": "--spec-min-p",
    "STRATA_ADAPT_EVERY": "--adapt-every",
    "STRATA_ADAPT_SWAPS": "--adapt-swaps",
    "STRATA_CONCURRENT_CACHE_MIB": "--conversation-cache-mib",
}

# Deliberate exceptions to the mirror scan, each with a written reason.
ALLOWED_MIRRORS = {
    "STRATA_PLE_GGUF": "test-fixture fallback in kernels/ple_parity.cpp; not a serve value",
}

SRC_DIRS = ("src", "include", "serve")
SRC_GLOBS = ("*.cpp", "*.hpp", "*.h", "*.cu", "*.cuh", "*.py")

GETENV_RE = re.compile(r'getenv\(\s*"([A-Za-z_0-9]+)"')
FLAG_RE = re.compile(r'"(--[a-z0-9][a-z0-9-]*)"')


def iter_sources(root: Path):
    for d in SRC_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for pat in SRC_GLOBS:
            for f in base.rglob(pat):
                if any(part in ("__pycache__", "third_party") for part in f.parts):
                    continue
                yield f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="repo root (default: derived from script path)")
    args = ap.parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]

    if not (root / "src").is_dir() or not (root / "include").is_dir():
        print(f"REFUSED: {root} does not look like the engine tree (no src/ or include/)")
        return 3

    total = 0
    fails = 0

    def check(name: str, ok: bool) -> None:
        nonlocal total, fails
        total += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails += 1

    sources = list(iter_sources(root))
    check(f"source files scanned ({len(sources)})", len(sources) > 0)
    if total == 1 and fails == 1:
        return 1

    # --- A. killed env names are gone from engine source -------------------
    texts = {}
    for f in sources:
        try:
            texts[f] = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    alive = {env: [str(f.relative_to(root)) for f, t in texts.items() if env in t] for env in KILLED_ENV}
    for env, hits in alive.items():
        check(f"A: {env} has no remaining env read (owner: {KILLED_ENV[env]})", not hits)
        if hits:
            print(f"        still read in: {', '.join(sorted(hits)[:5])}")

    # --- B. each owning flag is still parsed ------------------------------
    gen = texts.get(root / "src" / "program" / "generate.cpp", "")
    if not gen:
        gen = next((t for f, t in texts.items() if f.name == "generate.cpp"), "")
    for env, flag in KILLED_ENV.items():
        check(f"B: {flag} still parsed in generate.cpp", f'"{flag}"' in gen)

    # --- C. mirror scan -----------------------------------------------------
    parsed_flags = set(FLAG_RE.findall(gen)) if gen else set()
    flag_names = {f[2:] for f in parsed_flags}
    env_names = set()
    env_hits = {}
    for f, t in texts.items():
        for m in GETENV_RE.finditer(t):
            env_names.add(m.group(1))
            env_hits.setdefault(m.group(1), str(f.relative_to(root)))
    mirrors = []
    for env in sorted(env_names):
        if env in KILLED_ENV or env in ALLOWED_MIRRORS:
            continue
        base = env
        if base.startswith("STRATA_"):
            base = base[len("STRATA_"):]
        norm = base.lower().replace("_", "-")
        if norm in flag_names:
            mirrors.append((env, f"--{norm}", env_hits[env]))
    check("C: no new flag/env mirror pairs (allowlist: " + ", ".join(sorted(ALLOWED_MIRRORS)) + ")", not mirrors)
    for env, flag, where in mirrors:
        print(f"        mirror: {env} ({where}) duplicates {flag}")

    print(f"checks run: {total}")
    if fails:
        print(f"flag_owner_gate: FAIL ({fails} of {total} checks)")
        return 1
    print("flag_owner_gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
