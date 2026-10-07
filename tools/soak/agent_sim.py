#!/usr/bin/env python3
"""agent_sim.py - sustained 5-agent turn-taking traffic against the running Strata server.

Wave baseline primary: 5 agents x 128K (defaults: --agents 5 --ctx-cap 122880
--steady-min-ctx 80000). The 8 x ~100K stretch stays available as an explicit
option (--agents 8 --ctx-cap 96000 --steady-min-ctx 60000).

    python3 agent_sim.py --url http://127.0.0.1:8096 --model strata-coder \
        --serve-log /home/mhd67/strata-serving-lab/serve-8096.log \
        --engine-log /home/mhd67/strata-serving-lab/serve-combined-8-engine.log \
        --agents 5 --turns 64 --minutes 45 --max-tokens 256 --ctx-cap 122880 \
        --msg-tokens 2500 --seed 20261001 [--dry-run]

WHAT IT SIMULATES.  N agents (default 5) each keep their own growing conversation.  Every turn an
agent appends one synthetic user message (deterministic seeded corpus) and sends the FULL
conversation through the public loopback route (POST /v1/chat/completions, stream=true, greedy
temperature=0, fixed seed, bounded max_tokens).  A turn's reply is appended to that agent's
conversation, so contexts climb toward --ctx-cap; once a conversation would exceed the cap the
oldest exchanges are dropped (sliding window) so the load stays at the cap for the rest of the
run.  Agents fire their next turn the moment their previous one completes, so with N agents the
server carries N concurrent streaming requests continuously - the sustained shape the wave
target needs (primary 5x50=250 tok/s; 8-agent stretch 8x50=400 tok/s),
not a single-load window.

METRICS (one JSON object per completed turn in turns.jsonl; nothing modelled):
  decode_tps      the server's own done-line rate: "[strata request N] done: X tokens in Y s
                  (Z tok/s)" - decode-only (first token -> last), the SAME metric every sibling
                  kit (lane-perf-matrix, lane-verify-harness) scores on.
  prefill_s       the last "reading the prompt: ... <s> s so far" line for that request.
  prefill_tps     engine prompt tokens / prefill_s (effective rate: includes any prefix reuse).
  ctx_engine      the engine's own prompt token count for the turn (len(ids)), read from
                  /metrics active-request state while the turn is in flight; ctx_est is the
                  client-side chars/4 estimate.  The steady-window classification uses the
                  engine count when present, else the estimate (recorded per turn as ctx_basis).
  client side     wall_s (HTTP), ttft_client_s (first SSE content/reasoning delta), stream span
                  and event count (events/s is NOT tok/s - never scored).
  cross-checks    done-line "expert cache X% hit", plus (when /metrics history matches the turn)
                  the engine's own prompt_ms/decode_ms/decode_tok_s/reused counters.

HEADLINE (summary.md): sustained decode tok/s per agent = p50/p90 over steady turns (after
warmup), the aggregate over the steady window (sum of generated tokens / window wall - includes
prefill, so it is the conservative "sustained" reading), and the per-turn prefill cost.

STALL GUARDS (fail-fast, soak-kit autopsy): per-turn hard bound (--turn-bound, default 300 s),
soft stall when neither the SSE stream nor ANY progress channel advances for --stall-s (default
90 s) - the request's serve-log lines, raw serve-log growth, and the tracked /metrics fields all
count (a deep prefill legitimately holds the SSE stream empty for minutes while the engine reads),
and a global "no completed turn AND no progress channel advance for --global-stall s" guard. The
FIRST failure is terminal: the run stops, captures an autopsy (serve-log tail, engine-log tail,
watchdog lines, /metrics, ps, host/GPU) under the results dir, writes summary.md with FAIL, and
exits 2.

OUTPUT  results/agent-sim-<stamp>/{turns.jsonl, summary.md, meta.json, metrics-series.jsonl,
        metrics-end.json, fail-autopsy-*/}.  Stdout ends with "RESULTS_DIR=<path>" (the wrapper
        greps it).  This script NEVER starts, stops or restarts the server; it only sends
        requests and reads the logs it is given.  Exit: 0 completed, 2 fatal (autopsy), 3
        preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SERVE_LOG = "/home/mhd67/strata-serving-lab/serve-8096.log"
DEFAULT_ENGINE_LOG = "/home/mhd67/strata-serving-lab/serve-combined-8-engine.log"
DEFAULT_MODEL = "strata-coder"

ID_RE = re.compile(r"^\[strata(?:-agx)? request (\d+)\]")
DONE_RE = re.compile(r"^\[strata(?:-agx)? request (\d+)\] done: (\d+) tokens in (\d+) s \(([0-9.]+) tok/s\) "
                     r"\((\w+), cancel=(\w+)\)(?:, expert cache ([0-9.]+)% hit)?")
READ_RE = re.compile(r"^\[strata(?:-agx)? request (\d+)\] reading the prompt: ([\d,]+)(?: of ([\d,]+))? tokens, "
                     r"(\d+) s so far")
WATCH_RE = re.compile(r"watchdog|issue #29|no progress|stall|abort|killed", re.IGNORECASE)

# ---------------------------------------------------------------- deterministic corpus

WORDS = ("bucket batches sorted summit marked circuit signal window anchor ledger frame cursor "
         "packet digest stream buffer shard vector matrix tensor socket channel thread queue "
         "round trip latency budget capture probe sample record review mirror target measure "
         "plateau valley ridge canyon delta epsilon gamma sigma lambda theta kappa rho tau "
         "chiseled tempered woven braided coiled folded layered nested stacked chained linked "
         "aligned pinned parked resumed paused staged lifted shifted rotated scaled clipped "
         "quiet steady rugged hollow narrow wide shallow steep gentle brisk patient silent "
         "amber cobalt copper silver golden crimson violet indigo teal slate basalt granite "
         "cedar birch maple willow aspen juniper larch poplar cypress tamarisk rowan alder").split()


def make_message(seed: int, agent: int, turn: int, target_tokens: int) -> str:
    """One deterministic synthetic user message (~target_tokens est, chars/4)."""
    rng = random.Random(f"corpus:{seed}:{agent}:{turn}")
    header = f"[agent {agent} turn {turn}] Continue the sustained review. "
    tail = "\nReply with a short status line for this turn."
    target_chars = max(200, target_tokens * 4) - len(header) - len(tail)
    parts: list[str] = []
    n = 0
    while n < target_chars:
        words = [rng.choice(WORDS) for _ in range(rng.randint(8, 14))]
        sentence = words[0].capitalize() + " " + " ".join(words[1:]) + "."
        parts.append(sentence)
        n += len(sentence) + 1
    return header + " ".join(parts) + tail


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------- log watcher

class LogWatcher:
    """Incremental reader of the serve log: new request ids, done lines, prefill lines, liveness.

    Liveness hardening (a growing log must never go quiet under this reader):
      * the cursor resyncs from 0 when the file is truncated or replaced (server restart /
        copytruncate / logrotate) instead of seeking past EOF and reading nothing;
      * a line split across two polls is carried in `pending` until its newline arrives, so a
        reading-line reprint is never silently dropped;
      * `touch()` / `size_change_at()` are a parse-free, cursor-free raw-growth channel: the wall
        time the serve log last grew, valid even if line parsing breaks.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.off = self.path.stat().st_size if self.path.exists() else 0
        self.ids: list[int] = []
        self.done: dict[int, dict] = {}
        self.reading: dict[int, dict] = {}
        self.last_line: dict[int, float] = {}
        self.lock = threading.Lock()
        self.pending = b""                      # bytes of a line split across polls
        self.last_size_change_at = time.time()  # last observed raw size GROWTH (any writer)
        self._size = self.off
        self._ino = self.path.stat().st_ino if self.path.exists() else None

    def touch(self) -> None:
        """Parse-free liveness probe: record when the raw file size grows (works even if the
        cursor or the line parsing is somehow broken)."""
        with self.lock:
            try:
                st = self.path.stat()
            except OSError:
                return
            if self._ino is None:
                self._ino = st.st_ino
            if st.st_size > self._size:
                self.last_size_change_at = time.time()
            self._size = st.st_size

    def size_change_at(self) -> float:
        with self.lock:
            return self.last_size_change_at

    def poll(self) -> None:
        with self.lock:
            if not self.path.exists():
                return
            try:
                st = self.path.stat()
                if st.st_size < self.off or (self._ino not in (None, 0) and st.st_ino not in (0, self._ino)):
                    self.off = 0                 # truncated / replaced: restart from the top
                    self.pending = b""
                    self._size = st.st_size
                self._ino = st.st_ino
                with open(self.path, "rb") as f:
                    f.seek(self.off)
                    chunk = f.read()
            except OSError:
                return
            if not chunk:
                return
            self.off += len(chunk)
            self._size = self.off
            now = time.time()
            self.last_size_change_at = now
            data = self.pending + chunk
            self.pending = b""
            cut = data.rfind(b"\n")
            if cut < 0:
                self.pending = data              # no complete line yet; keep it for the next poll
                return
            self.pending = data[cut + 1:]
            for ln in data[:cut + 1].decode("utf-8", "replace").split("\n"):
                m = ID_RE.match(ln)
                if not m:
                    continue
                rid = int(m.group(1))
                if rid not in self.ids:
                    self.ids.append(rid)
                self.last_line[rid] = now
                m = DONE_RE.match(ln)
                if m:
                    self.done[int(m.group(1))] = {
                        "tokens": int(m.group(2)), "seconds": int(m.group(3)), "rate": float(m.group(4)),
                        "finish": m.group(5), "cancelled": m.group(6), "cache_hit_pct": m.group(7)}
                m = READ_RE.match(ln)
                if m:
                    self.reading[int(m.group(1))] = {
                        "read_so_far": int(m.group(2).replace(",", "")),
                        "total_hint": int(m.group(3).replace(",", "")) if m.group(3) else None,
                        "seconds": int(m.group(4))}

    def id_count(self) -> int:
        with self.lock:
            return len(self.ids)

    def id_at(self, i: int):
        with self.lock:
            return self.ids[i] if i < len(self.ids) else None

    def done_for(self, rid):
        with self.lock:
            return dict(self.done.get(rid) or {}) or None

    def reading_for(self, rid):
        with self.lock:
            return dict(self.reading.get(rid) or {}) or None

    def last_line_at(self, rid):
        with self.lock:
            return self.last_line.get(rid)


# ---------------------------------------------------------------- metrics poller

class MetricsPoller(threading.Thread):
    """1 s cadence /metrics snapshots: per-request engine state, live rates, totals, series file.

    Also tracks WHEN any tracked progress field last advanced (`advanced_at()`), and each series
    row records the raw live+totals fields the runners key on (prompt_tokens, prompt_read,
    prompt_total, generated, totals.prompt_tokens, totals.output_tokens), so a deep prefill leaves
    a readable record even while the decode counters are frozen - and the liveness check has a real
    prefill channel once the server exposes prompt_read/prompt_total.
    """

    def __init__(self, url: str, series_path: Path, interval: float = 1.0):
        super().__init__(daemon=True)
        self.url = url
        self.series_path = series_path
        self.interval = interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.seen: dict[int, dict] = {}      # rid -> best observed state (first non-null prompt_tokens)
        self.last_live: dict = {}
        self.last_totals: dict = {}
        self.engine: dict = {}
        self.errors = 0
        self.last_change_at = time.time()    # last advance of any tracked progress field
        self._prev_progress = None

    @staticmethod
    def progress_key(live: dict, totals: dict) -> tuple:
        """The prefill+decode progress fields: whichever advances counts as progress."""
        return (live.get("generated"), live.get("prompt_tokens"), live.get("prompt_read"),
                live.get("prompt_total"), totals.get("output_tokens"), totals.get("prompt_tokens"))

    def advanced_at(self) -> float:
        """Wall time any tracked progress field last moved (decode OR prefill channel)."""
        with self.lock:
            return self.last_change_at

    def _fetch(self, path: str, timeout: float = 5.0):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read())

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                m = self._fetch("/metrics")
            except (OSError, ValueError):
                with self.lock:
                    self.errors += 1
                self.stop.wait(self.interval)
                continue
            now = time.time()
            live = m.get("live") or {}
            totals = m.get("totals") or {}
            with self.lock:
                if m.get("engine"):
                    self.engine = m["engine"]
                self.last_live = live
                self.last_totals = totals
                key = self.progress_key(live, totals)
                if (self._prev_progress is None or key != self._prev_progress) \
                        and any(v is not None for v in key):
                    self._prev_progress = key
                    self.last_change_at = now
                for st in m.get("active_requests") or []:
                    rid = st.get("request_id")
                    if rid is None:
                        continue
                    prev = self.seen.get(rid) or {}
                    merged = dict(prev)
                    for k in ("prompt_tokens", "max_tokens", "phase", "generated", "started", "first_token"):
                        if st.get(k) is not None:
                            merged.setdefault(k, st.get(k))
                    self.seen[rid] = merged
            try:
                with open(self.series_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": round(now, 1), "state": live.get("state"),
                                        "active": live.get("active_requests"), "queued": live.get("queued"),
                                        "tok_s": live.get("tok_s"), "tok_s_mean": live.get("tok_s_mean"),
                                        "generated": live.get("generated"),
                                        "prompt_tokens": live.get("prompt_tokens"),
                                        "prompt_read": live.get("prompt_read"),
                                        "prompt_total": live.get("prompt_total"),
                                        "totals": {"prompt_tokens": totals.get("prompt_tokens"),
                                                   "output_tokens": totals.get("output_tokens")}},
                                       ensure_ascii=False) + "\n")
            except OSError:
                pass
            self.stop.wait(self.interval)

    def state_for(self, rid):
        with self.lock:
            return dict(self.seen.get(rid) or {}) or None

    def snapshot(self) -> dict:
        with self.lock:
            return {"engine": dict(self.engine), "live": dict(self.last_live),
                    "totals": dict(self.last_totals), "errors": self.errors}


# ---------------------------------------------------------------- helpers

def get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def sh(cmd: str, timeout: int = 12) -> str:
    import subprocess
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:  # noqa: BLE001
        return f"(cmd failed: {e})"


def tail_lines(path: Path, n: int) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError as e:
        return [f"(cannot read {path}: {e})"]


def autopsy(tag: str, reason: str, serve_log: Path, engine_log: Path, outdir: Path, url: str) -> Path:
    d = outdir / f"fail-autopsy-{tag}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "reason.txt").write_text(reason + "\n", encoding="utf-8")
    (d / "serve-log-tail.txt").write_text("\n".join(tail_lines(serve_log, 120)) + "\n", encoding="utf-8")
    (d / "engine-log-tail.txt").write_text("\n".join(tail_lines(engine_log, 160)) + "\n", encoding="utf-8")
    wd = [ln for ln in tail_lines(engine_log, 1200) if WATCH_RE.search(ln)]
    (d / "engine-watchdog-lines.txt").write_text("\n".join(wd) + "\n", encoding="utf-8")
    (d / "metrics.json").write_text(sh(f"curl -s --max-time 8 {url}/metrics"), encoding="utf-8")
    (d / "ps.txt").write_text(sh("ps -eo pid,etime,rss,cmd | grep '[b]uild/strata'"), encoding="utf-8")
    (d / "host.txt").write_text(sh("free -g; nvidia-smi --query-gpu=index,memory.used,utilization.gpu,"
                                   "temperature.gpu,power.draw --format=csv,noheader"), encoding="utf-8")
    return d


def pct(values: list[float], q: float):
    """simple order-statistic percentile (n small; documented, not interpolated)."""
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, max(0, int(math.ceil(q * len(v))) - 1))]


def med(values: list):
    return statistics.median(values) if values else None


# ---------------------------------------------------------------- run state

class RunState:
    def __init__(self, args, outdir: Path):
        self.args = args
        self.outdir = outdir
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.frozen = False
        self.turns: list[dict] = []
        self.fatal: dict | None = None
        self.inflight = 0
        self.peak_inflight = 0
        self.workers_done = 0

    def record(self, rec: dict) -> None:
        with self.lock:
            self.turns.append(rec)
            if not self.frozen:
                with open(self.outdir / "turns.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def set_fatal(self, reason: str) -> None:
        with self.lock:
            if self.fatal is None:
                self.fatal = {"reason": reason, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.stop.set()

    def enter(self) -> None:
        with self.lock:
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)

    def leave(self) -> None:
        with self.lock:
            self.inflight -= 1


# ---------------------------------------------------------------- one turn

def fire_request(url: str, body: dict, read_timeout: float, fire_lock: threading.Lock, watcher: LogWatcher):
    """Serialize fires, map the new [strata request N] id exactly, return (response, rid)."""
    with fire_lock:
        n0 = watcher.id_count()
        req = urllib.request.Request(url + "/v1/chat/completions",
                                     data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        resp = None
        http_err = None
        try:
            resp = urllib.request.urlopen(req, timeout=read_timeout)
        except urllib.error.HTTPError as e:
            http_err = f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace') if e.fp else e.reason}"
        rid = None
        while time.time() - t0 < 30:
            watcher.poll()
            if watcher.id_count() > n0:
                rid = watcher.id_at(n0)
                break
            time.sleep(0.02)
        if http_err is not None:
            if resp is not None:
                resp.close()
            raise RuntimeError(http_err)
        if rid is None:
            if resp is not None:
                resp.close()
            raise RuntimeError("no [strata request N] line within 30 s of fire")
        return resp, rid


def _revive_quiet_stream(resp) -> bool:
    """After ONE socket timeout CPython marks the response socket unreadable for good
    (SocketIO._timeout_occurred -> OSError('cannot read from timed out object')). A quiet-but-live
    prefill must keep reading, so clear that flag; returns False only when the flag is set and
    cannot be cleared (then the caller must stop)."""
    raw = getattr(getattr(resp, "fp", None), "raw", None)
    if raw is None or not getattr(raw, "_timeout_occurred", False):
        return True
    try:
        raw._timeout_occurred = False
        return True
    except Exception:                      # noqa: BLE001 - unknown object layout: do not continue
        return False


def read_stream(resp, rid, watcher: LogWatcher, turn_deadline: float, stall_s: float,
                poller: "MetricsPoller | None" = None):
    """Read the SSE stream. Returns (content, reasoning, t_first, t_last, events, usage, error).

    A quiet SSE socket is only a stall when NO progress channel moves: the request's serve-log
    lines, raw serve-log growth, or (when a poller is supplied) an advance of the tracked /metrics
    fields - a deep prefill legitimately streams nothing for minutes.
    """
    content: list[str] = []
    reasoning: list[str] = []
    t_first = t_last = None
    events = 0
    usage = None
    err = None
    while True:
        if time.time() > turn_deadline:
            err = f"turn bound exceeded ({turn_deadline:.0f} is past)"
            break
        try:
            raw = resp.readline()
        except socket.timeout:
            watcher.poll()
            watcher.touch()
            now = time.time()
            last = watcher.last_line_at(rid) if rid is not None else None
            fresh = (last is not None and now - last < stall_s) or now - watcher.size_change_at() < stall_s
            if not fresh and poller is not None:
                fresh = now - poller.advanced_at() < stall_s
            if fresh and _revive_quiet_stream(resp):
                continue                       # legit long prefill: a progress channel is moving
            if fresh:
                err = "stream read error: socket unusable after a timeout (cannot read from timed out object)"
                break
            err = f"stall: no SSE data and no progress on any channel for {stall_s:.0f} s"
            break
        except Exception as e:                 # noqa: BLE001 - an instrument must never hang here
            err = f"stream read error: {type(e).__name__}: {e}"
            break
        if not raw:
            err = "stream ended without [DONE]"
            break
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line or line.startswith(":"):
            continue                            # blank or ": keep-alive" SSE comment
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if isinstance(chunk, dict):
            if chunk.get("error"):
                err = "stream error: " + json.dumps(chunk["error"])[:300]
                break
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                c, r = d.get("content"), d.get("reasoning_content")
                if c or r:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    events += 1
                    if c:
                        content.append(c)
                    if r:
                        reasoning.append(r)
    return "".join(content), "".join(reasoning), t_first, t_last, events, usage, err


def run_agent(agent: int, st: RunState, watcher: LogWatcher, poller: MetricsPoller, fire_lock: threading.Lock):
    a = st.args
    conv: list[dict] = []                      # messages sent so far (user/assistant pairs)
    exchanges: list[dict] = []                 # per appended exchange: {user_est, reply_tokens}
    ctx_est = 0                                # best estimate of the CURRENT conversation tokens
    turn = 0
    try:
        while not st.stop.is_set() and turn < a.turns:
            turn += 1
            rng = random.Random(f"len:{a.seed}:{agent}:{turn}")
            max_tokens = rng.choice([a.max_tokens, max(a.max_tokens * 3 // 4, 1), max(a.max_tokens // 2, 1)])
            msg = make_message(a.seed, agent, turn, a.msg_tokens)
            msg_est = est_tokens(msg)
            projected = ctx_est + msg_est         # client estimate of this turn's full prompt
            dropped = 0
            while projected > a.ctx_cap and len(exchanges) > 1:
                ex = exchanges.pop(0)
                conv = conv[2:]
                ctx_est -= ex["user_est"] + ex["reply_tokens"]
                dropped += 1
                projected = ctx_est + msg_est
            trimmed = dropped > 0
            body = {"model": a.model, "messages": conv + [{"role": "user", "content": msg}],
                    "temperature": 0, "seed": 1234, "stream": True, "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False}}
            rec = {"run": st.outdir.name, "agent": agent, "turn": turn, "ok": False, "error": None,
                   "t_fire": round(time.time(), 3), "t_done": None, "wall_s": None, "req_id": None,
                   "ctx_est_tokens": projected, "msg_est_tokens": msg_est, "max_tokens": max_tokens,
                   "trimmed": trimmed, "dropped_exchanges": dropped,
                   "prompt_tokens": None, "prompt_read_hint": None, "gen_tokens": None,
                   "done_seconds": None, "decode_tps": None, "prefill_s": None, "prefill_tps": None,
                   "finish": None, "cancelled": None, "cache_hit_pct": None,
                   "ttft_client_s": None, "stream_span_s": None, "stream_events": None,
                   "content_chars": 0, "reasoning_chars": 0, "usage": None,
                   "engine_prompt_ms": None, "engine_decode_ms": None, "engine_decode_tps": None,
                   "reused_tokens": None, "history_matched": None}
            st.enter()
            try:
                try:
                    resp, rid = fire_request(a.url, body, a.stall_s, fire_lock, watcher)
                except (RuntimeError, OSError, urllib.error.URLError) as e:
                    rec["error"] = f"fire failed: {e}"
                    st.set_fatal(f"agent {agent} turn {turn}: {rec['error']}")
                    return
                rec["req_id"] = rid
                deadline = time.time() + a.turn_bound
                try:
                    content, reasoning, t_first, t_last, events, usage, err = read_stream(
                        resp, rid, watcher, deadline, a.stall_s, poller)
                finally:
                    resp.close()
                now = time.time()
                rec["t_done"] = round(now, 3)
                rec["wall_s"] = round(now - rec["t_fire"], 2)
                rec["stream_events"] = events
                rec["content_chars"] = len(content)
                rec["reasoning_chars"] = len(reasoning)
                rec["usage"] = usage
                if t_first is not None:
                    rec["ttft_client_s"] = round(t_first - rec["t_fire"], 2)
                if t_first is not None and t_last is not None:
                    rec["stream_span_s"] = round(t_last - t_first, 2)
                # wait (bounded) for the server's done line for THIS request id
                tw = time.time()
                done = None
                while time.time() - tw < 20:
                    watcher.poll()
                    done = watcher.done_for(rid)
                    if done:
                        break
                    time.sleep(0.1)
                rd = watcher.reading_for(rid)
                if rd:
                    rec["prefill_s"] = rd["seconds"]
                    rec["prompt_read_hint"] = rd["total_hint"] or rd["read_so_far"]
                stt = poller.state_for(rid)
                if stt and stt.get("prompt_tokens") is not None:
                    rec["prompt_tokens"] = stt["prompt_tokens"]
                if done:
                    rec["gen_tokens"] = done["tokens"]
                    rec["done_seconds"] = done["seconds"]
                    rec["decode_tps"] = done["rate"]
                    rec["finish"] = done["finish"]
                    rec["cancelled"] = done["cancelled"]
                    rec["cache_hit_pct"] = done["cache_hit_pct"]
                    if rec["prefill_s"] and rec["prompt_tokens"]:
                        rec["prefill_tps"] = round(rec["prompt_tokens"] / max(rec["prefill_s"], 1), 1)
                    elif rec["prefill_s"] and rec["prompt_read_hint"]:
                        rec["prefill_tps"] = round(rec["prompt_read_hint"] / max(rec["prefill_s"], 1), 1)
                if err:
                    rec["error"] = err
                if err is None and done is None:
                    rec["error"] = "no done line within 20 s of stream end"
                rec["ok"] = rec["error"] is None and done is not None and done.get("cancelled") != "True"
                # conversation update
                reply = content if content else reasoning
                if rec["ok"] and reply:
                    reply_tokens = rec["gen_tokens"] or est_tokens(reply)
                    conv = conv + [{"role": "user", "content": msg},
                                   {"role": "assistant", "content": reply}]
                    exchanges.append({"user_est": msg_est, "reply_tokens": reply_tokens})
                    if rec["prompt_tokens"] is not None:
                        ctx_est = rec["prompt_tokens"] + reply_tokens     # engine count + this reply
                    else:
                        ctx_est = ctx_est + msg_est + reply_tokens        # estimate chain
                elif rec["ok"]:
                    rec["ok"] = False
                    rec["error"] = "empty reply (no content/reasoning deltas)"
                st.record(rec)
                if not rec["ok"]:
                    st.set_fatal(f"agent {agent} turn {turn}: {rec['error']}")
                    return
            finally:
                st.leave()
            # pacing: next turn fires immediately (sustained N-way concurrency)
    finally:
        with st.lock:
            st.workers_done += 1


# ---------------------------------------------------------------- summary

def build_summary(st: RunState, poller: MetricsPoller, a, started: str, stop_reason: str,
                  totals_before: dict, totals_after: dict) -> str:
    turns = list(st.turns)
    ok = [t for t in turns if t["ok"]]
    steady = [t for t in ok if t.get("steady")]
    agents = sorted({t["agent"] for t in turns})
    lines: list[str] = []
    w = lines.append
    w(f"# agent-sim {st.outdir.name} - sustained {a.agents}-agent traffic")
    w("")
    w(f"started {started} | stop_reason={stop_reason} | url={a.url} model={a.model}")
    w(f"engine: {json.dumps(poller.snapshot()['engine'])[:300]}")
    w("")
    w("## Headline (steady window: engine context >= %d tok, turn > %d warmup)" % (a.steady_min_ctx, a.warmup_turns))
    w("")
    per: dict[int, dict] = {}
    for ag in agents:
        ss = [t for t in steady if t["agent"] == ag]
        rates = [t["decode_tps"] for t in ss if t["decode_tps"] is not None]
        per[ag] = {"n": len(ss), "rates": rates,
                   "p50": statistics.median(rates) if rates else None,
                   "p90": pct(rates, 0.9), "p10": pct(rates, 0.1),
                   "min": min(rates) if rates else None, "max": max(rates) if rates else None,
                   "tokens": sum(t["gen_tokens"] or 0 for t in ss),
                   "prefill_s": [t["prefill_s"] for t in ss if t["prefill_s"] is not None],
                   "prefill_tps": [t["prefill_tps"] for t in ss if t["prefill_tps"] is not None],
                   "ctx": [t["ctx_engine_tokens"] if t.get("ctx_engine_tokens") is not None else t["ctx_est_tokens"]
                           for t in ss]}
    if steady:
        w0 = min(t["t_fire"] for t in steady)
        w1 = max(t["t_done"] for t in steady)
        window = max(w1 - w0, 1e-9)
        agg = sum(t["gen_tokens"] or 0 for t in steady) / window
        w(f"- aggregate sustained (sum of generated tokens / steady window wall): **{agg:.1f} tok/s** "
          f"over {window:.0f} s, {len(steady)} turns")
        w(f"- sum of per-agent p50 decode: {sum(v['p50'] or 0 for v in per.values()):.1f} tok/s "
          f"(optimistic bound: p50s as if perfectly concurrent)")
        p50s = [v["p50"] for v in per.values() if v["p50"] is not None]
        w(f"- min-agent p50 decode: **{min(p50s):.1f} tok/s** (target {a.target_agent_tps}/agent)"
          if p50s else "- min-agent p50: -")
        w("")
        w("| agent | n steady | p50 | p90 | p10 | min | window tok/s | prefill med s | prefill med tok/s | ctx range |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for ag in agents:
            v = per[ag]
            if not v["rates"]:
                w(f"| a{ag} | 0 | - | - | - | - | - | - | - | - |")
                continue
            wt = v["tokens"] / window
            ctxr = f"{min(v['ctx'])}..{max(v['ctx'])}" if v["ctx"] else "-"
            pf_s, pf_t = med(v["prefill_s"]), med(v["prefill_tps"])
            pfs = f"{pf_s:.0f}" if pf_s is not None else "-"
            pft = f"{pf_t:.0f}" if pf_t is not None else "-"
            w(f"| a{ag} | {v['n']} | {v['p50']:.1f} | {v['p90']:.1f} | {v['p10']:.1f} | {v['min']:.1f} | "
              f"{wt:.1f} | {pfs} | {pft} | {ctxr} |")
        pf = [t["prefill_s"] for t in steady if t["prefill_s"] is not None]
        pft = [t["prefill_tps"] for t in steady if t["prefill_tps"] is not None]
        wall = sum(t["wall_s"] or 0 for t in steady)
        w("")
        w("## Prefill cost (steady turns)")
        if pf:
            share = 100 * sum(pf) / max(wall, 1e-9)
            note = " (prefill seconds are the server line's integer rounding; share clamped)" if share > 100 else ""
            w(f"- median prefill_s: {statistics.median(pf):.1f} s | median effective prefill tok/s: "
              f"{med(pft):.0f} | prefill share of turn wall: {min(share, 100.0):.1f}%{note}")
        else:
            w("- no prefill lines parsed (reading lines missing?)")
        reused = [t["reused_tokens"] for t in steady if t.get("reused_tokens") is not None]
        if reused:
            w(f"- median reused (prefix) tokens per turn: {statistics.median(reused):.0f}")
    else:
        w("- no steady turns yet (context never reached %d engine tokens)" % a.steady_min_ctx)
    w("")
    w("## Run facts")
    w(f"- turns recorded {len(turns)}, ok {len(ok)}, failed {len(turns) - len(ok)}, steady {len(steady)}")
    w(f"- peak concurrent in-flight turns: {st.peak_inflight} (expected {a.agents})")
    w(f"- stop_reason: {stop_reason}" + (f" | fatal: {st.fatal['reason']}" if st.fatal else ""))
    if totals_before or totals_after:
        try:
            d = int(totals_after.get("output_tokens", 0)) - int(totals_before.get("output_tokens", 0))
            w(f"- engine totals delta (output_tokens): {d} vs sum(gen_tokens) "
              f"{sum(t['gen_tokens'] or 0 for t in turns)}")
        except (TypeError, ValueError):
            pass
    w("")
    w("## Score it (the acceptance verdict)")
    w(f"    python3 score_target.py {st.outdir}")
    w("")
    w("Metric semantics: decode_tps = the server's done-line rate (decode-only, first->last token);")
    w("prefill_s = last 'reading the prompt' line; prefill_tps = engine prompt tokens / prefill_s")
    w("(effective, includes any prefix reuse); aggregate includes prefill (conservative).")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8096")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--serve-log", default=DEFAULT_SERVE_LOG)
    ap.add_argument("--engine-log", default=DEFAULT_ENGINE_LOG)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--turns", type=int, default=64, help="max turns per agent")
    ap.add_argument("--minutes", type=float, default=30.0, help="wall-clock cap (minutes)")
    ap.add_argument("--max-tokens", type=int, default=256, help="decode bound ceiling per turn")
    ap.add_argument("--ctx-cap", type=int, default=122880, help="per-agent context cap (client est tokens)")
    ap.add_argument("--msg-tokens", type=int, default=2500, help="est tokens per synthetic user message")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--steady-min-ctx", type=int, default=80000,
                    help="steady window min engine-token count (engine count lands ~87.5K at the "
                         "122880-est cap; 102400 was unreachable in time at the measured pace)")
    ap.add_argument("--warmup-turns", type=int, default=4)
    ap.add_argument("--min-steady-turns", type=int, default=12)
    ap.add_argument("--turn-bound", type=float, default=300.0, help="per-turn hard bound (s)")
    ap.add_argument("--stall-s", type=float, default=90.0, help="soft stall: no stream+log progress (s)")
    ap.add_argument("--global-stall", type=float, default=600.0, help="no completed turn anywhere (s)")
    ap.add_argument("--target-agent-tps", type=float, default=50.0, help="target per agent (for the summary line)")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan; send nothing")
    a = ap.parse_args()
    a.url = a.url.rstrip("/")

    if a.dry_run:
        print(f"dry run: agent-sim -> {a.url} model={a.model} (nothing sent, nothing written)")
        print(f"  agents={a.agents} turns/agent={a.turns} minutes={a.minutes} max_tokens={a.max_tokens}")
        print(f"  ctx_cap={a.ctx_cap} msg_tokens={a.msg_tokens} seed={a.seed} "
              f"steady_min_ctx={a.steady_min_ctx} warmup_turns={a.warmup_turns}")
        print(f"  bounds: turn_bound={a.turn_bound}s stall={a.stall_s}s global_stall={a.global_stall}s")
        print(f"  serve-log: {a.serve_log}")
        print(f"  engine-log: {a.engine_log}")
        print(f"  out: {a.out}/agent-sim-<stamp>/")
        return 0

    serve_log, engine_log = Path(a.serve_log), Path(a.engine_log)
    if not serve_log.exists():
        print(f"serve log not found: {serve_log} - is the server running? (this script never starts it)",
              file=sys.stderr)
        return 3
    try:
        m0 = get_json(a.url + "/metrics")
    except (OSError, ValueError) as e:
        print(f"server at {a.url} is not answering ({e}); start it first (see README)", file=sys.stderr)
        return 3

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    outdir = Path(a.out) / f"agent-sim-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    engine0 = m0.get("engine") or {}
    totals_before = dict(m0.get("totals") or {})
    print(f"agent-sim {stamp} start {started} agents={a.agents} turns={a.turns} minutes={a.minutes} "
          f"kv={engine0.get('kv')} concurrency={engine0.get('concurrency')} version={engine0.get('version')}",
          flush=True)
    print(f"  serve-log {serve_log} | engine-log {engine_log} | out {outdir}", flush=True)

    st = RunState(a, outdir)
    watcher = LogWatcher(serve_log)
    poller = MetricsPoller(a.url, outdir / "metrics-series.jsonl")
    poller.start()
    fire_lock = threading.Lock()

    threads = [threading.Thread(target=run_agent, args=(ag, st, watcher, poller, fire_lock), daemon=True)
               for ag in range(a.agents)]
    t_start = time.time()
    deadline = t_start + a.minutes * 60
    for t in threads:
        t.start()
    last_done_count = 0
    last_done_at = time.time()
    last_growth_seen = watcher.size_change_at()
    last_metrics_seen = poller.advanced_at()
    stop_reason = "turns"
    while True:
        time.sleep(1.0)
        alive = [t for t in threads if t.is_alive()]
        if not alive:
            stop_reason = "turns" if st.fatal is None else "fatal"
            break
        if st.fatal is not None:
            stop_reason = "fatal"
            break
        if time.time() >= deadline:
            stop_reason = "minutes"
            st.stop.set()
            break
        n = len(st.turns)
        if n > last_done_count:
            last_done_count = n
            last_done_at = time.time()
        # prefill liveness channels: a deep prefill completes no turns for minutes; raw serve-log
        # growth while turns are in flight and any advance of the tracked /metrics progress fields
        # are progress too, so a WORKING deep prefill is never declared stalled. A genuinely
        # frozen engine moves none of these and still trips --global-stall.
        watcher.touch()
        if watcher.size_change_at() > last_growth_seen:
            last_growth_seen = watcher.size_change_at()
            if st.inflight > 0:
                last_done_at = time.time()
        adv = poller.advanced_at()
        if adv > last_metrics_seen:
            last_metrics_seen = adv
            last_done_at = time.time()
        if time.time() - last_done_at > a.global_stall:
            stop_reason = "fatal"
            st.set_fatal(f"no completed turn anywhere for {a.global_stall:.0f} s")
            break
    # grace join: fatal -> quick (autopsy now); minutes/turns -> let the current turn finish
    grace = 20.0 if stop_reason == "fatal" else min(a.turn_bound + 30, 400)
    for t in threads:
        t.join(timeout=grace)
    with st.lock:
        st.frozen = True
    still = sum(1 for t in threads if t.is_alive())

    # enrich with /metrics history (best effort: same prompt/output token counts + time window)
    try:
        mh = get_json(a.url + "/metrics?requests=all", timeout=15)
        hist = mh.get("requests") or []
        totals_after = dict(mh.get("totals") or {})
        index: dict[tuple, list[dict]] = {}
        for h in hist:
            index.setdefault((h.get("prompt_tokens"), h.get("output_tokens")), []).append(h)
        for rec in st.turns:
            if rec["prompt_tokens"] is None or rec["gen_tokens"] is None:
                continue
            cands = index.get((rec["prompt_tokens"], rec["gen_tokens"])) or []
            best, bd = None, 1e9
            for h in cands:
                d = abs(float(h.get("time") or 0) - rec["t_fire"])
                if d < bd:
                    best, bd = h, d
            if best is not None and bd <= 60:
                rec["engine_prompt_ms"] = best.get("prompt_ms")
                rec["engine_decode_ms"] = best.get("decode_ms")
                rec["engine_decode_tps"] = best.get("decode_tok_s")
                rec["reused_tokens"] = best.get("reused")
                rec["history_matched"] = True
            else:
                rec["history_matched"] = False
        (outdir / "metrics-end.json").write_text(json.dumps(
            {"totals": totals_after, "requests_kept": mh.get("requests_kept"), "engine": mh.get("engine")},
            ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    except (OSError, ValueError) as e:
        totals_after = {}
        print(f"warn: /metrics?requests=all fetch failed ({e}); history enrichment skipped", flush=True)

    # steady classification (engine count when present, else client estimate) + write turns.jsonl
    for rec in st.turns:
        ctx = rec["prompt_tokens"] if rec["prompt_tokens"] is not None else rec["ctx_est_tokens"]
        rec["ctx_engine_tokens"] = rec["prompt_tokens"]
        rec["ctx_basis"] = "engine" if rec["prompt_tokens"] is not None else "est"
        rec["warmup"] = rec["turn"] <= a.warmup_turns
        rec["steady"] = bool(rec["ok"] and not rec["warmup"] and ctx >= a.steady_min_ctx)
    with open(outdir / "turns.jsonl", "w", encoding="utf-8") as f:
        for rec in st.turns:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    poller.stop.set()

    exit_code = 2 if stop_reason == "fatal" else 0
    autopsy_dir = None
    if stop_reason == "fatal":
        tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        autopsy_dir = autopsy(tag, (st.fatal or {}).get("reason", "unknown"), serve_log, engine_log, outdir, a.url)
        print(f"AGENT_SIM_FAIL: {(st.fatal or {}).get('reason')}", flush=True)
        print(f"  autopsy -> {autopsy_dir}/ (the engine's own watchdog report is the autopsy)", flush=True)
    summary = build_summary(st, poller, a, started, stop_reason, totals_before, totals_after)
    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    meta = {"run": outdir.name, "started": started,
            "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "url": a.url, "model": a.model, "serve_log": str(serve_log), "engine_log": str(engine_log),
            "agents": a.agents, "turns": a.turns, "minutes": a.minutes, "max_tokens": a.max_tokens,
            "ctx_cap": a.ctx_cap, "msg_tokens": a.msg_tokens, "seed": a.seed,
            "steady_min_ctx": a.steady_min_ctx, "warmup_turns": a.warmup_turns,
            "min_steady_turns": a.min_steady_turns, "turn_bound": a.turn_bound, "stall_s": a.stall_s,
            "target_agent_tps": a.target_agent_tps, "stop_reason": stop_reason,
            "fatal": st.fatal, "autopsy_dir": str(autopsy_dir) if autopsy_dir else None,
            "peak_inflight": st.peak_inflight, "workers_still_running_at_finalize": still,
            "engine_at_start": engine0, "exit_code": exit_code,
            "turns_sha256": hashlib.sha256((outdir / "turns.jsonl").read_bytes()).hexdigest()}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(summary, flush=True)
    print(f"RESULTS_DIR={outdir}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
