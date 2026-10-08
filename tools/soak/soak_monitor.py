#!/usr/bin/env python3
"""Soak monitor: sample engine health, RSS, VRAM and log sizes every interval; CSV + event log.
Ends when the stop-marker file appears. Pure stdlib; runs beside a soak sim."""
import csv, json, os, subprocess, sys, time, urllib.request
from datetime import datetime, timezone

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 300
STOP = os.path.join(OUT, "soak.stop")
URL = "http://127.0.0.1:8096"
REC = os.environ.get("SOAK_RECIPE", "")

os.makedirs(OUT, exist_ok=True)
csv_path = os.path.join(OUT, "soak-monitor.csv")
evt_path = os.path.join(OUT, "soak-events.log")
new = not os.path.exists(csv_path)
f = open(csv_path, "a", newline="")
w = csv.writer(f)
if new:
    w.writerow(["utc", "health", "engine_commit", "rss_mib", "gpu0_mib", "gpu1_mib",
                "server_log_mib", "engine_log_mib", "note"])

def evt(msg):
    with open(evt_path, "a") as e:
        e.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")

def sh_list(args, t=10):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=t).stdout.strip()
    except Exception as ex:
        return f"ERR {ex}"

def engine_rss_mib():
    out = sh_list(["ps", "-eo", "rss=,comm="])
    total = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "strata":
            try:
                total += int(parts[0])
            except ValueError:
                pass
    return total

def gpu_used():
    out = sh_list(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return [l.strip() for l in out.splitlines() if l.strip()]

print(f"soak monitor start interval={INTERVAL}s out={OUT}", flush=True)
while not os.path.exists(STOP):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    health, commit = "DOWN", ""
    try:
        with urllib.request.urlopen(URL + "/health", timeout=5) as r:
            j = json.loads(r.read().decode())
        health, commit = j.get("status", "?"), j.get("commit", "")
    except Exception:
        pass
    rss = engine_rss_mib()
    gpus = gpu_used()
    g0 = gpus[0] if len(gpus) > 0 else "?"
    g1 = gpus[1] if len(gpus) > 1 else "?"
    def mib(p):
        try:
            return round(os.path.getsize(p) / 1048576, 1)
        except Exception:
            return -1
    sl = mib(os.path.join(REC, "logs/server.log")) if REC else -1
    el = mib(os.path.join(REC, "logs/engine.log")) if REC else -1
    note = ""
    if health != "ok":
        note = "HEALTH-FAIL"
        evt("HEALTH-FAIL observed")
    if isinstance(rss, str) and rss.isdigit() and int(rss) == 0 and health == "ok":
        note = (note + " " if note else "") + "RSS-READ-ERR"
    w.writerow([ts, health, commit, rss, g0, g1, sl, el, note])
    f.flush()
    time.sleep(INTERVAL)
evt("monitor stop (marker seen)")
print("soak monitor end", flush=True)
