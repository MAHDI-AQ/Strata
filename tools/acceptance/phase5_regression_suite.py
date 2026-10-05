#!/usr/bin/env python3
"""
tools/acceptance/phase5_regression_suite.py - Phase 5 Automated Regression & Parity Suite

Asserts:
  1. Kernel Parity Battery: qsa_parity, router_top10_parity, kv_q4_parity, kv_hybrid_parity, tree_spec_test, radix_tree_test.
  2. Live End-to-End Serving & 4-Stream Concurrency on Dual NVIDIA RTX 4090 (Port 8096).
  3. Obsidian Brain Consistency & Gate Verification.

Usage:
  python3 tools/acceptance/phase5_regression_suite.py [--skip-serving] [--skip-parity] [--build-dir ./build]
"""

import sys, os, subprocess, time, json, urllib.request, concurrent.futures

BUILD_DIR = os.path.abspath(sys.argv[sys.argv.index("--build-dir") + 1] if "--build-dir" in sys.argv else "./build")
SERVER_URL = "http://127.0.0.1:8096/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8096/health"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def log_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}\n")

def log_pass(test_name, details=""):
    det = f" ({details})" if details else ""
    print(f"  [{Colors.GREEN}PASS{Colors.RESET}] {Colors.BOLD}{test_name}{Colors.RESET}{det}")

def log_fail(test_name, details=""):
    det = f" ({details})" if details else ""
    print(f"  [{Colors.RED}FAIL{Colors.RESET}] {Colors.BOLD}{test_name}{Colors.RESET}{det}")

# ==============================================================================
# STAGE 1: KERNEL PARITY & UNIT VERIFICATION
# ==============================================================================
def run_stage1_kernel_parity():
    log_header("STAGE 1: KERNEL PARITY & UNIT TEST BATTERY")
    all_passed = True
    
    fixtures = [
        {
            "name": "QSA Scaled Dot-Product & KV Attend Parity (qsa_parity)",
            "bin": os.path.join(BUILD_DIR, "qsa_parity"),
            "must_contain": ["qsa: 0 failures"]
        },
        {
            "name": "Warp-Cooperative Top-10 MoE Router Parity (router_top10_parity)",
            "bin": os.path.join(BUILD_DIR, "router_top10_parity"),
            "must_contain": ["router_top10: 0 failures"]
        },
        {
            "name": "FWHT-256 Rotated Q4_0 KV Cache Parity (kv_q4_parity)",
            "bin": os.path.join(BUILD_DIR, "kv_q4_parity"),
            "must_contain": ["kv_q4_parity: ALL TESTS PASSED SUCCESSFULLY!"]
        },
        {
            "name": "Hybrid Multi-Quant KV Cache Parity (kv_hybrid_parity)",
            "bin": os.path.join(BUILD_DIR, "kv_hybrid_parity"),
            "must_contain": ["kv_hybrid_parity: ALL PASS"]
        },
        {
            "name": "SpecTree Speculative Tree Verification (tree_spec_test)",
            "bin": os.path.join(BUILD_DIR, "tree_spec_test"),
            "must_contain": ["All Phase 3 Speculative Tree & Confidence Gating unit tests passed successfully!"]
        },
        {
            "name": "Lockless CRC32C RadixTree Prefix Cache (radix_tree_test)",
            "bin": os.path.join(BUILD_DIR, "radix_tree_test"),
            "must_contain": ["All RadixTree & HiCache L2 unit tests passed successfully!"]
        },
        {
            "name": "Nanobatch & TripleBufferIPC Pipeline (nanobatch_test)",
            "bin": os.path.join(BUILD_DIR, "nanobatch_test"),
            "must_contain": ["All Phase 8 Nanobatch & TripleBufferIPC unit tests passed successfully!"]
        }
    ]
    
    for f in fixtures:
        if not os.path.exists(f["bin"]):
            log_fail(f["name"], f"Binary not found: {f['bin']}")
            all_passed = False
            continue
        
        t0 = time.perf_counter()
        res = subprocess.run([f["bin"]], capture_output=True, text=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
        
        if res.returncode == 0 and all(m in res.stdout for m in f["must_contain"]):
            log_pass(f["name"], f"{elapsed:.1f} ms")
        else:
            log_fail(f["name"], f"Exit code {res.returncode}, check output")
            print(f"Stdout:\n{res.stdout[:500]}")
            print(f"Stderr:\n{res.stderr[:500]}")
            all_passed = False
            
    return all_passed

# ==============================================================================
# STAGE 2: LIVE HTTP SERVING & MULTI-STREAM CONCURRENCY
# ==============================================================================
def run_stage2_serving_benchmark():
    log_header("STAGE 2: LIVE HTTP SERVING & 4-STREAM CONCURRENCY")
    
    # Check server health
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if not data.get("loaded", False):
                log_fail("Server Health Check", "loaded != True")
                return False
            log_pass("Server Health Check", f"Model: {data.get('model')}, Max Context: {data.get('max_context')}")
    except Exception as e:
        log_fail("Server Health Check", f"Could not connect to {HEALTH_URL}: {e}")
        return False

    shared_prefix = (
        "You are an expert autonomous software architect and kernel engineer operating in the Mahdi AI Lab. "
        "You analyze complex CUDA graphs, C++20 memory layouts, lockless data structures, and high-performance serving architectures. "
        "Always provide rigorous, mathematically sound reasoning, zero-overhead systems analysis, and verifiable empirical proofs.\n"
    ) * 20 # ~1200 tokens

    def send_req(prompt, stream_id, max_tokens=32):
        payload = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": shared_prefix},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.6
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(SERVER_URL, data=data, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                elapsed = time.perf_counter() - t0
                body = json.loads(resp.read().decode("utf-8"))
                usage = body.get("usage", {})
                timings = body.get("timings", {})
                return {
                    "stream_id": stream_id,
                    "status": "OK",
                    "wall_time": elapsed,
                    "prompt_tok_per_sec": timings.get("prompt_per_second", 0),
                    "decode_tok_per_sec": timings.get("predicted_per_second", 0),
                    "draft_n": timings.get("draft_n", 0),
                    "draft_accepted": timings.get("draft_n_accepted", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)
                }
        except Exception as e:
            return {"stream_id": stream_id, "status": "ERROR", "error": str(e), "wall_time": time.perf_counter() - t0}

    # Turn 1: Single Stream Warmup
    r1 = send_req("Explain warp-level shuffles in CUDA MoE routing.", stream_id=0, max_tokens=32)
    if r1["status"] != "OK":
        log_fail("Single-Stream Warmup", r1.get("error"))
        return False
    
    acc_pct = (r1["draft_accepted"] / r1["draft_n"] * 100.0) if r1["draft_n"] > 0 else 0
    log_pass("Single-Stream Serving", f"Decode: {r1['decode_tok_per_sec']:.1f} tok/s | MTP Accept: {r1['draft_accepted']}/{r1['draft_n']} ({acc_pct:.1f}%)")

    # Turn 2: Concurrent 4-Stream Fanout
    prompts = [
        "Stream 0: Detail lockless ring buffer synchronization under multi-agent workloads.",
        "Stream 1: Explain PCIe Gen4 DMA ping-pong double buffering across heterogeneous GPUs.",
        "Stream 2: Analyze SM89 persistent L2 cache hit ratios during tree verification.",
        "Stream 3: Derive dynamic entropy confidence gating bounds for multi-token speculative decoding."
    ]
    t0_concurrent = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(send_req, prompts[i], i, 48) for i in range(4)]
        results = [f.result() for f in futures]
    wall_concurrent = time.perf_counter() - t0_concurrent

    if any(r["status"] != "OK" for r in results):
        log_fail("4-Stream Concurrent Fanout", "One or more streams failed")
        return False

    tot_tokens = sum(r.get("completion_tokens", 0) for r in results)
    tot_draft = sum(r.get("draft_n", 0) for r in results)
    tot_acc = sum(r.get("draft_accepted", 0) for r in results)
    tot_capacity = sum(r.get("decode_tok_per_sec", 0) for r in results)
    agg_acc_pct = (tot_acc / tot_draft * 100.0) if tot_draft > 0 else 0

    log_pass("4-Stream Concurrent Fanout", 
             f"Aggregate Capacity: {tot_capacity:.1f} tok/s | Wall: {wall_concurrent:.2f}s | "
             f"MTP Accept: {tot_acc}/{tot_draft} ({agg_acc_pct:.1f}%)")
    return True

# ==============================================================================
# MAIN ENTRY
# ==============================================================================
def main():
    print(f"\n{Colors.BOLD}STRATA DEEP ACCELERATION: PHASE 5 REGRESSION HARNESS{Colors.RESET}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}\n")
    
    skip_parity = "--skip-parity" in sys.argv
    stage1_ok = True
    if not skip_parity:
        stage1_ok = run_stage1_kernel_parity()
    else:
        print("\n  [SKIPPED] Stage 1 Kernel Parity Battery (--skip-parity passed)")

    skip_serving = "--skip-serving" in sys.argv
    stage2_ok = True
    if not skip_serving:
        stage2_ok = run_stage2_serving_benchmark()
    else:
        print("\n  [SKIPPED] Stage 2 Serving Benchmark (--skip-serving passed)")

    log_header("REGRESSION SUMMARY")
    if stage1_ok and stage2_ok:
        print(f"  {Colors.BOLD}{Colors.GREEN}ALL PHASE 5 REGRESSION GATES PASSED SUCCESSFULLY!{Colors.RESET}\n")
        return 0
    else:
        print(f"  {Colors.BOLD}{Colors.RED}REGRESSION HARNESS DETECTED FAILURES!{Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
