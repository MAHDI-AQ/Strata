#!/usr/bin/env python3
"""long_horizon_harness_test.py - Realistic multi-agent long-horizon harness benchmark.

Models a realistic Hermes agent workload:
  1. Base harness: ~35k tokens (system prompt + tool definitions + instructions).
  2. Long-horizon context accumulation: Main agent works, appends tool outputs, growing from 35k -> 50k -> 70k.
  3. Concurrent subagent fan-out: Main agent spins up 4 subagents sharing the accumulated harness prefix concurrently (c5 @ 128k context envelope).
  4. Measures: Cold vs Warm TTFT, prefix reuse count, per-agent decode TPS, and aggregate decode TPS.
"""
import argparse
import json
import random
import sys
import time
import urllib.request
import concurrent.futures
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8096"
WORDS = [
    "quantum", "compiler", "register", "tensor", "pipeline", "speculative", "memory", "bandwidth",
    "attention", "gradient", "transformer", "latency", "throughput", "buffer", "matrix", "vector",
    "kernel", "execution", "synchronize", "stream", "residual", "expert", "routing", "scatter"
]

def generate_text_tokens(approx_tokens, seed=42):
    rng = random.Random(seed)
    n_words = int(approx_tokens / 1.3)
    text = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return text

def send_chat_completion(url, messages, max_tokens=64, stream=True, request_id=None):
    body = {
        "model": "strata-coder",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t_start = time.perf_counter()
    ttft = None
    first_token_time = None
    generated_tokens = 0
    full_text = []

    if stream:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                ln = line.decode("utf-8").strip()
                if not ln or not ln.startswith("data: "):
                    continue
                payload = ln[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content") or delta.get("reasoning_content") or ""
                        if content:
                            if ttft is None:
                                ttft = time.perf_counter() - t_start
                                first_token_time = time.perf_counter()
                            generated_tokens += 1
                            full_text.append(content)
                except Exception:
                    pass
    else:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data_resp = json.loads(resp.read().decode("utf-8"))
            ttft = time.perf_counter() - t_start
            choices = data_resp.get("choices", [])
            if choices:
                full_text.append(choices[0].get("message", {}).get("content", ""))
                generated_tokens = data_resp.get("usage", {}).get("completion_tokens", max_tokens)

    t_end = time.perf_counter()
    total_wall = t_end - t_start
    decode_wall = (t_end - first_token_time) if first_token_time else total_wall
    decode_tps = (generated_tokens / decode_wall) if decode_wall > 0 and generated_tokens > 0 else 0.0

    return {
        "request_id": request_id,
        "total_wall_s": round(total_wall, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "decode_wall_s": round(decode_wall, 3),
        "generated_tokens": generated_tokens,
        "decode_tps": round(decode_tps, 2),
        "text": "".join(full_text)[:80],
        "full_content": "".join(full_text)
    }

def get_server_metrics(url):
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Multi-agent long horizon harness benchmark")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--base-tokens", type=int, default=35000, help="Base harness size (system prompt + tools)")
    parser.add_argument("--step1-tokens", type=int, default=15000, help="Context step 1 (accumulate to 50k)")
    parser.add_argument("--step2-tokens", type=int, default=20000, help="Context step 2 (accumulate to 70k)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Tokens to generate per turn")
    args = parser.parse_args()

    print(f"=== Long-Horizon Multi-Agent Harness Benchmark ===")
    print(f"Server URL: {args.url}")
    print(f"Base Harness: {args.base_tokens} tokens")
    print(f"Context Steps: +{args.step1_tokens} -> +{args.step2_tokens} (Total prefix ~{args.base_tokens + args.step1_tokens + args.step2_tokens})")
    print(f"Sub-agents: 4 concurrent subagents + 1 main agent (c5)\n")

    # Generate synthetic text blocks
    print("Generating synthetic prompt blocks...")
    base_harness = generate_text_tokens(args.base_tokens, seed=101)
    tool_output_1 = generate_text_tokens(args.step1_tokens, seed=102)
    tool_output_2 = generate_text_tokens(args.step2_tokens, seed=103)

    messages = [
        {"role": "system", "content": f"You are Hermes Agent with tool capabilities. HARNESS_DATA: {base_harness}"},
        {"role": "user", "content": "Begin primary task analysis."}
    ]

    # --- Phase 1: Cold Run (Main Agent Turn 1 @ ~35k) ---
    print("\n--- Phase 1: Cold Run (Main Agent Turn 1 @ ~35k context) ---")
    res1 = send_chat_completion(args.url, messages, max_tokens=args.max_tokens, request_id="main-turn-1")
    m2 = get_server_metrics(args.url)
    latest_req = m2.get("requests", [{}])[0] if m2.get("requests") else {}
    print(f"Cold Turn 1: Wall={res1['total_wall_s']}s, TTFT={res1['ttft_s']}s, Gen={res1['generated_tokens']} tok, Decode={res1['decode_tps']} tok/s")
    print(f"Engine Server Metrics: prompt_tokens={latest_req.get('prompt_tokens')}, reused={latest_req.get('reused')}, prompt_ms={latest_req.get('prompt_ms')}ms, decode_tok_s={latest_req.get('decode_tok_s')}")

    # Add actual assistant response from turn 1 and tool output 1
    messages.append({"role": "assistant", "content": res1.get("full_content", "OK")})
    messages.append({"role": "user", "content": f"Tool output: {tool_output_1}"})

    # --- Phase 2: Warm Turn 2 (Main Agent Context Accumulation @ ~50k) ---
    print("\n--- Phase 2: Warm Turn 2 (Main Agent Accumulation @ ~50k context) ---")
    res2 = send_chat_completion(args.url, messages, max_tokens=args.max_tokens, request_id="main-turn-2")
    m3 = get_server_metrics(args.url)
    latest_req = m3.get("requests", [{}])[0] if m3.get("requests") else {}
    print(f"Warm Turn 2: Wall={res2['total_wall_s']}s, TTFT={res2['ttft_s']}s, Gen={res2['generated_tokens']} tok, Decode={res2['decode_tps']} tok/s")
    print(f"Engine Server Metrics: prompt_tokens={latest_req.get('prompt_tokens')}, reused={latest_req.get('reused')}, prompt_ms={latest_req.get('prompt_ms')}ms, decode_tok_s={latest_req.get('decode_tok_s')}")

    # Add actual assistant response from turn 2 and tool output 2
    messages.append({"role": "assistant", "content": res2.get("full_content", "OK")})
    messages.append({"role": "user", "content": f"Tool output: {tool_output_2}"})

    # --- Phase 3: Warm Turn 3 (Main Agent Accumulation @ ~70k) ---
    print("\n--- Phase 3: Warm Turn 3 (Main Agent Accumulation @ ~70k context) ---")
    res3 = send_chat_completion(args.url, messages, max_tokens=args.max_tokens, request_id="main-turn-3")
    m4 = get_server_metrics(args.url)
    latest_req = m4.get("requests", [{}])[0] if m4.get("requests") else {}
    print(f"Warm Turn 3: Wall={res3['total_wall_s']}s, TTFT={res3['ttft_s']}s, Gen={res3['generated_tokens']} tok, Decode={res3['decode_tps']} tok/s")
    print(f"Engine Server Metrics: prompt_tokens={latest_req.get('prompt_tokens')}, reused={latest_req.get('reused')}, prompt_ms={latest_req.get('prompt_ms')}ms, decode_tok_s={latest_req.get('decode_tok_s')}")

    # Add actual assistant response from turn 3
    messages.append({"role": "assistant", "content": res3.get("full_content", "OK")})

    # --- Phase 4: Concurrent Fan-Out (Main Agent + 4 Sub-Agents @ 70k Prefix) ---
    print("\n--- Phase 4: Concurrent Multi-Agent Fan-Out (1 Main + 4 Sub-Agents @ ~70k Shared Prefix) ---")
    subagent_directives = [
        "Subagent 1: Audit codebase memory layout and report structures.",
        "Subagent 2: Diagnose CUDA kernel launch parameters and register pressure.",
        "Subagent 3: Inspect memory bandwidth saturation and tensor copy latency.",
        "Subagent 4: Verify test suite assertions and numerical stability.",
        "Main Agent: Coordinate subagent findings and synthesize global plan."
    ]

    fanout_requests = []
    for idx, directive in enumerate(subagent_directives):
        agent_msgs = list(messages) + [{"role": "user", "content": directive}]
        fanout_requests.append((f"agent-slot-{idx}", agent_msgs))

    t_fanout_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(send_chat_completion, args.url, req_msgs, args.max_tokens, True, req_id): req_id
            for req_id, req_msgs in fanout_requests
        }
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    t_fanout_end = time.perf_counter()

    m5 = get_server_metrics(args.url)
    fanout_server_reqs = m5.get("requests", [])[:5]

    print(f"\nFan-out Completed in {t_fanout_end - t_fanout_start:.2f}s total wall time:")
    total_tokens = 0
    for r in sorted(results, key=lambda x: x["request_id"]):
        total_tokens += r["generated_tokens"]
        print(f"  {r['request_id']}: Wall={r['total_wall_s']}s, TTFT={r['ttft_s']}s, Gen={r['generated_tokens']}, Decode={r['decode_tps']} tok/s")

    agg_decode_tps = total_tokens / (t_fanout_end - t_fanout_start)
    print(f"\nAggregate Concurrency Decode Throughput: {agg_decode_tps:.2f} tok/s")
    print("\nEngine Server Metrics for Fan-out requests:")
    for req in fanout_server_reqs:
        print(f"  Req: prompt_tokens={req.get('prompt_tokens')}, reused={req.get('reused')}, prompt_ms={req.get('prompt_ms')}ms, decode_tok_s={req.get('decode_tok_s')}")

if __name__ == "__main__":
    main()
