<h1 align="center">Strata AGX</h1>

<p align="center"><b>High-Performance Serving Engine for 125B MoE Models on Dual Consumer GPUs</b><br>
Dual NVIDIA GeForce RTX 4090 (48GB) · Dynamic RadixTree KV · Continuous Multi-Agent Batching · Bit-Exact Speculative Verification</p>

---

## Overview

**Strata AGX** is a dedicated downstream serving fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) tailored for **real-world multi-agent concurrency** and **low-latency serving** of massive Mixture-of-Experts models (such as **Qwen 3.8 Flash Next 125B**) on Dual NVIDIA GeForce RTX 4090 GPUs.

While upstream Strata demonstrated that 125B MoE models can run on dual 24GB GPUs in single-stream mode, Strata AGX delivers the production serving layer: continuous token-level scheduling for parallel agent workflows, zero-doorbell CUDA graph verification, dynamic RadixTree prefix reuse, automated GPUStack native lifecycle integration, and bit-exact generation yield.

---

## Key Capabilities & Production Features

1. **Continuous Multi-Agent Serving ($C=2$ Concurrent Streams @ $2 \times 262\text{k}$ KV Pool):**
   - Single unified continuous-batching queue serving agent turns and tool calls alike.
   - Dynamic lockless slot recycling (< 5 µs overhead) upon stream completion.
   - 524,288 total logical KV cache pool tokens ($2 \times 262,144$ full context slots) with automated FIFO queuing under burst load.

2. **Upstream 0.1.39b Merged Performance Core:**
   - **Zero-Doorbell Speculative Verify Graph:** Multi-token verification windows (1–4 tokens) captured into persistent CUDA graphs, completely bypassing host-driver submission bottlenecks.
   - **Sub-Warp Cooperative Expert Packing:** Warp-level cooperative expert launches and shared-memory staging, raising sustained decode throughput to **100–120+ tok/s**.
   - **Bit-Exactness Guarantee:** Bit-for-bit mathematical parity against reference ggml across all 18 quant formats with zero logit distortion.

3. **Multi-Turn Agent Prefix Caching (Dynamic RadixTree):**
   - Zero-copy prefix sharing across branching agent sessions and multi-turn workflows.
   - Pinned billing headers (Claude Code / Anthropic client compatibility) to ensure agent turns hit identical prefix hashes across consecutive prompts.

4. **GPUStack Native Lifecycle & Telemetry:**
   - Whitelisted as a first-class native engine in GPUStack with automated candidate verification.
   - Native `/health`, `/v1/models`, and `/v1/tokenize` endpoints with token-granular reservation contracts.
   - Real-time dual-GPU hardware monitoring dashboard reporting per-card VRAM, temperature, power, and MoE cache slot occupancy.

---

## Production Recipe Benchmark & Measured Serving Metrics

All figures below represent real physical hardware benchmarks measured on live multi-turn completions:

### Physical Hardware Environment
- **GPUs:** Dual NVIDIA GeForce RTX 4090 24GB (PCIe 4.0 x16 / x8, Total 48GB VRAM)
  - *Clocks:* Core Offset +150 MHz (~2,800 MHz boost) · GDDR6X Memory Offset +1000 MHz (11.5 GHz effective / ~1,104 GB/s per GPU)
- **CPU:** AMD Ryzen 9 5950X (16-Core / 32-Thread, CCD0 pinned: cores 0–7, threads 0–7, 16–23)
- **RAM:** 96 GB DDR4-3200 (Quad-channel configuration)
- **Storage:** WD_BLACK SN850X 4TB NVMe SSD
- **OS:** Ubuntu 24.04 LTS (Linux 6.8, CUDA 12.4)

---

### Production Serving Recipe: Qwen 3.8 Flash Next IQ3_XXS (262K Context)

- **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (125B MoE with 6B active parameters, 512 total experts, 3.06 bpw)
- **Layer Distribution:** Layer-split at 27 (Layers 0–26 on CUDA 0, Layers 27–47 + Output Head on CUDA 1)
- **KV Precision:** `q4_0` with FWHT-256 rotation
- **Speculative Verification:** MTP `--spec 4 --mtp-max-t 2 --suffix-draft 3 --spec-min-p 0.55`

### Transparent Serving Modes & Operational Performance

The engine provides two distinct operational profiles tailored for specific deployment goals:

| Serving Metric | Mode A: Dedicated Single-Stream ($C=1$) | Mode B: Dual-Agent Concurrency ($C=2$) | Operational Detail |
|---|---:|---:|---|
| **Primary Goal** | Maximum Solo Prefill Throughput | Simultaneous Multi-Developer Serving | Mode B prevents agent blocking under concurrency |
| **MoE VRAM Residency** | **20,739 / 24,576 slots (84.4%)** | **17,254 / 24,576 slots (70.2%)** | Mode B reserves 4.6 GiB dedicated prompt workspace |
| **VRAM Cache Allocation** | 17.9 GB (CUDA0) / 16.5 GB (CUDA1) | 14.9 GB (CUDA0) / 13.0 GB (CUDA1) | Zero thrashing across PCIe bus |
| **Cold Prefill (62k Hermes)** | **11.0 s (~5,631 tok/s)** | **19.7 s (~3,180 tok/s)** | Tested on 61,948 token real Hermes payload + 33 tools |
| **Extreme Cold Prefill (181k)** | **28.7 s (~6,325 tok/s)** | — | Synthetic continuous long-context probe |
| **Warm TTFT (Cached Prefix)** | **24 – 35 ms** | **24 – 35 ms** | Instant RadixTree prefix match (50k+ prefix hit) |
| **Sustained Decode Speed** | **105.0 – 120.4 tok/s** | **82.1 tok/s aggregate** | Mode B serves 2 simultaneous streams @ ~98 tok/s peak |
| **Peak Decode Speed** | **156.2 tok/s** | **120+ tok/s** | High speculative acceptance runs |
| **Speculative Acceptance Rate** | **78.2% – 86.4%** | **80.5% – 88.7%** | Fused MTP draft chain (`STRATA_MTP_FUSE_CHAIN=1`) |
| **Total VRAM Consumption** | **47.6 GB across dual GPUs** | **47.6 GB across dual GPUs** | 23.9 GB on GPU 0 + 23.9 GB on GPU 1 (97% VRAM) |

> **Architectural Note on Concurrency vs. Prefill**: In single-stream mode ($C=1$), prompt prefill dynamically borrows MoE cache slots via `lend()` and returns them immediately, allowing high prefill throughput. In true concurrent mode ($C=2$), simultaneous streams require independent prompt workspace buffers (`--concurrent-prefill 5120`). Sizing this buffer at 5k chunks reclaims ~5.95 GiB of VRAM compared to 12k chunks, restoring +4,030 resident experts to guarantee high sustained decode throughput without PCIe bottlenecking.

---

## Quickstart & Ready-To-Run Configuration

### 1. Build from Source

```bash
git clone https://github.com/MAHDI-AQ/Strata-AGX.git
cd Strata-AGX

# Configure with CUDA enabled for SM89 (RTX 4090)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=89

# Compile binary
cmake --build build --target strata -j 16
```

### 2. Verify Kernel Parity
Confirm bit-exactness and kernel execution integrity before launching serving:
```bash
./build/iq_multi_parity
./build/s2_expert_grouped_parity
./build/native_grouped_parity
./build/radix_tree_test
```

### 3. Launching Serving with the Lab Recipe Configuration

A production configuration template is checked in at [`configs/q3-xxs-dual-4090-serving.json`](configs/q3-xxs-dual-4090-serving.json).

```bash
# Start server using the reference config
python3 serve/server.py --engine strata --config configs/q3-xxs-dual-4090-serving.json --port 8096
```

#### Production Configuration Contents (`configs/q3-xxs-dual-4090-serving.json`):
```json
{
  "exe": "./build/strata",
  "cwd": ".",
  "tokenizer": "/path/to/models/packs/iq3_xxs/tokenizer",
  "gpu": [0, 1],
  "host": "127.0.0.1",
  "port": 8096,
  "env": {
    "STRATA_RADIX_VRAM_SLOTS": "8",
    "STRATA_RADIX_HOST_SLOTS": "8",
    "STRATA_WATCHDOG_S": "180",
    "STRATA_STAGE_OVERLAP": "1",
    "STRATA_PREFILL_CHAIN": "1",
    "STRATA_PREFILL_RING": "8",
    "STRATA_PLE_PREFETCH": "1",
    "STRATA_MMQ_BLOB": "1",
    "STRATA_STAGE_OVERLAP_CROSSDEV": "1",
    "STRATA_CONCURRENT_RETAIN": "1",
    "STRATA_CONCURRENT_PROFILE": "1",
    "STRATA_PASS_STALL_MS": "250",
    "STRATA_IPC_SHM": "/strata_ipc_8096",
    "STRATA_NVME_TIER_DIR": "/srv/lab/cache/strata/nvme_radix",
    "STRATA_SAMPLER_ONE_BLOCK": "1",
    "STRATA_PIN_LIMIT_GIB": "32",
    "STRATA_RADIX_VRAM_MIB": "2048",
    "STRATA_RADIX_DEEP_RESERVE_MIB": "1280",
    "STRATA_OLD_SAMPLER": "1",
    "STRATA_VERIFY_DEVICE_PLAN": "1",
    "STRATA_MTP_FUSE_CHAIN": "1",
    "STRATA_DRAFT_WAVEFRONT": "1"
  },
  "sampling": {
    "temperature": 0.6,
    "top_p": 0.95,
    "min_p": 0.05
  },
  "model_name": "Qwen3.8-Flash-Next-IQ3_XXS",
  "log": "./logs/engine.log",
  "api_monitor": true,
  "args": [
    "--pack", "/path/to/models/packs/iq3_xxs",
    "--native", "/path/to/models/flash-next-iq3_xxs/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/models/flash-next-iq3_xxs/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
    "--expert-cache", "auto",
    "--expert-profile", "./data/expert-profile.bin",
    "--prefill", "auto",
    "--spec", "4",
    "--mtp-max-t", "2",
    "--suffix-draft", "3",
    "--spec-min-p", "0.55",
    "--mtp", "/path/to/models/mtp/rt",
    "--max-context", "262144",
    "--concurrency", "2",
    "--unit-wait-ms", "0",
    "--batch-rows", "10",
    "--batch-policy", "fair",
    "--concurrent-prefill", "5120",
    "--batch-parallel", "1",
    "--batch-graphs", "16",
    "--batch-padding", "1",
    "--kv", "q4_0",
    "--layer-split", "27",
    "--split-device", "1",
    "--ple-row-cache", "320001536",
    "--vram-reserve-mib", "636",
    "--adapt-every", "0",
    "--nanobatch", "2",
    "--pcie-frac", "0.85"
  ],
  "cors_origins": ["http://127.0.0.1:8096", "http://localhost:8096"],
  "allowed_hosts": ["127.0.0.1", "localhost"]
}
```

---

## Client Usage (OpenAI Compatible)

The server exposes an OpenAI-compliant API on `http://127.0.0.1:8096/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8096/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen38-flashnext",
    messages=[
        {"role": "system", "content": "You are an autonomous engineering assistant."},
        {"role": "user", "content": "Implement a lock-free circular buffer in C++20."},
    ],
    temperature=0.6,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

---

## Cherry-Pick & Porting Status

Strata AGX merges the upstream **0.1.39b performance and bit-exactness core** (`11bb74cc`) while curating an active ledger of stability cherry-picks and explicit standing bans:

| Commit / Feature | Upstream Ref | Category | Description | Status |
|---|---|---|---|---|
| `deee447` | Upstream Core | **Performance** | Multi-token parallel resident plan on Dual RTX 4090s | **Merged (0.1.39b)** |
| `055122c` | Upstream Core | **Correctness** | Bit-exactness restore & zero-doorbell CUDA verify graph | **Merged (0.1.39b)** |
| `6f32ec07` | Upstream Core | **Performance** | Sub-warp cooperative expert packing & shared-memory staging | **Merged (0.1.39b)** |
| `37cc2589` | `98b7ea94` | **Stability** | Sanitizes literal control tokens (`<|im_start|>`, `<|im_end|>`) in user messages | **Landed** |
| `e8b4f84a` | `bae372b9` | **Agent Caching** | Pins Claude Code / Anthropic billing header stamps for prefix cache hits | **Landed** |
| `e740f956` | `c34dd571` | **Concurrency** | Sets `STRATA_HTTP_BACKLOG 256` to absorb high-burst client connections | **Landed** |
| `6c2bd1ab` | Fork Feature | **Serving API** | Restores `/v1/tokenize` endpoint with full RadixTree reservation facts | **Landed** |
| `12447529` | Fork Feature | **Observability** | Real-time dual-GPU independent hardware & MoE cache telemetry | **Landed** |
| `c628f801` | Fork Tuning | **Hardware Tuning** | Pins processes and thread pools to AMD Ryzen 5950X CCD0 (cores 0–7, 16–23); eliminates 30s 0% GPU freeze on 62k+ prompts | **Landed** |
| `5df35dcb` | Upstream Reject | **Prohibited** | `STRATA_ROUTE_RESIDENT` (MoE routing bias; perturbs logits & breaks bit-exactness) | **REJECTED / BANNED** |
| `--aux-slots` | Fork Experiment | **Prohibited** | Asymmetric slot slicing; caused scheduler starvation; replaced by uniform $C=2$ pool | **PURGED (`a3da023c`)** |

> For the comprehensive commit-by-commit porting analysis, technical rationale, and verification receipts, see the detailed documentation in [**`docs/cherry-pick-ledger.md`**](docs/cherry-pick-ledger.md).

---

## Documentation & Porting Ledgers

- [**`ATTRIBUTION.md`**](ATTRIBUTION.md) — Comprehensive upstream attribution, community contributor credits, and architectural lineage matrix.
- [**`docs/cherry-pick-ledger.md`**](docs/cherry-pick-ledger.md) — Detailed authoritative cherry-pick ledger, commit tracking, and exclusion justifications.

---

## License & Attribution

Strata is licensed under the **Apache License 2.0 / MIT License**.
See [**`ATTRIBUTION.md`**](ATTRIBUTION.md) for full contributor credits and research lineage.
