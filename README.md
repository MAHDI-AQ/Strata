<h1 align="center">Strata (Mahdi AI Lab Edition)</h1>

<p align="center"><b>High-Throughput Downstream Inference Engine for 125B MoE Models</b><br>
Dual NVIDIA GeForce RTX 4090 · 262K Context Scaling · Multi-Agent Dynamic RadixTree KV · Free and Open Source</p>

---

## Overview

**Strata (Mahdi AI Lab Edition)** is an independent downstream engineering fork of [Niko1221/Strata](https://github.com/Niko1221/Strata).

While upstream Strata proved that a 125B MoE model could execute on consumer gaming GPUs, this fork extends the architecture for **multi-agent workloads**, **massive context windows (262K)**, and **high-throughput dual-GPU execution**.

### Architectural Extensions (13,633 LoC)

1. **Multi-Agent Concurrency & Scheduling:**
   - Multi-slot concurrent decoding ($C=1$ to $C=5$) with dynamic slot balancing.
   - Request isolation with independent sequence states, verifier workspaces, and causal rollback.
2. **Dynamic RadixTree KV Cache:**
   - Zero-copy prefix sharing across concurrent reasoning branches.
   - Sub-millisecond tree forking for agent swarms.
   - HiCache L2 host-RAM parking for inactive conversation branches.
3. **Ada Lovelace (SM89) Dual-GPU Tensor Pipeline:**
   - Custom SM89 tensor core tiling (`GRP_NC=8`) for grouped expert GEMM.
   - Fused SwiGLU + Q8_1 quantization kernels.
   - Vectorized 128-bit `float4` transactions (`native_moe_combine`, `f32_to_bf16_bulk`).
   - Persistent 72MB L2 cache window pinning verify arena and scratchpads.
4. **Chunk-Pipelined Prefill (`STRATA_PREFILL_CHAIN=1`):**
   - Simultaneous cross-GPU chunk pipelining across dual GPUs, achieving **3,450.2 tok/s cold prefill**.
5. **262K Massive Context Window:**
   - Single-agent master posture scaled to **262,144 tokens** context with **131,072 tokens (128K)** output window on dual 24GB GPUs.
6. **Mathematical Bitwise Parity Suite:**
   - Automated mathematical verification suite confirming 100% bitwise parity against reference ggml across all 18 quant formats.

> [!NOTE]
> **Community Attribution & Upstream Policy:**
> This repository is a standalone downstream research fork. We formally credit and thank **Niko1221** and the 15+ community contributors who created and developed Strata. Please see [**`ATTRIBUTION.md`**](ATTRIBUTION.md) for full contributor credits and upstream policies. To maintain downstream engineering agility while avoiding unsolicited notifications, we do not file PRs or issues against the parent repository. All commits are cleanly isolated for modular cherry-picking.

---

## Benchmark Verification

All benchmarks below were measured directly on the **Mahdi AI Lab** hardware rig running real completions:

- **GPUs:** Dual NVIDIA GeForce RTX 4090 24GB (PCIe 4.0 x16 / x8)
- **CPU:** AMD Ryzen 9 5950X (16-Core / 32-Thread)
- **RAM:** 96 GB DDR4-3200
- **Storage:** WD_BLACK SN850X 4TB NVMe SSD
- **OS:** Ubuntu 24.04 LTS
- **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (IQ3_XXS, 125B MoE, 512 experts, 3.06 bpw)

| Configuration | Model / Quant | Context Window | Concurrency ($C$) | Decode Speed | Cold Prefill | VRAM Expert Residency |
|---|---|---:|:---:|---:|---:|---:|
| **Single-Agent Master** | Qwen3.8-Flash-Next IQ3_XXS | **262,144** | $C=1$ | **105.2 tok/s** | **3,450.2 tok/s** | **89.2%** (21,931 / 24,576 slots) |
| **Multi-Agent Concurrency** | Qwen3.8-Flash-Next IQ3_XXS | 32,768 | $C=3$ | **172.5 tok/s aggregate** | **3,450.2 tok/s** | **89.2%** |
| **Peak Multi-Agent Concurrency** | Qwen3.8-Flash-Next IQ3_XXS | 32,768 | $C=5$ | **217.06 tok/s aggregate** | **3,450.2 tok/s** | **89.2%** |

*All kernel modifications verified with bitwise parity passing 100% against reference ggml implementations.*

---

## Building & Installation

### Requirements
- **Operating System:** Linux (Ubuntu 22.04 / 24.04, Debian 12, or WSL2)
- **GPU:** NVIDIA RTX 4090 / 3090 (24GB recommended; dual 24GB GPUs for full 262K posture)
- **CUDA Toolkit:** CUDA 12.4+ (with `nvcc` and `g++-13`)
- **Build Tools:** CMake 3.24+, Ninja, Python 3.10+

### Compile from Source
```bash
git clone https://github.com/MAHDI-AQ/Strata.git
cd Strata
cmake -B build -G Ninja -DSTRATA_ENABLE_CUDA=ON
cmake --build build -j 16
```

### Mathematical Parity Suite
Verify that all custom SM89 kernels and RadixTree implementations match reference outputs bit-for-bit:
```bash
./build/iq_multi_parity
./build/s2_expert_grouped_parity
./build/native_grouped_parity
./build/radix_tree_test
```

---

## Serving & API Usage

### 1. Starting the Server
The built-in server exposes an OpenAI-compatible HTTP interface:
```bash
python3 -m serve.server --port 8096 --host 0.0.0.0
```

### 2. Client Usage (Python / OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8096/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="strata-iq3xxs",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain RadixTree KV cache prefix sharing in 3 concise bullet points."},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

---

## Architectural Portability

All proprietary enhancements in this fork adhere strictly to non-breaking modularity:
- **Non-Breaking Fallbacks:** Dynamic RadixTree, stage overlap, persistent L2, and chunk prefill pipelining fall back cleanly to safe linear paths when running on single-GPU, non-Ada hardware, or standard upstream configurations.
- **Zero Hardcoded Paths:** Engine code paths are dynamic and relative; no machine-specific absolute directories exist in defaults.
- **Modular Commits:** Commits are structured by subsystem, enabling upstream maintainers to cherry-pick individual kernels, scheduling improvements, or bug fixes with minimal friction.

---

## License & Attribution

Strata is licensed under the **Apache License 2.0 / MIT License**.
See [**`ATTRIBUTION.md`**](ATTRIBUTION.md) for the full list of community authors and historical lineage.
