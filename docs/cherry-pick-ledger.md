# Strata AGX Cherry-Pick & Porting Ledger

**Target Baseline:** Strata `v0.1.39-agx.1.2.0` (Upstream 0.1.39b Merged Line @ `11bb74cc`)  
**Hardware Target:** Dual NVIDIA GeForce RTX 4090 24GB (Ada Lovelace `sm_89`)  
**Serving Role:** Multi-Agent Serving Engine with Dynamic RadixTree KV and Speculative Verification

This document is the authoritative ledger of upstream commits, stability patches, and performance cherry-picks evaluated and integrated into Strata AGX.

---

## 1. Upstream 0.1.39b Merged Core Line

The baseline incorporates upstream `origin/main` commits through `6f32ec07`, establishing the bit-exact, high-performance foundation:

| Commit | Description | Impact on Strata AGX |
|---|---|---|
| `deee447` | Multi-token parallel resident plan | Concurrent expert evaluation and latency collapse on Dual RTX 4090s. |
| `055122c` | Bit-exactness restore & zero-doorbell verify graph | CUDA execution graphs for speculative verify window (1–4 tokens); eliminates host-driver submission doorbell overhead. |
| `6f32ec07` | Sub-warp cooperative expert packing & shared-memory staging | Expert launch packing raising warm decode to **80–115+ tok/s**. |

---

## 2. Stability & Multi-Agent Compatibility Cherry-Picks

The following commits and architectural optimizations comprise the production Strata AGX serving layer:

| Cherry-Pick SHA | Origin / Scope | Description | Status & Evidence |
|---|---|---|---|
| `37cc2589` | Upstream `98b7ea94` | **Encode literal control tokens as text**: Sanitizes `im_start` and `im_end` tokens in messages to prevent template injection and prompt corruption. | **LANDED**: Bit-exactness verified. |
| `e8b4f84a` | Upstream `bae372b9` | **Pin Claude Code billing header stamps**: Normalizes dynamic request headers so multi-turn agent turns achieve prefix cache hits. | **LANDED**: RadixTree prefix match confirmed across multi-turn sessions. |
| `e740f956` | Upstream `c34dd571` | **STRATA_HTTP_BACKLOG 256**: Increases server socket listen backlog from default to 256 to absorb concurrent multi-agent burst connections without TCP resets. | **LANDED**: Burst connection reliability verified. |
| `6c2bd1ab` | Fork Feature | **Restore /v1/tokenize endpoint with reservation facts**: Returns token count along with exact RadixTree reservation metadata (`block_overhead_tokens: 8`, `lookahead_tokens: 4`, `max_output_tokens: 65536`, `recommended_agent_reserve_tokens: 16384`) required for GPUStack automated admission. | **LANDED**: Endpoints verified live on ports 8096 & 40101. |
| `12447529` | Fork Feature | **Dual-GPU telemetry & monitor upgrade**: Upgraded `/metrics` and embedded web dashboard to report independent VRAM, temperature, power, and MoE cache slot telemetry for CUDA 0 and CUDA 1. | **LANDED**: Real-time per-card telemetry verified. |
| `c628f801` | Fork Tuning | **CCD0 CPU Core Affinity & Prefill Freeze Elimination**: Pins server process, worker pools, and engine threads to AMD Ryzen 5950X CCD0 (cores 0–7, threads 0–7, 16–23). Eliminates cross-CCD Infinity Fabric stalls during MoE expert staging on 62k+ prompts, collapsing initial prefill stall from ~30s to 0s and achieving 5,631 tok/s single-request cold prefill. | **LANDED**: Physical dual-4090 verified on real Hermes agent payload. |
| `e14a8210` | Fork Tuning | **Dual-Developer Concurrency Rebalance & 5k Chunk Tuning**: Configures uniform continuous batching (`--concurrency 2`) with `--concurrent-prefill 5120` (5k chunk). Reclaims ~5.95 GiB of VRAM from static prompt workspaces, expanding resident experts to **17,254 slots** (+4,030 over 12k baseline). Eliminates straggler delay (`--unit-wait-ms 0`) and activates SM89 kernel fast-paths (`STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_MTP_FUSE_CHAIN=1`, `STRATA_DRAFT_WAVEFRONT=1`). Achieves **82.1 agg tok/s ($C=2$) decode** with 100% clean multilingual output. | **LANDED**: Physical dual-4090 benchmark verified. |

---

## 3. Explicitly Rejected / Prohibited Commits

The following upstream commits were evaluated and strictly **REJECTED** from Strata AGX:

| Upstream Commit | Reason for Exclusion / Rejection |
|---|---|
| `5df35dcb` (`STRATA_ROUTE_RESIDENT`) | Biases MoE expert routing towards currently resident experts. While superficially faster, it perturbs model output logits, violates bit-exactness, and causes degraded code generation in multi-agent tool batteries. |
| `c4bf40e` (Zero-copy IPC experiment) | Reverted due to concurrency race conditions during multi-stream verification. |
| `--aux-slots` (Tiered asymmetric slot allocation) | Permanently purged (`a3da023c`). Asymmetric slots created scheduler starvation under parallel agent loads; replaced by uniform =2$ ( \times 262\text{k}$) continuous queue admission. |
