# Experimental concurrent serving

> [!NOTE]
> **Environment Context:** This document records early exploratory single-GPU testing on Windows (RTX 5090). For our production Dual NVIDIA GeForce RTX 4090 (Ubuntu 24.04) multi-agent benchmarks, context scaling (262K), and RadixTree performance, see the root [`README.md`](../README.md).

This branch adds configurable shared-model serving for **1–4 requests**, raised to **1–8 requests** in the pair-combine build (the qualification tables below were established at c=1–4). It is based on upstream 0.1.27, commit `a79080535d1b2a71a3419a0d97d8e7dca194b0f1`. Native Windows qualification on an RTX 5090 established exact token parity at c=1, 2, 3 and 4 under the matched numerical settings below. This remains a single-GPU experimental implementation; the qualification is specific to Swift IQ2_XS and the tested configuration, not a guarantee across all models and settings.

The subsequent throughput work adds up to sixteen shared expert rows, optional padded verification and parallel request projections. See [the throughput study](concurrency-throughput.md) for matched measurements, current qualification and recommended settings. The qualification tables later in this document describe the original implementation.

## Configuration

Add these engine arguments to the usual server JSON `args`:

```text
--concurrency 4 --batch-rows 8 --batch-policy fair --concurrent-prefill 256
```

| Option | Values | Meaning |
| --- | --- | --- |
| `--concurrency` | 1–8 in the pair-combine build, default 1 | Maximum active sequences; 1 retains the existing serving path |
| `--batch-rows` | 1–16, default 8 | Total target-verification rows in a scheduling round |
| `--batch-graphs` | 1–64, default 8 | LRU cache of graph layouts, bounded by available VRAM |
| `--batch-padding` | 0 or 1, default 0 | Pad short windows to the MTP width when the row budget permits; extra outputs are discarded |
| `--batch-parallel` | 0 or 1, default 0 | Overlap independent request projections in each layer before shared expert dispatch |
| `--batch-policy` | `fair` or `depth` | Share rows across ready requests, or prioritize longer windows |
| `--concurrent-prefill` | 256–1024, default 256 | Maximum prompt chunk before returning to ready decode work |

With four ready requests wanting four rows each, `fair` at eight rows gives each two rows; at sixteen rows all four retain their full windows. `depth` gives earlier requests longer windows, then rotates admission order. Smaller budgets also rotate, so requests get turns. A row is one target input token, including the real token and any speculative or discarded padding tokens.

MTP stays enabled and keeps the normal confidence threshold. Each request drafts independently. A short target window can leave some draft work unused; performance must be measured before choosing the best policy. The scheduler does not execute another request concurrently inside a running MTP graph.

## Supported first implementation

- Native Windows CUDA, one NVIDIA GPU, native experts and profile-filled expert cache.
- Text requests, resident INT8 KV, MTP with speculative windows of 2–8 rows.
- Per-request sampling, penalties, stop/cancellation and streaming responses through the existing Python HTTP server.
- One model/expert arena and shared immutable MTP weights. Each request has separate attention, recurrence, PLE history, draft KV, sampling and rollback state.
- Adaptive expert-cache updates at completed scheduling boundaries.

Concurrent mode rejects vision, ROCm, multi-GPU, streamed KV, control vectors and split-window verification. It does not reuse conversation-prefix checkpoints. The original single-request path remains available with `--concurrency 1`.

## What is batched

For each layer, the combined CUDA graph runs each request's mixer/router, packs the routed-expert inputs, dispatches experts once over the combined rows, scatters the results, then completes each request's layer. Identical experts across requests can share dispatch and fetch work. Attention, dense projections and MTP remain per request. Existing native GPU expert kernels are reused; this does not add a new matrix-matrix expert kernel.

Each request commits only its own accepted tokens. Graph layouts are cached by slot/window shape with a configurable LRU bound. Padding preserves the original acceptance limit and MTP work; causal outputs beyond that limit are discarded and their state rolled back. Parallel mode uses separate request streams and private PLE scratch, joins them before expert dispatch, and shares only immutable weights. Prompt work uses one dedicated shared workspace and at most one prompt chunk per round. Expert residency changes occur only after GPU work has completed.

## Memory and limitations

Concurrency adds independent sequence state and verifier workspaces. Slots are allocated before automatic expert-cache sizing, with extra headroom reserved for later verifier allocations. Allocation checks fail rather than deliberately consuming the requested VRAM reserve. This is not a hard process-wide VRAM cap: CUDA graphs and runtime allocations still require empirical measurement.

The expert arena remains in system RAM and VRAM experts remain cached copies. This branch does **not** implement exclusive RAM/VRAM expert placement or free cached experts from RAM. More slots may reduce expert cache capacity, so higher concurrency is not guaranteed to increase throughput. Start validation at a moderate context length, such as 32768 per request, rather than multiplying a 196608-token configuration by four.

HTTP output queues are bounded per request; a client that stops consuming output is canceled without blocking other streams. Engine failure terminates affected requests and requires an explicit server restart in concurrent mode. Live TPS uses a monotonic aggregate token counter; `/metrics` also exposes `active_requests` with individual counters. Progress log lines include request IDs. Completion histories and timings remain request-local. The existing monitor UI does not draw separate per-request charts.

## Lightweight verification

```text
python -m unittest discover -s tests/concurrency -p "test_*.py" -v
cmake -S . -B build-cpu -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONCURRENCY_TESTS=ON
cmake --build build-cpu --target strata-batch-schedule-test
ctest --test-dir build-cpu -R strata-batch-schedule-test --output-on-failure
```

The scheduler test checks 131,072 combinations, bounds, policy allocation and rotating progress. The Python tests use mocked engine pipes for four-stream isolation, cancellation, backpressure, EOF, aggregate live rates and legacy statistics. Neither test loads model weights or runs GPU inference.

The Windows development checks also run the upstream `serve.test_server` and `serve.test_mcp` suites. On the 0.1.27 base, all 63 upstream tests and all six concurrent-serving tests pass. The scheduler executable passes in Release mode with its assertions explicitly retained.

The qualified native Windows Release executable uses MSVC 19.32, **CUDA 13.0**, `STRATA_PORTABLE=ON`, and `CMAKE_CUDA_ARCHITECTURES=120-real`. Its help output exposes the new controls, and invalid concurrency/unsupported configuration checks exit before loading weights. Do not substitute the earlier CUDA 13.2 development executable: even the unmodified stock source built with that configuration failed repeatability.

### Exact-token qualification settings

Raw-token comparisons use Swift IQ2_XS, 32768 context, resident INT8 KV, MTP/spec 4 and confidence 0.5, a fixed profile/cache budget (10000 requested, 10501 actual slots), CPU misses and a 2048 MiB reserve. Both the published stock reference and modified engine use:

```text
--expert-cache 10000 --prefill 256 --short-read 0 --pcie-frac 0
--adapt-every 0 --suffix-draft 0 --prompt-cache 0 --no-prefill-borrow
--vram-reserve-mib 2048 --max-context 32768 --kv int8
--spec 4 --spec-min-p 0.5
```

Set these environment variables **for both sides of a strict A/B comparison** (the JSON server config accepts an `env` object):

```json
"env": {
  "STRATA_NO_IQ512": "1",
  "STRATA_NO_IQ256": "1",
  "STRATA_NO_IQ4NL": "1"
}
```

Upstream CPU expert dispatch selects different arithmetic for a singleton and a multi-token group ([issue #152](https://github.com/Niko1221/Strata/issues/152)). Those paths round differently. The flags select the same GGML row arithmetic at every width, at a potential throughput cost. The real-weight `strata-expert-width-test` reproduces the difference: 1687 differing FP32 cells in the tested layer with default dispatch, zero with these flags at widths 1, 2, 3, 4 and 8 across three tested layers. The test takes a compatible model shard path; it maps a few tensors and does not start GPU inference.

Cache residency also changes CPU/GPU arithmetic. Automatic cache sizing, adaptive migration, PCIe offload and different prompt chunking are not part of this strict-parity configuration. Exact token equality is not promised when changing those settings. MTP confidence and suffix-draft variations remain outside this qualification.

The 18-case suite includes prose, code, arithmetic, Chinese, repetition penalties, repeated prompts in different slots, seeded sampling, unequal output lengths and prompts up to 5692 tokens; outputs run to EOS or caps of 128/256 tokens. The official stock executable is the reference. A separate unmodified local CUDA 13.0/portable build matched the official executable on all eight initial cases, including a restart repeat. No throughput improvement is implied by token parity: dense operations and MTP remain per request.

| Modified configuration | Exact stock matches | Aggregate tokens/s |
| --- | ---: | ---: |
| c=1 | 18/18 | 60.06 |
| c=2, fair, 8 rows | 18/18 | 53.71 |
| c=3, fair, 8 rows | 18/18 | 50.75 |
| c=4, fair, 8 rows | 18/18 | 50.79 |
| c=4, depth, 8 rows | 18/18 | 48.96 |
| c=4, fair, 1 row | 18/18 | 44.88 |
| c=4, fair, 4 rows | 18/18 | 47.50 |

All 126 responses matched. Each run produced 2406 tokens. Timings include prefill and graph capture, exclude model loading, and use groups submitted together with a barrier between groups. They are single-run observations, not isolated decode measurements or a saturated arrival benchmark. Published stock measured 59.94 tokens/s in the same suite. This workload showed **no aggregate throughput gain** from concurrency. c=4/fair/8 executed 114 four-request target rounds; a one-row budget intentionally time-slices the requests.

The real-model lifecycle test passed all 19 checks: queued/prefill/decode cancellation, unaffected-request parity, twelve slot-reuse comparisons over three rounds, invalid-request rejection and valid-request recovery. The tested native executable's SHA-256 is `609b243b3375e413f565db2c25adfebbfa58a06e7981993c5bafb1d2d790dffa`; the official reference is `814c016245a6e407c45d070cdbd77cea4b92445d57c47001e8289b89f3d4b3e0`.

A real HTTP/SSE test observed four active requests simultaneously, four distinct response IDs, complete streams and exact decoded stock-reference text for all four requests. It used the built-in template on both sides; the ordinary server chooses the pack's template when present. All six HTTP assertions passed. The scheduler and real-weight CPU width tests were rebuilt and passed with the qualified CUDA 13.0/portable build configuration.

### Earlier failed controls

The initial CUDA 13.2/nonportable builds failed even unmodified-stock repeatability. Rebuilding with CUDA 13.0 and portable settings resolved the observed corruption; the compiler version and build options were changed together, so this does not isolate a compiler defect. A sanitizer diagnostic on the old stock build reported MMQ uninitialized reads and then hit the Windows timeout; it was incomplete and is not a proven cause. Shared-graph replay stalls were separately fixed with kernel row copies, explicit graph upload/synchronization and kernel expert copying.

On the development machine, **do not launch model validation until the user explicitly approves RAM/VRAM-heavy testing**. Closing an earlier model process does not waive that instruction.
