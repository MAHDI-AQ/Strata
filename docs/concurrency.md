# Experimental concurrent serving

This branch adds configurable shared-model serving for **1–4 requests**. It is based on upstream 0.1.27, commit `a79080535d1b2a71a3419a0d97d8e7dca194b0f1`. Model correctness and performance require GPU validation; compilation and CPU/mock tests do not establish inference parity or throughput.

## Configuration

Add these engine arguments to the usual server JSON `args`:

```text
--concurrency 4 --batch-rows 8 --batch-policy fair --concurrent-prefill 256
```

| Option | Values | Meaning |
| --- | --- | --- |
| `--concurrency` | 1–4, default 1 | Maximum active sequences; 1 retains the existing serving path |
| `--batch-rows` | 1–8, default 8 | Total target-verification rows in a scheduling round |
| `--batch-policy` | `fair` or `depth` | Share rows across ready requests, or prioritize longer windows |
| `--concurrent-prefill` | 256–1024, default 256 | Maximum prompt chunk before returning to ready decode work |

With four ready requests wanting four rows each, `fair` at eight rows gives each two rows. `depth` gives two requests four rows each, then rotates admission order. Smaller budgets also rotate, so requests get turns. A row is one target input token, including the real token and any speculative tokens. This is not a sixteen-row implementation.

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

Each request commits only its own accepted tokens. Graph layouts are cached by slot/window shape with an eight-layout bound. Prompt work uses one dedicated shared workspace and at most one prompt chunk per round. Expert residency changes occur only after GPU work has completed.

## Memory and limitations

Concurrency adds independent sequence state and verifier workspaces. Slots are allocated before automatic expert-cache sizing, with extra headroom reserved for later verifier allocations. Allocation checks fail rather than deliberately consuming the requested VRAM reserve. This is not a hard process-wide VRAM cap: CUDA graphs and runtime allocations still require empirical measurement.

The expert arena remains in system RAM and VRAM experts remain cached copies. This branch does **not** implement exclusive RAM/VRAM expert placement or free cached experts from RAM. More slots may reduce expert cache capacity, so higher concurrency is not guaranteed to increase throughput. Start validation at a moderate context length, such as 32768 per request, rather than multiplying a 196608-token configuration by four.

HTTP output queues are bounded per request; a client that stops consuming output is canceled without blocking other streams. Engine failure terminates affected requests and requires an explicit server restart in concurrent mode. The monitor's live text/rate fields are a shared latest-update view, not separate per-request charts; completion histories and timings are request-local.

## Lightweight verification

```text
python -m unittest discover -s tests/concurrency -p "test_*.py" -v
cmake -S . -B build-cpu -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONCURRENCY_TESTS=ON
cmake --build build-cpu --target strata-batch-schedule-test
ctest --test-dir build-cpu -R strata-batch-schedule-test --output-on-failure
```

The scheduler test checks 65,536 combinations, bounds, policy allocation and rotating progress. The Python tests use mocked engine pipes for four-stream isolation, cancellation, backpressure, EOF and legacy statistics. Neither test loads model weights or runs GPU inference.

The Windows development checks also run the upstream `serve.test_server` and `serve.test_mcp` suites. On the 0.1.27 base, all 63 upstream tests and all six concurrent-serving tests pass. The scheduler executable passes in Release mode with its assertions explicitly retained.

The complete native Windows Release executable builds successfully with MSVC 19.32, CUDA 13.2 and `CMAKE_CUDA_ARCHITECTURES=120-real`. Its help output exposes the new controls, and invalid concurrency/unsupported configuration checks exit before loading weights. No GPU inference or model parity/performance validation has been performed.

## Required model validation before everyday use

1. Compare greedy single-request outputs on the original and concurrent paths with matching model, context and sampling settings.
2. Compare c=2 and c=4 against independent reference requests, including unequal prompt lengths, penalties, early EOS, context limits, cancellation and slot reuse.
3. Exercise fair/depth policies, row budgets 1/4/8, MTP confidence changes, suffix drafting and adaptive cache updates.
4. Measure peak RAM/VRAM, time to first token, aggregate throughput and per-request latency, including a new long prompt arriving during decode.
5. Verify memory stability over repeated admission/cancellation and multiple graph shapes.

On the development machine, **do not launch model validation until the user explicitly approves RAM/VRAM-heavy testing**. Closing an earlier model process does not waive that instruction.
