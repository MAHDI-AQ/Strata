# Concurrency throughput on native Windows

The optimized c=4 path combines shared expert dispatch with overlapping request projections and reusable verification graphs. On the tested RTX 5090, the selected preset reached **230–233 aggregate decode tokens/s**, compared with **109–110** on the previous fork. These are totals across four requests, not per-request rates. A smaller prefill workspace measured 233–237 TPS but had substantially worse long-prompt TTFT.

## Configuration

Add to the existing native, single-GPU, text-only configuration:

```text
--concurrency 4 --max-context 98304 --vram-reserve-mib 2560
--batch-rows 16 --batch-padding 1 --batch-parallel 1 --batch-graphs 8
--concurrent-prefill 1024
```

The context limit is **98,304 tokens per request**, including input and output. Vision remains unsupported in concurrent mode. MTP remains enabled with spec 4 and confidence 0.5; padding does not request extra drafts or change acceptance decisions. The original c=1 path and c=1 vision preset remain available. Experimental options default off; the row budget still defaults to eight for compatibility.

## What changed and why

1. **Parallel copies.** The integer-copy kernel was also being used for large device tensor transfers between request workspaces and the expert coordinator. Its single block serialized those transfers on one SM. A bounded grid now handles large copies; small control copies remain small.
2. **Sixteen expert rows.** Four requests can retain four verification rows each instead of sharing eight. This required widening coordinator storage, expert plans and CPU workspace limits. Both the 128-entry expert-plan assumption and CPU kernels that silently processed only eight rows had to be fixed. Sixteen rows can route 160 expert entries.
3. **Stable graph shapes.** Confidence-controlled MTP windows vary in length, generating many combinations across four slots. Padding short windows to the MTP width makes graph reuse practical. Padding is causal, stays inside context and row limits, and its outputs are never accepted or committed. Original request window lengths still govern acceptance, output limits and draft catch-up.
4. **Parallel request projections.** Each layer forks independent request work onto separate CUDA streams, then joins before shared expert dispatch. This lets the GPU overlap otherwise serialized kernels. Attention/recurrent state remains independent; dense weights are shared read-only. PLE scratch had to become private per parallel slot. MTP and commit work remain per request.
5. **Bounded graph memory.** Graph caching uses LRU replacement. Automatic expert-cache sizing leaves additional space for wider workspaces and graph storage. Capturing a graph evicts cached layouts as needed; failure to retain the requested VRAM reserve produces an explicit error.
6. **Accurate serving telemetry.** Live TPS uses one monotonic aggregate token counter. Request admissions no longer reset another request's samples. `/metrics` exposes active request counters, and progress lines identify their request. The preset router can choose another private backend port when Windows retains the old port after a model switch; its public endpoint stays unchanged.

These changes reuse the existing native expert kernels. They do not implement a new tensor-core MoE kernel or pack all dense projections into one matrix multiplication. They also do not implement exclusive RAM/VRAM expert placement: the expert arena remains in RAM with cached GPU copies.

## Measurement method

Hardware: RTX 5090, Ryzen 9950X3D, 48 GB DDR5, native Windows. GPU voltage/clock settings were left unchanged. Build: MSVC 19.32, CUDA 13.0, portable CPU build, SM120. Model: Swift Qwen3.8 Flash Next GSQ-RCO IQ2_XS. The GGUF uses mixed expert formats; testing only the format named in the filename is insufficient.

The native-protocol harness is `tools/bench_concurrency.py`. It records executable hashes, arguments, relevant environment, prompt tokens, raw output IDs and token arrival timestamps. Each process loads one model, performs a separate warmup, then runs four related relativity essays capped at 1,024 output tokens each, repeated twice. Thinking is disabled and sampling is greedy. Context is 98,304, vision is off, expert cache is automatic and reserve is 2,560 MiB. c=1 runs the four requests sequentially; c=4 submits them together. Startup is excluded.

**Aggregate decode TPS** counts tokens during the common interval where all batch members are producing output, trimming 0.25 seconds from each end. **Wall TPS** divides all outputs by the batch makespan, including prefill and graph capture. These quantities must not be interchanged, nor should an aggregate rate be multiplied by concurrency.

| Configuration | Aggregate decode TPS | Wall TPS |
| --- | ---: | ---: |
| Previous fork, c=4, eight rows | 109–110 | 106–107 |
| Previous fork, c=1 (weighted mean) | 162.83 | 157.35 |
| Optimized fork, c=1 (weighted mean) | 162.96 | 157.97 |
| Corrected sixteen rows, dynamic shapes, 32 graphs | 120–128 | 116–124 |
| Sixteen rows, padded, serial projections | 154–156 | 148–149 |
| Sixteen rows, padded, parallel projections | 233–237 | 218–223 |
| Selected preset: same, prefill chunk 1,024 | 230–233 | 216–219 |

The previous fork is commit `700af00` (native binary SHA-256 `609b243b3375e413f565db2c25adfebbfa58a06e7981993c5bafb1d2d790dffa`). Its matched c=1 essay runs measured roughly 158–167 decode tokens/s. Measurements are two-run observations on this machine, not a universal scaling guarantee. Different request lengths, routing, prompt sizes, cache residency and foreground applications change results.

Across all 8,192 output tokens per configuration, weighted aggregate decode rates were 109.70 TPS for previous c=4 and 235.26 TPS for optimized c=4: **2.14×**. Optimized c=4 was **1.44×** optimized c=1 in decode and **1.40×** in wall throughput. The c=1 difference between builds was below one percent.

The selected 1,024-token prefill preset trades a little decode throughput for lower TTFT. Its weighted rates were **231.19 aggregate decode TPS** and **217.68 wall TPS**, about **2.1×** the previous c=4 decode rate and **42%** above optimized c=1. With one active request and a 7,086-token prompt, measured TTFT fell from **8.46 s to 3.28 s** when increasing the chunk from 256 to 1,024. This is a synthetic native-protocol measurement, not a DSH UI timing. The larger workspace reduces available expert-cache space. [Machine-readable throughput results](concurrency-throughput-results.json) include weighted means and executable hashes.

In the two-repeat optimized run, graph captures fell to seven, taking about 0.26 seconds. The wider dynamic run captured 546 layouts, taking about 15.4 seconds. The improvement comes from reducing this overhead and exposing independent GPU work, while retaining batched expert dispatch.

## Correctness and regression coverage

Both optimized c=1 and c=4 matched all **18 reference responses token for token** against the published stock binary under fixed numerical controls (2,409 output tokens per run). The suite covers prose, code, arithmetic, Chinese, seeded sampling, penalties, unequal lengths, repeated prompts and prompts up to 5,692 tokens. It uses 6,000 requested cache slots, 98,304 context, 256-token prefill chunks, no adaptation/PCIe misses/prefix cache/suffix drafts, and:

```text
STRATA_NO_IQ512=1
STRATA_NO_IQ256=1
STRATA_NO_IQ4NL=1
```

These controls select consistent CPU arithmetic across expert group widths. **Default adaptive operation does not promise token-identical results when concurrency changes cache residency or CPU/GPU routing.** Performance runs use normal settings and are separate from strict parity qualification.

The c=4 suite was then repeated with **1,024-token prefill chunks on both stock and fork**: all 18 responses also matched exactly. In total, the c=1/256, c=4/256 and c=4/1024 configurations passed 54 matched response comparisons. The final native executable has SHA-256 `4eca46ae4b31137a00845b44cf822d99cda34646ad411ab6da88ffe93b447dcc`.

CPU expert regression tests now execute the actual pool path, including Q2_0 down projections, across each format pair found in a shard. GPU tests exercise widths 1, 2, 3, 4, 8, 9, 12 and 16 with 160 routed entries and nontrivial token/output indexing. The copy test checks mapped-host and device sources, boundary sizes, guards and graph replay with changed inputs. Scheduler tests cover 131,072 cases. Python tests cover multiplexing, cancellation, backpressure, aggregate rates and preset routing.

All 19 real-model lifecycle checks passed: queued, prefill and decode cancellation; unaffected-request parity; twelve slot-reuse comparisons; invalid-request rejection; and successful recovery afterward. The Python suites passed 13 concurrency/router tests and 63 upstream tests (three additional upstream tests skipped). CPU/GPU width checks passed on both GGUF shards, and the elementwise/copy self-test reported zero failures.

A final API smoke test through the normal preset router confirmed four simultaneously active native requests. All four OpenAI-compatible streaming responses returned HTTP 200, 256 output tokens, valid usage and SSE `[DONE]`; the engine returned to zero active requests afterward. This verifies the deployed serving path separately from the matched native-protocol performance benchmark.

## Rejected measurements and approaches

Early sixteen-row experiments produced implausibly high throughput while generating incorrect/repetitive output. Those results are invalid and are not included above. Full token comparisons exposed the expert-plan limit and CPU Q2_0 truncation; the broader suite later exposed shared PLE scratch in parallel mode. All were fixed before accepting performance results.

A prototype that packed dense projections across requests gave little additional gain after fixing copies and was removed. An optional IQ2_XS-specific reuse kernel was also removed; it did not target the principal expert formats in this model. Simply allowing more graph layouts did not solve shape churn effectively enough.

## Remaining limits

- Only one NVIDIA GPU on native Windows was qualified. Concurrent ROCm, multi-GPU and vision remain rejected by the existing guards.
- Higher concurrency is limited to four requests; a larger row budget does not increase request capacity.
- Long-prompt TTFT still includes prompt processing. Concurrent mode has no conversation-prefix checkpoint reuse, so c=1 can be much faster on repeated chat prefixes. Model switching still requires unloading and loading the resident model.
- MTP drafting remains per request in the serving default. A later overlap experiment is available behind an opt-in flag but was not promoted because its long-form production-parity check failed. Larger batches can reduce the expert cache available per GPU and increase misses; throughput gains depend on workload and available VRAM.

Example benchmark invocation (substitute local paths):

```text
python tools/bench_concurrency.py --config local-c4.json --output c4.json --tokens 1024 --repeat 2 --set batch-rows=16 --set batch-padding=1 --set batch-parallel=1 --set batch-graphs=8
```

Model-loading tests consume substantial RAM/VRAM. The CPU-only Python and scheduler suites can be run separately without loading a model.

The [follow-up optimization campaign](concurrency-optimization-results.md) adds stable logical-slot packing and diagnostic tools. It preserves the serving default rather than promoting a short-benchmark gain that fails a broader parity gate. It also documents existing timing-dependent lookup/adaptation limits and the rejected expert-kernel experiment.
