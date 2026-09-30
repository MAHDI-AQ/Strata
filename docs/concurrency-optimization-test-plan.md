# Concurrency optimization: hypothesis and validation plan

Status: campaign complete; results, rejected promotion, explicitly deferred hypotheses and restored safe-default validation are recorded in [the campaign report](concurrency-optimization-results.md).

## Objective and scope

Improve aggregate decode throughput beyond the qualified c=4 build without sacrificing tokenwise correctness, c=1 performance, request isolation or usable latency. Learn from Ninfer and llama.cpp without assuming their fully GPU-resident models predict Flash Next performance.

Target: native Windows, one RTX 5090, current Swift Flash Next GGUF. Keep the user's voltage/clock settings, 2.5 GiB VRAM reserve, vision off, existing API/model IDs and 98,304-token c=4 context per request. Keep MTP settings unchanged unless an explicitly isolated experiment requires otherwise. No ROCm, multi-GPU, c>4, model requantization, DFlash or exclusive RAM/VRAM placement in this campaign. Prefix reuse is a separate TTFT experiment.

Operate one resident model at a time, stop the current backend before replacement, and serialize model-loading tests. Preserve the known-good binary and presets. Do not change GPU power controls or watchdog settings. Stop a run on memory pressure, allocation errors, watchdog events or corruption; do not count that run as valid performance evidence.

## Frozen references

- Current fork: `a2905719fa4e0820fe4babd463b8ab0f05a4d853`.
- Qualified executable SHA-256: `4eca46ae4b31137a00845b44cf822d99cda34646ad411ab6da88ffe93b447dcc`.
- Published stock reference: preserve the existing stock binary and record its hash before running comparisons.
- Ninfer inspected: `d44ab58408aa389728cd8b1ee50179527e1f3e0d`.
- llama.cpp inspected: `931351ea50dfdd3ee249606f655eef2e9a629daf`.
- Current c=4: batch rows 16, padding on, parallel projections on, graph cache 8, prefill chunk 1,024, automatic expert cache.
- Historical matched measurements: current c=1 162.96 aggregate decode TPS / 157.97 wall TPS; current c=4 231.19 / 217.68. These are references, not pass thresholds for a different workload or a different session.

Use the existing `tools/bench_concurrency.py`, `tools/summarize_concurrency.py`, width tests and local parity/lifecycle harnesses. Extend missing coverage rather than introduce another benchmark stack. See [qualified results](concurrency-throughput.md).

## Mandatory gate for every hypothesis and code change

Each independently testable change, including instrumentation and scheduling changes, follows this sequence. Do not accumulate several unqualified optimizations into one candidate.

1. Record the hypothesis, affected execution route, expected mechanism, baseline revision, candidate diff, settings and success criteria before testing.
2. Run relevant low-memory unit/kernel checks. Any new arithmetic path must also be checked against independently decoded weights and an FP32/FP64 mathematical reference; comparison against another optimized kernel alone is insufficient.
3. Run tokenwise parity before accepting performance results. Compare raw token IDs, lengths and finish reasons per request, not rendered text or aggregate hashes alone. Record the first mismatch, request, position, seed and route. Throughput from a mismatching run is invalid for promotion.
4. After correctness passes, run an isolated A/B performance comparison with instrumentation disabled. Retain the existing numerical and token evidence alongside timings.
5. Keep or reject the change. Save a qualified checkpoint before the next experiment. A small follow-up edit invalidates the affected checks and must pass them again.

### Three complementary parity tracks

**A. Stock parity:** rerun the established 18-case suite for candidate c=1 and c=4 at matched 1,024-token prefill. Each of the four simultaneously active slots must be compared with its own stock response. Rotate fixtures through all slots and include c=2/c=3 transitions where the changed code supports them. Preserve the existing fixed numerical controls: fixed cache profile, 6,000 requested slots, CPU misses, adaptation/suffix/prefix reuse/short-read/prefill-borrow disabled, and `STRATA_NO_IQ512=1`, `STRATA_NO_IQ256=1`, `STRATA_NO_IQ4NL=1`. Record exact resolved arguments rather than relying on defaults.

**B. Changed-route parity:** prove the optimized path actually executed. The strict stock configuration can bypass GPU kernels, so passing it alone does not qualify a GPU optimization. Compare candidate against the frozen fork using identical weights, cache residency, routing, activation quantization, sampling and prompt lengths. Exercise GPU hits, CPU misses and mixed dispatch separately, with fixed placement. Add route counters or a small deterministic route fixture when necessary. Compare per-layer/operation outputs as well as complete generated token streams; preserve per-token accumulation order where feasible.

**C. Realistic serving parity:** use normal settings and collect raw token streams for baseline and candidate at c=1..4, including uneven arrivals and lengths. Automatic residency and CPU/GPU rounding already permit differences, so a mismatch must be investigated and reproduced with matched placement before attribution. Do not silently label it harmless, relax the gate after seeing results, or claim unconditional production parity. If exact parity on the changed path remains unresolved, keep the candidate experimental and retain the known-good default. A change requiring a weaker correctness contract needs explicit user agreement.

The core suite includes prose, code, arithmetic, Chinese, greedy and seeded sampling, repetition/presence penalties, short and long prompts, unequal output lengths and repeated prompts. Add boundary cases relevant to each change. Re-run the full core suite after every retained hypothesis; use smaller targeted checks during its development, never as the final substitute.

## Measurement protocol

- Preserve prompts, seeds, output caps, context, model files, MTP settings and reserve within an A/B comparison. Record changes in resolved cache capacity/workspace size; do not hide their performance cost.
- Separate cold startup/capture, warmup, prefill and steady decode. Capture model hashes, binary hashes, configurations and observed clocks/memory pressure without changing the undervolt.
- Primary workload: the existing four related relativity essays, 1,024 output tokens each, at c=1/2/3/4. Add different-topic prompts to test expert-routing diversity and staggered/unequal requests to test churn.
- Confirmation workload: 8,192 output tokens per request where the model reaches that length, with actual active-batch intervals recorded. Include a 16,384-token-context diagnostic comparable in shape to Ninfer's saturation test, but judge deployment gains at the unchanged 98,304-token setting. Never claim a direct model-to-model benchmark.
- Run at least three paired A/B trials for retained candidates, alternate order, and use one model at a time. Start with the smallest useful screen; expand only for promising candidates or unresolved variance. If thermal/foreground activity makes pairs incomparable, retain and label them and repeat under comparable conditions.
- Report aggregate committed decode TPS, output tokens / complete wall time, per-request TPS and TTFT, inter-token stalls, MTP acceptance/tokens per round, actual active-batch distribution, RAM/VRAM peak and expert hit/miss information. Aggregate TPS is never multiplied by concurrency.
- Report each trial and the median/range. For percentile latency, use a separate workload with at least 20 requests; do not infer p95 from four responses. Keep prefill/ramp-up/drain in wall metrics even when excluded from steady decode.
- Predeclared practical screen: seek at least 5% median c=4 decode or wall improvement with consistent paired direction. Investigate a >3% c=1 regression or >5% TTFT/wall regression; do not automatically ship a tradeoff. Smaller gains may be retained only if repeatable, low-complexity and without material costs. These are engineering thresholds, not claims of statistical significance.

## H0: establish the target execution bottleneck

Hypothesis: remaining time is dominated by target work, but existing host-wait counters cannot distinguish GPU computation from coordination overhead.

Add or enable bounded stage measurements for projections/attention, expert routing, GPU experts, CPU misses, transfers, output head, commit, MTP and cache adaptation. Record expert group-size distributions, distinct experts per layer, cache hits/misses and all-resident versus mixed layers. Correlate critical-path timing; do not add overlapping CPU/GPU durations as if they were serial. GPU utilization alone is insufficient.

Parity: instrumentation on/off must produce identical IDs under fixed controls at c=1 and all c=4 slots. Verify profiling is disabled for final throughput tests and that the supposedly uninstrumented path has no added synchronization.

Decision: rank H1-H4 from measured critical-path opportunity. Historical phase counters put drafting near 10% of measured decode-stage time: eliminating drafting alone cannot justify a doubling claim. If a proposed component is tiny, defer it instead of implementing it for architectural neatness.

## H1: expert weight reuse across small token groups

Hypothesis: the current grouped expert kernels repeat weight decoding inside the per-entry loop. Reusing decoded fragments across several token accumulators may reduce work when multiple rows choose the same expert.

Prototype separate 1-token and small multi-token paths for the actual formats in both GGUF shards, including Q2_0 down projections. Dispatch by per-expert occupancy. Compare scalar/vector reuse first; test a tensor-core path only if occupancy and packing costs support it. Account for register pressure, scratch and cache-capacity changes. Do not optimize only the quantization name in the model filename.

Tests: every present format pair; widths 1,2,3,4,8,9,12,16; one shared expert versus dispersed experts; reversed token mapping; 160 routed entries; tail groups and output guards. Validate quantization boundaries and GPU graph replay with changed inputs. Require tracks A/B/C, including forced resident use of every new kernel. Measure microbenchmarks and complete requests; a microbenchmark win is not an end-to-end win.

Reject if gains disappear on diverse routing, correctness changes remain unresolved, or scratch reduces expert-cache capacity enough to erase throughput gains.

## H2: GPU-resident planning for the batch coordinator

Hypothesis: building resident expert plans on the GPU can reduce dependency on the host, especially on all-hit layers. Single-request resident planning is a starting point, not proof that the batch route is covered.

Implement an isolated resident fast path while preserving the existing mixed/CPU-miss fallback. Keep shared cache updates at safe boundaries. Preserve Windows-safe graph transfers and doorbell ordering; do not reintroduce the previous WDDM memcpy/doorbell deadlock pattern. Avoid broad asynchronous redesign until the isolated path is measured.

Tests: all-hit, one miss, many misses, duplicate experts across slots, format changes by layer, adaptation boundaries, cancellation and slot reuse. Run all parity tracks with forced hit/miss patterns and verify neither stale plans nor skipped CPU contributions occur. Measure host service work, GPU stalls, CPU load and final TPS. Retain only a measured critical-path benefit.

## H3: batched MTP and commit

Hypothesis: batched draft steps amortize projection/head work and synchronization across requests. Expected benefit is bounded by the measured draft/commit share.

First batch independent MTP work without changing draft count, confidence threshold, proposal vocabulary or acceptance rules. Keep per-request RNG, KV/recurrent state, positions and valid lengths separate. Keep commit changes separable from draft batching so their effects can be measured independently.

Tests: zero/partial/full draft acceptance; different accepted lengths in the same round; EOS inside a window; short output budgets and context boundaries; c=1..4; seeded sampling; cancellation and immediate slot reuse. Compare emitted IDs and subsequent continuation after rollback, not just draft agreement. Require all parity tracks. Record accepted tokens per round and draft time, not merely total TPS.

## H4: selected packed projections

Hypothesis: a measured projection bottleneck may benefit from one cross-request matrix operation instead of overlapping per-request operations. The earlier broad packing prototype gave little benefit and is not evidence to repeat it unchanged.

Select only a costly projection/head identified by H0. Compare current overlap, packed existing kernels, and a shape-appropriate kernel including gather/scatter and quantization cost. Preserve per-request attention/state boundaries and private scratch. Qualify rounding/accumulation differences with all parity tracks. Stop if the end-to-end benefit is absent; do not rewrite all dense layers speculatively.

## H5: graph reuse across changing slot identities

Hypothesis: supplying slot pointers/IDs as stable graph input data instead of graph-key identity improves churn and first-use latency. Current steady c=4 captures are already few, so this is principally a latency/turnover experiment.

Test active subsets 1..4, slot replacement, cancellation, variable speculative lengths, context envelopes and repeated changes beyond the graph-cache capacity. Check exact continuation parity, stale pointer isolation, capture counts, graph memory and first-token stalls. Account for pointer-indirection overhead in steady decode. Do not inflate graph cache or reduce reserve to manufacture a win.

## H6: concurrent prefix reuse (separate TTFT track)

Hypothesis: retaining a complete reusable prefix state reduces DSH follow-up TTFT. It does not explain Ninfer's published saturated decode scaling, whose prefix reuse was disabled.

Design the checkpoint around all continuation state required by Flash Next, including KV, recurrent/convolution state, positions, MTP alignment and any other model-specific state. A KV-only checkpoint is insufficient. Budget retained prefixes against expert-cache and active-request memory.

Test exact repeated prefixes, one-token divergence, shared prefixes, interleaved conversations, eviction/restore, cancellation, and four simultaneous continuations. Compare reused and full-prefill output IDs under matched numerical/chunking controls; investigate differences rather than waive parity because chunking changed. Report hot/cold TTFT, memory cost and any saturated throughput loss separately. Defer if this distracts from the throughput objective.

## Final integration and promotion

1. Combine only independently qualified changes. Repeat parity on the combination: separate passes do not establish combined correctness.
2. Repeat stock parity for c=1 and c=4, changed-route coverage, slot rotation and realistic serving checks. Re-run the 19 lifecycle checks plus any new cases; verify four simultaneous HTTP streams, valid usage, finish reasons and SSE completion.
3. Compare the integrated candidate against frozen `a290571` using the full retained performance matrix. Report improvements and regressions, rejected hypotheses, actual memory use and correctness limits.
4. Promote only after the parity/performance/resource gates pass. Preserve c=1 and vision presets and API compatibility. Update the report and launch configuration, then leave the requested qualified preset running. Commit/push when authorized for the execution campaign.

For every hypothesis retain a compact result: baseline/candidate hashes, hypothesis, route coverage, parity case/token counts and first mismatch if any, benchmark trials, resource changes, keep/reject decision and remaining uncertainty. Never change comparison criteria after seeing an unfavorable result.
