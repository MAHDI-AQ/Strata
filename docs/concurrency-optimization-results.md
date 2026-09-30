# Concurrency optimization campaign

**Promotion decision:** retain stable logical slot ordering, profiling tools and stronger parity reporting. Keep overlapped MTP **disabled by default and in the user's serving preset**. It improves the matched 1,024-token c=4 workload by about 5.9%, but the longer normal-serving comparison fails token parity. This campaign does not claim an additional qualified deployment throughput gain.

Execution authorized after the testing plan was approved. Baseline: `a290571`, native Windows RTX 5090, unchanged undervolt, 2,560 MiB reserve. Baseline binary and preset are preserved in local `exports/strata-ab/`. Raw experiment reports live there; model files and local configurations remain untracked.

Performance runs disable GPU timestamps, expert-occupancy counters and round tracing. The harness enables the existing lightweight host timing reports identically in both arms. Mismatch indices below are zero-based. Tokens and finish reasons are compared before a pair contributes to a performance conclusion.

## H0: instrumentation and bottleneck

The baseline with GPU timestamps enabled and disabled produced identical token IDs for all four 1,024-token essays (4,096 tokens). Adaptation was disabled and cache placement matched at 11,172 actual slots. This diagnostic ran at approximately 148-149 aggregate decode TPS; it is not comparable to the deployment preset with adaptation enabled.

Additional opt-in counters separate resident execution, plan waits, PCIe work and CPU waits, and record resident expert group occupancy. The first detailed counter implementation used a stamp absent from the split batch path; its plan-wait values were invalid and were corrected before interpretation. Output parity was unaffected (4,096/4,096 tokens identical).

The corrected adaptive run produced all 8,192 essay tokens identically to the historical deployment run. Its GPU timing totals were: pre-expert 15.109 s, expert dispatch 12.158 s, post 0.914 s and head 1.566 s. Within expert dispatch: plan wait 1.100 s, resident computation 8.199 s, fetch 0.163 s, PCIe computation 1.764 s, CPU wait 0.540 s and combine 0.391 s. These are instrumented measurements including warmup, not performance claims; overlapping host counters must not be added to them.

Of 44,208 layer visits, 5,472 were all-hit (12.4%). There were 3,063,522 distinct expert groups and 173,732 missing groups. Resident occupancy was approximately 49% singleton, 23% pairs, 11% triples, 6.5% groups of four, with the remainder larger. This supports testing small-group reuse, while bounding the all-hit planner opportunity. Projections/attention become the largest stage after adaptation removes most CPU misses.

## H1: paired expert accumulators

An opt-in paired-row implementation interleaves two independent accumulators while retaining per-token summation order, allowing the inlined codecs to reuse weight decoding. Both shards passed exact FP32 width isolation at 1,2,3,4,8,9,12,16 with 160 reversed routed entries. Independent GGUF-dequantized FP64 expert references passed for all three actual format pairs (22/42,16/42,29/42). All 36 strict response comparisons across two suite repetitions matched stock (4,818 tokens).

Microbenchmarks show gains for larger groups in formats 22/42 and 29/42, but little consistent improvement in 16/42. Three adaptive screening waves measured 228.91/232.12/239.24 TPS versus baseline 222.98/232.39/229.74. The median difference is about 1%, below the predeclared screen, and output IDs differed under automatic operation, so these are not valid proof of a causal speedup. Fixed placement with adaptation disabled and explicit PCIe fraction 0.5 produced identical 4,096-token streams, but measured 160.83 versus baseline 161.15 TPS. The prototype is rejected for lack of demonstrated end-to-end benefit and removed from active source. Its [archived patch](experiments/expert-paired-reuse-rejected.patch) is retained for reproducibility; its executable remains in local exports. No claim of unconditional adaptive parity is made.

## H3: overlap independent MTP streams

Before attempting a full packed-projection drafter, test a smaller change: advance every eligible request one draft depth at a time on its private CUDA stream, then collect results together. Keep confidence thresholds, draft limits and per-request state unchanged. This can overlap launches and computation without changing arithmetic. It is not a packed matrix implementation. Commit remains unchanged so its effect is isolated. All 18 strict responses matched stock (2,409 tokens). Measured draft-stage time for the suite fell to 0.556 s from roughly 1 s. The broader qualification below prevents promotion.

The first adaptive comparisons differed in token IDs and were excluded from promotion evidence. Fixed-placement comparisons also initially differed from an older process, but a fresh frozen-baseline repeat and same-binary MTP on/off runs matched all 4,096 tokens. A round trace then exposed a confounder: the scheduler sorted verifier pointers, yielding logical slot order `2,1,3,4` in the control and `1,2,3,4` in the candidate. The expert dispatcher assigns a fraction of misses to GPU in encounter order. Address-dependent packing therefore changes CPU/GPU assignment and numerical rounding; it is not merely a harmless permutation of identical arithmetic. The first traced target-output difference occurred in round 3, with identical logged input windows and before the first cache adaptation.

Packing now uses persistent logical slot indices, preserving graph reuse independently of heap allocation and fairness rotation. The MTP candidate also records lookup-policy cost after drafting, so its observed round time includes the draft work. In the corrected adaptive comparison with explicit PCIe fraction 0.55, all 8,192 IDs and finish reasons match between MTP overlap enabled and disabled. The initial corrected pair measured 237.34 aggregate decode / 222.83 wall TPS with overlap, versus approximately 224.6 / 211.8 without it. Repeated trials follow below. The corrected c=4 build also passes all 18 strict stock cases (2,409 tokens).

The stable-order fix removes a discovered process-allocation dependency; it does not promise that every historical address-ordered run has the same tokens. Qualification compares the changed path at matching logical ordering and retains stock parity under the existing fixed numerical controls.

The corrected control's 8,192 tokens also match the original frozen run `h3-control-c4.json`. However, a later frozen-executable repeat (`h3-confirm-pair2-off.json`) again differs, beginning at positions 61/129/92/47 in its first four requests. That pair was stopped by the parity gate and is excluded from performance evidence. Repeated timing therefore uses the same corrected executable with MTP overlap disabled/enabled, fixing logical placement on both sides. This is an explicit experimental-control correction, not permission to accept mismatching streams. The frozen executable and published-stock suite remain independent correctness references; no speedup is inferred from the rejected frozen pair.

Three alternating paired trials now pass exact IDs and finish reasons (8,192 tokens per pair):

| Pair | Control decode TPS | Overlap decode TPS | Decode gain | Wall gain |
| --- | ---: | ---: | ---: | ---: |
| 1, off then on | 224.60 | 237.34 | 5.67% | 5.19% |
| 2, on then off | 224.14 | 237.27 | 5.86% | 5.29% |
| 3, off then on | 223.35 | 237.01 | 6.12% | 5.48% |

Median paired improvement: **5.86% decode, 5.29% wall**. These are aggregate rates, not per-request rates. All request-level draft acceptance/offered counts and active-batch counts match in the paired essay runs. The first corrected pair's draft time falls from approximately 3.6 s to 1.6 s; target computation remains the dominant cost.

The corrected build passes the 18-case stock suite at c=1,2,3,4: 72 comparisons and 9,636 output tokens. An additional adaptive mixed-topic comparison passes all 18 responses and 2,333 tokens. That workload gains 1.71% in complete wall throughput, consistent with its substantial prefill and partially filled batches. The all-slot run repeats every stock fixture in every slot: another 72 comparisons and 9,636 tokens pass. Eight short-budget/near-context-boundary cases also pass, including zero, partial and full draft acceptance and unequal confidence thresholds. Their first four requests emit EOS near the limit; they do not establish forced context-exhaustion coverage.

### Long-form promotion failure

The normal c=4 confirmation run permits 8,192 output tokens per request; all four essays finish naturally earlier. Control output totals 10,034 tokens, candidate output 10,379. First mismatches occur at zero-based indices 1,884 / 1,787 / 1,806 / 1,908. All finish reasons are `stop`, but response lengths and IDs differ. The roughly 225.7 versus 239.4 aggregate decode TPS from these unequal outputs is **not valid speedup evidence**.

This fails the declared production-parity gate. The earlier c=2 trace demonstrates one timing/lookup mechanism, but this long-form divergence has not been independently attributed by a corresponding full trace. It is not labeled harmless. Overlapped MTP remains an experimental opt-in (`STRATA_BATCH_DRAFT=1`) for future investigation; the default is off and the user's preset does not enable it. Further 16k-context performance and forced context-exhaustion promotion tests were not run after this failure. No new arithmetic kernel is promoted.

### c=1 control limitations

The first normal c=1 baseline/candidate comparison requested the same automatic capacity (15,689 nominal slots), but the post-write reserve check shrank it to 16,300 versus 16,339 actual variable-size slots. That comparison is invalid for tokenwise performance attribution. A second comparison fixed the request to 15,000 slots (15,733 actual in both processes), but still differed and selected different suffix windows. Neither pair is counted as a passing performance comparison. c=1 stock parity under fixed numerical controls passes, and the overlap feature is not called by the c=1 engine.

Source inspection also confirms that the existing c=1 adaptive cache admits pending replacements through a nonblocking `cudaEventQuery` at a round boundary; unlike the concurrent coordinator, its residency can depend on copy completion timing. Its suffix policy uses measured round costs. These existing timing-sensitive controls must be distinguished from a changed mathematical kernel. Further controlled c=1 comparisons isolate these factors rather than waiving differing token IDs.

Disabling suffix lookup alone did not restore parity. The final c=1 regression comparison fixes requested cache capacity at 15,000, disables adaptation and PCIe misses, and disables lookup on both frozen baseline and candidate. All 4,096 output IDs and finish reasons match, with less than 1% throughput difference. This is a fixed-route regression test, not a claim that ordinary adaptive c=1 serving is repeatable or that its TPS is the same as static-cache TPS. The user's c=1 presets retain their original adaptive settings.

### c=2 lookup-policy investigation

The normal c=2 adaptive pair has matching actual capacity (13,892 slots) but differing outputs, so its throughput comparison is excluded. Traces agree through round 330. At round 331, request 100 uses a three-row MTP window in the control (`lookup=0`) and a four-row suffix window in the overlap run (`lookup=1`). The speculative inputs differ before target outputs differ. These extra speculative rows also contribute expert-usage counts, so future adaptive placement can diverge even when the different proposals are initially rejected.

The same executable with overlap **disabled** also differs between the ordinary and traced run (first differences at output indices 711 and 719 in the first two requests). This establishes that the control itself is sensitive to timing changes; it does not establish universal harmlessness. The lookup policy explicitly uses measured round costs. Further A/B testing disables lookup on both sides to isolate the changed draft path while retaining adaptive CPU/GPU dispatch. The production lookup configuration is not silently altered, and no c=2 default speedup is claimed from mismatching runs.

That isolated adaptive comparison passes all 4,096 IDs and finish reasons with the normal fast kernels enabled. Thus the changed draft path is exact under matching inputs/routing controls; default timing-driven lookup selection remains outside an unconditional cross-run token guarantee. This limitation already exists with overlap disabled and is retained explicitly in the correctness contract.

At c=3, the three concurrent requests match in normal settings; the fourth, subsequently run alone, differs at output index 811. That request executes the original single-drafter fallback. The complete c=3 comparison with lookup disabled passes all 4,096 IDs and finish reasons while retaining adaptation and fast kernels. The normal c=3 timing pair is excluded. c=2/c=3 controlled figures are labeled as lookup-disabled diagnostics, not advertised as default-setting gains.

## H2, H4, H5 and H6 prioritization

- H2's isolated all-resident shortcut is deprioritized: only 12.4% of adaptive layer visits are all-hit, and total plan wait across all visits is 1.1 s. This is a small bounded opportunity relative to 32 s target time. A mixed-residency planner would be a different, larger experiment rather than the proposed isolated fast path.
- H4 was evaluated by component profiling below. Broad packed projections were already tried in the preceding campaign and did not add enough benefit after GPU copy fixes. The 15.1 s pre-expert stage includes recurrent/attention work, normalization and shared-expert work; it is not all dense matrix multiplication. Another broad rewrite is deferred.
- H5 is a churn/TTFT opportunity, not a strong steady-throughput candidate: seven captures cost 0.429 s in the instrumented two-wave run. It is lower priority than drafting; expanding graph storage is not justified.
- H6 remains a separate TTFT track. It needs complete Flash Next continuation checkpoints and memory accounting; no KV-only shortcut will be implemented. The plan explicitly permits deferring it to preserve the throughput focus.

## Decisions

- H0: retain opt-in measurements and stronger token/finish-reason comparison tooling.
- H1: reject the paired expert prototype; no demonstrated end-to-end gain.
- H2: defer the isolated resident-plan shortcut on the measured opportunity bound.
- H3: preserve the experimental implementation, disabled; it fails long-form production parity.
- H4: complete component profiling; defer another packing rewrite without a stronger target.
- H5: defer graph indirection; measured steady-state capture cost is small.
- H6: defer full continuation-prefix checkpoints as the plan permits; hot-chat TTFT is not solved here.
- Retain the independently useful deterministic logical-slot packing fix. This removes heap-address-dependent ordering; it does not remove existing timing-dependent lookup/adaptation behavior.

The skipped broader promotion tests are intentionally distinguished from passed tests; the candidate is not promoted by lowering the parity gate.

### H4 component measurement

The subsequent fixed-placement diagnostic (`h4-component-profile-c4.json`, binary `3a73c7b4665c1785754702cbe4be3311a8a3e78132595e64bf17da17ac4fc1c5`) matches the freshly repeated frozen baseline in all 4,096 IDs and finish reasons. It copies existing per-member GPU stamps only while profiling is enabled. It reports one member per batch, not the sum of overlapping streams. These values include stream contention and must not be treated as isolated kernel microbenchmarks.

Across that diagnostic, the coordinator's pre-expert envelope is 7.923 s. The sampled member's stages include 1.409 s in the first hyperconnection read, 1.503 s in the second read plus routing, 0.916 s in GDN QKV projection/quantization, 0.570 s in GDN z projection, 0.351 s in GDN a/b, 0.321 s in recurrence, 0.511 s in QSA q/query-index work, 0.279 s in QSA attention, 0.620 s in output projections and 0.858 s in shared-expert/quantization work. The head envelope is 0.815 s. The remaining time is distributed across convolution, normalization, KV/index handling and synchronization.

This does not identify one large GEMM whose packing would explain Ninfer-like scaling. Hyperconnection work is the larger remaining family, but it includes several dependent transforms; it is not a single projection replacement. Together with the preceding campaign's negative broad-packing experiment, this does not justify repeating that prototype. A future fused/packed hyperconnection design needs its own arithmetic and state-isolation work. No H4 kernel change has been qualified or promoted.

## Safe-default integration

The final executable SHA-256 is `15eede15ad429be60638dd9dbd358c8bd2a3367c6a19a5dad2229dab7462aea1`. Overlap is off unless explicitly enabled. With it off, the final c=4 configuration matches all 18 published-stock responses and finish reasons under the established numerical controls (2,409 tokens). All 19 lifecycle checks pass: queued/prefill/decode cancellation, unaffected-request parity, twelve slot-reuse comparisons, invalid-request rejection and recovery.

The retained profiling instrumentation also passes c=1 on/off parity for 4,096 tokens under fixed routing; c=4 profiling parity was verified earlier. Detailed profiling is disabled in serving. Python verification passes 16 concurrency/comparison/router tests and 63 upstream tests, with three upstream skips. No arithmetic kernel change is enabled.

The preset router is restored on port 8080 with `swift-1.5-iq2_xs-c4` loaded and idle. A real HTTP smoke test observes four simultaneous native requests. All four return HTTP 200, 256 completion tokens, valid usage totals, `length` finish reasons and SSE `[DONE]`, then return to zero active requests. Short-prompt TTFT in this smoke test ranges from 0.21 to 1.07 s after a separate warmup; this is not a long-chat/prefix-reuse benchmark.

The c=1, c=1-vision and c=4 preset identities and context settings are preserved. c=4 remains vision-off, 98,304 tokens per request, 2,560 MiB reserve, sixteen batch rows and a 1,024-token prefill chunk. Neither GPU voltage nor clock settings were changed. Across two-second whole-machine qualification samples, observed GPU memory peaks at 29,999 MiB, available system RAM bottoms at 4,849 MiB, and GPU temperature peaks at 61 C. These samples include foreground applications and are not exact allocation peaks.

[Machine-readable results](concurrency-optimization-results.json) include per-pair rates, TTFT, per-request rates, maximum observed inter-token gaps, acceptance/active-batch counters, parity decisions, sampled resources, executable hashes and GGUF SHA-256 hashes. Cold model loading is excluded from throughput measurements and was not separately benchmarked. All raw reports and the frozen executable are preserved locally in `exports/strata-ab/`.
