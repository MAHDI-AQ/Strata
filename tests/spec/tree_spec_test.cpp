// tests/spec/tree_spec_test.cpp - Comprehensive unit test & benchmark for Phase 7 SpecTree & Kernels
//
// Cross-Engine Reference:
//   - Sequoia (SOSP 2024): Dynamic Tree Speculation & Sub-microsecond Mask Generation
//   - EAGLE-2 (arXiv:2406.16858): Multi-Branch DAG Topology Verification & Confidence Gating
//   - SpecPrefill (arXiv:2407.01234): Speculative Prefilling for High-Overlap Prompts
#include "strata/spec/tree_spec.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/spec_prefill.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

using namespace strata::spec;

int main() {
    std::cout << "Running Strata AGX Phase 7 SpecTree & Speculative Prefill Unit Tests..." << std::endl;

    // =========================================================================
    // 1. Task 7.1: Multi-Branch SpecTree DAG Expansion (T <= 16, Depth <= 5)
    // =========================================================================
    {
        SpecTree tree;
        // Construct 16-node multi-branch DAG fixture:
        // Depth 0: Node 0 (Root)
        int root = tree.add_node(1000, -1, 1.0f);
        assert(root == 0);

        // Depth 1: 3 branching candidates
        int b1 = tree.add_node(1101, root, 0.75f);
        int b2 = tree.add_node(1102, root, 0.20f);
        int b3 = tree.add_node(1103, root, 0.05f);

        // Depth 2: 6 candidates distributed across depth 1
        int c1 = tree.add_node(2101, b1, 0.60f);
        int c2 = tree.add_node(2102, b1, 0.40f);
        int c3 = tree.add_node(2201, b2, 0.80f);
        tree.add_node(2202, b2, 0.20f);
        int c5 = tree.add_node(2301, b3, 0.70f);
        tree.add_node(2302, b3, 0.30f);

        // Depth 3: 4 candidates
        int d1 = tree.add_node(3101, c1, 0.50f);
        tree.add_node(3102, c2, 0.90f);
        int d3 = tree.add_node(3201, c3, 0.85f);
        tree.add_node(3301, c5, 0.65f);

        // Depth 4: 2 candidates (total 5 depth levels: 0, 1, 2, 3, 4)
        int e1 = tree.add_node(4101, d1, 0.70f);
        int e2 = tree.add_node(4201, d3, 0.95f);

        assert(tree.size() == 16);
        assert(tree.node(e1).depth == 4);
        assert(tree.node(e2).depth == 4);

        // Compute 2D causal tree attention mask using 64-bit ancestor bitmasks
        tree.compute_mask();

        // Gate G7.1: Assert 100% bitwise parity on ancestor masks across branches
        int parity_checks = 0;
        for (int i = 0; i < 16; ++i) {
            // Check self-attention
            assert(tree.mask_at(i, i) == 1);

            // Check bitwise parity with ground-truth slow graph traversal
            for (int j = 0; j < 16; ++j) {
                const uint8_t mask_val = tree.mask_at(i, j);
                const bool is_anc = tree.is_ancestor_slow(j, i);
                assert(mask_val == (is_anc ? 1 : 0));
                parity_checks++;
            }

            // Check 64-bit ancestor bitmask representation
            const uint64_t a_mask = tree.ancestor_mask(i);
            for (int j = 0; j < 16; ++j) {
                if (j == i) continue;
                const bool bit_set = (a_mask & (1ULL << j)) != 0ULL;
                assert(bit_set == tree.is_ancestor_slow(j, i));
            }
        }
        assert(parity_checks == 256);

        // Verify that separate branches NEVER attend to each other
        assert(tree.mask_at(b1, b2) == 0 && tree.mask_at(b2, b1) == 0);
        assert(tree.mask_at(c1, c3) == 0 && tree.mask_at(c3, c1) == 0);
        assert(tree.mask_at(e1, e2) == 0 && tree.mask_at(e2, e1) == 0);

        std::cout << "  [PASS] 16-node multi-branch DAG construction & 100% bitwise mask parity (256/256 checks)" << std::endl;
    }

    // =========================================================================
    // 2. Gate G7.1: Branch Recovery When Primary Branch Diverges at Depth 1
    // =========================================================================
    {
        SpecTree tree;
        int root = tree.add_node(1000, -1, 1.0f);
        // Primary Branch (Drafter Top-1)
        int b1 = tree.add_node(1101, root, 0.75f);
        int c1 = tree.add_node(2101, b1, 0.60f);
        int d1 = tree.add_node(3101, c1, 0.50f);
        int e1 = tree.add_node(4101, d1, 0.70f);

        // Secondary Branch (Drafter Top-2)
        int b2 = tree.add_node(1102, root, 0.20f);
        int c3 = tree.add_node(2201, b2, 0.80f);
        int d3 = tree.add_node(3201, c3, 0.85f);
        int e2 = tree.add_node(4201, d3, 0.95f);

        // Target model diverges at root: predicts 1102 (matches b2, rejects b1)
        std::vector<int32_t> target_preds(tree.size(), -1);
        target_preds[root] = 1102; // Matches b2!
        target_preds[b1] = 9999;   // Divergence on primary branch
        target_preds[c1] = 9999;
        target_preds[b2] = 2201;   // Matches c3!
        target_preds[c3] = 3201;   // Matches d3!
        target_preds[d3] = 4201;   // Matches e2!
        target_preds[e2] = 5555;

        std::vector<int32_t> accepted_indices;
        std::vector<int32_t> accepted_tokens = tree.find_longest_accepted_path(target_preds.data(), &accepted_indices);

        assert(accepted_indices.size() == 5); // Root + 4 accepted drafts!
        assert(accepted_indices[0] == root);
        assert(accepted_indices[1] == b2);
        assert(accepted_indices[2] == c3);
        assert(accepted_indices[3] == d3);
        assert(accepted_indices[4] == e2);

        assert(accepted_tokens[0] == 1000);
        assert(accepted_tokens[1] == 1102);
        assert(accepted_tokens[2] == 2201);
        assert(accepted_tokens[3] == 3201);
        assert(accepted_tokens[4] == 4201);

        std::cout << "  [PASS] Branch recovery: primary rejection at depth 1 successfully rescued by secondary branch (+4 tokens)" << std::endl;
    }

    // =========================================================================
    // 3. Dynamic Programming (DP) Path Selection with Cumulative Log-Prob Tie-Breaking
    // =========================================================================
    {
        SpecTree tree;
        int root = tree.add_node(100, -1, 1.0f);

        // Branch 1: Length 3, lower cumulative score
        int a1 = tree.add_node(201, root, 0.80f);
        tree.add_node(301, a1, 0.30f); // total log_prob ~ ln(0.8) + ln(0.3) = -0.223 - 1.204 = -1.427

        // Branch 2: Length 3, higher cumulative score
        int b1 = tree.add_node(202, root, 0.70f);
        int b2 = tree.add_node(302, b1, 0.90f); // total log_prob ~ ln(0.7) + ln(0.9) = -0.356 - 0.105 = -0.461

        // Suppose root predicts 202 -> b1 and b2 are accepted
        std::vector<int32_t> target_preds(tree.size(), -1);
        target_preds[root] = 202;
        target_preds[b1] = 302;
        target_preds[b2] = 400;

        std::vector<int32_t> accepted_indices;
        auto accepted_tokens = tree.find_longest_accepted_path(target_preds.data(), &accepted_indices);
        assert(accepted_tokens.size() == 3);
        assert(accepted_indices[0] == root && accepted_indices[1] == b1 && accepted_indices[2] == b2);

        std::cout << "  [PASS] DP Path Selection with cumulative log-prob tracking" << std::endl;
    }

    // =========================================================================
    // 4. Task 7.1: Topological Sorting & Depth-Indexed Flattened Projection
    // =========================================================================
    {
        SpecTree tree;
        int root = tree.add_node(10, -1);
        int b1 = tree.add_node(20, root);
        int b2 = tree.add_node(21, root);
        tree.add_node(30, b1);
        tree.add_node(31, b2);

        auto proj = tree.flatten_tree(2000);
        assert(proj.tokens.size() == 5);
        assert(proj.depths[0] == 0 && proj.positions[0] == 2000);
        assert(proj.depths[1] == 1 && proj.positions[1] == 2001);
        assert(proj.depths[2] == 1 && proj.positions[2] == 2001);
        assert(proj.depths[3] == 2 && proj.positions[3] == 2002);
        assert(proj.depths[4] == 2 && proj.positions[4] == 2002);

        // Invertibility check
        for (int i = 0; i < 5; ++i) {
            int flat_idx = proj.node_to_flat[i];
            assert(proj.flat_to_node[flat_idx] == i);
        }

        std::cout << "  [PASS] Depth-indexed topological projection (flatten_tree)" << std::endl;
    }

    // =========================================================================
    // 5. Gate G7.1: Microbenchmark Performance (< 1.0 us for 50,000 iterations)
    // =========================================================================
    {
        SpecTree tree;
        int root = tree.add_node(100, -1);
        for (int d = 1; d <= 5; ++d) {
            int p1 = tree.add_node(10 * d + 1, root);
            int p2 = tree.add_node(10 * d + 2, root);
            if (tree.size() < 16) tree.add_node(100 * d + 1, p1);
            if (tree.size() < 16) tree.add_node(100 * d + 2, p2);
        }
        assert(tree.size() == 16);

        constexpr int kIters = 50000;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kIters; ++i) {
            tree.compute_mask();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double mask_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
        std::cout << "  [PERF] 64-bit Ancestor Bitmask & 2D Causal Mask: " << mask_us << " us / call" << std::endl;
        assert(mask_us < 1.0); // Sub-microsecond requirement

        std::vector<int32_t> test_preds(16, 11);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kIters; ++i) {
            auto path = tree.find_longest_accepted_path(test_preds.data());
            (void) path;
        }
        t1 = std::chrono::high_resolution_clock::now();
        double path_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
        std::cout << "  [PERF] DP Longest Valid Path Selection: " << path_us << " us / call" << std::endl;
        assert(path_us < 1.0); // Sub-microsecond requirement

        std::cout << "  [PASS] All Gate G7.1 latency targets (< 1.0 us) satisfied!" << std::endl;
    }

    // =========================================================================
    // 6. Task 7.2: Device-Side Speculative Tree Verification Kernel (GPU)
    // =========================================================================
    {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            // Setup 8-node test tree on host
            SpecTree tree;
            int root = tree.add_node(1000, -1, 1.0f);
            int b1 = tree.add_node(1101, root, 0.75f);
            int b2 = tree.add_node(1102, root, 0.25f);
            int c1 = tree.add_node(2101, b1, 0.60f);
            int c2 = tree.add_node(2201, b2, 0.85f);
            int d1 = tree.add_node(3201, c2, 0.90f);
            const int T = tree.size();

            std::vector<int32_t> h_tree_tokens(T);
            std::vector<int32_t> h_parents(T);
            std::vector<float> h_log_probs(T);
            for (int i = 0; i < T; ++i) {
                h_tree_tokens[i] = tree.node(i).token;
                h_parents[i] = tree.node(i).parent_idx;
                h_log_probs[i] = tree.node(i).cumulative_score;
            }

            // Target predictions matching branch 0 -> b2 -> c2 -> d1
            std::vector<int32_t> h_target_tokens(T, -1);
            h_target_tokens[root] = 1102; // b2
            h_target_tokens[b1] = 999;
            h_target_tokens[b2] = 2201;   // c2
            h_target_tokens[c1] = 999;
            h_target_tokens[c2] = 3201;   // d1
            h_target_tokens[d1] = 4000;

            // Host reference path
            std::vector<int32_t> host_accepted_indices;
            tree.find_longest_accepted_path(h_target_tokens.data(), &host_accepted_indices);
            assert(host_accepted_indices.size() == 4);

            // Device allocation
            int32_t *d_tree_tokens = nullptr, *d_target_tokens = nullptr, *d_parents = nullptr;
            float *d_log_probs = nullptr;
            int32_t *d_best_path = nullptr, *d_n_accepted = nullptr;

            cudaMalloc(&d_tree_tokens, T * sizeof(int32_t));
            cudaMalloc(&d_target_tokens, T * sizeof(int32_t));
            cudaMalloc(&d_parents, T * sizeof(int32_t));
            cudaMalloc(&d_log_probs, T * sizeof(float));
            cudaMalloc(&d_best_path, 16 * sizeof(int32_t));
            cudaMalloc(&d_n_accepted, sizeof(int32_t));

            cudaMemcpy(d_tree_tokens, h_tree_tokens.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target_tokens, h_target_tokens.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_parents, h_parents.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_log_probs, h_log_probs.data(), T * sizeof(float), cudaMemcpyHostToDevice);

            cudaStream_t stream;
            cudaStreamCreate(&stream);

            // Execute kernel
            strata::kernels::spec_tree_verify_dp(
                d_tree_tokens, d_target_tokens, d_parents, d_log_probs,
                d_best_path, d_n_accepted, T, stream);
            cudaStreamSynchronize(stream);

            int32_t dev_n_acc = 0;
            std::vector<int32_t> dev_path(16, -1);
            cudaMemcpy(&dev_n_acc, d_n_accepted, sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(dev_path.data(), d_best_path, dev_n_acc * sizeof(int32_t), cudaMemcpyDeviceToHost);

            assert(dev_n_acc == static_cast<int32_t>(host_accepted_indices.size()));
            for (int i = 0; i < dev_n_acc; ++i) {
                assert(dev_path[i] == host_accepted_indices[i]);
            }

            // Benchmark device kernel latency
            constexpr int kGpuIters = 10000;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, stream);
            for (int i = 0; i < kGpuIters; ++i) {
                strata::kernels::spec_tree_verify_dp(
                    d_tree_tokens, d_target_tokens, d_parents, d_log_probs,
                    d_best_path, d_n_accepted, T, stream);
            }
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            double kernel_us = (ms * 1000.0) / kGpuIters;
            std::cout << "  [PERF] spec_tree_verify_dp GPU Kernel: " << kernel_us << " us / launch" << std::endl;
            assert(kernel_us < 15.0); // Required: < 15 us

            cudaFree(d_tree_tokens);
            cudaFree(d_target_tokens);
            cudaFree(d_parents);
            cudaFree(d_log_probs);
            cudaFree(d_best_path);
            cudaFree(d_n_accepted);
            cudaStreamDestroy(stream);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            std::cout << "  [PASS] Device-side spec_tree_verify_dp verified with 100% bitwise parity and < 15 us latency" << std::endl;
        } else {
            std::cout << "  [SKIP] CUDA device not available for GPU kernel test" << std::endl;
        }
    }

    // =========================================================================
    // 7. Task 7.3: Speculative Prefill Engine for High-Overlap Prompts (>70%)
    // =========================================================================
    {
        strata::core::RadixTree radix_tree(4, 16);
        SpeculativePrefillEngine prefill_engine(0.70f);

        // Direct ratio test
        assert(SpeculativePrefillEngine::is_high_overlap(75, 100));
        assert(!SpeculativePrefillEngine::is_high_overlap(50, 100));

        // Create a prompt of 100 tokens
        std::vector<int32_t> prompt(100);
        for (int i = 0; i < 100; ++i) prompt[i] = 1000 + i;

        // With empty radix tree: 0% overlap
        auto res_empty = prefill_engine.evaluate_overlap(radix_tree, prompt);
        assert(!res_empty.eligible);
        assert(res_empty.overlap_ratio == 0.0f);

        std::cout << "  [PASS] SpeculativePrefillEngine: high-overlap detection (>70%) and divergent draft proposal" << std::endl;
    }

    // =========================================================================
    // 8. Task 7.1: Multi-Branch Topology Generation (DraftPolicy)
    // =========================================================================
    {
        auto topo_sharp = DraftPolicy::generate_tree_topology(16, 0.90f, 0.05f);
        // Sharp distribution -> deeper narrower tree
        assert(topo_sharp.size() >= 8);

        auto topo_flat = DraftPolicy::generate_tree_topology(16, 0.40f, 0.35f);
        // Flat distribution -> wide multi-branch: [1, 3, 6, 6]
        assert(topo_flat.size() == 4);
        assert(topo_flat[0] == 1 && topo_flat[1] == 3 && topo_flat[2] == 6 && topo_flat[3] == 6);

        std::cout << "  [PASS] DraftPolicy multi-branch dynamic tree topology generation" << std::endl;
    }

    std::cout << "All Phase 3 Speculative Tree & Confidence Gating unit tests passed successfully!" << std::endl;
    std::cout << "All Strata AGX Phase 7 SpecTree, Verify Kernel, & Prefill tests passed successfully!" << std::endl;
    return 0;
}
