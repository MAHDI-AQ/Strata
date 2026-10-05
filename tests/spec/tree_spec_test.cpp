// tests/spec/tree_spec_test.cpp - Comprehensive unit test & benchmark for Phase 3 SpecTree & ConfidenceGater
#include "strata/spec/tree_spec.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>

using namespace strata::spec;

int main() {
    std::cout << "Running Phase 3 Speculative Tree & Confidence Gating unit tests..." << std::endl;

    // 1. Test SpecTree construction and node hierarchy
    {
        SpecTree tree;
        assert(tree.empty());
        assert(tree.size() == 0);

        int root = tree.add_node(1001, -1, 1.0f);
        assert(root == 0);
        assert(tree.node(root).depth == 0);
        assert(tree.node(root).token == 1001);

        // Branch A (Top-1 candidate path)
        int a1 = tree.add_node(2001, root, 0.70f);
        int a2 = tree.add_node(3001, a1, 0.60f);

        // Branch B (Top-2 candidate path)
        int b1 = tree.add_node(2002, root, 0.28f);
        int b2 = tree.add_node(3002, b1, 0.85f);

        assert(tree.size() == 5);
        assert(tree.node(a1).depth == 1);
        assert(tree.node(a2).depth == 2);
        assert(tree.node(b1).depth == 1);
        assert(tree.node(b2).depth == 2);

        // Test ancestor logic
        assert(tree.is_ancestor(root, root));
        assert(tree.is_ancestor(root, a1));
        assert(tree.is_ancestor(root, a2));
        assert(tree.is_ancestor(a1, a2));
        assert(!tree.is_ancestor(a1, b1)); // Siblings cannot attend to each other!
        assert(!tree.is_ancestor(a2, b2)); // Distinct branches
        assert(tree.is_ancestor(root, b2));

        // Test 2D causal tree attention mask computation
        tree.compute_mask();
        assert(tree.mask_at(root, root) == 1);
        assert(tree.mask_at(a1, root) == 1);
        assert(tree.mask_at(a1, a1) == 1);
        assert(tree.mask_at(a1, b1) == 0); // Branch A does NOT attend to Branch B!
        assert(tree.mask_at(b1, a1) == 0); // Branch B does NOT attend to Branch A!
        assert(tree.mask_at(b2, root) == 1);
        assert(tree.mask_at(b2, b1) == 1);
        assert(tree.mask_at(b2, a2) == 0);

        std::cout << "  [PASS] SpecTree construction, branching, and 2D causal mask" << std::endl;
    }

    // 2. Test Longest Valid Path Selection: Branch Recovery over Linear Rejection
    {
        SpecTree tree;
        int root = tree.add_node(100, -1, 1.0f);
        // Branch A (e.g. drafter top-1 = "const", then "int")
        int a1 = tree.add_node(201, root, 0.55f);
        int a2 = tree.add_node(301, a1, 0.70f);

        // Branch B (e.g. drafter top-2 = "static", then "int")
        int b1 = tree.add_node(202, root, 0.45f);
        int b2 = tree.add_node(302, b1, 0.90f);

        // Scenario: Target model actually preferred "static" (202) over "const" (201).
        // Under linear speculative decoding, proposing Branch A alone results in 0 accepted drafts!
        // Target model outputs:
        // At root (100): target predicts 202 ("static")
        // At a1 (201): target predicts 999
        // At b1 (202): target predicts 302 ("int")
        // At b2 (302): target predicts 400
        std::vector<int32_t> target_preds(5);
        target_preds[root] = 202; // Matches b1!
        target_preds[a1] = 999;
        target_preds[a2] = 999;
        target_preds[b1] = 302;   // Matches b2!
        target_preds[b2] = 400;

        std::vector<int32_t> accepted_indices;
        std::vector<int32_t> accepted_tokens = tree.find_longest_accepted_path(target_preds.data(), &accepted_indices);

        assert(accepted_tokens.size() == 3); // Root, B1, B2 all accepted!
        assert(accepted_tokens[0] == 100);
        assert(accepted_tokens[1] == 202);
        assert(accepted_tokens[2] == 302);
        assert(accepted_indices[0] == root);
        assert(accepted_indices[1] == b1);
        assert(accepted_indices[2] == b2);

        std::cout << "  [PASS] Longest valid path recovery: branch B rescued (acceptance +200%)" << std::endl;
    }

    // 3. Test Flattening for GPU verification forward pass
    {
        SpecTree tree;
        int root = tree.add_node(10, -1);
        int a1 = tree.add_node(20, root);
        int a2 = tree.add_node(30, a1);
        int b1 = tree.add_node(21, root);

        std::vector<int32_t> flat_tokens;
        std::vector<int64_t> flat_positions;
        tree.flatten(flat_tokens, flat_positions, 1000);

        assert(flat_tokens.size() == 4);
        assert(flat_tokens[0] == 10 && flat_positions[0] == 1000);
        assert(flat_tokens[1] == 20 && flat_positions[1] == 1001);
        assert(flat_tokens[2] == 30 && flat_positions[2] == 1002);
        assert(flat_tokens[3] == 21 && flat_positions[3] == 1001); // b1 has depth 1, so pos 1001!

        std::cout << "  [PASS] Tree flattening and position indexing" << std::endl;
    }

    // 4. Test ConfidenceGater on sharp vs flat logits
    {
        ConfidenceGater gater(2.0f, 0.6f, 1.5f);

        // Case A: High confidence sharp distribution
        std::vector<float> sharp_logits(100, -10.0f);
        sharp_logits[42] = 8.5f;   // Top 1
        sharp_logits[7]  = 5.0f;   // Top 2 (Margin = 3.5, strong winner)

        ConfidenceMetrics cm_sharp = ConfidenceGater::evaluate_logits(sharp_logits.data(), 100, 10);
        assert(cm_sharp.logit_margin >= 3.4f);
        assert(cm_sharp.top1_prob > 0.90f);
        assert(cm_sharp.entropy < 0.6f);

        auto dec_sharp = gater.decide(cm_sharp, 4, 0.5f);
        assert(dec_sharp.recommended_t == 4);
        assert(dec_sharp.use_tree == true);

        // Case B: High entropy flat distribution (uncertain token)
        std::vector<float> flat_logits(100, 2.0f);
        flat_logits[10] = 2.1f;
        flat_logits[20] = 2.05f;

        ConfidenceMetrics cm_flat = ConfidenceGater::evaluate_logits(flat_logits.data(), 100, 10);
        assert(cm_flat.logit_margin < 0.2f);
        assert(cm_flat.entropy > 1.8f);

        auto dec_flat = gater.decide(cm_flat, 4, 0.5f);
        assert(dec_flat.recommended_t == 1);
        assert(dec_flat.use_tree == false);

        std::cout << "  [PASS] ConfidenceGater: sharp -> T=4 (tree), flat -> T=1 (throttled)" << std::endl;
    }

    // 5. Test Adaptive Online EMA calibration
    {
        ConfidenceGater gater(2.0f, 0.6f, 1.5f);
        float init_high = gater.high_margin();

        // Feed high acceptance rounds (>85%) -> should loosen thresholds
        for (int i = 0; i < 20; ++i) gater.observe(10, 9); // 90% acceptance
        assert(gater.high_margin() < init_high); // Threshold adapted downwards

        // Feed low acceptance rounds (<60%) -> should tighten thresholds
        float mid_high = gater.high_margin();
        for (int i = 0; i < 30; ++i) gater.observe(10, 3); // 30% acceptance
        assert(gater.high_margin() > mid_high); // Threshold tightened upwards

        std::cout << "  [PASS] Adaptive online EMA threshold calibration" << std::endl;
    }

    // 6. Performance Microbenchmark: Mask Computation & Path Selection Latency
    {
        SpecTree tree;
        int root = tree.add_node(1, -1);
        for (int d = 1; d <= 3; ++d) {
            tree.add_node(10 * d + 1, root);
            tree.add_node(10 * d + 2, root);
        }

        int iters = 50000;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            tree.compute_mask();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double mask_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        std::cout << "  [PERF] 2D Tree Mask computation: " << mask_us << " us / call" << std::endl;
        assert(mask_us < 1.0); // Sub-microsecond!

        std::vector<int32_t> test_preds(tree.size(), 11);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            auto path = tree.find_longest_accepted_path(test_preds.data());
            (void) path;
        }
        t1 = std::chrono::high_resolution_clock::now();
        double path_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        std::cout << "  [PERF] Longest Valid Path selection: " << path_us << " us / call" << std::endl;
        assert(path_us < 1.0); // Sub-microsecond!

        std::cout << "  [PASS] All microbenchmark latencies well below 1.0 us target!" << std::endl;
    }

    std::cout << "All Phase 3 Speculative Tree & Confidence Gating unit tests passed successfully!" << std::endl;
    return 0;
}
