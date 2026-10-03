#include "strata/core/radix_tree.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace strata::core;

int main() {
    std::cout << "Running RadixTree unit tests..." << std::endl;

    RadixTree tree(4); // Max 4 cached snapshots

    // 1. Initial match on empty tree
    std::vector<int32_t> prompt1 = {10, 20, 30, 40, 50};
    auto m1 = tree.match_prefix(prompt1);
    assert(m1.matched_tokens == 0);
    assert(m1.node == nullptr);
    std::cout << "  [PASS] Empty tree match" << std::endl;

    // 2. Tree property checks
    assert(tree.total_nodes() == 1); // root node
    assert(tree.cached_snapshot_count() == 0);
    std::cout << "  [PASS] Initial state check" << std::endl;

    // 3. Test insert & match without GPU (mock dummy states)
    // We create a realistic token sequence (e.g. 300 tokens)
    std::vector<int32_t> harness_prompt(500);
    for (size_t i = 0; i < harness_prompt.size(); ++i) harness_prompt[i] = (int32_t)(1000 + i);

    std::vector<int> stage_devs = {0};
    std::vector<const SessionState*> states;
    std::vector<const float*> R_ptrs;
    std::vector<void*> streams;
    ModelGeometry g{};

    // Note: insert requires states to populate device buffers if prefix_len >= 256.
    // If states are empty, insert returns nullptr gracefully.
    auto res_null = tree.insert(harness_prompt, 500, stage_devs, states, R_ptrs, g, streams);
    assert(res_null == nullptr);
    std::cout << "  [PASS] Graceful refusal on empty stage states" << std::endl;

    std::cout << "All RadixTree tests passed successfully!" << std::endl;
    return 0;
}
