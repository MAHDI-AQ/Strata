#include "strata/core/radix_tree.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace strata::core;

int main() {
    std::cout << "Running RadixTree unit tests..." << std::endl;

    RadixTree tree(4, 64); // Max 4 VRAM snapshots, max 64 Host-RAM snapshots

    // 1. Initial match on empty tree
    std::vector<int32_t> prompt1 = {10, 20, 30, 40, 50};
    auto m1 = tree.match_prefix(prompt1);
    assert(m1.matched_tokens == 0);
    assert(m1.node == nullptr);
    std::cout << "  [PASS] Empty tree match" << std::endl;

    // 2. Tree property checks
    assert(tree.total_nodes() == 1); // root node
    assert(tree.cached_snapshot_count() == 0);
    assert(tree.cached_host_snapshot_count() == 0);
    std::cout << "  [PASS] Initial state check" << std::endl;

    // 3. Test insert & match without GPU (mock dummy states)
    std::vector<int32_t> harness_prompt(500);
    for (size_t i = 0; i < harness_prompt.size(); ++i) harness_prompt[i] = (int32_t)(1000 + i);

    std::vector<int> stage_devs = {0};
    std::vector<const SessionState*> states;
    std::vector<const float*> R_ptrs;
    std::vector<void*> streams;
    ModelGeometry g{};

    auto res_null = tree.insert(harness_prompt, 500, stage_devs, states, R_ptrs, g, streams);
    assert(res_null == nullptr);
    std::cout << "  [PASS] Graceful refusal on empty stage states" << std::endl;

    // 4. Test HiCache L2 Host-RAM Parking and Lifecycle
    {
        auto node = std::make_shared<RadixNode>();
        node->id = 42;
        node->prefix_len = 1024;

        // Allocate mock device snapshot on GPU 0
        RadixStageSnapshot ss;
        ss.device = 0;
        ss.gdn_bytes = 1024 * sizeof(float);
        cudaMalloc(&ss.gdn_saved, ss.gdn_bytes);
        std::vector<float> h_mock(1024, 3.14159f);
        cudaMemcpy(ss.gdn_saved, h_mock.data(), ss.gdn_bytes, cudaMemcpyHostToDevice);

        ss.R_bytes = 512 * sizeof(float);
        cudaMalloc(&ss.R_saved, ss.R_bytes);
        cudaMemcpy(ss.R_saved, h_mock.data(), ss.R_bytes, cudaMemcpyHostToDevice);

        node->stage_snapshots.push_back(ss);
        assert(node->has_device_snapshot());
        assert(!node->has_host_snapshot());
        assert(node->has_snapshot());
        assert(node->total_vram_bytes() == ss.gdn_bytes + ss.R_bytes);

        // Park node to Host RAM (HiCache L2 Tier)
        node->park_to_host();
        assert(!node->has_device_snapshot());
        assert(node->has_host_snapshot());
        assert(node->has_snapshot());
        assert(node->total_vram_bytes() == 0); // VRAM completely freed!
        assert(node->total_host_bytes() == ss.gdn_bytes + ss.R_bytes);

        // Verify data fidelity in host snapshot
        const float* h_gdn = (const float*) node->stage_host_snapshots[0].gdn_data.data();
        assert(h_gdn[0] == 3.14159f);
        assert(h_gdn[1023] == 3.14159f);
        (void) h_gdn;

        node->free_host();
        assert(!node->has_snapshot());
        assert(node->total_host_bytes() == 0);
        std::cout << "  [PASS] HiCache L2 Host-RAM parking, data fidelity, and VRAM release" << std::endl;
    }

    std::cout << "All RadixTree tests passed successfully!" << std::endl;
    return 0;
}
