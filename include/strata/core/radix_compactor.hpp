// include/strata/core/radix_compactor.hpp - Dynamic KV Fragmentation Compaction
//
// Cross-Engine Attribution and Research Lineage:
//   - SGLang: Prefix-cache memory management & dynamic RadixAttention eviction (Zheng et al., LMSYS / UC Berkeley)
//   - vLLM: Virtual block defragmentation & page compaction (Kwon et al., SOSP 2023)
//   - Orca: Iteration-level memory recovery (OSDI 2022)
//
// Background page compaction and memory defragmentation coordinator.
// Reclaims dead orphaned leaf nodes, merges single-child linear chains (path compression),
// and eliminates virtual memory fragmentation across long-horizon multi-turn sessions.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace strata::core {

class RadixTree;
struct RadixNode;

struct CompactionStats {
    size_t nodes_scanned = 0;
    size_t dead_leaves_pruned = 0;
    size_t chains_merged = 0;
    size_t bytes_reclaimed = 0;
    double compaction_time_us = 0.0;
    double initial_fragmentation = 0.0;
    double final_fragmentation = 0.0;
};

/// Coordinator for dynamic background memory defragmentation and page compaction
class RadixCompactor {
public:
    explicit RadixCompactor(RadixTree* tree = nullptr)
        : tree_(tree) {}

    void set_tree(RadixTree* tree) {
        tree_ = tree;
    }

    RadixTree* tree() const { return tree_; }

    /// Calculates fragmentation ratio in [0.0, 1.0]
    /// Frag = 1.0 - (useful_tokens / allocated_capacity)
    static double compute_fragmentation(size_t active_nodes, size_t total_nodes,
                                       size_t live_bytes, size_t allocated_bytes) {
        if (total_nodes == 0 || allocated_bytes == 0) return 0.0;
        const double node_factor = (double)active_nodes / (double)total_nodes;
        const double mem_factor = (double)live_bytes / (double)allocated_bytes;
        const double utilization = 0.5 * node_factor + 0.5 * mem_factor;
        return std::clamp(1.0 - utilization, 0.0, 1.0);
    }

    /// Perform dynamic compaction pass on the provided RadixTree (or internal tree_)
    CompactionStats compact(RadixTree* tree = nullptr);

    uint64_t total_compactions() const { return total_compactions_.load(); }
    uint64_t total_leaves_pruned() const { return total_leaves_pruned_.load(); }
    uint64_t total_chains_merged() const { return total_chains_merged_.load(); }
    uint64_t total_bytes_reclaimed() const { return total_bytes_reclaimed_.load(); }

private:
    RadixTree* tree_ = nullptr;
    mutable std::mutex mutex_;
    std::atomic<uint64_t> total_compactions_{0};
    std::atomic<uint64_t> total_leaves_pruned_{0};
    std::atomic<uint64_t> total_chains_merged_{0};
    std::atomic<uint64_t> total_bytes_reclaimed_{0};
};

}  // namespace strata::core
