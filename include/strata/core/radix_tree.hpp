#pragma once

#include <cuda_runtime.h>
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/on_device.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

struct RadixQsaSlice {
    void* k_q = nullptr;
    void* v_q4 = nullptr;
    void* k_scale = nullptr;
    void* v_scale = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    size_t k_bytes = 0;
    size_t v_bytes = 0;
    size_t k_scale_bytes = 0;
    size_t v_scale_bytes = 0;

    void* idx_tail = nullptr;
    size_t idx_tail_bytes = 0;
    void* idx_dead = nullptr;
    size_t idx_dead_bytes = 0;
    void* idx_pooled = nullptr;
    size_t idx_pooled_bytes = 0;
    int32_t* idx_block_pos = nullptr;
};

struct RadixStageSnapshot {
    int device = -1;
    float* gdn_saved = nullptr;
    size_t gdn_bytes = 0;

    float* ple_saved = nullptr;
    size_t ple_bytes = 0;

    float* R_saved = nullptr;
    size_t R_bytes = 0;

    int32_t ple_prev_saved[2] = {-1, -1};
    int32_t ple_token_saved = -1;

    int64_t qsa_ord0 = 0;
    int64_t qsa_alloc = 0;
    std::vector<RadixQsaSlice> qsa_slices;

    void free_device();
};

struct RadixNode : public std::enable_shared_from_this<RadixNode> {
    int64_t id = 0;
    int64_t prefix_len = 0;
    std::vector<int32_t> edge_tokens;
    int ref_count = 0;
    std::chrono::steady_clock::time_point last_accessed;

    std::weak_ptr<RadixNode> parent;
    std::unordered_map<int32_t, std::shared_ptr<RadixNode>> children;

    std::vector<RadixStageSnapshot> stage_snapshots;

    bool has_snapshot() const {
        return !stage_snapshots.empty() && stage_snapshots[0].gdn_saved != nullptr;
    }

    size_t total_vram_bytes() const;
    void free_device();
    ~RadixNode();
};

struct RadixMatch {
    std::shared_ptr<RadixNode> node;
    int64_t matched_tokens = 0;
};

class RadixTree {
public:
    explicit RadixTree(size_t max_cached_snapshots = 8);
    ~RadixTree();

    RadixMatch match_prefix(const int32_t* tokens, size_t n) const;
    RadixMatch match_prefix(const int64_t* tokens, size_t n) const;
    RadixMatch match_prefix(const std::vector<int32_t>& tokens) const { return match_prefix(tokens.data(), tokens.size()); }
    RadixMatch match_prefix(const std::vector<int64_t>& tokens) const { return match_prefix(tokens.data(), tokens.size()); }

    std::shared_ptr<RadixNode> insert(
        const int32_t* tokens,
        size_t n,
        int64_t prefix_len,
        const std::vector<int>& stage_devices,
        const std::vector<const SessionState*>& states,
        const std::vector<const float*>& R_ptrs,
        const ModelGeometry& g,
        const std::vector<void*>& streams);

    std::shared_ptr<RadixNode> insert(
        const std::vector<int32_t>& tokens,
        int64_t prefix_len,
        const std::vector<int>& stage_devices,
        const std::vector<const SessionState*>& states,
        const std::vector<const float*>& R_ptrs,
        const ModelGeometry& g,
        const std::vector<void*>& streams) {
        return insert(tokens.data(), tokens.size(), prefix_len, stage_devices, states, R_ptrs, g, streams);
    }

    void acquire(const std::shared_ptr<RadixNode>& node);
    void release(const std::shared_ptr<RadixNode>& node);

    bool fork_to_session(
        const std::shared_ptr<RadixNode>& node,
        int64_t prefix_len,
        const std::vector<int>& stage_devices,
        std::vector<SessionState*>& child_states,
        const ModelGeometry& g,
        const std::vector<void*>& streams,
        std::string& err);

    size_t evict_lru(size_t max_snapshots, size_t min_free_vram_mib = 0);

    size_t cached_snapshot_count() const { return cached_snapshots_; }
    size_t total_nodes() const { return node_count_; }

private:
    std::shared_ptr<RadixNode> root_;
    size_t max_cached_snapshots_ = 8;
    size_t cached_snapshots_ = 0;
    size_t node_count_ = 0;
    int64_t next_node_id_ = 1;

    void collect_unreferenced_leaves(
        const std::shared_ptr<RadixNode>& curr,
        std::vector<std::shared_ptr<RadixNode>>& leaves);
};

}  // namespace strata::core
