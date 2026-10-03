#include "strata/core/radix_tree.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace strata::core {

void RadixStageSnapshot::free_device() {
    if (device >= 0) {
        const core::OnDevice on(device);
        if (gdn_saved) { cudaFree(gdn_saved); gdn_saved = nullptr; }
        if (ple_saved) { cudaFree(ple_saved); ple_saved = nullptr; }
        if (R_saved) { cudaFree(R_saved); R_saved = nullptr; }
        for (auto& s : qsa_slices) {
            if (s.k_q) { cudaFree(s.k_q); s.k_q = nullptr; }
            if (s.v_q4) { cudaFree(s.v_q4); s.v_q4 = nullptr; }
            if (s.k_scale) { cudaFree(s.k_scale); s.k_scale = nullptr; }
            if (s.v_scale) { cudaFree(s.v_scale); s.v_scale = nullptr; }
            if (s.k_pool) { cudaFree(s.k_pool); s.k_pool = nullptr; }
            if (s.v_pool) { cudaFree(s.v_pool); s.v_pool = nullptr; }
            if (s.idx_tail) { cudaFree(s.idx_tail); s.idx_tail = nullptr; }
            if (s.idx_dead) { cudaFree(s.idx_dead); s.idx_dead = nullptr; }
            if (s.idx_pooled) { cudaFree(s.idx_pooled); s.idx_pooled = nullptr; }
            if (s.idx_block_pos) { cudaFree(s.idx_block_pos); s.idx_block_pos = nullptr; }
        }
        qsa_slices.clear();
    }
    gdn_bytes = ple_bytes = R_bytes = 0;
}

size_t RadixNode::total_vram_bytes() const {
    size_t total = 0;
    for (const auto& ss : stage_snapshots) {
        total += ss.gdn_bytes + ss.ple_bytes + ss.R_bytes;
        for (const auto& qs : ss.qsa_slices) {
            total += qs.k_bytes + qs.v_bytes + qs.k_scale_bytes + qs.v_scale_bytes;
            total += qs.idx_tail_bytes + qs.idx_dead_bytes + qs.idx_pooled_bytes;
            if (qs.idx_block_pos) total += sizeof(int32_t);
        }
    }
    return total;
}

void RadixNode::free_device() {
    for (auto& ss : stage_snapshots) {
        ss.free_device();
    }
    stage_snapshots.clear();
}

RadixNode::~RadixNode() {
    free_device();
}

RadixTree::RadixTree(size_t max_cached_snapshots)
    : max_cached_snapshots_(max_cached_snapshots) {
    root_ = std::make_shared<RadixNode>();
    root_->id = 0;
    root_->prefix_len = 0;
    root_->last_accessed = std::chrono::steady_clock::now();
    node_count_ = 1;
}

RadixTree::~RadixTree() = default;

RadixMatch RadixTree::match_prefix(const int32_t* tokens, size_t n) const {
    RadixMatch best{nullptr, 0};
    std::shared_ptr<RadixNode> curr = root_;
    size_t matched_len = 0;

    while (curr && matched_len < n) {
        const int32_t next_tok = tokens[matched_len];
        auto it = curr->children.find(next_tok);
        if (it == curr->children.end()) break;

        auto child = it->second;
        const auto& edge = child->edge_tokens;
        size_t match_edge = 0;
        while (match_edge < edge.size() &&
               matched_len + match_edge < n &&
               edge[match_edge] == tokens[matched_len + match_edge]) {
            ++match_edge;
        }

        if (match_edge == edge.size()) {
            matched_len += edge.size();
            curr = child;
            if (curr->has_snapshot()) {
                best = RadixMatch{curr, (int64_t) matched_len};
            }
        } else {
            break;
        }
    }
    return best;
}

RadixMatch RadixTree::match_prefix(const int64_t* tokens, size_t n) const {
    RadixMatch best{nullptr, 0};
    std::shared_ptr<RadixNode> curr = root_;
    size_t matched_len = 0;

    while (curr && matched_len < n) {
        const int32_t next_tok = (int32_t) tokens[matched_len];
        auto it = curr->children.find(next_tok);
        if (it == curr->children.end()) break;

        auto child = it->second;
        const auto& edge = child->edge_tokens;
        size_t match_edge = 0;
        while (match_edge < edge.size() &&
               matched_len + match_edge < n &&
               (int64_t) edge[match_edge] == tokens[matched_len + match_edge]) {
            ++match_edge;
        }

        if (match_edge == edge.size()) {
            matched_len += edge.size();
            curr = child;
            if (curr->has_snapshot()) {
                best = RadixMatch{curr, (int64_t) matched_len};
            }
        } else {
            break;
        }
    }
    return best;
}

std::shared_ptr<RadixNode> RadixTree::insert(
    const int32_t* tokens,
    size_t n,
    int64_t prefix_len,
    const std::vector<int>& stage_devices,
    const std::vector<const SessionState*>& states,
    const std::vector<const float*>& R_ptrs,
    const ModelGeometry& g,
    const std::vector<void*>& streams) {

    if (prefix_len < 256 || (size_t) prefix_len > n ||
        states.size() != stage_devices.size() || streams.size() != stage_devices.size()) {
        return nullptr;
    }

    std::shared_ptr<RadixNode> curr = root_;
    size_t matched_len = 0;

    while (matched_len < (size_t) prefix_len) {
        const int32_t next_tok = tokens[matched_len];
        auto it = curr->children.find(next_tok);
        if (it == curr->children.end()) {
            auto new_node = std::make_shared<RadixNode>();
            new_node->id = next_node_id_++;
            new_node->prefix_len = (int64_t) prefix_len;
            new_node->edge_tokens.assign(tokens + matched_len, tokens + prefix_len);
            new_node->parent = curr;
            new_node->last_accessed = std::chrono::steady_clock::now();
            curr->children[next_tok] = new_node;
            ++node_count_;
            curr = new_node;
            matched_len = (size_t) prefix_len;
            break;
        }

        auto child = it->second;
        const auto& edge = child->edge_tokens;
        size_t match_edge = 0;
        const size_t max_comp = std::min(edge.size(), (size_t) prefix_len - matched_len);
        while (match_edge < max_comp && edge[match_edge] == tokens[matched_len + match_edge]) {
            ++match_edge;
        }

        if (match_edge == edge.size()) {
            matched_len += edge.size();
            curr = child;
        } else {
            auto split_node = std::make_shared<RadixNode>();
            split_node->id = next_node_id_++;
            split_node->prefix_len = curr->prefix_len + (int64_t) match_edge;
            split_node->edge_tokens.assign(edge.begin(), edge.begin() + match_edge);
            split_node->parent = curr;
            split_node->last_accessed = std::chrono::steady_clock::now();

            child->edge_tokens.erase(child->edge_tokens.begin(), child->edge_tokens.begin() + match_edge);
            child->parent = split_node;
            split_node->children[child->edge_tokens[0]] = child;

            curr->children[next_tok] = split_node;
            ++node_count_;

            matched_len += match_edge;

            if (matched_len == (size_t) prefix_len) {
                curr = split_node;
            } else {
                auto new_node = std::make_shared<RadixNode>();
                new_node->id = next_node_id_++;
                new_node->prefix_len = (int64_t) prefix_len;
                new_node->edge_tokens.assign(tokens + matched_len, tokens + prefix_len);
                new_node->parent = split_node;
                new_node->last_accessed = std::chrono::steady_clock::now();
                split_node->children[tokens[matched_len]] = new_node;
                ++node_count_;
                curr = new_node;
                matched_len = (size_t) prefix_len;
            }
            break;
        }
    }

    curr->last_accessed = std::chrono::steady_clock::now();

    if (!curr->has_snapshot()) {
        const size_t n_stages = stage_devices.size();
        curr->stage_snapshots.resize(n_stages);

        strata::kernels::QsaShapes qs = strata::kernels::qsa_real_shapes();
        qs.n_head = g.n_head;
        qs.n_head_kv = g.n_head_kv;
        qs.head_dim = g.head_dim;
        qs.idx_n_head = g.idx_q_heads;
        qs.idx_dim = g.idx_key_dim;

        const int64_t pages = (prefix_len + qs.page_size - 1) / qs.page_size;
        const size_t qsa_rows = (size_t) pages * qs.n_head_kv * qs.page_size;

        for (size_t st = 0; st < n_stages; ++st) {
            auto& ss = curr->stage_snapshots[st];
            ss.device = stage_devices[st];
            const core::OnDevice on(ss.device);
            cudaStream_t cs = (cudaStream_t) streams[st];
            const auto* s = states[st];

            if (s->gdn_alloc > 0 && s->gdn_state) {
                ss.gdn_bytes = (size_t) s->gdn_alloc * core::gdn_state_floats(g) * sizeof(float);
                cudaMalloc(&ss.gdn_saved, ss.gdn_bytes);
                cudaMemcpyAsync(ss.gdn_saved, s->gdn_state, ss.gdn_bytes, cudaMemcpyDeviceToDevice, cs);
            }

            if (s->ple_hist) {
                ss.ple_bytes = (size_t) core::ple_hist_bytes();
                cudaMalloc(&ss.ple_saved, ss.ple_bytes);
                cudaMemcpyAsync(ss.ple_saved, s->ple_hist, ss.ple_bytes, cudaMemcpyDeviceToDevice, cs);
                ss.ple_prev_saved[0] = s->ple_prev[0];
                ss.ple_prev_saved[1] = s->ple_prev[1];
                ss.ple_token_saved = s->ple_token;
            }

            const float* src_R = R_ptrs[st] ? R_ptrs[st] : s->block.R;
            if (src_R) {
                ss.R_bytes = (size_t) g.hc * g.n_embd * sizeof(float);
                cudaMalloc(&ss.R_saved, ss.R_bytes);
                cudaMemcpyAsync(ss.R_saved, src_R, ss.R_bytes, cudaMemcpyDeviceToDevice, cs);
            }

            ss.qsa_ord0 = s->qsa_ord0;
            ss.qsa_alloc = s->qsa_alloc;
            ss.qsa_slices.resize(s->qsa_alloc);

            for (int64_t j = 0; j < s->qsa_alloc; ++j) {
                const QsaState& qst = s->qsa_states[s->qsa_ord0 + j];
                auto& slice = ss.qsa_slices[j];

                if (qst.kv_q4) {
                    slice.k_bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    slice.v_bytes = slice.k_bytes;
                    if (qst.k_q4) {
                        cudaMalloc(&slice.k_q, slice.k_bytes);
                        cudaMemcpyAsync(slice.k_q, qst.k_q4, slice.k_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.v_q4) {
                        cudaMalloc(&slice.v_q4, slice.v_bytes);
                        cudaMemcpyAsync(slice.v_q4, qst.v_q4, slice.v_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                } else if (qst.kv_hybrid) {
                    slice.k_bytes = qsa_rows * qs.head_dim;
                    slice.k_scale_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    slice.v_bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    if (qst.k_q) {
                        cudaMalloc(&slice.k_q, slice.k_bytes);
                        cudaMemcpyAsync(slice.k_q, qst.k_q, slice.k_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.k_scale) {
                        cudaMalloc(&slice.k_scale, slice.k_scale_bytes);
                        cudaMemcpyAsync(slice.k_scale, qst.k_scale, slice.k_scale_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.v_q4) {
                        cudaMalloc(&slice.v_q4, slice.v_bytes);
                        cudaMemcpyAsync(slice.v_q4, qst.v_q4, slice.v_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                } else if (qst.kv_int8) {
                    slice.k_bytes = qsa_rows * qs.head_dim;
                    slice.k_scale_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    if (qst.k_q) {
                        cudaMalloc(&slice.k_q, slice.k_bytes);
                        cudaMemcpyAsync(slice.k_q, qst.k_q, slice.k_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.v_q) {
                        cudaMalloc(&slice.v_q4, slice.k_bytes);
                        cudaMemcpyAsync(slice.v_q4, qst.v_q, slice.k_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.k_scale) {
                        cudaMalloc(&slice.k_scale, slice.k_scale_bytes);
                        cudaMemcpyAsync(slice.k_scale, qst.k_scale, slice.k_scale_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.v_scale) {
                        cudaMalloc(&slice.v_scale, slice.k_scale_bytes);
                        cudaMemcpyAsync(slice.v_scale, qst.v_scale, slice.k_scale_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                }

                slice.idx_tail_bytes = (size_t) (qs.idx_block - 1) * qs.idx_dim * sizeof(float);
                slice.idx_dead_bytes = (size_t) qs.idx_dim * sizeof(float);
                if (qst.idx_tail) {
                    cudaMalloc(&slice.idx_tail, slice.idx_tail_bytes);
                    cudaMemcpyAsync(slice.idx_tail, qst.idx_tail, slice.idx_tail_bytes, cudaMemcpyDeviceToDevice, cs);
                }
                if (qst.idx_dead) {
                    cudaMalloc(&slice.idx_dead, slice.idx_dead_bytes);
                    cudaMemcpyAsync(slice.idx_dead, qst.idx_dead, slice.idx_dead_bytes, cudaMemcpyDeviceToDevice, cs);
                }
                if (qst.idx_block_pos) {
                    cudaMalloc(&slice.idx_block_pos, sizeof(int32_t));
                    cudaMemcpyAsync(slice.idx_block_pos, qst.idx_block_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice, cs);
                }
                const int64_t pooled_rows = prefix_len / qs.idx_block;
                if (pooled_rows > 0 && qst.idx_pooled) {
                    slice.idx_pooled_bytes = (size_t) (pooled_rows + 1) * qs.idx_dim * sizeof(float);
                    cudaMalloc(&slice.idx_pooled, slice.idx_pooled_bytes);
                    cudaMemcpyAsync(slice.idx_pooled, qst.idx_pooled, (size_t) pooled_rows * qs.idx_dim * sizeof(float), cudaMemcpyDeviceToDevice, cs);
                    cudaMemcpyAsync((char*) slice.idx_pooled + pooled_rows * qs.idx_dim * sizeof(float),
                                    qst.idx_dead, slice.idx_dead_bytes, cudaMemcpyDeviceToDevice, cs);
                }
            }
        }
        for (size_t st = 0; st < n_stages; ++st) {
            cudaStreamSynchronize((cudaStream_t) streams[st]);
        }
        ++cached_snapshots_;
        evict_lru(max_cached_snapshots_);
    }
    return curr;
}

void RadixTree::acquire(const std::shared_ptr<RadixNode>& node) {
    if (node) ++node->ref_count;
}

void RadixTree::release(const std::shared_ptr<RadixNode>& node) {
    if (node && node->ref_count > 0) --node->ref_count;
}

bool RadixTree::fork_to_session(
    const std::shared_ptr<RadixNode>& node,
    int64_t prefix_len,
    const std::vector<int>& stage_devices,
    std::vector<SessionState*>& child_states,
    const ModelGeometry& g,
    const std::vector<void*>& streams,
    std::string& err) {

    if (!node || !node->has_snapshot()) {
        err = "radix_fork: node has no device snapshot";
        return false;
    }

    const size_t n_stages = stage_devices.size();
    strata::kernels::QsaShapes qs = strata::kernels::qsa_real_shapes();
    qs.n_head = g.n_head;
    qs.n_head_kv = g.n_head_kv;
    qs.head_dim = g.head_dim;
    qs.idx_n_head = g.idx_q_heads;
    qs.idx_dim = g.idx_key_dim;

    const int64_t pages = (prefix_len + qs.page_size - 1) / qs.page_size;
    const size_t qsa_rows = (size_t) pages * qs.n_head_kv * qs.page_size;

    for (size_t st = 0; st < n_stages; ++st) {
        const auto& ss = node->stage_snapshots[st];
        auto* child = child_states[st];
        const core::OnDevice on(ss.device);
        cudaStream_t cs = (cudaStream_t) streams[st];

        if (ss.R_saved && child->block.R) {
            cudaMemcpyAsync(child->block.R, ss.R_saved, ss.R_bytes, cudaMemcpyDeviceToDevice, cs);
        }

        if (ss.gdn_saved && child->gdn_state) {
            cudaMemcpyAsync(child->gdn_state, ss.gdn_saved, ss.gdn_bytes, cudaMemcpyDeviceToDevice, cs);
        }

        if (ss.ple_saved && child->ple_hist) {
            cudaMemcpyAsync(child->ple_hist, ss.ple_saved, ss.ple_bytes, cudaMemcpyDeviceToDevice, cs);
            child->ple_prev[0] = ss.ple_prev_saved[0];
            child->ple_prev[1] = ss.ple_prev_saved[1];
            child->ple_token = ss.ple_token_saved;
        }

        for (int64_t j = 0; j < child->qsa_alloc && j < (int64_t) ss.qsa_slices.size(); ++j) {
            const auto& slice = ss.qsa_slices[j];
            QsaState& cst = child->qsa_states[child->qsa_ord0 + j];

            if (cst.kv_q4) {
                const size_t bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                if (slice.k_q && cst.k_q4) cudaMemcpyAsync(cst.k_q4, slice.k_q, bytes, cudaMemcpyDeviceToDevice, cs);
                if (slice.v_q4 && cst.v_q4) cudaMemcpyAsync(cst.v_q4, slice.v_q4, bytes, cudaMemcpyDeviceToDevice, cs);
            } else if (cst.kv_hybrid) {
                const size_t k_bytes = qsa_rows * qs.head_dim;
                const size_t sc_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                const size_t v_bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                if (slice.k_q && cst.k_q) cudaMemcpyAsync(cst.k_q, slice.k_q, k_bytes, cudaMemcpyDeviceToDevice, cs);
                if (slice.k_scale && cst.k_scale) cudaMemcpyAsync(cst.k_scale, slice.k_scale, sc_bytes, cudaMemcpyDeviceToDevice, cs);
                if (slice.v_q4 && cst.v_q4) cudaMemcpyAsync(cst.v_q4, slice.v_q4, v_bytes, cudaMemcpyDeviceToDevice, cs);
            } else if (cst.kv_int8) {
                const size_t bytes = qsa_rows * qs.head_dim;
                const size_t sc_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                if (slice.k_q && cst.k_q) cudaMemcpyAsync(cst.k_q, slice.k_q, bytes, cudaMemcpyDeviceToDevice, cs);
                if (slice.v_q4 && cst.v_q) cudaMemcpyAsync(cst.v_q, slice.v_q4, bytes, cudaMemcpyDeviceToDevice, cs);
                if (slice.k_scale && cst.k_scale) cudaMemcpyAsync(cst.k_scale, slice.k_scale, sc_bytes, cudaMemcpyDeviceToDevice, cs);
                if (slice.v_scale && cst.v_scale) cudaMemcpyAsync(cst.v_scale, slice.v_scale, sc_bytes, cudaMemcpyDeviceToDevice, cs);
            }

            if (slice.idx_tail && cst.idx_tail) {
                cudaMemcpyAsync(cst.idx_tail, slice.idx_tail, slice.idx_tail_bytes, cudaMemcpyDeviceToDevice, cs);
            }
            if (slice.idx_dead && cst.idx_dead) {
                cudaMemcpyAsync(cst.idx_dead, slice.idx_dead, slice.idx_dead_bytes, cudaMemcpyDeviceToDevice, cs);
            }
            if (slice.idx_block_pos && cst.idx_block_pos) {
                cudaMemcpyAsync(cst.idx_block_pos, slice.idx_block_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice, cs);
            }
            const int64_t pooled_rows = prefix_len / qs.idx_block;
            if (pooled_rows > 0 && slice.idx_pooled && cst.idx_pooled) {
                cudaMemcpyAsync(cst.idx_pooled, slice.idx_pooled, slice.idx_pooled_bytes, cudaMemcpyDeviceToDevice, cs);
            }
        }
    }
    for (size_t st = 0; st < n_stages; ++st) {
        const core::OnDevice on(stage_devices[st]);
        if (cudaStreamSynchronize((cudaStream_t) streams[st]) != cudaSuccess) {
            err = "radix_fork: cudaStreamSynchronize failed";
            return false;
        }
    }
    node->last_accessed = std::chrono::steady_clock::now();
    return true;
}

void RadixTree::collect_unreferenced_leaves(
    const std::shared_ptr<RadixNode>& curr,
    std::vector<std::shared_ptr<RadixNode>>& leaves) {
    if (!curr) return;
    if (curr != root_ && curr->ref_count == 0 && curr->children.empty()) {
        leaves.push_back(curr);
        return;
    }
    for (const auto& kv : curr->children) {
        collect_unreferenced_leaves(kv.second, leaves);
    }
}

size_t RadixTree::evict_lru(size_t max_snapshots, size_t min_free_vram_mib) {
    size_t evicted = 0;
    auto should_evict = [&]() -> bool {
        if (cached_snapshots_ > max_snapshots) return true;
        if (min_free_vram_mib > 0) {
            size_t free0 = 0, tot0 = 0, free1 = 0, tot1 = 0;
            { const core::OnDevice on(0); cudaMemGetInfo(&free0, &tot0); }
            { const core::OnDevice on(1); cudaMemGetInfo(&free1, &tot1); }
            const size_t reserve_bytes = min_free_vram_mib * 1048576ULL;
            if (free0 < reserve_bytes || free1 < reserve_bytes) return true;
        }
        return false;
    };

    while (cached_snapshots_ > 0 && should_evict()) {
        std::vector<std::shared_ptr<RadixNode>> leaves;
        collect_unreferenced_leaves(root_, leaves);
        if (leaves.empty()) break; // All leaves actively in use

        std::sort(leaves.begin(), leaves.end(), [](const auto& a, const auto& b) {
            return a->last_accessed < b->last_accessed;
        });

        auto victim = leaves.front();
        if (victim->has_snapshot()) {
            victim->free_device();
            if (cached_snapshots_ > 0) --cached_snapshots_;
            ++evicted;
        }

        if (auto p = victim->parent.lock()) {
            if (!victim->edge_tokens.empty()) {
                p->children.erase(victim->edge_tokens[0]);
                --node_count_;
            }
        }
    }
    return evicted;
}

}  // namespace strata::core
