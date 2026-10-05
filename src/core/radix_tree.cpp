// src/core/radix_tree.cpp - Lockless RadixTree Prefix Matching, Mutation & Pinned DMA Implementation
//
// Cross-Engine Attribution and Research Lineage:
//   - SGLang: RadixAttention tree prefix caching, hardware CRC32C chunking, and multi-tier memory management (arXiv:2312.07104)
//
#include "strata/core/radix_tree.hpp"
#include "strata/core/nvme_tier.hpp"
#include "strata/core/radix_compactor.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

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

size_t RadixNode::total_host_bytes() const {
    size_t total = 0;
    for (const auto& hss : stage_host_snapshots) {
        total += hss.gdn_data.size() + hss.ple_data.size() + hss.R_data.size();
        for (const auto& qs : hss.qsa_slices) {
            total += qs.k_q.size() + qs.v_q4.size() + qs.k_scale.size() + qs.v_scale.size();
            total += qs.idx_tail.size() + qs.idx_dead.size() + qs.idx_pooled.size();
            if (qs.has_idx_block_pos) total += sizeof(int32_t);
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

void RadixNode::free_host() {
    stage_host_snapshots.clear();
    is_host_parked = false;
}

void RadixNode::park_to_host() {
    if (stage_snapshots.empty()) return;
    stage_host_snapshots.resize(stage_snapshots.size());
    for (size_t st = 0; st < stage_snapshots.size(); ++st) {
        auto& ss = stage_snapshots[st];
        auto& hss = stage_host_snapshots[st];
        hss.device = ss.device;
        hss.ple_prev_saved[0] = ss.ple_prev_saved[0];
        hss.ple_prev_saved[1] = ss.ple_prev_saved[1];
        hss.ple_token_saved = ss.ple_token_saved;
        hss.qsa_ord0 = ss.qsa_ord0;
        hss.qsa_alloc = ss.qsa_alloc;

        if (ss.device >= 0) {
            const core::OnDevice on(ss.device);
            if (ss.gdn_saved && ss.gdn_bytes > 0) {
                hss.gdn_data.resize(ss.gdn_bytes);
                cudaMemcpy(hss.gdn_data.data(), ss.gdn_saved, ss.gdn_bytes, cudaMemcpyDeviceToHost);
            }
            if (ss.ple_saved && ss.ple_bytes > 0) {
                hss.ple_data.resize(ss.ple_bytes);
                cudaMemcpy(hss.ple_data.data(), ss.ple_saved, ss.ple_bytes, cudaMemcpyDeviceToHost);
            }
            if (ss.R_saved && ss.R_bytes > 0) {
                hss.R_data.resize(ss.R_bytes);
                cudaMemcpy(hss.R_data.data(), ss.R_saved, ss.R_bytes, cudaMemcpyDeviceToHost);
            }
            hss.qsa_slices.resize(ss.qsa_slices.size());
            for (size_t j = 0; j < ss.qsa_slices.size(); ++j) {
                const auto& src = ss.qsa_slices[j];
                auto& dst = hss.qsa_slices[j];
                if (src.k_q && src.k_bytes > 0) {
                    dst.k_q.resize(src.k_bytes);
                    cudaMemcpy(dst.k_q.data(), src.k_q, src.k_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.v_q4 && src.v_bytes > 0) {
                    dst.v_q4.resize(src.v_bytes);
                    cudaMemcpy(dst.v_q4.data(), src.v_q4, src.v_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.k_scale && src.k_scale_bytes > 0) {
                    dst.k_scale.resize(src.k_scale_bytes);
                    cudaMemcpy(dst.k_scale.data(), src.k_scale, src.k_scale_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.v_scale && src.v_scale_bytes > 0) {
                    dst.v_scale.resize(src.v_scale_bytes);
                    cudaMemcpy(dst.v_scale.data(), src.v_scale, src.v_scale_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.idx_tail && src.idx_tail_bytes > 0) {
                    dst.idx_tail.resize(src.idx_tail_bytes);
                    cudaMemcpy(dst.idx_tail.data(), src.idx_tail, src.idx_tail_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.idx_dead && src.idx_dead_bytes > 0) {
                    dst.idx_dead.resize(src.idx_dead_bytes);
                    cudaMemcpy(dst.idx_dead.data(), src.idx_dead, src.idx_dead_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.idx_pooled && src.idx_pooled_bytes > 0) {
                    dst.idx_pooled.resize(src.idx_pooled_bytes);
                    cudaMemcpy(dst.idx_pooled.data(), src.idx_pooled, src.idx_pooled_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.idx_block_pos) {
                    dst.has_idx_block_pos = true;
                    cudaMemcpy(&dst.idx_block_pos_val, src.idx_block_pos, sizeof(int32_t), cudaMemcpyDeviceToHost);
                }
            }
        }
        ss.free_device();
    }
    stage_snapshots.clear();
    is_host_parked = true;
}

bool RadixNode::offload_to_nvme(NVMeStorageTier& nvme) {
    if (!has_host_snapshot() && has_device_snapshot()) {
        park_to_host();
    }
    if (!has_host_snapshot()) return false;

    // Calculate total serialization size
    size_t total_payload = sizeof(NVMeNodeHeader);
    total_payload += edge_tokens.size() * sizeof(int32_t);
    total_payload += stage_host_snapshots.size() * sizeof(NVMeStageHeader);

    for (const auto& hss : stage_host_snapshots) {
        total_payload += hss.gdn_data.size();
        total_payload += hss.ple_data.size();
        total_payload += hss.R_data.size();
        total_payload += hss.qsa_slices.size() * sizeof(NVMeQsaSliceHeader);
        for (const auto& slice : hss.qsa_slices) {
            total_payload += slice.k_q.size();
            total_payload += slice.v_q4.size();
            total_payload += slice.k_scale.size();
            total_payload += slice.v_scale.size();
            total_payload += slice.idx_tail.size();
            total_payload += slice.idx_dead.size();
            total_payload += slice.idx_pooled.size();
        }
    }

    AlignedBuffer buffer(total_payload);
    uint8_t* p = buffer.data();
    size_t offset = 0;

    NVMeNodeHeader hdr;
    hdr.magic = kNVMeMagic;
    hdr.version = kNVMeVersion;
    hdr.node_id = id;
    hdr.prefix_len = prefix_len;
    hdr.n_edge_tokens = static_cast<uint32_t>(edge_tokens.size());
    hdr.n_stages = static_cast<uint32_t>(stage_host_snapshots.size());
    hdr.total_payload_bytes = total_payload;
    hdr.checksum = 0;

    std::memcpy(p + offset, &hdr, sizeof(hdr));
    offset += sizeof(hdr);

    if (!edge_tokens.empty()) {
        size_t bytes = edge_tokens.size() * sizeof(int32_t);
        std::memcpy(p + offset, edge_tokens.data(), bytes);
        offset += bytes;
    }

    for (const auto& hss : stage_host_snapshots) {
        NVMeStageHeader shdr;
        shdr.device = hss.device;
        shdr.ple_prev[0] = hss.ple_prev_saved[0];
        shdr.ple_prev[1] = hss.ple_prev_saved[1];
        shdr.ple_token = hss.ple_token_saved;
        shdr.gdn_bytes = hss.gdn_data.size();
        shdr.ple_bytes = hss.ple_data.size();
        shdr.R_bytes = hss.R_data.size();
        shdr.qsa_ord0 = hss.qsa_ord0;
        shdr.qsa_alloc = hss.qsa_alloc;
        shdr.n_qsa_slices = static_cast<uint32_t>(hss.qsa_slices.size());

        std::memcpy(p + offset, &shdr, sizeof(shdr));
        offset += sizeof(shdr);

        if (shdr.gdn_bytes > 0) {
            std::memcpy(p + offset, hss.gdn_data.data(), shdr.gdn_bytes);
            offset += shdr.gdn_bytes;
        }
        if (shdr.ple_bytes > 0) {
            std::memcpy(p + offset, hss.ple_data.data(), shdr.ple_bytes);
            offset += shdr.ple_bytes;
        }
        if (shdr.R_bytes > 0) {
            std::memcpy(p + offset, hss.R_data.data(), shdr.R_bytes);
            offset += shdr.R_bytes;
        }

        for (const auto& slice : hss.qsa_slices) {
            NVMeQsaSliceHeader qhdr;
            qhdr.k_bytes = slice.k_q.size();
            qhdr.v_bytes = slice.v_q4.size();
            qhdr.k_scale_bytes = slice.k_scale.size();
            qhdr.v_scale_bytes = slice.v_scale.size();
            qhdr.idx_tail_bytes = slice.idx_tail.size();
            qhdr.idx_dead_bytes = slice.idx_dead.size();
            qhdr.idx_pooled_bytes = slice.idx_pooled.size();
            qhdr.idx_block_pos_val = slice.idx_block_pos_val;
            qhdr.has_idx_block_pos = slice.has_idx_block_pos ? 1 : 0;

            std::memcpy(p + offset, &qhdr, sizeof(qhdr));
            offset += sizeof(qhdr);

            if (qhdr.k_bytes > 0) {
                std::memcpy(p + offset, slice.k_q.data(), qhdr.k_bytes);
                offset += qhdr.k_bytes;
            }
            if (qhdr.v_bytes > 0) {
                std::memcpy(p + offset, slice.v_q4.data(), qhdr.v_bytes);
                offset += qhdr.v_bytes;
            }
            if (qhdr.k_scale_bytes > 0) {
                std::memcpy(p + offset, slice.k_scale.data(), qhdr.k_scale_bytes);
                offset += qhdr.k_scale_bytes;
            }
            if (qhdr.v_scale_bytes > 0) {
                std::memcpy(p + offset, slice.v_scale.data(), qhdr.v_scale_bytes);
                offset += qhdr.v_scale_bytes;
            }
            if (qhdr.idx_tail_bytes > 0) {
                std::memcpy(p + offset, slice.idx_tail.data(), qhdr.idx_tail_bytes);
                offset += qhdr.idx_tail_bytes;
            }
            if (qhdr.idx_dead_bytes > 0) {
                std::memcpy(p + offset, slice.idx_dead.data(), qhdr.idx_dead_bytes);
                offset += qhdr.idx_dead_bytes;
            }
            if (qhdr.idx_pooled_bytes > 0) {
                std::memcpy(p + offset, slice.idx_pooled.data(), qhdr.idx_pooled_bytes);
                offset += qhdr.idx_pooled_bytes;
            }
        }
    }

    if (!nvme.write_node_direct(id, buffer.data(), total_payload)) {
        return false;
    }

    l3_bytes = total_payload;
    is_l3_offloaded = true;
    free_host();
    return true;
}

bool RadixNode::hydrate_from_nvme(NVMeStorageTier& nvme) {
    if (!is_l3_offloaded) return true;
    if (l3_bytes == 0) return false;

    AlignedBuffer buffer(l3_bytes);
    if (!nvme.read_node_direct(id, buffer.data(), l3_bytes)) {
        return false;
    }

    const uint8_t* p = buffer.data();
    size_t offset = 0;

    NVMeNodeHeader hdr;
    std::memcpy(&hdr, p + offset, sizeof(hdr));
    offset += sizeof(hdr);

    if (hdr.magic != kNVMeMagic || hdr.version != kNVMeVersion) {
        return false;
    }

    if (hdr.n_edge_tokens > 0) {
        size_t bytes = hdr.n_edge_tokens * sizeof(int32_t);
        edge_tokens.resize(hdr.n_edge_tokens);
        std::memcpy(edge_tokens.data(), p + offset, bytes);
        offset += bytes;
        update_chunk_hashes();
    }

    stage_host_snapshots.resize(hdr.n_stages);
    for (uint32_t st = 0; st < hdr.n_stages; ++st) {
        NVMeStageHeader shdr;
        std::memcpy(&shdr, p + offset, sizeof(shdr));
        offset += sizeof(shdr);

        auto& hss = stage_host_snapshots[st];
        hss.device = shdr.device;
        hss.ple_prev_saved[0] = shdr.ple_prev[0];
        hss.ple_prev_saved[1] = shdr.ple_prev[1];
        hss.ple_token_saved = shdr.ple_token;
        hss.qsa_ord0 = shdr.qsa_ord0;
        hss.qsa_alloc = shdr.qsa_alloc;

        if (shdr.gdn_bytes > 0) {
            hss.gdn_data.resize(shdr.gdn_bytes);
            std::memcpy(hss.gdn_data.data(), p + offset, shdr.gdn_bytes);
            offset += shdr.gdn_bytes;
        }
        if (shdr.ple_bytes > 0) {
            hss.ple_data.resize(shdr.ple_bytes);
            std::memcpy(hss.ple_data.data(), p + offset, shdr.ple_bytes);
            offset += shdr.ple_bytes;
        }
        if (shdr.R_bytes > 0) {
            hss.R_data.resize(shdr.R_bytes);
            std::memcpy(hss.R_data.data(), p + offset, shdr.R_bytes);
            offset += shdr.R_bytes;
        }

        hss.qsa_slices.resize(shdr.n_qsa_slices);
        for (uint32_t j = 0; j < shdr.n_qsa_slices; ++j) {
            NVMeQsaSliceHeader qhdr;
            std::memcpy(&qhdr, p + offset, sizeof(qhdr));
            offset += sizeof(qhdr);

            auto& slice = hss.qsa_slices[j];
            slice.idx_block_pos_val = qhdr.idx_block_pos_val;
            slice.has_idx_block_pos = qhdr.has_idx_block_pos != 0;

            if (qhdr.k_bytes > 0) {
                slice.k_q.resize(qhdr.k_bytes);
                std::memcpy(slice.k_q.data(), p + offset, qhdr.k_bytes);
                offset += qhdr.k_bytes;
            }
            if (qhdr.v_bytes > 0) {
                slice.v_q4.resize(qhdr.v_bytes);
                std::memcpy(slice.v_q4.data(), p + offset, qhdr.v_bytes);
                offset += qhdr.v_bytes;
            }
            if (qhdr.k_scale_bytes > 0) {
                slice.k_scale.resize(qhdr.k_scale_bytes);
                std::memcpy(slice.k_scale.data(), p + offset, qhdr.k_scale_bytes);
                offset += qhdr.k_scale_bytes;
            }
            if (qhdr.v_scale_bytes > 0) {
                slice.v_scale.resize(qhdr.v_scale_bytes);
                std::memcpy(slice.v_scale.data(), p + offset, qhdr.v_scale_bytes);
                offset += qhdr.v_scale_bytes;
            }
            if (qhdr.idx_tail_bytes > 0) {
                slice.idx_tail.resize(qhdr.idx_tail_bytes);
                std::memcpy(slice.idx_tail.data(), p + offset, qhdr.idx_tail_bytes);
                offset += qhdr.idx_tail_bytes;
            }
            if (qhdr.idx_dead_bytes > 0) {
                slice.idx_dead.resize(qhdr.idx_dead_bytes);
                std::memcpy(slice.idx_dead.data(), p + offset, qhdr.idx_dead_bytes);
                offset += qhdr.idx_dead_bytes;
            }
            if (qhdr.idx_pooled_bytes > 0) {
                slice.idx_pooled.resize(qhdr.idx_pooled_bytes);
                std::memcpy(slice.idx_pooled.data(), p + offset, qhdr.idx_pooled_bytes);
                offset += qhdr.idx_pooled_bytes;
            }
        }
    }

    is_host_parked = true;
    is_l3_offloaded = false;
    return true;
}

void RadixNode::free_nvme(NVMeStorageTier& nvme) {
    if (is_l3_offloaded) {
        nvme.remove_node(id);
        is_l3_offloaded = false;
        l3_bytes = 0;
    }
}

RadixNode::~RadixNode() {
    free_device();
    free_host();
}

RadixTree::RadixTree(size_t max_cached_snapshots, size_t max_host_snapshots)
    : max_cached_snapshots_(max_cached_snapshots), max_host_snapshots_(max_host_snapshots) {
    const char* env_l2 = std::getenv("STRATA_HICACHE_L2_SLOTS");
    if (env_l2) {
        max_host_snapshots_ = (size_t) std::strtoul(env_l2, nullptr, 10);
    }
    root_ = std::make_shared<RadixNode>();
    root_->id = 0;
    root_->prefix_len = 0;
    root_->last_accessed = std::chrono::steady_clock::now();
    node_count_ = 1;
}

RadixTree::~RadixTree() = default;

RadixMatch RadixTree::match_prefix(const int32_t* tokens, size_t n) const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    RadixMatch best{nullptr, 0};
    std::shared_ptr<RadixNode> curr = root_;
    size_t matched_len = 0;

    while (curr && matched_len < n) {
        const int32_t next_tok = tokens[matched_len];
        auto it = curr->children.find(next_tok);
        if (it == curr->children.end()) break;

        auto child = it->second;
        const auto& edge = child->edge_tokens;
        const auto& chunk_hashes = child->edge_chunk_hashes;
        size_t match_edge = 0;
        const size_t max_comp = std::min(edge.size(), n - matched_len);
        const size_t num_chunks = max_comp / kRadixChunkTokens;

        // Task 2.1: 64-token chunk hash comparison
        size_t c = 0;
        for (; c < num_chunks && c < chunk_hashes.size(); ++c) {
            const uint64_t q_hash = compute_token_chunk_hash(tokens + matched_len + c * kRadixChunkTokens, kRadixChunkTokens);
            if (chunk_hashes[c] != q_hash) break;
            match_edge += kRadixChunkTokens;
        }

        // Remainder scalar comparison
        while (match_edge < max_comp && edge[match_edge] == tokens[matched_len + match_edge]) {
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

RadixMatch RadixTree::match_prefix_overlap(const int32_t* tokens, size_t n, float* out_overlap_ratio) const {
    auto match = match_prefix(tokens, n);
    if (out_overlap_ratio) {
        *out_overlap_ratio = (n > 0) ? (static_cast<float>(match.matched_tokens) / static_cast<float>(n)) : 0.0f;
    }
    return match;
}

RadixMatch RadixTree::match_prefix(const int64_t* tokens, size_t n) const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    RadixMatch best{nullptr, 0};
    std::shared_ptr<RadixNode> curr = root_;
    size_t matched_len = 0;

    while (curr && matched_len < n) {
        const int32_t next_tok = (int32_t) tokens[matched_len];
        auto it = curr->children.find(next_tok);
        if (it == curr->children.end()) break;

        auto child = it->second;
        const auto& edge = child->edge_tokens;
        const auto& chunk_hashes = child->edge_chunk_hashes;
        size_t match_edge = 0;
        const size_t max_comp = std::min(edge.size(), n - matched_len);
        const size_t num_chunks = max_comp / kRadixChunkTokens;

        // Task 2.1: 64-token chunk hash comparison
        size_t c = 0;
        for (; c < num_chunks && c < chunk_hashes.size(); ++c) {
            const uint64_t q_hash = compute_token_chunk_hash(tokens + matched_len + c * kRadixChunkTokens, kRadixChunkTokens);
            if (chunk_hashes[c] != q_hash) break;
            match_edge += kRadixChunkTokens;
        }

        // Remainder scalar comparison
        while (match_edge < max_comp && (int64_t) edge[match_edge] == tokens[matched_len + match_edge]) {
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

    std::unique_lock<std::shared_mutex> lock(rw_lock_);
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
            new_node->update_chunk_hashes();
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
            split_node->update_chunk_hashes();
            split_node->parent = curr;
            split_node->last_accessed = std::chrono::steady_clock::now();

            child->edge_tokens.erase(child->edge_tokens.begin(), child->edge_tokens.begin() + match_edge);
            child->update_chunk_hashes();
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
                new_node->update_chunk_hashes();
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
            const core::OnDevice on(stage_devices[st]);
            cudaStreamSynchronize((cudaStream_t) streams[st]);
        }
        ++cached_snapshots_;
        evict_lru_locked(max_cached_snapshots_);
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

    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    if (!node || !node->has_snapshot()) {
        err = "radix_fork: node has no snapshot";
        return false;
    }

    if (node->is_l3_offloaded && nvme_tier_) {
        if (!node->hydrate_from_nvme(*nvme_tier_)) {
            err = "radix_fork: failed to hydrate node from NVMe L3";
            return false;
        }
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

    if (node->has_device_snapshot()) {
        for (size_t st = 0; st < n_stages && st < node->stage_snapshots.size(); ++st) {
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
    } else if (node->has_host_snapshot()) {
        for (size_t st = 0; st < n_stages && st < node->stage_host_snapshots.size(); ++st) {
            const auto& hss = node->stage_host_snapshots[st];
            auto* child = child_states[st];
            const core::OnDevice on(hss.device);
            cudaStream_t cs = (cudaStream_t) streams[st];

            if (!hss.R_data.empty() && child->block.R) {
                cudaMemcpyAsync(child->block.R, hss.R_data.data(), hss.R_data.size(), cudaMemcpyHostToDevice, cs);
            }

            if (!hss.gdn_data.empty() && child->gdn_state) {
                cudaMemcpyAsync(child->gdn_state, hss.gdn_data.data(), hss.gdn_data.size(), cudaMemcpyHostToDevice, cs);
            }

            if (!hss.ple_data.empty() && child->ple_hist) {
                cudaMemcpyAsync(child->ple_hist, hss.ple_data.data(), hss.ple_data.size(), cudaMemcpyHostToDevice, cs);
                child->ple_prev[0] = hss.ple_prev_saved[0];
                child->ple_prev[1] = hss.ple_prev_saved[1];
                child->ple_token = hss.ple_token_saved;
            }

            for (int64_t j = 0; j < child->qsa_alloc && j < (int64_t) hss.qsa_slices.size(); ++j) {
                const auto& slice = hss.qsa_slices[j];
                QsaState& cst = child->qsa_states[child->qsa_ord0 + j];

                if (cst.kv_q4) {
                    const size_t bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    if (!slice.k_q.empty() && cst.k_q4) cudaMemcpyAsync(cst.k_q4, slice.k_q.data(), bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.v_q4.empty() && cst.v_q4) cudaMemcpyAsync(cst.v_q4, slice.v_q4.data(), bytes, cudaMemcpyHostToDevice, cs);
                } else if (cst.kv_hybrid) {
                    const size_t k_bytes = qsa_rows * qs.head_dim;
                    const size_t sc_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    const size_t v_bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    if (!slice.k_q.empty() && cst.k_q) cudaMemcpyAsync(cst.k_q, slice.k_q.data(), k_bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.k_scale.empty() && cst.k_scale) cudaMemcpyAsync(cst.k_scale, slice.k_scale.data(), sc_bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.v_q4.empty() && cst.v_q4) cudaMemcpyAsync(cst.v_q4, slice.v_q4.data(), v_bytes, cudaMemcpyHostToDevice, cs);
                } else if (cst.kv_int8) {
                    const size_t bytes = qsa_rows * qs.head_dim;
                    const size_t sc_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    if (!slice.k_q.empty() && cst.k_q) cudaMemcpyAsync(cst.k_q, slice.k_q.data(), bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.v_q4.empty() && cst.v_q) cudaMemcpyAsync(cst.v_q, slice.v_q4.data(), bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.k_scale.empty() && cst.k_scale) cudaMemcpyAsync(cst.k_scale, slice.k_scale.data(), sc_bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.v_scale.empty() && cst.v_scale) cudaMemcpyAsync(cst.v_scale, slice.v_scale.data(), sc_bytes, cudaMemcpyHostToDevice, cs);
                }

                if (!slice.idx_tail.empty() && cst.idx_tail) {
                    cudaMemcpyAsync(cst.idx_tail, slice.idx_tail.data(), slice.idx_tail.size(), cudaMemcpyHostToDevice, cs);
                }
                if (!slice.idx_dead.empty() && cst.idx_dead) {
                    cudaMemcpyAsync(cst.idx_dead, slice.idx_dead.data(), slice.idx_dead.size(), cudaMemcpyHostToDevice, cs);
                }
                if (slice.has_idx_block_pos && cst.idx_block_pos) {
                    cudaMemcpyAsync(cst.idx_block_pos, &slice.idx_block_pos_val, sizeof(int32_t), cudaMemcpyHostToDevice, cs);
                }
                const int64_t pooled_rows = prefix_len / qs.idx_block;
                if (pooled_rows > 0 && !slice.idx_pooled.empty() && cst.idx_pooled) {
                    cudaMemcpyAsync(cst.idx_pooled, slice.idx_pooled.data(), slice.idx_pooled.size(), cudaMemcpyHostToDevice, cs);
                }
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
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    return evict_lru_locked(max_snapshots, min_free_vram_mib);
}

size_t RadixTree::evict_lru_locked(size_t max_snapshots, size_t min_free_vram_mib) {
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

    // Tier 1: Park excess VRAM snapshots to HiCache L2 Host-RAM
    while (cached_snapshots_ > 0 && should_evict()) {
        std::vector<std::shared_ptr<RadixNode>> leaves;
        collect_unreferenced_leaves(root_, leaves);
        if (leaves.empty()) break; // All leaves actively in use

        std::sort(leaves.begin(), leaves.end(), [](const auto& a, const auto& b) {
            return a->last_accessed < b->last_accessed;
        });

        std::shared_ptr<RadixNode> victim = nullptr;
        for (const auto& leaf : leaves) {
            if (leaf->has_device_snapshot()) {
                victim = leaf;
                break;
            }
        }
        if (!victim) break;

        victim->park_to_host();
        if (cached_snapshots_ > 0) --cached_snapshots_;
        ++cached_host_snapshots_;
        ++evicted;
    }

    // Tier 2: Prune host snapshots when exceeding max_host_snapshots_
    while (cached_host_snapshots_ > max_host_snapshots_) {
        std::vector<std::shared_ptr<RadixNode>> leaves;
        collect_unreferenced_leaves(root_, leaves);
        if (leaves.empty()) break;

        std::sort(leaves.begin(), leaves.end(), [](const auto& a, const auto& b) {
            return a->last_accessed < b->last_accessed;
        });

        std::shared_ptr<RadixNode> host_victim = nullptr;
        for (const auto& leaf : leaves) {
            if (leaf->has_host_snapshot()) {
                host_victim = leaf;
                break;
            }
        }
        if (nvme_tier_ && host_victim->has_host_snapshot()) {
            if (host_victim->offload_to_nvme(*nvme_tier_)) {
                ++cached_nvme_snapshots_;
            } else {
                host_victim->free_host();
            }
        } else {
            host_victim->free_host();
        }
        if (cached_host_snapshots_ > 0) --cached_host_snapshots_;

        if (host_victim->children.empty() && host_victim->ref_count == 0 && !host_victim->has_snapshot()) {
            if (auto p = host_victim->parent.lock()) {
                if (!host_victim->edge_tokens.empty()) {
                    p->children.erase(host_victim->edge_tokens[0]);
                    --node_count_;
                }
            }
        }
    }
    return evicted;
}

CompactionStats RadixTree::compact_tree() {
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    CompactionStats stats;
    const auto start = std::chrono::steady_clock::now();

    size_t active_nodes = 0;
    size_t vram_bytes = 0;
    std::vector<std::shared_ptr<RadixNode>> stack = {root_};
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node) continue;
        ++stats.nodes_scanned;
        if (node->ref_count > 0 || node->has_snapshot()) {
            ++active_nodes;
            vram_bytes += node->total_vram_bytes();
        }
        for (const auto& kv : node->children) {
            if (kv.second) stack.push_back(kv.second);
        }
    }
    stats.initial_fragmentation = RadixCompactor::compute_fragmentation(
        active_nodes, node_count_ > 0 ? node_count_ : 1,
        vram_bytes, (vram_bytes > 0 ? vram_bytes : 1) * 2);

    std::function<void(std::shared_ptr<RadixNode>)> prune_dead_leaves = [&](std::shared_ptr<RadixNode> curr) {
        if (!curr) return;
        std::vector<int32_t> dead_keys;
        for (auto& kv : curr->children) {
            prune_dead_leaves(kv.second);
            if (kv.second && kv.second->children.empty() && kv.second->ref_count == 0 && !kv.second->has_snapshot()) {
                dead_keys.push_back(kv.first);
            }
        }
        for (int32_t k : dead_keys) {
            auto victim = curr->children[k];
            if (victim) {
                stats.bytes_reclaimed += victim->total_vram_bytes() + victim->total_host_bytes();
            }
            curr->children.erase(k);
            if (node_count_ > 0) --node_count_;
            ++stats.dead_leaves_pruned;
        }
    };
    prune_dead_leaves(root_);

    std::function<void(std::shared_ptr<RadixNode>)> compress_paths = [&](std::shared_ptr<RadixNode> curr) {
        if (!curr) return;
        for (auto& kv : curr->children) {
            compress_paths(kv.second);
        }
        if (curr != root_ && curr->ref_count == 0 && !curr->has_snapshot() && curr->children.size() == 1) {
            auto child = curr->children.begin()->second;
            if (child) {
                curr->edge_tokens.insert(curr->edge_tokens.end(), child->edge_tokens.begin(), child->edge_tokens.end());
                curr->update_chunk_hashes();
                curr->children = std::move(child->children);
                for (auto& c_kv : curr->children) {
                    if (c_kv.second) c_kv.second->parent = curr;
                }
                curr->stage_snapshots = std::move(child->stage_snapshots);
                curr->stage_host_snapshots = std::move(child->stage_host_snapshots);
                curr->is_host_parked = child->is_host_parked;
                curr->is_l3_offloaded = child->is_l3_offloaded;
                curr->l3_bytes = child->l3_bytes;
                curr->prefix_len = child->prefix_len;
                curr->last_accessed = std::max(curr->last_accessed, child->last_accessed);
                if (node_count_ > 0) --node_count_;
                ++stats.chains_merged;
            }
        }
    };
    compress_paths(root_);

    const auto end = std::chrono::steady_clock::now();
    stats.compaction_time_us = std::chrono::duration<double, std::micro>(end - start).count();

    active_nodes = 0;
    vram_bytes = 0;
    stack = {root_};
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node) continue;
        if (node->ref_count > 0 || node->has_snapshot()) {
            ++active_nodes;
            vram_bytes += node->total_vram_bytes();
        }
        for (const auto& kv : node->children) {
            if (kv.second) stack.push_back(kv.second);
        }
    }
    stats.final_fragmentation = RadixCompactor::compute_fragmentation(
        active_nodes, node_count_ > 0 ? node_count_ : 1,
        vram_bytes, (vram_bytes > 0 ? vram_bytes : 1) * 2);

    return stats;
}

CompactionStats RadixCompactor::compact(RadixTree* tree) {
    std::lock_guard<std::mutex> lock(mutex_);
    RadixTree* target = tree ? tree : tree_;
    if (!target) return {};
    CompactionStats stats = target->compact_tree();
    ++total_compactions_;
    total_leaves_pruned_ += stats.dead_leaves_pruned;
    total_chains_merged_ += stats.chains_merged;
    total_bytes_reclaimed_ += stats.bytes_reclaimed;
    return stats;
}

}  // namespace strata::core
