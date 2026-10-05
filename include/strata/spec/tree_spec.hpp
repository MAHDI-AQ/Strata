// include/strata/spec/tree_spec.hpp - Phase 3: Speculative Tree Topology Verification & Confidence Gating
//
// Cross-Engine Reference: EAGLE-2 (Dynamic Draft Trees), Sequoia (Tree Mask Verification), Medusa.
//
// In multi-agent serving with speculative decoding, linear draft chains (x1 -> x2 -> x3) suffer from
// catastrophic early rejection when the target model diverges at x1, even if an alternate plausible
// branch exists. SpecTree represents candidate tokens as an acyclic tree T = (V, E) of up to K nodes.
// All nodes in the tree are evaluated simultaneously in a single target model forward pass by passing
// a 2D tree attention mask M[i, j] in {0, 1} where M[i, j] = 1 iff node j is an ancestor of node i.
#pragma once

#include <array>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

namespace strata::spec {

inline constexpr int kMaxTreeNodes = 16;
inline constexpr int kDefaultTreeNodes = 8;
inline constexpr int kMaxTreeDepth = 4;

struct SpecTreeNode {
    int32_t id = 0;              // node index [0 .. count-1]
    int32_t token = 0;           // candidate token ID
    int32_t parent_idx = -1;     // parent node index (-1 for root/input)
    int32_t depth = 0;           // 0 = root, 1 = first speculative draft, etc.
    float prob = 1.0f;           // drafter confidence / probability score
    float cumulative_score = 0.0f; // log-probability along path from root
    std::vector<int32_t> children;
};

class SpecTree {
public:
    SpecTree() { clear(); }

    void clear() {
        nodes_.clear();
        nodes_.reserve(kMaxTreeNodes);
        mask_.fill(0);
    }

    int size() const { return (int) nodes_.size(); }
    bool empty() const { return nodes_.empty(); }

    const SpecTreeNode& node(int idx) const { return nodes_[idx]; }
    SpecTreeNode& node(int idx) { return nodes_[idx]; }
    const std::vector<SpecTreeNode>& nodes() const { return nodes_; }

    /// Add a node to the speculative tree. Returns the new node's index.
    int add_node(int32_t token, int32_t parent_idx, float prob = 1.0f) {
        if ((int) nodes_.size() >= kMaxTreeNodes) return -1;
        const int id = (int) nodes_.size();
        SpecTreeNode n;
        n.id = id;
        n.token = token;
        n.parent_idx = parent_idx;
        if (parent_idx >= 0 && parent_idx < id) {
            n.depth = nodes_[parent_idx].depth + 1;
            const float p_clamped = std::max(1e-5f, std::min(1.0f, prob));
            n.cumulative_score = nodes_[parent_idx].cumulative_score + std::log(p_clamped);
            nodes_[parent_idx].children.push_back(id);
        } else {
            n.depth = 0;
            n.cumulative_score = 0.0f;
        }
        n.prob = prob;
        nodes_.push_back(std::move(n));
        return id;
    }

    /// Checks if node `ancestor` is an ancestor of node `descendant` (or ancestor == descendant).
    bool is_ancestor(int ancestor, int descendant) const {
        if (ancestor < 0 || ancestor >= (int) nodes_.size()) return false;
        if (descendant < 0 || descendant >= (int) nodes_.size()) return false;
        int curr = descendant;
        while (curr >= 0) {
            if (curr == ancestor) return true;
            curr = nodes_[curr].parent_idx;
        }
        return false;
    }

    /// Computes the 2D causal tree attention mask:
    /// mask[i * kMaxTreeNodes + j] == 1 if node j is an ancestor of node i, else 0.
    void compute_mask() {
        mask_.fill(0);
        const int N = (int) nodes_.size();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (is_ancestor(j, i)) {
                    mask_[i * kMaxTreeNodes + j] = 1;
                }
            }
        }
    }

    const uint8_t* mask_data() const { return mask_.data(); }
    uint8_t mask_at(int i, int j) const {
        if (i < 0 || i >= kMaxTreeNodes || j < 0 || j >= kMaxTreeNodes) return 0;
        return mask_[i * kMaxTreeNodes + j];
    }

    /// Given target model predictions for each node in the tree:
    /// target_predictions[u] is the token emitted by the target model when verifying node u.
    /// A branch transition from node u to child v is valid if target_predictions[u] == nodes_[v].token.
    /// Returns the vector of accepted token IDs along the longest valid branch.
    std::vector<int32_t> find_longest_accepted_path(
        const int32_t* target_predictions,
        std::vector<int32_t>* accepted_indices = nullptr) const {

        if (nodes_.empty()) return {};

        // Depth-first search to find longest valid branch with highest cumulative score
        std::vector<int32_t> best_path;
        float best_score = -1e9f;

        std::vector<int32_t> current_path;
        current_path.push_back(0); // Start at root (node 0)

        auto dfs = [&](auto self, int curr_idx, float curr_score) -> void {
            if (current_path.size() > best_path.size() ||
               (current_path.size() == best_path.size() && curr_score > best_score)) {
                best_path = current_path;
                best_score = curr_score;
            }

            const int32_t expected_next = target_predictions[curr_idx];
            for (int child_idx : nodes_[curr_idx].children) {
                if (nodes_[child_idx].token == expected_next) {
                    current_path.push_back(child_idx);
                    self(self, child_idx, curr_score + nodes_[child_idx].cumulative_score);
                    current_path.pop_back();
                }
            }
        };

        dfs(dfs, 0, 0.0f);

        std::vector<int32_t> accepted_tokens;
        accepted_tokens.reserve(best_path.size());
        for (int idx : best_path) {
            accepted_tokens.push_back(nodes_[idx].token);
        }

        if (accepted_indices) {
            *accepted_indices = best_path;
        }

        return accepted_tokens;
    }

    /// Flattens the tree nodes into sequential arrays for GPU verification forward passes.
    void flatten(std::vector<int32_t>& out_tokens,
                 std::vector<int64_t>& out_positions,
                 int64_t base_position) const {
        out_tokens.clear();
        out_positions.clear();
        const int N = (int) nodes_.size();
        out_tokens.reserve(N);
        out_positions.reserve(N);
        for (int i = 0; i < N; ++i) {
            out_tokens.push_back(nodes_[i].token);
            out_positions.push_back(base_position + nodes_[i].depth);
        }
    }

private:
    std::vector<SpecTreeNode> nodes_;
    std::array<uint8_t, kMaxTreeNodes * kMaxTreeNodes> mask_{};
};

// ============================================================================
// Task 3.3: Dynamic Confidence Gater & Entropy Extraction
// ============================================================================
struct ConfidenceMetrics {
    float top1_prob = 1.0f;
    float top2_prob = 0.0f;
    float logit_margin = 10.0f; // top1 - top2 logit difference
    float entropy = 0.0f;       // -sum p log p
};

class ConfidenceGater {
public:
    explicit ConfidenceGater(float high_margin = 2.0f, float low_margin = 0.6f, float max_entropy = 1.5f)
        : high_margin_(high_margin), low_margin_(low_margin), max_entropy_(max_entropy) {}

    /// Evaluates logits to extract top-1 prob, logit margin, and Shannon entropy.
    static ConfidenceMetrics evaluate_logits(const float* logits, int vocab_size, int top_k = 10) {
        ConfidenceMetrics cm;
        if (!logits || vocab_size <= 0) return cm;

        // Find top-2 logits
        float max1 = -1e9f, max2 = -1e9f;
        int idx1 = 0;
        for (int i = 0; i < vocab_size; ++i) {
            const float val = logits[i];
            if (val > max1) {
                max2 = max1;
                max1 = val;
                idx1 = i;
            } else if (val > max2) {
                max2 = val;
            }
        }
        cm.logit_margin = max1 - max2;

        // Compute softmax over top-k for calibrated entropy
        std::vector<std::pair<float, int>> top_items;
        top_items.reserve(top_k);
        top_items.push_back({max1, idx1});

        for (int i = 0; i < vocab_size; ++i) {
            if (i == idx1) continue;
            if (top_items.size() < (size_t) top_k) {
                top_items.push_back({logits[i], i});
                if (top_items.size() == (size_t) top_k) {
                    std::sort(top_items.begin(), top_items.end(), std::greater<std::pair<float, int>>());
                }
            } else if (logits[i] > top_items.back().first) {
                top_items.back() = {logits[i], i};
                for (int j = (int) top_items.size() - 1; j > 0 && top_items[j].first > top_items[j - 1].first; --j) {
                    std::swap(top_items[j], top_items[j - 1]);
                }
            }
        }

        // Softmax & Entropy
        float sum_exp = 0.0f;
        for (const auto& item : top_items) {
            sum_exp += std::exp(item.first - max1);
        }
        cm.top1_prob = 1.0f / sum_exp;
        if (top_items.size() > 1) {
            cm.top2_prob = std::exp(top_items[1].first - max1) / sum_exp;
        }

        float ent = 0.0f;
        for (const auto& item : top_items) {
            const float p = std::exp(item.first - max1) / sum_exp;
            if (p > 1e-6f) ent -= p * std::log(p);
        }
        cm.entropy = ent;

        return cm;
    }

    /// Selects optimal speculative depth T in [1 .. max_t] and whether to use tree branching.
    struct Decision {
        int recommended_t = 1;
        bool use_tree = false;
        const char* reason = "normal";
    };

    Decision decide(const ConfidenceMetrics& cm, int max_t, float spec_min_p = 0.5f) const {
        Decision d;
        // High confidence: expand speculation depth and enable speculative tree exploration
        if (cm.logit_margin >= high_margin_ && cm.entropy <= max_entropy_ && cm.top1_prob >= spec_min_p) {
            d.recommended_t = std::min(max_t, 4);
            d.use_tree = (d.recommended_t >= 3);
            d.reason = "high_confidence_expanded";
            return d;
        }
        // Low confidence / high entropy: throttle to 1 to avoid wasted draft & verify compute
        if (cm.logit_margin < low_margin_ || cm.entropy > max_entropy_ * 1.5f || cm.top1_prob < spec_min_p * 0.7f) {
            d.recommended_t = 1;
            d.use_tree = false;
            d.reason = "high_entropy_throttled";
            return d;
        }
        // Normal regime
        d.recommended_t = std::min(max_t, 2);
        d.use_tree = false;
        d.reason = "standard_regime";
        return d;
    }

    /// Online EMA calibration based on acceptance feedback
    void observe(int offered, int accepted) {
        if (offered <= 0) return;
        const float rate = (float) accepted / (float) offered;
        // If acceptance rate is exceptionally high (>85%), loosen thresholds to speculate more aggressively
        if (rate > 0.85f) {
            high_margin_ = std::max(1.2f, high_margin_ * 0.98f);
            low_margin_ = std::max(0.3f, low_margin_ * 0.98f);
        } else if (rate < 0.60f) {
            // If acceptance rate is low (<60%), tighten thresholds to prevent compute waste
            high_margin_ = std::min(3.5f, high_margin_ * 1.05f);
            low_margin_ = std::min(1.2f, low_margin_ * 1.05f);
        }
    }

    float high_margin() const { return high_margin_; }
    float low_margin() const { return low_margin_; }

private:
    float high_margin_ = 2.0f;
    float low_margin_ = 0.6f;
    float max_entropy_ = 1.5f;
};

} // namespace strata::spec
