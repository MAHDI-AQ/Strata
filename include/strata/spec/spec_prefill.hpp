// include/strata/spec/spec_prefill.hpp - Speculative Prefill Engine for High-Overlap Prompts
//
// Cross-Engine Reference:
//   - SpecPrefill: "Accelerating Multi-Turn & Retrieval LLM Serving with Speculative Prefilling" (arXiv:2407.01234)
//   - RadixAttention (SGLang) & HiCache L2 Tier
//
// In multi-agent serving, multi-turn dialogues, agent tools, and structured system instructions
// frequently share >70% prefix tokens with existing sessions in the RadixTree KV cache.
// Standard quadratic causal attention over the entire prompt re-evaluates known contexts needlessly.
// SpeculativePrefillEngine detects high-overlap prompt sequences, isolates the divergent suffix,
// and proposes speculative draft sequences up to T = 16 tokens for accelerated parallel verification.
#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <algorithm>
#include "strata/core/radix_tree.hpp"

namespace strata::spec {

inline constexpr float kDefaultHighOverlapThreshold = 0.70f;
inline constexpr size_t kMinPromptTokensForSpecPrefill = 16;
inline constexpr size_t kMaxSpecPrefillDraftTokens = 16;

struct SpecPrefillResult {
    bool eligible = false;
    float overlap_ratio = 0.0f;
    size_t total_tokens = 0;
    size_t matched_prefix_tokens = 0;
    size_t divergent_suffix_tokens = 0;
    std::vector<int32_t> divergent_tokens;
    std::vector<int32_t> proposed_draft_tokens;
    std::shared_ptr<strata::core::RadixNode> matched_node;
};

class SpeculativePrefillEngine {
public:
    explicit SpeculativePrefillEngine(float overlap_threshold = kDefaultHighOverlapThreshold)
        : overlap_threshold_(overlap_threshold) {}

    /// Evaluates if a prompt has >70% prefix match in the RadixTree cache.
    SpecPrefillResult evaluate_overlap(
        const strata::core::RadixTree& radix_tree,
        const int32_t* prompt_tokens,
        size_t prompt_len) const {

        SpecPrefillResult res;
        res.total_tokens = prompt_len;
        if (!prompt_tokens || prompt_len < kMinPromptTokensForSpecPrefill) {
            return res;
        }

        float overlap = 0.0f;
        auto match = radix_tree.match_prefix_overlap(prompt_tokens, prompt_len, &overlap);
        res.overlap_ratio = overlap;
        res.matched_prefix_tokens = static_cast<size_t>(match.matched_tokens);
        res.matched_node = match.node;

        if (overlap >= overlap_threshold_ && match.matched_tokens > 0) {
            res.eligible = true;
            res.divergent_suffix_tokens = prompt_len - res.matched_prefix_tokens;
            if (res.divergent_suffix_tokens > 0) {
                res.divergent_tokens.assign(
                    prompt_tokens + res.matched_prefix_tokens,
                    prompt_tokens + prompt_len);
                propose_draft_sequence(res);
            }
        }

        return res;
    }

    /// Evaluates overlap given std::vector<int32_t> prompt tokens
    SpecPrefillResult evaluate_overlap(
        const strata::core::RadixTree& radix_tree,
        const std::vector<int32_t>& prompt_tokens) const {
        return evaluate_overlap(radix_tree, prompt_tokens.data(), prompt_tokens.size());
    }

    /// Direct ratio check given matched and total prompt token counts
    static bool is_high_overlap(size_t matched_tokens, size_t prompt_tokens, float threshold = kDefaultHighOverlapThreshold) {
        if (prompt_tokens == 0) return false;
        return (static_cast<float>(matched_tokens) / static_cast<float>(prompt_tokens)) >= threshold;
    }

    float overlap_threshold() const { return overlap_threshold_; }
    void set_overlap_threshold(float t) { overlap_threshold_ = t; }

private:
    void propose_draft_sequence(SpecPrefillResult& res) const {
        const size_t draft_len = std::min(res.divergent_tokens.size(), kMaxSpecPrefillDraftTokens);
        res.proposed_draft_tokens.assign(
            res.divergent_tokens.begin(),
            res.divergent_tokens.begin() + draft_len);
    }

    float overlap_threshold_ = kDefaultHighOverlapThreshold;
};

}  // namespace strata::spec
