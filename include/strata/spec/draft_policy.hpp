// include/strata/spec/draft_policy.hpp - per verify round: the MTP's window, or a lookup (suffix) window?
//
// The suffix drafter (prompt lookup) proposes the tokens that followed an earlier repeat of the context. Taken
// whenever it proposes more than the MTP, it lost 2-8% on ordinary text: a long lookup window costs much more to
// verify than the MTP's usual 3-4 tokens, and it is only worth that when enough of it is accepted. llama.cpp's
// lookup decoding answers the same problem with confidence thresholds on its n-gram statistics; here the policy
// learns both sides online and compares expected committed tokens per millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup drafts whose
//           match was about as long (4 buckets of match length, decayed counts), at the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the measured round
// times per window size (EMA; sizes not seen yet are scaled from seen ones by a prior shape), so the policy adapts
// to the machine and the context length. It only chooses which drafts to verify: the output is unchanged.
#pragma once

#include "strata/spec/tree_spec.hpp"
#include <array>
#include <vector>

namespace strata::spec {

class DraftPolicy {
public:
    static constexpr int kMaxT = 16;
    static constexpr int kBuckets = 4;

    explicit DraftPolicy(int max_t, double margin = 0.03);

    struct Pick {
        bool lookup = false;
        int t = 1;                      // window size (1 + drafts)
    };
    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_mtp, int lookup_k, int match) const;
    /// After the round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    double lookup_rate(int match) const;   // current q for a match length
    double cost_ms(int t) const;           // measured or scaled round time of a window of t tokens

    /// Dynamic Confidence Gater & Entropy Throttling
    int decide_depth(float first_prob, int base_t, float spec_min_p) const;
    ConfidenceGater& confidence_gater() { return gater_; }
    const ConfidenceGater& confidence_gater() const { return gater_; }

    /// Task 7.1: Multi-Branch SpecTree Topology Generation (T <= 16)
    /// Task 7.1: Multi-Branch SpecTree Topology Generation (T <= 16)
    static inline std::vector<int> generate_tree_topology(int budget, float top1_prob, float top2_prob) {
        budget = std::clamp(budget, 1, kMaxT);
        std::vector<int> topology;
        if (budget <= 1) {
            topology.push_back(1);
            return topology;
        }
        topology.push_back(1); // Root node
        int remaining = budget - 1;

        if (top1_prob >= 0.85f) {
            // High confidence: deep narrow tree
            while (remaining > 0) {
                int branch = std::min(remaining, (top2_prob > 0.08f) ? 2 : 1);
                topology.push_back(branch);
                remaining -= branch;
            }
        } else if (top1_prob >= 0.60f) {
            // Moderate confidence: balanced tree
            int d = 1;
            while (remaining > 0) {
                int width = std::min(remaining, std::max(2, d * 2));
                topology.push_back(width);
                remaining -= width;
                d++;
            }
        } else {
            // Divergent / high entropy: wide multi-branch (e.g., [1, 3, 6, 6] for budget 16)
            int d = 1;
            while (remaining > 0) {
                int width = (d == 1) ? std::min(remaining, 3) : std::min(remaining, 6);
                topology.push_back(width);
                remaining -= width;
                d++;
            }
        }
        return topology;
    }

private:
    static int bucket(int match);
    double mtp_tokens(int t) const;

    int max_t_;
    double margin_;
    ConfidenceGater gater_{2.0f, 0.6f, 1.5f};
    std::array<double, kMaxT + 1> cost_{}, cost_n_{};      // round ms by window size
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
};

}  // namespace strata::spec
