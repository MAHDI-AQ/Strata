#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace strata::program {
// Requests are supplied in rotating fairness order. Zero means this request waits for the next iteration.
inline std::vector<int> schedule_rows(const std::vector<int>& wanted, int budget, bool depth) {
    if (budget < 1 || budget > 16 || wanted.size() > 4)
        throw std::invalid_argument("batch budget must be 1..16, requests <= 4");
    for (int n : wanted) if (n < 1 || n > 8) throw std::invalid_argument("window must be 1..8");
    std::vector<int> rows(wanted.size(), 0);
    if (depth) {
        for (size_t i = 0; i < wanted.size() && budget; ++i) {
            rows[i] = std::min(wanted[i], budget);
            budget -= rows[i];
        }
    } else {
        bool changed = true;
        while (budget && changed) {
            changed = false;
            for (size_t i = 0; i < wanted.size() && budget; ++i)
                if (rows[i] < wanted[i]) { ++rows[i]; --budget; changed = true; }
        }
    }
    return rows;
}
}
