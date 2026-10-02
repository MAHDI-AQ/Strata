#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/kernels/cpu/expert.hpp"   // MAXT: the batch row envelope this budget is capped by

namespace strata::program {
// Requests are supplied in rotating fairness order. Zero means this request waits for the next iteration.
inline std::vector<int> schedule_rows(const std::vector<int>& wanted, int budget, bool depth) {
    // The budget is the batch ROW ENVELOPE (MAXT): the pool's window tables, the verify workspace and
    // `--batch-rows` all follow that one constant - a cap is a lattice, keep them in lockstep.
    // The request lattice is 16 concurrent requests (M2 raise from 8; 16-legal/17-reject is covered in
    // the schedule tests).
    if (budget < 1 || budget > strata::kernels::cpu::MAXT || wanted.size() > 16)
        throw std::invalid_argument("batch budget must be 1.." + std::to_string(strata::kernels::cpu::MAXT) +
                                    ", requests <= 16");
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
