#include "strata/program/batch_schedule.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <numeric>
#include <iostream>
using strata::program::schedule_rows;
int main() {
    assert((schedule_rows({4,4,4,4}, 8, false) == std::vector<int>{2,2,2,2}));
    assert((schedule_rows({4,4,4,4}, 16, false) == std::vector<int>{4,4,4,4}));
    assert((schedule_rows({4,4,4,4,4,4,4,4}, 12, false) == std::vector<int>{2,2,2,2,1,1,1,1}));
    assert((schedule_rows({4,4,4,4,4,4,4,8}, 16, false) == std::vector<int>{2,2,2,2,2,2,2,2}));
    assert((schedule_rows({4,4,4,4}, 8, true) == std::vector<int>{4,4,0,0}));
    assert((schedule_rows({1,6,2}, 8, false) == std::vector<int>{1,5,2}));
    assert(schedule_rows({}, 8, false).empty());
    int checks = 0;
    for (int a = 1; a <= 8; ++a) for (int b = 1; b <= 8; ++b)
    for (int c = 1; c <= 8; ++c) for (int d = 1; d <= 8; ++d)
    for (int budget = 1; budget <= 16; ++budget) for (bool depth : {false, true}) {
        std::vector<int> wanted{a,b,c,d};
        const auto rows = schedule_rows(wanted, budget, depth);
        assert(std::accumulate(rows.begin(), rows.end(), 0) == std::min(budget, a+b+c+d));
        for (int i = 0; i < 4; ++i) assert(rows[i] >= 0 && rows[i] <= wanted[i]);
        if (!depth && budget >= 4) for (int n : rows) assert(n > 0);
        ++checks;
    }
    // Rotating the eligible list must give every request progress even with a one-row budget.
    for (bool depth : {false, true}) {
        int progress[4]{};
        for (int rotation = 0; rotation < 4; ++rotation) {
            const auto rows = schedule_rows({8,8,8,8}, 1, depth);
            for (int j = 0; j < 4; ++j) progress[(rotation+j)%4] += rows[j];
        }
        for (int n : progress) assert(n == 1);
    }
    for (int invalid : {0,17,32}) {
        bool rejected = false;
        try { schedule_rows({4,4}, invalid, false); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    {
        // The pair-combine raise: 8 requests are accepted, 9 are not.
        bool rejected = false;
        try { schedule_rows(std::vector<int>(9, 1), 16, false); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    std::cout << checks << " schedule cases passed\n";
}
