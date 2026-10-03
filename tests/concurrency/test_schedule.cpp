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
    for (int budget = 1; budget <= 48; ++budget) for (bool depth : {false, true}) {
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
    // The envelope raise: budget 48 (MAXT rows) is accepted, 49 is refused - failing closed above.
    for (int invalid : {0,49,64}) {
        bool rejected = false;
        try { schedule_rows({4,4}, invalid, false); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    {
        // The full 24-row envelope allocates across the agents: 3 x 8 = 24 rows (240 entries at k = 10).
        assert((schedule_rows({8,8,8}, 24, false) == std::vector<int>{8,8,8}));
        // Sprint 2: 5 agents with 8 speculative rows each = 40 rows:
        assert((schedule_rows({8,8,8,8,8}, 40, false) == std::vector<int>{8,8,8,8,8}));
        // Sprint 2: 6 agents with 8 speculative rows each = 48 rows:
        assert((schedule_rows({8,8,8,8,8,8}, 48, false) == std::vector<int>{8,8,8,8,8,8}));
        // With eight agents the 48 rows spread evenly (8 x 6 = 48).
        assert((schedule_rows({8,8,8,8,8,8,8,8}, 48, false) == std::vector<int>{6,6,6,6,6,6,6,6}));
    }
    // LANE m1m2 (M2 request lattice 8 -> 16): the c16 battery - the same 196,608-case shape as the
    // recorded 24-row envelope point (8^4 x 24 budgets x 2 policies), now over a 16-member wanted
    // vector (the four values cycle, then mirror).  Same invariants as the 4-wide sweep: rows sum to
    // min(budget, Sigma wanted); every row within [0, wanted[i]]; fair mode starves no member once
    // the budget covers the member count (budget >= 16).
    int checks16 = 0;
    for (int a = 1; a <= 8; ++a) for (int b = 1; b <= 8; ++b)
    for (int c = 1; c <= 8; ++c) for (int d = 1; d <= 8; ++d)
    for (int budget = 1; budget <= 48; ++budget) for (bool depth : {false, true}) {
        const int quad[4] = {a, b, c, d};
        std::vector<int> wanted(16);
        for (int i = 0; i < 16; ++i) wanted[(size_t) i] = quad[(i < 8 ? i : 15 - i) % 4];
        const auto rows = schedule_rows(wanted, budget, depth);
        int total = 0; for (int n : wanted) total += n;
        assert(std::accumulate(rows.begin(), rows.end(), 0) == std::min(budget, total));
        for (int i = 0; i < 16; ++i) assert(rows[(size_t) i] >= 0 && rows[(size_t) i] <= wanted[(size_t) i]);
        if (!depth && budget >= 16) for (int n : rows) assert(n > 0);
        ++checks16;
    }
    // Rotating the eligible list must give every one of 16 members progress with a one-row budget.
    for (bool depth : {false, true}) {
        int progress[16]{};
        for (int rotation = 0; rotation < 16; ++rotation) {
            const auto rows = schedule_rows(std::vector<int>(16, 8), 1, depth);
            for (int j = 0; j < 16; ++j) progress[(rotation+j)%16] += rows[(size_t) j];
        }
        for (int n : progress) assert(n == 1);
    }
    // The full 48-row envelope spreads across 16 members: all 16 get 3 rows (16 x 3 = 48).
    assert((schedule_rows(std::vector<int>(16, 8), 48, false) ==
            std::vector<int>(16, 3)));
    {
        // The request lattice: 16 requests are accepted, 17 are not (M2 raise).  (The 8-legal case the
        // old pair-combine test asserted is now covered by the 16-wide sweep and the named 16 case.)
        assert(schedule_rows(std::vector<int>(16, 1), 24, false).size() == 16);
        bool rejected = false;
        try { schedule_rows(std::vector<int>(17, 1), 24, false); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    {
        // The per-window bound (kVerifyMaxT = 8) did NOT move with the row envelope: a 9-token window is
        // refused, so no batch member can exceed the fixed per-window tables.
        bool rejected = false;
        try { schedule_rows({9,1}, 24, false); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    std::cout << checks << " schedule cases passed\n";
    std::cout << checks16 << " c16 schedule cases passed\n";
}
