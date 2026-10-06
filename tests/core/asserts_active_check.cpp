// tests/core/asserts_active_check.cpp - the asserts-liveness sentinel (CMakeLists.txt:
// STRATA_ENABLE_ASSERTS strips -DNDEBUG from the Release-class flags and builds this target).
//
//   compile time: NDEBUG defined here means the build class made assert() inert - refuse to build.
//   runtime:      with STRATA_ASSERTS_PROBE_TRIP set, assert(false) must abort the process -
//                 proves assert() is compiled in and fires (asserts_gate.sh runs both probes).
#ifdef NDEBUG
#error "asserts are inert: NDEBUG is defined in this build (STRATA_ENABLE_ASSERTS=ON should strip it)"
#endif

#include <cassert>
#include <cstdlib>

int main() {
    assert(true);   // a no-op when asserts are live; the compile above refuses the inert class
    if (std::getenv("STRATA_ASSERTS_PROBE_TRIP")) {
        assert(false);   // must abort: proves assert() is compiled in and fires
    }
    return 0;
}
