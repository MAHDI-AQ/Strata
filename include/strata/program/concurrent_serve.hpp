#pragma once
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_cache.hpp"
#include <memory>

namespace strata::program {
struct ConcurrentConfig {
    int requests = 1, rows = 8, window = 4, mtp_window_rows = 4, prefill_chunk = 256;
    bool depth = false;
    int64_t context = 32768, draft_context = 32768;
    int reserve_mib = 1536, suffix = 0;
    int adapt_every = 4, adapt_swaps = 96;
    float spec_min_p = 0.5f;
    std::string mtp_dir;
    std::vector<int64_t> eos;
};
class ConcurrentServe {
public:
    explicit ConcurrentServe(ConcurrentConfig config);
    ~ConcurrentServe();
    // Allocate sequence states and the shared prompt workspace BEFORE sizing the expert cache.
    bool prepare(const core::ModelGeometry&, core::SessionState&, core::MtpDrafter&, std::string&);
    int run(const core::WeightTable&, const core::NativeHead*, core::ExpertSource*, core::ExpertCache&,
            int32_t* host_res, const core::VerifyHits&, core::ExpertDispatch&,
            core::PoolMultiFn, void* user, std::string&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
