#pragma once
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_cache.hpp"
#include <memory>
#include <vector>

namespace strata::program {
struct ConcurrentConfig {
    int requests = 1, rows = 8, window = 4, mtp_window_rows = 4, prefill_chunk = 256;
    int graph_cache = 8;
    bool pad_batch = false;
    bool parallel_batch = false;
    bool depth = false;
    int64_t context = 32768, draft_context = 32768;
    int reserve_mib = 1536, suffix = 0;
    int adapt_every = 4, adapt_swaps = 96;
    // P1-cache-revive: prefix-cache surface for the concurrent path. Off by default
    // (0 = disabled, mirrors --conversation-cache-mib on the serial path). The #189 core
    // is in-tree since 0.1.30 but the concurrent/split save/restore lift is not landed, so
    // any nonzero value is refused at startup; the fields exist so the flag surface is stable
    // when the lift lands. Kill-switch: 0 (flag or STRATA_CONCURRENT_CACHE_MIB=0).
    int64_t conversation_cache_mib = 0;
    int conversation_cache_slots = 4;
    int64_t conversation_cache_min_free_mib = 2560;
    float spec_min_p = 0.5f;
    std::string mtp_dir;
    std::string kv = "int8";   // resident KV mode (int8, k8v4, q4_0); reported in INFO
    std::vector<int64_t> eos;
};
/// One device's share of a layer-split engine, built by the CLI exactly where it builds its
/// GpuStages.  stages[0] is CUDA0 and carries the CLI's primary objects, so stages.size()==1 is
/// the single-GPU engine this class has always run: every field then holds the very object the
/// old run() took as a separate argument.  (Combine build C1; per-stage behavior arrives with C4.)
struct ServeStage {
    int device = 0;                            ///< CUDA device this stage runs on
    const core::WeightTable* wt = nullptr;     ///< this stage's weights (N=1: run()'s wt)
    const core::NativeHead* head = nullptr;    ///< the head this stage runs (last stage; null before it)
    int64_t lb = 0;                            ///< first layer of this stage
    int64_t le = -1;                           ///< one past its last layer (-1 accepted as "to the end")
    core::SessionState* session = nullptr;     ///< the session slot 0 borrows on this stage (N=1: the primary)
    core::ExpertCache* cache = nullptr;        ///< this stage's expert cache (N=1: run()'s cache)
    int32_t* host_res = nullptr;               ///< the residency table (slots|-1; one table, split shares it)
    core::VerifyHits hits{};                   ///< this stage's d_res/cache_base/slot_off view
    core::ExpertDispatch* dispatch = nullptr;  ///< the adapter (split: every stage carries the SplitDrive base)
};

class ConcurrentServe {
public:
    explicit ConcurrentServe(ConcurrentConfig config);
    ~ConcurrentServe();
    // Allocate the per-stage sequence states and the stage-owned prompt workspaces BEFORE sizing the expert
    // cache.  `stages` is the same list run() takes: stage 0 borrows `primary`; stages >= 1 borrow their
    // ServeStage::session (the CLI GpuStage session the borrowed drafter was loaded against).
    bool prepare(const core::ModelGeometry&, core::SessionState&, core::MtpDrafter&,
                 const std::vector<ServeStage>&, std::string&);
    // stages.size()==1 == the single-GPU engine.  stage_plans = split_drive.plan (nullptr at N=1);
    // the per-round publish targets the C4 loops refresh.
    int run(const std::vector<ServeStage>& stages, core::ExpertSource* source,
            core::PoolMultiFn pool, void* user, core::GpuPlanSink** stage_plans, std::string&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
