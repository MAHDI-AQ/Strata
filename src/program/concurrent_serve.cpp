#include "strata/program/concurrent_serve.hpp"
#include "strata/program/batch_schedule.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"   // C4: kVerifyMaxT (hand-off buffer sizing)
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/spec/draft_policy.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_set>

namespace strata::program {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
bool gpu_alloc(void** p, size_t bytes, int reserve, std::string& err) {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess || free < bytes + (size_t) reserve * 1048576) {
        err = "concurrency: insufficient dedicated VRAM for request states and reserve; reduce concurrency/context";
        return false;
    }
    if (cudaMalloc(p, bytes) != cudaSuccess) { err = "concurrency: allocation failed"; return false; }
    return true;
}
struct Input {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false, closed = false;
};
struct Request {
    uint64_t id = 0;
    int64_t max_new = 0;
    std::vector<int64_t> tokens;
    kernels::SamplerParams sampling{};
    float spec_min_p = 0.5f;
};
bool parse_request(const std::string& line, const ConcurrentConfig& config, int64_t vocab, Request& r, std::string& err) {
    std::istringstream input(line);
    std::string command, id, maximum, token;
    input >> command >> id >> maximum;
    auto integer = [](const std::string& s) -> int64_t {
        size_t end = 0;
        const auto n = std::stoll(s, &end);
        if (end != s.size()) throw std::invalid_argument("integer");
        return n;
    };
    try {
        const auto number = integer(id);
        if (number < 1) throw std::invalid_argument("request id");
        r.id = (uint64_t) number;
        r.max_new = integer(maximum);
        if (command != "CGEN" || r.max_new < 1 || r.max_new > config.context) throw std::invalid_argument("max_new");
        r.sampling.greedy = true;
        r.sampling.temperature = 0;
        r.sampling.top_p = 1.0f;
        r.spec_min_p = config.spec_min_p;
        bool got_tokens = false;
        while (input >> token) {
            const auto eq = token.find('=');
            if (eq == std::string::npos) {
                if (got_tokens) throw std::invalid_argument("trailing data");
                got_tokens = true;
                std::istringstream ids(token);
                std::string cell;
                while (std::getline(ids, cell, ',')) {
                    const auto value = integer(cell);
                    if (value < 0 || value >= vocab || r.tokens.size() >= (size_t) config.context)
                        throw std::invalid_argument("token/context");
                    r.tokens.push_back(value);
                }
                if (token.back() == ',') throw std::invalid_argument("empty token");
                continue;
            }
            if (got_tokens) throw std::invalid_argument("keys after tokens");
            const std::string key = token.substr(0, eq), value = token.substr(eq + 1);
            size_t used = 0;
            const float f = std::stof(value, &used);
            if (used != value.size() || !std::isfinite(f)) throw std::invalid_argument("sampling value");
            if (key == "temperature") { if (f < 0) throw std::invalid_argument(key); r.sampling.temperature = f; r.sampling.greedy = f == 0; }
            else if (key == "top_p") { if (f <= 0 || f > 1) throw std::invalid_argument(key); r.sampling.top_p = f; }
            else if (key == "min_p") { if (f < 0 || f > 1) throw std::invalid_argument(key); r.sampling.min_p = f; }
            else if (key == "top_k") { const auto n = integer(value); if (n < 1 || n > 64) throw std::invalid_argument(key); r.sampling.top_k = (int) n; }
            else if (key == "seed") { const auto n = integer(value); if (n < 0) throw std::invalid_argument(key); r.sampling.seed = (uint64_t) n; }
            else if (key == "penalty_last_n") { const auto n = integer(value); if (n < 0 || n > 4096) throw std::invalid_argument(key); r.sampling.penalty_last_n = (int) n; }
            else if (key == "penalty_repeat") { if (f <= 0) throw std::invalid_argument(key); r.sampling.penalty_repeat = f; }
            else if (key == "penalty_freq") r.sampling.penalty_freq = f;
            else if (key == "penalty_present") r.sampling.penalty_present = f;
            else if (key == "spec_min_p") { if (f < 0 || f > 1) throw std::invalid_argument(key); r.spec_min_p = f; }
            else if (key == "cvec" && (f == 0 || f == 1)) {} // no control vector can be loaded in this mode
            else throw std::invalid_argument("unsupported key: " + key);
        }
        if (!got_tokens || r.tokens.empty()) throw std::invalid_argument("empty prompt");
        return true;
    } catch (const std::exception& e) { err = "bad concurrent request: " + std::string(e.what()); return false; }
}
void error(uint64_t id, const std::string& e) { std::printf("R %llu ERR %s\n", (unsigned long long) id, e.c_str()); std::fflush(stdout); }
}

struct ConcurrentServe::Impl {
    explicit Impl(ConcurrentConfig c) : config(std::move(c)) {}
    struct Slot {
        // C4: one entry per stage of the engine.  Slot 0 borrows the CLI's chain (the primary session and the
        // CLI stage sessions); every other slot owns one per-stage arena.  The drafter stays slot-level and
        // lives on the LAST stage's device; history_device stays slot-level too (run() allocates it there).
        struct StageSlot {
            core::SessionState owned;
            core::SessionState* state = nullptr;
            void* arena = nullptr;
            core::Verifier verify;
            prefill::Prefill prompt;
            void* ple_scratch = nullptr;   // only the stage that holds layer 1 (stage 0 for any legal split)
            // TODO(C4 run-side): per-stage destruction guard (a core::OnDevice member) is deferred - OnDevice
            // is neither default-constructible nor movable, and StageSlot lives in a std::vector.
        };
        // The stage count is fixed for the engine's life.  Sized here because a vector of StageSlot cannot be
        // resized (Verifier and Prefill are not movable: user-declared destructors with deleted copies).
        explicit Slot(size_t stage_count) : stages(stage_count) {}
        std::vector<StageSlot> stages;
        std::unique_ptr<core::MtpDrafter> draft_owner;
        core::MtpDrafter* draft = nullptr;
        int32_t* history_device = nullptr;
        std::vector<int32_t> history, consumed;
        spec::SuffixDrafter suffix;
        spec::DraftPolicy policy{8};
        Request request;
        bool active = false, first = true, lookup = false;
        int64_t read = 0, position = 0, generated = 0, offered = 0, accepted = 0;
        int32_t current = 0;
        int count = 0, match = 0;
        int32_t drafts[8]{}, lookup_tokens[8]{}, window[8]{}, output[8]{};
        float probability[8]{};
        double prompt_ms = 0;
        Clock::time_point decode_start{};
        ~Slot() {
            // TODO(C4 run-side): free under OnDevice(last stage / stage device) once stages can be non-zero.
            if (history_device) cudaFree(history_device);
            for (auto& gs : stages) if (gs.ple_scratch) cudaFree(gs.ple_scratch);
        }
    };
    ConcurrentConfig config;
    const core::ModelGeometry* geometry = nullptr;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<ServeStage> stages;             // stage list: prepare stores it, run() refreshes it
    core::GpuPlanSink** stage_plans = nullptr;  // C4: per-round publish targets (= split_drive.plan)
    // C4: one hand-off buffer per stage boundary (n_stages - 1), mapped pinned and portable: the stages on
    // either side read it from different devices.  All of a stage's slot verifiers and batch coordinators
    // point at the same buffer (C2's packed row0/phase-5 hand-off writes consume it there).
    struct Boundary { float* host = nullptr; float* dev = nullptr; };
    std::vector<Boundary> hand;
    // C1-B/C4: the prompt path is stage-owned, one entry per stage; N=1 keeps exactly one (same resources).
    struct StageRt {
        int device = 0;
        void* prompt_workspace = nullptr;
        uint64_t prompt_bytes = 0;
        cudaStream_t prompt_stream = nullptr;
    };
    std::vector<StageRt> stage_rt;
    ~Impl() {
        // Graphs and draft state must die before the sessions they reference. Slot 0's sessions are borrowed
        // from the CLI; the own per-(slot, stage) arenas are collected first and freed after slots.clear().
        struct OwnedArena { void* base; core::QsaState* qsa; int device; };
        std::vector<OwnedArena> allocations;
        for (const auto& s : slots)
            for (size_t st = 0; st < s->stages.size(); ++st) {
                auto& gs = s->stages[st];
                if (gs.arena) allocations.push_back({gs.arena, gs.owned.qsa_states,
                                                     st < stages.size() ? stages[st].device : 0});
            }
        slots.clear();
        for (const auto& a : allocations) {
            if (a.qsa) for (int64_t i = 0; i < geometry->n_qsa_layers(); ++i) {
                if (a.qsa[i].host_step) cudaFreeHost(a.qsa[i].host_step);
                if (a.qsa[i].host_pos) cudaFreeHost(a.qsa[i].host_pos);
            }
            delete[] a.qsa;
            const core::OnDevice on(a.device);   // the arena was allocated on its stage's device
            cudaFree(a.base);
        }
        for (auto& rt : stage_rt) {
            const core::OnDevice on(rt.device);  // the stream/workspace were created on that stage's device
            if (rt.prompt_stream) cudaStreamDestroy(rt.prompt_stream);
            if (rt.prompt_workspace) cudaFree(rt.prompt_workspace);
        }
        for (auto& h : hand) if (h.host) cudaFreeHost(h.host);   // C4: the per-boundary mapped hand-off buffers
    }
};
ConcurrentServe::ConcurrentServe(ConcurrentConfig c) : impl_(std::make_unique<Impl>(std::move(c))) {}
ConcurrentServe::~ConcurrentServe() = default;

bool ConcurrentServe::prepare(const core::ModelGeometry& g, core::SessionState& primary, core::MtpDrafter& draft,
                              const std::vector<ServeStage>& stages, std::string& err) {
    auto& m = *impl_;
    m.geometry = &g;
    m.stages = stages;   // C4: stage devices and borrowed sessions are read here; run() re-reads the rest
    const auto& c = m.config;
    static const core::ModelGeometry draft_geometry{};
    for (int i = 0; i < c.requests; ++i) {
        // Register ownership before any allocation that can fail partway through initialization.
        m.slots.push_back(std::make_unique<Impl::Slot>(stages.size()));
        auto& s = m.slots.back();
        s->suffix = spec::SuffixDrafter(std::max(1, c.suffix), 64, (size_t) c.context + 4096);
        for (size_t st = 0; st < stages.size(); ++st) {
            auto& gs = s->stages[st];
            const core::OnDevice on(stages[st].device);
            if (i == 0) {
                // Slot 0 borrows the CLI's chain: the primary on stage 0, the CLI stage sessions beyond it.
                gs.state = st == 0 ? &primary : stages[st].session;
            } else {
                // C4 follow-up (C5 reconciliation): an owned stage session carves ONLY its stage's layers -
                // same range convention as the CLI's per-stage carve (le == -1 resolves to the end).
                if (!gpu_alloc(&gs.arena, core::session_bytes(g, c.context, primary.k, stages[st].lb, stages[st].le), c.reserve_mib, err)) return false;
                gs.state = &gs.owned;
                if (!core::session_init(g, c.context, primary.k, gs.arena, gs.owned, stages[st].lb, stages[st].le)) { err = "concurrency: session initialization failed"; return false; }
                if (st == 0) {
                    // The PLE is a layer-1 module: only the stage that holds layer 1 (stage 0 for any legal
                    // split) wires it.  Stages >= 1 leave it unwired there (ready() == false).
                    gs.owned.ple = primary.ple;
                    gs.owned.ple.hist = gs.owned.ple_hist;
                    gs.owned.ple.prev = gs.owned.ple_prev;
                    gs.owned.ple.token = &gs.owned.ple_token;
                    // PLE weights/table are immutable, but its scratch is written during layer 1.
                    // Parallel member streams must not inherit the primary session's scratch pointer.
                    if (c.parallel_batch && primary.ple.ready()) {
                        if (!gpu_alloc(&gs.ple_scratch, (size_t) core::ple_run_scratch_bytes(), c.reserve_mib, err)) return false;
                        gs.owned.ple.scratch = static_cast<float*>(gs.ple_scratch);
                    }
                }
            }
        }
        if (i == 0) s->draft = &draft;
        else {
            s->draft_owner = std::make_unique<core::MtpDrafter>();
            s->draft = s->draft_owner.get();
            // The drafter reads the LAST stage's residual and weights, so it lives on that stage's device.
            const core::OnDevice on(stages.back().device);
            if (!s->draft->load(c.mtp_dir, draft_geometry, *s->stages.back().state, c.window, err, c.draft_context, &draft)) return false;
        }
        s->draft->set_max_drafts(c.mtp_window_rows - 1);
    }
    // C4: stage-owned prompt path, one entry per stage, sized like the stage list.  For a 1-entry list this is
    // C1-B's single workspace/stream (same allocations, same order as before).
    m.stage_rt.clear();
    m.stage_rt.resize(stages.size());
    for (size_t st = 0; st < stages.size(); ++st) {
        auto& rt = m.stage_rt[st];
        rt.device = stages[st].device;
        const core::OnDevice on(rt.device);
        rt.prompt_bytes = prefill::Prefill::bytes_needed(g, st == 0 ? primary : *stages[st].session, c.prefill_chunk);
        if (!gpu_alloc(&rt.prompt_workspace, (size_t) rt.prompt_bytes, c.reserve_mib, err)) return false;
        if (cudaStreamCreateWithFlags(&rt.prompt_stream, cudaStreamNonBlocking) != cudaSuccess) { err = "concurrency: prompt stream failed"; return false; }
    }
    // C4: one mapped hand-off buffer per stage boundary.  Sized for the widest round: a single window is at
    // most kVerifyMaxT rows, a batch round at most c.rows (the pool's n_tok*k envelope caps a real round
    // lower; the buffer is sized for the config, the cap is enforced by scheduling - C2).
    const size_t hand_rows = std::max<size_t>((size_t) strata::kernels::kVerifyMaxT, (size_t) c.rows);
    const size_t hand_bytes = hand_rows * (size_t) core::Verifier::handoff_floats(g) * sizeof(float);
    m.hand.clear();
    for (size_t b = 0; b + 1 < stages.size(); ++b) {
        float* host = nullptr;
        if (cudaHostAlloc((void**) &host, hand_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            err = "concurrency: the layer-split hand-off allocation failed";
            return false;
        }
        std::memset(host, 0, hand_bytes);
        float* dev = nullptr;
        if (cudaHostGetDevicePointer((void**) &dev, host, 0) != cudaSuccess) {
            cudaFreeHost(host);
            err = "concurrency: the layer-split hand-off mapping failed";
            return false;
        }
        m.hand.push_back({host, dev});
    }
    return true;
}

int ConcurrentServe::run(const std::vector<ServeStage>& stages, core::ExpertSource* source,
                         core::PoolMultiFn pool, void* user, core::GpuPlanSink** stage_plans, std::string& err) {
    auto& m = *impl_;
    const auto& c = m.config;
    const auto& g = *m.geometry;
    if (stages.empty() || !stages[0].wt || !stages[0].cache || !stages[0].dispatch) {
        err = "concurrency: no stage"; return 1;
    }
    // C1 boundary: at N=1 every binding below is the very object the old run() took as a separate
    // argument - same call sequence, same objects, same order (re-root, not a rewrite).
    const ServeStage& st0 = stages[0];
    const ServeStage& stl = stages.back();   // C4: the head, the drafter and the history live on the LAST stage
    const core::WeightTable& wt = *stl.wt;   // ... and output.weight (the vocab) is in ITS weight table
    const core::NativeHead* head = stl.head;
    core::ExpertCache& cache = *st0.cache;
    int32_t* host_res = st0.host_res;
    const core::VerifyHits& hits = st0.hits;
    core::ExpertDispatch& dispatch = *st0.dispatch;
    m.stages = stages;
    m.stage_plans = stage_plans;
    if (m.stage_rt.size() != stages.size() || m.hand.size() + 1 != stages.size()) {
        err = "concurrency: the stage prompt runtimes or hand-off buffers are missing (prepare)"; return 1;
    }
    if (!head || !head->loaded() || !wt.find("output.weight")) { err = "concurrency: native head required"; return 1; }
    if (!source || !host_res || !hits.d_res || cache.slots() < 1) {
        err = "concurrency: no profile-filled expert cache fits; reduce context/concurrency or increase available VRAM";
        return 1;
    }
    struct ProgressGuard { ~ProgressGuard() { core::progress().busy.store(false); } } progress_guard;
    std::jthread watchdog; // outlives the batch graph, including teardown after a failed GPU execution
    // C4: one batch coordinator per stage (at N=1 this is today's single `batch`).  Each carries its
    // stage's range, hit view and boundary buffers, chained like the slot verifiers; the chain drives them
    // from INSIDE run_batch (batch[0] replays, then next_->run_batch) - DEPENDS-C2: inert until C2's
    // run_batch accepts a chained stage (a multi-stage batch round is refused at the round dispatch below).
    std::vector<core::Verifier> batch(m.stages.size());
    for (size_t st = 0; st < m.stages.size(); ++st) {
        const auto& sg = m.stages[st];
        const core::OnDevice on(sg.device);
        const bool last = st + 1 == m.stages.size();
        auto& b = batch[st];
        b.set_batch_cache(c.graph_cache, c.reserve_mib);
        b.set_batch_parallel(c.parallel_batch);
        b.set_stage(sg.lb, last ? -1 : sg.le, st == 0 ? nullptr : m.hand[st - 1].dev,
                    last ? nullptr : m.hand[st].dev);
        if (!b.init(*sg.wt, g, *m.slots[0]->stages[st].state, sg.hits, sg.head, std::max(2, c.rows), err, true)) return 1;
        b.set_pcie_mode(2); // Match the stock Windows-safe kernel-copy path.
        if (!last) b.set_next(&batch[st + 1], user);
    }
    // C4: per-slot per-stage chains, wired like the single-request split path (generate.cpp 3641-3730):
    // each stage's verifier runs [lb, le) and hands its rows to the next one's boundary buffer, the prompt
    // paths chain the same way, and the drafter, the history and on_chunk belong to the LAST stage.
    for (auto& ptr : m.slots) {
        auto& s = *ptr;
        for (size_t st = 0; st < m.stages.size(); ++st) {
            const auto& sg = m.stages[st];
            const core::OnDevice on(sg.device);
            const bool last = st + 1 == m.stages.size();
            s.stages[st].verify.set_stage(sg.lb, last ? -1 : sg.le,   // set_stage BEFORE init (verify.hpp:114)
                                          st == 0 ? nullptr : m.hand[st - 1].dev,
                                          last ? nullptr : m.hand[st].dev);
            if (!s.stages[st].verify.init(*sg.wt, g, *s.stages[st].state, sg.hits, sg.head, c.window, err)) return 1;
            s.stages[st].verify.set_pcie_mode(2);
            if (!last) s.stages[st].verify.set_next(&s.stages[st + 1].verify, user);   // user = &split_drive
            s.stages[st].prompt.set_stage(sg.lb, last ? -1 : sg.le, last ? nullptr : &s.stages[st + 1].prompt);
            if (!s.stages[st].prompt.init(*sg.wt, g, *s.stages[st].state, source, sg.cache, sg.host_res,
                                          c.prefill_chunk, (void*) m.stage_rt[st].prompt_stream, err,
                                          m.stage_rt[st].prompt_workspace, m.stage_rt[st].prompt_bytes)) return 1;
        }
        // The drafter reads the chain's last residual and the LAST stage's weights/head (generate.cpp:3750).
        if (!s.draft->bind(*m.stages.back().wt, m.stages.back().head, s.stages[0].verify.final_R_all(), err)) return 1;
        s.history.resize((size_t) c.window * 4096, -1);
        // C4: the last stage's sampler reads it, so it is allocated on the last stage's device.
        if (const core::OnDevice on(m.stages.back().device);
            cudaMalloc(&s.history_device, s.history.size() * sizeof(int32_t)) != cudaSuccess) {
            err = "concurrency: penalty buffer allocation failed"; return 1;
        }
        auto* slot = &s;
        // C4: on_chunk belongs to the LAST stage (prefill.hpp:104; generate.cpp:3940 moves it there in the
        // single path) - it is called when the prompt chunk has crossed every stage.
        s.stages.back().prompt.on_chunk = [slot](const float* residual, int64_t n, int64_t position, std::string& e) {
            std::vector<int32_t> next((size_t) n);
            for (int64_t j = 0; j < n; ++j) next[(size_t) j] = (int32_t) slot->request.tokens[(size_t) (position + j + 1)];
            // Use the existing draft implementation; both slots share only immutable weights.
            return slot->draft->prefill(residual, next.data(), n, position, e);
        };
    }
    size_t free = 0, total = 0;
    for (const auto& sg : m.stages) {
        const core::OnDevice on(sg.device);   // every stage must keep the configured reserve on ITS device
        cudaMemGetInfo(&free, &total);
        if (free < (size_t) c.reserve_mib * 1048576) {
            err = "concurrency: remaining VRAM is below the requested reserve; lower expert cache/context"; return 1;
        }
    }
    const bool adaptive = c.adapt_every > 0 && c.adapt_swaps > 0;
    if (adaptive) dispatch.usage.assign((size_t) g.n_layers * g.n_expert, 0.0f);
    int64_t rounds = 0;
    int64_t batch_sizes[9]{};   // one slot per member count 1..8 (pair-combine raise)
    const bool profiling = std::getenv("STRATA_CONCURRENT_PROFILE") != nullptr;
    double target_ms = 0, draft_ms = 0, commit_ms = 0, adapt_ms = 0, prefill_ms = 0;
    int64_t produced = 0, target_rows = 0;
    auto report_profile = [&]() {
        if (!profiling || !rounds) return;
        // every stage of the chain waits - a split's stage 1 included: sum them all, not stage 0's only.
        double wait = 0, pool_ms = 0, host = 0;
        for (const auto& b : batch) { wait += b.ms_wait; pool_ms += b.ms_pool; host += b.ms_host; }
        for (const auto& s : m.slots)
            for (const auto& gs : s->stages) {
                wait += gs.verify.ms_wait; pool_ms += gs.verify.ms_pool; host += gs.verify.ms_host;
            }
        std::fprintf(stderr, "strata concurrent profile: rounds=%lld tokens=%lld rows=%lld target_ms=%.1f "
                     "draft_ms=%.1f commit_ms=%.1f adapt_ms=%.1f prefill_ms=%.1f "
                     "target_wait_ms=%.1f target_pool_ms=%.1f target_host_ms=%.1f\n",
                     (long long) rounds, (long long) produced, (long long) target_rows, target_ms,
                     draft_ms, commit_ms, adapt_ms, prefill_ms, wait, pool_ms, host);
        std::fflush(stderr);
        std::fprintf(stderr, "strata concurrent detail: captures=%lld capture_ms=%.1f gpu_pre_ms=%.1f "
                     "gpu_experts_ms=%.1f gpu_post_ms=%.1f gpu_head_ms=%.1f\n",
                     (long long) batch[0].batch_captures, batch[0].ms_batch_capture, batch[0].batch_gpu_ms[0],
                     batch[0].batch_gpu_ms[1], batch[0].batch_gpu_ms[2], batch[0].batch_gpu_ms[3]);
    };
    auto adapt = [&]() -> bool {
        // All target, commit, prefill and draft work has finished at this boundary. Updating both
        // residency tables here makes cached graph pointers safe without per-request invalidation.
        struct Swap { float gain; int64_t layer; int in, out; };
        std::vector<Swap> swaps;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            std::vector<std::pair<float, int>> candidates, victims;
            for (int e = 0; e < g.n_expert; ++e) {
                const size_t i = (size_t) l * g.n_expert + e;
                const float u = dispatch.usage[i];
                if (host_res[i] >= 0) victims.emplace_back(u, e);
                else if (u >= 2.0f) candidates.emplace_back(u, e);
            }
            std::sort(candidates.rbegin(), candidates.rend());
            std::sort(victims.begin(), victims.end());
            for (size_t i = 0; i < std::min(candidates.size(), victims.size()); ++i) {
                const float gain = candidates[i].first - victims[i].first;
                if (gain < 1.5f) break;
                swaps.push_back({gain, l, candidates[i].second, victims[i].second});
            }
        }
        std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
        if (swaps.size() > (size_t) c.adapt_swaps) swaps.resize((size_t) c.adapt_swaps);
        // C4: a swap stays in its layer's own stage: the fill goes through THAT stage's cache on ITS device,
        // and the full residency table is re-uploaded to every stage it touched (the host table is one shared
        // buffer; each stage has its own device copy in its hits.d_res).
        auto stage_of = [&](int64_t l) -> size_t {
            size_t st = 0;
            while (st + 1 < m.stages.size() && l >= (m.stages[st].le < 0 ? g.n_layers : m.stages[st].le)) ++st;
            return st;
        };
        std::vector<char> touched(m.stages.size(), 0);
        for (const auto& s : swaps) {
            const size_t st = stage_of(s.layer);
            const auto& sg = m.stages[st];
            const size_t incoming = (size_t) s.layer * g.n_expert + s.in, outgoing = (size_t) s.layer * g.n_expert + s.out;
            const int slot = sg.host_res[outgoing];   // the slot lives in that stage's cache
            const auto* blob = source->blob(s.layer, s.in);
            const core::OnDevice on(sg.device);
            if (!blob || !sg.cache->fill_slot_blocking(slot, blob, err, (int64_t) kernels::cpu::expert_layout().blob_bytes(s.layer))) return false;
            sg.host_res[outgoing] = core::kNotResident; sg.host_res[incoming] = slot; touched[st] = 1;
        }
        for (size_t st = 0; st < m.stages.size(); ++st) if (touched[st]) {
            const auto& sg = m.stages[st];
            const core::OnDevice on(sg.device);
            if (cudaMemcpy((void*) sg.hits.d_res, sg.host_res, (size_t) g.n_layers * g.n_expert * sizeof(int32_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) { err = "concurrency: residency upload failed"; return false; }
        }
        for (float& usage : dispatch.usage) usage *= 0.7f;
        return true;
    };
    std::fprintf(stderr, "strata concurrent: shared expert batching; independent MTP; adaptive cache %s; no conversation-prefix reuse\n", adaptive ? "on" : "off");
    std::printf("INFO engine=" STRATA_VERSION " concurrency=%d batch_rows=%d batch_policy=%s context=%lld kv=%s lookup=%d expert_policy=%s\n",
                c.requests, c.rows, c.depth ? "depth" : "fair", (long long) c.context, c.kv.c_str(), c.suffix, adaptive ? "adaptive" : "static");
    std::printf("READY %lld stop multiplex\n", (long long) c.context);
    std::fflush(stdout);
    auto input = std::make_shared<Input>();
    std::thread([input] {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::unique_lock<std::mutex> lock(input->mutex);
            input->cv.wait(lock, [&] { return input->lines.size() < 32 || input->closed; });
            if (input->closed) return;
            const bool quit = line == "QUIT";
            input->lines.push_back(std::move(line));
            input->cv.notify_all();
            if (quit) break;
        }
        std::lock_guard<std::mutex> lock(input->mutex);
        input->eof = true;
        input->cv.notify_all();
    }).detach();
    struct InputGuard { std::shared_ptr<Input> input; ~InputGuard() { std::lock_guard<std::mutex> l(input->mutex); input->closed = true; input->cv.notify_all(); } } guard{input};
    std::deque<Request> pending;
    std::unordered_set<uint64_t> live;
    size_t rotation = 0, prompt_rotation = 0;
    const char* watchdog_env = std::getenv("STRATA_WATCHDOG_S");
    const int watchdog_seconds = watchdog_env ? std::max(0, std::atoi(watchdog_env)) : 60;
    watchdog = std::jthread([watchdog_seconds](std::stop_token stop) {
        auto& progress = core::progress();
        uint64_t last = progress.beats.load();
        auto since = Clock::now();
        while (!stop.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const auto now = Clock::now();
            const auto beat = progress.beats.load();
            if (!progress.busy.load() || beat != last) { last = beat; since = now; continue; }
            if (watchdog_seconds > 0 && now - since >= std::chrono::seconds(watchdog_seconds)) {
                std::fprintf(stderr, "strata concurrent: no progress for %d seconds (%s); stopping stalled engine\n",
                             watchdog_seconds, progress.where.load());
                if (auto diag = core::diag_verify_fn().load()) diag(stderr);
                if (auto diag = core::diag_pool_fn().load()) diag(stderr);
                std::fflush(stderr);
                std::abort();
            }
        }
    });
    auto finish = [&](Impl::Slot& s, const char* reason) {
        const double decode = s.first ? 0 : elapsed(s.decode_start);
        std::printf("R %llu DONE %lld %zu %.1f %.1f %s %lld %lld 0\n", (unsigned long long) s.request.id,
                    (long long) s.generated, s.request.tokens.size(), s.prompt_ms, decode, reason,
                    (long long) s.accepted, (long long) s.offered);
        std::fflush(stdout);
        live.erase(s.request.id);
        s.active = false;
    };
    bool quitting = false;
    while (!quitting) {
        core::progress().busy.store(!live.empty());
        std::deque<std::string> commands;
        {
            std::unique_lock<std::mutex> lock(input->mutex);
            if (live.empty() && pending.empty()) input->cv.wait(lock, [&] { return !input->lines.empty() || input->eof; });
            commands.swap(input->lines);
            if (input->eof && commands.empty()) quitting = true;
            input->cv.notify_all();
        }
        for (const auto& line : commands) {
            if (line == "QUIT") { quitting = true; break; }
            if (line.rfind("CSTOP ", 0) == 0) {
                uint64_t id = 0; std::istringstream(line.substr(6)) >> id;
                for (auto& s : m.slots) if (s->active && s->request.id == id) finish(*s, "cancel");
                for (auto p = pending.begin(); p != pending.end();) {
                    if (p->id == id) { error(id, "cancelled before admission"); live.erase(id); p = pending.erase(p); }
                    else ++p;
                }
                continue;
            }
            Request request;
            std::string reason;
            if (!parse_request(line, c, wt.find("output.weight")->ne1, request, reason)) { error(request.id, reason); continue; }
            if (live.count(request.id)) { error(request.id, "duplicate request id"); continue; }
            if (pending.size() >= 16) { error(request.id, "request queue is full"); continue; }
            live.insert(request.id);
            pending.push_back(std::move(request));
        }
        if (quitting) break;
        core::progress().busy.store(!live.empty());
        for (auto& ptr : m.slots) if (!ptr->active && !pending.empty()) {
            auto& s = *ptr;
            s.request = std::move(pending.front()); pending.pop_front();
            s.read = s.generated = s.offered = s.accepted = 0;
            s.prompt_ms = 0; s.first = true; s.active = true;
            s.position = (int64_t) s.request.tokens.size() - 1;
            s.current = (int32_t) s.request.tokens.back();
            s.consumed.clear();
            std::fill(std::begin(s.probability), std::end(s.probability), 0.0f);
            s.suffix.reset(); s.policy = spec::DraftPolicy{c.window};
            for (auto token : s.request.tokens) s.suffix.append((int32_t) token);
            for (size_t st = 0; st < m.stages.size(); ++st) {   // C4: every stage's state resets on its device
                const core::OnDevice on(m.stages[st].device);
                core::session_zero(*s.stages[st].state, g, nullptr, m.stage_rt[st].prompt_stream);
            }
            for (size_t st = 0; st < m.stage_rt.size(); ++st) {
                core::progress_at("concurrency: resetting a stage", (int64_t) st);   // the watchdog's view
                if (cudaStreamSynchronize(m.stage_rt[st].prompt_stream) != cudaSuccess) { err = "concurrency: reset failed"; return 1; }
            }
            s.draft->reset(); s.draft->set_prompt_len((int64_t) s.request.tokens.size());
            s.stages[0].verify.set_sampling(s.request.sampling);
        }
        // At most ONE bounded prompt chunk before returning to ready decoders.
        core::progress().busy.store(!live.empty());
        for (size_t j = 0; j < m.slots.size(); ++j) {
            const size_t i = (prompt_rotation + j) % m.slots.size();
            auto& s = *m.slots[i];
            if (!s.active || s.read >= s.position) continue;
            const auto start = Clock::now();
            const auto n = std::min<int64_t>(c.prefill_chunk, s.position - s.read);
            if (!s.stages[0].prompt.run(s.request.tokens.data() + s.read, n, s.read, err)) return 1;
            for (int64_t t = 0; t < n; ++t) s.consumed.push_back((int32_t) s.request.tokens[(size_t) (s.read + t)]);
            s.read += n; s.prompt_ms += elapsed(start);
            prefill_ms += elapsed(start);
            std::printf("R %llu PP %lld %zu\n", (unsigned long long) s.request.id, (long long) s.read, s.request.tokens.size());
            std::fflush(stdout);
            prompt_rotation = (i + 1) % m.slots.size();
            break;
        }
        std::vector<Impl::Slot*> ready;
        std::vector<int> wanted;
        for (size_t j = 0; j < m.slots.size(); ++j) {
            auto& s = *m.slots[(rotation + j) % m.slots.size()];
            if (!s.active || s.read < s.position) continue;
            if (s.position >= c.context) { finish(s, "length"); continue; }
            int n = s.first ? 1 : c.mtp_window_rows;
            if (!s.first && s.request.spec_min_p > 0) {
                n = 1;
                while (n < c.mtp_window_rows && s.probability[n - 1] >= s.request.spec_min_p) ++n;
            }
            s.lookup = false; s.match = 0;
            if (!s.first && c.suffix > 0) {
                const int k = s.suffix.propose(c.window - 1, s.lookup_tokens);
                s.match = s.suffix.last_match();
                if (k > 0 && s.lookup_tokens[0] == s.drafts[0]) {
                    const auto pick = s.policy.choose(n, k, s.match);
                    if (pick.lookup) { n = pick.t; s.lookup = true; }
                }
            }
            n = (int) std::min<int64_t>(n, std::min(c.context - s.position, s.request.max_new - s.generated));
            if (n < 1) { finish(s, "length"); continue; }
            ready.push_back(&s); wanted.push_back(n);
        }
        const auto allocation = schedule_rows(wanted, c.rows, c.depth);
        std::vector<core::Verifier::BatchWindow> windows;
        std::vector<Impl::Slot*> win_slots;   // C4: parallel to `windows`; a window's chain head is its slot's stage 0
        for (size_t i = 0; i < ready.size(); ++i) {
            auto& s = *ready[i]; s.count = allocation[i];
            if (!s.count) continue;
            if (s.first) s.decode_start = Clock::now();
            s.window[0] = s.current;
            for (int t = 1; t < s.count; ++t) s.window[t] = s.lookup ? s.lookup_tokens[t - 1] : s.drafts[t - 1];
            // Outputs beyond the real window are never emitted, accepted or committed.
            // Fixed shapes can amortize graph construction without asking MTP for more drafts.
            for (int t = s.count; t < c.window; ++t) s.window[t] = s.current;
            const int history = s.request.sampling.penalty_last_n;
            if (history > 0) {
                kernels::penalty_rows(s.consumed.data(), (int64_t) s.consumed.size(), s.window, s.count, history, s.history.data());
                if (cudaMemcpy(s.history_device, s.history.data(), (size_t) s.count * history * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "concurrency: history upload failed"; return 1;
                }
            }
            s.stages[0].verify.set_history(history ? s.history_device : nullptr, history);
            windows.push_back({&s.stages[0].verify, s.count, s.window, s.position, s.output});
            win_slots.push_back(&s);
        }
        if (!windows.empty()) {
            ++batch_sizes[windows.size()];
            if (c.pad_batch && windows.size() > 1) {
                int padded_rows = 0;
                for (const auto& w : windows)
                    padded_rows += (int) std::min<int64_t>(std::max(w.count, c.mtp_window_rows), c.context - w.position);
                if (padded_rows <= c.rows)
                    for (auto& w : windows)
                        w.count = (int) std::min<int64_t>(std::max(w.count, c.mtp_window_rows), c.context - w.position);
            }
            // Stable packing order avoids recapturing a graph merely because fairness rotated the request order.
            std::sort(windows.begin(), windows.end(), [](const auto& a, const auto& b) {
                return std::less<core::Verifier*>{}(a.verifier, b.verifier);
            });
            const auto start = Clock::now();
            dispatch.failed = false;
            // C3/C4 (design-wiring §5): refresh EVERY stage's publish target for THIS round before it runs.
            // A stale sink is not a hang: the empty-plan fallback silently drops the resident experts'
            // contribution (verify.cpp:1073-1081), and round parity is the only detector.  A single-window
            // round publishes through the window's slot chain, a batch round through its stage coordinators.
            for (size_t st = 0; st < m.stages.size(); ++st)
                if (stage_plans)
                    stage_plans[st] = windows.size() == 1 ? win_slots.front()->stages[st].verify.plan_sink()
                                                          : batch[st].plan_sink();
            dispatch.plan = windows.size() == 1 ? win_slots.front()->stages[0].verify.plan_sink()
                                                : batch[0].plan_sink();
            // N1 (C2 acceptance kit, env-gated, default off): the stale-sink negative control.  Point stage
            // `st`'s publish target at ANOTHER stage's sink - a verifier that is not executing this round -
            // so the empty-plan fallback fires and the round completes with silently wrong tokens, no hang.
            if (const char* v = std::getenv("STRATA_PLAN_SINK_STALE");
                v != nullptr && stage_plans != nullptr && m.stages.size() > 1) {
                const int stale = std::abs(std::atoi(v)) % (int) m.stages.size();
                stage_plans[stale] = stage_plans[(stale + 1) % (int) m.stages.size()];
            }
            bool ok;
            if (windows.size() == 1) {
                const auto& w = windows.front();
                ok = win_slots.front()->stages[0].verify.run(w.count, w.tokens, w.position, pool, user, w.output, err);
            } else {
                // C2 landed: run_batch chains through the stages itself (A1 contiguity, A5 continuation).
                // The coordinators are chained and share the per-boundary hand buffers (C4 + verify.cpp A2).
                ok = batch[0].run_batch(windows, pool, user, err);
            }
            if (!ok || dispatch.failed) {
                if (dispatch.failed) err = dispatch.fail ? dispatch.fail : "expert dispatch failed";
                return 1;
            }
            target_ms += elapsed(start);
            for (const auto& w : windows) target_rows += w.count;
            for (auto* ptr : ready) {
                auto& s = *ptr; if (!s.count) continue;
                int accepted = 0;
                while (accepted < s.count - 1 && s.window[accepted + 1] == s.output[accepted]) ++accepted;
                int keep = accepted + 1;
                bool eos = false;
                for (int t = 0; t < keep; ++t) if (std::find(c.eos.begin(), c.eos.end(), s.output[t]) != c.eos.end()) {
                    keep = t + 1; eos = true; break;
                }
                const auto commit_start = Clock::now();
                if (!s.stages[0].verify.commit(keep, err)) return 1;
                commit_ms += elapsed(commit_start);
                produced += keep;
                s.offered += s.count - 1; s.accepted += keep - 1;
                for (int t = 0; t < keep; ++t) {
                    s.consumed.push_back(s.window[t]); s.suffix.append(s.output[t]); ++s.generated;
                    std::printf("R %llu T %d\n", (unsigned long long) s.request.id, s.output[t]);
                }
                std::fflush(stdout);
                s.first = false;
                if (eos || s.generated >= s.request.max_new || s.position + keep >= c.context) {
                    finish(s, eos ? "stop" : "length"); continue;
                }
                // Catch-up consumes the verified window; limit the extra speculative chain near the context boundary.
                s.draft->set_max_drafts((int) std::min<int64_t>(c.mtp_window_rows - 1, c.context - (s.position + keep)));
                const auto draft_start = Clock::now();
                if (!s.draft->draft(s.count, s.output, s.position, keep - 1, s.drafts, err, s.probability, s.request.spec_min_p)) return 1;
                draft_ms += elapsed(draft_start);
                s.policy.observe(s.lookup, s.count, keep - 1, s.match, elapsed(start));
                s.current = s.output[keep - 1]; s.position += keep; s.read = s.position;
            }
            ++rounds;
            if (adaptive && rounds % c.adapt_every == 0) {
                const auto adapt_start = Clock::now();
                if (!adapt()) return 1;
                adapt_ms += elapsed(adapt_start);
            }
            if (rounds % 64 == 0) report_profile();
        }
        rotation = (rotation + 1) % m.slots.size();
    }
    for (auto& s : m.slots) if (s->active) finish(*s, "cancel");
    for (const auto& r : pending) error(r.id, "server shutting down");
    report_profile();
    std::fprintf(stderr, "strata concurrent: target rounds by active batch size: 1=%lld 2=%lld 3=%lld 4=%lld 5=%lld 6=%lld 7=%lld 8=%lld\n",
                 (long long) batch_sizes[1], (long long) batch_sizes[2], (long long) batch_sizes[3], (long long) batch_sizes[4],
                 (long long) batch_sizes[5], (long long) batch_sizes[6], (long long) batch_sizes[7], (long long) batch_sizes[8]);
    return 0;
}
}
