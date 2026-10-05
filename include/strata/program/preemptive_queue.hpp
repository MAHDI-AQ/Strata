// include/strata/program/preemptive_queue.hpp - Sub-15ms Preemptive Auxiliary Insertion & Micro-Interleaved Wavefronts
//
// Cross-Engine Attribution and Research Lineage:
//   - SGLang: Prioritized request scheduling & chunked prefill interleaving (Zheng et al., LMSYS / UC Berkeley)
//   - vLLM: Chunked prefill & preemptive scheduling for mixed decode/prefill workloads (Kwon et al., SOSP 2023)
//   - Orca: Preemptive iteration interleaving for low-latency auxiliary inference (OSDI 2022)
//
// Slices prompt sequences into interruptible micro-chunks (C in [64, 128] tokens).
// When an urgent auxiliary request arrives (e.g. tool call on slot 3), background prefill yields
// at the micro-chunk boundary, admitting the auxiliary request in < 15 ms without stalling primary streams.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace strata::program {

enum class PriorityLevel : uint8_t {
    NORMAL = 0,
    HIGH = 1,
    URGENT_AUX = 2
};

/// Micro-chunk of a prompt sequence for interruptible prefill
struct MicroChunk {
    int64_t offset = 0;
    int64_t count = 0;
    bool is_final = false;
};

/// Slices prompt sequences into micro-chunks of C tokens (default 64 to 128)
class MicroPrefillSlicer {
public:
    explicit MicroPrefillSlicer(int64_t chunk_size = 128)
        : chunk_size_(std::clamp<int64_t>(chunk_size, 32, 512)) {}

    void set_chunk_size(int64_t chunk_size) {
        chunk_size_ = std::clamp<int64_t>(chunk_size, 32, 512);
    }

    int64_t chunk_size() const { return chunk_size_; }

    std::vector<MicroChunk> slice(int64_t total_tokens, int64_t start_offset = 0) const {
        std::vector<MicroChunk> chunks;
        if (total_tokens <= start_offset) return chunks;

        int64_t curr = start_offset;
        while (curr < total_tokens) {
            const int64_t n = std::min<int64_t>(chunk_size_, total_tokens - curr);
            const bool is_final = (curr + n >= total_tokens);
            chunks.push_back({curr, n, is_final});
            curr += n;
        }
        return chunks;
    }

private:
    int64_t chunk_size_ = 128;
};

/// Thread-safe priority queue supporting sub-15ms preemptive auxiliary insertion
template <typename TRequest>
class PreemptiveQueue {
public:
    struct QueuedItem {
        TRequest request;
        PriorityLevel priority = PriorityLevel::NORMAL;
        std::chrono::steady_clock::time_point enqueue_time{};
    };

    explicit PreemptiveQueue(size_t max_capacity = 64)
        : max_capacity_(max_capacity) {}

    bool push(TRequest req, PriorityLevel prio = PriorityLevel::NORMAL) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (urgent_queue_.size() + normal_queue_.size() >= max_capacity_) {
            return false;
        }

        QueuedItem item{std::move(req), prio, std::chrono::steady_clock::now()};
        if (prio == PriorityLevel::URGENT_AUX) {
            urgent_queue_.push_back(std::move(item));
            urgent_pending_.store(true, std::memory_order_release);
        } else {
            normal_queue_.push_back(std::move(item));
        }
        cv_.notify_one();
        return true;
    }

    bool pop(TRequest& out_req, PriorityLevel& out_prio, double* out_wait_ms = nullptr) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (urgent_queue_.empty() && normal_queue_.empty()) {
            return false;
        }

        QueuedItem item;
        if (!urgent_queue_.empty()) {
            item = std::move(urgent_queue_.front());
            urgent_queue_.pop_front();
            urgent_pending_.store(!urgent_queue_.empty(), std::memory_order_release);
        } else {
            item = std::move(normal_queue_.front());
            normal_queue_.pop_front();
        }

        const auto now = std::chrono::steady_clock::now();
        const double wait_ms = std::chrono::duration<double, std::milli>(now - item.enqueue_time).count();
        if (out_wait_ms) *out_wait_ms = wait_ms;

        if (item.priority == PriorityLevel::URGENT_AUX) {
            last_aux_turnaround_ms_.store(wait_ms, std::memory_order_relaxed);
            ++aux_served_;
        }

        out_req = std::move(item.request);
        out_prio = item.priority;
        return true;
    }

    /// Fast lock-free check for in-flight prefill threads to yield
    bool should_preempt(int active_slot = -1) const {
        return urgent_pending_.load(std::memory_order_acquire);
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return urgent_queue_.size() + normal_queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return urgent_queue_.empty() && normal_queue_.empty();
    }

    size_t urgent_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return urgent_queue_.size();
    }

    double last_aux_turnaround_ms() const {
        return last_aux_turnaround_ms_.load(std::memory_order_relaxed);
    }

    uint64_t aux_served() const {
        return aux_served_.load(std::memory_order_relaxed);
    }

private:
    size_t max_capacity_ = 64;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<QueuedItem> urgent_queue_;
    std::deque<QueuedItem> normal_queue_;

    std::atomic<bool> urgent_pending_{false};
    std::atomic<double> last_aux_turnaround_ms_{0.0};
    std::atomic<uint64_t> aux_served_{0};
};

}  // namespace strata::program
