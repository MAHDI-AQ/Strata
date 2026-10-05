// include/strata/core/hydration_queue.hpp - Predictive Asynchronous Prefix Pre-Hydration Queue
//
// Cross-Engine Attribution and Research Lineage:
//   - LMDeploy: "Predictive Prefix Preloading for High-Concurrency Swarms"
//   - SGLang: "Async Disk-to-Host Pipeline Streaming"
//
#pragma once

#include "strata/core/radix_tree.hpp"
#include "strata/core/nvme_tier.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

struct HydrationRequest {
    std::string session_id;
    int64_t node_id = 0;
    std::shared_ptr<RadixNode> node;
    std::chrono::steady_clock::time_point enqueued_at;
    bool completed = false;
};

class HydrationQueue {
public:
    explicit HydrationQueue(std::shared_ptr<RadixTree> tree, std::shared_ptr<NVMeStorageTier> nvme)
        : tree_(std::move(tree)), nvme_(std::move(nvme)) {
        start();
    }

    ~HydrationQueue() {
        stop();
    }

    void start() {
        if (running_.exchange(true)) return;
        worker_thread_ = std::thread(&HydrationQueue::worker_loop, this);
    }

    void stop() {
        if (!running_.exchange(false)) return;
        cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    /// Enqueue a node for background asynchronous pre-hydration from NVMe L3 into Host L2.
    void enqueue_hydration(const std::shared_ptr<RadixNode>& node, const std::string& session_id = "") {
        if (!node) return;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            HydrationRequest req;
            req.session_id = session_id;
            req.node_id = node->id;
            req.node = node;
            req.enqueued_at = std::chrono::steady_clock::now();
            queue_.push_back(std::move(req));
        }
        cv_.notify_one();
    }

    size_t pending_requests() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return queue_.size();
    }

    uint64_t total_hydrated() const {
        return total_hydrated_.load(std::memory_order_relaxed);
    }

    double last_hydration_latency_ms() const {
        return last_latency_ms_.load(std::memory_order_relaxed);
    }

private:
    std::shared_ptr<RadixTree> tree_;
    std::shared_ptr<NVMeStorageTier> nvme_;
    std::atomic<bool> running_{false};
    std::thread worker_thread_;

    mutable std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::deque<HydrationRequest> queue_;

    std::atomic<uint64_t> total_hydrated_{0};
    std::atomic<double> last_latency_ms_{0.0};

    void worker_loop() {
        while (running_.load(std::memory_order_relaxed)) {
            HydrationRequest req;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] {
                    return !running_.load(std::memory_order_relaxed) || !queue_.empty();
                });
                if (!running_.load(std::memory_order_relaxed) && queue_.empty()) {
                    break;
                }
                if (!queue_.empty()) {
                    req = std::move(queue_.front());
                    queue_.pop_front();
                }
            }

            if (req.node && nvme_) {
                auto t0 = std::chrono::high_resolution_clock::now();
                // Hydrate node from NVMe L3 to Host L2
                if (req.node->is_l3_offloaded) {
                    req.node->hydrate_from_nvme(*nvme_);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                last_latency_ms_.store(ms, std::memory_order_relaxed);
                total_hydrated_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
};

}  // namespace strata::core
