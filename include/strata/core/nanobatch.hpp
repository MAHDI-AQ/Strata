// include/strata/core/nanobatch.hpp - NanoFlow-Style Device-Level Nanobatching & Overlapped PCIe Pipeline
//
// Cross-Engine Reference: NanoFlow (arXiv:2408.12757), DeepSeek-AI Pipeline Parallelism, Megatron-LM.
// Partitions active batch requests into K in [2..4] fine-grained micro-slices (nanobatches)
// to formulate a 3-stage temporal execution pipeline:
//   1. Compute Stage 0 (GPU 0): layer slice for nanobatch N + 1
//   2. Transfer Stage (D2D PCIe): asynchronous CUDA DMA stream transfers activations R of nanobatch N
//   3. Compute Stage 1 (GPU 1): layer slice for nanobatch N - 1
// 100% hides cross-GPU PCIe Gen4 x8 transfer latency behind intra-GPU layer compute.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// A partitioned nanobatch micro-slice representing a subset of the active batch windows.
struct NanobatchSlice {
    int slice_idx = 0;              ///< Index of this nanobatch slice (0 .. K-1)
    size_t window_offset = 0;       ///< First window index in batch
    size_t window_count = 0;        ///< Number of windows in this slice
    int row_offset = 0;             ///< Offset into batch flattened rows [0, total_rows)
    int row_count = 0;              ///< Number of rows in this slice
    size_t byte_offset = 0;         ///< Byte offset in cross-device transfer buffer
    size_t byte_count = 0;          ///< Byte count in cross-device transfer buffer
};

/// 3-stage temporal execution pipeline step
struct PipelineStep {
    int step = 0;
    int stage0_slice = -1;      ///< Nanobatch index running compute Stage 0 (-1 if none)
    int transfer_slice = -1;    ///< Nanobatch index in-flight over DMA (-1 if none)
    int stage1_slice = -1;      ///< Nanobatch index running compute Stage 1 (-1 if none)
};

/// NanobatchScheduler partitions batch requests into fine-grained micro-slices (K in [2..4])
/// and coordinates the 3-stage execution pipeline across dual GPUs.
class NanobatchScheduler {
public:
    explicit NanobatchScheduler(int K = 2) : K_(std::clamp(K, 2, 4)) {}

    void set_K(int K) {
        K_ = std::clamp(K, 2, 4);
    }

    int K() const { return K_; }

    /// Partitions a batch of window row counts into K slices.
    /// Distributes rows as evenly as possible across slices.
    std::vector<NanobatchSlice> partition(const std::vector<int>& window_row_counts, size_t row_bytes) const {
        if (window_row_counts.empty()) return {};

        int total_rows = 0;
        for (int r : window_row_counts) total_rows += r;
        if (total_rows <= 0) return {};

        const size_t total_windows = window_row_counts.size();
        const int effective_k = (total_windows < (size_t) K_) ? (int) total_windows : K_;

        std::vector<NanobatchSlice> slices;
        slices.reserve(effective_k);

        int target_rows_per_slice = (total_rows + effective_k - 1) / effective_k;
        size_t win_idx = 0;
        int current_row_off = 0;

        for (int k = 0; k < effective_k; ++k) {
            if (win_idx >= total_windows) break;

            NanobatchSlice sl;
            sl.slice_idx = k;
            sl.window_offset = win_idx;
            sl.row_offset = current_row_off;

            int accumulated_rows = 0;
            size_t w_count = 0;

            while (win_idx < total_windows) {
                if (k == effective_k - 1) {
                    accumulated_rows += window_row_counts[win_idx];
                    win_idx++;
                    w_count++;
                    continue;
                }

                if (w_count > 0 && (accumulated_rows + window_row_counts[win_idx] > target_rows_per_slice)) {
                    break;
                }

                accumulated_rows += window_row_counts[win_idx];
                win_idx++;
                w_count++;
            }

            sl.window_count = w_count;
            sl.row_count = accumulated_rows;
            sl.byte_offset = (size_t) sl.row_offset * row_bytes;
            sl.byte_count = (size_t) sl.row_count * row_bytes;

            current_row_off += accumulated_rows;
            slices.push_back(sl);
        }

        return slices;
    }

    /// Computes the sequence of temporal pipeline steps for a batch of num_slices nanobatches.
    /// Total steps = num_slices + 2.
    static std::vector<PipelineStep> generate_schedule(int num_slices) {
        if (num_slices <= 0) return {};
        const int total_steps = num_slices + 2;
        std::vector<PipelineStep> steps;
        steps.reserve(total_steps);

        for (int s = 0; s < total_steps; ++s) {
            PipelineStep step;
            step.step = s;
            step.stage0_slice   = (s < num_slices) ? s : -1;
            step.transfer_slice = (s >= 1 && s - 1 < num_slices) ? (s - 1) : -1;
            step.stage1_slice   = (s >= 2 && s - 2 < num_slices) ? (s - 2) : -1;
            steps.push_back(step);
        }
        return steps;
    }

private:
    int K_ = 2;
};

}  // namespace strata::core
