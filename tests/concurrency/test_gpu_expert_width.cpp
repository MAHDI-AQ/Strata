// Exact GPU row isolation across expert group widths, using real GGUF weights.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>

static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) return 2;
    const bool bench = argc == 3 && std::strcmp(argv[2], "--bench") == 0;
    strata::GgufFile file(argv[1]);
    constexpr int rows = 16, entries = rows * 10, H = 2560, F = 640, expert = 7;
    int failures = 0;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::set<std::pair<int, int>> tested;
    for (int layer = 0; layer < 48; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".ffn_";
        auto* g = file.find(prefix + "gate_exps.weight");
        auto* u = file.find(prefix + "up_exps.weight");
        auto* d = file.find(prefix + "down_exps.weight");
        if (!g || !u || !d) continue;
        if (!tested.insert({(int) g->type, (int) d->type}).second) continue;
        auto layout = strata::kernels::native_expert_layout((int) g->type, (int) d->type, H, F);
        std::vector<unsigned char> blob(layout.bytes);
        std::memcpy(blob.data(), file.tensor_data(*g) + expert * layout.up_off, layout.up_off);
        std::memcpy(blob.data() + layout.up_off, file.tensor_data(*u) + expert * layout.up_off, layout.up_off);
        std::memcpy(blob.data() + layout.down_off, file.tensor_data(*d) + expert * (layout.bytes - layout.down_off), layout.bytes - layout.down_off);
        std::vector<float> input(rows * H), output(entries * H), reference;
        int32_t indices[entries], token_rows[entries];
        for (int i = 0; i < entries; ++i) {
            indices[i] = entries - i - 1;
            token_rows[i] = indices[i] % rows;
        }
        for (int t = 0; t < rows; ++t) {
            for (int i = 0; i < H; ++i) input[t * H + i] = std::sin(float(i * 13 + t * 31 + 1) * 0.013f);
        }
        void *weights, *xq, *scratch;
        float *x, *out;
        int32_t *starts, *count, *index, *tokens;
        unsigned long long* pointers;
        check(cudaMalloc(&weights, blob.size()));
        check(cudaMalloc(&xq, rows * H / 32 * 36));
        const auto scratch_bytes = strata::kernels::native_expert_scratch_bytes(entries, F);
        check(cudaMalloc(&scratch, scratch_bytes));
        check(cudaMemset(scratch, 0, scratch_bytes));
        check(cudaMalloc(&x, input.size() * 4)); check(cudaMalloc(&out, output.size() * 4));
        check(cudaMalloc(&starts, 8)); check(cudaMalloc(&count, 4));
        check(cudaMalloc(&index, entries * 4)); check(cudaMalloc(&tokens, entries * 4)); check(cudaMalloc(&pointers, 8));
        const unsigned long long pointer = (unsigned long long) weights;
        const int32_t one = 1;
        check(cudaMemcpy(weights, blob.data(), blob.size(), cudaMemcpyHostToDevice));
        check(cudaMemcpy(x, input.data(), input.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(pointers, &pointer, 8, cudaMemcpyHostToDevice));
        check(cudaMemcpy(count, &one, 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(index, indices, entries * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(tokens, token_rows, entries * 4, cudaMemcpyHostToDevice));
        strata::kernels::quantize_q8_1_rows(x, rows, H, xq, stream);
        check(cudaStreamSynchronize(stream));
        for (int width : {1, 2, 3, 4, 8, 9, 12, 16}) {
            check(cudaMemset(out, 0xFF, output.size() * 4));
            for (int start = 0; start < entries; start += width) {
                const int32_t bounds[2] = {start, (std::min)(entries, start + width)};
                check(cudaMemcpy(starts, bounds, 8, cudaMemcpyHostToDevice));
                strata::kernels::native_expert_grouped(layout, pointers, starts, count, index, tokens,
                                                      1, entries, xq, scratch, out, stream);
                check(cudaStreamSynchronize(stream));
            }
            check(cudaMemcpy(output.data(), out, output.size() * 4, cudaMemcpyDeviceToHost));
            if (width == 1) reference = output;
            int different = 0;
            for (size_t i = 0; i < output.size(); ++i)
                different += !std::isfinite(output[i]) || std::memcmp(&output[i], &reference[i], 4) != 0;
            std::printf("GPU layer %d formats %d/%d width %d entries %d: %d differing FP32 cells\n", layer, (int) g->type, (int) d->type, width, entries, different);
            failures += different != 0;
            if (bench) {
                const int32_t bounds[2] = {0, width};
                check(cudaMemcpy(starts, bounds, 8, cudaMemcpyHostToDevice));
                cudaGraph_t graph;
                cudaGraphExec_t executable;
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                strata::kernels::native_expert_grouped(layout, pointers, starts, count, index, tokens,
                                                      1, width, xq, scratch, out, stream);
                check(cudaStreamEndCapture(stream, &graph));
                check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                for (int i = 0; i < 10; ++i) check(cudaGraphLaunch(executable, stream));
                cudaEvent_t begin, end;
                check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
                check(cudaEventRecord(begin, stream));
                for (int i = 0; i < 200; ++i) check(cudaGraphLaunch(executable, stream));
                check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
                float ms = 0;
                check(cudaEventElapsedTime(&ms, begin, end));
                std::printf("GPU micro formats %d/%d group %d: %.3f us/expert\n", (int) g->type, (int) d->type, width, ms * 5);
                check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
                check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
            }
        }
        for (void* p : {weights, xq, scratch, (void*) x, (void*) out, (void*) starts,
                        (void*) count, (void*) index, (void*) tokens, (void*) pointers}) check(cudaFree(p));
    }
    check(cudaStreamDestroy(stream));
    return failures || tested.empty() ? 1 : 0;
}
