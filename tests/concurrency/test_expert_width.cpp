// Real-weight CPU regression: changing an expert's token group must not change its arithmetic.
// This maps only a few expert tensors; it does not load the model or allocate GPU state.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: strata-expert-width-test <GGUF shard>\n"); return 2; }
    namespace cpu = strata::kernels::cpu;
    strata::GgufFile file(argv[1]);
    constexpr int rows = 8, H = 2560, F = 640, expert = 7;
    int failures = 0;
    for (int layer : {0, 1, 2}) {
        const auto prefix = "blk." + std::to_string(layer) + ".ffn_";
        auto* g = file.find(prefix + "gate_exps.weight");
        auto* u = file.find(prefix + "up_exps.weight");
        auto* d = file.find(prefix + "down_exps.weight");
        if (!g || !u || !d) { std::fprintf(stderr, "missing layer %d\n", layer); return 2; }
        cpu::NativeFmt fmt;
        std::string err;
        if (!cpu::native_fmt((int)g->type, (int)d->type, H, F, fmt, err)) {
            std::fprintf(stderr, "%s\n", err.c_str()); return 2;
        }
        std::vector<unsigned char> blob(fmt.bytes);
        std::memcpy(blob.data(), file.tensor_data(*g) + expert * fmt.up_off, fmt.up_off);
        std::memcpy(blob.data() + fmt.up_off, file.tensor_data(*u) + expert * fmt.up_off, fmt.up_off);
        std::memcpy(blob.data() + fmt.down_off, file.tensor_data(*d) + expert * (fmt.bytes - fmt.down_off), fmt.bytes - fmt.down_off);
        std::vector<std::vector<unsigned char>> aq(rows, std::vector<unsigned char>(cpu::kNativeActBytes));
        std::vector<std::vector<unsigned char>> hq(rows, std::vector<unsigned char>(cpu::kNativeHBytes));
        std::vector<float> input(rows * H), ff(rows * F), output(rows * H), reference;
        const void* ap[rows]; const void* hp[rows]; float* fp[rows]; float* op[rows];
        for (int t = 0; t < rows; ++t) {
            for (int i = 0; i < H; ++i) input[t * H + i] = std::sin(float(i * 13 + t * 31 + 1) * 0.013f);
            cpu::native_quant_act(fmt, input.data() + t * H, aq[t].data());
            ap[t] = aq[t].data(); hp[t] = hq[t].data();
            fp[t] = ff.data() + t * F; op[t] = output.data() + t * H;
        }
        for (int width : {1, 2, 3, 4, 8}) {
            for (int start = 0; start < rows; start += width) {
                const int n = (std::min)(width, rows - start);
                cpu::native_gu_rows(fmt, blob.data(), ap + start, n, fp + start, 0, F);
                for (int t = start; t < start + n; ++t) cpu::native_quant_h(fmt, fp[t], hq[t].data());
                cpu::native_down_rows(fmt, blob.data(), hp + start, n, op + start, 0, H);
            }
            if (width == 1) reference = output;
            int different = 0;
            for (size_t i = 0; i < output.size(); ++i)
                different += std::memcmp(&output[i], &reference[i], sizeof(float)) != 0;
            std::printf("layer %d width %d: %d differing FP32 cells\n", layer, width, different);
            failures += different != 0;
        }
    }
    return failures ? 1 : 0;
}
