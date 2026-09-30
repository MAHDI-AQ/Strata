// Real-weight CPU regression: changing an expert's token group must not change its arithmetic.
// This maps only a few expert tensors; it does not load the model or allocate GPU state.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <set>

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: strata-expert-width-test <GGUF shard>\n"); return 2; }
    namespace cpu = strata::kernels::cpu;
    strata::GgufFile file(argv[1]);
    constexpr int rows = 16, H = 2560, F = 640, expert = 7;
    int failures = 0;
    cpu::ExpertPool pool(2, false, true);
    std::set<std::pair<int, int>> tested;
    for (int layer = 0; layer < 48; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".ffn_";
        auto* g = file.find(prefix + "gate_exps.weight");
        auto* u = file.find(prefix + "up_exps.weight");
        auto* d = file.find(prefix + "down_exps.weight");
        if (!g || !u || !d) continue; // tensors may be in another shard
        if (!tested.insert({(int) g->type, (int) d->type}).second) continue;
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
        std::vector<float> input(rows * H), output(rows * H), reference;
        std::vector<cpu::ActQ> q2(rows);
        const void* ap[rows]; float* op[rows];
        for (int t = 0; t < rows; ++t) {
            for (int i = 0; i < H; ++i) input[t * H + i] = std::sin(float(i * 13 + t * 31 + 1) * 0.013f);
            cpu::native_quant_act(fmt, input.data() + t * H, aq[t].data());
            if (fmt.gu_type == 42) cpu::act_quant_any(input.data() + t * H, H, q2[t]);
            ap[t] = aq[t].data(); op[t] = output.data() + t * H;
        }
        for (int width : {1, 2, 3, 4, 8, 9, 12, 16}) {
            for (int start = 0; start < rows; start += width) {
                const int n = (std::min)(width, rows - start);
                cpu::ExpertJobMulti job;
                job.blob = blob.data(); job.nt = n;
                for (int t = 0; t < n; ++t) {
                    job.nact[t] = ap[start + t]; job.act[t] = &q2[start + t]; job.out[t] = op[start + t];
                }
                pool.run_split_multi_native(fmt, &job, 1);
            }
            if (width == 1) reference = output;
            int different = 0;
            for (size_t i = 0; i < output.size(); ++i)
                different += std::memcmp(&output[i], &reference[i], sizeof(float)) != 0;
            std::printf("layer %d formats %d/%d width %d: %d differing FP32 cells\n", layer, fmt.gu_type, fmt.d_type, width, different);
            failures += different != 0;
        }
    }
    return failures || tested.empty() ? 1 : 0;
}
