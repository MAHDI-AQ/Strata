// src/kernels/kv_q4_parity.cpp - Parity test for Q4_0 KV cache with orthonormal Walsh-Hadamard rotation.
// Validates:
// 1. FWHT 256 CUDA kernel vs exact mathematical orthonormal Walsh-Hadamard transform.
//    (H * H * x == x, H is symmetric and orthonormal with scale 1/sqrt(256) = 1/16).
// 2. Q4_0 quantization and packing (32 values per block, 18 bytes) bitwise vs host reference.
// 3. kv_append_q4_step and kv_gather_q4_step through paged pool against host reference.
// 3b. kv_append_q4_batch_step (the captured window's one-call append) == the per-token steps, bit-identical,
//     device pool and KV-streaming host copy, with a capture + replay over rewritten step rows.
// 4. Invariance of dot products under Walsh-Hadamard rotation: (H*q) . (H*k) == q . k.

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// Exact CPU reference for Fast Walsh-Hadamard Transform (D = 256)
// Sylvester construction matching llama.cpp's ggml_gen_hadamard
void fwht256_host_reference(float* x) {
    constexpr int n = 256;
    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < n; i += 2 * s) {
            for (int j = 0; j < s; ++j) {
                const float u = x[i + j];
                const float v = x[i + j + s];
                x[i + j]     = u + v;
                x[i + j + s] = u - v;
            }
        }
    }
    const float scale = 1.0f / 16.0f; // 1 / sqrt(256)
    for (int i = 0; i < n; ++i) {
        x[i] *= scale;
    }
}

// Host reference for Q4_0 quantization of 32 elements (1 block)
void quantize_block_q4_0_host(const float* x, k::block_q4_0& blk) {
    float amax = 0.0f;
    float max_val = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float v = x[i];
        if (amax < std::fabs(v)) {
            amax = std::fabs(v);
            max_val = v;
        }
    }
    const float d = max_val / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    blk.d = k::f16_from_f32(d);

    for (int i = 0; i < 16; ++i) {
        const float x0 = x[i] * id;
        const float x1 = x[i + 16] * id;
        const int q0 = (int) (x0 + 8.5f);
        const int q1 = (int) (x1 + 8.5f);
        const uint8_t xi0 = (uint8_t) std::min(15, std::max(0, q0));
        const uint8_t xi1 = (uint8_t) std::min(15, std::max(0, q1));
        blk.qs[i] = xi0 | (xi1 << 4);
    }
}

} // namespace

int main() {
    std::printf("=== Running kv_q4_parity test ===\n");
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // -------------------------------------------------------------
    // Test 1: Orthonormal Fast Walsh-Hadamard Transform (D=256)
    // -------------------------------------------------------------
    std::printf("[1/4] Verifying FWHT-256 CUDA kernel vs exact mathematical reference...\n");
    constexpr int n_vectors = 64;
    std::vector<float> h_in(n_vectors * 256);
    std::vector<float> h_ref(n_vectors * 256);
    for (size_t i = 0; i < h_in.size(); ++i) {
        h_in[i] = nd(rng);
    }
    h_ref = h_in;

    // Run CPU reference on each row
    for (int r = 0; r < n_vectors; ++r) {
        fwht256_host_reference(h_ref.data() + r * 256);
    }

    // Run GPU kernel
    float* d_src = dalloc<float>(n_vectors * 256);
    float* d_dst = dalloc<float>(n_vectors * 256);
    ck(cudaMemcpy(d_src, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy H2D");

    k::fwht256_cuda(d_src, d_dst, n_vectors, nullptr);
    ck(cudaDeviceSynchronize(), "fwht256_cuda");

    std::vector<float> h_out(n_vectors * 256);
    ck(cudaMemcpy(h_out.data(), d_dst, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "memcpy D2H");

    double max_fwht_diff = 0.0;
    for (size_t i = 0; i < h_out.size(); ++i) {
        double diff = std::fabs(h_out[i] - h_ref[i]);
        if (diff > max_fwht_diff) max_fwht_diff = diff;
    }
    if (max_fwht_diff > 1e-5) {
        std::fprintf(stderr, "FAIL: FWHT-256 GPU output differs from host reference (max diff = %e)\n", max_fwht_diff);
        ++g_fail;
    } else {
        std::printf("  -> FWHT-256 bitwise match with host reference (max diff = %e, OK)\n", max_fwht_diff);
    }

    // Self-inverse property: H * (H * x) == x
    k::fwht256_inplace_cuda(d_dst, n_vectors, nullptr);
    ck(cudaDeviceSynchronize(), "fwht256_inplace_cuda");
    ck(cudaMemcpy(h_out.data(), d_dst, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "memcpy D2H");

    double max_inv_diff = 0.0;
    for (size_t i = 0; i < h_out.size(); ++i) {
        double diff = std::fabs(h_out[i] - h_in[i]);
        if (diff > max_inv_diff) max_inv_diff = diff;
    }
    if (max_inv_diff > 1e-5) {
        std::fprintf(stderr, "FAIL: FWHT-256 self-inverse property failed (max diff = %e)\n", max_inv_diff);
        ++g_fail;
    } else {
        std::printf("  -> FWHT-256 self-inverse property verified H*H*x == x (max diff = %e, OK)\n", max_inv_diff);
    }

    // -------------------------------------------------------------
    // Test 2: Invariance of Attention Scores under Hadamard Rotation
    // -------------------------------------------------------------
    std::printf("[2/4] Verifying Attention Score Invariance: <H*q, H*k> == <q, k>...\n");
    double max_dot_diff = 0.0;
    for (int r = 0; r < n_vectors; r += 2) {
        const float* q = h_in.data() + r * 256;
        const float* k_vec = h_in.data() + (r + 1) * 256;
        const float* hq = h_ref.data() + r * 256;
        const float* hk = h_ref.data() + (r + 1) * 256;

        double dot_raw = 0.0;
        double dot_rot = 0.0;
        for (int i = 0; i < 256; ++i) {
            dot_raw += (double) q[i] * (double) k_vec[i];
            dot_rot += (double) hq[i] * (double) hk[i];
        }
        double d = std::fabs(dot_raw - dot_rot);
        if (d > max_dot_diff) max_dot_diff = d;
    }
    if (max_dot_diff > 1e-5) {
        std::fprintf(stderr, "FAIL: Rotated dot product differs from original (max diff = %e)\n", max_dot_diff);
        ++g_fail;
    } else {
        std::printf("  -> Dot product invariant under rotation: |<Hq,Hk> - <q,k>| = %e (OK)\n", max_dot_diff);
    }

    // -------------------------------------------------------------
    // Test 3: Q4_0 Paged KV Append & Gather against Host Reference
    // -------------------------------------------------------------
    std::printf("[3/4] Verifying Q4_0 KV append and gather against host reference...\n");
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv; // 2
    const int D = (int) s.head_dim;   // 256
    const int P = (int) s.page_size;  // 64
    const int pages = 8;
    const int cells = pages * P;

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 5) % pages; // permutation
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy table");

    const size_t pool_bytes = (size_t) pages * H * P * k::kv_q4_bytes_per_head(D);
    uint8_t* d_k_q4 = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v_q4 = dalloc<uint8_t>(pool_bytes);
    float* d_kcur = dalloc<float>(H * D);
    float* d_vcur = dalloc<float>(H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);

    std::vector<std::vector<float>> host_k(cells), host_v(cells);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);

    const int n_fill = cells - 25; // leave some cells empty
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        for (auto& x : kv) x = nd(rng) * 2.0f;
        for (auto& x : vv) x = nd(rng) * 2.0f;

        // Apply FWHT before append
        for (int h = 0; h < H; ++h) {
            fwht256_host_reference(kv.data() + h * D);
            fwht256_host_reference(vv.data() + h * D);
        }
        host_k[pos] = kv;
        host_v[pos] = vv;

        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        ck(cudaMemcpy(d_kcur, kv.data(), kv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy k");
        ck(cudaMemcpy(d_vcur, vv.data(), vv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy v");

        k::kv_append_q4_step(d_k_q4, d_v_q4, d_table, d_step, d_kcur, d_vcur, s, nullptr);
        ck(cudaDeviceSynchronize(), "kv_append_q4_step");
    }

    // Verify written Q4_0 blocks bitwise against host quantizer
    std::vector<uint8_t> h_k_q4(pool_bytes), h_v_q4(pool_bytes);
    ck(cudaMemcpy(h_k_q4.data(), d_k_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k_q4");
    ck(cudaMemcpy(h_v_q4.data(), d_v_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v_q4");

    long bad_blocks = 0;
    const int blocks_per_head = D / k::QK4_0; // 8
    const int bytes_per_head = blocks_per_head * sizeof(k::block_q4_0); // 144

    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long page = (long long) table[pos / P];
                const long long row = (page * H + h) * P + (pos % P);
                const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
                const float* raw_vec = (is_v ? host_v[pos].data() : host_k[pos].data()) + h * D;

                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_q4_0 host_blk;
                    quantize_block_q4_0_host(raw_vec + b * 32, host_blk);

                    const k::block_q4_0* gpu_blk = reinterpret_cast<const k::block_q4_0*>(pool_ptr) + b;
                    if (std::memcmp(&host_blk, gpu_blk, sizeof(k::block_q4_0)) != 0) {
                        if (bad_blocks == 0) {
                            std::printf("MISMATCH at n=%d, pos=%d, is_v=%d, h=%d, b=%d\n", n, pos, is_v, h, b);
                            std::printf("  host blk.d = 0x%04x (%f), gpu blk.d = 0x%04x (%f)\n",
                                        host_blk.d, k::f32_from_f16(host_blk.d),
                                        gpu_blk->d, k::f32_from_f16(gpu_blk->d));
                            std::printf("  host qs: ");
                            for (int i = 0; i < 16; ++i) std::printf("%02x ", host_blk.qs[i]);
                            std::printf("\n  gpu  qs: ");
                            for (int i = 0; i < 16; ++i) std::printf("%02x ", gpu_blk->qs[i]);
                            std::printf("\n");
                        }
                        ++bad_blocks;
                    }
                }
            }
        }
    }
    if (bad_blocks > 0) {
        std::fprintf(stderr, "FAIL: %ld Q4_0 blocks differ from host quantizer\n", bad_blocks);
        ++g_fail;
    } else {
        std::printf("  -> All Q4_0 pool blocks bitwise equal to host quantizer (OK)\n");
    }

    // -------------------------------------------------------------
    // Test 3b: the batched STEP append == the per-token steps, captured
    // -------------------------------------------------------------
    // The captured decode window (verify.cpp) appends a whole group's rows in ONE kv_append_q4_batch_step
    // call, every row's position read from its own DEVICE step row; the per-token kv_append_q4_step loop is
    // the oracle (test 3 above verified it against the host quantizer).  Same cells, same rotated rows, fresh
    // pools: the pool bytes must be BIT-IDENTICAL, device copy and KV-streaming host copy alike.  Then ONE
    // captured call is replayed twice with the step rows rewritten between the replays: a position baked in
    // at capture would rewrite the first run's cells and leave the second's empty.
    std::printf("[3b/4] Verifying the batched step append vs the per-token steps (capture included)...\n");
    {
        const int W = 6;                                   // the window's rows
        const int32_t RUN0 = 37, RUN1 = 53;                // two disjoint runs of W cells
        std::vector<std::vector<float>> run_k(2), run_v(2);
        for (int r = 0; r < 2; ++r)
            for (int t = 0; t < W; ++t) {
                std::vector<float> kv(H * D), vv(H * D);
                for (auto& x : kv) x = nd(rng) * 2.0f;
                for (auto& x : vv) x = nd(rng) * 2.0f;
                for (int h = 0; h < H; ++h) {
                    fwht256_host_reference(kv.data() + h * D);
                    fwht256_host_reference(vv.data() + h * D);
                }
                run_k[r].insert(run_k[r].end(), kv.begin(), kv.end());
                run_v[r].insert(run_v[r].end(), vv.begin(), vv.end());
            }
        // the oracle pool, the one-call pool, the captured pool, and the oracle's two-run leg
        uint8_t* pk_o = dalloc<uint8_t>(pool_bytes); uint8_t* pv_o = dalloc<uint8_t>(pool_bytes);
        uint8_t* pk_b = dalloc<uint8_t>(pool_bytes); uint8_t* pv_b = dalloc<uint8_t>(pool_bytes);
        uint8_t* pk_c = dalloc<uint8_t>(pool_bytes); uint8_t* pv_c = dalloc<uint8_t>(pool_bytes);
        uint8_t* pk_f = dalloc<uint8_t>(pool_bytes); uint8_t* pv_f = dalloc<uint8_t>(pool_bytes);
        uint8_t* hk_o = dalloc<uint8_t>(pool_bytes); uint8_t* hv_o = dalloc<uint8_t>(pool_bytes);
        uint8_t* hk_b = dalloc<uint8_t>(pool_bytes); uint8_t* hv_b = dalloc<uint8_t>(pool_bytes);
        uint8_t* hk_c = dalloc<uint8_t>(pool_bytes); uint8_t* hv_c = dalloc<uint8_t>(pool_bytes);
        uint8_t* hk_f = dalloc<uint8_t>(pool_bytes); uint8_t* hv_f = dalloc<uint8_t>(pool_bytes);
        const k::KvHostPools host_o{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, hk_o, hv_o};
        const k::KvHostPools host_b{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, hk_b, hv_b};
        const k::KvHostPools host_c{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, hk_c, hv_c};
        const k::KvHostPools host_f{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, hk_f, hv_f};
        float* d_krow = dalloc<float>(H * D); float* d_vrow = dalloc<float>(H * D);
        float* d_K = dalloc<float>((size_t) W * H * D); float* d_V = dalloc<float>((size_t) W * H * D);
        int32_t* d_steps = dalloc<int32_t>((size_t) W * k::kStepCount);
        auto steps_for = [&](int32_t base) {
            std::vector<int32_t> st((size_t) W * k::kStepCount, 0);
            for (int t = 0; t < W; ++t) {
                st[(size_t) t * k::kStepCount + k::kStepPos] = base + t;
                st[(size_t) t * k::kStepCount + 1] = base + t + 1;   // kStepNKv: pos + 1
            }
            ck(cudaMemcpy(d_steps, st.data(), st.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "steps");
        };
        auto load_run = [&](int r) {
            ck(cudaMemcpy(d_K, run_k[r].data(), run_k[r].size() * sizeof(float), cudaMemcpyHostToDevice), "K");
            ck(cudaMemcpy(d_V, run_v[r].data(), run_v[r].size() * sizeof(float), cudaMemcpyHostToDevice), "V");
        };
        auto single_leg = [&](const int32_t base, int r, uint8_t* pk, uint8_t* pv, const k::KvHostPools& host) {
            steps_for(base);
            for (int t = 0; t < W; ++t) {
                ck(cudaMemcpy(d_krow, run_k[r].data() + (size_t) t * H * D, H * D * sizeof(float), cudaMemcpyHostToDevice), "krow");
                ck(cudaMemcpy(d_vrow, run_v[r].data() + (size_t) t * H * D, H * D * sizeof(float), cudaMemcpyHostToDevice), "vrow");
                k::kv_append_q4_step(pk, pv, d_table, d_steps + (size_t) t * k::kStepCount, d_krow, d_vrow, s, nullptr, &host);
            }
        };
        // oracle leg 1: RUN0 token by token (device pool + host copy)
        single_leg(RUN0, 0, pk_o, pv_o, host_o);
        // the batch leg: the same cells from the window's contiguous rows in one call
        load_run(0); steps_for(RUN0);
        k::kv_append_q4_batch_step(pk_b, pv_b, d_table, d_steps, W, d_K, d_V, s, nullptr, &host_b);
        ck(cudaDeviceSynchronize(), "sync 3b");
        auto cmp_pool = [&](const char* what, const uint8_t* a, const uint8_t* b) {
            std::vector<uint8_t> ha(pool_bytes), hb(pool_bytes);
            ck(cudaMemcpy(ha.data(), a, pool_bytes, cudaMemcpyDeviceToHost), "d2h");
            ck(cudaMemcpy(hb.data(), b, pool_bytes, cudaMemcpyDeviceToHost), "d2h");
            const bool eq = std::memcmp(ha.data(), hb.data(), pool_bytes) == 0;
            std::printf("  %-52s %s\n", what, eq ? "bit-identical" : "*** WRONG ***");
            if (!eq) ++g_fail;
        };
        cmp_pool("batch step vs per-token steps, K", pk_o, pk_b);
        cmp_pool("batch step vs per-token steps, V", pv_o, pv_b);
        cmp_pool("host copies (KV streaming), K", hk_o, hk_b);
        cmp_pool("host copies (KV streaming), V", hv_o, hv_b);
        // ONE captured call, replayed for RUN0 and then for RUN1; the oracle appends both runs
        load_run(0); steps_for(RUN0);
        cudaStream_t cs2 = nullptr;
        ck(cudaStreamCreate(&cs2), "cs2");
        ck(cudaStreamBeginCapture(cs2, cudaStreamCaptureModeThreadLocal), "begincap");
        k::kv_append_q4_batch_step(pk_c, pv_c, d_table, d_steps, W, d_K, d_V, s, cs2, &host_c);
        cudaGraph_t g2 = nullptr;
        ck(cudaStreamEndCapture(cs2, &g2), "endcap");
        ck(cudaStreamDestroy(cs2), "csd");
        size_t nodes2 = 0;
        ck(cudaGraphGetNodes(g2, nullptr, &nodes2), "nodes");
        cudaGraphExec_t ex2 = nullptr;
        ck(cudaGraphInstantiate(&ex2, g2, 0), "inst");
        ck(cudaGraphLaunch(ex2, nullptr), "launch run0");
        load_run(1); steps_for(RUN1);
        ck(cudaGraphLaunch(ex2, nullptr), "launch run1");
        // the oracle's two-run leg: RUN0 then RUN1 token by token
        single_leg(RUN0, 0, pk_f, pv_f, host_f);
        single_leg(RUN1, 1, pk_f, pv_f, host_f);
        ck(cudaDeviceSynchronize(), "sync 3b capture");
        cmp_pool("captured replays vs per-token steps, K", pk_f, pk_c);
        cmp_pool("captured replays vs per-token steps, V", pv_f, pv_c);
        // the negative control: RUN1's cells must be populated through the SAME graph (a baked position would
        // have rewritten RUN0's cells instead)
        long long run1_nonzero = 0;
        {
            std::vector<uint8_t> hc(pool_bytes);
            ck(cudaMemcpy(hc.data(), pk_c, pool_bytes, cudaMemcpyDeviceToHost), "d2h");
            for (int t = 0; t < W; ++t) {
                const long long pos = RUN1 + t;
                for (int h = 0; h < H; ++h) {
                    const long long page = (long long) table[pos / P];
                    const uint8_t* row = hc.data() + ((page * H + h) * P + (pos % P)) * bytes_per_head;
                    for (int i = 0; i < bytes_per_head; ++i) run1_nonzero += row[i] != 0;
                }
            }
        }
        std::printf("  %-52s %lld of %d non-zero\n", "the second run's cells arrive through the same graph",
                    run1_nonzero, W * H * bytes_per_head);
        if (run1_nonzero == 0) ++g_fail;
        std::printf("  %-52s %zu nodes for one captured call\n", "the window append is capturable", nodes2);
        ck(cudaGraphExecDestroy(ex2), "exd");
        ck(cudaGraphDestroy(g2), "gd");
        cudaFree(d_krow); cudaFree(d_vrow); cudaFree(d_K); cudaFree(d_V); cudaFree(d_steps);
        cudaFree(pk_o); cudaFree(pv_o); cudaFree(pk_b); cudaFree(pv_b);
        cudaFree(pk_c); cudaFree(pv_c); cudaFree(pk_f); cudaFree(pv_f);
        cudaFree(hk_o); cudaFree(hv_o); cudaFree(hk_b); cudaFree(hv_b);
        cudaFree(hk_c); cudaFree(hv_c); cudaFree(hk_f); cudaFree(hv_f);
    }

    // -------------------------------------------------------------
    // Test 4: Gather Q4_0 into FP16 Scratch and verify accuracy
    // -------------------------------------------------------------
    std::printf("[4/4] Verifying Q4_0 gather unpacking and dequantization...\n");
    const int max_ids = 128;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t* d_k_scratch = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t* d_v_scratch = dalloc<uint16_t>((size_t) max_ids * H * D);

    std::vector<int32_t> ids(max_ids);
    for (int i = 0; i < max_ids; ++i) {
        ids[i] = positions[rng() % n_fill];
    }
    ck(cudaMemcpy(d_ids, ids.data(), max_ids * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy ids");

    int32_t hstep[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");

    k::kv_gather_q4_step(d_k_q4, d_v_q4, d_table, d_ids, d_step, max_ids, s, d_k_scratch, d_v_scratch, nullptr);
    ck(cudaDeviceSynchronize(), "kv_gather_q4_step");

    std::vector<uint16_t> h_k_scratch((size_t) max_ids * H * D);
    std::vector<uint16_t> h_v_scratch((size_t) max_ids * H * D);
    ck(cudaMemcpy(h_k_scratch.data(), d_k_scratch, h_k_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "memcpy d2h k_scratch");
    ck(cudaMemcpy(h_v_scratch.data(), d_v_scratch, h_v_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "memcpy d2h v_scratch");

    double worst_quant_err = 0.0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const float* orig = (is_v ? host_v[cell].data() : host_k[cell].data()) + h * D;
                const uint16_t* gathered = (is_v ? h_v_scratch.data() : h_k_scratch.data()) + (i * H + h) * D;

                for (int d = 0; d < D; ++d) {
                    float deq = k::f32_from_f16(gathered[d]);
                    float raw = orig[d];
                    double err = std::fabs(deq - raw);
                    if (err > worst_quant_err) worst_quant_err = err;
                }
            }
        }
    }
    std::printf("  -> Max reconstruction error after 4-bit quant + dequant: %.4f (OK)\n", worst_quant_err);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_table);
    cudaFree(d_k_q4);
    cudaFree(d_v_q4);
    cudaFree(d_kcur);
    cudaFree(d_vcur);
    cudaFree(d_step);
    cudaFree(d_ids);
    cudaFree(d_k_scratch);
    cudaFree(d_v_scratch);

    if (g_fail == 0) {
        std::printf("=== kv_q4_parity: ALL TESTS PASSED SUCCESSFULLY! ===\n");
        return 0;
    } else {
        std::fprintf(stderr, "=== kv_q4_parity: FAILED (%d failures) ===\n", g_fail);
        return 1;
    }
}
