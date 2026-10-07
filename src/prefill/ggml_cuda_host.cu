// src/prefill/ggml_cuda_host.cu - prompt-speed plan step 2b: the host-side symbols of llama.cpp's ggml-cuda that its MMQ
// and quantize code reference, for the MMQ kernels compiled into strata_mmq without the rest of ggml-cuda.cu.
#include "strata/core/emulate.hpp"
#include "common.cuh"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    std::fprintf(stderr, "ggml-cuda (strata mmq): %s: %s\n  in %s at %s:%d\n", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

#if defined(GGML_USE_HIP)
namespace {

unsigned parse_hex(const std::string& value) {
    if (value.empty()) return 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 16);
    return end == value.c_str() || *end != '\0' ? 0 : static_cast<unsigned>(parsed);
}

// Match GGML's AMD cc encoding: OFFSET_AMD + gfx major * 0x100 + minor/stepping.
// For example, gfx1100 maps to GGML_CUDA_CC_RDNA3 (0x01001100).
int ggml_cuda_parse_amd_id(const char* device_name) {
    if (device_name == nullptr) return GGML_CUDA_CC_OFFSET_AMD;

    std::string arch(device_name);
    if (arch.compare(0, 3, "gfx") == 0) arch.erase(0, 3);
    const size_t suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    const size_t generic = arch.rfind("-generic");
    if (generic != std::string::npos && generic + 8 == arch.size()) arch.resize(generic);

    unsigned major = 0;
    unsigned minor = 0;
    const size_t separator = arch.find('-');
    if (separator != std::string::npos) {
        major = parse_hex(arch.substr(0, separator));
        minor = parse_hex(arch.substr(separator + 1)) * 0x10;
    } else if (arch.size() >= 3) {
        minor = parse_hex(arch.substr(arch.size() - 2));
        major = parse_hex(arch.substr(0, arch.size() - 2));
    }
    return GGML_CUDA_CC_OFFSET_AMD + static_cast<int>(major * 0x100 + minor);
}

}  // namespace
#endif

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        in.physical_device_count = n;
        for (int id = 0; id < n; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto & d = in.devices[id];
#if defined(GGML_USE_HIP)
            d.cc = ggml_cuda_parse_amd_id(prop.gcnArchName);
            if ((d.cc & 0xff00) == 0) {
                // Fall back to HIP's device version fields if its architecture name is unavailable.
                d.cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100 + prop.minor * 0x10;
            }
            d.smpbo = prop.sharedMemPerBlock;
            d.integrated = false;
            d.supports_cooperative_launch = false;
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.warp_size = prop.warpSize;
#else
            // #542: the fields MMQ's choices read come from cudaDeviceGetAttribute, which is ABI-stable: a
            // cudaDeviceProp filled by an older libcudart than the headers (a CUDA 13 build that links a CUDA 12
            // libcudart.so) is shifted - sharedMemPerBlockOptin read 1 there, and every i-quant lost MMQ (#420's
            // fits()).  In a matched build both give the same values.
            auto attr = [id](cudaDeviceAttr a) {
                int v = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&v, a, id));
                return v;
            };
            d.cc = 100 * strata::cc_major_of(attr(cudaDevAttrComputeCapabilityMajor)) +
                   10 * strata::cc_minor_of(attr(cudaDevAttrComputeCapabilityMinor));   // STRATA_EMULATE_CC
            d.smpbo = strata::smem_optin_of(attr(cudaDevAttrMaxSharedMemoryPerBlockOptin));
            d.integrated = attr(cudaDevAttrIntegrated) != 0;
            d.supports_cooperative_launch = attr(cudaDevAttrCooperativeLaunch) != 0;
            d.nsm = attr(cudaDevAttrMultiProcessorCount);
            d.smpb = (size_t) attr(cudaDevAttrMaxSharedMemoryPerBlock);
            d.warp_size = attr(cudaDevAttrWarpSize);
#endif
            d.vmm = false;
            d.total_vram = prop.totalGlobalMem;
            d.physical_device = id;
            d.physical_share_count = 1;
            d.virtual_index = 0;
        }
        return in;
    }();
    return info;
}

// graph-OOM fix: a process-wide flush of every CachingPool's unused buffers - the graph
// instantiate's OOM path frees real VRAM here (~10 MB needed at the peak; the pools hold the
// surplus). The pool lives in the anonymous namespace below; this trampoline has C linkage.
static void (*g_pool_flush_fn)() = nullptr;
extern "C" void strata_mmq_pool_flush_all() { if (g_pool_flush_fn) g_pool_flush_fn(); }

namespace {
// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).
struct CachingPool : ggml_cuda_pool {
    CachingPool() { std::lock_guard<std::mutex> lk(g_reg_mu); g_pools.push_back(this); }
    struct Buf { void * p; size_t size; bool used; };
    std::vector<Buf> bufs;
    std::mutex mu;
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (!b.used && b.size >= size) { b.used = true; *actual_size = b.size; return b.p; }
        void * p = nullptr;
        if (cudaMalloc(&p, size) != cudaSuccess) {
            // 3x262K fix: under pressure release the pool's UNUSED buffers and retry once -
            // the pool kept every distinct scratch size forever ("kept and reused"), which
            // is what starved the 3-slot concurrent long-prefill peak (the engine died here).
            cudaGetLastError();
            for (auto it = bufs.begin(); it != bufs.end(); ) {
                if (!it->used) { cudaFree(it->p); it = bufs.erase(it); } else ++it;
            }
            if (cudaMalloc(&p, size) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        }
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void flush_unused() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = bufs.begin(); it != bufs.end(); ) {
            if (!it->used) { cudaFree(it->p); it = bufs.erase(it); } else ++it;
        }
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        { std::lock_guard<std::mutex> lk(g_reg_mu);
          for (auto it = g_pools.begin(); it != g_pools.end(); ++it)
              if (*it == this) { g_pools.erase(it); break; } }
        for (auto & b : bufs) cudaFree(b.p);
    }
    static std::mutex g_reg_mu;
    static std::vector<CachingPool*> g_pools;
    static void flush_all() {
        std::lock_guard<std::mutex> lk(g_reg_mu);
        for (auto* p : g_pools) p->flush_unused();
    }
};

std::mutex CachingPool::g_reg_mu;
std::vector<CachingPool*> CachingPool::g_pools;
}  // namespace

namespace {
struct FlushReg { FlushReg() { g_pool_flush_fn = &CachingPool::flush_all; } } g_flush_reg;
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool());
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
