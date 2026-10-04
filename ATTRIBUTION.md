# Open-Source Contributor Attribution & Lineage Registry

Strata was created by **Niko1221** and advanced by an extraordinary community of open-source engineers, systems researchers, and hardware performance specialists.

This repository is a downstream innovation fork maintained by the **Mahdi AI Lab** (`MAHDI-AQ`). While this fork introduces deep architectural extensions (Ada Lovelace tensor core kernels, GSQ-RCO 512-expert streaming, Dynamic RadixTree KV caching, chunk-pipelined prefill, and multi-agent concurrency), **none of this would be possible without the foundational breakthroughs, architectural elegance, and sustained contributions of the original Strata community.**

---

## The Strata Community Contributors

We formally and respectfully credit the following authors and contributors whose commits, pull requests, reviews, and bug reports form the foundation of Strata:

| Contributor / Handle | Contact / Profile | Key Architectural & Code Contributions |
|---|---|---|
| **Niko1221** | [`Niko1221`](https://github.com/Niko1221) | **Creator of Strata**. Designed the original inference engine, Swift support, IQ3_S native kernels, state machine, and core runtime architecture. |
| **tntcannon5000** | Niranjan Kewalramani ([`tntcannon5000`](https://github.com/tntcannon5000)) | Pioneered multi-GPU layer splitting, engine streaming, prompt chunk pipelining, and performance engineering. |
| **Guillaume PUTIER** | [`guillaume@putier.fr`](mailto:guillaume@putier.fr) | Engine stability, memory allocation improvements, and Linux build pipeline fixes. |
| **Andy** | [`q8atnight@gmail.com`](mailto:q8atnight@gmail.com) | Architectural guidance, quantization calibration methodology, and engine review. |
| **Jakub Luwierski** | [`j.luwierski@gmail.com`](mailto:j.luwierski@gmail.com) | Per-request temperature sampling, persistent Philox draw counter, `min_p` sampling, and penalty bitmap sampling. |
| **code-martin** | [`code-martin`](https://github.com/code-martin) | Q4_0 KV cache mode with FWHT-256 Walsh-Hadamard rotation (`--kv q4_0`). |
| **Mirtraxxx** | [`Mirtraxxx`](https://github.com/Mirtraxxx) | Conversation cache architecture, CPU idle sleep between requests (#4), and short prompt decode optimization. |
| **btechnet** | [`coolio986@hotmail.com`](mailto:coolio986@hotmail.com) | Zero-token reply engine race condition fix (#7) and unset `max_tokens` context sizing. |
| **Vistawizard** | [`Vistawizard`](https://github.com/Vistawizard) | CPU thread pool dangling-else fix in Linux `physical_cores()` (#14). |
| **Enky** | `enky@localhost` | Local engine testing, hardware validation, and patch review. |
| **samuelishida** | [`samuelishida`](https://github.com/samuelishida) | Build script resilience, environment variable isolation, and packaging. |
| **Oluwabori Olaleye** | [`borexola`](https://github.com/borexola) | Model packaging tooling and multi-shard GGUF parsing. |
| **GioStrives** | [`giostrives@gmail.com`](mailto:giostrives@gmail.com) | Community usability testing and CLI parameter ergonomics. |
| **Pravesh Khatana** | [`pravesh.khatana@gmail.com`](mailto:pravesh.khatana@gmail.com) | Evaluation tooling, tokenizer compliance, and prompt format validation. |
| **Mahdi AI Lab** | [`MAHDI-AQ`](https://github.com/MAHDI-AQ) | SM89 Ada Lovelace kernel tiling, GSQ-RCO IQ3_XXS 512-expert streaming, Dynamic RadixTree KV, 262K context scaling, and agentic benchmark receipts. |

---

## Fork Governance & Community Policy

1. **Standalone Downstream Lab:** This fork operates as an independent downstream research lab.
2. **Upstream PR Boundary:** Because this fork introduces radical architectural alterations tailored for multi-GPU agentic labs, **we do not submit unsolicited upstream PRs or file issues against `Niko1221/Strata`**. Upstream maintainers are welcome to cherry-pick any modular features, kernels, or bug fixes from our clean commits at their discretion.
3. **Open Access:** All proprietary Mahdi AI Lab innovations (13,633+ LoC) are provided openly under the repository's open-source license for the benefit of the local LLM and agentic engineering community.

---

## Ecosystem Cross-Pollination & Architectural Lineage

In pursuit of maximal serving acceleration and latency reduction across consumer hardware (Dual RTX 4090), Strata openly cross-pollinates and synthesizes architectural breakthroughs from the broader LLM systems and compiler research community. 

We formally acknowledge and attribute the following foundational projects, architectures, and research papers:

### 1. SGLang (LMSYS Org / UC Berkeley)
- **Repository:** [`sgl-project/sglang`](https://github.com/sgl-project/sglang)
- **Key Researchers:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Shen, Zhanghao Wu, Dachuan Li, Hao Zhang, Joseph E. Gonzalez, Ion Stoica (UC Berkeley / LMSYS).
- **Paper:** *"SGLang: Efficient Execution of Structured Language Model Programs"* (arXiv:2312.07104).
- **Attributed Architectural Patterns:**
  - **RadixAttention (Tree-Structured Prefix Caching):** Strata adapts the core insight of maintaining KV caches in a dynamic radix tree rather than a linear hash map, enabling zero-copy cache sharing across multi-turn agent conversations, tool-calling chains, and shared system prompts.
  - **Chunk-Level Hash Fingerprinting:** SGLang's chunk-based hashing pattern adapted to 64-token chunks using hardware-accelerated SSE4.2 CRC32C, enabling $O(1)$ block jumps that collapse 32K token lookups to sub-microsecond latency.
  - **Multi-Tier Memory Architecture (HiCache L2):** Hierarchical KV eviction and parking (L1 VRAM <-> L2 Pinned Host RAM) using page-locked pinned memory arenas (`cudaHostAllocPortable`) for wire-speed PCIe DMA restoration.
  - **Zero-Overhead Event Loop Scheduling:** The architectural separation of high-frequency token-level queue arbitration from GPU kernel execution pipelines to prevent CPU scheduling bottlenecks.
  - **Shared-Memory IPC Protocol:** SGLang's pattern of lock-free POSIX shared-memory channels for token streaming and logits between the Python API layer and the native engine.

### 2. vLLM (vLLM Team / UC Berkeley)
- **Repository:** [`vllm-project/vllm`](https://github.com/vllm-project/vllm)
- **Key Researchers:** Woosuk Kwon, Zhuohan Li, Siyuan Shen, Mellun Zhang, Lianmin Zheng, Charles Chen, Binhang Yuan, Ion Stoica (UC Berkeley).
- **Paper:** *"Efficient Memory Management for Large Language Model Serving with PagedAttention"* (SOSP 2023).
- **Attributed Architectural Patterns:**
  - **Asynchronous Output Processing:** Decoupling token detokenization, sampling logic, and SSE wire streaming from the synchronous GPU forward-pass critical path.
  - **Static CUDA Graph Re-binding:** The methodology of executing static CUDA graphs with dynamic batch sizes and sequence lengths by updating mapped device pointers and block tables rather than triggering expensive graph re-captures.
  - **Chunked Prefill & Decode Piggybacking:** Interleaving bounded prompt evaluation chunks with latency-sensitive decode steps to prevent frame-drops on active multi-agent streams.

### 3. FlashInfer (FlashInfer Team / University of Washington)
- **Repository:** [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer)
- **Key Researchers:** Zihao Ye, Ruihang Lai, Lianmin Zheng, Yineng Zhang, Joseph E. Gonzalez, Arvind Krishnamurthy (University of Washington / UC Berkeley).
- **Paper:** *"FlashInfer: Efficient and Customizable Attention Kernels for LLM Serving"*.
- **Attributed Architectural Patterns:**
  - **Fused GEMM Epilogues for Ada Lovelace (SM89):** Fusing normalization (RMSNorm) and bias addition directly into the GEMM input/output epilogues, eliminating intermediate global memory read/write passes.
  - **Grouped GEMM for Mixture-of-Experts:** Parallelizing execution of multiple routed expert GEMMs across device SMs in a unified grid launch to maximize hardware utilization under low batch sizes.

### 4. EAGLE-2 & Sequoia (Speculative Decoding Research)
- **Papers & Repositories:**
  - *EAGLE-2:* Yuhui Li, Fangcheng Fu, Ling Shen, Xu Chen, Shenggui Li, Wentao Zhang (Peking University) — *"EAGLE-2: Faster Sub-step Speculative Decoding with Dynamic Draft Trees"* (arXiv:2406.16858).
  - *Sequoia:* Zhuohan Li, Siyuan Shen, Lianmin Zheng, Woosuk Kwon, Ion Stoica (UC Berkeley) — *"Sequoia: 4.8x Faster Speculative Decoding with Dynamic Trees"* (arXiv:2402.12374).
  - *DeepSeek MTP:* DeepSeek-AI — *"DeepSeek-V2 / DeepSeek-V3 Technical Report"* (Multi-Token Prediction architecture).
- **Attributed Architectural Patterns:**
  - **Speculative Tree Topology Verification:** Verifying speculative candidate trees via customized 2D attention masks in a single forward pass, expanding acceptance rates beyond linear chain limits on branching reasoning paths.
  - **Confidence-Gated Speculative Depth:** Dynamic evaluation of token generation uncertainty (logit entropy / top-1 margin) to dynamically throttle speculative draft length $T$, conserving compute on low-confidence branches.

### 5. llama.cpp & GGML (Georgi Gerganov & Community)
- **Repository:** [`ggerganov/llama.cpp`](https://github.com/ggerganov/llama.cpp)
- **Key Architect:** Georgi Gerganov and the open-source GGML community.
- **Attributed Architectural Patterns:**
  - **Quantization Calibration (IQ3_XXS / Q4_0 / GSQ-RCO):** High-efficiency low-bit weight representation formats, block-quantized scales, and fast integer SIMD/warp-level dequantization kernels.
  - **Memory-Mapped Weight Ingestion (`mmap`):** Direct file-backed tensor paging enabling zero-overhead model loading and instant process initialization.

### 6. Nanoflow (MegaScale / DeepSeek-AI Research)
- **Repository / Paper:** *"NanoFlow: Towards Optimal Large Language Model Serving Through Device-Level Nanobatch Execution"* (arXiv:2408.12757).
- **Key Researchers:** DeepSeek-AI, Tsinghua University, Peking University.
- **Attributed Architectural Patterns:**
  - **Asynchronous Device-Level DMA Pipelines:** Overlapping host-to-device and device-to-host memory copy pipelines across multiple hardware copy engines (`copy_stream_`), saturating bidirectional PCIe bandwidth without stalling compute SMs.
  - **Dual-GPU Concurrent Transfer Scheduling:** Splitting multi-stage KV snapshots across heterogeneous or split-bus topologies (GPU 0 layers 0-26, GPU 1 layers 27-47) via concurrent stream dispatch.
