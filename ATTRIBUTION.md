# Open-Source Contributor Attribution & Lineage Registry

Strata was created by **Niko1221** and advanced by an extraordinary community of open-source engineers, systems researchers, and hardware performance specialists.

This repository is a downstream innovation fork maintained by the **M&F AI Lab** (`MAHDI-AQ`). While this fork introduces deep architectural extensions (Ada Lovelace tensor core kernels, GSQ-RCO 512-expert streaming, Dynamic RadixTree KV caching, chunk-pipelined prefill, and multi-agent concurrency), **none of this would be possible without the foundational breakthroughs, architectural elegance, and sustained contributions of the original Strata community.**

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
| **M&F AI Lab** | [`MAHDI-AQ`](https://github.com/MAHDI-AQ) | SM89 Ada Lovelace kernel tiling, GSQ-RCO IQ3_XXS 512-expert streaming, Dynamic RadixTree KV, 262K context scaling, and agentic benchmark receipts. |

---

## Fork Governance & Community Policy

1. **Standalone Downstream Lab:** This fork operates as an independent downstream research lab.
2. **Upstream PR Boundary:** Because this fork introduces radical architectural alterations tailored for multi-GPU agentic labs, **we do not submit unsolicited upstream PRs or file issues against `Niko1221/Strata`**. Upstream maintainers are welcome to cherry-pick any modular features, kernels, or bug fixes from our clean commits at their discretion.
3. **Open Access:** All proprietary M&F AI Lab innovations (13,633+ LoC) are provided openly under the repository's open-source license for the benefit of the local LLM and agentic engineering community.
