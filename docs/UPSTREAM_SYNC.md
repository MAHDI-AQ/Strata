# Strata AGX Upstream Synchronization Registry & Cherry-Pick Ledger

**Project:** Strata AGX (Agentic Graph eXecution)  
**Maintained By:** Mahdi AI Lab (`Mahdi <MAHDI-AQ@users.noreply.github.com>`)  
**Parent Upstream Repository:** [`Niko1221/Strata`](https://github.com/Niko1221/Strata) (`origin/main`)  
**Fork Base Coordinate:** `99f3dbd0b21d1401b3769e0c0d963913607f380b` (`v0.1.38` baseline)  
**Current AGX Release:** `v0.1.38-agx.1.0.3`  
**Last Updated:** 2026-10-05

---

## 1. Synchronization Architecture & Policy

Strata AGX is an independent high-performance downstream fork engineered for multi-agent workloads, deep context scaling (262K), and dual-GPU Ada Lovelace (RTX 4090) acceleration. Because the downstream code diverged by **13,600+ LoC** (implementing zero-copy SHM IPC, lockless CRC32C RadixTree prefix caching, HiCache L2 host-RAM parking, Sequoia/EAGLE-2 speculative trees, and warp-cooperative Top-10 SM89 routers), **blanket upstream rebasing is strictly prohibited**.

Instead, Strata AGX employs a **Selective Cherry-Pick & Ledger Architecture**:
1. **Upstream Remote Tracking:** `origin` tracks `https://github.com/Niko1221/Strata.git`.
2. **Modular Cherry-Picks:** Upstream bug fixes, kernel micro-optimizations, and quantization formats are evaluated individually and cherry-picked with clean commit isolation.
3. **Formal Ledger:** Every evaluated upstream PR or commit series is recorded in this document with its SHA, title, AGX decision, and technical rationale.
4. **Zero Upstream Inconvenience:** To preserve development agility while avoiding unsolicited notifications or issues for creator Niko1221, all synchronization remains internal to the fork.

---

## 2. Versioning Standard & Brakes-On Discipline (Mahdi Rule, 2026-10-05)

Strata AGX follows the compound versioning scheme:
```text
v<upstream-version>-agx.<major>.<minor>.<patch>
```

- `<upstream-version>`: Baseline upstream release tracked (e.g. `0.1.38`).
- `<major>`: Ground-up engine generation in Strata AGX (locked at `1`). Never bumped unless an irreversible breaking rewrite occurs.
- `<minor>`: Monumental consolidated program deliveries (e.g. an entire multi-phase master plan sealed and verified). Minor versions are NEVER bumped for individual phases, tasks, or PRs.
- `<patch>`: ALL active incremental work — research phases, kernel fusions, sync cherry-picks, PRs, and daily fixes (`v0.1.38-agx.1.0.1`, `1.0.2`, ..., `1.0.9999`). Patch numbers grow naturally without premature milestone inflation.

*Baseline Milestone:* **`v0.1.38-agx.1.0.0`** (2026-10-05: Phases 1–5 complete).
*Phase 6 Milestone:* **`v0.1.38-agx.1.0.1`** (2026-10-05: Fused GDN Recurrence, Direct Q8_0 Epilogue, and SM89 Split-K Flash-Decoding).
*Phase 7 Milestone:* **`v0.1.38-agx.1.0.2`** (2026-10-05: Multi-Branch SpecTree DAG Expansion T <= 16, Device DP Verify Kernel, and Speculative Prefill).
*Phase 8 Milestone:* **`v0.1.38-agx.1.0.3`** (2026-10-05: NanoFlow-Style Device-Level Nanobatching K in [2..4], Lockless TripleBufferIPC 3-Slot Ring, and Overlapped PCIe Pipeline).

---

## 3. Upstream Commit & PR Evaluation Ledger

Tracking upstream commits on `origin/main` beyond fork base `99f3dbd` (October 3–5, 2026):

| Upstream Commit | Author / PR | Subject | AGX Classification | Rationale & Architectural Impact |
|---|---|---|---|---|
| `6f32ec0` | Niko1221 | setup: tip for images on 12 GB card: `--vram-reserve-mib 1000` | **EVALUATING** | Setup script tip only; non-intrusive. Can be cherry-picked in next setup polish. |
| `c2d1f19` | Niko1221 | setup: Unsloth's UD-IQ4_XS is regular choice | **EVALUATING** | Relates to automated model downloading in setup.py. Harmless utility update. |
| `8b36fa0` | Niko1221 | docs: README refined + six translations | **REJECTED (CUSTOM)** | Strata AGX maintains dedicated README.md reflecting AGX architecture and dual RTX 4090 benchmarks. |
| `1189218` | Niko1221 | Merge branch 'f-139b-par' into f-139-rel | **EVALUATING** | Preparation branch for upstream 0.1.39 release. Candidate for evaluation upon 0.1.39 tag. |
| `cb97e6c` | Niko1221 | Merge branch 'f-139b-spd' into f-139-rel | **EVALUATING** | Upstream speed improvements branch. To be bisected for relevant CUDA kernels. |
| `5f19911` | Niko1221 | #583: byte-budget ring only where it gains; short prompts | **EVALUATING** | Prompt ring buffer heuristic. Requires profiling against AGX chunked prefill engine. |
| `bc0fc49` | Niko1221 | setup: CUDA 12 engine driver floor per OS (Linux 525) | **EVALUATING** | Upstream setup driver requirements. Harmless. |
| `537588f` | Niko1221 | Parking with a layer split: draft ring with last stage, on GPU | **SUPERSEDED** | Strata AGX implements lockless CRC32C RadixTree with HiCache L2 host-RAM parking, which natively handles multi-stream concurrent parking across multiple GPUs. Upstream single-stream parking is superseded. |
| `f67bf51` | Niko1221 | Parking with a layer split: draft ring once, checkpoints moved | **SUPERSEDED** | Superseded by RadixTree prefix-cache architecture in `src/core/radix_tree.cpp`. |
| `e11166b` | Niko1221 | Conversation parking with `--layer-split` | **SUPERSEDED** | Superseded by RadixTree prefix-cache architecture in `src/core/radix_tree.cpp`. |
| `ba707d8` | Niko1221 | Several conversations at once: batch slots, pipelined layer split | **SUPERSEDED** | Strata AGX Phase 1 & 2 implemented full multi-stream wavefront concurrent dispatch (`src/program/concurrent_serve.cpp`), zero-copy SHM IPC, and independent MTP draftees. Upstream concurrent batching is superseded. |
| `e50f266` | Niko1221 | layer split: with explicit split points each GPU loads own weights | **ACCEPTED (IN-TREE)** | Strata AGX natively incorporates explicit layer split allocations (`--layer-split K`) across dual GPUs. |
| `e30dd85` | Niko1221 | STRATA_ARENA_MMAP=1: expert arena as read-only mapped file | **ACCEPTED (IN-TREE)** | Integrated in memory manager for host-RAM fallback. |
| `4f3973a` | Niko1221 | serve: read the /load and /unload body before replying | **ACCEPTED (CHERRY-PICKED)** | Clean HTTP handling fix for server.py. |
| `4ba35fd` – `7021c80` (48 commits) | Community / Intel | Intel Arc / SYCL Port & oneAPI backend integration | **OUT-OF-SCOPE** | Focuses on Intel Arc (B70 / Battlemage) and SYCL runtime. Strata AGX is strictly specialized for high-throughput NVIDIA Ada Lovelace / CUDA SM89 dual-GPU architectures. |

---

## 4. Synchronization Procedure for AGX Maintainers

When performing a synchronization pass from upstream:

1. **Fetch Latest Upstream Commits:**
   ```bash
   git fetch origin
   git log --oneline 99f3dbd..origin/main
   ```

2. **Inspect & Isolate Candidate Commit:**
   ```bash
   git log -p -1 <commit-sha>
   ```

3. **Apply Cherry-Pick to Local Branch:**
   ```bash
   git cherry-pick -x <commit-sha>
   ```

4. **Execute Full Parity & Serving Regression:**
   ```bash
   bash tools/acceptance/run_full_regression.sh
   ```

5. **Update This Ledger & Version:**
   - Record the commit in the table above with its evaluation status.
   - If functional changes occurred, increment patch version in `CMakeLists.txt` and `serve/server.py` (e.g. `v0.1.38-agx.1.0.1`).
   - Commit with author `Mahdi <MAHDI-AQ@users.noreply.github.com>`.

