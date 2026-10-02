# patches/

Vendor patches for the FetchContent clone of llama.cpp (base GIT_TAG
`3cf03257f219afbe7334045ff7c6a06ac68c627d`).  Applied by `CMakeLists.txt`'s
`PATCH_COMMAND`, idempotently: a reverse-check first (re-populate of an
already-patched tree = no-op), and a failed apply in BOTH directions aborts the
configure rather than leaving an engine silently unpatched.

- `strata-llamacpp-mmq-x-ptrs.patch` — `ggml/src/ggml-cuda/mmq.cuh`:
  `mmq_args.x_ptrs` (per-channel absolute bases) threaded through
  `launch_mul_mat_q` into `mul_mat_q`'s three tile-offset sites.  Needed by the
  fork's zero-copy MoE read path (`STRATA_MMQ_BLOB`; `src/prefill/prefill.cpp`
  -> `moe_mmq.cu` `Product::w_tab`).  Default-off: with `x_ptrs == null` the
  layout is byte-for-byte the stock uniform-stride one.

## Already-populated build dir (no re-populate)

A configure that reuses an existing `_deps/strata_llamacpp-src` runs no
PATCH_COMMAND.  Apply the same patch by hand before building from such a tree:

```sh
cd <repo>/build/_deps/strata_llamacpp-src
P=<repo>/patches/strata-llamacpp-mmq-x-ptrs.patch
git apply -R --check "$P" 2>/dev/null || git apply "$P"
```

`git apply -R --check` succeeding means the tree already carries the patch (a
no-op).  Any other failure is loud and must be resolved, never bypassed — a
silently unpatched vendor tree means `mmq_args` has no `x_ptrs` member and the
fork side will not compile (that is the intended fail-loud, not a mystery).
