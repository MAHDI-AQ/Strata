# patches/

Vendor patches for the FetchContent clone of llama.cpp (base GIT_TAG
`3cf03257f219afbe7334045ff7c6a06ac68c627d`).  Applied by `CMakeLists.txt`'s
`PATCH_COMMAND`, idempotently: a reverse-check first (re-populate of an
already-patched tree = no-op), and a failed apply in BOTH directions aborts the
configure rather than leaving an engine silently unpatched.

- `strata-llamacpp-mmq-x-ptrs-additive.patch` — `ggml/src/ggml-cuda/mmq.cuh`
  plus one appended instantiation line in each of the 9 compiled
  `template-instances/mmq-instance-*.cu` files.  The patch is ADDITIVE ONLY:
  it appends `mul_mat_q_ptrs` (a vendored variant of the `mul_mat_q` kernel),
  its launcher/dispatch, and `mul_mat_q_case_ptrs`; every previously existing
  kernel, launcher and dispatch body stays byte-untouched (object-level proof:
  the pre-existing symbols' SASS is byte-identical between a pristine compile
  and a compile of the additive tree).  Needed by the fork's zero-copy MoE
  read path (`STRATA_MMQ_BLOB`; `src/prefill/prefill.cpp` -> `moe_mmq.cu`
  `Product::w_tab` -> `mul_mat_q_case_ptrs`).  Default-off: with no table the
  stock symbols run, byte-for-byte.
  It SUPERSEDES `strata-llamacpp-mmq-x-ptrs.patch` (removed from this dir):
  that earlier non-additive patch recompiled the whole stock `mul_mat_q`
  family (an `mmq_args`/kernel parameter change), which is exactly the surface
  the additive form keeps untouched.

## Already-populated build dir (no re-populate)

A configure that reuses an existing `_deps/strata_llamacpp-src` runs no
PATCH_COMMAND.  Apply the same patch by hand before building from such a tree:

```sh
cd <repo>/build/_deps/strata_llamacpp-src
P=<repo>/patches/strata-llamacpp-mmq-x-ptrs-additive.patch
git apply -R --check "$P" 2>/dev/null || git apply "$P"
```

If the tree still carries the SUPERSEDED non-additive patch (check:
`grep -c 'x_ptrs ? x_ptrs\[' ggml/src/ggml-cuda/mmq.cuh` is non-zero), restore
it to pristine first, then apply:

```sh
cd <repo>/build/_deps/strata_llamacpp-src
git checkout -- ggml/src/ggml-cuda/mmq.cuh          # drop the superseded edit
git apply "<repo>/patches/strata-llamacpp-mmq-x-ptrs-additive.patch"
```

`git apply -R --check` succeeding means the tree already carries the additive
patch (a no-op).  Any other failure is loud and must be resolved, never
bypassed — a silently unpatched vendor tree means the fork side will reference
`mul_mat_q_case_ptrs` it cannot see and the build will not compile (that is the
intended fail-loud, not a mystery).
