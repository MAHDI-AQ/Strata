# Serving Configuration Reference

How a Strata AGX server is configured and booted: the server config file, the engine
flags that matter for serving, and the environment gates. Every serving value has
exactly **one owner** — either a CLI flag (values that belong to a boot) or an
environment gate (A/B experiments and diagnostics). Two owners for one value is a
defect: the reader can no longer tell what the engine will do.

## 1. Server config file (JSON)

`serve/server.py` reads the config passed as `--config`. Schema:

| key | type | effect |
|---|---|---|
| `exe` | string | engine binary. Relative paths resolve against `cwd`. The engine's version label is read from `BUILD.json` beside the binary. |
| `cwd` | string | working directory for the engine process (model/data relative paths resolve here). |
| `tokenizer` | string | tokenizer directory (`vocab.json` + `token_type.json`); overrides the server CLI default. |
| `gpu` | int or list | CUDA device pinning, `nvidia-smi` numbering. A list (`[0,1]`) is the layer-split pair. |
| `host` | string | bind address (default `127.0.0.1`; the server CLI `--host` wins). |
| `port` | int | intended listen port (kept for reference). The server binds `--port` from its CLI — it does not read this key; keep the two consistent. |
| `env` | object | key/values merged into the **engine** process environment (this is where a recipe's env gates live). |
| `sampling` | object | `temperature` / `top_p` / `min_p` defaults applied to requests that do not carry their own. |
| `model_name` | string | label reported by `/health` and `/v1/models` (`q3-xxs-3x262k`-style). |
| `log` | string | engine output capture. Opened in **append** mode — rotate it yourself (see §6); a recipe boot script should rotate before launch so growth stays bounded across boots. |
| `args` | list | the engine argv (everything except `--serve`, which the server appends itself). |

Skeleton:

```json
{
  "exe": "build/strata",
  "cwd": ".",
  "tokenizer": "path/to/tokenizer",
  "gpu": [0, 1],
  "host": "127.0.0.1",
  "port": 8096,
  "env": { "STRATA_WATCHDOG_S": "180" },
  "sampling": { "temperature": 0.6, "top_p": 0.95, "min_p": 0.05 },
  "model_name": "strata-serving",
  "log": "logs/engine.log",
  "args": ["--pack", "...", "--concurrency", "3"]
}
```

Boot the server with the port explicit: `python3 serve/server.py --engine strata --config config.json --port 8096`.

## 2. Single-owner values (flags)

These settings belong to their flag; the matching environment overrides were removed so a
recipe cannot mean two things at once. If you carry an older recipe, delete the env form:

| value | owner flag | former env form (removed) |
|---|---|---|
| nanobatch width K ∈ [2..4] | `--nanobatch` | `STRATA_NANOBATCH` |
| speculative min-p | `--spec-min-p` | `STRATA_SPEC_MIN_P` |
| adapt cadence | `--adapt-every` | `STRATA_ADAPT_EVERY` |
| adapt swap budget | `--adapt-swaps` | `STRATA_ADAPT_SWAPS` |
| concurrent conversation-cache budget (MiB) | `--conversation-cache-mib` | `STRATA_CONCURRENT_CACHE_MIB` |

`tools/acceptance/flag_owner_gate.py` enforces this class mechanically (no value with two
owners; no new flag/env mirror pairs).

## 3. Serving flags

Grouped by function; `--help` lists the full CLI surface including one-shot generate and
diagnostic flags. Effect notes marked "measured" come from engine receipts on dual RTX 4090.

### Model & data
| flag | effect |
|---|---|
| `--pack DIR` | pack directory (tokenizer-adjacent metadata, expert manifest; default `pack/full`). |
| `--native SHARD` | enables every full-context native path (stream-token, GR MMVF, BF16, head, dense + PLE key, MoE combine, GDN, router, QSA, indexer, RoPE, PLE postops, CPU q8_0 contract unless the expert cache is on). Individual `--native-*` flags remain for A/B. |
| `--native-*-gguf PATH` | native head / dense-shard weights from the model shards (repeat per shard). |
| `--ple-gguf PATH` | the PLE (n-gram dictionary) table — required; the second GGUF shard. |
| `--mtp DIR` | the MTP draft head assets. |
| `--expert-profile PATH` | ranked expert-pair profile used to pre-fill the expert caches at boot. |
| `--kv fp16\|int8\|q4_0\|k8v4` | KV storage format. `q4_0` ≈ 9,024 B/token (capacity lever); `k8v4` = INT8 K + rotated Q4 V (fidelity lever); `int8` = half of fp16; default `fp16`. |
| `--kv-resident N` | stream KV: keep N cells per QSA layer in VRAM (min 20480), rest pinned in host RAM; `0` = all VRAM. Not with `--kv k8v4`. |

### Capacity & residency
| flag | effect |
|---|---|
| `--max-context N` | KV/state capacity per slot (tokens). |
| `--concurrency N` | serving slots (1..16). A cap is a lattice: every fixed-size array is sized to it. Queue beyond N is server-side (standard continuous batching). |
| `--expert-cache N\|auto` | VRAM expert-cache size. `auto` sizes from live free VRAM at boot; an explicit N byte-packs pairs (the packed pair count differs from N — N is the max-blob count). An explicit size defeats the auto shrink-on-refill path. |
| `--expert-cache-device1\|2\|3 N` | per-device cache overrides for multi-device expert placement. |
| `--vram-reserve-mib N` | free-VRAM reserve kept below the expert cache + graph captures. |
| `--ple-row-cache N` | bounded host row cache of the PLE table (90 B/row); the whole-table value for this model is 320,001,536 rows. Independent of `--ple-io`. |
| `--ple-io direct\|mmap\|ram` | PLE read path: `direct` = unbuffered 4 KiB reads (default; table never enters RAM); `ram` = mmap + mlock the whole table (very slow to lock; verify on your box). |

### Prefill & scheduling
| flag | effect |
|---|---|
| `--prefill auto\|N` | the single-request (C=1) prefill chunking: `auto` → up to 16,384-token chunks. |
| `--concurrent-prefill N` | the **concurrent path**'s chunk size, 256..4096. See §4 — this, not `--prefill`, is what a serving recipe pins. |
| `--batch-rows N` | rows processed per round (≤ kernel `MAXT`). Wider rounds raise decode throughput until the kernel envelope is hit. |
| `--batch-policy fair\|depth` | row scheduler: `fair` = even spread across active slots; `depth` = depth-first per slot. |
| `--batch-parallel 0\|1` | per-member fork/join streams for the batch chain (measured decode win on the split path). |
| `--batch-graphs N` | CUDA-graph cache entries for batch shapes. |
| `--batch-padding 0\|1` | pad batch rows to full width (graph capture friendliness). |
| `--nanobatch K` | fine-grained micro-partitioning K ∈ [2..4] for cross-stage overlap (see §5 gates for when it engages). |

### Speculative decoding
| flag | effect |
|---|---|
| `--spec T` | verify-window token budget (draft tree width). |
| `--mtp-max-t N` | cap on MTP draft tokens per step. |
| `--suffix-draft N` | suffix-match drafts per step. |
| `--spec-min-p P` | minimum draft confidence to submit a drafted token. |

### Split & placement
| flag | effect |
|---|---|
| `--layer-split K` | stage-0 layers on the first card; the remainder on the second (requires `--serve`). Below the stage-time balance point the cross-slot stage-1 overlap can fault — pick K by measurement, not taste. |
| `--split-device D` | which device runs stage 1. |
| `--split-skip-if-fits` | collapse to one device if the model fits one card. |
| `--shared-expert-arena` | one host expert arena shared across engines (multi-lane setups). |

### Gate & environment knobs used at serving (env `env{}` in the recipe)
| gate | effect |
|---|---|
| `STRATA_WATCHDOG_S` | engine self-abort after N seconds without token progress (0 = off). 180 is the shipped serving value. |
| `STRATA_STAGE_OVERLAP` | run stage0[next] with stage1[cur] (default on; same-device keeps historic default). |
| `STRATA_STAGE_OVERLAP_CROSSDEV` | cross-device variant of the stage overlap. |
| `STRATA_PREFILL_CHAIN` | cross-chunk prefill stage pipeline (the 3,450 tok/s class receipts were chained). |
| `STRATA_PLE_PREFETCH` | PLE read-ahead overlapping the previous chunk's compute. |
| `STRATA_MTP_FUSE_CHAIN` | fused MTP draft chain (measured decode premium). |
| `STRATA_DRAFT_WAVEFRONT` | draft wavefront scheduling. |
| `STRATA_CONCURRENT_RETAIN` | live prefix retention across turns on the concurrent path (multi-turn resume; parking is not supported on this path). |
| `STRATA_CONCURRENT_PROFILE` | emit the per-round concurrent profile lines (diagnostics; off for rate runs). |
| `STRATA_SLOT_LAZY` | allocate slot sessions lazily (shrinks boot-time sizing for idle slots; refused with `--batch-parallel 1` in some revisions — check your tree). |
| `STRATA_IPC_SHM` | SHM name for the client IPC ring (`/strata_ipc_<port>`). |
| `STRATA_PASS_STALL_MS` | stage pass stall bound (ms). |
| `STRATA_SAMPLER_ONE_BLOCK` | one-block sampler path. |
| `STRATA_MMQ_BLOB` | MMQ blob expert path toggle (A/B arm; check the receipt in `docs/` before enabling). |
| `STRATA_VERIFY_DEVICE_PLAN` | verifier device-plan validation arm. |
| `STRATA_BATCH_HEAD` | batched head path. |

~130 `STRATA_*` gates exist in total; the rest are A/B experiments, kernel arm selectors,
and diagnostics. Find any of them with `grep -rn "STRATA_" src include serve`. **Rule:
if the value has a CLI flag, the flag owns it.**

## 4. The prefill pair — a worked ownership example

Two flags govern prompt chunking, one per path:

- `--prefill auto` → the **single-request** path chunks up to 16,384 tokens. The README's
  "cold prefill throughput" row (3,450.2 tok/s on IQ3_XXS) was measured here.
- `--concurrent-prefill N` → the **concurrent serving path** takes its chunk size from this
  flag (validated 256..4096). A serving recipe pins 4096.

Same-machine measurements at chunk 4096 on the concurrent path: deep solo prefill
≈2,114–2,327 tok/s (38.9K / 108.3K prompts). The gap to the 16,384-chunk number is the
chunk cap itself — raising the concurrent cap is a named engine lever, not a config knob.

## 5. When the cross-stage overlap engages

`STRATA_STAGE_OVERLAP*` + `--nanobatch K` partition eligible slots into micro-batches so
stage 0 and stage 1 can ping-pong across the two cards. The engine keeps a "one stage-1
pass in flight" law; width and overlap experiments must keep it (faults otherwise surface
as `prefill copy_i32` illegal-access class aborts). Measure duty per card before and after
any change here.

## 6. Boot discipline (recipes)

- Boot through a recipe script that is **TERM-first**: signal the old engine, wait for
  processes to exit AND both GPUs to return to idle (<600 MiB used) before launching the
  next engine. Never SIGKILL an engine and relaunch within seconds — on a 2×4090 box that
  class has produced a GPU at the driver level ("GPU requires reset", Xid 120/154) whose
  only recovery is a reboot.
- Rotate `log` before launch (keep one previous generation); the engine log is append-mode.
- Wait for `/health` `"loaded": true` before sending traffic; treat a suspiciously fast
  ready as suspect until the engine's own log confirms the boot.
