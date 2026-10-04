# Strata Configuration Templates

This directory contains clean, ready-to-run configuration templates for Strata (Mahdi AI Lab Edition).

## Available Templates

1. **`c1-262k-master.template.json`**:
   - Single-agent master posture ($C=1$).
   - Full 262,144 (262K) context window with `--kv q4_0` on dual 24GB GPUs (RTX 4090 / 3090).
   - Tuned for maximum context depth, speculative decoding (`--spec 2 --mtp-max-t 2`), and chunk-pipelined prefill.

2. **`c5-concurrent.template.json`**:
   - Multi-agent swarm concurrency posture ($C=5$).
   - 131,072 (128K) shared context pool with dynamic slot balancing.
   - Pipelined batch scheduling (`--batch-rows 20 --batch-graphs 24 --batch-padding 1`).
   - Scales aggregate throughput to >220-253 tok/s on dual RTX 4090s.

## Usage

1. Copy the desired template:
   ```bash
   cp configs/c1-262k-master.template.json my-server-config.json
   ```
2. Replace `/path/to/models/...` with the absolute paths to your local GGUF models, pack directory, tokenizer, and MTP weights.
3. Launch the server:
   ```bash
   python serve/server.py --engine strata --config my-server-config.json --port 8080
   ```
