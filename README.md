# ⚡ TurboQuant Benchmark Suite

A local-AI benchmarking project to test **Google's TurboQuant** KV-cache compression
(PolarQuant + QJL, ICLR 2026) against standard quantization modes using `llama.cpp`.

```
q8_0 (8-bit baseline)  →  q4_0 (4-bit)  →  turbo3 (3-bit, 4.6× KV)  →  turbo2 (2-bit, 6.4× KV)
```

---

## What TurboQuant Actually Does

Standard quantization compresses model **weights** at load time.  
TurboQuant compresses the **KV cache** (keys + values) *at runtime*, online, with:

| Component | Role |
|-----------|------|
| **PolarQuant** | Randomly rotates vectors into polar coords → uniform distribution → better scalar quantization |
| **QJL** (1-bit) | Applies a 1-bit Quantized Johnson-Lindenstrauss transform on the residual → eliminates bias in attention scores |

**Result:** 3-bit KV cache with zero accuracy loss and zero memory overhead
(traditional methods need 1–2 extra bits to store quantization constants).

---

## Project Structure

```
turboquant-bench/
├── setup.sh              ← install + build + download model
├── benchmark.py          ← main throughput benchmark (all cache modes)
├── perplexity_eval.py    ← accuracy test (PPL comparison)
├── dashboard.html        ← interactive visual dashboard (pre-seeded)
├── results/
│   ├── bench_results.json  ← written after benchmark.py
│   ├── perplexity_results.json
│   └── report.html        ← auto-generated after benchmark
└── models/               ← GGUF model lives here
```

---

## Quick Start

### Step 1 — Setup (one-time)
```bash
chmod +x setup.sh
./setup.sh
```
This will:
- Clone `turboquant_plus` (llama.cpp fork with TurboQuant support)
- Build llama.cpp with Metal/CUDA if available
- Download **Qwen3-1.7B-Q4_K_M** (~1.2 GB) from HuggingFace

### Step 2 — Run benchmark
```bash
# Compare all five cache modes on short + medium prompts
python3 benchmark.py

# Custom: only TurboQuant modes, long context
python3 benchmark.py --modes q8_0,turbo3,turbo2 --prompts short,medium,long

# Dry run (no binary needed — simulated numbers)
python3 benchmark.py --dry-run
```

### Step 3 — Check accuracy
```bash
python3 perplexity_eval.py --modes q8_0,q4_0,turbo3
```

### Step 4 — View results
```bash
# Pre-seeded dashboard (always works):
open dashboard.html

# Live report (generated after step 2):
open results/report.html
```

---

## Benchmark Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `models/Qwen3-1.7B-Q4_K_M.gguf` | Path to any GGUF model |
| `--modes` | `q8_0,q4_0,turbo3` | Cache modes (comma-sep) |
| `--prompts` | `short,medium` | Prompt IDs: short/medium/long |
| `--predict` | `256` | Tokens to generate |
| `--threads` | auto | CPU threads |
| `--dry-run` | off | Simulate (no binary) |

---

## Cache Modes Explained

| Mode   | Bits/value | KV Compression | Description |
|--------|-----------|----------------|-------------|
| `q8_0`   | 8-bit | 1× (baseline) | Standard llama.cpp 8-bit |
| `q4_0`   | 4-bit | 2× | Standard 4-bit block quant |
| `turbo4` | 4-bit | 3.8× | TurboQuant 4-bit (no overhead) |
| `turbo3` | 3-bit | 4.6× | TurboQuant 3-bit (sweet spot) |
| `turbo2` | 2-bit | 6.4× | TurboQuant 2-bit (extreme) |

**Why turbo4 > q4_0 at same bits?**  
Standard q4_0 stores per-block scale factors (≈1–2 extra bits overhead).
TurboQuant eliminates this overhead via PolarQuant, so turbo4 fits more in the same memory.

---

## Hardware Notes

| Hardware | Expected turbo3 gain vs q8_0 |
|----------|------------------------------|
| MacBook M-series (Metal) | +45–55% decode at long ctx |
| NVIDIA RTX (CUDA)         | +60–80% decode at long ctx |
| CPU only                  | +15–25% decode (memory-bound) |

---

## References

- Paper: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)  
- Google blog: [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)  
- Implementation: [turboquant_plus](https://github.com/TheTom/turboquant_plus)  
- vLLM tracking: [Issue #38171](https://github.com/vllm-project/vllm/issues/38171)
