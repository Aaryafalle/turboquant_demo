# ⚡ TurboQuant Benchmark Suite

> Test Google's new **TurboQuant** KV-cache compression (ICLR 2026) on your local machine — measure real speed, memory, and accuracy differences across compression modes.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?style=flat-square&logo=windows)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 🧠 What is TurboQuant?

When an AI model generates text, it saves calculations in memory called the **KV cache** (Key-Value cache). As conversations get longer, this cache grows and becomes the main bottleneck — slow to read and eats RAM.

**TurboQuant** is a new compression algorithm by Google Research (ICLR 2026) that compresses this cache to just **3 bits per value** using two techniques:

| Technique | What it does |
|-----------|-------------|
| **PolarQuant** | Rotates data vectors into polar coordinates so they compress cleanly without extra overhead |
| **QJL** (1-bit) | Applies a 1-bit Quantized Johnson-Lindenstrauss transform on the residual to eliminate rounding errors |

**Result:** 4.6x smaller KV cache, ~50% faster generation, zero quality loss — all at inference time with no retraining needed.

---

## What This Project Does

This benchmark suite runs the same AI model with different KV cache modes and measures:

- Decode speed — how fast the model generates text (tokens/sec)
- Prefill speed — how fast it reads your prompt (tokens/sec)
- Peak RAM usage — how much memory the KV cache consumes
- Perplexity (PPL) — accuracy check, does compression hurt quality?

Then generates a visual HTML dashboard comparing all modes side by side.

---

## Cache Modes Compared

| Mode | Bits | KV Compression | Description |
|------|------|---------------|-------------|
| `q8_0` | 8-bit | 1x (baseline) | Standard llama.cpp, no compression |
| `q4_0` | 4-bit | 2x | Traditional 4-bit block quantization |
| `turbo4` | 4-bit | 3.8x | TurboQuant 4-bit (no memory overhead) |
| `turbo3` | 3-bit | 4.6x | TurboQuant 3-bit (sweet spot) |
| `turbo2` | 2-bit | 6.4x | TurboQuant 2-bit (extreme compression) |

> **Why does turbo4 beat q4_0 at the same bit-width?**
> Standard q4_0 stores scaling constants per block (~1-2 extra bits overhead). TurboQuant eliminates this overhead entirely via PolarQuant.

---

## Project Structure

```
turboquant-bench/
│
├── setup.bat               <- Windows setup (builds engine + downloads model)
├── setup.ps1               <- PowerShell alternative
├── benchmark.py            <- Main benchmark harness
├── perplexity_eval.py      <- Accuracy evaluation (PPL test)
├── dashboard.html          <- Pre-seeded visual dashboard (works offline)
│
├── turboquant_plus/        <- llama.cpp fork with TurboQuant (auto-cloned)
├── models/                 <- AI model stored here (auto-downloaded)
│   └── Qwen3-1.7B-Q4_K_M.gguf
│
└── results/                <- Generated after running benchmark
    ├── bench_results.json
    ├── perplexity_results.json
    └── report.html
```

---

## Quick Start (Windows)

### Prerequisites

| Tool | Download | Note |
|------|----------|------|
| **Git** | https://git-scm.com/download/win | |
| **CMake 3.x+** | https://cmake.org/download | Tick "Add to PATH" during install |
| **Python 3.10+** | https://python.org | Tick "Add to PATH" during install |
| **VS Build Tools** | https://aka.ms/vs/17/release/vs_BuildTools.exe | Select: Desktop development with C++ |

### Step 1 — Clone this repo
```cmd
git clone https://github.com/YOUR_USERNAME/turboquant-bench.git
cd turboquant-bench
```

### Step 2 — Run setup
```cmd
setup.bat
```
This will:
- Clone `turboquant_plus` (llama.cpp fork with TurboQuant support)
- Build `llama-cli.exe` using CMake + Visual Studio (~5 min)
- Download Qwen3-1.7B-Q4_K_M model (~1.2 GB)
- Install Python dependencies

### Step 3 — Run the benchmark
```cmd
python benchmark.py --modes q8_0,q4_0,turbo3 --prompts short,medium
```

### Step 4 — View results
```cmd
python -m http.server 8080
```
Open browser → `http://localhost:8080/results/report.html`

---

## Test Without Building (Dry Run)

Don't have CMake or VS Build Tools yet? Test with simulated data:
```cmd
python benchmark.py --dry-run
```
Results are based on real TurboQuant paper numbers with small random variation.

---

## Benchmark Options

```
python benchmark.py [options]

  --model     Path to GGUF model (default: models/Qwen3-1.7B-Q4_K_M.gguf)
  --modes     Cache modes, comma-separated (default: q8_0,q4_0,turbo3)
              Choices: q8_0, q4_0, turbo4, turbo3, turbo2
  --prompts   Prompt IDs, comma-separated (default: short,medium)
              Choices: short (512 ctx), medium (2048 ctx), long (8192 ctx)
  --predict   Tokens to generate per run (default: 256)
  --threads   CPU threads (default: auto)
  --dry-run   Simulate results without needing the binary
```

### Examples
```cmd
# Quick test
python benchmark.py --modes q8_0,turbo3 --prompts short

# Full benchmark — all modes, all prompts
python benchmark.py --modes q8_0,q4_0,turbo4,turbo3,turbo2 --prompts short,medium,long

# Accuracy check
python perplexity_eval.py --modes q8_0,q4_0,turbo3
```

---

## Expected Results

| Metric | q8_0 vs turbo3 |
|--------|---------------|
| Decode speed | +40-57% faster |
| KV Cache RAM | 4.6x less |
| Output quality | No change (PPL within +-0.1) |

Gains become more dramatic at longer context lengths (8K+ tokens) where the KV cache dominates memory.

---

## How It Works

```
benchmark.py
    |
    |-- Reads --modes and --prompts flags
    |
    |-- For each (cache_mode x prompt) combination:
    |       |-- Calls llama-cli.exe with --cache-type-k [mode]
    |       |-- Parses speed numbers from output
    |       |-- Stores result in RunResult object
    |
    |-- Prints results table to terminal
    |-- Saves bench_results.json
    |-- Generates report.html with Chart.js charts
```

The only thing that changes between runs is the `--cache-type-k` flag.
Everything else — model, prompt, hardware — stays identical for a fair comparison.

---

## References

- TurboQuant Paper: https://arxiv.org/abs/2504.19874
- Google Research Blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- turboquant_plus (community llama.cpp fork): https://github.com/TheTom/turboquant_plus
- vLLM integration tracking: https://github.com/vllm-project/vllm/issues/38171

---

## Built With

- llama.cpp — Local LLM inference engine
- turboquant_plus — TurboQuant fork of llama.cpp
- Qwen3-1.7B — Test model by Alibaba
- Chart.js — Dashboard charts
- Rich — Terminal UI

---

## Author

**Aarya** — Engineering Student, RMD Sinhagad School of Engineering, Pune

Built as a hands-on exploration of Google's TurboQuant compression research (ICLR 2026).

---

## License

MIT — free to use, modify, and share.
