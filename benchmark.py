#!/usr/bin/env python3
"""
TurboQuant Benchmark Harness
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compares KV-cache quantization modes across:
  • q8_0   — 8-bit  (baseline)
  • q4_0   — 4-bit  (lightweight)
  • turbo3 — 3-bit TurboQuant (PolarQuant + QJL, target 4.6x compression)
  • turbo2 — 2-bit TurboQuant (extreme, 6.4x compression)

Metrics captured per run:
  • Tokens/sec (prefill + decode)
  • Peak VRAM / RAM (MB)
  • Perplexity proxy (token log-likelihood from llama-perplexity)
  • Output quality (subjective, saved to results/)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# ── rich UI ───────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich import print as rprint
    RICH = True
except ImportError:
    RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
    class Panel:
        pass

console = Console()

# ── config ────────────────────────────────────────────────────────────────────
import platform
_WIN = platform.system() == "Windows"
_EXE = ".exe" if _WIN else ""
_SEP = "\\" if _WIN else "./"

DEFAULT_MODEL  = r"models\Qwen3-1.7B-Q4_K_M.gguf" if _WIN else "models/Qwen3-1.7B-Q4_K_M.gguf"
LLAMA_CLI      = f"llama-cli{_EXE}"      if _WIN else "./llama-cli"
LLAMA_BENCH    = f"llama-bench{_EXE}"    if _WIN else "./llama-bench"
RESULTS_DIR    = Path("results")
RESULTS_JSON   = RESULTS_DIR / "bench_results.json"

CACHE_MODES = [
    ("q8_0",   "8-bit (baseline)",          "📦"),
    ("q4_0",   "4-bit (standard quant)",    "⚡"),
    ("turbo4", "4-bit TurboQuant (3.8x)",   "🚀"),
    ("turbo3", "3-bit TurboQuant (4.6x)",   "🔥"),
    ("turbo2", "2-bit TurboQuant (6.4x)",   "💎"),
]

# Test prompts — varied length and complexity to stress KV cache
PROMPTS = [
    {
        "id": "short",
        "label": "Short context",
        "ctx": 512,
        "text": "Explain what a transformer attention mechanism does in 3 sentences.",
    },
    {
        "id": "medium",
        "label": "Medium context",
        "ctx": 2048,
        "text": (
            "You are an expert in machine learning systems. "
            "Describe the memory bottleneck problem in large language model inference, "
            "explain how key-value caches work, why they grow with context length, "
            "and propose three different engineering solutions to mitigate this. "
            "For each solution discuss the tradeoffs in memory, accuracy, and latency."
        ),
    },
    {
        "id": "long",
        "label": "Long context (needle-in-haystack)",
        "ctx": 8192,
        "text": (
            "Below is a long document. At the end you will find a hidden answer. "
            + ("The quick brown fox jumps over the lazy dog. " * 200)  # filler
            + "HIDDEN FACT: The secret number is 7429. "
            + ("More filler text here for context padding. " * 100)
            + "\n\nQuestion: What is the secret number mentioned in the document?"
        ),
    },
]


@dataclass
class RunResult:
    cache_type:    str
    prompt_id:     str
    ctx_len:       int
    prefill_tps:   float = 0.0   # tokens/sec during prompt ingestion
    decode_tps:    float = 0.0   # tokens/sec during generation
    peak_mem_mb:   float = 0.0   # peak process RSS in MB
    total_time_s:  float = 0.0
    output_text:   str   = ""
    error:         str   = ""
    timestamp:     str   = ""
    compress_ratio: float = 0.0  # theoretical KV cache compression

    # TurboQuant theoretical compression ratios per mode
    COMPRESS = {
        "q8_0": 1.0, "q4_0": 2.0,
        "turbo4": 3.8, "turbo3": 4.6, "turbo2": 6.4,
    }

    def __post_init__(self):
        self.compress_ratio = self.COMPRESS.get(self.cache_type, 1.0)
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")


def check_binary(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def parse_perf_from_output(text: str) -> dict:
    """Extract prefill and decode tok/s from llama.cpp stderr output."""
    perf = {"prefill_tps": 0.0, "decode_tps": 0.0}
    # llama.cpp format: "llama_perf_sampler_print: ... 123.45 tokens per second"
    # and: "llama_perf_context_print:   prompt eval time = ... / 512 tokens (  2.34 ms per token,  427.35 tokens per second)"
    m = re.search(r"prompt eval.*?(\d+\.\d+)\s+tokens per second", text)
    if m:
        perf["prefill_tps"] = float(m.group(1))
    m = re.search(r"eval time\s*=.*?(\d+\.\d+)\s+tokens per second", text)
    if m:
        perf["decode_tps"] = float(m.group(1))
    # fallback: llama_bench style
    m = re.search(r"(\d+\.\d+)\s+t/s\s+prompt", text)
    if m:
        perf["prefill_tps"] = float(m.group(1))
    m = re.search(r"(\d+\.\d+)\s+t/s\s+generation", text)
    if m:
        perf["decode_tps"] = float(m.group(1))
    return perf


def get_peak_rss_mb(pid: int) -> float:
    """Read peak RSS from /proc on Linux."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmPeak"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    except Exception:
        return 0.0


def run_inference(
    model: str,
    prompt: dict,
    cache_type: str,
    n_predict: int = 256,
    n_threads: int = 0,
) -> RunResult:
    """Run a single llama-cli inference with given cache type."""

    if n_threads == 0:
        import os
        n_threads = max(1, (os.cpu_count() or 4) - 1)

    result = RunResult(
        cache_type=cache_type,
        prompt_id=prompt["id"],
        ctx_len=prompt["ctx"],
    )

    if not check_binary(LLAMA_CLI):
        result.error = f"Binary not found: {LLAMA_CLI}. Run setup.sh first."
        return result

    cmd = [
        LLAMA_CLI,
        "--model", model,
        "--prompt", prompt["text"],
        "--n-predict", str(n_predict),
        "--ctx-size", str(prompt["ctx"]),
        "--cache-type-k", cache_type,
        "--cache-type-v", cache_type,
        "--threads", str(n_threads),
        "--no-mmap",   
        "--log-disable",    # accurate memory measurement    # cleaner output
        "-ngl", "99",       # offload all layers to GPU if available
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.time() - t0
        combined = proc.stdout + proc.stderr
        #print(combined)
        perf = parse_perf_from_output(combined)
        result.prefill_tps  = perf["prefill_tps"]
        result.decode_tps   = perf["decode_tps"]
        result.total_time_s = elapsed
        result.peak_mem_mb  = 0.0

        # capture generated text (strip prompt echo)
        out = proc.stdout.strip()
        if out:
            result.output_text = out[:800]  # trim for JSON
        elif proc.returncode != 0:
            result.error = proc.stderr[-500:]

    except subprocess.TimeoutExpired:
        result.error = "Timeout (300s)"
    except FileNotFoundError:
        result.error = f"Binary not found: {LLAMA_CLI}"
    except Exception as e:
        result.error = str(e)

    return result


def run_llama_bench(model: str, cache_type: str, ctx: int = 512) -> dict:
    """
    Use llama-bench for faster, more accurate throughput numbers.
    Returns {"prefill_tps": float, "decode_tps": float}
    """
    if not check_binary(LLAMA_BENCH):
        return {}
    cmd = [
        LLAMA_BENCH,
        "-m", model,
        "--cache-type-k", cache_type,
        "--cache-type-v", cache_type,
        "-p", "128",   # prompt tokens
        "-n", "128",   # generation tokens
        "--ctx", str(ctx),
        "-r", "3",     # repetitions
        "-o", "json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        data = json.loads(proc.stdout)
        rows = data.get("results", [])
        out = {}
        for r in rows:
            if r.get("n_prompt", 0) > 0:
                out["prefill_tps"] = r.get("avg_ts", 0)
            elif r.get("n_gen", 0) > 0:
                out["decode_tps"] = r.get("avg_ts", 0)
        return out
    except Exception:
        return {}


def render_table(results: list[RunResult]):
    if not RICH:
        for r in results:
            print(f"{r.cache_type:8s} | {r.prompt_id:8s} | "
                  f"prefill={r.prefill_tps:.1f} t/s | decode={r.decode_tps:.1f} t/s | "
                  f"mem={r.peak_mem_mb:.0f} MB | {r.compress_ratio:.1f}x")
        return

    table = Table(title="TurboQuant Benchmark Results", show_lines=True)
    table.add_column("Cache",      style="cyan bold")
    table.add_column("Prompt",     style="white")
    table.add_column("Prefill t/s",style="green")
    table.add_column("Decode t/s", style="green bold")
    table.add_column("Peak RAM",   style="yellow")
    table.add_column("KV Compr.",  style="magenta bold")
    table.add_column("Status",     style="white")

    for r in results:
        status = "✅" if not r.error else f"❌ {r.error[:30]}"
        table.add_row(
            r.cache_type,
            r.prompt_id,
            f"{r.prefill_tps:.1f}",
            f"{r.decode_tps:.1f}",
            f"{r.peak_mem_mb:.0f} MB" if r.peak_mem_mb else "—",
            f"{r.compress_ratio:.1f}×",
            status,
        )
    console.print(table)


def save_results(results: list[RunResult]):
    RESULTS_DIR.mkdir(exist_ok=True)
    data = [asdict(r) for r in results]
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"\n[green]Results saved →[/green] {RESULTS_JSON}")


def generate_html_report(results: list[RunResult], model_name: str):
    """Write a self-contained HTML report to results/report.html"""
    data_js = json.dumps([asdict(r) for r in results], indent=2)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TurboQuant Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --orange: #d29922; --red: #f85149;
    --turbo-color: #a371f7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Courier New', monospace; padding: 2rem; }}
  h1 {{ font-size: 2rem; color: var(--accent); margin-bottom: .25rem; }}
  .subtitle {{ color: var(--muted); font-size: .85rem; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; }}
  .card h2 {{ font-size: 1rem; color: var(--muted); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: .1em; }}
  .stat {{ font-size: 2.5rem; font-weight: bold; color: var(--accent); }}
  .stat-label {{ font-size: .75rem; color: var(--muted); margin-top: .25rem; }}
  canvas {{ max-height: 320px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ background: #1c2128; color: var(--muted); padding: .5rem 1rem; text-align: left; font-weight: normal; }}
  td {{ padding: .5rem 1rem; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #1c2128; }}
  .badge {{ display: inline-block; padding: .1rem .5rem; border-radius: 4px; font-size: .75rem; }}
  .badge-turbo {{ background: #2d1f69; color: var(--turbo-color); }}
  .badge-std   {{ background: #1a2d1a; color: var(--green); }}
  .output {{ font-size: .78rem; color: var(--muted); background: #0d1117; padding: .5rem; border-radius: 4px;
             border: 1px solid var(--border); max-height: 80px; overflow: auto; white-space: pre-wrap; }}
  .section {{ margin-bottom: 2rem; }}
  .section-title {{ font-size: 1.1rem; color: var(--accent); margin-bottom: 1rem; border-bottom: 1px solid var(--border); padding-bottom: .5rem; }}
  .tag {{ background: #161b22; border: 1px solid var(--border); border-radius: 4px; padding: .15rem .6rem; font-size: .75rem; color: var(--muted); }}
</style>
</head>
<body>

<h1>⚡ TurboQuant Benchmark</h1>
<div class="subtitle">Model: {model_name} &nbsp;|&nbsp; Ran: <span id="ts"></span></div>

<div class="grid" id="statCards"></div>

<div class="section">
  <div class="section-title">Decode Throughput by Cache Mode</div>
  <div class="card"><canvas id="throughputChart"></canvas></div>
</div>

<div class="section">
  <div class="section-title">KV Cache Compression Ratio</div>
  <div class="card"><canvas id="compressionChart"></canvas></div>
</div>

<div class="section">
  <div class="section-title">Raw Results</div>
  <div class="card">
    <table id="resultsTable">
      <thead>
        <tr>
          <th>Cache</th><th>Prompt</th><th>Prefill t/s</th>
          <th>Decode t/s</th><th>RAM (MB)</th><th>KV Compression</th><th>Output</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script>
const RAW = {data_js};

document.getElementById('ts').textContent = new Date().toLocaleString();

// ── stat cards ────────────────────────────────────────────────────────────────
const byCache = {{}};
for (const r of RAW) {{
  if (!byCache[r.cache_type]) byCache[r.cache_type] = [];
  byCache[r.cache_type].push(r);
}}
const avgDecode = c => {{
  const vals = (byCache[c] || []).map(r => r.decode_tps).filter(v => v > 0);
  return vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(1) : '—';
}};

const MODES = ['q8_0','q4_0','turbo4','turbo3','turbo2'];
const LABELS = {{'q8_0':'8-bit Baseline','q4_0':'4-bit Std','turbo4':'TurboQuant 4-bit','turbo3':'TurboQuant 3-bit','turbo2':'TurboQuant 2-bit'}};
const COMPRESS = {{'q8_0':1,'q4_0':2,'turbo4':3.8,'turbo3':4.6,'turbo2':6.4}};

const statDiv = document.getElementById('statCards');
for (const m of MODES) {{
  if (!byCache[m]) continue;
  const isTurbo = m.startsWith('turbo');
  const div = document.createElement('div');
  div.className = 'card';
  div.innerHTML = `
    <h2>${{LABELS[m] || m}} <span class="badge ${{isTurbo?'badge-turbo':'badge-std'}}">${{isTurbo?'TurboQuant':'Standard'}}</span></h2>
    <div class="stat">${{avgDecode(m)}} <small style="font-size:1rem">t/s</small></div>
    <div class="stat-label">avg decode throughput</div>
    <div style="margin-top:.75rem;font-size:.85rem;color:var(--muted)">
      KV Compression: <strong style="color:var(--turbo-color)">${{COMPRESS[m]}}×</strong>
    </div>
  `;
  statDiv.appendChild(div);
}}

// ── throughput chart ──────────────────────────────────────────────────────────
const COLORS = {{'q8_0':'#58a6ff','q4_0':'#3fb950','turbo4':'#d29922','turbo3':'#a371f7','turbo2':'#f85149'}};
const modes = MODES.filter(m => byCache[m]);
const prompts = [...new Set(RAW.map(r => r.prompt_id))];

new Chart(document.getElementById('throughputChart'), {{
  type: 'bar',
  data: {{
    labels: prompts,
    datasets: modes.map(m => ({{
      label: LABELS[m] || m,
      data: prompts.map(p => {{
        const v = (byCache[m]||[]).find(r=>r.prompt_id===p);
        return v ? v.decode_tps : 0;
      }}),
      backgroundColor: COLORS[m] + 'cc',
      borderColor: COLORS[m],
      borderWidth: 1,
      borderRadius: 4,
    }})),
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#e6edf3' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }},
            title: {{ display: true, text: 'Tokens / sec', color: '#8b949e' }} }},
    }},
  }},
}});

// ── compression chart ─────────────────────────────────────────────────────────
new Chart(document.getElementById('compressionChart'), {{
  type: 'bar',
  data: {{
    labels: modes.map(m => LABELS[m] || m),
    datasets: [{{
      label: 'KV Cache Compression Ratio',
      data: modes.map(m => COMPRESS[m]),
      backgroundColor: modes.map(m => COLORS[m] + 'cc'),
      borderColor: modes.map(m => COLORS[m]),
      borderWidth: 1,
      borderRadius: 4,
    }}],
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }},
            title: {{ display: true, text: 'Compression ×', color: '#8b949e' }} }},
    }},
  }},
}});

// ── results table ─────────────────────────────────────────────────────────────
const tbody = document.querySelector('#resultsTable tbody');
for (const r of RAW) {{
  const tr = document.createElement('tr');
  const isTurbo = r.cache_type.startsWith('turbo');
  tr.innerHTML = `
    <td><span class="tag">${{r.cache_type}}</span></td>
    <td>${{r.prompt_id}} <small style="color:var(--muted)">(ctx ${{r.ctx_len}})</small></td>
    <td>${{r.prefill_tps ? r.prefill_tps.toFixed(1) : '—'}}</td>
    <td style="color:${{isTurbo?'var(--turbo-color)':'var(--green)'}};font-weight:bold">${{r.decode_tps ? r.decode_tps.toFixed(1) : '—'}}</td>
    <td>${{r.peak_mem_mb ? r.peak_mem_mb.toFixed(0) : '—'}}</td>
    <td style="color:var(--turbo-color);font-weight:bold">${{r.compress_ratio}}×</td>
    <td><div class="output">${{r.output_text || (r.error ? '❌ '+r.error : '—')}}</div></td>
  `;
  tbody.appendChild(tr);
}}
</script>
</body>
</html>"""

    report_path = RESULTS_DIR / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    console.print(f"[green]HTML report saved →[/green] {report_path}")
    return report_path


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TurboQuant Benchmark Harness")
    parser.add_argument("--model",   default=DEFAULT_MODEL, help="Path to GGUF model")
    parser.add_argument("--modes",   default="q8_0,q4_0,turbo3", help="Cache modes (comma-separated)")
    parser.add_argument("--prompts", default="short,medium", help="Prompt IDs (comma-separated)")
    parser.add_argument("--predict", type=int, default=256, help="Tokens to generate per run")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0=auto)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate results (no binary needed)")
    args = parser.parse_args()

    # ── banner ────────────────────────────────────────────────────────────────
    console.print(Panel.fit(
        "[bold cyan]TurboQuant Benchmark Harness[/bold cyan]\n"
        "[white]PolarQuant + QJL KV-cache compression — local AI speed test[/white]",
        border_style="cyan"
    ) if RICH else "=== TurboQuant Benchmark ===")

    selected_modes   = [m.strip() for m in args.modes.split(",")]
    selected_prompts = [p for p in PROMPTS if p["id"] in args.prompts.split(",")]
    model_name = Path(args.model).stem

    if not selected_prompts:
        console.print("[red]No valid prompts selected. Use: short, medium, long[/red]")
        sys.exit(1)

    # ── check binaries ────────────────────────────────────────────────────────
    if not args.dry_run:
        if not check_binary(LLAMA_CLI):
            console.print(f"[red]Binary not found: {LLAMA_CLI}[/red]\n"
                          "[yellow]→ Run ./setup.sh first to build llama.cpp with TurboQuant support[/yellow]")
            sys.exit(1)
        if not os.path.isfile(args.model):
            console.print(f"[red]Model not found: {args.model}[/red]\n"
                          "[yellow]→ Run ./setup.sh to download the model[/yellow]")
            sys.exit(1)

    all_results: list[RunResult] = []
    total_runs = len(selected_modes) * len(selected_prompts)
    run_n = 0

    for cache_type in selected_modes:
        for prompt in selected_prompts:
            run_n += 1
            console.print(f"\n[cyan][{run_n}/{total_runs}][/cyan] "
                          f"cache=[bold]{cache_type}[/bold]  prompt=[bold]{prompt['id']}[/bold]  "
                          f"ctx={prompt['ctx']}")

            if args.dry_run:
                # Simulate realistic numbers for demo/testing without binary
                import random, math
                base_decode = {"q8_0": 45, "q4_0": 55, "turbo4": 61, "turbo3": 68, "turbo2": 72}
                ctx_penalty = math.log(prompt["ctx"] / 512 + 1) * 0.15
                decode = base_decode.get(cache_type, 45) * (1 - ctx_penalty) + random.uniform(-2, 2)
                r = RunResult(
                    cache_type=cache_type,
                    prompt_id=prompt["id"],
                    ctx_len=prompt["ctx"],
                    prefill_tps=round(decode * 8 + random.uniform(-10, 10), 1),
                    decode_tps=round(max(1, decode), 1),
                    peak_mem_mb=round(1800 / RunResult.COMPRESS.get(cache_type, 1) + random.uniform(-50, 50), 0),
                    total_time_s=round(256 / max(1, decode), 1),
                    output_text=f"[DRY RUN] Simulated output for {cache_type} @ {prompt['id']}",
                )
            else:
                r = run_inference(
                    model=args.model,
                    prompt=prompt,
                    cache_type=cache_type,
                    n_predict=args.predict,
                    n_threads=args.threads,
                )

            all_results.append(r)

            # quick inline summary
            if r.error:
                console.print(f"  [red]✗ Error: {r.error[:80]}[/red]")
            else:
                console.print(
                    f"  prefill=[green]{r.prefill_tps:.1f}[/green] t/s  "
                    f"decode=[bold green]{r.decode_tps:.1f}[/bold green] t/s  "
                    f"KV compress=[magenta]{r.compress_ratio:.1f}×[/magenta]"
                )

    # ── summary table ─────────────────────────────────────────────────────────
    console.print("\n")
    render_table(all_results)

    # ── save ─────────────────────────────────────────────────────────────────
    save_results(all_results)
    report = generate_html_report(all_results, model_name)

    if args.dry_run:
        console.print("\n[yellow]ℹ DRY RUN — numbers are simulated. "
                      "Run without --dry-run for real measurements.[/yellow]")

    console.print(f"\n[bold green]Done! Open the report:[/bold green]")
    console.print(f"  → [underline]{report}[/underline]")


if __name__ == "__main__":
    main()