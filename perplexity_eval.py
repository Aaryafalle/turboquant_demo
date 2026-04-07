#!/usr/bin/env python3
"""
perplexity_eval.py — TurboQuant accuracy check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs llama-perplexity on a wikitext-style passage for each
cache mode and compares PPL scores.

Lower PPL = better quality preservation.
A PPL within ~±0.1 of q8_0 is considered zero-loss.

Usage:
  python3 perplexity_eval.py --model models/Qwen3-1.7B-Q4_K_M.gguf
"""

import argparse, json, re, subprocess, sys, time
from pathlib import Path

import platform
_WIN = platform.system() == "Windows"
LLAMA_PERP = (r"turboquant_plus\build\bin\Release\llama-perplexity.exe"
              if _WIN else
              "./turboquant_plus/build/bin/llama-perplexity")
RESULTS_DIR = Path("results")

# Standard wikitext chunk (public domain — from WikiText-103 test set)
WIKITEXT_SAMPLE = """
 = Valkyria Chronicles III = 

 Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit .
Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan ,
is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation
Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series .
Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story
runs parallel to the first game and follows the " Nameless " , a penal military unit serving the
nation of Gallia during the Second Europan War who perform secret black operations and are not
officially recognized by the Gallian Army . The game began development in 2010 , carrying over
Senjō no Valkyria 's mechanics and refining them for the new entry . Unlike previous entries ,
Valkyria Chronicles III was not localized , with the developer citing low sales of the previous
Valkyria Chronicles II outside Japan as a reason .
""".strip()

CACHE_MODES = ["q8_0", "q4_0", "turbo4", "turbo3", "turbo2"]

def run_perplexity(model: str, cache_type: str, text_file: str) -> dict:
    cmd = [
        LLAMA_PERP,
        "--model", model,
        "--file", text_file,
        "--cache-type-k", cache_type,
        "--cache-type-v", cache_type,
        "--ctx-size", "512",
        "--threads", "4",
        "--log-disable",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - t0
        output = proc.stdout + proc.stderr

        # parse PPL: "Final estimate: PPL = 12.3456"
        m = re.search(r"Final estimate.*?PPL\s*=\s*([\d.]+)", output)
        ppl = float(m.group(1)) if m else None

        return {"cache_type": cache_type, "ppl": ppl, "time_s": round(elapsed, 2), "error": None}
    except subprocess.TimeoutExpired:
        return {"cache_type": cache_type, "ppl": None, "time_s": 300, "error": "Timeout"}
    except Exception as e:
        return {"cache_type": cache_type, "ppl": None, "time_s": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="models/Qwen3-1.7B-Q4_K_M.gguf")
    parser.add_argument("--modes",  default="q8_0,q4_0,turbo3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    text_file = "/tmp/tq_ppl_input.txt"
    with open(text_file, "w") as f:
        f.write(WIKITEXT_SAMPLE)

    print("=" * 58)
    print("  TurboQuant — Perplexity Evaluation (accuracy proxy)")
    print("=" * 58)
    print(f"  Model : {args.model}")
    print(f"  Modes : {modes}\n")

    results = []
    baseline_ppl = None

    for mode in modes:
        print(f"  [{mode}] Running… ", end="", flush=True)
        if args.dry_run:
            import random
            BASE = {"q8_0": 12.31, "q4_0": 12.45, "turbo4": 12.33, "turbo3": 12.37, "turbo2": 12.54}
            r = {"cache_type": mode, "ppl": BASE.get(mode, 12.5) + random.uniform(-0.05, 0.05),
                 "time_s": round(random.uniform(5, 12), 1), "error": None}
        else:
            r = run_perplexity(args.model, mode, text_file)
        results.append(r)

        if r["error"]:
            print(f"ERROR — {r['error']}")
        else:
            if mode == "q8_0":
                baseline_ppl = r["ppl"]
            delta = ""
            if baseline_ppl and mode != "q8_0" and r["ppl"]:
                d = r["ppl"] - baseline_ppl
                delta = f"  Δ{d:+.3f} vs q8_0"
            print(f"PPL = {r['ppl']:.4f}   ({r['time_s']:.1f}s){delta}")

    # save
    RESULTS_DIR.mkdir(exist_ok=True)
    outfile = RESULTS_DIR / "perplexity_results.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results → {outfile}")
    print("\n  TurboQuant accuracy goal: PPL within ±0.1 of q8_0 = ✅ zero-loss")

    if args.dry_run:
        print("  [DRY RUN — simulated values]\n")


if __name__ == "__main__":
    main()