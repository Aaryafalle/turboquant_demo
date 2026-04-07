#!/usr/bin/env bash
# =============================================================================
#  TurboQuant Bench — Setup Script
#  Clones turboquant_plus (llama.cpp fork with TurboQuant KV cache support),
#  builds it, and downloads a small model for benchmarking.
# =============================================================================

set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[SETUP]${RESET} $*"; }
ok()   { echo -e "${GREEN}[  OK ]${RESET} $*"; }
warn() { echo -e "${YELLOW}[ WARN]${RESET} $*"; }
die()  { echo -e "${RED}[FAIL ]${RESET} $*"; exit 1; }

# ── config ────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/TheTom/turboquant_plus.git"
LLAMA_DIR="turboquant_plus"
MODEL_DIR="models"
# Qwen3-1.7B-Q4_K_M — small enough to run on most laptops, still meaningful
MODEL_URL="https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf"
MODEL_FILE="$MODEL_DIR/Qwen3-1.7B-Q4_K_M.gguf"

# ── pre-flight ────────────────────────────────────────────────────────────────
for cmd in git cmake make python3 curl; do
  command -v "$cmd" &>/dev/null || die "Missing dependency: $cmd"
done
ok "All dependencies found"

# ── clone / update repo ───────────────────────────────────────────────────────
if [ -d "$LLAMA_DIR/.git" ]; then
  log "turboquant_plus already cloned — pulling latest…"
  git -C "$LLAMA_DIR" pull --ff-only
else
  log "Cloning turboquant_plus…"
  git clone --depth 1 "$REPO_URL" "$LLAMA_DIR"
fi
ok "Source ready"

# ── build ─────────────────────────────────────────────────────────────────────
BUILD_ARGS="-DGGML_NATIVE=ON"

# Apple Silicon: enable Metal
if [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
  BUILD_ARGS="$BUILD_ARGS -DGGML_METAL=ON"
  warn "Detected Apple Silicon — Metal GPU enabled"
fi

# NVIDIA: enable CUDA if nvcc available
if command -v nvcc &>/dev/null; then
  BUILD_ARGS="$BUILD_ARGS -DGGML_CUDA=ON"
  warn "Detected CUDA — CUDA GPU enabled"
fi

log "Building llama.cpp with TurboQuant support (this may take a few minutes)…"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" $BUILD_ARGS -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON > /tmp/cmake.log 2>&1 \
  || { cat /tmp/cmake.log; die "cmake configure failed"; }

NPROC=$(python3 -c "import os; print(max(1, os.cpu_count()-1))")
cmake --build "$LLAMA_DIR/build" -j "$NPROC" > /tmp/build.log 2>&1 \
  || { tail -40 /tmp/build.log; die "Build failed — see /tmp/build.log"; }
ok "Build complete"

# symlink key binaries
ln -sf "$LLAMA_DIR/build/bin/llama-cli" llama-cli 2>/dev/null || true
ln -sf "$LLAMA_DIR/build/bin/llama-bench" llama-bench 2>/dev/null || true

# ── download model ────────────────────────────────────────────────────────────
mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_FILE" ]; then
  ok "Model already downloaded: $MODEL_FILE"
else
  log "Downloading Qwen3-1.7B (Q4_K_M, ~1.2 GB)…"
  curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL" \
    || die "Model download failed. Check your internet connection."
  ok "Model saved to $MODEL_FILE"
fi

# ── Python deps ───────────────────────────────────────────────────────────────
log "Installing Python benchmark dependencies…"
python3 -m pip install -q rich psutil matplotlib numpy tabulate

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}${GREEN}  Setup complete! Run the benchmark:${RESET}"
echo -e "${BOLD}  python3 benchmark.py${RESET}"
echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
