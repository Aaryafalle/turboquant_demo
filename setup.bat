@echo off
setlocal enabledelayedexpansion

:: =============================================================================
::  TurboQuant Bench — Windows CMD Setup
::  Run with:  setup.bat
:: =============================================================================

set REPO_URL=https://github.com/TheTom/turboquant_plus.git
set LLAMA_DIR=turboquant_plus
set MODEL_DIR=models
set MODEL_URL=https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf
set MODEL_FILE=models\Qwen3-1.7B-Q4_K_M.gguf

echo.
echo   ==================================================
echo     TurboQuant Benchmark  --  Windows Setup
echo   ==================================================
echo.

:: ── Check git ────────────────────────────────────────────────────────────────
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] git not found.
    echo        Install from: https://git-scm.com/download/win
    pause & exit /b 1
)
echo [ OK ] git found

:: ── Check cmake ──────────────────────────────────────────────────────────────
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] cmake not found.
    echo        Install from: https://cmake.org/download
    echo        Tick "Add CMake to system PATH" during install.
    pause & exit /b 1
)
echo [ OK ] cmake found

:: ── Check python ─────────────────────────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] python not found.
    echo        Install from: https://python.org
    echo        Tick "Add Python to PATH" during install.
    pause & exit /b 1
)
echo [ OK ] python found

:: ── Check curl (built into Win10/11) ────────────────────────────────────────
where curl >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] curl not found -- will use PowerShell for download fallback
    set USE_PS_DOWNLOAD=1
) else (
    echo [ OK ] curl found
    set USE_PS_DOWNLOAD=0
)

echo.

:: ── Clone repo ────────────────────────────────────────────────────────────────
if exist "%LLAMA_DIR%\.git" (
    echo [SETUP] turboquant_plus already cloned, pulling latest...
    git -C %LLAMA_DIR% pull --ff-only
) else (
    echo [SETUP] Cloning turboquant_plus...
    git clone --depth 1 %REPO_URL% %LLAMA_DIR%
    if %errorlevel% neq 0 ( echo [FAIL] Clone failed. Check internet. & pause & exit /b 1 )
)
echo [ OK ] Source ready

:: ── CMake configure ───────────────────────────────────────────────────────────
echo.
echo [SETUP] Configuring build...
if not exist "%LLAMA_DIR%\build" mkdir "%LLAMA_DIR%\build"

set CMAKE_EXTRA=

:: Check for NVIDIA GPU
where nvcc >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARN] CUDA detected -- enabling GPU support
    set CMAKE_EXTRA=-DGGML_CUDA=ON
) else (
    echo [SETUP] No CUDA found -- CPU-only build
)

cmake -S %LLAMA_DIR% -B %LLAMA_DIR%\build ^
    -DGGML_NATIVE=ON ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=ON ^
    %CMAKE_EXTRA%

if %errorlevel% neq 0 (
    echo.
    echo [FAIL] CMake configure failed.
    echo        Make sure Visual Studio Build Tools are installed:
    echo        https://aka.ms/vs/17/release/vs_BuildTools.exe
    echo        Select: Desktop development with C++
    pause & exit /b 1
)

:: ── Build ────────────────────────────────────────────────────────────────────
echo.
echo [SETUP] Building llama.cpp with TurboQuant support...
echo         This will take 3-8 minutes. Please wait.
echo.

:: Count logical CPUs for parallel build
for /f "tokens=2 delims==" %%i in ('wmic cpu get NumberOfLogicalProcessors /value ^| find "="') do set NPROC=%%i
set NPROC=%NPROC: =%
if "%NPROC%"=="" set NPROC=4

cmake --build %LLAMA_DIR%\build --config Release -j %NPROC%
if %errorlevel% neq 0 (
    echo [FAIL] Build failed. See output above.
    pause & exit /b 1
)
echo [ OK ] Build complete

:: ── Copy binaries to project root ────────────────────────────────────────────
set BIN_DIR=%LLAMA_DIR%\build\bin\Release
if not exist "%BIN_DIR%\llama-cli.exe" set BIN_DIR=%LLAMA_DIR%\build\Release

if exist "%BIN_DIR%\llama-cli.exe" (
    copy /Y "%BIN_DIR%\llama-cli.exe"          "llama-cli.exe"          >nul
    copy /Y "%BIN_DIR%\llama-bench.exe"         "llama-bench.exe"        >nul 2>&1
    copy /Y "%BIN_DIR%\llama-perplexity.exe"    "llama-perplexity.exe"   >nul 2>&1
    echo [ OK ] Binaries copied to project folder
) else (
    echo [WARN] Could not find binaries at %BIN_DIR%
    echo        Check build output above.
)

:: ── Download model ────────────────────────────────────────────────────────────
echo.
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

if exist "%MODEL_FILE%" (
    echo [ OK ] Model already downloaded: %MODEL_FILE%
) else (
    echo [SETUP] Downloading Qwen3-1.7B Q4_K_M (~1.2 GB)...
    echo         Do not close this window.
    echo.
    if %USE_PS_DOWNLOAD%==1 (
        powershell -Command "Invoke-WebRequest -Uri '%MODEL_URL%' -OutFile '%MODEL_FILE%'"
    ) else (
        curl -L --progress-bar -o "%MODEL_FILE%" "%MODEL_URL%"
    )
    if %errorlevel% neq 0 (
        echo [FAIL] Download failed. Check internet connection.
        pause & exit /b 1
    )
    echo [ OK ] Model saved to %MODEL_FILE%
)

:: ── Python deps ───────────────────────────────────────────────────────────────
echo.
echo [SETUP] Installing Python dependencies...
python -m pip install -q rich psutil matplotlib numpy tabulate
echo [ OK ] Python packages ready

:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo   ==================================================
echo     Setup complete!
echo   ==================================================
echo.
echo     Run the benchmark:
echo       python benchmark.py
echo.
echo     No binary yet? Test with simulated data:
echo       python benchmark.py --dry-run
echo.
echo     Open dashboard.html in your browser anytime.
echo   ==================================================
echo.
pause