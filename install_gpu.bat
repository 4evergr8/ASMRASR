@echo off
setlocal
cls

echo ======================================================
echo           ASMRASR NVIDIA GPU Installer v2
echo ======================================================
echo.

set PYTHON_EXE=
for %%p in ("env\Scripts\python.exe" "env\python.exe" "venv\Scripts\python.exe" "python.exe") do (
    if not defined PYTHON_EXE if exist "%%~p" set "PYTHON_EXE=%%~p"
)

if "%PYTHON_EXE%"=="" (
    set "PYTHON_EXE=python"
    echo [WARNING] No venv found, using system python.
)

echo [INFO] Using: %PYTHON_EXE%

:menu
echo 1. CUDA 12.1 (Stable - Recommended)
echo 2. CUDA 12.4 (Latest)
echo 3. CUDA 11.8 (Old)
echo 4. Just dependencies
echo Q. Quit
set /p choice=Choice: 

if /i "%choice%"=="1" (set "URL=https://download.pytorch.org/whl/cu121" & goto start_install)
if /i "%choice%"=="2" (set "URL=https://download.pytorch.org/whl/cu124" & goto start_install)
if /i "%choice%"=="3" (set "URL=https://download.pytorch.org/whl/cu118" & goto start_install)
if /i "%choice%"=="4" goto deps_only
if /i "%choice%"=="q" exit /b
goto menu

:start_install
echo [INFO] Step 1/3: Installing generic dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo [INFO] Step 2/3: Installing GPU Torch components (Force)...
"%PYTHON_EXE%" -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url %URL%

echo [INFO] Step 3/3: Installing GPU Accelerators...
"%PYTHON_EXE%" -m pip install --upgrade onnxruntime-gpu faster-whisper
goto finish

:deps_only
"%PYTHON_EXE%" -m pip install -r requirements.txt
goto finish

:finish
echo.
echo ======================================================
echo [CHECK] Final GPU Test...
"%PYTHON_EXE%" -c "import torch; print('---'); print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU handle:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('---')"
echo ======================================================
echo.
pause
exit /b