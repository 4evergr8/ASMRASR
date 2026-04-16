# ======================================================
#           ASMRASR 启动器 (NVIDIA GPU 加速)
# ======================================================
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "          ASMRASR 启动器 (NVIDIA GPU 加速)" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[提示] 正在初始化环境，准备启动 ASR 引擎..." -ForegroundColor Gray
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "          ASMRASR Runner (NVIDIA GPU Acceleration)"
Write-Host "======================================================" -ForegroundColor Cyan

# 1. 自动注入 FFmpeg 路径
$ffmpegPath = "$env:LOCALAPPDATA\ffmpegio\ffmpeg-downloader\ffmpeg\bin"
if (Test-Path $ffmpegPath) {
    $env:PATH = "$ffmpegPath;$env:PATH"
    Write-Host "[INFO] FFmpeg path injected: $ffmpegPath" -ForegroundColor Gray
}

# 2. 定位 Python
$python = "python"
if (Test-Path "env\Scripts\python.exe") { $python = "env\Scripts\python.exe" }
elseif (Test-Path "venv\Scripts\python.exe") { $python = "venv\Scripts\python.exe" }

Write-Host "[INFO] Using: $python"

# 3. 执行流程
Write-Host "`n[1/4] 正在提取音频..." -ForegroundColor Yellow
& $python a_提取音频.py

Write-Host "`n[2/4] 正在分离人声 (GPU 加速)..." -ForegroundColor Yellow
& $python b_分离人声.py

Write-Host "`n[3/4] 正在进行人声检测..." -ForegroundColor Yellow
& $python c_人声检测.py

Write-Host "`n[4/4] 正在进行语音识别 (GPU 加速)..." -ForegroundColor Yellow
& $python d_语音识别.py

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "[DONE] 全部处理完成！请查看 3asr 文件夹。" -ForegroundColor Green
Read-Host "按回车键退出..."