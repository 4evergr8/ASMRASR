Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "          ASMRASR Runner (Resuming from Step 4 - MIRROR MODE)"
Write-Host "======================================================" -ForegroundColor Cyan

# 注入环境 - FFmpeg
$ffmpegPath = "$env:LOCALAPPDATA\ffmpegio\ffmpeg-downloader\ffmpeg\bin"
if (Test-Path $ffmpegPath) { $env:PATH = "$ffmpegPath;$env:PATH" }

# 注入镜像加速 - HuggingFace Mirror
$env:HF_ENDPOINT = "https://hf-mirror.com"
Write-Host "[INFO] Mirror Acceleration Enabled: hf-mirror.com" -ForegroundColor Yellow

$python = "python"
if (Test-Path "env\Scripts\python.exe") { $python = "env\Scripts\python.exe" }
elseif (Test-Path "venv\Scripts\python.exe") { $python = "venv\Scripts\python.exe" }

Write-Host "[INFO] Detected Step 1-3 are done. Skipping to Step 4..." -ForegroundColor Green

Write-Host "`n[4/4] 正在进行语音识别 (GPU 加速)..." -ForegroundColor Yellow
& $python d_语音识别.py

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "[DONE] 全部处理完成！请查看 3asr 文件夹。" -ForegroundColor Green
Read-Host "按回车键退出..."