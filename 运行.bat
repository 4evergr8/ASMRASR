@echo off
chcp 65001
setlocal
.venv\Scripts\python.exe a_提取音频.py
env\python.exe b_分离人声.py
env\python.exe c_人声检测.py
env\python.exe d_语音识别.py
popd
pause
endlocal