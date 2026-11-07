@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run the virtual-environment setup in docs\BUILD_GUIDE_JA.md first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe app.py
if errorlevel 1 pause
