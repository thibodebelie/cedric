@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "VENV_PY=%SCRIPT_DIR%venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  set "PY=%VENV_PY%"
) else (
  set "PY=python"
)

"%PY%" "%SCRIPT_DIR%run_all.py" %*
