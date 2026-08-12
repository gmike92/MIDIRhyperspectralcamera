@echo off
REM Hypercube viewer (view_hyperspectral.py) via the 'midir' conda env. Args pass through.
cd /d "%~dp0"
call "%~dp0_find_conda.bat"
if not defined CONDACMD (
  echo [ERROR] conda not found. Open the Anaconda Prompt and run:
  echo         conda activate midir  ^&  python view_hyperspectral.py %%*
  pause & exit /b 1
)
"%CONDACMD%" run -n midir --no-capture-output python view_hyperspectral.py %*
if errorlevel 1 pause
