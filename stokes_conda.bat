@echo off
REM Stokes-polarimetry app (stokes_app.py) via the 'midir' conda env. Args pass through.
cd /d "%~dp0"
call "%~dp0_find_conda.bat"
if not defined CONDACMD (
  echo [ERROR] conda not found. Open the Anaconda Prompt and run:
  echo         conda activate midir  ^&  python stokes_app.py %%*
  pause & exit /b 1
)
"%CONDACMD%" run -n midir --no-capture-output python stokes_app.py %*
if errorlevel 1 pause
