@echo off
REM Launch the acquisition app (mock camera + real SmarAct MCS2 stage available)
REM using the 'midir' conda env. Uncheck "Simulate" in Stages>TWINS for the real stage.
cd /d "%~dp0"
call "%~dp0_find_conda.bat"
if not defined CONDACMD (
  echo [ERROR] conda not found. Open the Anaconda Prompt and run:
  echo         conda activate midir  ^&  python main.py --mode mock
  pause & exit /b 1
)
"%CONDACMD%" run -n midir --no-capture-output python main.py --mode mock
if errorlevel 1 pause
