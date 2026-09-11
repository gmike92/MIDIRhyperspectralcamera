@echo off
REM MCS2 stage + SR865A lock-in step-scan acquisition
cd /d "%~dp0"
".venv\Scripts\python.exe" main_lockin.py %*
pause
