@echo off
REM Standalone hyperspectral Z-series analysis app.
REM Optionally drag a folder or .npz files onto this file to open them.
cd /d "%~dp0"
".venv\Scripts\python.exe" analysis_app.py %*
pause
