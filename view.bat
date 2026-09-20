@echo off
REM Standalone hyperspectral viewer. Optional: drag a .npz onto this file.
cd /d "%~dp0"
".venv\Scripts\python.exe" view_hyperspectral.py %*
pause
