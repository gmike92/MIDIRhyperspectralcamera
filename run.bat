@echo off
REM IRC806 MWIR live viewer
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py --mode irc806 --fps 120
pause
