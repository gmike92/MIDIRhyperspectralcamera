@echo off
REM Stokes polarimetry app: pick 4 QWP measurements (one per angle),
REM average over a wavelength range, compute/plot the Stokes parameters.
REM Optionally drag up to 4 .npz files onto this file to preload the slots.
cd /d "%~dp0"
".venv\Scripts\python.exe" stokes_app.py %*
pause
