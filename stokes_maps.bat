@echo off
REM Stokes-from-hypercubes app: load the THREE measurement .npz hypercubes (M1,
REM M2, M3) and optionally a 4th for phasing. Pick a wavelength with the slider;
REM the app runs the analysis_app Phase-panel DFT to show amplitude (row 1),
REM phase in units of pi (row 2), and the phase-free Stokes maps (row 3).
cd /d "%~dp0"
".venv\Scripts\python.exe" stokes_maps_app.py %*
pause
