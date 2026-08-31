@echo off
REM Stokes VIEWER: load the M1/M2/M3 (+ phasing / I45) folders, press
REM "Compute all z" once, then scrub the z and wavelength sliders instantly.
REM Same layout as stokes_maps_app, but the DFTs are precomputed for every z so
REM the maps, mask, phase-x-amp and export update with no recomputation.
cd /d "%~dp0"
".venv\Scripts\python.exe" stokes_viewer_app.py %*
pause
