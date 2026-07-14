@echo off
REM ============================================================================
REM  Build a SELF-CONTAINED HyperspectralAnalyzer.exe
REM
REM  Produces  dist\HyperspectralAnalyzer.exe  -- a single file that bundles
REM  Python, every library, and the TWINS calibration data. Your colleague just
REM  double-clicks it (or drags a results folder / .npz file onto it). Nothing
REM  needs to be installed on their PC.
REM
REM  Just run this file once on THIS machine.
REM ============================================================================
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo ERROR: %PY% not found. Run this from the gui\ folder of the project.
    pause & exit /b 1
)

echo.
echo [1/2] Installing PyInstaller into the build venv (one-time)...
"%PY%" -m pip install --quiet --upgrade pyinstaller || (echo pip failed & pause & exit /b 1)

echo.
echo [2/2] Building HyperspectralAnalyzer.exe  (takes a few minutes)...
"%PY%" -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name HyperspectralAnalyzer ^
  --add-data "Twins\ASRC calibration\parameters_cal.txt;Twins\ASRC calibration" ^
  --add-data "Twins\ASRC calibration\parameters_int.txt;Twins\ASRC calibration" ^
  --collect-submodules instruments ^
  --hidden-import pyqtgraph.exporters ^
  --hidden-import PIL.Image ^
  --hidden-import scipy.cluster.vq ^
  --hidden-import scipy.signal ^
  --hidden-import scipy.interpolate ^
  --hidden-import scipy.ndimage ^
  --hidden-import pandas ^
  --exclude-module matplotlib ^
  --exclude-module tkinter ^
  --exclude-module PyQt5 ^
  --exclude-module PySide6 ^
  analysis_app.py
if errorlevel 1 (echo BUILD FAILED & pause & exit /b 1)

echo.
echo ============================================================================
echo  DONE.  Give your colleague this single file:
echo      %CD%\dist\HyperspectralAnalyzer.exe
echo  They can double-click it, or drag a results folder / .npz onto it.
echo ============================================================================
pause
