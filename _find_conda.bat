@echo off
REM Helper: locate a usable conda and set CONDACMD. Used by the *_conda.bat launchers.
REM Leaves CONDACMD empty (and returns errorlevel 1) if conda can't be found.
set "CONDACMD="

REM 1) conda already on PATH (e.g. launched from an initialized shell)?
where conda >nul 2>&1 && ( set "CONDACMD=conda" & exit /b 0 )

REM 2) conda has exported CONDA_EXE into the environment?
if defined CONDA_EXE if exist "%CONDA_EXE%" ( set "CONDACMD=%CONDA_EXE%" & exit /b 0 )

REM 3) search the usual install locations
for %%R in (
  "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3"
  "%USERPROFILE%\Anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3"
  "C:\ProgramData\anaconda3" "C:\ProgramData\miniconda3"
) do if exist "%%~R\Scripts\conda.exe" ( set "CONDACMD=%%~R\Scripts\conda.exe" & exit /b 0 )

exit /b 1
