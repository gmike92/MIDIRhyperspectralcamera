# Hyperspectral Camera — Hamamatsu Orca + TWINS (SmarAct MCS2)

> **This build is an adaptation of the MIDIR app for a different bench.**
> It keeps the full TWINS Fourier-transform hyperspectral workflow, DSP,
> HyperViewer and analysis tools unchanged, and swaps only the two hardware
> backends:
>
> | | Original (MIDIR) | This build |
> |---|---|---|
> | Camera | IRCameras IRC806 (MWIR, 14-bit) | **Hamamatsu Orca Flash** (sCMOS, 16-bit) via DCAM |
> | TWINS wedge stage | SmarAct **SCU3D** (ctypes) | SmarAct **MCS2** / SLC-2460 (`smaract.ctl`, closed-loop, pm) |
>
> **Run it now with no hardware:** `python main.py --mode mock` — the camera is a
> synthetic beam and every stage defaults to its Simulate mode, so the whole
> acquire → per-pixel FFT → viewer path works offline.
>
> **What you must supply for real measurements** (the DSP is otherwise unchanged):
> 1. **Your TWINS calibration files** — `Twins/ASRC calibration/parameters_cal.txt`
>    (stage position → wavelength) and `parameters_int.txt` (motor nonlinearity).
>    The ones in the repo are for the original MWIR TWINS unit and will give wrong
>    wavelengths for your interferometer/detector.
> 2. **The correct spectral band + ZPD** for your Orca+TWINS. The Orca is a
>    visible/NIR sCMOS camera, so the app's default λ range (**3.8–4.4 µm**, MWIR)
>    and ZPD/scan-range defaults in the *TWINS* and *Measure* tabs must be set to
>    your actual band before results are meaningful.
> 3. **MCS2 wiring** — the wedge channel index / device locator and the
>    HOME/SAFE positions in `instruments/twins_stage.py`.
>
> Everything below describes the original MIDIR app; it still applies except for
> the camera/stage specifics above.

---

# MIDIR Hyperspectral Camera

A PyQt6 acquisition + analysis application for **static hyperspectral
imaging**: an area camera combined
with a **NIREOS TWINS** common-path birefringent interferometer. The TWINS wedge
is stepped, N frames are grabbed per step, and a per-pixel interferogram is built
and Fourier-transformed into a spectral cube. MWIR band of interest **3.8–4.4 µm**.

Optional motorized axes extend a single map into a series: a **Thorlabs delay
stage** (Z) and a **Thorlabs rotation mount** (angle), both on KDC101 K-Cube
controllers, can be scanned to acquire a hypercube at every (Z, angle) point.

> The camera talks GigE Vision through the **Pleora eBUS** .NET API (via
> pythonnet). A synthetic **mock** camera lets the whole app run with no
> hardware — `--mode mock` needs nothing installed.

## Features

**Live camera**
- Live image with Inferno / Viridis / Magma / Grey / Turbo / Coolwarm colormaps.
- Auto or fixed Min/Max colorbar; click a pixel for horizontal/vertical profiles.
- Integration time 0.01–8 ms, averaging, background capture/subtraction, snapshot.
- On-image draggable **ROI** + binning, shared by the measurement panels.
- FPGA/board + FPA temperature readout (serial-over-GigE), NUC/BPR correction.

**Stages**
- **Thorlabs Stage** (delay, mm) and **Rotator** (deg) tabs — connect / home /
  go-to / jog, each with a no-hardware "Simulate" mode.
- **TWINS** (SmarAct SCU3D wedge) tab with a live 1-D interferogram scan.

**Hyperspectral measurement (Measure tab)**
- Steps the TWINS wedge, grabs frame stacks, computes a **per-pixel DFT** →
  spectral cube, motor-nonlinearity calibrated, auto-saved per position.
- **Z-scan** and **angle-scan**, each specified as up to three enable-able
  **zones** (independent spacing), acquired as a nested **Z × angle grid**.
- Built-in **HyperViewer**: λ-scrub / peak-λ / peak-intensity / SAM /
  **continuum-line** maps, per-pixel spectra, colormaps.

**Analysis**
- A standalone Z-series analyzer (`analysis_app.py`) and a lightweight cube
  viewer (`view_hyperspectral.py`) for the saved `.npz` hypercubes.

## Layout

```
gui/
  main.py               entry point  (--mode irc806|mock|auto, --fps N)
  worker_camera.py      camera worker process (frames -> shared memory + queue)
  camera/               camera abstraction + backends
    camera_interface.py   CameraInterface ABC + CameraStatus/MeasurementResult
    irc806_camera.py      IRC806 backend (Pleora eBUS .NET via pythonnet)
    mock_camera.py        synthetic drifting-beam camera (no hardware)
    factory.py            create_camera(mode)
  ui/
    main_window.py        orchestrator: live view, controls, background, ROI, save
    stages.py             delay + rotator + TWINS control panels
    twins_scan.py         live 1-D TWINS interferogram scan
    measure_kspace.py     the hyperspectral experiment + HyperViewer
  instruments/          drivers + shared DSP
    stage_driver.py       Thorlabs delay stage (KDC101)
    rotator_stage.py      Thorlabs rotation mount (KDC101, degrees)
    twins_stage.py        NIREOS TWINS SmarAct SCU3D driver
    subtwinslv.py         step-scan engine (scan / scan_cube)
    hyperspectral.py      2-D per-pixel DFT (compute_hyperspectral)
    spectrum_processor.py 1-D interferogram -> spectrum
    calibration.py dsp.py analysis.py walkoff.py   shared processing
  Twins/ASRC calibration/ parameters_{cal,int}.txt  spectral + motor calibration
  analysis_app.py       standalone Z-series hyperspectral analyzer
  view_hyperspectral.py standalone cube viewer
  docs/                 ACQUISITION_APP.md (architecture) + CONTINUUM_SUBTRACTION.md
```

See **[docs/ACQUISITION_APP.md](docs/ACQUISITION_APP.md)** for a full technical
walk-through of the architecture, and
**[docs/CONTINUUM_SUBTRACTION.md](docs/CONTINUUM_SUBTRACTION.md)** for the
per-pixel continuum-subtraction method used to isolate the resonant line image.

## Setup (one time)

```bat
cd gui
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Python 3.8–3.12, 64-bit. `mock` mode needs only the pip packages. The real
hardware additionally needs, installed system-wide:

- **Pleora eBUS runtime** — `C:\Program Files\Common Files\Pleora\eBUS SDK\PvDotNet.dll` (camera)
- **Thorlabs Kinesis** — `C:\Program Files\Thorlabs\Kinesis` (delay stage + rotator, KDC101)
- **SmarAct SCU3D** — `SCU3DControl.dll` (TWINS wedge)

## Run

```bat
run.bat                                       REM main.py --mode irc806 --fps 120
.venv\Scripts\python main.py --mode mock      REM no camera/hardware needed
view.bat  path\to\cube.npz                    REM standalone cube viewer
analyze.bat                                   REM standalone Z-series analyzer
```

## Hardware notes

- The camera is **single-access**: close eBUS Player first. After stopping, wait
  ~10 s for the GigE heartbeat to clear before relaunching, and close the app
  with its window **X** (force-killing can orphan the worker process).
- The IRC806 `ExposureTime` node is in **seconds** with no valid firmware
  min/max, so the UI hard-clamps integration to 0.01–8 ms (a huge value freezes
  the stream). Frames are uint16 with **14-bit** data (saturation = 16383).
- Both the delay stage and rotator use **KDC101** controllers, so each is pinned
  to its serial to avoid grabbing the other's device. Assign the actuator
  (Z825B / PRM1-Z8) to its controller in the Kinesis app first, or moves come out
  in raw device units.
- Camera IP is auto-discovered; no IP needs to be set.
