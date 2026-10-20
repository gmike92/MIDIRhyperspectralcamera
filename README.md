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

# VIS-NIR Hyperspectral Camera

A PyQt6 acquisition + analysis application for **static hyperspectral
imaging**: an area camera combined
with a **NIREOS TWINS** common-path birefringent interferometer. The TWINS wedge
is stepped, N frames are grabbed per step, and a per-pixel interferogram is built
and Fourier-transformed into a spectral cube. Spectral band of interest **0.35–3.5 µm**.

Optional motorized axes extend a single map into a series: a **Thorlabs delay
stage** (Z) and a **Thorlabs rotation mount** (angle), both on KDC101 K-Cube
controllers, can be scanned to acquire a hypercube at every (Z, angle) point.

> A synthetic **mock** camera lets the whole app run with no
> hardware — `--mode mock` needs nothing installed.

## Features

**Live camera**
- Live image with Inferno / Viridis / Magma / Grey / Turbo / Coolwarm colormaps.
- Auto or fixed Min/Max colorbar; click a pixel for horizontal/vertical profiles.
- Integration time 0.01–8 ms, averaging, background capture/subtraction, snapshot.
- On-image draggable **ROI** applied as the camera subarray, plus N×N camera binning.
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

  main.py               entry point
  worker_camera.py      camera worker process (frames -> shared memory + queue)
  camera/               camera abstraction + backends
    camera_interface.py   CameraInterface ABC + CameraStatus/MeasurementResult
    irc806_camera.py      IRC806 backend (Pleora eBUS .NET via pythonnet)
    mock_camera.py        synthetic drifting-beam camera (no hardware)
    hamamatsu_backend.py 
    orca_camera.py
    factory.py            create_camera(mode)
  ui/
    main_window.py        orchestrator: live view, controls, background, ROI, save
    stages.py             delay + rotator + TWINS control panels
    twins_scan.py         live 1-D TWINS interferogram scan
    measure_hyperspectral.py  the hyperspectral experiment + HyperViewer
  instruments/          drivers + shared DSP
    stage_driver.py       Thorlabs delay stage (KDC101)
    rotator_stage.py      Thorlabs rotation mount (KDC101, degrees)
    twins_stage.py        NIREOS TWINS SmarAct SCU3D driver
    subtwinslv.py         step-scan engine (scan / scan_cube)
    hyperspectral.py      2-D per-pixel DFT (compute_hyperspectral)
    spectrum_processor.py 1-D interferogram -> spectrum
    calibration.py dsp.py analysis.py   shared processing
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
Clone this branch folder. 
Create virtual environment (e.g., midir) with the necessary libraries:
```bat
cd MIDIRhyperspectralcamera
conda create -n midir python=3.12
conda activate midir
conda install -c conda-forge pyqt        :: brings a self-consistent Qt6
pip install pyqtgraph numpy scipy pandas pillow tifffile loguru cffi
pip install "C:\your_Smaract_folder\smaract_ctl-1.6.2.zip"  :: change your_Smaract_folder with your directory for the smaract_ctl-1.6.2.zip filexx
```

Python 3.8–3.12, 64-bit. `mock` mode needs only the pip packages. The real
hardware additionally needs, installed system-wide:

- **Pleora eBUS runtime** — `C:\Program Files\Common Files\Pleora\eBUS SDK\PvDotNet.dll` (camera)
- **Thorlabs Kinesis** — `C:\Program Files\Thorlabs\Kinesis` (delay stage + rotator, KDC101)
- **SmarAct SCU3D** — `SCU3DControl.dll` (TWINS wedge)
- **Hamamatsu Orca**
- **Smaract MSC2** - 'C:\SmarAct\MCS2\SDK\Python\packages\smaract_ctl-1.6.2.zip'

## Moving the app to another computer or rig

Nothing in this app auto-detects which bench it is running on. Serial numbers,
SDK folders, travel limits and the TWINS calibration are **hard-coded for the
bench it was written on**, so a new PC, a different pair of Thorlabs
controllers or another TWINS unit needs the edits below.

Work through the sections in order. Each one says *what* to change, *where* it
lives, and *how to find the right value for your rig*. You can run
`python main.py --mode mock` at any point: mock mode ignores all of it.

### 1. Thorlabs serial numbers (rotator + delay stage)

The rig has **two KDC101 K-Cube controllers** — one for the rotation mount, one
for the delay stage. The rotator is pinned to its serial so the two can never
grab each other's controller; the delay stage then takes whichever other KDC101
is present.

| What | Where | Value on the original bench |
|---|---|---|
| Rotator serial | `ui/stages.py`, in `StagesPanel.__init__` (~line 88) | `RotatorStage(serial="27267060")` |
| Delay-stage serial | not pinned — `_delay_connect()` in `ui/stages.py` auto-picks the first KDC101 that is *not* the rotator | — |

**Find your serials.** Close the Kinesis app first (see section 3 — USB access
is exclusive), then:

```bat
python -c "from pylablib.devices import Thorlabs; print(Thorlabs.list_kinesis_devices())"
```

This prints `(serial, description)` for every attached Thorlabs controller. The
serial is also printed on the label on the KDC101 itself and shown in the
Kinesis device list. Put the rotator's serial into `ui/stages.py`.

If your bench has **more than two** KDC101 controllers, pin the delay stage as
well: in `_delay_connect()` pass the serial explicitly, e.g.
`self.delay.connect(simulate=sim, serial="12345678", exclude=other)`.

> **Symptom of a wrong serial:** `[RotatorStage] no hardware found` even though
> the Kinesis app moves the mount perfectly. The driver tried to open a serial
> that is not attached.

### 2. Actuator type assigned to each controller

A KDC101 only knows its counts-per-degree (or per-mm) once an **actuator is
assigned to it in the Kinesis app** — select the controller, then
*Settings → Device*, and pick the actuator: the rotation mount (e.g. `PRM1Z8`)
or the delay actuator (e.g. `Z825B`). This is a one-time setup per controller,
per PC.

The same name must then appear in the code, where it is pushed to the
controller as `DeviceSettingsName`:

| What | Where | Default |
|---|---|---|
| Rotation-mount type | `RotatorStage.__init__`, `instruments/rotator_stage.py` (~line 49) | `stage_name="PRM1-Z8"` |
| Delay-actuator counts/mm | `DEFAULT_COUNTS_PER_MM`, `instruments/stage_driver.py` (~line 43) | `20000` |

> **Symptoms of a wrong actuator:** "Go to" lands at the wrong angle, jog moves
> by random amounts, or the position readout shows huge numbers — the values are
> raw encoder counts, not degrees/mm. With no actuator assigned at all,
> connecting raises a configuration `NullReference`.

### 3. Thorlabs libraries: which backend, which folder, which bitness

The two Thorlabs drivers try three backends in order and use the first that
works: **pylablib → thorlabs_apt → pythonnet (.NET Kinesis DLLs)**.

**The simplest setup is `pip install pylablib`.** It talks to the KDC101 over
the FTDI virtual COM port, needs none of the Kinesis .NET DLLs, and avoids the
bitness trap below entirely. The USB driver that comes with the Kinesis install
is enough.

If you rely on the **.NET backend** instead, two things must line up:

1. **The folder.** Both drivers hard-code
   `C:\Program Files\Thorlabs\Kinesis` — see `_connect_dotnet()` in
   `instruments/rotator_stage.py` (~line 113) and `instruments/stage_driver.py`
   (~line 115). The **32-bit** Kinesis installs to
   `C:\Program Files (x86)\Thorlabs\Kinesis` instead; if that is what you have,
   both lines must be changed.
2. **The bitness.** A 64-bit Python **cannot** load the 32-bit Kinesis DLLs, no
   matter how correct the path is. With 64-bit Python (the normal case here),
   install the **64-bit Kinesis**, which lands in
   `C:\Program Files\Thorlabs\Kinesis` — the path the code already expects.

> **Symptom of a missing or mismatched Kinesis:**
> `_connect_dotnet failed: Could not load file or assembly
> 'Thorlabs.MotionControl.DeviceManagerCLI'`. Together with
> `No module named 'pylablib'` and `No module named 'thorlabs_apt'` on the lines
> above it, you get `no hardware found` with the hardware sitting right there.

> **Always close the Kinesis app before connecting from Python.** Thorlabs USB
> access is exclusive: while Kinesis (or APT User) holds the controller, Python
> sees no devices at all. Check the system tray and Task Manager — it can keep
> running after its window is closed.

### 4. SmarAct MCS2 (TWINS wedge stage)

| What | Where | Value on the original bench |
|---|---|---|
| `smaract.ctl` Python package | installed from the MCS2 SDK zip; the path differs per PC | `C:\SmarAct\MCS2\SDK\Python\packages\smaract_ctl-1.6.2.zip` |
| Controller locator | auto-discovered via `FindDevices()`; pass an explicit locator to `connect()` if several MCS2 units are attached | e.g. `usb:sn:MCS2-00000123` |
| Wedge channel index | `DEFAULT_CHANNEL`, `instruments/twins_stage.py` (~line 71) | `0` |
| Travel limits (mm) | `TRAVEL_MIN_MM` / `TRAVEL_MAX_MM`, same file (~lines 57–58) | `-13.2` / `18.8` |
| Park positions (mm) | `HOME_POSITION_MM` / `SAFE_POSITION_MM` (~lines 64–65) | `0.0` / `0.0` |
| Move velocity / acceleration | `DEFAULT_VELOCITY_MM_S` / `DEFAULT_ACCEL_MM_S2` (~lines 68–69) | `3.0` mm/s, `10.0` mm/s² |

The travel limits are **enforced in software only** — this controller reports no
range limit of its own, so a wrong value here lets the positioner drive into its
end stop. Measure the real end points on your rig before the first scan. Both
park positions must lie inside the limits; that is checked at import.

### 5. Camera

| What | Where | Note |
|---|---|---|
| Default backend | `--mode` in `main.py` (`orca`, `irc806`, `mock`, `auto`) | `run.bat` launches `--mode irc806`; the Connection-mode dropdown switches at runtime |
| Hamamatsu Orca | needs the DCAM runtime (`dcamapi.dll`) installed system-wide | test it on its own with `python diag_dcam.py` |
| IRCameras IRC806 | `EBUS_DIR` in `camera/irc806_camera.py` (~line 23) | `C:\Program Files\Common Files\Pleora\eBUS SDK`; close eBUS Player first — it holds the camera exclusively |

### 6. TWINS calibration files — do this before trusting any spectrum

Both live in `Twins/ASRC calibration/` and both are **tab-separated with exactly
two rows of equal length**:

| File | Row 0 | Row 1 |
|---|---|---|
| `parameters_cal.txt` | wavelength (µm) | stage spatial frequency (cycles per mm of wedge travel, mm⁻¹) |
| `parameters_int.txt` | nominal stage position (mm) | reference interferogram of a known source |

`parameters_cal.txt` maps the stage Fourier frequencies onto real wavelengths;
`parameters_int.txt` removes the wedge motor's reproducible nonlinearity. The
files shipped in the repo belong to the original interferometer and detector, so
they **will give wrong wavelengths on another TWINS unit**.

The loader parses only the first two rows and rejects anything else: a file in
the wrong layout prints a `[WARN]` line and the app falls back to a plain
`1/frequency` conversion instead of hanging. Mistakes it catches are data
written in columns instead of rows, a header line, and the two arrays
concatenated into a single row.

There is also a stale fallback path in `instruments/spectrum_processor.py`
(~line 96) pointing at another user's Desktop. It is harmless, because the
package-relative path is tried first, but you may want to delete it.

### 7. Save folders

These defaults point at drives that may not exist on your PC. All of them can be
changed in the GUI at run time; these are the values the app starts with:

| Default | Where |
|---|---|
| `C:\CAMERA` | `ui/main_window.py` (~line 69) — camera snapshots, and the root the Measure tab saves under |
| `C:\CAMERA\twins` | `ui/twins_scan.py` (~line 40) — 1-D TWINS scans |
| `D:\CAMERA\hyperspectral` | `ui/measure_hyperspectral.py` (~line 501) — fallback when the camera folder is unset |
| `C:\CAMERA\spectral_calib` | `spectral_calib_smaract.py` (~line 78) |
| `D:\CAMERA` | opening folder of the file dialogs in `stokes_app.py` and `stokes_maps_app.py` (used only if it exists) |

### 8. Spectral band, scan range and ZPD

These defaults describe the original bench and must match **your** optics,
detector and TWINS unit before results mean anything. They live in
`instruments/hyperspectral.py` (~lines 17–38) and can also be set per
measurement in the *TWINS* and *Measure* tabs:

| Constant | Default | Meaning |
|---|---|---|
| `DEFAULT_WL_START` / `DEFAULT_WL_STOP` | `0.4` / `3.5` µm | spectral window of the reconstruction |
| `DEFAULT_START_MM` / `DEFAULT_STOP_MM` | `0` / `2` mm | wedge scan range |
| `DEFAULT_N_STEPS` | `100` | steps per scan |
| `DEFAULT_ZPD_MM` | `-6` mm | zero-path-difference position of your interferometer |
| `DEFAULT_ZPD_WINDOW_MM` | `0.1` mm | search window around the ZPD |

### 9. Values the app remembers per Windows user

Scan parameters, wavelength windows and colormap choices are stored with
`QSettings` under the organisation **`MIR_CAMERA`** (in the registry, under
`HKEY_CURRENT_USER\Software\MIR_CAMERA`). They are not part of the repo, so a
fresh PC starts from the defaults above — and conversely, a PC you have used
before keeps your previous numbers even after you edit the defaults in the code.
Delete that registry key to start clean.

### 10. Check it before a measurement

```bat
python main.py --mode mock                 REM GUI works, no hardware needed
python diag_dcam.py                        REM Hamamatsu DCAM sees the camera
python -c "from pylablib.devices import Thorlabs; print(Thorlabs.list_kinesis_devices())"
python -c "import smaract.ctl as ctl; print(ctl.FindDevices())"
```

Then, in the app: connect each stage with **Simulate** unticked, home the
rotator and the delay stage, and confirm the position readouts are in degrees
and millimetres rather than huge raw counts. Finally run one short TWINS scan
and check that the spectrum peaks where your source actually emits — that is the
quickest test that `parameters_cal.txt` belongs to your interferometer.

## Run

```bat
run.bat                                       REM main.py --mode irc806 --fps 120
.venv\Scripts\python main.py --mode mock      REM no camera/hardware needed
view.bat  path\to\cube.npz                    REM standalone cube viewer
analyze.bat                                   REM standalone Z-series analyzer
```
Alternative

```bat
python main.py --mode mock  :: python main.py if real camera is available for connection
```
