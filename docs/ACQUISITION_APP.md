# MIR_CAMERA — Acquisition App: technical handoff

Audience: an AI agent (or developer) taking over the **live acquisition app** under
`gui/`. This explains the architecture and the *logic*, not every line. Companion
docs: `memory.md` (running change log) and the analyzer is documented separately
(`gui/analysis_app.py` / `AnalysisApp/`).

> Scope: **static hyperspectral imaging** — an IRCameras **IRC G806 MWIR** camera +
> a **NIREOS TWINS** common-path birefringent interferometer. Step the TWINS wedge,
> grab N frames per step, build an interferogram per pixel, per-pixel DFT → spectral
> cube. MWIR band of interest **3.8–4.4 µm**; TWINS ZPD (centerburst) ≈ **24.33 mm**.
> NOT pump-probe (no lock-in). The Thorlabs delay stage is an optional Z-series axis.

Line numbers below are anchors at time of writing; trust **names** over numbers.

---

## 1. Big picture

```
main.py  ──spawns──►  camera_worker (separate PROCESS)  ──► real IRC806 (eBUS/.NET) or MockCamera
   │                        │  frames → shared memory + frame_queue;  status → frame_queue
   │ builds                 │  commands ◄── control_queue
   ▼                        ▼
MainWindow (ui/main_window.py)  ── QTimer 8 ms poll → self.latest_frame (fresh .copy each frame)
   ├── Camera tab      (exposure, NUC, ROI, colormap, background capture, save image)
   ├── Thorlabs tab    (StagesPanel.delay_group)
   ├── TWINS tab       (StagesPanel.twins_group  +  TwinsScanPanel: live 1-D scan)
   └── Measure tab     (MeasurePanel: the K-space HYPERSPECTRAL experiment)
                          frame_source = lambda: self.latest_frame
                          bg_provider, save_dir_provider, meta_provider, roi_provider
```

Three subsystems:
1. **Camera** — its own OS process; frames via shared memory + a queue; commands via a queue.
2. **Main window** — polls frames, displays them, owns `latest_frame`, exposes camera controls, background, ROI, save dir/filename, and the camera-metadata provider.
3. **Stages + acquisition** — TWINS wedge driver, optional Thorlabs Z stage, the live 1-D TWINS scan, and the **MeasurePanel** hyperspectral experiment (the main scientific output).

Two processing depths share the same calibration + math:
- **1-D** (`instruments/spectrum_processor.py`) — one ROI-mean value per wedge step → one spectrum (the TWINS tab live scan).
- **2-D** (`instruments/hyperspectral.py`) — full ROI image per wedge step → per-pixel spectral cube (the Measure tab).

---

## 2. Camera subsystem

Files: `worker_camera.py`, `camera/camera_interface.py`, `camera/factory.py`,
`camera/irc806_camera.py`, `camera/mock_camera.py`.

### Process & transport
- The camera runs in a **separate `multiprocessing.Process`** (`camera_worker()` in
  `worker_camera.py`), spawned by `main.py`. Rationale: GenICam/eBUS grabbing must not
  stall the Qt event loop.
- `main.py` creates: `frame_queue = mp.Queue(maxsize=4)`, `control_queue = mp.Queue()`,
  and a **`SharedMemory`** block sized `(4096, 4096)` uint16 (over-allocated to fit any
  ROI). Config dict carries `mode`, `target_fps`, and the shared-memory name/shape.
- **Frames → UI:** worker copies the frame into shared memory and pushes a packet
  `{"type":"frame","shape":(h,w),"shared":True,"measurement":{...}}` onto `frame_queue`.
  If shared memory is absent it embeds `"frame": ndarray` instead. Queue-full → drop oldest.
- **Status → UI:** same `frame_queue`, `{"type":"status","status":{...}}` (connected,
  acquiring, backend, serial, width/height, exposure_ms, average_count, board_temp_c,
  fpa_temp_k, message). Pushed when idle / after commands / on temp reads.
- **Commands UI → worker:** `control_queue`, dicts `{"type":..., "value":...}`:
  `start`, `pause`, `stop`, `set_exposure`, `set_average`, `connect`, `read_temp`,
  `load_state`, `set_correction` (NUC+BPR output mux), `set_bpr`, `set_nuc_slot`, `snapshot`.
- **Auto-heal:** if no frame for ~4 s while streaming, worker `camera.reconnect()`
  (throttled ~6 s). Temps refreshed every ~5 min.

### Abstraction & backends
- `camera/camera_interface.py` — `CameraInterface` ABC: `connect/disconnect/
  start_acquisition/stop_acquisition/get_frame/set_exposure/get_status` (+ optional
  `set_average`). Dataclasses `CameraStatus`, `MeasurementResult`.
- `camera/factory.py` — `create_camera(mode)`: `irc806`/`auto` → try `Irc806Camera`,
  fall back to `MockCamera` if eBUS/.NET import fails; `mock` → `MockCamera`.
- `camera/irc806_camera.py` — the real backend:
  - **Pleora eBUS 5.1.5** GenICam via **pythonnet** (`PvDotNet.dll` under
    `C:\Program Files\Common Files\Pleora\eBUS SDK`). Discover GigE → `PvDevice.CreateAndConnect`.
  - Frames: `PvStream` + queued buffers; `get_frame()` = `RetrieveBuffer` →
    `_image_to_numpy` → **uint16** (Mono16 container, **14-bit** data). FPA read from
    GenICam Width/Height.
  - **Exposure**: GenICam `ExposureTime` node is in **seconds** with bogus min/max →
    hard-clamp in software **0.01–8.0 ms** (default 0.3 ms); >8 ms freezes the stream.
  - **Temperature, NUC/BPR, state file**: not GenICam — tunneled over a **serial-over-GigE**
    register protocol (TX/RX regs `0xA400..0xA410`, CRC16, byte-stuffing). `refresh_temperatures()`
    reads board °C + FPA K with retries (GVSP starves GVCP). `set_correction(True)` selects
    the NUC+BPR-corrected output mux (this is what makes the live frame flat-field corrected).
    `load_state(xml)` loads a calibration profile from camera flash.
  - **Heartbeat** raised to 15 s to survive link glitches; `reconnect()` heals dropped links.
- `camera/mock_camera.py` — synthetic drifting Gaussian beam (320×256, clipped 12-bit),
  used when hardware/eBUS is unavailable. Lets the whole app run with no camera.

### Key camera facts for downstream code
- Frame dtype **uint16**, **14-bit** range → **saturation = 16383 counts**.
- FPA is read at runtime (IRC806 ~640×512 class; mock 320×256). Don't hard-code resolution.

---

## 3. Main window (`ui/main_window.py`) — the orchestrator

`MainWindow.__init__` attaches to the shared frame + queues, builds the UI, and starts
a **`QTimer` every ~8 ms** → `update_from_worker()`.

### Frame pipeline (the most important invariant)
- `update_from_worker()` drains `frame_queue`: status packets → `_apply_status`, frame
  packets → keep the latest. For a shared frame it slices `self.shared_frame[:h,:w].copy()`
  — **a fresh numpy array every poll**.
- `_apply_frame(frame, measurement)` sets **`self.latest_frame = frame`** (the single
  source of truth), handles background capture/subtraction for *display*, pushes the image
  to pyqtgraph, and every 3rd frame updates colorbar levels, X/Y profiles, crosshair,
  beam measurements, FPS.
- **Object-identity freshness:** because each new frame is a brand-new `.copy()`, scanners
  can detect "a new frame arrived" by checking `frame is last_frame`. The MeasurePanel /
  TwinsScanner averaging relies on this (see §6).
- **`frame_source = lambda: self.latest_frame`** is handed to both TwinsScanPanel and
  MeasurePanel. They call it whenever they need the current frame.

### Control surface → `control_queue`
Exposure slider/spin → `set_exposure`; averaging → `set_average`; Start/Pause/Snapshot;
Connect → `connect`; NUC checkbox → `set_correction`; BPR → `set_bpr`; NUC slot →
`set_nuc_slot`; Load state → `load_state`; Poll temp → `read_temp`. Colormap / auto-scale /
display range are display-only (don't touch the camera).

### ROI, background, save, metadata, status (the callbacks MeasurePanel depends on)
- **ROI**: a draggable cyan box on the image; `get_roi_bounds()` → `(r0,r1,c0,c1)` clipped,
  or `None` (full frame). Passed as `roi_provider`.
- **Background**: "Capture" averages `bg_average_frames` (16) into `self.background_frame`
  (float32). `bg_provider = lambda: (self.background_frame, self.use_bg_subtraction)`.
- **Save**: `save_dir_edit` (default `D:\CAMERA`) + `filename_edit`. `save_dir_provider`
  and the camera's "Save image" (TIFF uint16 + NPY + colormapped PNG, stamped
  `YYYYMMDD_HHMMSS.<name>`).
- **Metadata**: `meta_provider = self._kspace_metadata` returns
  `{camera_serial, exposure_ms, averaging, fpa_temp_k, board_temp_c, nuc_corrected,
  save_filename_camera}` from `self.latest_status` — embedded in saved hypercubes.
- **Status**: `_apply_status` stores `self.latest_status` and updates labels (temps shown
  as °C / K, NaN-guarded).

---

## 4. Stages (`ui/stages.py`, `instruments/twins_stage.py`, `instruments/stage_driver.py`)

`StagesPanel` owns both drivers and exposes two QGroupBoxes (placed in the Thorlabs and
TWINS tabs). It runs a 500 ms position-poll timer and wraps blocking moves in a
`StageController` (background thread + Qt signals) so the UI never freezes.

- **`self.twins` — `TwinsStage`** (NIREOS wedge, SmarAct **SCU3D** via `SCU3DControl.dll`,
  ctypes). `connect(home=True)` references and **parks at HOME 19 mm**; `disconnect(safe=False)`
  leaves the wedge in place. `move_to(mm)`, `move_by(mm)`, `wait_for_stop()` (waits for the
  move to *begin* then to settle — guards the async status lag), `get_position()`, `is_moving()`.
  - **Status-code gotcha (already fixed):** moving reports `2=MOVING`/`3=TARGETING`;
    `6=MOVING_TO_REFERENCE`. The original port used `STATUS_MOVING=6`, so `wait_for_stop`
    returned instantly and the scan recorded *during* motion. `_IN_MOTION=(1,2,3,5,6)` now.
  - `POSITION_SCALE = 10000` raw units/mm; `HOME_POSITION_MM = 19`.
- **`self.delay` — `DelayStage`** (Thorlabs **KDC101** K-Cube + **Z825B** actuator, Kinesis
  .NET via pythonnet). `connect`, `home`, `move_to_mm`, `move_relative_mm`, `get_position_mm`,
  `is_moving`, `is_homed`. **One-time setup**: the Z825B actuator must be assigned to the
  KDC101 in the Kinesis app, else `CreateKCubeDCServo` throws a config NullReference.
- **`StagesPanel.freeze(True/False)`** — pauses the poll timer and disables manual controls
  during a scan (avoids DLL contention). MeasurePanel/Twins scan call this around runs.

---

## 5. Live 1-D TWINS scan (`ui/twins_scan.py` + `instruments/subtwinslv.py` + `spectrum_processor.py`)

The **TWINS tab** does a quick scalar interferogram for alignment / single-spectrum.
- `TwinsScanPanel`: params (start/stop mm, steps, frames/point), live interferogram +
  spectrum plots, persisted via `QSettings("ts_*")`. `Scan` → `_start_scan()` builds a
  `TwinsScanner(twins, frame_source)` and runs it on the controller thread.
- `TwinsScanner.scan(...)` (`subtwinslv.py`): for each `linspace(start,stop,n)` target →
  `move_to` → `wait_for_stop` → settle → `_read_scalar(roi, frames_avg)` (averages **fresh**
  frames, drops the first in-flight one) → progress callback. Returns `(positions, ifg_1d)`.
- `compute_spectrum()` → `SpectrumProcessor.compute_spectrum()`: baseline removal →
  `calibrate_position_axis` (motor nonlinearity) → `find_centerburst` (Hilbert envelope ZPD,
  searched ±`search_mm` around 24.33) → apodization → explicit DFT (matrix multiply) →
  `_freq_to_wavelength` (via `parameters_cal.txt`). Same math as the 2-D path.

---

## 6. The hyperspectral experiment — `MeasurePanel` (`ui/measure_kspace.py`)

This is the scientific core. It runs a worker thread that drives the TWINS wedge (and
optionally the Thorlabs Z stage), grabs frame stacks, computes per-pixel spectra, and
auto-saves everything. Key classes in the file: `MeasurePanel`, `HyperViewer`,
`LiveInterferogram`; helpers `load_kspace_npz`, `kspace_metadata`.

### Acquisition flow (two-phase worker `_worker`)
A single **Acquire** = one *run*. At `_start`, a per-run folder
`<camera_folder>/<run_stamp>.<filename>/` is created (`run_stamp = YYYYMMDD_HHMMSS`
captured once; `_run_target()` returns `(folder, stamp, fname)`). Z targets =
`[None]` for a single map, or `linspace(z0,z1,zn)` for a Z-scan.

- **Phase 1 — ACQUIRE every Z position back-to-back (NO FFT between steps).**
  For each Z: move the Thorlabs stage → `TwinsScanner.scan_cube(...)` returns
  `(positions, datacube)` where `datacube` is `(n_pos, h, w)` over the **binned ROI**.
  Each step is **immediately safety-saved** raw-only & **uncompressed**
  (`_save_position_npz(..., compress=False)`, metadata `processing_stage="raw_acquired"`),
  so a later crash never loses acquired data.
- **Phase 2 — TRANSFORM every acquired cube + save.** For each: optional saturation mask,
  `resolve_n_points` (Auto = 1.5× steps, clamped [512,4096]), `compute_hyperspectral(...)`,
  optional SVD denoise, then **overwrite** the same file with the spectrum (compressed,
  `processing_stage="complete"`). Single maps autosave the whole cube; Z-scans keep the
  per-position files and `_on_done` skips the whole-set autosave.

`scan_cube` (`subtwinslv.py`): per step, `wait_for_stop` + settle, then `_read_roi_slice`
averages **distinct** frames (object-identity, drop first in-flight), subtracts the
captured **background** (full-frame, before ROI crop+bin) if enabled, crops to ROI, bins by
`bin_factor`. Records the **real measured** wedge position (`stage.get_position()`). On a
camera freeze it `_wait_for_stream` then re-acquires that step. `progress(i,tot,pos,value)`
drives the live interferogram preview.

### Calibration of the wedge axis (important)
- `compute_hyperspectral` **always** applies the motor-nonlinearity correction
  `calibrate_position_axis(positions)` (from `parameters_int.txt`) before the DFT, unless
  told the axis is already calibrated (`positions_calibrated=True`). This linearizes the
  reproducible wedge motor error (same method as the user's `Desktop/calibrate_piezo.py`).
- **Saved axes:** every file stores BOTH `twins_positions_mm` (raw measured) and
  `twins_positions_calibrated_mm` (the axis the DFT used). Metadata records
  `motor_calibration_applied`, `motor_calibration_file`, `motor_calibration_max_dev_um`,
  `position_axis_used` ("calibrated"/"raw_measured"). On reload the analyzer reads these to
  avoid **double-calibration** (feeds the calibrated axis with `positions_calibrated=True`).

### The per-pixel transform (`instruments/hyperspectral.py::compute_hyperspectral`)
- Operates on the **ROI (binned) cube only** — the full frame is never transformed.
- Preprocess: moving-average baseline removal; `find_centerburst` (signed spatial sum →
  Hilbert envelope, searched ±`search_mm` around `expected_zero_mm=24.33`); optional
  symmetrize. Apodization via `dsp.apodization_window` — now **asymmetric-aware** (per-wing
  taper to each edge, keeps the full long tail; `gaussian` is a separate NIREOS two-sided
  branch). **FT-window** selection: `ft_region` full/center/tails + `ft_width_mm`, or explicit
  `ft_window_mm=(lo,hi)` (boxcar if the window excludes the ZPD, else taper×window).
- DFT: a single **vectorised matrix multiply (NUDFT)** —
  `phase_kernel = exp(-2πj·pos·freq)`; `spec_flat = phase_kernel.conj().T @ (weighted pixels)`
  → `(n_freq, h·w)` → magnitude (`np.abs`, correct for spatially-varying ZPD). Output cube is
  **float32**. Frequency grid → wavelength via `parameters_cal.txt`
  (`_get_frequency_limits` / `_freq_to_wavelength`).
- Returns `(wavelengths (n_freq,), spectrum_cube (n_freq,h,w))`.

### Saved file schema (per-position `.npz`)
`wavelengths`, `spectrum_cube` (n_freq,h,w float32), `z_value_mm`, `z_unit`,
`raw_interferogram` (n_pos,h,w — the per-pixel interferogram, **always saved**),
`twins_positions_mm`, `twins_positions_calibrated_mm`, optional `saturation_mask`,
optional `background`+`background_subtracted`, and `metadata` (np object dict) +
`metadata_json`. Whole-cube/manual saves use the **stacked** form: `spectrum_cubes`
(n_maps,n_freq,h,w), `z_values`, `raw_interferograms`/`raw_positions`/`raw_positions_calibrated`.
Filenames: `<stamp>.<name>.npz` (single) or `<stamp>.<name>_z<pos>mm.npz` (per Z).

### Viewer
`HyperViewer` (in-app "Open Viewer" + standalone `view_hyperspectral.py`) shows the cube:
λ + Z sliders, maps (λ-slice/Peak-λ/Peak-intensity/SAM), colormap/gamma, pixel spectra,
and a calibration badge via `set_calibration_note(meta)`. `set_result` (in-RAM) /
`set_result_lazy` (per-Z lazy load). The big **analysis** app is separate (`analysis_app.py`).

### Metadata captured at scan start (`_scan_meta` + `_cam_meta`)
start/stop/steps/step_um, frames/point, binning, ROI, apodization+width, wl range,
n_freq setting, ZPD, walk-off, background_subtracted, saturation level, svd, zscan,
filename — plus the camera dict from `meta_provider`.

---

## 7. Shared processing modules (`instruments/`)

- `calibration.py` — loads `parameters_cal.txt` (spectral: row0 wavelength µm, row1
  reciprocal) and `parameters_int.txt` (motor: row0 position, row1 reference IFG), both under
  `gui/Twins/ASRC calibration/`. `calibrate_position_axis` (motor nonlinearity via
  analytic-signal phase), `position_calibration_status`. **Frozen-aware** (finds data via
  `sys._MEIPASS` when packaged).
- `dsp.py` — apodization window library (Happ-Genzel, Blackman-Harris 3/4, triangular,
  super-gaussian) **asymmetric-aware**; FWHM-based resolution estimate.
- `analysis.py` — cube analysis: `saturation_mask`, `roi_average`, `svd_denoise`,
  `svd_explained_variance`, peak maps, `spectral_derivative`, `spectral_angle_map`.
- `walkoff.py` — TWINS wedge walk-off correction (per-frame parametric shift +
  phase-correlation registration). Wired into `compute_hyperspectral(walkoff=...)`.
- `spectrum_processor.py` / `hyperspectral.py` — the 1-D and 2-D processors (same math,
  different output dimensionality).

---

## 8. Conventions, gotchas, invariants

- **`self.latest_frame` is replaced by a fresh `.copy()` each poll** — averaging code relies
  on object identity to detect new frames and drop in-flight ones. Don't mutate it in place.
- **Saturation = 16383** (14-bit). Saturation mask is computed on the RAW counts (add the
  background back if it was subtracted).
- **TWINS `wait_for_stop`** must really wait (the `STATUS_MOVING=6` bug recorded during motion).
- **Exposure clamp 0.01–8 ms** is mandatory (firmware min/max are invalid; >8 ms freezes).
- **The DFT is ROI+binned only** — to speed up: tighter ROI, higher bin, fewer `n_freq`, or a
  `complex64` kernel (~2×; currently complex128).
- **Calibrated axis is applied at compute time, never double-applied on reload** (metadata flag
  + `positions_calibrated`). Raw axis is preserved so files can be re-derived with a new cal.
- **Raw interferogram is always saved** → every file is reprocessable (don't re-add a gate).
- **Two code trees**: edit `gui/` (canonical). `AnalysisApp/` is the synced analyzer
  distributable; the camera app isn't packaged as an exe (run via `gui/run.bat`).
- Persistence via `QSettings` orgs "MIR_CAMERA": apps "KSpace"/"TwinsScan"/"HyperViewer".

---

## 9. Run / entry points

- `gui/run.bat` → `python gui/main.py --mode irc806 --fps 120` (use `--mode mock` with no camera).
- `gui/.venv` holds the deps (PyQt6, pyqtgraph, numpy, scipy, pandas, pythonnet, Pillow).
- Hardware DLLs: Pleora eBUS (camera), `SCU3DControl.dll` (TWINS SmarAct), Thorlabs Kinesis (Z stage).

## 10. File map (acquisition app)

| File | Role |
|---|---|
| `gui/main.py` | entry: spawns camera process, shared mem + queues, builds MainWindow |
| `gui/worker_camera.py` | camera worker process: frame/status out, command in, auto-heal |
| `gui/camera/{camera_interface,factory,irc806_camera,mock_camera}.py` | camera abstraction + backends |
| `gui/ui/main_window.py` | orchestrator: frame poll, latest_frame, controls, background, save, metadata |
| `gui/ui/stages.py` | StagesPanel: TWINS + Thorlabs UI, StageController threading, freeze() |
| `gui/ui/twins_scan.py` | live 1-D TWINS scan UI |
| `gui/ui/measure_kspace.py` | **MeasurePanel**: the hyperspectral experiment + HyperViewer |
| `gui/instruments/twins_stage.py` | NIREOS TWINS SCU3D driver |
| `gui/instruments/stage_driver.py` | Thorlabs KDC101/Z825B driver |
| `gui/instruments/subtwinslv.py` | TwinsScanner: step-scan engine (`scan`, `scan_cube`) |
| `gui/instruments/hyperspectral.py` | 2-D per-pixel DFT (`compute_hyperspectral`) |
| `gui/instruments/spectrum_processor.py` | 1-D interferogram → spectrum |
| `gui/instruments/{calibration,dsp,analysis,walkoff}.py` | shared processing |
| `gui/Twins/ASRC calibration/parameters_{cal,int}.txt` | spectral + motor calibration data |
