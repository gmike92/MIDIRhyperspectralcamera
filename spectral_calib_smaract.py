"""
spectral_calib_smaract.py -- spectrometer-vs-stage calibration scan.

Port of spectral_calib_spectrometer.py (PI motor + Hamamatsu/OceanOptics) to
this rig: an **OceanOptics spectrometer** read through python-seabreeze and the
**SmarAct MCS2** TWINS wedge stage (instruments/twins_stage.py, the same driver
the main app uses in Stages > TWINS and in the TWINS scan).

What it does
    * connects the spectrometer (seabreeze) and the stage (MCS2), each with a
      "Simulate" fallback so the whole tool runs with no hardware;
    * shows the LIVE spectrum in a pyqtgraph plot -- also while a scan runs
      (the live reader and the scan share the device through a lock, so the
      plot never stops updating);
    * scans the stage from Start to End in Step increments; at each commanded
      position it waits for the stage to stop, settles, reads the position back
      from the stage sensor and acquires `Averages` spectra with the chosen
      integration time, cropped to the wavelength range of interest;
    * saves ONE dataset per run in <save dir>/spectral_calib_YYYYmmdd_HHMMSS/:
          spectra.npz      positions_mm (measured), targets_mm, wavelengths_nm,
                           spectra (n_positions x n_wavelengths), timestamps,
                           integration_time_ms, averages + metadata (json str)
          spectra.csv      column 0 = wavelength [nm], one column per position
                           (header = measured position in mm)
          positions.csv    index, target_mm, measured_mm, timestamp
          metadata.json    everything about the run (devices, settings, times)

Dependencies:  PyQt6, pyqtgraph, numpy (the app's requirements.txt) plus
               `pip install seabreeze pyusb libusb-package` and the SmarAct MCS2
               SDK (`smaract.ctl`).
               The spectrometer is read with seabreeze's *pyseabreeze* backend
               (pure Python over libusb, works with the WinUSB driver that
               OceanView installs). The Ocean Insight **Ocean NR** (USB
               0x0999:0x1007, 512-px InGaAs, 900-1700 nm) is missing from
               seabreeze's device table, so it is registered here (see
               _register_ocean_nr) before enumerating -- verified on the unit.
               Both hardware libraries are imported lazily: the tool starts
               (in simulation) without them.

Run:  .venv\\Scripts\\python spectral_calib_smaract.py
      python spectral_calib_smaract.py --save-dir D:\\data   (optional)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from instruments.twins_stage import TwinsStage, TRAVEL_MIN_MM, TRAVEL_MAX_MM

# ---------------------------------------------------------------------------
# DEFAULT parameters (positions in mm, times in ms, wavelengths in nm)
# ---------------------------------------------------------------------------
START_MM = 0.0
END_MM = 2.0
STEP_MM = 0.05
SETTLE_S = 0.2                  # extra wait after the stage reports "stopped"

T_ACQ_MS = 100.0
MEAN_N = 1
W_MIN_NM = 400.0
W_MAX_NM = 1000.0

LIVE_PERIOD_S = 0.05            # pause between live reads (on top of t_acq)
DEFAULT_SAVE_DIR = r"C:\CAMERA\spectral_calib"


SEABREEZE_BACKEND = "pyseabreeze"   # 'cseabreeze' knows no 0x0999 devices; see --backend

# Ocean NR: measured on the real unit (s/n NR1700501). 512 px -> the GET_SPECTRUM
# reply is 32 B metadata + 1024 B; integration times from 1 ms to 2 s were timed
# as honoured, 10 s is a conservative cap (every value is acknowledged).
OCEAN_NR_USB_PRODUCT_ID = 0x1007
OCEAN_NR_PIXELS = 512
OCEAN_NR_ITIME_MIN_US = 1_000
OCEAN_NR_ITIME_MAX_US = 10_000_000


def _register_ocean_nr() -> None:
    """Add the Ocean Insight 'Ocean NR' to pyseabreeze's device registry.

    python-seabreeze (<= 2.11) lists the 0x0999 family SR2/SR4/SR6/HR2/HR4/HR6/ST
    (all OBP2 protocol) but not the NR. Its class only differs in product id and
    pixel count, so it is defined here exactly like HR6. No-op when already
    registered (or when a future seabreeze ships it).
    """
    from seabreeze.pyseabreeze import devices as dv
    from seabreeze.pyseabreeze.transport import USBTransport
    if (0x0999, OCEAN_NR_USB_PRODUCT_ID) in USBTransport.vendor_product_ids:
        return
    from seabreeze.pyseabreeze.protocol import OBP2Protocol
    import seabreeze.pyseabreeze.features as sbf

    class NR(dv.SeaBreezeDevice):          # noqa: N801  (seabreeze naming)
        model_name = "NR"
        transport = (USBTransport,)
        usb_vendor_id = 0x0999
        usb_product_id = OCEAN_NR_USB_PRODUCT_ID
        usb_endpoint_map = dv.EndPointMap(ep_out=0x01, highspeed_in=0x81)
        usb_protocol = OBP2Protocol
        dark_pixel_indices = dv.DarkPixelIndices.from_ranges()
        integration_time_min = OCEAN_NR_ITIME_MIN_US
        integration_time_max = OCEAN_NR_ITIME_MAX_US
        integration_time_base = 1
        spectrum_num_pixel = OCEAN_NR_PIXELS
        spectrum_raw_length = OCEAN_NR_PIXELS * 2 + 32
        spectrum_max_value = 65535
        trigger_modes = dv.TriggerMode.supported("OBP_NORMAL")
        feature_classes = (sbf.spectrometer.SeaBreezeSpectrometerFeatureHR6,)

    print("[Spectrometer] registered Ocean NR (0x0999:0x1007) with pyseabreeze")


def _import_seabreeze(backend: str):
    """Select the seabreeze backend (must happen before the first import of
    seabreeze.spectrometers) and register the extra devices."""
    import seabreeze
    if "seabreeze.spectrometers" not in sys.modules:
        seabreeze.use(backend)
    if backend == "pyseabreeze":
        _register_ocean_nr()
    import seabreeze.spectrometers as sb
    return sb


# ===========================================================================
# Spectrometer: OceanOptics via python-seabreeze, or a simulated device
# ===========================================================================
class OceanOpticsSpectrometer:
    """Thin wrapper around seabreeze.spectrometers.Spectrometer.

    API used by the GUI:
        connect(simulate=False, serial=None) -> bool
        disconnect()
        wavelengths() -> ndarray [nm]
        set_integration_time_ms(ms)
        integration_limits_ms -> (min_ms, max_ms)
        read(averages=1) -> ndarray  (mean of `averages` spectra, counts)
        is_connected, backend ('seabreeze' | 'sim'), model, serial, max_intensity

    `position_source` (callable -> mm) is only used by the SIMULATED device to
    modulate the synthetic spectrum with the stage position, so a simulated
    scan looks like a real TWINS calibration (fringes vs. wavelength).
    """

    def __init__(self, position_source=None) -> None:
        self.spec = None
        self.is_connected = False
        self.backend = None
        self.model = "-"
        self.serial = "-"
        self.max_intensity = 65535.0
        self.integration_limits_ms = (1.0, 60_000.0)
        self._t_acq_ms = T_ACQ_MS
        self._wl = None
        self._position_source = position_source
        self.lock = threading.Lock()     # one reader at a time (live + scan)

    # -- connection ----------------------------------------------------------
    def connect(self, simulate: bool = False, serial: str | None = None) -> bool:
        if self.is_connected:
            return True
        if simulate:
            self.backend = "sim"
            self.model = "SIM-2000"
            self.serial = "SIM"
            self._wl = np.linspace(190.0, 1100.0, 2048)
            self.is_connected = True
            print("[Spectrometer] connected (SIMULATED)")
            return True
        try:
            sb = _import_seabreeze(SEABREEZE_BACKEND)
        except Exception as exc:  # noqa: BLE001
            print(f"[Spectrometer] python-seabreeze not available: {exc}\n"
                  "               pip install seabreeze pyusb libusb-package")
            return False
        try:
            devices = sb.list_devices()
            print(f"[Spectrometer] seabreeze ({SEABREEZE_BACKEND}) sees: {devices}")
            if serial:
                self.spec = sb.Spectrometer.from_serial_number(serial)
            else:
                self.spec = sb.Spectrometer.from_first_available()
        except Exception as exc:  # noqa: BLE001
            print(f"[Spectrometer] no OceanOptics spectrometer found: {exc}\n"
                  "               (unknown model? check the USB vendor:product id in "
                  "Device Manager; other backend: --backend cseabreeze)")
            self.spec = None
            return False
        self.backend = "seabreeze"
        self.is_connected = True
        try:
            self.model = str(self.spec.model)
            self.serial = str(self.spec.serial_number)
            self.max_intensity = float(self.spec.max_intensity)
            lo, hi = self.spec.integration_time_micros_limits
            self.integration_limits_ms = (lo / 1000.0, hi / 1000.0)
            self._wl = np.asarray(self.spec.wavelengths(), dtype=float)
            self.set_integration_time_ms(self._t_acq_ms)
        except Exception as exc:  # noqa: BLE001
            print(f"[Spectrometer] configuration warning: {exc}")
        print(f"[Spectrometer] connected: {self.model} s/n {self.serial}, "
              f"{len(self._wl)} px, t_int {self.integration_limits_ms[0]:g}.."
              f"{self.integration_limits_ms[1]:g} ms")
        return True

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        with self.lock:
            if self.spec is not None:
                try:
                    self.spec.close()
                except Exception as exc:  # noqa: BLE001
                    print(f"[Spectrometer] close warning: {exc}")
            self.spec = None
            self.is_connected = False
        print("[Spectrometer] disconnected")

    # -- settings / data -----------------------------------------------------
    def wavelengths(self) -> np.ndarray:
        return self._wl if self._wl is not None else np.zeros(0)

    def set_integration_time_ms(self, ms: float) -> None:
        lo, hi = self.integration_limits_ms
        ms = float(min(max(ms, lo), hi))
        self._t_acq_ms = ms
        if self.spec is not None:
            self.spec.integration_time_micros(int(round(ms * 1000.0)))

    @property
    def integration_time_ms(self) -> float:
        return self._t_acq_ms

    def read(self, averages: int = 1) -> np.ndarray:
        """Mean of `averages` fresh spectra (full pixel range, counts).
        Call with the device lock held, or use read_locked()."""
        n = max(1, int(averages))
        if self.backend == "sim":
            acc = np.zeros_like(self._wl)
            for _ in range(n):
                time.sleep(self._t_acq_ms / 1000.0)
                acc += self._simulated_spectrum()
            return acc / n
        acc = None
        for _ in range(n):
            frame = np.asarray(self.spec.intensities(), dtype=float)
            acc = frame if acc is None else acc + frame
        return acc / n

    def read_locked(self, averages: int = 1) -> np.ndarray:
        with self.lock:
            if not self.is_connected:
                raise RuntimeError("spectrometer not connected")
            return self.read(averages)

    def _simulated_spectrum(self) -> np.ndarray:
        """Broadband source + two lines, modulated by TWINS-like fringes whose
        period in wavelength depends on the stage position."""
        wl = self._wl
        x = 0.0
        if self._position_source is not None:
            try:
                x = float(self._position_source())
            except Exception:  # noqa: BLE001
                x = 0.0
        base = (np.exp(-((wl - 620.0) / 180.0) ** 2)
                + 0.5 * np.exp(-((wl - 532.0) / 3.0) ** 2)
                + 0.3 * np.exp(-((wl - 780.0) / 4.0) ** 2))
        # optical path difference (nm) ~ 0.01 * x(mm) * 1e6 nm/mm
        opd_nm = 0.01 * x * 1.0e6
        fringes = 0.5 * (1.0 + np.cos(2.0 * np.pi * opd_nm / wl))
        scale = 0.6 * self.max_intensity * self._t_acq_ms / 100.0
        sig = scale * base * fringes + 800.0
        sig += np.random.normal(0.0, 15.0, wl.shape)
        return np.clip(sig, 0.0, self.max_intensity)


def crop_range(wl: np.ndarray, spectrum: np.ndarray, w_min: float, w_max: float):
    """Keep w_min <= wavelength <= w_max (same mask on both arrays)."""
    mask = (wl >= w_min) & (wl <= w_max)
    if not mask.any():
        return wl, spectrum
    return wl[mask], spectrum[mask]


# ===========================================================================
# Live-plot reader: keeps the spectrum on screen up to date
# ===========================================================================
class LiveReader(QtCore.QObject):
    """Background thread reading the spectrometer continuously; each spectrum
    is emitted (thread-safe) to the GUI. Settings are shared with the scan
    (same t_acq / averages / range) through `settings`, a dict the GUI updates."""

    spectrum = QtCore.pyqtSignal(object, object)     # wavelengths, intensities
    error = QtCore.pyqtSignal(str)

    def __init__(self, spectrometer: OceanOpticsSpectrometer, settings: dict) -> None:
        super().__init__()
        self.spectrometer = spectrometer
        self.settings = settings
        self._running = False
        self._thread = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        print("[Live] start")
        while self._running:
            if not self.spectrometer.is_connected:
                time.sleep(0.2)
                continue
            try:
                intensities = self.spectrometer.read_locked(self.settings["averages"])
            except Exception as exc:  # noqa: BLE001
                self.error.emit(f"live read error: {exc}")
                time.sleep(0.5)
                continue
            wl, intensities = crop_range(self.spectrometer.wavelengths(), intensities,
                                         self.settings["w_min"], self.settings["w_max"])
            self.spectrum.emit(wl, intensities)
            time.sleep(LIVE_PERIOD_S)
        print("[Live] ended")


# ===========================================================================
# Scan worker: stage sweep + one (averaged) spectrum per position
# ===========================================================================
class ScanWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, float, object, object)  # i, n, pos, wl, I
    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)                          # run dir or None

    def __init__(self, stage: TwinsStage, spectrometer: OceanOpticsSpectrometer,
                 settings: dict, targets: np.ndarray, settle_s: float,
                 save_dir: str) -> None:
        super().__init__()
        self.stage = stage
        self.spectrometer = spectrometer
        self.settings = dict(settings)     # frozen for the whole scan
        self.targets = np.asarray(targets, dtype=float)
        self.settle_s = float(settle_s)
        self.save_dir = save_dir
        self._abort = False
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def abort(self) -> None:
        self._abort = True

    def _run(self) -> None:
        n = len(self.targets)
        print(f"[Scan] {n} positions: {self.targets[0]:.4f} -> {self.targets[-1]:.4f} mm")
        t_start = datetime.now()
        wl_full = self.spectrometer.wavelengths()
        wl, _ = crop_range(wl_full, wl_full, self.settings["w_min"], self.settings["w_max"])
        positions, spectra, stamps, taken_targets = [], [], [], []
        error = None
        try:
            for i, target in enumerate(self.targets):
                if self._abort:
                    self.status.emit("scan aborted")
                    break
                if not self.stage.move_to(float(target)):
                    error = f"stage refused move to {target:.4f} mm"
                    break
                if not self.stage.wait_for_stop():
                    error = f"stage did not stop at {target:.4f} mm"
                    break
                if self.settle_s > 0:
                    time.sleep(self.settle_s)
                pos = float(self.stage.get_position())
                intensities = self.spectrometer.read_locked(self.settings["averages"])
                _, intensities = crop_range(wl_full, intensities,
                                            self.settings["w_min"], self.settings["w_max"])
                positions.append(pos)
                spectra.append(intensities)
                stamps.append(time.time())
                taken_targets.append(float(target))
                self.progress.emit(i + 1, n, pos, wl, intensities)
        except Exception as exc:  # noqa: BLE001
            error = f"scan error: {exc}"

        if error:
            self.status.emit(error)
        if not positions:
            self.finished.emit(None)
            return
        try:
            run_dir = self._save(wl, np.asarray(positions), np.asarray(taken_targets),
                                 np.asarray(spectra), np.asarray(stamps), t_start,
                                 aborted=self._abort or bool(error))
            self.status.emit(f"saved {len(positions)} spectra -> {run_dir}")
            self.finished.emit(run_dir)
        except Exception as exc:  # noqa: BLE001
            self.status.emit(f"save error: {exc}")
            self.finished.emit(None)

    def _save(self, wl, positions, targets, spectra, stamps, t_start, aborted) -> str:
        stamp = t_start.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.save_dir) / f"spectral_calib_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "started": t_start.isoformat(timespec="seconds"),
            "finished": datetime.now().isoformat(timespec="seconds"),
            "aborted": bool(aborted),
            "n_positions": int(len(positions)),
            "n_wavelengths": int(len(wl)),
            "scan": {
                "start_mm": float(self.targets[0]),
                "end_mm": float(self.targets[-1]),
                "step_mm": round(float(self.targets[1] - self.targets[0]), 9) if len(self.targets) > 1 else 0.0,
                "n_targets": int(len(self.targets)),
                "settle_s": self.settle_s,
            },
            "spectrometer": {
                "backend": self.spectrometer.backend,
                "model": self.spectrometer.model,
                "serial": self.spectrometer.serial,
                "integration_time_ms": float(self.spectrometer.integration_time_ms),
                "averages": int(self.settings["averages"]),
                "wavelength_min_nm": float(self.settings["w_min"]),
                "wavelength_max_nm": float(self.settings["w_max"]),
                "max_intensity": float(self.spectrometer.max_intensity),
            },
            "stage": {
                "backend": self.stage.backend,
                "locator": self.stage.locator,
                "channel": self.stage.channel,
                "travel_mm": [TRAVEL_MIN_MM, TRAVEL_MAX_MM],
            },
        }

        np.savez(run_dir / "spectra.npz",
                 positions_mm=positions, targets_mm=targets, wavelengths_nm=wl,
                 spectra=spectra, timestamps=stamps,
                 integration_time_ms=meta["spectrometer"]["integration_time_ms"],
                 averages=meta["spectrometer"]["averages"],
                 metadata=json.dumps(meta))

        # spectra.csv: wavelength column + one column per measured position
        header = "wavelength_nm," + ",".join(f"{p:.6f}" for p in positions)
        np.savetxt(run_dir / "spectra.csv",
                   np.column_stack([wl, spectra.T]), delimiter=",",
                   header=header, comments="", fmt="%.6f")
        np.savetxt(run_dir / "positions.csv",
                   np.column_stack([np.arange(len(positions)), targets, positions, stamps]),
                   delimiter=",", header="index,target_mm,measured_mm,timestamp",
                   comments="", fmt=("%d", "%.6f", "%.6f", "%.3f"))
        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[Scan] saved {run_dir}")
        return str(run_dir)


def scan_targets(start: float, end: float, step: float) -> np.ndarray:
    """Positions start, start+step, ... up to and including end (direction from
    the sign of end-start). Like the reference np.arange, but the last point is
    never past `end` because of float round-off."""
    step = abs(float(step))
    if step <= 0 or start == end:
        return np.array([float(start)])
    n = int(np.floor(abs(end - start) / step + 1e-9)) + 1
    sign = 1.0 if end >= start else -1.0
    return float(start) + sign * step * np.arange(n)


# ===========================================================================
# Main window
# ===========================================================================
class MainWindow(QMainWindow):
    def __init__(self, save_dir: str = DEFAULT_SAVE_DIR) -> None:
        super().__init__()
        self.setWindowTitle("Spectral calibration: OceanOptics spectrometer + SmarAct stage")

        self.stage = TwinsStage()
        self.spectrometer = OceanOpticsSpectrometer(position_source=self._sim_position)
        # shared acquisition settings (read by the live thread and by the scan)
        self.settings = {"averages": MEAN_N, "w_min": W_MIN_NM, "w_max": W_MAX_NM}
        self.live = LiveReader(self.spectrometer, self.settings)
        self.live.spectrum.connect(self._plot_spectrum)
        self.live.error.connect(self._set_status)
        self.scan = None
        self._busy = False              # a connect / go-to is running in a thread

        central = QWidget()
        root = QVBoxLayout(central)
        top = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self._build_stage_group())
        left.addWidget(self._build_scan_group())
        top.addLayout(left, 1)
        top.addWidget(self._build_spectrometer_group(), 1)
        root.addLayout(top)
        root.addWidget(self._build_plot(), 1)
        root.addLayout(self._build_footer())
        self.setCentralWidget(central)

        self.save_dir_edit.setText(save_dir)

        # persist the user's parameters across runs
        self._qs = QtCore.QSettings("MIR_CAMERA", "SpectralCalib")
        self._restore_settings()
        for w in self._persisted().values():
            w.valueChanged.connect(self._save_settings)
        self.save_dir_edit.editingFinished.connect(self._save_settings)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_position)
        self._poll_timer.start(500)

        self._refresh_enabled()
        self._update_n_points()
        self._toggle_live(self.live_spect.isChecked())   # checkbox is set before its signal is wired

    # -- persistence ---------------------------------------------------------
    def _persisted(self) -> dict:
        return {"start": self.start_in, "end": self.end_in, "step": self.step_in,
                "settle": self.settle_in, "t_acq": self.acq_time, "averages": self.mean_n,
                "w_min": self.w_min, "w_max": self.w_max}

    def _restore_settings(self) -> None:
        for key, w in self._persisted().items():
            val = self._qs.value(key, None)
            if val is None:
                continue
            try:
                w.setValue(type(w.value())(val))
            except (TypeError, ValueError):
                pass
        d = self._qs.value("save_dir", None)
        if d:
            self.save_dir_edit.setText(str(d))

    def _save_settings(self, *_) -> None:
        for key, w in self._persisted().items():
            self._qs.setValue(key, w.value())
        self._qs.setValue("save_dir", self.save_dir_edit.text())

    # -- widgets -------------------------------------------------------------
    def _build_stage_group(self) -> QGroupBox:
        g = QGroupBox("SmarAct stage (MCS2)")
        grid = QGridLayout(g)

        self.stage_sim = QCheckBox("Simulate (no hardware)")
        self.stage_sim.setChecked(True)
        grid.addWidget(self.stage_sim, 0, 0, 1, 2)

        self.stage_connect = QPushButton("Connect")
        self.stage_connect.clicked.connect(self._toggle_stage)
        grid.addWidget(self.stage_connect, 1, 0)
        self.stage_pos = QLabel("-- mm")
        self.stage_pos.setStyleSheet("font-weight:600;")
        grid.addWidget(self.stage_pos, 1, 1)

        grid.addWidget(QLabel("Go to"), 2, 0)
        row = QHBoxLayout()
        self.goto_in = QDoubleSpinBox()
        self.goto_in.setRange(TRAVEL_MIN_MM, TRAVEL_MAX_MM)
        self.goto_in.setDecimals(4)
        self.goto_in.setSingleStep(0.1)
        self.goto_in.setSuffix(" mm")
        self.goto_in.setToolTip(f"Software travel limits (instruments/twins_stage.py): "
                                f"{TRAVEL_MIN_MM} .. {TRAVEL_MAX_MM} mm")
        row.addWidget(self.goto_in)
        self.goto_btn = QPushButton("Go")
        self.goto_btn.clicked.connect(self._goto)
        row.addWidget(self.goto_btn)
        grid.addLayout(row, 2, 1)

        self.stage_status = QLabel("offline")
        self.stage_status.setStyleSheet("color:#888; font-size:11px;")
        self.stage_status.setWordWrap(True)
        grid.addWidget(self.stage_status, 3, 0, 1, 2)
        return g

    def _build_scan_group(self) -> QGroupBox:
        g = QGroupBox("Scan")
        grid = QGridLayout(g)

        def mm_spin(value, step=0.1, decimals=4):
            s = QDoubleSpinBox()
            s.setRange(TRAVEL_MIN_MM, TRAVEL_MAX_MM)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setValue(value)
            s.setSuffix(" mm")
            s.valueChanged.connect(self._update_n_points)
            return s

        self.start_in = mm_spin(START_MM)
        self.end_in = mm_spin(END_MM)
        self.step_in = QDoubleSpinBox()
        self.step_in.setRange(0.0001, TRAVEL_MAX_MM - TRAVEL_MIN_MM)
        self.step_in.setDecimals(4)
        self.step_in.setSingleStep(0.01)
        self.step_in.setValue(STEP_MM)
        self.step_in.setSuffix(" mm")
        self.step_in.valueChanged.connect(self._update_n_points)
        self.settle_in = QDoubleSpinBox()
        self.settle_in.setRange(0.0, 10.0)
        self.settle_in.setDecimals(2)
        self.settle_in.setSingleStep(0.05)
        self.settle_in.setValue(SETTLE_S)
        self.settle_in.setSuffix(" s")
        self.n_points_l = QLabel("--")
        self.n_points_l.setStyleSheet("font-weight:600;")

        grid.addWidget(QLabel("Start"), 0, 0)
        grid.addWidget(self.start_in, 0, 1)
        grid.addWidget(QLabel("End"), 1, 0)
        grid.addWidget(self.end_in, 1, 1)
        grid.addWidget(QLabel("Step"), 2, 0)
        grid.addWidget(self.step_in, 2, 1)
        grid.addWidget(QLabel("Points"), 3, 0)
        grid.addWidget(self.n_points_l, 3, 1)
        grid.addWidget(QLabel("Settle after move"), 4, 0)
        grid.addWidget(self.settle_in, 4, 1)

        grid.addWidget(QLabel("Save dir"), 5, 0)
        row = QHBoxLayout()
        self.save_dir_edit = QLineEdit()
        row.addWidget(self.save_dir_edit)
        browse = QPushButton("...")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse_save_dir)
        row.addWidget(browse)
        grid.addLayout(row, 5, 1)

        btns = QHBoxLayout()
        self.go_btn = QPushButton("Acquire")
        self.go_btn.clicked.connect(self._start_scan)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_scan)
        btns.addWidget(self.go_btn)
        btns.addWidget(self.stop_btn)
        grid.addLayout(btns, 6, 0, 1, 2)
        return g

    def _build_spectrometer_group(self) -> QGroupBox:
        g = QGroupBox("Spectrometer (OceanOptics / seabreeze)")
        grid = QGridLayout(g)

        self.spec_sim = QCheckBox("Simulate (no hardware)")
        self.spec_sim.setChecked(True)
        grid.addWidget(self.spec_sim, 0, 0, 1, 2)

        self.spec_connect = QPushButton("Connect")
        self.spec_connect.clicked.connect(self._toggle_spectrometer)
        grid.addWidget(self.spec_connect, 1, 0)
        self.spec_info = QLabel("offline")
        self.spec_info.setStyleSheet("color:#888; font-size:11px;")
        self.spec_info.setWordWrap(True)
        grid.addWidget(self.spec_info, 1, 1)

        self.acq_time = QDoubleSpinBox()
        self.acq_time.setRange(1.0, 60_000.0)
        self.acq_time.setDecimals(1)
        self.acq_time.setValue(T_ACQ_MS)
        self.acq_time.setSuffix(" ms")
        self.acq_time.valueChanged.connect(self._apply_integration_time)
        grid.addWidget(QLabel("Integration time"), 2, 0)
        grid.addWidget(self.acq_time, 2, 1)

        self.mean_n = QSpinBox()
        self.mean_n.setRange(1, 1000)
        self.mean_n.setValue(MEAN_N)
        self.mean_n.valueChanged.connect(lambda v: self.settings.__setitem__("averages", int(v)))
        grid.addWidget(QLabel("Averages"), 3, 0)
        grid.addWidget(self.mean_n, 3, 1)

        self.w_min = QDoubleSpinBox()
        self.w_min.setRange(100.0, 2500.0)
        self.w_min.setDecimals(1)
        self.w_min.setValue(W_MIN_NM)
        self.w_min.setSuffix(" nm")
        self.w_min.valueChanged.connect(self._apply_range)
        self.w_max = QDoubleSpinBox()
        self.w_max.setRange(100.0, 2500.0)
        self.w_max.setDecimals(1)
        self.w_max.setValue(W_MAX_NM)
        self.w_max.setSuffix(" nm")
        self.w_max.valueChanged.connect(self._apply_range)
        grid.addWidget(QLabel("From"), 4, 0)
        grid.addWidget(self.w_min, 4, 1)
        grid.addWidget(QLabel("To"), 5, 0)
        grid.addWidget(self.w_max, 5, 1)

        self.live_spect = QCheckBox("Live spectrum")
        self.live_spect.setChecked(True)
        self.live_spect.toggled.connect(self._toggle_live)
        grid.addWidget(self.live_spect, 6, 0, 1, 2)
        grid.setRowStretch(7, 1)
        return g

    def _build_plot(self) -> pg.PlotWidget:
        self.plot = pg.PlotWidget(title="Spectrum")
        self.plot.setLabel("left", "Intensity [counts]")
        self.plot.setLabel("bottom", "Wavelength [nm]")
        self.plot.showGrid(x=True, y=True)
        self.plot.setMinimumHeight(250)
        self.line = self.plot.plot([], [], pen=pg.mkPen("#1c7ed6", width=1))
        return self.plot

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_l = QLabel("idle")
        self.status_l.setStyleSheet("color:#888;")
        row.addWidget(self.progress, 1)
        row.addWidget(self.status_l, 2)
        return row

    # -- helpers ---------------------------------------------------------------
    def _sim_position(self) -> float:
        return self.stage.get_position() if self.stage.is_connected else 0.0

    def _set_status(self, text: str) -> None:
        self.status_l.setText(text)

    def _run_async(self, fn, done_msg: str, label: QLabel) -> None:
        """Run a blocking stage/spectrometer call on a thread (UI never freezes)."""
        if self._busy:
            return
        self._busy = True
        self._refresh_enabled()
        sig = _AsyncDone(self)

        def _work():
            try:
                fn()
                sig.done.emit(done_msg)
            except Exception as exc:  # noqa: BLE001
                sig.done.emit(f"error: {exc}")

        def _finish(msg):
            self._busy = False
            label.setText(msg)
            self._on_devices_changed()
            self._refresh_enabled()

        sig.done.connect(_finish)
        threading.Thread(target=_work, daemon=True).start()

    def _refresh_enabled(self) -> None:
        scanning = self.scan is not None
        st_conn, sp_conn = self.stage.is_connected, self.spectrometer.is_connected
        free = not self._busy and not scanning
        self.stage_connect.setText("Disconnect" if st_conn else "Connect")
        self.stage_connect.setEnabled(free)
        self.stage_sim.setEnabled(free and not st_conn)
        for w in (self.goto_in, self.goto_btn):
            w.setEnabled(free and st_conn)
        self.spec_connect.setText("Disconnect" if sp_conn else "Connect")
        self.spec_connect.setEnabled(free)
        self.spec_sim.setEnabled(free and not sp_conn)
        # integration time / averages / range stay editable for the LIVE view;
        # the scan freezes its own copy at start (so they are locked while scanning)
        for w in (self.acq_time, self.mean_n, self.w_min, self.w_max):
            w.setEnabled(not scanning)
        for w in (self.start_in, self.end_in, self.step_in, self.settle_in, self.save_dir_edit):
            w.setEnabled(not scanning)
        self.go_btn.setEnabled(free and st_conn and sp_conn)
        self.stop_btn.setEnabled(scanning)

    def _on_devices_changed(self) -> None:
        if self.spectrometer.is_connected:
            lo, hi = self.spectrometer.integration_limits_ms
            self.acq_time.setRange(lo, hi)
            wl = self.spectrometer.wavelengths()
            if len(wl):
                self.w_min.setRange(float(np.floor(wl.min())), float(np.ceil(wl.max())))
                self.w_max.setRange(float(np.floor(wl.min())), float(np.ceil(wl.max())))
            self._apply_integration_time(self.acq_time.value())
            self._apply_range()
            self.spec_info.setText(f"{self.spectrometer.model}  s/n {self.spectrometer.serial}"
                                   f"  ({len(wl)} px, {self.spectrometer.backend})")
        else:
            self.acq_time.setRange(1.0, 60_000.0)
            self.w_min.setRange(100.0, 2500.0)
            self.w_max.setRange(100.0, 2500.0)
            self.spec_info.setText("offline")
        if not self.stage.is_connected:
            self.stage_pos.setText("-- mm")

    # -- device slots ----------------------------------------------------------
    def _toggle_stage(self) -> None:
        if self.stage.is_connected:
            # leave the wedge where it is (no park move), as in the main app
            self._run_async(lambda: self.stage.disconnect(safe=False), "disconnected",
                            self.stage_status)
        else:
            sim = self.stage_sim.isChecked()
            self._run_async(lambda: self._connect_stage(sim), "connected", self.stage_status)

    def _connect_stage(self, sim: bool) -> None:
        # connect() references the positioner and parks at HOME: the stage MOVES
        if not self.stage.connect(simulate=sim, home=True):
            raise RuntimeError("stage not found (check MCS2 SDK / USB, or tick Simulate)")

    def _goto(self) -> None:
        target = self.goto_in.value()

        def _move():
            if not self.stage.move_to(target):
                raise RuntimeError(f"move to {target:.4f} mm refused")
            self.stage.wait_for_stop()

        self._run_async(_move, f"moved to {target:.4f} mm", self.stage_status)

    def _toggle_spectrometer(self) -> None:
        if self.spectrometer.is_connected:
            self._run_async(self.spectrometer.disconnect, "disconnected", self.spec_info)
        else:
            sim = self.spec_sim.isChecked()

            def _connect():
                if not self.spectrometer.connect(simulate=sim):
                    raise RuntimeError("spectrometer not found (seabreeze / USB, or tick Simulate)")

            self._run_async(_connect, "connected", self.spec_info)

    def _apply_integration_time(self, ms: float) -> None:
        if not self.spectrometer.is_connected:
            return
        with self.spectrometer.lock:
            try:
                self.spectrometer.set_integration_time_ms(float(ms))
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"integration time error: {exc}")

    def _apply_range(self, *_) -> None:
        lo, hi = self.w_min.value(), self.w_max.value()
        if lo >= hi:            # keep the range non-empty
            if self.sender() is self.w_min:
                self.w_max.setValue(lo + 1.0)
            else:
                self.w_min.setValue(hi - 1.0)
            return
        self.settings["w_min"], self.settings["w_max"] = lo, hi

    def _toggle_live(self, on: bool) -> None:
        if on:
            self.live.start()
        else:
            self.live.stop()

    def _poll_position(self) -> None:
        if self._busy or self.scan is not None or not self.stage.is_connected:
            return
        try:
            self.stage_pos.setText(f"{self.stage.get_position():.4f} mm")
        except Exception:  # noqa: BLE001
            pass

    def _browse_save_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Save directory",
                                             self.save_dir_edit.text() or DEFAULT_SAVE_DIR)
        if d:
            self.save_dir_edit.setText(d)
            self._save_settings()

    def _update_n_points(self, *_) -> None:
        n = len(scan_targets(self.start_in.value(), self.end_in.value(), self.step_in.value()))
        self.n_points_l.setText(f"{n}")

    # -- plot ------------------------------------------------------------------
    @QtCore.pyqtSlot(object, object)
    def _plot_spectrum(self, wl, intensities) -> None:
        self.line.setData(np.asarray(wl), np.asarray(intensities))

    # -- scan ------------------------------------------------------------------
    def _start_scan(self) -> None:
        if self.scan is not None or self._busy:
            return
        if not (self.stage.is_connected and self.spectrometer.is_connected):
            self._set_status("connect the stage AND the spectrometer first")
            return
        targets = scan_targets(self.start_in.value(), self.end_in.value(), self.step_in.value())
        save_dir = self.save_dir_edit.text().strip() or DEFAULT_SAVE_DIR
        self._apply_integration_time(self.acq_time.value())

        self.scan = ScanWorker(self.stage, self.spectrometer, self.settings, targets,
                               self.settle_in.value(), save_dir)
        self.scan.progress.connect(self._on_progress)
        self.scan.status.connect(self._set_status)
        self.scan.finished.connect(self._on_scan_done)
        self.progress.setRange(0, len(targets))
        self.progress.setValue(0)
        self._set_status(f"scanning {len(targets)} positions...")
        self._refresh_enabled()
        self.scan.start()

    def _stop_scan(self) -> None:
        if self.scan is not None:
            self.scan.abort()
            self._set_status("stopping after the current position...")

    @QtCore.pyqtSlot(int, int, float, object, object)
    def _on_progress(self, i, n, pos, wl, intensities) -> None:
        self.progress.setValue(i)
        self.stage_pos.setText(f"{pos:.4f} mm")
        self._set_status(f"point {i}/{n} @ {pos:.4f} mm")
        self._plot_spectrum(wl, intensities)

    @QtCore.pyqtSlot(object)
    def _on_scan_done(self, run_dir) -> None:
        self.scan = None
        self.last_run_dir = run_dir
        if run_dir is None and "saved" not in self.status_l.text():
            self._set_status(self.status_l.text() + "  (nothing saved)")
        self._refresh_enabled()

    # -- shutdown --------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        if self.scan is not None:
            self.scan.abort()
        self.live.stop()
        self._poll_timer.stop()
        try:
            self.spectrometer.disconnect()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.stage.is_connected:
                self.stage.disconnect(safe=False)
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)


class _AsyncDone(QtCore.QObject):
    """One-shot signal carrier so a worker thread can hand its result to the GUI."""
    done = QtCore.pyqtSignal(str)


def main(argv=None) -> int:
    global SEABREEZE_BACKEND
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save-dir", default=None,
                        help=f"root folder for the datasets (default {DEFAULT_SAVE_DIR}, "
                             "remembered between runs)")
    parser.add_argument("--backend", choices=("pyseabreeze", "cseabreeze"),
                        default=SEABREEZE_BACKEND,
                        help="seabreeze backend (default pyseabreeze: libusb-based, "
                             "supports the Ocean NR and the WinUSB driver)")
    args = parser.parse_args(argv)
    SEABREEZE_BACKEND = args.backend

    app = QApplication(sys.argv)
    win = MainWindow(save_dir=args.save_dir or DEFAULT_SAVE_DIR)
    if args.save_dir:
        win.save_dir_edit.setText(args.save_dir)
    win.resize(1100, 800)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
