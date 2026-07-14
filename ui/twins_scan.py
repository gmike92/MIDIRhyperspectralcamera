"""TWINS interferogram scan + FTIR spectrum panel for the GUI.

Sweeps the TWINS wedge stage across [start, stop] in N steps (showing the live
step size), reads the live-camera ROI mean at each step to build the
interferogram, then runs the verbatim repo SpectrumProcessor DFT to a spectrum.

The scan runs through the stage's StageController (so its position-poll timer
pauses during the scan -- no concurrent SCU3D DLL access) on a background
thread; progress/results return via Qt signals.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QGroupBox, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from instruments.subtwinslv import TwinsScanner
from instruments.spectrum_processor import (
    DEFAULT_START_MM, DEFAULT_STOP_MM, DEFAULT_N_STEPS, DEFAULT_APODIZATION,
    DEFAULT_WL_START, DEFAULT_WL_STOP,
)


class TwinsScanPanel(QWidget):
    sig_progress = QtCore.pyqtSignal(int, int, float, float)
    sig_scan_done = QtCore.pyqtSignal(object, object)
    sig_status = QtCore.pyqtSignal(str)

    def __init__(self, twins_ctl, frame_source, roi_provider=None,
                 save_dir: str = r"D:\CAMERA\twins") -> None:
        super().__init__()
        self.twins_ctl = twins_ctl              # StageController (has .driver, .run, .busy)
        self.frame_source = frame_source        # () -> 2D frame or None
        self.roi_provider = roi_provider        # () -> (r0,r1,c0,c1) or None (full frame)
        self.save_dir = save_dir
        self.scanner = None
        self._abort = False
        self._positions = None
        self._interferogram = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_spectrum_group())
        layout.addWidget(self._build_plots())

        self.sig_progress.connect(self._on_progress)
        self.sig_scan_done.connect(self._on_scan_done)
        self.sig_status.connect(self.lbl_status.setText)

        # Persist scan parameters across app restarts (restore, then bind saves).
        self._settings = QtCore.QSettings("MIR_CAMERA", "TwinsScan")
        self._restore_settings()
        for widget, _cast in self._persisted_spins().values():
            widget.valueChanged.connect(self._save_settings)

        self._update_step()

    # -- persistence ---------------------------------------------------------
    def _persisted_spins(self) -> dict:
        """key -> (widget, cast). QSettings stores strings, so cast on restore."""
        return {
            "ts_start": (self.spin_start, float),
            "ts_stop": (self.spin_stop, float),
            "ts_steps": (self.spin_steps, int),
            "ts_frames": (self.spin_frames, int),
            "ts_apod": (self.spin_apod, float),
            "ts_wl0": (self.spin_wl0, float),
            "ts_wl1": (self.spin_wl1, float),
            "ts_npoints": (self.spin_npoints, int),
        }

    def _restore_settings(self) -> None:
        for key, (widget, cast) in self._persisted_spins().items():
            val = self._settings.value(key, None)
            if val is None:
                continue
            try:
                widget.setValue(cast(val))
            except (TypeError, ValueError):
                pass

    def _save_settings(self, *args) -> None:
        for key, (widget, _cast) in self._persisted_spins().items():
            self._settings.setValue(key, widget.value())

    # -- acquisition ---------------------------------------------------------
    def _build_scan_group(self) -> QGroupBox:
        g = QGroupBox("TWINS Scan")
        grid = QGridLayout(g)

        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0.0, 50.0)
        self.spin_start.setDecimals(3)
        self.spin_start.setSingleStep(0.1)
        self.spin_start.setValue(DEFAULT_START_MM)
        self.spin_start.setSuffix(" mm")
        self.spin_start.valueChanged.connect(self._update_step)
        grid.addWidget(QLabel("Start"), 0, 0)
        grid.addWidget(self.spin_start, 0, 1)

        self.spin_stop = QDoubleSpinBox()
        self.spin_stop.setRange(0.0, 50.0)
        self.spin_stop.setDecimals(3)
        self.spin_stop.setSingleStep(0.1)
        self.spin_stop.setValue(DEFAULT_STOP_MM)
        self.spin_stop.setSuffix(" mm")
        self.spin_stop.valueChanged.connect(self._update_step)
        grid.addWidget(QLabel("Stop"), 1, 0)
        grid.addWidget(self.spin_stop, 1, 1)

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(2, 10000)
        self.spin_steps.setValue(DEFAULT_N_STEPS)
        self.spin_steps.valueChanged.connect(self._update_step)
        grid.addWidget(QLabel("Steps"), 2, 0)
        grid.addWidget(self.spin_steps, 2, 1)

        self.lbl_step = QLabel("-- µm")
        self.lbl_step.setStyleSheet("font-weight:600;")
        grid.addWidget(QLabel("Step size"), 3, 0)
        grid.addWidget(self.lbl_step, 3, 1)

        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(1, 200)
        self.spin_frames.setValue(1)
        grid.addWidget(QLabel("Frames/point"), 4, 0)
        grid.addWidget(self.spin_frames, 4, 1)

        self.lbl_roi = QLabel("full frame")
        self.lbl_roi.setToolTip("Enable the ROI box from the Measure tab "
                                "(Show ROI) to restrict the scan; else full frame.")
        grid.addWidget(QLabel("Signal region"), 5, 0)
        grid.addWidget(self.lbl_roi, 5, 1)

        btn_row = QHBoxLayout()
        self.btn_scan = QPushButton("Scan")
        self.btn_scan.clicked.connect(self._start_scan)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_scan)
        btn_row.addWidget(self.btn_scan)
        btn_row.addWidget(self.btn_stop)
        grid.addLayout(btn_row, 6, 0, 1, 2)

        self.progress = QProgressBar()
        grid.addWidget(self.progress, 7, 0, 1, 2)
        self.lbl_status = QLabel("idle")
        self.lbl_status.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        grid.addWidget(self.lbl_status, 8, 0, 1, 2)
        return g

    # -- spectrum params -----------------------------------------------------
    def _build_spectrum_group(self) -> QGroupBox:
        g = QGroupBox("Spectrum (DFT)")
        grid = QGridLayout(g)

        self.spin_apod = QDoubleSpinBox()
        self.spin_apod.setRange(0.01, 5.0)
        self.spin_apod.setSingleStep(0.05)
        self.spin_apod.setValue(DEFAULT_APODIZATION)
        grid.addWidget(QLabel("Apodization"), 0, 0)
        grid.addWidget(self.spin_apod, 0, 1)

        self.spin_wl0 = QDoubleSpinBox()
        self.spin_wl0.setRange(0.1, 100.0)
        self.spin_wl0.setValue(DEFAULT_WL_START)
        self.spin_wl0.setSuffix(" µm")
        grid.addWidget(QLabel("λ start"), 1, 0)
        grid.addWidget(self.spin_wl0, 1, 1)

        self.spin_wl1 = QDoubleSpinBox()
        self.spin_wl1.setRange(0.1, 100.0)
        self.spin_wl1.setValue(DEFAULT_WL_STOP)
        self.spin_wl1.setSuffix(" µm")
        grid.addWidget(QLabel("λ stop"), 2, 0)
        grid.addWidget(self.spin_wl1, 2, 1)

        self.spin_npoints = QSpinBox()
        self.spin_npoints.setRange(50, 10000)
        self.spin_npoints.setValue(1000)
        grid.addWidget(QLabel("N points"), 3, 0)
        grid.addWidget(self.spin_npoints, 3, 1)

        btn_row = QHBoxLayout()
        self.btn_compute = QPushButton("Compute Spectrum")
        self.btn_compute.clicked.connect(self._compute_spectrum)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._save)
        btn_row.addWidget(self.btn_compute)
        btn_row.addWidget(self.btn_save)
        grid.addLayout(btn_row, 4, 0, 1, 2)
        return g

    def _build_plots(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)

        self.plot_ifg = pg.PlotWidget(title="Interferogram")
        self.plot_ifg.setLabel("bottom", "Stage position", units="mm")
        self.plot_ifg.setMinimumHeight(150)
        self.curve_ifg = self.plot_ifg.plot(pen=pg.mkPen("#1c7ed6", width=1))
        v.addWidget(self.plot_ifg)

        self.plot_spec = pg.PlotWidget(title="Spectrum")
        self.plot_spec.setLabel("bottom", "Wavelength", units="µm")
        self.plot_spec.setMinimumHeight(150)
        self.curve_spec = self.plot_spec.plot(pen=pg.mkPen("#e8590c", width=1))
        v.addWidget(self.plot_spec)
        return w

    # -- step-size readout ---------------------------------------------------
    def _update_step(self) -> None:
        n = self.spin_steps.value()
        if n > 1:
            step_um = abs(self.spin_stop.value() - self.spin_start.value()) / (n - 1) * 1000
            self.lbl_step.setText(f"{step_um:.1f} µm")
        else:
            self.lbl_step.setText("-- µm")

    def _current_roi(self):
        roi = self.roi_provider() if self.roi_provider else None
        if roi is None:
            self.lbl_roi.setText("full frame")
        else:
            r0, r1, c0, c1 = roi
            self.lbl_roi.setText(f"ROI {r1-r0}×{c1-c0} px")
        return roi

    # -- scan control --------------------------------------------------------
    def _start_scan(self) -> None:
        drv = self.twins_ctl.driver
        if not getattr(drv, "is_connected", False):
            self.sig_status.emit("TWINS stage not connected")
            return
        if self.twins_ctl.busy:
            self.sig_status.emit("stage busy")
            return
        start = self.spin_start.value()
        stop = self.spin_stop.value()
        n = self.spin_steps.value()
        roi = self._current_roi()
        self._scan_roi = roi          # ROI averaged per point, saved with the scan
        frames = self.spin_frames.value()

        self.scanner = TwinsScanner(drv, self.frame_source)
        self._abort = False
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setMaximum(n)
        self.progress.setValue(0)
        self.curve_ifg.setData([], [])
        self.curve_spec.setData([], [])

        def scan_fn():
            def prog(i, total, pos, val):
                self.sig_progress.emit(i, total, pos, val)
            try:
                pos, ifg = self.scanner.scan(
                    start, stop, n, roi=roi, frames_avg=frames,
                    progress=prog, should_abort=lambda: self._abort)
                self.sig_scan_done.emit(pos, ifg)
            except Exception as e:  # noqa: BLE001
                self.sig_status.emit(f"scan error: {e}")
                self.sig_scan_done.emit(None, None)

        # Run via the controller so its poll timer pauses (no DLL contention).
        self.twins_ctl.run(scan_fn)

    def _stop_scan(self) -> None:
        self._abort = True
        self.sig_status.emit("stopping...")

    @QtCore.pyqtSlot(int, int, float, float)
    def _on_progress(self, i: int, n: int, pos: float, val: float) -> None:
        self.progress.setValue(i)
        self.lbl_status.setText(f"point {i}/{n}  @ {pos:.3f} mm  =  {val:.1f}")
        if self.scanner is not None and self.scanner.positions is not None:
            p = self.scanner.positions
            g = self.scanner.interferogram
            if p is not None and g is not None and len(p):
                self.curve_ifg.setData(np.asarray(p[:i]), np.asarray(g[:i]))

    @QtCore.pyqtSlot(object, object)
    def _on_scan_done(self, positions, interferogram) -> None:
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if positions is None or interferogram is None or len(positions) == 0:
            self.sig_status.emit("scan aborted / no data")
            return
        self._positions = np.asarray(positions)
        self._interferogram = np.asarray(interferogram)
        self.curve_ifg.setData(self._positions, self._interferogram)
        self.sig_status.emit(f"scan done: {len(self._positions)} points")
        self._compute_spectrum()

    # -- spectrum ------------------------------------------------------------
    def _compute_spectrum(self) -> None:
        if self.scanner is None or self.scanner.positions is None or len(self.scanner.positions) == 0:
            self.sig_status.emit("no scan to process")
            return
        try:
            wl, spec = self.scanner.compute_spectrum(
                wl_start=self.spin_wl0.value(), wl_stop=self.spin_wl1.value(),
                apod_width=self.spin_apod.value(), n_points=self.spin_npoints.value())
        except Exception as e:  # noqa: BLE001
            self.sig_status.emit(f"spectrum error: {e}")
            return
        if wl is None or spec is None:
            self.sig_status.emit("spectrum unavailable")
            return
        self.curve_spec.setData(np.asarray(wl), np.asarray(spec))
        peak = float(wl[int(np.argmax(spec))])
        self.sig_status.emit(f"spectrum: peak ~{peak:.2f} µm")

    def _save(self) -> None:
        if self.scanner is None or self.scanner.positions is None or len(self.scanner.positions) == 0:
            self.sig_status.emit("nothing to save")
            return
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = os.path.join(self.save_dir, f"twins_scan_{stamp}")
            with open(stem + ".csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["position_mm", "interferogram"])
                for p, v in zip(self.scanner.positions, self.scanner.interferogram):
                    w.writerow([p, v])
            roi = getattr(self, "_scan_roi", None)
            np.savez(stem + ".npz",
                     positions=self.scanner.positions,
                     interferogram=self.scanner.interferogram,
                     wavelengths=self.scanner.processor.wavelengths,
                     spectrum=self.scanner.processor.spectrum,
                     roi=np.asarray(roi if roi is not None else [], dtype=float))
            self.sig_status.emit(f"saved {os.path.basename(stem)}.csv/.npz")
        except Exception as e:  # noqa: BLE001
            self.sig_status.emit(f"save error: {e}")
