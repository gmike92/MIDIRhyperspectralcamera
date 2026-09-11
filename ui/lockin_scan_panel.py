"""Scan panel: sweep the MCS2 stage, record the SR865A at every position.

Owns the scan parameters, the run/stop buttons and saving. The scan itself runs
on a worker thread (`instruments.lockin_scan.LockInScanner`); progress comes
back through Qt signals so the main window can grow the trace live, point by
point, exactly the way the camera app grew its interferogram.

Both instrument panels are frozen for the duration of a scan: their poll timers
stop and their controls grey out, so the scan thread is the only thing talking
to either device.
"""
from __future__ import annotations

import csv
import json
import os
import threading
from datetime import datetime

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from instruments.lockin_scan import LockInScanner, recommended_settle_factor

DEFAULT_SAVE_DIR = r"D:\LOCKIN"


class LockInScanPanel(QWidget):
    """Scan setup, execution and saving."""

    sig_progress = QtCore.pyqtSignal(int, int, float, float)
    sig_scan_done = QtCore.pyqtSignal(object, object)
    sig_status = QtCore.pyqtSignal(str)
    sig_running = QtCore.pyqtSignal(bool)

    def __init__(self, stage_panel, lockin_panel,
                 save_dir: str = DEFAULT_SAVE_DIR) -> None:
        super().__init__()
        self.stage_panel = stage_panel
        self.lockin_panel = lockin_panel
        self.save_dir = save_dir
        self.scanner = None
        self._abort = False
        self._running = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_timing_group())
        layout.addWidget(self._build_save_group())
        layout.addStretch()

        self.sig_progress.connect(self._on_progress)
        self.sig_scan_done.connect(self._on_scan_done)
        self.sig_status.connect(self.lbl_status.setText)

        self._settings = QtCore.QSettings("MIR_CAMERA", "LockInScan")
        self._restore_settings()
        for widget in self._persisted().values():
            widget.valueChanged.connect(self._save_settings)
        self._update_derived()

    # -- scan parameters -----------------------------------------------------
    def _build_scan_group(self) -> QGroupBox:
        g = QGroupBox("Scan")
        grid = QGridLayout(g)

        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(-100.0, 100.0)
        self.spin_start.setDecimals(5)
        self.spin_start.setSingleStep(0.01)
        self.spin_start.setSuffix(" mm")
        self.spin_start.setValue(-0.05)
        self.spin_start.valueChanged.connect(self._update_derived)
        grid.addWidget(QLabel("Start"), 0, 0)
        grid.addWidget(self.spin_start, 0, 1)
        btn_here_start = QPushButton("Here")
        btn_here_start.setFixedWidth(46)
        btn_here_start.setToolTip("Set from the stage's current position")
        btn_here_start.clicked.connect(lambda: self._use_current(self.spin_start))
        grid.addWidget(btn_here_start, 0, 2)

        self.spin_stop = QDoubleSpinBox()
        self.spin_stop.setRange(-100.0, 100.0)
        self.spin_stop.setDecimals(5)
        self.spin_stop.setSingleStep(0.01)
        self.spin_stop.setSuffix(" mm")
        self.spin_stop.setValue(0.05)
        self.spin_stop.valueChanged.connect(self._update_derived)
        grid.addWidget(QLabel("Stop"), 1, 0)
        grid.addWidget(self.spin_stop, 1, 1)
        btn_here_stop = QPushButton("Here")
        btn_here_stop.setFixedWidth(46)
        btn_here_stop.setToolTip("Set from the stage's current position")
        btn_here_stop.clicked.connect(lambda: self._use_current(self.spin_stop))
        grid.addWidget(btn_here_stop, 1, 2)

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(2, 100000)
        self.spin_steps.setValue(201)
        self.spin_steps.valueChanged.connect(self._update_derived)
        grid.addWidget(QLabel("Steps"), 2, 0)
        grid.addWidget(self.spin_steps, 2, 1, 1, 2)

        self.lbl_step = QLabel("-- µm")
        self.lbl_step.setStyleSheet("font-weight:600;")
        grid.addWidget(QLabel("Step size"), 3, 0)
        grid.addWidget(self.lbl_step, 3, 1, 1, 2)

        self.lbl_channel = QLabel("R")
        self.lbl_channel.setToolTip("Set on the Lock-in tab (Reading → Channel)")
        grid.addWidget(QLabel("Recording"), 4, 0)
        grid.addWidget(self.lbl_channel, 4, 1, 1, 2)

        btn_row = QHBoxLayout()
        self.btn_scan = QPushButton("Scan")
        self.btn_scan.setMinimumHeight(34)
        self.btn_scan.clicked.connect(self._start_scan)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setMinimumHeight(34)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_scan)
        btn_row.addWidget(self.btn_scan)
        btn_row.addWidget(self.btn_stop)
        grid.addLayout(btn_row, 5, 0, 1, 3)

        self.progress = QProgressBar()
        grid.addWidget(self.progress, 6, 0, 1, 3)

        self.lbl_status = QLabel("idle")
        self.lbl_status.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        grid.addWidget(self.lbl_status, 7, 0, 1, 3)
        return g

    # -- timing --------------------------------------------------------------
    def _build_timing_group(self) -> QGroupBox:
        g = QGroupBox("Synchronisation")
        grid = QGridLayout(g)

        self.spin_settle_factor = QDoubleSpinBox()
        self.spin_settle_factor.setRange(0.0, 50.0)
        self.spin_settle_factor.setDecimals(1)
        self.spin_settle_factor.setSingleStep(0.5)
        self.spin_settle_factor.setValue(5.0)
        self.spin_settle_factor.setSuffix(" × τ")
        self.spin_settle_factor.setToolTip(
            "Wait this many lock-in time constants after each move before "
            "reading. The output is the input through a low-pass filter, so it "
            "needs several τ to forget the previous position.\n"
            "Rule of thumb: 3τ at 6 dB/oct, 6τ at 24 dB/oct.")
        self.spin_settle_factor.valueChanged.connect(self._update_derived)
        grid.addWidget(QLabel("Settle"), 0, 0)
        grid.addWidget(self.spin_settle_factor, 0, 1)

        self.btn_auto_settle = QPushButton("Auto")
        self.btn_auto_settle.setFixedWidth(46)
        self.btn_auto_settle.setToolTip(
            "Set the settle factor from the filter slope on the instrument")
        self.btn_auto_settle.clicked.connect(self._auto_settle_factor)
        grid.addWidget(self.btn_auto_settle, 0, 2)

        self.spin_extra_settle = QDoubleSpinBox()
        self.spin_extra_settle.setRange(0.0, 60.0)
        self.spin_extra_settle.setDecimals(3)
        self.spin_extra_settle.setSingleStep(0.05)
        self.spin_extra_settle.setSuffix(" s")
        self.spin_extra_settle.setToolTip(
            "Fixed extra dwell added on top of the τ-based settle (mechanical "
            "ringing, chopper re-lock, ...)")
        self.spin_extra_settle.valueChanged.connect(self._update_derived)
        grid.addWidget(QLabel("Extra dwell"), 1, 0)
        grid.addWidget(self.spin_extra_settle, 1, 1, 1, 2)

        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(1, 1000)
        self.spin_samples.setValue(1)
        self.spin_samples.setToolTip(
            "Reads averaged per position. Their spread is saved as the point's "
            "error bar.")
        self.spin_samples.valueChanged.connect(self._update_derived)
        grid.addWidget(QLabel("Samples/point"), 2, 0)
        grid.addWidget(self.spin_samples, 2, 1, 1, 2)

        self.spin_sample_interval = QDoubleSpinBox()
        self.spin_sample_interval.setRange(0.0, 10.0)
        self.spin_sample_interval.setDecimals(3)
        self.spin_sample_interval.setSingleStep(0.01)
        self.spin_sample_interval.setSuffix(" s")
        self.spin_sample_interval.setSpecialValueText("auto (1 × τ)")
        self.spin_sample_interval.setToolTip(
            "Gap between averaged samples. 0 = one time constant, so the "
            "samples are (nearly) independent instead of N copies of the same "
            "filtered value.")
        self.spin_sample_interval.valueChanged.connect(self._update_derived)
        grid.addWidget(QLabel("Sample gap"), 3, 0)
        grid.addWidget(self.spin_sample_interval, 3, 1, 1, 2)

        self.chk_xyr = QCheckBox("Also record X, Y, R")
        self.chk_xyr.setChecked(True)
        self.chk_xyr.setToolTip(
            "One extra SNAP? per point. Costs a round trip, but saves the full "
            "quadrature information so the scan can be re-phased afterwards.")
        grid.addWidget(self.chk_xyr, 4, 0, 1, 3)

        self.chk_return = QCheckBox("Return to start when done")
        self.chk_return.setChecked(False)
        grid.addWidget(self.chk_return, 5, 0, 1, 3)

        self.lbl_estimate = QLabel("--")
        self.lbl_estimate.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_estimate.setWordWrap(True)
        grid.addWidget(QLabel("Estimate"), 6, 0)
        grid.addWidget(self.lbl_estimate, 6, 1, 1, 2)
        return g

    # -- save ----------------------------------------------------------------
    def _build_save_group(self) -> QGroupBox:
        g = QGroupBox("Save")
        v = QVBoxLayout(g)

        v.addWidget(QLabel("Folder"))
        dir_row = QHBoxLayout()
        self.edit_save_dir = QLineEdit(self.save_dir)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_save_dir)
        dir_row.addWidget(self.edit_save_dir)
        dir_row.addWidget(browse)
        v.addLayout(dir_row)

        v.addWidget(QLabel("Filename"))
        self.edit_filename = QLineEdit("lockin_scan")
        v.addWidget(self.edit_filename)

        self.chk_autosave = QCheckBox("Save automatically when a scan finishes")
        self.chk_autosave.setChecked(True)
        v.addWidget(self.chk_autosave)

        self.btn_save = QPushButton("Save now")
        self.btn_save.setMinimumHeight(30)
        self.btn_save.clicked.connect(lambda: self.save())
        v.addWidget(self.btn_save)

        self.lbl_saved = QLabel("nothing saved yet")
        self.lbl_saved.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_saved.setWordWrap(True)
        v.addWidget(self.lbl_saved)
        return g

    def _browse_save_dir(self) -> None:
        start = self.edit_save_dir.text().strip() or self.save_dir
        chosen = QFileDialog.getExistingDirectory(self, "Select save folder", start)
        if chosen:
            self.edit_save_dir.setText(chosen)
            self._save_settings()

    # -- derived readouts ----------------------------------------------------
    def _use_current(self, spin: QDoubleSpinBox) -> None:
        pos = self.stage_panel.use_current_position()
        if pos != pos:                       # NaN: stage offline
            self.sig_status.emit("stage position unknown (not connected)")
            return
        spin.setValue(pos)

    def _auto_settle_factor(self) -> None:
        lockin = self.lockin_panel.lockin
        if not lockin.is_connected:
            self.sig_status.emit("lock-in not connected")
            return
        try:
            factor = recommended_settle_factor(lockin.filter_slope)
        except Exception as e:  # noqa: BLE001
            self.sig_status.emit(f"could not read filter slope: {e}")
            return
        self.spin_settle_factor.setValue(factor)

    def _current_tau(self) -> float:
        """The time constant the estimate is based on: live from the instrument
        when connected, otherwise whatever the Lock-in tab is showing."""
        lockin = self.lockin_panel.lockin
        if lockin.is_connected:
            try:
                return float(lockin.time_constant)
            except Exception:  # noqa: BLE001
                pass
        data = self.lockin_panel.combo_tc.currentData()
        return float(data) if data else 0.1

    def _update_derived(self, *_a) -> None:
        n = self.spin_steps.value()
        span = abs(self.spin_stop.value() - self.spin_start.value())
        self.lbl_step.setText(f"{span / (n - 1) * 1000:.3f} µm" if n > 1 else "-- µm")
        self.lbl_channel.setText(self.lockin_panel.channel)

        tau = self._current_tau()
        gap = self.spin_sample_interval.value() or tau
        settle = self.spin_settle_factor.value() * tau + self.spin_extra_settle.value()
        per_point = settle + max(0, self.spin_samples.value() - 1) * gap
        # ~40 ms of stage move + handshake per point is what this rig measures;
        # it is only a rough guide, so it stays a constant rather than a setting.
        total = n * (per_point + 0.04)
        self.lbl_estimate.setText(
            f"τ = {tau:g} s → {settle:.3f} s settle, {per_point:.3f} s/point, "
            f"≈ {_fmt_duration(total)} total")

    # -- run -----------------------------------------------------------------
    def _start_scan(self) -> None:
        stage = self.stage_panel.stage
        lockin = self.lockin_panel.lockin
        if not stage.is_connected:
            self.sig_status.emit("stage not connected")
            return
        if not lockin.is_connected:
            self.sig_status.emit("lock-in not connected")
            return
        if self._running:
            return

        start = self.spin_start.value()
        stop = self.spin_stop.value()
        n = self.spin_steps.value()
        parameter = self.lockin_panel.channel
        samples = self.spin_samples.value()
        gap = self.spin_sample_interval.value() or None      # 0 -> auto (1 tau)
        settle_factor = self.spin_settle_factor.value()
        extra = self.spin_extra_settle.value()
        record_xyr = self.chk_xyr.isChecked()
        return_to_start = self.chk_return.isChecked()

        self._save_settings()
        self.scanner = LockInScanner(stage, lockin)
        self._abort = False
        self._set_running(True)
        self.progress.setMaximum(n)
        self.progress.setValue(0)

        def _work():
            try:
                pos, val = self.scanner.scan(
                    start, stop, n,
                    parameter=parameter,
                    samples=samples,
                    sample_interval_s=gap,
                    settle_factor=settle_factor,
                    extra_settle_s=extra,
                    record_xyr=record_xyr,
                    progress=lambda i, t, p, v: self.sig_progress.emit(i, t, p, v),
                    should_abort=lambda: self._abort,
                    status=lambda msg: self.sig_status.emit(msg))
                if return_to_start and not self._abort:
                    self.sig_status.emit("returning to start...")
                    stage.move_to(start)
                    stage.wait_for_stop()
                self.sig_scan_done.emit(pos, val)
            except Exception as e:  # noqa: BLE001
                self.sig_status.emit(f"scan error: {e}")
                self.sig_scan_done.emit(None, None)

        threading.Thread(target=_work, daemon=True).start()

    def _stop_scan(self) -> None:
        self.abort()
        self.sig_status.emit("stopping after this point...")

    def abort(self) -> None:
        """Ask a running scan to stop at the next point boundary."""
        self._abort = True

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.btn_scan.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.sig_running.emit(running)

    @QtCore.pyqtSlot(int, int, float, float)
    def _on_progress(self, i: int, n: int, position: float, value: float) -> None:
        self.progress.setValue(i)
        self.lbl_status.setText(f"point {i}/{n} @ {position:.5f} mm = {value:.6e}")

    @QtCore.pyqtSlot(object, object)
    def _on_scan_done(self, positions, values) -> None:
        self._set_running(False)
        if positions is None or len(positions) == 0:
            self.lbl_status.setText("scan aborted / no data")
            return
        requested = self.scanner.metadata.get("n_steps", len(positions))
        verb = "stopped at" if len(positions) < requested else "scan done:"
        self.lbl_status.setText(
            f"{verb} {len(positions)}/{requested} points in "
            f"{_fmt_duration(self.scanner.metadata.get('duration_s', 0))}")
        if self.chk_autosave.isChecked():
            self.save()

    @property
    def running(self) -> bool:
        return self._running

    def current_trace(self):
        """(positions, values) of the scan in progress or the last finished one."""
        if self.scanner is None:
            return None, None
        if self._running:
            return self.scanner.live_trace()
        return self.scanner.positions, self.scanner.values

    # -- saving --------------------------------------------------------------
    def save(self, quiet: bool = False) -> str:
        """Write <stamp>.<name>.csv / .npz / .json. Returns the stem, or ''."""
        if self.scanner is None or not self.scanner.has_data():
            if not quiet:
                self.lbl_saved.setText("nothing to save")
            return ""
        save_dir = self.edit_save_dir.text().strip() or self.save_dir
        name = self.edit_filename.text().strip() or "lockin_scan"
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            self.lbl_saved.setText(f"cannot create folder: {e}")
            return ""

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = os.path.join(save_dir, f"{stamp}.{name}")
        data = self.scanner.as_dict()
        meta = dict(self.scanner.metadata)
        meta["stage"] = self.stage_panel.stage.describe()
        meta["saved_at"] = stamp

        try:
            with open(stem + ".csv", "w", newline="") as f:
                writer = csv.writer(f)
                columns = ["position_mm", "target_mm", "value", "error", "time_s"]
                arrays = [data["positions_mm"], data["targets_mm"], data["values"],
                          data["errors"], data["timestamps_s"]]
                if "x" in data:
                    columns += ["x", "y", "r"]
                    arrays += [data["x"], data["y"], data["r"]]
                writer.writerow(columns)
                writer.writerows(zip(*arrays))
            np.savez(stem + ".npz", **{k: v for k, v in data.items() if v is not None},
                     metadata=json.dumps(meta))
            with open(stem + ".json", "w") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            self.lbl_saved.setText(f"save error: {e}")
            return ""

        self.lbl_saved.setText(f"saved {os.path.basename(stem)}.csv/.npz/.json")
        return stem

    # -- persistence ---------------------------------------------------------
    def _persisted(self) -> dict:
        return {
            "start": self.spin_start,
            "stop": self.spin_stop,
            "steps": self.spin_steps,
            "settle_factor": self.spin_settle_factor,
            "extra_settle": self.spin_extra_settle,
            "samples": self.spin_samples,
            "sample_interval": self.spin_sample_interval,
        }

    def _restore_settings(self) -> None:
        s = self._settings
        for key, widget in self._persisted().items():
            val = s.value(key, None)
            if val is None:
                continue
            cast = int if isinstance(widget, QSpinBox) else float
            try:
                widget.setValue(cast(val))
            except (TypeError, ValueError):
                pass
        for key, widget in (("save_dir", self.edit_save_dir),
                            ("filename", self.edit_filename)):
            val = s.value(key, None)
            if val:
                widget.setText(str(val))
        for key, box in (("record_xyr", self.chk_xyr),
                         ("return_to_start", self.chk_return),
                         ("autosave", self.chk_autosave)):
            val = s.value(key, None)
            if val is not None:
                box.setChecked(str(val).lower() in ("true", "1"))

    def _save_settings(self, *_a) -> None:
        s = self._settings
        for key, widget in self._persisted().items():
            s.setValue(key, widget.value())
        s.setValue("save_dir", self.edit_save_dir.text())
        s.setValue("filename", self.edit_filename.text())
        s.setValue("record_xyr", self.chk_xyr.isChecked())
        s.setValue("return_to_start", self.chk_return.isChecked())
        s.setValue("autosave", self.chk_autosave.isChecked())


def _fmt_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{int(seconds // 60)} min {int(seconds % 60)} s"
    return f"{int(seconds // 3600)} h {int((seconds % 3600) // 60)} min"
