"""Measure tab -- extended-resolution TWINS scans in Thorlabs-motor configurations.

The TWINS (SmarAct) wedge is scanned several times, each time with a different
set of optics inserted in the beam by three Thorlabs MTS50-Z8 motors (one
optic per motor). Every motor has two positions, defined ONCE and remembered:

    Out  -> the motor's optic is clear of the beam (nothing in the interferometer)
    In   -> the optic is in the beam (the fixed position z1 / z2 / z3)

One Acquire runs the seven configurations below in order. For each one the
three motors are driven to their Out/In targets, then the wedge is scanned
over the (changeable) Start..Stop range and the full ROI frame is stored at
every wedge step. Each configuration is saved to its own .npz (raw
interferogram cube + measured wedge positions + motor positions) as soon as it
finishes, so an aborted run keeps everything acquired so far. No spectrum is
computed here: the per-configuration interferograms are combined offline.

    1  none       all three motors Out
    2  M1         motor 1 at z1
    3  M2         motor 2 at z2
    4  M3         motor 3 at z3
    5  M1 + M3    motors 1 and 3 at z1, z3
    6  M2 + M3    motors 2 and 3 at z2, z3
    7  M1+M2+M3   all three at z1, z2, z3
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from instruments.subtwinslv import TwinsScanner
from instruments.hyperspectral import DEFAULT_START_MM, DEFAULT_STOP_MM, DEFAULT_N_STEPS
from instruments.twins_stage import TRAVEL_MIN_MM, TRAVEL_MAX_MM
from ui.measure_hyperspectral import (
    LiveInterferogram, _free_gb, LOW_DISK_WARN_GB, LOW_DISK_ABORT_GB,
)

# The configurations, in acquisition order: (label, which motors are IN).
CONFIGURATIONS = [
    ("none  (all motors out)", (False, False, False)),
    ("M1  (z1)", (True, False, False)),
    ("M2  (z2)", (False, True, False)),
    ("M3  (z3)", (False, False, True)),
    ("M1 + M3  (z1, z3)", (True, False, True)),
    ("M2 + M3  (z2, z3)", (False, True, True)),
    ("M1 + M2 + M3  (z1, z2, z3)", (True, True, True)),
]
N_MOTORS = 3

# A motor counts as "arrived" when it reports no motion and sits within this
# distance of its target; the three motors move simultaneously.
MOTOR_SETTLE_TOL_MM = 0.002
MOTOR_MOVE_TIMEOUT_S = 90.0


class MeasurePanel(QWidget):
    sig_status = QtCore.pyqtSignal(str)
    sig_done = QtCore.pyqtSignal(object)                 # list of per-config results
    sig_point = QtCore.pyqtSignal(float, float, bool)    # pos, ROI-mean, first-of-scan
    sig_warn = QtCore.pyqtSignal(str, str)               # (title, message) -> modal warning

    def __init__(self, stages_panel, frame_source,
                 bg_provider=None, save_dir_provider=None,
                 meta_provider=None, save_dir: str = r"D:\CAMERA\extended") -> None:
        super().__init__()
        self.sp = stages_panel
        self.frame_source = frame_source
        self.bg_provider = bg_provider        # () -> (background_frame|None, subtract_bool)
        self.save_dir_provider = save_dir_provider  # () -> current save folder (camera)
        self.meta_provider = meta_provider    # () -> dict of camera metadata
        self.save_dir = save_dir              # fallback if no provider
        self._scan_meta = {}
        self._cam_meta = {}
        self.background_map = None
        self.background_subtracted = False
        self._abort = False
        self._paused = False
        self._low_disk_warned = False
        self._scan_thread = None
        self.live_monitor = None
        self.results = []                     # last run: one dict per configuration
        self._run_folder = None
        self._run_stamp = None
        self._save_fname = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_positions_group())
        layout.addWidget(self._build_config_group())
        layout.addWidget(self._build_run_group())

        # Persist everything between runs: restore first, THEN bind the saves.
        self._settings = QtCore.QSettings("MIR_CAMERA", "ExtendedResolution")
        self._restore_settings()
        for widget in self._persisted_spins().values():
            widget.valueChanged.connect(self._save_settings)
        for chk in self._persisted_checks().values():
            chk.toggled.connect(self._save_settings)
        self.edit_filename.editingFinished.connect(self._save_settings)
        self.edit_dir.editingFinished.connect(self._save_settings)

        self.sig_status.connect(self._on_status)
        self.sig_done.connect(self._on_done)
        self.sig_point.connect(self._on_point)
        self.sig_warn.connect(self._on_warn)
        self._update_step()
        self._update_config_label()

    # -- groups --------------------------------------------------------------
    def _build_scan_group(self) -> QGroupBox:
        g = QGroupBox("TWINS interferometer scan")
        grid = QGridLayout(g)
        self.spin_start = self._mm_spin(DEFAULT_START_MM)
        self.spin_stop = self._mm_spin(DEFAULT_STOP_MM)
        self.spin_steps = QSpinBox(); self.spin_steps.setRange(3, 1000000)
        self.spin_steps.setValue(DEFAULT_N_STEPS)
        for s in (self.spin_start, self.spin_stop, self.spin_steps):
            s.valueChanged.connect(self._update_step)
        self.lbl_step = QLabel("-- µm"); self.lbl_step.setStyleSheet("font-weight:600;")
        self.spin_frames = QSpinBox(); self.spin_frames.setRange(1, 200); self.spin_frames.setValue(1)
        grid.addWidget(QLabel("Start"), 0, 0); grid.addWidget(self.spin_start, 0, 1)
        grid.addWidget(QLabel("Stop"), 1, 0); grid.addWidget(self.spin_stop, 1, 1)
        grid.addWidget(QLabel("Steps"), 2, 0); grid.addWidget(self.spin_steps, 2, 1)
        grid.addWidget(QLabel("Step size"), 3, 0); grid.addWidget(self.lbl_step, 3, 1)
        grid.addWidget(QLabel("Frames/point"), 4, 0); grid.addWidget(self.spin_frames, 4, 1)
        return g

    def _build_positions_group(self) -> QGroupBox:
        g = QGroupBox("Thorlabs motor positions (mm)")
        grid = QGridLayout(g)
        note = QLabel("Defined once, remembered: Out = optic clear of the beam, "
                      "In = optic in the beam (z1, z2, z3).")
        note.setStyleSheet("color:#888; font-size:11px;")
        note.setWordWrap(True)
        grid.addWidget(note, 0, 0, 1, 3)
        grid.addWidget(QLabel("Motor"), 1, 0)
        grid.addWidget(QLabel("Out (no optics)"), 1, 1)
        grid.addWidget(QLabel("In (z)"), 1, 2)
        self.spin_out, self.spin_in = [], []
        motors = list(getattr(self.sp, "motors", []))
        for i in range(N_MOTORS):
            travel = motors[i].travel_mm if i < len(motors) else 50.0
            name = motors[i].name if i < len(motors) else f"Motor {i + 1}"
            s_out = self._motor_spin(0.0, travel)
            s_in = self._motor_spin(0.0, travel)
            s_in.setToolTip(f"z{i + 1}: fixed position of {name} when its optic is in the beam")
            grid.addWidget(QLabel(f"{name}  (z{i + 1})"), 2 + i, 0)
            grid.addWidget(s_out, 2 + i, 1)
            grid.addWidget(s_in, 2 + i, 2)
            self.spin_out.append(s_out)
            self.spin_in.append(s_in)
        return g

    def _build_config_group(self) -> QGroupBox:
        g = QGroupBox("Configurations (acquired in this order)")
        v = QVBoxLayout(g)
        self.chk_cfg = []
        for k, (label, _mask) in enumerate(CONFIGURATIONS):
            chk = QCheckBox(f"{k + 1}.  {label}")
            chk.setChecked(True)
            chk.toggled.connect(self._update_config_label)
            v.addWidget(chk)
            self.chk_cfg.append(chk)
        self.lbl_cfg = QLabel("")
        self.lbl_cfg.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_cfg.setWordWrap(True)
        v.addWidget(self.lbl_cfg)
        return g

    def _build_run_group(self) -> QGroupBox:
        g = QGroupBox("Run")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Acquire")
        self.btn_run.clicked.connect(self._start)
        self.btn_pause = QPushButton("Pause"); self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip("Hold the run before the next configuration "
                                  "(the current wedge scan finishes first). "
                                  "Press again to resume.")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_run); row.addWidget(self.btn_pause); row.addWidget(self.btn_stop)
        v.addLayout(row)
        row_dir = QHBoxLayout()
        self.edit_dir = QLineEdit(self._camera_dir())
        self.edit_dir.setToolTip(
            "Where this measurement's files are written. Each Acquire creates "
            "<date>.<filename>/ inside it. Leave empty to follow the camera "
            "panel's save folder.")
        self.btn_dir = QPushButton("Browse...")
        self.btn_dir.clicked.connect(self._browse_dir)
        row_dir.addWidget(QLabel("Save folder"))
        row_dir.addWidget(self.edit_dir, 1)
        row_dir.addWidget(self.btn_dir)
        v.addLayout(row_dir)
        row4 = QHBoxLayout()
        self.edit_filename = QLineEdit("extended")
        self.edit_filename.setToolTip("Base filename; configuration k is saved as "
                                      "<date>.<filename>_c<k>.npz in the run folder.")
        row4.addWidget(QLabel("Filename")); row4.addWidget(self.edit_filename, 1)
        v.addLayout(row4)
        self.progress = QProgressBar(); v.addWidget(self.progress)
        self.lbl_status = QLabel("idle"); self.lbl_status.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_status.setWordWrap(True); v.addWidget(self.lbl_status)
        return g

    # -- helpers -------------------------------------------------------------
    def _mm_spin(self, val):
        """Wedge position: limited to the software travel (may be negative)."""
        s = QDoubleSpinBox(); s.setRange(TRAVEL_MIN_MM, TRAVEL_MAX_MM)
        s.setDecimals(3); s.setSingleStep(0.1); s.setValue(val); s.setSuffix(" mm")
        s.setToolTip(f"Travel limits (instruments/twins_stage.py): "
                     f"{TRAVEL_MIN_MM} .. {TRAVEL_MAX_MM} mm")
        return s

    def _motor_spin(self, val, travel):
        s = QDoubleSpinBox(); s.setRange(0.0, float(travel)); s.setDecimals(4)
        s.setSingleStep(0.1); s.setValue(val); s.setSuffix(" mm")
        return s

    def _camera_dir(self) -> str:
        return (self.save_dir_provider() if self.save_dir_provider
                else None) or self.save_dir

    def _measure_dir(self) -> str:
        typed = self.edit_dir.text().strip() if hasattr(self, "edit_dir") else ""
        return typed or self._camera_dir()

    def _browse_dir(self) -> None:
        start = self._measure_dir()
        if not os.path.isdir(start):
            start = ""
        chosen = QFileDialog.getExistingDirectory(
            self, "Folder for extended-resolution measurements", start)
        if chosen:
            self.edit_dir.setText(os.path.normpath(chosen))
            self._save_settings()

    def _update_step(self) -> None:
        n = self.spin_steps.value()
        start, stop = self.spin_start.value(), self.spin_stop.value()
        if n > 1:
            self.lbl_step.setText(f"{abs(stop - start) / (n - 1) * 1000:.3g} µm")
        else:
            self.lbl_step.setText("-- µm")
        self._update_config_label()

    def _selected_configs(self) -> list:
        """[(index, label, in_mask), ...] for the ticked configurations, in order."""
        return [(k, label, mask) for k, (label, mask) in enumerate(CONFIGURATIONS)
                if self.chk_cfg[k].isChecked()]

    def _update_config_label(self) -> None:
        if not hasattr(self, "lbl_cfg"):
            return
        n_cfg = len(self._selected_configs())
        n = self.spin_steps.value()
        self.lbl_cfg.setText(
            f"{n_cfg} configuration(s) x {n} wedge steps = {n_cfg * n} points")

    # -- persistence ---------------------------------------------------------
    def _persisted_spins(self) -> dict:
        d = {"ext_start": self.spin_start, "ext_stop": self.spin_stop,
             "ext_steps": self.spin_steps, "ext_frames": self.spin_frames}
        for i in range(N_MOTORS):
            d[f"ext_motor{i}_out"] = self.spin_out[i]
            d[f"ext_motor{i}_in"] = self.spin_in[i]
        return d

    def _persisted_checks(self) -> dict:
        return {f"ext_cfg{k}_on": chk for k, chk in enumerate(self.chk_cfg)}

    def _restore_settings(self) -> None:
        for key, widget in self._persisted_spins().items():
            val = self._settings.value(key, None)
            if val is None:
                continue
            try:
                cast = int if isinstance(widget, QSpinBox) else float
                widget.setValue(cast(val))
            except (TypeError, ValueError):
                pass
        for key, chk in self._persisted_checks().items():
            v = self._settings.value(key, None)
            if v is not None:
                chk.setChecked(str(v).lower() == "true")
        fn = self._settings.value("ext_filename", None)
        if fn is not None:
            self.edit_filename.setText(str(fn))
        d = self._settings.value("ext_dir", None)
        if d is not None and str(d).strip():
            self.edit_dir.setText(str(d))

    def _save_settings(self, *args) -> None:
        for key, widget in self._persisted_spins().items():
            self._settings.setValue(key, widget.value())
        for key, chk in self._persisted_checks().items():
            self._settings.setValue(key, chk.isChecked())
        self._settings.setValue("ext_filename", self.edit_filename.text())
        self._settings.setValue("ext_dir", self.edit_dir.text())

    # -- run -----------------------------------------------------------------
    def _start(self) -> None:
        if not getattr(self.sp.twins, "is_connected", False):
            self.sig_status.emit("TWINS stage not connected")
            return
        motors = list(getattr(self.sp, "motors", []))[:N_MOTORS]
        if len(motors) < N_MOTORS:
            self.sig_status.emit("the stages panel does not provide three Thorlabs motors")
            return
        missing = [m.name for m in motors if not getattr(m, "is_connected", False)]
        if missing:
            self.sig_status.emit(f"Thorlabs motor(s) not connected: {', '.join(missing)}")
            return
        configs = self._selected_configs()
        if not configs:
            self.sig_status.emit("no configuration ticked")
            return
        if self.frame_source() is None:
            self.sig_status.emit("No live frame -- start the camera first")
            return
        bg, bg_sub = self.bg_provider() if self.bg_provider else (None, False)
        bg = None if bg is None else np.asarray(bg, dtype=np.float32)
        self._cam_meta = (self.meta_provider() or {}) if self.meta_provider else {}

        params = dict(
            start=self.spin_start.value(), stop=self.spin_stop.value(),
            n=self.spin_steps.value(), frames=self.spin_frames.value(),
            background=bg, bg_subtract=bool(bg_sub and bg is not None),
            out=[s.value() for s in self.spin_out],
            z=[s.value() for s in self.spin_in],
            configs=configs,
        )
        nsteps = params["n"]
        self._save_fname = self.edit_filename.text().strip() or "extended"
        self._scan_meta = dict(
            measurement="extended_resolution",
            start_mm=params["start"], stop_mm=params["stop"], n_steps=nsteps,
            step_um=(abs(params["stop"] - params["start"]) / (nsteps - 1) * 1000
                     if nsteps > 1 else None),
            frames_per_point=params["frames"],
            binning=self._cam_meta.get("binning"),
            background_subtracted=params["bg_subtract"],
            motor_names=[m.name for m in motors],
            motor_serials=[m.serial for m in motors],
            motor_out_mm=list(params["out"]), motor_in_mm=list(params["z"]),
            configurations=[dict(index=k + 1, label=label, motors_in=list(mask))
                            for k, label, mask in configs],
            filename=self._save_fname,
        )
        camera_folder = self._measure_dir()
        frame = self.frame_source()
        hb, wb = (int(frame.shape[0]), int(frame.shape[1])) if frame is not None else (0, 0)
        est_gb = len(configs) * nsteps * hb * wb * 4 / 1e9
        free_gb = _free_gb(camera_folder)
        if free_gb is not None and (free_gb < est_gb * 1.1 or free_gb < LOW_DISK_ABORT_GB):
            drive = os.path.splitdrive(os.path.abspath(camera_folder))[0] or camera_folder
            ans = QMessageBox.warning(
                self, "Low disk space",
                f"This acquisition needs roughly {est_gb:.1f} GB, but only "
                f"{free_gb:.1f} GB is free on {drive}\\.\n\nData may fail to save "
                f"mid-scan. Proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                self.sig_status.emit(
                    f"acquisition cancelled -- only {free_gb:.1f} GB free (need ~{est_gb:.1f} GB)")
                return
        self._low_disk_warned = False
        self._run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_folder = os.path.join(camera_folder, f"{self._run_stamp}.{self._save_fname}")
        self._abort = False
        self._paused = False
        self.results = []
        self.btn_run.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(True)
        self.sp.freeze(True)
        self.progress.setMaximum(len(configs))
        self.progress.setValue(0)
        if self.live_monitor is None:
            self.live_monitor = LiveInterferogram()
        self.live_monitor.reset()
        self.live_monitor.show(); self.live_monitor.raise_()
        self._scan_thread = threading.Thread(target=self._worker, args=(params, motors),
                                             daemon=True)
        self._scan_thread.start()

    @QtCore.pyqtSlot(float, float, bool)
    def _on_point(self, pos: float, value: float, is_first: bool) -> None:
        if self.live_monitor is None:
            return
        if is_first:
            self.live_monitor.reset()
        self.live_monitor.add_point(pos, value)

    def _stop(self) -> None:
        self._abort = True
        self._paused = False
        self.sig_status.emit("stopping...")

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.btn_pause.setText("Resume" if self._paused else "Pause")
        self.sig_status.emit("paused" if self._paused else "resuming...")

    def _wait_if_paused(self) -> None:
        while self._paused and not self._abort:
            time.sleep(0.1)

    # -- motors --------------------------------------------------------------
    def _move_motors(self, motors, targets) -> list:
        """Send all motors to their targets at once, then wait until every one
        has stopped within MOTOR_SETTLE_TOL_MM. Returns the measured positions.
        Raises on a refused move or a timeout so the run stops rather than
        scanning with an optic half-way in."""
        for m, t in zip(motors, targets):
            if not m.move_to_mm(float(t), wait=False):
                raise RuntimeError(f"{m.name}: move to {t:.4f} mm refused "
                                   "(not homed or outside travel)")
        deadline = time.time() + MOTOR_MOVE_TIMEOUT_S
        while True:
            pos = [m.get_position_mm() for m in motors]
            arrived = [abs(pm - t) <= MOTOR_SETTLE_TOL_MM and not m.is_moving()
                       for m, pm, t in zip(motors, pos, targets)]
            if all(arrived):
                return pos
            if self._abort:
                for m in motors:
                    m.stop()
                raise RuntimeError("aborted while moving the motors")
            if time.time() > deadline:
                late = ", ".join(f"{m.name} at {pm:.4f} (target {t:.4f})"
                                 for m, pm, t, ok in zip(motors, pos, targets, arrived) if not ok)
                raise RuntimeError(f"motor move timed out: {late}")
            time.sleep(0.1)

    # -- worker --------------------------------------------------------------
    def _worker(self, p: dict, motors) -> None:
        results = []
        try:
            scanner = TwinsScanner(self.sp.twins, self.frame_source)
            bg = p["background"]
            bg_for_scan = bg if p["bg_subtract"] else None
            self.background_subtracted = bool(p["bg_subtract"] and bg is not None)
            self.background_map = None if bg is None else np.asarray(bg, dtype=np.float32)
            os.makedirs(self._run_folder, exist_ok=True)

            for m in motors:
                if not getattr(m, "is_homed", True):
                    self.sig_status.emit(f"homing {m.name}...")
                    m.home()

            total = len(p["configs"])
            for k, (idx, label, mask) in enumerate(p["configs"]):
                self._wait_if_paused()
                if self._abort:
                    break
                targets = [p["z"][i] if mask[i] else p["out"][i] for i in range(N_MOTORS)]
                self.sig_status.emit(
                    f"pt {k + 1}/{total} cfg {idx + 1} [{label}]: moving motors to "
                    + ", ".join(f"{t:.3f}" for t in targets) + " mm")
                actual = self._move_motors(motors, targets)

                def prog(i, tot, pos, value, _k=k, _idx=idx, _l=label, _t=total):
                    self.sig_status.emit(
                        f"pt {_k + 1}/{_t} cfg {_idx + 1} [{_l}]: wedge point {i}/{tot}")
                    self.sig_point.emit(float(pos), float(value), i == 1)

                positions, datacube = scanner.scan_cube(
                    p["start"], p["stop"], p["n"], None,
                    frames_avg=p["frames"], background=bg_for_scan,
                    progress=prog, should_abort=lambda: self._abort,
                    status_cb=lambda m: self.sig_status.emit(m))
                if datacube is None or len(positions) < 3:
                    continue
                res = dict(index=idx, label=label, motors_in=list(mask),
                           motor_targets_mm=list(targets), motor_positions_mm=list(actual),
                           positions=np.asarray(positions, dtype=float),
                           datacube=np.asarray(datacube))
                try:
                    res["path"] = self._save_config_npz(res)
                    self.sig_status.emit(
                        f"pt {k + 1}/{total} cfg {idx + 1}: saved {os.path.basename(res['path'])}")
                except Exception as e:  # noqa: BLE001
                    self.sig_status.emit(f"pt {k + 1}/{total} cfg {idx + 1}: SAVE FAILED: {e}")
                results.append(res)

                free = _free_gb(self._run_folder)
                if free is not None and free < LOW_DISK_ABORT_GB:
                    self.sig_warn.emit(
                        "Disk full - acquisition stopped",
                        f"Only {free:.2f} GB free on the save drive. Stopping to avoid "
                        f"corrupt saves. {k + 1} of {total} configurations were acquired.")
                    self.sig_status.emit(f"ABORTED: disk full ({free:.2f} GB free)")
                    self._abort = True
                    break
                if (free is not None and free < LOW_DISK_WARN_GB
                        and not self._low_disk_warned):
                    self._low_disk_warned = True
                    self.sig_warn.emit(
                        "Low disk space",
                        f"Only {free:.1f} GB free on the save drive - free up space "
                        f"or this acquisition may fail to finish saving.")
        except Exception as e:  # noqa: BLE001
            self.sig_status.emit(f"error: {e}")
        self.sig_done.emit(results)

    # -- saving --------------------------------------------------------------
    def _save_config_npz(self, res: dict) -> str:
        """One configuration -> <run-stamp>.<filename>_c<k>.npz in the run folder."""
        stem = os.path.join(self._run_folder,
                            f"{self._run_stamp}.{self._save_fname}_c{res['index'] + 1}")
        datacube = np.asarray(res["datacube"])
        meta = dict(self._scan_meta or {})
        meta.update(config_index=res["index"] + 1, config_label=res["label"],
                    motors_in=list(res["motors_in"]),
                    motor_targets_mm=[float(v) for v in res["motor_targets_mm"]],
                    motor_positions_mm=[float(v) for v in res["motor_positions_mm"]],
                    n_positions=int(datacube.shape[0]), cube_shape=list(datacube.shape),
                    saved_local_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self._cam_meta:
            meta["camera"] = self._cam_meta
        kw = dict(
            raw_interferogram=datacube,                       # (n_pos, h, w)
            twins_positions_mm=np.asarray(res["positions"]),  # measured wedge positions
            roi_mean=datacube.reshape(datacube.shape[0], -1).mean(axis=1),
            config_index=int(res["index"] + 1),
            config_label=str(res["label"]),
            motors_in=np.asarray(res["motors_in"], bool),
            motor_targets_mm=np.asarray(res["motor_targets_mm"], float),
            motor_positions_mm=np.asarray(res["motor_positions_mm"], float),
            metadata=np.array(meta, dtype=object),
            metadata_json=json.dumps(meta, default=str, indent=2))
        if self.background_map is not None:
            kw["background"] = np.asarray(self.background_map)
            kw["background_subtracted"] = bool(self.background_subtracted)
        np.savez(stem + ".npz", **kw)   # uncompressed: fast, raw barely compresses
        return stem + ".npz"

    def _save_run_json(self, results) -> str:
        meta = dict(self._scan_meta or {})
        meta.update(
            run_folder=self._run_folder,
            n_acquired=len(results), aborted=bool(self._abort),
            files=[os.path.basename(r.get("path", "")) for r in results],
            acquired=[dict(config_index=r["index"] + 1, label=r["label"],
                           motor_positions_mm=[float(v) for v in r["motor_positions_mm"]],
                           n_positions=int(len(r["positions"])))
                      for r in results],
            saved_local_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self._cam_meta:
            meta["camera"] = self._cam_meta
        path = os.path.join(self._run_folder, f"{self._run_stamp}.{self._save_fname}_run.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, default=str, indent=2)
        return path

    # -- signal slots --------------------------------------------------------
    @QtCore.pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)
        if msg.startswith("pt "):
            try:
                self.progress.setValue(int(msg.split()[1].split("/")[0]))
            except Exception:  # noqa: BLE001
                pass

    @QtCore.pyqtSlot(str, str)
    def _on_warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    @QtCore.pyqtSlot(object)
    def _on_done(self, results) -> None:
        self.btn_run.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self._paused = False
        self.btn_stop.setEnabled(False)
        self.sp.freeze(False)
        self.results = list(results or [])
        if not self.results:
            self.lbl_status.setText("no data (aborted or scan failed)")
            return
        try:
            self._save_run_json(self.results)
        except Exception as e:  # noqa: BLE001
            self.lbl_status.setText(f"done, but run summary not written: {e}")
            return
        shape = self.results[0]["datacube"].shape
        self.lbl_status.setText(
            f"done: {len(self.results)} configuration(s), cube {shape} (pos x h x w)"
            f"{' -- ABORTED early' if self._abort else ''}  |  saved in {self._run_folder}")

    def shutdown(self) -> None:
        """Abort a running acquisition and join its thread BEFORE the caller
        disconnects the stages (the worker drives the TWINS + Thorlabs DLLs)."""
        self._abort = True
        t = self._scan_thread
        if t is not None and t.is_alive():
            t.join(timeout=10.0)
        if self.live_monitor is not None:
            self.live_monitor.close()
