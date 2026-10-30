"""Stage control panels for the GUI: Thorlabs motors + TWINS (SmarAct).

Connect and manual jog/move for every stage. Blocking operations (connect,
home, move) run in background threads so the UI never freezes; position is
polled on a timer. A "Simulate" checkbox lets you exercise the controls with
no hardware (default ON until the stages are wired up).

Thorlabs motors: THREE identical MTS50-Z8 stages, each on its own KDC101.
The controllers are told apart only by serial number, so every motor has its
own serial field (pre-filled from THORLABS_MOTORS below) and is pinned to it on
connect; on auto-pick (empty serial) the serials of the other motors and of
any other K-Cube on the PC (OTHER_KINESIS_SERIALS) are excluded so two drivers
can never open the same controller.

The actual TWINS interferogram scan (sweep stage + read camera ROI -> spectrum)
lives in its own panel; this one covers the connections + manual control.
"""
from __future__ import annotations

import threading

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from instruments.stage_driver import KinesisStage
from instruments.twins_stage import TwinsStage, TRAVEL_MIN_MM, TRAVEL_MAX_MM

# (label, KDC101 serial number) for each of the three MTS50-Z8 motors. These
# are only the FIRST-RUN defaults: the serial typed in the panel is remembered
# per PC (QSettings "MIR_CAMERA/Stages", same store as the other panels) and
# restored at the next start. An empty serial means "first free Thorlabs
# device" (the other motors' and the rotator's serials are excluded).
THORLABS_MOTORS = [
    ("Motor 1", ""),
    ("Motor 2", ""),
    ("Motor 3", ""),
]
THORLABS_STAGE_NAME = "MTS50-Z8"
# Other Thorlabs K-Cubes that may be plugged into this PC (e.g. the PRM1-Z8
# rotator's KDC101): never auto-picked for a motor.
OTHER_KINESIS_SERIALS = ["27267060"]


class StageController(QtCore.QObject):
    """Wraps a stage driver; runs blocking ops on a thread, emits updates."""

    status = QtCore.pyqtSignal(str)
    position = QtCore.pyqtSignal(str)
    busy_changed = QtCore.pyqtSignal(bool)

    def __init__(self, driver, fmt) -> None:
        super().__init__()
        self.driver = driver
        self._fmt = fmt
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def run(self, fn, done_msg="") -> None:
        """Run ``fn`` on a worker thread. ``done_msg`` is a string or a callable
        evaluated after ``fn`` succeeds (so it can report e.g. the serial)."""
        if self._busy:
            return
        self._busy = True
        self.busy_changed.emit(True)

        def _work():
            try:
                fn()
                msg = done_msg() if callable(done_msg) else done_msg
                if msg:
                    self.status.emit(msg)
            except Exception as e:  # noqa: BLE001
                self.status.emit(f"error: {e}")
            finally:
                self._busy = False
                self.busy_changed.emit(False)
                self._emit_pos()

        threading.Thread(target=_work, daemon=True).start()

    def poll(self) -> None:
        if not self._busy:
            self._emit_pos()

    def _emit_pos(self) -> None:
        try:
            if getattr(self.driver, "is_connected", False):
                self.position.emit(self._fmt(self.driver))
        except Exception:  # noqa: BLE001
            pass


class StagesPanel(QWidget):
    """Owns the Thorlabs-motor and TWINS-stage drivers, controllers and the
    position-poll timer. The control groups (``thorlabs_group``,
    ``twins_group``) are exposed so they can be placed into separate tabs;
    this widget itself stays hidden and just keeps the timer/threads alive.
    """

    def __init__(self) -> None:
        super().__init__()
        # Three identical Thorlabs motors (MTS50-Z8 on KDC101), told apart by
        # serial number. Serials are remembered across runs.
        self._settings = QtCore.QSettings("MIR_CAMERA", "Stages")
        self.motors = [
            KinesisStage(serial=self._load_motor_serial(i, default) or None,
                         stage_name=THORLABS_STAGE_NAME, name=label)
            for i, (label, default) in enumerate(THORLABS_MOTORS)]
        self.motor_ctls = [
            StageController(m, lambda d: f"{d.get_position_mm():.4f} mm")
            for m in self.motors]
        # Motor 1 doubles as the "delay" axis the Measure panel's Z-scan drives.
        self.delay = self.motors[0]
        self.delay_ctl = self.motor_ctls[0]

        self.twins = TwinsStage()
        self.twins_ctl = StageController(
            self.twins, lambda d: f"{d.get_position():.4f} mm")

        self._mw = []                       # per-motor widget dicts
        self.motor_groups = [self._build_motor_group(i) for i in range(len(self.motors))]
        self.thorlabs_group = self._build_thorlabs_group()
        self.twins_group = self._build_twins_group()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(500)

    # -- Thorlabs motors (KDC101 + MTS50-Z8, mm) -----------------------------
    def _build_thorlabs_group(self) -> QWidget:
        """Container stacking the three motor groups (one tab)."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        for g in self.motor_groups:
            v.addWidget(g)
        v.addStretch(1)
        return w

    def _build_motor_group(self, i: int) -> QGroupBox:
        drv, ctl = self.motors[i], self.motor_ctls[i]
        g = QGroupBox(f"{drv.name} ({drv.stage_name}, mm)")
        v = QVBoxLayout(g)
        w = {}

        ser = QHBoxLayout()
        ser.addWidget(QLabel("Serial"))
        w["serial"] = QLineEdit(drv.preferred_serial or "")
        w["serial"].setPlaceholderText("KDC101 serial (empty = first free)")
        w["serial"].setToolTip("Serial number printed on this motor's KDC101. "
                               "Three identical motors are told apart only by it.")
        # Remember the serial as soon as it is edited (and again on connect).
        w["serial"].editingFinished.connect(lambda i=i: self._save_motor_serial(i))
        ser.addWidget(w["serial"])
        v.addLayout(ser)

        w["sim"] = QCheckBox("Simulate (no hardware)")
        w["sim"].setChecked(True)
        v.addWidget(w["sim"])

        row = QHBoxLayout()
        w["connect"] = QPushButton("Connect")
        w["connect"].clicked.connect(lambda _=False, i=i: self._motor_connect(i))
        w["home"] = QPushButton("Home")
        w["home"].clicked.connect(lambda _=False, i=i: ctl.run(drv.home, "homed"))
        row.addWidget(w["connect"])
        row.addWidget(w["home"])
        v.addLayout(row)

        w["pos"] = QLabel("-- mm")
        w["pos"].setStyleSheet("font-weight:600;")
        v.addWidget(w["pos"])

        go = QHBoxLayout()
        go.addWidget(QLabel("Go to"))
        w["goto"] = QDoubleSpinBox()
        w["goto"].setRange(0.0, drv.travel_mm)
        w["goto"].setDecimals(4)
        w["goto"].setSingleStep(0.1)
        w["goto"].setSuffix(" mm")
        w["goto"].setToolTip(f"Travel 0 .. {drv.travel_mm:g} mm")
        go.addWidget(w["goto"])
        w["go"] = QPushButton("Go")
        w["go"].clicked.connect(lambda _=False, i=i: self._motor_goto(i))
        go.addWidget(w["go"])
        v.addLayout(go)

        jog = QHBoxLayout()
        jog.addWidget(QLabel("Jog"))
        w["step"] = QDoubleSpinBox()
        w["step"].setRange(0.0, drv.travel_mm)
        w["step"].setDecimals(4)
        w["step"].setSingleStep(0.01)
        w["step"].setValue(0.1)
        w["step"].setSuffix(" mm")
        jog.addWidget(w["step"])
        w["minus"] = QPushButton("-")
        w["minus"].setFixedWidth(32)
        w["minus"].clicked.connect(lambda _=False, i=i: self._motor_jog(i, -1.0))
        w["plus"] = QPushButton("+")
        w["plus"].setFixedWidth(32)
        w["plus"].clicked.connect(lambda _=False, i=i: self._motor_jog(i, +1.0))
        jog.addWidget(w["minus"])
        jog.addWidget(w["plus"])
        v.addLayout(jog)

        w["status"] = QLabel("offline")
        w["status"].setStyleSheet("color:#888; font-size:11px;")
        w["status"].setWordWrap(True)
        v.addWidget(w["status"])

        self._mw.append(w)
        ctl.position.connect(w["pos"].setText)
        ctl.status.connect(w["status"].setText)
        ctl.busy_changed.connect(lambda *_, i=i: self._refresh_motor(i))
        self._refresh_motor(i)
        return g

    def _motor_serial(self, i: int) -> str:
        return self._mw[i]["serial"].text().strip()

    def _load_motor_serial(self, i: int, default: str = "") -> str:
        return str(self._settings.value(f"motor{i}/serial", default, type=str)).strip()

    def _save_motor_serial(self, i: int) -> None:
        self._settings.setValue(f"motor{i}/serial", self._motor_serial(i))
        self._settings.sync()

    def _motor_connect(self, i: int) -> None:
        drv, ctl, w = self.motors[i], self.motor_ctls[i], self._mw[i]
        if drv.is_connected:
            ctl.run(drv.disconnect, "disconnected")
            return
        sim = w["sim"].isChecked()
        serial = self._motor_serial(i) or None
        drv.preferred_serial = serial
        self._save_motor_serial(i)
        # Never let an auto-pick grab another motor's KDC101 (or any other
        # K-Cube on the PC): exclude their serials and whatever is already open.
        others = [self._motor_serial(j) for j in range(len(self.motors)) if j != i]
        others += [m.serial for m in self.motors if m.is_connected and m is not drv]
        others += list(OTHER_KINESIS_SERIALS)

        def _connect():
            if not drv.connect(serial=serial, simulate=sim, exclude=others):
                raise RuntimeError(
                    f"no Thorlabs device for {drv.name}"
                    + (f" (serial {serial})" if serial else "") + " -- see console")

        ctl.run(_connect, lambda: f"connected via {drv.backend} (serial {drv.serial})")

    def _motor_goto(self, i: int) -> None:
        drv, ctl, w = self.motors[i], self.motor_ctls[i], self._mw[i]
        target = w["goto"].value()

        def _move():
            if not drv.move_to_mm(target):
                raise RuntimeError("move refused (not homed or outside travel)")

        ctl.run(_move, f"moved to {target:.4f} mm")

    def _motor_jog(self, i: int, sign: float) -> None:
        drv, ctl, w = self.motors[i], self.motor_ctls[i], self._mw[i]
        delta = sign * w["step"].value()

        def _move():
            if not drv.move_relative_mm(delta):
                raise RuntimeError("move refused (not homed or outside travel)")

        ctl.run(_move, f"jogged {delta:+.4f} mm")

    def _refresh_motor(self, i: int) -> None:
        drv, ctl, w = self.motors[i], self.motor_ctls[i], self._mw[i]
        conn, busy = drv.is_connected, ctl.busy
        w["connect"].setText("Disconnect" if conn else "Connect")
        w["connect"].setEnabled(not busy)
        w["sim"].setEnabled(not conn and not busy)
        w["serial"].setEnabled(not conn and not busy)
        for key in ("home", "go", "minus", "plus", "goto", "step"):
            w[key].setEnabled(conn and not busy)
        if not conn:
            w["pos"].setText("-- mm")

    # -- TWINS (SmarAct) stage ----------------------------------------------
    def _build_twins_group(self) -> QGroupBox:
        g = QGroupBox("TWINS Stage (SmarAct)")
        v = QVBoxLayout(g)

        self.t_sim = QCheckBox("Simulate (no hardware)")
        self.t_sim.setChecked(True)
        v.addWidget(self.t_sim)

        self.t_connect = QPushButton("Connect")
        self.t_connect.clicked.connect(self._twins_connect)
        v.addWidget(self.t_connect)

        self.t_pos = QLabel("-- mm")
        self.t_pos.setStyleSheet("font-weight:600;")
        v.addWidget(self.t_pos)

        go = QHBoxLayout()
        go.addWidget(QLabel("Go to"))
        self.t_goto = QDoubleSpinBox()
        # Software travel limits (the controller enforces none) -- the wedge
        # coordinates can be negative, so never assume a 0 lower bound.
        self.t_goto.setRange(TRAVEL_MIN_MM, TRAVEL_MAX_MM)
        self.t_goto.setToolTip(f"Travel limits (instruments/twins_stage.py): "
                               f"{TRAVEL_MIN_MM} .. {TRAVEL_MAX_MM} mm")
        self.t_goto.setDecimals(3)
        self.t_goto.setValue(24.0)
        self.t_goto.setSuffix(" mm")
        go.addWidget(self.t_goto)
        self.t_go = QPushButton("Go")
        self.t_go.clicked.connect(
            lambda: self.twins_ctl.run(
                lambda: (self.twins.move_to(self.t_goto.value()),
                         self.twins.wait_for_stop()),
                f"moved to {self.t_goto.value():.3f} mm"))
        go.addWidget(self.t_go)
        v.addLayout(go)

        # Jog by +/- a step in micrometres (relative move).
        jog = QHBoxLayout()
        jog.addWidget(QLabel("Jog"))
        self.t_step = QDoubleSpinBox()
        self.t_step.setRange(0.1, 5000.0)
        self.t_step.setDecimals(1)
        self.t_step.setSingleStep(1.0)
        self.t_step.setValue(10.0)
        self.t_step.setSuffix(" µm")
        jog.addWidget(self.t_step)
        self.t_minus = QPushButton("−")
        self.t_minus.clicked.connect(lambda: self._twins_jog(-1.0))
        self.t_plus = QPushButton("+")
        self.t_plus.clicked.connect(lambda: self._twins_jog(+1.0))
        jog.addWidget(self.t_minus)
        jog.addWidget(self.t_plus)
        v.addLayout(jog)

        self.t_status = QLabel("offline")
        self.t_status.setStyleSheet("color:#888; font-size:11px;")
        self.t_status.setWordWrap(True)
        v.addWidget(self.t_status)

        self.twins_ctl.position.connect(self.t_pos.setText)
        self.twins_ctl.status.connect(self.t_status.setText)
        self.twins_ctl.busy_changed.connect(lambda *_: self._refresh_twins())
        self._refresh_twins()
        return g

    def _twins_connect(self) -> None:
        if self.twins.is_connected:
            # Leave the wedge where it is on disconnect (no move to SAFE 25 mm).
            self.twins_ctl.run(lambda: self.twins.disconnect(safe=False), "disconnected")
        else:
            # Reference (so absolute positions are valid) then park at HOME 19 mm,
            # matching the reference repo. Connecting moves the wedge.
            sim = self.t_sim.isChecked()
            self.twins_ctl.run(
                lambda: self.twins.connect(simulate=sim, home=True), "connected")

    def _twins_jog(self, sign: float) -> None:
        # micrometres -> mm; refresh the cached position first so the relative
        # move is taken from the stage's actual current position.
        dmm = sign * self.t_step.value() / 1000.0
        self.twins_ctl.run(
            lambda: (self.twins.get_position(),
                     self.twins.move_by(dmm),
                     self.twins.wait_for_stop()),
            f"jogged {sign * self.t_step.value():+.1f} µm")

    def _refresh_twins(self) -> None:
        conn = self.twins.is_connected
        busy = self.twins_ctl.busy
        self.t_connect.setText("Disconnect" if conn else "Connect")
        self.t_connect.setEnabled(not busy)
        self.t_sim.setEnabled(not conn and not busy)
        for w in (self.t_go, self.t_goto, self.t_step, self.t_minus, self.t_plus):
            w.setEnabled(conn and not busy)
        if not conn:
            self.t_pos.setText("-- mm")

    def _poll(self) -> None:
        for ctl in self.motor_ctls:
            ctl.poll()
        self.twins_ctl.poll()

    def freeze(self, frozen: bool) -> None:
        """Pause position polling and lock the manual controls during a Measure
        scan that drives the stages from its own thread (avoids DLL contention)."""
        if frozen:
            self.timer.stop()
        else:
            self.timer.start(500)
        self.thorlabs_group.setEnabled(not frozen)
        self.twins_group.setEnabled(not frozen)

    def shutdown(self) -> None:
        for drv in (*self.motors, self.twins):
            try:
                if drv.is_connected:
                    drv.disconnect()
            except Exception:  # noqa: BLE001
                pass
