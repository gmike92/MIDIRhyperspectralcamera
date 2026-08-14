"""Stage control panels for the GUI: Thorlabs delay stage + TWINS (SmarAct).

Connect and manual jog/move for both stages. Blocking operations (connect,
home, move) run in background threads so the UI never freezes; position is
polled on a timer. A "Simulate" checkbox lets you exercise the controls with
no hardware (default ON until the stages are wired up).

The actual TWINS interferogram scan (sweep stage + read camera ROI -> spectrum)
is a later step; this panel covers the connections + manual control.
"""
from __future__ import annotations

import threading

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from instruments.stage_driver import DelayStage
from instruments.rotator_stage import RotatorStage
from instruments.twins_stage import TwinsStage


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

    def run(self, fn, done_msg: str = "") -> None:
        if self._busy:
            return
        self._busy = True
        self.busy_changed.emit(True)

        def _work():
            try:
                fn()
                if done_msg:
                    self.status.emit(done_msg)
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
    """Owns the delay-stage and TWINS-stage drivers, controllers and the
    position-poll timer. The two control groups (``delay_group`` and
    ``twins_group``) are exposed so they can be placed into separate tabs;
    this widget itself stays hidden and just keeps the timer/threads alive.
    """

    def __init__(self) -> None:
        super().__init__()
        self.delay = DelayStage()
        self.twins = TwinsStage()
        # This rig has TWO KDC101 controllers; pin the rotator to its serial so
        # it never collides with the delay stage's KDC101 (and vice-versa).
        self.rotator = RotatorStage(serial="27256774")
        self.delay_ctl = StageController(
            self.delay,
            lambda d: f"{d.get_position_mm():.4f} mm")
        self.twins_ctl = StageController(
            self.twins, lambda d: f"{d.get_position():.4f} mm")
        self.rotator_ctl = StageController(
            self.rotator, lambda d: f"{d.get_position_deg():.4f} deg")

        self.delay_group = self._build_delay_group()
        self.twins_group = self._build_twins_group()
        self.rotator_group = self._build_rotator_group()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(500)

    # -- Thorlabs delay stage ------------------------------------------------
    def _build_delay_group(self) -> QGroupBox:
        g = QGroupBox("Thorlabs Stage (mm)")
        v = QVBoxLayout(g)

        self.d_sim = QCheckBox("Simulate (no hardware)")
        self.d_sim.setChecked(True)
        v.addWidget(self.d_sim)

        row = QHBoxLayout()
        self.d_connect = QPushButton("Connect")
        self.d_connect.clicked.connect(self._delay_connect)
        self.d_home = QPushButton("Home")
        self.d_home.clicked.connect(
            lambda: self.delay_ctl.run(self.delay.home, "homed"))
        row.addWidget(self.d_connect)
        row.addWidget(self.d_home)
        v.addLayout(row)

        self.d_pos = QLabel("-- mm")
        self.d_pos.setStyleSheet("font-weight:600;")
        v.addWidget(self.d_pos)

        go = QHBoxLayout()
        go.addWidget(QLabel("Go to"))
        self.d_goto = QDoubleSpinBox()
        # Allow negative absolute targets (LTS300 reconfigured for a negative
        # coordinate range). The controller still enforces its own soft limits.
        self.d_goto.setRange(-300.0, 300.0)
        self.d_goto.setDecimals(4)
        self.d_goto.setSingleStep(0.1)
        self.d_goto.setSuffix(" mm")
        go.addWidget(self.d_goto)
        self.d_go = QPushButton("Go")
        self.d_go.clicked.connect(
            lambda: self.delay_ctl.run(
                lambda: self.delay.move_to_mm(self.d_goto.value()),
                f"moved to {self.d_goto.value():.4f} mm"))
        go.addWidget(self.d_go)
        v.addLayout(go)

        jog = QHBoxLayout()
        jog.addWidget(QLabel("Jog"))
        self.d_step = QDoubleSpinBox()
        self.d_step.setRange(0.0, 50.0)
        self.d_step.setDecimals(4)
        self.d_step.setSingleStep(0.01)
        self.d_step.setValue(0.1)
        self.d_step.setSuffix(" mm")
        jog.addWidget(self.d_step)
        self.d_minus = QPushButton("-")
        self.d_minus.setFixedWidth(32)
        self.d_minus.clicked.connect(
            lambda: self.delay_ctl.run(
                lambda: self.delay.move_relative_mm(-self.d_step.value())))
        self.d_plus = QPushButton("+")
        self.d_plus.setFixedWidth(32)
        self.d_plus.clicked.connect(
            lambda: self.delay_ctl.run(
                lambda: self.delay.move_relative_mm(self.d_step.value())))
        jog.addWidget(self.d_minus)
        jog.addWidget(self.d_plus)
        v.addLayout(jog)

        self.d_status = QLabel("offline")
        self.d_status.setStyleSheet("color:#888; font-size:11px;")
        self.d_status.setWordWrap(True)
        v.addWidget(self.d_status)

        self.delay_ctl.position.connect(self.d_pos.setText)
        self.delay_ctl.status.connect(self.d_status.setText)
        self.delay_ctl.busy_changed.connect(lambda *_: self._refresh_delay())
        self._refresh_delay()
        return g

    def _delay_connect(self) -> None:
        if self.delay.is_connected:
            self.delay_ctl.run(self.delay.disconnect, "disconnected")
        else:
            sim = self.d_sim.isChecked()
            # Exclude the rotator's KDC101 so auto-pick never grabs it.
            other = [self.rotator.preferred_serial] if self.rotator.preferred_serial else None
            self.delay_ctl.run(
                lambda: self.delay.connect(simulate=sim, exclude=other), "connected")

    def _refresh_delay(self) -> None:
        conn = self.delay.is_connected
        busy = self.delay_ctl.busy
        self.d_connect.setText("Disconnect" if conn else "Connect")
        self.d_sim.setEnabled(not conn and not busy)
        for w in (self.d_home, self.d_go, self.d_minus, self.d_plus,
                  self.d_goto, self.d_step):
            w.setEnabled(conn and not busy)
        self.d_connect.setEnabled(not busy)
        if not conn:
            self.d_pos.setText("-- mm")

    # -- Thorlabs rotator (KDC101, degrees) ---------------------------------
    def _build_rotator_group(self) -> QGroupBox:
        g = QGroupBox("Thorlabs Rotator (deg)")
        v = QVBoxLayout(g)

        self.r_sim = QCheckBox("Simulate (no hardware)")
        self.r_sim.setChecked(True)
        v.addWidget(self.r_sim)

        row = QHBoxLayout()
        self.r_connect = QPushButton("Connect")
        self.r_connect.clicked.connect(self._rotator_connect)
        self.r_home = QPushButton("Home")
        self.r_home.clicked.connect(
            lambda: self.rotator_ctl.run(self.rotator.home, "homed"))
        row.addWidget(self.r_connect)
        row.addWidget(self.r_home)
        v.addLayout(row)

        self.r_pos = QLabel("-- deg")
        self.r_pos.setStyleSheet("font-weight:600;")
        v.addWidget(self.r_pos)

        go = QHBoxLayout()
        go.addWidget(QLabel("Go to"))
        self.r_goto = QDoubleSpinBox()
        self.r_goto.setRange(0.0, 360.0)
        self.r_goto.setDecimals(4)
        self.r_goto.setSingleStep(1.0)
        self.r_goto.setSuffix(" deg")
        go.addWidget(self.r_goto)
        self.r_go = QPushButton("Go")
        self.r_go.clicked.connect(
            lambda: self.rotator_ctl.run(
                lambda: self.rotator.move_to_deg(self.r_goto.value()),
                f"moved to {self.r_goto.value():.4f} deg"))
        go.addWidget(self.r_go)
        v.addLayout(go)

        jog = QHBoxLayout()
        jog.addWidget(QLabel("Jog"))
        self.r_step = QDoubleSpinBox()
        self.r_step.setRange(0.0, 180.0)
        self.r_step.setDecimals(3)
        self.r_step.setSingleStep(0.5)
        self.r_step.setValue(1.0)
        self.r_step.setSuffix(" deg")
        jog.addWidget(self.r_step)
        self.r_minus = QPushButton("−")
        self.r_minus.setFixedWidth(32)
        self.r_minus.clicked.connect(
            lambda: self.rotator_ctl.run(
                lambda: self.rotator.move_relative_deg(-self.r_step.value())))
        self.r_plus = QPushButton("+")
        self.r_plus.setFixedWidth(32)
        self.r_plus.clicked.connect(
            lambda: self.rotator_ctl.run(
                lambda: self.rotator.move_relative_deg(self.r_step.value())))
        jog.addWidget(self.r_minus)
        jog.addWidget(self.r_plus)
        v.addLayout(jog)

        self.r_status = QLabel("offline")
        self.r_status.setStyleSheet("color:#888; font-size:11px;")
        self.r_status.setWordWrap(True)
        v.addWidget(self.r_status)

        self.rotator_ctl.position.connect(self.r_pos.setText)
        self.rotator_ctl.status.connect(self.r_status.setText)
        self.rotator_ctl.busy_changed.connect(lambda *_: self._refresh_rotator())
        self._refresh_rotator()
        return g

    def _rotator_connect(self) -> None:
        if self.rotator.is_connected:
            self.rotator_ctl.run(self.rotator.disconnect, "disconnected")
        else:
            sim = self.r_sim.isChecked()
            self.rotator_ctl.run(lambda: self.rotator.connect(simulate=sim), "connected")

    def _refresh_rotator(self) -> None:
        conn = self.rotator.is_connected
        busy = self.rotator_ctl.busy
        self.r_connect.setText("Disconnect" if conn else "Connect")
        self.r_sim.setEnabled(not conn and not busy)
        for w in (self.r_home, self.r_go, self.r_minus, self.r_plus,
                  self.r_goto, self.r_step):
            w.setEnabled(conn and not busy)
        self.r_connect.setEnabled(not busy)
        if not conn:
            self.r_pos.setText("-- deg")

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
        self.t_goto.setRange(0, 30)
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
        self.delay_ctl.poll()
        self.twins_ctl.poll()
        self.rotator_ctl.poll()

    def freeze(self, frozen: bool) -> None:
        """Pause position polling and lock the manual controls during a Measure
        scan that drives the stages from its own thread (avoids DLL contention)."""
        if frozen:
            self.timer.stop()
        else:
            self.timer.start(500)
        self.delay_group.setEnabled(not frozen)
        self.twins_group.setEnabled(not frozen)
        self.rotator_group.setEnabled(not frozen)

    def shutdown(self) -> None:
        for drv in (self.delay, self.twins, self.rotator):
            try:
                if drv.is_connected:
                    drv.disconnect()
            except Exception:  # noqa: BLE001
                pass
