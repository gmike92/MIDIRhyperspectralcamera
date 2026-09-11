"""SmarAct MCS2 stage panel: connect, reference, jog and absolute moves.

Owns the `MCS2Stage` driver and its `DeviceController`. Blocking calls run on a
worker thread; the position is polled on a 500 ms timer that `freeze()` stops
for the duration of a scan, so the scan thread is the only caller talking to
SmarActCTL while it runs.
"""
from __future__ import annotations

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from instruments.mcs2_stage import MCS2Stage
from ui.device_controller import DeviceController


class MCS2Panel(QWidget):
    """Connection + manual control for one MCS2 channel."""

    sig_status = QtCore.pyqtSignal(str)
    sig_position = QtCore.pyqtSignal(float)     # mm, for the main window readout

    def __init__(self) -> None:
        super().__init__()
        self.stage = MCS2Stage()
        self.ctl = DeviceController(self.stage, lambda d: f"{d.get_position():.6f} mm")
        self.latest_position = float("nan")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.connection_group = self._build_connection_group()
        self.motion_group = self._build_motion_group()
        layout.addWidget(self.connection_group)
        layout.addWidget(self.motion_group)
        layout.addStretch()

        self.ctl.status.connect(self._set_status)
        self.ctl.reading.connect(self._on_reading)
        self.ctl.busy_changed.connect(lambda *_: self._refresh())

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.ctl.poll)
        self.timer.start(500)

        self._settings = QtCore.QSettings("MIR_CAMERA", "MCS2Panel")
        self._restore_settings()
        self._refresh()

    # -- connection ----------------------------------------------------------
    def _build_connection_group(self) -> QGroupBox:
        g = QGroupBox("Stage (SmarAct MCS2)")
        grid = QGridLayout(g)

        self.combo_locator = QComboBox()
        self.combo_locator.setEditable(True)
        self.combo_locator.addItem("Simulate (no hardware)")
        self.combo_locator.setToolTip(
            "MCS2 locator, e.g. usb:sn:MCS2-00001234 or network:192.168.1.200. "
            '"Scan" fills this from the controllers found on this PC.')
        grid.addWidget(QLabel("Device"), 0, 0)
        grid.addWidget(self.combo_locator, 0, 1)

        self.spin_channel = QSpinBox()
        self.spin_channel.setRange(0, 63)
        self.spin_channel.setToolTip("MCS2 channel index the positioner is on")
        grid.addWidget(QLabel("Channel"), 1, 0)
        grid.addWidget(self.spin_channel, 1, 1)

        row = QHBoxLayout()
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._toggle_connect)
        self.btn_scan_devices = QPushButton("Scan")
        self.btn_scan_devices.clicked.connect(self._scan_devices)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_scan_devices)
        grid.addLayout(row, 2, 0, 1, 2)

        self.lbl_status = QLabel("offline")
        self.lbl_status.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        grid.addWidget(self.lbl_status, 3, 0, 1, 2)
        return g

    def _scan_devices(self) -> None:
        found = MCS2Stage.find_devices()
        current = self.combo_locator.currentText()
        self.combo_locator.clear()
        self.combo_locator.addItem("Simulate (no hardware)")
        for loc in found:
            self.combo_locator.addItem(loc)
        if found:
            self.combo_locator.setCurrentIndex(1)
            self._set_status(f"found {len(found)} MCS2 device(s)")
        else:
            self.combo_locator.setCurrentText(current)
            self._set_status("no MCS2 devices found")

    def _toggle_connect(self) -> None:
        if self.stage.is_connected:
            self.ctl.run(self.stage.disconnect, "disconnected")
            return
        text = self.combo_locator.currentText().strip()
        simulate = text.lower().startswith("simulate") or not text
        locator = None if simulate else text
        channel = self.spin_channel.value()
        self._save_settings()

        def _connect():
            if not self.stage.connect(simulate=simulate, locator=locator,
                                      channel=channel):
                raise RuntimeError(self.stage.last_error or "connection refused")
            self.sig_status.emit(self.stage.describe())
            self._push_limits()

        self.ctl.run(_connect)

    def _push_limits(self) -> None:
        """Widen the spin boxes to the stage's configured range limits."""
        lo, hi = self.stage.get_limits_mm()
        if lo == hi:                    # 0/0 means "no software limits set"
            lo, hi = -100.0, 100.0
        QtCore.QMetaObject.invokeMethod(
            self, "_apply_limits", QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(float, float(lo)), QtCore.Q_ARG(float, float(hi)))

    @QtCore.pyqtSlot(float, float)
    def _apply_limits(self, lo: float, hi: float) -> None:
        self.spin_goto.setRange(lo, hi)
        self.lbl_limits.setText(f"{lo:.3f} … {hi:.3f} mm")

    # -- motion --------------------------------------------------------------
    def _build_motion_group(self) -> QGroupBox:
        g = QGroupBox("Motion")
        grid = QGridLayout(g)

        self.lbl_position = QLabel("-- mm")
        self.lbl_position.setStyleSheet("font-size:20px; font-weight:600;")
        grid.addWidget(self.lbl_position, 0, 0, 1, 2)

        ref_row = QHBoxLayout()
        self.btn_reference = QPushButton("Find reference")
        self.btn_reference.setToolTip(
            "The MCS2 sensors are incremental: absolute positions only mean "
            "something after referencing once per power-up.")
        self.btn_reference.clicked.connect(
            lambda: self.ctl.run(self.stage.find_reference, "referenced"))
        self.btn_calibrate = QPushButton("Calibrate")
        self.btn_calibrate.setToolTip(
            "Sensor calibration sequence — moves up to several mm. Only needed "
            "after the mechanics or the positioner type changed.")
        self.btn_calibrate.clicked.connect(self._calibrate)
        ref_row.addWidget(self.btn_reference)
        ref_row.addWidget(self.btn_calibrate)
        grid.addLayout(ref_row, 1, 0, 1, 2)

        self.lbl_limits = QLabel("--")
        self.lbl_limits.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(QLabel("Range"), 2, 0)
        grid.addWidget(self.lbl_limits, 2, 1)

        go = QHBoxLayout()
        self.spin_goto = QDoubleSpinBox()
        self.spin_goto.setRange(-100.0, 100.0)
        self.spin_goto.setDecimals(5)
        self.spin_goto.setSingleStep(0.01)
        self.spin_goto.setSuffix(" mm")
        self.btn_go = QPushButton("Go")
        self.btn_go.clicked.connect(self._go_to)
        go.addWidget(self.spin_goto)
        go.addWidget(self.btn_go)
        grid.addWidget(QLabel("Go to"), 3, 0)
        grid.addLayout(go, 3, 1)

        jog = QHBoxLayout()
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.001, 100000.0)
        self.spin_step.setDecimals(3)
        self.spin_step.setSingleStep(1.0)
        self.spin_step.setValue(10.0)
        self.spin_step.setSuffix(" µm")
        self.btn_minus = QPushButton("−")
        self.btn_minus.setFixedWidth(34)
        self.btn_minus.clicked.connect(lambda: self._jog(-1.0))
        self.btn_plus = QPushButton("+")
        self.btn_plus.setFixedWidth(34)
        self.btn_plus.clicked.connect(lambda: self._jog(+1.0))
        jog.addWidget(self.spin_step)
        jog.addWidget(self.btn_minus)
        jog.addWidget(self.btn_plus)
        grid.addWidget(QLabel("Jog"), 4, 0)
        grid.addLayout(jog, 4, 1)

        self.spin_velocity = QDoubleSpinBox()
        self.spin_velocity.setRange(0.001, 100.0)
        self.spin_velocity.setDecimals(3)
        self.spin_velocity.setValue(1.0)
        self.spin_velocity.setSuffix(" mm/s")
        self.spin_velocity.editingFinished.connect(self._apply_profile)
        grid.addWidget(QLabel("Velocity"), 5, 0)
        grid.addWidget(self.spin_velocity, 5, 1)

        self.spin_acceleration = QDoubleSpinBox()
        self.spin_acceleration.setRange(0.001, 1000.0)
        self.spin_acceleration.setDecimals(3)
        self.spin_acceleration.setValue(10.0)
        self.spin_acceleration.setSuffix(" mm/s²")
        self.spin_acceleration.editingFinished.connect(self._apply_profile)
        grid.addWidget(QLabel("Acceleration"), 6, 0)
        grid.addWidget(self.spin_acceleration, 6, 1)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setToolTip("Abort any motion and release the position hold")
        self.btn_stop.clicked.connect(self.stage.stop)
        grid.addWidget(self.btn_stop, 7, 0, 1, 2)
        return g

    def _calibrate(self) -> None:
        answer = QMessageBox.question(
            self, "Calibrate",
            "Calibration moves the positioner by up to several millimetres.\n"
            "Make sure it can move freely. Continue?")
        if answer == QMessageBox.StandardButton.Yes:
            self.ctl.run(self.stage.calibrate, "calibrated")

    def _go_to(self) -> None:
        target = self.spin_goto.value()
        self.ctl.run(lambda: (self.stage.move_to(target), self.stage.wait_for_stop()),
                     f"moved to {target:.5f} mm")

    def _jog(self, sign: float) -> None:
        delta_mm = sign * self.spin_step.value() / 1000.0
        self.ctl.run(lambda: (self.stage.move_by(delta_mm), self.stage.wait_for_stop()),
                     f"jogged {sign * self.spin_step.value():+.3f} µm")

    def _apply_profile(self) -> None:
        self._save_settings()
        if not self.stage.is_connected:
            self.stage.velocity_mm_s = self.spin_velocity.value()
            self.stage.acceleration_mm_s2 = self.spin_acceleration.value()
            return
        self.ctl.run(lambda: self.stage.set_profile(self.spin_velocity.value(),
                                                    self.spin_acceleration.value()))

    def use_current_position(self) -> float:
        """Latest polled position in mm (used by the scan panel's 'Use current')."""
        return self.latest_position

    # -- state ---------------------------------------------------------------
    def _on_reading(self, text: str) -> None:
        self.lbl_position.setText(text)
        try:
            self.latest_position = float(text.split()[0])
            self.sig_position.emit(self.latest_position)
        except (ValueError, IndexError):
            pass

    def _set_status(self, text: str) -> None:
        self.lbl_status.setText(text)
        self.sig_status.emit(text)

    def _refresh(self) -> None:
        conn = self.stage.is_connected
        busy = self.ctl.busy
        self.btn_connect.setText("Disconnect" if conn else "Connect")
        self.btn_connect.setEnabled(not busy)
        self.combo_locator.setEnabled(not conn and not busy)
        self.spin_channel.setEnabled(not conn and not busy)
        self.btn_scan_devices.setEnabled(not conn and not busy)
        for w in (self.btn_reference, self.btn_calibrate, self.btn_go, self.spin_goto,
                  self.spin_step, self.btn_minus, self.btn_plus,
                  self.spin_velocity, self.spin_acceleration):
            w.setEnabled(conn and not busy)
        self.btn_stop.setEnabled(conn)
        if not conn:
            self.lbl_position.setText("-- mm")
            self.latest_position = float("nan")

    def freeze(self, frozen: bool) -> None:
        """Pause position polling and lock manual control during a scan."""
        if frozen:
            self.timer.stop()
        else:
            self.timer.start(500)
        self.connection_group.setEnabled(not frozen)
        self.motion_group.setEnabled(not frozen)

    def shutdown(self) -> None:
        self.timer.stop()
        try:
            if self.stage.is_connected:
                self.stage.disconnect()
        except Exception:  # noqa: BLE001
            pass

    # -- persistence ---------------------------------------------------------
    def _restore_settings(self) -> None:
        s = self._settings
        loc = s.value("locator", None)
        if loc:
            self.combo_locator.setCurrentText(str(loc))
        for key, widget, cast in (("channel", self.spin_channel, int),
                                  ("velocity", self.spin_velocity, float),
                                  ("acceleration", self.spin_acceleration, float),
                                  ("step_um", self.spin_step, float)):
            val = s.value(key, None)
            if val is None:
                continue
            try:
                widget.setValue(cast(val))
            except (TypeError, ValueError):
                pass

    def _save_settings(self) -> None:
        s = self._settings
        s.setValue("locator", self.combo_locator.currentText())
        s.setValue("channel", self.spin_channel.value())
        s.setValue("velocity", self.spin_velocity.value())
        s.setValue("acceleration", self.spin_acceleration.value())
        s.setValue("step_um", self.spin_step.value())
