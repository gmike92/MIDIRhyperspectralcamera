"""SR865A lock-in control panel -- the acquisition app's signal source.

This panel replaces the camera panel of the imaging app: connection, the
demodulator settings that matter for a step scan (time constant, sensitivity,
filter slope, phase) and the choice of which channel the scan records.

The panel owns the `LockInSR860` driver and a `DeviceController`; the main
window reads `panel.lockin` for the scan and `panel.channel` for the parameter
to record. A monitor timer keeps a live reading flowing whenever the instrument
is connected and no scan is running -- that reading is what the main window's
live strip chart plots.
"""
from __future__ import annotations

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from instruments.lockin_sr860 import (
    FILTER_SLOPES, LockInSR860, TIME_CONSTANTS, VOLTAGE_SENSITIVITIES,
    channel_names, channel_unit, probe_visa,
)
from ui.device_controller import DeviceController

#: Interface picker: label -> (driver interface key, address placeholder).
INTERFACE_CHOICES = [
    ("LAN (VXI-11)", "vxi11", "192.168.1.10"),
    ("VISA (USB/GPIB)", "visa", "USB0::0xB506::0x2000::00xxxxx::INSTR"),
    ("LAN (raw TCP)", "tcp", "192.168.1.10:23"),
    ("Simulate (no hardware)", "sim", ""),
]

#: The example addresses, so switching interface can overwrite a leftover one.
_OTHER_HINTS = {hint for _label, _key, hint in INTERFACE_CHOICES if hint}


def format_value(value: float, unit: str = "") -> str:
    """Engineering-notation readout, e.g. 1.234 mV / -12.35 deg."""
    if value is None or value != value:          # None or NaN
        return "--"
    if unit in ("deg", "") or value == 0:
        return f"{value:.4f} {unit}".strip()
    mag = abs(value)
    for scale, prefix in ((1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"),
                          (1e-12, "p")):
        if mag >= scale or scale == 1e-12:
            return f"{value / scale:.4f} {prefix}{unit}"
    return f"{value:.4e} {unit}"


class LockInPanel(QWidget):
    """Connection + settings + live readout for the SR865A."""

    sig_status = QtCore.pyqtSignal(str)
    sig_reading = QtCore.pyqtSignal(float)      # latest monitor value

    def __init__(self) -> None:
        super().__init__()
        self.lockin = LockInSR860()
        self.ctl = DeviceController(self.lockin)
        self.latest_value = float("nan")
        self._monitoring = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.connection_group = self._build_connection_group()
        self.settings_group = self._build_settings_group()
        self.reading_group = self._build_reading_group()
        for g in (self.connection_group, self.settings_group, self.reading_group):
            layout.addWidget(g)
        layout.addStretch()

        self.ctl.status.connect(self._set_status)
        self.ctl.busy_changed.connect(lambda *_: self._refresh())

        # Live readout poll. Deliberately slow (5 Hz): every read is a round trip
        # over LAN/USB and the scan thread competes for the same link.
        self.monitor_timer = QtCore.QTimer(self)
        self.monitor_timer.timeout.connect(self._poll_reading)
        self.monitor_timer.start(200)

        self._settings = QtCore.QSettings("MIR_CAMERA", "LockInPanel")
        self._restore_settings()
        self._refresh()

    # -- connection ----------------------------------------------------------
    def _build_connection_group(self) -> QGroupBox:
        g = QGroupBox("Lock-in (SR865A)")
        grid = QGridLayout(g)

        self.combo_interface = QComboBox()
        for label, _key, _hint in INTERFACE_CHOICES:
            self.combo_interface.addItem(label)
        self.combo_interface.setCurrentIndex(0)
        self.combo_interface.currentIndexChanged.connect(self._on_interface_changed)
        grid.addWidget(QLabel("Interface"), 0, 0)
        grid.addWidget(self.combo_interface, 0, 1)

        self.edit_address = QLineEdit(INTERFACE_CHOICES[0][2])
        self.edit_address.setToolTip("IP address, or VISA resource string")
        grid.addWidget(QLabel("Address"), 1, 0)
        grid.addWidget(self.edit_address, 1, 1)

        row = QHBoxLayout()
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._toggle_connect)
        self.btn_find = QPushButton("Find VISA")
        self.btn_find.setToolTip("List VISA resources attached to this PC")
        self.btn_find.clicked.connect(self._find_visa)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_find)
        grid.addLayout(row, 2, 0, 1, 2)

        self.lbl_status = QLabel("offline")
        self.lbl_status.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        grid.addWidget(self.lbl_status, 3, 0, 1, 2)
        return g

    def _on_interface_changed(self, index: int) -> None:
        _label, key, hint = INTERFACE_CHOICES[index]
        self.edit_address.setPlaceholderText(hint)
        self.edit_address.setEnabled(key != "sim")
        current = self.edit_address.text().strip()
        if key == "sim":
            self.edit_address.clear()
        elif not current or current in _OTHER_HINTS:
            # Replace an address only when it is blank or is another interface's
            # example. Leaving a VISA resource string in the box after switching
            # to a LAN interface is what produces "getaddrinfo failed" -- Windows
            # tries to resolve it as a hostname. Anything typed is left alone.
            self.edit_address.setText(hint)

    def _find_visa(self) -> None:
        found, message = probe_visa()
        if not found:
            # `message` distinguishes pyvisa-missing from no-VISA-runtime from
            # runtime-present-but-nothing-attached; they need different fixes.
            self._set_status(message)
            return
        self._set_status("VISA: " + ", ".join(found))
        # Pick the first SRS-looking resource to save a paste.
        for res in found:
            if "0xB506" in res or "SR86" in res.upper():
                self.combo_interface.setCurrentIndex(1)
                self.edit_address.setText(res)
                return
        self.combo_interface.setCurrentIndex(1)
        self.edit_address.setText(found[0])

    def _toggle_connect(self) -> None:
        if self.lockin.is_connected:
            self.ctl.run(self.lockin.disconnect, "disconnected")
            return
        key = INTERFACE_CHOICES[self.combo_interface.currentIndex()][1]
        address = self.edit_address.text().strip()
        self._save_settings()

        def _connect():
            if not self.lockin.connect(key, address):
                raise RuntimeError(self.lockin.last_error or "connection refused")
            self.sig_status.emit(self.lockin.identity)
            self._read_settings_from_instrument()

        self.ctl.run(_connect)

    # -- settings ------------------------------------------------------------
    def _build_settings_group(self) -> QGroupBox:
        g = QGroupBox("Demodulator")
        grid = QGridLayout(g)

        self.combo_tc = QComboBox()
        for tc in TIME_CONSTANTS:
            self.combo_tc.addItem(_fmt_seconds(tc), tc)
        self.combo_tc.setCurrentIndex(TIME_CONSTANTS.index(0.1))
        self.combo_tc.activated.connect(self._apply_time_constant)
        grid.addWidget(QLabel("Time constant"), 0, 0)
        grid.addWidget(self.combo_tc, 0, 1)

        self.combo_slope = QComboBox()
        for slope in FILTER_SLOPES:
            self.combo_slope.addItem(f"{slope} dB/oct", slope)
        self.combo_slope.setCurrentIndex(FILTER_SLOPES.index(24))
        self.combo_slope.activated.connect(self._apply_filter_slope)
        grid.addWidget(QLabel("Filter slope"), 1, 0)
        grid.addWidget(self.combo_slope, 1, 1)

        self.combo_sens = QComboBox()
        for sens in VOLTAGE_SENSITIVITIES:
            self.combo_sens.addItem(format_value(sens, "V"), sens)
        self.combo_sens.activated.connect(self._apply_sensitivity)
        grid.addWidget(QLabel("Sensitivity"), 2, 0)
        grid.addWidget(self.combo_sens, 2, 1)

        self.spin_phase = QDoubleSpinBox()
        self.spin_phase.setRange(-360.0, 360.0)
        self.spin_phase.setDecimals(3)
        self.spin_phase.setSingleStep(1.0)
        self.spin_phase.setSuffix(" deg")
        self.spin_phase.editingFinished.connect(self._apply_phase)
        grid.addWidget(QLabel("Phase"), 3, 0)
        grid.addWidget(self.spin_phase, 3, 1)

        auto = QHBoxLayout()
        self.btn_auto_phase = QPushButton("Auto phase")
        self.btn_auto_phase.clicked.connect(
            lambda: self._instrument_action(self.lockin.auto_phase, "auto phase"))
        self.btn_auto_range = QPushButton("Auto range")
        self.btn_auto_range.clicked.connect(
            lambda: self._instrument_action(self.lockin.auto_range, "auto range"))
        self.btn_auto_scale = QPushButton("Auto scale")
        self.btn_auto_scale.clicked.connect(
            lambda: self._instrument_action(self.lockin.auto_scale, "auto scale"))
        auto.addWidget(self.btn_auto_phase)
        auto.addWidget(self.btn_auto_range)
        auto.addWidget(self.btn_auto_scale)
        grid.addLayout(auto, 4, 0, 1, 2)

        self.btn_read_back = QPushButton("Read settings from instrument")
        self.btn_read_back.clicked.connect(
            lambda: self.ctl.run(self._read_settings_from_instrument))
        grid.addWidget(self.btn_read_back, 5, 0, 1, 2)
        return g

    def _instrument_action(self, fn, label: str) -> None:
        """Run a one-shot instrument command, then refresh the local settings.

        Auto phase/range/scale all change the instrument state, so the panel
        re-reads it afterwards instead of showing stale values.
        """
        def _work():
            fn()
            self._read_settings_from_instrument()
        self.ctl.run(_work, f"{label} done")

    def _read_settings_from_instrument(self) -> None:
        """Pull the live settings into the combo boxes (worker thread safe:
        the widget writes go through a queued signal)."""
        if not self.lockin.is_connected:
            return
        snap = {}
        for key, fn in (("tc", lambda: self.lockin.time_constant),
                        ("slope", lambda: self.lockin.filter_slope),
                        ("sens", lambda: self.lockin.sensitivity),
                        ("phase", lambda: self.lockin.phase),
                        ("freq", lambda: self.lockin.frequency)):
            try:
                snap[key] = fn()
            except Exception:  # noqa: BLE001
                snap[key] = None
        QtCore.QMetaObject.invokeMethod(
            self, "_apply_settings_snapshot", QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, snap))

    @QtCore.pyqtSlot(object)
    def _apply_settings_snapshot(self, snap: dict) -> None:
        if snap.get("tc") is not None and snap["tc"] in TIME_CONSTANTS:
            self.combo_tc.setCurrentIndex(TIME_CONSTANTS.index(snap["tc"]))
        if snap.get("slope") is not None and snap["slope"] in FILTER_SLOPES:
            self.combo_slope.setCurrentIndex(FILTER_SLOPES.index(snap["slope"]))
        if snap.get("sens") is not None and snap["sens"] in VOLTAGE_SENSITIVITIES:
            self.combo_sens.setCurrentIndex(VOLTAGE_SENSITIVITIES.index(snap["sens"]))
        if snap.get("phase") is not None:
            self.spin_phase.blockSignals(True)
            self.spin_phase.setValue(float(snap["phase"]))
            self.spin_phase.blockSignals(False)
        if snap.get("freq") is not None:
            self.lbl_frequency.setText(f"{snap['freq']:.6g} Hz")

    def _apply_time_constant(self, *_a) -> None:
        tc = self.combo_tc.currentData()
        self._write_setting(lambda: setattr(self.lockin, "time_constant", tc),
                            f"time constant {_fmt_seconds(tc)}")

    def _apply_filter_slope(self, *_a) -> None:
        slope = self.combo_slope.currentData()
        self._write_setting(lambda: setattr(self.lockin, "filter_slope", slope),
                            f"filter {slope} dB/oct")

    def _apply_sensitivity(self, *_a) -> None:
        sens = self.combo_sens.currentData()
        self._write_setting(lambda: setattr(self.lockin, "sensitivity", sens),
                            f"sensitivity {format_value(sens, 'V')}")

    def _apply_phase(self) -> None:
        deg = self.spin_phase.value()
        self._write_setting(lambda: setattr(self.lockin, "phase", deg),
                            f"phase {deg:.3f} deg")

    def _write_setting(self, fn, label: str) -> None:
        self._save_settings()
        if not self.lockin.is_connected:
            return          # offline: the combo just remembers the choice
        self.ctl.run(fn, label)

    # -- live reading --------------------------------------------------------
    def _build_reading_group(self) -> QGroupBox:
        g = QGroupBox("Reading")
        grid = QGridLayout(g)

        self.combo_channel = QComboBox()
        for name in channel_names():
            self.combo_channel.addItem(name)
        self.combo_channel.setCurrentText("R")
        self.combo_channel.currentTextChanged.connect(self._on_channel_changed)
        self.combo_channel.setToolTip(
            "X/Y/R/Theta and the Aux inputs are read with OUTP?; Data 1-4 are the "
            "front-panel display slots, read with OUTR?.")
        grid.addWidget(QLabel("Channel"), 0, 0)
        grid.addWidget(self.combo_channel, 0, 1)

        self.lbl_value = QLabel("--")
        self.lbl_value.setStyleSheet("font-size:20px; font-weight:600;")
        grid.addWidget(self.lbl_value, 1, 0, 1, 2)

        self.lbl_frequency = QLabel("-- Hz")
        grid.addWidget(QLabel("Ref. frequency"), 2, 0)
        grid.addWidget(self.lbl_frequency, 2, 1)

        self.lbl_overload = QLabel("--")
        grid.addWidget(QLabel("Input level"), 3, 0)
        grid.addWidget(self.lbl_overload, 3, 1)

        self.chk_monitor = QCheckBox("Live monitor")
        self.chk_monitor.setChecked(True)
        self.chk_monitor.setToolTip(
            "Poll the selected channel at 5 Hz and plot it against time. "
            "Automatically paused while a scan is running.")
        self.chk_monitor.toggled.connect(self._on_monitor_toggled)
        grid.addWidget(self.chk_monitor, 4, 0, 1, 2)
        return g

    def _on_channel_changed(self, _name: str) -> None:
        self._save_settings()
        self.sig_status.emit(f"recording channel: {self.channel}")

    def _on_monitor_toggled(self, on: bool) -> None:
        self._monitoring = on
        if not on:
            self.lbl_value.setText("--")

    @property
    def channel(self) -> str:
        return self.combo_channel.currentText()

    @property
    def channel_unit(self) -> str:
        return channel_unit(self.channel)

    def set_monitor_enabled(self, on: bool) -> None:
        """Main window pauses monitoring for the duration of a scan, so the scan
        thread has the instrument link to itself."""
        self._monitoring = bool(on) and self.chk_monitor.isChecked()

    def _poll_reading(self) -> None:
        if not (self._monitoring and self.lockin.is_connected) or self.ctl.busy:
            return
        try:
            value = self.lockin.read_channel(self.channel)
        except Exception as e:  # noqa: BLE001
            self._set_status(f"read error: {e}")
            return
        self.latest_value = value
        self.lbl_value.setText(format_value(value, self.channel_unit))
        self.sig_reading.emit(value)
        # The input-level indicator is cheap and catches the classic "my signal
        # is flat" cause: the front end is overloaded or the range is too small.
        try:
            level = self.lockin.signal_strength
            self.lbl_overload.setText(
                ["low", "ok", "ok", "high", "OVERLOAD"][min(4, max(0, level))])
            self.lbl_overload.setStyleSheet(
                "color:#c92a2a; font-weight:600;" if level >= 4 else "")
        except Exception:  # noqa: BLE001
            pass

    # -- state ---------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        self.lbl_status.setText(text)
        self.sig_status.emit(text)

    def _refresh(self) -> None:
        conn = self.lockin.is_connected
        busy = self.ctl.busy
        self.btn_connect.setText("Disconnect" if conn else "Connect")
        self.btn_connect.setEnabled(not busy)
        self.combo_interface.setEnabled(not conn and not busy)
        self.edit_address.setEnabled(
            not conn and not busy
            and INTERFACE_CHOICES[self.combo_interface.currentIndex()][1] != "sim")
        self.btn_find.setEnabled(not conn and not busy)
        for w in (self.btn_auto_phase, self.btn_auto_range, self.btn_auto_scale,
                  self.btn_read_back):
            w.setEnabled(conn and not busy)
        if not conn:
            self.lbl_value.setText("--")
            self.lbl_frequency.setText("-- Hz")
            self.lbl_overload.setText("--")

    def freeze(self, frozen: bool) -> None:
        """Lock the panel while a scan drives the instrument from its own thread."""
        self.connection_group.setEnabled(not frozen)
        self.settings_group.setEnabled(not frozen)
        self.combo_channel.setEnabled(not frozen)
        self.set_monitor_enabled(not frozen)

    def shutdown(self) -> None:
        self.monitor_timer.stop()
        try:
            if self.lockin.is_connected:
                self.lockin.disconnect()
        except Exception:  # noqa: BLE001
            pass

    # -- persistence ---------------------------------------------------------
    def _restore_settings(self) -> None:
        s = self._settings
        idx = s.value("interface", None)
        if idx is not None:
            try:
                self.combo_interface.setCurrentIndex(int(idx))
            except (TypeError, ValueError):
                pass
        addr = s.value("address", None)
        if addr:
            self.edit_address.setText(str(addr))
        chan = s.value("channel", None)
        if chan and self.combo_channel.findText(str(chan)) >= 0:
            self.combo_channel.setCurrentText(str(chan))
        for key, combo, table in (("tc", self.combo_tc, TIME_CONSTANTS),
                                  ("slope", self.combo_slope, FILTER_SLOPES),
                                  ("sens", self.combo_sens, VOLTAGE_SENSITIVITIES)):
            val = s.value(key, None)
            if val is None:
                continue
            try:
                target = type(table[0])(val)
                if target in table:
                    combo.setCurrentIndex(table.index(target))
            except (TypeError, ValueError):
                pass
        self._on_interface_changed(self.combo_interface.currentIndex())

    def _save_settings(self) -> None:
        s = self._settings
        s.setValue("interface", self.combo_interface.currentIndex())
        s.setValue("address", self.edit_address.text())
        s.setValue("channel", self.combo_channel.currentText())
        s.setValue("tc", self.combo_tc.currentData())
        s.setValue("slope", self.combo_slope.currentData())
        s.setValue("sens", self.combo_sens.currentData())


def _fmt_seconds(seconds: float) -> str:
    """1e-3 -> '1 ms', 30000 -> '30 ks' -- for the time-constant picker."""
    for scale, prefix in ((1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ")):
        if seconds >= scale:
            return f"{seconds / scale:g} {prefix}s"
    return f"{seconds:g} s"
