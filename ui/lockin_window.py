"""Main window of the lock-in acquisition app.

Same shape as the camera app's `MainWindow` -- instrument tabs on the left, the
live view filling the rest -- but the live view is a TRACE, not an image: the
lock-in reading plotted against the commanded stage position, growing point by
point as the scan runs.

Left column
    Lock-in | Stage | Scan          (LockInPanel / MCS2Panel / LockInScanPanel)

Right column
    Scan trace   -- value vs stage position, live during a scan, with an
                    optional ghost of the previous scan for comparison.
    Live monitor -- value vs time, running whenever the lock-in is connected
                    and no scan is in progress. This is the strip chart that
                    stands in for the camera's live view when lining the
                    experiment up by hand.
"""
from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from ui.lockin_panel import LockInPanel, format_value
from ui.lockin_scan_panel import DEFAULT_SAVE_DIR, LockInScanPanel
from ui.mcs2_panel import MCS2Panel

MONITOR_HISTORY = 600          # samples kept in the live strip chart (~2 min at 5 Hz)
#: The scan trace is plotted in MICROMETRES: a typical scan spans tens of um,
#: which in mm is an axis full of 0.0500-style numbers. Positions travel in mm
#: everywhere else (drivers, scan panel, saved files) and are converted only
#: here, on their way onto the plot.
UM_PER_MM = 1000.0


class LockInMainWindow(QMainWindow):
    """MCS2 stage + SR865A lock-in step-scan acquisition."""

    def __init__(self, save_dir: str = DEFAULT_SAVE_DIR,
                 simulate: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("MCS2 + SR865A Lock-in Acquisition")
        self.resize(1280, 820)

        self.save_dir = save_dir
        self._monitor_t = deque(maxlen=MONITOR_HISTORY)
        self._monitor_v = deque(maxlen=MONITOR_HISTORY)
        self._monitor_t0 = time.time()
        self._previous_scan = None      # (positions, values) of the last finished scan

        self.lockin_panel = LockInPanel()
        self.stage_panel = MCS2Panel()
        self.scan_panel = LockInScanPanel(self.stage_panel, self.lockin_panel,
                                          save_dir=save_dir)

        self._build_ui()
        self._connect_signals()

        if simulate:
            # --simulate wires both panels to their fake backends so the whole
            # scan path can be exercised with nothing plugged in.
            self.lockin_panel.combo_interface.setCurrentIndex(
                self.lockin_panel.combo_interface.count() - 1)   # "Simulate"
            self.stage_panel.combo_locator.setCurrentIndex(0)  # "Simulate"

        # Repaint the scan trace while a scan runs. The scan thread emits one
        # progress signal per point, but at 100+ points/s redrawing on every one
        # would starve the GUI, so the curve is refreshed on a timer instead.
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._refresh_scan_curve)
        self.plot_timer.start(100)

    # -- layout --------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        tabs = QTabWidget()
        tabs.setMinimumWidth(340)
        tabs.setMaximumWidth(380)
        tabs.addTab(self._make_tab(self.lockin_panel), "Lock-in")
        tabs.addTab(self._make_tab(self.stage_panel), "Stage")
        tabs.addTab(self._make_tab(self.scan_panel), "Scan")
        tabs.setCurrentIndex(2)
        root.addWidget(tabs, 0)

        right = QVBoxLayout()
        root.addLayout(right, 1)

        title = QLabel("Lock-in Trace")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        right.addWidget(title)

        self.status_label = QLabel("Connect the lock-in and the stage, then scan.")
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)

        # Big live readout: what the lock-in reads right now, and where the
        # stage is. Deliberately large -- it is read from across the lab bench.
        readout = QHBoxLayout()
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("font-size: 30px; font-weight: 600;")
        self.position_label = QLabel("-- mm")
        self.position_label.setStyleSheet(
            "font-size: 22px; font-weight: 600; color:#1c7ed6;")
        readout.addWidget(self.value_label)
        readout.addStretch()
        readout.addWidget(self.position_label)
        right.addLayout(readout)

        right.addLayout(self._build_plot_controls())

        self.scan_plot = pg.PlotWidget(title="Scan trace")
        self._style_plot(self.scan_plot)
        self.scan_plot.setLabel("bottom", "Stage position", units="µm")
        self.scan_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ghost_curve = self.scan_plot.plot(
            pen=pg.mkPen("#adb5bd", width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.scan_curve = self.scan_plot.plot(pen=pg.mkPen("#1c7ed6", width=2))
        # Marker on the point being measured, so it is obvious where the scan is.
        self.scan_marker = pg.ScatterPlotItem(size=9, brush=pg.mkBrush("#e8590c"),
                                              pen=None)
        self.scan_plot.addItem(self.scan_marker)
        right.addWidget(self.scan_plot, 3)

        self.monitor_plot = pg.PlotWidget(title="Live monitor")
        self._style_plot(self.monitor_plot)
        self.monitor_plot.setLabel("bottom", "Time", units="s")
        self.monitor_plot.showGrid(x=True, y=True, alpha=0.2)
        self.monitor_curve = self.monitor_plot.plot(pen=pg.mkPen("#37b24d", width=1))
        self.monitor_plot.setMinimumHeight(150)
        self.monitor_plot.setMaximumHeight(220)
        right.addWidget(self.monitor_plot, 1)

        self._update_axis_labels()
        self.statusBar().showMessage("ready")

    def _build_plot_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.chk_ghost = QCheckBox("Keep previous scan")
        self.chk_ghost.setChecked(True)
        self.chk_ghost.setToolTip(
            "Draw the last finished scan behind the current one, so a change in "
            "alignment shows up immediately")
        self.chk_ghost.toggled.connect(self._refresh_ghost_curve)
        row.addWidget(self.chk_ghost)

        self.chk_autoscale = QCheckBox("Auto-scale Y")
        self.chk_autoscale.setChecked(True)
        self.chk_autoscale.toggled.connect(
            lambda on: self.scan_plot.enableAutoRange(axis="y", enable=on))
        row.addWidget(self.chk_autoscale)

        row.addSpacing(16)
        row.addWidget(QLabel("Monitor history"))
        self.spin_history = QSpinBox()
        self.spin_history.setRange(20, 20000)
        self.spin_history.setValue(MONITOR_HISTORY)
        self.spin_history.setSuffix(" pts")
        self.spin_history.valueChanged.connect(self._resize_monitor_history)
        row.addWidget(self.spin_history)

        self.btn_clear_monitor = QPushButton("Clear")
        self.btn_clear_monitor.clicked.connect(self._clear_monitor)
        row.addWidget(self.btn_clear_monitor)

        self.btn_export = QPushButton("Export plot")
        self.btn_export.setToolTip("Save the scan trace as a PNG next to the data")
        self.btn_export.clicked.connect(self._export_plot)
        row.addWidget(self.btn_export)

        row.addStretch()
        return row

    def _make_tab(self, widget: QWidget) -> QScrollArea:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(widget)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(container)
        return area

    def _style_plot(self, plot: pg.PlotWidget) -> None:
        plot.setBackground("w")
        axis_pen = pg.mkPen("#666666", width=1)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen("#333333")
            axis.setStyle(tickLength=6)

    # -- signals -------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.lockin_panel.sig_status.connect(self._set_status)
        self.lockin_panel.sig_reading.connect(self._on_reading)
        self.lockin_panel.combo_channel.currentTextChanged.connect(
            self._on_channel_changed)
        self.stage_panel.sig_status.connect(self._set_status)
        self.stage_panel.sig_position.connect(self._on_position)
        self.scan_panel.sig_status.connect(self._set_status)
        self.scan_panel.sig_running.connect(self._on_scan_running)
        self.scan_panel.sig_scan_done.connect(self._on_scan_done)
        self.scan_panel.sig_progress.connect(self._on_scan_progress)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.statusBar().showMessage(text, 6000)

    def _on_channel_changed(self, _name: str) -> None:
        self._update_axis_labels()
        self._clear_monitor()

    def _update_axis_labels(self) -> None:
        channel = self.lockin_panel.channel
        unit = self.lockin_panel.channel_unit
        for plot in (self.scan_plot, self.monitor_plot):
            plot.setLabel("left", channel, units=unit or None)

    # -- live monitor --------------------------------------------------------
    def _on_reading(self, value: float) -> None:
        self.value_label.setText(
            format_value(value, self.lockin_panel.channel_unit))
        self._monitor_t.append(time.time() - self._monitor_t0)
        self._monitor_v.append(value)
        self.monitor_curve.setData(np.fromiter(self._monitor_t, dtype=float),
                                   np.fromiter(self._monitor_v, dtype=float))

    def _on_position(self, position_mm: float) -> None:
        self.position_label.setText(f"{position_mm:.5f} mm")

    def _clear_monitor(self) -> None:
        self._monitor_t.clear()
        self._monitor_v.clear()
        self._monitor_t0 = time.time()
        self.monitor_curve.setData([], [])

    def _resize_monitor_history(self, n: int) -> None:
        self._monitor_t = deque(self._monitor_t, maxlen=n)
        self._monitor_v = deque(self._monitor_v, maxlen=n)

    # -- scan trace ----------------------------------------------------------
    def _on_scan_running(self, running: bool) -> None:
        # Hand both instruments over to the scan thread for the duration.
        self.lockin_panel.freeze(running)
        self.stage_panel.freeze(running)
        if running:
            if self.chk_ghost.isChecked():
                self._previous_scan = self._current_curve_data()
            self._refresh_ghost_curve()
            self.scan_curve.setData([], [])
            self.scan_marker.setData([], [])
            self._set_status("scanning...")

    def _current_curve_data(self):
        x, y = self.scan_curve.getData()
        if x is None or len(x) == 0:
            return None
        return np.array(x, copy=True), np.array(y, copy=True)

    def _refresh_ghost_curve(self, *_a) -> None:
        if self.chk_ghost.isChecked() and self._previous_scan is not None:
            self.ghost_curve.setData(*self._previous_scan)
        else:
            self.ghost_curve.setData([], [])

    def _on_scan_progress(self, i: int, n: int, position: float, value: float) -> None:
        # Cheap per-point work only: the marker. The curve itself is redrawn by
        # the plot timer so a fast scan cannot flood the event loop.
        self.scan_marker.setData([position * UM_PER_MM], [value])
        self.position_label.setText(f"{position:.5f} mm")
        self.value_label.setText(
            format_value(value, self.lockin_panel.channel_unit))
        self.statusBar().showMessage(f"point {i}/{n}", 2000)

    def _refresh_scan_curve(self) -> None:
        if not self.scan_panel.running:
            return
        positions, values = self.scan_panel.current_trace()
        if positions is None or len(positions) == 0:
            return
        self.scan_curve.setData(np.asarray(positions) * UM_PER_MM,
                                np.asarray(values))

    def _on_scan_done(self, positions, values) -> None:
        if positions is None or len(positions) == 0:
            return
        self.scan_curve.setData(np.asarray(positions) * UM_PER_MM,
                                np.asarray(values))
        self.scan_marker.setData([], [])
        values = np.asarray(values, dtype=float)
        if np.all(np.isnan(values)):
            self._set_status(f"{len(positions)} points (no valid readings)")
            return
        peak = int(np.nanargmax(np.abs(values)))
        self._set_status(
            f"{len(positions)} points | extremum "
            f"{format_value(float(values[peak]), self.lockin_panel.channel_unit)} "
            f"at {positions[peak] * UM_PER_MM:.3f} µm")

    def _export_plot(self) -> None:
        save_dir = self.scan_panel.edit_save_dir.text().strip() or self.save_dir
        name = self.scan_panel.edit_filename.text().strip() or "lockin_scan"
        try:
            os.makedirs(save_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(save_dir, f"{stamp}.{name}.png")
            self.scan_plot.grab().save(path)
        except Exception as e:  # noqa: BLE001
            self._set_status(f"export failed: {e}")
            return
        self._set_status(f"saved {os.path.basename(path)}")

    # -- shutdown ------------------------------------------------------------
    def closeEvent(self, event) -> None:
        self.plot_timer.stop()
        self.scan_panel.abort()
        self.lockin_panel.shutdown()
        self.stage_panel.shutdown()
        super().closeEvent(event)
