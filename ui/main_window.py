from __future__ import annotations

import os
import queue
import time
from datetime import datetime
from multiprocessing import shared_memory

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QRect, QSize
from PyQt6.QtGui import QPainter, QPdfWriter
from PyQt6.QtSvg import QSvgGenerator
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QScrollArea,
    QSlider,
    QTabWidget,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.stages import StagesPanel
from ui.twins_scan import TwinsScanPanel
from ui.measure_kspace import MeasurePanel


class MainWindow(QMainWindow):
    def __init__(
        self,
        frame_queue,
        control_queue,
        initial_mode: str = "irc806",
        shared_frame_name: str = "",
        shared_frame_shape: tuple[int, int] = (0, 0),
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.control_queue = control_queue
        self.current_mode = initial_mode
        self.shared_frame_shape = tuple(shared_frame_shape)
        self.shared_frame_handle = None
        self.shared_frame = None
        if shared_frame_name and len(self.shared_frame_shape) == 2:
            self.shared_frame_handle = shared_memory.SharedMemory(name=shared_frame_name)
            self.shared_frame = np.ndarray(self.shared_frame_shape, dtype=np.uint16, buffer=self.shared_frame_handle.buf)

        self.setWindowTitle("IRC806 MWIR Live Viewer")
        self.resize(1280, 820)

        self.background_frame = None
        self.use_bg_subtraction = False
        self._bg_capture_remaining = 0   # frames left to average into background
        self._bg_accum = None            # accumulator during capture
        self.bg_average_frames = 16      # frames averaged for a background
        self.latest_frame = None
        # Bad-pixel (cold/hot) mask: masked pixels are replaced by their nearest
        # good neighbour BEFORE latest_frame is set, so they are neither plotted
        # nor saved (the scan reads latest_frame). Precomputed index maps make the
        # per-frame fix O(n_bad).
        self.bad_pixel_mask = None          # bool (H, W)
        self.badpix_enabled = False
        self._badpix_bad = None             # (yy, xx) of masked pixels
        self._badpix_src = None             # (yy, xx) of their nearest good pixel
        self.mask_paint_mode = "none"       # "none" | "add" | "erase"
        self.latest_status = {}        # most recent camera status (for metadata)
        self.last_display_frame = None
        self._display_levels = (0.0, 16383.0)
        self.profile_pixel = None  # (row, col) for X/Y profiles; None = center
        self.save_dir = r"D:\CAMERA"
        self.save_filename = "filename"
        self.auto_scale_display = False
        self.frame_count = 0
        self.analysis_frame_counter = 0
        self.last_fps_update = time.time()

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_from_worker)
        self.timer.start(8)   # ~120 Hz display poll to match the camera rate

    def _build_ui(self) -> None:
        pg.setConfigOptions(imageAxisOrder="row-major")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.stages_panel = StagesPanel()

        controls_tabs = QTabWidget()
        controls_tabs.setMinimumWidth(320)
        controls_tabs.setMaximumWidth(340)
        controls_tabs.addTab(self._make_tab([
            self._build_camera_group(),
            self._build_badpixel_group(),
            self._build_processing_group(),
            self._build_correction_group(),
            self._build_save_group(),
            self._build_measurements_group(),
        ]), "Camera")
        controls_tabs.addTab(
            self._make_tab([self.stages_panel.delay_group]), "Thorlabs Stage")
        controls_tabs.addTab(
            self._make_tab([self.stages_panel.rotator_group]), "Rotator")
        self.twins_scan_panel = TwinsScanPanel(
            self.stages_panel.twins_ctl,
            frame_source=lambda: self.latest_frame,
            roi_provider=self.get_roi_bounds,
            save_dir=os.path.join(self.save_dir, "twins"))
        controls_tabs.addTab(
            self._make_tab([self.stages_panel.twins_group, self.twins_scan_panel]),
            "TWINS")
        self.measure_panel = MeasurePanel(
            self.stages_panel,
            frame_source=lambda: self.latest_frame,
            roi_provider=self.get_roi_bounds,
            roi_show=self.set_roi_visible,
            bg_provider=lambda: (self.background_frame, self.use_bg_subtraction),
            # Save K-space files into the SAME folder set in the camera Save group.
            save_dir_provider=lambda: self.save_dir_edit.text().strip() or self.save_dir,
            meta_provider=self._kspace_metadata,
            save_dir=self.save_dir)
        controls_tabs.addTab(self._make_tab([self.measure_panel]), "Measure")
        root.addWidget(controls_tabs, 0)

        viewer_column = QVBoxLayout()
        root.addLayout(viewer_column, 1)

        title = QLabel("Beam Viewer")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        viewer_column.addWidget(title)

        self.status_label = QLabel("Waiting for camera worker...")
        self.status_label.setWordWrap(True)
        viewer_column.addWidget(self.status_label)

        # ROI selector lives on the live view: tick to show the cyan box, drag
        # it over the region of interest; measurements (TWINS / hyperspectral)
        # read and save these bounds.
        roi_row = QHBoxLayout()
        self.roi_checkbox = QCheckBox("Show measurement ROI (drag the box on the image)")
        self.roi_checkbox.toggled.connect(self.set_roi_visible)
        self.roi_bounds_label = QLabel("ROI: full frame")
        self.roi_bounds_label.setStyleSheet("color:#888;")
        roi_row.addWidget(self.roi_checkbox)
        roi_row.addWidget(self.roi_bounds_label)
        roi_row.addStretch()
        viewer_column.addLayout(roi_row)

        self.export_panel = QWidget()
        export_layout = QVBoxLayout(self.export_panel)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(10)
        viewer_column.addWidget(self.export_panel, 1)

        # Temperature readout — sits just above the intensity (color) bar,
        # right-aligned. Updates every 5 min, or on demand via the Poll button.
        temp_row = QHBoxLayout()
        temp_row.addStretch()
        self.temp_value_label = QLabel("FPGA/Board: -- °C   FPA: -- K")
        self.temp_value_label.setStyleSheet("font-weight: 600; font-size: 13px;")
        temp_row.addWidget(self.temp_value_label)
        self.btn_poll_temp = QPushButton("Poll")
        self.btn_poll_temp.setFixedWidth(50)
        self.btn_poll_temp.setToolTip("Read the camera temperature now")
        self.btn_poll_temp.clicked.connect(
            lambda: self.control_queue.put({"type": "read_temp"}))
        temp_row.addWidget(self.btn_poll_temp)
        temp_row.addSpacing(26)   # nudge over the colorbar
        export_layout.addLayout(temp_row, 0)

        self.viewer_widget = pg.GraphicsLayoutWidget()
        export_layout.addWidget(self.viewer_widget, 5)

        self.image_plot = self.viewer_widget.addPlot(row=0, col=0)
        self.image_plot.setAspectLocked(True)
        # Images are row-major with row 0 at the TOP; pyqtgraph's y-axis points
        # up by default (row 0 at bottom = upside-down). Invert Y so the display
        # matches sensor orientation (watermark upright, clicks map correctly).
        self.image_plot.invertY(True)
        self.image_plot.hideAxis("left")
        self.image_plot.hideAxis("bottom")
        self.image_plot.setMenuEnabled(False)
        self.image_item = pg.ImageItem()
        self.image_item.setAutoDownsample(True)
        self.image_plot.addItem(self.image_item)

        # Crosshair marking the pixel whose row/column drive the profile plots.
        cross_pen = pg.mkPen("#39ff14", width=1)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self.image_plot.addItem(self.v_line, ignoreBounds=True)
        self.image_plot.addItem(self.h_line, ignoreBounds=True)

        # Draggable rectangular ROI for measurements (TWINS scan / hyperspectral).
        # Hidden until "Show ROI" is enabled; bounds read via get_roi_bounds().
        self.roi = pg.RectROI([260, 200], [120, 120],
                              pen=pg.mkPen("c", width=2))
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.image_plot.addItem(self.roi)
        self.roi.setVisible(False)
        self.roi.sigRegionChanged.connect(self._update_roi_label)

        # Click the image to choose the profile pixel.
        self.viewer_widget.scene().sigMouseClicked.connect(self.on_image_clicked)

        self.color_map = self._make_color_map("inferno")
        self.color_bar = pg.ColorBarItem(
            values=(0.0, 16383.0),  # 14-bit default; overridden by display range
            colorMap=self.color_map,
            interactive=False,
            width=18,
        )
        self.color_bar.setLabel("right", "Counts")
        self.color_bar.setImageItem(self.image_item, insert_in=self.image_plot)

        plots = QHBoxLayout()
        export_layout.addLayout(plots, 1)

        self.x_profile_plot = pg.PlotWidget(title="Horizontal Profile")
        self._style_profile_plot(self.x_profile_plot)
        self.x_profile_curve = self.x_profile_plot.plot(pen=pg.mkPen("#d9485f", width=2))
        self.x_profile_plot.setMinimumHeight(210)
        self.x_profile_plot.setMaximumHeight(230)
        plots.addWidget(self.x_profile_plot, 1)

        self.y_profile_plot = pg.PlotWidget(title="Vertical Profile")
        self._style_profile_plot(self.y_profile_plot)
        self.y_profile_curve = self.y_profile_plot.plot(pen=pg.mkPen("#1c7ed6", width=2))
        self.y_profile_plot.setMinimumHeight(210)
        self.y_profile_plot.setMaximumHeight(230)
        plots.addWidget(self.y_profile_plot, 1)

        self.fps_label = QLabel("FPS: 0.0")
        self.statusBar().addPermanentWidget(self.fps_label)
        self.apply_color_map(self.color_map_combo.currentText())

    def _make_tab(self, groups: list[QWidget]) -> QScrollArea:
        """Stack the given group widgets vertically inside a scrollable tab."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        for widget in groups:
            layout.addWidget(widget)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(container)
        return scroll

    def _is_goldeye_mode(self) -> bool:
        """The SP1203/Goldeye-specific controls (option dropdowns, 1 s log
        integration slider) apply only to these modes; the IRC806 keeps its
        original UI."""
        return (self.current_mode or "").lower() in (
            "ophir", "sp1203", "goldeye", "gige", "beamgage")

    def _build_badpixel_group(self) -> QGroupBox:
        group = QGroupBox("Bad pixels (cold/hot)")
        v = QVBoxLayout(group)
        self.badpix_check = QCheckBox("Apply mask (masked pixels not plotted/saved)")
        self.badpix_check.toggled.connect(self._on_badpix_enabled)
        v.addWidget(self.badpix_check)
        self.badpix_count_label = QLabel("Masked: 0 px")
        v.addWidget(self.badpix_count_label)

        det_row = QHBoxLayout()
        self.badpix_detect_btn = QPushButton("Auto-detect hot/cold")
        self.badpix_detect_btn.setToolTip(
            "Flag pixels that deviate from a 3x3 median by more than N-sigma "
            "(robust MAD) -- catches stuck-hot and dead-cold pixels -- and add "
            "them to the mask.")
        self.badpix_detect_btn.clicked.connect(self._detect_bad_pixels)
        det_row.addWidget(self.badpix_detect_btn)
        self.badpix_sigma_spin = QDoubleSpinBox()
        self.badpix_sigma_spin.setRange(2.0, 30.0)
        self.badpix_sigma_spin.setSingleStep(0.5)
        self.badpix_sigma_spin.setValue(6.0)
        self.badpix_sigma_spin.setPrefix("σ ")
        det_row.addWidget(self.badpix_sigma_spin)
        v.addLayout(det_row)

        paint_row = QHBoxLayout()
        self.badpix_paint_btn = QPushButton("Paint")
        self.badpix_paint_btn.setCheckable(True)
        self.badpix_paint_btn.setToolTip("Click pixels on the image to MASK them.")
        self.badpix_erase_btn = QPushButton("Erase")
        self.badpix_erase_btn.setCheckable(True)
        self.badpix_erase_btn.setToolTip("Click pixels on the image to UNMASK them.")
        self.badpix_paint_btn.clicked.connect(
            lambda on: self._set_paint_mode("add" if on else "none"))
        self.badpix_erase_btn.clicked.connect(
            lambda on: self._set_paint_mode("erase" if on else "none"))
        paint_row.addWidget(self.badpix_paint_btn)
        paint_row.addWidget(self.badpix_erase_btn)
        paint_row.addWidget(QLabel("Brush (px)"))
        self.badpix_brush_spin = QSpinBox()
        self.badpix_brush_spin.setRange(0, 25)
        self.badpix_brush_spin.setValue(0)
        paint_row.addWidget(self.badpix_brush_spin)
        v.addLayout(paint_row)

        file_row = QHBoxLayout()
        for label, slot in (("Clear", self._clear_badpix),
                            ("Save", self._save_badpix), ("Load", self._load_badpix)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            file_row.addWidget(b)
        v.addLayout(file_row)
        return group

    # -- bad-pixel mask logic ------------------------------------------------
    def _badpix_ensure_shape(self, shape) -> None:
        if self.bad_pixel_mask is None or self.bad_pixel_mask.shape != shape:
            self.bad_pixel_mask = np.zeros(shape, dtype=bool)

    def _rebuild_badpix_map(self) -> None:
        """Precompute, for each masked pixel, the nearest GOOD pixel to copy from."""
        m = self.bad_pixel_mask
        if m is None or not m.any() or m.all():
            self._badpix_bad = self._badpix_src = None
        else:
            from scipy.ndimage import distance_transform_edt
            inds = distance_transform_edt(m, return_distances=False, return_indices=True)
            bad = np.where(m)
            self._badpix_bad = bad
            self._badpix_src = (inds[0][bad], inds[1][bad])
        if hasattr(self, "badpix_count_label"):
            n = int(m.sum()) if m is not None else 0
            self.badpix_count_label.setText(f"Masked: {n} px")

    def _correct_bad_pixels(self, frame: np.ndarray) -> np.ndarray:
        if (not self.badpix_enabled or self._badpix_bad is None
                or self.bad_pixel_mask is None
                or self.bad_pixel_mask.shape != frame.shape):
            return frame
        out = frame.copy()
        out[self._badpix_bad] = frame[self._badpix_src]
        return out

    def _on_badpix_enabled(self, on: bool) -> None:
        self.badpix_enabled = bool(on)
        self.status_label.setText(
            f"bad-pixel mask {'ON' if on else 'off'}")

    def _detect_bad_pixels(self) -> None:
        f = self.latest_frame
        if f is None:
            self.status_label.setText("no frame to detect bad pixels")
            return
        from scipy.ndimage import median_filter
        ff = np.asarray(f, dtype=np.float32)
        diff = ff - median_filter(ff, size=3)
        mad = float(np.median(np.abs(diff - np.median(diff)))) + 1e-6
        thr = float(self.badpix_sigma_spin.value()) * 1.4826 * mad
        new = np.abs(diff) > thr
        self._badpix_ensure_shape(ff.shape)
        added = int((new & ~self.bad_pixel_mask).sum())
        self.bad_pixel_mask |= new
        self._rebuild_badpix_map()
        if not self.badpix_check.isChecked():
            self.badpix_check.setChecked(True)    # auto-enable so it takes effect
        self.status_label.setText(f"detected {added} new bad pixel(s)")

    def _paint_mask(self, row: int, col: int) -> None:
        if self.latest_frame is None:
            return
        self._badpix_ensure_shape(self.latest_frame.shape)
        r = int(self.badpix_brush_spin.value())
        add = self.mask_paint_mode == "add"
        if r <= 0:
            self.bad_pixel_mask[row, col] = add
        else:
            h, w = self.bad_pixel_mask.shape
            yy, xx = np.ogrid[:h, :w]
            self.bad_pixel_mask[(xx - col) ** 2 + (yy - row) ** 2 <= r * r] = add
        self._rebuild_badpix_map()
        if add and not self.badpix_check.isChecked():
            self.badpix_check.setChecked(True)

    def _set_paint_mode(self, mode: str) -> None:
        self.mask_paint_mode = mode
        self.badpix_paint_btn.setChecked(mode == "add")
        self.badpix_erase_btn.setChecked(mode == "erase")

    def _clear_badpix(self, *_a) -> None:
        if self.bad_pixel_mask is not None:
            self.bad_pixel_mask[:] = False
        self._rebuild_badpix_map()
        self.status_label.setText("bad-pixel mask cleared")

    def _save_badpix(self, *_a) -> None:
        if self.bad_pixel_mask is None or not self.bad_pixel_mask.any():
            self.status_label.setText("no bad-pixel mask to save")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save bad-pixel mask", "bad_pixels.npy", "NumPy mask (*.npy)")
        if path:
            np.save(path, self.bad_pixel_mask.astype(np.uint8))
            self.status_label.setText(f"saved mask: {os.path.basename(path)}")

    def _load_badpix(self, *_a) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load bad-pixel mask", "", "NumPy mask (*.npy)")
        if not path:
            return
        try:
            m = np.load(path).astype(bool)
        except Exception as e:  # noqa: BLE001
            self.status_label.setText(f"mask load failed: {e}")
            return
        if m.ndim != 2:
            self.status_label.setText("mask file is not 2D")
            return
        self.bad_pixel_mask = m
        self._rebuild_badpix_map()
        self.badpix_check.setChecked(True)
        self.status_label.setText(
            f"loaded mask: {os.path.basename(path)} ({int(m.sum())} px)")

    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("Camera")
        layout = QVBoxLayout(group)

        self.backend_label = QLabel("Backend: unknown")
        self.mode_label = QLabel("Requested mode: -")
        self.resolution_label = QLabel("Resolution: -")
        self.serial_label = QLabel("Serial: -")
        self.average_label = QLabel("Averaging: 1")
        self.exposure_status_label = QLabel("Exposure: - ms")
        for _w in (self.backend_label, self.mode_label, self.resolution_label,
                   self.serial_label, self.average_label, self.exposure_status_label):
            layout.addWidget(_w)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["irc806", "ophir", "mock", "auto"])
        self.mode_combo.setCurrentText(self.current_mode)
        layout.addWidget(QLabel("Connection mode"))
        layout.addWidget(self.mode_combo)

        # Integration time. Goldeye/SP1203: LOG-scale 1 us .. 1 s. IRC806 (and
        # other modes): the ORIGINAL linear 0.01 .. 8 ms (1 frame at 120 Hz) --
        # so the IRC806 is operated exactly as before the Goldeye was added.
        goldeye = self._is_goldeye_mode()
        self._int_log = goldeye
        if goldeye:
            self.INT_MIN_MS, self.INT_MAX_MS, self._int_decimals = 0.001, 1000.0, 3
        else:
            self.INT_MIN_MS, self.INT_MAX_MS, self._int_decimals = 0.01, 8.0, 2
        self.integration_label = QLabel("Integration time: 0.30 ms")
        self.integration_slider = QSlider(Qt.Orientation.Horizontal)
        self.integration_slider.setRange(0, 1000)   # mapped to INT_MIN..MAX ms
        self.integration_slider.valueChanged.connect(self.on_integration_slider_changed)
        self.integration_spin = QDoubleSpinBox()
        self.integration_spin.setDecimals(self._int_decimals)
        self.integration_spin.setRange(self.INT_MIN_MS, self.INT_MAX_MS)
        self.integration_spin.setSingleStep(0.05)
        self.integration_spin.valueChanged.connect(self.on_integration_spin_changed)
        layout.addWidget(self.integration_label)
        layout.addWidget(self.integration_slider)
        layout.addWidget(self.integration_spin)
        self._set_integration_value(0.3, emit=False)   # sync both widgets

        self.average_spin = QSpinBox()
        self.average_spin.setRange(1, 256)
        self.average_spin.setValue(1)
        self.average_spin.valueChanged.connect(self.on_average_changed)
        layout.addWidget(QLabel("Averaging"))
        layout.addWidget(self.average_spin)

        # -- SP1203 / Goldeye GenICam options: shown ONLY in Goldeye/Ophir mode,
        # so the IRC806 UI is unchanged from before the Goldeye was added.
        self.opt_combos = {}
        self.framerate_spin = None
        if goldeye:
            layout.addWidget(QLabel("— SP1203 / Goldeye options —"))
            for label, node, options in (
                ("Integration mode", "IntegrationMode",
                 ["IntegrateThenRead", "IntegrateWhileRead"]),
                ("Auto exposure", "ExposureAuto", ["Off", "Once", "Continuous"]),
                ("Sensor gain", "SensorGain", ["Gain0", "Gain1", "Gain2"]),
            ):
                row = QHBoxLayout()
                row.addWidget(QLabel(label))
                cb = QComboBox()
                cb.addItems(options)
                cb.currentTextChanged.connect(
                    lambda val, n=node: self.control_queue.put(
                        {"type": "set_option", "name": n, "value": val}))
                row.addWidget(cb)
                layout.addLayout(row)
                self.opt_combos[node] = cb

            fr_row = QHBoxLayout()
            fr_row.addWidget(QLabel("Frame rate (Hz)"))
            self.framerate_spin = QDoubleSpinBox()
            self.framerate_spin.setRange(0.03, 231.0)
            self.framerate_spin.setDecimals(2)
            self.framerate_spin.setValue(30.0)
            self.framerate_spin.valueChanged.connect(
                lambda v: self.control_queue.put(
                    {"type": "set_option", "name": "AcquisitionFrameRate", "value": float(v)}))
            fr_row.addWidget(self.framerate_spin)
            layout.addLayout(fr_row)

        start_button = QPushButton("Start")
        start_button.clicked.connect(lambda: self.control_queue.put({"type": "start"}))
        pause_button = QPushButton("Pause")
        pause_button.clicked.connect(lambda: self.control_queue.put({"type": "pause"}))
        snapshot_button = QPushButton("Snapshot")
        snapshot_button.clicked.connect(lambda: self.control_queue.put({"type": "snapshot"}))
        connect_button = QPushButton("Connect / Reconnect")
        connect_button.clicked.connect(self.connect_selected_mode)
        disconnect_button = QPushButton("Disconnect")
        disconnect_button.setToolTip(
            "Stop the stream and release the camera, staying offline (no "
            "auto-reconnect). Use this if the image splits / shows wrong pixels "
            "(GigE stream desync), then click Connect / Reconnect to recover.")
        disconnect_button.clicked.connect(self.disconnect_camera)
        top_buttons = QHBoxLayout()
        top_buttons.addWidget(connect_button)
        top_buttons.addWidget(start_button)
        bottom_buttons = QHBoxLayout()
        bottom_buttons.addWidget(disconnect_button)
        bottom_buttons.addWidget(pause_button)
        bottom_buttons.addWidget(snapshot_button)
        layout.addLayout(top_buttons)
        layout.addLayout(bottom_buttons)

        return group

    def _build_processing_group(self) -> QGroupBox:
        group = QGroupBox("Processing")
        layout = QVBoxLayout(group)

        # --- Background removal: capture a reference (e.g. lens capped / beam
        #     blocked), then subtract it from the live image ---
        bg_row = QHBoxLayout()
        self.btn_capture_bg = QPushButton("Capture background")
        self.btn_capture_bg.clicked.connect(self.capture_background)
        self.btn_clear_bg = QPushButton("Clear")
        self.btn_clear_bg.clicked.connect(self.clear_background)
        bg_row.addWidget(self.btn_capture_bg)
        bg_row.addWidget(self.btn_clear_bg)
        layout.addLayout(bg_row)

        self.bg_checkbox = QCheckBox("Subtract background")
        self.bg_checkbox.toggled.connect(self.on_bg_toggled)
        layout.addWidget(self.bg_checkbox)

        self.bg_status_label = QLabel("Background: none")
        layout.addWidget(self.bg_status_label)

        self.color_map_combo = QComboBox()
        self.color_map_combo.addItems(
            ["inferno", "viridis", "magma", "turbo", "coolwarm", "grey"])
        self.color_map_combo.currentTextChanged.connect(self.apply_color_map)
        layout.addWidget(QLabel("Color map"))
        layout.addWidget(self.color_map_combo)

        self.auto_scale_checkbox = QCheckBox("Auto colorbar (scale to frame)")
        self.auto_scale_checkbox.setChecked(self.auto_scale_display)
        self.auto_scale_checkbox.toggled.connect(self.on_auto_scale_toggled)
        layout.addWidget(self.auto_scale_checkbox)

        # With background subtraction, floor the colorbar at 0 so zero is the
        # bottom of the bar (instead of a symmetric ±range with zero in the
        # middle). Uncheck for the symmetric diverging view.
        self.zero_floor_display = True
        self.zero_floor_checkbox = QCheckBox("Colorbar starts at 0 (subtraction)")
        self.zero_floor_checkbox.setChecked(self.zero_floor_display)
        self.zero_floor_checkbox.toggled.connect(self.on_zero_floor_toggled)
        layout.addWidget(self.zero_floor_checkbox)

        # Fixed display range (used when Auto colorbar is OFF)
        layout.addWidget(QLabel("Fixed colorbar range (counts)"))
        range_row = QHBoxLayout()
        self.display_min_spin = QDoubleSpinBox()
        self.display_min_spin.setDecimals(0)
        self.display_min_spin.setRange(-65535, 65535)   # negative floor allowed (subtraction)
        self.display_min_spin.setValue(0)
        self.display_min_spin.setPrefix("Min ")
        self.display_max_spin = QDoubleSpinBox()
        self.display_max_spin.setDecimals(0)
        self.display_max_spin.setRange(1, 65535)
        self.display_max_spin.setValue(16383)
        self.display_max_spin.setPrefix("Max ")
        self.display_min_spin.valueChanged.connect(self.on_display_range_changed)
        self.display_max_spin.valueChanged.connect(self.on_display_range_changed)
        range_row.addWidget(self.display_min_spin)
        range_row.addWidget(self.display_max_spin)
        layout.addLayout(range_row)
        # initial enabled state matches auto toggle (auto starts off -> manual on)
        self.display_min_spin.setEnabled(not self.auto_scale_display)
        self.display_max_spin.setEnabled(not self.auto_scale_display)

        save_svg_button = QPushButton("Save snapshot as SVG")
        save_svg_button.clicked.connect(self.save_snapshot_svg)
        layout.addWidget(save_svg_button)

        save_pdf_button = QPushButton("Save snapshot as PDF")
        save_pdf_button.clicked.connect(self.save_snapshot_pdf)
        layout.addWidget(save_pdf_button)

        return group

    def _build_correction_group(self) -> QGroupBox:
        group = QGroupBox("Image Correction (NUC / BPR)")
        layout = QVBoxLayout(group)

        # Load the factory NUC state file (loads NUC + matched params, then
        # switches the output to corrected). Easiest way to get NUC'd data.
        layout.addWidget(QLabel("NUC state file (on camera)"))
        self.state_file_edit = QLineEdit("93534-01_120Hz_1.2ms.xml")
        layout.addWidget(self.state_file_edit)
        self.btn_load_state = QPushButton("Load NUC state")
        self.btn_load_state.setMinimumHeight(32)
        self.btn_load_state.clicked.connect(self.on_load_state)
        layout.addWidget(self.btn_load_state)

        self.nuc_checkbox = QCheckBox("Apply NUC + BPR (corrected output)")
        self.nuc_checkbox.toggled.connect(
            lambda c: self.control_queue.put({"type": "set_correction", "value": bool(c)}))
        layout.addWidget(self.nuc_checkbox)

        self.bpr_checkbox = QCheckBox("Bad-pixel replacement")
        self.bpr_checkbox.toggled.connect(
            lambda c: self.control_queue.put({"type": "set_bpr", "value": bool(c)}))
        layout.addWidget(self.bpr_checkbox)

        slot_row = QHBoxLayout()
        slot_row.addWidget(QLabel("NUC slot"))
        self.nuc_slot_spin = QSpinBox()
        self.nuc_slot_spin.setRange(0, 11)
        self.nuc_slot_spin.valueChanged.connect(
            lambda v: self.control_queue.put({"type": "set_nuc_slot", "value": int(v)}))
        slot_row.addWidget(self.nuc_slot_spin)
        slot_row.addStretch()
        layout.addLayout(slot_row)

        note = QLabel("NUC needs a calibration file loaded in the active slot "
                      "(via WinIRC/IRComm). Raw output = both off.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-size:11px;")
        layout.addWidget(note)
        return group

    def on_load_state(self) -> None:
        fn = self.state_file_edit.text().strip()
        if not fn:
            return
        self.status_label.setText(f"Loading NUC state '{fn}' (a few seconds)...")
        self.control_queue.put({"type": "load_state", "value": fn})
        self.nuc_checkbox.blockSignals(True)   # corrected output will be on
        self.nuc_checkbox.setChecked(True)
        self.nuc_checkbox.blockSignals(False)

    def _build_save_group(self) -> QGroupBox:
        group = QGroupBox("Save Image")
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("Folder"))
        dir_row = QHBoxLayout()
        self.save_dir_edit = QLineEdit(self.save_dir)
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse_save_dir)
        dir_row.addWidget(self.save_dir_edit)
        dir_row.addWidget(browse)
        layout.addLayout(dir_row)

        layout.addWidget(QLabel("Filename"))
        self.filename_edit = QLineEdit(self.save_filename)
        layout.addWidget(self.filename_edit)

        self.btn_save_image = QPushButton("Save image")
        self.btn_save_image.clicked.connect(self.save_image)
        self.btn_save_image.setMinimumHeight(34)
        layout.addWidget(self.btn_save_image)

        self.save_status_label = QLabel("Saved as: <date>.<filename>")
        self.save_status_label.setWordWrap(True)
        layout.addWidget(self.save_status_label)

        return group

    def browse_save_dir(self) -> None:
        start = self.save_dir_edit.text().strip() or self.save_dir
        chosen = QFileDialog.getExistingDirectory(self, "Select save folder", start)
        if chosen:
            self.save_dir_edit.setText(chosen)

    def save_image(self) -> None:
        if self.latest_frame is None:
            QMessageBox.information(self, "No data", "No frame to save yet.")
            return
        save_dir = self.save_dir_edit.text().strip() or self.save_dir
        fname = self.filename_edit.text().strip() or self.save_filename
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Folder error", f"Cannot create folder:\n{save_dir}\n{e}")
            return

        # Final name = <date>.<filename>  (date = YYYYMMDD_HHMMSS to avoid clashes)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{stamp}.{fname}"
        raw = self.latest_frame
        if raw.dtype != np.uint16:
            raw16 = np.clip(raw, 0, 65535).astype(np.uint16)
        else:
            raw16 = raw
        saved = []

        # 16-bit TIFF (lossless raw data)
        try:
            import tifffile
            tiff_path = os.path.join(save_dir, base + ".tiff")
            tifffile.imwrite(tiff_path, raw16)
            saved.append(os.path.basename(tiff_path))
        except Exception as e:  # noqa: BLE001
            print(f"[save] TIFF failed: {e}")

        # Raw NumPy (for analysis)
        try:
            np.save(os.path.join(save_dir, base + ".npy"), raw)
            saved.append(base + ".npy")
        except Exception as e:  # noqa: BLE001
            print(f"[save] npy failed: {e}")

        # Colormapped PNG matching the current display (levels + colormap)
        try:
            disp = self.last_display_frame if self.last_display_frame is not None else raw
            lo, hi = self._display_levels
            lut = self.color_map.getLookupTable(0.0, 1.0, 256)
            argb, alpha = pg.makeARGB(np.asarray(disp), lut=lut, levels=(lo, hi))
            qimg = pg.makeQImage(argb, alpha, transpose=False)
            png_path = os.path.join(save_dir, base + ".png")
            qimg.save(png_path)
            saved.append(base + ".png")
        except Exception as e:  # noqa: BLE001
            print(f"[save] PNG failed: {e}")

        if saved:
            msg = f"Saved {base}.* to {save_dir}"
            self.save_status_label.setText(msg)
            self.statusBar().showMessage(msg, 4000)
        else:
            self.save_status_label.setText("Save FAILED (see console)")

    def _build_measurements_group(self) -> QGroupBox:
        group = QGroupBox("Measurements")
        layout = QFormLayout(group)

        self.measurement_labels = {}
        for key in (
            "peak",
            "total_power",
            "centroid_x",
            "centroid_y",
            "beam_width_x",
            "beam_width_y",
        ):
            label = QLabel("-")
            layout.addRow(key.replace("_", " ").title(), label)
            self.measurement_labels[key] = label

        return group

    def _ms_to_slider(self, ms: float) -> int:
        ms = float(np.clip(ms, self.INT_MIN_MS, self.INT_MAX_MS))
        if getattr(self, "_int_log", True):
            frac = np.log10(ms / self.INT_MIN_MS) / np.log10(self.INT_MAX_MS / self.INT_MIN_MS)
        else:
            frac = (ms - self.INT_MIN_MS) / (self.INT_MAX_MS - self.INT_MIN_MS)
        return int(round(frac * 1000.0))

    def _slider_to_ms(self, s: int) -> float:
        frac = s / 1000.0
        if getattr(self, "_int_log", True):
            return float(self.INT_MIN_MS * (self.INT_MAX_MS / self.INT_MIN_MS) ** frac)
        return float(self.INT_MIN_MS + frac * (self.INT_MAX_MS - self.INT_MIN_MS))

    def _set_integration_value(self, integration_ms: float, *, emit: bool) -> None:
        integration_ms = float(np.clip(integration_ms, self.INT_MIN_MS, self.INT_MAX_MS))
        dec = getattr(self, "_int_decimals", 2)
        self.integration_label.setText(f"Integration time: {integration_ms:.{dec}f} ms")

        slider_value = self._ms_to_slider(integration_ms)
        if self.integration_slider.value() != slider_value:
            self.integration_slider.blockSignals(True)
            self.integration_slider.setValue(slider_value)
            self.integration_slider.blockSignals(False)

        if abs(self.integration_spin.value() - integration_ms) > 1e-6:
            self.integration_spin.blockSignals(True)
            self.integration_spin.setValue(integration_ms)
            self.integration_spin.blockSignals(False)

        if emit:
            self.control_queue.put({"type": "set_exposure", "value": integration_ms})

    def on_integration_slider_changed(self, value: int) -> None:
        self._set_integration_value(self._slider_to_ms(value), emit=True)

    def on_integration_spin_changed(self, value: float) -> None:
        self._set_integration_value(value, emit=True)

    def on_average_changed(self, value: int) -> None:
        self.control_queue.put({"type": "set_average", "value": int(value)})

    def connect_selected_mode(self) -> None:
        self.current_mode = self.mode_combo.currentText()
        self.status_label.setText(f"Connecting using {self.current_mode} mode...")
        self.control_queue.put({"type": "connect", "mode": self.current_mode})

    def disconnect_camera(self) -> None:
        """Manually drop the camera and stay offline (no auto-reconnect) until the
        user hits Connect / Reconnect -- for recovering a split/desynced stream."""
        self.status_label.setText("Disconnecting camera...")
        self.control_queue.put({"type": "disconnect"})

    def capture_background(self, *_args) -> None:
        """Average the next N incoming frames into the background reference."""
        self._bg_accum = None
        self._bg_capture_remaining = max(1, int(self.bg_average_frames))
        self.bg_status_label.setText(
            f"Background: capturing ({self._bg_capture_remaining})...")

    def on_bg_toggled(self, checked: bool) -> None:
        self.use_bg_subtraction = checked
        if checked and self.background_frame is None:
            self.bg_status_label.setText("Background: none — click 'Capture background'")

    def clear_background(self, *_args) -> None:
        self.background_frame = None
        self._bg_accum = None
        self._bg_capture_remaining = 0
        self.bg_status_label.setText("Background: none")

    def on_auto_scale_toggled(self, checked: bool) -> None:
        self.auto_scale_display = checked
        self.display_min_spin.setEnabled(not checked)
        self.display_max_spin.setEnabled(not checked)
        self._refresh_color_levels()

    def on_zero_floor_toggled(self, checked: bool) -> None:
        self.zero_floor_display = checked
        self._refresh_color_levels()   # fixed mode applies now; auto applies next frame

    def on_display_range_changed(self, *_args) -> None:
        self._refresh_color_levels()

    def _refresh_color_levels(self) -> None:
        """Apply the fixed colorbar range immediately (auto updates per frame)."""
        if self.auto_scale_display:
            return
        lo = float(self.display_min_spin.value())
        hi = float(self.display_max_spin.value())
        if hi <= lo:
            hi = lo + 1.0
        self.color_bar.setLevels((lo, hi))

    def on_image_clicked(self, event) -> None:
        """Pick the pixel whose row/column drive the horizontal/vertical profiles."""
        if self.latest_frame is None:
            return
        scene_pos = event.scenePos()
        if not self.image_plot.sceneBoundingRect().contains(scene_pos):
            return
        image_pos = self.image_item.mapFromScene(scene_pos)
        col = int(np.floor(image_pos.x()))
        row = int(np.floor(image_pos.y()))
        height, width = self.latest_frame.shape
        if col < 0 or row < 0 or col >= width or row >= height:
            return
        # In paint/erase mode a click edits the bad-pixel mask instead of moving
        # the profile cursor (brush radius = the "Brush (px)" spin).
        if self.mask_paint_mode != "none":
            self._paint_mask(row, col)
            return
        self.profile_pixel = (row, col)
        self._update_crosshair()
        # refresh immediately so the profiles jump to the new pixel
        if self.latest_frame is not None:
            self._update_profiles(self.latest_frame)

    def _profile_rc(self, shape) -> tuple[int, int]:
        height, width = shape
        if self.profile_pixel is None:
            return height // 2, width // 2
        row, col = self.profile_pixel
        return int(np.clip(row, 0, height - 1)), int(np.clip(col, 0, width - 1))

    def _update_crosshair(self) -> None:
        if self.latest_frame is None:
            return
        row, col = self._profile_rc(self.latest_frame.shape)
        self.v_line.setPos(col + 0.5)
        self.h_line.setPos(row + 0.5)

    def set_roi_visible(self, visible: bool) -> None:
        self.roi.setVisible(visible)
        self._update_roi_label()

    def _update_roi_label(self) -> None:
        roi = self.get_roi_bounds()
        if roi is None:
            self.roi_bounds_label.setText("ROI: full frame")
        else:
            r0, r1, c0, c1 = roi
            self.roi_bounds_label.setText(
                f"ROI: rows {r0}-{r1}, cols {c0}-{c1}  ({r1-r0}×{c1-c0} px)")

    def get_roi_bounds(self):
        """Current ROI as (row0, row1, col0, col1) clamped to the frame, or None
        when the ROI box is hidden / there is no frame yet (-> full-frame mean)."""
        if self.latest_frame is None or not self.roi.isVisible():
            return None
        try:
            st = self.roi.state
            x0 = max(0, int(st["pos"].x()))
            y0 = max(0, int(st["pos"].y()))
            rw = max(1, int(st["size"].x()))
            rh = max(1, int(st["size"].y()))
        except Exception:  # noqa: BLE001
            return None
        h, w = self.latest_frame.shape
        r0, r1 = min(y0, h - 1), min(y0 + rh, h)
        c0, c1 = min(x0, w - 1), min(x0 + rw, w)
        if r1 <= r0 or c1 <= c0:
            return None
        return (r0, r1, c0, c1)

    def _update_profiles(self, display_frame: np.ndarray) -> None:
        row, col = self._profile_rc(display_frame.shape)
        self.x_profile_curve.setData(display_frame[row, :])
        self.y_profile_curve.setData(display_frame[:, col])
        self.x_profile_plot.setTitle(f"Horizontal Profile (row {row})")
        self.y_profile_plot.setTitle(f"Vertical Profile (col {col})")

    def _style_profile_plot(self, plot: pg.PlotWidget) -> None:
        plot.setBackground("w")
        axis_pen = pg.mkPen("#666666", width=1)
        text_pen = "#333333"
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(text_pen)
            axis.setStyle(tickLength=6)

    def _make_color_map(self, name: str) -> pg.ColorMap:
        gradients = {
            "inferno": [
                (0.0, (0, 0, 4, 255)),
                (0.35, (87, 15, 109, 255)),
                (0.7, (187, 55, 84, 255)),
                (1.0, (252, 255, 164, 255)),
            ],
            "viridis": [
                (0.0, (68, 1, 84, 255)),
                (0.35, (59, 82, 139, 255)),
                (0.7, (33, 145, 140, 255)),
                (1.0, (253, 231, 37, 255)),
            ],
            "magma": [
                (0.0, (0, 0, 4, 255)),
                (0.4, (81, 18, 123, 255)),
                (0.75, (182, 55, 121, 255)),
                (1.0, (252, 253, 191, 255)),
            ],
            "turbo": [
                (0.0, (48, 18, 59, 255)),
                (0.25, (33, 144, 241, 255)),
                (0.5, (124, 255, 121, 255)),
                (0.75, (250, 186, 57, 255)),
                (1.0, (122, 4, 3, 255)),
            ],
            "coolwarm": [   # diverging: blue -> white -> red (zero = white mid)
                (0.0, (59, 76, 192, 255)),
                (0.5, (221, 221, 221, 255)),
                (1.0, (180, 4, 38, 255)),
            ],
            "grey": [
                (0.0, (0, 0, 0, 255)),
                (1.0, (255, 255, 255, 255)),
            ],
        }
        ticks = gradients[name]
        return pg.ColorMap(pos=[tick[0] for tick in ticks], color=[tick[1] for tick in ticks])

    def apply_color_map(self, name: str) -> None:
        self.color_map = self._make_color_map(name)
        self.image_item.setColorMap(self.color_map)
        self.color_bar.setColorMap(self.color_map)

    def _compute_measurements(self, frame: np.ndarray) -> dict[str, float]:
        weights = np.clip(frame.astype(np.float64), 0.0, None)
        total_power = float(weights.sum())
        peak = float(np.nanmax(frame)) if frame.size else 0.0

        if total_power <= 0:
            return {
                "peak": peak,
                "total_power": 0.0,
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "beam_width_x": 0.0,
                "beam_width_y": 0.0,
            }

        yy, xx = np.indices(frame.shape, dtype=np.float64)
        centroid_x = float((weights * xx).sum() / total_power)
        centroid_y = float((weights * yy).sum() / total_power)
        var_x = float((weights * (xx - centroid_x) ** 2).sum() / total_power)
        var_y = float((weights * (yy - centroid_y) ** 2).sum() / total_power)

        return {
            "peak": peak,
            "total_power": total_power,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "beam_width_x": 2.0 * np.sqrt(max(var_x, 0.0)),
            "beam_width_y": 2.0 * np.sqrt(max(var_y, 0.0)),
        }

    def _render_export_panel(self, painter: QPainter) -> None:
        self.export_panel.render(painter)

    def save_snapshot_svg(self) -> None:
        if self.latest_frame is None:
            QMessageBox.information(self, "No data", "No frame is available yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save SVG Snapshot",
            "wincam_snapshot.svg",
            "SVG files (*.svg)",
        )
        if not path:
            return

        generator = QSvgGenerator()
        generator.setFileName(path)
        size = self.export_panel.size()
        generator.setSize(QSize(max(size.width(), 1), max(size.height(), 1)))
        generator.setViewBox(QRect(0, 0, max(size.width(), 1), max(size.height(), 1)))
        generator.setTitle("WinCamD snapshot")
        painter = QPainter(generator)
        self._render_export_panel(painter)
        painter.end()

    def save_snapshot_pdf(self) -> None:
        if self.latest_frame is None:
            QMessageBox.information(self, "No data", "No frame is available yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF Snapshot",
            "wincam_snapshot.pdf",
            "PDF files (*.pdf)",
        )
        if not path:
            return

        writer = QPdfWriter(path)
        writer.setResolution(300)
        painter = QPainter(writer)
        self._render_export_panel(painter)
        painter.end()

    def update_from_worker(self) -> None:
        latest_frame_packet = None
        try:
            while not self.frame_queue.empty():
                packet = self.frame_queue.get_nowait()
                if packet["type"] == "status":
                    self._apply_status(packet["status"])
                elif packet["type"] == "frame":
                    latest_frame_packet = packet
        except queue.Empty:
            pass

        if latest_frame_packet is None:
            return

        frame = latest_frame_packet.get("frame")
        if latest_frame_packet.get("shared") and self.shared_frame is not None:
            frame_shape = tuple(latest_frame_packet.get("shape", (0, 0)))
            if len(frame_shape) == 2 and frame_shape[0] > 0 and frame_shape[1] > 0:
                frame = self.shared_frame[: frame_shape[0], : frame_shape[1]].copy()
        if frame is not None:
            self._apply_frame(frame, latest_frame_packet["measurement"])

    def _kspace_metadata(self) -> dict:
        """Camera state embedded into the saved K-space hypercube metadata."""
        s = self.latest_status or {}
        return {
            "camera_serial": s.get("serial_number"),
            "exposure_ms": s.get("exposure_ms"),
            "averaging": s.get("average_count"),
            "fpa_temp_k": s.get("fpa_temp_k"),
            "board_temp_c": s.get("board_temp_c"),
            "nuc_corrected": (self.nuc_checkbox.isChecked()
                              if hasattr(self, "nuc_checkbox") else None),
            "save_filename_camera": self.save_filename,
        }

    def _apply_status(self, status: dict) -> None:
        self.latest_status = status     # keep for K-space metadata
        self.backend_label.setText(f"Backend: {status.get('backend', 'unknown')}")
        requested_mode = status.get("requested_mode", self.current_mode) or self.current_mode
        self.current_mode = requested_mode
        self.mode_label.setText(f"Requested mode: {requested_mode}")
        if self.mode_combo.currentText() != requested_mode:
            self.mode_combo.setCurrentText(requested_mode)

        width = status.get("width", 0)
        height = status.get("height", 0)
        resolution = f"{width} x {height}" if width and height else "-"
        self.resolution_label.setText(f"Resolution: {resolution}")
        self.serial_label.setText(f"Serial: {status.get('serial_number') or '-'}")

        average_count = int(status.get("average_count", 1) or 1)
        exposure_ms = float(status.get("exposure_ms", 0.3) or 0.3)
        self.average_label.setText(f"Averaging: {average_count}")
        self.exposure_status_label.setText(f"Exposure: {exposure_ms:.2f} ms")
        if self.average_spin.value() != average_count:
            self.average_spin.blockSignals(True)
            self.average_spin.setValue(average_count)
            self.average_spin.blockSignals(False)
        self._set_integration_value(exposure_ms, emit=False)

        # Temperatures (board/FPGA in °C, FPA in K). NaN -> "--".
        board = float(status.get("board_temp_c", float("nan")))
        fpa = float(status.get("fpa_temp_k", float("nan")))
        b_txt = f"{board:.1f} °C" if board == board else "-- °C"
        f_txt = f"{fpa:.1f} K" if fpa == fpa else "-- K"
        self.temp_value_label.setText(f"FPGA/Board: {b_txt}   FPA: {f_txt}")

        state = "connected" if status.get("connected") else "offline"
        acquisition = "acquiring" if status.get("acquiring") else "idle"
        self.status_label.setText(f"Camera is {state}, {acquisition}. {status.get('message', '')}")

    def _apply_frame(self, frame: np.ndarray, measurement: dict) -> None:
        # Replace masked cold/hot pixels with their nearest good neighbour so they
        # are neither displayed nor saved (both paths read latest_frame).
        frame = self._correct_bad_pixels(frame)
        self.latest_frame = frame

        # Background capture: average N incoming frames into background_frame.
        if self._bg_capture_remaining > 0:
            f32 = frame.astype(np.float32)
            self._bg_accum = f32 if self._bg_accum is None else self._bg_accum + f32
            self._bg_capture_remaining -= 1
            if self._bg_capture_remaining == 0:
                n = max(1, int(self.bg_average_frames))
                self.background_frame = self._bg_accum / n
                self._bg_accum = None
                self.bg_status_label.setText(f"Background: captured (avg {n})")
            else:
                self.bg_status_label.setText(
                    f"Background: capturing ({self._bg_capture_remaining})...")

        subtracting = self.use_bg_subtraction and self.background_frame is not None
        if subtracting:
            display_frame = frame.astype(np.float32) - self.background_frame
        else:
            display_frame = frame
        self.last_display_frame = display_frame

        self.image_item.setImage(display_frame, autoLevels=False)
        self.analysis_frame_counter += 1
        if self.analysis_frame_counter % 3 == 0:
            if subtracting:
                if self.auto_scale_display:
                    if self.zero_floor_display:
                        # Floor at 0: colorbar runs 0..max, so zero is the bottom.
                        fmax = float(np.nanmax(display_frame)) if display_frame.size else 1.0
                        self.color_bar.setLevels((0.0, max(fmax, 1.0)))
                    else:
                        # Symmetric ±range (diverging view, zero in the middle).
                        max_abs = float(np.nanmax(np.abs(display_frame))) if display_frame.size else 1.0
                        max_abs = max(max_abs, 1.0)
                        self.color_bar.setLevels((-max_abs, max_abs))
                else:
                    # Fixed range: honor the user's Min (their zero) and Max.
                    lo = float(self.display_min_spin.value())
                    hi = float(self.display_max_spin.value())
                    if hi <= lo:
                        hi = lo + 1.0
                    self.color_bar.setLevels((lo, hi))
            else:
                if self.auto_scale_display:
                    fmin = float(np.nanmin(display_frame)) if display_frame.size else 0.0
                    fmax = float(np.nanmax(display_frame)) if display_frame.size else 1.0
                    if fmax <= fmin:
                        fmax = fmin + 1.0
                    self.color_bar.setLevels((fmin, fmax))
                else:
                    lo = float(self.display_min_spin.value())
                    hi = float(self.display_max_spin.value())
                    if hi <= lo:
                        hi = lo + 1.0
                    self.color_bar.setLevels((lo, hi))
            try:
                self._display_levels = tuple(self.color_bar.levels())
            except Exception:  # noqa: BLE001
                pass
            self._update_crosshair()
            self._update_profiles(display_frame)

            processed_measurement = self._compute_measurements(display_frame)
            for key, label in self.measurement_labels.items():
                value = processed_measurement.get(key, 0.0)
                if key in {"centroid_x", "centroid_y", "beam_width_x", "beam_width_y"}:
                    label.setText(f"{value:.2f} px")
                elif key == "total_power":
                    label.setText(f"{value:.0f}")
                else:
                    label.setText(f"{value:.1f}")

        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_update >= 1.0:
            fps = self.frame_count / (now - self.last_fps_update)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_update = now

    def closeEvent(self, event) -> None:
        if hasattr(self, "measure_panel"):
            self.measure_panel.shutdown()
        if hasattr(self, "stages_panel"):
            self.stages_panel.shutdown()
        self.control_queue.put({"type": "stop"})
        if self.shared_frame_handle is not None:
            self.shared_frame_handle.close()
        super().closeEvent(event)
