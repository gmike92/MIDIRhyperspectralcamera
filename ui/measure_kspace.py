"""Measure tab -- K-space hyperspectral (per-pixel TWINS spectra) + Z-scan.

A TWINS wedge scan stores the full 2-D ROI at every position (a datacube), then
HyperspectralProcessor runs an independent DFT per pixel -> a spectrum cube
(n_freq, h, w). You can scrub wavelength to see the spatial map at each λ and
click any pixel to see its spectrum.

Z-scan: optionally wrap the whole thing in an outer loop over Thorlabs
delay-stage positions (fs) -- one hyperspectral map per delay -> a 4-D
(z, λ, y, x) dataset.

Controls live in the sidebar tab; the map+spectrum open in a separate resizable
window (HyperViewer), like the repo's standalone measurement windows.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

# Disk-space guards for saving hypercubes (large files).
LOW_DISK_WARN_GB = 3.0     # warn once when free space drops below this during a scan
LOW_DISK_ABORT_GB = 0.5    # abort the scan to avoid failed / truncated saves


def _free_gb(path):
    """Free space (GB) on the volume that holds `path`, walking up to the nearest
    existing directory. Returns None if it can't be determined."""
    try:
        p = os.path.abspath(path)
        while p and not os.path.isdir(p):
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        return shutil.disk_usage(p or ".").free / 1e9
    except Exception:  # noqa: BLE001
        return None

from instruments.subtwinslv import TwinsScanner
from instruments.hyperspectral import (
    HyperspectralProcessor, DEFAULT_START_MM, DEFAULT_STOP_MM, DEFAULT_N_STEPS,
    DEFAULT_APODIZATION, DEFAULT_WL_START, DEFAULT_WL_STOP,
    ZEROFILL_FACTOR, ZEROFILL_MIN, ZEROFILL_MAX, resolve_n_points,
    DEFAULT_ZPD_MM, DEFAULT_ZPD_WINDOW_MM,
)
from instruments.dsp import APOD_TYPES


# grey and jet aren't bundled in pyqtgraph and need matplotlib (absent here), so
# build them by hand; viridis/inferno/magma/turbo are pyqtgraph built-ins.
_MANUAL_CMAPS = {
    "grey": ([0.0, 1.0], [(0, 0, 0), (255, 255, 255)]),
    "jet": ([0.0, 0.125, 0.375, 0.625, 0.875, 1.0],
            [(0, 0, 128), (0, 0, 255), (0, 255, 255),
             (255, 255, 0), (255, 0, 0), (128, 0, 0)]),
}


def _get_cmap(name):
    """Return a pyqtgraph ColorMap by name, or None if unavailable."""
    key = str(name).lower()
    if key == "gray":
        key = "grey"
    if key in _MANUAL_CMAPS:
        pos, cols = _MANUAL_CMAPS[key]
        return pg.ColorMap(pos=pos, color=cols)
    try:
        cm = pg.colormap.get(key)        # pyqtgraph built-ins (viridis, etc.)
        if cm is not None:
            return cm
    except Exception:  # noqa: BLE001
        pass
    return None


# ===========================================================================
# Pop-out viewer: wavelength-scrub map + per-pixel spectrum + z selector
# ===========================================================================
class HyperViewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("K-Space Hyperspectral Viewer")
        self.resize(900, 520)
        self.wavelengths = None
        self.cubes = []          # list of (n_freq, h, w)  (in-RAM mode)
        self.z_values = []       # list of z (mm) or None
        self.sat_masks = []      # list of (h, w) bool or None
        self.sel_px = None       # (row, col) in ROI
        self._cont_regions = []  # shaded band overlays on the spectrum plot
        # Lazy mode: a callable z_index -> cube, used for big Z-series files so
        # only the current position's cube is held in RAM.
        self.cube_loader = None
        self._lazy_cache = {}    # z_index -> cube (kept tiny)

        root = QHBoxLayout(self)

        # left: spatial map with wavelength slider (ImageView time axis = λ)
        left = QVBoxLayout()
        self.lbl_cal = QLabel("")          # wedge-axis calibration status (from metadata)
        self.lbl_cal.setStyleSheet("font-weight:600;")
        left.addWidget(self.lbl_cal)
        self.map_view = pg.ImageView()
        self.map_view.view.invertY(True)     # match live-view orientation
        self.map_view.scene.sigMouseClicked.connect(self._on_click)
        # Hide pyqtgraph's built-in timeline/ROI/menu -- we drive the wavelength
        # with our own slider below. (Leaving the built-in one connected made the
        # two sliders chase each other / drift.)
        for _w in (self.map_view.ui.roiBtn, self.map_view.ui.menuBtn,
                   self.map_view.ui.roiPlot):
            _w.hide()
        left.addWidget(self.map_view, 1)

        ctl = QHBoxLayout()
        self.combo_map = QComboBox()
        self.combo_map.addItems(["λ scrub", "Peak λ", "Peak intensity",
                                 "SAM (selected px)", "Continuum line"])
        self.combo_map.setToolTip("λ scrub: spatial map at each wavelength.\n"
                                  "Peak λ/intensity: per-pixel spectral peak.\n"
                                  "SAM: spectral-angle distance to the selected pixel.\n"
                                  "Continuum line: per-pixel line emission with the\n"
                                  "  broadband (thermal) continuum subtracted -- a linear\n"
                                  "  baseline from two shoulder bands is removed under the\n"
                                  "  line band (see CONTINUUM_SUBTRACTION.md).")
        self.combo_map.currentTextChanged.connect(self._refresh_map)
        ctl.addWidget(QLabel("Map:")); ctl.addWidget(self.combo_map, 1)
        self.combo_deriv = QComboBox()
        self.combo_deriv.addItems(["raw", "d/dλ", "d²/dλ²"])
        self.combo_deriv.setToolTip("Spectral derivative of the pixel spectrum.")
        self.combo_deriv.currentTextChanged.connect(self._update_spectrum)
        ctl.addWidget(QLabel("Spectrum:")); ctl.addWidget(self.combo_deriv, 1)
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["grey", "inferno", "viridis", "magma", "turbo", "jet"])
        self.combo_cmap.setToolTip("Map colormap.")
        self.combo_cmap.currentTextChanged.connect(self._apply_cmap)
        ctl.addWidget(QLabel("Colors:")); ctl.addWidget(self.combo_cmap, 1)
        left.addLayout(ctl)

        # Continuum-line bands (line + two shoulders, µm). Only relevant/visible
        # when Map = "Continuum line"; a linear continuum estimated from the two
        # shoulders is subtracted under the line band, per pixel.
        def _band(lo, hi):
            a = QDoubleSpinBox(); b = QDoubleSpinBox()
            for s, v in ((a, lo), (b, hi)):
                s.setDecimals(3); s.setRange(0.1, 100.0); s.setSuffix(" µm")
                s.setSingleStep(0.005); s.setValue(v)
                s.valueChanged.connect(self._on_continuum_changed)
            return a, b
        self.cont_row_w = QWidget()
        crow = QHBoxLayout(self.cont_row_w); crow.setContentsMargins(0, 0, 0, 0)
        self.spin_line0, self.spin_line1 = _band(3.965, 4.050)
        self.spin_lsh0, self.spin_lsh1 = _band(3.900, 3.955)
        self.spin_rsh0, self.spin_rsh1 = _band(4.060, 4.120)
        crow.addWidget(QLabel("Line")); crow.addWidget(self.spin_line0); crow.addWidget(self.spin_line1)
        crow.addWidget(QLabel("L")); crow.addWidget(self.spin_lsh0); crow.addWidget(self.spin_lsh1)
        crow.addWidget(QLabel("R")); crow.addWidget(self.spin_rsh0); crow.addWidget(self.spin_rsh1)
        self.cont_row_w.setVisible(False)
        left.addWidget(self.cont_row_w)

        # Remember the colormap choice across sessions (and in the standalone viewer).
        self._cmap_settings = QtCore.QSettings("MIR_CAMERA", "HyperViewer")
        saved_cmap = self._cmap_settings.value("cmap", "grey")
        self.combo_cmap.blockSignals(True)
        self.combo_cmap.setCurrentText(str(saved_cmap))
        self.combo_cmap.blockSignals(False)
        self._apply_cmap()

        # dedicated wavelength slider (active in λ-scrub mode)
        wlrow = QHBoxLayout()
        self.wl_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.wl_slider.valueChanged.connect(self._on_wl_slider)
        self.lbl_wl_val = QLabel("-- µm")
        self.lbl_wl_val.setStyleSheet("font-weight:600;")
        wlrow.addWidget(QLabel("Wavelength:"))
        wlrow.addWidget(self.wl_slider, 1)
        wlrow.addWidget(self.lbl_wl_val)
        left.addLayout(wlrow)

        # delay (z) slider -- hidden entirely unless a Z-scan was acquired
        self.zrow_w = QWidget()
        zrow = QHBoxLayout(self.zrow_w); zrow.setContentsMargins(0, 0, 0, 0)
        self.lbl_z = QLabel("Z (mm):")
        self.z_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        self.lbl_z_val = QLabel("-")
        zrow.addWidget(self.lbl_z)
        zrow.addWidget(self.z_slider, 1)
        zrow.addWidget(self.lbl_z_val)
        self.zrow_w.setVisible(False)
        left.addWidget(self.zrow_w)
        root.addLayout(left, 2)

        # right: spectrum of the selected pixel
        right = QVBoxLayout()
        self.spec_plot = pg.PlotWidget(title="Pixel spectrum")
        self.spec_plot.setLabel("bottom", "Wavelength", units="µm")
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen("#e8590c", width=2))
        right.addWidget(self.spec_plot, 1)
        self.lbl_pixel = QLabel("Click the map to pick a pixel")
        right.addWidget(self.lbl_pixel)
        root.addLayout(right, 1)

    def set_calibration_note(self, meta) -> None:
        """Show whether the loaded spectra were computed on the motor-calibrated
        wedge axis, read from the file/scan metadata."""
        meta = meta or {}
        if meta.get("position_axis_used") == "calibrated" or meta.get("motor_calibration_applied"):
            f = meta.get("motor_calibration_file") or "parameters_int.txt"
            self.lbl_cal.setText(f"Wedge axis: CALIBRATED  ({f})")
            self.lbl_cal.setStyleSheet("font-weight:600; color:#2f9e44;")
        elif meta:
            self.lbl_cal.setText("Wedge axis: raw measured (no motor calibration)")
            self.lbl_cal.setStyleSheet("font-weight:600; color:#e8590c;")
        else:
            self.lbl_cal.setText("")

    def set_result(self, wavelengths, cubes, z_values, sat_masks=None) -> None:
        self.wavelengths = np.asarray(wavelengths)
        self.cubes = list(cubes)
        self.z_values = list(z_values)
        self.sat_masks = list(sat_masks) if sat_masks else [None] * len(self.cubes)
        nf = len(self.wavelengths)
        self.wl_slider.blockSignals(True)
        self.wl_slider.setMinimum(0)
        self.wl_slider.setMaximum(max(0, nf - 1))
        self.wl_slider.setValue(0)
        self.wl_slider.blockSignals(False)
        self.cube_loader = None      # in-RAM mode
        self._lazy_cache = {}
        self._configure_sliders(len(self.cubes))

    def set_result_lazy(self, wavelengths, z_values, cube_loader,
                        sat_masks=None) -> None:
        """Z-series mode: cubes are loaded on demand via cube_loader(z_index),
        so only the current position is held in RAM (for big per-Z files)."""
        self.wavelengths = np.asarray(wavelengths)
        self.cubes = []
        self.z_values = list(z_values)
        self.sat_masks = list(sat_masks) if sat_masks else [None] * len(z_values)
        self.cube_loader = cube_loader
        self._lazy_cache = {}
        nf = len(self.wavelengths)
        self.wl_slider.blockSignals(True)
        self.wl_slider.setMinimum(0); self.wl_slider.setMaximum(max(0, nf - 1))
        self.wl_slider.setValue(0)
        self.wl_slider.blockSignals(False)
        self._configure_sliders(len(z_values))

    def _configure_sliders(self, n_z: int) -> None:
        self.zrow_w.setVisible(n_z > 1)
        self.z_slider.blockSignals(True)
        self.z_slider.setMinimum(0); self.z_slider.setMaximum(max(0, n_z - 1))
        self.z_slider.setValue(0)
        self.z_slider.blockSignals(False)
        self._show_cube(0)

    def _current_cube(self):
        zi = self.z_slider.value()
        if self.cube_loader is not None:
            if zi not in self._lazy_cache:
                self._lazy_cache.clear()          # keep only the current cube
                try:
                    self._lazy_cache[zi] = self.cube_loader(zi)
                except Exception:  # noqa: BLE001
                    return None
            return self._lazy_cache.get(zi)
        if not self.cubes:
            return None
        return self.cubes[zi]

    def _current_mask(self):
        zi = self.z_slider.value()
        if self.sat_masks and zi < len(self.sat_masks):
            return self.sat_masks[zi]
        return None

    def _reference_spectrum(self, cube):
        """Reference spectrum for SAM: the selected pixel, else the ROI mean."""
        if self.sel_px is not None:
            row, col = self.sel_px
            _, h, w = cube.shape
            if 0 <= row < h and 0 <= col < w:
                return cube[:, row, col]
        from instruments.analysis import roi_average
        return roi_average(cube, self._current_mask())[0]

    def _show_cube(self, z_idx: int) -> None:
        z = self.z_values[z_idx] if z_idx < len(self.z_values) else None
        self.lbl_z_val.setText("-" if z is None else f"{z:.4f} mm")
        self._refresh_map()
        self._update_spectrum()

    def _refresh_map(self) -> None:
        cube = self._current_cube()
        if cube is None:
            return
        mode = self.combo_map.currentText()
        self.cont_row_w.setVisible(mode == "Continuum line")
        self._update_continuum_overlay()
        if mode == "λ scrub":
            # cube is (n_freq, h, w); axis 0 = wavelength, driven by our slider.
            self.map_view.setImage(cube, xvals=self.wavelengths, autoLevels=True)
            self.map_view.ui.roiPlot.hide()   # setImage re-shows it; keep it hidden
            self.wl_slider.setEnabled(True)
            idx = max(0, min(self.wl_slider.value(), len(self.wavelengths) - 1))
            self.map_view.setCurrentIndex(idx)
            self._set_wl_label(idx)
            return
        from instruments import analysis as A
        if mode == "Peak λ":
            img = A.peak_wavelength_map(cube, self.wavelengths)
        elif mode == "Peak intensity":
            img = A.peak_intensity_map(cube)
        elif mode == "Continuum line":
            ln, ls, rs = self._continuum_bands()
            img = A.continuum_line_image(cube, self.wavelengths, ln, ls, rs)
            if img is None:
                self.wl_slider.setEnabled(False)
                self.lbl_wl_val.setText("bands out of range")
                return
        else:  # SAM
            img = A.spectral_angle_map(cube, self._reference_spectrum(cube))
        img = np.asarray(img, dtype=float)
        mask = self._current_mask()
        if mask is not None:
            img = img.copy(); img[np.asarray(mask, bool)] = np.nan
        self.map_view.setImage(img, autoLevels=True)
        # No wavelength axis on a 2-D map -> the slider doesn't apply.
        self.wl_slider.setEnabled(False)
        self.lbl_wl_val.setText(f"{mode} map")

    def _continuum_bands(self):
        """(line, left_shoulder, right_shoulder) bands (µm), each (lo, hi)."""
        return (tuple(sorted((self.spin_line0.value(), self.spin_line1.value()))),
                tuple(sorted((self.spin_lsh0.value(), self.spin_lsh1.value()))),
                tuple(sorted((self.spin_rsh0.value(), self.spin_rsh1.value()))))

    def _on_continuum_changed(self, *args) -> None:
        if self.combo_map.currentText() == "Continuum line":
            self._refresh_map()

    def _update_continuum_overlay(self) -> None:
        """Shade the line + shoulder bands on the pixel spectrum plot so they can
        be positioned on the resonance / clean continuum. Cleared in other modes."""
        for reg in self._cont_regions:
            self.spec_plot.removeItem(reg)
        self._cont_regions = []
        if self.combo_map.currentText() != "Continuum line":
            return
        ln, ls, rs = self._continuum_bands()
        for (lo, hi), rgba in ((ln, (232, 89, 12, 45)),      # line = orange
                               (ls, (77, 171, 247, 40)),     # shoulders = blue
                               (rs, (77, 171, 247, 40))):
            reg = pg.LinearRegionItem([lo, hi], movable=False,
                                      brush=pg.mkBrush(*rgba), pen=pg.mkPen(None))
            reg.setZValue(-10)
            self.spec_plot.addItem(reg)
            self._cont_regions.append(reg)

    def _on_z_changed(self, idx: int) -> None:
        # In lazy mode self.cubes is empty -> count by z_values instead.
        n = len(self.z_values)
        if 0 <= idx < n:
            self._show_cube(idx)

    def _apply_cmap(self, *args) -> None:
        cm = _get_cmap(self.combo_cmap.currentText())
        if cm is not None:
            self.map_view.setColorMap(cm)
        self._cmap_settings.setValue("cmap", self.combo_cmap.currentText())

    def _set_wl_label(self, idx: int) -> None:
        if self.wavelengths is None or len(self.wavelengths) == 0:
            return
        idx = max(0, min(idx, len(self.wavelengths) - 1))
        self.lbl_wl_val.setText(f"{self.wavelengths[idx]:.3f} µm")

    def _on_wl_slider(self, idx: int) -> None:
        """Dedicated wavelength slider -> drive the ImageView's scrub index."""
        if self.combo_map.currentText() != "λ scrub":
            return
        self.map_view.setCurrentIndex(int(idx))
        self._set_wl_label(int(idx))

    def _on_click(self, ev) -> None:
        cube = self._current_cube()
        if cube is None:
            return
        ii = self.map_view.getImageItem()
        pos = ev.scenePos()
        if not ii.sceneBoundingRect().contains(pos):
            return
        p = ii.mapFromScene(pos)
        col, row = int(p.x()), int(p.y())     # row-major image: x=col, y=row
        _, h, w = cube.shape
        if 0 <= row < h and 0 <= col < w:
            self.sel_px = (row, col)
            self._update_spectrum()
            # SAM is referenced to the selected pixel -> recompute the map.
            if self.combo_map.currentText().startswith("SAM"):
                self._refresh_map()

    def _update_spectrum(self) -> None:
        cube = self._current_cube()
        if cube is None or self.sel_px is None or self.wavelengths is None:
            return
        row, col = self.sel_px
        _, h, w = cube.shape
        if not (0 <= row < h and 0 <= col < w):
            return
        y = np.asarray(cube[:, row, col], dtype=float)
        order = {"raw": 0, "d/dλ": 1, "d²/dλ²": 2}.get(self.combo_deriv.currentText(), 0)
        for _ in range(order):
            y = np.gradient(y, self.wavelengths)
        self.spec_curve.setData(self.wavelengths, y)
        self.lbl_pixel.setText(f"Pixel (row {row}, col {col}) in ROI")


# ===========================================================================
# Live interferogram preview (ROI mean vs wedge position, during the scan)
# ===========================================================================
class LiveInterferogram(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Live interferogram")
        self.resize(560, 320)
        lay = QVBoxLayout(self)
        self.plot = pg.PlotWidget(title="ROI mean vs wedge position")
        self.plot.setLabel("bottom", "Wedge position", units="mm")
        self.plot.setLabel("left", "ROI mean")
        self.curve = self.plot.plot(pen=pg.mkPen("#1c7ed6", width=2),
                                    symbol="o", symbolSize=3, symbolBrush="#1c7ed6")
        lay.addWidget(self.plot)
        self._x: list = []
        self._y: list = []

    def reset(self) -> None:
        self._x, self._y = [], []
        self.curve.setData([], [])

    def add_point(self, pos: float, value: float) -> None:
        self._x.append(pos); self._y.append(value)
        self.curve.setData(self._x, self._y)


def load_kspace_npz(path: str):
    """Read a saved K-space .npz -> (wavelengths, cubes, z_values, sat_masks).

    Works for files saved by MeasurePanel (spectral cube, optional saturation
    masks and raw interferogram). Returns None if the spectral cube is absent.
    """
    d = np.load(path, allow_pickle=True)
    if "wavelengths" not in d or "spectrum_cubes" not in d:
        return None
    wl = np.asarray(d["wavelengths"])
    cubes = [np.asarray(c) for c in d["spectrum_cubes"]]
    zv = d["z_values"] if "z_values" in d else np.full(len(cubes), np.nan)
    z_values = [None if np.isnan(v) else float(v) for v in np.asarray(zv).ravel()]
    masks = None
    if "saturation_masks" in d:
        masks = [np.asarray(m, bool) for m in d["saturation_masks"]]
    return wl, cubes, z_values, masks


def kspace_metadata(path: str) -> dict:
    """The embedded metadata dict of a saved K-space .npz (or {}). Lets a viewer
    show whether the spectra were computed on the calibrated wedge axis."""
    try:
        with np.load(path, allow_pickle=True) as d:
            if "metadata" in d.files:
                return dict(d["metadata"].item())
    except Exception:  # noqa: BLE001
        pass
    return {}


# ===========================================================================
# Sidebar controls
# ===========================================================================
class MeasurePanel(QWidget):
    sig_status = QtCore.pyqtSignal(str)
    sig_done = QtCore.pyqtSignal(object, object, object, object)  # wl, cubes, z, sat_masks
    sig_point = QtCore.pyqtSignal(float, float, bool)  # pos, ROI-mean, is_first_of_scan
    sig_warn = QtCore.pyqtSignal(str, str)  # (title, message) -> modal warning on the GUI thread

    def __init__(self, stages_panel, frame_source, roi_provider,
                 roi_show=None, bg_provider=None, save_dir_provider=None,
                 meta_provider=None, save_dir: str = r"D:\CAMERA\kspace") -> None:
        super().__init__()
        self.sp = stages_panel
        self.frame_source = frame_source
        self.roi_provider = roi_provider      # () -> (r0,r1,c0,c1) or None (full frame)
        self.roi_show = roi_show              # (bool) -> toggle the on-image ROI box
        self.bg_provider = bg_provider        # () -> (background_frame|None, subtract_bool)
        self.save_dir_provider = save_dir_provider  # () -> current save folder (camera)
        self.meta_provider = meta_provider    # () -> dict of camera metadata
        self.save_dir = save_dir              # fallback if no provider
        self._scan_meta = {}                  # scan parameters captured at scan start
        self.background_map = None             # binned-ROI background saved with the cube
        self.background_subtracted = False
        self._abort = False
        self._paused = False                   # Pause holds the worker at loop boundaries
        self._low_disk_warned = False
        self.viewer = None
        self.wavelengths = None
        self.cubes = []
        self.z_values = []
        self.sat_masks = []
        # Raw interferogram cubes + positions (per z), kept for optional saving
        # and for walk-off calibration from a sharp-sample scan.
        self.raw_cubes = []
        self.raw_positions = []
        self._last_datacube = None
        self._last_positions = None
        # Per-run save folder (set on each Acquire in _start); all of a run's
        # files land in <run-stamp>.<filename>/ under the camera folder.
        self._run_folder = None
        self._run_stamp = None
        self._save_folder = None
        self._save_fname = None
        self.live_monitor = None
        # A processor instance used only for live scan-parameter estimation
        # (resolution / max-step); loads the calibration once.
        self.est_proc = HyperspectralProcessor()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_roi_group())
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_spectrum_group())
        layout.addWidget(self._build_postproc_group())
        layout.addWidget(self._build_walkoff_group())
        layout.addWidget(self._build_zscan_group())
        layout.addWidget(self._build_ascan_group())
        layout.addWidget(self._build_run_group())

        # Persist scan/spectrum params (incl. the wavelength window) between
        # measurements. Restore first, THEN bind saves so restoring doesn't
        # immediately rewrite the same values.
        self._settings = QtCore.QSettings("MIR_CAMERA", "KSpace")
        self._restore_settings()
        for widget, _cast in self._persisted_spins().values():
            widget.valueChanged.connect(self._save_settings)
        self.combo_apod.currentTextChanged.connect(self._save_settings)
        self.combo_ftregion.currentTextChanged.connect(self._save_settings)
        self.chk_walkoff.toggled.connect(self._save_settings)
        self.combo_save.currentTextChanged.connect(self._save_settings)
        self.chk_save_raw.toggled.connect(self._save_settings)
        for chk in self._persisted_checks().values():   # sat, svd, zscan, zones
            chk.toggled.connect(self._save_settings)
        self.edit_filename.editingFinished.connect(self._save_settings)

        self.sig_status.connect(self._on_status)
        self.sig_done.connect(self._on_done)
        self.sig_point.connect(self._on_point)
        self.sig_warn.connect(self._on_warn)
        # The master scan checkboxes drive the (Z × angle) grid-size readout.
        self.chk_zscan.toggled.connect(self._update_grid_label)
        self.chk_ascan.toggled.connect(self._update_grid_label)
        self._update_step()
        self._update_zstep()
        self._update_astep()

        # Keep the ROI px readout live while the box is dragged.
        self._roi_timer = QtCore.QTimer(self)
        self._roi_timer.timeout.connect(self._refresh_roi_label)
        self._roi_timer.start(700)

    # -- groups --------------------------------------------------------------
    def _build_roi_group(self) -> QGroupBox:
        g = QGroupBox("ROI + binning")
        grid = QGridLayout(g)
        hint = QLabel("Set the ROI on the live view (tick \"Show measurement ROI\" "
                      "and drag the box). It is saved with the measurement.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(hint, 0, 0, 1, 2)

        self.spin_bin = QSpinBox()
        self.spin_bin.setRange(1, 64)
        self.spin_bin.setValue(1)
        self.spin_bin.setToolTip("Bin NxN pixels into one super-pixel (better SNR, "
                                 "smaller cube). 1 = no binning.")
        self.spin_bin.valueChanged.connect(self._refresh_roi_label)
        grid.addWidget(QLabel("Binning (NxN)"), 1, 0)
        grid.addWidget(self.spin_bin, 1, 1)

        self.lbl_roi = QLabel("ROI: full frame")
        self.lbl_roi.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(self.lbl_roi, 2, 0, 1, 2)
        return g

    def _refresh_roi_label(self) -> None:
        roi = self.roi_provider() if self.roi_provider else None
        b = self.spin_bin.value()
        if roi is None:
            self.lbl_roi.setText("ROI: full frame")
        else:
            r0, r1, c0, c1 = roi
            h, w = (r1 - r0) // b, (c1 - c0) // b
            self.lbl_roi.setText(f"ROI: {r1-r0}×{c1-c0} px  →  {h}×{w} after bin {b}")

    def _build_scan_group(self) -> QGroupBox:
        g = QGroupBox("TWINS cube scan")
        grid = QGridLayout(g)
        self.spin_start = self._mm_spin(DEFAULT_START_MM)
        self.spin_stop = self._mm_spin(DEFAULT_STOP_MM)
        self.spin_steps = QSpinBox(); self.spin_steps.setRange(3, 10000)
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

    def _build_spectrum_group(self) -> QGroupBox:
        g = QGroupBox("Spectrum (per-pixel DFT)")
        grid = QGridLayout(g)
        # Apodization window: 'gaussian' is the NIREOS position-space window
        # (uses Gauss width below); the rest are standard FTIR windows.
        self.combo_apod = QComboBox(); self.combo_apod.addItems(APOD_TYPES)
        self.combo_apod.setCurrentText("gaussian")
        self.combo_apod.setToolTip("Apodization window. Affects the spectral "
                                   "lineshape and the resolution estimate below.")
        self.combo_apod.currentTextChanged.connect(self._update_step)
        self.spin_apod = QDoubleSpinBox(); self.spin_apod.setRange(0.01, 5.0)
        self.spin_apod.setSingleStep(0.05); self.spin_apod.setValue(DEFAULT_APODIZATION)
        self.spin_apod.setToolTip("Gaussian apodization width (only for the "
                                  "'gaussian' window type).")
        # MWIR band for this InSb camera (repo defaults are LWIR 8-14 µm).
        # Overridden by the last-used value once a measurement has been run.
        self.spin_wl0 = self._um_spin(3.8)
        self.spin_wl1 = self._um_spin(4.4)
        self.spin_wl0.valueChanged.connect(self._update_step)
        self.spin_wl1.valueChanged.connect(self._update_step)
        # 0 = "Auto": ZEROFILL_FACTOR x steps, clamped to [MIN, MAX]. This is
        # interpolation/oversampling of the spectrum, NOT true resolution (that
        # is fixed by the scan length). A manual value > 0 overrides Auto.
        self.spin_nfreq = QSpinBox(); self.spin_nfreq.setRange(0, ZEROFILL_MAX)
        self.spin_nfreq.setValue(0); self.spin_nfreq.setSpecialValueText("Auto")
        self.spin_nfreq.setToolTip(
            f"Spectral output points (interpolation only -- does not change the\n"
            f"true resolution, which is set by the scan length).\n"
            f"Auto = {ZEROFILL_FACTOR}x steps, clamped to [{ZEROFILL_MIN}, {ZEROFILL_MAX}].")
        self.spin_nfreq.valueChanged.connect(self._update_step)
        self.lbl_nfreq_auto = QLabel("")
        self.lbl_nfreq_auto.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(QLabel("Apod type"), 0, 0); grid.addWidget(self.combo_apod, 0, 1)
        grid.addWidget(QLabel("Gauss width"), 1, 0); grid.addWidget(self.spin_apod, 1, 1)
        grid.addWidget(QLabel("λ start"), 2, 0); grid.addWidget(self.spin_wl0, 2, 1)
        grid.addWidget(QLabel("λ stop"), 3, 0); grid.addWidget(self.spin_wl1, 3, 1)
        grid.addWidget(QLabel("N freq"), 4, 0); grid.addWidget(self.spin_nfreq, 4, 1)
        grid.addWidget(self.lbl_nfreq_auto, 5, 0, 1, 2)

        # Calibration-aware estimates: spectral resolution from scan range +
        # apodization window, and the max stage step that still samples the
        # shortest λ at 5 pts/cycle.
        self.lbl_resolution = QLabel("-- nm"); self.lbl_resolution.setStyleSheet("font-weight:600;")
        self.lbl_min_step = QLabel("-- µm"); self.lbl_min_step.setStyleSheet("font-weight:600;")
        grid.addWidget(QLabel("Resolution"), 6, 0); grid.addWidget(self.lbl_resolution, 6, 1)
        grid.addWidget(QLabel("Max step (5/cyc)"), 7, 0); grid.addWidget(self.lbl_min_step, 7, 1)

        # FT window: which part of the interferogram to transform.
        self.combo_ftregion = QComboBox()
        self.combo_ftregion.addItems(["full", "center", "tails"])
        self.combo_ftregion.setToolTip(
            "Which part of the interferogram to Fourier-transform (relative to ZPD):\n"
            "  full   = whole interferogram\n"
            "  center = burst only (±width) -> broad spectral features\n"
            "  tails  = wings only (beyond ±width) -> high-Q narrow resonances")
        self.spin_ftwidth = QDoubleSpinBox()
        self.spin_ftwidth.setRange(0.001, 50.0); self.spin_ftwidth.setDecimals(3)
        self.spin_ftwidth.setSingleStep(0.01); self.spin_ftwidth.setValue(0.10)
        self.spin_ftwidth.setSuffix(" mm")
        self.spin_ftwidth.setToolTip("Half-width around ZPD separating center from tails.")
        grid.addWidget(QLabel("FT region"), 8, 0); grid.addWidget(self.combo_ftregion, 8, 1)
        grid.addWidget(QLabel("FT width (±ZPD)"), 9, 0); grid.addWidget(self.spin_ftwidth, 9, 1)
        return g

    def _build_postproc_group(self) -> QGroupBox:
        g = QGroupBox("Post-processing")
        grid = QGridLayout(g)
        self.chk_sat = QCheckBox("Mask saturated pixels")
        self.chk_sat.setChecked(True)
        self.chk_sat.setToolTip("Exclude pixels that clipped at any wedge "
                                "position from the ROI average and the maps.")
        grid.addWidget(self.chk_sat, 0, 0, 1, 2)
        self.spin_sat = QSpinBox(); self.spin_sat.setRange(1, 65535)
        self.spin_sat.setValue(16383); self.spin_sat.setSuffix(" cts")
        self.spin_sat.setToolTip("Saturation count level (14-bit full scale = 16383).")
        grid.addWidget(QLabel("Saturation level"), 1, 0); grid.addWidget(self.spin_sat, 1, 1)

        self.chk_svd = QCheckBox("SVD denoise (low-rank)")
        self.chk_svd.setToolTip("Keep the k strongest SVD components of the cube; "
                                "removes spatially-incoherent noise.")
        grid.addWidget(self.chk_svd, 2, 0, 1, 2)
        self.spin_svd_k = QSpinBox(); self.spin_svd_k.setRange(1, 64); self.spin_svd_k.setValue(6)
        self.spin_svd_k.setToolTip("Number of spectral components to keep.")
        grid.addWidget(QLabel("Components (k)"), 3, 0); grid.addWidget(self.spin_svd_k, 3, 1)
        return g

    def _build_walkoff_group(self) -> QGroupBox:
        g = QGroupBox("Walk-off correction (TWINS image drift)")
        grid = QGridLayout(g)
        hint = QLabel("Calibrate the per-frame image shift ONCE on a sharp, "
                      "high-contrast target, then apply it to every scan.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(hint, 0, 0, 1, 2)

        self.chk_walkoff = QCheckBox("Apply walk-off correction")
        grid.addWidget(self.chk_walkoff, 1, 0, 1, 2)

        self.spin_wo_y = self._wo_spin(); self.spin_wo_x = self._wo_spin()
        grid.addWidget(QLabel("Rate Y (px/mm)"), 2, 0); grid.addWidget(self.spin_wo_y, 2, 1)
        grid.addWidget(QLabel("Rate X (px/mm)"), 3, 0); grid.addWidget(self.spin_wo_x, 3, 1)

        self.btn_wo_cal = QPushButton("Calibrate from last scan")
        self.btn_wo_cal.setToolTip("Register the frames of the most recent scan "
                                   "(use a sharp sample) and fit the shift rate.")
        self.btn_wo_cal.clicked.connect(self._calibrate_walkoff)
        grid.addWidget(self.btn_wo_cal, 4, 0, 1, 2)

        self.lbl_wo = QLabel("not calibrated")
        self.lbl_wo.setWordWrap(True); self.lbl_wo.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(self.lbl_wo, 5, 0, 1, 2)
        return g

    def _wo_spin(self):
        s = QDoubleSpinBox(); s.setRange(-1000.0, 1000.0); s.setDecimals(4)
        s.setSingleStep(0.1); s.setValue(0.0); s.setSuffix(" px/mm"); return s

    def _calibrate_walkoff(self) -> None:
        cube = self._last_datacube
        pos = self._last_positions
        if cube is None or pos is None:
            self.lbl_wo.setText("Run a scan on a sharp sample first, then calibrate.")
            return
        self.lbl_wo.setText("calibrating (registering frames)...")
        self.btn_wo_cal.setEnabled(False)
        try:
            from instruments.walkoff import estimate_shift_rate
            est = estimate_shift_rate(cube, pos)
            self.spin_wo_y.setValue(est["rate_y"])
            self.spin_wo_x.setValue(est["rate_x"])
            self.chk_walkoff.setChecked(True)
            self.lbl_wo.setText(
                f"rate_y={est['rate_y']:.3f} (r²={est['r2_y']:.2f}), "
                f"rate_x={est['rate_x']:.3f} (r²={est['r2_x']:.2f}) px/mm — "
                f"low r² ⇒ not a clean linear drift / use a sharper sample.")
        except Exception as e:  # noqa: BLE001
            self.lbl_wo.setText(f"calibration error: {e}")
        finally:
            self.btn_wo_cal.setEnabled(True)

    # Default zones: (enabled, start_mm, stop_mm, step_um). Three fixed zones so
    # you can e.g. sample finely near focus and coarsely far from it in one scan.
    _ZONE_DEFAULTS = ((True, 0.0, 0.5, 10.0),
                      (False, 0.5, 2.0, 50.0),
                      (False, 2.0, 5.0, 100.0))

    def _build_zscan_group(self) -> QGroupBox:
        g = QGroupBox("Z-scan (Thorlabs stage, mm)")
        grid = QGridLayout(g)
        self.chk_zscan = QCheckBox("Enable Z-scan (map per z position)")
        grid.addWidget(self.chk_zscan, 0, 0, 1, 5)
        for col, text in enumerate(("Zone", "Z start", "Z stop", "Step", "Points")):
            grid.addWidget(QLabel(text), 1, col)
        # Each enabled zone contributes its own evenly-spaced points; the enabled
        # zones are concatenated, sorted and de-duplicated into one Z target list.
        self.zone_rows = []
        for i, (en, z0, z1, st) in enumerate(self._ZONE_DEFAULTS):
            row = 2 + i
            chk = QCheckBox(str(i + 1)); chk.setChecked(en)
            s0 = self._zmm_spin(z0); s1 = self._zmm_spin(z1)
            su = self._zstep_um_spin(st)
            lbl = QLabel("-- pts")
            chk.toggled.connect(self._update_zstep)
            for w in (s0, s1, su):
                w.valueChanged.connect(self._update_zstep)
            grid.addWidget(chk, row, 0)
            grid.addWidget(s0, row, 1); grid.addWidget(s1, row, 2)
            grid.addWidget(su, row, 3); grid.addWidget(lbl, row, 4)
            self.zone_rows.append(dict(chk=chk, z0=s0, z1=s1, step=su, lbl=lbl))
        self.lbl_zstep = QLabel("-- pts"); self.lbl_zstep.setStyleSheet("font-weight:600;")
        grid.addWidget(QLabel("Total Z"), 5, 0, 1, 3)
        grid.addWidget(self.lbl_zstep, 5, 3, 1, 2)
        return g

    def _zone_points(self, zr) -> list:
        """Evenly-spaced Z positions (mm) for one zone row (endpoints inclusive)."""
        lo, hi = zr["z0"].value(), zr["z1"].value()
        if hi < lo:
            lo, hi = hi, lo
        if hi <= lo:
            return [lo]
        step_mm = max(zr["step"].value() / 1000.0, 1e-6)
        n = int(round((hi - lo) / step_mm)) + 1
        return list(np.linspace(lo, hi, max(n, 2)))

    def _zone_targets(self) -> list:
        """Sorted, de-duplicated Z positions (mm) from all ENABLED zones."""
        pts = []
        for zr in self.zone_rows:
            if zr["chk"].isChecked():
                pts.extend(self._zone_points(zr))
        if not pts:
            return []
        pts.sort()
        out = [pts[0]]
        for v in pts[1:]:
            if v - out[-1] > 5e-4:   # merge points closer than 0.5 µm
                out.append(v)
        return out

    # Default angle zones: (enabled, start_deg, stop_deg, step_deg). Mirrors the
    # Z-scan zones so you can sample some angle ranges finely and others coarsely.
    _ANGLE_DEFAULTS = ((True, 0.0, 180.0, 10.0),
                       (False, 80.0, 100.0, 2.0),
                       (False, 180.0, 360.0, 30.0))

    def _build_ascan_group(self) -> QGroupBox:
        g = QGroupBox("Angle-scan (Thorlabs rotator, deg)")
        grid = QGridLayout(g)
        self.chk_ascan = QCheckBox("Enable angle-scan (map per angle)")
        grid.addWidget(self.chk_ascan, 0, 0, 1, 5)
        for col, text in enumerate(("Zone", "Start", "Stop", "Step", "Points")):
            grid.addWidget(QLabel(text), 1, col)
        # Same zone model as the Z-scan; enabled zones are concatenated, sorted
        # and de-duplicated into one angle target list (in degrees).
        self.angle_rows = []
        for i, (en, a0, a1, st) in enumerate(self._ANGLE_DEFAULTS):
            row = 2 + i
            chk = QCheckBox(str(i + 1)); chk.setChecked(en)
            s0 = self._adeg_spin(a0); s1 = self._adeg_spin(a1)
            su = self._astep_deg_spin(st)
            lbl = QLabel("-- pts")
            chk.toggled.connect(self._update_astep)
            for w in (s0, s1, su):
                w.valueChanged.connect(self._update_astep)
            grid.addWidget(chk, row, 0)
            grid.addWidget(s0, row, 1); grid.addWidget(s1, row, 2)
            grid.addWidget(su, row, 3); grid.addWidget(lbl, row, 4)
            self.angle_rows.append(dict(chk=chk, a0=s0, a1=s1, step=su, lbl=lbl))
        self.lbl_astep = QLabel("-- pts"); self.lbl_astep.setStyleSheet("font-weight:600;")
        grid.addWidget(QLabel("Total angles"), 5, 0, 1, 3)
        grid.addWidget(self.lbl_astep, 5, 3, 1, 2)
        self.lbl_grid = QLabel(""); self.lbl_grid.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_grid.setWordWrap(True)
        grid.addWidget(self.lbl_grid, 6, 0, 1, 5)
        return g

    def _angle_points(self, ar) -> list:
        """Evenly-spaced angles (deg) for one zone row (endpoints inclusive)."""
        lo, hi = ar["a0"].value(), ar["a1"].value()
        if hi < lo:
            lo, hi = hi, lo
        if hi <= lo:
            return [lo]
        step = max(ar["step"].value(), 1e-4)
        n = int(round((hi - lo) / step)) + 1
        return list(np.linspace(lo, hi, max(n, 2)))

    def _angle_targets(self) -> list:
        """Sorted, de-duplicated angles (deg) from all ENABLED zones."""
        pts = []
        for ar in self.angle_rows:
            if ar["chk"].isChecked():
                pts.extend(self._angle_points(ar))
        if not pts:
            return []
        pts.sort()
        out = [pts[0]]
        for v in pts[1:]:
            if v - out[-1] > 1e-3:   # merge angles closer than 0.001 deg
                out.append(v)
        return out

    def _build_run_group(self) -> QGroupBox:
        g = QGroupBox("Run")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Acquire")
        self.btn_run.clicked.connect(self._start)
        self.btn_pause = QPushButton("Pause"); self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip("Hold the scan at the next interferogram / DFT "
                                  "boundary (the current cube finishes first). "
                                  "Press again to resume.")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_run); row.addWidget(self.btn_pause)
        row.addWidget(self.btn_stop)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.btn_recompute = QPushButton("Recompute")
        self.btn_recompute.clicked.connect(self._recompute)
        self.btn_recompute.setToolTip("Re-run the DFT on the LAST scan's raw "
                                      "interferogram with the current settings (FT region, "
                                      "apodization, λ, denoise) -- no re-scan.")
        self.btn_view = QPushButton("Open Viewer"); self.btn_view.clicked.connect(self._open_viewer)
        self.btn_load = QPushButton("Load"); self.btn_load.clicked.connect(self._load)
        self.btn_load.setToolTip("Open a saved K-space .npz in the viewer.")
        self.btn_save = QPushButton("Save"); self.btn_save.clicked.connect(self._save)
        row2.addWidget(self.btn_recompute); row2.addWidget(self.btn_view)
        row2.addWidget(self.btn_load); row2.addWidget(self.btn_save)
        v.addLayout(row2)
        row3 = QHBoxLayout()
        self.combo_save = QComboBox()
        self.combo_save.addItems(["Full cube (every pixel)", "ROI average", "Both"])
        self.combo_save.setCurrentText("Both")
        self.combo_save.setToolTip(
            "What 'Save' writes:\n"
            "  Full cube  -> every pixel's spectrum (.npz, λ×h×w)\n"
            "  ROI average -> mean±std spectrum over the acquired ROI (.csv)\n"
            "  Both -> both files.")
        row3.addWidget(QLabel("Save as")); row3.addWidget(self.combo_save, 1)
        v.addLayout(row3)
        row4 = QHBoxLayout()
        self.edit_filename = QLineEdit("kspace")
        self.edit_filename.setToolTip("Base filename; files are saved as "
                                      "<date>.<filename> in the camera's save folder.")
        row4.addWidget(QLabel("Filename")); row4.addWidget(self.edit_filename, 1)
        v.addLayout(row4)
        self.chk_save_raw = QCheckBox("Raw interferogram saved for reprocessing (always)")
        self.chk_save_raw.setToolTip("The raw (positions, datacube) is ALWAYS stored "
                                     "in the .npz so you can reprocess offline (FT window / "
                                     "apodization / ZPD) without re-scanning. It is small "
                                     "next to the spectrum (n_pos << n_freq).")
        self.chk_save_raw.setChecked(True)
        self.chk_save_raw.setEnabled(False)
        v.addWidget(self.chk_save_raw)
        self.progress = QProgressBar(); v.addWidget(self.progress)
        self.lbl_status = QLabel("idle"); self.lbl_status.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_status.setWordWrap(True); v.addWidget(self.lbl_status)
        return g

    # -- small spin helpers --------------------------------------------------
    def _mm_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(0.0, 50.0); s.setDecimals(3)
        s.setSingleStep(0.1); s.setValue(val); s.setSuffix(" mm"); return s

    def _um_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(0.1, 100.0); s.setDecimals(2)
        s.setSingleStep(0.1); s.setValue(val); s.setSuffix(" µm"); return s

    def _zmm_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(0.0, 300.0); s.setDecimals(4)
        s.setSingleStep(0.1); s.setValue(val); s.setSuffix(" mm"); return s

    def _zstep_um_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(0.1, 10000.0); s.setDecimals(2)
        s.setSingleStep(1.0); s.setValue(val); s.setSuffix(" µm"); return s

    def _adeg_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(0.0, 360.0); s.setDecimals(3)
        s.setSingleStep(1.0); s.setValue(val); s.setSuffix(" deg"); return s

    def _astep_deg_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(0.001, 360.0); s.setDecimals(3)
        s.setSingleStep(0.5); s.setValue(val); s.setSuffix(" deg"); return s

    # -- persistence (remember params between measurements) ------------------
    def _persisted_spins(self) -> dict:
        """key -> (widget, cast). QSettings stores strings, so cast on restore."""
        zones = {}
        for i, zr in enumerate(self.zone_rows):
            zones[f"ks_zone{i}_z0"] = (zr["z0"], float)
            zones[f"ks_zone{i}_z1"] = (zr["z1"], float)
            zones[f"ks_zone{i}_step"] = (zr["step"], float)
        for i, ar in enumerate(self.angle_rows):
            zones[f"ks_azone{i}_a0"] = (ar["a0"], float)
            zones[f"ks_azone{i}_a1"] = (ar["a1"], float)
            zones[f"ks_azone{i}_step"] = (ar["step"], float)
        return {**zones,
            "ks_start": (self.spin_start, float),
            "ks_stop": (self.spin_stop, float),
            "ks_steps": (self.spin_steps, int),
            "ks_frames": (self.spin_frames, int),
            "ks_bin": (self.spin_bin, int),
            "ks_apod": (self.spin_apod, float),
            "ks_wl0": (self.spin_wl0, float),
            "ks_wl1": (self.spin_wl1, float),
            "ks_nfreq": (self.spin_nfreq, int),
            "ks_wo_y": (self.spin_wo_y, float),
            "ks_wo_x": (self.spin_wo_x, float),
            "ks_sat_level": (self.spin_sat, int),
            "ks_svd_k": (self.spin_svd_k, int),
            "ks_ftwidth": (self.spin_ftwidth, float),
        }

    def _persisted_checks(self) -> dict:
        """key -> checkbox widgets persisted between measurements."""
        checks = {"ks_sat_on": self.chk_sat, "ks_svd_on": self.chk_svd,
                  "ks_zscan_on": self.chk_zscan, "ks_ascan_on": self.chk_ascan}
        for i, zr in enumerate(self.zone_rows):
            checks[f"ks_zone{i}_on"] = zr["chk"]
        for i, ar in enumerate(self.angle_rows):
            checks[f"ks_azone{i}_on"] = ar["chk"]
        return checks

    def _restore_settings(self) -> None:
        for key, (widget, cast) in self._persisted_spins().items():
            val = self._settings.value(key, None)
            if val is None:
                continue
            try:
                widget.setValue(cast(val))
            except (TypeError, ValueError):
                pass
        apod = self._settings.value("ks_apod_type", None)
        if apod is not None:
            self.combo_apod.setCurrentText(str(apod))
        ftr = self._settings.value("ks_ftregion", None)
        if ftr is not None:
            self.combo_ftregion.setCurrentText(str(ftr))
        wo = self._settings.value("ks_walkoff_on", None)
        if wo is not None:
            self.chk_walkoff.setChecked(str(wo).lower() == "true")
        save_mode = self._settings.value("ks_save_mode", None)
        if save_mode is not None:
            self.combo_save.setCurrentText(str(save_mode))
        for key, chk in self._persisted_checks().items():
            v = self._settings.value(key, None)
            if v is not None:
                chk.setChecked(str(v).lower() == "true")
        fn = self._settings.value("ks_filename", None)
        if fn is not None:
            self.edit_filename.setText(str(fn))

    def _save_settings(self, *args) -> None:
        for key, (widget, _cast) in self._persisted_spins().items():
            self._settings.setValue(key, widget.value())
        self._settings.setValue("ks_apod_type", self.combo_apod.currentText())
        self._settings.setValue("ks_ftregion", self.combo_ftregion.currentText())
        self._settings.setValue("ks_walkoff_on", self.chk_walkoff.isChecked())
        self._settings.setValue("ks_save_mode", self.combo_save.currentText())
        for key, chk in self._persisted_checks().items():
            self._settings.setValue(key, chk.isChecked())
        self._settings.setValue("ks_filename", self.edit_filename.text())

    def _update_step(self) -> None:
        n = self.spin_steps.value()
        start, stop = self.spin_start.value(), self.spin_stop.value()
        step_um = None
        if n > 1:
            step_um = abs(stop - start) / (n - 1) * 1000
            self.lbl_step.setText(f"{step_um:.1f} µm")
        else:
            self.lbl_step.setText("-- µm")
        # Show what "Auto" resolves to (interpolation bins), so the readout
        # tracks the scan length live.
        if hasattr(self, "lbl_nfreq_auto"):
            manual = self.spin_nfreq.value()
            resolved = resolve_n_points(n, manual=manual)
            if manual > 0:
                self.lbl_nfreq_auto.setText(f"N freq = {resolved} (manual)")
            else:
                self.lbl_nfreq_auto.setText(f"Auto → {resolved} bins ({ZEROFILL_FACTOR}×{n} steps)")

        # Calibration-aware resolution + sampling-adequacy check.
        if hasattr(self, "lbl_resolution"):
            wl0, wl1 = self.spin_wl0.value(), self.spin_wl1.value()
            wl_center = 0.5 * (wl0 + wl1)
            apod_type = self.combo_apod.currentText()
            res_nm = self.est_proc.estimate_resolution_nm(
                abs(stop - start), wl_center, apod_type=apod_type)
            self.lbl_resolution.setText(
                "-- nm" if res_nm is None else f"~{res_nm:.0f} nm @ {wl_center:.1f} µm")

            wl_short = min(wl0, wl1)
            max_um = self.est_proc.max_step_um(wl_short, samples_per_cycle=5)
            if max_um is None:
                self.lbl_min_step.setText("-- µm")
                self.lbl_min_step.setStyleSheet("font-weight:600;")
            else:
                self.lbl_min_step.setText(f"{max_um:.2f} µm @ {wl_short:.1f} µm")
                # Red if the chosen step under-samples the shortest λ, else green.
                if step_um is not None and step_um > max_um:
                    self.lbl_min_step.setStyleSheet("font-weight:600; color:#f44336;")
                else:
                    self.lbl_min_step.setStyleSheet("font-weight:600; color:#4CAF50;")

    def _update_zstep(self) -> None:
        for zr in self.zone_rows:
            if zr["chk"].isChecked():
                zr["lbl"].setText(f"{len(self._zone_points(zr))} pts")
            else:
                zr["lbl"].setText("off")
        total = len(self._zone_targets())
        self.lbl_zstep.setText(f"{total} pts" if total else "-- pts")
        self._update_grid_label()

    def _update_astep(self) -> None:
        for ar in self.angle_rows:
            if ar["chk"].isChecked():
                ar["lbl"].setText(f"{len(self._angle_points(ar))} pts")
            else:
                ar["lbl"].setText("off")
        total = len(self._angle_targets())
        self.lbl_astep.setText(f"{total} pts" if total else "-- pts")
        self._update_grid_label()

    def _update_grid_label(self) -> None:
        """Show the total number of acquired maps = (Z points) × (angle points)."""
        if not hasattr(self, "lbl_grid"):
            return
        nz = len(self._zone_targets()) if self.chk_zscan.isChecked() else 0
        na = len(self._angle_targets()) if self.chk_ascan.isChecked() else 0
        if not nz and not na:
            self.lbl_grid.setText("")
            return
        gz, ga = max(nz, 1), max(na, 1)
        parts = []
        if nz:
            parts.append(f"{nz} Z")
        if na:
            parts.append(f"{na} angle")
        self.lbl_grid.setText(f"Grid: {' × '.join(parts)} = {gz * ga} map(s) acquired")

    # -- run -----------------------------------------------------------------
    def _start(self) -> None:
        if not getattr(self.sp.twins, "is_connected", False):
            self.sig_status.emit("TWINS stage not connected")
            return
        zscan = self.chk_zscan.isChecked()
        z_targets = self._zone_targets() if zscan else []
        if zscan and not getattr(self.sp.delay, "is_connected", False):
            self.sig_status.emit("Delay stage not connected (needed for Z-scan)")
            return
        if zscan and not z_targets:
            self.sig_status.emit("Z-scan enabled but no zones produce points -- enable a zone")
            return
        ascan = self.chk_ascan.isChecked()
        a_targets = self._angle_targets() if ascan else []
        rotator = getattr(self.sp, "rotator", None)
        if ascan and not getattr(rotator, "is_connected", False):
            self.sig_status.emit("Rotator not connected (needed for angle-scan)")
            return
        if ascan and not a_targets:
            self.sig_status.emit("Angle-scan enabled but no zones produce points -- enable a zone")
            return
        if self.frame_source() is None:
            self.sig_status.emit("No live frame -- start the camera first")
            return
        roi = self.roi_provider() if self.roi_provider else None   # None = full frame
        self._scan_roi = roi          # saved with the measurement
        self._scan_bin = self.spin_bin.value()
        if roi is not None:
            r0, r1, c0, c1 = roi
            self.sig_status.emit(f"ROI saved for scan: rows {r0}-{r1}, cols {c0}-{c1}")

        walkoff = (dict(rate_y=self.spin_wo_y.value(), rate_x=self.spin_wo_x.value())
                   if self.chk_walkoff.isChecked() else None)
        # Snapshot the captured background (full frame) + whether to subtract it,
        # taken now so it can't change mid-scan.
        bg, bg_sub = self.bg_provider() if self.bg_provider else (None, False)
        bg = None if bg is None else np.asarray(bg, dtype=np.float32)
        if roi is not None and bg is not None:
            self.sig_status.emit("background " + ("subtracted" if bg_sub else "saved (not subtracted)"))
        params = dict(
            start=self.spin_start.value(), stop=self.spin_stop.value(),
            n=self.spin_steps.value(), frames=self.spin_frames.value(), roi=roi,
            bin=self.spin_bin.value(),
            apod=self.spin_apod.value(), apod_type=self.combo_apod.currentText(),
            wl0=self.spin_wl0.value(),
            wl1=self.spin_wl1.value(), nfreq=self.spin_nfreq.value(),
            walkoff=walkoff,
            background=bg, bg_subtract=bool(bg_sub and bg is not None),
            sat_on=self.chk_sat.isChecked(), sat_level=self.spin_sat.value(),
            svd_on=self.chk_svd.isChecked(), svd_k=self.spin_svd_k.value(),
            ft_region=self.combo_ftregion.currentText(), ft_width=self.spin_ftwidth.value(),
            zscan=zscan, z_targets=z_targets,
            ascan=ascan, a_targets=a_targets,
        )
        # Capture scan parameters as metadata (no arrays) for the saved hypercube.
        nsteps = params["n"]
        step_um = (abs(params["stop"] - params["start"]) / (nsteps - 1) * 1000
                   if nsteps > 1 else None)
        self._scan_meta = dict(
            start_mm=params["start"], stop_mm=params["stop"], n_steps=nsteps,
            step_um=step_um, frames_per_point=params["frames"], binning=params["bin"],
            roi=list(roi) if roi is not None else None,
            apodization=params["apod_type"], apod_width=params["apod"],
            wl_start_um=params["wl0"], wl_stop_um=params["wl1"],
            n_freq_setting=params["nfreq"], expected_zpd_mm=DEFAULT_ZPD_MM,
            walkoff=walkoff, background_subtracted=params["bg_subtract"],
            saturation_masking=params["sat_on"], saturation_level=params["sat_level"],
            svd_denoise=params["svd_on"], svd_k=params["svd_k"],
            ft_region=params["ft_region"], ft_width_mm=params["ft_width"],
            zscan=zscan, filename=self.edit_filename.text().strip() or "kspace",
            z_n_positions=len(z_targets), z_positions_mm=list(z_targets),
            z_zones=[dict(start_mm=zr["z0"].value(), stop_mm=zr["z1"].value(),
                          step_um=zr["step"].value())
                     for zr in self.zone_rows if zr["chk"].isChecked()],
            ascan=ascan, a_n_positions=len(a_targets), a_positions_deg=list(a_targets),
            a_zones=[dict(start_deg=ar["a0"].value(), stop_deg=ar["a1"].value(),
                          step_deg=ar["step"].value())
                     for ar in self.angle_rows if ar["chk"].isChecked()],
        )
        # Each Acquire = one experiment "run": save ALL its files into a folder
        # named <run-timestamp>.<filename> under the camera folder.
        camera_folder = (self.save_dir_provider() if self.save_dir_provider
                         else None) or self.save_dir
        # --- Disk-space check: estimate the data size and warn if the save volume
        # is low (do this BEFORE freezing the UI / starting the thread). ---
        frame = self.frame_source()
        if roi is not None:
            hh, ww = roi[1] - roi[0], roi[3] - roi[2]
        elif frame is not None:
            hh, ww = int(frame.shape[0]), int(frame.shape[1])
        else:
            hh = ww = 0
        binf = max(1, params["bin"])
        hb, wb = max(1, hh // binf), max(1, ww // binf)
        n_freq_est = resolve_n_points(params["n"], manual=params["nfreq"])
        grid_total = ((len(z_targets) if zscan else 1) * (len(a_targets) if ascan else 1))
        # per cube = raw (n_pos planes) + spectrum (n_freq planes), float32. Files
        # are saved UNCOMPRESSED, so this matches actual on-disk size closely.
        est_gb = grid_total * (params["n"] + n_freq_est) * hb * wb * 4 / 1e9
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
        self._low_disk_warned = False   # mid-scan low-disk warning fires once per run
        self._save_fname = self.edit_filename.text().strip() or "kspace"
        self._run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_folder = os.path.join(camera_folder, f"{self._run_stamp}.{self._save_fname}")
        self._save_folder = self._run_folder   # per-position saves go here
        self._cam_meta = (self.meta_provider() or {}) if self.meta_provider else {}
        self._save_raw_flag = self.chk_save_raw.isChecked()
        self._per_position_saved = False
        self._abort = False
        self._paused = False
        self.raw_cubes, self.raw_positions = [], []
        self.btn_run.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(True)
        self.sp.freeze(True)
        grid_total = (len(z_targets) if zscan else 1) * (len(a_targets) if ascan else 1)
        self.progress.setMaximum(grid_total)
        self.progress.setValue(0)
        # Live interferogram preview (centerburst forming as the wedge scans).
        if self.live_monitor is None:
            self.live_monitor = LiveInterferogram()
        self.live_monitor.reset()
        self.live_monitor.show(); self.live_monitor.raise_()
        # Keep the handle so shutdown() can abort + join this scan before the
        # stages are disconnected (else close-during-scan races the stage DLLs).
        self._scan_thread = threading.Thread(target=self._worker, args=(params,), daemon=True)
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
        self._paused = False   # release the worker if it's parked in a pause
        self.sig_status.emit("stopping...")

    def _toggle_pause(self) -> None:
        """Pause/resume the running scan. The worker parks at the next loop
        boundary (finishing the in-flight cube / DFT first), so the stages are
        never stopped mid-move. Stop still works while paused."""
        self._paused = not self._paused
        self.btn_pause.setText("Resume" if self._paused else "Pause")
        self.sig_status.emit("paused" if self._paused else "resuming...")

    def _wait_if_paused(self) -> None:
        """Block the worker thread while paused (polling so Stop still aborts)."""
        while self._paused and not self._abort:
            time.sleep(0.1)

    def _recompute(self) -> None:
        """Re-run the DFT on the last scan's RAW interferogram with the current
        settings (FT region, apodization, λ, denoise) -- no re-scan. Lets you
        compare e.g. center vs tails on already-acquired data."""
        if not getattr(self, "raw_cubes", None):
            self.lbl_status.setText("no raw data to recompute -- run a scan first")
            return
        p = dict(
            wl0=self.spin_wl0.value(), wl1=self.spin_wl1.value(),
            nfreq=self.spin_nfreq.value(), apod=self.spin_apod.value(),
            apod_type=self.combo_apod.currentText(),
            walkoff=(dict(rate_y=self.spin_wo_y.value(), rate_x=self.spin_wo_x.value())
                     if self.chk_walkoff.isChecked() else None),
            sat_on=self.chk_sat.isChecked(), sat_level=self.spin_sat.value(),
            svd_on=self.chk_svd.isChecked(), svd_k=self.spin_svd_k.value(),
            ft_region=self.combo_ftregion.currentText(), ft_width=self.spin_ftwidth.value(),
        )
        # Keep the saved metadata in step with what was recomputed.
        self._scan_meta.update(
            ft_region=p["ft_region"], ft_width_mm=p["ft_width"],
            apodization=p["apod_type"], apod_width=p["apod"],
            wl_start_um=p["wl0"], wl_stop_um=p["wl1"], n_freq_setting=p["nfreq"],
            svd_denoise=p["svd_on"], svd_k=p["svd_k"], recomputed=True)
        self._per_position_saved = False   # recompute result is saved as one set
        self.btn_run.setEnabled(False)
        self.btn_recompute.setEnabled(False)
        self.lbl_status.setText(f"recomputing from raw (FT={p['ft_region']})...")
        threading.Thread(target=self._recompute_worker, args=(p,), daemon=True).start()

    def _recompute_worker(self, p: dict) -> None:
        try:
            from instruments.analysis import saturation_mask, svd_denoise
            proc = HyperspectralProcessor()
            cubes, masks, wls = [], [], None
            for positions, datacube in zip(self.raw_positions, self.raw_cubes):
                datacube = np.asarray(datacube)
                sat_mask = None
                if p["sat_on"]:
                    sat_src = datacube
                    if self.background_subtracted and self.background_map is not None:
                        sat_src = datacube + self.background_map[None, :, :]
                    sat_mask = saturation_mask(sat_src, p["sat_level"])
                n_freq = resolve_n_points(len(positions), manual=p["nfreq"])
                wl, cube = proc.compute_hyperspectral(
                    positions, datacube, wl_start=p["wl0"], wl_stop=p["wl1"],
                    apod_width=p["apod"], n_freq=n_freq,
                    expected_zero_mm=DEFAULT_ZPD_MM, search_mm=DEFAULT_ZPD_WINDOW_MM,
                    apod_type=p["apod_type"], walkoff=p["walkoff"],
                    ft_region=p["ft_region"], ft_width_mm=p["ft_width"])
                if cube is None:
                    continue
                if p["svd_on"]:
                    cube = svd_denoise(cube, p["svd_k"])
                wls = wl
                cubes.append(cube)
                masks.append(sat_mask)
            self.sig_done.emit(wls, cubes, self.z_values, masks)
        except Exception as e:  # noqa: BLE001
            self.sig_status.emit(f"recompute error: {e}")
            self.sig_done.emit(None, None, None, None)

    def _position_calibration(self, positions):
        """(calibrated_axis, info) for a raw measured wedge axis -- the SAME
        motor-nonlinearity correction compute_hyperspectral applies internally,
        recomputed here so the calibrated axis can be saved alongside the raw one.
        `info` records whether it was applied, the cal file, and the peak shift."""
        info = {"motor_calibration_applied": False, "motor_calibration_file": None,
                "position_axis_used": "raw_measured"}
        if positions is None:
            return None, info
        raw = np.asarray(positions, dtype=float)
        try:
            from instruments.calibration import (calibrate_position_axis,
                                                  position_calibration_status)
            avail, fname = position_calibration_status()
            cal = np.asarray(calibrate_position_axis(raw), dtype=float)
            info["motor_calibration_applied"] = bool(avail)
            info["motor_calibration_file"] = fname
            # Which axis the saved spectra were actually computed on. The DFT
            # runs on the calibrated axis whenever the cal file is available.
            info["position_axis_used"] = "calibrated" if avail else "raw_measured"
            if avail and cal.shape == raw.shape:
                info["motor_calibration_max_dev_um"] = float(
                    np.max(np.abs(cal - raw)) * 1e3)
            return cal, info
        except Exception as e:  # noqa: BLE001
            info["motor_calibration_error"] = str(e)
            return raw, info

    def _save_position_npz(self, wl, cube, sat_mask, z_value, z_index,
                           datacube=None, positions=None, compress=False,
                           a_value=None, a_index=None) -> str:
        """Save ONE stage position to its own .npz in the run folder.

        Called twice per position in a Z-scan: first in the ACQUIRE phase with
        cube=None -- a fast raw-interferogram-only safety save so no acquired step
        is ever lost -- then in the TRANSFORM phase with the computed spectrum,
        OVERWRITING the same file. The path is deterministic in (run stamp,
        filename, z) so the second call upgrades the first. Thread-safe (uses
        values snapshotted in _start, no GUI access).

        Saved UNCOMPRESSED by default (compress=False): the spectrum cube barely
        compresses (~0.9) yet zlib DEFLATE dominates the per-position wall-clock
        (~90 s vs ~2 s to write). Disk headroom is guarded by the low-disk checks
        in _start / the acquire loop instead."""
        folder = getattr(self, "_save_folder", None) or self.save_dir
        fname = getattr(self, "_save_fname", None) or "kspace"
        os.makedirs(folder, exist_ok=True)
        stamp = getattr(self, "_run_stamp", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
        # Filename tag: include whichever axes were scanned so grid points are
        # uniquely named, e.g. ..._z1.5000mm_a45.000deg.npz. Falls back to a
        # running index when neither axis has a value (shouldn't happen in a scan).
        tag = ""
        if z_value is not None:
            tag += f"_z{z_value:.4f}mm"
        if a_value is not None:
            tag += f"_a{a_value:.3f}deg"
        if not tag:
            tag = f"_p{z_index:03d}"
        stem = os.path.join(folder, f"{stamp}.{fname}{tag}")
        has_spectrum = cube is not None and wl is not None
        cal_positions, cal_info = self._position_calibration(positions)
        meta = dict(self._scan_meta or {})
        meta.update(z_index=int(z_index),
                    z_value_mm=(None if z_value is None else float(z_value)),
                    angle_index=(None if a_index is None else int(a_index)),
                    angle_value_deg=(None if a_value is None else float(a_value)),
                    saved_local_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if has_spectrum:
            cube = np.asarray(cube)
            meta.update(processing_stage="complete",
                        cube_shape=list(cube.shape), n_freq=int(cube.shape[0]),
                        wl_min_um=float(np.min(wl)), wl_max_um=float(np.max(wl)))
        else:
            # FFT not done yet -- this is the per-step safety save (raw only).
            meta["processing_stage"] = "raw_acquired"
        meta.update(cal_info)
        if getattr(self, "_cam_meta", None):
            meta["camera"] = self._cam_meta
        kw = dict(
            z_value_mm=(np.nan if z_value is None else float(z_value)),
            z_unit="mm",
            angle_value_deg=(np.nan if a_value is None else float(a_value)),
            angle_unit="deg",
            metadata=np.array(meta, dtype=object),
            metadata_json=json.dumps(meta, default=str, indent=2))
        if has_spectrum:
            kw["wavelengths"] = wl
            kw["spectrum_cube"] = cube
        if sat_mask is not None:
            kw["saturation_mask"] = np.asarray(sat_mask, bool)
        if getattr(self, "background_map", None) is not None:
            kw["background"] = np.asarray(self.background_map)
            kw["background_subtracted"] = bool(self.background_subtracted)
        # ALWAYS store the raw interferogram cube (n_pos, h, w) -- this IS the
        # per-pixel / per-wedge-step interferogram -- and the REAL measured TWINS
        # positions, so every position's full data is preserved for reprocessing.
        if datacube is not None:
            kw["raw_interferogram"] = np.asarray(datacube)
            kw["twins_positions_mm"] = np.asarray(positions)  # real measured wedge positions
            # The motor-nonlinearity-corrected axis the DFT was actually computed
            # on (parameters_int.txt). Equals twins_positions_mm if no cal file.
            if cal_positions is not None:
                kw["twins_positions_calibrated_mm"] = np.asarray(cal_positions)
        (np.savez_compressed if compress else np.savez)(stem + ".npz", **kw)
        return stem + ".npz"

    def _worker(self, p: dict) -> None:
        try:
            from instruments.analysis import saturation_mask, svd_denoise
            from instruments.subtwinslv import bin_image
            scanner = TwinsScanner(self.sp.twins, self.frame_source)
            proc = HyperspectralProcessor()
            cubes, zvals, masks, wls = [], [], [], None
            raw_cubes, raw_positions = [], []
            # Background: subtract from each frame (if enabled) and record the
            # binned-ROI background in the cube geometry for saving.
            bg = p["background"]
            bg_for_scan = bg if p["bg_subtract"] else None
            self.background_subtracted = bool(p["bg_subtract"] and bg is not None)
            if bg is not None:
                _roi = p["roi"]
                _crop = bg if _roi is None else bg[_roi[0]:_roi[1], _roi[2]:_roi[3]]
                self.background_map = bin_image(np.asarray(_crop, dtype=np.float32), p["bin"])
            else:
                self.background_map = None
            # Scan axes. [None] means "that axis is not scanned" -> a single
            # position. The full acquisition is the NESTED grid: outer Z, inner
            # angle (for each Z, sweep every angle).
            z_targets = ([None] if not p["zscan"]
                         else [float(z) for z in p["z_targets"]])
            a_targets = ([None] if not p["ascan"]
                         else [float(a) for a in p["a_targets"]])
            is_scan = p["zscan"] or p["ascan"]
            total = len(z_targets) * len(a_targets)
            rotator = getattr(self.sp, "rotator", None)

            def grid_label(z, a):
                parts = []
                if z is not None:
                    parts.append(f"z={z:.4f}mm")
                if a is not None:
                    parts.append(f"a={a:.3f}deg")
                return " ".join(parts)

            if p["zscan"] and not getattr(self.sp.delay, "is_homed", True):
                self.sig_status.emit("homing delay stage...")
                self.sp.delay.home()
            if p["ascan"] and rotator is not None and not getattr(rotator, "is_homed", True):
                self.sig_status.emit("homing rotator...")
                rotator.home()

            # ---- Phase 1: ACQUIRE every (Z, angle) grid point's raw
            # interferogram first. No per-pixel DFT here -- the stages march
            # through the whole grid, then everything is transformed (Phase 2).
            acquired = []   # [{z, zi, a, ai, gi, positions, datacube}, ...]
            gi = 0
            for zi, z in enumerate(z_targets):
                self._wait_if_paused()   # park before advancing to the next Z
                if self._abort:
                    break
                if z is not None:
                    self.sig_status.emit(f"moving Z to {z:.4f} mm")
                    self.sp.delay.move_to_mm(float(z))  # wait=True blocks until done
                for ai, a in enumerate(a_targets):
                    self._wait_if_paused()   # park here between cubes if paused
                    if self._abort:
                        break
                    g, label = gi, grid_label(z, a)
                    gi += 1
                    if a is not None:
                        self.sig_status.emit(
                            f"pt {g+1}/{total} {label}: rotating to {a:.3f} deg")
                        rotator.move_to_deg(float(a))  # wait=True blocks until done

                    def prog(i, tot, pos, value, _g=g, _l=label, _t=total):
                        self.sig_status.emit(
                            f"pt {_g+1}/{_t} {_l}: cube point {i}/{tot}")
                        self.sig_point.emit(float(pos), float(value), i == 1)

                    positions, datacube = scanner.scan_cube(
                        p["start"], p["stop"], p["n"], p["roi"],
                        frames_avg=p["frames"], bin_factor=p["bin"], background=bg_for_scan,
                        progress=prog, should_abort=lambda: self._abort,
                        status_cb=lambda m: self.sig_status.emit(m))

                    if datacube is None or len(positions) < 3:
                        continue
                    # Keep the last raw cube so walk-off can be calibrated from it
                    # later (use a sharp-sample scan).
                    self._last_datacube = datacube
                    self._last_positions = positions
                    acquired.append({"z": z, "zi": zi, "a": a, "ai": ai, "gi": g,
                                     "positions": np.asarray(positions),
                                     "datacube": np.asarray(datacube)})
                    # Disk-space guard: warn once when low, ABORT before the volume
                    # fills so we don't write corrupt / truncated files.
                    free = _free_gb(self._save_folder)
                    if free is not None and free < LOW_DISK_ABORT_GB:
                        self.sig_warn.emit(
                            "Disk full — acquisition stopped",
                            f"Only {free:.2f} GB free on the save drive. Stopping to "
                            f"avoid corrupt saves. {g} of {total} points were acquired.")
                        self.sig_status.emit(f"ABORTED: disk full ({free:.2f} GB free)")
                        self._abort = True
                        break
                    if (free is not None and free < LOW_DISK_WARN_GB
                            and not self._low_disk_warned):
                        self._low_disk_warned = True
                        self.sig_warn.emit(
                            "Low disk space",
                            f"Only {free:.1f} GB free on the save drive — free up space "
                            f"or this acquisition may fail to finish saving.")
                    # Safety save: write THIS grid point's raw interferogram
                    # immediately (uncompressed = fast), so a later crash never
                    # loses acquired data. Phase 2 overwrites it with the spectrum.
                    if is_scan:
                        try:
                            self._save_position_npz(None, None, None, z, zi,
                                                    datacube, positions, compress=False,
                                                    a_value=a, a_index=ai)
                            self._per_position_saved = True
                            self.sig_status.emit(f"pt {g+1}/{total} {label}: raw saved")
                        except Exception as e:  # noqa: BLE001
                            self.sig_status.emit(f"pt {g+1}: RAW SAVE FAILED: {e}")

            # ---- Phase 2: now transform every acquired cube (per-pixel DFT) + save.
            n_acq = len(acquired)
            if n_acq and not self._abort:
                self.sig_status.emit(
                    f"acquisition done ({n_acq} cube{'s' if n_acq != 1 else ''}); "
                    f"computing per-pixel DFTs...")
            for k, item in enumerate(acquired):
                self._wait_if_paused()   # park between DFTs if paused
                if self._abort:
                    break
                z, zi = item["z"], item["zi"]
                a, ai = item["a"], item["ai"]
                positions, datacube = item["positions"], item["datacube"]
                # Saturation mask from the RAW counts: if a background was
                # subtracted, add it back so a clipped pixel still reads >= the
                # saturation level (otherwise subtraction would mask it).
                sat_mask = None
                if p["sat_on"]:
                    sat_src = datacube
                    if self.background_subtracted and self.background_map is not None:
                        sat_src = datacube + self.background_map[None, :, :]
                    sat_mask = saturation_mask(sat_src, p["sat_level"])
                # "Auto" (nfreq == 0) -> ZEROFILL_FACTOR x steps, clamped.
                n_freq = resolve_n_points(len(positions), manual=p["nfreq"])
                self.sig_status.emit(
                    f"FFT {k+1}/{n_acq}: per-pixel DFT ({n_freq} bins)...")
                wl, cube = proc.compute_hyperspectral(
                    positions, datacube, wl_start=p["wl0"], wl_stop=p["wl1"],
                    apod_width=p["apod"], n_freq=n_freq,
                    expected_zero_mm=DEFAULT_ZPD_MM, search_mm=DEFAULT_ZPD_WINDOW_MM,
                    apod_type=p["apod_type"], walkoff=p["walkoff"],
                    ft_region=p["ft_region"], ft_width_mm=p["ft_width"])
                if cube is None:
                    continue
                if p["svd_on"]:
                    self.sig_status.emit(f"FFT {k+1}/{n_acq}: SVD denoise (k={p['svd_k']})...")
                    cube = svd_denoise(cube, p["svd_k"])
                wls = wl
                cubes.append(cube)
                # Viewer slider value: the Z position, or the angle for an
                # angle-only scan (per-file metadata records both exactly).
                zvals.append(z if z is not None else a)
                masks.append(sat_mask)
                raw_cubes.append(datacube)
                raw_positions.append(positions)
                # Full experiment: save THIS grid point's hypercube.
                if is_scan:
                    try:
                        path = self._save_position_npz(
                            wl, cube, sat_mask, z, zi, datacube, positions,
                            a_value=a, a_index=ai)
                        self._per_position_saved = True
                        self.sig_status.emit(
                            f"FFT {k+1}/{n_acq}: saved {os.path.basename(path)}")
                    except Exception as e:  # noqa: BLE001
                        self.sig_status.emit(f"pt {k+1}: SAVE FAILED: {e}")
            self.raw_cubes = raw_cubes
            self.raw_positions = raw_positions
            self.sig_done.emit(wls, cubes, zvals, masks)
        except Exception as e:  # noqa: BLE001
            self.sig_status.emit(f"error: {e}")
            self.sig_done.emit(None, None, None, None)

    # -- signal slots --------------------------------------------------------
    @QtCore.pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)
        # Acquisition progress: "pt <i>/<total> ..." marks each grid point.
        if msg.startswith("pt "):
            try:
                cur = int(msg.split()[1].split("/")[0])
                self.progress.setValue(cur)
            except Exception:  # noqa: BLE001
                pass

    @QtCore.pyqtSlot(str, str)
    def _on_warn(self, title: str, message: str) -> None:
        """Modal warning raised from the scan thread (via sig_warn)."""
        QMessageBox.warning(self, title, message)

    @QtCore.pyqtSlot(object, object, object, object)
    def _on_done(self, wavelengths, cubes, z_values, sat_masks) -> None:
        self.btn_run.setEnabled(True)
        self.btn_recompute.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self._paused = False
        self.btn_stop.setEnabled(False)
        self.sp.freeze(False)
        if not cubes:
            self.lbl_status.setText("no data (aborted or scan failed)")
            return
        self.wavelengths, self.cubes, self.z_values = wavelengths, cubes, z_values
        self.sat_masks = sat_masks or [None] * len(cubes)
        nsat = sum(int(m.sum()) for m in self.sat_masks if m is not None)
        sat_txt = f", {nsat} saturated px" if nsat else ""
        msg = f"done: {len(cubes)} map(s), cube {cubes[0].shape} (λ×h×w){sat_txt}"
        if getattr(self, "_per_position_saved", False):
            # Z-scan: each position was already saved individually as it ran.
            msg += f"  |  {len(cubes)} position(s) auto-saved individually"
        else:
            # Single map: auto-save the hypercube (+ metadata) so no scan is lost.
            try:
                path = self._autosave_cube()
                msg += f"  |  auto-saved {os.path.basename(path)}"
            except Exception as e:  # noqa: BLE001
                msg += f"  |  AUTO-SAVE FAILED: {e}"
        self.lbl_status.setText(msg)
        self._open_viewer()

    def _run_target(self):
        """(folder, stamp, fname) for the current experiment's files. Creates the
        <run-stamp>.<filename> run folder under the camera folder. Falls back to a
        fresh run folder if there is no active scan (e.g. a manual save after Load)."""
        folder = getattr(self, "_run_folder", None)
        stamp = getattr(self, "_run_stamp", None)
        fname = getattr(self, "_save_fname", None) or self.edit_filename.text().strip() or "kspace"
        if not folder or not stamp:
            base = (self.save_dir_provider() if self.save_dir_provider else None) or self.save_dir
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(base, f"{stamp}.{fname}")
        os.makedirs(folder, exist_ok=True)
        return folder, stamp, fname

    def _autosave_cube(self) -> str:
        """Write the full hypercube npz into the run folder, named
        <run-stamp>.<filename>.npz -- called automatically after every scan."""
        folder, stamp, fname = self._run_target()
        return self._save_cube_npz(os.path.join(folder, f"{stamp}.{fname}"))

    def _open_viewer(self) -> None:
        if not self.cubes:
            self.lbl_status.setText("nothing to view yet")
            return
        if self.viewer is None:
            self.viewer = HyperViewer()
        self.viewer.set_result(self.wavelengths, self.cubes, self.z_values,
                               getattr(self, "sat_masks", None))
        try:
            self.viewer.set_calibration_note(self._build_metadata())
        except Exception:  # noqa: BLE001
            pass
        self.viewer.show()
        self.viewer.raise_()

    def _roi_average(self):
        """Spatial mean & std spectrum over each cube (the acquired ROI),
        excluding saturated pixels. Returns (wavelengths, means, stds) with
        means/stds shaped (n_z, n_freq)."""
        from instruments.analysis import roi_average
        wl = np.asarray(self.wavelengths)
        masks = getattr(self, "sat_masks", None) or [None] * len(self.cubes)
        ms = [roi_average(c, masks[i] if i < len(masks) else None)
              for i, c in enumerate(self.cubes)]
        means = np.array([m[0] for m in ms])
        stds = np.array([m[1] for m in ms])
        return wl, means, stds

    def _save_roi_average_csv(self, stem: str) -> str:
        """Write the ROI-average spectrum/spectra as a CSV: wavelength_um, then a
        mean & std column per z map."""
        wl, means, stds = self._roi_average()
        cols = [wl]
        header = ["wavelength_um"]
        for zi, z in enumerate(self.z_values):
            tag = "" if z is None else f"_z{z:.4f}mm"
            cols += [means[zi], stds[zi]]
            header += [f"mean{tag}", f"std{tag}"]
        np.savetxt(stem + "_roi_avg.csv", np.column_stack(cols),
                   delimiter=",", header=",".join(header), comments="")
        return stem + "_roi_avg.csv"

    def _build_metadata(self) -> dict:
        """Scan parameters + camera state describing the saved hypercube."""
        meta = dict(self._scan_meta or {})
        if self.cubes:
            c0 = np.asarray(self.cubes[0])
            meta["cube_shape"] = list(c0.shape)        # (n_freq, h, w)
            meta["n_freq"] = int(c0.shape[0])
            meta["n_maps"] = len(self.cubes)
        if self.wavelengths is not None and len(self.wavelengths):
            meta["wl_min_um"] = float(np.min(self.wavelengths))
            meta["wl_max_um"] = float(np.max(self.wavelengths))
        meta["z_values_mm"] = [None if z is None else float(z) for z in self.z_values]
        meta["saturated_px"] = sum(int(m.sum()) for m in (self.sat_masks or [])
                                   if m is not None)
        # Whether the spectra were computed on the motor-nonlinearity-corrected
        # wedge axis (parameters_int.txt), with the peak shift it introduced.
        rawpos0 = (self.raw_positions[0] if getattr(self, "raw_positions", None)
                   else None)
        _, cal_info = self._position_calibration(rawpos0)
        meta.update(cal_info)
        meta["saved_local_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.meta_provider:
            try:
                meta["camera"] = self.meta_provider() or {}
            except Exception:  # noqa: BLE001
                pass
        return meta

    def _save_cube_npz(self, stem: str) -> str:
        roi = getattr(self, "_scan_roi", None)
        meta = self._build_metadata()
        kw = dict(
            wavelengths=self.wavelengths,
            spectrum_cubes=np.asarray(self.cubes),
            z_values=np.asarray([np.nan if z is None else z for z in self.z_values]),
            z_unit="mm",
            roi=np.asarray(roi if roi is not None else [], dtype=float),
            binning=int(getattr(self, "_scan_bin", 1)),
            metadata=np.array(meta, dtype=object),          # dict (load allow_pickle=True)
            metadata_json=json.dumps(meta, default=str, indent=2))  # portable, human-readable
        masks = getattr(self, "sat_masks", None)
        if masks and any(m is not None for m in masks):
            h, w = np.asarray(self.cubes[0]).shape[1:]
            kw["saturation_masks"] = np.array([
                (np.zeros((h, w), bool) if m is None else np.asarray(m, bool))
                for m in masks])
        # The captured background (binned-ROI geometry, aligns with the cube) +
        # whether it was subtracted from the interferogram.
        if getattr(self, "background_map", None) is not None:
            kw["background"] = np.asarray(self.background_map)
            kw["background_subtracted"] = bool(self.background_subtracted)
        # ALWAYS store the raw interferogram cube(s) + positions, so the file can
        # be reprocessed offline (different apodization / ZPD / FT window) without
        # re-scanning. The raw is small vs the spectrum (n_pos << n_freq).
        if getattr(self, "raw_cubes", None):
            try:
                kw["raw_interferograms"] = np.asarray(self.raw_cubes)
                kw["raw_positions"] = np.asarray(self.raw_positions)
                # The calibrated (motor-corrected) axis per z, as actually used.
                kw["raw_positions_calibrated"] = np.asarray(
                    [self._position_calibration(p)[0] for p in self.raw_positions])
            except Exception:  # noqa: BLE001 (ragged per-z shapes -> skip raw)
                pass
        np.savez(stem + ".npz", **kw)  # uncompressed: fast write, cube barely compresses
        return stem + ".npz"

    def _save(self) -> None:
        if not self.cubes:
            self.lbl_status.setText("nothing to save")
            return
        mode = self.combo_save.currentText()
        want_cube = mode.startswith("Full cube") or mode == "Both"
        want_roi = mode.startswith("ROI average") or mode == "Both"
        # Save into the run folder (<run-stamp>.<filename>/) so all of this
        # experiment's files stay together.
        try:
            folder, stamp, fname = self._run_target()
            stem = os.path.join(folder, f"{stamp}.{fname}")
            saved = []
            if want_cube:
                saved.append(os.path.basename(self._save_cube_npz(stem)))
            if want_roi:
                saved.append(os.path.basename(self._save_roi_average_csv(stem)))
            self.lbl_status.setText(f"saved {', '.join(saved)} to {folder}")
        except Exception as e:  # noqa: BLE001
            self.lbl_status.setText(f"save error: {e}")

    def _load(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        folder = (self.save_dir_provider() if self.save_dir_provider
                  else None) or self.save_dir
        start = folder if os.path.isdir(folder) else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load K-space measurement", start, "NumPy archive (*.npz)")
        if not path:
            return
        try:
            res = load_kspace_npz(path)
            if res is None:
                self.lbl_status.setText("file has no spectral cube")
                return
            wl, cubes, z_values, masks = res
            self.wavelengths, self.cubes, self.z_values = wl, cubes, z_values
            self.sat_masks = masks or [None] * len(cubes)
            self.raw_cubes, self.raw_positions = [], []  # not reloaded for viewing
            self.lbl_status.setText(
                f"loaded {os.path.basename(path)}: {len(cubes)} map(s), cube {cubes[0].shape}")
            self._open_viewer()
        except Exception as e:  # noqa: BLE001
            self.lbl_status.setText(f"load error: {e}")

    def shutdown(self) -> None:
        # Abort a running acquisition and wait for its worker thread to unwind
        # BEFORE the caller (MainWindow.closeEvent) disconnects the stages -- the
        # scan thread drives the TWINS/Thorlabs stages, so releasing the DLLs out
        # from under an in-flight move would race. should_abort=lambda: self._abort
        # is polled per step, so the worker stops at the next step boundary.
        self._abort = True
        t = getattr(self, "_scan_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=10.0)
        if self.viewer is not None:
            self.viewer.close()
        if self.live_monitor is not None:
            self.live_monitor.close()
