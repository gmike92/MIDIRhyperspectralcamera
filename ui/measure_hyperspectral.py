"""Hyperspectral viewer + shared scan widgets.

HyperViewer shows a saved per-pixel TWINS spectral cube: scrub wavelength to see
the spatial map at each λ and click any pixel to see its spectrum
(view_hyperspectral.py opens one from an .npz). LiveInterferogram is the pop-out
ROI-mean-vs-wedge-position plot the Measure tab updates while a scan runs.

The hyperspectral *acquisition* panel (per-pixel DFT) is
not part of this branch: the Measure tab is ui/measure_extended.py.
"""
from __future__ import annotations

import os
import shutil
import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget,
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
        self.setWindowTitle("Hyperspectral Viewer")
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

def load_hyperspectral_npz(path: str):
    """Read a saved hyperspectral .npz -> (wavelengths, cubes, z_values, sat_masks).

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

def hyperspectral_metadata(path: str) -> dict:
    """The embedded metadata dict of a saved hyperspectral .npz (or {}). Lets a viewer
    show whether the spectra were computed on the calibrated wedge axis."""
    try:
        with np.load(path, allow_pickle=True) as d:
            if "metadata" in d.files:
                return dict(d["metadata"].item())
    except Exception:  # noqa: BLE001
        pass
    return {}
