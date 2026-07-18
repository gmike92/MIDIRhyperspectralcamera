"""
Hyperspectral Z-series analysis app (standalone).

A richer analyzer for our per-Z hypercubes, in the spirit of the hsiAnalysis
HyperspectralAnalysisApp (PoliMi) -- reimplemented in Python for our .npz data.
Loads the per-Z files saved at each Thorlabs position as one Z-series and lets
you:

  - scrub Z (mm) and wavelength (µm); pick the spatial map (wavelength slice,
    peak-λ, peak-intensity, SAM, or false-colour RGB)
  - draw rectangular ROIs and read their average spectra (mean ± std), overlaid
  - click a pixel for its spectrum
  - normalise spectra: raw / area / reflectivity (÷ a reference ROI)
  - SVD-denoise the current cube; choose the colormap and gamma
  - read each position's full metadata
  - export ROI/pixel spectra (current Z, or vs Z), and the current map image

Per-Z files are big, so cubes load LAZILY -- only the current position is in RAM
(float32). Run:  .venv\\Scripts\\python analysis_app.py [folder | files...]
"""
import datetime
import glob
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

# Interpret image arrays as (row, col) -- same as the acquisition app
# (main_window.py sets this too). This makes the analyzer show cubes the SAME way
# up as the live camera view, and makes the ROI/click x=col,y=row mapping correct.
# Must run before any ImageView/ImageItem is created.
pg.setConfigOptions(imageAxisOrder="row-major")

from instruments import analysis as A
from instruments.hyperspectral import (
    HyperspectralProcessor, resolve_n_points, DEFAULT_ZPD_MM, DEFAULT_ZPD_WINDOW_MM)

# Stokes polarimetry panel (embedded as a toolbar-toggled page). Guarded so the
# analyzer still starts if stokes_app.py is ever absent -- the Stokes button is
# just omitted in that case.
try:
    from stokes_app import StokesApp
except Exception:  # noqa: BLE001
    StokesApp = None


# ---------------------------------------------------------------------------
# File helpers (read metadata cheaply; load cube on demand)
# ---------------------------------------------------------------------------
def _npz_info(path):
    info = {"path": path, "z": float("nan"), "angle": float("nan"),
            "wavelengths": None, "metadata": {}, "mask": None, "has_cube": False}
    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)
        info["has_cube"] = "spectrum_cube" in files
        if "wavelengths" in files:
            info["wavelengths"] = np.asarray(d["wavelengths"])
        for key, npz_key in (("z", "z_value_mm"), ("angle", "angle_value_deg")):
            if npz_key in files:
                try:
                    info[key] = float(d[npz_key])
                except Exception:  # noqa: BLE001
                    pass
        if "metadata" in files:
            try:
                info["metadata"] = d["metadata"].item()
            except Exception:  # noqa: BLE001
                pass
        # fall back to metadata for z / angle if the top-level keys were absent
        if np.isnan(info["z"]):
            info["z"] = float(info["metadata"].get("z_value_mm", float("nan")) or float("nan"))
        if np.isnan(info["angle"]):
            info["angle"] = float(info["metadata"].get("angle_value_deg", float("nan")) or float("nan"))
        if "saturation_mask" in files:
            info["mask"] = np.asarray(d["saturation_mask"])
    info["map_index"] = None
    return info


def _infos_from_file(path):
    """One or more Z-entry infos from a saved .npz, supporting BOTH layouts:
    per-position files (`spectrum_cube`, one map) and whole-cube files
    (`spectrum_cubes` stacked over z -- written by the single-map / Z-set
    autosave & manual Save). Whole-cube files expand to one info per map, each
    tagged with a `map_index`. Returns a list."""
    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)
        if "spectrum_cubes" in files:
            wl = np.asarray(d["wavelengths"]) if "wavelengths" in files else None
            meta = {}
            if "metadata" in files:
                try:
                    meta = dict(d["metadata"].item())
                except Exception:  # noqa: BLE001
                    meta = {}
            zvals = (np.asarray(d["z_values"]).ravel() if "z_values" in files
                     else np.array([np.nan]))
            masks = d["saturation_masks"] if "saturation_masks" in files else None
            out = []
            for i in range(len(zvals)):
                z = float(zvals[i])
                out.append({"path": path, "map_index": i, "z": z,
                            "angle": float("nan"),   # stacked cubes carry no angle
                            "wavelengths": wl, "metadata": meta, "has_cube": True,
                            "mask": (np.asarray(masks[i]) if masks is not None
                                     and i < len(masks) else None)})
            return out
    return [_npz_info(path)]   # per-position (single-cube) file


def _load_cube(path, map_index=None):
    """Load ONE spectral cube (n_freq, h, w). Supports per-position files
    (`spectrum_cube`) and whole-cube files (`spectrum_cubes` stacked over z,
    selected by `map_index`)."""
    with np.load(path, allow_pickle=True) as d:
        if "spectrum_cube" in d.files:
            return np.asarray(d["spectrum_cube"], dtype=np.float32)
        if "spectrum_cubes" in d.files:
            return np.asarray(d["spectrum_cubes"][map_index or 0], dtype=np.float32)
    return None


def _file_meta(d):
    """The metadata dict from a loaded npz (or {})."""
    if "metadata" in d.files:
        try:
            return dict(d["metadata"].item())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _read_axis(d, map_index=None):
    """(position_axis, already_calibrated) from a loaded npz, honoring metadata.

    If the file's metadata says the spectra were computed on the calibrated
    wedge axis AND the calibrated axis is stored, return THAT axis flagged
    calibrated=True (so recompute uses the exact same axis and does NOT correct
    it again). Otherwise return the raw measured axis flagged False so compute
    applies the current motor calibration. Supports per-position (singular keys)
    and whole-cube (plural keys, indexed by `map_index`) layouts.
    """
    files = set(d.files)
    applied = bool(_file_meta(d).get("motor_calibration_applied", False))
    mi = map_index or 0
    if applied:
        if "twins_positions_calibrated_mm" in files:
            return np.asarray(d["twins_positions_calibrated_mm"], dtype=float), True
        if "raw_positions_calibrated" in files:
            return np.asarray(d["raw_positions_calibrated"][mi], dtype=float), True
    if "twins_positions_mm" in files:
        return np.asarray(d["twins_positions_mm"], dtype=float), False
    if "raw_positions" in files:
        return np.asarray(d["raw_positions"][mi], dtype=float), False
    return None, False


def _read_raw(d, map_index=None):
    """(raw_interferogram, position_axis, already_calibrated) from a loaded npz,
    supporting per-position (`raw_interferogram`) and whole-cube
    (`raw_interferograms` stacked, indexed by `map_index`) layouts. Returns a
    None raw if the file has no stored interferogram."""
    files = set(d.files)
    raw = None
    if "raw_interferogram" in files:
        raw = np.asarray(d["raw_interferogram"], dtype=np.float32)
    elif "raw_interferograms" in files:
        raw = np.asarray(d["raw_interferograms"][map_index or 0], dtype=np.float32)
    pos, calibrated = _read_axis(d, map_index)
    return raw, pos, calibrated


def _get_cmap(name):
    manual = {
        "grey": ([0.0, 1.0], [(0, 0, 0), (255, 255, 255)]),
        "jet": ([0.0, 0.125, 0.375, 0.625, 0.875, 1.0],
                [(0, 0, 128), (0, 0, 255), (0, 255, 255),
                 (255, 255, 0), (255, 0, 0), (128, 0, 0)]),
    }
    key = "grey" if str(name).lower() == "gray" else str(name).lower()
    if key in manual:
        pos, cols = manual[key]
        return pg.ColorMap(pos=pos, color=cols)
    try:
        return pg.colormap.get(key)
    except Exception:  # noqa: BLE001
        return None


def _bwr_cmap():
    """Diverging blue-white-red map (blue = -π, white = 0, red = +π)."""
    return pg.ColorMap(pos=[0.0, 0.5, 1.0],
                       color=[(0, 0, 255), (255, 255, 255), (255, 0, 0)])


# Selectable colormaps for the wrapped-phase image. 'blue-white-red' is the
# default (signed, diverging); 'cyclic (CET-C1)' is topologically correct for
# wrapped phase; the rest fall through to the shared resolver / pyqtgraph.
PHASE_CMAPS = ["blue-white-red", "cyclic (CET-C1)", "turbo", "jet", "grey"]


def _phase_cmap(name):
    key = str(name).lower()
    if key in ("blue-white-red", "bwr"):
        return _bwr_cmap()
    if key.startswith("cyclic"):
        try:
            cm = pg.colormap.get("CET-C1")
            if cm is not None:
                return cm
        except Exception:  # noqa: BLE001
            pass
    cm = _get_cmap(name)
    return cm if cm is not None else _bwr_cmap()


_ROI_COLORS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#e64980"]

# LRU cube/raw caches: keep at most this many recent cubes, and never exceed this
# many total bytes (a single giant cube still loads, but old ones are evicted).
_CACHE_MAX_ITEMS = 6
_CACHE_MAX_BYTES = 1_500_000_000        # ~1.5 GB across all cached cubes
_SEL_DEBOUNCE_MS = 90                    # Z/angle: heavy (cube reload) -> longer
_WL_DEBOUNCE_MS = 25                     # λ-slice: light (cached band) -> short

# Palette for clicked-pixel spectra / interferograms (kept distinct from the ROI
# colours so pixels and regions never clash). Each selected pixel keeps the same
# colour in both the spectra plot and the interferogram plot.
_PIXEL_COLORS = ["#e8590c", "#7048e8", "#0ca678", "#f03e3e", "#1098ad",
                 "#d6336c", "#f59f00", "#4263eb", "#37b24d", "#c2255c"]


def _pixel_color(i):
    return _PIXEL_COLORS[i % len(_PIXEL_COLORS)]

# Distinct, bright categorical palette for K-means clusters -- used for BOTH the
# map and the per-cluster spectra so they match, and none is black/dark (which
# would vanish on the dark spectra plot). 16 colours = max cluster count.
_KMEANS_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
    "#fffac8", "#e377c2", "#aaffc3", "#ffe119",
]


def _mk_group(name, run):
    n = len(run); zs = [r["z"] for r in run if r["z"] == r["z"]]
    ts0 = run[0]["ts"]
    date = (f"{ts0[:4]}-{ts0[4:6]}-{ts0[6:8]} {ts0[9:11]}:{ts0[11:13]}"
            if len(ts0) >= 13 else (ts0 or "?"))
    zr = f"z {min(zs):.2f}–{max(zs):.2f} mm" if zs else "single map"
    label = f"{name or '(unnamed)'}  —  {date}, {n} file(s), {zr}"
    return (label, [r["path"] for r in run])


def _group_experiments(paths):
    """Group per-Z .npz files into experiments. Filenames are
    `<date>.<name>_z<pos>mm.npz`; group by <name>, then split a same-name group
    into separate runs whenever the z grid REPEATS (a re-run) or there is a big
    time gap between files (a separate session). Returns [(label, [paths])]."""
    recs = []
    for p in paths:
        stem = os.path.basename(p)[:-4]
        ts, dot, rest = stem.partition(".")
        if not dot:
            rest, ts = stem, ""
        # strip a trailing `_a<angle>deg` then `_z<pos>mm` tag to get the base name;
        # the angle tag (if present) sits AFTER the z tag: ..._z8.0000mm_a45.000deg
        am = re.search(r"_a(-?\d+(?:\.\d+)?)deg$", rest)
        afn = float(am.group(1)) if am else float("nan")
        base = rest[:am.start()] if am else rest
        zm = re.search(r"_z(-?\d+(?:\.\d+)?)mm$", base)
        zfn = float(zm.group(1)) if zm else float("nan")
        name = base[:zm.start()] if zm else base
        try:
            t = datetime.datetime.strptime(ts[:15], "%Y%m%d_%H%M%S")
        except Exception:  # noqa: BLE001
            t = None
        recs.append({"name": name, "ts": ts, "t": t, "z": zfn, "angle": afn, "path": p})

    byname = OrderedDict()
    for r in sorted(recs, key=lambda r: (r["name"], r["ts"])):
        byname.setdefault(r["name"], []).append(r)

    groups = []
    for name, items in byname.items():
        gaps = [(items[i]["t"] - items[i - 1]["t"]).total_seconds()
                for i in range(1, len(items)) if items[i]["t"] and items[i - 1]["t"]]
        med = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
        thresh = max(120.0, 6.0 * med)        # "big gap" = new session
        run, seen, prev = [], set(), None
        for r in items:
            big_gap = (prev is not None and r["t"] and prev["t"]
                       and (r["t"] - prev["t"]).total_seconds() > thresh)
            # split on a repeated (z, angle) grid point -> a genuine re-run
            zkey = (round(r["z"], 4) if r["z"] == r["z"] else None,
                    round(r["angle"], 4) if r["angle"] == r["angle"] else None)
            if run and ((zkey != (None, None) and zkey in seen) or big_gap):
                groups.append(_mk_group(name, run)); run, seen = [], set()
            run.append(r)
            if zkey != (None, None):
                seen.add(zkey)
            prev = r
        if run:
            groups.append(_mk_group(name, run))
    return groups


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class ZSeriesAnalyzer(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hyperspectral Z-Series Analyzer")
        self.resize(1320, 760)
        self.infos = []
        self.wavelengths = None
        self._saved_wavelengths = None  # axis of the stored spectrum cubes
        self.rois = []                 # list of pg.RectROI
        self.ref_spectrum = None       # reference spectrum (recomputed per refresh)
        self.ref_roi = None            # ROI used as the reflectivity reference
        self.sam_ref_roi = None        # ROI whose average is the SAM reference
        self.sam_ref_ext = None        # (wl, value) external SAM reference spectrum
        self.bg_roi = None             # ROI whose average spectrum is subtracted
        self._bg_cube = None           # cached background-subtracted cube
        self._bg_sig = None            # signature of (cube, bg ROI) for that cache
        self._z_axis = []              # sorted unique Z values across the loaded cubes
        self._a_axis = []              # sorted unique angle values
        self._grid = {}                # (z_index, angle_index) -> flat index into self.infos
        self._kmeans_labels = None     # (h, w) cluster labels from the last K-means map
        self._kmeans_k = 0             # number of clusters in that map
        self._cluster_means = []       # cached per-cluster mean spectra
        self._cluster_key = None       # signature for the cluster-means cache
        self._svd_cum = None           # cached cumulative explained variance
        # Bounded LRU caches (OrderedDict = insertion order): keep the last few
        # decompressed cubes / raw interferograms so revisiting a Z/angle is
        # instant instead of re-reading + decompressing a big .npz. Capped by
        # BOTH item count and total bytes so huge cubes can't blow up RAM.
        self._cube_cache = OrderedDict()   # z_index -> (wl, cube)
        self._raw_cache = OrderedDict()    # z_index -> (positions, raw_interferogram, calibrated)
        self.sel_pixels = []           # list of (row, col) clicked pixels
        self.recompute_on = False      # compute map from raw interferogram window
        self.proc = None               # HyperspectralProcessor (lazy)
        self.phase_mode = False        # True while the phase-only page is shown
        self.phase_bkg = None          # (ref_raw_cube, ref_positions, path) background
        self.phase_bkg_on = False      # subtract the background phase (per-pixel)
        self._stokes_dirty = True      # Stokes panel needs (re)fill from loaded folder
        self.work_roi = None           # pg.RectROI defining the analysis crop (or None)
        self._full_hw = None           # (h, w) of the FULL current frame (pre-crop)

        # Debounce: slider drags fire valueChanged rapidly; coalesce the heavy
        # refresh (cube reload + map/spectra/DFT) so only the settled value runs.
        self._sel_timer = QtCore.QTimer(self); self._sel_timer.setSingleShot(True)
        self._sel_timer.timeout.connect(self._do_selection_refresh)
        self._wl_timer = QtCore.QTimer(self); self._wl_timer.setSingleShot(True)
        self._wl_timer.timeout.connect(self._do_wl_refresh)

        self._build_ui()

    # -- UI -------------------------------------------------------------------
    def _build_ui(self):
        # toolbar: exactly six controls -- Load folder | Load files | [view
        # selector: Hypercube · Phase · Stokes] | Save/Export dropdown.
        tb = self.addToolBar("File")
        tb.setMovable(False)
        tb.addAction("Load folder…").triggered.connect(self.load_folder)
        tb.addAction("Load files…").triggered.connect(self.load_files)
        tb.addSeparator()

        # View selector: three mutually-exclusive toggles picking the central page.
        # Exactly one is always active, so "Hypercube" is the always-available way
        # back from the Phase / Stokes views.
        self.view_group = QtGui.QActionGroup(self)
        self.view_group.setExclusive(True)
        self.act_hyper = tb.addAction("Hypercube")
        self.act_hyper.setCheckable(True)
        self.act_hyper.setChecked(True)                 # default view (before connect)
        self.act_hyper.setToolTip("Main hypercube view: spatial map, spectra, "
                                  "process and metadata.")
        self.act_hyper.toggled.connect(self._toggle_hyper_view)
        self.view_group.addAction(self.act_hyper)
        self.act_phase = tb.addAction("Phase")
        self.act_phase.setCheckable(True)
        self.act_phase.setToolTip("Show only the amplitude + wrapped-phase maps "
                                  "(recomputed complex DFT at the current λ).")
        self.act_phase.toggled.connect(self._toggle_phase_view)
        self.view_group.addAction(self.act_phase)
        self.act_stokes = None
        if StokesApp is not None:
            self.act_stokes = tb.addAction("Stokes")
            self.act_stokes.setCheckable(True)
            self.act_stokes.setToolTip("Stokes polarimetry: compute S0..S3 per "
                                       "pixel from four QWP-angle measurements.")
            self.act_stokes.toggled.connect(self._toggle_stokes_view)
            self.view_group.addAction(self.act_stokes)
        tb.addSeparator()

        # Save / Export dropdown: all saving + export actions in one menu button.
        save_btn = QtWidgets.QToolButton()
        save_btn.setText("Save / Export ▾")
        save_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        save_menu = QtWidgets.QMenu(save_btn)
        save_menu.addAction("Save image…").triggered.connect(self.export_image)
        save_menu.addAction("Export GIF…").triggered.connect(self.export_gif)
        save_menu.addAction("Export spectra…").triggered.connect(self.export_spectra)
        save_menu.addAction("Export ROI vs Z…").triggered.connect(self.export_roi_vs_z)
        save_menu.addSeparator()
        save_menu.addAction("Recompute → save all Z…").triggered.connect(self.batch_recompute)
        save_btn.setMenu(save_menu)
        tb.addWidget(save_btn)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_split = split   # page 0 of the central stack (built at the end)

        # left: image + sliders + display controls
        leftw = QtWidgets.QWidget(); left = QtWidgets.QVBoxLayout(leftw)
        self.iv = pg.ImageView()
        self.iv.view.invertY(True)
        # Downsample big images for display (renders at screen resolution instead
        # of pushing every pixel through the GPU each frame).
        self.iv.getImageItem().setAutoDownsample(True)
        for w in (self.iv.ui.roiBtn, self.iv.ui.menuBtn, self.iv.ui.roiPlot):
            w.hide()
        self.iv.scene.sigMouseClicked.connect(self._on_click)
        left.addWidget(self.iv, 1)

        row = QtWidgets.QHBoxLayout()
        self.wl_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.wl_slider.valueChanged.connect(self._on_wl)
        self.lbl_wl = QtWidgets.QLabel("-- µm"); self.lbl_wl.setStyleSheet("font-weight:600;")
        row.addWidget(QtWidgets.QLabel("λ:")); row.addWidget(self.wl_slider, 1); row.addWidget(self.lbl_wl)
        left.addLayout(row)

        self.zrow = QtWidgets.QWidget(); zr = QtWidgets.QHBoxLayout(self.zrow); zr.setContentsMargins(0,0,0,0)
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_slider.valueChanged.connect(self._on_selection_changed)
        self.lbl_z = QtWidgets.QLabel("--")
        zr.addWidget(QtWidgets.QLabel("Z (mm):")); zr.addWidget(self.z_slider, 1); zr.addWidget(self.lbl_z)
        self.zrow.setVisible(False); left.addWidget(self.zrow)

        # Angle slider (datasets acquired at multiple angles: total cubes = Z × angle)
        self.arow = QtWidgets.QWidget(); ar = QtWidgets.QHBoxLayout(self.arow); ar.setContentsMargins(0,0,0,0)
        self.angle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.angle_slider.valueChanged.connect(self._on_selection_changed)
        self.lbl_a = QtWidgets.QLabel("--")
        ar.addWidget(QtWidgets.QLabel("Angle (°):")); ar.addWidget(self.angle_slider, 1); ar.addWidget(self.lbl_a)
        self.arow.setVisible(False); left.addWidget(self.arow)

        ctl = QtWidgets.QHBoxLayout()
        self.combo_map = QtWidgets.QComboBox()
        self.combo_map.addItems(["λ slice", "Peak λ", "Peak intensity", "SAM", "RGB",
                                 "K-means", "Continuum line"])
        self.combo_map.currentTextChanged.connect(self._on_map_changed)
        self.combo_cmap = QtWidgets.QComboBox()
        self.combo_cmap.addItems(["grey", "inferno", "viridis", "magma", "turbo", "jet"])
        self.combo_cmap.currentTextChanged.connect(self._apply_cmap)
        self.spin_gamma = QtWidgets.QDoubleSpinBox(); self.spin_gamma.setRange(0.1, 5.0)
        self.spin_gamma.setSingleStep(0.1); self.spin_gamma.setValue(1.0)
        self.spin_gamma.valueChanged.connect(self.refresh_map)
        ctl.addWidget(QtWidgets.QLabel("Map:")); ctl.addWidget(self.combo_map, 1)
        ctl.addWidget(QtWidgets.QLabel("Colors:")); ctl.addWidget(self.combo_cmap, 1)
        ctl.addWidget(QtWidgets.QLabel("γ:")); ctl.addWidget(self.spin_gamma)
        left.addLayout(ctl)

        # Analysis ROI: crop EVERY panel (map, spectra, clicked pixels, phase,
        # raw-recompute) to a rectangle so nothing outside it is ever computed.
        # Uncheck to see the full frame + drag the box; check to crop to it.
        wrow = QtWidgets.QHBoxLayout()
        self.chk_work = QtWidgets.QCheckBox("Limit to ROI")
        self.chk_work.setToolTip("Crop every panel to the ROI box so out-of-interest "
                                 "regions are never recomputed. Uncheck to reposition it.")
        self.chk_work.toggled.connect(self._on_work_toggled)
        self.btn_work_reset = QtWidgets.QPushButton("Reset box")
        self.btn_work_reset.setToolTip("Recentre the analysis-ROI box on the full frame.")
        self.btn_work_reset.clicked.connect(self._reset_work_roi)
        self.lbl_work = QtWidgets.QLabel("full frame")
        self.lbl_work.setStyleSheet("color:#888;")
        wrow.addWidget(self.chk_work); wrow.addWidget(self.btn_work_reset)
        wrow.addWidget(self.lbl_work, 1)
        left.addLayout(wrow)

        # Intensity scale (the colour scalebar is the histogram beside the image;
        # these lock it to a fixed min/max so it stays constant while scrubbing).
        ctl2 = QtWidgets.QHBoxLayout()
        self.chk_lock_lev = QtWidgets.QCheckBox("Lock intensity scale")
        self.chk_lock_lev.toggled.connect(self._on_lock_levels)
        self.spin_lmin = QtWidgets.QDoubleSpinBox(); self.spin_lmax = QtWidgets.QDoubleSpinBox()
        for s in (self.spin_lmin, self.spin_lmax):
            s.setDecimals(3); s.setRange(-1e15, 1e15); s.setEnabled(False)
            s.valueChanged.connect(self._on_level_spin)
        self.btn_full_lev = QtWidgets.QToolButton(); self.btn_full_lev.setText("Full")
        self.btn_full_lev.setToolTip("Set min/max to the current map's data range.")
        self.btn_full_lev.clicked.connect(self._levels_from_data)
        ctl2.addWidget(self.chk_lock_lev)
        ctl2.addWidget(QtWidgets.QLabel("min")); ctl2.addWidget(self.spin_lmin, 1)
        ctl2.addWidget(QtWidgets.QLabel("max")); ctl2.addWidget(self.spin_lmax, 1)
        ctl2.addWidget(self.btn_full_lev)
        left.addLayout(ctl2)
        split.addWidget(leftw)

        # right: tabs (Spectra / Process / Metadata)
        tabs = QtWidgets.QTabWidget()
        self.tabs = tabs
        split.addWidget(tabs); split.setSizes([820, 500])

        # Spectra tab
        spw = QtWidgets.QWidget(); sp = QtWidgets.QVBoxLayout(spw)
        self.spec = pg.PlotWidget(title="Spectra")
        self.spec.setLabel("bottom", "Wavelength", units="µm")
        self.spec.addLegend()
        sp.addWidget(self.spec, 1)
        rb = QtWidgets.QHBoxLayout()
        b_add = QtWidgets.QPushButton("Add ROI"); b_add.clicked.connect(self.add_roi)
        b_poly = QtWidgets.QPushButton("Add polygon"); b_poly.clicked.connect(self.add_poly_roi)
        b_clr = QtWidgets.QPushButton("Clear ROIs"); b_clr.clicked.connect(self.clear_rois)
        b_ref = QtWidgets.QPushButton("ROI 1 = reference"); b_ref.clicked.connect(self.set_reference)
        rb.addWidget(b_add); rb.addWidget(b_poly); rb.addWidget(b_clr); rb.addWidget(b_ref)
        sp.addLayout(rb)
        nb = QtWidgets.QHBoxLayout()
        self.combo_norm = QtWidgets.QComboBox()
        self.combo_norm.addItems(["raw", "area-normalised", "reflectivity (÷ reference)",
                                  "transmission (÷ reference)"])
        self.combo_norm.currentTextChanged.connect(self.update_spectra)
        self.combo_deriv = QtWidgets.QComboBox(); self.combo_deriv.addItems(["raw", "d/dλ", "d²/dλ²"])
        self.combo_deriv.currentTextChanged.connect(self.update_spectra)
        nb.addWidget(QtWidgets.QLabel("Normalise:")); nb.addWidget(self.combo_norm, 1)
        nb.addWidget(QtWidgets.QLabel("Deriv:")); nb.addWidget(self.combo_deriv)
        sp.addLayout(nb)
        self.lbl_pixel = QtWidgets.QLabel("Click the map for a pixel spectrum; Add ROI for region averages.")
        self.lbl_pixel.setStyleSheet("color:#888;"); sp.addWidget(self.lbl_pixel)
        tabs.addTab(spw, "Spectra / ROI")

        # Process tab
        prw = QtWidgets.QWidget(); pr = QtWidgets.QFormLayout(prw)
        # --- SVD denoise: tick to reveal the explained-variance (scree) plot,
        #     THEN choose how many components to keep; the cube is rebuilt from
        #     exactly those components.
        self.chk_svd = QtWidgets.QCheckBox("SVD denoise (current Z)")
        self.chk_svd.toggled.connect(self._on_svd_toggled)
        self.svd_plot = pg.PlotWidget(); self.svd_plot.setMaximumHeight(170)
        self.svd_plot.setLabel("bottom", "components kept")
        self.svd_plot.setLabel("left", "cumulative variance", units="%")
        self.svd_plot.showGrid(x=True, y=True, alpha=0.2)
        self.svd_curve = self.svd_plot.plot(pen=pg.mkPen("#1c7ed6", width=2),
                                            symbol="o", symbolSize=4)
        self.svd_kline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#e8590c", width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.svd_plot.addItem(self.svd_kline)
        self.svd_plot.setVisible(False)
        self.spin_k = QtWidgets.QSpinBox(); self.spin_k.setRange(1, 64); self.spin_k.setValue(6)
        self.spin_k.setEnabled(False)
        self.spin_k.valueChanged.connect(self._on_svd_k_changed)
        pr.addRow(self.chk_svd)
        pr.addRow(self.svd_plot)
        pr.addRow("Components k", self.spin_k)

        # --- SAM: per-pixel spectral-angle map vs a reference spectrum.
        self.lbl_sam = QtWidgets.QLabel("SAM reference: none (uses ROI 1 / average)")
        self.lbl_sam.setStyleSheet("color:#888;")
        b_sam_roi = QtWidgets.QPushButton("SAM ref = ROI 1 average")
        b_sam_roi.clicked.connect(self._sam_ref_from_roi)
        b_sam_file = QtWidgets.QPushButton("SAM ref from file…")
        b_sam_file.clicked.connect(self._sam_ref_from_file)
        b_sam_show = QtWidgets.QPushButton("Show SAM map")
        b_sam_show.clicked.connect(self._show_sam)
        pr.addRow(b_sam_roi); pr.addRow(b_sam_file); pr.addRow(b_sam_show)
        pr.addRow(self.lbl_sam)

        # --- Background subtraction: subtract the average spectrum of a chosen ROI
        #     from every pixel's spectrum (affects all maps + spectra).
        self.chk_bg = QtWidgets.QCheckBox("Subtract background ROI")
        self.chk_bg.toggled.connect(self._on_bg_toggled)
        b_bg = QtWidgets.QPushButton("Set background = ROI 1")
        b_bg.setToolTip("Draw an ROI over a background region (Spectra/ROI tab), then "
                        "click this: its average spectrum is subtracted from every pixel.")
        b_bg.clicked.connect(self._bg_from_roi)
        self.lbl_bg = QtWidgets.QLabel("Background: none")
        self.lbl_bg.setStyleSheet("color:#888;")
        pr.addRow(self.chk_bg)
        pr.addRow(b_bg)
        pr.addRow(self.lbl_bg)

        self.spin_clusters = QtWidgets.QSpinBox(); self.spin_clusters.setRange(2, 16)
        self.spin_clusters.setValue(4)
        self.spin_clusters.valueChanged.connect(
            lambda *_: self.refresh_map() if self.combo_map.currentText() == "K-means" else None)
        pr.addRow("K-means clusters", self.spin_clusters)

        # --- Continuum subtraction (Map = "Continuum line"): per pixel, subtract a
        #     linear continuum estimated from two shoulder bands and average the
        #     residual over the line band -> isolates a narrow emission line.
        def _band(lo, hi):
            a = QtWidgets.QDoubleSpinBox(); b = QtWidgets.QDoubleSpinBox()
            for s, v in ((a, lo), (b, hi)):
                s.setDecimals(3); s.setRange(0.1, 100.0); s.setSuffix(" µm"); s.setValue(v)
                s.valueChanged.connect(self._on_continuum_changed)
            row = QtWidgets.QWidget(); box = QtWidgets.QHBoxLayout(row)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(a); box.addWidget(QtWidgets.QLabel("–")); box.addWidget(b)
            return row, a, b
        line_row, self.spin_line0, self.spin_line1 = _band(3.965, 4.050)
        lsh_row, self.spin_lsh0, self.spin_lsh1 = _band(3.900, 3.955)
        rsh_row, self.spin_rsh0, self.spin_rsh1 = _band(4.060, 4.120)
        pr.addRow("Line band", line_row)
        pr.addRow("Left shoulder", lsh_row)
        pr.addRow("Right shoulder", rsh_row)

        # False-colour RGB (Map = "RGB"): each pixel's spectrum is projected onto
        # the hsiAnalysis R/G/B response curves (RGB_Transmission), so colour
        # reflects the whole spectral shape (over the current λ range). No per-band
        # picking needed -- click the button to render it in the left viewer.
        self.btn_show_rgb = QtWidgets.QPushButton("Show false-colour RGB map")
        self.btn_show_rgb.clicked.connect(self._show_rgb)
        pr.addRow(self.btn_show_rgb)

        self.spin_crop0 = QtWidgets.QDoubleSpinBox(); self.spin_crop0.setRange(0.1, 100); self.spin_crop0.setSuffix(" µm")
        self.spin_crop1 = QtWidgets.QDoubleSpinBox(); self.spin_crop1.setRange(0.1, 100); self.spin_crop1.setSuffix(" µm")
        self.chk_crop = QtWidgets.QCheckBox("Limit λ range (map + spectra)")
        self.chk_crop.toggled.connect(self._on_crop_changed)
        for sp in (self.spin_crop0, self.spin_crop1):
            sp.valueChanged.connect(self._on_crop_changed)
        pr.addRow(self.chk_crop)
        pr.addRow("λ from", self.spin_crop0); pr.addRow("λ to", self.spin_crop1)
        tabs.addTab(prw, "Process")

        # Interferogram / recompute tab
        igw = QtWidgets.QWidget(); ig = QtWidgets.QVBoxLayout(igw)
        self.ifg_plot = pg.PlotWidget(
            title="Interferogram vs wedge position (ROI/frame mean + clicked pixel)")
        self.ifg_plot.setLabel("bottom", "Wedge position", units="mm")
        self.ifg_plot.addLegend(offset=(-10, 10))
        self.ifg_curve = self.ifg_plot.plot(pen=pg.mkPen("#1c7ed6", width=1),
                                            name="ROI/frame mean")
        # One interferogram curve per clicked pixel (raw[:, row, col]); rebuilt on
        # each refresh from self.sel_pixels.
        self.ifg_px_curves = []
        self.win_region = pg.LinearRegionItem(brush=(80, 160, 255, 45))
        self.win_region.sigRegionChangeFinished.connect(self._on_window_changed)
        self.ifg_plot.addItem(self.win_region)
        ig.addWidget(self.ifg_plot, 1)
        # Which traces to show, and clear the pixel selection. Ctrl/Shift-click on
        # the map adds pixels (plain click selects a single one).
        vis = QtWidgets.QHBoxLayout()
        vis.addWidget(QtWidgets.QLabel("Show:"))
        self.chk_show_mean = QtWidgets.QCheckBox("ROI/frame mean"); self.chk_show_mean.setChecked(True)
        self.chk_show_pixels = QtWidgets.QCheckBox("Pixels"); self.chk_show_pixels.setChecked(True)
        self.chk_show_mean.toggled.connect(self.update_interferogram)
        self.chk_show_pixels.toggled.connect(self.update_interferogram)
        btn_clear_px = QtWidgets.QPushButton("Clear pixels")
        btn_clear_px.clicked.connect(self._clear_pixels)
        vis.addWidget(self.chk_show_mean); vis.addWidget(self.chk_show_pixels)
        vis.addWidget(btn_clear_px); vis.addStretch(1)
        ig.addLayout(vis)
        self.chk_recompute = QtWidgets.QCheckBox(
            "Recompute map/spectra from the interferogram window (below)")
        self.chk_recompute.toggled.connect(self._toggle_recompute)
        ig.addWidget(self.chk_recompute)
        pb = QtWidgets.QHBoxLayout()
        for name, fn in [("Full", self._win_full), ("Centre", self._win_centre),
                         ("Left tail", self._win_left), ("Right tail", self._win_right)]:
            b = QtWidgets.QPushButton(name); b.clicked.connect(fn); pb.addWidget(b)
        ig.addLayout(pb)
        form = QtWidgets.QFormLayout()
        self.r_apod = QtWidgets.QComboBox()
        self.r_apod.addItems(["gaussian", "happ-genzel", "blackman-harris-3",
                              "blackman-harris-4", "boxcar"])
        self.r_wl0 = QtWidgets.QDoubleSpinBox(); self.r_wl0.setRange(0.1, 100); self.r_wl0.setSuffix(" µm")
        self.r_wl1 = QtWidgets.QDoubleSpinBox(); self.r_wl1.setRange(0.1, 100); self.r_wl1.setSuffix(" µm")
        self.r_nfreq = QtWidgets.QSpinBox(); self.r_nfreq.setRange(0, 8192)
        self.r_nfreq.setSpecialValueText("Auto")
        for wdg in (self.r_apod, self.r_wl0, self.r_wl1, self.r_nfreq):
            sig = (wdg.currentTextChanged if isinstance(wdg, QtWidgets.QComboBox)
                   else wdg.valueChanged)
            sig.connect(self._recompute_if_on)
        form.addRow("Apodization", self.r_apod)
        form.addRow("λ from", self.r_wl0); form.addRow("λ to", self.r_wl1)
        form.addRow("N freq", self.r_nfreq)
        ig.addLayout(form)
        self.lbl_axis = QtWidgets.QLabel("")
        self.lbl_axis.setStyleSheet("font-weight:bold;")
        ig.addWidget(self.lbl_axis)
        self.lbl_win = QtWidgets.QLabel(
            "Drag the shaded window (or the buttons). A LONGER window → finer "
            "spectral detail; the centre burst → broad features, the tails → high-Q lines.")
        self.lbl_win.setWordWrap(True); self.lbl_win.setStyleSheet("color:#888;")
        ig.addWidget(self.lbl_win)
        tabs.addTab(igw, "Interferogram / Recompute")

        # Phase tab: recompute the COMPLEX per-pixel DFT at the current λ / plane
        # and show XY amplitude + wrapped phase side by side. The saved cube keeps
        # only the DFT magnitude (np.abs), so the interferometric phase is lost;
        # this re-runs the identical forward transform (same FT window as the
        # Interferogram/Recompute tab + its apodization) and keeps the phase.
        self.phase_page = QtWidgets.QWidget(); ph = QtWidgets.QVBoxLayout(self.phase_page)
        self._last_phase = None
        # Slim slider bar so the phase-only page is self-contained: λ (+ z / angle
        # when present) mirror the main sliders both ways (see _mirror_phase_sliders).
        pbar = QtWidgets.QHBoxLayout()
        self.ph_wl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ph_lbl_wl = QtWidgets.QLabel("-- µm"); self.ph_lbl_wl.setStyleSheet("font-weight:600;")
        pbar.addWidget(QtWidgets.QLabel("λ:")); pbar.addWidget(self.ph_wl, 1); pbar.addWidget(self.ph_lbl_wl)
        # Z as a dropdown of the actual positions (like the Stokes panel), rather
        # than a slider -- lists each z in mm and jumps straight to it.
        self.ph_z_combo = QtWidgets.QComboBox(); self.ph_z_combo.setMinimumWidth(110)
        self.ph_zrow = QtWidgets.QWidget(); _pz = QtWidgets.QHBoxLayout(self.ph_zrow); _pz.setContentsMargins(0,0,0,0)
        _pz.addWidget(QtWidgets.QLabel("Z:")); _pz.addWidget(self.ph_z_combo, 1)
        self.ph_a = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ph_lbl_a = QtWidgets.QLabel("--")
        self.ph_arow = QtWidgets.QWidget(); _pa = QtWidgets.QHBoxLayout(self.ph_arow); _pa.setContentsMargins(0,0,0,0)
        _pa.addWidget(QtWidgets.QLabel("Angle:")); _pa.addWidget(self.ph_a, 1); _pa.addWidget(self.ph_lbl_a)
        self.ph_zrow.setVisible(False); self.ph_arow.setVisible(False)
        pbar.addWidget(self.ph_zrow, 1); pbar.addWidget(self.ph_arow, 1)
        ph.addLayout(pbar)
        # Two-way mirror: moving a phase slider drives the master (which updates the
        # maps); moving the master mirrors back. setValue only re-emits on a real
        # change, so the two converge with no feedback loop.
        self.ph_wl.valueChanged.connect(self.wl_slider.setValue)
        self.ph_a.valueChanged.connect(self.angle_slider.setValue)
        self.wl_slider.valueChanged.connect(self.ph_wl.setValue)
        self.angle_slider.valueChanged.connect(self.ph_a.setValue)
        # Z dropdown <-> master z slider (index == z position). setCurrentIndex /
        # setValue only re-emit on a real change, so the two converge, no loop.
        self.ph_z_combo.currentIndexChanged.connect(self._on_ph_z_combo)
        self.z_slider.valueChanged.connect(self.ph_z_combo.setCurrentIndex)
        self.ph_glw = pg.GraphicsLayoutWidget()
        self.ph_glw.addLabel("Amplitude", row=0, col=0, colspan=2)
        self.ph_glw.addLabel("Wrapped phase [-π, π]", row=0, col=2, colspan=2)
        vb_a = self.ph_glw.addViewBox(row=1, col=0)
        vb_a.setAspectLocked(True); vb_a.invertY(True)
        self.ph_img_amp = pg.ImageItem(); vb_a.addItem(self.ph_img_amp)
        self.ph_cbar_amp = pg.ColorBarItem(interactive=False, colorMap=_get_cmap("turbo"))
        self.ph_glw.addItem(self.ph_cbar_amp, row=1, col=1)
        self.ph_cbar_amp.setImageItem(self.ph_img_amp)
        vb_p = self.ph_glw.addViewBox(row=1, col=2)
        vb_p.setAspectLocked(True); vb_p.invertY(True)
        vb_p.setBackgroundColor(pg.mkColor(200, 200, 200))
        self.ph_img_phase = pg.ImageItem(); vb_p.addItem(self.ph_img_phase)
        self.ph_cbar_phase = pg.ColorBarItem(interactive=False,
                                             colorMap=_phase_cmap(PHASE_CMAPS[0]),
                                             values=(-np.pi, np.pi))
        self.ph_glw.addItem(self.ph_cbar_phase, row=1, col=3)
        self.ph_cbar_phase.setImageItem(self.ph_img_phase)
        ph.addWidget(self.ph_glw, 1)
        pc = QtWidgets.QHBoxLayout()
        pc.addWidget(QtWidgets.QLabel("Phase colors"))
        self.ph_cmap_combo = QtWidgets.QComboBox()
        self.ph_cmap_combo.addItems(PHASE_CMAPS)
        self.ph_cmap_combo.currentTextChanged.connect(self._on_phase_cmap)
        pc.addWidget(self.ph_cmap_combo)
        pc.addWidget(QtWidgets.QLabel("Phase mask below"))
        self.ph_thresh = QtWidgets.QDoubleSpinBox()
        self.ph_thresh.setRange(0.0, 100.0); self.ph_thresh.setValue(5.0)
        self.ph_thresh.setSuffix(" % max amp"); self.ph_thresh.setDecimals(2)
        self.ph_thresh.valueChanged.connect(self._update_phase)
        pc.addWidget(self.ph_thresh)
        btn_ph_exp = QtWidgets.QPushButton("Export amp+phase…")
        btn_ph_exp.clicked.connect(self._export_phase)
        pc.addWidget(btn_ph_exp); pc.addStretch(1)
        ph.addLayout(pc)

        # Background phase correction: load a background interferogram and subtract
        # its per-pixel phase (removes the interferometer's instrumental phase,
        # leaving the true spectral phase). Can phase + save the whole Z×θ stack.
        pb = QtWidgets.QHBoxLayout()
        btn_bkg = QtWidgets.QPushButton("Load background…")
        btn_bkg.setToolTip("Load a background interferogram (.npz with raw_interferogram) "
                           "of the SAME ROI + wedge scan. Its per-pixel phase is subtracted.")
        btn_bkg.clicked.connect(self._phase_load_bkg)
        self.chk_phase_bkg = QtWidgets.QCheckBox("Subtract background phase")
        self.chk_phase_bkg.setEnabled(False)
        self.chk_phase_bkg.toggled.connect(self._on_phase_bkg_toggled)
        btn_phase_stack = QtWidgets.QPushButton("Phase Z×θ stack → save…")
        btn_phase_stack.setToolTip("Phase-correct every cube in the loaded stack with the "
                                   "background and save the phased spectrum cubes to a folder.")
        btn_phase_stack.clicked.connect(self._phase_stack_save)
        pb.addWidget(btn_bkg); pb.addWidget(self.chk_phase_bkg)
        pb.addWidget(btn_phase_stack); pb.addStretch(1)
        ph.addLayout(pb)
        self.lbl_phase_bkg = QtWidgets.QLabel("Background: none loaded.")
        self.lbl_phase_bkg.setStyleSheet("color:#888;"); self.lbl_phase_bkg.setWordWrap(True)
        ph.addWidget(self.lbl_phase_bkg)

        self.ph_info = QtWidgets.QLabel(
            "Move the λ slider to change wavelength. The FT window + apodization "
            "are taken from the Interferogram / Recompute tab.")
        self.ph_info.setWordWrap(True); self.ph_info.setStyleSheet("color:#666;")
        ph.addWidget(self.ph_info)
        # (phase_page is NOT a tab -- it is page 1 of the central stack, shown via
        # the "Phase" toolbar toggle.)

        # Metadata tab
        self.meta = QtWidgets.QPlainTextEdit(); self.meta.setReadOnly(True)
        self.meta.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        tabs.addTab(self.meta, "Metadata")

        tabs.currentChanged.connect(self._on_tab_changed)

        # Central stack: page 0 = normal analysis view, page 1 = phase-only page
        # (just the amplitude + wrapped-phase maps). The "Phase" toolbar toggle
        # switches between them.
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.main_split)   # index 0
        self.stack.addWidget(self.phase_page)   # index 1
        # Stokes page: embed the standalone StokesApp (its own self-contained UI
        # + file loading), so the panel behaves exactly like stokes_app.py.
        self.stokes_app = StokesApp(embedded=True) if StokesApp is not None else None
        if self.stokes_app is not None:
            self.stack.addWidget(self.stokes_app)   # index 2
        self.setCentralWidget(self.stack)

        self.statusBar().showMessage("No data — File ▸ Load folder…")

    # -- loading --------------------------------------------------------------
    def load_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder of per-Z hypercubes")
        if not d:
            return
        # Each run is saved into its own <run-stamp>.<name>/ subfolder, so search
        # the chosen folder AND its subfolders -- works whether the user picks a
        # single run folder or the parent camera folder holding many runs.
        # ("**" with recursive=True also matches files directly in d.)
        paths = sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True))
        if not paths:
            self.statusBar().showMessage("No .npz files in this folder (or its subfolders).")
            return
        groups = _group_experiments(paths)
        if len(groups) <= 1:
            self.load_paths(paths)
            return
        labels = [g[0] for g in groups] + ["— All files (mixed) —"]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Choose experiment",
            f"{len(groups)} experiments found in this folder — pick one:",
            labels, 0, False)
        if not ok:
            return
        if choice.startswith("— All"):
            self.load_paths(paths)
        else:
            self.load_paths(next((g[1] for g in groups if g[0] == choice), paths))

    def load_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load per-Z hypercubes", "", "NumPy archive (*.npz)")
        if paths:
            self.load_paths(list(paths))

    def load_paths(self, paths):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            infos = []
            for p in paths:
                try:
                    for i in _infos_from_file(p):
                        if i["has_cube"] and i["wavelengths"] is not None:
                            infos.append(i)
                except Exception as e:  # noqa: BLE001
                    print("skip", p, e)
            if not infos:
                self.statusBar().showMessage("No per-Z hypercubes found.")
                return
            infos.sort(key=lambda i: (np.isnan(i["z"]), i["z"],
                                      np.isnan(i["angle"]), i["angle"]))
            self.infos = infos
            self._build_grid(infos)
            self.wavelengths = np.asarray(infos[0]["wavelengths"])
            self._saved_wavelengths = self.wavelengths
            self._cube_cache.clear(); self._raw_cache.clear()
            self._kmeans_labels = None; self._cluster_key = None
            self._reset_work_crop()        # new dataset -> back to full frame
            self.recompute_on = False
            self.chk_recompute.blockSignals(True)
            self.chk_recompute.setChecked(False); self.chk_recompute.setEnabled(True)
            self.chk_recompute.blockSignals(False)
            for sp, v in ((self.r_wl0, self.wavelengths.min()),
                          (self.r_wl1, self.wavelengths.max())):
                sp.blockSignals(True); sp.setValue(float(v)); sp.blockSignals(False)
            nf = len(self.wavelengths)
            self.spin_crop0.setValue(float(self.wavelengths.min()))
            self.spin_crop1.setValue(float(self.wavelengths.max()))
            self.wl_slider.blockSignals(True)
            self.wl_slider.setMinimum(0)
            self.wl_slider.setMaximum(max(0, nf - 1)); self.wl_slider.setValue(nf // 2)
            self.wl_slider.blockSignals(False)
            self.zrow.setVisible(len(self._z_axis) > 1)
            self.arow.setVisible(len(self._a_axis) > 1)
            for slider, axis in ((self.z_slider, self._z_axis),
                                 (self.angle_slider, self._a_axis)):
                slider.blockSignals(True)
                slider.setMaximum(max(0, len(axis) - 1)); slider.setValue(0)
                slider.blockSignals(False)
            self._apply_cmap()
            self._update_slider_labels()
            self.refresh_map()
            self.update_spectra()
            self.update_interferogram()
            self._update_meta()
            self._mirror_phase_sliders()   # phase-page sliders track the new data
            self._update_phase()
            zr = (f"{self._z_axis[0]:.3f}…{self._z_axis[-1]:.3f} mm"
                  if len(self._z_axis) > 1 else
                  ("single Z" if self._z_axis[0] != self._z_axis[0]
                   else f"z={self._z_axis[0]:.3f} mm"))
            if len(self._a_axis) > 1:
                zr += f", {len(self._a_axis)} angles {self._a_axis[0]:.1f}…{self._a_axis[-1]:.1f}°"
            self.statusBar().showMessage(f"Loaded {len(infos)} cube(s): {zr}")
            # A new dataset -> the Stokes panel must refill from it (lazily on next
            # open, or right away if it is the page currently shown).
            self._stokes_dirty = True
            if (self.stokes_app is not None
                    and self.stack.currentWidget() is self.stokes_app):
                self._refresh_stokes_from_main()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # -- (Z, angle) selection grid -------------------------------------------
    def _build_grid(self, infos):
        """Sorted unique Z / angle axes + a (z_index, angle_index) -> flat-index
        grid, so the two sliders together select one cube. NaN (missing) axis
        collapses to a single bucket -> backward compatible with Z-only data."""
        def axis(key):
            vals = sorted({round(float(i[key]), 4) for i in infos if i[key] == i[key]})
            return vals if vals else [float("nan")]
        self._z_axis = axis("z")
        self._a_axis = axis("angle")
        z_of = {v: k for k, v in enumerate(self._z_axis)}
        a_of = {v: k for k, v in enumerate(self._a_axis)}
        self._grid = {}
        for idx, info in enumerate(infos):
            zv = round(float(info["z"]), 4) if info["z"] == info["z"] else None
            av = round(float(info["angle"]), 4) if info["angle"] == info["angle"] else None
            self._grid[(z_of.get(zv, 0), a_of.get(av, 0))] = idx

    def _cur(self):
        """Flat index into self.infos for the current (Z, angle) selection, or None."""
        if not self.infos:
            return None
        return self._grid.get((self.z_slider.value(), self.angle_slider.value()))

    def _update_slider_labels(self):
        zi, ai = self.z_slider.value(), self.angle_slider.value()
        z = self._z_axis[zi] if 0 <= zi < len(self._z_axis) else float("nan")
        a = self._a_axis[ai] if 0 <= ai < len(self._a_axis) else float("nan")
        self.lbl_z.setText("--" if z != z else f"{z:.4f} mm")
        self.lbl_a.setText("--" if a != a else f"{a:.2f}°")

    # -- analysis ROI (crop every panel to a region of interest) --------------
    def _work_active(self):
        return self.chk_work.isChecked() and self.work_roi is not None

    def _work_bounds(self, h, w):
        """(r0,r1,c0,c1) crop of a full (h,w) frame from the analysis ROI box,
        clipped to the frame; None when the crop is off / degenerate."""
        if not self._work_active():
            return None
        pos, size = self.work_roi.pos(), self.work_roi.size()
        c0 = int(np.clip(round(float(pos[0])), 0, w - 1))
        r0 = int(np.clip(round(float(pos[1])), 0, h - 1))
        c1 = int(np.clip(round(float(pos[0]) + float(size[0])), c0 + 1, w))
        r1 = int(np.clip(round(float(pos[1]) + float(size[1])), r0 + 1, h))
        return (r0, r1, c0, c1) if (r1 > r0 and c1 > c0) else None

    def _work_crop3(self, arr):
        """Crop a FULL (n,h,w) array to the analysis ROI (identity if crop off)."""
        if arr is None or arr.ndim != 3:
            return arr
        b = self._work_bounds(arr.shape[1], arr.shape[2])
        if b is None:
            return arr
        r0, r1, c0, c1 = b
        return arr[:, r0:r1, c0:c1]

    def _ensure_work_roi(self, h, w):
        if self.work_roi is not None:
            return
        sw, sh = max(2, int(w * 0.6)), max(2, int(h * 0.6))
        roi = pg.RectROI([(w - sw) // 2, (h - sh) // 2], [sw, sh],
                         pen=pg.mkPen("#ffd43b", width=2))
        roi.setZValue(20)
        roi.sigRegionChanged.connect(self._update_work_label)
        self.iv.getView().addItem(roi)
        roi.setVisible(False)
        self.work_roi = roi

    def _update_work_label(self, *a):
        b = self._work_bounds(*self._full_hw) if (self._work_active() and self._full_hw) else None
        if b:
            r0, r1, c0, c1 = b
            self.lbl_work.setText(f"ROI {c1-c0}×{r1-r0} px @ ({c0},{r0})")
        elif self.work_roi is not None and not self.chk_work.isChecked():
            pos, size = self.work_roi.pos(), self.work_roi.size()
            self.lbl_work.setText(f"box {int(size[0])}×{int(size[1])} px "
                                  f"(check 'Limit to ROI' to apply)")
        else:
            self.lbl_work.setText("full frame")

    def _clear_rois_silent(self):
        """Drop all measurement ROIs + their references without recursing into a
        refresh (coordinates change when the crop toggles)."""
        for roi in self.rois:
            try:
                self.iv.getView().removeItem(roi)
            except Exception:  # noqa: BLE001
                pass
        self.rois = []
        self.ref_roi = self.bg_roi = self.sam_ref_roi = None
        self._bg_sig = None
        if hasattr(self, "lbl_bg"):
            self.lbl_bg.setText("Background: none")

    def _on_work_toggled(self, on):
        """Enter/leave the ROI-cropped view. Coordinates change, so cached cubes
        and measurement ROIs/pixels are reset."""
        if on:
            if not self._full_hw:
                self.chk_work.blockSignals(True); self.chk_work.setChecked(False)
                self.chk_work.blockSignals(False)
                self.statusBar().showMessage("Load a dataset first.")
                return
            self._ensure_work_roi(*self._full_hw)
            self.work_roi.setVisible(False)        # cropped view -> hide the box
        elif self.work_roi is not None:
            self.work_roi.setVisible(True)         # full frame -> box for repositioning
        # region changed -> invalidate cube-shaped caches + local (per-frame) state
        self._cube_cache.clear(); self._bg_cube = None; self._bg_sig = None
        self.sel_pixels = []
        self._clear_rois_silent()
        self._kmeans_labels = None; self._cluster_key = None
        self._update_work_label()
        self.refresh_map(); self.update_spectra()
        self.update_interferogram(); self._update_meta(); self._update_phase()

    def _reset_work_crop(self):
        """New dataset -> discard the analysis ROI (its dims may differ) and return
        to the full frame."""
        self.chk_work.blockSignals(True); self.chk_work.setChecked(False)
        self.chk_work.blockSignals(False)
        if self.work_roi is not None:
            try:
                self.iv.getView().removeItem(self.work_roi)
            except Exception:  # noqa: BLE001
                pass
            self.work_roi = None
        self._full_hw = None
        if hasattr(self, "lbl_work"):
            self.lbl_work.setText("full frame")

    def _reset_work_roi(self):
        if not self._full_hw:
            return
        h, w = self._full_hw
        self._ensure_work_roi(h, w)
        sw, sh = max(2, int(w * 0.6)), max(2, int(h * 0.6))
        self.work_roi.setSize([sw, sh]); self.work_roi.setPos([(w - sw) // 2, (h - sh) // 2])
        self.work_roi.setVisible(not self.chk_work.isChecked())
        if self.chk_work.isChecked():
            self._on_work_toggled(True)            # re-crop to the recentred box
        self._update_work_label()

    def _on_selection_changed(self, *a):
        """Z or angle slider moved. Update the label instantly and DEBOUNCE the
        heavy refresh (a fresh cube reload) so dragging across a stack only loads
        the position you land on, not every one in between."""
        self._update_slider_labels()
        self._sel_timer.start(_SEL_DEBOUNCE_MS)

    def _do_selection_refresh(self):
        """Deferred heavy refresh for a settled Z/angle selection."""
        if self.infos and self._cur() is None:
            self.statusBar().showMessage("No cube acquired at this Z / angle combination.")
        self.refresh_map(); self.update_spectra()
        self.update_interferogram(); self._update_meta()
        self._update_phase()

    # -- cube access (lazy, with denoise) ------------------------------------
    def _base_cube(self, zi):
        """(wl, cube) for z-index `zi` BEFORE SVD denoise -- the raw loaded cube
        (or the recomputed one). Used by both current_cube and the scree plot."""
        if self.recompute_on:
            return self._compute_from_raw(zi)
        cube = _load_cube(self.infos[zi]["path"], self.infos[zi].get("map_index"))
        wl = self.infos[zi].get("wavelengths")
        wl = np.asarray(wl) if wl is not None else self._saved_wavelengths
        return wl, cube

    @staticmethod
    def _trim_cache(cache):
        """Evict oldest entries until within the item/byte caps (keep >=1)."""
        def nbytes(v):
            arr = v[1]
            return int(arr.nbytes) if arr is not None else 0
        while len(cache) > 1 and (
                len(cache) > _CACHE_MAX_ITEMS
                or sum(nbytes(v) for v in cache.values()) > _CACHE_MAX_BYTES):
            cache.popitem(last=False)   # drop least-recently-used

    def current_cube(self):
        if not self.infos:
            return None
        zi = self._cur()
        if zi is None:
            return None
        if zi in self._cube_cache:
            self._cube_cache.move_to_end(zi)          # mark most-recently-used
        else:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                wl, cube = self._base_cube(zi)
                if cube is not None:
                    # Analysis ROI: crop the loaded FULL cube. (When recompute_on,
                    # _base_cube already returns a cube computed from the cropped
                    # raw, so it must NOT be cropped again -- hence the guard. Full
                    # dims are tracked from the full cube / full raw for the ROI.)
                    if not self.recompute_on:
                        self._full_hw = cube.shape[1:]
                        cube = self._work_crop3(cube)
                    if self.chk_svd.isChecked():
                        cube = A.svd_denoise(cube, self.spin_k.value())
                # Cache the cube WITH its own wavelength axis so they never drift
                # apart (files can differ in n_freq; recompute changes the axis).
                self._cube_cache[zi] = (np.asarray(wl) if wl is not None else None, cube)
                self._trim_cache(self._cube_cache)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        wl, cube = self._cube_cache.get(zi, (None, None))
        if wl is not None:
            self.wavelengths = wl
            self._sync_wl_slider(len(wl))
        return self._apply_background(cube)

    def _apply_background(self, cube):
        """Subtract the chosen background ROI's average spectrum from every pixel.
        Cached by (cube, ROI geometry) so it only recomputes when one changes; the
        base (load+SVD) cube stays cached, so this never re-reads from disk."""
        if (cube is None or not self.chk_bg.isChecked()
                or self.bg_roi is None or self.bg_roi not in self.rois):
            return cube
        sig = (id(cube), id(self.bg_roi), str(self.bg_roi.saveState()))
        if self._bg_sig != sig or self._bg_cube is None:
            bg = self._roi_spectrum(self.bg_roi, cube)[0]
            self._bg_cube = cube - np.asarray(bg, dtype=cube.dtype)[:, None, None]
            self._bg_sig = sig
        return self._bg_cube

    # -- raw interferogram + recompute ---------------------------------------
    def current_interferogram(self):
        zi = self._cur()
        if zi is None:
            return None, None
        if zi in self._raw_cache:
            self._raw_cache.move_to_end(zi)
        else:
            pos = raw = None; calibrated = False
            mi = self.infos[zi].get("map_index")
            try:
                with np.load(self.infos[zi]["path"], allow_pickle=True) as d:
                    raw, pos, calibrated = _read_raw(d, mi)
            except Exception as e:  # noqa: BLE001
                print("raw load:", e)
            self._raw_cache[zi] = (pos, raw, calibrated)
            self._trim_cache(self._raw_cache)
        pos, raw, _ = self._raw_cache[zi]
        # The cache holds the FULL raw; track full dims and return the analysis-ROI
        # crop so phase / recompute only work on the region of interest.
        if raw is not None:
            self._full_hw = raw.shape[1:]
        return pos, self._work_crop3(raw)

    def current_positions_calibrated(self):
        """True if the current z's position axis is already the calibrated one
        (so recompute must NOT re-correct it)."""
        entry = self._raw_cache.get(self._cur())
        return bool(entry[2]) if entry else False

    def _phase_ref_cube(self, shape):
        """The background reference interferogram to phase-correct against, if one
        is loaded, phasing is enabled, and its shape matches (else None)."""
        if not self.phase_bkg_on or self.phase_bkg is None:
            return None
        # Crop the (full-frame) background to the SAME analysis ROI so it matches
        # the cropped measurement raw before its phase is subtracted.
        ref_raw = self._work_crop3(self.phase_bkg[0])
        return ref_raw if getattr(ref_raw, "shape", None) == tuple(shape) else None

    def _compute_from_raw(self, zi):
        pos, raw = self.current_interferogram()
        if raw is None or pos is None:
            return None, None
        if self.proc is None:
            self.proc = HyperspectralProcessor()
        nfreq = resolve_n_points(len(pos), manual=self.r_nfreq.value())
        return self.proc.compute_hyperspectral(
            pos, raw, wl_start=self.r_wl0.value(), wl_stop=self.r_wl1.value(),
            n_freq=nfreq, apod_type=self.r_apod.currentText(),
            ft_window_mm=self.win_region.getRegion(),
            expected_zero_mm=DEFAULT_ZPD_MM, search_mm=DEFAULT_ZPD_WINDOW_MM,
            positions_calibrated=self.current_positions_calibrated(),
            reference_cube=self._phase_ref_cube(raw.shape))

    def _sync_wl_slider(self, nf=None):
        self._apply_wl_slider_range()

    # -- λ limit (display-only crop of the bands the map uses) -----------------
    def _crop_indices(self, wl=None, n=None):
        """[i0, i1) band range for the λ-limit, or None when off / empty."""
        if not getattr(self, "chk_crop", None) or not self.chk_crop.isChecked():
            return None
        wl = self.wavelengths if wl is None else wl
        if wl is None:
            return None
        wl = np.asarray(wl, float)
        n = len(wl) if n is None else int(n)
        lo, hi = sorted((self.spin_crop0.value(), self.spin_crop1.value()))
        idx = np.where((wl >= lo) & (wl <= hi))[0]
        if idx.size < 1:
            return None
        i0, i1 = int(idx[0]), min(int(idx[-1]) + 1, n)
        return (i0, i1) if i1 - i0 >= 1 else None

    def _crop_slice(self, cube):
        """(subcube, subwl) restricted to the λ-limit, else the full cube/axis."""
        wl = np.asarray(self.wavelengths)
        rng = self._crop_indices(wl, cube.shape[0])
        if rng is None:
            return cube, wl
        i0, i1 = rng
        return cube[i0:i1], wl[i0:i1]

    def _apply_wl_slider_range(self):
        """Bound the λ slider to the full cube, or to the λ-limit band range."""
        if self.wavelengths is None:
            return
        n = len(self.wavelengths)
        rng = self._crop_indices(self.wavelengths, n)
        lo, hi = (0, max(0, n - 1)) if rng is None else (rng[0], rng[1] - 1)
        self.wl_slider.blockSignals(True)
        self.wl_slider.setMinimum(lo); self.wl_slider.setMaximum(hi)
        v = self.wl_slider.value()
        if not (lo <= v <= hi):
            self.wl_slider.setValue((lo + hi) // 2)
        self.wl_slider.blockSignals(False)

    def _on_crop_changed(self, *a):
        self._apply_wl_slider_range()
        self.refresh_map()
        self.update_spectra()
        self._update_phase()

    def update_interferogram(self):
        if not self.infos:
            return
        pos, raw = self.current_interferogram()
        if raw is None or pos is None:
            self.ifg_curve.setData([], [])
            self._clear_pixel_curves()
            self.chk_recompute.setChecked(False)
            self.chk_recompute.setEnabled(False)
            self.lbl_axis.setText("")
            self.lbl_win.setText("This file has no stored raw interferogram — recompute unavailable.")
            return
        self.chk_recompute.setEnabled(True)
        # Tell the user (from the file's metadata) which wedge axis this is, so it
        # is clear the recompute won't double-correct an already-calibrated axis.
        _ci = self._cur()
        meta = self.infos[_ci].get("metadata", {}) if _ci is not None else {}
        if self.current_positions_calibrated():
            f = meta.get("motor_calibration_file", "parameters_int.txt")
            self.lbl_axis.setText(f"Axis: motor-CALIBRATED  ({f}) — recompute uses it as-is")
            self.lbl_axis.setStyleSheet("font-weight:bold; color:#2f9e44;")
        else:
            applied = meta.get("motor_calibration_applied", None)
            extra = "" if applied is None else "  (no cal file at acquisition)"
            self.lbl_axis.setText(f"Axis: raw measured — recompute applies current calibration{extra}")
            self.lbl_axis.setStyleSheet("font-weight:bold; color:#e8590c;")
        # ROI/frame mean trace (toggle with the "ROI/frame mean" checkbox).
        if self.chk_show_mean.isChecked():
            if self.rois:
                r0, r1, c0, c1 = self._roi_bounds(self.rois[0], raw.shape[1], raw.shape[2])
                ifg = raw[:, r0:r1, c0:c1].mean(axis=(1, 2))
            else:
                ifg = raw.mean(axis=(1, 2))
            self.ifg_curve.setData(pos, ifg)
        else:
            self.ifg_curve.setData([], [])
        # One trace per clicked pixel (toggle with the "Pixels" checkbox), each
        # coloured to match its spectrum in the Spectra tab.
        self._clear_pixel_curves()
        if self.chk_show_pixels.isChecked():
            for i, (r, c) in enumerate(self.sel_pixels):
                if 0 <= r < raw.shape[1] and 0 <= c < raw.shape[2]:
                    cv = self.ifg_plot.plot(pos, raw[:, r, c],
                                            pen=pg.mkPen(_pixel_color(i), width=1),
                                            name=f"px({r},{c})")
                    self.ifg_px_curves.append(cv)
        lo_b, hi_b = float(np.min(pos)), float(np.max(pos))
        self.win_region.setBounds([lo_b, hi_b])
        lo, hi = self.win_region.getRegion()
        if not (lo_b <= lo < hi <= hi_b):
            self.win_region.blockSignals(True)
            self.win_region.setRegion([lo_b, hi_b])
            self.win_region.blockSignals(False)

    def _clear_pixel_curves(self):
        """Remove the per-pixel interferogram curves (and their legend entries)."""
        for cv in getattr(self, "ifg_px_curves", []):
            self.ifg_plot.removeItem(cv)
        self.ifg_px_curves = []

    def _clear_pixels(self):
        """Drop all clicked-pixel selections and refresh the spectra + interferogram."""
        self.sel_pixels = []
        self.update_spectra()
        self.update_interferogram()

    def _on_window_changed(self):
        lo, hi = self.win_region.getRegion()
        self.lbl_win.setText(f"FT window: {lo:.4f} … {hi:.4f} mm  (length {hi - lo:.4f} mm)")
        if self.recompute_on:
            self._cube_cache.clear()
            self.refresh_map(); self.update_spectra()
        elif self.chk_recompute.isEnabled():
            self.statusBar().showMessage(
                "Tick 'Recompute…' to apply this interferogram window.", 5000)
        self._update_phase()   # phase always tracks the FT window

    def _toggle_recompute(self, on):
        self.recompute_on = bool(on)
        if not on and self._saved_wavelengths is not None:
            self.wavelengths = self._saved_wavelengths
            self._sync_wl_slider(len(self.wavelengths))
        self._cube_cache.clear()
        self.refresh_map(); self.update_spectra()
        self._update_phase()

    def _recompute_if_on(self, *a):
        self._update_phase()   # apodization also drives the phase view
        if self.recompute_on:
            self._cube_cache.clear()
            self.refresh_map(); self.update_spectra()
        elif self.chk_recompute.isEnabled():
            self.statusBar().showMessage(
                "Tick 'Recompute map/spectra from the interferogram window' to apply "
                "apodization / window / λ / N-freq changes.", 5000)
        else:
            self.statusBar().showMessage(
                "This file has no stored raw interferogram — it cannot be recomputed "
                "(re-acquire with raw saved).", 5000)

    def _zpd_mm(self, pos, raw):
        from instruments.hyperspectral import find_centerburst
        ifg = raw.mean(axis=(1, 2))
        idx = find_centerburst(ifg - ifg.mean(), pos, DEFAULT_ZPD_MM, DEFAULT_ZPD_WINDOW_MM)
        return float(pos[idx])

    def _set_window(self, lo, hi):
        self.win_region.setRegion([float(lo), float(hi)])
        self._on_window_changed()

    def _win_full(self):
        pos, raw = self.current_interferogram()
        if pos is not None:
            self._set_window(pos.min(), pos.max())

    def _win_centre(self):
        pos, raw = self.current_interferogram()
        if pos is not None:
            z = self._zpd_mm(pos, raw); self._set_window(z - 0.2, z + 0.2)

    def _win_left(self):
        pos, raw = self.current_interferogram()
        if pos is not None:
            self._set_window(pos.min(), self._zpd_mm(pos, raw) - 0.05)

    def _win_right(self):
        pos, raw = self.current_interferogram()
        if pos is not None:
            self._set_window(self._zpd_mm(pos, raw) + 0.05, pos.max())

    def current_mask(self):
        zi = self._cur()
        return self.infos[zi]["mask"] if zi is not None else None

    def _invalidate_and_refresh(self, *a):
        self._cube_cache.clear()
        self.refresh_map(); self.update_spectra()

    # -- SVD denoise (scree plot -> choose components) ------------------------
    def _on_svd_toggled(self, on):
        """Show the explained-variance plot when SVD is enabled; the user can
        only choose k once the scree plot is visible."""
        self.svd_plot.setVisible(on)
        self.spin_k.setEnabled(on)
        if on:
            self._update_svd_plot()
        self._invalidate_and_refresh()      # rebuild cube with / without denoise

    def _on_svd_k_changed(self, *a):
        if not self.chk_svd.isChecked():
            return
        k = self.spin_k.value()
        self.svd_kline.setValue(k)
        cum = getattr(self, "_svd_cum", None)
        if cum is not None and len(cum):
            self.svd_plot.setTitle(
                f"{k} components → {cum[min(k, len(cum)) - 1] * 100:.1f}% of variance")
        self._invalidate_and_refresh()      # rebuild the cube from k components

    def _update_svd_plot(self):
        """Compute the singular-value spectrum of the current (pre-denoise) cube
        and plot cumulative explained variance vs number of components."""
        if not self.infos:
            return
        ci = self._cur()
        if ci is None:
            return
        wl, base = self._base_cube(ci)
        if base is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            _, _, cum = A.svd_explained_variance(base)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._svd_cum = cum
        x = np.arange(1, len(cum) + 1)
        self.svd_curve.setData(x, cum * 100.0)
        self.spin_k.blockSignals(True)
        self.spin_k.setMaximum(int(len(cum)))
        self.spin_k.setValue(min(self.spin_k.value(), len(cum)))
        self.spin_k.blockSignals(False)
        self.svd_kline.setValue(self.spin_k.value())
        k = self.spin_k.value()
        self.svd_plot.setTitle(
            f"{k} components → {cum[min(k, len(cum)) - 1] * 100:.1f}% of variance")

    # -- map display ----------------------------------------------------------
    def _on_map_changed(self, *a):
        # Switching map mode refreshes both the image and the spectra panel (so
        # e.g. the per-cluster spectra / continuum bands appear with their map).
        self.refresh_map()
        self.update_spectra()

    def _continuum_bands(self):
        """(line, left_shoulder, right_shoulder) wavelength bands, each (lo, hi)."""
        return (tuple(sorted((self.spin_line0.value(), self.spin_line1.value()))),
                tuple(sorted((self.spin_lsh0.value(), self.spin_lsh1.value()))),
                tuple(sorted((self.spin_rsh0.value(), self.spin_rsh1.value()))))

    def _on_continuum_changed(self, *a):
        if self.combo_map.currentText() == "Continuum line":
            self.refresh_map()
            self.update_spectra()          # move the shaded band overlays

    def _apply_cmap(self, *a):
        cm = _get_cmap(self.combo_cmap.currentText())
        if cm is not None:
            self.iv.setColorMap(cm)
        self._update_phase()

    def _show_rgb(self):
        """Render the false-colour RGB map in the left viewer (over the current
        λ range -- respects the λ-limit if it is on)."""
        if self.current_cube() is None:
            self.statusBar().showMessage("Load a cube first.")
            return
        self.combo_map.setCurrentText("RGB")
        self.refresh_map()

    def refresh_map(self, *a):
        cube = self.current_cube()
        if cube is None:
            return
        mode = self.combo_map.currentText()
        gamma = self.spin_gamma.value()
        # "Limit λ" restricts the bands the λ-aggregating maps use (a pure display
        # choice -- no recompute). The λ-slice slider is bounded to the same range.
        ccube, cwl = self._crop_slice(cube)
        if mode == "λ slice":
            i = min(max(self.wl_slider.value(), 0), cube.shape[0] - 1)
            self._set_map_image(self._gamma(cube[i], gamma))
        elif mode == "Peak λ":
            self._set_map_image(A.peak_wavelength_map(ccube, cwl))
        elif mode == "Peak intensity":
            self._set_map_image(self._gamma(A.peak_intensity_map(ccube), gamma))
        elif mode == "SAM":
            self._set_map_image(A.spectral_angle_map(ccube, self._reference_for_sam(ccube, cwl)))
        elif mode == "RGB":
            self.iv.setImage(self._rgb(ccube, cwl, gamma), autoLevels=False, levels=(0, 1))
        elif mode == "K-means":
            self._kmeans_map(ccube)        # cluster + store labels
            self.iv.setImage(self._labels_to_rgb(self._kmeans_labels, self._kmeans_k),
                             autoLevels=False, levels=(0, 1))
        elif mode == "Continuum line":
            # uses absolute wavelength bands -> the FULL cube/axis, not the λ-limit
            ln, ls, rs = self._continuum_bands()
            img = A.continuum_line_image(cube, self.wavelengths, ln, ls, rs)
            if img is None:
                self.statusBar().showMessage(
                    "Continuum bands are outside the dataset's wavelength range.")
            else:
                self._set_map_image(self._gamma(img, gamma))
        self.iv.ui.roiPlot.hide()
        self.lbl_wl.setText(f"{self.wavelengths[self.wl_slider.value()]:.3f} µm"
                            if mode == "λ slice" else f"[{mode}]")

    def _set_map_image(self, img, lockable=True):
        """Display a scalar map honoring the locked intensity scale. When not
        locked, autoscale and reflect the resulting min/max in the spin boxes so
        the user can grab and lock them."""
        if lockable and self.chk_lock_lev.isChecked():
            lo, hi = self.spin_lmin.value(), self.spin_lmax.value()
            if hi <= lo:
                hi = lo + 1e-9
            self.iv.setImage(np.asarray(img), autoLevels=False, levels=(lo, hi))
        else:
            self.iv.setImage(np.asarray(img), autoLevels=True)
            if lockable:
                self._sync_level_spins()

    def _sync_level_spins(self):
        lo, hi = self.iv.ui.histogram.getLevels()
        for s, v in ((self.spin_lmin, lo), (self.spin_lmax, hi)):
            s.blockSignals(True); s.setValue(float(v)); s.blockSignals(False)

    def _on_lock_levels(self, on):
        self.spin_lmin.setEnabled(on); self.spin_lmax.setEnabled(on)
        if on:                       # seed from whatever is currently displayed
            self._sync_level_spins()
        self.refresh_map()

    def _on_level_spin(self, *a):
        if self.chk_lock_lev.isChecked():
            self.refresh_map()

    def _levels_from_data(self):
        """Set min/max to the current map's full data range and lock."""
        cube = self.current_cube()
        if cube is None:
            return
        ccube, cwl = self._crop_slice(cube)
        mode = self.combo_map.currentText()
        if mode == "Peak intensity":
            img = A.peak_intensity_map(ccube)
        elif mode == "Peak λ":
            img = A.peak_wavelength_map(ccube, cwl)
        else:
            img = cube[min(self.wl_slider.value(), cube.shape[0] - 1)]
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
        self.chk_lock_lev.setChecked(True)
        for s, v in ((self.spin_lmin, lo), (self.spin_lmax, hi)):
            s.blockSignals(True); s.setValue(v); s.blockSignals(False)
        self.refresh_map()

    @staticmethod
    def _gamma(img, g):
        if g == 1.0:
            return img
        img = np.asarray(img, dtype=float)
        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
        if mx <= mn:
            return img
        return ((img - mn) / (mx - mn)) ** g

    def _rgb(self, cube, wl, gamma):
        """False-colour RGB exactly as in hsiAnalysis HyperspectralAnalysis_Spectrum:
        each pixel's spectrum is PROJECTED onto the R/G/B response curves
        (RGB_Transmission), i.e. channel = response · spectrum. The image is then
        scaled to its max and gamma-corrected. The visible response curves are
        mapped onto the dataset's band so MWIR data renders in colour."""
        from instruments.rgb_response import response_on_axis
        Rd, Gd, Bd = response_on_axis(np.asarray(wl, float))
        n_freq, h, w = cube.shape
        flat = np.asarray(cube, dtype=float).reshape(n_freq, -1)
        r = (Rd @ flat).reshape(h, w)
        g = (Gd @ flat).reshape(h, w)
        b = (Bd @ flat).reshape(h, w)
        rgb = np.stack([r, g, b], axis=-1)
        mx = float(np.nanmax(rgb))
        if mx > 0:
            rgb = rgb / mx                      # ImmagineRGB ./ max_image
        rgb = np.clip(rgb, 0.0, 1.0)
        return rgb ** gamma if gamma != 1.0 else rgb

    def _kmeans_map(self, cube):
        """Cluster pixels by spectral SHAPE (intensity-normalised) into k groups;
        return the (h, w) cluster-label map."""
        from scipy.cluster.vq import kmeans2
        n, h, w = cube.shape
        # subsample bands for speed on big cubes (clustering by spectral SHAPE
        # doesn't need every bin)
        step = max(1, n // 256)
        X = cube[::step].reshape(-1, h * w).T.astype(float)   # (pixels, bands)
        # L2-normalise each pixel spectrum to a unit vector. kmeans2 only does
        # (squared) EUCLIDEAN distance -- there is no cosine option in scipy.vq --
        # but for unit vectors ||a-b||^2 = 2(1 - cos theta), so Euclidean distance
        # on normalised spectra is monotonic in the COSINE similarity (= spectral
        # angle, as in SAM). Hence clustering is by spectral SHAPE, not intensity.
        # (Approximate spherical k-means: centroids are the plain mean of the unit
        # vectors and are NOT re-normalised each iteration, as true spherical
        # k-means would.)
        norm = np.linalg.norm(X, axis=1, keepdims=True); norm[norm == 0] = 1.0
        Xn = X / norm                                     # spectral shape
        k = self.spin_clusters.value()
        try:
            # 'points' init (random data points) avoids the cov/cholesky path;
            # 'warn' returns labels even if a cluster ends up empty.
            _, labels = kmeans2(Xn, k, seed=0, minit="points", missing="warn")
        except Exception as e:  # noqa: BLE001
            print("k-means failed:", e)
            labels = np.zeros(h * w, int)
        labels = labels.reshape(h, w)
        # Keep the labels so update_spectra can plot one average spectrum per
        # cluster, coloured to match this map.
        self._kmeans_labels = labels.astype(int)
        self._kmeans_k = int(k)
        return labels.astype(float)

    def _reference_for_sam(self, cube, wl):
        """The reference spectrum the SAM map is computed against, matched to the
        CURRENT cube's wavelength axis so it never goes out of sync."""
        # external file reference -> interpolate onto this cube's wavelengths
        ext = getattr(self, "sam_ref_ext", None)
        if ext is not None:
            rwl, rval = ext
            return np.interp(np.asarray(wl, float), rwl, rval)
        # chosen ROI reference -> recompute its average on THIS cube (axis-safe)
        sref = getattr(self, "sam_ref_roi", None)
        if sref is not None and sref in self.rois:
            return self._roi_spectrum(sref, cube)[0]
        # fallbacks: ROI 1 / clicked pixel / whole-frame average
        if self.rois:
            return self._roi_spectrum(self.rois[0], cube)[0]
        if self.sel_pixels:
            r, c = self.sel_pixels[0]
            _, h, w = cube.shape
            if 0 <= r < h and 0 <= c < w:
                return cube[:, r, c]
        return A.roi_average(cube, self.current_mask())[0]

    def _sam_ref_from_roi(self):
        """Use ROI 1's average spectrum as the SAM reference."""
        if not self.rois:
            self.statusBar().showMessage("Add an ROI first — its average becomes the SAM reference.")
            return
        self.sam_ref_ext = None
        self.sam_ref_roi = self.rois[0]
        self.lbl_sam.setText("SAM reference: ROI 1 average")
        self._show_sam()

    def _sam_ref_from_file(self):
        """Load an external reference spectrum (2 columns: wavelength_µm, value)."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Reference spectrum (wavelength_um, value)", "",
            "Text/CSV (*.csv *.txt *.dat);;All files (*)")
        if not path:
            return
        try:
            data = None
            for delim in (None, ",", ";", "\t"):
                try:
                    data = np.loadtxt(path, delimiter=delim, comments=("#", "%"))
                    if data.ndim == 2 and data.shape[1] >= 2:
                        break
                except Exception:  # noqa: BLE001
                    continue
            if data is None or data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("need 2 columns: wavelength_um, value")
            order = np.argsort(data[:, 0])
            rwl, rval = data[order, 0], data[order, 1]
            self.sam_ref_ext = (np.asarray(rwl, float), np.asarray(rval, float))
            self.sam_ref_roi = None
            self.lbl_sam.setText(f"SAM reference: {os.path.basename(path)} "
                                 f"({rwl.min():.3f}–{rwl.max():.3f} µm)")
            self._show_sam()
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"could not read reference spectrum: {e}")

    def _show_sam(self):
        """Switch the map to the spectral-angle map and render it."""
        self.combo_map.setCurrentText("SAM")
        self.refresh_map()

    # -- sliders / click ------------------------------------------------------
    def _on_wl(self, *a):
        # Update the λ label instantly; debounce the map/phase recompute so a
        # slider drag stays smooth (short delay -- a λ-slice is a cheap band pick).
        if self.wavelengths is not None and len(self.wavelengths):
            i = int(np.clip(self.wl_slider.value(), 0, len(self.wavelengths) - 1))
            self.lbl_wl.setText(f"{np.asarray(self.wavelengths)[i]:.3f} µm")
        self._wl_timer.start(_WL_DEBOUNCE_MS)

    def _do_wl_refresh(self):
        if self.combo_map.currentText() == "λ slice":
            self.refresh_map()
        self._update_phase()

    # -- phase view (complex DFT at one wavelength) ---------------------------
    def _on_tab_changed(self, *a):
        # Phase is no longer a tab (it is the toolbar-toggled stack page); nothing
        # to do here, kept so the currentChanged connection stays valid.
        pass

    # The three view toggles share an exclusive QActionGroup, so exactly one is
    # active; switching one off fires when another is switched on. Each handler
    # just re-derives the page from the current checked states via _show_view.
    def _toggle_hyper_view(self, on):
        """Back to the main hypercube view."""
        if on:
            self._show_view()

    def _toggle_phase_view(self, on):
        """Show the phase-only page (amplitude + wrapped phase)."""
        self.phase_mode = bool(on)
        self._show_view()
        if on:
            self._mirror_phase_sliders()
            self._update_phase()

    def _toggle_stokes_view(self, on):
        """Show the embedded Stokes polarimetry page. On entry, (re)fill it from
        the folder loaded in the analyzer so the user does not pick a folder
        again."""
        if on and self._stokes_dirty:
            self._refresh_stokes_from_main()
        self._show_view()

    def _refresh_stokes_from_main(self):
        """Feed the Stokes panel the .npz files currently loaded in the analyzer
        (File ▸ Load folder), so it auto-assigns the four QWP slots by angle."""
        if self.stokes_app is None:
            return
        if not self.infos:
            self.stokes_app.status.showMessage(
                "No dataset loaded — use File ▸ Load folder in the analyzer first.")
            return
        paths = list(dict.fromkeys(i["path"] for i in self.infos))   # unique, ordered
        self.stokes_app.populate_from_paths(paths)
        self._stokes_dirty = False

    def _show_view(self):
        """Pick the central page from the view toggles (Hypercube = main view)."""
        if getattr(self, "stack", None) is None:
            return                                  # toolbar built before the stack
        if self.act_phase.isChecked():
            self.stack.setCurrentWidget(self.phase_page)
        elif self.act_stokes is not None and self.act_stokes.isChecked():
            self.stack.setCurrentWidget(self.stokes_app)
        else:
            self.stack.setCurrentWidget(self.main_split)

    def _on_ph_z_combo(self, idx):
        """Phase-page Z dropdown -> select that Z (drives the whole refresh)."""
        if idx >= 0:
            self.z_slider.setValue(idx)

    def _mirror_phase_sliders(self):
        """Copy the master λ/angle sliders' ranges/values/visibility onto the
        phase-page controls, and fill the Z dropdown from the loaded positions
        (signals blocked so it never fights the two-way connections)."""
        for src, dst in ((self.wl_slider, self.ph_wl), (self.angle_slider, self.ph_a)):
            dst.blockSignals(True)
            dst.setMinimum(src.minimum()); dst.setMaximum(src.maximum())
            dst.setValue(src.value())
            dst.blockSignals(False)
        # Z dropdown: one entry per position (in mm), current = the master z index.
        self.ph_z_combo.blockSignals(True)
        self.ph_z_combo.clear()
        for z in self._z_axis:
            self.ph_z_combo.addItem("--" if z != z else f"{z:.4f} mm")
        self.ph_z_combo.setCurrentIndex(min(max(self.z_slider.value(), 0),
                                            len(self._z_axis) - 1))
        self.ph_z_combo.blockSignals(False)
        # Base visibility on the data (the main panel is hidden while the phase
        # page is shown, so its rows' isVisible() would read False here).
        self.ph_zrow.setVisible(len(self._z_axis) > 1)
        self.ph_arow.setVisible(len(self._a_axis) > 1)
        self.ph_lbl_wl.setText(self.lbl_wl.text())
        self.ph_lbl_a.setText(self.lbl_a.text())

    def _on_phase_cmap(self, *a):
        """Recolor the wrapped-phase map (colormap only -- no DFT recompute)."""
        cm = _phase_cmap(self.ph_cmap_combo.currentText())
        self.ph_cbar_phase.setColorMap(cm)
        self.ph_img_phase.setColorMap(cm)

    def _update_phase(self, *a):
        """Recompute the complex per-pixel DFT at the current λ / plane and draw
        the XY amplitude + wrapped-phase maps. Lazy: only runs while the phase-only
        page is shown (a full DFT per λ move is otherwise wasted)."""
        if not getattr(self, "phase_mode", False):
            return
        # Keep the phase-page controls in step with the master view.
        self.ph_lbl_a.setText(self.lbl_a.text())
        if (self.ph_z_combo.count() and self.ph_z_combo.currentIndex() != self.z_slider.value()):
            self.ph_z_combo.blockSignals(True)
            self.ph_z_combo.setCurrentIndex(min(max(self.z_slider.value(), 0),
                                                self.ph_z_combo.count() - 1))
            self.ph_z_combo.blockSignals(False)
        if self.wavelengths is not None and len(self.wavelengths):
            _i = int(np.clip(self.wl_slider.value(), 0, len(self.wavelengths) - 1))
            self.ph_lbl_wl.setText(f"{float(np.asarray(self.wavelengths)[_i]):.3f} µm")
        if not self.infos or self.wavelengths is None:
            return
        pos, raw = self.current_interferogram()
        if raw is None or pos is None:
            self.ph_img_amp.clear(); self.ph_img_phase.clear()
            self._last_phase = None
            self.ph_info.setText("This file has no stored raw interferogram — "
                                 "the phase view is unavailable (re-acquire with raw saved).")
            return
        idx = int(np.clip(self.wl_slider.value(), 0, len(self.wavelengths) - 1))
        lam = float(np.asarray(self.wavelengths)[idx])
        if self.proc is None:
            self.proc = HyperspectralProcessor()
        try:
            cmap, info = self.proc.compute_complex_map(
                pos, raw, lam, apod_type=self.r_apod.currentText(),
                ft_window_mm=self.win_region.getRegion(),
                expected_zero_mm=DEFAULT_ZPD_MM, search_mm=DEFAULT_ZPD_WINDOW_MM,
                positions_calibrated=self.current_positions_calibrated(),
                reference_cube=self._phase_ref_cube(raw.shape))
        except Exception as e:  # noqa: BLE001
            self.ph_info.setText(f"Phase compute failed: {e}")
            return
        if cmap is None:
            self.ph_info.setText("Too few scan positions for a DFT.")
            return
        amp = np.abs(cmap).astype(np.float32)
        phase = np.angle(cmap).astype(np.float32)
        amax = float(amp.max()) if amp.size else 0.0
        hi = amax if amax > 0 else 1.0
        thr = self.ph_thresh.value() / 100.0
        ph_masked = phase.copy()
        if amax > 0:
            ph_masked[amp < thr * amax] = np.nan     # NaN -> transparent (grey bg)
        cm_amp = _get_cmap(self.combo_cmap.currentText())
        self.ph_img_amp.setImage(amp, autoLevels=False)
        self.ph_cbar_amp.setColorMap(cm_amp)
        self.ph_cbar_amp.setLevels((0.0, hi))
        self.ph_img_phase.setImage(ph_masked, autoLevels=False, levels=(-np.pi, np.pi))
        self._last_phase = (amp, phase)
        axis_txt = ("calibrated axis (as-is)" if self.current_positions_calibrated()
                    else "raw axis + motor cal")
        corr = ("BACKGROUND-CORRECTED phase" if info.get("phase_corrected")
                else "raw phase (interferometer phase included)")
        lo_w, hi_w = self.win_region.getRegion()
        self.ph_info.setText(
            f"λ = {lam:.4f} µm   |   {corr}   |   ZPD {info['center_mm']:.4f} mm   |   "
            f"FT window {lo_w:.3f}–{hi_w:.3f} mm   |   apod {self.r_apod.currentText()}   |   "
            f"{axis_txt}   |   phase masked below {self.ph_thresh.value():.3g}% of max amplitude")

    def _export_phase(self):
        lp = getattr(self, "_last_phase", None)
        if lp is None:
            self.statusBar().showMessage("No phase map yet — open the Phase tab first.", 4000)
            return
        amp, phase = lp
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export amplitude + phase", "phase_map.tsv",
            "Tab-separated (*.tsv *.txt)")
        if not path:
            return
        h, w = amp.shape
        rows, cols = np.mgrid[0:h, 0:w]
        data = np.column_stack([cols.ravel(), rows.ravel(),
                                amp.ravel(), phase.ravel()])
        np.savetxt(path, data, fmt="%.6g", delimiter="\t",
                   header="col\trow\tamplitude\tphase_rad", comments="")
        self.statusBar().showMessage(f"Saved {data.shape[0]} rows to {path}", 5000)

    # -- background phase correction ------------------------------------------
    def _phase_load_bkg(self):
        """Load a background interferogram (.npz with raw_interferogram) to phase
        the measurement against (per-pixel instrumental-phase removal)."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load background interferogram", "", "NumPy archive (*.npz)")
        if not path:
            return
        try:
            with np.load(path, allow_pickle=True) as d:
                raw, pos, _cal = _read_raw(d, None)
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"could not read background: {e}", 6000)
            return
        if raw is None or pos is None:
            self.lbl_phase_bkg.setText("Background: that file has no raw interferogram.")
            return
        self.phase_bkg = (np.asarray(raw, dtype=np.float32), np.asarray(pos, float), path)
        self.chk_phase_bkg.setEnabled(True)
        self._describe_phase_bkg()
        if self.chk_phase_bkg.isChecked():
            self._on_phase_bkg_toggled(True)          # already on -> just refresh
        else:
            self.chk_phase_bkg.setChecked(True)       # -> toggles on + refreshes

    def _describe_phase_bkg(self):
        if self.phase_bkg is None:
            self.lbl_phase_bkg.setText("Background: none loaded."); return
        raw, _pos, path = self.phase_bkg
        compat = ""
        if self.infos:
            cur = self.current_interferogram()[1]          # cropped to the analysis ROI
            if cur is not None:
                refc = self._work_crop3(raw)               # crop the ref the same way
                compat = ("  —  compatible ✓" if refc.shape == cur.shape
                          else f"  —  SHAPE MISMATCH ✗ (data is {tuple(cur.shape)})")
        state = "ON" if self.phase_bkg_on else "off"
        self.lbl_phase_bkg.setText(
            f"Background: {os.path.basename(path)}  {tuple(raw.shape)}{compat}   [correction {state}]")

    def _on_phase_bkg_toggled(self, on):
        self.phase_bkg_on = bool(on) and self.phase_bkg is not None
        self._describe_phase_bkg()
        if self.recompute_on:          # main view re-derives phased spectra
            self._invalidate_and_refresh()
        self._update_phase()           # phase page shows the corrected phase

    def _phase_stack_save(self):
        """Phase-correct EVERY cube in the loaded (Z × θ) stack with the loaded
        background and save the phased spectrum cubes to a new folder."""
        if self.phase_bkg is None:
            self.statusBar().showMessage("Load a background first.", 4000)
            return
        if not self.infos:
            return
        dest = QtWidgets.QFileDialog.getExistingDirectory(self, "Save phased Z×θ stack to…")
        if not dest:
            return
        ref_raw = self.phase_bkg[0]
        if self.proc is None:
            self.proc = HyperspectralProcessor()
        wl0, wl1 = self.r_wl0.value(), self.r_wl1.value()
        apod = self.r_apod.currentText(); win = list(self.win_region.getRegion())
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        saved = skipped = 0
        try:
            for k, info in enumerate(self.infos):
                self.statusBar().showMessage(f"phasing {k+1}/{len(self.infos)}…")
                QtWidgets.QApplication.processEvents()
                mi = info.get("map_index")
                with np.load(info["path"], allow_pickle=True) as d:
                    raw, pos, calibrated = _read_raw(d, mi)
                    meta = _file_meta(d)
                    keep = {key: d[key] for key in (
                        "z_value_mm", "z_unit", "angle_value_deg", "angle_unit",
                        "background", "background_subtracted", "saturation_mask",
                        "twins_positions_mm", "twins_positions_calibrated_mm")
                        if key in d.files}
                if raw is None or pos is None or ref_raw.shape != raw.shape:
                    skipped += 1
                    continue
                nfreq = resolve_n_points(len(pos), manual=self.r_nfreq.value())
                wl, cube = self.proc.compute_hyperspectral(
                    pos, raw, wl_start=wl0, wl_stop=wl1, n_freq=nfreq, apod_type=apod,
                    ft_window_mm=win, expected_zero_mm=DEFAULT_ZPD_MM,
                    search_mm=DEFAULT_ZPD_WINDOW_MM, positions_calibrated=calibrated,
                    reference_cube=ref_raw)
                if cube is None:
                    skipped += 1
                    continue
                meta = dict(meta)
                meta.update(phase_corrected=True,
                            phase_background=os.path.basename(self.phase_bkg[2]),
                            recompute_window_mm=win, recompute_apod=apod,
                            recompute_nfreq=int(nfreq))
                kw = dict(wavelengths=wl, spectrum_cube=cube.astype(np.float32),
                          raw_interferogram=np.asarray(raw, np.float32),
                          metadata=np.array(meta, dtype=object),
                          metadata_json=json.dumps(meta, default=str, indent=2))
                kw.update(keep)
                base = os.path.splitext(os.path.basename(info["path"]))[0]
                np.savez(os.path.join(dest, base + ".npz"), **kw)  # uncompressed: fast write
                saved += 1
            msg = f"phased + saved {saved} cube(s) to {dest}"
            if skipped:
                msg += f"  ({skipped} skipped — no raw / shape mismatch)"
            self.statusBar().showMessage(msg)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_click(self, ev):
        cube = self.current_cube()
        if cube is None:
            return
        ii = self.iv.getImageItem()
        if not ii.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = ii.mapFromScene(ev.scenePos())
        col, row = int(p.x()), int(p.y())
        _, h, w = cube.shape
        if 0 <= row < h and 0 <= col < w:
            px = (row, col)
            # Ctrl/Shift-click adds (or toggles off) a pixel; a plain click selects
            # just that one -- so you can overlay several pixels or reset to one.
            mods = ev.modifiers()
            add = bool(mods & (QtCore.Qt.KeyboardModifier.ControlModifier
                               | QtCore.Qt.KeyboardModifier.ShiftModifier))
            if add:
                if px in self.sel_pixels:
                    self.sel_pixels.remove(px)
                else:
                    self.sel_pixels.append(px)
            else:
                self.sel_pixels = [px]
            self.update_spectra()
            self.update_interferogram()

    # -- ROIs -----------------------------------------------------------------
    def add_roi(self):
        cube = self.current_cube()
        if cube is None:
            return
        _, h, w = cube.shape
        s = max(4, min(h, w) // 5)
        col = _ROI_COLORS[len(self.rois) % len(_ROI_COLORS)]
        roi = pg.RectROI([w // 2 - s // 2, h // 2 - s // 2], [s, s],
                         pen=pg.mkPen(col, width=2), removable=True)
        roi.sigRegionChanged.connect(self._on_roi_changed)
        self.iv.getView().addItem(roi)
        self.rois.append(roi)
        self.update_spectra()

    def clear_rois(self):
        for roi in self.rois:
            self.iv.getView().removeItem(roi)
        self.rois = []
        self.ref_spectrum = None
        self.ref_roi = None
        self.bg_roi = None; self._bg_sig = None
        self.lbl_bg.setText("Background: none")
        self.refresh_map(); self.update_spectra()

    def _on_roi_changed(self, *a):
        """ROI moved/resized -> update spectra; also re-render the map if it
        depends on an ROI (background subtraction)."""
        self.update_spectra()
        if self.chk_bg.isChecked():
            self.refresh_map()          # _apply_background re-subtracts if the bg ROI moved

    def _bg_from_roi(self):
        """Use ROI 1's average spectrum as the background to subtract."""
        if not self.rois:
            self.statusBar().showMessage("Add an ROI over a background region first.")
            return
        self.bg_roi = self.rois[0]
        self.lbl_bg.setText("Background: ROI 1 average (subtracted)")
        self._bg_sig = None
        self.chk_bg.blockSignals(True); self.chk_bg.setChecked(True); self.chk_bg.blockSignals(False)
        self.refresh_map(); self.update_spectra()

    def _on_bg_toggled(self, *a):
        self._bg_sig = None             # force re-evaluation (no expensive reload)
        self.refresh_map(); self.update_spectra()

    def set_reference(self):
        if not self.rois:
            self.statusBar().showMessage("Add an ROI first; ROI 1 becomes the reference.")
            return
        self.ref_roi = self.rois[0]   # spectrum recomputed per refresh (axis-safe)
        self.statusBar().showMessage("ROI 1 set as reflectivity/transmission reference.")
        self.update_spectra()

    def add_poly_roi(self):
        cube = self.current_cube()
        if cube is None:
            return
        _, h, w = cube.shape
        s = max(4, min(h, w) // 5); cx, cy = w / 2, h / 2
        col = _ROI_COLORS[len(self.rois) % len(_ROI_COLORS)]
        pts = [[cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s]]
        roi = pg.PolyLineROI(pts, closed=True, pen=pg.mkPen(col, width=2), removable=True)
        roi.sigRegionChanged.connect(self._on_roi_changed)
        self.iv.getView().addItem(roi)
        self.rois.append(roi)
        self.update_spectra()

    def _roi_bounds(self, roi, h, w):
        x0, y0 = roi.pos(); sw, sh = roi.size()
        c0, c1 = sorted((int(round(x0)), int(round(x0 + sw))))
        r0, r1 = sorted((int(round(y0)), int(round(y0 + sh))))
        return (max(0, r0), min(h, max(r0 + 1, r1)),
                max(0, c0), min(w, max(c0 + 1, c1)))

    def _poly_vertices(self, roi):
        return [(float(q.x()), float(q.y()))
                for q in (roi.mapToParent(p) for p in roi.getState()["points"])]

    @staticmethod
    def _poly_mask(verts, h, w):
        ys, xs = np.mgrid[0:h, 0:w]
        xs = xs.ravel().astype(float) + 0.5; ys = ys.ravel().astype(float) + 0.5
        inside = np.zeros(xs.shape, bool); n = len(verts); j = n - 1
        for i in range(n):
            xi, yi = verts[i]; xj, yj = verts[j]
            cond = ((yi > ys) != (yj > ys)) & (xs < (xj - xi) * (ys - yi) / ((yj - yi) + 1e-12) + xi)
            inside ^= cond; j = i
        return inside.reshape(h, w)

    def _roi_spectrum(self, roi, cube):
        n, h, w = cube.shape
        if isinstance(roi, pg.PolyLineROI):
            mask = self._poly_mask(self._poly_vertices(roi), h, w)
            if not mask.any():
                return np.zeros(n), np.zeros(n)
            block = cube[:, mask]
        else:
            r0, r1, c0, c1 = self._roi_bounds(roi, h, w)
            block = cube[:, r0:r1, c0:c1].reshape(n, -1)
        return block.mean(axis=1), block.std(axis=1)

    # -- spectra plot ---------------------------------------------------------
    def _post(self, y):
        """Apply normalisation + derivative to a spectrum."""
        wl = self.wavelengths
        norm = self.combo_norm.currentText()
        y = np.asarray(y, dtype=float)
        if norm.startswith("area"):
            trap = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
            area = float(trap(np.abs(y), wl)) if trap else float(np.sum(np.abs(y)))
            if area:
                y = y / area
        elif (norm.startswith("reflect") or norm.startswith("transmission")) \
                and self.ref_spectrum is not None:
            ref = np.asarray(self.ref_spectrum, dtype=float)
            if ref.shape == y.shape:
                y = y / np.where(ref == 0, np.nan, ref)
        order = {"raw": 0, "d/dλ": 1, "d²/dλ²": 2}.get(self.combo_deriv.currentText(), 0)
        for _ in range(order):
            y = np.gradient(y, wl)
        return y

    def update_spectra(self, *a):
        self.spec.clear()
        if self.wavelengths is None:
            return
        cube = self.current_cube()
        if cube is None:
            return
        # reference spectrum for reflectivity/transmission, recomputed on the
        # CURRENT cube so it always matches the (possibly recomputed) λ axis
        self.ref_spectrum = None
        if self.ref_roi is not None and self.ref_roi in self.rois:
            self.ref_spectrum = self._roi_spectrum(self.ref_roi, cube)[0]
        wl = self.wavelengths
        if wl is None or len(wl) != cube.shape[0]:
            return                             # axis/cube out of sync -> skip safely
        # X axis = the dataset's spectral limits (or the λ-limit window) -- never
        # the default [0,1]. Y auto-scales to the plotted spectra.
        if self.chk_crop.isChecked():
            lo, hi = sorted((self.spin_crop0.value(), self.spin_crop1.value()))
        else:
            lo, hi = float(np.min(wl)), float(np.max(wl))
        self.spec.setXRange(lo, hi, padding=0.02)
        self.spec.enableAutoRange(axis="y")
        # ROI average spectra (mean + shaded ±std band)
        for i, roi in enumerate(self.rois):
            mean, std = self._roi_spectrum(roi, cube)
            y = self._post(mean)
            col = _ROI_COLORS[i % len(_ROI_COLORS)]
            cu = self.spec.plot(wl, self._post(mean + std), pen=None)
            cl = self.spec.plot(wl, self._post(mean - std), pen=None)
            band = pg.FillBetweenItem(cu, cl, brush=pg.mkBrush(QtGui.QColor(col)))
            band.setOpacity(0.18); self.spec.addItem(band)
            self.spec.plot(wl, y, pen=pg.mkPen(col, width=2), name=f"ROI {i+1}")
        # K-means: one average spectrum per cluster, coloured to match the map.
        if (self.combo_map.currentText() == "K-means"
                and self._kmeans_labels is not None
                and self._kmeans_labels.shape == cube.shape[1:]):
            self._plot_cluster_spectra(cube, wl)
        # Continuum-line mode: shade the line band + the two shoulder bands so the
        # user can position them on the resonance / clean continuum.
        if self.combo_map.currentText() == "Continuum line":
            ln, ls, rs = self._continuum_bands()
            for (lo, hi), rgba in ((ln, (232, 89, 12, 45)),      # line = orange
                                   (ls, (77, 171, 247, 40)),     # shoulders = blue
                                   (rs, (77, 171, 247, 40))):
                reg = pg.LinearRegionItem([lo, hi], movable=False,
                                          brush=pg.mkBrush(*rgba), pen=pg.mkPen(None))
                reg.setZValue(-10)
                self.spec.addItem(reg)
        # selected pixel spectra (one dashed trace per clicked pixel, coloured to
        # match its interferogram in the Interferogram tab)
        _, h, w = cube.shape
        shown = []
        for i, (r, c) in enumerate(self.sel_pixels):
            if 0 <= r < h and 0 <= c < w:
                self.spec.plot(wl, self._post(cube[:, r, c]),
                               pen=pg.mkPen(_pixel_color(i), width=1,
                                            style=QtCore.Qt.PenStyle.DashLine),
                               name=f"px({r},{c})")
                shown.append(f"({r},{c})")
        if shown:
            self.lbl_pixel.setText("Pixels: " + ", ".join(shown)
                                   + "   (Ctrl/Shift-click to add)")
        else:
            self.lbl_pixel.setText("Click the map for a pixel spectrum; "
                                   "Ctrl/Shift-click to add more.")
        # keep the interferogram's ROI-mean curve in step with ROI 1
        if hasattr(self, "ifg_curve"):
            self.update_interferogram()

    def _plot_cluster_spectra(self, cube, wl):
        """Plot one average spectrum per K-means cluster, coloured to match the
        cluster colours in the map (same colormap, label 0..k-1). The raw means
        are cached by (cube, labels) so only _post (norm/derivative) re-runs."""
        lab = self._kmeans_labels
        k = max(1, int(self._kmeans_k))
        key = (id(cube), id(lab), k, cube.shape)
        if self._cluster_key != key:
            flat = np.asarray(cube, float).reshape(cube.shape[0], -1)
            flab = lab.ravel()
            self._cluster_means = []
            for c in range(k):
                sel = flab == c
                npx = int(sel.sum())
                mean = flat[:, sel].mean(axis=1) if npx else None
                self._cluster_means.append((c, npx, mean))
            self._cluster_key = key
        for c, npx, mean in self._cluster_means:
            if mean is None:
                continue
            col = _KMEANS_COLORS[c % len(_KMEANS_COLORS)]   # matches the map
            self.spec.plot(wl, self._post(mean), pen=pg.mkPen(col, width=2),
                           name=f"cluster {c} (n={npx})")

    @staticmethod
    def _labels_to_rgb(labels, k):
        """(h, w, 3) image colouring each K-means label with the categorical
        palette (so clusters are distinct and visible, matching the spectra)."""
        rgb = np.zeros(np.asarray(labels).shape + (3,), dtype=float)
        for c in range(max(1, int(k))):
            qc = QtGui.QColor(_KMEANS_COLORS[c % len(_KMEANS_COLORS)])
            rgb[labels == c] = (qc.redF(), qc.greenF(), qc.blueF())
        return rgb

    def batch_recompute(self):
        """Recompute every Z from its raw interferogram with the current window/
        apod/λ/N-freq settings and save a new .npz series."""
        if not self.infos:
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(self, "Save recomputed Z-series to…")
        if not out:
            return
        win = list(self.win_region.getRegion())
        wl0, wl1, nfset = self.r_wl0.value(), self.r_wl1.value(), self.r_nfreq.value()
        apod = self.r_apod.currentText()
        if self.proc is None:
            self.proc = HyperspectralProcessor()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        saved = 0
        try:
            for k, info in enumerate(self.infos):
                self.statusBar().showMessage(
                    f"recomputing {k+1}/{len(self.infos)} (z={info['z']:.3f} mm)…")
                QtWidgets.QApplication.processEvents()
                pos = raw = mask = None; meta = {}; calibrated = False
                mi = info.get("map_index")
                with np.load(info["path"], allow_pickle=True) as d:
                    raw, pos, calibrated = _read_raw(d, mi)
                    if raw is None:
                        continue
                    meta = _file_meta(d)
                    if "saturation_mask" in d.files:
                        mask = np.asarray(d["saturation_mask"])
                    elif "saturation_masks" in d.files and mi is not None:
                        mask = np.asarray(d["saturation_masks"][mi])
                if pos is None or raw is None:
                    continue
                nfreq = resolve_n_points(len(pos), manual=nfset)
                wl, cube = self.proc.compute_hyperspectral(
                    pos, raw, wl_start=wl0, wl_stop=wl1, n_freq=nfreq, apod_type=apod,
                    ft_window_mm=win, expected_zero_mm=DEFAULT_ZPD_MM,
                    search_mm=DEFAULT_ZPD_WINDOW_MM, positions_calibrated=calibrated)
                meta.update(recompute_window_mm=win, recompute_apod=apod,
                            recompute_wl_um=[wl0, wl1], recompute_nfreq=int(nfreq))
                kw = dict(wavelengths=wl, spectrum_cube=cube.astype(np.float32),
                          z_value_mm=info["z"], z_unit="mm",
                          metadata=np.array(meta, dtype=object),
                          metadata_json=json.dumps(meta, default=str, indent=2))
                if mask is not None:
                    kw["saturation_mask"] = np.asarray(mask, bool)
                base = os.path.splitext(os.path.basename(info["path"]))[0]
                np.savez(os.path.join(out, base + "_rewin.npz"), **kw)  # uncompressed: fast write
                saved += 1
            self.statusBar().showMessage(f"recomputed + saved {saved} position(s) to {out}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # -- metadata + exports ---------------------------------------------------
    def _update_meta(self):
        zi = self._cur()
        if zi is not None:
            self.meta.setPlainText(json.dumps(self.infos[zi].get("metadata", {}),
                                              default=str, indent=2))
        else:
            self.meta.setPlainText("")

    def export_image(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save map image", "map.png",
                                                        "PNG (*.png);;TIFF (*.tif)")
        if path:
            self.iv.getImageItem().getViewBox().scene().items()  # noop, ensure rendered
            exp = pg.exporters.ImageExporter(self.iv.getView())
            exp.export(path)
            self.statusBar().showMessage(f"saved {os.path.basename(path)}")

    # -- RGB pickers + GIF ----------------------------------------------------
    def _frame_to_pil(self):
        """Grab the currently displayed map (colormap + orientation as on screen)
        as a PIL image, via the image view widget."""
        from PIL import Image
        qimg = self.iv.ui.graphicsView.grab().toImage().convertToFormat(
            QtGui.QImage.Format.Format_RGBA8888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits(); ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape(h, w, 4).copy()
        return Image.fromarray(arr[..., :3])

    def export_gif(self):
        """Animate the map over wavelength (or Z) and save an animated GIF, using
        the current colormap, gamma and intensity scale."""
        cube = self.current_cube()
        if not self.infos or cube is None:
            self.statusBar().showMessage("Load a cube first.")
            return
        axes = (["Wavelength (λ sweep)"]
                + (["Z position sweep"] if len(self._z_axis) > 1 else [])
                + (["Angle sweep"] if len(self._a_axis) > 1 else []))
        axis, ok = QtWidgets.QInputDialog.getItem(
            self, "Export GIF", "Animate over:", axes, 0, False)
        if not ok:
            return
        fps, ok = QtWidgets.QInputDialog.getInt(self, "Export GIF", "Frames per second:",
                                                12, 1, 60)
        if not ok:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export GIF", "animation.gif",
                                                        "GIF (*.gif)")
        if not path:
            return
        gamma = self.spin_gamma.value()
        save_band = self.wl_slider.value()
        save_z, save_a = self.z_slider.value(), self.angle_slider.value()
        # Consistent intensity scale across frames (no flicker): the locked levels
        # if set, else the global data range of the current cube.
        if self.chk_lock_lev.isChecked():
            lo, hi = self.spin_lmin.value(), self.spin_lmax.value()
        else:
            lo, hi = float(np.nanmin(cube)), float(np.nanmax(cube))
        if hi <= lo:
            hi = lo + 1e-9

        def render(img2d):
            x = np.clip((np.asarray(img2d, float) - lo) / (hi - lo), 0, 1)
            if gamma != 1.0:
                x = x ** gamma
            self.iv.setImage(x, autoLevels=False, levels=(0, 1))
            QtWidgets.QApplication.processEvents()
            return self._frame_to_pil()

        frames = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            if axis.startswith("Wavelength"):
                n = cube.shape[0]
                step = max(1, n // 150)            # cap ~150 frames
                bands = list(range(0, n, step))
                for k, i in enumerate(bands):
                    self.statusBar().showMessage(f"GIF frame {k+1}/{len(bands)}…")
                    frames.append(render(cube[i]))
            else:                                   # Z or Angle sweep, at current band
                band = self.wl_slider.value()
                if axis.startswith("Z"):             # sweep Z at the current angle
                    ai = self.angle_slider.value()
                    idxs = [self._grid.get((zi, ai)) for zi in range(len(self._z_axis))]
                else:                                # sweep angle at the current Z
                    zi = self.z_slider.value()
                    idxs = [self._grid.get((zi, ai)) for ai in range(len(self._a_axis))]
                idxs = [k for k in idxs if k is not None]
                for j, idx in enumerate(idxs):
                    self.statusBar().showMessage(f"GIF frame {j+1}/{len(idxs)}…")
                    c = _load_cube(self.infos[idx]["path"], self.infos[idx].get("map_index"))
                    if c is None:
                        continue
                    frames.append(render(c[min(band, c.shape[0] - 1)]))
            frames = [f for f in frames if f is not None]
            if frames:
                frames[0].save(path, save_all=True, append_images=frames[1:],
                               duration=int(round(1000 / fps)), loop=0, disposal=2)
                self.statusBar().showMessage(
                    f"saved {os.path.basename(path)} ({len(frames)} frames)")
            else:
                self.statusBar().showMessage("GIF export produced no frames.")
        finally:
            self.wl_slider.blockSignals(True); self.wl_slider.setValue(save_band)
            self.wl_slider.blockSignals(False)
            for slider, v in ((self.z_slider, save_z), (self.angle_slider, save_a)):
                slider.blockSignals(True); slider.setValue(v); slider.blockSignals(False)
            self.refresh_map()
            QtWidgets.QApplication.restoreOverrideCursor()

    def export_spectra(self):
        """Export the currently plotted ROI/pixel spectra (current Z) to CSV."""
        cube = self.current_cube()
        if cube is None:
            return
        cols = [self.wavelengths]; header = ["wavelength_um"]
        for i, roi in enumerate(self.rois):
            cols.append(self._post(self._roi_spectrum(roi, cube)[0])); header.append(f"ROI{i+1}")
        _, h, w = cube.shape
        for r, c in self.sel_pixels:
            if 0 <= r < h and 0 <= c < w:
                cols.append(self._post(cube[:, r, c])); header.append(f"px_{r}_{c}")
        if len(cols) == 1:
            self.statusBar().showMessage("Nothing to export — add an ROI or click a pixel.")
            return
        zi = self._cur(); info = self.infos[zi] if zi is not None else {}
        z = info.get("z", float("nan")); a = info.get("angle", float("nan"))
        tag = f"z{z:.3f}mm" if z == z else "map"
        if a == a:
            tag += f"_a{a:.1f}deg"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export spectra", f"spectra_{tag}.csv", "CSV (*.csv)")
        if path:
            np.savetxt(path, np.column_stack(cols), delimiter=",",
                       header=",".join(header), comments="")
            self.statusBar().showMessage(f"saved {os.path.basename(path)}")

    def export_roi_vs_z(self):
        """ROI-1 average spectrum at EVERY Z (loads each cube) -> CSV."""
        if not self.rois or not self.infos:
            self.statusBar().showMessage("Add an ROI first.")
            return
        roi = self.rois[0]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export ROI vs Z",
                                                        "roi1_vs_z.csv", "CSV (*.csv)")
        if not path:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            nref = len(self.wavelengths)
            cols = [self.wavelengths]; header = ["wavelength_um"]
            skipped = 0
            for info in self.infos:
                cube = _load_cube(info["path"], info.get("map_index"))
                if cube is None:
                    continue
                spec = self._roi_spectrum(roi, cube)[0]
                if spec.shape[0] != nref:      # mixed n_freq -> can't share one wl axis
                    skipped += 1
                    continue
                cols.append(spec); header.append(f"z{info['z']:.4f}mm")
            np.savetxt(path, np.column_stack(cols), delimiter=",",
                       header=",".join(header), comments="")
            msg = f"saved {os.path.basename(path)}"
            if skipped:
                msg += f" ({skipped} Z skipped: spectral length != {nref})"
            self.statusBar().showMessage(msg)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import pyqtgraph.exporters  # noqa: F401  (register exporters)

    argv = [a for a in sys.argv[1:] if a != "--selftest"]
    selftest = "--selftest" in sys.argv
    paths = [a for a in argv if os.path.exists(a)]

    app = QtWidgets.QApplication(sys.argv)
    win = ZSeriesAnalyzer()

    def _open(ps):
        if len(ps) == 1 and os.path.isdir(ps[0]):    # a folder -> search it + subfolders
            win.load_paths(sorted(glob.glob(os.path.join(ps[0], "**", "*.npz"),
                                            recursive=True)))
        elif ps:
            win.load_paths(ps)

    if selftest:   # headless build smoke test: construct, load, render, exit (no GUI loop)
        if paths:
            _open(paths); win.refresh_map()
        # Confirm the bundled TWINS calibration data is found (frozen builds).
        from instruments.calibration import (get_spectral_calibration,
                                             position_calibration_status)
        sc = get_spectral_calibration()[0] is not None
        pc = position_calibration_status()[0]
        print(f"SELFTEST OK: app constructed"
              + (f", loaded {len(win.infos)} map(s)" if win.infos else " (no file given)")
              + f"; spectral_cal={'FOUND' if sc else 'MISSING'}"
              + f", position_cal={'FOUND' if pc else 'MISSING'}")
        return

    win.show()
    _open(paths)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
