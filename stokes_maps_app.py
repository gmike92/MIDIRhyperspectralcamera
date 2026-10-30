"""
Stokes-from-hypercubes app -- phase-free Stokes retrieval from three measurements.

Workflow
--------
Load the THREE measurement hypercubes (.npz saved by the acquisition app, each
carrying a `raw_interferogram` + position axis + `wavelengths`) as M1, M2, M3,
and optionally a FOURTH hypercube used only for the phasing (background phase
correction). Pick a wavelength with the slider; the app does a per-pixel complex DFT 
at that wavelength -- and shows, in a 3x3 grid:

    row 1 : amplitude |field|   for M1, M2, M3
    row 2 : phase (rad)         for M1, M2, M3   (colour scale labelled in pi)
    row 3 : the retrieved Stokes maps from the PHASE-FREE formulae

      S1^2 = 1/2 (M2^2 + M3^2 - M1^2)
      S2^2 = 1/2 (M1^2 + M3^2 - M2^2)
      S3^2 = 1/2 (M1^2 + M2^2 - M3^2)

|S_i| = sqrt(clip(S_i^2, 0)). By default the SIGNED parameters are shown:
|S_i| times a per-pixel sign  sign(cos(P2)), sign(cos(P1)), sign(sin(P1)).

Run:  gui/.venv/Scripts/python stokes_maps_app.py
"""
import os
import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from instruments.hyperspectral import (
    HyperspectralProcessor, DEFAULT_ZPD_MM, DEFAULT_ZPD_WINDOW_MM)

# Show images as (row, col), like the other apps / imagesc.
pg.setConfigOptions(imageAxisOrder="row-major")

APOD_TYPES = ["gaussian", "happ-genzel", "blackman-harris-3",
              "blackman-harris-4", "boxcar"]


def _load_meas_npz(path):
    """Load a measurement .npz -> dict(raw, pos, cal, wl).

    Mirrors analysis_app's raw/axis reader: the per-pixel `raw_interferogram`
    (n_pos, h, w), the wedge position axis (calibrated one when the file's
    metadata says so, else the raw axis, flagged accordingly) and the stored
    `wavelengths` axis. Supports singular and stacked (plural) key layouts.
    """
    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)
        meta = {}
        if "metadata" in files:
            try:
                meta = dict(d["metadata"].item())
            except Exception:  # noqa: BLE001
                meta = {}
        if "raw_interferogram" in files:
            raw = np.asarray(d["raw_interferogram"], np.float32)
        elif "raw_interferograms" in files:
            raw = np.asarray(d["raw_interferograms"][0], np.float32)
        else:
            raise ValueError("no 'raw_interferogram' in this .npz "
                             "(re-acquire with raw saved).")
        applied = bool(meta.get("motor_calibration_applied", False))
        pos, cal = None, False
        if applied and "twins_positions_calibrated_mm" in files:
            pos, cal = np.asarray(d["twins_positions_calibrated_mm"], float), True
        elif applied and "raw_positions_calibrated" in files:
            pos, cal = np.asarray(d["raw_positions_calibrated"][0], float), True
        elif "twins_positions_mm" in files:
            pos, cal = np.asarray(d["twins_positions_mm"], float), False
        elif "raw_positions" in files:
            pos, cal = np.asarray(d["raw_positions"][0], float), False
        wl = np.asarray(d["wavelengths"], float).ravel() if "wavelengths" in files else None
    if raw.ndim != 3:
        raise ValueError(f"raw_interferogram must be 3-D (n_pos,h,w); got {raw.shape}.")
    if pos is None:
        pos = np.arange(raw.shape[0], dtype=float)
    return dict(raw=raw, pos=pos, cal=bool(cal), wl=wl)


def _bwr_cmap():
    """Diverging blue-white-red (blue = low / -pi, white = 0, red = high / +pi)."""
    return pg.ColorMap(pos=[0.0, 0.5, 1.0],
                       color=[(0, 0, 255), (255, 255, 255), (255, 0, 0)])


def _cyclic_cmap():
    """Cyclic colormap for WRAPPED PHASE (CET-C1), topologically correct so -pi
    and +pi share the same colour -- matching the analysis_app Phase panel.
    Falls back to blue-white-red if CET-C1 is unavailable."""
    try:
        cm = pg.colormap.get("CET-C1")
        if cm is not None:
            return cm
    except Exception:  # noqa: BLE001
        pass
    return _bwr_cmap()


_PI_TICKS = [[(-np.pi, "\u2212\u03c0"), (-np.pi / 2, "\u2212\u03c0/2"),
              (0.0, "0"), (np.pi / 2, "\u03c0/2"), (np.pi, "\u03c0")]]


def _set_pi_ticks(cbar):
    """Label a colorbar's axis in multiples of pi (phase maps are in radians)."""
    ax = getattr(cbar, "axis", None)
    if ax is not None:
        try:
            ax.setTicks(_PI_TICKS)
        except Exception:  # noqa: BLE001
            pass


def _sign(x):
    """Sign map with 0 values converted to +1, so an exactly-zero expression doesn't null a pixel."""
    return np.where(np.asarray(x, float) < 0.0, -1.0, 1.0)


def _robust_peak(amp):
    """A hot-pixel-proof 'peak' amplitude: the 99.5th percentile of the finite
    values (a single saturated pixel must not set the mask threshold). 0 if empty."""
    finite = np.asarray(amp, float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, 99.5)) if finite.size else 0.0


def _nanmax_safe(a, default=1.0):
    """max over finite entries, or `default` if there are none (no All-NaN warning)."""
    finite = np.asarray(a, float)
    finite = finite[np.isfinite(finite)]
    return float(finite.max()) if finite.size else default


def stokes_from_amp_phase(M1, M2, M3, P1, P2, P3):
    """Phase-free Stokes from three amplitudes + three phases -> (mag, signs).

    Shape-agnostic: the map view passes (h, w) arrays, the ROI-spectra window
    passes (n_freq,) spectra, so BOTH views use the same convention and cannot
    drift apart.

        S1^2 = 1/2 (M2^2 + M3^2 - M1^2)
        S2^2 = 1/2 (M1^2 + M3^2 - M2^2)
        S3^2 = 1/2 (M1^2 + M2^2 - M3^2)

    |S_i| = sqrt(clip(S_i^2, 0)); the signed value is mag[i] * signs[i].
    """
    mag = [np.sqrt(np.clip(0.5 * (M2**2 + M3**2 - M1**2), 0.0, None)),
           np.sqrt(np.clip(0.5 * (M1**2 + M3**2 - M2**2), 0.0, None)),
           np.sqrt(np.clip(0.5 * (M1**2 + M2**2 - M3**2), 0.0, None))]
    # Signs are determined by the phases of the maps.
    signs = [_sign(np.cos(P2)), _sign(np.cos(P1)), _sign(np.sin(P1))]
    # Variants tried earlier, kept for reference:
    # signs = [_sign(np.cos(P2))*_sign(np.sin(P3)), _sign(np.cos(P1))*_sign(np.cos(P3)), _sign(np.sin(P1))*-_sign(np.sin(P2))]
    # signs = [_sign(M2 + P3), _sign(M1 + M3), _sign(P1 - P2)]
    return mag, signs


# Stokes titles. M_k = amplitude map k, phi_k = phase map k. The magnitude is
# |S_i| = sqrt(clip(S_i^2, 0)); the SIGNED value multiplies it by a per-pixel
# sign taken from the expressions the user specified.
TITLES_SIGNED = [
    "S₁ = sgn(cos(φ₂))·√[½(M₂² + M₃² − M₁²)]",
    "S₂ = sgn(cos(φ₁))·√[½(M₁² + M₃² − M₂²)]",
    "S₃ = sgn(sin(φ₁))·√[½(M₁² + M₂² − M₃²)]",
]
TITLES_UNSIGNED = [
    "|S₁| = √[½(M₂² + M₃² − M₁²)]",
    "|S₂| = √[½(M₁² + M₃² − M₂²)]",
    "|S₃| = √[½(M₁² + M₂² − M₃²)]",
]


class StokesMapsApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stokes from hypercubes (phase-free, per-λ DFT)")
        self.resize(1300, 940)
        self.slots = [None, None, None]     # dict(raw,pos,cal,wl,name) or None
        self.ref = None                     # 4th cube for phasing, or None
        self.i45 = None                     # optional separate cube whose |field| is I45
        self._i45_amp = None                # I45 amplitude at the current lambda
        self.proc = None                    # HyperspectralProcessor (lazy)
        self._pixel_win = None              # PixelSpectraWindow, lazily created
        self.wl = None                      # wavelength axis for the slider
        self._last_dir = "D:\\CAMERA" if os.path.isdir("D:\\CAMERA") else ""
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True); self._timer.setInterval(200)
        self._timer.timeout.connect(self._recompute)
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # --- load row: 3 measurements + phasing reference ---
        top = QtWidgets.QHBoxLayout()
        self.slot_lbl = []
        for k in range(3):
            b = QtWidgets.QPushButton(f"Load M{k+1} (.npz)\u2026")
            b.clicked.connect(lambda _c, kk=k: self._load_slot(kk))
            top.addWidget(b)
            lbl = QtWidgets.QLabel("(empty)"); lbl.setStyleSheet("color:#888;")
            top.addWidget(lbl); self.slot_lbl.append(lbl)
        top.addSpacing(16)
        b_ref = QtWidgets.QPushButton("Load phasing ref (.npz)\u2026")
        b_ref.setToolTip("Optional 4th hypercube: its raw interferogram is used as "
                         "the background for per-pixel phase correction (same as the "
                         "analysis_app Phase panel).")
        b_ref.clicked.connect(self._load_ref)
        top.addWidget(b_ref)
        self.ref_lbl = QtWidgets.QLabel("phasing: none"); self.ref_lbl.setStyleSheet("color:#888;")
        top.addWidget(self.ref_lbl)
        self.chk_phase = QtWidgets.QCheckBox("Apply phasing")
        self.chk_phase.setEnabled(False)
        self.chk_phase.toggled.connect(self._schedule)
        top.addWidget(self.chk_phase)
        top.addStretch(1)
        self.b_pixel = QtWidgets.QPushButton("Pixel / ROI spectra…")
        self.b_pixel.setToolTip(
            "Open the inverse view: pick a pixel or a region on the average raw "
            "image and see amplitude, phase and the Stokes parameters as "
            "functions of wavelength for that region.")
        self.b_pixel.setEnabled(False)
        self.b_pixel.clicked.connect(self._open_pixel_window)
        top.addWidget(self.b_pixel)
        b_export = QtWidgets.QPushButton("Export maps…")
        b_export.setToolTip("Choose which maps to export and the file format "
                            "(PNG figure, float TIFF, NumPy .npy, CSV, or an .npz bundle).")
        b_export.clicked.connect(self._open_export_dialog)
        top.addWidget(b_export)
        root.addLayout(top)

        # --- transmission-correction row ---
        # Per-cube scalar gain: the measurements are taken through different
        # optics/polarisers, so their |field| maps are not directly comparable
        # until each is divided by its own transmission. M1 is the normalisation
        # reference and stays at 1 -- correcting the other cubes relative to it
        # is equivalent to correcting all four.
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("Transmission correction"))
        self.sp_gain = {}
        for key, label, tip in (
            (1, "M₂",
             "Multiply the M₂ complex map by this factor (amplitude and every "
             "Stokes map that uses it; the phase is unchanged by a positive "
             "scalar)."),
            (2, "M₃",
             "Multiply the M₃ complex map by this factor (amplitude and every "
             "Stokes map that uses it; the phase is unchanged by a positive "
             "scalar)."),
            ("ref", "phasing ref",
             "Multiply the phasing-reference complex map by this factor. This "
             "matters only while 'I₄₅ from' = phasing ref, i.e. when that cube is "
             "the normalisation source for S₀ = 2·I₄₅ − S₂ and for the amplitude "
             "mask." + chr(10) +
             "The phasing itself is NOT affected: the reference enters as "
             "exp(-i·angle(ref)), a unit-modulus phasor, so any positive scalar "
             "cancels exactly."),
            ("i45", "I₄₅ cube",
             "Multiply the separate I₄₅ cube's complex map by this factor. Used "
             "only while 'I₄₅ from' = separate cube."),
        ):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(1e-6, 1e4); s.setDecimals(4)
            s.setSingleStep(0.01); s.setValue(1.0)
            s.setToolTip(tip)
            s.valueChanged.connect(self._schedule)
            trow.addWidget(QtWidgets.QLabel(label)); trow.addWidget(s)
            self.sp_gain[key] = s
        self.lbl_m1_fixed = QtWidgets.QLabel("(M₁ ≡ 1, reference)")
        self.lbl_m1_fixed.setStyleSheet("color:#888;")
        trow.addWidget(self.lbl_m1_fixed)
        trow.addStretch(1)
        root.addLayout(trow)

        # --- S0 / normalisation row ---
        s0row = QtWidgets.QHBoxLayout()
        self.chk_norm = QtWidgets.QCheckBox("Normalise by S₀ = 2·I₄₅ − S₂")
        self.chk_norm.setToolTip("Compute S₀ from the I₄₅ amplitude and show "
                                 "S₁/S₀, S₂/S₀, S₃/S₀ (signed).")
        self.chk_norm.toggled.connect(self._schedule)
        s0row.addWidget(self.chk_norm)
        s0row.addWidget(QtWidgets.QLabel("I₄₅ from"))
        self.combo_i45src = QtWidgets.QComboBox()
        self.combo_i45src.addItems(["phasing ref", "separate cube"])
        self.combo_i45src.currentIndexChanged.connect(self._schedule)
        s0row.addWidget(self.combo_i45src)
        b_i45 = QtWidgets.QPushButton("Load I₄₅ cube (.npz)…")
        b_i45.setToolTip("Load a separate hypercube whose |field| is used as I₄₅ "
                         "(only used when 'I₄₅ from' = separate cube).")
        b_i45.clicked.connect(self._load_i45)
        s0row.addWidget(b_i45)
        self.i45_lbl = QtWidgets.QLabel("I₄₅: none"); self.i45_lbl.setStyleSheet("color:#888;")
        s0row.addWidget(self.i45_lbl)
        s0row.addSpacing(16)
        s0row.addWidget(QtWidgets.QLabel("Amp mask"))
        self.sp_mask = QtWidgets.QDoubleSpinBox()
        self.sp_mask.setRange(0.0, 100.0); self.sp_mask.setValue(5.0)
        self.sp_mask.setSuffix(" % peak"); self.sp_mask.setDecimals(2)
        self.sp_mask.setToolTip("Mask pixels below this % of the I₄₅ peak amplitude "
                                "(or, when no I₄₅ is loaded, each map's own peak). Applied "
                                "to phase + Stokes; the amplitude maps stay full.")
        self.sp_mask.valueChanged.connect(self._schedule)
        s0row.addWidget(self.sp_mask)
        self.chk_showmask = QtWidgets.QCheckBox("Mark masked px")
        self.chk_showmask.setToolTip("Tint the masked-out pixels red on the amplitude "
                                     "maps so you can see which pixels the mask removes.")
        self.chk_showmask.setChecked(True)
        self.chk_showmask.toggled.connect(self._refresh_mask_overlay)
        s0row.addWidget(self.chk_showmask)
        s0row.addStretch(1)
        root.addLayout(s0row)

        # --- wavelength slider ---
        wlrow = QtWidgets.QHBoxLayout()
        wlrow.addWidget(QtWidgets.QLabel("λ:"))
        self.sl_wl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_wl.setEnabled(False)
        self.sl_wl.valueChanged.connect(self._on_lambda)
        wlrow.addWidget(self.sl_wl, 1)
        self.lbl_wl = QtWidgets.QLabel("-- µm"); self.lbl_wl.setStyleSheet("font-weight:600;")
        wlrow.addWidget(self.lbl_wl)
        root.addLayout(wlrow)

        # --- DFT controls (mirror the analysis_app Phase panel) ---
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Apod"))
        self.combo_apod = QtWidgets.QComboBox(); self.combo_apod.addItems(APOD_TYPES)
        self.combo_apod.currentIndexChanged.connect(self._schedule)
        ctl.addWidget(self.combo_apod)
        ctl.addWidget(QtWidgets.QLabel("Centre"))
        self.combo_center = QtWidgets.QComboBox()
        # Barycentre first = the default: each pixel's own I² barycentre follows a
        # ZPD that drifts across the field, and (unlike the field-wide envelope)
        # it does not depend on which pixels are in the frame, so the maps and the
        # ROI-spectra window agree exactly.
        self.combo_center.addItems(["barycentre (per-pixel)", "envelope (field)"])
        self.combo_center.setToolTip(
            "Where the symmetric apodization window is centred:" + chr(10)
            + "• barycentre (per-pixel, default): each pixel's own I² "
              "barycentre." + chr(10)
            + "• envelope (field): one centre-burst for the whole frame, from "
              "the envelope of the signed spatial sum.")
        self.combo_center.currentIndexChanged.connect(self._schedule)
        ctl.addWidget(self.combo_center)
        self.chk_ftwin = QtWidgets.QCheckBox("Limit FT window")
        self.chk_ftwin.toggled.connect(self._schedule)
        ctl.addWidget(self.chk_ftwin)
        self.sp_ft0 = QtWidgets.QDoubleSpinBox(); self.sp_ft1 = QtWidgets.QDoubleSpinBox()
        for s in (self.sp_ft0, self.sp_ft1):
            s.setRange(-1e4, 1e4); s.setDecimals(4); s.setSuffix(" mm")
            s.valueChanged.connect(self._schedule)
        ctl.addWidget(self.sp_ft0); ctl.addWidget(QtWidgets.QLabel("to")); ctl.addWidget(self.sp_ft1)
        ctl.addSpacing(12)
        ctl.addWidget(QtWidgets.QLabel("Stokes"))
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["Signed Stokes parameters", "Unsigned Stokes parameters"])
        self.combo_mode.setToolTip(
            "Signed (default): |Sᵢ| = √(clip(Sᵢ²,0)) times a per-pixel sign — "
            "sgn(M₂+φ₃) for S₁, sgn(M₁+M₃) for S₂, sgn(φ₁−φ₂) for S₃.\n"
            "Unsigned: the magnitudes |Sᵢ| only.")
        self.combo_mode.currentIndexChanged.connect(self._recompute_stokes_only)
        ctl.addWidget(self.combo_mode)
        ctl.addStretch(1)
        root.addLayout(ctl)

        self.status = QtWidgets.QLabel("Load three measurement hypercubes to begin.")
        self.status.setStyleSheet("color:#888;")
        root.addWidget(self.status)

        # --- 3x4 grid: col 0 = I45/S0 reference column; cols 1-3 = M1,M2,M3.
        #     rows: amplitude / phase / Stokes.
        gridw = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(gridw)
        self.amp_img = []; self.amp_cbar = []; self.amp_title = []
        self.ph_img = []; self.ph_cbar = []; self.ph_title = []
        self.s_img = []; self.s_cbar = []; self.s_title = []
        self.export_panels = []          # [{key, label, img, glw}] for the exporter
        turbo, bwr, cyc = pg.colormap.get("turbo"), _bwr_cmap(), _cyclic_cmap()

        def _reg(key, label, img, glw):
            self.export_panels.append(dict(key=key, label=label, img=img, glw=glw))

        # Reference column (col 0): I45 amplitude / I45 phase / S0.
        glw, self.i45_amp_img, self.i45_amp_cbar, self.i45_amp_title = self._make_panel("I\u2084\u2085  |field|", turbo)
        grid.addWidget(glw, 0, 0); _reg("I45_amplitude", "I\u2084\u2085 amplitude", self.i45_amp_img, glw)
        self.i45_amp_ov = pg.ImageItem(); self.i45_amp_ov.setZValue(10)
        self.i45_amp_img.getViewBox().addItem(self.i45_amp_ov)
        glw, self.i45_ph_img, self.i45_ph_cbar, self.i45_ph_title = self._make_panel("phase I\u2084\u2085  (scale in \u03c0)", cyc)
        self.i45_ph_cbar.setLevels((-np.pi, np.pi)); _set_pi_ticks(self.i45_ph_cbar)
        grid.addWidget(glw, 1, 0); _reg("I45_phase", "I\u2084\u2085 phase", self.i45_ph_img, glw)
        glw, self.s0_img, self.s0_cbar, self.s0_title = self._make_panel("S\u2080 = 2\u00b7I\u2084\u2085 \u2212 S\u2082", turbo, title_size="9pt")
        grid.addWidget(glw, 2, 0); _reg("S0", "S\u2080", self.s0_img, glw)

        # Measurement columns (1..3).
        self.amp_mask_ov = []
        for k in range(3):
            glw, img, cbar, lbl = self._make_panel(f"M{k+1}", turbo)
            self.amp_img.append(img); self.amp_cbar.append(cbar); self.amp_title.append(lbl)
            grid.addWidget(glw, 0, k + 1); _reg(f"M{k+1}_amplitude", f"M{k+1} amplitude", img, glw)
            ov = pg.ImageItem(); ov.setZValue(10)
            img.getViewBox().addItem(ov)
            self.amp_mask_ov.append(ov)
        for k in range(3):
            glw, img, cbar, lbl = self._make_panel(f"phase M{k+1}  (rad, scale in \u03c0)", cyc)
            cbar.setLevels((-np.pi, np.pi)); _set_pi_ticks(cbar)
            self.ph_img.append(img); self.ph_cbar.append(cbar); self.ph_title.append(lbl)
            grid.addWidget(glw, 1, k + 1); _reg(f"M{k+1}_phase", f"M{k+1} phase", img, glw)
        for k in range(3):
            glw, img, cbar, lbl = self._make_panel(TITLES_SIGNED[k], bwr, title_size="9pt")
            self.s_img.append(img); self.s_cbar.append(cbar); self.s_title.append(lbl)
            grid.addWidget(glw, 2, k + 1); _reg(f"S{k+1}", f"S{k+1}", img, glw)
        for r in range(3):
            grid.setRowStretch(r, 1)
        for c in range(4):
            grid.setColumnStretch(c, 1)
        root.addWidget(gridw, 1)

    def _make_panel(self, title, cmap, title_size=None):
        glw = pg.GraphicsLayoutWidget()
        glw.addLabel(title, row=0, col=0, colspan=2, size=title_size or "11pt")
        vb = glw.addViewBox(row=1, col=0)
        vb.setAspectLocked(True); vb.invertY(True)
        lbl = glw.getItem(0, 0)
        img = pg.ImageItem(); vb.addItem(img)
        cbar = pg.ColorBarItem(interactive=False, colorMap=cmap)
        glw.addItem(cbar, row=1, col=1); cbar.setImageItem(img)
        return glw, img, cbar, lbl

    # -------------------------------------------------------------- loading
    def _load_slot(self, k):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load measurement M{k+1} (.npz)", self._last_dir,
            "Hypercube (*.npz);;All files (*)")
        if not path:
            return
        try:
            slot = _load_meas_npz(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Could not load hypercube",
                                           f"{os.path.basename(path)}:\n{e}")
            return
        slot["name"] = os.path.basename(path)
        self._last_dir = os.path.dirname(path)
        self.slots[k] = slot
        self.slot_lbl[k].setText(f"{slot['name']}  {slot['raw'].shape}")
        self.slot_lbl[k].setStyleSheet("color:#2f9e44;")
        self._init_axes_from(slot)
        self.b_pixel.setEnabled(True)
        self._recompute()
        self._pixel_refresh()

    def _load_ref(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load phasing reference (.npz)", self._last_dir,
            "Hypercube (*.npz);;All files (*)")
        if not path:
            return
        try:
            self.ref = _load_meas_npz(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Could not load reference", f"{e}")
            return
        self.ref["name"] = os.path.basename(path)
        self._last_dir = os.path.dirname(path)
        self.ref_lbl.setText(f"phasing: {self.ref['name']}  {self.ref['raw'].shape}")
        self.ref_lbl.setStyleSheet("color:#2f9e44;")
        self.chk_phase.setEnabled(True)
        self.chk_phase.blockSignals(True); self.chk_phase.setChecked(True)
        self.chk_phase.blockSignals(False)
        self.b_pixel.setEnabled(True)
        self._recompute()
        self._pixel_refresh()

    def _load_i45(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load I₄₅ cube (.npz)", self._last_dir,
            "Hypercube (*.npz);;All files (*)")
        if not path:
            return
        try:
            self.i45 = _load_meas_npz(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Could not load I45 cube", f"{e}")
            return
        self.i45["name"] = os.path.basename(path)
        self._last_dir = os.path.dirname(path)
        self.i45_lbl.setText(f"I₄₅: {self.i45['name']}  {self.i45['raw'].shape}")
        self.i45_lbl.setStyleSheet("color:#2f9e44;")
        self.combo_i45src.setCurrentIndex(1)          # -> use the separate cube
        self.b_pixel.setEnabled(True)
        self._recompute()
        self._pixel_refresh()

    def _open_pixel_window(self):
        """Show the ROI-spectra window (one instance, re-raised if already open)."""
        if self._pixel_win is None:
            self._pixel_win = PixelSpectraWindow(self)
        self._pixel_win.show()
        self._pixel_win.raise_()
        self._pixel_win.activateWindow()

    def _pixel_refresh(self):
        """Tell the ROI-spectra window a cube changed, if it is open."""
        win = self._pixel_win
        if win is not None and win.isVisible():
            win._refresh_background()
            win._schedule()

    def _init_axes_from(self, slot):
        """Set the λ slider range + FT-window defaults from the first cube loaded."""
        if self.wl is None and slot.get("wl") is not None and len(slot["wl"]):
            self.wl = np.asarray(slot["wl"], float)
            self.sl_wl.blockSignals(True)
            self.sl_wl.setMinimum(0); self.sl_wl.setMaximum(len(self.wl) - 1)
            self.sl_wl.setValue(len(self.wl) // 2)
            self.sl_wl.setEnabled(True)
            self.sl_wl.blockSignals(False)
            self._update_wl_label()
        pos = slot["pos"]
        if pos is not None and len(pos) and self.sp_ft0.value() == self.sp_ft1.value():
            self.sp_ft0.blockSignals(True); self.sp_ft1.blockSignals(True)
            self.sp_ft0.setValue(float(np.min(pos))); self.sp_ft1.setValue(float(np.max(pos)))
            self.sp_ft0.blockSignals(False); self.sp_ft1.blockSignals(False)

    # ------------------------------------------------------------- compute
    def _on_lambda(self, *a):
        self._update_wl_label()
        self._schedule()
        win = self._pixel_win
        if win is not None and win.isVisible():
            win._sync_cursor_from_app()

    def _update_wl_label(self):
        if self.wl is not None and len(self.wl):
            i = int(np.clip(self.sl_wl.value(), 0, len(self.wl) - 1))
            self.lbl_wl.setText(f"{self.wl[i]:.4f} µm")

    def _schedule(self, *a):
        self._timer.start()

    def _current_lambda(self):
        if self.wl is None or not len(self.wl):
            return None
        i = int(np.clip(self.sl_wl.value(), 0, len(self.wl) - 1))
        return float(self.wl[i])

    def _i45_source(self):
        """The cube whose |field| is I45: the phasing ref or the separate cube."""
        return self.ref if self.combo_i45src.currentIndex() == 0 else self.i45

    def _gain(self, key):
        """User transmission-correction factor for one cube (1.0 if there is none).

        `key` is 0/1/2 for M1/M2/M3, "ref" for the phasing reference or "i45" for
        the separate I45 cube. M1 carries no box -- it is the normalisation
        reference. Applied to the COMPLEX map: the DFT is linear, so scaling the
        map equals scaling the raw interferogram, without copying the cube.
        """
        s = self.sp_gain.get(key)
        return 1.0 if s is None else float(s.value())

    def _i45_gain(self):
        """Gain of whichever cube currently feeds I45."""
        return self._gain("ref" if self.combo_i45src.currentIndex() == 0 else "i45")

    def _recompute(self, *args):
        lam = self._current_lambda()
        if lam is None:
            return
        if self.proc is None:
            self.proc = HyperspectralProcessor()
        apod = self.combo_apod.currentText()
        center = ("barycenter" if self.combo_center.currentText().startswith("bary")
                  else "envelope")
        ftwin = ((self.sp_ft0.value(), self.sp_ft1.value())
                 if self.chk_ftwin.isChecked() else None)
        phasing = self.chk_phase.isChecked() and self.ref is not None
        maskpct = self.sp_mask.value()
        notes = []

        def _dft(src, ref_raw=None):
            # Compute the complex map at the current λ, optionally using a reference cube for phasing.
            # This is based on HyperspectralProcessor.compute_complex_map, which handles the DFT, apodization,
            # ZPD search, and optional phasing. Phasing consists of computing the complex map of the reference
            # cube at the same λ and using it to correct the phase of the source cube.
            c, _ = self.proc.compute_complex_map(
                src["pos"], src["raw"], lam, apod_type=apod, ft_window_mm=ftwin,
                expected_zero_mm=DEFAULT_ZPD_MM, search_mm=DEFAULT_ZPD_WINDOW_MM,
                positions_calibrated=src["cal"], reference_cube=ref_raw,
                center_method=center)
            return c

        # --- I45 reference: its |field| is the mask source and S0 uses it. ---
        # (Magnitude is unchanged by phasing, so no reference is applied here.)
        self._i45_amp = None
        self._i45_valid = None
        gmask = None
        src = self._i45_source()
        if src is not None:
            try:
                c45 = _dft(src)
            except Exception as e:  # noqa: BLE001
                c45 = None; notes.append(f"I₄₅: {e}")
            if c45 is not None:
                c45 = c45 * self._i45_gain()
                self._i45_amp = np.abs(c45).astype(np.float64)
                i45_ph = np.angle(c45).astype(np.float64)
                pk = _robust_peak(self._i45_amp)
                v = np.isfinite(self._i45_amp)
                if maskpct > 0 and pk > 0:
                    v &= self._i45_amp >= (maskpct / 100.0) * pk
                self._i45_valid = v
                gmask = v
                self.i45_amp_img.setImage(self._i45_amp, autoLevels=False)
                self.i45_amp_cbar.setLevels((0.0, pk if pk > 0 else 1.0))
                self.i45_ph_img.setImage(np.where(v, i45_ph, np.nan),
                                         autoLevels=False, levels=(-np.pi, np.pi))
                self.i45_ph_cbar.setLevels((-np.pi, np.pi)); _set_pi_ticks(self.i45_ph_cbar)
        if self._i45_amp is None:
            self.i45_amp_img.clear(); self.i45_ph_img.clear()
            self.i45_amp_ov.setVisible(False)

        prepared = []
        for k in range(3):
            slot = self.slots[k]
            if slot is None:
                self.amp_img[k].clear(); self.ph_img[k].clear()
                self.amp_title[k].setText(f"M{k+1}")
                prepared.append(None); continue
            raw = slot["raw"]
            ref_raw = None
            if phasing:
                if self.ref["raw"].shape == raw.shape:
                    ref_raw = self.ref["raw"]
                else:
                    notes.append(f"M{k+1}: ref shape {self.ref['raw'].shape} ≠ "
                                 f"{raw.shape}, phasing skipped")
            try:
                cmap = _dft(slot, ref_raw)
            except Exception as e:  # noqa: BLE001
                notes.append(f"M{k+1}: {e}")
                prepared.append(None); continue
            if cmap is None:
                notes.append(f"M{k+1}: too few scan positions for a DFT")
                prepared.append(None); continue
            g = self._gain(k)                 # M1 has no box -> 1.0 (reference)
            if g != 1.0:
                cmap = cmap * g
            amp = np.abs(cmap).astype(np.float64)
            phase = np.angle(cmap).astype(np.float64)
            # Mask from I45 when available (single mask), else this map's own peak.
            # Applied to PHASE (row 2) + Stokes; the amplitude map stays full.
            valid = np.isfinite(amp)
            if gmask is not None and gmask.shape == amp.shape:
                valid &= gmask
            else:
                pkm = _robust_peak(amp)
                if maskpct > 0 and pkm > 0:
                    valid &= amp >= (maskpct / 100.0) * pkm
            peak = _robust_peak(amp)
            self.amp_img[k].setImage(amp, autoLevels=False)         # row 1: unmasked
            self.amp_cbar[k].setLevels((0.0, peak if peak > 0 else 1.0))
            self.ph_img[k].setImage(np.where(valid, phase, np.nan),  # row 2: masked
                                    autoLevels=False, levels=(-np.pi, np.pi))
            self.ph_cbar[k].setLevels((-np.pi, np.pi)); _set_pi_ticks(self.ph_cbar[k])
            self.amp_title[k].setText(
                f"M{k+1}  |field|"
                + (f"  ×{g:g}" if g != 1.0 else "")
                + ("  [phased]" if ref_raw is not None else ""))
            prepared.append((amp, valid, phase))

        self._prepared = prepared
        self._refresh_mask_overlay()
        n_loaded = sum(p is not None for p in prepared)
        base = (f"λ = {lam:.4f} µm   |   apod {apod}   |   centre {center}   |   "
                f"{n_loaded}/3 maps"
                + ("   |   phasing ON" if phasing else ""))
        # Make a non-unity transmission correction visible: a scaled map must
        # never be mistaken for a raw one.
        gains = [("M₂", self._gain(1)), ("M₃", self._gain(2))]
        if src is not None:
            gains.append(("I₄₅", self._i45_gain()))
        active = [f"{lbl}×{g:g}" for lbl, g in gains if g != 1.0]
        if active:
            base += "   |   gain " + " ".join(active)
        if notes:
            base += "   |   " + "; ".join(notes)
        self._status_base = base
        self._compute_stokes(prepared)

    def _recompute_stokes_only(self, *a):
        """Toggle signed/unsigned without recomputing the DFTs."""
        if getattr(self, "_prepared", None) is not None:
            self._compute_stokes(self._prepared)

    def _refresh_mask_overlay(self, *a):
        """Tint the masked-out pixels (red, translucent) on the amplitude maps so
        the mask is visible on row 1. Cheap: rebuilt from the stored valid masks,
        no DFT recompute."""
        prep = getattr(self, "_prepared", None)
        show = self.chk_showmask.isChecked()

        def _tint(ov, valid):
            if not show or valid is None:
                ov.setVisible(False); return
            masked = ~np.asarray(valid, bool)
            rgba = np.zeros(masked.shape + (4,), np.ubyte)
            rgba[masked] = (255, 0, 0, 130)      # translucent red = removed by the mask
            ov.setImage(rgba); ov.setVisible(True)

        _tint(self.i45_amp_ov, getattr(self, "_i45_valid", None))
        for k in range(3):
            _tint(self.amp_mask_ov[k],
                  None if prep is None or prep[k] is None else prep[k][1])

    def _status(self, note):
        self.status.setText(getattr(self, "_status_base", "") + note)

    def _compute_stokes(self, prepared):
        # If any of the three maps is missing, clear the Stokes row and return.
        if any(p is None for p in prepared):
            for k in range(3):
                self.s_img[k].clear()
            self.s0_img.clear()
            self._status("   |   load all three maps for Stokes")
            return
        # Check that all three maps have the same shape; if not, clear the Stokes row and return.
        shapes = {p[0].shape for p in prepared} # creating a set of shapes of the amplitude maps, i.e., dimensions must be identical 
        if len(shapes) != 1:
            for k in range(3):
                self.s_img[k].clear()
            self.s0_img.clear()
            self._status("   |   maps differ in size "
                         f"({', '.join(str(s) for s in shapes)}) — Stokes needs "
                         "identical dimensions")
            return

        # S0 needs a matching-shape I45 amplitude; normalisation divides by it.
        i45 = self._i45_amp
        have_i45 = i45 is not None and i45.shape == prepared[0][0].shape
        normalize = self.chk_norm.isChecked() and have_i45
        # Normalised Stokes are inherently signed; the toggle only matters otherwise.
        signed = normalize or (self.combo_mode.currentIndex() == 0)

        # Amplitudes and phases of the three maps
        M1, M2, M3 = (p[0] for p in prepared)
        P1, P2, P3 = (p[2] for p in prepared)

        # Combined mask: only pixels that pass all three amplitude masks are valid. 
        # prepared[k][1] is the valid mask for map k, so we take the logical AND of all three.
        # Calculation continues only if there aare valid pixels; otherwise return after clearing the Stokes row.
        combined = prepared[0][1] & prepared[1][1] & prepared[2][1]
        n_ok = int(np.count_nonzero(combined))
        if n_ok == 0:
            for k in range(3):
                self.s_img[k].clear()
            self.s0_img.clear()
            self._status("   |   0 pixels pass the amplitude mask — lower 'Amp mask %'")
            return

        # Stokes parameters from the amplitudes and phases of the three maps.
        # The formulae + the sign convention live in stokes_from_amp_phase(), so
        # the map view and the ROI-spectra window share one definition.
        mag, signs = stokes_from_amp_phase(M1, M2, M3, P1, P2, P3)
        s_signed = [mag[k] * signs[k] for k in range(3)]

        # S0 = 2*I45 - S2, shown in the reference column whenever I45 is available.
        s0 = (2.0 * i45 - s_signed[1]) if have_i45 else None
        if have_i45:
            s0_disp = np.where(combined, s0, np.nan)
            self.s0_img.setImage(s0_disp, autoLevels=False)
            hi0 = _nanmax_safe(s0_disp, default=1.0)
            self.s0_cbar.setLevels((0.0, hi0 if hi0 > 0 else 1.0))
            self.s0_title.setText("S₀ = 2·I₄₅ − S₂", size="9pt")
        else:
            self.s0_img.clear()
            self.s0_title.setText("S₀ (load I₄₅)", size="9pt")

        if normalize:
            eps = max(1e-12 * _nanmax_safe(np.abs(s0)), 1e-30)
            s0_safe = np.where(np.abs(s0) <= eps, np.nan, s0)
            vals = [s_signed[k] / s0_safe for k in range(3)]
            titles = ["S₁/S₀", "S₂/S₀", "S₃/S₀"]
            cmap = _bwr_cmap()
        elif signed:
            vals = s_signed
            titles = TITLES_SIGNED
            cmap = _bwr_cmap()
        else:
            vals = mag
            titles = TITLES_UNSIGNED
            cmap = pg.colormap.get("turbo")

        for k in range(3):
            self.s_title[k].setText(titles[k], size="9pt")
            self.s_cbar[k].setColorMap(cmap)
            img = np.where(combined, vals[k], np.nan)
            self.s_img[k].setImage(img, autoLevels=False)
            if normalize:                             # S_i/S0 is bounded to [-1, 1]
                self.s_cbar[k].setLevels((-1.0, 1.0))
            elif signed:                              # diverging, symmetric about 0
                m = _nanmax_safe(np.abs(img), default=1.0)
                self.s_cbar[k].setLevels((-m if m > 0 else -1.0, m if m > 0 else 1.0))
            else:
                hi = _nanmax_safe(img, default=1.0)
                self.s_cbar[k].setLevels((0.0, hi if hi > 0 else 1.0))
        self._status(f"   |   Stokes over {n_ok} px"
                     + ("  (normalised S₁/S₀…)" if normalize else ""))

    # --------------------------------------------------------------- export
    def _open_export_dialog(self):
        """Pick which maps to export and the file format."""
        avail = [p for p in self.export_panels
                 if getattr(p["img"], "image", None) is not None]
        if not avail:
            QtWidgets.QMessageBox.information(
                self, "Export maps", "Nothing to export yet — load maps first.")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export maps")
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel("Maps to export:"))
        cb_all = QtWidgets.QCheckBox("Select all"); cb_all.setChecked(True)
        v.addWidget(cb_all)
        checks = []
        for p in avail:
            cb = QtWidgets.QCheckBox(p["label"]); cb.setChecked(True)
            v.addWidget(cb); checks.append((cb, p))
        cb_all.toggled.connect(lambda on: [cb.setChecked(on) for cb, _ in checks])
        fmt_row = QtWidgets.QHBoxLayout()
        fmt_row.addWidget(QtWidgets.QLabel("Format:"))
        combo = QtWidgets.QComboBox()
        combo.addItems(["PNG image (rendered figure)", "TIFF (float data)",
                        "NumPy .npy (data)", "CSV (data)", "NPZ bundle (all-in-one)"])
        fmt_row.addWidget(combo, 1); v.addLayout(fmt_row)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        selected = [p for cb, p in checks if cb.isChecked()]
        if not selected:
            return
        self._do_export(selected, combo.currentIndex())

    def _do_export(self, selected, fmt):
        # fmt: 0 PNG (rendered), 1 TIFF float, 2 npy, 3 csv, 4 npz bundle.
        if fmt == 4:                                     # single .npz bundle
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save NPZ bundle", os.path.join(self._last_dir, "stokes_maps.npz"),
                "NumPy archive (*.npz)")
            if not path:
                return
            data = {p["key"]: np.asarray(p["img"].image) for p in selected}
            try:
                np.savez(path, **data)
            except Exception as e:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "Export failed", str(e)); return
            self.status.setText(f"Exported {len(data)} maps → {os.path.basename(path)}")
            return

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export maps to folder", self._last_dir)
        if not folder:
            return
        ext = {0: ".png", 1: ".tif", 2: ".npy", 3: ".csv"}[fmt]
        done, errs = 0, []
        for p in selected:
            base = os.path.join(folder, p["key"])
            try:
                if fmt == 0:                             # rendered figure via pyqtgraph
                    import pyqtgraph.exporters as pgx
                    pgx.ImageExporter(p["glw"].scene()).export(base + ext)
                else:
                    arr = np.asarray(p["img"].image, dtype=np.float32)
                    if fmt == 1:
                        _save_tiff(base + ext, arr)
                    elif fmt == 2:
                        np.save(base + ext, arr)
                    elif fmt == 3:
                        np.savetxt(base + ext, arr, delimiter=",", fmt="%.6g")
                done += 1
            except Exception as e:  # noqa: BLE001
                errs.append(f"{p['key']}: {e}")
        msg = f"Exported {done}/{len(selected)} maps ({ext}) → {folder}"
        if errs:
            msg += "   |   " + "; ".join(errs)
        self.status.setText(msg)


# ---------------------------------------------------------- ROI spectra window
# Curve colours, shared by the plots and the legends.
PEN_M = {"M1": "#1f77b4", "M2": "#d62728", "M3": "#2ca02c", "I45": "#9467bd"}
PEN_S = ["#1f77b4", "#d62728", "#2ca02c"]
PEN_S0 = "#888888"


class PixelSpectraWindow(QtWidgets.QMainWindow):
    """Spectra of one pixel / one ROI, the inverse view of the maps.

    The maps show every pixel at ONE wavelength; this window shows ONE region at
    every wavelength. Order of operations (chosen deliberately):

        1. crop the ROI out of each raw interferogram cube
        2. per-pixel complex DFT over the whole lambda range, with the SAME
           apodization, ZPD, FT window and reference phasing as the maps
        3. average the ROI **after** the DFT -- averaging the interferograms
           first would be coherent averaging, which cancels fringes wherever the
           ZPD or the phase varies across the region
        4. plot |M| and phi for M1, M2, M3 and the I45 cube
        5. plot the Stokes parameters from the same phase-free formulae as the
           map view -- stokes_from_amp_phase(), one shared helper, so the sign
           convention cannot drift between the two views

    Everything except the lambda grid and the ROI is read live from the main
    window, so the curves always correspond to the maps on screen.

    One difference from the maps, and ONLY in "envelope" centre mode: there the
    ZPD comes from the signed spatial sum of the SELECTED pixels, not of the
    whole frame, so a small ROI can land the apodization window a sample or two
    away from where the map view puts it (a few percent in amplitude). In
    "barycentre (per-pixel)" mode the centre is each pixel's own I^2 barycentre,
    which the crop cannot change, so the two views agree to float precision; a
    boxcar window removes the sensitivity either way.
    """

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Pixel / ROI spectra")
        self.resize(1320, 860)
        self._syncing = False          # guards the cursor <-> lambda-slider loop
        self._curves = []              # [{key, label, group, y}] for the exporter
        self._wl = None                # lambda axis of the last computation
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._recompute)
        self._build_ui()
        self._connect_app()
        self._refresh_background()
        self._init_lambda_range()
        self._schedule()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # ---------------- left: the ROI picker on the average raw image -----
        leftw = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(leftw)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("Image from"))
        self.combo_bg = QtWidgets.QComboBox()
        self.combo_bg.addItems(["M₁", "M₂", "M₃", "I₄₅"])
        self.combo_bg.setToolTip("Which cube's position-averaged raw interferogram "
                                 "is shown as the background for picking the ROI. "
                                 "The ROI itself is applied to every cube.")
        self.combo_bg.currentIndexChanged.connect(self._refresh_background)
        srow.addWidget(self.combo_bg)
        srow.addSpacing(12)
        srow.addWidget(QtWidgets.QLabel("Select"))
        self.combo_sel = QtWidgets.QComboBox()
        self.combo_sel.addItems(["ROI (region)", "Pixel (1x1)"])
        self.combo_sel.currentIndexChanged.connect(self._on_sel_mode)
        srow.addWidget(self.combo_sel)
        srow.addStretch(1)
        lv.addLayout(srow)

        glw = pg.GraphicsLayoutWidget()
        vb = glw.addViewBox(row=0, col=0)
        vb.setAspectLocked(True)
        vb.invertY(True)
        self.bg_img = pg.ImageItem()
        vb.addItem(self.bg_img)
        self.roi = pg.RectROI([0, 0], [8, 8], pen=pg.mkPen("#ffcc00", width=2))
        vb.addItem(self.roi)
        self._vb = vb
        self.roi.sigRegionChanged.connect(self._update_roi_label)
        self.roi.sigRegionChangeFinished.connect(self._schedule)
        lv.addWidget(glw, 1)
        self.lbl_roi = QtWidgets.QLabel("ROI: --")
        self.lbl_roi.setStyleSheet("font-weight:600;")
        lv.addWidget(self.lbl_roi)

        arow = QtWidgets.QHBoxLayout()
        arow.addWidget(QtWidgets.QLabel("ROI average"))
        self.combo_avg = QtWidgets.QComboBox()
        self.combo_avg.addItems(["complex field (coherent)",
                                 "amplitude & phase (as displayed)"])
        self.combo_avg.setToolTip(
            "How the ROI is reduced AFTER the per-pixel DFT.\n"
            "complex field: mean of the complex spectra, then |.| and arg -- "
            "self-consistent, best noise rejection (the reference phasing has "
            "already removed the instrumental phase, so the pixels add in "
            "phase).\n"
            "amplitude & phase: mean of |c| and the circular mean of arg(c) -- "
            "literally the average of what the amplitude and phase maps show.")
        self.combo_avg.currentIndexChanged.connect(self._schedule)
        arow.addWidget(self.combo_avg, 1)
        lv.addLayout(arow)
        split.addWidget(leftw)

        # ---------------- right: the three stacked spectra ------------------
        rightw = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(rightw)
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(QtWidgets.QLabel("λ"))
        self.sp_wl0 = QtWidgets.QDoubleSpinBox()
        self.sp_wl1 = QtWidgets.QDoubleSpinBox()
        for s in (self.sp_wl0, self.sp_wl1):
            s.setRange(0.1, 100.0)
            s.setDecimals(3)
            s.setSuffix(" µm")
            s.setSingleStep(0.1)
            s.valueChanged.connect(self._schedule)
        crow.addWidget(self.sp_wl0)
        crow.addWidget(QtWidgets.QLabel("to"))
        crow.addWidget(self.sp_wl1)
        crow.addSpacing(12)
        crow.addWidget(QtWidgets.QLabel("N points"))
        self.sp_nfreq = QtWidgets.QSpinBox()
        self.sp_nfreq.setRange(16, 4096)
        self.sp_nfreq.setValue(256)
        self.sp_nfreq.setToolTip("Number of wavelengths in the DFT grid. Cost "
                                 "scales as N x scan points x ROI pixels.")
        self.sp_nfreq.valueChanged.connect(self._schedule)
        crow.addWidget(self.sp_nfreq)
        crow.addSpacing(12)
        self.chk_unwrap = QtWidgets.QCheckBox("Unwrap phase")
        self.chk_unwrap.setToolTip("DISPLAY ONLY: plot np.unwrap(phi) so the "
                                   "±π jumps disappear. The Stokes signs are "
                                   "always taken from the wrapped phase.")
        self.chk_unwrap.toggled.connect(self._replot_phase)
        crow.addWidget(self.chk_unwrap)
        crow.addStretch(1)
        b_exp = QtWidgets.QPushButton("Export curves…")
        b_exp.setToolTip("Choose which curves to write out, as one CSV table or "
                         "an .npz bundle.")
        b_exp.clicked.connect(self._open_export_dialog)
        crow.addWidget(b_exp)
        rv.addLayout(crow)

        self.p_amp = pg.PlotWidget()
        self.p_ph = pg.PlotWidget()
        self.p_st = pg.PlotWidget()
        for p, ylabel in ((self.p_amp, "|field| (a.u.)"),
                          (self.p_ph, "phase (rad)"),
                          (self.p_st, "Stokes")):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("left", ylabel)
            p.addLegend(offset=(-10, 10))
        self.p_ph.getAxis("left").setTicks(_PI_TICKS)
        self.p_st.setLabel("bottom", "wavelength (µm)")
        self.p_ph.setXLink(self.p_amp)
        self.p_st.setXLink(self.p_amp)
        for p in (self.p_amp, self.p_ph, self.p_st):
            rv.addWidget(p, 1)

        # Shared vertical cursor: dragging it in any plot moves it in all three
        # and drives the main window's lambda slider, so the maps follow.
        self.cursors = []
        for p in (self.p_amp, self.p_ph, self.p_st):
            ln = pg.InfiniteLine(angle=90, movable=True,
                                 pen=pg.mkPen("#ff8800", width=1,
                                              style=QtCore.Qt.PenStyle.DashLine))
            ln.setZValue(10)
            ln.sigPositionChanged.connect(self._on_cursor_moved)
            p.addItem(ln, ignoreBounds=True)
            self.cursors.append(ln)
        self.lbl_cursor = QtWidgets.QLabel("cursor: --")
        self.lbl_cursor.setStyleSheet("font-weight:600;")
        rv.addWidget(self.lbl_cursor)
        split.addWidget(rightw)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([440, 880])
        root.addWidget(split, 1)
        self.status = QtWidgets.QLabel("Pick a pixel or a ROI.")
        self.status.setStyleSheet("color:#888;")
        root.addWidget(self.status)

    def _connect_app(self):
        """Recompute whenever a main-window setting the spectra depend on changes.

        Deliberately NOT connected to the lambda slider or the amplitude mask:
        neither affects these curves, and a slider drag must not trigger DFTs.
        """
        app = self.app
        for sig in (app.combo_apod.currentIndexChanged,
                    app.combo_center.currentIndexChanged,
                    app.chk_ftwin.toggled,
                    app.sp_ft0.valueChanged,
                    app.sp_ft1.valueChanged,
                    app.chk_phase.toggled,
                    app.combo_i45src.currentIndexChanged,
                    app.chk_norm.toggled,
                    app.combo_mode.currentIndexChanged):
            sig.connect(self._schedule)
        for s in app.sp_gain.values():
            s.valueChanged.connect(self._schedule)

    # -------------------------------------------------------------- helpers
    def _schedule(self, *a):
        self._timer.start()

    def _bg_slot(self):
        i = self.combo_bg.currentIndex()
        return self.app._i45_source() if i == 3 else self.app.slots[i]

    def _cube_list(self):
        """[(key, label, slot, gain)] for M1, M2, M3 and the I45 source."""
        app = self.app
        out = [(f"M{k+1}", f"M{chr(0x2080 + k + 1)}", app.slots[k], app._gain(k))
               for k in range(3)]
        out.append(("I45", "I₄₅", app._i45_source(), app._i45_gain()))
        return out

    def _ref_shape(self):
        """Frame shape the ROI is expressed in: the background cube's, or the
        first cube that is loaded if that slot is still empty."""
        for slot in [self._bg_slot()] + [s for _k, _l, s, _g in self._cube_list()]:
            if slot is not None:
                return slot["raw"].shape[1:]
        return None

    def _refresh_background(self, *a):
        slot = self._bg_slot()
        if slot is None:
            self.bg_img.clear()
            self.status.setText("That slot is empty — load the cube, or pick another.")
            return
        img = np.asarray(slot["raw"], np.float32).mean(axis=0)
        self.bg_img.setImage(img, autoLevels=True)
        h, w = img.shape
        self.roi.maxBounds = QtCore.QRectF(0, 0, w, h)
        # First show: park a sensible ROI in the middle of the frame.
        if self.roi.size().x() <= 0 or self.roi.pos().x() + self.roi.size().x() > w:
            s = max(1, min(8, min(h, w)))
            self.roi.setPos((w - s) // 2, (h - s) // 2)
            self.roi.setSize((s, s))
        self._vb.autoRange()
        self._update_roi_label()

    def _init_lambda_range(self):
        wl = self.app.wl
        if wl is None or not len(wl):
            self.sp_wl0.setValue(3.8)
            self.sp_wl1.setValue(4.4)
            return
        for s, v in ((self.sp_wl0, float(np.min(wl))), (self.sp_wl1, float(np.max(wl)))):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self.sp_nfreq.blockSignals(True)
        self.sp_nfreq.setValue(int(np.clip(len(wl), 16, 4096)))
        self.sp_nfreq.blockSignals(False)

    def _on_sel_mode(self, *a):
        pixel = self.combo_sel.currentIndex() == 1
        if pixel:
            self.roi.setSize((1, 1))
        for h in self.roi.getHandles():
            h.setVisible(not pixel)
        self._update_roi_label()
        self._schedule()

    def _roi_indices(self, h, w):
        """ROI -> (r0, r1, c0, c1), clipped, always at least one pixel.

        pyqtgraph is in row-major image order here, so the ROI's x is the column
        and its y is the row.
        """
        p, s = self.roi.pos(), self.roi.size()
        c0 = int(np.floor(p.x()))
        r0 = int(np.floor(p.y()))
        c1 = c0 + max(1, int(round(s.x())))
        r1 = r0 + max(1, int(round(s.y())))
        c0 = int(np.clip(c0, 0, max(0, w - 1)))
        r0 = int(np.clip(r0, 0, max(0, h - 1)))
        c1 = int(np.clip(c1, c0 + 1, w))
        r1 = int(np.clip(r1, r0 + 1, h))
        return r0, r1, c0, c1

    def _update_roi_label(self, *a):
        shape = self._ref_shape()
        if shape is None:
            self.lbl_roi.setText("ROI: --")
            return
        h, w = shape
        r0, r1, c0, c1 = self._roi_indices(h, w)
        n = (r1 - r0) * (c1 - c0)
        self.lbl_roi.setText(f"rows {r0}:{r1}   cols {c0}:{c1}   ({n} px)")

    def _roi_reduce(self, spec):
        """(n_freq, hr, wr) complex -> (amplitude, phase), averaged over the ROI."""
        flat = spec.reshape(spec.shape[0], -1)
        if self.combo_avg.currentIndex() == 0:          # coherent complex mean
            c = np.nanmean(flat, axis=1)
            return np.abs(c), np.angle(c)
        amp = np.nanmean(np.abs(flat), axis=1)          # incoherent amplitude
        ph = np.angle(np.nanmean(np.exp(1j * np.angle(flat)), axis=1))
        return amp, ph

    # -------------------------------------------------------------- compute
    def _recompute(self, *a):
        app = self.app
        shape = self._ref_shape()
        if shape is None:
            self.status.setText("Load a cube first.")
            return
        if app.proc is None:
            app.proc = HyperspectralProcessor()
        h, w = shape
        r0, r1, c0, c1 = self._roi_indices(h, w)
        npx = (r1 - r0) * (c1 - c0)
        wl0, wl1 = sorted((self.sp_wl0.value(), self.sp_wl1.value()))
        if wl1 - wl0 <= 0:
            self.status.setText("Empty wavelength range.")
            return
        nf = self.sp_nfreq.value()
        apod = app.combo_apod.currentText()
        center = ("barycenter" if app.combo_center.currentText().startswith("bary")
                  else "envelope")
        ftwin = ((app.sp_ft0.value(), app.sp_ft1.value())
                 if app.chk_ftwin.isChecked() else None)
        phasing = app.chk_phase.isChecked() and app.ref is not None
        notes = []
        t0 = time.perf_counter()

        amps, phases, wl_axis = {}, {}, None
        for key, label, slot, gain in self._cube_list():
            if slot is None:
                continue
            raw = slot["raw"]
            if raw.shape[1:] != shape:
                notes.append(f"{key}: {raw.shape[1:]} != {shape}, skipped")
                continue
            sub = raw[:, r0:r1, c0:c1]
            # The I45 cube keeps its RAW phase: the maps read its |field| without
            # phasing, and phasing the reference against itself would flatten its
            # own phase curve to zero.
            ref_sub = None
            if phasing and key != "I45":
                if app.ref["raw"].shape == raw.shape:
                    ref_sub = app.ref["raw"][:, r0:r1, c0:c1]
                else:
                    notes.append(f"{key}: ref shape mismatch, unphased")
            try:
                wl, spec = app.proc.compute_hyperspectral(
                    slot["pos"], sub, wl_start=wl0, wl_stop=wl1, n_freq=nf,
                    reference_cube=ref_sub,
                    expected_zero_mm=DEFAULT_ZPD_MM, search_mm=DEFAULT_ZPD_WINDOW_MM,
                    apod_type=apod, ft_window_mm=ftwin,
                    positions_calibrated=slot["cal"], center_method=center,
                    complex_out=True)
            except Exception as e:  # noqa: BLE001
                notes.append(f"{key}: {e}")
                continue
            if spec is None:
                notes.append(f"{key}: too few scan positions for a DFT")
                continue
            a_, p_ = self._roi_reduce(np.asarray(spec))
            amps[key] = a_ * gain
            phases[key] = p_
            wl_axis = np.asarray(wl, float)

        if wl_axis is None:
            self.status.setText("Nothing to transform — load at least one cube.")
            return
        self._wl = wl_axis
        self._amps, self._phases = amps, phases
        self._plot_fields()
        self._plot_stokes()
        dt = time.perf_counter() - t0
        base = (f"{npx} px   |   {len(wl_axis)} λ in [{wl0:.3f}, {wl1:.3f}] µm"
                f"   |   apod {apod}   |   centre {center}"
                + ("   |   phasing ON" if phasing else "")
                + f"   |   {dt:.2f} s")
        if notes:
            base += "   |   " + "; ".join(notes)
        self.status.setText(base)
        self._sync_cursor_from_app()

    # --------------------------------------------------------------- plotting
    def _plot_fields(self):
        self._curves = [dict(key="wavelength_um", label="wavelength (µm)",
                             group="axis", y=self._wl)]
        self.p_amp.clear()
        for key, label, _s, _g in self._cube_list():
            if key not in self._amps:
                continue
            self.p_amp.plot(self._wl, self._amps[key], name=f"|{label}|",
                            pen=pg.mkPen(PEN_M[key], width=2))
            self._curves.append(dict(key=f"amp_{key}", label=f"|{label}|",
                                     group="amplitude", y=self._amps[key]))
            self._curves.append(dict(key=f"phase_{key}", label=f"φ({label})",
                                     group="phase", y=self._phases[key]))
        self._replot_phase()
        self._readd_cursors()

    def _replot_phase(self, *a):
        """Redraw only the phase panel (the unwrap toggle is display-only)."""
        if self._wl is None or not getattr(self, "_phases", None):
            return
        unwrap = self.chk_unwrap.isChecked()
        self.p_ph.clear()
        for key, label, _s, _g in self._cube_list():
            if key not in self._phases:
                continue
            y = self._phases[key]
            self.p_ph.plot(self._wl, np.unwrap(y) if unwrap else y,
                           name=f"φ({label})", pen=pg.mkPen(PEN_M[key], width=2))
        ax = self.p_ph.getAxis("left")
        if unwrap:
            ax.setTicks(None)
            self.p_ph.enableAutoRange(axis="y")
        else:
            ax.setTicks(_PI_TICKS)
            self.p_ph.setYRange(-np.pi, np.pi)
        self._readd_cursors()

    def _plot_stokes(self):
        self.p_st.clear()
        have = [k for k in ("M1", "M2", "M3") if k in self._amps]
        if len(have) < 3:
            self.p_st.setTitle("load M₁, M₂ and M₃ for Stokes")
            self._readd_cursors()
            return
        self.p_st.setTitle(None)
        app = self.app
        M1, M2, M3 = (self._amps[k] for k in ("M1", "M2", "M3"))
        P1, P2, P3 = (self._phases[k] for k in ("M1", "M2", "M3"))
        mag, signs = stokes_from_amp_phase(M1, M2, M3, P1, P2, P3)
        s_signed = [mag[k] * signs[k] for k in range(3)]
        i45 = self._amps.get("I45")
        have_i45 = i45 is not None
        normalize = app.chk_norm.isChecked() and have_i45
        signed = normalize or (app.combo_mode.currentIndex() == 0)
        s0 = (2.0 * i45 - s_signed[1]) if have_i45 else None

        if normalize:
            eps = max(1e-12 * _nanmax_safe(np.abs(s0)), 1e-30)
            s0_safe = np.where(np.abs(s0) <= eps, np.nan, s0)
            vals = [s_signed[k] / s0_safe for k in range(3)]
            names = ["S₁/S₀", "S₂/S₀", "S₃/S₀"]
        elif signed:
            vals = s_signed
            names = ["S₁", "S₂", "S₃"]
        else:
            vals = mag
            names = ["|S₁|", "|S₂|", "|S₃|"]
        for k in range(3):
            self.p_st.plot(self._wl, vals[k], name=names[k],
                           pen=pg.mkPen(PEN_S[k], width=2))
            self._curves.append(dict(key=f"S{k+1}", label=names[k],
                                     group="stokes", y=np.asarray(vals[k], float)))
        if have_i45 and not normalize:
            self.p_st.plot(self._wl, s0, name="S₀",
                           pen=pg.mkPen(PEN_S0, width=1,
                                        style=QtCore.Qt.PenStyle.DashLine))
        if have_i45:
            self._curves.append(dict(key="S0", label="S₀ = 2·I₄₅ − S₂",
                                     group="stokes", y=np.asarray(s0, float)))
        self._readd_cursors()

    def _readd_cursors(self):
        """plot.clear() drops the cursor lines, so put them back after a redraw."""
        for p, ln in zip((self.p_amp, self.p_ph, self.p_st), self.cursors):
            if ln.scene() is None:
                p.addItem(ln, ignoreBounds=True)

    # ----------------------------------------------------------- the cursor
    def _on_cursor_moved(self, line):
        if self._syncing or self._wl is None:
            return
        self._syncing = True
        try:
            x = float(line.value())
            for ln in self.cursors:
                if ln is not line:
                    ln.setValue(x)
            i = int(np.argmin(np.abs(self._wl - x)))
            self._update_cursor_label(i)
            # Drive the main window's lambda slider so the maps follow the cursor.
            app = self.app
            if app.wl is not None and len(app.wl):
                j = int(np.argmin(np.abs(np.asarray(app.wl, float) - self._wl[i])))
                if j != app.sl_wl.value():
                    app.sl_wl.setValue(j)
        finally:
            self._syncing = False

    def _sync_cursor_from_app(self):
        """Park the cursor at the main window's current lambda."""
        if self._syncing or self._wl is None or not len(self._wl):
            return
        lam = self.app._current_lambda()
        self._syncing = True
        try:
            x = float(self._wl[len(self._wl) // 2]) if lam is None else float(lam)
            for ln in self.cursors:
                ln.setValue(x)
            self._update_cursor_label(int(np.argmin(np.abs(self._wl - x))))
        finally:
            self._syncing = False

    def _update_cursor_label(self, i):
        bits = [f"λ = {self._wl[i]:.4f} µm"]
        for c in self._curves:
            if c["group"] in ("amplitude", "stokes") and i < len(c["y"]):
                bits.append(f"{c['label']} = {c['y'][i]:.4g}")
        self.lbl_cursor.setText("cursor:   " + "    ".join(bits))

    # ----------------------------------------------------------- export
    def _open_export_dialog(self):
        avail = [c for c in self._curves if c["group"] != "axis"]
        if self._wl is None or not avail:
            QtWidgets.QMessageBox.information(
                self, "Export curves", "Nothing to export yet — compute some curves "
                                       "first.")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export curves")
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel("Curves to export (the wavelength axis is "
                                     "always included):"))
        cb_all = QtWidgets.QCheckBox("Select all")
        cb_all.setChecked(True)
        v.addWidget(cb_all)
        checks = []
        for group, title in (("amplitude", "Amplitudes"), ("phase", "Phases"),
                             ("stokes", "Stokes")):
            rows = [c for c in avail if c["group"] == group]
            if not rows:
                continue
            lab = QtWidgets.QLabel(title)
            lab.setStyleSheet("font-weight:600; margin-top:6px;")
            v.addWidget(lab)
            for c in rows:
                cb = QtWidgets.QCheckBox(c["label"])
                cb.setChecked(True)
                v.addWidget(cb)
                checks.append((cb, c))
        cb_all.toggled.connect(lambda on: [cb.setChecked(on) for cb, _ in checks])
        frow = QtWidgets.QHBoxLayout()
        frow.addWidget(QtWidgets.QLabel("Format:"))
        combo = QtWidgets.QComboBox()
        combo.addItems(["CSV (one table, one column per curve)",
                        "NPZ bundle (named arrays)"])
        frow.addWidget(combo, 1)
        v.addLayout(frow)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        sel = [c for cb, c in checks if cb.isChecked()]
        if not sel:
            return
        self._do_export(sel, combo.currentIndex())

    def _do_export(self, sel, fmt):
        app = self.app
        shape = self._ref_shape()
        if shape is None:
            return
        r0, r1, c0, c1 = self._roi_indices(*shape)
        stem = f"roi_spectra_r{r0}-{r1}_c{c0}-{c1}"
        if fmt == 1:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save NPZ bundle",
                os.path.join(app._last_dir, stem + ".npz"), "NumPy archive (*.npz)")
            if not path:
                return
            data = {"wavelength_um": self._wl}
            data.update({c["key"]: np.asarray(c["y"], float) for c in sel})
            try:
                np.savez(path, roi=np.array([r0, r1, c0, c1]), **data)
            except Exception as e:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
                return
            self.status.setText(f"Exported {len(sel)} curves → "
                                f"{os.path.basename(path)}")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV table", os.path.join(app._last_dir, stem + ".csv"),
            "CSV (*.csv)")
        if not path:
            return
        cols = [np.asarray(self._wl, float)] + [np.asarray(c["y"], float) for c in sel]
        names = ["wavelength_um"] + [c["key"] for c in sel]
        try:
            np.savetxt(path, np.column_stack(cols), delimiter=",", fmt="%.6g",
                       header=",".join(names), comments="")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
            return
        self.status.setText(f"Exported {len(sel)} curves → "
                            f"{os.path.basename(path)}")


def _save_tiff(path, arr):
    """Save a 2-D float array as a 32-bit float TIFF (values preserved, NaN kept).
    Uses tifffile if present, else Pillow's 'F' mode."""
    arr = np.asarray(arr, dtype=np.float32)
    try:
        import tifffile
        tifffile.imwrite(path, arr)
        return
    except Exception:  # noqa: BLE001
        pass
    from PIL import Image
    Image.fromarray(arr, mode="F").save(path)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = StokesMapsApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
