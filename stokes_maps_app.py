"""
Stokes-from-hypercubes app -- phase-free Stokes retrieval from three measurements.

Workflow
--------
Load the THREE measurement hypercubes (.npz saved by the acquisition app, each
carrying a `raw_interferogram` + position axis + `wavelengths`) as M1, M2, M3,
and optionally a FOURTH hypercube used only for the phasing (background phase
correction). Pick a wavelength with the slider; the app then does exactly what
the analysis_app Phase panel does -- a per-pixel complex DFT at that wavelength
(same apodization / ZPD / FT-window / phase-reference machinery via
HyperspectralProcessor.compute_complex_map) -- and shows, in a 3x3 grid:

    row 1 : amplitude |field|   for M1, M2, M3
    row 2 : phase (rad)         for M1, M2, M3   (colour scale labelled in pi)
    row 3 : the retrieved Stokes maps from the PHASE-FREE formulae

      S1^2 = 1/2 (M2^2 + M3^2 - M1^2)
      S2^2 = 1/2 (M1^2 + M3^2 - M2^2)
      S3^2 = 1/2 (M1^2 + M2^2 - M3^2)

|S_i| = sqrt(clip(S_i^2, 0)). By default the SIGNED parameters are shown:
|S_i| times a per-pixel sign  sgn(M2+phi3), sgn(M1+M3), sgn(phi1-phi2).

Run:  gui/.venv/Scripts/python stokes_maps_app.py
"""
import os
import sys

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
    """Sign map with 0 -> +1 (so an exactly-zero expression doesn't null a pixel)."""
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


# Stokes titles. M_k = amplitude map k, phi_k = phase map k. The magnitude is
# |S_i| = sqrt(clip(S_i^2, 0)); the SIGNED value multiplies it by a per-pixel
# sign taken from the expressions the user specified.
TITLES_SIGNED = [
    "S₁ = sgn(M₂+φ₃)·√[½(M₂² + M₃² − M₁²)]",
    "S₂ = sgn(M₁+M₃)·√[½(M₁² + M₃² − M₂²)]",
    "S₃ = sgn(φ₁−φ₂)·√[½(M₁² + M₂² − M₃²)]",
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
        b_export = QtWidgets.QPushButton("Export maps…")
        b_export.setToolTip("Choose which maps to export and the file format "
                            "(PNG figure, float TIFF, NumPy .npy, CSV, or an .npz bundle).")
        b_export.clicked.connect(self._open_export_dialog)
        top.addWidget(b_export)
        root.addLayout(top)

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
        self.combo_center.addItems(["envelope (field)", "barycentre (per-pixel)"])
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
        self._recompute()

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
        self._recompute()

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
        self._recompute()

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
                f"M{k+1}  |field|" + ("  [phased]" if ref_raw is not None else ""))
            prepared.append((amp, valid, phase))

        self._prepared = prepared
        self._refresh_mask_overlay()
        n_loaded = sum(p is not None for p in prepared)
        base = (f"λ = {lam:.4f} µm   |   apod {apod}   |   centre {center}   |   "
                f"{n_loaded}/3 maps"
                + ("   |   phasing ON" if phasing else ""))
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
        if any(p is None for p in prepared):
            for k in range(3):
                self.s_img[k].clear()
            self.s0_img.clear()
            self._status("   |   load all three maps for Stokes")
            return
        shapes = {p[0].shape for p in prepared}
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

        M1, M2, M3 = (p[0] for p in prepared)
        P1, P2, P3 = (p[2] for p in prepared)
        combined = prepared[0][1] & prepared[1][1] & prepared[2][1]
        n_ok = int(np.count_nonzero(combined))
        if n_ok == 0:
            for k in range(3):
                self.s_img[k].clear()
            self.s0_img.clear()
            self._status("   |   0 pixels pass the amplitude mask — lower 'Amp mask %'")
            return

        mag = [np.sqrt(np.clip(0.5 * (M2**2 + M3**2 - M1**2), 0.0, None)),
               np.sqrt(np.clip(0.5 * (M1**2 + M3**2 - M2**2), 0.0, None)),
               np.sqrt(np.clip(0.5 * (M1**2 + M2**2 - M3**2), 0.0, None))]
        signs = [_sign(M2 + P3), _sign(M1 + M3), _sign(P1 - P2)]
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
