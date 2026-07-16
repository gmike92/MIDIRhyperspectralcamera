"""
Stokes polarimetry app for hyperspectral MWIR measurements.

Loads a stack of FOUR hyperspectral measurements taken through a quarter-wave
plate at four angles (default 0, 22.5, 67.5, 90 deg), averages each cube over a
user-chosen wavelength range to get one intensity image per angle, and computes
the Stokes parameters S0, S1, S2, S3 per pixel.

Each measurement is a .npz saved by the acquisition app: a spectral cube
`spectrum_cube` (n_wavelengths, h, w) and a `wavelengths` axis (um). The QWP
angle is NOT stored in the file (it is in the run/folder name), so each slot has
an editable angle field -- the four slots feed the Stokes formulas as I1..I4:

    I_k = mean_lambda( spectrum_cube_k[range] )           # one image per angle
    S0  = I1 + I4
    S1  = (2*I2 - I1 - I4) / S0
    S2  = ((I1 - I4)*sqrt(2) - I1 - 2*I2 + 4*I3 - I4) / S0
    S3  = (I1 - I4) / S0

(verbatim from the user's reference code). An optional flat-field correction
divides each intensity image by that cube's frame at a chosen wavelength
(default = longest / last, the ~4.4 um blackbody frame). When it is enabled the
Stokes parameters are computed from the CORRECTED intensities; when off, from
the raw averaged intensities.

Run:  gui/.venv/Scripts/python stokes_app.py   (or stokes.bat)
"""
import glob
import json
import os
import sys

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

# Show images as (row, col) like the other apps / imagesc.
pg.setConfigOptions(imageAxisOrder="row-major")

# The analysis script works in the measurement frame = QWP filename angle MINUS
# 45 deg (a fast-axis / rotator zero offset). Two calculation methods are
# supported; each uses a different set of four angles + Stokes formulas.
ANGLE_OFFSET_DEG = -45.0


def _stokes_A(I1, I2, I3, I4):
    """Method A -- filename angles 0, 45, 67.5, 90 (frame -45, 0, 22.5, 45)."""
    S0 = I1 + I4
    S1 = (2 * I2 - I1 - I4) / S0
    S2 = ((I1 - I4) * np.sqrt(2) - I1 - 2 * I2 + 4 * I3 - I4) / S0
    S3 = (I1 - I4) / S0
    return S0, S1, S2, S3


def _stokes_B(I1, I2, I3, I4):
    """Method B -- filename angles 0, 22.5, 67.5, 90 (frame -45, -22.5, 22.5, 45)."""
    S0 = I1 + I4
    S1 = (2 * I2 + 2 * I3 - 2 * I1 - 2 * I4) / S0
    S2 = ((I1 - I4) * np.sqrt(2) - 2 * I2 + 2 * I3) / S0
    S3 = (I1 - I4) / S0
    return S0, S1, S2, S3


# Box titles show the exact formula each method computes (I_k = averaged
# intensity in slot k; S0 is the denominator for S1..S3).
TITLES_A = [
    "S₀ = I₁ + I₄",
    "S₁ = (2·I₂ − I₁ − I₄) / S₀",
    "S₂ = [(I₁ − I₄)·√2 − I₁ − 2·I₂ + 4·I₃ − I₄] / S₀",
    "S₃ = (I₁ − I₄) / S₀",
]
TITLES_B = [
    "S₀ = I₁ + I₄",
    "S₁ = 2·(I₂ + I₃ − I₁ − I₄) / S₀",
    "S₂ = [(I₁ − I₄)·√2 − 2·I₂ + 2·I₃] / S₀",
    "S₃ = (I₁ − I₄) / S₀",
]

METHODS = [
    {"name": "0, 45, 67.5, 90°  →  frame −45, 0, 22.5, 45",
     "filename_angles": [0.0, 45.0, 67.5, 90.0], "titles": TITLES_A, "stokes": _stokes_A},
    {"name": "0, 22.5, 67.5, 90°  →  frame −45, −22.5, 22.5, 45",
     "filename_angles": [0.0, 22.5, 67.5, 90.0], "titles": TITLES_B, "stokes": _stokes_B},
]


def _bwr_cmap():
    """Diverging blue-white-red colormap (blue = low, white = mid, red = high)."""
    return pg.ColorMap(pos=[0.0, 0.5, 1.0],
                       color=[(0, 0, 255), (255, 255, 255), (255, 0, 0)])


def _load_measurement(path):
    """(cube (n_wl,h,w) float32, wl (n_wl,) float64, angle_or_None) from a .npz.

    Supports per-position files (`spectrum_cube`) and stacked whole-cube files
    (`spectrum_cubes`, first map used). Reads the stored angle_value_deg if the
    file happens to carry one (angle-scan acquisitions)."""
    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)
        if "spectrum_cube" in files:
            cube = np.asarray(d["spectrum_cube"], dtype=np.float32)
        elif "spectrum_cubes" in files:
            cube = np.asarray(d["spectrum_cubes"], dtype=np.float32)
            if cube.ndim == 4:
                cube = cube[0]
        else:
            raise ValueError("No 'spectrum_cube' (or 'spectrum_cubes') in this file.")
        if "wavelengths" not in files:
            raise ValueError("No 'wavelengths' axis in this file.")
        wl = np.asarray(d["wavelengths"], dtype=float).ravel()
        angle = None
        if "angle_value_deg" in files:
            try:
                a = float(d["angle_value_deg"])
                if np.isfinite(a):
                    angle = a
            except Exception:  # noqa: BLE001
                pass
    if cube.ndim != 3:
        raise ValueError(f"spectrum_cube must be 3-D (n_wl,h,w); got shape {cube.shape}.")
    return cube, wl, angle


def _read_meta(path):
    """(angle_deg, z_mm) from a .npz WITHOUT loading the big cube -- reads only the
    small angle/z members (np.load is lazy). NaN when a value is absent."""
    angle, z = float("nan"), float("nan")
    try:
        with np.load(path, allow_pickle=True) as d:
            files = set(d.files)
            if "angle_value_deg" in files:
                try:
                    angle = float(d["angle_value_deg"])
                except Exception:  # noqa: BLE001
                    pass
            if "z_value_mm" in files:
                try:
                    z = float(d["z_value_mm"])
                except Exception:  # noqa: BLE001
                    pass
            if (not np.isfinite(angle) or not np.isfinite(z)) and "metadata_json" in files:
                m = json.loads(str(d["metadata_json"]))
                if not np.isfinite(angle):
                    angle = float(m.get("angle_value_deg", np.nan) or np.nan)
                if not np.isfinite(z):
                    z = float(m.get("z_value_mm", np.nan) or np.nan)
    except Exception:  # noqa: BLE001
        pass
    return angle, z


class StokesApp(QtWidgets.QMainWindow):
    def __init__(self, embedded=False):
        super().__init__()
        # `embedded` = shown as a panel inside the analyzer, which already loaded a
        # folder -> hide this panel's own "Load folder" button and instead get its
        # files from the host via populate_from_paths().
        self.embedded = embedded
        self.setWindowTitle("Stokes Polarimetry — hyperspectral QWP analyzer")
        self.resize(1300, 860)
        # Per-slot state: dict(cube, wl, name) or None; angles from the spin boxes.
        self.slots = [None, None, None, None]
        self._folder_metas = []      # [(path, angle, z)] from the last loaded folder
        self._last_dir = "D:\\CAMERA" if os.path.isdir("D:\\CAMERA") else ""
        self._ranges_init = False
        self._syncing = False
        self._first_stokes = True
        self.I = [None] * 4          # raw averaged intensity images
        self.Ibkg = [None] * 4       # background-divided intensity images
        self.S = [None] * 4          # Stokes maps
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # --- method selector + auto-load from a folder ---
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Formula:"))
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems([m["name"] for m in METHODS])
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        top.addWidget(self.method_combo)
        top.addSpacing(16)
        if not self.embedded:
            btn_folder = QtWidgets.QPushButton("Load folder (auto-assign by angle)…")
            btn_folder.clicked.connect(self._load_folder)
            top.addWidget(btn_folder)
            hint = QtWidgets.QLabel(
                "reads each .npz's angle and fills I1..I4 at the selected formula's "
                "angles; or load each slot by hand below")
        else:
            hint = QtWidgets.QLabel(
                "Auto-filled from the folder loaded in the analyzer (File ▸ Load "
                "folder). Adjust individual slots below if needed.")
        hint.setStyleSheet("color:#888;")
        top.addWidget(hint)
        top.addSpacing(16)
        top.addWidget(QtWidgets.QLabel("z-position:"))
        self.z_combo = QtWidgets.QComboBox()
        self.z_combo.setEnabled(False); self.z_combo.setMinimumWidth(110)
        self.z_combo.currentIndexChanged.connect(self._on_z_changed)
        top.addWidget(self.z_combo)
        top.addStretch(1)
        root.addLayout(top)

        # --- slot loaders (one group box per angle) ---
        slot_row = QtWidgets.QHBoxLayout()
        self.angle_spin = []
        self.file_lbl = []
        for k in range(4):
            gb = QtWidgets.QGroupBox(f"I{k+1}")
            gl = QtWidgets.QGridLayout(gb)
            btn = QtWidgets.QPushButton("Load…")
            btn.clicked.connect(lambda _c, kk=k: self._load_slot(kk))
            asp = QtWidgets.QDoubleSpinBox()
            asp.setRange(-360.0, 360.0); asp.setDecimals(1); asp.setSuffix(" °")
            asp.setValue(self._frame_angles()[k])
            asp.valueChanged.connect(self._recompute)
            lbl = QtWidgets.QLabel("(not loaded)")
            lbl.setStyleSheet("color:#888;"); lbl.setWordWrap(True)
            lbl.setMinimumWidth(150)
            gl.addWidget(btn, 0, 0)
            gl.addWidget(QtWidgets.QLabel("QWP"), 0, 1)
            gl.addWidget(asp, 0, 2)
            gl.addWidget(lbl, 1, 0, 1, 3)
            self.angle_spin.append(asp)
            self.file_lbl.append(lbl)
            slot_row.addWidget(gb)
        root.addLayout(slot_row)

        # --- wavelength range + flat-field controls ---
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Average λ from"))
        self.sp_wl0 = QtWidgets.QDoubleSpinBox()
        self.sp_wl0.setRange(0.0, 1e4); self.sp_wl0.setDecimals(4); self.sp_wl0.setSuffix(" µm")
        self.sp_wl1 = QtWidgets.QDoubleSpinBox()
        self.sp_wl1.setRange(0.0, 1e4); self.sp_wl1.setDecimals(4); self.sp_wl1.setSuffix(" µm")
        ctl.addWidget(self.sp_wl0); ctl.addWidget(QtWidgets.QLabel("to")); ctl.addWidget(self.sp_wl1)
        ctl.addSpacing(20)
        self.chk_bkg = QtWidgets.QCheckBox("Flat-field correction (÷ frame at λ)")
        self.sp_wlbkg = QtWidgets.QDoubleSpinBox()
        self.sp_wlbkg.setRange(0.0, 1e4); self.sp_wlbkg.setDecimals(4); self.sp_wlbkg.setSuffix(" µm")
        ctl.addWidget(self.chk_bkg); ctl.addWidget(self.sp_wlbkg)
        note = QtWidgets.QLabel("slot angle = filename QWP angle − 45°")
        note.setStyleSheet("color:#888;")
        ctl.addSpacing(16); ctl.addWidget(note)
        ctl.addStretch(1)
        for w in (self.sp_wl0, self.sp_wl1, self.sp_wlbkg):
            w.valueChanged.connect(self._recompute)
        self.chk_bkg.toggled.connect(self._recompute)
        root.addLayout(ctl)

        # --- tabs: Intensities + Stokes ---
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        # Intensities tab: 2x2 average-intensity images.
        itab = QtWidgets.QWidget(); igrid = QtWidgets.QGridLayout(itab)
        self.i_img = []; self.i_cbar = []; self.i_title = []
        vir = pg.colormap.get("viridis")
        for k in range(4):
            glw, img, cbar, lbl = self._make_panel(f"I{k+1}", vir, interactive=False)
            self.i_img.append(img); self.i_cbar.append(cbar); self.i_title.append(lbl)
            igrid.addWidget(glw, k // 2, k % 2)
        self.tabs.addTab(itab, "Intensities")

        # Stokes tab: 2x2 S0..S3 with blue-white-red + adjustable colorbars.
        stab = QtWidgets.QWidget(); sroot = QtWidgets.QVBoxLayout(stab)
        sgrid_w = QtWidgets.QWidget(); sgrid = QtWidgets.QGridLayout(sgrid_w)
        self.s_img = []; self.s_cbar = []; self.s_title = []
        bwr = _bwr_cmap()
        titles = self._method()["titles"]
        for k in range(4):
            # S0 is a plain (positive) intensity -> viridis; S1..S3 are signed
            # -> diverging blue-white-red centered on 0.
            cmap = pg.colormap.get("viridis") if k == 0 else bwr
            glw, img, cbar, lbl = self._make_panel(titles[k], cmap,
                                                   interactive=True, title_size="10pt")
            cbar.sigLevelsChanged.connect(lambda _c=None, kk=k: self._on_cbar(kk))
            self.s_img.append(img); self.s_cbar.append(cbar); self.s_title.append(lbl)
            sgrid.addWidget(glw, k // 2, k % 2)
        sroot.addWidget(sgrid_w, 1)
        # per-map colour-limit controls (min / max / Auto)
        clim_row = QtWidgets.QHBoxLayout()
        self.s_min = []; self.s_max = []
        for k in range(4):
            box = QtWidgets.QHBoxLayout()
            box.addWidget(QtWidgets.QLabel(f"S{k}:"))
            smin = QtWidgets.QDoubleSpinBox(); smax = QtWidgets.QDoubleSpinBox()
            for s in (smin, smax):
                s.setRange(-1e9, 1e9); s.setDecimals(4); s.setSingleStep(0.05)
            smin.valueChanged.connect(lambda _v=None, kk=k: self._on_spin(kk))
            smax.valueChanged.connect(lambda _v=None, kk=k: self._on_spin(kk))
            btn = QtWidgets.QPushButton("Auto")
            btn.clicked.connect(lambda _c=None, kk=k: self._auto_clim(kk))
            box.addWidget(QtWidgets.QLabel("min")); box.addWidget(smin)
            box.addWidget(QtWidgets.QLabel("max")); box.addWidget(smax)
            box.addWidget(btn)
            self.s_min.append(smin); self.s_max.append(smax)
            clim_row.addLayout(box)
            if k < 3:
                clim_row.addSpacing(12)
        sroot.addLayout(clim_row)
        self.tabs.addTab(stab, "Stokes")

        self.status = self.statusBar()
        self.status.showMessage("Load four QWP measurements (one per angle).")

    def _make_panel(self, title, cmap, interactive, title_size=None):
        """A single image + colorbar in its own GraphicsLayoutWidget."""
        glw = pg.GraphicsLayoutWidget()
        if title_size:
            glw.addLabel(title, row=0, col=0, colspan=2, size=title_size)
        else:
            glw.addLabel(title, row=0, col=0, colspan=2)
        vb = glw.addViewBox(row=1, col=0)
        vb.setAspectLocked(True); vb.invertY(True)
        lbl = glw.getItem(0, 0)              # the LabelItem, for later retitling
        img = pg.ImageItem()
        vb.addItem(img)
        cbar = pg.ColorBarItem(interactive=interactive, colorMap=cmap)
        glw.addItem(cbar, row=1, col=1)
        cbar.setImageItem(img)
        return glw, img, cbar, lbl

    # ------------------------------------------------------------- method
    def _method(self):
        return METHODS[self.method_combo.currentIndex()] if hasattr(self, "method_combo") else METHODS[0]

    def _frame_angles(self):
        """Slot (measurement-frame) angles for the current method = filename − 45."""
        return [a + ANGLE_OFFSET_DEG for a in self._method()["filename_angles"]]

    def _on_method_changed(self, *args):
        """Formula changed -> retitle the boxes, rescale colorbars, and reload the
        associated files (from the scanned folder) or just recompute."""
        for k, t in enumerate(self._method()["titles"]):
            self.s_title[k].setText(t, size="10pt")
        self._first_stokes = True                 # re-autoscale for the new formula
        if self._folder_metas:
            zsel = self.z_combo.currentData() if self.z_combo.count() else None
            self._assign_from_folder(zsel)
        else:
            fr = self._frame_angles()
            for k in range(4):
                if self.slots[k] is None:         # keep loaded slots' real angles
                    self.angle_spin[k].blockSignals(True)
                    self.angle_spin[k].setValue(fr[k])
                    self.angle_spin[k].blockSignals(False)
            self._recompute()

    # ------------------------------------------------------------- loading
    def _assign_slot(self, k, path):
        """Load one measurement into slot k (shared by manual + folder loading).
        Sets the filename label and the frame angle; does NOT recompute."""
        try:
            cube, wl, angle = _load_measurement(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Load error", f"{os.path.basename(path)}:\n{e}")
            return False
        self.slots[k] = {"cube": cube, "wl": wl, "name": os.path.basename(path)}
        self.file_lbl[k].setText(os.path.basename(path))
        self.file_lbl[k].setStyleSheet("color:#2f9e44;")
        self.file_lbl[k].setToolTip(path)
        if angle is not None:
            # Stored angle is the QWP filename angle; the script frame is -45 deg.
            self.angle_spin[k].blockSignals(True)
            self.angle_spin[k].setValue(angle + ANGLE_OFFSET_DEG)
            self.angle_spin[k].blockSignals(False)
        if not self._ranges_init:                 # seed λ range from the first cube
            lo, hi = float(np.min(wl)), float(np.max(wl))
            for sp, v in ((self.sp_wl0, lo), (self.sp_wl1, hi), (self.sp_wlbkg, hi)):
                sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
            self._ranges_init = True
        return True

    def _load_slot(self, k):
        """Manual per-slot load."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"QWP measurement for slot I{k+1}", self._last_dir,
            "Measurement (*.npz)")
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        if self._assign_slot(k, path):
            self._recompute()

    def _load_folder(self):
        """Scan a folder, read every .npz's angle (and z), and auto-fill I1..I4
        with the files at the target QWP angles. If several z-positions exist,
        ask which one to use."""
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Folder holding the QWP angle stack", self._last_dir)
        if not d:
            return
        self._last_dir = d
        paths = sorted(glob.glob(os.path.join(d, "*.npz")))
        if not paths:
            QtWidgets.QMessageBox.information(
                self, "No files", "No .npz files directly in that folder.")
            return
        self.populate_from_paths(paths)

    def populate_from_paths(self, paths):
        """Auto-assign I1..I4 from a given list of .npz paths (used by the
        standalone folder dialog AND by the analyzer to reuse its loaded folder).
        Reads each file's angle/z, fills the QWP slots, and populates the
        z-selector. No modal dialogs -- reports issues to the status bar."""
        paths = [p for p in paths if p]
        if not paths:
            return
        metas = [(p, *_read_meta(p)) for p in paths]           # (path, angle, z)
        if not any(np.isfinite(a) for _, a, _ in metas):
            self.status.showMessage(
                "Loaded files carry no angle (angle_value_deg) — cannot auto-assign "
                "QWP slots. Load the four measurements manually below.")
            return
        self._folder_metas = metas
        # Populate the in-window z-position selector (a z-stack -> several z).
        zs = sorted({round(z, 4) for _, _, z in metas if np.isfinite(z)})
        self.z_combo.blockSignals(True)
        self.z_combo.clear()
        for z in zs:
            self.z_combo.addItem(f"{z:.4f} mm", z)
        self.z_combo.setCurrentIndex(0 if zs else -1)
        self.z_combo.setEnabled(len(zs) > 1)
        self.z_combo.blockSignals(False)
        self._assign_from_folder(zs[0] if zs else None)

    def _on_z_changed(self, idx):
        """z-position dropdown changed -> reload the four angle files at that z."""
        if idx < 0 or not self._folder_metas:
            return
        self._assign_from_folder(self.z_combo.currentData())

    def _assign_from_folder(self, zsel):
        """Fill I1..I4 from the scanned folder with the files at the target QWP
        angles and the given z (zsel=None -> ignore z)."""
        metas = self._folder_metas
        targets = self._method()["filename_angles"]     # e.g. 0,45,67.5,90
        loaded, missing = [], []
        for k, ta in enumerate(targets):
            cand = [(p, a, z) for (p, a, z) in metas
                    if np.isfinite(a) and (zsel is None or abs(z - zsel) < 1e-3)]
            best = min(cand, key=lambda t: abs(t[1] - ta)) if cand else None
            if best is not None and abs(best[1] - ta) <= 1.0 and self._assign_slot(k, best[0]):
                loaded.append(f"I{k+1}←{best[1]:g}°")
            else:
                missing.append(f"{ta:g}°")
        self._recompute()
        zmsg = f" at z = {zsel:.4f} mm" if zsel is not None else ""
        msg = f"Auto-loaded {len(loaded)}/4{zmsg}: " + ", ".join(loaded)
        if missing:
            msg += "  |  no file near angle(s): " + ", ".join(missing)
        self.status.showMessage(msg)

    # ------------------------------------------------------------- compute
    def _recompute(self, *args):
        if any(s is None for s in self.slots):
            n = sum(s is not None for s in self.slots)
            self.status.showMessage(f"Loaded {n}/4 measurements — load all four to compute Stokes.")
            return
        shapes = [s["cube"].shape[1:] for s in self.slots]
        if len({shapes[0], *shapes}) != 1:
            self.status.showMessage(f"Image sizes differ across slots: {shapes} — cannot combine.")
            return
        wl0 = self.sp_wl0.value(); wl1 = self.sp_wl1.value(); wlb = self.sp_wlbkg.value()
        self.I = [None] * 4; self.Ibkg = [None] * 4
        for k, s in enumerate(self.slots):
            wl, cube = s["wl"], s["cube"]
            i0 = int(np.argmin(np.abs(wl - wl0)))
            i1 = int(np.argmin(np.abs(wl - wl1)))
            lo, hi = min(i0, i1), max(i0, i1)
            Ii = cube[lo:hi + 1].mean(axis=0).astype(np.float64)     # (h, w)
            ib = int(np.argmin(np.abs(wl - wlb)))
            with np.errstate(divide="ignore", invalid="ignore"):
                Ib = Ii / cube[ib].astype(np.float64)
            self.I[k] = Ii
            self.Ibkg[k] = Ib
        # Stokes uses the flat-field-corrected intensities when the correction is
        # on, otherwise the raw averaged intensities; formula = the chosen method.
        use_ff = self.chk_bkg.isChecked()
        I1, I2, I3, I4 = self.Ibkg if use_ff else self.I
        with np.errstate(divide="ignore", invalid="ignore"):
            S0, S1, S2, S3 = self._method()["stokes"](I1, I2, I3, I4)
        self.S = [np.where(np.isfinite(m), m, np.nan) for m in (S0, S1, S2, S3)]
        self._update_intensities()
        self._update_stokes()
        n0, n1 = sorted((i0, i1))
        self.status.showMessage(
            f"λ average {wl[n0]:.4f}–{wl[n1]:.4f} µm ({hi - lo + 1} bands) | "
            f"flat-field correction {'ON' if self.chk_bkg.isChecked() else 'off'} "
            f"(÷ frame at {wl[ib]:.4f} µm) | Stokes from "
            f"{'flat-fielded' if self.chk_bkg.isChecked() else 'raw'} intensities")

    def _update_intensities(self):
        disp = self.Ibkg if self.chk_bkg.isChecked() else self.I
        for k in range(4):
            d = disp[k]
            self.i_img[k].setImage(d, autoLevels=False)
            fin = d[np.isfinite(d)]
            lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
            if hi <= lo:
                hi = lo + 1.0
            self.i_cbar[k].setLevels((lo, hi))
            ang = self.angle_spin[k].value()
            self.i_title[k].setText(
                f"I{k+1} — frame {ang:g}°"
                + ("  (flat-fielded)" if self.chk_bkg.isChecked() else ""))

    def _update_stokes(self):
        for k in range(4):
            self.s_img[k].setImage(self.S[k], autoLevels=False)
            if self._first_stokes:
                self._auto_clim(k)
        self._first_stokes = False

    # ----------------------------------------------------- colour limits
    def _auto_clim(self, k):
        d = self.S[k]
        fin = d[np.isfinite(d)]
        if fin.size == 0:
            lo, hi = -1.0, 1.0
        elif k == 0:                                  # S0: intensity -> data range
            lo, hi = float(fin.min()), float(fin.max())
        else:                                         # S1..S3: symmetric about 0
            m = float(np.max(np.abs(fin)))
            m = m if (m > 0 and np.isfinite(m)) else 1.0
            lo, hi = -m, m
        if hi <= lo:
            hi = lo + 1e-9
        self._syncing = True
        self.s_min[k].setValue(lo); self.s_max[k].setValue(hi)
        self.s_cbar[k].setLevels((lo, hi))
        self._syncing = False

    def _on_spin(self, k):
        if self._syncing:
            return
        lo, hi = self.s_min[k].value(), self.s_max[k].value()
        if hi > lo:
            self._syncing = True
            self.s_cbar[k].setLevels((lo, hi))
            self._syncing = False

    def _on_cbar(self, k):
        if self._syncing:
            return
        lo, hi = self.s_cbar[k].levels()
        self._syncing = True
        self.s_min[k].setValue(float(lo)); self.s_max[k].setValue(float(hi))
        self._syncing = False


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = StokesApp()
    win.show()
    # Optional: files passed on the command line fill the slots in order.
    for k, arg in enumerate(sys.argv[1:5]):
        if os.path.isfile(arg):
            try:
                cube, wl, angle = _load_measurement(arg)
                win.slots[k] = {"cube": cube, "wl": wl, "name": os.path.basename(arg)}
                win.file_lbl[k].setText(os.path.basename(arg))
                win.file_lbl[k].setStyleSheet("color:#2f9e44;")
                if angle is not None:
                    win.angle_spin[k].setValue(angle + ANGLE_OFFSET_DEG)
            except Exception as e:  # noqa: BLE001
                print("skip", arg, e)
    if any(s is not None for s in win.slots):
        if not win._ranges_init:
            wl = next(s["wl"] for s in win.slots if s is not None)
            win.sp_wl0.setValue(float(np.min(wl))); win.sp_wl1.setValue(float(np.max(wl)))
            win.sp_wlbkg.setValue(float(np.max(wl))); win._ranges_init = True
        win._recompute()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
