"""
Stokes VIEWER -- precomputed z-stack for fast scrubbing.

Same structure and controls as stokes_maps_app.py, but the heavy per-pixel DFTs
are done ONCE for every z of the loaded folders (full complex spectrum, all
wavelengths, via HyperspectralProcessor.compute_hyperspectral(complex_out=True)).
After "Compute all z", moving the z or λ slider is a pure array lookup, so the
maps update instantly. The amplitude mask, phase×amp, signed/unsigned, S0
normalisation and the export button all work with no recomputation -- they read
the cached complex spectra.

Changing a DFT parameter (apodization, ZPD centre, FT window) marks the cache
stale; press "Compute all z" to rebuild it.

Run:  gui/.venv/Scripts/python stokes_viewer_app.py
"""
import sys

import numpy as np
from PyQt6 import QtCore, QtWidgets

from instruments.hyperspectral import (
    DEFAULT_ZPD_MM, DEFAULT_ZPD_WINDOW_MM, HyperspectralProcessor)
from stokes_maps_app import StokesMapsApp, _robust_peak


class StokesViewerApp(StokesMapsApp):
    def __init__(self):
        self._spec = {}          # z -> dict(G=[G1,G2,G3], ref, i45, phased)
        self._wl_axis = None
        super().__init__()
        self.setWindowTitle("Stokes viewer — precomputed z-stack (fast scrub)")
        self._add_compute_row()
        self._rewire_for_viewer()

    # ---- extra UI + rewiring on top of the shared layout -------------------
    def _add_compute_row(self):
        root = self.centralWidget().layout()
        row = QtWidgets.QHBoxLayout()
        self.btn_compute = QtWidgets.QPushButton("Compute all z")
        self.btn_compute.setToolTip("Run the per-pixel DFTs once for every z of the "
                                    "loaded folders; afterwards the sliders are instant.")
        self.btn_compute.clicked.connect(self._precompute_all)
        row.addWidget(self.btn_compute)
        self.lbl_compute = QtWidgets.QLabel("not computed")
        self.lbl_compute.setStyleSheet("color:#e8590c; font-weight:600;")
        row.addWidget(self.lbl_compute); row.addStretch(1)
        root.insertLayout(1, row)          # just under the load row

    def _rewire_for_viewer(self):
        # DFT-defining controls invalidate the cache instead of triggering a
        # (now display-only) recompute.
        for sig in (self.combo_apod.currentIndexChanged,
                    self.combo_center.currentIndexChanged,
                    self.chk_ftwin.toggled,
                    self.sp_ft0.valueChanged, self.sp_ft1.valueChanged):
            try:
                sig.disconnect()
            except Exception:  # noqa: BLE001
                pass
            sig.connect(self._mark_stale)
        self._timer.setInterval(0)          # slider scrubbing is a lookup -> instant

    def _mark_stale(self, *a):
        self._spec = {}
        self.lbl_compute.setText("params changed — press Compute all z")
        self.lbl_compute.setStyleSheet("color:#e8590c; font-weight:600;")

    # instant sliders (no DFT, no debounce needed)
    def _on_z(self, *a):
        self._update_z_label(); self._recompute()

    def _on_lambda(self, *a):
        self._update_wl_label(); self._recompute()

    # ---- precompute --------------------------------------------------------
    def _precompute_all(self):
        if self.wl is None or not self._z_axis:
            QtWidgets.QMessageBox.information(self, "Compute", "Load the folders first.")
            return
        if self.proc is None:
            self.proc = HyperspectralProcessor()
        apod = self.combo_apod.currentText()
        center = ("barycenter" if self.combo_center.currentText().startswith("bary")
                  else "envelope")
        ftwin = ((self.sp_ft0.value(), self.sp_ft1.value())
                 if self.chk_ftwin.isChecked() else None)
        wl0, wl1, nf = float(self.wl.min()), float(self.wl.max()), int(len(self.wl))

        def _spec(meas, ref_raw):
            wls, G = self.proc.compute_hyperspectral(
                meas["pos"], meas["raw"], wl_start=wl0, wl_stop=wl1, n_freq=nf,
                apod_type=apod, ft_window_mm=ftwin, expected_zero_mm=DEFAULT_ZPD_MM,
                search_mm=DEFAULT_ZPD_WINDOW_MM, positions_calibrated=meas["cal"],
                reference_cube=ref_raw, center_method=center, complex_out=True)
            self._wl_axis = wls
            return G

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._spec = {}
        try:
            for zi, z in enumerate(self._z_axis):
                self.lbl_compute.setText(f"computing z {zi+1}/{len(self._z_axis)}…")
                QtWidgets.QApplication.processEvents()
                ref_meas = self._resolve(self.ref, z)
                ref_raw = ref_meas["raw"] if ref_meas is not None else None
                entry = {"G": [None, None, None], "ref": None, "i45": None,
                         "phased": ref_raw is not None}
                for k in range(3):
                    m = self._resolve(self.slots[k], z)
                    if m is not None:
                        entry["G"][k] = _spec(m, ref_raw)      # phased spectrum
                if ref_meas is not None:                        # |ref| = I45 candidate
                    entry["ref"] = _spec(ref_meas, None)
                i45m = self._resolve(self.i45, z)
                if i45m is not None:
                    entry["i45"] = _spec(i45m, None)
                self._spec[z] = entry
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        # Adopt the processor's actual wavelength grid so the λ slider/labels and
        # every cached frame refer to exactly the same wavelengths.
        if self._wl_axis is not None:
            self.wl = np.asarray(self._wl_axis, float)
            self._update_wl_label()
        self.lbl_compute.setText(f"computed — {len(self._spec)} z (scrub freely)")
        self.lbl_compute.setStyleSheet("color:#2f9e44; font-weight:600;")
        self._recompute()

    # ---- fast display (override the DFT recompute with a lookup) ------------
    def _recompute(self, *a):
        if self.wl is None or not self._z_axis:
            return
        z = self._current_z()
        entry = self._spec.get(z)
        li = int(np.clip(self.sl_wl.value(), 0, len(self.wl) - 1))
        if entry is None:
            for k in range(3):
                self.amp_img[k].clear(); self.ph_img[k].clear()
            self.i45_amp_img.clear(); self.i45_ph_img.clear(); self.i45_amp_ov.setVisible(False)
            self.s0_img.clear()
            for k in range(3):
                self.s_img[k].clear()
            self.status.setText("Press 'Compute all z' to build the cache.")
            return

        maskpct = self.sp_mask.value()
        # --- I45 reference frame at this (z, λ) ---
        self._i45_amp = None; self._i45_phase = None; self._i45_valid = None
        gmask = None
        i45cube = entry["ref"] if self.combo_i45src.currentIndex() == 0 else entry["i45"]
        if i45cube is not None:
            Gi = i45cube[li]
            self._i45_amp = np.abs(Gi).astype(np.float64)
            self._i45_phase = np.angle(Gi).astype(np.float64)
            pk = _robust_peak(self._i45_amp)
            v = np.isfinite(self._i45_amp)
            if maskpct > 0 and pk > 0:
                v &= self._i45_amp >= (maskpct / 100.0) * pk
            self._i45_valid = v; gmask = v
            self.i45_amp_img.setImage(self._i45_amp, autoLevels=False)
            self.i45_amp_cbar.setLevels((0.0, pk if pk > 0 else 1.0))
        else:
            self.i45_amp_img.clear(); self.i45_ph_img.clear()
            self.i45_amp_ov.setVisible(False)

        phased = entry["phased"] and self.chk_phase.isChecked()
        prepared = []
        for k in range(3):
            G = entry["G"][k]
            if G is None:
                self.amp_img[k].clear(); self.ph_img[k].clear()
                self.amp_title[k].setText(f"M{k+1}")
                prepared.append(None); continue
            Gk = G[li]
            amp = np.abs(Gk).astype(np.float64)
            phase = np.angle(Gk).astype(np.float64)
            valid = np.isfinite(amp)
            if gmask is not None and gmask.shape == amp.shape:
                valid &= gmask
            else:
                pkm = _robust_peak(amp)
                if maskpct > 0 and pkm > 0:
                    valid &= amp >= (maskpct / 100.0) * pkm
            peak = _robust_peak(amp)
            self.amp_img[k].setImage(amp, autoLevels=False)
            self.amp_cbar[k].setLevels((0.0, peak if peak > 0 else 1.0))
            self.amp_title[k].setText(f"M{k+1}  |field|" + ("  [phased]" if phased else ""))
            prepared.append((amp, valid, phase, phased))

        self._prepared = prepared
        self._refresh_mask_overlay()
        self._refresh_phase()
        n_loaded = sum(p is not None for p in prepared)
        self._status_base = (f"z = {z:.4f} mm   |   λ = {self.wl[li]:.4f} µm   |   "
                             f"{n_loaded}/3 maps (precomputed)"
                             + ("   |   phasing ON" if phased else ""))
        self._compute_stokes(prepared)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = StokesViewerApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
