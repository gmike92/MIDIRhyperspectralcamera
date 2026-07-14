"""
Standalone K-space hyperspectral viewer.

Open a saved measurement (.npz written by the Measure tab) WITHOUT launching the
full camera app -- to review or re-inspect past scans.

Usage:
    .venv\\Scripts\\python view_hyperspectral.py [path\\to\\kspace_*.npz]

With no argument a file dialog opens. The viewer supports the wavelength slider,
per-pixel spectra, peak-λ / peak-intensity / SAM maps, and the spectral
derivative -- the same widget the live app uses.
"""
import os
import sys

from PyQt6 import QtWidgets


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from ui.measure_kspace import HyperViewer, load_kspace_npz, kspace_metadata

    app = QtWidgets.QApplication(sys.argv)
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open K-space measurement", "", "NumPy archive (*.npz)")
    if not path:
        return
    res = load_kspace_npz(path)
    if res is None:
        QtWidgets.QMessageBox.warning(None, "K-Space Viewer",
                                      "This file has no spectral cube.")
        return
    wl, cubes, z_values, masks = res
    viewer = HyperViewer()
    viewer.setWindowTitle(f"K-Space Hyperspectral Viewer — {os.path.basename(path)}")
    viewer.set_result(wl, cubes, z_values, masks)
    viewer.set_calibration_note(kspace_metadata(path))   # read cal state from metadata
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
