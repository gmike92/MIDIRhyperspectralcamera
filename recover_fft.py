"""recover_fft.py -- finish an interrupted K-space acquisition.

When an acquisition is stopped during the ACQUIRE phase, its per-position files
are saved raw-only (`processing_stage="raw_acquired"`: they have the
`raw_interferogram` but no `spectrum_cube`). This tool reads each such file,
runs the SAME per-pixel DFT the acquisition app's transform phase runs (using the
scan parameters stored in the file's own metadata: apodization, wavelength range,
n_freq, FT window, motor calibration, saturation mask), and writes a COMPLETE
file (raw interferogram + spectrum cube) into a new folder. Already-complete
files are copied through unchanged.

Usage (run from the gui/ folder so instruments + calibration resolve):
    .venv\\Scripts\\python.exe recover_fft.py "<source folder>" ["<dest folder>"]
Default dest = "<source folder>_recovered".
"""
import os
import sys
import glob
import json
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instruments.hyperspectral import (HyperspectralProcessor, resolve_n_points,
                                       DEFAULT_ZPD_MM, DEFAULT_ZPD_WINDOW_MM)
from instruments.analysis import saturation_mask, svd_denoise


def _f(v, default=None):
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return default


def recover_file(path, dest_dir, proc):
    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)
        meta = dict(d["metadata"].item()) if "metadata" in files else {}
        base = os.path.basename(path)
        out = os.path.join(dest_dir, base)

        # already processed -> copy through unchanged
        if "spectrum_cube" in files:
            np.savez(out, **{k: d[k] for k in files})
            return "copied (already complete)"
        if "raw_interferogram" not in files:
            return "skipped (no raw interferogram)"

        datacube = np.asarray(d["raw_interferogram"], dtype=float)      # (n_pos, h, w), ROI-binned
        positions = np.asarray(d["twins_positions_mm"], dtype=float)    # raw measured wedge axis
        cal_positions = (np.asarray(d["twins_positions_calibrated_mm"])
                         if "twins_positions_calibrated_mm" in files else None)
        background = np.asarray(d["background"]) if "background" in files else None
        bg_sub = bool(d["background_subtracted"]) if "background_subtracted" in files else False
        z_value = _f(d["z_value_mm"]) if "z_value_mm" in files else None
        a_value = _f(d["angle_value_deg"]) if "angle_value_deg" in files else None

    # --- scan parameters (from the file's own metadata) ---
    wl0 = _f(meta.get("wl_start_um"), 3.8)
    wl1 = _f(meta.get("wl_stop_um"), 4.4)
    apod_width = _f(meta.get("apod_width"), 0.2)
    apod_type = str(meta.get("apodization", "gaussian"))
    ft_region = str(meta.get("ft_region", "full"))
    ft_width = _f(meta.get("ft_width_mm"), 0.1)
    walkoff = meta.get("walkoff", None)
    nfreq_set = int(meta.get("n_freq_setting", 0) or 0)
    n_freq = resolve_n_points(len(positions), manual=nfreq_set)

    # --- saturation mask, EXACTLY as the acquisition phase 2 does ---
    sat_mask = None
    if meta.get("saturation_masking"):
        sat_src = datacube
        if bg_sub and background is not None:
            sat_src = datacube + np.asarray(background, dtype=float)[None, :, :]
        sat_mask = saturation_mask(sat_src, meta.get("saturation_level", 16383))

    # --- the per-pixel DFT (motor calibration applied inside, positions raw) ---
    wl, cube = proc.compute_hyperspectral(
        positions, datacube, wl_start=wl0, wl_stop=wl1,
        apod_width=apod_width, n_freq=n_freq,
        expected_zero_mm=DEFAULT_ZPD_MM, search_mm=DEFAULT_ZPD_WINDOW_MM,
        apod_type=apod_type, walkoff=walkoff,
        ft_region=ft_region, ft_width_mm=ft_width)
    if cube is None:
        return "FAILED (compute returned None)"
    if meta.get("svd_denoise"):
        cube = svd_denoise(cube, int(meta.get("svd_k", 6)))

    # --- save a COMPLETE file (same schema as _save_position_npz) ---
    meta = dict(meta)
    meta.update(processing_stage="complete",
                cube_shape=list(np.asarray(cube).shape), n_freq=int(cube.shape[0]),
                wl_min_um=float(np.min(wl)), wl_max_um=float(np.max(wl)),
                recovered_from_raw=True,
                recovered_local_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    kw = dict(
        wavelengths=wl, spectrum_cube=np.asarray(cube, dtype=np.float32),
        z_value_mm=(np.nan if z_value is None else z_value), z_unit="mm",
        angle_value_deg=(np.nan if a_value is None else a_value), angle_unit="deg",
        metadata=np.array(meta, dtype=object),
        metadata_json=json.dumps(meta, default=str, indent=2),
        raw_interferogram=np.asarray(datacube, dtype=np.float32),
        twins_positions_mm=positions)
    if cal_positions is not None:
        kw["twins_positions_calibrated_mm"] = cal_positions
    if sat_mask is not None:
        kw["saturation_mask"] = np.asarray(sat_mask, bool)
    if background is not None:
        kw["background"] = background
        kw["background_subtracted"] = bool(bg_sub)
    np.savez(out, **kw)  # uncompressed: fast write; cube barely compresses
    return f"FFT -> {cube.shape} ({n_freq} bins)"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    src = sys.argv[1]
    dest = sys.argv[2] if len(sys.argv) > 2 else src.rstrip("\\/") + "_recovered"
    os.makedirs(dest, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(src, "*.npz")))
    print(f"Recovering {len(paths)} file(s)\n  from {src}\n  to   {dest}\n")
    proc = HyperspectralProcessor()
    for i, p in enumerate(paths, 1):
        try:
            msg = recover_file(p, dest, proc)
        except Exception as e:  # noqa: BLE001
            msg = f"ERROR: {e}"
        print(f"[{i}/{len(paths)}] {os.path.basename(p)}: {msg}")
    print(f"\nDone. Complete dataset in: {dest}")


if __name__ == "__main__":
    main()
