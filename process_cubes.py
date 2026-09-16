"""
process_cubes.py -- batch pre-processing of spectral hypercubes (.npz).

A small, guided command-line tool that mirrors the analysis_app Recompute + Process
panels so results match the GUI. For a folder of per-Z/angle .npz cubes it:

  (1) checks each cube reaches the target wavelength (default 4.3 µm) and, if not
      (or if you force it), RECOMPUTES the spectrum from the stored raw
      interferogram using HyperspectralProcessor -- you choose the wavelength
      range, number of spectral points, ZPD centre method and apodization,
      exactly as in the Recompute panel;
  (2) optionally applies a rectangular or circular ROI;
  (3) optionally applies the unitary flat-field (÷ frame_at_λ / max) from the
      Process panel.

Each input file is written to the output folder as <name>_proc.npz with the
processed spectrum_cube, wavelengths, z/angle and a 'processing' metadata record.

Run:  gui/.venv/Scripts/python process_cubes.py  [input_folder]
"""
import glob
import json
import os
import sys

import numpy as np

from instruments.hyperspectral import (
    HyperspectralProcessor, resolve_n_points, DEFAULT_ZPD_MM, DEFAULT_ZPD_WINDOW_MM)

TARGET_WL_UM = 4.3           # we want the spectra to cover at least this wavelength
_KEEP_KEYS = ("z_value_mm", "z_unit", "angle_value_deg", "angle_unit",
              "twins_positions_mm", "twins_positions_calibrated_mm")


# ------------------------------------------------------------------ I/O
def read_cube(path):
    """Load one .npz -> dict(cube, wl, raw, pos, cal, meta, keep)."""
    with np.load(path, allow_pickle=True) as d:
        f = set(d.files)
        meta = {}
        if "metadata" in f:
            try:
                meta = dict(d["metadata"].item())
            except Exception:  # noqa: BLE001
                meta = {}
        cube = (np.asarray(d["spectrum_cube"], np.float32) if "spectrum_cube" in f
                else np.asarray(d["spectrum_cubes"][0], np.float32) if "spectrum_cubes" in f
                else None)
        wl = np.asarray(d["wavelengths"], float).ravel() if "wavelengths" in f else None
        raw = (np.asarray(d["raw_interferogram"], np.float32) if "raw_interferogram" in f
               else np.asarray(d["raw_interferograms"][0], np.float32) if "raw_interferograms" in f
               else None)
        applied = bool(meta.get("motor_calibration_applied", False))
        pos, cal = None, False
        if applied and "twins_positions_calibrated_mm" in f:
            pos, cal = np.asarray(d["twins_positions_calibrated_mm"], float), True
        elif "twins_positions_mm" in f:
            pos, cal = np.asarray(d["twins_positions_mm"], float), False
        keep = {k: d[k] for k in _KEEP_KEYS if k in f}
    return dict(cube=cube, wl=wl, raw=raw, pos=pos, cal=cal, meta=meta, keep=keep)


def save_cube(out_dir, base, wl, cube, keep, proc, raw=None):
    meta = dict(proc.get("_src_meta", {}) or {}); meta.pop("_src_meta", None)
    meta["processing"] = {k: v for k, v in proc.items() if k != "_src_meta"}
    kw = dict(spectrum_cube=np.asarray(cube, np.float32),
              metadata=np.array(meta, dtype=object),
              metadata_json=json.dumps(meta, default=str, indent=2))
    if wl is not None:
        kw["wavelengths"] = np.asarray(wl)
    if raw is not None:                               # so the Stokes app can DFT it
        kw["raw_interferogram"] = np.asarray(raw, np.float32)
    for k, v in keep.items():
        kw[k] = v
    np.savez(os.path.join(out_dir, base + "_proc.npz"), **kw)


# ---------------------------------------------------------- processing steps
def needs_recompute(wl, target=TARGET_WL_UM):
    """True if the axis is missing or does not reach the target wavelength."""
    return wl is None or not len(wl) or float(np.nanmax(wl)) < target - 1e-6


def recompute(proc, rec, params):
    """(wl, cube) from the raw interferogram, mirroring the Recompute panel."""
    if rec["raw"] is None or rec["pos"] is None:
        return rec["wl"], rec["cube"]              # nothing to recompute from
    n_freq = resolve_n_points(len(rec["pos"]), manual=params["n_freq"])
    wl, cube = proc.compute_hyperspectral(
        rec["pos"], rec["raw"], wl_start=params["wl_start"], wl_stop=params["wl_stop"],
        n_freq=n_freq, apod_type=params["apod"], ft_window_mm=params["ft_window_mm"],
        expected_zero_mm=DEFAULT_ZPD_MM, search_mm=DEFAULT_ZPD_WINDOW_MM,
        positions_calibrated=rec["cal"], center_method=params["center"])
    return wl, cube


def apply_roi(cube, roi, mask_outside=True):
    """Crop/mask a (n, h, w) array. roi=None -> unchanged. rect -> crop;
    circle -> a disc CENTRED in the frame, radius = frac * min(h, w)/2, cropped to
    its bounding box. `mask_outside` NaNs pixels outside the circle (use for the
    spectrum cube; set False for the raw interferogram, which a DFT can't NaN)."""
    if roi is None or cube is None:
        return cube
    h, w = cube.shape[1:]
    if roi["type"] == "rect":
        x0 = max(0, int(roi["x"])); y0 = max(0, int(roi["y"]))
        x1 = min(w, x0 + int(roi["w"])); y1 = min(h, y0 + int(roi["h"]))
        return cube[:, y0:y1, x0:x1]
    if roi["type"] == "circle":
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0            # frame centre
        r = float(roi["frac"]) * (min(h, w) / 2.0)       # radius from the fraction
        x0, x1 = max(0, int(np.floor(cx - r))), min(w, int(np.ceil(cx + r)) + 1)
        y0, y1 = max(0, int(np.floor(cy - r))), min(h, int(np.ceil(cy + r)) + 1)
        sub = cube[:, y0:y1, x0:x1]
        if mask_outside:
            sub = sub.astype(np.float32).copy()
            yy, xx = np.mgrid[y0:y1, x0:x1]
            sub[:, (xx - cx) ** 2 + (yy - cy) ** 2 > r * r] = np.nan
        return sub
    return cube


def apply_flatfield(cube, wl, wl_ff):
    """Divide every band by the UNITARY flat = frame(λ_ff)/max(frame) (peak = 1),
    correcting illumination shape while preserving the intensity scale."""
    if cube is None or wl is None or not len(wl):
        return cube
    idx = int(np.argmin(np.abs(np.asarray(wl, float) - wl_ff)))
    frame = cube[idx].astype(np.float32)
    fmax = float(np.nanmax(np.abs(frame))) if frame.size else 0.0
    if fmax <= 0:
        return cube
    unit = frame / fmax
    safe = np.where(np.abs(unit) < 1e-6, np.nan, unit)
    return (cube / safe[None, :, :]).astype(np.float32)


# ------------------------------------------------------------------ prompts
def ask(prompt, default=None):
    d = "" if default is None else f" [{default}]"
    r = input(f"{prompt}{d}: ").strip()
    return r if r else ("" if default is None else str(default))


def ask_yesno(prompt, default=False):
    r = ask(prompt + " (y/n)", "y" if default else "n").lower()
    return r.startswith("y")


def ask_float(prompt, default):
    while True:
        try:
            return float(ask(prompt, default))
        except ValueError:
            print("  please enter a number.")


def ask_int(prompt, default):
    while True:
        try:
            return int(ask(prompt, default))
        except ValueError:
            print("  please enter an integer.")


def ask_choice(prompt, options, default):
    r = ask(f"{prompt} {options}", default).lower()
    return r if r in options else default


# ------------------------------------------------------------------ main
def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else ask("Input folder of .npz cubes")
    if not os.path.isdir(in_dir):
        print("Not a folder:", in_dir); return
    paths = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
    if not paths:
        print("No .npz files in", in_dir); return
    out_dir = ask("Output folder", os.path.join(in_dir, "processed"))
    os.makedirs(out_dir, exist_ok=True)

    proc = HyperspectralProcessor()

    # (1) wavelength check --------------------------------------------------
    cubes = [read_cube(p) for p in paths]
    short = [os.path.basename(p) for p, c in zip(paths, cubes) if needs_recompute(c["wl"])]
    have_raw = all(c["raw"] is not None for c in cubes)
    print(f"\n{len(paths)} cube(s). Target coverage: {TARGET_WL_UM} µm.")
    if short:
        print(f"{len(short)} cube(s) do NOT reach {TARGET_WL_UM} µm, e.g. {short[:3]}")
    else:
        print(f"All cubes already reach {TARGET_WL_UM} µm.")
    do_recompute = ask_yesno("Recompute spectra from the raw interferogram?",
                             default=bool(short))
    rec_params = None
    if do_recompute:
        if not have_raw:
            print("  NOTE: some files have no raw_interferogram; those are kept as-is.")
        rec_params = dict(
            wl_start=ask_float("  λ start (µm)", 3.8),
            wl_stop=ask_float("  λ stop (µm)", 4.4),
            n_freq=ask_int("  N spectral points (0 = Auto)", 0),
            center=("barycenter" if ask_choice("  ZPD centre", ["envelope", "barycenter"],
                                               "envelope").startswith("bary") else "envelope"),
            apod=ask_choice("  apodization",
                            ["gaussian", "happ-genzel", "blackman-harris-3",
                             "blackman-harris-4", "boxcar"], "gaussian"),
            ft_window_mm=None)
        if rec_params["wl_stop"] < TARGET_WL_UM:
            print(f"  WARNING: λ stop < {TARGET_WL_UM} µm; the result won't reach the target.")

    # (2) ROI ---------------------------------------------------------------
    roi = None
    if ask_yesno("\nApply a ROI before processing?", default=False):
        t = ask_choice("  ROI type", ["rect", "circle"], "rect")
        if t == "rect":
            roi = dict(type="rect", x=ask_int("  x0", 0), y=ask_int("  y0", 0),
                       w=ask_int("  width", 100), h=ask_int("  height", 100))
        else:
            roi = dict(type="circle",
                       frac=ask_float("  radius (fraction of frame, 0-1)", 0.5))

    # (3) flat-field --------------------------------------------------------
    flat_wl = None
    if ask_yesno("\nApply flat-field (÷ unitary frame at λ)?", default=False):
        flat_wl = ask_float("  flat-field λ (µm)", TARGET_WL_UM)

    # keep the raw interferogram so the output also loads in the Stokes app
    keep_raw = ask_yesno("\nKeep the raw interferogram in each output file "
                         "(needed to load the folder in the Stokes app)?", default=True)
    if keep_raw and not have_raw:
        print("  NOTE: some files have no raw_interferogram; those outputs won't "
              "be loadable by the Stokes app.")

    # process ---------------------------------------------------------------
    print()
    saved = 0
    for path, c in zip(paths, cubes):
        base = os.path.splitext(os.path.basename(path))[0]
        wl, cube = c["wl"], c["cube"]
        steps = {"_src_meta": c["meta"]}
        if do_recompute and rec_params is not None:   # recompute all -> uniform axis
            wl2, cube2 = recompute(proc, c, rec_params)
            if cube2 is not None:
                wl, cube = wl2, cube2
                steps["recompute"] = {k: rec_params[k] for k in
                                      ("wl_start", "wl_stop", "n_freq", "center", "apod")}
            elif needs_recompute(wl):
                print(f"  WARNING {base}: no raw interferogram, still < {TARGET_WL_UM} µm")
        if cube is None:
            print(f"  skip {base}: no spectrum cube"); continue
        if roi is not None:
            cube = apply_roi(cube, roi); steps["roi"] = roi
        if flat_wl is not None:
            cube = apply_flatfield(cube, wl, flat_wl); steps["flat_field_wl_um"] = flat_wl
        # raw interferogram, ROI-cropped the same way (no NaN mask so a DFT works)
        raw_out = None
        if keep_raw and c["raw"] is not None:
            raw_out = (apply_roi(c["raw"], roi, mask_outside=False)
                       if roi is not None else c["raw"])
            steps["raw_kept"] = True
        save_cube(out_dir, base, wl, cube, c["keep"], steps, raw=raw_out)
        saved += 1
        rawtxt = f" +raw{raw_out.shape}" if raw_out is not None else ""
        print(f"  [{saved}/{len(paths)}] {base} -> {base}_proc.npz  cube{cube.shape}{rawtxt}")
    print(f"\nDone. Wrote {saved} processed cube(s) to {out_dir}")


if __name__ == "__main__":
    main()
