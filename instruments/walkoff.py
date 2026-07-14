"""
walkoff.py -- TWINS birefringent walk-off (image-shift) correction.

As the TWINS wedge translates, the imaged scene drifts across the FPA (lateral
beam walk-off). For per-pixel hyperspectral DFT this is corrupting: pixel (i,j)
must see the SAME scene point at every wedge position. The drift is reproducible
and (to good approximation) linear in wedge position -- exactly the model used
in the hsiAnalysis MATLAB suite (TempHyp_Generator.m), where it is corrected by
a per-frame sub-pixel shift at a calibrated `shift_rate` [px / mm of travel].

This module provides three pieces, mirroring that suite:
  - apply_walkoff_correction(): parametric linear correction at a known rate
    (the operational path; rate measured once on a sharp sample).
  - estimate_shift_rate(): measure the rate from a sharp-sample scan by
    registering each frame and fitting shift vs. position (the calibration).
  - register_cube(): data-driven per-frame registration as a fallback when no
    rate is available (the MATLAB "registration" option / imregcorr analogue).

Registration uses FFT phase correlation with sub-pixel parabolic refinement
(pure numpy/scipy -- no scikit-image dependency).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _hann2d(shape):
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def _parab(a, b, c):
    """Sub-sample peak offset in [-0.5, 0.5] from 3 samples around the peak."""
    d = a - 2.0 * b + c
    if d == 0:
        return 0.0
    return 0.5 * (a - c) / d


def phase_correlation(ref, img, window=True):
    """Sub-pixel shift (dy, dx) of `img` relative to `ref`.

    Positive (dy, dx) means `img` is displaced by (+dy, +dx) px w.r.t. `ref`;
    undo it with scipy.ndimage.shift(img, (-dy, -dx)). Robust to global intensity
    scaling (the TWINS fringe modulation), which is why phase correlation is used
    rather than plain cross-correlation.
    """
    ref = np.asarray(ref, dtype=float)
    img = np.asarray(img, dtype=float)
    ref = ref - ref.mean()
    img = img - img.mean()
    if window:
        w = _hann2d(ref.shape)
        ref = ref * w
        img = img * w
    R = np.fft.fft2(ref)
    I = np.fft.fft2(img)
    cross = R * np.conj(I)
    mag = np.abs(cross)
    mag[mag == 0] = 1.0
    surf = np.fft.ifft2(cross / mag).real

    ny, nx = surf.shape
    py, px = np.unravel_index(int(np.argmax(surf)), surf.shape)
    dy = py + _parab(surf[(py - 1) % ny, px], surf[py, px], surf[(py + 1) % ny, px])
    dx = px + _parab(surf[py, (px - 1) % nx], surf[py, px], surf[py, (px + 1) % nx])
    # Wrap to signed shifts.
    if dy > ny / 2:
        dy -= ny
    if dx > nx / 2:
        dx -= nx
    # The correlation peak sits at -displacement; negate so the returned (dy, dx)
    # is the true displacement of `img` relative to `ref`.
    return -float(dy), -float(dx)


def register_cube(datacube, reference=0, window=True):
    """Register every frame of an (n_pos, h, w) cube to a reference frame.

    Returns (shifts, corrected_cube): shifts is (n_pos, 2) of measured (dy, dx);
    corrected_cube has each frame shifted back onto the reference grid.
    """
    cube = np.asarray(datacube, dtype=float)
    n = cube.shape[0]
    ref_idx = int(reference)
    ref = cube[ref_idx]
    shifts = np.zeros((n, 2))
    out = np.empty_like(cube)
    out[ref_idx] = cube[ref_idx]
    for k in range(n):
        if k == ref_idx:
            continue
        dy, dx = phase_correlation(ref, cube[k], window=window)
        shifts[k] = (dy, dx)
        out[k] = ndimage.shift(cube[k], (-dy, -dx), order=1, mode="nearest")
    return shifts, out


def _r2(x, y, slope, intercept):
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def estimate_shift_rate(datacube, positions, reference=0, window=True):
    """Measure the walk-off rate from a sharp-sample scan.

    Registers each frame to the reference, then fits image shift vs. stage
    position (mm) linearly for each axis. Returns a dict:
        rate_y, rate_x  : px per mm of stage travel
        ref_mm          : stage position of the reference frame (shift = 0 here)
        r2_y, r2_x      : fit quality (1 = perfectly linear/reproducible)
        shifts          : (n, 2) measured per-frame (dy, dx)
        positions       : the stage positions used
    """
    pos = np.asarray(positions, dtype=float).ravel()
    shifts, _ = register_cube(datacube, reference=reference, window=window)
    dp = pos - pos[int(reference)]
    rate_y, b_y = np.polyfit(dp, shifts[:, 0], 1)
    rate_x, b_x = np.polyfit(dp, shifts[:, 1], 1)
    return {
        "rate_y": float(rate_y),
        "rate_x": float(rate_x),
        "ref_mm": float(pos[int(reference)]),
        "r2_y": _r2(dp, shifts[:, 0], rate_y, b_y),
        "r2_x": _r2(dp, shifts[:, 1], rate_x, b_x),
        "shifts": shifts,
        "positions": pos,
    }


def apply_walkoff_correction(datacube, positions, rate_y=0.0, rate_x=0.0,
                             ref_mm=None):
    """Parametric walk-off correction: shift each frame by -rate*(pos - ref_mm).

    rate_y / rate_x are px per mm (from estimate_shift_rate, measured once on a
    sharp sample). ref_mm is the position where the shift is zero (defaults to
    the first frame). Returns the corrected cube (float). A no-op if both rates
    are 0.
    """
    cube = np.asarray(datacube, dtype=float)
    pos = np.asarray(positions, dtype=float).ravel()
    if not rate_y and not rate_x:
        return cube
    if ref_mm is None:
        ref_mm = pos[0]
    out = np.empty_like(cube)
    for k in range(cube.shape[0]):
        dp = pos[k] - float(ref_mm)
        sy, sx = rate_y * dp, rate_x * dp
        if sy == 0.0 and sx == 0.0:
            out[k] = cube[k]
        else:
            out[k] = ndimage.shift(cube[k], (-sy, -sx), order=1, mode="nearest")
    return out
