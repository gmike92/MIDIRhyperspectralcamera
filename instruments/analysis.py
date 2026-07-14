"""
analysis.py -- hyperspectral cube analysis, ported from the hsiAnalysis MATLAB
suite (github.com/hyperpolimi/hsiAnalysis):

  - saturation_mask        : flag pixels that clipped anywhere in the scan
  - roi_average            : masked spatial mean/std spectrum (excludes saturated)
  - svd_denoise            : low-rank spectral denoise (SVD score truncation)
  - peak_wavelength_map    : per-pixel wavelength of the spectral maximum
  - peak_intensity_map     : per-pixel spectral maximum
  - spectral_derivative    : 1st/2nd derivative along wavelength
  - spectral_angle_map     : SAM distance of every pixel to a reference spectrum

All operate on a real spectral cube of shape (n_freq, h, w) and a 1-D
wavelength axis (n_freq,). Pure numpy.
"""
from __future__ import annotations

import numpy as np


def saturation_mask(datacube, sat_level):
    """(h, w) boolean: True where the pixel reached `sat_level` counts at ANY
    wedge position during the scan. `datacube` is the raw (n_pos, h, w) cube."""
    cube = np.asarray(datacube)
    if cube.ndim != 3:
        return None
    return np.any(cube >= float(sat_level), axis=0)


def roi_average(cube, mask=None):
    """Spatial mean & std spectrum over the cube, excluding masked pixels.

    `mask` (h, w) True = exclude (e.g. saturated). Returns (mean, std), each
    (n_freq,). Pixels are NaN-averaged so partially-empty ROIs still work.
    """
    cube = np.asarray(cube, dtype=float)
    n_freq, h, w = cube.shape
    flat = cube.reshape(n_freq, h * w)
    if mask is not None:
        valid = ~np.asarray(mask, dtype=bool).reshape(-1)
        if not valid.any():
            nan = np.full(n_freq, np.nan)
            return nan, nan
        flat = flat[:, valid]
    return flat.mean(axis=1), flat.std(axis=1)


def svd_denoise(cube, k):
    """Low-rank denoise: keep the k strongest SVD components of the (pixels x
    bands) matrix and reconstruct. Removes spatially-incoherent noise while
    preserving the dominant spectral structure. k>=min(pixels,bands) is a no-op.
    """
    cube = np.asarray(cube, dtype=np.float32)   # keep the spectrum float32
    n_freq, h, w = cube.shape
    X = cube.reshape(n_freq, h * w).T          # (pixels, bands)
    k = int(max(1, min(k, min(X.shape))))
    if k >= min(X.shape):
        return cube
    mu = X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Xk = (U[:, :k] * S[:k]) @ Vt[:k] + mu
    return Xk.T.reshape(n_freq, h, w).astype(np.float32)


def svd_explained_variance(cube, max_components=50, sample_pixels=8000):
    """Singular-value spectrum of the (pixels x bands) data, for a scree plot.

    Returns (singular_values, explained_ratio, cumulative_ratio), each truncated
    to `max_components`. Pixels are subsampled (the singular-value spectrum is
    well estimated from a subset) and the matrix is mean-centred -- matching
    svd_denoise -- so the variance fractions correspond to the components
    svd_denoise keeps.
    """
    cube = np.asarray(cube, dtype=np.float32)
    n_freq, h, w = cube.shape
    X = cube.reshape(n_freq, h * w).T               # (pixels, bands)
    if X.shape[0] > sample_pixels:
        X = X[:: X.shape[0] // sample_pixels]
    mu = X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(X - mu, compute_uv=False)    # descending singular values
    var = sv.astype(np.float64) ** 2
    total = var.sum()
    ev = var / total if total > 0 else var
    cum = np.cumsum(ev)
    n = int(min(max_components, len(sv)))
    return sv[:n], ev[:n], cum[:n]


def peak_wavelength_map(cube, wavelengths):
    """(h, w) wavelength (same units as `wavelengths`) of the per-pixel max."""
    cube = np.asarray(cube, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)
    idx = np.argmax(cube, axis=0)
    return wl[idx]


def peak_intensity_map(cube):
    """(h, w) per-pixel spectral maximum."""
    return np.max(np.asarray(cube, dtype=float), axis=0)


def spectral_derivative(cube, wavelengths, order=1):
    """d/dλ (order 1) or d²/dλ² (order 2) of the cube along the spectral axis.

    Uses np.gradient (handles the non-uniform wavelength axis) so the output
    keeps the same shape and wavelength grid. Returns the derivative cube.
    """
    cube = np.asarray(cube, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)
    d = cube
    for _ in range(int(order)):
        d = np.gradient(d, wl, axis=0)
    return d


def continuum_line_image(cube, wavelengths, line, left, right):
    """Per-pixel continuum-subtracted line image (see CONTINUUM_SUBTRACTION.md).

    Estimates the local continuum UNDER a narrow emission line from two "shoulder"
    bands, subtracts that linear baseline across the line band, and averages the
    residual -> (h, w) line-emission map. This removes a smooth broadband (thermal)
    continuum and isolates the resonant line structure.

    `line`, `left`, `right` are (lo, hi) wavelength bands in the SAME units as
    `wavelengths`. Returns None if any band contains no samples (out of range).
    """
    cube = np.asarray(cube, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)
    lm = (wl >= line[0]) & (wl <= line[1])           # line band
    lsh = (wl >= left[0]) & (wl <= left[1])          # left shoulder
    rsh = (wl >= right[0]) & (wl <= right[1])        # right shoulder
    if not (lm.any() and lsh.any() and rsh.any()):
        return None
    wl_l, wl_r = wl[lsh].mean(), wl[rsh].mean()
    b_L = cube[lsh].mean(axis=0)                      # (h, w) continuum @ wl_l
    b_R = cube[rsh].mean(axis=0)                      # (h, w) continuum @ wl_r
    slope = (b_R - b_L) / (wl_r - wl_l + 1e-12)       # (h, w) baseline slope
    wll = wl[lm]                                      # in-line wavelengths
    baseline = b_L[None] + slope[None] * (wll[:, None, None] - wl_l)
    return (cube[lm] - baseline).mean(axis=0)         # (h, w)


def spectral_angle_map(cube, reference):
    """Spectral Angle Mapper: per-pixel angle (radians) between each spectrum
    and `reference` (n_freq,). 0 = identical spectral shape; larger = different.
    Invariant to per-pixel intensity scaling.
    """
    cube = np.asarray(cube, dtype=float)
    ref = np.asarray(reference, dtype=float).ravel()
    n_freq, h, w = cube.shape
    X = cube.reshape(n_freq, h * w)                 # (bands, pixels)
    num = ref @ X                                   # (pixels,)
    den = np.linalg.norm(ref) * np.linalg.norm(X, axis=0)
    den[den == 0] = np.nan
    cos = np.clip(num / den, -1.0, 1.0)
    return np.arccos(cos).reshape(h, w)
