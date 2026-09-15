"""
dsp.py -- shared interferogram DSP: apodization window library + the
apodization-broadened resolution estimate.

Ported from the hsiAnalysis MATLAB suite (Politecnico di Milano,
github.com/hyperpolimi/hsiAnalysis): `Apodization.m` (window library) and
`FWHM_apodization.m` (instrument-lineshape FWHM = FFT of the window). These give
the standard FTIR windows (Happ-Genzel, Blackman-Harris, ...) and a resolution
estimate that accounts for the window's broadening, not just 1/L.
"""
from __future__ import annotations

import numpy as np

# Apodization window types. 'gaussian' is the NIREOS position-space window kept
# in the processors themselves; the rest are the standard index-space FTIR
# windows from Apodization.m.
APOD_TYPES = [
    "happ-genzel",
    "blackman-harris-3",
    "blackman-harris-4",
    "triangular",
    "boxcar",
]


def apodization_window(apod_type, size, center):
    """SYMMETRIC apodization window of length `size`, centred on the ZPD sample
    `center`.

    Verbatim port of Apodization.m (hsiAnalysis / HyperMeasurementApp 'inspire'):
    `x = index - center`; the cosine windows have period `size`, so BOTH wings
    taper at the SAME rate no matter where the ZPD sits -- there is NO per-wing /
    tail-length adjustment. Happ-Genzel is zeroed beyond +/- size/2; triangular is
    a symmetric triangle whose half-width is the SHORTER wing. There is NO width
    parameter. `boxcar`/unknown -> ones (no apodization).
    """
    apod_type = str(apod_type).lower()
    size = int(size)
    x = np.arange(size, dtype=float) - float(center)

    if apod_type == "happ-genzel":
        y = 0.54 + 0.46 * np.cos(2.0 * np.pi * x / size)
        y[(x < -size / 2.0) | (x > size / 2.0)] = 0.0
        return y
    if apod_type == "blackman-harris-3":
        return (0.42323 + 0.49755 * np.cos(2.0 * np.pi * x / size)
                + 0.07922 * np.cos(2.0 * 2.0 * np.pi * x / size))
    if apod_type == "blackman-harris-4":
        return (0.35875 + 0.48829 * np.cos(2.0 * np.pi * x / size)
                + 0.14128 * np.cos(2.0 * 2.0 * np.pi * x / size)
                + 0.01168 * np.cos(3.0 * 2.0 * np.pi * x / size))
    if apod_type == "triangular":
        peak = x[-1] if center > round(size / 2.0) else float(center)
        y = peak - np.abs(x)
        m = y.max()
        if m > 0:
            y = y / m
        y[y < 0] = 0.0
        return y

    # boxcar / gaussian / supergaussian / unknown -> no apodization.
    return np.ones(size)


def apodization_window_map(apod_type, size, center):
    """Per-pixel SYMMETRIC apodization: `center` is an (h, w) ZPD-index map,
    returns an (size, h, w) window centred on each pixel's OWN ZPD.

    Same window families as apodization_window (verbatim Apodization.m port,
    symmetric, no width, no per-wing scaling), vectorised across the field.
    `boxcar`/unknown -> ones.
    """
    apod_type = str(apod_type).lower()
    size = int(size)
    center = np.asarray(center, dtype=float)                 # (h, w)
    x = np.arange(size, dtype=float)[:, None, None] - center[None]     # (size,h,w)
    if apod_type == "happ-genzel":
        y = 0.54 + 0.46 * np.cos(2.0 * np.pi * x / size)
        y[(x < -size / 2.0) | (x > size / 2.0)] = 0.0
        return y
    if apod_type == "blackman-harris-3":
        return (0.42323 + 0.49755 * np.cos(2.0 * np.pi * x / size)
                + 0.07922 * np.cos(2.0 * 2.0 * np.pi * x / size))
    if apod_type == "blackman-harris-4":
        return (0.35875 + 0.48829 * np.cos(2.0 * np.pi * x / size)
                + 0.14128 * np.cos(2.0 * 2.0 * np.pi * x / size)
                + 0.01168 * np.cos(3.0 * 2.0 * np.pi * x / size))
    if apod_type == "triangular":
        peak = np.where(center > np.round(size / 2.0), x[-1], center)
        y = peak[None] - np.abs(x)
        m = y.max(axis=0, keepdims=True)
        y = np.where(m > 0, y / np.where(m > 0, m, 1.0), y)
        y[y < 0] = 0.0
        return y
    return np.ones((size,) + center.shape)


def _fourier_dir(t, s, nu):
    """Explicit matrix DFT, FourierDir.m: (Dt*s) @ exp(-2j*pi*t'*nu)."""
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = np.append(dt, dt[-1] if dt.size else 0.0)
    return (dt * s) @ np.exp(-2j * np.pi * np.outer(t, nu))


# Cache of the dimensionless FWHM constant per apod_type: the FWHM of the
# window's transform for a unit-length scan. By Fourier scaling the FWHM for a
# real scan of length L is just C / L, so we compute C once.
_fwhm_const_cache: dict = {}


def _fwhm_constant(apod_type):
    key = str(apod_type).lower()
    if key in _fwhm_const_cache:
        return _fwhm_const_cache[key]
    # Unit scan length: t in [-0.5, 0.5]. Frequency grid wide enough to bracket
    # the main lobe; dense enough to interpolate the half-max crossings.
    n_t = 512
    t = np.linspace(-0.5, 0.5, n_t)
    f_max = 20.0
    f = np.linspace(-f_max, f_max, 8001)
    if key in ("boxcar", "none", "rectangular"):
        apod = np.ones(n_t)
    else:
        apod = apodization_window(apod_type, n_t, n_t / 2.0)
    A = np.abs(_fourier_dir(t, apod, f))
    A /= A.max()
    pos, neg = f > 0, f < 0
    # interp the f where A crosses 0.5 on each side of the peak
    hi = np.interp(0.5, A[pos][::-1], f[pos][::-1])   # ascending A needed
    lo = np.interp(0.5, A[neg], f[neg])
    c = float(hi - lo)
    _fwhm_const_cache[key] = c
    return c


def apodization_fwhm(apod_type, delta_t):
    """FWHM of the instrument lineshape (in 1/[delta_t units]) for a scan of
    length `delta_t` and the given apodization window.

    Port of FWHM_apodization.m, but using Fourier scaling (FWHM = C/L) so it is
    cheap to call live. For a stage scan of `delta_t` mm this returns the
    spectral FWHM in 1/mm (stage pseudo-frequency), which the processors then
    map to nm via the calibration slope.
    """
    if not delta_t or delta_t <= 0:
        return None
    return _fwhm_constant(apod_type) / float(delta_t)
