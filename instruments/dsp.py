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
    "gaussian",
    "happ-genzel",
    "blackman-harris-3",
    "blackman-harris-4",
    "triangular",
    "supergaussian",
    "boxcar",
]


def _supergauss(x, x1, x2, index, fraction=200.0):
    """superGauss from Apodization.m: exponent 2*index, =1/fraction at x1,x2."""
    x0 = (x1 + x2) / 2.0
    denom = np.log(fraction) ** (1.0 / (2.0 * index))
    tau = abs(x1 - x0) / denom if denom != 0 else 1.0
    if tau == 0:
        return np.ones_like(x)
    return np.exp(-((x - x0) / tau) ** (2 * index))


def apodization_window(apod_type, size, center, sg_index=7):
    """Index-space apodization window of length `size`, peaked at the ZPD `center`.

    ASYMMETRIC-AWARE: the window peaks (=1) at the ZPD and tapers each wing
    INDEPENDENTLY down to its edge value at that wing's array end -- the left
    wing is scaled over `center` samples, the right wing over `size-1-center`.
    So for an off-centre / single-sided interferogram nothing is zeroed, the full
    long tail is kept (resolution preserved) and both ends taper smoothly to ~0
    (sidelobes suppressed). For a centred ZPD this reduces to the standard
    symmetric Happ-Genzel / Blackman-Harris / super-gaussian windows of
    Apodization.m (hsiAnalysis). `boxcar` and the position-space `gaussian`
    (handled in the processors) are unchanged.
    """
    apod_type = str(apod_type).lower()
    size = int(size)

    if apod_type in ("boxcar", "none", "rectangular"):
        return np.ones(size)

    # Signed normalised distance from the ZPD: u = 0 at the burst, u = -1 at the
    # first sample, u = +1 at the last -- each wing stretched over its own length.
    center = float(center)
    left = center if center > 0 else 1.0
    right = (size - 1 - center) if (size - 1 - center) > 0 else 1.0
    d = np.arange(size, dtype=float) - center
    u = np.clip(np.where(d <= 0, d / left, d / right), -1.0, 1.0)
    au = np.abs(u)

    if apod_type == "happ-genzel":            # edge value 0.08
        return 0.54 + 0.46 * np.cos(np.pi * u)
    if apod_type == "blackman-harris-3":      # edge ~0.005
        return (0.42323 + 0.49755 * np.cos(np.pi * u)
                + 0.07922 * np.cos(2 * np.pi * u))
    if apod_type == "blackman-harris-4":      # edge ~0.00006
        return (0.35875 + 0.48829 * np.cos(np.pi * u)
                + 0.14128 * np.cos(2 * np.pi * u)
                + 0.01168 * np.cos(3 * np.pi * u))
    if apod_type == "triangular":             # peak 1 at ZPD, 0 at both edges
        return 1.0 - au
    if apod_type == "supergaussian":          # edge value 1/200
        return np.exp(-np.log(200.0) * au ** (2 * sg_index))

    # Unknown -> boxcar (no apodization).
    return np.ones(size)


def apodization_window_map(apod_type, size, center):
    """Per-pixel apodization: `center` is an (h, w) ZPD-index map, returns an
    (size, h, w) window with each pixel peaked (=1) at its OWN ZPD.

    Identical window families and asymmetric-wing scaling as apodization_window
    (which handles the scalar-centre case), just vectorised across the field so a
    per-pixel centre map can be used. `boxcar`/unknown -> ones.
    """
    apod_type = str(apod_type).lower()
    size = int(size)
    center = np.asarray(center, dtype=float)                 # (h, w)
    if apod_type in ("boxcar", "none", "rectangular"):
        return np.ones((size,) + center.shape)
    # Signed normalised distance from each pixel's ZPD, each wing over its own
    # length (u = 0 at the burst, -1 at the first sample, +1 at the last).
    left = np.where(center > 0, center, 1.0)                 # (h, w)
    right = np.where((size - 1 - center) > 0, size - 1 - center, 1.0)
    d = np.arange(size, dtype=float)[:, None, None] - center[None]     # (size,h,w)
    u = np.clip(np.where(d <= 0, d / left[None], d / right[None]), -1.0, 1.0)
    au = np.abs(u)
    if apod_type == "happ-genzel":
        return 0.54 + 0.46 * np.cos(np.pi * u)
    if apod_type == "blackman-harris-3":
        return (0.42323 + 0.49755 * np.cos(np.pi * u)
                + 0.07922 * np.cos(2 * np.pi * u))
    if apod_type == "blackman-harris-4":
        return (0.35875 + 0.48829 * np.cos(np.pi * u)
                + 0.14128 * np.cos(2 * np.pi * u)
                + 0.01168 * np.cos(3 * np.pi * u))
    if apod_type == "triangular":
        return 1.0 - au
    if apod_type == "supergaussian":
        return np.exp(-np.log(200.0) * au ** (2 * 7))
    return np.ones((size,) + center.shape)


def _fourier_dir(t, s, nu):
    """Explicit matrix DFT, FourierDir.m: (Dt*s) @ exp(-2j*pi*t'*nu)."""
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = np.append(dt, dt[-1] if dt.size else 0.0)
    return (dt * s) @ np.exp(-2j * np.pi * np.outer(t, nu))


# Cache of the dimensionless FWHM constant per (apod_type, sg_index): the FWHM
# of the window's transform for a unit-length scan. By Fourier scaling the FWHM
# for a real scan of length L is just C / L, so we compute C once.
_fwhm_const_cache: dict = {}


def _fwhm_constant(apod_type, sg_index=7):
    key = (str(apod_type).lower(), int(sg_index))
    if key in _fwhm_const_cache:
        return _fwhm_const_cache[key]
    # Unit scan length: t in [-0.5, 0.5]. Frequency grid wide enough to bracket
    # the main lobe; dense enough to interpolate the half-max crossings.
    n_t = 512
    t = np.linspace(-0.5, 0.5, n_t)
    f_max = 20.0
    f = np.linspace(-f_max, f_max, 8001)
    if key[0] in ("boxcar", "none", "rectangular"):
        apod = np.ones(n_t)
    else:
        apod = apodization_window(apod_type, n_t, n_t / 2.0, sg_index)
    A = np.abs(_fourier_dir(t, apod, f))
    A /= A.max()
    pos, neg = f > 0, f < 0
    # interp the f where A crosses 0.5 on each side of the peak
    hi = np.interp(0.5, A[pos][::-1], f[pos][::-1])   # ascending A needed
    lo = np.interp(0.5, A[neg], f[neg])
    c = float(hi - lo)
    _fwhm_const_cache[key] = c
    return c


def apodization_fwhm(apod_type, delta_t, sg_index=7):
    """FWHM of the instrument lineshape (in 1/[delta_t units]) for a scan of
    length `delta_t` and the given apodization window.

    Port of FWHM_apodization.m, but using Fourier scaling (FWHM = C/L) so it is
    cheap to call live. For a stage scan of `delta_t` mm this returns the
    spectral FWHM in 1/mm (stage pseudo-frequency), which the processors then
    map to nm via the calibration slope.
    """
    if not delta_t or delta_t <= 0:
        return None
    return _fwhm_constant(apod_type, sg_index) / float(delta_t)
