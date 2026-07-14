"""
calibration.py -- shared TWINS / Gemini calibration loaders.

Ported from the reference repo (gmike92/Hyperspectral_MIDIR_pumpprobe,
calibration.py) so the wavelength axis and the motor-nonlinearity correction
here are identical to the reference setup. Two tab-separated 2-row files live
under ``Twins/ASRC calibration/``:

    parameters_cal.txt   Spectral calibration: row0 = wavelength (µm),
                         row1 = reciprocal (stage pseudo-frequency). Maps the
                         stage-Fourier frequencies onto real wavelengths.

    parameters_int.txt   Motor-nonlinearity reference: row0 = nominal stage
                         position (mm), row1 = interferogram of a known source.
                         Lets us recover the *real* position axis (analytic-
                         signal phase trick) and remove the wedge motor's
                         reproducible nonlinearity from every scan.

Both files are cached at module load (opened at most once per process). Paths
are resolved relative to this package (``gui/Twins/ASRC calibration``) first,
then the current working directory -- so it works no matter where Python is
launched from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# This file is gui/instruments/calibration.py -> package root is gui/.
_PKG_ROOT = Path(__file__).resolve().parent.parent
_CAL_DIR = Path("Twins") / "ASRC calibration"
# When frozen by PyInstaller the data files are extracted under _MEIPASS.
_BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", _PKG_ROOT))

_SPECTRAL_PATHS = [
    _BUNDLE_ROOT / _CAL_DIR / "parameters_cal.txt",     # bundled (frozen) / package
    _PKG_ROOT / _CAL_DIR / "parameters_cal.txt",        # gui/Twins/ASRC calibration/...
    Path(".") / _CAL_DIR / "parameters_cal.txt",        # CWD-relative
    _PKG_ROOT.parent / _CAL_DIR / "parameters_cal.txt",  # project-root-relative
]

_POSITION_PATHS = [
    _BUNDLE_ROOT / _CAL_DIR / "parameters_int.txt",
    _PKG_ROOT / _CAL_DIR / "parameters_int.txt",
    Path(".") / _CAL_DIR / "parameters_int.txt",
    _PKG_ROOT.parent / _CAL_DIR / "parameters_int.txt",
]

# Module-level caches: None = not yet attempted, (a, b) = loaded,
# (None, None) = tried and failed (don't retry every call).
_spectral_cache = None
_position_cache = None
_position_path = None    # name of the position-cal file actually loaded


def _read_two_row_file(path: Path):
    """Read a tab-separated file with two rows: row 0 = x-axis, row 1 = y-axis."""
    import pandas as pd
    ref = pd.read_csv(path, sep="\t", header=None)
    return (ref.iloc[0].to_numpy(dtype="float64"),
            ref.iloc[1].to_numpy(dtype="float64"))


def get_spectral_calibration():
    """Return (wavelength_cal_um, reciprocal_cal) or (None, None)."""
    global _spectral_cache
    if _spectral_cache is not None:
        return _spectral_cache
    for p in _SPECTRAL_PATHS:
        if p.exists():
            try:
                wl, rk = _read_two_row_file(p)
                print(f"[OK] Loaded spectral calibration: {p.name}  "
                      f"({wl.min():.2f}-{wl.max():.2f} µm)")
                _spectral_cache = (wl, rk)
                return _spectral_cache
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Could not read spectral calibration {p}: {e}")
    print("[WARN] parameters_cal.txt not found -- using simple 1/freq conversion.")
    _spectral_cache = (None, None)
    return _spectral_cache


def get_position_calibration():
    """Return (position_ref_mm, amplitude_ref) or (None, None).

    The nominal-stage positions and the reference-source interferogram used by
    calibrate_position_axis().
    """
    global _position_cache, _position_path
    if _position_cache is not None:
        return _position_cache
    for p in _POSITION_PATHS:
        if p.exists():
            try:
                pos, amp = _read_two_row_file(p)
                print(f"[OK] Loaded position calibration: {p.name}")
                _position_cache = (pos, amp)
                _position_path = p.name
                return _position_cache
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Could not read position calibration {p}: {e}")
    print("[WARN] parameters_int.txt not found -- motor jitter correction disabled.")
    _position_cache = (None, None)
    return _position_cache


def position_calibration_status():
    """(available: bool, filename: str|None) for the motor-nonlinearity cal.

    `available` is True only when parameters_int.txt loaded successfully, i.e.
    calibrate_position_axis() will actually correct the axis (not pass it
    through unchanged). Lets callers record in metadata whether the saved
    spectra were computed on the calibrated wedge axis.
    """
    pos, amp = get_position_calibration()
    return (pos is not None and amp is not None), _position_path


def get_real_position_axis(reference) -> np.ndarray:
    """Recover the real (normalized) position axis from a reference interferogram.

    Analytic-signal phase trick (NIREOS get_real_position_axis):
        FFT -> zero negative freqs -> iFFT -> unwrap(-angle) -> normalize [0,1].
    """
    ref = np.asarray(reference).squeeze()
    fft_ref = np.fft.fft(ref)
    half = int(np.floor(len(ref) / 2) - 1)
    if half > 0:
        fft_ref[:half] = 0.0
    phase = np.unwrap(-np.angle(np.fft.ifft(fft_ref)))
    a, b = float(phase.min()), float(phase.max())
    if b - a == 0:
        return np.linspace(0.0, 1.0, len(ref))
    return (phase - a) / (b - a)


def calibrate_position_axis(position_axis) -> np.ndarray:
    """Apply motor-nonlinearity correction to a nominal stage axis.

    Interpolates the stored reference IFG onto the user's (oversampled) axis,
    runs get_real_position_axis to extract the corrected axis, then downsamples
    back to the input length. Returns the input unchanged if the position
    calibration isn't available or anything goes wrong.
    """
    pos_ref, amp_ref = get_position_calibration()
    if pos_ref is None or amp_ref is None:
        return np.asarray(position_axis)

    try:
        from scipy import interpolate as _interp
        position_axis = np.asarray(position_axis).squeeze()
        if position_axis.size < 4:
            return position_axis

        d_user = float(np.mean(np.diff(position_axis)))
        d_ref = float(np.mean(np.diff(pos_ref)))
        if d_ref == 0:
            return position_axis
        factor = int(max(1, np.ceil(abs(d_user / d_ref))))

        oversmp_pos = np.linspace(position_axis[0], position_axis[-1],
                                  factor * position_axis.size)
        f_ref = _interp.interp1d(pos_ref, amp_ref, kind="cubic",
                                 bounds_error=False,
                                 fill_value=(amp_ref[0], amp_ref[-1]))
        ref_interp = f_ref(oversmp_pos)

        calib_norm = get_real_position_axis(ref_interp)
        calib_overs = calib_norm * (position_axis[-1] - position_axis[0]) + position_axis[0]
        calibrated = calib_overs[0:-1:factor]
        if calibrated.size >= position_axis.size:
            return calibrated[:position_axis.size]
        pad = np.full(position_axis.size - calibrated.size,
                      calibrated[-1] if calibrated.size else 0.0)
        return np.concatenate([calibrated, pad])
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Position calibration failed, using raw axis: {e}")
        return np.asarray(position_axis)


if __name__ == "__main__":
    wl, rk = get_spectral_calibration()
    pos, amp = get_position_calibration()
    if pos is not None:
        nominal = np.linspace(23.8, 24.8, 120)
        corrected = calibrate_position_axis(nominal)
        dev_um = (corrected - nominal) * 1000.0
        print(f"[demo] motor correction over 120 pts: "
              f"max deviation {np.abs(dev_um).max():.2f} µm")
