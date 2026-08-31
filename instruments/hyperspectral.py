"""
hyperspectral.py -- per-pixel TWINS DFT (K-space hyperspectral).

Verbatim port of the repo's HyperspectralProcessor
(gmike92/Labview-pumprobepython, sub_kspace_lw.py): takes a 3-D datacube
(n_positions, h, w) of ROI frames acquired while scanning the TWINS wedge and
runs an independent DFT for every pixel, returning a spectrum cube
(n_freq, h, w) plus the wavelength axis (µm via the NIREOS calibration).

Heavy step is `phase_kernel.conj().T @ flat`; keep the ROI modest (a spectrum
cube is n_freq * h * w complex128 during compute).
"""
import numpy as np
from pathlib import Path


DEFAULT_START_MM = 23.8
DEFAULT_STOP_MM = 24.8
DEFAULT_N_STEPS = 100
DEFAULT_APODIZATION = 0.2
DEFAULT_WL_START = 8.0       # µm
DEFAULT_WL_STOP = 14.0       # µm

DEFAULT_CALIBRATION_FILE = r".\Twins\ASRC calibration\parameters_cal.txt"

# Spectral output sampling ("Auto"), imported from the reference repo. Zero-
# filling does NOT add true spectral resolution (that is fixed by the scan
# length / max OPD) -- it only interpolates the spectrum so the curve looks
# smooth. Auto n_freq = ZEROFILL_FACTOR x scan steps, clamped to a sane band.
ZEROFILL_FACTOR = 1.5
ZEROFILL_MIN = 512
ZEROFILL_MAX = 4096

# Centerburst (ZPD) search defaults (NIREOS TWINS wedge): the burst sits ~here
# with small run-to-run drift; detection is the envelope max within +/- window.
DEFAULT_ZPD_MM = 24.33
DEFAULT_ZPD_WINDOW_MM = 0.1


def resolve_n_points(n_steps, manual=None):
    """Number of spectral output bins (interpolation only).

    A manual value > 0 wins; otherwise "Auto" = ZEROFILL_FACTOR x n_steps,
    clamped to [ZEROFILL_MIN, ZEROFILL_MAX]. Does NOT change the true spectral
    resolution -- that is set by the scan length.
    """
    if manual and manual > 0:
        return int(manual)
    return int(np.clip(ZEROFILL_FACTOR * int(n_steps), ZEROFILL_MIN, ZEROFILL_MAX))


def find_centerburst(signal_1d, positions, expected_zero_mm=None, search_mm=None):
    """Locate the ZPD (center burst) index from a 1-D interferogram.

    Uses the analytic-signal (Hilbert) envelope rather than argmax(|signal|):
    the envelope is smooth, so it picks the true burst instead of jumping to the
    tallest individual fringe or to a baseline edge artifact. If
    ``expected_zero_mm`` is given the search is limited to +/- ``search_mm``
    around it (default 5% of the scan span); otherwise the outer ~3% of points
    are excluded so baseline roll-off at the ends can't win.
    """
    s = np.asarray(signal_1d, dtype=float).ravel()
    n = s.size
    if n < 4:
        return int(np.argmax(np.abs(s))) if n else 0
    try:
        from scipy.signal import hilbert
        env = np.abs(hilbert(s - s.mean()))
    except Exception:  # noqa: BLE001
        env = np.abs(s - s.mean())

    pos = np.asarray(positions, dtype=float).ravel()
    span = abs(pos[-1] - pos[0]) if n > 1 else 0.0

    mask = np.ones(n, dtype=bool)
    if expected_zero_mm is not None and span > 0:
        hw = search_mm if search_mm is not None else max(0.05 * span, 3.0 * span / n)
        mask = np.abs(pos - float(expected_zero_mm)) <= hw
        if not mask.any():
            mask = np.ones(n, dtype=bool)
    else:
        guard = max(1, int(0.03 * n))
        mask[:guard] = False
        mask[-guard:] = False
    return int(np.argmax(np.where(mask, env, -np.inf)))


def barycenter_map(sig):
    """Per-pixel ZPD index via the I^2 barycentre (MATLAB
    center = round((1:N)*(I.^2)/(sum(I.^2)+1))), computed INDEPENDENTLY for every
    pixel so a ZPD that shifts across the field of view is followed per pixel.

    `sig` is the (n_pos, h, w) baseline-removed interferogram; returns an (h, w)
    integer index map (0-based, clipped to the valid range). The `+1` in the
    denominator regularises signal-free pixels (they collapse toward index 0).
    """
    w2 = np.asarray(sig, dtype=float) ** 2
    n = w2.shape[0]
    k = np.arange(n, dtype=float)[:, None, None]
    c = np.round(np.sum(k * w2, axis=0) / (np.sum(w2, axis=0) + 1.0)).astype(int)
    return np.clip(c, 0, n - 1)


class HyperspectralProcessor:
    """
    Process a 3D datacube (n_positions, h, w) into a spectrum cube.
    Each pixel gets its own DFT independently.
    """

    def __init__(self, calibration_file=None):
        self.calibration_file = calibration_file or DEFAULT_CALIBRATION_FILE
        self.wavelength_cal = None
        self.reciprocal_cal = None
        self._load_calibration()

    def _load_calibration(self):
        try:
            import pandas as pd
            cal_path = Path(self.calibration_file)
            if not cal_path.exists():
                import sys
                rel = Path("Twins") / "ASRC calibration" / "parameters_cal.txt"
                # _MEIPASS = PyInstaller bundle dir when frozen; else the package root.
                base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
                alt_paths = [
                    base / rel,                                  # bundled / package-relative
                    Path(__file__).resolve().parent.parent / rel,
                    Path(".") / rel,                             # CWD-relative
                ]
                for alt in alt_paths:
                    if alt.exists():
                        cal_path = alt
                        break
            if cal_path.exists():
                ref = pd.read_csv(cal_path, sep="\t", header=None)
                self.wavelength_cal = ref.iloc[0].to_numpy(dtype='float64')
                self.reciprocal_cal = ref.iloc[1].to_numpy(dtype='float64')
                print(f"[OK] KSpace: Loaded calibration: {cal_path.name}")
        except Exception as e:
            print(f"[WARN] KSpace calibration: {e}")

    def _get_frequency_limits(self, wl_start, wl_stop):
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(1.0 / self.wavelength_cal, self.reciprocal_cal,
                          kind="linear", fill_value="extrapolate")
            return float(fn(1.0 / wl_stop)), float(fn(1.0 / wl_start))
        return 1.0 / wl_stop, 1.0 / wl_start

    def _freq_to_wavelength(self, frequencies):
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(self.reciprocal_cal, 1.0 / self.wavelength_cal,
                          kind="linear", fill_value="extrapolate")
            return 1.0 / fn(frequencies)
        return 1.0 / frequencies

    # -- scan-parameter estimation (imported from sub_twins_lw) --------------
    def max_step_um(self, wl_short_um, samples_per_cycle=5):
        """Maximum stage step (µm) for `samples_per_cycle` points per optical
        cycle at the shortest wavelength (2 = Nyquist, 5 = safe oversampling)."""
        if not wl_short_um or wl_short_um <= 0:
            return None
        if not samples_per_cycle or samples_per_cycle <= 0:
            return None
        if self.wavelength_cal is None or self.reciprocal_cal is None:
            return wl_short_um * 1000.0 / float(samples_per_cycle)
        try:
            from scipy.interpolate import interp1d
            fn = interp1d(1.0 / self.wavelength_cal, self.reciprocal_cal,
                          kind="linear", fill_value="extrapolate")
            k_max = float(fn(1.0 / wl_short_um))
            if k_max <= 0:
                return None
            return (1.0 / (float(samples_per_cycle) * k_max)) * 1000.0
        except Exception:  # noqa: BLE001
            return None

    def nyquist_step_um(self, wl_short_um):
        """Nyquist (2 samples/cycle) stage step at the shortest wavelength."""
        return self.max_step_um(wl_short_um, samples_per_cycle=2)

    def estimate_resolution_nm(self, scan_range_mm, wl_center_um, apod_type="gaussian"):
        """Spectral resolution (FWHM, nm) from the stage scan range, using the
        calibration's local slope (accounts for TWINS birefringence). The stage
        pseudo-frequency FWHM is 1/L for the legacy gaussian, or the apodization-
        broadened FWHM (FFT of the window) for a named FTIR window."""
        if not scan_range_mm or scan_range_mm <= 0:
            return None
        if not wl_center_um or wl_center_um <= 0:
            return None
        delta_recip = 1.0 / scan_range_mm
        if apod_type and str(apod_type).lower() != "gaussian":
            try:
                from instruments.dsp import apodization_fwhm
                fwhm = apodization_fwhm(apod_type, scan_range_mm)
                if fwhm:
                    delta_recip = fwhm
            except Exception:  # noqa: BLE001
                pass
        if self.wavelength_cal is None or self.reciprocal_cal is None:
            return (wl_center_um ** 2) * delta_recip * 1000.0
        try:
            from scipy.interpolate import interp1d
            fn = interp1d(1.0 / self.wavelength_cal, self.reciprocal_cal,
                          kind="linear", fill_value="extrapolate")
            inv_lambda_c = 1.0 / wl_center_um
            eps = max(inv_lambda_c * 1e-3, 1e-6)
            slope = float((fn(inv_lambda_c + eps) - fn(inv_lambda_c - eps)) / (2 * eps))
            if slope == 0:
                return None
            delta_inv_lambda = abs(delta_recip / slope)
            return (wl_center_um ** 2) * delta_inv_lambda * 1000.0
        except Exception:  # noqa: BLE001
            return None

    def compute_hyperspectral(self, positions, datacube,
                               wl_start=8.0, wl_stop=14.0,
                               apod_width=0.2, n_freq=200, reference_cube=None, invert=False,
                               expected_zero_mm=None, search_mm=None,
                               apod_type="gaussian", walkoff=None,
                               ft_region="full", ft_width_mm=0.1, ft_window_mm=None,
                               positions_calibrated=False, center_method="envelope",
                               complex_out=False):
        """
        Compute per-pixel DFT on a (n_pos, h, w) datacube.

        `complex_out=True` returns the full COMPLEX spectrum G(ω) = F·e^{-iΦ}
        (phased when a reference is given, else the raw spectrum) instead of the
        real/absorptive magnitude -- same forward transform, all wavelengths at
        once, for callers that need the quadratures (e.g. the Stokes viewer).

        `center_method`: how the apodization ZPD centre is found --
        "envelope" (default) = one Hilbert-envelope centre-burst for the whole
        field (signed spatial sum); "barycenter" = an independent I^2 barycentre
        per pixel, so a ZPD that varies across the field is followed per pixel.
        If reference_cube is provided, extracts spectral phase and returns the Absorptive (Real) part.

        `positions_calibrated`: set True when `positions` is ALREADY the
        motor-corrected axis (e.g. reloaded from a saved file's
        twins_positions_calibrated_mm), so the nonlinearity correction is NOT
        applied a second time. Default False = raw measured axis -> calibrate.

        Returns:
            wavelengths : 1D array (n_freq,)
            spectrum_cube : 3D array (n_freq, h, w)
        """
        positions = np.asarray(positions, dtype=float)
        datacube = np.asarray(datacube, dtype=float)
        n_pos, h, w = datacube.shape
        if n_pos < 3:
            return None, None

        # Remove the TWINS wedge motor's reproducible nonlinearity: replace the
        # nominal stage positions with the calibrated axis (parameters_int.txt).
        # No-op if the position calibration file isn't present, or if the caller
        # says the axis is already calibrated (avoids double-correction).
        if not positions_calibrated:
            try:
                from instruments.calibration import calibrate_position_axis
                positions = np.asarray(calibrate_position_axis(positions), dtype=float)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] KSpace: motor calibration skipped: {e}")

        # Walk-off correction: shift every frame back onto a common grid so each
        # pixel sees the same scene point across the scan (parametric rate from a
        # sharp-sample calibration). walkoff = {rate_y, rate_x, ref_mm}.
        if walkoff:
            try:
                from instruments.walkoff import apply_walkoff_correction
                ry = float(walkoff.get("rate_y", 0.0))
                rx = float(walkoff.get("rate_x", 0.0))
                rm = walkoff.get("ref_mm", None)
                datacube = apply_walkoff_correction(datacube, positions, ry, rx, rm)
                if reference_cube is not None:
                    reference_cube = apply_walkoff_correction(
                        np.asarray(reference_cube, dtype=float), positions, ry, rx, rm)
                print(f"[K-Space] walk-off applied: rate_y={ry:.3f} rate_x={rx:.3f} px/mm")
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] KSpace: walk-off correction skipped: {e}")

        if invert:
            datacube = -datacube
            if reference_cube is not None:
                reference_cube = -reference_cube

        sym_flag = hasattr(self, 'chk_asymmetric') and self.chk_asymmetric.isChecked()

        per_pixel = str(center_method).lower().startswith("bary")

        # Helper for baseline & apodization
        def preprocess(cube, force_center=None, c_pos=None):
            if c_pos is None:
                c_pos = positions

            window = max(1, len(c_pos) // 5)
            from scipy.ndimage import uniform_filter1d
            # 'nearest' (not constant/0): keeps the baseline sane at the scan
            # ends instead of inflating the edge samples, which used to drag the
            # burst detector to index 0.
            baseline = uniform_filter1d(cube, size=window, axis=0, mode='nearest')
            sig = cube - baseline

            # ZPD centre: either forced (reference-shared), one field-wide value
            # (envelope), or an independent per-pixel barycentre map.
            if force_center is not None:
                center = force_center
            elif per_pixel:
                center = barycenter_map(sig)                 # (h, w) index map
            else:
                # Collapse to a 1-D interferogram by SIGNED spatial sum (the
                # common-path fringe phase is shared across the field, so signed
                # summing reinforces the burst and cancels DC), then take the
                # analytic-envelope ZPD.
                interf_1d = np.sum(sig, axis=(1, 2))
                center = find_centerburst(interf_1d, c_pos, expected_zero_mm, search_mm)

            scalar = np.ndim(center) == 0

            # Symmetrisation only makes sense for one shared centre; skip it for a
            # per-pixel map (the position axis is common to all pixels).
            if sym_flag and scalar:
                center_idx = int(center)
                left_len = center_idx
                right_len = len(sig) - 1 - center_idx

                if right_len > left_len:
                    tail = sig[center_idx + 1:]
                    sym_signal = np.concatenate([tail[::-1], sig[center_idx:center_idx+1], tail], axis=0)
                    pos_diffs = c_pos[center_idx + 1:] - c_pos[center_idx]
                    mirrored_pos = c_pos[center_idx] - pos_diffs[::-1]
                    sym_positions = np.concatenate([mirrored_pos, [c_pos[center_idx]], c_pos[center_idx + 1:]])
                else:
                    tail = sig[:center_idx]
                    sym_signal = np.concatenate([tail, sig[center_idx:center_idx+1], tail[::-1]], axis=0)
                    pos_diffs = c_pos[center_idx] - c_pos[:center_idx]
                    mirrored_pos = c_pos[center_idx] + pos_diffs[::-1]
                    sym_positions = np.concatenate([c_pos[:center_idx], [c_pos[center_idx]], mirrored_pos])

                sig = sym_signal
                c_pos = sym_positions
                center = center_idx = len(sig) // 2

            cpos_c = c_pos[center]                  # scalar, or (h, w) per-pixel
            try:
                _c = float(cpos_c) if scalar else float(np.median(cpos_c))
                print(f"[K-Space PP] ZPD centre ({center_method}): {_c:.4f} mm"
                      + ("" if scalar else " (per-pixel median)"))
            except Exception:
                pass

            if str(apod_type).lower() == "gaussian":
                sigma = abs(c_pos[-1] - c_pos[0]) * apod_width
                if sigma > 0:
                    if scalar:
                        apod = np.exp(-(c_pos - cpos_c)**2 / (2.0 * sigma**2))
                    else:
                        apod = np.exp(-(c_pos[:, None, None] - cpos_c[None])**2
                                      / (2.0 * sigma**2))
                else:
                    apod = np.ones(len(c_pos))
            elif scalar:
                from instruments.dsp import apodization_window
                apod = apodization_window(apod_type, len(c_pos), center)
            else:
                from instruments.dsp import apodization_window_map
                apod = apodization_window_map(apod_type, len(c_pos), center)

            # FT-window: which part of the interferogram to transform.
            #   center -> broad spectral features (low resolution)
            #   tails  -> high-Q narrow resonances (drop the centerburst)
            # ft_window_mm=(lo,hi) gives an explicit position window (overrides
            # ft_region); keep the apodization taper if the window contains the
            # ZPD, else use a plain boxcar so a tail window isn't suppressed.
            if ft_window_mm is not None:
                lo, hi = sorted(float(v) for v in ft_window_mm)
                inwin = (c_pos >= lo) & (c_pos <= hi)
                if scalar:
                    apod = (apod * inwin) if (lo <= cpos_c <= hi) else inwin.astype(float)
                else:
                    in_c = (cpos_c >= lo) & (cpos_c <= hi)          # (h, w)
                    box = np.broadcast_to(inwin[:, None, None].astype(float), apod.shape)
                    apod = np.where(in_c[None], apod * inwin[:, None, None], box)
            elif ft_region and str(ft_region).lower() != "full":
                delta = (np.abs(c_pos - cpos_c) if scalar
                         else np.abs(c_pos[:, None, None] - cpos_c[None]))
                if str(ft_region).lower() == "center":
                    apod = apod * (delta <= ft_width_mm)
                else:  # "tails": keep the wings, drop the ZPD-centered taper
                    apod = (delta > ft_width_mm).astype(float)
            apod3 = apod[:, np.newaxis, np.newaxis] if apod.ndim == 1 else apod
            return sig * apod3, center, c_pos

        if reference_cube is not None:
            reference_cube = np.asarray(reference_cube, dtype=float)
            ref_signal, fixed_center, ref_pos = preprocess(reference_cube)
            signal, _, final_pos = preprocess(datacube, force_center=fixed_center, c_pos=ref_pos)
        else:
            signal, _, final_pos = preprocess(datacube)

        # Frequency grid
        start_freq, end_freq = self._get_frequency_limits(wl_start, wl_stop)
        # Generate frequencies in descending order so that wavelengths (which are inversely proportional) are ascending
        frequencies = np.linspace(end_freq, start_freq, n_freq)
        wavelengths = self._freq_to_wavelength(frequencies)

        # DFT: for each frequency, sum over positions
        # phase: (n_pos, n_freq)
        pos = final_pos.reshape(-1, 1)
        dpos = np.diff(final_pos)
        dpos = np.append(dpos, dpos[-1] if len(dpos) > 0 else 0)

        phase_kernel = np.exp(-2j * np.pi * pos * frequencies)  # (n_pos, n_freq)

        n_pos_final = len(final_pos)
        # DFT of signal
        weighted = signal * dpos[:, np.newaxis, np.newaxis]  # (n_pos, h, w)
        flat = weighted.reshape(n_pos_final, -1)     # (n_pos, h*w)
        spec_flat = phase_kernel.conj().T @ flat  # (n_freq, h*w)

        # Phase correction
        if reference_cube is not None:
            ref_weighted = ref_signal * dpos[:, np.newaxis, np.newaxis]
            ref_flat = ref_weighted.reshape(n_pos_final, -1)
            ref_spec_flat = phase_kernel.conj().T @ ref_flat

            phase_correction = np.angle(ref_spec_flat)
            phased_spec_flat = spec_flat * np.exp(-1j * phase_correction)
            if complex_out:
                return wavelengths, phased_spec_flat.reshape(n_freq, h, w).astype(np.complex64)
            spectrum_cube = np.real(phased_spec_flat).reshape(n_freq, h, w)
        else:
            if complex_out:
                return wavelengths, spec_flat.reshape(n_freq, h, w).astype(np.complex64)
            spectrum_cube = np.abs(spec_flat).reshape(n_freq, h, w)

        # Magnitude/absorptive spectra don't need float64 -- float32 halves the
        # cube size in RAM and on disk with no meaningful precision loss.
        return wavelengths, spectrum_cube.astype(np.float32)

    def compute_complex_map(self, positions, datacube, wavelength_um,
                            apod_width=0.2, apod_type="gaussian",
                            expected_zero_mm=None, search_mm=None,
                            ft_window_mm=None, positions_calibrated=False,
                            reference_cube=None, center_method="envelope"):
        """Per-pixel COMPLEX DFT at ONE wavelength -> (complex_map (h,w), info).

        The saved spectrum cube keeps only the DFT magnitude (np.abs), throwing
        the interferometric PHASE away. This runs the SAME forward transform as
        compute_hyperspectral -- identical baseline removal, centre-burst (ZPD)
        detection, apodization and FT window -- but for a single frequency and
        WITHOUT taking the magnitude, so both the amplitude and the wrapped phase
        map are available (thermal-hologram view).

        `reference_cube` (a background interferogram of the same shape): the
        per-pixel BACKGROUND phase is subtracted (complex_map *= exp(-i*angle(ref)))
        to remove the interferometer's instrumental phase, leaving the true
        spectral phase -- the same reference phase-correction compute_hyperspectral
        does. The reference defines the ZPD centre (both use it).

        `info` = {center_mm, freq, n_used, phase_corrected}. (None, {}) if too few.
        """
        positions = np.asarray(positions, dtype=float)
        datacube = np.asarray(datacube, dtype=float)
        if datacube.ndim != 3 or datacube.shape[0] < 3:
            return None, {}
        n_pos, h, w = datacube.shape
        ref = None
        if reference_cube is not None:
            ref = np.asarray(reference_cube, dtype=float)
            if ref.shape != datacube.shape:
                ref = None            # incompatible -> ignore, no phase correction

        # Motor-nonlinearity correction (unless the axis is already calibrated) --
        # same rule as compute_hyperspectral, so the phase matches the saved cube.
        if not positions_calibrated:
            try:
                from instruments.calibration import calibrate_position_axis
                positions = np.asarray(calibrate_position_axis(positions), dtype=float)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] KSpace phase: motor calibration skipped: {e}")

        # Baseline removal (moving average) + signed-sum centre-burst. When a
        # reference is given it DEFINES the ZPD centre (both cubes share it).
        from scipy.ndimage import uniform_filter1d
        window = max(1, len(positions) // 5)
        sig = datacube - uniform_filter1d(datacube, size=window, axis=0, mode='nearest')
        ref_sig = (ref - uniform_filter1d(ref, size=window, axis=0, mode='nearest')
                   if ref is not None else None)
        burst_src = ref_sig if ref_sig is not None else sig
        # ZPD centre: one field-wide envelope value, or an independent per-pixel
        # barycentre map (same choice as compute_hyperspectral, kept in step so
        # the phase view matches the recomputed cube).
        if str(center_method).lower().startswith("bary"):
            center = barycenter_map(burst_src)             # (h, w) index map
        else:
            center = find_centerburst(np.sum(burst_src, axis=(1, 2)), positions,
                                      expected_zero_mm, search_mm)
        scalar = np.ndim(center) == 0
        cpos_c = positions[center]                         # scalar or (h, w)

        # Apodization (gaussian in position space, or a named FTIR window).
        if str(apod_type).lower() == "gaussian":
            sigma = abs(positions[-1] - positions[0]) * apod_width
            if sigma <= 0:
                apod = np.ones(len(positions))
            elif scalar:
                apod = np.exp(-(positions - cpos_c) ** 2 / (2.0 * sigma ** 2))
            else:
                apod = np.exp(-(positions[:, None, None] - cpos_c[None]) ** 2
                              / (2.0 * sigma ** 2))
        elif scalar:
            from instruments.dsp import apodization_window
            apod = apodization_window(apod_type, len(positions), center)
        else:
            from instruments.dsp import apodization_window_map
            apod = apodization_window_map(apod_type, len(positions), center)

        # FT window: keep the taper if the window contains the ZPD, else a boxcar
        # (matches compute_hyperspectral's ft_window_mm branch).
        if ft_window_mm is not None:
            lo, hi = sorted(float(v) for v in ft_window_mm)
            inwin = (positions >= lo) & (positions <= hi)
            if scalar:
                apod = (apod * inwin) if (lo <= cpos_c <= hi) else inwin.astype(float)
            else:
                in_c = (cpos_c >= lo) & (cpos_c <= hi)     # (h, w)
                box = np.broadcast_to(inwin[:, None, None].astype(float), apod.shape)
                apod = np.where(in_c[None], apod * inwin[:, None, None], box)

        # Single-frequency DFT for the requested wavelength.
        freq = self._get_frequency_limits(wavelength_um, wavelength_um)[0]
        dpos = np.diff(positions)
        dpos = np.append(dpos, dpos[-1] if len(dpos) > 0 else 0.0)
        wkernel = ((dpos * apod)[:, np.newaxis, np.newaxis] if apod.ndim == 1
                   else dpos[:, np.newaxis, np.newaxis] * apod)
        kernel = np.exp(-2j * np.pi * positions * freq)          # (n_pos,)
        spec_flat = np.conj(kernel) @ (sig * wkernel).reshape(n_pos, -1)  # (h*w,)
        complex_map = spec_flat.reshape(h, w)
        if ref_sig is not None:
            ref_spec = np.conj(kernel) @ (ref_sig * wkernel).reshape(n_pos, -1)
            complex_map = complex_map * np.exp(-1j * np.angle(ref_spec.reshape(h, w)))
        info = {"center_mm": float(cpos_c) if scalar else float(np.median(cpos_c)),
                "freq": float(freq), "per_pixel_center": not scalar,
                "n_used": int(np.count_nonzero(apod.any(axis=(1, 2)) if apod.ndim == 3
                                               else apod)),
                "phase_corrected": ref_sig is not None}
        return complex_map, info
