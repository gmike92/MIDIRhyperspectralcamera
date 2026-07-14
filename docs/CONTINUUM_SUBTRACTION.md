# Per-pixel continuum subtraction — isolating the q-BIC line image

## Why

Each hyperspectral cube `spectrum_cube[wavelength, y, x]` contains, at every
pixel, the emission spectrum. Two things live in it:

1. a **broadband thermal (Planck) continuum** — the hot metasurface/substrate
   glowing across the whole 3.8–4.4 µm window. It is spatially **smooth** (a
   big disk that fills the ROI) and carries no vortex/spiral structure.
2. the **narrow q-BIC resonant line** at ~4.00 µm (Q ≈ 40, FWHM ≈ 0.1 µm) —
   this is the structured emission (focus, spiral, s3 texture) we care about.

If you just take a narrowband slice at 4.0 µm you still get (1)+(2), and the
broad continuum **swamps** the resonant structure (it dominates the total
counts and makes every intensity metric measure the thermal disk, not the
q‑BIC emission).

**Fix:** at each pixel, estimate the continuum *underneath* the line from two
"shoulder" bands just outside it, subtract that baseline, and keep only the
residual line emission. Done per pixel, this removes the smooth thermal disk
and leaves the resonant q‑BIC image. (Validation: on the bare-substrate control
— which has no resonance — the result is featureless noise, i.e. ≈ 0.)

## Method (per pixel)

Define three wavelength bands:

| band | range (µm) | role |
|------|-----------|------|
| line  | **3.965 – 4.050** | the q-BIC emission line to integrate |
| left shoulder  (L) | **3.900 – 3.955** | continuum sample below the line |
| right shoulder (R) | **4.060 – 4.120** | continuum sample above the line |

For each pixel:

1. `b_L` = mean of the spectrum over the **left** shoulder, at mean wavelength `λ_L`.
2. `b_R` = mean over the **right** shoulder, at mean wavelength `λ_R`.
3. Fit a straight line (the local continuum) through the two points
   `(λ_L, b_L)` and `(λ_R, b_R)`:  `slope = (b_R − b_L) / (λ_R − λ_L)`.
4. For every wavelength `λ` inside the **line** band, subtract the baseline
   `b_L + slope·(λ − λ_L)` and **average** the residual over the line band.

That average residual is the continuum-subtracted line intensity at that pixel.
Repeat for all pixels → the q-BIC line image.

```
spectrum
  |            .-''-.   <- q-BIC line (integrated over 'line' band)
  |      _____/      \_____
  |  b_L●----+--------+----●  b_R   <- linear continuum baseline (subtracted)
  |     [ L ]  [line]  [ R ]
  +----------------------------------> wavelength
     3.90  3.955 3.965 4.05 4.06 4.12
```

## Notes / how to choose the bands

- **Units:** `wavelengths` here are in **micrometres**. Match your band limits
  to whatever units your `wavelengths` array uses.
- **Line band** should span the resonance (center ± ~1 FWHM). Ours: 4.01 µm
  center, ~0.1 µm FWHM → 3.965–4.05 µm.
- **Shoulders** should sit **just outside** the line, on clean continuum:
  wide enough to average down noise, but avoiding (a) the line wings and
  (b) other spectral features. Here the right shoulder stops at 4.12 µm to stay
  clear of the atmospheric CO₂ dip near 4.2–4.25 µm and the steeply rising
  thermal tail beyond ~4.15 µm.
- The cube must already be **dark/background corrected** (ours has
  `background_subtracted = True`); this step removes the *spectral* continuum,
  not the detector offset.
- A linear baseline is enough because the continuum is smooth over this narrow
  window. If your continuum is more curved, use more/wider shoulder bands and
  fit a low-order polynomial instead of a line.

## Code (NumPy only — drop-in, no other dependencies)

```python
import numpy as np

def continuum_subtract(cube, wl,
                       line=(3.965, 4.050),
                       lsh =(3.900, 3.955),
                       rsh =(4.060, 4.120)):
    """
    Per-pixel continuum-subtracted line image.

    cube : ndarray (n_wavelength, Ny, Nx)  -- the spectral cube
    wl   : ndarray (n_wavelength,)         -- wavelengths, SAME units as the bands
    line/lsh/rsh : (lo, hi) wavelength bands for the line and the two shoulders

    returns : ndarray (Ny, Nx)  -- resonant line emission, continuum removed
    """
    cube = np.asarray(cube, dtype=float)
    wl   = np.asarray(wl,   dtype=float)

    lm = (wl >= line[0]) & (wl <= line[1])          # line band
    l  = (wl >= lsh[0])  & (wl <= lsh[1])           # left shoulder
    r  = (wl >= rsh[0])  & (wl <= rsh[1])           # right shoulder

    wl_l, wl_r = wl[l].mean(), wl[r].mean()
    b_L = cube[l].mean(axis=0)                       # (Ny,Nx) continuum @ wl_l
    b_R = cube[r].mean(axis=0)                       # (Ny,Nx) continuum @ wl_r
    slope = (b_R - b_L) / (wl_r - wl_l)              # (Ny,Nx) baseline slope

    wll = wl[lm]                                     # in-line wavelengths
    baseline = b_L[None] + slope[None] * (wll[:, None, None] - wl_l)  # (nline,Ny,Nx)
    return (cube[lm] - baseline).mean(axis=0)        # (Ny,Nx)
```

### Example (loading one of the .npz files)

```python
import numpy as np
d    = np.load('..._z17.5000mm.npz', allow_pickle=True)
cube = d['spectrum_cube']          # (n_wl, Ny, Nx), already background-subtracted
wl   = d['wavelengths']            # (n_wl,) in micrometres
img  = continuum_subtract(cube, wl)   # -> continuum-free q-BIC line image
```

## Quick sanity checks

- Run it on the **bare-substrate** cube → the output should be flat noise
  (≈ 0), confirming the resonant structure comes from the metasurface, not the
  thermal background.
- The output can be **slightly negative** in places (noise around a zero
  baseline) — that's expected; do not clip to zero before differencing/averaging
  or you bias the result. Clip only for display.
- To locate the line in a new dataset, plot the ROI-mean spectrum of the sample
  and of a no-resonance control; the line is where their **ratio** peaks (ours
  peaks at 4.00 µm). Put the shoulders on either side of that peak.
```
