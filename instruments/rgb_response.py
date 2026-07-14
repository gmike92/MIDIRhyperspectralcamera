"""rgb_response.py -- R/G/B spectral response curves for false-colour RGB.

The reference hsiAnalysis HyperspectralAnalysis_Spectrum builds RGB by projecting
each pixel spectrum onto smooth, overlapping R/G/B response curves (its
RGB_Transmission.mat, divided by the same normalisation constants). These are
those curves (wavelength in nm). Embedded so no external data file is needed.
"""
import numpy as np

WL_NM = np.array([300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100])
R = np.array([1.70768e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.41536e-08, 3.32997e-06, 2.26097e-05, 5.98029e-05, 9.52373e-05, 0.000131099, 0.000152923, 0.000163391, 0.000166294, 0.000164518, 0.000166174, 0.00016556, 0.000160317, 0.000101129, 3.5127e-05, 8.36763e-06, 3.07382e-06, 1.36614e-06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
G = np.array([2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 2.17004e-08, 1.08502e-06, 4.6873e-06, 2.10494e-05, 6.46673e-05, 9.98221e-05, 0.000144308, 0.000171998, 0.000208107, 0.000206632, 0.000210277, 0.000205937, 0.000184237, 0.000128467, 8.96228e-05, 5.58135e-05, 2.25685e-05, 8.02917e-06, 3.03806e-06, 1.51903e-06, 6.51013e-07, 6.51013e-07, 4.34009e-07, 4.34009e-07, 4.34009e-07, 4.34009e-07, 2.17004e-08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
B = np.array([2.12031e-08, 2.12031e-08, 2.12031e-08, 2.12031e-08, 2.12031e-08, 2.12031e-08, 2.12031e-08, 2.12031e-08, 2.12031e-08, 5.93686e-06, 0.000178742, 0.000193584, 0.000195492, 0.000195492, 0.000199097, 0.000200793, 0.000200793, 0.000201005, 0.000196128, 0.000185103, 0.000129763, 8.01476e-05, 3.54091e-05, 6.99701e-06, 1.48421e-06, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 2.12031e-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Core visible window of the curves (B/G/R peaks span ~450-600 nm). For data
# outside the curves (e.g. MWIR) this window is mapped onto the data band.
_MAP_LO, _MAP_HI = 380.0, 700.0


def response_on_axis(wl_data, map_to_range=True):
    """(R, G, B) response vectors sampled on the data wavelength axis.

    map_to_range / non-overlapping data -> the visible window [380,700] nm is
    linearly mapped onto the data range (B->short end, R->long end) so the
    projection produces colour across the whole band. Otherwise the curves are
    used at their literal nm positions (faithful for visible/NIR data).
    """
    wl = np.asarray(wl_data, float)
    lo, hi = float(np.min(wl)), float(np.max(wl))
    overlap = (hi >= WL_NM.min()) and (lo <= WL_NM.max())
    if map_to_range or not overlap:
        u = (wl - lo) / (hi - lo + 1e-12)
        x = _MAP_LO + u * (_MAP_HI - _MAP_LO)
    else:
        x = wl
    return (np.interp(x, WL_NM, R), np.interp(x, WL_NM, G), np.interp(x, WL_NM, B))
