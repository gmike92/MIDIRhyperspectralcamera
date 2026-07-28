"""
Hamamatsu Orca Flash camera wrapper.

Thin wrapper around HamamatsuDevice (CameraDevice.py) that exposes the SAME
set of device functions used by your existing CameraHardware.py /
hyperspectral_measure.py, behind a clean open/close API the rest of the app
shares. Nothing here changes camera behaviour — every call forwards 1:1 to
the matching HamamatsuDevice method.

Device methods forwarded (see CameraDevice.py):
    setExposure / getExposure
    setBinning / getBinning
    setSubarrayH / setSubarrayV / setSubarrayHpos / setSubarrayVpos /
        setSubArrayMode / getSubarray*           (ROI)
    setTriggerSource / setTriggerMode / setTriggerPolarity / setTriggerActive
    getInternalFrameRate / getTemperature / getModelInfo
    startAcquisition / stopAcquisition / stopAcquisitionNotReleasing
    getLastFrame / getFrames
"""

import sys
import os
import time
import contextlib
import numpy as np


@contextlib.contextmanager
def _suppress_stdout():
    """Silence prints from the device layer (e.g. CameraDevice.stopAcquisition's
    'max camera backlog ...' line) without editing the shared CameraDevice.py."""
    old = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [_HERE, os.path.join(_HERE, '..')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def add_path(path):
    """Add a folder under the 'ScopeFoundry Projects' root to sys.path.

    This file is at  <ScopeFoundry Projects>/HyperMeasurementApp/hardware/camera.py
    so the projects root is three levels up; `path` (e.g. 'Hamamatsu_ScopeFoundry')
    is a sibling of HyperMeasurementApp.
    """
    here = os.path.abspath(__file__)
    projects_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    target = os.path.join(projects_root, path)
    if not os.path.isdir(target):
        print(f"[Camera] WARNING: expected device folder not found: {target}")
    if target not in sys.path:
        sys.path.append(target)


class HamamatsuCamera:
    """
    Wrapper around HamamatsuDevice.

    The wrapper keeps a small amount of cached state (exposure, binning, ROI)
    so the UI can show values before the camera is opened; once opened, every
    getter reads back from the hardware.
    """

    def __init__(self,
                 camera_id: int = 0,
                 frame_x: int = 2048,
                 frame_y: int = 2048,
                 exposure: float = 0.01,         # seconds
                 binning: int = 1,
                 trsource: str = 'internal',
                 trmode: str = 'normal',
                 trpolarity: str = 'positive',
                 tractive: str = 'edge',
                 subarrayh_pos: int = 0,
                 subarrayv_pos: int = 0):

        self.camera_id = camera_id
        # Full sensor size — also the default (full-frame) subarray size.
        self.sensor_h = frame_x
        self.sensor_v = frame_y

        # Cached settings (mirrored to hardware once open).
        self.exposure = exposure
        self.binning = binning
        self.trsource = trsource
        self.trmode = trmode
        self.trpolarity = trpolarity
        self.tractive = tractive
        self.subarrayh = frame_x
        self.subarrayv = frame_y
        self.subarrayh_pos = subarrayh_pos
        self.subarrayv_pos = subarrayv_pos

        self.hamamatsu = None
        self._open = False
        self._acquiring = False

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    def open(self) -> bool:
        try:
            add_path('Hamamatsu_ScopeFoundry')
            from CameraDevice import HamamatsuDevice
            self.hamamatsu = HamamatsuDevice(
                camera_id=self.camera_id,
                frame_x=self.subarrayh,
                frame_y=self.subarrayv,
                acquisition_mode='fixed_length',
                number_frames=1,
                exposure=self.exposure,
                trsource=self.trsource,
                trmode=self.trmode,
                trpolarity=self.trpolarity,
                tractive=self.tractive,
                subarrayh_pos=self.subarrayh_pos,
                subarrayv_pos=self.subarrayv_pos,
                binning=self.binning,
                hardware=None,
            )
            self._open = True
            print(f"[Camera] Opened: {self.hamamatsu.getModelInfo()}")
            return True
        except Exception as e:
            print(f"[Camera] open() failed: {e}")
            self._open = False
            return False

    def close(self):
        if self.hamamatsu and self._open:
            try:
                self.hamamatsu.shutdown()
            except Exception as e:
                print(f"[Camera] shutdown error: {e}")
        self._open = False
        self._acquiring = False
        self.hamamatsu = None

    @property
    def is_open(self) -> bool:
        return self._open

    def get_model_info(self) -> str:
        if self._open:
            try:
                return self.hamamatsu.getModelInfo()
            except Exception:
                pass
        return "Hamamatsu Orca Flash 4V3"

    # ------------------------------------------------------------------ #
    #  Exposure
    # ------------------------------------------------------------------ #

    def set_exposure(self, seconds: float):
        """HamamatsuDevice.setExposure (seconds)."""
        self.exposure = seconds
        if self._open:
            self.hamamatsu.setExposure(seconds)
            self.exposure = self.hamamatsu.getExposure()

    def get_exposure(self) -> float:
        if self._open:
            self.exposure = self.hamamatsu.getExposure()
        return self.exposure

    def get_internal_frame_rate(self) -> float:
        """Maximum internal frame rate for the current ROI/exposure."""
        if self._open:
            try:
                return float(self.hamamatsu.getInternalFrameRate())
            except Exception:
                return 0.0
        return 0.0

    def get_temperature(self) -> float:
        if self._open:
            try:
                return float(self.hamamatsu.getTemperature())
            except Exception:
                return float('nan')
        return float('nan')

    # ------------------------------------------------------------------ #
    #  Binning
    # ------------------------------------------------------------------ #

    def set_binning(self, binning: int):
        """HamamatsuDevice.setBinning. Only valid while idle (not acquiring)."""
        self.binning = int(binning)
        if self._open:
            self.hamamatsu.setBinning(int(binning))
            self.binning = self.hamamatsu.getBinning()

    def get_binning(self) -> int:
        if self._open:
            self.binning = self.hamamatsu.getBinning()
        return self.binning

    # ------------------------------------------------------------------ #
    #  ROI (subarray) — order matters, mirrors CameraHardware.py
    # ------------------------------------------------------------------ #

    def set_roi(self, hsize: int, vsize: int, hpos: int = 0, vpos: int = 0):
        """
        Set the camera ROI. Sizes are forced to multiples of 4 by the device.
        Must be called while the camera is idle (not acquiring).
        """
        self.subarrayh, self.subarrayv = int(hsize), int(vsize)
        self.subarrayh_pos, self.subarrayv_pos = int(hpos), int(vpos)
        if not self._open:
            return
        # Sizes first (each resets its position to 0), then enable mode, then positions.
        self.hamamatsu.setSubarrayH(int(hsize))
        self.hamamatsu.setSubarrayV(int(vsize))
        self.hamamatsu.setSubArrayMode()
        self.hamamatsu.setSubarrayHpos(int(hpos))
        self.hamamatsu.setSubarrayVpos(int(vpos))
        self.hamamatsu.setSubArrayMode()
        self._refresh_roi()

    def set_full_frame(self):
        """Reset ROI to the full sensor."""
        self.set_roi(self.sensor_h, self.sensor_v, 0, 0)

    def get_roi(self) -> dict:
        """Returns {'hsize','vsize','hpos','vpos'} in unbinned sensor pixels."""
        if self._open:
            self._refresh_roi()
        return {'hsize': self.subarrayh, 'vsize': self.subarrayv,
                'hpos': self.subarrayh_pos, 'vpos': self.subarrayv_pos}

    def _refresh_roi(self):
        try:
            self.subarrayh = int(self.hamamatsu.getSubarrayH())
            self.subarrayv = int(self.hamamatsu.getSubarrayV())
            self.subarrayh_pos = int(self.hamamatsu.getSubarrayHpos())
            self.subarrayv_pos = int(self.hamamatsu.getSubarrayVpos())
        except Exception:
            pass

    @property
    def eff_h(self) -> int:
        """Effective horizontal frame size after binning."""
        return int(self.subarrayh // self.binning)

    @property
    def eff_v(self) -> int:
        """Effective vertical frame size after binning."""
        return int(self.subarrayv // self.binning)

    # ------------------------------------------------------------------ #
    #  Trigger
    # ------------------------------------------------------------------ #

    def set_trigger_source(self, source: str):
        """source: 'internal' or 'external'."""
        self.trsource = source
        if self._open:
            self.hamamatsu.setTriggerSource(source)

    def get_trigger_source(self) -> str:
        if self._open:
            try:
                return self.hamamatsu.getTriggerSource()
            except Exception:
                pass
        return self.trsource

    # ------------------------------------------------------------------ #
    #  Acquisition control (1:1 passthroughs to HamamatsuDevice)
    # ------------------------------------------------------------------ #

    def set_acquisition(self, mode: str, n_frames: int = 1):
        """mode: 'fixed_length' or 'run_till_abort'."""
        if self._open:
            self.hamamatsu.acquisition_mode = mode
            self.hamamatsu.number_frames = int(n_frames)

    def start_acquisition(self):
        if self._open:
            self.hamamatsu.startAcquisition()
            self._acquiring = True

    def stop_acquisition(self):
        # Idempotent: a no-op if nothing is acquiring, so the worker's safety-net
        # stop won't re-stop / re-release after the scan's own stop.
        if self._open and self._acquiring:
            try:
                with _suppress_stdout():
                    self.hamamatsu.stopAcquisition()
            except Exception as e:
                print(f"[Camera] stopAcquisition error: {e}")
            self._acquiring = False

    def stop_acquisition_not_releasing(self):
        # Stops capture but keeps the buffer; leaves _acquiring True so a later
        # stop_acquisition() still releases the buffer.
        if self._open:
            with _suppress_stdout():
                self.hamamatsu.stopAcquisitionNotReleasing()

    def get_last_frame(self) -> np.ndarray | None:
        """Most-recent frame as a 2-D uint16 array (vertical, horizontal)."""
        if not self._open:
            return None
        [frame, dims] = self.hamamatsu.getLastFrame()
        data = frame.getData()
        return np.reshape(data, (dims[1], dims[0]))

    def get_frames(self) -> tuple[list[np.ndarray], tuple[int, int]]:
        """All buffered frames (used by the external-trigger scan)."""
        if not self._open:
            return [], (0, 0)
        [frames, dims] = self.hamamatsu.getFrames()
        imgs = [np.reshape(f.getData(), (dims[1], dims[0])) for f in frames]
        return imgs, (dims[1], dims[0])

    # ------------------------------------------------------------------ #
    #  Convenience helpers (used by the live view + camera panel)
    # ------------------------------------------------------------------ #

    def snap(self) -> np.ndarray | None:
        """Single frame, fixed_length / internal trigger (mirrors measure())."""
        if not self._open:
            return None
        try:
            self.set_acquisition('fixed_length', 1)
            self.start_acquisition()
            img = self.get_last_frame()
            self.stop_acquisition()
            return img
        except Exception as e:
            print(f"[Camera] snap() error: {e}")
            self.stop_acquisition()
            return None

    def start_live(self):
        """run_till_abort acquisition for live preview."""
        if not self._open:
            return
        self.set_acquisition('run_till_abort', 100)
        self.start_acquisition()

    def get_live_frame(self) -> np.ndarray | None:
        if not self._open:
            return None
        try:
            return self.get_last_frame()
        except Exception as e:
            print(f"[Camera] get_live_frame() error: {e}")
            return None

    def stop_live(self):
        self.stop_acquisition()


# ------------------------------------------------------------------ #
#  Mock camera — for UI development without hardware
# ------------------------------------------------------------------ #

class MockCamera:
    """Animated synthetic Hamamatsu-like frames for offline UI testing.

    Mirrors the HamamatsuCamera API (binning, ROI, acquisition passthroughs)
    so every panel works identically with or without hardware.
    """

    def __init__(self, frame_x=2048, frame_y=2048, binning=1):
        self.sensor_h = frame_x
        self.sensor_v = frame_y
        self.subarrayh = frame_x
        self.subarrayv = frame_y
        self.subarrayh_pos = 0
        self.subarrayv_pos = 0
        self.binning = binning
        self._exposure = 0.01
        self._open = False
        self._acquiring = False
        self._frame_count = 0
        self.trsource = 'internal'

    # --- connection ---
    def open(self) -> bool:
        self._open = True
        print("[MockCamera] Opened (simulation — Hamamatsu Orca Flash 4V3)")
        return True

    def close(self):
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def get_model_info(self) -> str:
        return "Hamamatsu Orca Flash 4V3 (MOCK)"

    # --- exposure / status ---
    def set_exposure(self, seconds: float):
        self._exposure = seconds

    def get_exposure(self) -> float:
        return self._exposure

    def get_internal_frame_rate(self) -> float:
        return 1.0 / max(self._exposure, 1e-4)

    def get_temperature(self) -> float:
        return -10.0

    # --- binning ---
    def set_binning(self, binning: int):
        self.binning = int(binning)

    def get_binning(self) -> int:
        return self.binning

    # --- ROI ---
    def set_roi(self, hsize, vsize, hpos=0, vpos=0):
        self.subarrayh = int(hsize) - int(hsize) % 4
        self.subarrayv = int(vsize) - int(vsize) % 4
        self.subarrayh_pos = int(hpos) - int(hpos) % 4
        self.subarrayv_pos = int(vpos) - int(vpos) % 4

    def set_full_frame(self):
        self.set_roi(self.sensor_h, self.sensor_v, 0, 0)

    def get_roi(self) -> dict:
        return {'hsize': self.subarrayh, 'vsize': self.subarrayv,
                'hpos': self.subarrayh_pos, 'vpos': self.subarrayv_pos}

    @property
    def eff_h(self) -> int:
        return int(self.subarrayh // self.binning)

    @property
    def eff_v(self) -> int:
        return int(self.subarrayv // self.binning)

    # --- trigger ---
    def set_trigger_source(self, source: str):
        self.trsource = source

    def get_trigger_source(self) -> str:
        return self.trsource

    # --- acquisition passthroughs ---
    def set_acquisition(self, mode: str, n_frames: int = 1):
        pass

    def start_acquisition(self):
        self._acquiring = True

    def stop_acquisition(self):
        self._acquiring = False

    def stop_acquisition_not_releasing(self):
        pass

    def get_last_frame(self) -> np.ndarray | None:
        if not self._open:
            return None
        self._frame_count += 1
        return self._make_frame()

    def get_frames(self):
        imgs = [self._make_frame() for _ in range(5)]
        return imgs, (self.eff_v, self.eff_h)

    # --- convenience ---
    def snap(self) -> np.ndarray | None:
        if not self._open:
            return None
        self._frame_count += 1
        return self._make_frame()

    def start_live(self):
        pass

    def get_live_frame(self) -> np.ndarray | None:
        if not self._open:
            return None
        self._frame_count += 1
        return self._make_frame()

    def stop_live(self):
        pass

    def _make_frame(self) -> np.ndarray:
        t = self._frame_count * 0.05
        h, w = self.eff_v, self.eff_h

        # Work on a small grid then upscale for speed.
        scale = max(1, min(h, w) // 64)
        sh, sw = max(1, h // scale), max(1, w // scale)
        frame = np.random.normal(200, 20, (sh, sw)).astype(np.float32)

        y, x = np.ogrid[:sh, :sw]
        for amp, sigma, cx_f, cy_f, fx, fy in [
            (50000, 0.10, 0.5, 0.5, 0.5, 0.7),
            (20000, 0.06, 0.3, 0.35, 0.4, 0.3),
        ]:
            cx = sw * (cx_f + 0.1 * np.cos(t * fx))
            cy = sh * (cy_f + 0.1 * np.sin(t * fy))
            sig = sigma * min(sh, sw)
            frame += amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sig**2))

        frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
        frame = frame[:h, :w]
        if frame.shape != (h, w):  # pad if rounding left it short
            out = np.zeros((h, w), dtype=np.float32)
            out[:frame.shape[0], :frame.shape[1]] = frame
            frame = out
        return np.clip(frame, 0, 65535).astype(np.uint16)
