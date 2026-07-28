"""
orca_camera.py -- Hamamatsu Orca Flash (DCAM) backend for the hyperspectral app.

Implements the app's CameraInterface on top of the proven HamamatsuCamera DCAM
wrapper (camera/hamamatsu_backend.py), so the entire acquisition/DSP/viewer
stack works unchanged with the Orca in place of the IRC806.

Interface contract (see camera/camera_interface.py):
    connect() -> CameraStatus
    disconnect() -> None
    start_acquisition() / stop_acquisition() -> None
    get_frame() -> np.ndarray | None        # 2-D uint16 (height, width)
    set_exposure(exposure_ms: float) -> None
    set_average(int) -> None                 # optional, host-side averaging
    get_status() -> CameraStatus
    self.status : CameraStatus               # live attribute the worker reads

Optional hooks the camera worker calls via hasattr():
    refresh_temperatures()  -> populate status.board_temp_c (sensor °C)
    reconnect()             -> re-open the DCAM link after a drop

The Orca is 16-bit, so frames are uint16 in [0, 65535] (vs the IRC806's 14-bit).

If DCAM / the Hamamatsu device layer is not available on this PC, connect()
returns a status with connected=False and a helpful message; use --mode mock for
offline development.
"""
from __future__ import annotations

import time
import numpy as np

from .camera_interface import CameraInterface, CameraStatus, copy_camera_status

# Host-side averaging cap (frames combined per get_frame when average > 1).
_MAX_AVERAGE = 256


class OrcaCamera(CameraInterface):
    """Hamamatsu Orca Flash camera exposed through the app's CameraInterface."""

    def __init__(self,
                 camera_id: int = 0,
                 exposure_ms: float = 10.0,
                 binning: int = 1) -> None:
        self._cam = None                 # HamamatsuCamera (lazy: needs DCAM)
        self._camera_id = camera_id
        self._binning = int(binning)
        self.exposure_ms = float(exposure_ms)
        self.average_count = 1
        self.connected = False
        self.acquiring = False
        self.frame_index = 0

        self.status = CameraStatus(
            connected=False,
            acquiring=False,
            backend="orca",
            message="Orca camera idle",
            width=0,
            height=0,
            serial_number="",
            requested_mode="orca",
            startup_profile="orca",
            selected_device_index=camera_id,
            average_count=1,
            exposure_ms=self.exposure_ms,
            binning=self._binning,
        )

    # -- connection ----------------------------------------------------------
    def connect(self) -> CameraStatus:
        try:
            from .hamamatsu_backend import HamamatsuCamera
        except Exception as exc:  # noqa: BLE001
            self._fail(f"Hamamatsu backend import failed: {exc}")
            return self.get_status()

        try:
            self._cam = HamamatsuCamera(
                camera_id=self._camera_id,
                exposure=self.exposure_ms / 1000.0,   # DCAM wrapper wants seconds
                binning=self._binning,
            )
            ok = self._cam.open()
        except Exception as exc:  # noqa: BLE001
            self._fail(f"Orca open() raised: {exc}")
            return self.get_status()

        if not ok:
            self._fail("Orca open() failed -- check DCAM driver / camera power")
            return self.get_status()

        self.connected = True
        self.status.connected = True
        self.status.serial_number = self._safe(self._cam.get_model_info, "Hamamatsu Orca")
        self._refresh_geometry()
        self.status.message = f"Orca connected -- {self.status.serial_number}"
        return self.get_status()

    def disconnect(self) -> None:
        self.acquiring = False
        self.status.acquiring = False
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:  # noqa: BLE001
                pass
        self._cam = None
        self.connected = False
        self.status.connected = False
        self.status.message = "Orca disconnected"

    def reconnect(self) -> bool:
        """Re-open the DCAM link after a drop (called by the worker's auto-heal)."""
        try:
            self.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self.connect()
        if self.connected:
            self.start_acquisition()
        return self.connected

    # -- acquisition ---------------------------------------------------------
    def start_acquisition(self) -> None:
        if not (self.connected and self._cam is not None):
            return
        try:
            self._cam.start_live()          # run_till_abort ring buffer
            self.acquiring = True
            self.status.acquiring = True
            self.status.message = "Orca acquisition running"
        except Exception as exc:  # noqa: BLE001
            self.status.message = f"Orca start_acquisition failed: {exc}"

    def stop_acquisition(self) -> None:
        self.acquiring = False
        self.status.acquiring = False
        if self._cam is not None:
            try:
                self._cam.stop_live()
            except Exception:  # noqa: BLE001
                pass
        self.status.message = "Orca acquisition stopped"

    def get_frame(self) -> np.ndarray | None:
        if not (self.connected and self.acquiring and self._cam is not None):
            return None
        try:
            frame = self._cam.get_live_frame()
        except Exception:  # noqa: BLE001
            return None
        if frame is None:
            return None

        if self.average_count > 1:
            frame = self._averaged(frame)

        self.frame_index += 1
        self.status.frame_counter = self.frame_index
        self.status.raw_peak_count = float(frame.max()) if frame.size else 0.0
        # Orca is 16-bit; ensure uint16 as the whole pipeline expects.
        if frame.dtype != np.uint16:
            frame = np.clip(frame, 0, 65535).astype(np.uint16)
        return frame

    def _averaged(self, first: np.ndarray) -> np.ndarray:
        """Host-side averaging: combine N distinct frames (identity-checked)."""
        acc = first.astype(np.float64)
        last = first
        got = 1
        deadline = time.time() + 2.0
        while got < self.average_count and time.time() < deadline:
            f = self._cam.get_live_frame()
            if f is None or f is last:
                continue
            acc += f
            last = f
            got += 1
        return (acc / got)

    # -- settings ------------------------------------------------------------
    def set_exposure(self, exposure_ms: float) -> None:
        self.exposure_ms = max(float(exposure_ms), 1e-3)
        self.status.exposure_ms = self.exposure_ms
        if self._cam is not None:
            try:
                self._cam.set_exposure(self.exposure_ms / 1000.0)
                self.exposure_ms = self._cam.get_exposure() * 1000.0
                self.status.exposure_ms = self.exposure_ms
            except Exception as exc:  # noqa: BLE001
                self.status.message = f"set_exposure failed: {exc}"

    def set_average(self, average_count: int) -> None:
        self.average_count = int(np.clip(average_count, 1, _MAX_AVERAGE))
        self.status.average_count = self.average_count
        self.status.message = f"Orca averaging set to {self.average_count}"

    # -- hardware binning / ROI ---------------------------------------------
    # DCAM only accepts these while idle; the worker stops acquisition around
    # the change and restarts it, so these just forward to the DCAM wrapper.
    def set_binning(self, binning: int) -> None:
        self._binning = int(binning)
        if self._cam is not None:
            try:
                self._cam.set_binning(int(binning))
            except Exception as exc:  # noqa: BLE001
                self.status.message = f"set_binning failed: {exc}"
        self._refresh_geometry()
        self.status.message = f"Orca binning set to {self.status.binning}"

    def get_binning(self) -> int:
        return self.status.binning

    def set_roi(self, hsize: int, vsize: int, hpos: int = 0, vpos: int = 0) -> None:
        if self._cam is not None:
            try:
                self._cam.set_roi(int(hsize), int(vsize), int(hpos), int(vpos))
            except Exception as exc:  # noqa: BLE001
                self.status.message = f"set_roi failed: {exc}"
        self._refresh_geometry()
        self.status.message = (f"Orca ROI {self.status.roi_hsize}x{self.status.roi_vsize} "
                               f"@ ({self.status.roi_hpos},{self.status.roi_vpos})")

    def set_full_frame(self) -> None:
        if self._cam is not None:
            try:
                self._cam.set_full_frame()
            except Exception as exc:  # noqa: BLE001
                self.status.message = f"set_full_frame failed: {exc}"
        self._refresh_geometry()
        self.status.message = "Orca ROI reset to full frame"

    def get_roi(self) -> dict:
        return {"hsize": self.status.roi_hsize, "vsize": self.status.roi_vsize,
                "hpos": self.status.roi_hpos, "vpos": self.status.roi_vpos}

    def refresh_temperatures(self) -> None:
        """Populate sensor temperature (°C) into board_temp_c if DCAM reports it."""
        if self._cam is None:
            return
        try:
            t = float(self._cam.get_temperature())
            if np.isfinite(t):
                self.status.board_temp_c = t
        except Exception:  # noqa: BLE001
            pass

    def get_status(self) -> CameraStatus:
        return copy_camera_status(self.status)

    # -- helpers -------------------------------------------------------------
    def _refresh_geometry(self) -> None:
        cam = self._cam
        if cam is None:
            return
        try:
            self.status.binning = int(cam.get_binning())
        except Exception:  # noqa: BLE001
            self.status.binning = self._binning
        try:
            roi = cam.get_roi()
            self.status.roi_hsize = int(roi["hsize"])
            self.status.roi_vsize = int(roi["vsize"])
            self.status.roi_hpos = int(roi["hpos"])
            self.status.roi_vpos = int(roi["vpos"])
        except Exception:  # noqa: BLE001
            pass
        try:
            self.status.sensor_width = int(cam.sensor_h)    # wrapper: sensor_h = horizontal
            self.status.sensor_height = int(cam.sensor_v)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.status.width = int(cam.eff_h)              # binned frame size
            self.status.height = int(cam.eff_v)
        except Exception:  # noqa: BLE001
            self.status.width = int(getattr(cam, "subarrayh", 0))
            self.status.height = int(getattr(cam, "subarrayv", 0))

    def _fail(self, message: str) -> None:
        self._cam = None
        self.connected = False
        self.status.connected = False
        self.status.acquiring = False
        self.status.message = message

    @staticmethod
    def _safe(fn, default):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            return default


if __name__ == "__main__":
    cam = OrcaCamera()
    st = cam.connect()
    print("connected:", st.connected, "|", st.message)
    if st.connected:
        cam.start_acquisition()
        time.sleep(0.2)
        f = cam.get_frame()
        print("frame:", None if f is None else (f.shape, f.dtype))
        cam.stop_acquisition()
        cam.disconnect()
