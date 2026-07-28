from __future__ import annotations

import math
import numpy as np

from .camera_interface import CameraInterface, CameraStatus, copy_camera_status


class MockCamera(CameraInterface):
    """Software camera that mimics a drifting beam spot, with hardware-style
    binning (1/2/4) and a subarray ROI so those controls work offline."""

    def __init__(self, width: int = 320, height: int = 256) -> None:
        # Full sensor (unbinned) geometry.
        self.sensor_w = width
        self.sensor_h = height
        # Active subarray (unbinned sensor px) and binning.
        self.roi_hsize = width
        self.roi_vsize = height
        self.roi_hpos = 0
        self.roi_vpos = 0
        self.binning = 1

        self.exposure_ms = 10.0
        self.connected = False
        self.acquiring = False
        self.frame_index = 0
        self.average_count = 1
        self.status = CameraStatus(
            connected=False,
            acquiring=False,
            backend="mock",
            message="Mock camera idle",
            width=width,
            height=height,
            serial_number="MOCK-ORCA-SIM",
            requested_mode="mock",
            startup_profile="mock",
            selected_device_index=0,
            host_visible=False,
            job_file_path="",
            average_count=1,
            exposure_ms=10.0,
            binning=1,
            roi_hsize=width,
            roi_vsize=height,
            sensor_width=width,
            sensor_height=height,
        )

        # Beam grid spans the FULL sensor; the ROI crops into it.
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

    def connect(self) -> CameraStatus:
        self.connected = True
        self.status.connected = True
        self.status.message = "Mock camera connected"
        self._sync_geometry()
        return self.get_status()

    def disconnect(self) -> None:
        self.connected = False
        self.acquiring = False
        self.status.connected = False
        self.status.acquiring = False
        self.status.message = "Mock camera disconnected"

    def start_acquisition(self) -> None:
        self.acquiring = True
        self.status.acquiring = True
        self.status.message = "Mock acquisition running"

    def stop_acquisition(self) -> None:
        self.acquiring = False
        self.status.acquiring = False
        self.status.message = "Mock acquisition stopped"

    def set_exposure(self, exposure_ms: float) -> None:
        self.exposure_ms = float(np.clip(exposure_ms, 0.01, 1000.0))
        self.status.exposure_ms = self.exposure_ms
        self.status.message = f"Mock exposure set to {self.exposure_ms:.2f} ms"

    def set_average(self, average_count: int) -> None:
        self.average_count = max(1, int(average_count))
        self.status.average_count = self.average_count
        self.status.message = f"Mock averaging set to {self.average_count}"

    # -- hardware binning / ROI ---------------------------------------------
    def set_binning(self, binning: int) -> None:
        self.binning = int(binning) if int(binning) in (1, 2, 4) else 1
        self._sync_geometry()
        self.status.message = f"Mock binning set to {self.binning}"

    def get_binning(self) -> int:
        return self.binning

    def set_roi(self, hsize: int, vsize: int, hpos: int = 0, vpos: int = 0) -> None:
        # Snap to multiples of 4 and clamp inside the sensor (mirrors DCAM rules).
        def snap(v):
            return max(0, int(v) - int(v) % 4)
        hsize, vsize, hpos, vpos = map(snap, (hsize, vsize, hpos, vpos))
        hsize = max(4, min(hsize or self.sensor_w, self.sensor_w))
        vsize = max(4, min(vsize or self.sensor_h, self.sensor_h))
        hpos = min(hpos, self.sensor_w - hsize)
        vpos = min(vpos, self.sensor_h - vsize)
        self.roi_hsize, self.roi_vsize = hsize, vsize
        self.roi_hpos, self.roi_vpos = hpos, vpos
        self._sync_geometry()
        self.status.message = f"Mock ROI {hsize}x{vsize} @ ({hpos},{vpos})"

    def set_full_frame(self) -> None:
        self.set_roi(self.sensor_w, self.sensor_h, 0, 0)

    def get_roi(self) -> dict:
        return {"hsize": self.roi_hsize, "vsize": self.roi_vsize,
                "hpos": self.roi_hpos, "vpos": self.roi_vpos}

    def _sync_geometry(self) -> None:
        b = self.binning
        self.status.binning = b
        self.status.roi_hsize = self.roi_hsize
        self.status.roi_vsize = self.roi_vsize
        self.status.roi_hpos = self.roi_hpos
        self.status.roi_vpos = self.roi_vpos
        self.status.sensor_width = self.sensor_w
        self.status.sensor_height = self.sensor_h
        self.status.width = self.roi_hsize // b
        self.status.height = self.roi_vsize // b

    def get_status(self) -> CameraStatus:
        return copy_camera_status(self.status)

    def get_frame(self) -> np.ndarray | None:
        if not (self.connected and self.acquiring):
            return None

        self.frame_index += 1
        t = self.frame_index / 18.0

        cx = 0.35 * math.sin(t * 0.7)
        cy = 0.28 * math.cos(t * 0.5)
        sx = 0.15 + 0.02 * math.sin(t * 0.9)
        sy = 0.10 + 0.03 * math.cos(t * 0.6)
        amplitude = 1200.0 * min(self.exposure_ms / 10.0, 8.0)

        beam = amplitude * np.exp(
            -(
                ((self.grid_x - cx) ** 2) / (2.0 * sx**2)
                + ((self.grid_y - cy) ** 2) / (2.0 * sy**2)
            )
        )
        secondary = 160.0 * np.exp(
            -(
                ((self.grid_x + 0.42) ** 2) / (2.0 * 0.08**2)
                + ((self.grid_y - 0.35) ** 2) / (2.0 * 0.05**2)
            )
        )
        ripple = 35.0 * (np.sin(12 * self.grid_x + t) + np.cos(10 * self.grid_y - t))
        noise = np.random.normal(20.0, 8.0, size=(self.sensor_h, self.sensor_w))
        frame = beam + secondary + ripple + noise

        # Crop to the active subarray, then apply hardware-style binning.
        r0, c0 = self.roi_vpos, self.roi_hpos
        frame = frame[r0:r0 + self.roi_vsize, c0:c0 + self.roi_hsize]
        frame = self._bin(frame, self.binning)
        return np.clip(frame, 0, 65535).astype(np.uint16)

    @staticmethod
    def _bin(frame: np.ndarray, b: int) -> np.ndarray:
        if b <= 1:
            return frame
        h, w = frame.shape
        h2, w2 = h // b, w // b
        if h2 == 0 or w2 == 0:
            return frame
        return frame[:h2 * b, :w2 * b].reshape(h2, b, w2, b).sum(axis=(1, 3))
