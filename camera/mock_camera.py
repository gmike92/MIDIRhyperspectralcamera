from __future__ import annotations

import math
import numpy as np

from .camera_interface import CameraInterface, CameraStatus, copy_camera_status


class MockCamera(CameraInterface):
    """Software camera that mimics a drifting IR beam spot."""

    def __init__(self, width: int = 320, height: int = 256) -> None:
        self.width = width
        self.height = height
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
            serial_number="MOCK-WINCAMD-IR-BB",
            requested_mode="mock",
            startup_profile="mock",
            selected_device_index=0,
            host_visible=False,
            job_file_path="",
            average_count=1,
            exposure_ms=10.0,
        )

        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

    def connect(self) -> CameraStatus:
        self.connected = True
        self.status.connected = True
        self.status.message = "Mock camera connected"
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
        self.exposure_ms = float(np.clip(exposure_ms, 0.1, 250.0))
        self.status.exposure_ms = self.exposure_ms
        self.status.message = f"Mock exposure set to {self.exposure_ms:.2f} ms"

    def set_average(self, average_count: int) -> None:
        self.average_count = max(1, int(average_count))
        self.status.average_count = self.average_count
        self.status.message = f"Mock averaging set to {self.average_count}"

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
        noise = np.random.normal(20.0, 8.0, size=(self.height, self.width))

        frame = beam + secondary + ripple + noise
        return np.clip(frame, 0, 4095).astype(np.uint16)
