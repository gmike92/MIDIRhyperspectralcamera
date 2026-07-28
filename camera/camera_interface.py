from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraStatus:
    connected: bool = False
    acquiring: bool = False
    backend: str = "unknown"
    message: str = ""
    width: int = 0
    height: int = 0
    serial_number: str = ""
    requested_mode: str = ""
    startup_profile: str = ""
    selected_device_index: int = -1
    host_visible: bool = False
    job_file_path: str = ""
    average_count: int = 1
    exposure_ms: float = 10.0
    binning: int = 1                     # hardware binning factor (1, 2, 4)
    roi_hsize: int = 0                   # hardware subarray, unbinned sensor px
    roi_vsize: int = 0                   # (0 x 0 = full frame)
    roi_hpos: int = 0
    roi_vpos: int = 0
    sensor_width: int = 0                # full-sensor size (unbinned), for ROI limits
    sensor_height: int = 0
    frame_counter: int = 0
    raw_peak_count: float = 0.0
    board_temp_c: float = float("nan")   # FPGA/electronics board temperature (°C)
    fpa_temp_k: float = float("nan")     # focal-plane array temperature (K)


@dataclass(slots=True)
class MeasurementResult:
    peak: float
    total_power: float
    centroid_x: float
    centroid_y: float
    beam_width_x: float
    beam_width_y: float
    timestamp: float
    extras: dict[str, Any] = field(default_factory=dict)


def copy_camera_status(status: CameraStatus) -> CameraStatus:
    return CameraStatus(**asdict(status))


class CameraInterface(ABC):
    @abstractmethod
    def connect(self) -> CameraStatus:
        """Initialize the backend and return the current camera status."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release all backend resources."""

    @abstractmethod
    def start_acquisition(self) -> None:
        """Start streaming frames."""

    @abstractmethod
    def stop_acquisition(self) -> None:
        """Stop streaming frames."""

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        """Capture and return a single frame, or None if unavailable."""

    @abstractmethod
    def set_exposure(self, exposure_ms: float) -> None:
        """Update the exposure in milliseconds."""

    def set_average(self, average_count: int) -> None:
        """Update the averaging count if supported."""

    @abstractmethod
    def get_status(self) -> CameraStatus:
        """Return the latest backend status."""
