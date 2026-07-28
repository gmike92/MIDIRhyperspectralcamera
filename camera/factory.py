from __future__ import annotations

from .camera_interface import CameraInterface
from .mock_camera import MockCamera


def create_camera(mode: str = "auto") -> CameraInterface:
    """Create a camera backend.

    mode: "orca" (Hamamatsu Orca Flash via DCAM), "mock" (synthetic beam),
    or "auto" (try the Orca, fall back to mock if it can't load/connect).
    """
    normalized = (mode or "auto").lower()

    if normalized == "mock":
        return MockCamera()

    if normalized in ("orca", "hamamatsu", "dcam", "auto"):
        try:
            from .orca_camera import OrcaCamera
            return OrcaCamera()
        except Exception:  # noqa: BLE001  (DCAM not available -> mock)
            return MockCamera()

    return MockCamera()
