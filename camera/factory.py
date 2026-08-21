from __future__ import annotations

from .camera_interface import CameraInterface
from .mock_camera import MockCamera


def create_camera(mode: str = "auto") -> CameraInterface:
    """Create a camera backend.

    mode: "orca" (Hamamatsu Orca Flash via DCAM), "irc806" (IRCameras IRC806
    via Pleora eBUS), "mock" (synthetic beam), or "auto" (try the Orca, fall
    back to mock if it can't load/connect).
    """
    normalized = (mode or "auto").lower()

    if normalized == "mock":
        return MockCamera()

    if normalized in ("irc806", "irc", "ircameras"):
        try:
            from .irc806_camera import Irc806Camera
            return Irc806Camera()
        except Exception:  # noqa: BLE001  (eBUS/pythonnet not available -> mock)
            return MockCamera()

    if normalized in ("orca", "hamamatsu", "dcam", "auto"):
        try:
            from .orca_camera import OrcaCamera
            return OrcaCamera()
        except Exception:  # noqa: BLE001  (DCAM not available -> mock)
            return MockCamera()

    return MockCamera()
