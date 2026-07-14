from __future__ import annotations

from .camera_interface import CameraInterface
from .mock_camera import MockCamera


def create_camera(mode: str = "auto") -> CameraInterface:
    """Create a camera backend.

    mode: "irc806" (IRCameras MWIR via Pleora eBUS), "mock" (synthetic beam),
    or "auto" (try IRC806, fall back to mock if it can't load/connect).
    """
    normalized = (mode or "auto").lower()

    if normalized == "mock":
        return MockCamera()

    if normalized in ("irc806", "ircameras", "pleora", "ir", "auto"):
        try:
            from .irc806_camera import Irc806Camera
            return Irc806Camera()
        except Exception:  # noqa: BLE001  (eBUS/.NET not available -> mock)
            return MockCamera()

    return MockCamera()
