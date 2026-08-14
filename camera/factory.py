from __future__ import annotations

from .camera_interface import CameraInterface
from .mock_camera import MockCamera


def create_camera(mode: str = "auto") -> CameraInterface:
    """Create a camera backend.

    mode: "irc806" (IRCameras MWIR via Pleora eBUS), "ophir" (Ophir/Spiricon
    beam-profiler via BeamGage Automation, e.g. BGP-GIGE-SP1203), "mock"
    (synthetic beam), or "auto" (try IRC806, fall back to mock).
    """
    normalized = (mode or "auto").lower()

    if normalized == "mock":
        return MockCamera()

    # The "Ophir SP1203" is physically an Allied Vision Goldeye G-033 -- a standard
    # GigE Vision camera. Prefer the DIRECT eBUS backend (120 fps, 14-bit, real
    # exposure control) over BeamGage. "beamgage" forces the BeamGage path.
    if normalized in ("ophir", "sp1203", "goldeye", "gige"):
        try:
            from .goldeye_camera import GoldeyeCamera
            return GoldeyeCamera()
        except Exception:  # noqa: BLE001  (eBUS/.NET not available -> mock)
            return MockCamera()

    if normalized in ("beamgage", "spiricon"):
        try:
            from .ophir_camera import OphirBeamgageCamera
            return OphirBeamgageCamera()
        except Exception:  # noqa: BLE001  (BeamGage/.NET not available -> mock)
            return MockCamera()

    if normalized in ("irc806", "ircameras", "pleora", "ir", "auto"):
        try:
            from .irc806_camera import Irc806Camera
            return Irc806Camera()
        except Exception:  # noqa: BLE001  (eBUS/.NET not available -> mock)
            return MockCamera()

    return MockCamera()
