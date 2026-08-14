"""Ophir/Spiricon BeamGage camera backend (e.g. BGP-GIGE-SP1203).

There is no standalone GigE-Vision path to these cameras on this PC -- they are
served by **BeamGage Professional**. We drive BeamGage's .NET Automation API
(`Spiricon.Automation.AutomatedBeamGage`, in Spiricon.BeamGage.Automation.dll)
via pythonnet, exactly the pattern in Spiricon's own C# example
(BeamGage Professional\\Automation\\Examples\\C#): construct an AutomatedBeamGage
client, Start() the data source, and read `ResultsPriorityFrame.DoubleData`
(+ Width/Height) each new frame.

Notes / gotchas:
  * Constructing AutomatedBeamGage LAUNCHES a BeamGage engine that OWNS the
    camera. Any separately-open BeamGage GUI must be CLOSED first, or the two
    fight over the SP1203.
  * BeamGage Professional 64-bit -> its Automation DLL is 64-bit -> loads fine in
    the app's 64-bit venv (no 32-bit bridge needed).
  * Exposure / gain / Ultracal are driven from the BeamGage window (SHOW_GUI); we
    only pull frames here. set_exposure is a logged no-op.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from loguru import logger

from .camera_interface import CameraInterface, CameraStatus, copy_camera_status

BEAMGAGE_DIR = r"C:\Program Files\Spiricon\BeamGage Professional"
# Show the BeamGage window: it becomes the camera control panel (exposure, gain,
# Ultracal, data-source selection) while our app does the TWINS scan. Set False
# for a headless engine once the workflow is dialed in.
SHOW_GUI = True


class OphirBeamgageCamera(CameraInterface):
    def __init__(self) -> None:
        self.status = CameraStatus(backend="ophir", exposure_ms=10.0)
        self._bg = None                 # AutomatedBeamGage instance
        self._frame = None              # IAFrame (ResultsPriorityFrame)
        self._AutomatedBeamGage = None
        self._loaded = False
        self._last_id = -1

    # -- loading -------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import clr  # pythonnet
        if BEAMGAGE_DIR not in sys.path:
            sys.path.append(BEAMGAGE_DIR)
        try:
            os.add_dll_directory(BEAMGAGE_DIR)   # resolve dependent Spiricon DLLs
        except Exception:  # noqa: BLE001
            pass
        clr.AddReference(os.path.join(BEAMGAGE_DIR, "Spiricon.Automation.dll"))
        clr.AddReference(os.path.join(BEAMGAGE_DIR, "Spiricon.BeamGage.Automation.dll"))
        from Spiricon.Automation import AutomatedBeamGage
        self._AutomatedBeamGage = AutomatedBeamGage
        self._loaded = True

    # -- CameraInterface -----------------------------------------------------
    def connect(self) -> CameraStatus:
        try:
            self._ensure_loaded()
        except Exception as e:  # noqa: BLE001
            self.status.connected = False
            self.status.message = f"BeamGage Automation load failed: {e}"
            logger.error(self.status.message)
            return self.get_status()

        try:
            # Launch our own BeamGage automation client (it owns the camera).
            self._bg = self._AutomatedBeamGage("MIR_CAMERA", SHOW_GUI)
        except Exception as e:  # noqa: BLE001
            self._bg = None
            self.status.connected = False
            self.status.message = (f"BeamGage launch failed: {e}. "
                                   "Close any open BeamGage window first.")
            logger.error(self.status.message)
            return self.get_status()

        # The current-frame handle (example uses ResultsPriorityFrame).
        self._frame = (getattr(self._bg, "ResultsPriorityFrame", None)
                       or getattr(self._bg, "FramePriorityFrame", None))
        self.status.connected = True
        try:
            self.status.serial_number = str(self._bg.DataSource.DataSource or "")
        except Exception:  # noqa: BLE001
            pass
        w, h = self._dims()
        self.status.width, self.status.height = w, h
        self.status.message = f"BeamGage connected ({w}x{h}) src={self.status.serial_number}"
        logger.info(self.status.message)
        return self.get_status()

    def disconnect(self) -> None:
        if self._bg is not None:
            try:
                self._bg.Instance.Shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("BeamGage shutdown failed")
        self._bg = None
        self._frame = None
        self.status.connected = False
        self.status.acquiring = False

    def start_acquisition(self) -> None:
        if self._bg is None:
            return
        try:
            self._bg.DataSource.Start()
            self.status.acquiring = True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"BeamGage DataSource.Start failed: {e}")

    def stop_acquisition(self) -> None:
        if self._bg is None:
            return
        try:
            self._bg.DataSource.Stop()
        except Exception:  # noqa: BLE001
            pass
        self.status.acquiring = False

    def get_frame(self) -> np.ndarray | None:
        if self._bg is None or self._frame is None:
            return None
        try:
            # Only return genuinely NEW frames (ID advances each acquisition).
            try:
                fid = int(self._bg.FrameInfoResults.ID)
                if fid == self._last_id:
                    return None
                self._last_id = fid
            except Exception:  # noqa: BLE001  (no frame-info yet)
                pass
            if not bool(self._frame.HasData):
                return None
            w = int(self._frame.Width)
            h = int(self._frame.Height)
            data = self._frame.DoubleData        # System.Double[]
            if data is None or w <= 0 or h <= 0:
                return None
            arr = self._to_numpy(data, w * h)
            if arr is None or arr.size != w * h:
                return None
            frame = arr.reshape(h, w)
            self.status.frame_counter = self._last_id
            self.status.raw_peak_count = float(frame.max()) if frame.size else 0.0
            # Match the integer-count convention of the other backends.
            return np.clip(frame, 0, 65535).astype(np.uint16)
        except Exception:  # noqa: BLE001
            logger.exception("Ophir get_frame failed")
            return None

    def set_exposure(self, exposure_ms: float) -> None:
        # Exposure is set in the BeamGage window (ProgrammableSettings); driving it
        # from here is not wired yet. Record the request for status/metadata.
        self.status.exposure_ms = float(exposure_ms)
        logger.info(f"[ophir] set exposure in the BeamGage window "
                    f"(requested {exposure_ms:.2f} ms)")

    def get_status(self) -> CameraStatus:
        return copy_camera_status(self.status)

    # -- helpers -------------------------------------------------------------
    def _dims(self):
        try:
            return int(self._frame.Width), int(self._frame.Height)
        except Exception:  # noqa: BLE001
            return 0, 0

    @staticmethod
    def _to_numpy(net_double_array, n: int):
        """Fast-copy a System.Double[] into a float64 numpy array via Marshal."""
        from System import IntPtr
        from System.Runtime.InteropServices import Marshal
        out = np.empty(int(n), dtype=np.float64)
        Marshal.Copy(net_double_array, 0, IntPtr(int(out.ctypes.data)), int(n))
        return out
