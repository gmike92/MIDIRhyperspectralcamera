"""Direct GigE Vision backend for the Allied Vision Goldeye G-033 (the camera
BeamGage serves as "Ophir SP1203"). Streams straight over Pleora eBUS -- the same
stack as the IRC806 -- bypassing BeamGage entirely.

Why not BeamGage: the Goldeye is a standard GigE Vision camera. Direct eBUS gives
120 fps, 14-bit (Mono14), and direct ExposureTime/Gain control via GenICam --
versus BeamGage's throttled ~0.15 fps and awkward saturation config. We only need
raw frames for the TWINS scan, not BeamGage's beam-profiling math.

Device selection: picks the camera whose model contains "Goldeye" (so it never
grabs the IRC806 on the other NIC). Exposure is standard GenICam microseconds.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np
from loguru import logger

from .camera_interface import CameraInterface, CameraStatus, copy_camera_status

EBUS_DIR = r"C:\Program Files\Common Files\Pleora\eBUS SDK"
DEFAULT_EXPOSURE_MS = 1.0
EXP_MIN_MS = 0.001      # 1 us (camera minimum)
EXP_MAX_MS = 1000.0     # up to 1 s for dim beams (frame rate drops accordingly;
                        # camera range is 1 us .. ~30 s, it clamps beyond this)
MODEL_HINT = "Goldeye"  # select this camera; avoids grabbing the IRC806


class GoldeyeCamera(CameraInterface):
    def __init__(self, buffer_count: int = 16, fetch_timeout_ms: int = 1000) -> None:
        self.buffer_count = buffer_count
        self.fetch_timeout_ms = fetch_timeout_ms
        self._pv = None
        self._device = None
        self._stream = None
        self._buffers = []
        self._connection_id = ""
        self._desired_exposure_ms = DEFAULT_EXPOSURE_MS
        self.status = CameraStatus(backend="goldeye", message="Goldeye idle",
                                   exposure_ms=DEFAULT_EXPOSURE_MS)

    # -- eBUS .NET bootstrap -------------------------------------------------
    def _ensure_loaded(self):
        if self._pv is not None:
            return
        if os.path.isdir(EBUS_DIR):
            try:
                os.add_dll_directory(EBUS_DIR)
            except Exception:  # noqa: BLE001
                pass
            os.environ["PATH"] = EBUS_DIR + os.pathsep + os.environ.get("PATH", "")
        import clr  # noqa: F401  (pythonnet)
        from System.Reflection import Assembly
        Assembly.LoadFrom(os.path.join(EBUS_DIR, "PvDotNet.dll"))
        import PvDotNet as pv
        self._pv = pv

    # -- CameraInterface -----------------------------------------------------
    def connect(self) -> CameraStatus:
        try:
            self._ensure_loaded()
        except Exception as e:  # noqa: BLE001
            self.status.connected = False
            self.status.message = f"eBUS/.NET load failed: {e}"
            logger.error(self.status.message)
            return self.get_status()

        conn = self._discover()
        if not conn:
            self.status.connected = False
            self.status.message = ("Goldeye not found (powered? BeamGage/Spiricon "
                                   "services still holding it?)")
            return self.get_status()

        pv = self._pv
        try:
            res = pv.PvDevice.CreateAndConnect(conn)
            self._device = res[0] if isinstance(res, tuple) else res
        except Exception as e:  # noqa: BLE001
            self._device = None
            self.status.connected = False
            self.status.message = (f"Connect failed: {e}. Close BeamGage / stop "
                                   "Spiricon services (they own the camera).")
            logger.error(self.status.message)
            return self.get_status()

        self._connection_id = conn
        self.status.connected = True
        try:
            p = self._device.Parameters
            self.status.width = int(self._read(p, "Width") or 0)
            self.status.height = int(self._read(p, "Height") or 0)
            self.status.serial_number = str(self._read(p, "DeviceSerialNumber") or
                                            self._read(p, "DeviceModelName") or "Goldeye")
        except Exception:  # noqa: BLE001
            pass
        # Re-apply the remembered exposure (fresh process -> DEFAULT).
        self.set_exposure(self._desired_exposure_ms)
        self.status.message = (f"Goldeye connected ({self.status.width}x"
                               f"{self.status.height}) {conn}")
        logger.info(self.status.message)
        return self.get_status()

    def disconnect(self) -> None:
        try:
            self.stop_acquisition()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._stream is not None:
                self._stream.Close()
            if self._device is not None:
                self._device.Disconnect()
        except Exception:  # noqa: BLE001
            pass
        self._stream = None
        self._device = None
        self._buffers = []
        self.status.connected = False
        self.status.acquiring = False
        self.status.message = "Goldeye disconnected"

    def start_acquisition(self) -> None:
        if self._device is None or self.status.acquiring:
            return
        pv = self._pv
        try:
            so = pv.PvStream.CreateAndOpen(self._connection_id)
            self._stream = so[0] if isinstance(so, tuple) else so
            try:
                self._device.NegotiatePacketSize()
                self._device.SetStreamDestination(self._stream.LocalIPAddress,
                                                  self._stream.LocalPort)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"GEV stream config: {e}")

            # Free-run at full rate: BeamGage may leave the camera in a slow
            # triggered mode, and Allied Vision's DeviceLinkThroughputLimit paces
            # the stream way down (the ~1 fps we saw). Force continuous streaming
            # and lift the throughput cap. Best-effort (nodes vary by firmware).
            for node, val in (("TriggerMode", "Off"),
                              ("DeviceLinkThroughputLimitMode", "Off")):
                try:
                    self._device.Parameters.SetEnumValue(node, val)
                except Exception:  # noqa: BLE001
                    pass

            size = int(self._device.PayloadSize)
            count = min(int(self._stream.QueuedBufferMaximum), self.buffer_count)
            self._buffers = []
            for _ in range(count):
                b = pv.PvBuffer()
                b.Alloc(size)
                self._buffers.append(b)
                self._stream.QueueBuffer(b)

            self._device.StreamEnable()
            self._device.Parameters.ExecuteCommand("AcquisitionStart")
            self.status.acquiring = True
            self.status.message = "Goldeye streaming"
            logger.info(f"Acquisition started ({count} buffers x {size} bytes)")
        except Exception as e:  # noqa: BLE001
            self.status.acquiring = False
            self.status.message = f"start_acquisition failed: {e}"
            logger.error(self.status.message)

    def stop_acquisition(self) -> None:
        if self._device is None or not self.status.acquiring:
            return
        try:
            self._device.Parameters.ExecuteCommand("AcquisitionStop")
            self._device.StreamDisable()
            if self._stream is not None:
                self._stream.AbortQueuedBuffers()
                drained = 0
                while int(self._stream.QueuedBufferCount) > 0 and drained < len(self._buffers) + 4:
                    self._retrieve(100)
                    drained += 1
        except Exception as e:  # noqa: BLE001
            logger.debug(f"stop_acquisition: {e}")
        self.status.acquiring = False
        self.status.message = "Goldeye acquisition stopped"

    def get_frame(self) -> np.ndarray | None:
        if self._stream is None or not self.status.acquiring:
            return None
        pv = self._pv
        res, pvbuffer, opres = self._retrieve(self.fetch_timeout_ms)
        if res is None or not res.IsOK:
            return None
        try:
            if opres.IsOK and int(pvbuffer.PayloadType) == int(pv.PvPayloadType.Image):
                arr = self._image_to_numpy(pvbuffer.Image)
                if arr is not None:
                    self.status.frame_counter += 1
                    self.status.raw_peak_count = float(arr.max())
                    return arr
            return None
        except Exception:  # noqa: BLE001  (never let a bad frame crash the worker)
            return None
        finally:
            if pvbuffer is not None:
                self._stream.QueueBuffer(pvbuffer)

    def set_exposure(self, exposure_ms: float) -> None:
        ms = float(np.clip(exposure_ms, EXP_MIN_MS, EXP_MAX_MS))
        self.status.exposure_ms = ms
        self._desired_exposure_ms = ms
        if self._device is None:
            return
        try:
            # Standard GenICam ExposureTime is in MICROSECONDS.
            r = self._device.Parameters.SetFloatValue("ExposureTime", ms * 1000.0)
            ok = r.IsOK if hasattr(r, "IsOK") else True
            self.status.message = f"Exposure {ms:.3f} ms ({'set' if ok else 'rejected'})"
        except Exception as e:  # noqa: BLE001
            self.status.message = f"set_exposure failed: {e}"
            logger.warning(self.status.message)

    # Options settable from the UI: enum nodes (symbolic string) + float nodes.
    _FLOAT_OPTIONS = ("AcquisitionFrameRate",)

    def set_option(self, name: str, value) -> None:
        """Set a GenICam node (IntegrationMode, ExposureAuto, SensorGain,
        AcquisitionFrameRate, ...). Enum nodes take the symbolic string."""
        if self._device is None:
            return
        try:
            p = self._device.Parameters
            if name in self._FLOAT_OPTIONS:
                p.SetFloatValue(name, float(value))
            else:
                p.SetEnumValue(name, str(value))
            self.status.message = f"{name} = {value}"
            logger.info(f"[goldeye] {name} = {value}")
        except Exception as e:  # noqa: BLE001
            self.status.message = f"set {name} failed: {e}"
            logger.warning(self.status.message)

    def get_status(self) -> CameraStatus:
        return copy_camera_status(self.status)

    # -- helpers (generic eBUS, shared shape with the IRC806 driver) ----------
    def _discover(self):
        pv = self._pv
        system = pv.PvSystem()
        system.Find()
        fallback = ""
        for i in range(system.InterfaceCount):
            itf = system.GetInterface(i)
            try:
                n = itf.DeviceCount
            except Exception:  # noqa: BLE001
                n = 0
            for j in range(n):
                di = itf.GetDeviceInfo(j)
                cid = str(di.ConnectionID)
                model = ""
                try:
                    model = str(di.ModelName)
                except Exception:  # noqa: BLE001
                    pass
                if MODEL_HINT.lower() in model.lower():
                    return cid                    # the Goldeye -- take it
                if "compact science" not in model.lower():
                    fallback = fallback or cid    # any non-IRC806 device
        return fallback

    @staticmethod
    def _read(params, name):
        try:
            prm = params.Get(name)
            if prm is None:
                return None
            v = prm.ToString()
            return v[-1] if isinstance(v, (list, tuple)) else v
        except Exception:  # noqa: BLE001
            return None

    def _retrieve(self, timeout_ms):
        pv = self._pv
        try:
            buf = pv.PvBuffer()
            opres = pv.PvResult(pv.PvResultCode.OK)
            ret = self._stream.RetrieveBuffer(buf, opres, timeout_ms)
            if isinstance(ret, tuple):
                return ret[0], (ret[1] if len(ret) > 1 else buf), (ret[2] if len(ret) > 2 else opres)
            return ret, buf, opres
        except Exception:  # noqa: BLE001
            return None, None, None

    @staticmethod
    def _pointer_address(ptr):
        import clr as _clr
        from System import IntPtr
        from System.Runtime.Serialization import (SerializationInfo, FormatterConverter,
                                                  StreamingContext, StreamingContextStates,
                                                  ISerializable)
        si = SerializationInfo(ptr.GetType(), FormatterConverter())
        ISerializable.GetObjectData(ptr, si, StreamingContext(StreamingContextStates.All))
        ip = si.GetValue("_ptr", _clr.GetClrType(IntPtr))
        return ip.ToInt64()

    def _image_to_numpy(self, pvimage):
        w = int(pvimage.Width)
        h = int(pvimage.Height)
        bpp = int(pvimage.BitsPerPixel)
        # Reject incomplete/garbage frames: their Width/Height come back bogus
        # (which would try to allocate absurd arrays). Trust the sensor geometry
        # read at connect; also bound-check as a backstop.
        ew, eh = self.status.width, self.status.height
        if ew and eh and (w != ew or h != eh):
            return None
        if not (0 < w <= 8192 and 0 < h <= 8192):
            return None
        addr = self._pointer_address(pvimage.DataPointer)
        if not addr:
            return None
        if bpp <= 8:
            cbuf = (ctypes.c_uint8 * (w * h)).from_address(addr)
            return np.frombuffer(cbuf, dtype=np.uint8).reshape(h, w).copy()
        cbuf = (ctypes.c_uint16 * (w * h)).from_address(addr)
        return np.frombuffer(cbuf, dtype=np.uint16).reshape(h, w).copy()
