"""IRC806 (IRCameras "Compact Science Camera") backend via Pleora eBUS.

Implements CameraInterface for the kspace/newcamera GUI. eBUS 5.1.5 has no
SWIG Python binding, so this drives its .NET API (PvDotNet.dll) through
pythonnet. Frames come back as 16-bit (Mono16). Until IRCameras supplies a
valid Pleora eBUS license, the image carries the eBUS evaluation watermark
(top-left) -- a licensing issue, not a code issue.

Requires (in the GUI's 64-bit venv): pythonnet, numpy, and the eBUS runtime
installed system-wide (Common Files\\Pleora\\eBUS SDK + PvDotNet.dll).
"""
from __future__ import annotations

import ctypes
import os
import struct

import numpy as np
from loguru import logger

from .camera_interface import CameraInterface, CameraStatus, copy_camera_status

EBUS_DIR = r"C:\Program Files\Common Files\Pleora\eBUS SDK"

# Dedicated temperature diagnostic log. Every temperature read (raw serial
# frame + parsed value + sanity check) is written here so a bad/jumpy/stuck
# reading can be inspected after the fact. One sink per process.
TEMP_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "logs", "temperature.log")
# Plausible ranges for sanity flagging (not enforced, just logged).
BOARD_TEMP_RANGE_C = (-10.0, 90.0)
FPA_TEMP_RANGE_K = (60.0, 320.0)
_temp_sink_added = False


def _ensure_temp_log() -> None:
    global _temp_sink_added
    if _temp_sink_added:
        return
    try:
        os.makedirs(os.path.dirname(TEMP_LOG_PATH), exist_ok=True)
        logger.add(TEMP_LOG_PATH,
                   filter=lambda r: r["extra"].get("temp", False),
                   rotation="5 MB", retention=10, level="DEBUG",
                   enqueue=True,
                   format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {message}")
        _temp_sink_added = True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not open temperature log {TEMP_LOG_PATH}: {e}")


def _hx(b: bytes) -> str:
    return b.hex(" ") if b else "<empty>"

# IMPORTANT: this camera's ExposureTime node is in SECONDS (unit 's'), and the
# firmware reports NO usable min/max (returns +/-FLT_MAX). So we MUST clamp in
# software -- sending a large value (e.g. ms treated as us) makes integration
# time enormous and freezes the stream (looks like a crash). Max integration is
# ~one frame time (120 Hz -> ~8 ms); keep a safe ceiling below that.
EXP_SECONDS_PER_MS = 1.0e-3      # ms -> seconds
EXP_MIN_MS = 0.01                # 10 us
EXP_MAX_MS = 8.0                 # ~1 frame at 120 Hz; avoids freezing
DEFAULT_EXPOSURE_MS = 0.3        # starting integration time

# GVCP heartbeat timeout (ms). Long so a brief link glitch / busy moment does
# not drop the control channel; the worker also auto-reconnects if it does.
HEARTBEAT_TIMEOUT_MS = 15000

# --- Serial-command-over-GigE bridge (ICD 319-003-070) ---------------------
# Reads values that aren't GenICam nodes (board/FPA temperature) by sending
# IRCameras serial packets through these GenICam registers.
_REG_TX_CTRL = 0x0000A400   # write 1=start packet, 2=end packet
_REG_TX_DATA = 0x0000A404   # write data byte (low 8 bits)
_REG_RX_CTRL = 0x0000A408   # write 1=start read, 2=end read; read 2=packet avail
_REG_RX_DATA = 0x0000A40C   # read data byte (low 8 bits)
_REG_RX_SIZE = 0x0000A410   # read: number of available bytes
_CRC_POLY = 0x755B
_CRC_TABLE = []
for _i in range(256):
    _crc = 0
    _c = _i
    for _ in range(8):
        if (_crc ^ (_c << 8)) & 0x8000:
            _crc = ((_crc << 1) ^ _CRC_POLY) & 0xFFFF
        else:
            _crc = (_crc << 1) & 0xFFFF
        _c = (_c << 1) & 0xFF
    _CRC_TABLE.append(_crc)


def _irc_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for c in data:
        crc ^= (c << 8) & 0xFFFF
        crc = ((crc << 8) ^ _CRC_TABLE[(crc >> 8) & 0xFF]) & 0xFFFF
    return (~crc) & 0xFFFF


class Irc806Camera(CameraInterface):
    def __init__(self, buffer_count: int = 16, fetch_timeout_ms: int = 1000) -> None:
        self.buffer_count = buffer_count
        self.fetch_timeout_ms = fetch_timeout_ms
        self._pv = None
        self._device = None
        self._stream = None
        self._buffers = []
        self._connection_id = ""
        self._serial_debug = False     # when True, _serial_command logs raw frames
        # Remembered imaging state so a reconnect/heal restores what the user
        # had set, instead of snapping back to defaults / raw output.
        self._desired_exposure_ms = DEFAULT_EXPOSURE_MS
        self._desired_correction = None   # None=untouched, True/False once set
        self._desired_nuc_slot = None
        _ensure_temp_log()
        self.status = CameraStatus(
            backend="irc806",
            message="IRC806 idle",
            exposure_ms=DEFAULT_EXPOSURE_MS,
            width=0,
            height=0,
        )

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

        pv = self._pv
        conn = self._discover()
        if not conn:
            self.status.connected = False
            self.status.message = "No GigE camera found (is it powered/cooled?)"
            return self.get_status()

        try:
            dev = pv.PvDevice.CreateAndConnect(conn)
            self._device = dev[0] if isinstance(dev, tuple) else dev
        except Exception as e:  # noqa: BLE001
            self._device = None
            self.status.connected = False
            self.status.message = (f"Connect failed: {e}. "
                                   "(Close eBUS Player if it holds the camera.)")
            logger.error(self.status.message)
            return self.get_status()

        self._connection_id = conn
        self.status.connected = True
        # Lengthen the GVCP heartbeat so a brief link glitch doesn't drop the
        # control channel (the cause of the "register 0xa000" stop errors).
        self._set_heartbeat_timeout(HEARTBEAT_TIMEOUT_MS)
        try:
            p = self._device.Parameters
            self.status.width = int(self._read(p, "Width") or 0)
            self.status.height = int(self._read(p, "Height") or 0)
            self.status.serial_number = str(self._read(p, "DeviceID") or
                                            self._read(p, "DeviceModelName") or "IRC806")
        except Exception:  # noqa: BLE001
            pass
        # Re-apply the desired integration time. On a fresh process this is the
        # default (0.3 ms); after a reconnect/heal it restores whatever the user
        # had set. The hard clamp in set_exposure still guards against a
        # previous session leaving a stream-freezing value.
        self.set_exposure(self._desired_exposure_ms)
        # Restore NUC slot + correction (the "same background") so frames after a
        # reconnect match those before it -- no discontinuity mid-scan.
        if self._desired_nuc_slot is not None:
            self.set_nuc_slot(self._desired_nuc_slot)
        if self._desired_correction:
            self.set_correction(True)
        self.status.message = (f"Connected {conn} "
                               f"({self.status.width}x{self.status.height}) "
                               f"@ {self._desired_exposure_ms:.2f} ms")
        logger.info(self.status.message)
        return self.get_status()

    def _set_heartbeat_timeout(self, ms: int) -> None:
        """Best-effort: lengthen the GVCP heartbeat timeout so brief network
        stalls/link glitches don't tear down the control connection."""
        if self._device is None:
            return
        try:
            self._device.Parameters.SetIntegerValue("GevHeartbeatTimeout", int(ms))
            return
        except Exception:  # noqa: BLE001
            pass
        try:  # GEV communication-parameter fallback (name varies by SDK build)
            comm = self._device.GetCommunicationParameters()
            for name in ("HeartbeatTimeout", "DefaultHeartbeatTimeout"):
                try:
                    comm.SetIntegerValue(name, int(ms))
                    return
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass

    def is_device_connected(self) -> bool:
        """Whether the eBUS device still reports an open control connection."""
        if self._device is None:
            return False
        try:
            fn = getattr(self._device, "IsConnected", None)
            if callable(fn):
                return bool(fn())
        except Exception:  # noqa: BLE001
            return False
        return self.status.connected

    def reconnect(self) -> bool:
        """Tear down and re-establish device + stream after a link drop.
        Returns True if streaming resumed."""
        logger.warning("Camera reconnect: re-establishing link...")
        try:
            self.disconnect()
        except Exception:  # noqa: BLE001
            pass
        st = self.connect()
        if st.connected:
            self.start_acquisition()
        if self.status.connected and self.status.acquiring:
            logger.info("Camera reconnect: streaming resumed")
            return True
        return False

    def disconnect(self) -> None:
        self.stop_acquisition()
        try:
            if self._stream is not None:
                self._stream.Close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._device is not None:
                self._device.Disconnect()
        except Exception:  # noqa: BLE001
            pass
        self._stream = None
        self._device = None
        self._buffers = []
        self.status.connected = False
        self.status.acquiring = False
        self.status.message = "IRC806 disconnected"

    def start_acquisition(self) -> None:
        if self._device is None or self.status.acquiring:
            return
        pv = self._pv
        try:
            so = pv.PvStream.CreateAndOpen(self._connection_id)
            self._stream = so[0] if isinstance(so, tuple) else so
            try:
                self._device.NegotiatePacketSize()
                try:
                    psize = int(self._read(self._device.Parameters,
                                           "GevSCPSPacketSize") or 0)
                    logger.info(f"GEV packet size negotiated: {psize} bytes"
                                f"{' (JUMBO)' if psize > 1500 else ''}")
                except Exception:  # noqa: BLE001
                    pass
                self._device.SetStreamDestination(self._stream.LocalIPAddress,
                                                  self._stream.LocalPort)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"GEV stream config: {e}")

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
            self.status.message = "IRC806 streaming"
            logger.info(f"Acquisition started ({count} buffers x {size} bytes)")
        except Exception as e:  # noqa: BLE001
            self.status.acquiring = False
            self.status.message = f"start_acquisition failed: {e}"
            logger.error(self.status.message)

    def stop_acquisition(self) -> None:
        if self._device is None or not self.status.acquiring:
            return
        connected = self.is_device_connected()
        try:
            if connected:
                self._device.Parameters.ExecuteCommand("AcquisitionStop")
                self._device.StreamDisable()
            if self._stream is not None:
                self._stream.AbortQueuedBuffers()
                drained = 0
                while int(self._stream.QueuedBufferCount) > 0 and drained < len(self._buffers) + 4:
                    self._retrieve(100)
                    drained += 1
        except Exception as e:  # noqa: BLE001
            # If the device already dropped its link, the AcquisitionStop write
            # (reg 0xa000) can't reach it -- that's expected during cleanup, not
            # an error worth alarming about.
            if connected:
                logger.warning(f"stop_acquisition: {e}")
            else:
                logger.debug(f"stop_acquisition (device already disconnected): {e}")
        self.status.acquiring = False
        self.status.message = "IRC806 acquisition stopped"

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
        finally:
            if pvbuffer is not None:
                self._stream.QueueBuffer(pvbuffer)

    def set_exposure(self, exposure_ms: float) -> None:
        # Clamp HARD: the firmware accepts any value but huge ones freeze the
        # stream. Convert ms -> seconds (this camera's ExposureTime unit).
        ms = float(np.clip(exposure_ms, EXP_MIN_MS, EXP_MAX_MS))
        self.status.exposure_ms = ms
        self._desired_exposure_ms = ms     # remember across reconnect/heal
        if self._device is None:
            return
        try:
            p = self._device.Parameters
            r = p.SetFloatValue("ExposureTime", ms * EXP_SECONDS_PER_MS)
            ok = r.IsOK if hasattr(r, "IsOK") else True
            self.status.message = f"Exposure {ms:.2f} ms ({'set' if ok else 'rejected'})"
        except Exception as e:  # noqa: BLE001
            self.status.message = f"set_exposure failed: {e}"
            logger.warning(self.status.message)

    def get_status(self) -> CameraStatus:
        return copy_camera_status(self.status)

    # -- temperature (serial-over-GigE, ICD 319-003-070) ---------------------
    def refresh_temperatures(self) -> None:
        """Read board (°C) and FPA (K) temperatures into status. Safe to call
        while streaming; failures leave the previous value untouched.

        Every read is logged to logs/temperature.log (raw serial payload +
        parsed value + sanity flag) so a bad/jumpy/stuck reading can be
        inspected. See _log_temp_read for the parse/validation."""
        if self._device is None:
            return
        tlog = logger.bind(temp=True)
        self._serial_debug = True
        try:
            b = self._read_temp_retry(tlog, "board_degC", 0x10, 0x25, BOARD_TEMP_RANGE_C)
            if b is not None:
                self.status.board_temp_c = b
            f = self._read_temp_retry(tlog, "fpa_K", 0x11, 0x01, FPA_TEMP_RANGE_K)
            if f is not None:
                self.status.fpa_temp_k = f
        finally:
            self._serial_debug = False

    def _read_temp_retry(self, tlog, label, op_hi, op_lo, valid_range,
                         attempts: int = 8, deadline_s: float = 2.5):
        """Read a temperature float, retrying until the frame parses to a value
        inside `valid_range`. The serial bridge returns truncated/misaligned
        frames while the image stream is busy (GVSP starves GVCP), so a clean
        frame usually arrives within a few tries. Returns the value, or None if
        no valid reading was obtained (caller keeps the previous value). Capped
        by `attempts` and `deadline_s` so it can't stall the stream."""
        import time
        lo, hi = valid_range
        start = time.time()
        for k in range(attempts):
            if time.time() - start > deadline_s:
                break
            try:
                payload = self._serial_command(op_hi, op_lo, recv_timeout_s=0.3)
            except Exception as e:  # noqa: BLE001
                tlog.warning(f"{label} try {k+1}: EXCEPTION {e}")
                continue
            if payload is None or len(payload) < 4:
                tlog.debug(f"{label} try {k+1}: short/empty payload={_hx(payload or b'')}")
                continue
            value = struct.unpack("<f", payload[:4])[0]
            if lo <= value <= hi:
                tlog.info(f"{label} = {value:.3f} [OK] (try {k+1}) raw={_hx(payload)}")
                return value
            tlog.debug(f"{label} try {k+1}: out-of-range {value:.3f} raw={_hx(payload)}")
        tlog.warning(f"{label}: no valid reading in {k+1} tries -- keeping previous "
                     f"({getattr(self.status, 'board_temp_c' if 'board' in label else 'fpa_temp_k', float('nan'))})")
        return None

    # -- camera state file (loads NUC + matched params) ----------------------
    def load_state(self, filename: str) -> bool:
        """Load a camera state .xml on the camera (opcode 01 01). The state
        applies its stored NUC and parameters. Filename only, no path."""
        if self._device is None:
            return False
        try:
            resp = self._serial_command(0x01, 0x01, filename.encode("ascii") + b"\x00",
                                        recv_timeout_s=20.0)   # flash/NUC load is slow
            ok = len(resp) >= 1 and resp[0] == 0xA0
            self.status.message = (f"Loaded state {filename}" if ok
                                   else f"Load state failed (resp {resp.hex()})")
            logger.info(self.status.message)
            return ok
        except Exception as e:  # noqa: BLE001
            self.status.message = f"load_state error: {e}"
            logger.warning(self.status.message)
            return False

    # -- NUC / bad-pixel correction (serial-over-GigE) -----------------------
    def set_correction(self, enabled: bool) -> None:
        """Output mux (opcode 20 02): 2 = NUC'd + BPR-corrected, 0 = raw.
        This is what makes the live GigE image non-uniformity corrected."""
        self._desired_correction = bool(enabled)   # remember across reconnect/heal
        if self._device is None:
            return
        try:
            self._serial_command(0x20, 0x02, struct.pack("<i", 2 if enabled else 0))
            self.status.message = f"NUC/BPR output {'ON' if enabled else 'OFF (raw)'}"
        except Exception as e:  # noqa: BLE001
            logger.warning(f"set_correction failed: {e}")

    def set_bpr(self, enabled: bool) -> None:
        """Bad-pixel replacement module (opcode 20 12)."""
        if self._device is None:
            return
        try:
            self._serial_command(0x20, 0x12, bytes([1 if enabled else 0]))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"set_bpr failed: {e}")

    def set_nuc_slot(self, slot: int) -> None:
        """Active NUC calibration slot 0-11 (opcode 20 30)."""
        self._desired_nuc_slot = int(slot)   # remember across reconnect/heal
        if self._device is None:
            return
        try:
            self._serial_command(0x20, 0x30, struct.pack("<i", int(slot)))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"set_nuc_slot failed: {e}")

    def _reg_write(self, addr: int, val: int) -> None:
        self._device.WriteRegister(int(addr), int(val) & 0xFFFFFFFF)

    def _reg_read(self, addr: int) -> int:
        ret = self._device.ReadRegister(int(addr), 0)   # ByRef out
        return int(ret[-1]) if isinstance(ret, (list, tuple)) else int(ret)

    def _serial_command(self, op_hi: int, op_lo: int, data: bytes = b"",
                        recv_timeout_s: float = 1.0) -> bytes:
        """Send an IRCameras serial packet via the GigE register bridge and
        return the response payload (bytes after the echoed opcode).
        recv_timeout_s: bump for slow commands (state/NUC load can take seconds)."""
        import time
        payload = bytes([op_hi, op_lo]) + data
        crc = _irc_crc16(bytes([0x00, 0xFF]) + payload)     # nak + type + payload
        esc = bytearray()
        for b in payload:                                   # escape FF/5C in payload
            if b in (0xFF, 0x5C):
                esc.append(0x5C)
            esc.append(b)
        wire = bytes([0x3E, 0x00, 0xFF]) + bytes(esc) + \
            bytes([(crc >> 8) & 0xFF, crc & 0xFF, 0x3E])
        # transmit
        self._reg_write(_REG_TX_CTRL, 1)
        for b in wire:
            self._reg_write(_REG_TX_DATA, b)
        self._reg_write(_REG_TX_CTRL, 2)
        # receive (poll until a packet is available or timeout)
        deadline = time.time() + recv_timeout_s
        while time.time() < deadline:
            if self._reg_read(_REG_RX_CTRL) == 2:
                break
            time.sleep(0.01)
        else:
            return b""
        self._reg_write(_REG_RX_CTRL, 1)
        n = self._reg_read(_REG_RX_SIZE)
        raw = bytes(self._reg_read(_REG_RX_DATA) & 0xFF for _ in range(n))
        self._reg_write(_REG_RX_CTRL, 2)
        inner = raw[1:-1] if len(raw) >= 2 and raw[0] == 0x3E and raw[-1] == 0x3E else raw
        de = bytearray()
        i = 0
        while i < len(inner):
            if inner[i] == 0x5C and i + 1 < len(inner):
                i += 1
            de.append(inner[i])
            i += 1
        # de = [nak][FF][op_hi][op_lo][response...][crc_hi][crc_lo]
        payload = bytes(de[4:-2]) if len(de) > 6 else b""
        if self._serial_debug:
            logger.bind(temp=True).debug(
                f"serial op={op_hi:02X}{op_lo:02X} tx={_hx(wire)} "
                f"rx_n={n} raw={_hx(raw)} de={_hx(bytes(de))} payload={_hx(payload)}")
        return payload

    # -- helpers -------------------------------------------------------------
    def _discover(self):
        pv = self._pv
        system = pv.PvSystem()
        system.Find()
        for i in range(system.InterfaceCount):
            itf = system.GetInterface(i)
            try:
                n = itf.DeviceCount
            except Exception:  # noqa: BLE001
                n = 0
            for j in range(n):
                return str(itf.GetDeviceInfo(j).ConnectionID)
        return ""

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
        """RetrieveBuffer takes ByRef args; pass real instances, read back tuple."""
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
        """PvImage.DataPointer is a byte* wrapped as System.Reflection.Pointer.
        Recover its address via ISerializable ("_ptr" is a System.IntPtr)."""
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
        if w == 0 or h == 0:
            return None
        addr = self._pointer_address(pvimage.DataPointer)
        if not addr:
            return None
        if bpp <= 8:
            cbuf = (ctypes.c_uint8 * (w * h)).from_address(addr)
            return np.frombuffer(cbuf, dtype=np.uint8).reshape(h, w).copy()
        cbuf = (ctypes.c_uint16 * (w * h)).from_address(addr)
        return np.frombuffer(cbuf, dtype=np.uint16).reshape(h, w).copy()
