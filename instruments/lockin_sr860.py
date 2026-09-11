"""
lockin_sr860.py -- Stanford Research SR865A / SR865 / SR860 lock-in driver.

Hardware-only (NO GUI), same shape as the other instrument drivers in this repo
(connect / read / disconnect + a `simulate=True` mode so the scan logic runs
with nothing attached).

Transport
    Everything the app needs is plain SCPI text, so the driver is built on a
    single `query(cmd)` / `send(cmd)` pair and can speak over any of:

        'vxi11'  -- LAN, VXI-11 protocol   (address = IP)          [recommended]
        'visa'   -- NI/Keysight VISA       (address = resource string, e.g.
                                            'USB0::0xB506::0x2000::002xxx::INSTR')
        'tcp'    -- raw socket on port 23  (address = IP or IP:port)
        'sim'    -- no hardware

    The vendored `srsinst.sr860` package (instruments/srsinst.sr860) is used for
    'vxi11' and 'visa' when it is importable -- it brings the SRS-maintained
    interface layer. If it (or its `srsgui` dependency) is missing, the driver
    falls back to talking to the same interfaces directly through `vxi11` /
    `pyvisa`, and finally to a raw socket. Nothing about the rest of the app
    changes either way; `backend` records what was actually used.

Reading a value
    Two families of "channel" exist on an SR865A and both are exposed here:

      * OUTPUT PARAMETERS (`OUTP? n`) -- X, Y, R, theta, Aux In 1-4, ...
        Always available, independent of what the front panel is displaying.
      * DISPLAY DATA CHANNELS (`OUTR? n`) -- Data 1-4, i.e. whatever the four
        front-panel display slots are currently configured to show.

    `read_value("R")` covers the first, `read_display("Data 1")` the second, and
    `read_channel(name)` accepts either name.

Usage:
    from instruments.lockin_sr860 import LockInSR860
    li = LockInSR860()
    li.connect("vxi11", "192.168.1.10")     # or connect("sim", "")
    li.time_constant = 0.1
    print(li.read_channel("R"), "V")
    li.disconnect()
"""
from __future__ import annotations

import math
import os
import random
import socket
import sys
import threading
import time

# ---------------------------------------------------------------------------
# srsinst.sr860 normally comes from the venv (requirements.txt). A git clone of
# it also sits next to this file; that checkout is only a fallback, for a machine
# where the package was never pip-installed.
# ---------------------------------------------------------------------------
_VENDORED_SRSINST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "srsinst.sr860")


def _import_srsinst():
    """Import srsinst.sr860. Returns the SR860 class, or None if unavailable.

    The INSTALLED package wins: it is tried first and the vendored checkout is
    only put on sys.path if that fails, so a proper `pip install` is never
    shadowed by the clone sitting in this directory.
    """
    try:
        from srsinst.sr860 import SR860  # noqa: PLC0415
        return SR860
    except Exception:  # noqa: BLE001
        pass
    if os.path.isdir(_VENDORED_SRSINST):
        if _VENDORED_SRSINST not in sys.path:
            sys.path.append(_VENDORED_SRSINST)
        try:
            from srsinst.sr860 import SR860  # noqa: PLC0415
            return SR860
        except Exception:  # noqa: BLE001
            pass
    return None


#: Where to get a VISA backend, quoted in several error messages.
#: USB needs TWO things -- a VISA implementation, and a Windows driver bound to
#: the instrument. NI-VISA/Keysight ship both; pyvisa-py supplies only the first
#: (libusb cannot reach a device that is still unbound, so the SR865A stays
#: invisible until Zadig rebinds it to WinUSB).
VISA_RUNTIME_HINT = (
    "install NI-VISA (ni.com) or Keysight IO Libraries -- they provide the VISA "
    "implementation AND the USBTMC driver Windows binds to the instrument. "
    "(pyvisa-py is an alternative backend, but on Windows USB it also needs the "
    "device rebound to WinUSB with Zadig.) LAN (VXI-11) needs neither.")


def probe_visa() -> tuple[list[str], str]:
    """Look for VISA instruments. Returns (resources, human-readable status).

    Separates the three states that all otherwise present as "no devices":
    pyvisa missing, pyvisa present but no VISA runtime behind it, and a working
    runtime that simply sees nothing attached.
    """
    try:
        import pyvisa  # noqa: PLC0415
    except ImportError:
        return [], "pyvisa is not installed in this environment (pip install pyvisa)"
    try:
        rm = pyvisa.ResourceManager()
    except Exception as e:  # noqa: BLE001
        # The classic one: "Could not locate a VISA implementation".
        return [], f"pyvisa is installed but found no VISA runtime: {e}. " \
                   f"To use USB, {VISA_RUNTIME_HINT}"
    try:
        resources = [r for r in rm.list_resources() if not r.startswith("ASRL")]
    except Exception as e:  # noqa: BLE001
        return [], f"VISA resource scan failed: {e}"
    if not resources:
        return [], ("VISA is working but sees no instruments. Check the USB cable, "
                    "and that the lock-in appears in Device Manager without a "
                    "warning triangle (a device shown with an error has no "
                    "USBTMC driver bound -- installing NI-VISA provides it).")
    backend = getattr(rm.visalib, "library_path", "") or type(rm.visalib).__name__
    return resources, f"{len(resources)} VISA resource(s) via {backend}"


def _srsinst_is_connected(inst) -> bool:
    """Did an srsinst SR860 actually open its link?

    Needed because srsinst's interfaces report failure by leaving the object
    unconnected rather than by raising -- see `_connect_vxi11`.
    """
    try:
        state = inst.is_connected
        return bool(state() if callable(state) else state)
    except Exception:  # noqa: BLE001
        return True     # unknown API shape: let the *IDN? handshake decide


def explain_connection_error(error: BaseException, address: str = "") -> str:
    """Turn a transport exception into something a person can act on.

    Socket errors in particular arrive as bare errnos ("[Errno 11001]
    getaddrinfo failed") that say nothing about which knob to turn.
    """
    errno = getattr(error, "errno", None)
    text = str(error)
    where = f" '{address}'" if address else ""
    if errno == 11001 or "getaddrinfo" in text:
        return (f"cannot resolve{where}: not a valid IP address or hostname. "
                "Enter the lock-in's IP (SR865A front panel: System → Ethernet).")
    if errno in (10060, 110) or "timed out" in text.lower():
        return (f"no answer from{where}: the address resolves but nothing "
                "replied. Check the lock-in is powered and on this network, and "
                "that its remote interface is enabled.")
    if errno in (10061, 111) or "refused" in text.lower():
        return (f"connection refused by{where}: something is there but not "
                "listening on that port.")
    if errno in (10065, 113) or "unreachable" in text.lower():
        return f"{address or 'host'} is unreachable: check the subnet and cabling."
    return text


# ---------------------------------------------------------------------------
# Command tables (SR860 series programming manual)
# ---------------------------------------------------------------------------

#: `OUTP? n` output parameters -- name -> index.
OUTPUT_PARAMETERS = {
    "X": 0,
    "Y": 1,
    "R": 2,
    "Theta": 3,
    "Aux In 1": 4,
    "Aux In 2": 5,
    "Aux In 3": 6,
    "Aux In 4": 7,
    "X noise": 8,
    "Y noise": 9,
    "Aux Out 1": 10,
    "Aux Out 2": 11,
    "Phase": 12,
    "Sine amplitude": 13,
    "DC level": 14,
    "Int. frequency": 15,
    "Ext. frequency": 16,
}

#: `OUTR? n` front-panel display data channels -- name -> index.
DISPLAY_CHANNELS = {
    "Data 1": 0,
    "Data 2": 1,
    "Data 3": 2,
    "Data 4": 3,
}

#: Units of each output parameter, for axis labels and saved metadata.
PARAMETER_UNITS = {
    "X": "V", "Y": "V", "R": "V", "Theta": "deg",
    "Aux In 1": "V", "Aux In 2": "V", "Aux In 3": "V", "Aux In 4": "V",
    "X noise": "V", "Y noise": "V",
    "Aux Out 1": "V", "Aux Out 2": "V",
    "Phase": "deg", "Sine amplitude": "V", "DC level": "V",
    "Int. frequency": "Hz", "Ext. frequency": "Hz",
}

#: `OFLT i` time constants in seconds, index-ordered (1e-6 .. 30e3).
TIME_CONSTANTS = [float(f"{j * 10 ** (i - 6):.1e}")
                  for i in range(11) for j in (1.0, 3.0)]

#: `SCAL i` voltage sensitivities in volts, index-ordered (1 V .. 1 nV).
VOLTAGE_SENSITIVITIES = [1.0] + [float(f"{j * 10 ** -(i + 1):.1e}")
                                 for i in range(9) for j in (5.0, 2.0, 1.0)]

#: `OFSL i` low-pass filter slopes in dB/oct.
FILTER_SLOPES = [6, 12, 18, 24]

#: `IRNG i` voltage input ranges in volts.
INPUT_RANGES = [1.0, 0.3, 0.1, 0.03, 0.01]


def _nearest_index(table: list, value: float) -> int:
    """Index of the table entry closest to `value` (log distance for decades)."""
    return min(range(len(table)),
               key=lambda i: abs(math.log10(table[i] / value))
               if table[i] > 0 and value > 0 else abs(table[i] - value))


class LockInSR860:
    """SR865A / SR865 / SR860 lock-in amplifier over VXI-11, VISA, TCP or sim."""

    INTERFACES = ("vxi11", "visa", "tcp", "sim")

    def __init__(self) -> None:
        self.is_connected = False
        self.backend = None         # 'srsinst-vxi11' | 'vxi11' | 'visa' | 'tcp' | 'sim'
        self.interface = None       # what the caller asked for
        self.address = ""
        self.identity = ""
        self._inst = None           # srsinst SR860 / vxi11.Instrument / pyvisa resource
        self._sock = None           # raw TCP socket
        self._lock = threading.RLock()   # one query at a time (GUI poll vs scan thread)
        self.timeout_s = 5.0
        self._last_error = ""
        # Simulator state: a Gaussian-enveloped fringe pattern vs. "position",
        # so the scan panel draws something interferogram-shaped with no hardware.
        self._sim_position_mm = 0.0

    # -- connection ----------------------------------------------------------
    @staticmethod
    def find_visa_resources() -> list[str]:
        """VISA resource strings for anything currently attached (USB/GPIB/LAN)."""
        return probe_visa()[0]

    def connect(self, interface: str = "vxi11", address: str = "",
                timeout_s: float = 5.0) -> bool:
        """Open the instrument. `interface` is one of LockInSR860.INTERFACES."""
        if self.is_connected:
            return True
        interface = (interface or "vxi11").lower()
        self.interface = interface
        self.address = address.strip()
        self.timeout_s = float(timeout_s)
        self._last_error = ""

        try:
            if interface == "sim":
                self.backend = "sim"
            elif interface == "vxi11":
                self._connect_vxi11()
            elif interface == "visa":
                self._connect_visa()
            elif interface == "tcp":
                self._connect_tcp()
            else:
                raise ValueError(f"unknown interface '{interface}'")
        except Exception as e:  # noqa: BLE001
            self._last_error = explain_connection_error(e, self.address)
            print(f"[LockIn] connect failed ({interface}): {self._last_error}")
            self._close_transport()
            return False

        self.is_connected = True
        if self.backend == "sim":
            self.identity = "Stanford_Research_Systems,SR865A,SIMULATED,v1.00"
        else:
            try:
                self.identity = self.query("*IDN?")
            except Exception as e:  # noqa: BLE001
                self._last_error = (
                    "the link opened but the lock-in did not answer *IDN?: "
                    + explain_connection_error(e, self.address))
                print(f"[LockIn] *IDN? failed: {self._last_error}")
                self.is_connected = False
                self._close_transport()
                return False
        print(f"[LockIn] connected via {self.backend}: {self.identity}")
        return True

    def _check_host_address(self, what: str) -> None:
        """Reject an address that is not a usable host before we try to dial it.

        A VISA resource string pasted into a LAN field is the common way to get
        the opaque "[Errno 11001] getaddrinfo failed" -- Windows tries to resolve
        `USB0::0xB506::...` as a hostname and fails DNS. Catching it here says
        what is actually wrong.
        """
        if not self.address:
            raise ValueError(f"{what} needs the instrument's IP address "
                             "(e.g. 192.168.1.10)")
        if "::" in self.address:
            raise ValueError(
                f"'{self.address}' is a VISA resource string, but the interface "
                f"is set to {what}. Either switch the interface to VISA "
                "(USB/GPIB), or enter the lock-in's IP address instead.")

    def _connect_vxi11(self) -> None:
        self._check_host_address("VXI-11")
        SR860 = _import_srsinst()
        if SR860 is not None:
            inst = SR860("vxi11", self.address)
            # srsinst's Vxi11Interface.connect() SWALLOWS connection errors: it
            # logs and returns, leaving a dead object behind. Without this check
            # the failure would not surface until the first query.
            if not _srsinst_is_connected(inst):
                raise ConnectionError(
                    f"no VXI-11 response from {self.address} -- check the address, "
                    "and that the lock-in is powered and reachable "
                    "(SR865A: System → Ethernet shows its IP)")
            self._inst = inst
            self.backend = "srsinst-vxi11"
            return
        import vxi11  # noqa: PLC0415  (python-vxi11)
        self._inst = vxi11.Instrument(self.address)
        self._inst.timeout = self.timeout_s
        self._inst.open()
        self.backend = "vxi11"

    def _connect_visa(self) -> None:
        if not self.address:
            raise ValueError("VISA needs a resource string "
                             "(e.g. USB0::0xB506::0x2000::002xxxx::INSTR)")
        SR860 = _import_srsinst()
        if SR860 is not None:
            inst = SR860("visa", self.address)
            if not _srsinst_is_connected(inst):
                raise ConnectionError(
                    f"VISA could not open '{self.address}' -- check the resource "
                    "string (use Find VISA) and that a VISA runtime is installed")
            self._inst = inst
            self.backend = "srsinst-visa"
            return
        import pyvisa  # noqa: PLC0415
        try:
            rm = pyvisa.ResourceManager()
        except Exception as e:  # noqa: BLE001
            raise ConnectionError(
                f"no VISA runtime behind pyvisa ({e}). To use USB, "
                f"{VISA_RUNTIME_HINT}") from e
        self._inst = rm.open_resource(self.address)
        self._inst.timeout = int(self.timeout_s * 1000)
        self.backend = "visa"

    def _connect_tcp(self) -> None:
        self._check_host_address("raw TCP")
        host, _, port = self.address.partition(":")
        sock = socket.create_connection((host, int(port or 23)), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        self._sock = sock
        self.backend = "tcp"
        # The SR865A greets a raw telnet connection with a banner; drop it so it
        # is not mistaken for the answer to the first query.
        try:
            sock.settimeout(0.5)
            sock.recv(4096)
        except (socket.timeout, OSError):
            pass
        finally:
            sock.settimeout(self.timeout_s)

    def _close_transport(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._inst is not None:
                # srsinst SR860, vxi11.Instrument and pyvisa resources all
                # expose one of these.
                for name in ("disconnect", "close"):
                    fn = getattr(self._inst, name, None)
                    if callable(fn):
                        fn()
                        break
        except Exception:  # noqa: BLE001
            pass
        self._inst = None
        self._sock = None

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self._close_transport()
        self.is_connected = False
        self.backend = None
        print("[LockIn] disconnected")

    # -- raw SCPI ------------------------------------------------------------
    def send(self, cmd: str) -> None:
        """Write a command with no reply."""
        if not self.is_connected:
            raise RuntimeError("lock-in not connected")
        with self._lock:
            if self.backend == "sim":
                return
            if self._sock is not None:
                self._sock.sendall((cmd + "\n").encode("ascii"))
            elif self.backend.startswith("srsinst"):
                self._inst.send(cmd)
            elif self.backend == "vxi11":
                self._inst.write(cmd)
            else:                       # pyvisa
                self._inst.write(cmd)

    def query(self, cmd: str) -> str:
        """Write a query and return its reply, stripped."""
        if not self.is_connected:
            raise RuntimeError("lock-in not connected")
        with self._lock:
            if self.backend == "sim":
                return self._sim_query(cmd)
            if self._sock is not None:
                self._sock.sendall((cmd + "\n").encode("ascii"))
                chunks = []
                while True:
                    chunk = self._sock.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
                return b"".join(chunks).decode("ascii", "replace").strip()
            if self.backend.startswith("srsinst"):
                return str(self._inst.query_text(cmd)).strip()
            if self.backend == "vxi11":
                return str(self._inst.ask(cmd)).strip()
            return str(self._inst.query(cmd)).strip()   # pyvisa

    def query_float(self, cmd: str) -> float:
        return float(self.query(cmd))

    def query_int(self, cmd: str) -> int:
        return int(float(self.query(cmd)))

    # -- simulator -----------------------------------------------------------
    def set_sim_position(self, position_mm: float) -> None:
        """Tell the simulator where the stage is, so simulated reads trace a
        realistic interferogram instead of pure noise."""
        self._sim_position_mm = float(position_mm)

    def _sim_value(self) -> float:
        x = self._sim_position_mm
        envelope = math.exp(-((x / 0.05) ** 2))          # 50 um coherence length
        fringes = math.cos(2 * math.pi * x / 0.004)      # 4 um fringe period
        return 1e-3 * envelope * fringes + random.gauss(0, 2e-6)

    def _sim_query(self, cmd: str) -> str:
        head = cmd.split("?")[0].split()[0].upper()
        if head == "*IDN":
            return "Stanford_Research_Systems,SR865A,SIMULATED,v1.00"
        if head in ("OUTP", "OUTR"):
            return f"{self._sim_value():.6e}"
        if head == "SNAP":
            v = self._sim_value()
            n = len(cmd.split("?", 1)[1].split(","))
            vals = [v, v * 0.1, abs(v), 0.0][:max(2, n)]
            return ",".join(f"{q:.6e}" for q in vals)
        if head == "SNAPD":
            v = self._sim_value()
            return ",".join(f"{q:.6e}" for q in (v, v * 0.1, abs(v), 0.0))
        if head == "OFLT":
            return "10"         # 100 ms
        if head == "SCAL":
            return "18"        # 1 uV full scale
        if head == "OFSL":
            return "1"
        if head == "IRNG":
            return "0"
        if head == "ILVL":
            return "2"
        if head in ("FREQ", "FREQINT"):
            return "1.000000e+03"
        if head in ("PHAS", "SLVL", "ENBW", "CUROVLDSTAT"):
            return "0.0"
        return "0"

    # -- settings ------------------------------------------------------------
    @property
    def time_constant(self) -> float:
        """Low-pass time constant in seconds."""
        return TIME_CONSTANTS[self.query_int("OFLT?")]

    @time_constant.setter
    def time_constant(self, seconds: float) -> None:
        self.send(f"OFLT {_nearest_index(TIME_CONSTANTS, float(seconds))}")

    @property
    def sensitivity(self) -> float:
        """Full-scale sensitivity in volts (or amps in current mode)."""
        return VOLTAGE_SENSITIVITIES[self.query_int("SCAL?")]

    @sensitivity.setter
    def sensitivity(self, volts: float) -> None:
        self.send(f"SCAL {_nearest_index(VOLTAGE_SENSITIVITIES, float(volts))}")

    @property
    def filter_slope(self) -> int:
        """Low-pass filter roll-off in dB/oct."""
        return FILTER_SLOPES[self.query_int("OFSL?")]

    @filter_slope.setter
    def filter_slope(self, db_per_oct: int) -> None:
        idx = min(range(len(FILTER_SLOPES)),
                  key=lambda i: abs(FILTER_SLOPES[i] - int(db_per_oct)))
        self.send(f"OFSL {idx}")

    @property
    def phase(self) -> float:
        """Reference phase shift in degrees."""
        return self.query_float("PHAS?")

    @phase.setter
    def phase(self, degrees: float) -> None:
        self.send(f"PHAS {float(degrees):.6f}")

    @property
    def frequency(self) -> float:
        """Reference frequency in Hz (internal or measured external)."""
        return self.query_float("FREQ?")

    @property
    def signal_strength(self) -> int:
        """Front-panel input-level indicator, 0 (low) .. 4 (overload)."""
        return self.query_int("ILVL?")

    @property
    def equivalent_noise_bandwidth(self) -> float:
        return self.query_float("ENBW?")

    def is_overloaded(self) -> bool:
        """True if any input/filter/output overload latch is set."""
        try:
            return self.query_int("CUROVLDSTAT?") != 0
        except Exception:  # noqa: BLE001
            return False

    def auto_phase(self) -> None:
        self.send("APHS")

    def auto_range(self) -> None:
        self.send("ARNG")

    def auto_scale(self) -> None:
        self.send("ASCL")

    def settings_snapshot(self) -> dict:
        """Everything worth saving alongside a scan. Never raises."""
        out = {}
        for key, fn in (
            ("identity", lambda: self.identity),
            ("time_constant_s", lambda: self.time_constant),
            ("sensitivity_V", lambda: self.sensitivity),
            ("filter_slope_dB_oct", lambda: self.filter_slope),
            ("phase_deg", lambda: self.phase),
            ("frequency_Hz", lambda: self.frequency),
            ("enbw_Hz", lambda: self.equivalent_noise_bandwidth),
        ):
            try:
                out[key] = fn()
            except Exception:  # noqa: BLE001
                out[key] = None
        return out

    # -- reading -------------------------------------------------------------
    def read_value(self, parameter: str = "R") -> float:
        """Read one output parameter (`OUTP?`): 'X', 'Y', 'R', 'Theta', ..."""
        idx = OUTPUT_PARAMETERS[parameter]
        return self.query_float(f"OUTP? {idx}")

    def read_display(self, channel: str = "Data 1") -> float:
        """Read one front-panel display data channel (`OUTR?`): 'Data 1'..'Data 4'."""
        idx = DISPLAY_CHANNELS[channel]
        return self.query_float(f"OUTR? {idx}")

    def read_channel(self, name: str) -> float:
        """Read by name, accepting an output parameter OR a display channel."""
        if name in DISPLAY_CHANNELS:
            return self.read_display(name)
        return self.read_value(name)

    def snap(self, *parameters: str) -> list[float]:
        """Read 2 or 3 output parameters simultaneously (`SNAP?`).

        A SNAP is atomic: the values come from the same instant, which a series
        of separate OUTP? queries cannot guarantee.
        """
        if not 2 <= len(parameters) <= 3:
            raise ValueError("SNAP? takes 2 or 3 parameters")
        idx = ",".join(str(OUTPUT_PARAMETERS[p]) for p in parameters)
        return [float(v) for v in self.query(f"SNAP? {idx}").split(",")]

    def snap_displays(self) -> list[float]:
        """All four display data channels at once (`SNAPD?`)."""
        return [float(v) for v in self.query("SNAPD?").split(",")]

    def read_average(self, name: str, samples: int = 1,
                     interval_s: float = 0.0) -> tuple[float, float]:
        """Average `samples` reads of `name`, `interval_s` apart.

        Returns (mean, standard deviation). The spread is the honest error bar
        for the point and is saved with the scan.
        """
        n = max(1, int(samples))
        vals = []
        for k in range(n):
            if k and interval_s > 0:
                time.sleep(interval_s)
            vals.append(self.read_channel(name))
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return mean, 0.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return mean, math.sqrt(var)

    @property
    def last_error(self) -> str:
        return self._last_error


def channel_names() -> list[str]:
    """Every readable channel name, output parameters first."""
    return list(OUTPUT_PARAMETERS) + list(DISPLAY_CHANNELS)


def channel_unit(name: str) -> str:
    """Axis unit for a channel name ('' when the display channel is free-form)."""
    return PARAMETER_UNITS.get(name, "")


if __name__ == "__main__":
    li = LockInSR860()
    li.connect("sim", "")
    print(li.identity)
    for pos in (0.0, 0.001, 0.002):
        li.set_sim_position(pos)
        print(f"{pos:.4f} mm -> {li.read_channel('R'):.6e} V")
    li.disconnect()
