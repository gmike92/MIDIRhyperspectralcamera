"""
mcs2_stage.py -- SmarAct MCS2 closed-loop translation-stage driver.

Drives one channel of an MCS2 controller through SmarAct's `smaract.ctl`
Python API (SmarActCTL.dll). Used by the lock-in acquisition app to step the
delay/translation stage while the SR865A reads the demodulated signal.

Hardware-only (NO GUI). A `simulate=True` mode lets the scan logic be exercised
with no controller attached.

Units
    The MCS2 API works in PICOMETRES (pm) for linear positioners; this class
    exposes millimetres everywhere (1 mm = 1e9 pm), matching the other stage
    drivers in this repo (twins_stage.py, stage_driver.py).

Position holding
    After a closed-loop move the controller keeps servoing the positioner only
    for `hold_time_ms`. A step scan dwells at each point for as long as the
    lock-in needs (settle + averaging), which can be seconds, so the default is
    an INFINITE hold: the position is actively held until the next move or
    until `stop()` / `disconnect()`. Set `hold_time_ms` to a finite value (or 0)
    if the servo noise matters more than the position drift.

Usage:
    from instruments.mcs2_stage import MCS2Stage
    st = MCS2Stage(channel=0)
    st.connect(simulate=True)        # or connect() with the real controller
    st.move_to(1.5); st.wait_for_stop()
    print(st.get_position(), "mm")
    st.disconnect()

Install (when hardware is here): the MCS2 SDK ships the wheel at
    C:\\SmarAct\\MCS2\\SDK\\Python\\packages\\smaract_ctl-*.zip
"""
from __future__ import annotations

import time
from typing import Optional

PM_PER_MM = 1_000_000_000      # picometres per millimetre (MCS2 linear unit)

# Channel-state bits we care about, resolved lazily from smaract.ctl so this
# module imports cleanly with no SDK present.
_ctl = None


def _load_ctl():
    """Import smaract.ctl once; returns the module or None if unavailable."""
    global _ctl
    if _ctl is None:
        try:
            import smaract.ctl as ctl  # noqa: PLC0415
            _ctl = ctl
        except Exception as e:  # noqa: BLE001
            print(f"[MCS2Stage] smaract.ctl unavailable: {e}")
            _ctl = False
    return _ctl or None


class MCS2Stage:
    """One closed-loop channel of a SmarAct MCS2 controller, in millimetres."""

    def __init__(self, channel: int = 0, locator: Optional[str] = None) -> None:
        self.channel = int(channel)
        self.locator = locator          # e.g. "usb:sn:MCS2-00001234"; None = auto
        self.is_connected = False
        self.backend = None             # 'mcs2' | 'sim'
        self.handle = None
        self.ctl = None
        # Motion profile. MCS2 takes pm/s and pm/s^2; these are the mm-flavoured
        # values written on connect (and whenever they are changed).
        self.velocity_mm_s = 1.0
        self.acceleration_mm_s2 = 10.0
        self.hold_time_ms = None        # None = infinite hold (see module docstring)
        self.max_cl_frequency_hz = 6000
        self._sim_pos_mm = 0.0
        self._sim_target_mm = 0.0
        self._last_error = ""

    # -- connection ----------------------------------------------------------
    @staticmethod
    def find_devices() -> list[str]:
        """Locator strings of every MCS2 controller reachable from this PC."""
        ctl = _load_ctl()
        if ctl is None:
            return []
        try:
            buffer = ctl.FindDevices()
        except Exception as e:  # noqa: BLE001
            print(f"[MCS2Stage] FindDevices failed: {e}")
            return []
        return [loc for loc in buffer.split("\n") if loc.strip()]

    def connect(self, simulate: bool = False, locator: Optional[str] = None,
                channel: Optional[int] = None) -> bool:
        if self.is_connected:
            return True
        if channel is not None:
            self.channel = int(channel)
        if locator:
            self.locator = locator

        if simulate:
            self.backend = "sim"
            self.is_connected = True
            self._sim_pos_mm = self._sim_target_mm = 0.0
            print("[MCS2Stage] connected (SIMULATED)")
            return True

        ctl = _load_ctl()
        if ctl is None:
            self._last_error = "smaract.ctl not installed"
            print("[MCS2Stage] smaract.ctl not installed. Use connect(simulate=True).")
            return False

        loc = self.locator
        if not loc:
            found = self.find_devices()
            if not found:
                self._last_error = "no MCS2 devices found"
                print("[MCS2Stage] no MCS2 devices found.")
                return False
            loc = found[0]
        try:
            self.handle = ctl.Open(loc)
        except Exception as e:  # noqa: BLE001
            self._last_error = str(e)
            print(f"[MCS2Stage] open failed ({loc}): {e}")
            return False

        self.ctl = ctl
        self.locator = loc
        self.backend = "mcs2"
        self.is_connected = True
        try:
            self._configure_channel()
        except Exception as e:  # noqa: BLE001
            print(f"[MCS2Stage] channel setup warning: {e}")
        print(f"[MCS2Stage] connected to {loc} channel {self.channel}")
        return True

    def _configure_channel(self) -> None:
        """Put the channel into closed-loop absolute mode with a sane profile."""
        ctl, h, ch = self.ctl, self.handle, self.channel
        ch_type = ctl.GetProperty_i32(h, ch, ctl.Property.CHANNEL_TYPE)
        if ch_type == ctl.ChannelModuleType.STICK_SLIP_PIEZO_DRIVER:
            # maxCLF is not persistent across power cycles -- set it every time.
            ctl.SetProperty_i32(h, ch, ctl.Property.MAX_CL_FREQUENCY,
                                int(self.max_cl_frequency_hz))
        elif ch_type in (ctl.ChannelModuleType.PIEZO_SCANNER_DRIVER,
                         ctl.ChannelModuleType.MAGNETIC_DRIVER):
            ctl.SetProperty_i32(h, ch, ctl.Property.AMPLIFIER_ENABLED, ctl.TRUE)
        self._apply_hold_time()
        ctl.SetProperty_i32(h, ch, ctl.Property.MOVE_MODE, ctl.MoveMode.CL_ABSOLUTE)
        self._apply_profile()

    def _apply_hold_time(self) -> None:
        ctl, h, ch = self.ctl, self.handle, self.channel
        hold = (ctl.HOLD_TIME_INFINITE if self.hold_time_ms is None
                else int(self.hold_time_ms))
        ctl.SetProperty_i32(h, ch, ctl.Property.HOLD_TIME, hold)

    def _apply_profile(self) -> None:
        ctl, h, ch = self.ctl, self.handle, self.channel
        ctl.SetProperty_i64(h, ch, ctl.Property.MOVE_VELOCITY,
                            int(self.velocity_mm_s * PM_PER_MM))
        ctl.SetProperty_i64(h, ch, ctl.Property.MOVE_ACCELERATION,
                            int(self.acceleration_mm_s2 * PM_PER_MM))

    def set_profile(self, velocity_mm_s: Optional[float] = None,
                    acceleration_mm_s2: Optional[float] = None) -> None:
        if velocity_mm_s is not None:
            self.velocity_mm_s = float(velocity_mm_s)
        if acceleration_mm_s2 is not None:
            self.acceleration_mm_s2 = float(acceleration_mm_s2)
        if self.backend == "mcs2":
            try:
                self._apply_profile()
            except Exception as e:  # noqa: BLE001
                print(f"[MCS2Stage] profile write failed: {e}")

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        if self.backend == "mcs2":
            try:
                # Release the (possibly infinite) position hold before closing,
                # otherwise the positioner keeps servoing with nobody watching.
                self.ctl.Stop(self.handle, self.channel)
                self.ctl.Close(self.handle)
            except Exception as e:  # noqa: BLE001
                print(f"[MCS2Stage] disconnect warning: {e}")
        self.handle = None
        self.is_connected = False
        print("[MCS2Stage] disconnected")

    # -- device info ---------------------------------------------------------
    def describe(self) -> str:
        """One-line identification of the connected controller/positioner."""
        if not self.is_connected:
            return "offline"
        if self.backend == "sim":
            return "simulated MCS2"
        try:
            name = self.ctl.GetProperty_s(self.handle, 0, self.ctl.Property.DEVICE_NAME)
            serial = self.ctl.GetProperty_s(self.handle, 0,
                                            self.ctl.Property.DEVICE_SERIAL_NUMBER)
            ptype = self.ctl.GetProperty_s(self.handle, self.channel,
                                           self.ctl.Property.POSITIONER_TYPE_NAME)
            return f"{name} ({serial}) ch{self.channel}: {ptype}"
        except Exception as e:  # noqa: BLE001
            return f"MCS2 ch{self.channel} (info unavailable: {e})"

    def channel_count(self) -> int:
        if not self.is_connected or self.backend == "sim":
            return 1
        try:
            return int(self.ctl.GetProperty_i32(self.handle, 0,
                                                self.ctl.Property.NUMBER_OF_CHANNELS))
        except Exception:  # noqa: BLE001
            return 1

    def _state(self) -> int:
        return int(self.ctl.GetProperty_i32(self.handle, self.channel,
                                            self.ctl.Property.CHANNEL_STATE))

    @property
    def is_referenced(self) -> bool:
        if not self.is_connected:
            return False
        if self.backend == "sim":
            return True
        try:
            return bool(self._state() & self.ctl.ChannelState.IS_REFERENCED)
        except Exception:  # noqa: BLE001
            return False

    @property
    def has_sensor(self) -> bool:
        if not self.is_connected:
            return False
        if self.backend == "sim":
            return True
        try:
            return bool(self._state() & self.ctl.ChannelState.SENSOR_PRESENT)
        except Exception:  # noqa: BLE001
            return False

    # -- referencing / calibration ------------------------------------------
    def find_reference(self, wait: bool = True, timeout_s: float = 60.0) -> bool:
        """Establish the absolute zero. The sensors are incremental, so this is
        needed once per power-up before absolute positions mean anything."""
        if not self.is_connected:
            return False
        if self.backend == "sim":
            self._sim_pos_mm = self._sim_target_mm = 0.0
            return True
        try:
            ctl, h, ch = self.ctl, self.handle, self.channel
            ctl.SetProperty_i32(h, ch, ctl.Property.REFERENCING_OPTIONS, 0)
            self._apply_profile()
            ctl.Reference(h, ch)
        except Exception as e:  # noqa: BLE001
            self._last_error = str(e)
            print(f"[MCS2Stage] reference failed: {e}")
            return False
        if wait:
            return self.wait_for_stop(timeout_s=timeout_s)
        return True

    def calibrate(self, wait: bool = True, timeout_s: float = 120.0) -> bool:
        """Run the sensor calibration sequence (moves up to a few mm!).

        Only needed after the mechanical setup or positioner type changes; the
        result is stored in the controller's non-volatile memory.
        """
        if not self.is_connected:
            return False
        if self.backend == "sim":
            return True
        try:
            ctl, h, ch = self.ctl, self.handle, self.channel
            ctl.SetProperty_i32(h, ch, ctl.Property.CALIBRATION_OPTIONS, 0)
            ctl.Calibrate(h, ch)
        except Exception as e:  # noqa: BLE001
            self._last_error = str(e)
            print(f"[MCS2Stage] calibrate failed: {e}")
            return False
        if wait:
            return self.wait_for_stop(timeout_s=timeout_s)
        return True

    # -- motion --------------------------------------------------------------
    def move_to(self, position_mm: float) -> bool:
        """Start a closed-loop absolute move. Returns immediately (non-blocking);
        call wait_for_stop() to block until the target is reached."""
        if not self.is_connected:
            return False
        if self.backend == "sim":
            self._sim_target_mm = float(position_mm)
            self._sim_pos_mm = float(position_mm)
            return True
        try:
            ctl, h, ch = self.ctl, self.handle, self.channel
            ctl.SetProperty_i32(h, ch, ctl.Property.MOVE_MODE, ctl.MoveMode.CL_ABSOLUTE)
            ctl.Move(h, ch, int(round(position_mm * PM_PER_MM)), 0)
            return True
        except Exception as e:  # noqa: BLE001
            self._last_error = str(e)
            print(f"[MCS2Stage] move failed: {e}")
            return False

    def move_by(self, delta_mm: float) -> bool:
        if not self.is_connected:
            return False
        if self.backend == "sim":
            return self.move_to(self._sim_pos_mm + float(delta_mm))
        try:
            ctl, h, ch = self.ctl, self.handle, self.channel
            ctl.SetProperty_i32(h, ch, ctl.Property.MOVE_MODE, ctl.MoveMode.CL_RELATIVE)
            ctl.Move(h, ch, int(round(delta_mm * PM_PER_MM)), 0)
            return True
        except Exception as e:  # noqa: BLE001
            self._last_error = str(e)
            print(f"[MCS2Stage] relative move failed: {e}")
            return False

    def stop(self) -> None:
        if not self.is_connected or self.backend == "sim":
            return
        try:
            self.ctl.Stop(self.handle, self.channel)
        except Exception as e:  # noqa: BLE001
            print(f"[MCS2Stage] stop warning: {e}")

    def get_position(self) -> float:
        """Current position in mm (live sensor read)."""
        if not self.is_connected:
            return float("nan")
        if self.backend == "sim":
            return self._sim_pos_mm
        try:
            raw = self.ctl.GetProperty_i64(self.handle, self.channel,
                                           self.ctl.Property.POSITION)
            return float(raw) / PM_PER_MM
        except Exception as e:  # noqa: BLE001
            print(f"[MCS2Stage] position read warning: {e}")
            return float("nan")

    def get_limits_mm(self) -> tuple[float, float]:
        """Configured software range limits (min, max) in mm. (0, 0) = disabled."""
        if not self.is_connected or self.backend == "sim":
            return (-100.0, 100.0)
        try:
            lo = self.ctl.GetProperty_i64(self.handle, self.channel,
                                          self.ctl.Property.RANGE_LIMIT_MIN)
            hi = self.ctl.GetProperty_i64(self.handle, self.channel,
                                          self.ctl.Property.RANGE_LIMIT_MAX)
            return (float(lo) / PM_PER_MM, float(hi) / PM_PER_MM)
        except Exception:  # noqa: BLE001
            return (-100.0, 100.0)

    def is_moving(self) -> bool:
        if not self.is_connected or self.backend == "sim":
            return False
        try:
            st = self._state()
            cs = self.ctl.ChannelState
            return bool(st & (cs.ACTIVELY_MOVING | cs.REFERENCING | cs.CALIBRATING))
        except Exception:  # noqa: BLE001
            return False

    def wait_for_stop(self, timeout_s: float = 30.0, poll_s: float = 0.005) -> bool:
        """Block until the channel finishes moving. False on timeout/failure.

        Note that MCS2 movement commands are asynchronous: `Move` returns before
        the positioner has even started. The controller sets ACTIVELY_MOVING
        synchronously with the command, so unlike the SCU3D there is no need to
        first wait for motion to *begin*.
        """
        if not self.is_connected or self.backend == "sim":
            return True
        deadline = time.time() + timeout_s
        cs = self.ctl.ChannelState
        while True:
            try:
                st = self._state()
            except Exception as e:  # noqa: BLE001
                print(f"[MCS2Stage] state read failed: {e}")
                return False
            if st & cs.MOVEMENT_FAILED:
                reason = []
                if st & cs.END_STOP_REACHED:
                    reason.append("end stop reached")
                if st & cs.RANGE_LIMIT_REACHED:
                    reason.append("range limit reached")
                if st & cs.FOLLOWING_LIMIT_REACHED:
                    reason.append("following limit reached")
                self._last_error = "movement failed" + (
                    f" ({', '.join(reason)})" if reason else "")
                print(f"[MCS2Stage] {self._last_error}")
                return False
            if not (st & (cs.ACTIVELY_MOVING | cs.REFERENCING | cs.CALIBRATING)):
                return True
            if time.time() > deadline:
                self._last_error = "motion timeout"
                print("[MCS2Stage] motion timeout")
                return False
            time.sleep(poll_s)

    @property
    def last_error(self) -> str:
        return self._last_error


if __name__ == "__main__":
    print("MCS2 devices:", MCS2Stage.find_devices())
    st = MCS2Stage()
    st.connect(simulate=True)
    st.move_to(1.234); st.wait_for_stop()
    print("pos:", st.get_position(), "mm")
    st.disconnect()
