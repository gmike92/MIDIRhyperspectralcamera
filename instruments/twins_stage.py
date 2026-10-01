"""
twins_stage.py -- TWINS interferometer wedge stage, SmarAct MCS2 backend.

This is the motorized wedge stage *inside* the NIREOS TWINS interferometer.
Stepping it changes the birefringent optical path difference (the interferometric
delay); subtwinslv.py scans it and the app Fourier-transforms the per-pixel
interferogram into a spectrum.

Hardware here is a SmarAct **MCS2** controller driving a closed-loop linear
positioner (SLC-2460 with sensor module), via SmarAct's MCS2 SDK
(`import smaract.ctl as ctl`, SmarActCTL.dll). Positions are in **millimetres**
at this class's API; the MCS2 works in **picometres** internally.

This module is hardware-only (NO GUI) and keeps the SAME public API the rest of
the app already calls, so the scan engine (subtwinslv.py) and the stage UI
(ui/stages.py) need no changes:

    connect(simulate=False, home=True) -> bool
    disconnect(safe=True) -> None
    move_to(position_mm) -> bool          # closed-loop absolute, mm
    move_by(delta_mm) -> bool             # relative, mm
    get_position() -> float               # mm
    is_moving() -> bool
    wait_for_stop(timeout_s=30.0) -> bool
    # attributes: is_connected (bool), backend ('mcs2' | 'sim')

`simulate=True` gives a full software stage so the whole TWINS scan / DSP path
runs with no controller present.

Usage:
    from instruments.twins_stage import TwinsStage
    st = TwinsStage()
    st.connect(simulate=True)             # or connect() with the real MCS2
    st.move_to(24.0); st.wait_for_stop()
    print(st.get_position())
    st.disconnect()
"""
from __future__ import annotations

import time
from typing import Optional

# -- unit / geometry constants ------------------------------------------------
PM_PER_MM = 1_000_000_000        # picometres per millimetre (MCS2 linear units)

# -- travel limits (mm) -------------------------------------------------------
# The MCS2 does NOT know the usable travel of this wedge: RANGE_LIMIT_MIN/MAX
# and DEFAULT_RANGE_LIMIT_MIN/MAX all read 0 on this controller, and the manual
# states no range limit is set by default (a limit exists only once MAX is set
# above MIN). So the span is defined HERE and checked in software before every
# move -- the positioner would otherwise happily drive into its end stop.
#
# Positions are in the CURRENT referenced coordinates (they can be negative).
# Measure the real end points on the rig and tune these to it.
TRAVEL_MIN_MM = -16.5           # most negative position a move may target
TRAVEL_MAX_MM = 18.8            # most positive position a move may target

# TWINS wedge park positions (mm). These are interferometer-specific -- tune
# HOME/SAFE and the scan range to YOUR TWINS unit and its ZPD (see the Measure
# tab defaults and instruments/calibration.py). Both must lie inside the travel
# limits above; _check_travel_constants() verifies that at import.
HOME_POSITION_MM = 0.0          # parked/working position after referencing
SAFE_POSITION_MM = 0.0          # parked position on disconnect

# -- MCS2 motion defaults (converted to pm at connect) ------------------------
DEFAULT_VELOCITY_MM_S = 5.0      # closed-loop move velocity
DEFAULT_ACCEL_MM_S2 = 20.0       # closed-loop move acceleration
DEFAULT_HOLD_TIME_MS = 1000      # hold target position after a move (ms)
DEFAULT_CHANNEL = 0              # MCS2 channel index of the wedge positioner


def travel_range_mm() -> tuple:
    """(min_mm, max_mm) the wedge may be commanded to -- the software limits."""
    return (TRAVEL_MIN_MM, TRAVEL_MAX_MM)


def in_travel_range(position_mm: float) -> bool:
    """True when `position_mm` is a legal move target."""
    return TRAVEL_MIN_MM <= float(position_mm) <= TRAVEL_MAX_MM


def _check_travel_constants() -> None:
    """The park positions must be reachable, or connect/disconnect would refuse
    to move. Caught at import instead of mid-experiment."""
    if TRAVEL_MAX_MM <= TRAVEL_MIN_MM:
        raise ValueError(f"TRAVEL_MAX_MM ({TRAVEL_MAX_MM}) must exceed "
                         f"TRAVEL_MIN_MM ({TRAVEL_MIN_MM})")
    for name, value in (("HOME_POSITION_MM", HOME_POSITION_MM),
                        ("SAFE_POSITION_MM", SAFE_POSITION_MM)):
        if not in_travel_range(value):
            raise ValueError(f"{name} ({value} mm) is outside the travel limits "
                             f"{TRAVEL_MIN_MM} .. {TRAVEL_MAX_MM} mm")


_check_travel_constants()


class TwinsStage:
    """SmarAct MCS2 single-channel closed-loop stage (the TWINS wedge translator)."""

    def __init__(self) -> None:
        self.ctl = None                # the smaract.ctl module (loaded on connect)
        self.handle = None             # MCS2 device handle
        self.locator = None            # e.g. "usb:sn:MCS2-00000123"
        self.channel = DEFAULT_CHANNEL
        self.is_connected = False
        self.backend = None            # 'mcs2' | 'sim'
        self._position_mm = 0.0        # cached / simulated position

    # -- connection ----------------------------------------------------------
    def connect(self, locator: Optional[str] = None, simulate: bool = False,
                home: bool = True, channel: Optional[int] = None,
                dll_path: Optional[str] = None, reference: bool = True) -> bool:
        """Open the MCS2 (or a simulated stage). `locator` may be an explicit
        MCS2 device string; if None the first discovered device is used.

        `reference=False` opens the controller WITHOUT referencing it -- nothing
        moves, so the reference state can be read (`is_referenced()`). The
        `home` park move is then skipped too unless the axis already holds its
        reference, since an absolute move is only meaningful once referenced."""
        if self.is_connected:
            return True
        if channel is not None:
            self.channel = int(channel)

        if simulate:
            self.backend = "sim"
            self.is_connected = True
            self._position_mm = HOME_POSITION_MM
            print("[TwinsStage] connected (SIMULATED)")
            return True

        try:
            import smaract.ctl as ctl
        except Exception as exc:  # noqa: BLE001
            print(f"[TwinsStage] SmarAct MCS2 SDK (smaract.ctl) not available: {exc}\n"
                  "             Install the MCS2 SDK or use connect(simulate=True).")
            return False
        self.ctl = ctl

        try:
            self.locator = locator or self._find_device()
            if not self.locator:
                print("[TwinsStage] no MCS2 device found. Use connect(simulate=True).")
                return False
            self.handle = ctl.Open(self.locator)
        except Exception as exc:  # noqa: BLE001
            print(f"[TwinsStage] MCS2 open failed ({self.locator}): {exc}")
            return False

        self.backend = "mcs2"
        self.is_connected = True
        try:
            self._configure_channel()
            if reference:
                self._ensure_referenced()
            if home and self.is_referenced():
                self.move_to(HOME_POSITION_MM)
                self.wait_for_stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[TwinsStage] MCS2 configuration warning: {exc}")
        print(f"[TwinsStage] connected via MCS2 ({self.locator}, channel {self.channel})")
        return True

    def _find_device(self) -> Optional[str]:
        """Return the first MCS2 locator string, or None."""
        try:
            buffer = self.ctl.FindDevices()
        except Exception as exc:  # noqa: BLE001
            print(f"[TwinsStage] FindDevices failed: {exc}")
            return None
        locators = [s for s in str(buffer).replace("\r", "\n").split("\n") if s.strip()]
        return locators[0] if locators else None

    def _configure_channel(self) -> None:
        """Set closed-loop move mode, velocity, acceleration and hold time."""
        ctl = self.ctl
        ch = self.channel
        ctl.SetProperty_i32(self.handle, ch, ctl.Property.MOVE_MODE,
                            ctl.MoveMode.CL_ABSOLUTE)
        ctl.SetProperty_i64(self.handle, ch, ctl.Property.MOVE_VELOCITY,
                            int(DEFAULT_VELOCITY_MM_S * PM_PER_MM))
        ctl.SetProperty_i64(self.handle, ch, ctl.Property.MOVE_ACCELERATION,
                            int(DEFAULT_ACCEL_MM_S2 * PM_PER_MM))
        ctl.SetProperty_i32(self.handle, ch, ctl.Property.HOLD_TIME,
                            int(DEFAULT_HOLD_TIME_MS))

    def is_referenced(self) -> bool:
        """True when the positioner holds a valid reference (absolute positions
        are meaningful). Pure property read -- never moves the stage."""
        if not self.is_connected:
            return False
        if self.backend == "sim":
            return True
        try:
            state = self.ctl.GetProperty_i32(self.handle, self.channel,
                                             self.ctl.Property.CHANNEL_STATE)
            return bool(state & self.ctl.ChannelState.IS_REFERENCED)
        except Exception:  # noqa: BLE001
            return False

    def _ensure_referenced(self) -> None:
        """Reference the positioner so closed-loop absolute positions are valid."""
        if self.is_referenced():
            return
        ctl = self.ctl
        ch = self.channel
        print("[TwinsStage] referencing MCS2 positioner...")
        try:
            # Reference at moderate speed; options=0 keeps the controller default.
            ctl.SetProperty_i32(self.handle, ch, ctl.Property.REFERENCING_OPTIONS, 0)
            ctl.SetProperty_i64(self.handle, ch, ctl.Property.MOVE_VELOCITY,
                                int(DEFAULT_VELOCITY_MM_S * PM_PER_MM))
            ctl.Reference(self.handle, ch)
            self.wait_for_stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[TwinsStage] referencing warning: {exc}")

    def disconnect(self, safe: bool = True) -> None:
        if not self.is_connected:
            return
        if self.backend == "mcs2":
            try:
                if safe:
                    self.move_to(SAFE_POSITION_MM)
                    self.wait_for_stop()
                if self.handle is not None:
                    self.ctl.Close(self.handle)
            except Exception as exc:  # noqa: BLE001
                print(f"[TwinsStage] disconnect warning: {exc}")
        self.handle = None
        self.ctl = None
        self.is_connected = False
        print("[TwinsStage] disconnected")

    # -- motion --------------------------------------------------------------
    def move_to(self, position_mm: float) -> bool:
        """Closed-loop absolute move to `position_mm` (mm).

        Refused (returns False, nothing is commanded) when the target lies
        outside TRAVEL_MIN_MM .. TRAVEL_MAX_MM -- the controller has no range
        limit of its own, so this is the only thing standing between a typo and
        the end stop."""
        if not self.is_connected:
            return False
        if not in_travel_range(position_mm):
            print(f"[TwinsStage] REFUSED move to {float(position_mm):.4f} mm: "
                  f"outside travel {TRAVEL_MIN_MM} .. {TRAVEL_MAX_MM} mm")
            return False
        if self.backend == "sim":
            self._position_mm = float(position_mm)
            return True
        try:
            ctl = self.ctl
            ch = self.channel
            ctl.SetProperty_i32(self.handle, ch, ctl.Property.MOVE_MODE,
                                ctl.MoveMode.CL_ABSOLUTE)
            ctl.Move(self.handle, ch, int(round(position_mm * PM_PER_MM)), 0)
            self._position_mm = float(position_mm)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[TwinsStage] move failed: {exc}")
            return False

    def move_by(self, delta_mm: float) -> bool:
        """Relative move by `delta_mm` (mm).

        Checked against the travel limits using the stage's CURRENT position, so
        jogging cannot walk the wedge past an end point one step at a time."""
        if not self.is_connected:
            return False
        target = self.get_position() + float(delta_mm)
        if not in_travel_range(target):
            print(f"[TwinsStage] REFUSED jog of {float(delta_mm):+.4f} mm: "
                  f"target {target:.4f} mm is outside travel "
                  f"{TRAVEL_MIN_MM} .. {TRAVEL_MAX_MM} mm")
            return False
        if self.backend == "sim":
            self._position_mm += float(delta_mm)
            return True
        # Use a closed-loop relative move so it is independent of the cached value.
        try:
            ctl = self.ctl
            ch = self.channel
            ctl.SetProperty_i32(self.handle, ch, ctl.Property.MOVE_MODE,
                                ctl.MoveMode.CL_RELATIVE)
            ctl.Move(self.handle, ch, int(round(delta_mm * PM_PER_MM)), 0)
            self._position_mm += float(delta_mm)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[TwinsStage] relative move failed: {exc}")
            return False

    def get_position(self) -> float:
        """Current position in mm (read back from the sensor when on hardware)."""
        if not self.is_connected or self.backend == "sim":
            return self._position_mm
        try:
            pos_pm = self.ctl.GetProperty_i64(self.handle, self.channel,
                                              self.ctl.Property.POSITION)
            self._position_mm = pos_pm / PM_PER_MM
        except Exception as exc:  # noqa: BLE001
            print(f"[TwinsStage] position read warning: {exc}")
        return self._position_mm

    def is_moving(self) -> bool:
        if not self.is_connected or self.backend == "sim":
            return False
        try:
            state = self.ctl.GetProperty_i32(self.handle, self.channel,
                                             self.ctl.Property.CHANNEL_STATE)
            return bool(state & self.ctl.ChannelState.ACTIVELY_MOVING)
        except Exception:  # noqa: BLE001
            return False

    def stop(self) -> None:
        if self.is_connected and self.backend == "mcs2":
            try:
                self.ctl.Stop(self.handle, self.channel)
            except Exception:  # noqa: BLE001
                pass

    def wait_for_stop(self, timeout_s: float = 30.0) -> bool:
        """Wait for a move to begin, then settle. Same semantics as before so the
        scan only records once the wedge is stopped. The MCS2 CHANNEL_STATE
        ACTIVELY_MOVING bit is reliable, but we keep the 'wait to begin' guard so
        a status read immediately after Move() can't return instantly."""
        if not self.is_connected or self.backend == "sim":
            return True
        t0 = time.time()
        while not self.is_moving():
            if time.time() - t0 > 0.5:
                break
            time.sleep(0.005)
        while self.is_moving():
            if time.time() - t0 > timeout_s:
                print("[TwinsStage] motion timeout")
                return False
            time.sleep(0.01)
        return True


if __name__ == "__main__":
    st = TwinsStage()
    sim = False

    if sim:
        # Simulation
        if st.connect(simulate=True):
            st.move_to(24.0); st.wait_for_stop()
            print("pos:", st.get_position(), "mm")
            st.disconnect(safe=False)
    else:
        # Real device -- READ ONLY: open the controller without referencing it
        # and without the home/safe park moves, so nothing on the stage moves.
        if not st.connect(home=False, reference=False):
            raise SystemExit("Smaract stage NOT connected")
        print("Smaract stage connected")
        referenced = st.is_referenced()
        print("referenced:", referenced)
        print("travel limits:", f"{TRAVEL_MIN_MM:.4f} .. {TRAVEL_MAX_MM:.4f} mm "
              f"(software, set in this file)")
        print("position:", f"{st.get_position():.6f} mm"
              + ("" if referenced else "  (not referenced -- value is meaningless)"))
        st.disconnect(safe=False)
