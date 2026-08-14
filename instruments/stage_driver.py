"""
stage_driver.py -- Thorlabs delay-line stage driver (pump-probe delay).

System 1 of the pump-probe setup: a Thorlabs Kinesis/APT translation stage on
a retroreflector that sets the pump-probe time delay.

Decoupled, hardware-only (NO GUI). Hardware libraries are imported lazily so
this module imports cleanly even with nothing installed/connected. A built-in
`simulate=True` mode lets you develop and test the scan logic now, before the
stage is wired up.

Delay-line physics (double pass on a retroreflector):
    c = 0.000299792458 mm/fs
    delay (fs) = 2 * x (mm) / c          (double pass)
    x (mm)     = delay (fs) * c / 2

Backends tried, in order (first that works wins):
    1. pylablib  -> Thorlabs.KinesisMotor
    2. thorlabs_apt
    3. pythonnet  -> Thorlabs.MotionControl Kinesis .NET DLLs
    4. simulate   (no hardware)

Usage:
    from stage_driver import DelayStage
    s = DelayStage()
    s.connect(simulate=True)         # or connect() for real hardware
    s.home()
    s.move_to_fs(500)                # go to +500 fs
    print(s.get_position_fs(), "fs")
    s.disconnect()

Install (when hardware is here): `pip install pylablib`  and/or Thorlabs
Kinesis (https://www.thorlabs.com/kinesis). thorlabs_apt is optional.
"""
from __future__ import annotations

import time
from typing import Optional

SPEED_OF_LIGHT_MM_FS = 0.000299792458  # mm per fs

# Encoder scaling for the pylablib backend (counts per mm). Override per stage.
# NOTE: the LTS150/LTS300 (409600 counts/mm) is handled via the .NET backend
# instead, which uses real mm through LoadMotorConfiguration -- this constant
# does not apply to it.
DEFAULT_COUNTS_PER_MM = 20000


class DelayStage:
    """Thorlabs delay-line stage with mm and fs (delay) interfaces."""

    def __init__(self, counts_per_mm: int = DEFAULT_COUNTS_PER_MM,
                 double_pass: bool = True) -> None:
        self.counts_per_mm = counts_per_mm
        self.double_pass = double_pass
        self.zero_position_mm = 0.0     # mm position that corresponds to 0 fs

        self.is_connected = False
        self.is_homed = False
        # The user homes the stage (in Kinesis) BEFORE connecting and the LTS300
        # retains its home reference while powered, so on connect we TRUST it is
        # homed rather than block moves on a stale/late Status.IsHomed read. The
        # app never auto-homes. Set False to require a positive homed status.
        self.assume_homed = True
        # Set the controller's soft-limit approach policy to AllowAllMoves on
        # connect so moves outside its stock 0..300 soft limits (this stage runs
        # in a NEGATIVE coordinate range) aren't rejected with "Cannot move to
        # requested position". HARDWARE limit switches still protect the stage.
        # Set False to keep the controller's stock limit policy.
        self.override_limits = True
        # Move velocity profile. If the controller comes up with velocity=0 (a
        # common "connects + reads position but won't move" cause) MoveTo blocks
        # forever, so we set a safe profile on connect. LTS300 max is ~50 mm/s.
        self.max_velocity_mm_s = 20.0
        self.acceleration_mm_s2 = 20.0
        self.backend = None             # 'kinesis' | 'apt' | 'dotnet' | 'sim'
        self.serial = None
        self._stage = None              # underlying device object
        self._sim_pos_mm = 0.0
        self._exclude = set()           # serials to skip on auto-pick (other KDC101s)

    # -- connection ----------------------------------------------------------
    def connect(self, serial: Optional[str] = None, simulate: bool = False,
                exclude=None) -> bool:
        if self.is_connected:
            return True
        # This rig has TWO KDC101 controllers (delay + rotator); when we auto-pick
        # (no explicit serial) skip the OTHER controller's serial so the delay
        # stage never grabs the rotator's device (or vice-versa).
        self._exclude = set(str(s) for s in (exclude or []) if s)
        if simulate:
            self.backend = "sim"
            self.serial = serial or "SIM-DELAY"
            self.is_connected = True
            print("[DelayStage] connected (SIMULATED)")
            return True
        for attempt in (self._connect_kinesis, self._connect_apt, self._connect_dotnet):
            try:
                if attempt(serial):
                    self.is_connected = True
                    print(f"[DelayStage] connected via {self.backend} (serial {self.serial})")
                    return True
            except Exception as e:  # noqa: BLE001
                print(f"[DelayStage] {attempt.__name__} failed: {e}")
        print("[DelayStage] no hardware found. Use connect(simulate=True) to develop.")
        return False

    def _connect_kinesis(self, serial: Optional[str]) -> bool:
        from pylablib.devices import Thorlabs
        devices = Thorlabs.list_kinesis_devices()
        devices = [d for d in devices if str(d[0]) not in self._exclude]
        # LTS150/LTS300 (serial prefix "45") need 409600 counts/mm, not the
        # pylablib default here -- route them to the .NET path (mm via
        # LoadMotorConfiguration) so moves are correctly scaled.
        if not serial:
            devices = [d for d in devices if not str(d[0]).startswith("45")]
        if not devices:
            return False
        sn = serial or devices[0][0]
        self._stage = Thorlabs.KinesisMotor(sn)
        self.backend, self.serial = "kinesis", str(sn)
        return True

    def _connect_apt(self, serial: Optional[str]) -> bool:
        import thorlabs_apt as apt
        devices = apt.list_available_devices()
        devices = [d for d in devices if str(d[1]) not in self._exclude]
        if not devices:
            return False
        sn = int(serial) if serial else devices[0][1]
        self._stage = apt.Motor(sn)
        self.backend, self.serial = "apt", str(sn)
        return True

    def _connect_dotnet(self, serial: Optional[str]) -> bool:
        import os
        import sys
        import clr  # pythonnet
        kinesis = r"C:\Program Files\Thorlabs\Kinesis"
        if kinesis not in sys.path:
            sys.path.append(kinesis)
        try:
            os.add_dll_directory(kinesis)   # let dependent native DLLs resolve
        except Exception:  # noqa: BLE001
            pass
        clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
        clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        DeviceManagerCLI.BuildDeviceList()
        devices = [str(d) for d in DeviceManagerCLI.GetDeviceList()]
        devices = [d for d in devices if d not in self._exclude]
        if not devices:
            return False
        # Prefer an integrated-stepper long-travel stage (LTS150/LTS300, serial
        # prefix "45") for the delay axis; else the first available controller.
        sn = (str(serial) if serial
              else next((d for d in devices if d.startswith("45")), devices[0]))
        dev = self._create_dotnet_device(sn)
        dev.Connect(sn)
        dev.WaitForSettingsInitialized(5000)
        dev.StartPolling(250)
        time.sleep(0.25)
        dev.EnableDevice()
        time.sleep(0.25)
        # Load the stage calibration so MoveTo/Position use real mm.
        dev.LoadMotorConfiguration(sn)
        if self.override_limits:
            self._apply_limit_override(dev)
        self._apply_velocity(dev)
        self._stage = dev
        self.backend, self.serial = "dotnet", str(sn)
        # Read the REAL homed status (best-effort, with retries -- polling needs a
        # moment after EnableDevice) for logging, but never auto-home. With
        # assume_homed we trust the user's pre-connect homing so moves aren't
        # blocked by a stale IsHomed=False right after connect.
        raw = self._dotnet_is_homed(dev)
        self.is_homed = True if self.assume_homed else bool(raw)
        print(f"[DelayStage] dotnet {sn}: Status.IsHomed={raw}; is_homed="
              f"{self.is_homed} (assume_homed={self.assume_homed}, never auto-homes)")
        return True

    def _apply_velocity(self, dev) -> None:
        """Log the controller's velocity profile and, if it is zero/unset (=> the
        stage would never actually move), set a safe max-velocity/acceleration."""
        from System import Decimal
        cur_v = None
        try:
            vp = dev.GetVelocityParams()
            cur_v = float(Decimal.ToDouble(vp.MaxVelocity))
            cur_a = float(Decimal.ToDouble(vp.Acceleration))
            print(f"[DelayStage] velocity on connect: v={cur_v} mm/s, a={cur_a} mm/s^2")
        except Exception as e:  # noqa: BLE001
            print(f"[DelayStage] GetVelocityParams failed: {e}")
        if not cur_v:   # None or 0 -> stage won't move; set a usable profile
            try:
                dev.SetVelocityParams(Decimal(float(self.max_velocity_mm_s)),
                                      Decimal(float(self.acceleration_mm_s2)))
                print(f"[DelayStage] set velocity v={self.max_velocity_mm_s} mm/s, "
                      f"a={self.acceleration_mm_s2} mm/s^2")
            except Exception as e:  # noqa: BLE001
                print(f"[DelayStage] SetVelocityParams failed: {e}")

    def _apply_limit_override(self, dev) -> None:
        """Set the controller's SOFTWARE limit approach policy to AllowAllMoves so
        moves outside the stock soft limits aren't rejected by VerifyDeviceMovement
        ("Cannot move to requested position"). This LTS300 is parked/operated in a
        NEGATIVE coordinate range that sits outside the default 0..300 soft limits,
        so without this every move is refused. HARDWARE limit switches still
        protect the stage. The enum type is taken from the getter's own return
        value (nested LimitSettings+LimitsSoftwareApproachPolicies), so it works
        across Kinesis versions without a hard-coded namespace. Best-effort/logged."""
        from System import Enum
        try:
            cur = dev.GetLimitsSoftwareApproachPolicy()
            allow = Enum.Parse(cur.GetType(), "AllowAllMoves")
            dev.SetLimitsSoftwareApproachPolicy(allow)
            print(f"[DelayStage] soft-limit policy: {cur} -> "
                  f"{dev.GetLimitsSoftwareApproachPolicy()} (out-of-limit moves allowed)")
        except Exception as e:  # noqa: BLE001
            print(f"[DelayStage] could not set soft-limit policy: {str(e).splitlines()[0]}")

    @staticmethod
    def _dotnet_is_homed(dev):
        """Best-effort read of the .NET stage's homed flag. Polling (StartPolling)
        updates dev.Status asynchronously, so retry for ~2 s. Returns bool, or
        None if the status never became readable."""
        for _ in range(8):
            try:
                return bool(dev.Status.IsHomed)
            except Exception:  # noqa: BLE001
                time.sleep(0.25)
        return None

    @staticmethod
    def _create_dotnet_device(sn: str):
        """Create the right Kinesis .NET controller for this serial: an LTS150/
        LTS300 long-travel integrated stepper (prefix 45) vs a KDC101 K-Cube DC
        servo (prefix 27). Both expose the same GenericMotor MoveTo/Home/Position
        API, so the rest of the driver is backend-agnostic."""
        import clr  # pythonnet
        prefix = str(sn)[:2]
        if prefix == "45":   # LTS150 / LTS300 -> LongTravelStage
            clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
            from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import LongTravelStage
            try:
                return LongTravelStage.CreateLongTravelStage(sn)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"LTS {sn}: cannot create LongTravelStage ({e}). Check the "
                    "stage is powered and its actuator is assigned in Kinesis.") from e
        clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
        from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
        try:
            return KCubeDCServo.CreateKCubeDCServo(sn)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"KDC101 {sn}: cannot load device configuration ({e}). Open the "
                "Thorlabs Kinesis app, select this controller, assign the actuator "
                "(e.g. Z825B) under Settings -> Device, then retry.") from e

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        try:
            if self.backend == "kinesis" and self._stage is not None:
                self._stage.close()
            elif self.backend == "apt" and hasattr(self._stage, "_cleanup"):
                pass
            elif self.backend == "dotnet" and self._stage is not None:
                self._stage.StopPolling()
                self._stage.Disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[DelayStage] disconnect warning: {e}")
        self._stage = None
        self.is_connected = False
        self.is_homed = False
        print("[DelayStage] disconnected")

    # -- homing / motion -----------------------------------------------------
    def home(self, timeout_s: float = 60.0) -> bool:
        if not self.is_connected:
            return False
        if self.backend == "sim":
            self._sim_pos_mm = 0.0
        elif self.backend == "kinesis":
            self._stage.home(sync=True, timeout=timeout_s)
        elif self.backend == "apt":
            self._stage.move_home(blocking=True)
        elif self.backend == "dotnet":
            self._stage.Home(int(timeout_s * 1000))
        self.is_homed = True
        print("[DelayStage] homed")
        return True

    def move_to_mm(self, position_mm: float, wait: bool = True) -> bool:
        if not self._can_move():
            return False
        if self.backend == "sim":
            self._sim_pos_mm = position_mm
        elif self.backend == "kinesis":
            self._stage.move_to(position_mm * self.counts_per_mm)
            if wait:
                self._stage.wait_move()
        elif self.backend == "apt":
            self._stage.move_to(float(position_mm), blocking=wait)
        elif self.backend == "dotnet":
            from System import Decimal
            before = self.get_position_mm()
            try:
                self._stage.MoveTo(Decimal(float(position_mm)), 60000 if wait else 0)
            except Exception as e:  # noqa: BLE001
                print(f"[DelayStage] MoveTo({position_mm:.4f} mm) FAILED: {e}")
                raise
            after = self.get_position_mm()
            print(f"[DelayStage] MoveTo {position_mm:.4f} mm: "
                  f"{before:.4f} -> {after:.4f} mm")
        return True

    def move_relative_mm(self, delta_mm: float, wait: bool = True) -> bool:
        return self.move_to_mm(self.get_position_mm() + delta_mm, wait)

    def move_to_fs(self, delay_fs: float, wait: bool = True) -> bool:
        return self.move_to_mm(self.zero_position_mm + self.fs_to_mm(delay_fs), wait)

    def move_relative_fs(self, delta_fs: float, wait: bool = True) -> bool:
        return self.move_relative_mm(self.fs_to_mm(delta_fs), wait)

    def get_position_mm(self) -> float:
        if not self.is_connected:
            return self._sim_pos_mm
        try:
            if self.backend == "sim":
                return self._sim_pos_mm
            if self.backend == "kinesis":
                return self._stage.get_position() / self.counts_per_mm
            if self.backend == "apt":
                return float(self._stage.position)
            if self.backend == "dotnet":
                from System import Decimal
                return float(Decimal.ToDouble(self._stage.Position))
        except Exception as e:  # noqa: BLE001
            print(f"[DelayStage] position read warning: {e}")
        return self._sim_pos_mm

    def get_position_fs(self) -> float:
        return self.mm_to_fs(self.get_position_mm() - self.zero_position_mm)

    def is_moving(self) -> bool:
        if not self.is_connected or self.backend == "sim":
            return False
        try:
            if hasattr(self._stage, "is_moving"):
                return bool(self._stage.is_moving())
            if hasattr(self._stage, "IsDeviceBusy"):
                return bool(self._stage.IsDeviceBusy)
        except Exception:  # noqa: BLE001
            pass
        return False

    def _can_move(self) -> bool:
        if not self.is_connected:
            print("[DelayStage] not connected")
            return False
        if not self.is_homed:
            print("[DelayStage] not homed -- call home() first")
            return False
        return True

    # -- unit conversions ----------------------------------------------------
    def mm_to_fs(self, x_mm: float) -> float:
        factor = 2.0 if self.double_pass else 1.0
        return x_mm * factor / SPEED_OF_LIGHT_MM_FS

    def fs_to_mm(self, t_fs: float) -> float:
        factor = 2.0 if self.double_pass else 1.0
        return t_fs * SPEED_OF_LIGHT_MM_FS / factor


if __name__ == "__main__":
    s = DelayStage()
    print("1 mm =", s.mm_to_fs(1.0), "fs ; 100 fs =", s.fs_to_mm(100), "mm")
    s.connect(simulate=True)
    s.home()
    s.move_to_fs(500.0)
    print("pos:", round(s.get_position_mm(), 6), "mm =", round(s.get_position_fs(), 1), "fs")
    s.disconnect()
