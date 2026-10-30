"""
stage_driver.py -- Thorlabs Kinesis linear-stage driver (KDC101 + MTS50-Z8).

Generic driver for a Thorlabs brushed-DC-servo linear stage on a KDC101 (or
TDC001) controller, with a **millimetre** interface. Several identical stages
can be used at once: each driver instance is pinned to its controller's serial
number, so three MTS50-Z8 on three KDC101s never grab each other's device.

Decoupled, hardware-only (NO GUI). Hardware libraries are imported lazily so
this module imports cleanly even with nothing installed/connected. A built-in
`simulate=True` mode lets you develop and test scan logic with no stage wired.

Backends tried, in order (first that works wins):
    1. pylablib  -> Thorlabs.KinesisMotor(scale=counts_per_mm)   (mm units)
    2. thorlabs_apt
    3. pythonnet -> Thorlabs.MotionControl Kinesis .NET DLLs (KCube.DCServo)
    4. simulate   (no hardware)

Stage calibration: the encoder counts per mm and the travel are taken from
STAGE_TABLE by `stage_name` (default "MTS50-Z8": 34304 counts/mm, 50 mm). For
the .NET backend the same name is pushed to the controller as its
DeviceSettingsName so MoveTo/Position report real mm; without it the KDC101
works in raw device units.

One-time setup (real hardware, .NET backend): open the Thorlabs Kinesis app,
select each KDC101 and assign its actuator (MTS50-Z8) under Settings -> Device,
else CreateKCubeDCServo throws a configuration error.

Usage:
    from instruments.stage_driver import KinesisStage
    m = KinesisStage(serial="27000001", stage_name="MTS50-Z8")
    m.connect()                      # or connect(simulate=True)
    m.home()
    m.move_to_mm(12.5)
    print(m.get_position_mm(), "mm")
    m.disconnect()

`DelayStage` (below) is the same driver plus a femtosecond delay-line interface
(double pass on a retroreflector), kept for the pump-probe Z-scan.

Install (when hardware is here): `pip install pylablib`  and/or Thorlabs
Kinesis (https://www.thorlabs.com/kinesis). thorlabs_apt is optional.
"""
from __future__ import annotations

import time
from typing import Optional

SPEED_OF_LIGHT_MM_FS = 0.000299792458  # mm per fs

# Kinesis stage name -> (encoder counts per mm, travel in mm).
# Counts from the Thorlabs APT/Kinesis motor programming guide (Z8xx family:
# 512 counts/rev encoder x 67.49:1 gearbox on a 1 mm lead screw).
STAGE_TABLE = {
    "MTS50-Z8": (34304, 50.0),
    "MTS25-Z8": (34304, 25.0),
    "Z825B": (34304, 25.0),
    "Z812B": (34304, 12.0),
    "Z806": (34304, 6.0),
    "Z625B": (24600, 25.0),
    "Z612B": (24600, 12.0),
    "Z606": (24600, 6.0),
}
DEFAULT_STAGE = "MTS50-Z8"

# Full-travel homing of a 50 mm stage at the Kinesis default homing velocity
# takes ~1 min; give it headroom.
DEFAULT_HOME_TIMEOUT_S = 120.0


class KinesisStage:
    """Thorlabs KDC101-driven linear stage with a millimetre interface."""

    def __init__(self, serial: Optional[str] = None,
                 stage_name: str = DEFAULT_STAGE,
                 counts_per_mm: Optional[int] = None,
                 travel_mm: Optional[float] = None,
                 name: str = "") -> None:
        # Preferred serial: with several KDC101s on the same PC each driver
        # MUST target its own controller rather than auto-pick the first found.
        self.preferred_serial = str(serial) if serial else None
        self.stage_name = stage_name
        table_counts, table_travel = STAGE_TABLE.get(stage_name, (None, None))
        self.counts_per_mm = int(counts_per_mm or table_counts or 34304)
        self.travel_mm = float(travel_mm or table_travel or 50.0)
        self.name = name or stage_name
        self._tag = f"[{self.name}]"

        self.is_connected = False
        self.is_homed = False
        self.backend = None             # 'kinesis' | 'apt' | 'dotnet' | 'sim'
        self.serial = None
        self._stage = None              # underlying device object
        self._sim_pos_mm = 0.0
        self._exclude = set()           # serials to skip on auto-pick

    # -- connection ----------------------------------------------------------
    def connect(self, serial: Optional[str] = None, simulate: bool = False,
                exclude=None) -> bool:
        if self.is_connected:
            return True
        serial = str(serial) if serial else self.preferred_serial
        # Serials of the OTHER Thorlabs controllers on this rig: when we have to
        # auto-pick (no serial) never grab one of theirs.
        self._exclude = set(str(s) for s in (exclude or []) if s)
        self._exclude.discard(serial)
        if simulate:
            self.backend = "sim"
            self.serial = serial or f"SIM-{self.name}"
            self.is_connected = True
            print(f"{self._tag} connected (SIMULATED)")
            return True
        for attempt in (self._connect_kinesis, self._connect_apt, self._connect_dotnet):
            try:
                if attempt(serial):
                    self.is_connected = True
                    print(f"{self._tag} connected via {self.backend} (serial {self.serial})")
                    return True
            except Exception as e:  # noqa: BLE001
                print(f"{self._tag} {attempt.__name__} failed: {e}")
        print(f"{self._tag} no hardware found. Use connect(simulate=True) to develop.")
        return False

    def _pick_serial(self, serial: Optional[str], present: list) -> Optional[str]:
        """Resolve the serial to open from the devices actually present."""
        present = [str(s) for s in present]
        if serial:
            if serial not in present:
                raise RuntimeError(
                    f"serial {serial} not found (Thorlabs devices present: {present}). "
                    "Check the USB connection and that the KDC101 is powered.")
            return serial
        free = [s for s in present if s not in self._exclude]
        return free[0] if free else None

    def _connect_kinesis(self, serial: Optional[str]) -> bool:
        from pylablib.devices import Thorlabs
        devices = Thorlabs.list_kinesis_devices()
        sn = self._pick_serial(serial, [d[0] for d in devices])
        if sn is None:
            return False
        # scale=counts_per_mm -> pylablib works in user units of 1 mm, so
        # move_to/get_position are in mm (velocity in mm/s).
        self._stage = Thorlabs.KinesisMotor(sn, scale=self.counts_per_mm)
        self.backend, self.serial = "kinesis", str(sn)
        return True

    def _connect_apt(self, serial: Optional[str]) -> bool:
        import thorlabs_apt as apt
        devices = apt.list_available_devices()
        sn = self._pick_serial(serial, [d[1] for d in devices])
        if sn is None:
            return False
        self._stage = apt.Motor(int(sn))
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
        clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
        DeviceManagerCLI.BuildDeviceList()
        sn = self._pick_serial(serial, list(DeviceManagerCLI.GetDeviceList()))
        if sn is None:
            return False
        try:
            dev = KCubeDCServo.CreateKCubeDCServo(sn)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"KDC101 {sn}: cannot load device configuration ({e}). Open the "
                "Thorlabs Kinesis app, select this controller, assign the actuator "
                f"({self.stage_name}) under Settings -> Device, then retry.") from e
        dev.Connect(sn)
        dev.WaitForSettingsInitialized(5000)
        dev.StartPolling(250)
        time.sleep(0.25)
        dev.EnableDevice()
        time.sleep(0.25)
        # Load the actuator calibration and pin it to our stage so MoveTo /
        # Position use real mm instead of raw device units.
        config = dev.LoadMotorConfiguration(sn)
        if self.stage_name:
            try:
                config.DeviceSettingsName = self.stage_name
                config.UpdateCurrentConfiguration()
                try:
                    dev.SetSettings(dev.MotorDeviceSettings, True, False)
                except Exception:  # noqa: BLE001
                    pass
                print(f"{self._tag} applied stage settings '{self.stage_name}'")
            except Exception as e:  # noqa: BLE001
                print(f"{self._tag} WARNING: could not apply stage settings "
                      f"'{self.stage_name}' ({e}); positions may be in device units")
        self._stage = dev
        self.backend, self.serial = "dotnet", str(sn)
        return True

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        try:
            if self.backend == "kinesis" and self._stage is not None:
                self._stage.close()
            elif self.backend == "dotnet" and self._stage is not None:
                self._stage.StopPolling()
                self._stage.Disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"{self._tag} disconnect warning: {e}")
        self._stage = None
        self.is_connected = False
        self.is_homed = False
        print(f"{self._tag} disconnected")

    # -- homing / motion -----------------------------------------------------
    def home(self, timeout_s: float = DEFAULT_HOME_TIMEOUT_S) -> bool:
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
        print(f"{self._tag} homed")
        return True

    def in_travel(self, position_mm: float) -> bool:
        return 0.0 <= float(position_mm) <= self.travel_mm

    def move_to_mm(self, position_mm: float, wait: bool = True) -> bool:
        if not self._can_move():
            return False
        if not self.in_travel(position_mm):
            print(f"{self._tag} target {position_mm:.4f} mm outside travel "
                  f"0 .. {self.travel_mm:g} mm -- move refused")
            return False
        if self.backend == "sim":
            self._sim_pos_mm = float(position_mm)
        elif self.backend == "kinesis":
            self._stage.move_to(float(position_mm))
            if wait:
                self._stage.wait_move()
        elif self.backend == "apt":
            self._stage.move_to(float(position_mm), blocking=wait)
        elif self.backend == "dotnet":
            from System import Decimal
            self._stage.MoveTo(Decimal(float(position_mm)), 60000 if wait else 0)
        return True

    def move_relative_mm(self, delta_mm: float, wait: bool = True) -> bool:
        return self.move_to_mm(self.get_position_mm() + delta_mm, wait)

    def get_position_mm(self) -> float:
        if not self.is_connected:
            return self._sim_pos_mm
        try:
            if self.backend == "sim":
                return self._sim_pos_mm
            if self.backend == "kinesis":
                return float(self._stage.get_position())
            if self.backend == "apt":
                return float(self._stage.position)
            if self.backend == "dotnet":
                from System import Decimal
                return float(Decimal.ToDouble(self._stage.Position))
        except Exception as e:  # noqa: BLE001
            print(f"{self._tag} position read warning: {e}")
        return self._sim_pos_mm

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

    def stop(self) -> None:
        if not self.is_connected:
            return
        try:
            if self.backend == "kinesis":
                self._stage.stop()
            elif self.backend == "apt":
                self._stage.stop_profiled()
            elif self.backend == "dotnet":
                self._stage.StopImmediate()
        except Exception as e:  # noqa: BLE001
            print(f"{self._tag} stop warning: {e}")

    def _can_move(self) -> bool:
        if not self.is_connected:
            print(f"{self._tag} not connected")
            return False
        if not self.is_homed:
            print(f"{self._tag} not homed -- call home() first")
            return False
        return True


class DelayStage(KinesisStage):
    """KinesisStage used as a pump-probe delay line (double pass on a
    retroreflector), adding a femtosecond interface.

        c = 0.000299792458 mm/fs
        delay (fs) = 2 * x (mm) / c          (double pass)
        x (mm)     = delay (fs) * c / 2
    """

    def __init__(self, serial: Optional[str] = None,
                 stage_name: str = DEFAULT_STAGE,
                 double_pass: bool = True, **kwargs) -> None:
        super().__init__(serial=serial, stage_name=stage_name,
                         name=kwargs.pop("name", "DelayStage"), **kwargs)
        self.double_pass = double_pass
        self.zero_position_mm = 0.0     # mm position that corresponds to 0 fs

    def move_to_fs(self, delay_fs: float, wait: bool = True) -> bool:
        return self.move_to_mm(self.zero_position_mm + self.fs_to_mm(delay_fs), wait)

    def move_relative_fs(self, delta_fs: float, wait: bool = True) -> bool:
        return self.move_relative_mm(self.fs_to_mm(delta_fs), wait)

    def get_position_fs(self) -> float:
        return self.mm_to_fs(self.get_position_mm() - self.zero_position_mm)

    # -- unit conversions ----------------------------------------------------
    def mm_to_fs(self, x_mm: float) -> float:
        factor = 2.0 if self.double_pass else 1.0
        return x_mm * factor / SPEED_OF_LIGHT_MM_FS

    def fs_to_mm(self, t_fs: float) -> float:
        factor = 2.0 if self.double_pass else 1.0
        return t_fs * SPEED_OF_LIGHT_MM_FS / factor


if __name__ == "__main__":
    motors = [KinesisStage(serial=f"2700000{i}", name=f"Motor {i}") for i in (1, 2, 3)]
    for m in motors:
        m.connect(simulate=True)
        m.home()
    motors[0].move_to_mm(12.5)
    motors[1].move_to_mm(60.0)        # refused: outside 50 mm travel
    motors[2].move_relative_mm(-1.0)  # refused: would go below 0
    for m in motors:
        print(m.name, m.serial, round(m.get_position_mm(), 4), "mm")
        m.disconnect()
    d = DelayStage()
    print("1 mm =", d.mm_to_fs(1.0), "fs ; 100 fs =", d.fs_to_mm(100), "mm")
