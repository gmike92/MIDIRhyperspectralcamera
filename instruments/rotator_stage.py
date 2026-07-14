"""
rotator_stage.py -- Thorlabs rotation-mount driver (angle axis, degrees).

A Thorlabs rotation mount (e.g. PRM1Z8 / PRMTZ8) driven by a **KDC101** K-Cube
DC-servo controller over USB -- the SAME controller family as the delay stage
(see stage_driver.py), so the .NET Kinesis backend is identical apart from the
unit being **degrees** instead of mm. The actuator calibration assigned to the
controller in the Thorlabs Kinesis app makes MoveTo/Position report real degrees.

Decoupled, hardware-only (NO GUI). Hardware libraries are imported lazily so the
module imports cleanly with nothing installed. A `simulate=True` mode lets the
scan logic be developed with no hardware attached.

Backends tried, in order (first that works wins):
    1. pylablib  -> Thorlabs.KinesisMotor(scale="stage")   (real degrees)
    2. thorlabs_apt
    3. pythonnet  -> Thorlabs.MotionControl KCube.DCServo .NET DLLs
    4. simulate   (no hardware)

Usage:
    from rotator_stage import RotatorStage
    r = RotatorStage()
    r.connect(simulate=True)          # or connect() for real hardware
    r.home()
    r.move_to_deg(45.0)
    print(r.get_position_deg(), "deg")
    r.disconnect()

One-time setup (real hardware): open the Thorlabs Kinesis app, select the
KDC101, and assign its rotation actuator (e.g. PRM1Z8) under Settings -> Device,
else CreateKCubeDCServo throws a configuration NullReference (same gotcha as the
delay stage).
"""
from __future__ import annotations

import time
from typing import Optional


def wrap_deg(angle: float) -> float:
    """Normalise an angle to [0, 360)."""
    return float(angle) % 360.0


class RotatorStage:
    """Thorlabs rotation mount (KDC101) with a degrees interface."""

    def __init__(self, serial: Optional[str] = None,
                 stage_name: str = "PRM1-Z8") -> None:
        # Preferred serial: this rig has TWO KDC101 controllers (this rotator +
        # the delay stage), so the rotator MUST target its own serial rather than
        # auto-picking the first device found.
        self.preferred_serial = str(serial) if serial else None
        # Kinesis stage-settings name for the rotation mount. This is what tells
        # the controller its counts-per-degree; WITHOUT it MoveTo/Position are in
        # raw device units (Go-to lands at the wrong angle, jog moves randomly).
        self.stage_name = stage_name
        self.is_connected = False
        self.is_homed = False
        self.backend = None             # 'kinesis' | 'apt' | 'dotnet' | 'sim'
        self.serial = None
        self._stage = None              # underlying device object
        self._sim_pos_deg = 0.0

    # -- connection ----------------------------------------------------------
    def connect(self, serial: Optional[str] = None, simulate: bool = False) -> bool:
        if self.is_connected:
            return True
        serial = serial or self.preferred_serial
        if simulate:
            self.backend = "sim"
            self.serial = serial or "SIM-ROT"
            self.is_connected = True
            print("[RotatorStage] connected (SIMULATED)")
            return True
        for attempt in (self._connect_kinesis, self._connect_apt, self._connect_dotnet):
            try:
                if attempt(serial):
                    self.is_connected = True
                    print(f"[RotatorStage] connected via {self.backend} (serial {self.serial})")
                    return True
            except Exception as e:  # noqa: BLE001
                print(f"[RotatorStage] {attempt.__name__} failed: {e}")
        print("[RotatorStage] no hardware found. Use connect(simulate=True) to develop.")
        return False

    def _connect_kinesis(self, serial: Optional[str]) -> bool:
        from pylablib.devices import Thorlabs
        devices = Thorlabs.list_kinesis_devices()
        if not devices:
            return False
        sn = serial or devices[0][0]
        # scale="stage" -> pylablib applies the device's stored calibration, so
        # positions/moves are in real degrees rather than raw encoder counts.
        self._stage = Thorlabs.KinesisMotor(sn, scale="stage")
        self.backend, self.serial = "kinesis", str(sn)
        return True

    def _connect_apt(self, serial: Optional[str]) -> bool:
        import thorlabs_apt as apt
        devices = apt.list_available_devices()
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
        clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
        DeviceManagerCLI.BuildDeviceList()
        devices = list(DeviceManagerCLI.GetDeviceList())
        if not devices:
            return False
        sns = [str(d) for d in devices]
        if serial and serial not in sns:
            raise RuntimeError(
                f"rotator serial {serial} not found (Thorlabs devices present: {sns}). "
                "Check the USB connection and that the KDC101 is powered.")
        sn = str(serial) if serial else str(devices[0])
        try:
            dev = KCubeDCServo.CreateKCubeDCServo(sn)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"KDC101 {sn}: cannot load device configuration ({e}). Open the "
                "Thorlabs Kinesis app, select this controller, assign the rotation "
                "actuator (e.g. PRM1Z8) under Settings -> Device, then retry.") from e
        dev.Connect(sn)
        dev.WaitForSettingsInitialized(5000)
        dev.StartPolling(250)
        time.sleep(0.25)
        dev.EnableDevice()
        time.sleep(0.25)
        # Load the actuator calibration so MoveTo/Position use real units
        # (degrees) instead of raw device units. LoadMotorConfiguration returns a
        # MotorConfiguration; we must set its DeviceSettingsName to the rotation
        # mount (e.g. "PRM1-Z8") and push it, otherwise the controller has no
        # counts-per-degree -> Go-to lands at the wrong angle and jog is random.
        config = dev.LoadMotorConfiguration(sn)
        if self.stage_name:
            try:
                config.DeviceSettingsName = self.stage_name
                config.UpdateCurrentConfiguration()
                # Apply the freshly-configured settings to the live device.
                try:
                    dev.SetSettings(dev.MotorDeviceSettings, True, False)
                except Exception:  # noqa: BLE001
                    pass
                print(f"[RotatorStage] applied stage settings '{self.stage_name}'")
            except Exception as e:  # noqa: BLE001
                print(f"[RotatorStage] WARNING: could not apply stage settings "
                      f"'{self.stage_name}' ({e}); positions may be in device units")
        self._stage = dev
        self.backend, self.serial = "dotnet", str(sn)
        # Sanity log: with the stage applied this should read a plausible degrees
        # value; if it reads huge, the settings did not take.
        try:
            print(f"[RotatorStage] position now {self.get_position_deg():.4f} deg")
        except Exception:  # noqa: BLE001
            pass
        return True

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
            print(f"[RotatorStage] disconnect warning: {e}")
        self._stage = None
        self.is_connected = False
        self.is_homed = False
        print("[RotatorStage] disconnected")

    # -- homing / motion -----------------------------------------------------
    def home(self, timeout_s: float = 60.0) -> bool:
        if not self.is_connected:
            return False
        if self.backend == "sim":
            self._sim_pos_deg = 0.0
        elif self.backend == "kinesis":
            self._stage.home(sync=True, timeout=timeout_s)
        elif self.backend == "apt":
            self._stage.move_home(blocking=True)
        elif self.backend == "dotnet":
            self._stage.Home(int(timeout_s * 1000))
        self.is_homed = True
        print("[RotatorStage] homed")
        return True

    def move_to_deg(self, angle_deg: float, wait: bool = True) -> bool:
        if not self._can_move():
            return False
        target = wrap_deg(angle_deg)
        if self.backend == "sim":
            self._sim_pos_deg = target
        elif self.backend == "kinesis":
            self._stage.move_to(target)          # scale="stage" -> degrees
            if wait:
                self._stage.wait_move()
        elif self.backend == "apt":
            self._stage.move_to(float(target), blocking=wait)
        elif self.backend == "dotnet":
            from System import Decimal
            self._stage.MoveTo(Decimal(float(target)), 60000 if wait else 0)
        return True

    def move_relative_deg(self, delta_deg: float, wait: bool = True) -> bool:
        return self.move_to_deg(self.get_position_deg() + delta_deg, wait)

    def get_position_deg(self) -> float:
        if not self.is_connected:
            return self._sim_pos_deg
        try:
            if self.backend == "sim":
                return self._sim_pos_deg
            if self.backend == "kinesis":
                return float(self._stage.get_position())     # degrees
            if self.backend == "apt":
                return float(self._stage.position)
            if self.backend == "dotnet":
                from System import Decimal
                return float(Decimal.ToDouble(self._stage.Position))
        except Exception as e:  # noqa: BLE001
            print(f"[RotatorStage] position read warning: {e}")
        return self._sim_pos_deg

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
            print("[RotatorStage] not connected")
            return False
        if not self.is_homed:
            print("[RotatorStage] not homed -- call home() first")
            return False
        return True


if __name__ == "__main__":
    r = RotatorStage()
    r.connect(simulate=True)
    r.home()
    r.move_to_deg(45.0)
    print("pos:", round(r.get_position_deg(), 4), "deg")
    r.move_to_deg(370.0)
    print("wrapped 370 ->", round(r.get_position_deg(), 4), "deg")
    r.disconnect()
