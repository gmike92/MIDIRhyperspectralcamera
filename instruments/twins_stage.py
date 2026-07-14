"""
twins_stage.py -- NIREOS Gemini (SmarAct SCU3D) stage driver.

This is the motorized wedge stage *inside* the TWINS interferometer. It is
driven through SmarAct's SCU3DControl.dll via ctypes. Used by subtwinslv.py to
scan the interferometer delay.

Hardware-only (NO GUI). Provides a `simulate=True` mode so the TWINS scan
logic can be developed without the DLL/hardware present.

Usage:
    from twins_stage import TwinsStage
    st = TwinsStage()
    st.connect(simulate=True)        # or connect() with the real DLL
    st.move_to(24.0); st.wait_for_stop()
    print(st.get_position())
    st.disconnect()
"""
from __future__ import annotations

import ctypes
import time
from ctypes import POINTER, c_int, c_uint
from pathlib import Path
from typing import Optional

# Candidate locations of SmarAct's SCU3DControl.dll (edit to match this PC).
DLL_SEARCH_PATHS = [
    r"C:\SmarAct\SCU\SDK\lib64\SCU3DControl.dll",   # 64-bit SCU install (this PC)
    r"C:\SmarAct\SCU\SDK\lib\SCU3DControl.dll",      # 32-bit fallback
    r".\SCU3DControl.dll",
    r".\Twins\Python Scripts\SCU3DControl.dll",
    r"C:\Program Files\SmarAct\SCU\SCU3DControl.dll",
]

POSITION_SCALE = 10000     # raw device units per mm
# SCU3D channel status codes (SCU3DControl.h). A normal position move reports
# MOVING(2)/TARGETING(3); 6 is MOVING_TO_REFERENCE (only during referencing).
# The stage is settled when it reads STOPPED(0) or HOLDING(4).
STATUS_STOPPED = 0
STATUS_MOVING = 2
STATUS_TARGETING = 3
STATUS_HOLDING = 4
STATUS_MOVING_TO_REFERENCE = 6
# "in motion" = anything that isn't idle (STOPPED) or settled-at-target (HOLDING):
# SETTING_AMPLITUDE(1), MOVING(2), TARGETING(3), CALIBRATING(5), MOVE_TO_REF(6).
_IN_MOTION = (1, 2, 3, 5, 6)
HOME_POSITION_MM = 19.0
SAFE_POSITION_MM = 25.0


class TwinsStage:
    """SmarAct SCU3D single-channel stage (the TWINS wedge translator)."""

    def __init__(self) -> None:
        self.dll = None
        self.dll_path = None
        self.is_connected = False
        self.backend = None            # 'scu' | 'sim'
        self.system_index = c_uint(0)
        self.channel_index = c_uint(0)
        self._position_mm = 0.0

    # -- connection ----------------------------------------------------------
    def connect(self, dll_path: Optional[str] = None, simulate: bool = False,
                home: bool = True) -> bool:
        if self.is_connected:
            return True
        if simulate:
            self.backend = "sim"
            self.is_connected = True
            self._position_mm = HOME_POSITION_MM
            print("[TwinsStage] connected (SIMULATED)")
            return True

        if not self._load_dll(dll_path):
            print("[TwinsStage] SCU3DControl.dll not found. Use connect(simulate=True).")
            return False
        try:
            self.dll.SA_InitDevices.argtypes = [c_uint]
            self.dll.SA_InitDevices.restype = c_int
            if self.dll.SA_InitDevices(c_uint(0)) != 0:
                print("[TwinsStage] SA_InitDevices failed -- power-cycle the GEMINI.")
                return False
        except Exception as e:  # noqa: BLE001
            print(f"[TwinsStage] init failed: {e}")
            return False

        self.backend = "scu"
        self.is_connected = True
        self._ensure_referenced()
        if home:
            self.move_to(HOME_POSITION_MM)
            self.wait_for_stop()
        print("[TwinsStage] connected via SCU3D")
        return True

    def _load_dll(self, dll_path: Optional[str]) -> bool:
        for path in ([dll_path] if dll_path else []) + DLL_SEARCH_PATHS:
            if path and Path(path).exists():
                try:
                    self.dll = ctypes.CDLL(path)
                    self.dll_path = path
                    return True
                except OSError as e:
                    print(f"[TwinsStage] load failed {path}: {e}")
        return False

    def _ensure_referenced(self) -> None:
        try:
            known = c_uint(0)
            self.dll.SA_GetPhysicalPositionKnown_S.argtypes = [c_uint, c_uint, POINTER(c_uint)]
            self.dll.SA_GetPhysicalPositionKnown_S.restype = c_int
            self.dll.SA_GetPhysicalPositionKnown_S(self.system_index, self.channel_index,
                                                   ctypes.byref(known))
            if known.value == 0:
                print("[TwinsStage] finding reference mark...")
                self.dll.SA_MoveToReference_S.argtypes = [c_uint, c_uint, c_uint, c_uint]
                self.dll.SA_MoveToReference_S.restype = c_int
                self.dll.SA_MoveToReference_S(self.system_index, self.channel_index,
                                              c_uint(0), c_uint(0))
                self.wait_for_stop()
        except Exception as e:  # noqa: BLE001
            print(f"[TwinsStage] reference check warning: {e}")

    def disconnect(self, safe: bool = True) -> None:
        if not self.is_connected:
            return
        if self.backend == "scu":
            try:
                if safe:
                    self.move_to(SAFE_POSITION_MM)
                    self.wait_for_stop()
                self.dll.SA_ReleaseDevices.restype = c_int
                self.dll.SA_ReleaseDevices()
            except Exception as e:  # noqa: BLE001
                print(f"[TwinsStage] disconnect warning: {e}")
        self.dll = None
        self.is_connected = False
        print("[TwinsStage] disconnected")

    # -- motion --------------------------------------------------------------
    def move_to(self, position_mm: float) -> bool:
        if not self.is_connected:
            return False
        if self.backend == "sim":
            self._position_mm = position_mm
            return True
        try:
            raw = c_int(int(position_mm * POSITION_SCALE))
            self.dll.SA_MovePositionAbsolute_S.argtypes = [c_uint, c_uint, c_int, c_uint]
            self.dll.SA_MovePositionAbsolute_S.restype = c_int
            if self.dll.SA_MovePositionAbsolute_S(self.system_index, self.channel_index,
                                                  raw, c_uint(60000)) != 0:
                return False
            self._position_mm = position_mm
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[TwinsStage] move failed: {e}")
            return False

    def move_by(self, delta_mm: float) -> bool:
        return self.move_to(self._position_mm + delta_mm)

    def get_position(self) -> float:
        if not self.is_connected or self.backend == "sim":
            return self._position_mm
        try:
            pos = c_int(0)
            self.dll.SA_GetPosition_S.argtypes = [c_uint, c_uint, POINTER(c_int)]
            self.dll.SA_GetPosition_S.restype = c_int
            if self.dll.SA_GetPosition_S(self.system_index, self.channel_index,
                                         ctypes.byref(pos)) == 0:
                self._position_mm = pos.value / POSITION_SCALE
        except Exception as e:  # noqa: BLE001
            print(f"[TwinsStage] position read warning: {e}")
        return self._position_mm

    def is_moving(self) -> bool:
        if not self.is_connected or self.backend == "sim":
            return False
        try:
            status = c_int(0)
            self.dll.SA_GetStatus_S.argtypes = [c_uint, c_uint, POINTER(c_int)]
            self.dll.SA_GetStatus_S.restype = c_int
            if self.dll.SA_GetStatus_S(self.system_index, self.channel_index,
                                       ctypes.byref(status)) == 0:
                return status.value in _IN_MOTION
        except Exception:  # noqa: BLE001
            pass
        return False

    def wait_for_stop(self, timeout_s: float = 30.0) -> bool:
        if not self.is_connected or self.backend == "sim":
            return True
        t0 = time.time()
        # First wait for the move to actually BEGIN: the SCU reports status
        # asynchronously, so it can still read STOPPED for a few ms after the
        # move command -- without this, wait_for_stop would return instantly and
        # the scan would record during the move. Bounded so a no-op move can't stall.
        while not self.is_moving():
            if time.time() - t0 > 0.5:
                break
            time.sleep(0.005)
        # Then wait for it to settle (STOPPED or HOLDING at target).
        while self.is_moving():
            if time.time() - t0 > timeout_s:
                print("[TwinsStage] motion timeout")
                return False
            time.sleep(0.01)
        return True


if __name__ == "__main__":
    st = TwinsStage()
    st.connect(simulate=True)
    st.move_to(24.0); st.wait_for_stop()
    print("pos:", st.get_position(), "mm")
    st.disconnect()
