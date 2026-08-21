"""
diag_dcam.py -- minimal Hamamatsu DCAM probe (no app, no CameraDevice).

Run in the same env you launch the app with:
    conda activate midir
    python diag_dcam.py

It loads dcamapi.dll directly, calls dcamapi_init, and prints how many cameras
DCAM sees + the model of camera 0. Use it to tell apart:
  * device count 0  -> DCAM can't see the camera (busy / bitness / interface)
  * device count >0 -> camera is visible; the app should be able to open it
"""
import ctypes
import struct


def hexerr(code):
    return f"{code} (0x{code & 0xFFFFFFFF:08X})"


class DCAMAPI_INIT(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("iDeviceCount", ctypes.c_int32),
                ("reserved", ctypes.c_int32),
                ("initoptionbytes", ctypes.c_int32),
                ("initoption", ctypes.POINTER(ctypes.c_int32)),
                ("guid", ctypes.POINTER(ctypes.c_int32))]


class DCAMDEV_STRING(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("iString", ctypes.c_int32),
                ("text", ctypes.c_char_p),
                ("textbytes", ctypes.c_int32)]


DCAMERR_NOERROR = 1
DCAM_IDSTR_MODEL = 0x04000104

print(f"Python: {struct.calcsize('P') * 8}-bit")

try:
    dcam = ctypes.windll.dcamapi
except Exception as e:
    print("Could not load dcamapi.dll:", e)
    raise SystemExit(1)

init = DCAMAPI_INIT(0, 0, 0, 0, None, None)
init.size = ctypes.sizeof(init)
code = dcam.dcamapi_init(ctypes.byref(init))
print("dcamapi_init ->", hexerr(code))
print("iDeviceCount ->", init.iDeviceCount)

if init.iDeviceCount > 0:
    buf = ctypes.create_string_buffer(64)
    s = DCAMDEV_STRING(0, DCAM_IDSTR_MODEL, ctypes.cast(buf, ctypes.c_char_p), 64)
    s.size = ctypes.sizeof(s)
    r = dcam.dcamdev_getstring(ctypes.c_int32(0), ctypes.byref(s))
    print("camera 0 model ->", buf.value.decode(errors="replace"), "(getstring", hexerr(r), ")")

try:
    dcam.dcamapi_uninit()
except Exception:
    pass
