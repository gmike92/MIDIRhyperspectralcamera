"""Direct eBUS (GigE Vision) probe of the Allied Vision Goldeye G-033 that
BeamGage serves as "SP1203" -- bypassing BeamGage entirely. Connects, reads the
control nodes we'd need for a direct backend (exposure, gain, frame rate, pixel
format), then disconnects. Requires the camera FREE (stop BeamGage/Spiricon
services first). Usage:  .venv\\Scripts\\python.exe goldeye_probe.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera.irc806_camera import Irc806Camera

cam = Irc806Camera()
cam._ensure_loaded()          # load PvDotNet (eBUS)
pv = cam._pv

system = pv.PvSystem()
system.Find()
conn = None
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
        if "Goldeye" in model or cid.startswith("169.254.29"):
            conn = cid
            print(f"found: {model} @ {cid}")
if not conn:
    print("Goldeye NOT found via eBUS (still held by Spiricon services?)")
    sys.exit(1)

try:
    res = pv.PvDevice.CreateAndConnect(conn)
    dev = res[0] if isinstance(res, tuple) else res
except Exception as e:  # noqa: BLE001
    print(f"connect FAILED: {e}  (camera busy -> stop BeamGage/Spiricon first)")
    sys.exit(1)

p = dev.Parameters
def rd(name):
    try:
        prm = p.Get(name)
        return prm.ToString() if prm is not None else None
    except Exception:  # noqa: BLE001
        return None

print("\n=== control nodes (direct GigE Vision) ===")
for n in ("DeviceModelName", "DeviceVendorName", "DeviceSerialNumber",
          "Width", "Height", "WidthMax", "HeightMax", "PixelFormat",
          "ExposureTime", "ExposureAuto", "ExposureMode",
          "Gain", "GainAuto", "GainRaw", "SensorGain",
          "AcquisitionMode", "AcquisitionFrameRate", "AcquisitionFrameRateEnable",
          "AcquisitionFrameRateAbs", "TriggerMode", "TriggerSource",
          "DeviceTemperature", "SensorCoolingPower", "BitDepth", "PayloadSize"):
    v = rd(n)
    if v is not None:
        print(f"   {n:28s} = {v}")

try:
    dev.Disconnect()
except Exception:  # noqa: BLE001
    try:
        dev.Free()
    except Exception:  # noqa: BLE001
        pass
print("\ndone.")
