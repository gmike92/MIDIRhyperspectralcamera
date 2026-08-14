"""Enumerate ALL GigE Vision devices visible via Pleora eBUS (both NICs), so we
can see whether the Ophir SP1203 is a standard GigE Vision camera we can stream
from with the existing eBUS stack (and how to tell it apart from the IRcam).
Connectionless discovery -- does NOT connect, safe to run while the app holds the
IRcam. Usage:  .venv\\Scripts\\python.exe discover_gige.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera.irc806_camera import Irc806Camera

cam = Irc806Camera()
cam._ensure_loaded()          # load PvDotNet (eBUS) into cam._pv
pv = cam._pv

system = pv.PvSystem()
system.Find()
print("interface count:", system.InterfaceCount)

def get(obj, *names):
    for nm in names:
        try:
            v = getattr(obj, nm)
            v = v() if callable(v) else v
            if v is not None and str(v) != "":
                return v
        except Exception:  # noqa: BLE001
            pass
    return None

for i in range(system.InterfaceCount):
    itf = system.GetInterface(i)
    iid = get(itf, "DisplayID", "GetDisplayID", "Name") or f"itf{i}"
    mac = get(itf, "MACAddress", "GetMACAddress")
    ip = get(itf, "IPAddress", "GetIPAddress")
    try:
        n = itf.DeviceCount
    except Exception:  # noqa: BLE001
        n = 0
    print(f"\n[interface {i}] {iid}  ip={ip} mac={mac}  devices={n}")
    for j in range(n):
        di = itf.GetDeviceInfo(j)
        print(f"   --- device {j} ---")
        for attr in ("ConnectionID", "DisplayID", "ModelName", "VendorName",
                     "SerialNumber", "UserDefinedName", "IPAddress", "MACAddress",
                     "DeviceClass"):
            val = get(di, attr, "Get" + attr)
            if val is not None:
                print(f"      {attr:16s} = {val}")
print("\ndone.")
