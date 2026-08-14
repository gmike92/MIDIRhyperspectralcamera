"""One-shot: connect to the IRC806/812 and read sensor geometry, binning /
decimation, and full device info to determine the TRUE physical resolution &
pixel pitch (640x512@20um IRC806  vs  1280x1024@12um IRC812 windowed/binned).
Read-only: sets nothing. Run only with the main app CLOSED (camera is exclusive).
Usage:  .venv\\Scripts\\python.exe query_sensor.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera.irc806_camera import Irc806Camera

cam = Irc806Camera()
st = cam.connect()
print("connected:", st.connected, "|", st.message)
if not st.connected:
    sys.exit(1)
p = cam._device.Parameters

named = [
    # delivered geometry
    "Width", "Height", "WidthMax", "HeightMax", "OffsetX", "OffsetY",
    "SensorWidth", "SensorHeight", "PixelFormat", "PayloadSize", "LinePitch",
    # binning / decimation -> reveals a 1280x1024 read out as 640x512
    "BinningHorizontal", "BinningVertical", "BinningHorizontalMode",
    "BinningVerticalMode", "DecimationHorizontal", "DecimationVertical",
    "BinningSelector", "ReverseX", "ReverseY",
    # identity
    "DeviceModelName", "DeviceVendorName", "DeviceManufacturerInfo",
    "DeviceVersion", "DeviceFirmwareVersion", "DeviceSerialNumber",
    "DeviceID", "DeviceScanType",
]
print("\n=== named nodes ===")
for name in named:
    print(f"  {name:24s} = {cam._read(p, name)}")

# Dump EVERY readable node to a file + surface geometry-relevant ones.
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_nodes_dump.txt")
hits = []
try:
    n = p.GetCount() if hasattr(p, "GetCount") else p.Count
    with open(out, "w", encoding="utf-8") as fh:
        for i in range(int(n)):
            prm = p.Get(i) if hasattr(p, "Get") else p[i]
            try:
                nm = str(prm.GetName())
            except Exception:
                nm = str(getattr(prm, "Name", ""))
            try:
                val = prm.ToString()
            except Exception:
                val = "?"
            fh.write(f"{nm} = {val}\n")
            low, sval = nm.lower(), str(val)
            if (any(k in low for k in ("pixel", "sensor", "bin", "decim", "width",
                                       "height", "size", "pitch", "resolution",
                                       "format", "reverse", "offset"))
                    or "1280" in sval or "1024" in sval):
                hits.append(f"  {nm:28s} = {val}")
    print(f"\n=== geometry / binning / 1280 / 1024 nodes  (full dump -> {out}) ===")
    print("\n".join(hits) if hits else "  (none)")
except Exception as e:  # noqa: BLE001
    print("  (enumeration failed:", e, ")")

try:
    cam.disconnect()
except Exception:
    pass
print("\ndone.")
