"""Diagnostic for the Goldeye eBUS stream: packet size, complete vs incomplete
frame counts, lost-packet stats, and whether a lower frame rate restores
reliable streaming. Needs the camera FREE. Usage:
    .venv\\Scripts\\python.exe goldeye_test.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera.goldeye_camera import GoldeyeCamera

cam = GoldeyeCamera()
st = cam.connect()
print("connected:", st.connected, "|", st.message)
if not st.connected:
    sys.exit(1)
p = cam._device.Parameters

def rd(n):
    try:
        v = p.Get(n)
        return v.ToString() if v is not None else None
    except Exception:  # noqa: BLE001
        return None

print("BEFORE:  TriggerMode =", rd("TriggerMode"), "| TriggerSource =", rd("TriggerSource"),
      "| FrameRate =", rd("AcquisitionFrameRate"))
# Free-run: turn the trigger OFF so the camera streams continuously.
for setter, node, val in ((p.SetEnumValue, "TriggerMode", "Off"),
                          (p.SetFloatValue, "AcquisitionFrameRate", 30.0)):
    try:
        setter(node, val)
    except Exception as e:  # noqa: BLE001
        print(f"set {node} failed:", e)
print("AFTER:   TriggerMode =", rd("TriggerMode"), "| FrameRate =", rd("AcquisitionFrameRate"))
print("=== bandwidth / rate nodes ===")
for n in ("DeviceLinkThroughputLimitMode", "DeviceLinkThroughputLimit",
          "AcquisitionFrameRateEnable", "AcquisitionFrameRateLimit",
          "GevSCPD", "GevSCPSPacketSize", "StreamBytesPerSecond", "ExposureTime"):
    print(f"   {n} = {rd(n)}")
# Try to lift the throughput cap + enable the frame-rate control.
for setter, node, val in ((p.SetEnumValue, "DeviceLinkThroughputLimitMode", "Off"),
                          (p.SetBooleanValue, "AcquisitionFrameRateEnable", True),
                          (p.SetIntegerValue, "GevSCPD", 0)):
    try:
        setter(node, val)
        print(f"   set {node} -> {rd(node)}")
    except Exception as e:  # noqa: BLE001
        print(f"   set {node} failed: {str(e).splitlines()[0]}")

cam.start_acquisition()
print("packet size (GevSCPSPacketSize):", rd("GevSCPSPacketSize"))

# Raw retrieve loop so we can count complete vs incomplete.
pv = cam._pv
complete = incomplete = other = 0
t0 = time.time()
while time.time() - t0 < 4.0:
    res, buf, opres = cam._retrieve(500)
    if res is None or not res.IsOK:
        continue
    try:
        if not opres.IsOK:
            incomplete += 1
        elif int(buf.PayloadType) == int(pv.PvPayloadType.Image):
            im = buf.Image
            if int(im.Width) == st.width and int(im.Height) == st.height:
                complete += 1
            else:
                incomplete += 1
        else:
            other += 1
    finally:
        cam._stream.QueueBuffer(buf)
dt = time.time() - t0

print(f"\nin {dt:.1f}s: complete={complete} ({complete/dt:.1f} fps), "
      f"incomplete={incomplete}, other={other}")

# eBUS stream statistics (lost packets / resends).
sp = cam._stream.Parameters
for n in ("ImagesCount", "BlocksDropped", "LostPacketCount", "ResendPacketCount",
          "ImagesDropped", "PipelineBlocksDropped"):
    v = None
    try:
        prm = sp.Get(n)
        v = prm.ToString() if prm is not None else None
    except Exception:  # noqa: BLE001
        pass
    if v is not None:
        print(f"   stream.{n} = {v}")

cam.stop_acquisition()
cam.disconnect()
print("done.")
