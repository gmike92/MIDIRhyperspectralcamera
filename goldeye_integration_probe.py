"""Inspect the Goldeye's integration/exposure controls (typed): current value,
valid RANGE, and enum OPTIONS. Then a short exposure sweep to see signal level.
Needs the camera free. Usage:  .venv\\Scripts\\python.exe goldeye_integration_probe.py
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


def uw(r):
    return r[1] if isinstance(r, tuple) else r


def show_float(name):
    try:
        pf = p.GetFloat(name)
        v, mn, mx = uw(pf.GetValue()), uw(pf.GetMin()), uw(pf.GetMax())
        unit = ""
        try:
            unit = uw(pf.GetUnit())
        except Exception:  # noqa: BLE001
            pass
        print(f"  {name:22s} = {v}  range=[{mn}, {mx}] {unit}")
    except Exception as e:  # noqa: BLE001
        print(f"  {name:22s} : not a float ({str(e).splitlines()[0]})")


def show_enum(name):
    try:
        pe = p.GetEnum(name)
        cnt = int(uw(pe.GetEntriesCount()))
        opts = []
        for i in range(cnt):
            e = uw(pe.GetEntryByIndex(i))
            nm = uw(e.GetName())
            try:
                nm += "" if bool(uw(e.IsAvailable())) else " (n/a)"
            except Exception:  # noqa: BLE001
                pass
            opts.append(nm)
        print(f"  {name:22s} = {pe.ToString()}   options={opts}")
    except Exception as e:  # noqa: BLE001
        print(f"  {name:22s} : not an enum ({str(e).splitlines()[0]})")


print("\n=== integration / exposure controls ===")
show_float("ExposureTime")
show_enum("ExposureAuto")
show_enum("ExposureMode")
show_enum("IntegrationMode")
show_enum("SensorGain")
show_float("AcquisitionFrameRate")

# Exposure sweep: what integration time gives a good (unsaturated) signal?
print("\n=== exposure sweep (peak counts, 14-bit full scale = 16383) ===")
cam.start_acquisition()
for ms in (0.1, 0.5, 1.0, 2.0, 4.0):
    cam.set_exposure(ms)
    time.sleep(0.3)
    peak = mean = 0
    got = 0
    t0 = time.time()
    while got < 5 and time.time() - t0 < 1.5:
        f = cam.get_frame()
        if f is not None:
            got += 1
            peak = max(peak, int(f.max()))
            mean = float(f.mean())
    pct = 100.0 * peak / 16383
    flag = "  <-- SATURATED" if peak >= 16300 else ("  <-- good" if 40 <= pct <= 90 else "")
    print(f"  {ms:5.2f} ms -> peak={peak:5d} ({pct:5.1f}% full)  mean={mean:6.1f}{flag}")

cam.stop_acquisition()
cam.disconnect()
print("\ndone.")
