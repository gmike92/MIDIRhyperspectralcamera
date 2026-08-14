"""Standalone smoke test for the Ophir/BeamGage backend. LAUNCHES a BeamGage
engine (owns the SP1203), grabs a few frames, prints shape/stats, shuts down.

IMPORTANT: CLOSE any open BeamGage window first (else the two fight over the
camera). Usage:  .venv\\Scripts\\python.exe ophir_test.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera.ophir_camera import OphirBeamgageCamera

cam = OphirBeamgageCamera()
st = cam.connect()
print("connected:", st.connected, "|", st.message)
if not st.connected:
    sys.exit(1)
cam.start_acquisition()
print("acquiring:", cam.get_status().acquiring, "| dims:", st.width, "x", st.height)

got = 0
t0 = time.time()
while got < 5 and time.time() - t0 < 20:
    f = cam.get_frame()
    if f is not None:
        got += 1
        print(f"frame {got}: shape={f.shape} dtype={f.dtype} "
              f"min={f.min()} max={f.max()} mean={f.mean():.1f}")
    time.sleep(0.05)

print(f"\ngot {got} frame(s) in {time.time()-t0:.1f}s")
cam.stop_acquisition()
cam.disconnect()
print("done.")
