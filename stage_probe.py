"""stage_probe.py -- find how to move the LTS300 WITHOUT re-homing.

Kinesis blocks absolute moves when Status.IsHomed is False (VerifyDeviceMovement
-> "Cannot move to requested position"), even though the stage is already
referenced. This introspects the LongTravelStage .NET object to find the right
knob (CanMoveWithoutHomingFirst) / whether relative moves bypass the block, and
does ONE tiny 50 um test move.

Run with the stage DISCONNECTED in the app (click Disconnect on the Thorlabs
Stage panel -- no need to close the whole app / camera). Then:
    .venv\\Scripts\\python.exe stage_probe.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instruments.stage_driver import DelayStage

s = DelayStage()
s.override_limits = False          # keep the log clean; not testing limits here
ok = s.connect(exclude=["27256774"])   # exclude the rotator KDC101
print("connected:", ok, "| backend:", s.backend, "| serial:", s.serial)
if not ok or s.backend != "dotnet":
    print("Need the dotnet backend + a free stage. Is the stage disconnected in the app?")
    sys.exit(1)
dev = s._stage
from System import Decimal

def val(attr):
    try:
        return getattr(dev, attr)
    except Exception as e:  # noqa: BLE001
        return f"<err: {e}>"

print("\n=== homing / move API surface ===")
for n in sorted(d for d in dir(dev) if any(k in d.lower() for k in
        ("home", "move", "canmove", "limit", "needs", "jog", "reference"))):
    print("  ", n)

print("\n=== homing state ===")
for a in ("CanMoveWithoutHomingFirst", "NeedsHoming", "IsHomingRequired"):
    print(f"  {a} = {val(a)}")
try:
    st = dev.Status
    print(f"  Status.IsHomed={st.IsHomed}  IsHoming={getattr(st,'IsHoming','?')}  "
          f"Position={val('Position')}")
except Exception as e:  # noqa: BLE001
    print("  Status err:", e)

print("\n=== current limits + approach policy ===")
try:
    lim = dev.MotorPositionLimits
    print(f"  MotorPositionLimits: Min={val('MotorPositionLimits') and Decimal.ToDouble(lim.MinValue)} "
          f"Max={Decimal.ToDouble(lim.MaxValue)}")
except Exception as e:  # noqa: BLE001
    print(f"  MotorPositionLimits err: {e}")
try:
    print(f"  GetLimitsSoftwareApproachPolicy() = {dev.GetLimitsSoftwareApproachPolicy()}")
except Exception as e:  # noqa: BLE001
    print(f"  GetLimitsSoftwareApproachPolicy err: {e}")

print("\n=== set approach policy = AllowAllMoves (via Enum.Parse of getter type) ===")
from System import Enum
cur = dev.GetLimitsSoftwareApproachPolicy()
etype = cur.GetType()
print(f"  policy enum type: {etype.FullName}")
print(f"  values: {list(Enum.GetNames(etype))}")
try:
    allow = Enum.Parse(etype, "AllowAllMoves")
    dev.SetLimitsSoftwareApproachPolicy(allow)
    print(f"  SET OK -> now {dev.GetLimitsSoftwareApproachPolicy()}")
except Exception as e:  # noqa: BLE001
    print(f"  SET FAILED: {str(e).splitlines()[0]}")

pos0 = float(Decimal.ToDouble(dev.Position))
print(f"\n=== absolute MoveTo test from pos0={pos0:.4f} mm (-50 um) ===")
try:
    dev.MoveTo(Decimal(pos0 - 0.05), 20000)
    print(f"   MoveTo({pos0-0.05:.4f}): OK -> pos {float(Decimal.ToDouble(dev.Position)):.4f}")
except Exception as e:  # noqa: BLE001
    print(f"   MoveTo: {str(e).splitlines()[0]}")

s.disconnect()
print("\ndone.")
