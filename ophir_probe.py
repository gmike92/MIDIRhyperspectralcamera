"""Reflection-only probe of the BeamGage Automation assembly (no engine launch,
does not touch the camera). Lists exported types + the members of the main
automation class so we can write the backend. Usage:
    .venv\\Scripts\\python.exe ophir_probe.py
"""
import os, sys, clr  # noqa: F401

BG = r"C:\Program Files\Spiricon\BeamGage Professional"
sys.path.append(BG)
try:
    os.add_dll_directory(BG)
except Exception:  # noqa: BLE001
    pass

from System.Reflection import Assembly, BindingFlags  # noqa: E402
Assembly.LoadFrom(os.path.join(BG, "Spiricon.Automation.dll"))
asm = Assembly.LoadFrom(os.path.join(BG, "Spiricon.BeamGage.Automation.dll"))

try:
    types = list(asm.GetExportedTypes())
except Exception as e:  # noqa: BLE001  (ReflectionTypeLoadException)
    types = [t for t in getattr(e, "Types", []) if t is not None]

print(f"=== {len(types)} exported types ===")
for t in sorted(types, key=lambda x: x.FullName):
    print("  ", t.FullName)

# Find the main automation class + anything frame/result related.
def dump(t):
    print(f"\n=== {t.FullName} ===")
    bf = BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly
    props = t.GetProperties(bf)
    print("  -- properties --")
    for p in sorted(props, key=lambda x: x.Name):
        print(f"     {p.Name} : {p.PropertyType.Name}")
    print("  -- methods --")
    seen = set()
    for m in sorted(t.GetMethods(bf), key=lambda x: x.Name):
        if m.Name.startswith(("get_", "set_", "add_", "remove_")) or m.Name in seen:
            continue
        seen.add(m.Name)
        args = ", ".join(f"{p.ParameterType.Name}" for p in m.GetParameters())
        print(f"     {m.Name}({args}) : {m.ReturnType.Name}")
    print("  -- events --")
    for ev in t.GetEvents(bf):
        print(f"     {ev.Name} : {ev.EventHandlerType.Name}")

TARGETS = ("AutomatedBeamGage",)
by_name = {t.Name: t for t in types}
for name in TARGETS:
    if name in by_name:
        dump(by_name[name])
    else:
        print(f"\n(no type {name})")
print("\ndone.")
