"""
Verify the Kotlin port has not drifted from the Python it was ported from.

    python scripts/check_port_parity.py

=============================================================================
WHY THIS EXISTS
=============================================================================

The Android app is a hand port of the Python. Two copies of the same logic
drift, and the drift is SILENT: nothing crashes, no test fails, the phone just
quietly behaves differently from the version that was measured and tuned.

This was not hypothetical. Porting lighting.py, the DARK branch

    mean < 45 or (face_mean < 40 and contrast < 60)

was written in Kotlin as

    mean < 45 || faceMean < 40

dropping the contrast term. The effect: a normally-lit frame whose CENTRE
happens to be dark - dark clothing, a beard, a shadow - would be classified as
night on the phone and median-blurred for no reason, while the desktop
classified the same frame as NORMAL. Nothing would have reported that.

So the constants and the landmark indices are compared mechanically. Those are
the parts where a silent one-character difference does the most damage: every
threshold was tuned against measured data, and the landmark indices decide
which points on the face are even being looked at.

Logic still has to be reviewed by eye. This catches the mechanical half.
"""
import re
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DrowsyCfg          # noqa: E402
from src import geometry as G             # noqa: E402
from src import lighting as L             # noqa: E402

KT_DROWSY = ROOT / "android/app/src/main/java/com/umar/drowsy/Drowsiness.kt"
KT_GEOM = ROOT / "android/app/src/main/java/com/umar/drowsy/Geometry.kt"
KT_LIGHT = ROOT / "android/app/src/main/java/com/umar/drowsy/Lighting.kt"

# The lighting stage's decision constants. lighting.py defines each as a
# module-level name; Lighting.kt keeps the same names in `object LightCfg`.
LIGHT_CONSTS = [
    "COMFORT_LOW", "COMFORT_HIGH", "GAMMA_MIN", "GAMMA_MAX", "GAMMA_DEADBAND",
    "GAMMA_SMOOTH", "CLAHE_STRONG_CLIP", "CLAHE_SOFT_CLIP", "CLAHE_TILES",
    "BACKLIT_CLIP_FRAC", "BACKLIT_FACE_DROP", "DARK_MEAN", "DARK_FACE_MEAN",
    "DARK_MIN_CONTRAST", "DIM_MEAN", "BRIGHT_MEAN", "BRIGHT_CLIP_FRAC",
]

# Desktop-only settings with no phone equivalent, and why.
SKIP = {
    # The phone has no saved calibration file yet, so the ratio is used but
    # the grace period for a lost face is handled by CameraX lifecycle.
}


def parse_kotlin_consts(path: Path) -> dict:
    """Pull `const val NAME = 1.23` out of the Kotlin source."""
    text = path.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"const\s+val\s+([A-Z0-9_]+)\s*[:\w]*\s*=\s*([-\d.]+)", text):
        raw = m.group(2)
        out[m.group(1)] = float(raw) if "." in raw else int(raw)
    return out


def parse_kotlin_int_arrays(path: Path) -> dict:
    """Pull `val NAME = intArrayOf(1, 2, 3)` out of the Kotlin source."""
    text = path.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"val\s+([A-Z0-9_]+)\s*=\s*intArrayOf\(([^)]*)\)",
                         text, re.S):
        nums = [int(x) for x in re.findall(r"-?\d+", m.group(2))]
        out[m.group(1)] = nums
    return out


def main():
    problems = []
    checked = 0

    # ---- 1. thresholds -----------------------------------------------
    kt = parse_kotlin_consts(KT_DROWSY)
    if not kt:
        sys.exit(f"parsed no constants from {KT_DROWSY} - has it moved?")

    print("thresholds  (config.py DrowsyCfg  vs  Drowsiness.kt Cfg)")
    print("-" * 66)
    for f in fields(DrowsyCfg):
        if f.name in SKIP:
            continue
        kt_name = f.name.upper()
        py_val = f.default
        checked += 1
        if kt_name not in kt:
            problems.append(f"{f.name}: MISSING from Kotlin (expected {kt_name})")
            print(f"  {f.name:26s} {py_val!s:>8}  ->  MISSING")
            continue
        kt_val = kt[kt_name]
        same = abs(float(py_val) - float(kt_val)) < 1e-9
        print(f"  {f.name:26s} {py_val!s:>8}  ->  {kt_val!s:<8} "
              f"{'ok' if same else 'MISMATCH'}")
        if not same:
            problems.append(f"{f.name}: python {py_val} != kotlin {kt_val}")

    # ---- 2. landmark indices -----------------------------------------
    # A single wrong index here silently measures the wrong part of the face.
    print("\nlandmark indices  (geometry.py  vs  Geometry.kt)")
    print("-" * 66)
    kt_arrays = parse_kotlin_int_arrays(KT_GEOM)
    for name in ("RIGHT_EYE_EAR", "LEFT_EYE_EAR", "RIGHT_EYE_CONTOUR",
                 "LEFT_EYE_CONTOUR", "MOUTH_UPPER", "MOUTH_LOWER",
                 "MOUTH_CORNERS"):
        py_val = list(getattr(G, name))
        checked += 1
        if name not in kt_arrays:
            problems.append(f"{name}: MISSING from Kotlin")
            print(f"  {name:20s} MISSING from Kotlin")
            continue
        kt_val = kt_arrays[name]
        same = py_val == kt_val
        print(f"  {name:20s} {len(py_val):2d} points  "
              f"{'ok' if same else 'MISMATCH'}")
        if not same:
            problems.append(f"{name}: python {py_val} != kotlin {kt_val}")

    # POSE_LANDMARKS is private in Kotlin, so parse it separately.
    m = re.search(r"POSE_LANDMARKS\s*=\s*intArrayOf\(([^)]*)\)",
                  KT_GEOM.read_text(encoding="utf-8"))
    checked += 1
    if m:
        kt_pose = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
        same = kt_pose == list(G.POSE_LANDMARKS)
        print(f"  {'POSE_LANDMARKS':20s} {len(kt_pose):2d} points  "
              f"{'ok' if same else 'MISMATCH'}")
        if not same:
            problems.append(f"POSE_LANDMARKS: python {G.POSE_LANDMARKS} "
                            f"!= kotlin {kt_pose}")
    else:
        problems.append("POSE_LANDMARKS: MISSING from Kotlin")

    # ---- 3. the 3D head model used for pose --------------------------
    kt_text = KT_GEOM.read_text(encoding="utf-8")
    kt_pts = re.findall(r"Point3\(([-\d., ]+)\)", kt_text)
    checked += 1
    if len(kt_pts) == len(G.MODEL_POINTS_3D):
        ok = True
        for row, kt_row in zip(G.MODEL_POINTS_3D, kt_pts):
            vals = [float(x) for x in kt_row.split(",")]
            if any(abs(a - b) > 1e-6 for a, b in zip(row, vals)):
                ok = False
        print(f"  {'MODEL_POINTS_3D':20s} {len(kt_pts):2d} points  "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            problems.append("MODEL_POINTS_3D differs between python and kotlin")
    else:
        problems.append(f"MODEL_POINTS_3D: {len(G.MODEL_POINTS_3D)} points in "
                        f"python, {len(kt_pts)} in kotlin")

    # ---- 4. the lighting stage ---------------------------------------
    # This is the file whose port drifted first (the dropped contrast term),
    # and until now nothing here looked at it.
    print("\nlighting constants  (lighting.py  vs  Lighting.kt LightCfg)")
    print("-" * 66)
    kt_light = parse_kotlin_consts(KT_LIGHT)
    for name in LIGHT_CONSTS:
        py_val = getattr(L, name)
        checked += 1
        if name not in kt_light:
            problems.append(f"{name}: MISSING from Lighting.kt")
            print(f"  {name:26s} {py_val!s:>8}  ->  MISSING")
            continue
        kt_val = kt_light[name]
        same = abs(float(py_val) - float(kt_val)) < 1e-9
        print(f"  {name:26s} {py_val!s:>8}  ->  {kt_val!s:<8} "
              f"{'ok' if same else 'MISMATCH'}")
        if not same:
            problems.append(f"{name}: python {py_val} != kotlin {kt_val}")

    # ---- verdict -----------------------------------------------------
    print("\n" + "=" * 66)
    if problems:
        print(f"{len(problems)} PARITY PROBLEM(S) - the phone will behave "
              f"differently from the desktop:\n")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"{checked} values checked, python and kotlin agree on all of them.")


if __name__ == "__main__":
    main()
