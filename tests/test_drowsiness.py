"""
Tests for the temporal logic, driven by SIMULATED time.

    python -m tests.test_drowsiness

Why this file exists, and why it is the most useful test in the project:

DrowsinessMonitor.update() accepts an explicit `now` argument instead of always
calling time.time(). That one design choice means we can feed it a fake clock
and simulate a 60-second drive in about 3 milliseconds -- no webcam, no waiting,
no sitting in front of a camera pretending to fall asleep.

Testing "does it alarm after 0.8 s of closed eyes?" by actually closing your
eyes for 0.8 s is slow, unrepeatable, and impossible to run automatically.
Testing it with a fake clock is instant and exact.

The general lesson: whenever code depends on the current time, pass the time IN
rather than reading the clock inside. It costs one parameter and makes the code
testable forever.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG                            # noqa: E402
from src.drowsiness import DrowsinessMonitor, Level   # noqa: E402

D = CFG.drowsy   # durations come from config, never hard-coded here

FPS = 30.0
DT = 1.0 / FPS

_passed, _failed = 0, 0


def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}   {detail}")


def run(monitor, seconds, t0, *, ear, mar=0.1, pitch=0.0, yaw=0.0,
        closed_prob=None):
    """Feed `seconds` of identical frames at 30 FPS. Returns (state, end_time)."""
    st = None
    n = int(seconds * FPS)
    for i in range(n):
        st = monitor.update(face_found=True, closed_prob=closed_prob, ear=ear,
                            mar=mar, pitch=pitch, yaw=yaw, now=t0 + i * DT)
    return st, t0 + n * DT


EAR_OPEN, EAR_SHUT = 0.30, 0.08


def test_awake():
    print("\n[1] eyes open for 20 s -> stays AWAKE")
    m = DrowsinessMonitor()
    st, _ = run(m, 20, 1000.0, ear=EAR_OPEN)
    check("level is AWAKE", st.level == Level.AWAKE, f"got {st.level.name}")
    check("PERCLOS ~0", st.perclos < 0.01, f"got {st.perclos:.3f}")
    check("not flagged closed", not st.closed)


def test_normal_blinks_do_not_alarm():
    print("\n[2] realistic blinking (0.2 s every 4 s) -> must NOT alarm")
    m = DrowsinessMonitor()
    t = 1000.0
    for _ in range(8):
        _, t = run(m, 3.8, t, ear=EAR_OPEN)
        st, t = run(m, 0.2, t, ear=EAR_SHUT)
    check("still AWAKE", m.state.level == Level.AWAKE,
          f"got {m.state.level.name}")
    check("PERCLOS below warn", m.state.perclos < 0.15,
          f"got {m.state.perclos:.3f}")
    check("blinks counted", m.state.blinks >= 6, f"got {m.state.blinks}")
    print(f"        -> PERCLOS {m.state.perclos*100:.1f}%, "
          f"{m.state.blinks} blinks counted")


def test_microsleep():
    secs = D.microsleep_sec + 0.4
    print(f"\n[3] eyes shut {secs:.1f} s -> CRITICAL "
          f"(threshold {D.microsleep_sec:.1f} s)")
    m = DrowsinessMonitor()
    _, t = run(m, 5, 1000.0, ear=EAR_OPEN)
    st, _ = run(m, secs, t, ear=EAR_SHUT)
    check("microsleep flag set", st.microsleep)
    check("level is CRITICAL", st.level == Level.CRITICAL,
          f"got {st.level.name}")
    check("reason mentions microsleep",
          any("MICROSLEEP" in r for r in st.reasons), f"got {st.reasons}")
    check("not counted as a blink", st.blinks == 0, f"got {st.blinks}")


def test_perclos_slow_slide():
    print("\n[4] slow fatigue: 0.5 s closed every 1.5 s -> DROWSY via PERCLOS")
    m = DrowsinessMonitor()
    t = 1000.0
    for _ in range(20):
        _, t = run(m, 1.0, t, ear=EAR_OPEN)
        _, t = run(m, 0.5, t, ear=EAR_SHUT)
    st = m.state
    check("PERCLOS above warn threshold", st.perclos >= 0.15,
          f"got {st.perclos:.3f}")
    check("level at least DROWSY", st.level >= Level.DROWSY,
          f"got {st.level.name}")
    print(f"        -> PERCLOS {st.perclos*100:.1f}% (expected ~33%)")


def test_yawns():
    secs = D.yawn_min_sec + 0.6
    print(f"\n[5] three {secs:.1f} s yawns in a minute -> DROWSY")
    m = DrowsinessMonitor()
    t = 1000.0
    for _ in range(3):
        _, t = run(m, secs, t, ear=EAR_OPEN, mar=0.75)
        _, t = run(m, 4.0, t, ear=EAR_OPEN, mar=0.10)
    st = m.state
    check("3 yawns counted", st.yawns == 3, f"got {st.yawns}")
    check("level at least DROWSY", st.level >= Level.DROWSY,
          f"got {st.level.name}")


def test_talking_is_not_a_yawn():
    print("\n[6] talking (0.3 s mouth movements) -> must NOT count as yawns")
    m = DrowsinessMonitor()
    t = 1000.0
    for _ in range(15):
        _, t = run(m, 0.3, t, ear=EAR_OPEN, mar=0.70)
        _, t = run(m, 0.4, t, ear=EAR_OPEN, mar=0.15)
    check("zero yawns counted", m.state.yawns == 0, f"got {m.state.yawns}")
    print("        -> the yawn_min_sec duration gate rejected all of them")


def test_head_nod():
    secs = D.nod_min_sec + 0.5
    print(f"\n[7] head down {secs:.1f} s -> nodding "
          f"(threshold {D.nod_min_sec:.1f} s)")
    m = DrowsinessMonitor()
    _, t = run(m, 3, 1000.0, ear=EAR_OPEN)
    st, _ = run(m, secs, t, ear=EAR_OPEN, pitch=-25.0)
    check("nodding flag set", st.nodding)
    check("level at least DROWSY", st.level >= Level.DROWSY,
          f"got {st.level.name}")


def test_looking_away():
    secs = D.distract_min_sec + 0.5
    print(f"\n[8] head turned 45 deg for {secs:.1f} s -> DISTRACTED, not DROWSY")
    m = DrowsinessMonitor()
    _, t = run(m, 3, 1000.0, ear=EAR_OPEN)
    st, _ = run(m, secs, t, ear=EAR_OPEN, yaw=45.0)
    check("distracted flag set", st.distracted)
    check("level is DISTRACTED", st.level == Level.DISTRACTED,
          f"got {st.level.name}")


def test_frame_rate_independence():
    secs = D.microsleep_sec + 0.3
    print(f"\n[9] SAME {secs:.1f} s closure at 10 / 30 / 60 FPS "
          f"-> identical verdict")
    results = {}
    for fps in (10, 30, 60):
        m = DrowsinessMonitor()
        dt, t = 1.0 / fps, 1000.0
        for i in range(int(5 * fps)):
            m.update(face_found=True, ear=EAR_OPEN, now=t + i * dt)
        t += 5.0
        st = None
        for i in range(int(secs * fps)):
            st = m.update(face_found=True, ear=EAR_SHUT, now=t + i * dt)
        results[fps] = st.level
    check("all frame rates agree", len(set(results.values())) == 1,
          f"got {[(k, v.name) for k, v in results.items()]}")
    check("all report CRITICAL",
          all(v == Level.CRITICAL for v in results.values()),
          f"got {[(k, v.name) for k, v in results.items()]}")
    print(f"        -> {[(k, v.name) for k, v in results.items()]}")
    print("        this is WHY we measure seconds and not frames")


def test_cnn_ear_fusion():
    print("\n[10] fusion: CNN and EAR disagreeing")
    m = DrowsinessMonitor(cnn_weight=0.65)
    st = m.update(face_found=True, closed_prob=0.99, ear=EAR_SHUT, now=1000.0)
    check("both say closed -> closed", st.closed, f"score {st.closed_score:.2f}")
    m.reset()
    st = m.update(face_found=True, closed_prob=0.01, ear=EAR_OPEN, now=1000.0)
    check("both say open -> open", not st.closed, f"score {st.closed_score:.2f}")
    m.reset()
    # CNN carries 65% of the weight, so it should win a disagreement.
    st = m.update(face_found=True, closed_prob=0.97, ear=EAR_OPEN, now=1000.0)
    check("CNN outvotes EAR at weight 0.65", st.closed,
          f"score {st.closed_score:.2f}")


def test_perclos_warmup():
    print(f"\n[12] PERCLOS must not fire before "
          f"{D.perclos_min_obs_sec:.0f} s of data exist")
    m = DrowsinessMonitor()
    # 3 s open then 1.5 s closed = 33% PERCLOS, but only 4.5 s observed.
    _, t = run(m, 3.0, 1000.0, ear=EAR_OPEN)
    st, _ = run(m, 1.5, t, ear=EAR_SHUT)
    check("PERCLOS reads high", st.perclos > 0.25, f"got {st.perclos:.2f}")
    check("but did NOT trigger on PERCLOS",
          not any("PERCLOS" in r and "warming" not in r for r in st.reasons),
          f"got {st.reasons}")
    print(f"        -> PERCLOS {st.perclos*100:.0f}% ignored: {st.reasons[-1]}")


def test_face_lost():
    print("\n[11] face disappears -> NO_FACE after the grace period")
    m = DrowsinessMonitor()
    _, t = run(m, 3, 1000.0, ear=EAR_OPEN)
    st = m.update(face_found=False, now=t + 0.5)
    check("grace period holds previous state", st.level != Level.NO_FACE,
          f"got {st.level.name}")
    st = m.update(face_found=False, now=t + 5.0)
    check("eventually reports NO_FACE", st.level == Level.NO_FACE,
          f"got {st.level.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("DROWSINESS STATE MACHINE - simulated-time tests")
    print("=" * 60)
    for fn in (test_awake, test_normal_blinks_do_not_alarm, test_microsleep,
               test_perclos_slow_slide, test_yawns, test_talking_is_not_a_yawn,
               test_head_nod, test_looking_away, test_frame_rate_independence,
               test_cnn_ear_fusion, test_perclos_warmup, test_face_lost):
        fn()
    print("\n" + "=" * 60)
    print(f"{_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)
