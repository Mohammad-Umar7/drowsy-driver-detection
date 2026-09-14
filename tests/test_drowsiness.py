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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import check, summary             # noqa: E402
from src.config import CFG                            # noqa: E402
from src.drowsiness import (DrowsinessMonitor, EarCalibrator,  # noqa: E402
                            Level)

D = CFG.drowsy   # durations come from config, never hard-coded here

FPS = 30.0
DT = 1.0 / FPS


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


def test_alarm_releases_when_eyes_reopen():
    print("\n[13] CRITICAL must release the moment the eyes reopen")
    m = DrowsinessMonitor()
    # A long closure: PERCLOS pinned at 100%, alarm screaming.
    st, t = run(m, 15.0, 1000.0, ear=EAR_SHUT)
    check("screams while eyes are shut", st.level == Level.CRITICAL,
          f"got {st.level.name}")

    # Eyes open. PERCLOS is still ~75% because it averages 30 seconds, but the
    # driver is visibly awake, so it must stop shouting WAKE UP.
    st, t = run(m, 5.0, t, ear=EAR_OPEN)
    check("PERCLOS still high", st.perclos > 0.5, f"got {st.perclos:.2f}")
    check("but no longer CRITICAL", st.level != Level.CRITICAL,
          f"got {st.level.name}")
    check("still warns DROWSY", st.level == Level.DROWSY,
          f"got {st.level.name}")
    check("reason says recovering",
          any("recovering" in r for r in st.reasons), f"got {st.reasons}")
    print(f"        -> PERCLOS {st.perclos*100:.0f}% but "
          f"{st.reasons[0]} (not screaming)")

    # A normal blink during recovery must not flip it back to CRITICAL.
    st, t = run(m, 0.2, t, ear=EAR_SHUT)
    check("a blink during recovery does not re-trigger",
          st.level != Level.CRITICAL, f"got {st.level.name}")

    # Genuinely closing them again must scream immediately.
    st, _ = run(m, D.microsleep_sec + 0.5, t, ear=EAR_SHUT)
    check("closing them again screams again", st.level == Level.CRITICAL,
          f"got {st.level.name}")


def test_head_turn_is_not_a_closure():
    print("\n[14] a head turn must NEVER be read as closed eyes")
    m = DrowsinessMonitor()
    _, t = run(m, 6.0, 1000.0, ear=EAR_OPEN)

    # Head turned far. face.py reports eyes_reliable=False, and the numbers
    # arriving are garbage -- measured at yaw +70 deg the far eye was 4.6 px
    # wide and returned EAR 1.05 with the CNN scoring it 0.68 "closed".
    # Feed exactly that garbage and confirm it is ignored, not believed.
    st = None
    for i in range(int(5.0 * FPS)):
        st = m.update(face_found=True, closed_prob=0.68, ear=1.05,
                      yaw=70.0, eyes_reliable=False, now=t + i * DT)
    t += 5.0

    check("not flagged closed", not st.closed, f"score {st.closed_score:.2f}")
    check("no microsleep from a head turn", not st.microsleep)
    check("never reaches CRITICAL", st.level != Level.CRITICAL,
          f"got {st.level.name}")
    check("reports looking away instead", st.level == Level.DISTRACTED,
          f"got {st.level.name}")
    check("PERCLOS did not accumulate", st.perclos < 0.05,
          f"got {st.perclos:.3f}")
    check("says why", any("not visible" in r or "looking away" in r
                          for r in st.reasons), f"got {st.reasons}")
    print(f"        -> {st.reasons}")

    # Facing forward again with genuinely shut eyes must still work.
    st, _ = run(m, D.microsleep_sec + 0.4, t, ear=EAR_SHUT)
    check("real closure after the turn still fires",
          st.level == Level.CRITICAL, f"got {st.level.name}")


def test_distraction_alarm_is_gentler():
    print("\n[15] looking away alarms softly, and far less often")
    m = DrowsinessMonitor()
    _, t = run(m, 5.0, 1000.0, ear=EAR_OPEN)
    fired = []
    n = int(30.0 * FPS)
    for i in range(n):
        st = m.update(face_found=True, ear=EAR_OPEN, yaw=50.0,
                      now=t + i * DT)
        if st.should_alarm:
            fired.append(st.alarm_kind)
    check("it does alert", len(fired) > 0, "never alerted at all")
    check("uses the gentle sound", all(k == "distract" for k in fired),
          f"got {set(fired)}")
    # 30 s at a 9 s cooldown is at most 4 nudges, versus 7 on the urgent path.
    check("nags at most 4 times in 30 s", len(fired) <= 4,
          f"fired {len(fired)} times")
    print(f"        -> {len(fired)} gentle nudges in 30 s "
          f"(cooldown {D.distract_alarm_cooldown_sec:.0f}s)")


def test_calibration_waits_for_a_face():
    print("\n[16] calibration must survive an empty room, not abort silently")
    c = EarCalibrator(seconds=0.05, min_samples=30)
    c.start()
    # Nobody in front of the camera yet, so feed() is never called.
    time.sleep(0.15)
    check("still armed after a faceless wait", c.active)

    out = None
    for _ in range(60):
        out = c.feed(0.30)
        if out is not None:
            break
        time.sleep(0.003)
    check("calibrates once the face appears", out is not None,
          f"failed: {c.failed_reason!r}")
    if out is not None:
        expect = 0.30 * D.ear_calib_ratio
        check("threshold derived from the measured EAR",
              abs(out - expect) < 1e-6, f"got {out:.4f}, expected {expect:.4f}")
        print(f"        -> open EAR 0.300 -> threshold {out:.3f}")


def test_calibration_needs_enough_samples():
    print("\n[17] a couple of stray frames must not end calibration")
    c = EarCalibrator(seconds=0.02, min_samples=30)
    c.start()
    time.sleep(0.05)                 # the duration alone is already satisfied
    out = c.feed(0.30)
    check("3 samples is not enough", out is None)
    for _ in range(3):
        out = c.feed(0.30)
    check("still waiting", out is None and c.active)


def test_calibration_refuses_closed_eyes():
    print("\n[18] calibrating on SHUT eyes must be refused, and say so")
    c = EarCalibrator(seconds=0.02, min_samples=30)
    c.start()
    out = None
    for _ in range(60):
        out = c.feed(0.05)           # a shut eye
        time.sleep(0.001)
    check("refuses the bad calibration", out is None)
    check("no longer armed", not c.active)
    check("explains why", "CLOSED" in c.failed_reason, c.failed_reason)
    print(f"        -> {c.failed_reason}")


def test_drowsy_driver_slumping_out_of_frame_keeps_alarming():
    print("\n[19] a DROWSY driver who vanishes from view must NOT silence the alarm")
    m = DrowsinessMonitor()
    _, t = run(m, 5.0, 1000.0, ear=EAR_OPEN)
    st, t = run(m, D.microsleep_sec + 0.5, t, ear=EAR_SHUT)
    check("driver is CRITICAL", st.level == Level.CRITICAL, f"got {st.level.name}")

    # Now the face is gone - the driver has slumped out of the camera's view.
    # Old behaviour: NO_FACE, which ranks below AWAKE and never alarms.
    fired = 0
    st = None
    for i in range(int(10.0 * FPS)):
        st = m.update(face_found=False, now=t + i * DT)
        if st.should_alarm:
            fired += 1
    check("level is HELD at CRITICAL, not dropped to NO_FACE",
          st.level == Level.CRITICAL, f"got {st.level.name}")
    check("alarm keeps firing while the face is missing", fired >= 2,
          f"fired {fired} times in 10 s")
    check("reason says why", any("FACE LOST" in r for r in st.reasons),
          f"got {st.reasons}")
    print(f"        -> {st.reasons[0]}, alarm fired {fired}x in 10 s")

    # After the hold period it must eventually release.
    for i in range(int(10.0 * FPS), int((D.face_lost_hold_sec + 4.0) * FPS)):
        st = m.update(face_found=False, now=t + i * DT)
    check("releases after the hold period", st.level != Level.CRITICAL,
          f"got {st.level.name}")


def test_awake_driver_out_of_view_gets_a_nudge_not_a_siren():
    print("\n[20] an AWAKE driver out of view for a while gets a gentle nudge")
    m = DrowsinessMonitor()
    _, t = run(m, 5.0, 1000.0, ear=EAR_OPEN)
    kinds = set()
    st = None
    for i in range(int((D.face_lost_nudge_sec + 3.0) * FPS)):
        st = m.update(face_found=False, now=t + i * DT)
        if st.should_alarm:
            kinds.add(st.alarm_kind)
    check("level is DISTRACTED", st.level == Level.DISTRACTED,
          f"got {st.level.name}")
    check("only the gentle sound was used", kinds == {"distract"},
          f"got {kinds}")


def test_yawn_timer_does_not_survive_face_loss():
    print("\n[21] a yawn that started before a dropout is not completed on return")
    m = DrowsinessMonitor()
    _, t = run(m, 3.0, 1000.0, ear=EAR_OPEN)
    # Mouth opens, then the face is lost for longer than the grace period.
    _, t = run(m, 0.3, t, ear=EAR_OPEN, mar=0.75)
    for i in range(int((D.face_lost_grace_sec + 1.0) * FPS)):
        m.update(face_found=False, now=t + i * DT)
    t += D.face_lost_grace_sec + 1.0
    # Face returns with the mouth CLOSED. Old code: yawn_since was still set
    # from before the dropout, so this frame "completed" a 3 s yawn.
    st, _ = run(m, 0.5, t, ear=EAR_OPEN, mar=0.10)
    check("no yawn was counted", st.yawns == 0, f"got {st.yawns}")


def test_long_blinks_are_an_early_warning():
    print("\n[22] three 0.8 s closures in a minute -> DROWSY via long blinks")
    m = DrowsinessMonitor()
    t = 1000.0
    for _ in range(3):
        _, t = run(m, 5.2, t, ear=EAR_OPEN)
        _, t = run(m, 0.8, t, ear=EAR_SHUT)     # > blink_max, < microsleep
    st, _ = run(m, 1.0, t, ear=EAR_OPEN)
    check("counted as long blinks", st.long_blinks == 3, f"got {st.long_blinks}")
    check("NOT counted as ordinary blinks", st.blinks == 0, f"got {st.blinks}")
    check("no microsleep fired", not st.microsleep)
    check("PERCLOS alone is below its warn line", st.perclos < D.perclos_warn,
          f"got {st.perclos:.3f}")
    check("level is DROWSY", st.level == Level.DROWSY, f"got {st.level.name}")
    check("reason names the long blinks",
          any("long blinks" in r for r in st.reasons), f"got {st.reasons}")
    print(f"        -> {st.reasons}")


def test_no_nudge_before_anyone_has_been_seen():
    print("\n[23] an empty seat at startup must not be nagged to look at the road")
    m = DrowsinessMonitor()
    fired = 0
    st = None
    # The app has just been opened; nobody has sat down yet.
    for i in range(int((D.face_lost_nudge_sec * 3) * FPS)):
        st = m.update(face_found=False, now=1000.0 + i * DT)
        fired += st.should_alarm
    check("reports NO_FACE, not DISTRACTED", st.level == Level.NO_FACE,
          f"got {st.level.name}")
    check("never alarmed", fired == 0, f"fired {fired} times")

    # Once a driver HAS been seen, vanishing for a while does earn the nudge.
    t = 1000.0 + D.face_lost_nudge_sec * 3
    _, t = run(m, 2.0, t, ear=EAR_OPEN)
    fired = 0
    for i in range(int((D.face_lost_nudge_sec + 2.0) * FPS)):
        st = m.update(face_found=False, now=t + i * DT)
        fired += st.should_alarm
    check("nudges after the driver has been seen and then left",
          fired >= 1 and st.level == Level.DISTRACTED,
          f"fired {fired}, level {st.level.name}")


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
               test_cnn_ear_fusion, test_perclos_warmup,
               test_alarm_releases_when_eyes_reopen,
               test_head_turn_is_not_a_closure,
               test_distraction_alarm_is_gentler,
               test_calibration_waits_for_a_face,
               test_calibration_needs_enough_samples,
               test_calibration_refuses_closed_eyes,
               test_drowsy_driver_slumping_out_of_frame_keeps_alarming,
               test_awake_driver_out_of_view_gets_a_nudge_not_a_siren,
               test_yawn_timer_does_not_survive_face_loss,
               test_long_blinks_are_an_early_warning,
               test_no_nudge_before_anyone_has_been_seen,
               test_face_lost):
        fn()
    summary("drowsiness")
