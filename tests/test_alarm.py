"""
Tests for the alarm: the synthesised sounds and the escalation logic.

    python -m tests.test_alarm

Nothing here makes a noise. The player and the speaker are replaced with
fakes that record what they were asked to play, so the escalation, the
"never block the video loop" rule and the never-overlap guard can be checked
exactly, and the waveforms are inspected as numbers.
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import check, summary, wait_until            # noqa: E402
from src import alarm as A                                        # noqa: E402
from src.alarm import Alarm, render, to_wav, duration_sec         # noqa: E402


def dominant_hz(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1.0 / A.SAMPLE_RATE)[int(spec.argmax())])


def test_waveforms():
    print("\n[1] the three patterns are the right length and click-free")
    d = render("distract")
    w = render("drowsy")
    c0 = render("critical", 0)
    c2 = render("critical", 2)
    check("distract is a short soft chirp", 0.2 < duration_sec(d) < 0.5,
          f"{duration_sec(d):.2f}s")
    check("distract is quiet", 0.2 < np.abs(d).max() <= 0.4, f"peak {np.abs(d).max():.2f}")
    check("drowsy is under a second", 0.7 < duration_sec(w) < 1.0, f"{duration_sec(w):.2f}s")
    check("critical level 0 is two siren cycles",
          abs(duration_sec(c0) - (2 * 2 * A.SIREN_SWEEP_MS + 40) / 1000) < 0.01,
          f"{duration_sec(c0):.2f}s")
    check("critical level 2 is four cycles", duration_sec(c2) > duration_sec(c0) * 1.9,
          f"{duration_sec(c2):.2f}s vs {duration_sec(c0):.2f}s")
    check("critical is full scale", 0.9 < np.abs(c0).max() <= 1.0)
    for name, x in (("distract", d), ("drowsy", w), ("critical", c0)):
        check(f"{name} never clips", np.abs(x).max() <= 1.0)
        check(f"{name} starts and ends silent (no click)",
              abs(x[0]) < 0.02 and abs(x[-1]) < 0.02, f"{x[0]:.3f} .. {x[-1]:.3f}")
        check(f"{name} is float32", x.dtype == np.float32)


def test_siren_actually_sweeps():
    print("\n[2] the critical siren sweeps low -> high -> low")
    c = render("critical", 0)
    sweep = int(A.SAMPLE_RATE * A.SIREN_SWEEP_MS / 1000)
    win = int(A.SAMPLE_RATE * 0.06)
    start = dominant_hz(c[:win])
    peak = dominant_hz(c[sweep - win:sweep])
    end = dominant_hz(c[2 * sweep - win:2 * sweep])
    check("starts near the low pitch", abs(start - A.SIREN_LOW_HZ) < 120, f"{start:.0f} Hz")
    check("reaches the high pitch", abs(peak - A.SIREN_HIGH_HZ) < 150, f"{peak:.0f} Hz")
    check("comes back down", abs(end - A.SIREN_LOW_HZ) < 120, f"{end:.0f} Hz")


def test_wav_container():
    print("\n[3] the WAV bytes are a valid mono 16-bit file")
    import io
    import wave
    b = to_wav(render("drowsy"))
    with wave.open(io.BytesIO(b), "rb") as w:
        check("mono", w.getnchannels() == 1)
        check("16-bit", w.getsampwidth() == 2)
        check("sample rate", w.getframerate() == A.SAMPLE_RATE)
        check("frame count matches", w.getnframes() == len(render("drowsy")))


class FakePlayer:
    def __init__(self, hold=0.0):
        self.played = []
        self.hold = hold

    def __call__(self, wav):
        self.played.append(len(wav))
        time.sleep(self.hold)


def test_escalation():
    print("\n[4] consecutive critical alarms escalate, and reset after a quiet spell")
    spoken = []
    a = Alarm(player=FakePlayer(), speaker=spoken.append)
    levels = []
    t = 1000.0
    for _ in range(5):
        a.fire("critical", now=t)
        levels.append(a.last[1])
        wait_until(lambda: not a.playing)
        t += 4.0                                  # the alarm cooldown
    top = A.CRITICAL_REPEATS_MAX - A.CRITICAL_REPEATS_MIN
    check("levels climb and cap", levels == [0, 1, top, top, top], f"{levels}")
    check("the voice escalates too", spoken[0] == Alarm.VOICE[0] and spoken[1] == Alarm.VOICE[1],
          f"{spoken[:2]}")
    a.fire("critical", now=t + A.ESCALATION_WINDOW_SEC + 1.0)
    check("a quiet spell resets to level 0", a.last[1] == 0, f"{a.last}")
    wait_until(lambda: not a.playing)
    a.fire("drowsy", now=t + 100.0)
    wait_until(lambda: not a.playing)
    check("drowsy and distract never speak", len(spoken) == 6, f"{len(spoken)} utterances")


def test_never_blocks_and_never_overlaps():
    print("\n[5] fire() returns at once, and a pattern in progress is not stacked on")
    p = FakePlayer(hold=1.0)
    a = Alarm(player=p, speaker=lambda s: None)
    t0 = time.monotonic()
    started = a.fire("drowsy")
    dt = time.monotonic() - t0
    check("started", started)
    # The fake player holds for a full second; a fire() that blocked on it
    # could not possibly return this fast, even on a heavily loaded machine.
    check("returned immediately", dt < 0.3, f"{dt * 1000:.0f} ms")
    check("a second fire while playing is dropped", not a.fire("drowsy"))
    check("the pattern finishes", wait_until(lambda: not a.playing, timeout=10.0))
    check("only one pattern was played", len(p.played) == 1, f"{len(p.played)}")
    check("fires again once free", a.fire("distract"))
    check("second pattern played", wait_until(lambda: len(p.played) == 2, timeout=10.0))


def test_muted_and_dead_thread():
    print("\n[6] muted plays nothing; a thread that cannot start does not kill the alarm")
    p = FakePlayer()
    # Every Alarm in this file gets a fake speaker: the real one on Windows
    # spawns PowerShell to pre-render speech, which is load the suite does
    # not need and which made the timing checks flaky.
    a = Alarm(enabled=False, player=p, speaker=lambda s: None)
    check("muted fire() plays nothing", not a.fire("critical") and not p.played)
    a.toggle()
    real_thread = threading.Thread

    class Broken(real_thread):
        def start(self):
            raise RuntimeError("can't start new thread")
    A.threading.Thread = Broken
    try:
        check("failed thread start is reported", not a.fire("drowsy"))
        check("and the playing flag is released", not a.playing)
    finally:
        A.threading.Thread = real_thread
    check("the next fire works", a.fire("drowsy"))
    check("and played", wait_until(lambda: len(p.played) == 1))


class StoppablePlayer:
    """Holds for `hold` seconds unless stop() cuts it short, like a real device."""

    def __init__(self, hold):
        self.hold = hold
        self.played = []            # how long each pattern actually ran
        self.stops = 0
        self._cut = threading.Event()

    def __call__(self, wav):
        self._cut.clear()
        t0 = time.monotonic()
        self._cut.wait(self.hold)
        self.played.append(time.monotonic() - t0)

    def stop(self):
        self.stops += 1
        self._cut.set()


def test_critical_preempts_a_lesser_pattern():
    print("\n[8] a CRITICAL siren cuts a drowsy nudge short; nothing else interrupts")
    p = StoppablePlayer(hold=0.8)
    a = Alarm(player=p, speaker=lambda s: None)
    check("drowsy starts", a.fire("drowsy"))
    time.sleep(0.1)
    check("critical is accepted while the nudge plays", a.fire("critical"))
    check("but a drowsy fire during the siren is dropped", not a.fire("drowsy"))
    check("and a critical never interrupts a critical", not a.fire("critical"))
    check("both patterns ran", wait_until(lambda: len(p.played) == 2, timeout=6.0)
          and wait_until(lambda: not a.playing, timeout=6.0))
    check("the nudge was cut short", p.played[0] < 0.5, f"ran {p.played[0]:.2f}s")
    check("by exactly one stop()", p.stops == 1, f"{p.stops}")
    check("the siren played in full", p.played[1] >= 0.75, f"ran {p.played[1]:.2f}s")
    check("last played is the siren", a.last == ("critical", 0), f"{a.last}")


def test_a_player_that_raises_is_survived():
    print("\n[7] audio errors never propagate, and the flag is always released")
    def bad(wav):
        raise OSError("no audio device")
    a = Alarm(player=bad, speaker=lambda s: None)
    a.fire("critical")
    check("flag released after the error", wait_until(lambda: not a.playing))
    check("can fire again", a.fire("drowsy"))


if __name__ == "__main__":
    print("=" * 60)
    print("ALARM - synthesis, escalation, non-blocking guarantee")
    print("=" * 60)
    for fn in (test_waveforms, test_siren_actually_sweeps, test_wav_container,
               test_escalation, test_never_blocks_and_never_overlaps,
               test_muted_and_dead_thread, test_critical_preempts_a_lesser_pattern,
               test_a_player_that_raises_is_survived):
        fn()
    summary("alarm")
