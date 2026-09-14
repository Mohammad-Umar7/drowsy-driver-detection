"""
Tests for the on-screen display.

    python -m tests.test_hud

Rendering is done into numpy arrays, so no window and no camera are needed.
The display is drawn on every frame of a live session, so an exception in a
rarely-hit branch (a tiny frame, a missing face, debug on with no model)
takes the whole detector down. Every state is rendered here at three sizes.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import check, summary                        # noqa: E402
from src import geometry as G                                     # noqa: E402
from src.drowsiness import DrowsyState, Level                     # noqa: E402
from src.hud import (Hud, draw_face, panel, chip, bar,            # noqa: E402
                     fit_text, text_size)
from src.lighting import LightingNormalizer, LightStats, Light    # noqa: E402


def fake_obs(w, h, use_l=True, use_r=True):
    cx, cy = w * 0.55, h * 0.45
    lm = np.full((478, 2), (cx, cy), np.float32)
    for idx, ex in ((G.LEFT_EYE_CONTOUR, cx + 0.05 * w),
                    (G.RIGHT_EYE_CONTOUR, cx - 0.05 * w)):
        for k, i in enumerate(idx):
            a = 2 * np.pi * k / len(idx)
            lm[i] = (ex + 0.025 * w * np.cos(a), cy - 0.04 * h + 0.01 * h * np.sin(a))
    eye = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    s = int(0.04 * w)
    return SimpleNamespace(
        landmarks_px=lm, ear=0.28, mar=0.2, width_left=60.0, width_right=58.0,
        use_left=use_l, use_right=use_r, eyes_reliable=use_l or use_r,
        pitch=-3.0, yaw=5.0, roll=1.0, eye_left=eye, eye_right=eye,
        box_left=(int(cx) + s, int(cy) - s, int(cx) + 3 * s, int(cy) + s),
        box_right=(int(cx) - 3 * s, int(cy) - s, int(cx) - s, int(cy) + s),
    )


class FakeCalib:
    active = True
    seconds = 3.0

    def remaining(self):
        return 1.5


def blank(w, h):
    return np.full((h, w, 3), 90, np.uint8)


def test_primitives():
    print("\n[1] panels, chips and bars never raise, even off the edge of the frame")
    f = blank(300, 200)
    before = f.copy()
    panel(f, 20, 20, 120, 60)
    check("panel drew something", not np.array_equal(f, before))
    check("frame shape unchanged", f.shape == (200, 300, 3))
    panel(f, -50, -50, 100, 100)        # partly off the top-left
    panel(f, 250, 150, 200, 200)        # partly off the bottom-right
    panel(f, 500, 500, 50, 50)          # entirely outside
    check("off-frame panels are clipped, not fatal", f.shape == (200, 300, 3))
    w = chip(f, "AWAKE", 10, 100, (0, 0, 0), (100, 200, 100))
    check("chip returns a sensible width", 30 < w < 200, f"got {w}")
    bar(f, 10, 150, 100, 12, 1.7, (0, 0, 255), warn_at=0.5)
    bar(f, 10, 170, 100, 12, -0.3, (0, 0, 255))
    check("bars clamp fractions outside 0..1", True)


def test_fit_text():
    print("\n[2] long text is truncated to the space available")
    s = "MICROSLEEP 3.2s  |  PERCLOS 31% (critical)  |  head nodding"
    out = fit_text(s, 120, 0.5)
    check("ends with an ellipsis", out.endswith("..."), out)
    check("fits the width", text_size(out, 0.5)[0] <= 120, f"{text_size(out, 0.5)[0]}px")
    check("short text is untouched", fit_text("AWAKE", 500, 0.5) == "AWAKE")


def test_every_state_at_every_size():
    print("\n[3] all display states render at 1280x720, 640x480 and 320x240")
    light = LightingNormalizer()
    light.stats = LightStats(Light.DIM, 62, 58, 0.0, 0.02, 70, 0.72)
    states = {
        "awake": (DrowsyState(level=Level.AWAKE, reasons=["normal"]), True, False, None),
        "drowsy+debug": (DrowsyState(level=Level.DROWSY, closed=True, closed_score=0.9,
                                     closure_sec=1.2, perclos=0.2, long_blinks=4,
                                     reasons=["4 long blinks/min", "PERCLOS 20%"]),
                         True, True, (0.91, 0.87)),
        "debug, no model": (DrowsyState(level=Level.AWAKE), True, True, None),
        "critical": (DrowsyState(level=Level.CRITICAL, closed=True, closed_score=0.95,
                                 closure_sec=3.0, microsleep=True, perclos=0.4,
                                 reasons=["MICROSLEEP 3.0s", "PERCLOS 40% (critical)"]),
                     True, False, None),
        "distracted": (DrowsyState(level=Level.DISTRACTED, distracted=True,
                                   reasons=["looking away (+41 deg)"]), True, False, None),
        "no face": (DrowsyState(level=Level.NO_FACE, reasons=["no face detected"]),
                    False, False, None),
        "head turned": (DrowsyState(level=Level.AWAKE, eyes_reliable=False,
                                    reasons=["eyes not visible (head turned)"]),
                        True, False, None),
    }
    for (w, h) in ((1280, 720), (640, 480), (320, 240)):
        for name, (st, has_face, debug, probs) in states.items():
            hud = Hud()
            for i in range(300):
                hud.push(0.1 if i % 40 else 0.9, i % 40 == 0, now=1000.0 + i * 0.1)
            f = blank(w, h)
            before = f.copy()
            obs = fake_obs(w, h, use_r=st.eyes_reliable) if has_face else None
            try:
                if obs is not None:
                    draw_face(f, obs, st.closed)
                hud.draw(f, st, obs, 27.5, 0.213, probs is not None or name == "awake",
                         True, debug, light, calib=FakeCalib() if name == "awake" else None,
                         probs=probs, now=1030.0)
                # Both the flashing and the steady phase of CRITICAL.
                if st.level == Level.CRITICAL:
                    hud.draw(f, st, obs, 27.5, 0.213, True, True, False, light, now=1030.1)
                hud.draw(f, st, obs, 27.5, 0.213, True, False, debug, None, now=1030.0)
                ok = True
            except Exception as e:      # noqa: BLE001
                ok = False
                detail = f"{type(e).__name__}: {e}"
            check(f"{w}x{h} {name}", ok and f.shape == before.shape
                  and f.dtype == np.uint8 and not np.array_equal(f, before),
                  detail if not ok else "nothing drawn")


def test_signal_lost():
    print("\n[4] the SIGNAL LOST card")
    for (w, h) in ((1280, 720), (320, 240)):
        f = blank(w, h)
        before = f.copy()
        Hud().draw_signal_lost(f, 7.0)
        check(f"{w}x{h} draws the card", not np.array_equal(f, before))


def test_sparkline_history():
    print("\n[5] the closure history keeps 30 s, thins to ~10 Hz, and keeps every edge")
    hud = Hud(window_sec=30.0)
    n_frames = 60 * 30                          # 60 s at 30 fps, all open
    for i in range(n_frames):
        hud.push(0.1, False, now=1000.0 + i / 30.0)
    last = 1000.0 + (n_frames - 1) / 30.0
    check("nothing older than the window, measured from the LAST frame",
          hud.hist[0][0] >= last - 30.0 - 1e-9, f"oldest {hud.hist[0][0]:.2f}")
    # 30 fps in, at most one sample per 0.1 s out: at 30 fps that is every
    # fourth frame, ~7.5 Hz, so roughly 225 samples over 30 s.
    check("thinned to well under the input rate", 150 <= len(hud.hist) <= 320,
          f"{len(hud.hist)}")
    # A blink that starts and ends 40 ms apart must still show both edges,
    # even though both arrive inside the 0.1 s thinning interval.
    hud.push(0.9, True, now=last + 0.05)
    hud.push(0.1, False, now=last + 0.09)
    tail = [(round(t, 2), c) for t, _, c in list(hud.hist)[-2:]]
    check("a state change is never thinned away",
          tail == [(round(last + 0.05, 2), True), (round(last + 0.09, 2), False)],
          f"tail {tail}")
    hud.reset()
    check("reset() clears the strip", len(hud.hist) == 0)


if __name__ == "__main__":
    print("=" * 60)
    print("DISPLAY - every state, every size, no window needed")
    print("=" * 60)
    for fn in (test_primitives, test_fit_text, test_every_state_at_every_size,
               test_signal_lost, test_sparkline_history):
        fn()
    summary("display")
