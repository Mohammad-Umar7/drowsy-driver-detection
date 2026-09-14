"""
Tests for VideoSource: the one class that touches cameras, files and streams.

    python -m tests.test_source

No camera is needed. The file tests write a tiny clip with OpenCV first, and
the stream tests replace the grab routine with a fake, so the reconnect and
dead-source paths run in milliseconds instead of waiting on real hardware.
"""
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import check, summary                       # noqa: E402
from src.source import VideoSource, is_stream, parse_spec        # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="drowsy_src_"))


def write_clip(name="clip.mp4", n=10, fps=20.0, w=64, h=48):
    """A clip whose frames are numbered by brightness, left half dark."""
    p = _TMP / name
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 40 + i * 20, np.uint8)
        f[:, : w // 2] = 10                  # a left/right asymmetry
        vw.write(f)
    vw.release()
    return p


def test_spec_parsing():
    print("\n[1] source specs: device index vs file vs network stream")
    check("'0' is device 0", parse_spec("0") == 0)
    check("an int stays an int", parse_spec(3) == 3)
    check("a filename stays a string", parse_spec("clip.mp4") == "clip.mp4")
    check("http is a stream", is_stream("http://192.168.1.7:8080/video"))
    check("rtsp is a stream", is_stream("rtsp://cam/live"))
    check("a file is not a stream", not is_stream("clip.mp4"))
    check("a Windows path is not a stream", not is_stream(r"C:\clips\drive.mp4"))
    check("a device index is not a stream", not is_stream(0))


def test_file_plays_to_the_end():
    print("\n[2] a video file ends cleanly instead of looking like a lost camera")
    cap = VideoSource(str(write_clip()), verbose=False)
    check("opened", cap.opened)
    check("recognised as a file", cap.is_file and not cap.is_net)
    check("files are not threaded (nothing to drop)", not cap.threaded)
    check("files are not mirrored", not cap.mirror)
    check("not ended before reading", not cap.ended)
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
    check("all 10 frames delivered", n == 10, f"got {n}")
    check("ended is set at EOF", cap.ended)
    check("reads after EOF stay (False, None)", cap.read() == (False, None))
    cap.release()


def test_file_is_paced_to_its_frame_rate():
    print("\n[3] a 20 fps clip takes about half a second for 10 frames, not 0 ms")
    cap = VideoSource(str(write_clip(fps=20.0)), verbose=False)
    t0 = time.monotonic()
    while cap.read()[0]:
        pass
    dt = time.monotonic() - t0
    # 9 intervals of 50 ms plus the first frame. Generous upper bound for a
    # loaded machine; the point is that it is not near zero.
    check("took roughly 0.45 s", 0.40 <= dt <= 1.5, f"took {dt:.2f}s")
    check("describe() reports the pacing", "paced" in cap.describe())
    cap.release()


def test_rotation_and_mirroring():
    print("\n[4] --rotate and --mirror do what they say on a file")
    p = str(write_clip())
    plain = VideoSource(p, verbose=False)
    ok, ref = plain.read()
    plain.release()
    check("reference frame read", ok and ref is not None and ref.shape == (48, 64, 3))

    rot = VideoSource(p, rotate=90, verbose=False)
    ok, f = rot.read()
    rot.release()
    check("rotate=90 swaps width and height", ok and f.shape == (64, 48, 3),
          f"got {None if f is None else f.shape}")
    check("rotated frame matches cv2.rotate",
          ok and np.array_equal(f, cv2.rotate(ref, cv2.ROTATE_90_CLOCKWISE)))

    mir = VideoSource(p, mirror=True, verbose=False)
    ok, f = mir.read()
    mir.release()
    check("mirror=True flips left/right",
          ok and np.array_equal(f, cv2.flip(ref, 1)))
    check("the dark half really moved sides",
          ok and f[:, :32].mean() > f[:, 32:].mean())


def test_missing_file_is_reported_not_hung():
    print("\n[5] a file that does not exist is simply 'not opened'")
    cap = VideoSource(str(_TMP / "does_not_exist.mp4"), verbose=False)
    check("not opened", not cap.opened)
    t0 = time.monotonic()
    ok, _ = cap.read()
    check("read returns False immediately", not ok and time.monotonic() - t0 < 0.5)
    cap.release()


class _Fake(VideoSource):
    """A threaded source whose grab routine we script from the test."""

    def __init__(self, script, **kw):
        self._script = list(script)      # per call: "ok", "fail" or "raise"
        self.grabs = 0
        super().__init__("http://fake/stream", threaded=True, verbose=False, **kw)

    def _open(self):
        return True

    def _grab(self):
        self.grabs += 1
        step = self._script.pop(0) if self._script else "ok"
        if step == "raise":
            raise RuntimeError("corrupt packet")
        if step == "fail":
            return False, None
        return True, np.full((8, 8, 3), self.grabs, np.uint8)


def test_dead_source_does_not_block_readers():
    print("\n[6] a source that cannot reconnect is reported dead at once")
    cap = _Fake(["fail"] * 100, reconnect=False)
    t0 = time.monotonic()
    ok, _ = cap.read(wait=2.0)
    dt = time.monotonic() - t0
    check("read returned False", not ok)
    check("without waiting out the 2 s timeout", dt < 0.5, f"took {dt:.2f}s")
    cap.release()


def test_reader_survives_an_exception_and_reconnects():
    print("\n[7] a backend that RAISES is treated as a failed read, then recovers")
    cap = _Fake(["raise", "raise", "ok"], reconnect=True)
    ok, f = cap.read(wait=5.0)
    check("a frame arrived after the reconnect backoff", ok and f is not None)
    check("the thread is still alive", cap._thread.is_alive())
    ok2, f2 = cap.read(wait=2.0)
    check("and keeps delivering NEW frames", ok2 and int(f2[0, 0, 0]) > int(f[0, 0, 0]))
    cap.release()


def test_newest_frame_wins():
    print("\n[8] read() hands out the newest frame and never the same one twice")
    cap = _Fake([], reconnect=True)
    time.sleep(0.05)                     # let the pump race ahead
    ok, a = cap.read(wait=1.0)
    ok2, b = cap.read(wait=1.0)
    check("two reads give two different frames", ok and ok2 and not np.array_equal(a, b))
    check("the second is newer", ok and ok2 and int(b[0, 0, 0]) > int(a[0, 0, 0]))
    cap.release()
    check("release stops the pump", not cap._thread.is_alive())


if __name__ == "__main__":
    print("=" * 60)
    print("VIDEO SOURCE - files, rotation, streams, dead sources")
    print("=" * 60)
    for fn in (test_spec_parsing, test_file_plays_to_the_end,
               test_file_is_paced_to_its_frame_rate, test_rotation_and_mirroring,
               test_missing_file_is_reported_not_hung,
               test_dead_source_does_not_block_readers,
               test_reader_survives_an_exception_and_reconnects,
               test_newest_frame_wins):
        fn()
    summary("video source")
