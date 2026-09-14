"""
One video source abstraction: USB webcam, video file, or a PHONE over WiFi.

    VideoSource(0)                              # built-in webcam
    VideoSource("clip.mp4")                     # recorded file
    VideoSource("http://192.168.1.7:8080/video")  # Android IP Webcam app
    VideoSource("http://192.168.1.7:4747/video")  # DroidCam
    VideoSource("rtsp://192.168.1.7:8554/live")   # RTSP camera

=============================================================================
WHY A NETWORK CAMERA NEEDS ITS OWN CODE PATH
=============================================================================

A USB webcam hands you a frame the moment you ask. A phone over WiFi does not,
and two things go wrong if you treat them the same:

1. LATENCY ACCUMULATION -- the killer.
   The stream arrives at a fixed rate whether or not you are ready. Frames
   queue up inside OpenCV's buffer. If the network delivers 30 fps and you
   process 20, you fall behind by 10 frames every second, and the backlog only
   grows. After a minute you are looking at video from 30 seconds ago.

   For a drowsiness alarm that is not a glitch, it is a total failure: you
   would be warned about a microsleep that happened half a minute back.

   THE FIX: a background thread reads continuously and keeps only the NEWEST
   frame, throwing older ones away. Dropping frames is correct here -- for a
   real-time safety signal, fresh beats complete. We always work on the most
   recent view of the driver, and the queue can never build up.

2. DROPOUTS.
   WiFi in a moving car is unreliable. A USB camera basically never
   disconnects; a phone stream will. So a read failure must trigger a
   reconnect with backoff, not kill the app.

Also handled here: PHONE ROTATION. A phone clamped to a windscreen mount is
usually sideways, and the stream arrives rotated. MediaPipe needs an upright
face, so we rotate frames before anything else sees them.
"""
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Rotation constants, indexed by degrees.
_ROT = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def is_stream(spec) -> bool:
    """True when the source is a network URL rather than a device or file."""
    return isinstance(spec, str) and spec.split("://", 1)[0].lower() in (
        "http", "https", "rtsp", "rtmp", "udp", "tcp")


def parse_spec(spec: str) -> Union[int, str]:
    """
    "0" -> 0 (device index),  anything else stays a string.

    Argparse gives us strings; OpenCV needs an int for a local device and a
    string for a file or URL.
    """
    if isinstance(spec, int):
        return spec
    s = str(spec).strip()
    return int(s) if s.isdigit() else s


class VideoSource:
    """
    Uniform frame source. Use as a context manager, or call release().

    read() returns (ok, frame) exactly like cv2.VideoCapture, so it drops into
    existing code unchanged.
    """

    def __init__(self, spec=0, width: int = 1280, height: int = 720,
                 rotate: int = 0, mirror: Optional[bool] = None,
                 threaded: Optional[bool] = None, reconnect: bool = True,
                 verbose: bool = True):
        self.spec = parse_spec(spec)
        self.width, self.height = width, height
        self.rotate = int(rotate) % 360
        self.reconnect = reconnect
        self.verbose = verbose

        self.is_net = is_stream(self.spec)
        self.is_file = isinstance(self.spec, str) and not self.is_net

        # Mirror the built-in webcam so moving right moves you right on screen.
        # A phone filming you in a car is NOT a mirror, and a video file must
        # never be flipped, so both default to off.
        self.mirror = (not self.is_file and not self.is_net) if mirror is None \
            else bool(mirror)

        # Threading exists to drop stale frames. A file has no "stale" -- it
        # waits for us -- and threading it would race through to the end.
        self.threaded = (not self.is_file) if threaded is None else bool(threaded)

        self._cap = None
        self._lock = threading.Lock()
        # Waiters block on this instead of polling. The pump notifies once
        # per captured frame, so a consumer sleeps properly between frames
        # rather than waking 500 times a second to check.
        self._cond = threading.Condition(self._lock)
        self._frame = None
        self._seq = 0            # increments per captured frame
        self._last_seq = -1      # last sequence number handed out
        self._stop = threading.Event()
        self._thread = None
        self._fail_count = 0

        self._open()
        if self.threaded:
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------
    def _open(self) -> bool:
        if self._cap is not None:
            self._cap.release()

        if self.is_net:
            # FFMPEG handles http/rtsp; the default backend often cannot.
            self._cap = cv2.VideoCapture(self.spec, cv2.CAP_FFMPEG)
        elif self.is_file:
            self._cap = cv2.VideoCapture(self.spec)
        else:
            # DSHOW opens a Windows webcam almost instantly; the default MSMF
            # backend can take several seconds.
            self._cap = cv2.VideoCapture(self.spec, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            return False

        if not self.is_file:
            # Ask OpenCV to hold a single frame. Support is driver-dependent
            # and often ignored, which is exactly why the reader thread exists
            # as the real defence against latency build-up.
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass

        if not self.is_file and not self.is_net:
            # Ask for MJPG before setting the size. Most USB webcams stream raw
            # YUY2 by default, and raw 720p needs so much USB bandwidth that
            # the camera silently drops to ~10 fps. MJPG is compressed on the
            # camera, so the same 720p arrives at full frame rate. This must be
            # set BEFORE the resolution or the driver ignores it.
            try:
                self._cap.set(cv2.CAP_PROP_FOURCC,
                              cv2.VideoWriter_fourcc(*"MJPG"))
            except cv2.error:
                pass
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            got = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if self.verbose:
                if got != (self.width, self.height):
                    # Cameras do NOT error on an unsupported mode, they quietly
                    # substitute one. Silence here once cost us half our eye
                    # resolution, so always report the mismatch.
                    print(f"[warn] asked for {self.width}x{self.height}, got "
                          f"{got[0]}x{got[1]} - smaller eyes, lower accuracy")
                else:
                    print(f"[ok] capture {got[0]}x{got[1]}")
        return True

    def _grab(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        if self.rotate in _ROT:
            frame = cv2.rotate(frame, _ROT[self.rotate])
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return True, frame

    def _pump(self):
        """
        Background reader. Its entire job is to keep `self._frame` fresh and
        throw away everything older.
        """
        backoff = 0.5
        while not self._stop.is_set():
            ok, frame = self._grab()
            if ok:
                self._fail_count = 0
                backoff = 0.5
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
                continue

            self._fail_count += 1
            if not self.reconnect:
                # Mark the source dead BEFORE leaving. Without this the thread
                # exits with _stop still clear, so every later read() blocks
                # for its full timeout and returns nothing, forever, with no
                # way for the caller to tell "dead" from "slow". Set the flag
                # and wake any waiter so read() returns (False, None) at once.
                self._stop.set()
                with self._cond:
                    self._cond.notify_all()
                break
            # Exponential backoff, capped, so a dead source does not spin the
            # CPU while a briefly flaky one recovers quickly.
            if self.verbose and self._fail_count in (1, 5, 20):
                print(f"[stream] read failed ({self._fail_count}), "
                      f"reconnecting in {backoff:.1f}s ...")
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 1.6, 5.0)
            self._open()

    # ------------------------------------------------------------------
    def read(self, wait: float = 2.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Newest available frame.

        Threaded mode returns the freshest frame the pump has captured, and
        blocks only while waiting for a genuinely new one, so a slow network
        cannot make us process the same frame twice.
        """
        if not self.threaded:
            return self._grab()

        # monotonic, not time.time(): an NTP adjustment or a DST change would
        # otherwise stretch or collapse this deadline.
        deadline = time.monotonic() + wait
        with self._cond:
            while True:
                if self._frame is not None and self._seq != self._last_seq:
                    self._last_seq = self._seq
                    # No copy. cv2.read(), rotate() and flip() each return a
                    # NEW array, and the pump only ever REASSIGNS self._frame,
                    # never writes into an existing one -- so the array handed
                    # out here is never touched again by the pump. The copy
                    # this replaces was 2.7 MB of memcpy per frame at 720p:
                    # ~77 MB/s of pure waste at 28 fps.
                    return True, self._frame
                if self._stop.is_set():
                    return False, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                # Sleeps until the pump notifies or the deadline passes. The
                # old loop was `time.sleep(0.002)` in a busy loop: ~500 wakeups
                # a second on a core that had nothing to do.
                self._cond.wait(remaining)

    @property
    def size(self) -> Tuple[int, int]:
        with self._lock:
            if self._frame is not None:
                h, w = self._frame.shape[:2]
                return w, h
        if self._cap is None:
            return 0, 0
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def describe(self) -> str:
        kind = "phone/network stream" if self.is_net else (
            "video file" if self.is_file else "usb webcam")
        w, h = self.size
        bits = [f"{kind} {self.spec}", f"{w}x{h}"]
        if self.rotate:
            bits.append(f"rotated {self.rotate} deg")
        if self.mirror:
            bits.append("mirrored")
        return "  ".join(bits)

    def release(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
