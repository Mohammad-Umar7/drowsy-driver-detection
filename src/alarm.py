"""
Non-blocking audible alarm.

The one rule that matters here: NEVER block the video loop.

If you call a 1-second beep directly in the main loop, the camera stops being
read for that whole second. Frames pile up in the driver buffer, latency spikes,
PERCLOS timing distorts, and the app visibly stutters exactly when the driver
most needs it working.

So every sound is played on a BACKGROUND THREAD. The main loop fires it and
immediately carries on to the next frame.

daemon=True means these threads will not keep Python alive if the main program
exits -- without it, closing the app while a beep is playing would hang.
"""
import threading
import time

try:
    import winsound                    # Windows only, part of the stdlib
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False


class Alarm:
    """
    Escalating alarm patterns.

    DROWSY   : two mid-pitch beeps  - a nudge
    CRITICAL : fast high-pitched siren - impossible to ignore

    A single _playing flag prevents overlapping sounds when triggers stack up.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._playing = False
        self._lock = threading.Lock()

    def _play(self, pattern):
        try:
            for freq, ms in pattern:
                if _HAS_WINSOUND:
                    winsound.Beep(freq, ms)
                else:
                    # Terminal bell: works on macOS/Linux, silent on some
                    # terminals, but never crashes.
                    print("\a", end="", flush=True)
                    time.sleep(ms / 1000.0)
        except Exception:
            pass                        # audio must never crash the detector
        finally:
            with self._lock:
                self._playing = False

    def fire(self, critical: bool = False):
        if not self.enabled:
            return
        with self._lock:
            if self._playing:
                return
            self._playing = True
        pattern = ([(1500, 180), (1900, 180), (1500, 180), (1900, 300)]
                   if critical else [(900, 200), (700, 200)])
        threading.Thread(target=self._play, args=(pattern,), daemon=True).start()

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled
