"""
The temporal brain: per-frame observations -> an actual drowsiness decision.

=============================================================================
THE CENTRAL INSIGHT OF THIS PROJECT
=============================================================================

    A closed eye is NOT drowsiness.
    A closed eye held for two seconds IS.

You blink roughly every 4 seconds and a blink lasts 100-400 ms. A system that
alarms on any closed frame fires constantly, the driver switches it off, and it
has now made the road LESS safe than having no system at all.

So this file never asks "is the eye closed?". It asks "what has this eye been
doing for the last 30 seconds?".

Four signals are tracked, each catching a different stage of falling asleep:

  1. PERCLOS      the fraction of the last 30 s the eyes were closed.
                  This is the real automotive-industry standard metric and the
                  most heavily validated fatigue measure in the literature.
                  Above ~15% = drowsy. Catches the SLOW slide into sleep.

  2. MICROSLEEP   one unbroken closure >= microsleep_sec (2.0 s by default).
                  This is a person actually falling asleep for a moment. It
                  fires INSTANTLY -- no window, no averaging. Catches the
                  SUDDEN event.

  3. YAWN RATE    yawns per minute. Yawning is an EARLY warning that appears
                  well before the eyes start closing, so it buys warning time.

  4. HEAD NOD     head pitch dropping and staying down. The classic
                  head-bob of someone losing consciousness.

  5. LONG BLINKS  closures longer than a blink (0.45 s) but shorter than a
                  microsleep (2 s). Blink duration climbs well before the
                  eyes stay shut, so a cluster of these is an EARLY warning.

Everything is measured in SECONDS, never in frames. If you count frames, your
thresholds silently change meaning when the frame rate changes -- a laptop on
battery drops to 15 FPS and suddenly "20 frames closed" means 1.3 s instead of
0.66 s. Timestamps make the system frame-rate independent.
"""
import dataclasses
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from .config import CFG, DrowsyCfg


class Level(IntEnum):
    """Ordered so that `max()` and `>` comparisons work naturally."""
    NO_FACE = 0
    AWAKE = 1
    DISTRACTED = 2
    DROWSY = 3
    CRITICAL = 4


LEVEL_TEXT = {
    Level.NO_FACE: "NO FACE",
    Level.AWAKE: "AWAKE",
    Level.DISTRACTED: "LOOK AT THE ROAD",
    Level.DROWSY: "DROWSY",
    Level.CRITICAL: "WAKE UP!",
}

# BGR colours for the on-screen display (OpenCV uses BGR, not RGB).
LEVEL_COLOR = {
    Level.NO_FACE: (150, 150, 150),
    Level.AWAKE: (80, 200, 80),
    Level.DISTRACTED: (0, 200, 255),
    Level.DROWSY: (0, 140, 255),
    Level.CRITICAL: (0, 0, 255),
}


@dataclass
class DrowsyState:
    """A snapshot of everything the monitor currently believes."""
    level: Level = Level.AWAKE
    closed: bool = False
    closed_score: float = 0.0        # fused 0..1, >0.5 means closed
    perclos: float = 0.0             # 0..1 over the rolling window
    closure_sec: float = 0.0         # length of the CURRENT closure
    microsleep: bool = False
    blinks: int = 0
    blink_rate: float = 0.0          # blinks per minute
    long_blinks: int = 0             # closures between blink and microsleep
    long_blink_rate: float = 0.0     # per minute
    yawns: int = 0
    yawn_rate: float = 0.0           # yawns per minute
    yawning: bool = False
    nodding: bool = False
    distracted: bool = False
    reasons: list = field(default_factory=list)
    should_alarm: bool = False
    alarm_kind: str = ""        # 'critical' | 'drowsy' | 'distract'
    eyes_reliable: bool = True


class DrowsinessMonitor:
    """
    Feed it one observation per frame; it returns the current state.

    Deliberately knows nothing about cameras, models or drawing. That makes it
    unit-testable with synthetic timestamps -- see tests/test_drowsiness.py,
    which simulates a fake 8-second microsleep without any webcam at all.
    """

    def __init__(self, cfg: Optional[DrowsyCfg] = None,
                 ear_thresh: Optional[float] = None,
                 cnn_thresh: Optional[float] = None,
                 cnn_weight: Optional[float] = None):
        self.cfg = cfg or CFG.drowsy
        self.ear_thresh = ear_thresh if ear_thresh is not None else self.cfg.ear_thresh
        self.cnn_thresh = cnn_thresh if cnn_thresh is not None else self.cfg.cnn_closed_thresh
        self.cnn_weight = cnn_weight if cnn_weight is not None else self.cfg.fusion_cnn_weight
        self.reset()

    def reset(self):
        # Each entry: (timestamp, was_closed, dt_seconds).
        # Storing dt lets us weight by real elapsed time, so a dropped frame
        # does not distort PERCLOS.
        self._win = deque()
        self._yawn_times = deque()
        self._blink_times = deque()
        self._long_blink_times = deque()

        self._last_t = None
        self._closed_since = None      # when the current closure began
        self._yawn_since = None
        self._nod_since = None
        self._distract_since = None
        self._last_face_t = None
        self._lost_since = None
        self._level_at_loss = Level.AWAKE
        self._last_alarm_t = 0.0
        self._last_distract_alarm_t = 0.0

        self.blinks = 0
        self.long_blinks = 0
        self.yawns = 0
        self.state = DrowsyState()

    # ------------------------------------------------------------------
    def _fuse(self, closed_prob: Optional[float], ear: float) -> float:
        """
        Combine the CNN probability with the geometric EAR into one score.

        Why fuse instead of picking one:
          - the CNN is accurate but knows nothing about THIS person's eye shape
          - EAR is person-specific and calibratable, but breaks with glasses,
            heavy shadows, and off-angle faces
        They fail in different situations, so averaging them is more robust
        than either alone. This is called sensor fusion.

        EAR is mapped to a 0..1 "closedness" score that reads 0.5 exactly at the
        threshold, so both signals live on the same scale before mixing.
        """
        span = max(self.ear_thresh * 0.6, 1e-6)
        ear_score = 0.5 + 0.5 * ((self.ear_thresh - ear) / span)
        ear_score = min(1.0, max(0.0, ear_score))

        if closed_prob is None:          # no model loaded -> geometry only
            return ear_score

        # Re-centre the CNN probability on its validated threshold, so a
        # threshold of e.g. 0.42 from evaluate.py still maps to 0.5 here.
        if self.cnn_thresh <= 0 or self.cnn_thresh >= 1:
            cnn_score = closed_prob
        elif closed_prob <= self.cnn_thresh:
            cnn_score = 0.5 * closed_prob / self.cnn_thresh
        else:
            cnn_score = 0.5 + 0.5 * (closed_prob - self.cnn_thresh) / (1 - self.cnn_thresh)

        w = self.cnn_weight
        return w * cnn_score + (1 - w) * ear_score

    def _prune(self, now: float):
        cut = now - self.cfg.perclos_window_sec
        while self._win and self._win[0][0] < cut:
            self._win.popleft()
        cut_y = now - self.cfg.yawn_window_sec
        while self._yawn_times and self._yawn_times[0] < cut_y:
            self._yawn_times.popleft()
        while self._blink_times and self._blink_times[0] < cut_y:
            self._blink_times.popleft()
        while self._long_blink_times and self._long_blink_times[0] < cut_y:
            self._long_blink_times.popleft()

    # ------------------------------------------------------------------
    def update(self, face_found: bool, closed_prob: Optional[float] = None,
               ear: float = 0.3, mar: float = 0.0, pitch: float = 0.0,
               yaw: float = 0.0, eyes_reliable: bool = True,
               now: Optional[float] = None) -> DrowsyState:
        # monotonic: time.time() can jump backwards on an NTP correction or a
        # DST change, which would produce a negative dt (clamped to 0, so a
        # closure appears to freeze) or a huge one (a 1 s blink read as an
        # hour). Nothing here needs wall-clock time.
        now = time.monotonic() if now is None else now
        dt = 0.0 if self._last_t is None else max(0.0, min(1.0, now - self._last_t))
        self._last_t = now
        s = DrowsyState()

        # ---- no face -------------------------------------------------
        if not face_found:
            # A brief tracking dropout (a hand passes the camera) should not
            # reset everything, so we allow a grace period before giving up.
            if self._last_face_t is not None and \
                    now - self._last_face_t < self.cfg.face_lost_grace_sec:
                # A COPY, not the stored object: mutating self.state in place
                # would silently rewrite history that the caller may still
                # hold a reference to.
                s = dataclasses.replace(
                    self.state, reasons=["face lost (grace period)"],
                    should_alarm=False, alarm_kind="")
                return s

            # Genuinely gone. Freeze the closure timer and the mouth/head
            # timers: a yawn or nod that began before the dropout must not be
            # "completed" the instant the face returns.
            seen_before = self._last_face_t is not None
            if self._lost_since is None:
                self._lost_since = self._last_face_t if seen_before else now
                self._level_at_loss = self.state.level
            gone = now - self._lost_since
            self._closed_since = None
            self._yawn_since = None
            self._nod_since = None
            self._distract_since = None

            level, reasons = Level.NO_FACE, ["no face detected"]
            should_alarm, alarm_kind = False, ""

            if (self._level_at_loss >= Level.DROWSY and
                    gone <= self.cfg.face_lost_hold_sec):
                # THE SAFETY CASE. The most dangerous way to lose a face is a
                # drowsy driver slumping out of frame. NO_FACE ranks below
                # AWAKE and never alarms, so the old code went SILENT at
                # exactly the moment it mattered most. Hold the last level and
                # keep alarming until the driver is seen again.
                level = self._level_at_loss
                reasons = [f"FACE LOST while {LEVEL_TEXT[level]} ({gone:.0f}s)"]
                if now - self._last_alarm_t >= self.cfg.alarm_cooldown_sec:
                    should_alarm = True
                    alarm_kind = ("critical" if level == Level.CRITICAL
                                  else "drowsy")
                    self._last_alarm_t = now
            elif seen_before and gone >= self.cfg.face_lost_nudge_sec:
                # Not drowsy, but the driver has been out of view for a while.
                # A gentle nudge on the distraction channel, not a siren.
                #
                # Only once a driver has actually been seen. Before that there
                # is nobody to nudge: the app has just been opened and the
                # phone is still being clipped into its mount, and beeping
                # "look at the road" at an empty seat every 9 s is exactly the
                # kind of nagging that gets the whole thing switched off.
                level = Level.DISTRACTED
                reasons = [f"driver not visible ({gone:.0f}s)"]
                if now - self._last_distract_alarm_t >= \
                        self.cfg.distract_alarm_cooldown_sec:
                    should_alarm = True
                    alarm_kind = "distract"
                    self._last_distract_alarm_t = now

            # Counters are kept so the driver cannot clear a bad PERCLOS by
            # ducking out of frame.
            s = DrowsyState(level=level, blinks=self.blinks, yawns=self.yawns,
                            long_blinks=self.long_blinks,
                            reasons=reasons, should_alarm=should_alarm,
                            alarm_kind=alarm_kind, eyes_reliable=False)
            self.state = s
            return s

        self._last_face_t = now
        self._lost_since = None
        self._prune(now)

        # ---- 1. is the eye closed right now? -------------------------
        # When the head is turned far enough that no eye is properly visible,
        # the numbers coming in are nonsense (a 4 px-wide eye returned EAR 1.05
        # in real testing). Guessing from nonsense is worse than abstaining:
        # we PAUSE the eye pipeline instead. PERCLOS stops accumulating rather
        # than filling with false closures, and the closure timer resets so a
        # head turn can never be mistaken for a microsleep.
        if not eyes_reliable:
            self._closed_since = None
            score, closed = 0.0, False
        else:
            score = self._fuse(closed_prob, ear)
            closed = score > 0.5
            self._win.append((now, closed, dt))

        # ---- 2. PERCLOS ----------------------------------------------
        # Time-weighted, NOT a simple frame count: sum the seconds spent
        # closed and divide by the total seconds observed.
        tot = sum(d for _, _, d in self._win)
        cls = sum(d for _, c, d in self._win if c)
        perclos = (cls / tot) if tot > 0.5 else 0.0
        # Only trust the percentage once we have observed enough time -- see
        # perclos_min_obs_sec in config.py for why.
        perclos_ready = tot >= self.cfg.perclos_min_obs_sec

        # ---- 3. closure tracking: blink vs microsleep ----------------
        microsleep = False
        if closed:
            if self._closed_since is None:
                self._closed_since = now
            closure = now - self._closed_since
            if closure >= self.cfg.microsleep_sec:
                microsleep = True
        else:
            if self._closed_since is not None:
                closure = now - self._closed_since
                # A SHORT closure that just ended was a normal blink.
                # A long one was a microsleep and is not counted as a blink.
                if closure <= self.cfg.blink_max_sec:
                    self.blinks += 1
                    self._blink_times.append(now)
                elif closure < self.cfg.microsleep_sec:
                    # Too long for a blink, too short for a microsleep.
                    # Previously this fell through both branches and was
                    # simply forgotten.
                    self.long_blinks += 1
                    self._long_blink_times.append(now)
            self._closed_since = None
        closure_sec = (now - self._closed_since) if self._closed_since else 0.0

        # ---- 4. yawning ----------------------------------------------
        # A yawn must be BIG and SUSTAINED. Talking makes the mouth open wide
        # but only briefly, so the duration requirement filters speech out.
        yawning = False
        if mar >= self.cfg.mar_thresh:
            if self._yawn_since is None:
                self._yawn_since = now
            elif now - self._yawn_since >= self.cfg.yawn_min_sec:
                yawning = True
        else:
            if self._yawn_since is not None and \
                    now - self._yawn_since >= self.cfg.yawn_min_sec:
                self.yawns += 1                  # count it once, on release
                self._yawn_times.append(now)
            self._yawn_since = None

        # ---- 5. head pose --------------------------------------------
        nodding = False
        if pitch <= self.cfg.pitch_nod_deg:
            if self._nod_since is None:
                self._nod_since = now
            elif now - self._nod_since >= self.cfg.nod_min_sec:
                nodding = True
        else:
            self._nod_since = None

        distracted = False
        if abs(yaw) >= self.cfg.yaw_distract_deg:
            if self._distract_since is None:
                self._distract_since = now
            elif now - self._distract_since >= self.cfg.distract_min_sec:
                distracted = True
        else:
            self._distract_since = None

        # ---- 6. decide the level -------------------------------------
        # Ordered most-severe first. Every trigger records a human-readable
        # reason, so the HUD can explain WHY it is alarming instead of just
        # flashing red. An unexplained alarm gets switched off.
        reasons, level = [], Level.AWAKE

        if not eyes_reliable:
            reasons.append("eyes not visible (head turned)")

        if microsleep:
            level = Level.CRITICAL
            reasons.append(f"MICROSLEEP {closure_sec:.1f}s")

        # Are the eyes shut RIGHT NOW, for longer than an ordinary blink?
        # Requiring more than blink_max_sec stops a single blink during
        # recovery from flipping the alarm back on.
        closed_now = closed and closure_sec >= self.cfg.blink_max_sec

        if not perclos_ready:
            reasons.append(f"PERCLOS warming up ({tot:.0f}/"
                           f"{self.cfg.perclos_min_obs_sec:.0f}s)")
        elif perclos >= self.cfg.perclos_critical:
            # PERCLOS is a 30-SECOND AVERAGE, so after a long closure it stays
            # high for up to 30 s even once the driver is wide awake again.
            # Shouting WAKE UP at someone whose eyes are demonstrably open is
            # exactly how a safety device gets switched off -- so CRITICAL also
            # requires the eyes to be shut at this moment. The trend still
            # reports DROWSY the whole time it decays, which is the honest
            # reading: recently very drowsy, currently awake.
            if closed_now:
                level = max(level, Level.CRITICAL)
                reasons.append(f"PERCLOS {perclos*100:.0f}% (critical)")
            else:
                level = max(level, Level.DROWSY)
                reasons.append(f"PERCLOS {perclos*100:.0f}% (recovering)")
        elif perclos >= self.cfg.perclos_warn:
            level = max(level, Level.DROWSY)
            reasons.append(f"PERCLOS {perclos*100:.0f}%")

        yawn_rate = len(self._yawn_times) * 60.0 / self.cfg.yawn_window_sec
        if len(self._yawn_times) >= self.cfg.yawn_rate_warn:
            level = max(level, Level.DROWSY)
            reasons.append(f"{len(self._yawn_times)} yawns/min")

        n_long = len(self._long_blink_times)
        if n_long >= self.cfg.long_blink_rate_warn:
            level = max(level, Level.DROWSY)
            reasons.append(f"{n_long} long blinks/min")

        if nodding:
            level = max(level, Level.DROWSY)
            reasons.append("head nodding")
        if distracted and level < Level.DROWSY:
            level = max(level, Level.DISTRACTED)
            reasons.append(f"looking away ({yaw:+.0f} deg)")

        # ---- 7. alarm, with a cooldown so it does not machine-gun -----
        # Two independent alarm channels, because they mean different things:
        #   drowsy/critical -> urgent, 4 s apart
        #   distracted      -> a nudge, 9 s apart and a softer sound
        # Nagging someone every 4 s while they check a mirror is exactly how a
        # safety system ends up switched off.
        should_alarm, alarm_kind = False, ""
        if level >= Level.DROWSY:
            if now - self._last_alarm_t >= self.cfg.alarm_cooldown_sec:
                should_alarm = True
                alarm_kind = ("critical" if level == Level.CRITICAL
                              else "drowsy")
                self._last_alarm_t = now
        elif level == Level.DISTRACTED:
            if now - self._last_distract_alarm_t >= \
                    self.cfg.distract_alarm_cooldown_sec:
                should_alarm = True
                alarm_kind = "distract"
                self._last_distract_alarm_t = now

        blink_rate = len(self._blink_times) * 60.0 / self.cfg.yawn_window_sec

        s = DrowsyState(
            level=level, closed=closed, closed_score=score, perclos=perclos,
            closure_sec=closure_sec, microsleep=microsleep,
            blinks=self.blinks, blink_rate=blink_rate,
            long_blinks=self.long_blinks,
            long_blink_rate=n_long * 60.0 / self.cfg.yawn_window_sec,
            yawns=self.yawns, yawn_rate=yawn_rate, yawning=yawning,
            nodding=nodding, distracted=distracted,
            reasons=reasons or ["normal"], should_alarm=should_alarm,
            alarm_kind=alarm_kind, eyes_reliable=eyes_reliable,
        )
        self.state = s
        return s


class EarCalibrator:
    """
    Measure THIS person's open-eye EAR, then set their threshold from it.

    Why this is necessary: EAR is a ratio of eye height to width, and that
    ratio genuinely differs between people. Someone with narrow eyes might sit
    at 0.19 while wide awake. A hard-coded 0.21 threshold would report them as
    asleep permanently.

    Fix: watch them with eyes open for a few seconds, take the MEDIAN EAR
    (median, not mean -- one accidental blink during calibration would drag a
    mean down but barely moves a median), and set

        threshold = 0.75 * their_open_EAR

    Now the system adapts to the individual instead of demanding the individual
    match the system. Press 'c' in the live app to run it.
    """

    def __init__(self, seconds: float = 3.0, ratio: float = None,
                 min_samples: int = 30):
        self.seconds = seconds
        self.ratio = ratio if ratio is not None else CFG.drowsy.ear_calib_ratio
        self.min_samples = min_samples
        self.samples = []
        self.t0 = None
        self._armed = False
        self.failed_reason = ""

    def start(self):
        self.samples = []
        self.t0 = None          # clock starts at the FIRST SAMPLE, not here
        self._armed = True
        self.failed_reason = ""

    @property
    def active(self) -> bool:
        return self._armed

    def feed(self, ear: float) -> Optional[float]:
        """
        Feed one frame's EAR. Returns the new threshold once we have enough
        data, otherwise None.

        Only call this when a face is actually present.

        THE BUG THIS AVOIDS: the first version started its countdown in
        start(). If no face appeared within those 3 seconds -- the user was not
        at the desk yet -- the window expired with almost no samples, the
        calibrator gave up, and it did so SILENTLY. The app then ran a whole
        session on the wrong threshold with no indication anything had failed.

        Two rules fix it:
          1. the clock starts on the first sample, so an empty room just means
             calibration has not begun yet rather than that it failed;
          2. BOTH a minimum duration and a minimum sample count are required,
             so a few stray frames cannot end it early.
        """
        if not self._armed:
            return None
        if self.t0 is None:
            self.t0 = time.monotonic()
        self.samples.append(ear)

        if time.monotonic() - self.t0 < self.seconds:
            return None
        if len(self.samples) < self.min_samples:
            return None         # keep waiting; do NOT give up

        self._armed = False
        import statistics
        open_ear = statistics.median(self.samples)

        # Sanity check. If the median looks like a shut eye, the user was
        # blinking, squinting or looking away. Calibrating to that would set a
        # threshold so low the detector could never fire again -- far worse
        # than keeping the default. Refuse, and say so.
        if open_ear < 0.10:
            self.failed_reason = (f"measured EAR {open_ear:.3f} looks like a "
                                  f"CLOSED eye - keep your eyes open and "
                                  f"press 'c' to retry")
            return None
        return float(open_ear * self.ratio)

    def remaining(self) -> float:
        if not self._armed:
            return 0.0
        if self.t0 is None:
            return self.seconds          # waiting for a face to appear
        return max(0.0, self.seconds - (time.monotonic() - self.t0))
