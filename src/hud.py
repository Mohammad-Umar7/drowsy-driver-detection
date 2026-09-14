"""
The on-screen display for the desktop app.

Everything the driver (or the person demoing this) sees is drawn here, on top
of the camera frame, with plain OpenCV calls. Two things are worth knowing
about how that works, because OpenCV is a computer-vision library and not a
UI toolkit:

  1. OpenCV HAS NO ALPHA. You cannot draw a translucent rectangle. What you can
     do is draw an opaque one on a COPY of the region, then blend the copy with
     the original: out = (1-a)*original + a*copy. Blending only the panel's
     own pixels keeps it cheap - a full-frame blend at 720p would cost more
     than the face detector.

  2. OPENCV HAS NO ROUNDED RECTANGLES EITHER. A rounded rectangle is two plain
     rectangles in a cross shape plus four filled circles in the corners.

Layout is measured with cv2.getTextSize rather than guessed from character
counts, so a long reason string is truncated to fit instead of being drawn
straight through the status label next to it.

The one stateful element is the closure SPARKLINE: the last 30 seconds of the
fused eye-closure score, red where the eyes were shut. A blink is a narrow
spike; a microsleep is a wide red block; a slow slide into fatigue shows up as
spikes that get wider and closer together. That is the whole detector in one
picture, and it makes "why did it just alarm?" answerable at a glance.
"""
import time
from collections import deque

import cv2
import numpy as np

from .config import CFG
from .drowsiness import Level, LEVEL_TEXT
from .lighting import Light, LIGHT_TEXT

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# Palette, in BGR because that is what OpenCV speaks.
PANEL = (16, 18, 24)
TRACK = (40, 44, 54)
INK = (238, 240, 242)
MUTED = (150, 160, 172)
DIM = (100, 108, 120)
GOOD = (110, 205, 120)       # green
WARN = (40, 190, 255)        # amber
HOT = (30, 140, 255)         # orange
BAD = (70, 70, 235)          # red
COOL = (235, 200, 120)       # light blue, for informational chips
GREY = (165, 165, 165)

LEVEL_COLOR = {
    Level.NO_FACE: GREY,
    Level.AWAKE: GOOD,
    Level.DISTRACTED: WARN,
    Level.DROWSY: HOT,
    Level.CRITICAL: BAD,
}


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def rounded_rect(img, x0, y0, x1, y1, r, color, thickness=-1):
    """Rectangle with rounded corners: two rects in a cross plus four discs."""
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if r == 0:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        return
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r),
                       (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)
        for cx, cy, a in ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270),
                          (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thickness,
                        cv2.LINE_AA)


def panel(img, x, y, w, h, alpha=0.66, radius=14, color=PANEL):
    """
    Translucent rounded panel. Only the panel's own region is blended.

    Outside the rounded rectangle the copy is identical to the original, so
    the blend leaves those pixels untouched - no mask needed.
    """
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    rounded_rect(over, x - x0, y - y0, x - x0 + w - 1, y - y0 + h - 1,
                 radius, color, -1)
    cv2.addWeighted(roi, 1 - alpha, over, alpha, 0, dst=roi)


def text_size(s, scale, thick=1, font=FONT):
    (tw, th), base = cv2.getTextSize(s, font, scale, thick)
    return tw, th, base


def text(img, s, x, y, scale, color, thick=1, font=FONT):
    """Draw text with its baseline at y. Returns the width drawn."""
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, thick, cv2.LINE_AA)
    return text_size(s, scale, thick, font)[0]


def fit_text(s, max_w, scale, thick=1, font=FONT):
    """Truncate with '...' so the string fits in max_w pixels."""
    if text_size(s, scale, thick, font)[0] <= max_w:
        return s
    while len(s) > 1 and text_size(s + "...", scale, thick, font)[0] > max_w:
        s = s[:-1]
    return s.rstrip() + "..."


def chip(img, s, x, y, fg, bg, scale=0.5, pad_x=10, h=None, thick=1,
         font=FONT, alpha=1.0):
    """A filled pill with a label. Returns its width."""
    tw, th, base = text_size(s, scale, thick, font)
    h = h or th + base + 12
    w = tw + 2 * pad_x
    if alpha < 1.0:
        panel(img, x, y, w, h, alpha, radius=h // 2, color=bg)
    else:
        rounded_rect(img, x, y, x + w - 1, y + h - 1, h // 2, bg, -1)
    text(img, s, x + pad_x, y + (h + th) // 2, scale, fg, thick, font)
    return w


def bar(img, x, y, w, h, frac, color, warn_at=None):
    """Rounded progress track with a fill and an optional threshold tick."""
    rounded_rect(img, x, y, x + w, y + h, h // 2, TRACK, -1)
    fill = int(w * max(0.0, min(1.0, frac)))
    if fill >= h:
        rounded_rect(img, x, y, x + fill, y + h, h // 2, color, -1)
    elif fill > 0:
        cv2.circle(img, (x + h // 2, y + h // 2), h // 2, color, -1, cv2.LINE_AA)
    if warn_at is not None:
        wx = x + int(w * warn_at)
        cv2.line(img, (wx, y - 3), (wx, y + h + 3), INK, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
class Hud:
    """Draws the display, and remembers the last 30 s of closure for the sparkline."""

    def __init__(self, window_sec: float = None):
        self.window = window_sec or CFG.drowsy.perclos_window_sec
        # (time, fused closure score, closed?) at ~10 Hz. Storing every frame
        # would mean ~900 line segments per redraw for nothing visible.
        self.hist = deque()
        self._last_push = None

    # -- state ------------------------------------------------------------
    def reset(self):
        """Forget the closure history, e.g. when the counters are reset."""
        self.hist.clear()
        self._last_push = None

    def push(self, score: float, closed: bool, now: float = None):
        now = time.monotonic() if now is None else now
        # Thin to ~10 Hz, but never drop a sample that changes the closed
        # flag: a 40 ms blink must still show both of its edges.
        same = bool(self.hist) and self.hist[-1][2] == closed
        if not (self._last_push is not None and now - self._last_push < 0.1
                and same):
            self.hist.append((now, float(score), bool(closed)))
            self._last_push = now
        # Prune on EVERY call, accepted or thinned, so the window is measured
        # from the current frame rather than from the last accepted sample.
        cut = now - self.window
        while self.hist and self.hist[0][0] < cut:
            self.hist.popleft()

    # -- pieces -----------------------------------------------------------
    def _sparkline(self, img, x, y, w, h, now, s=1.0):
        rounded_rect(img, x, y, x + w, y + h, 6, TRACK, -1)
        # A caption strip along the top that the plot never enters, so the
        # label cannot collide with a spike.
        cap = int(15 * s)
        text(img, "eye closure, last %.0f s" % self.window, x + int(8 * s),
             y + int(11 * s), 0.34 * s, DIM)
        py0, py1 = y + cap, y + h - 2          # plot area, top to bottom
        if len(self.hist) < 2:
            return
        ty = (py0 + py1) // 2
        cv2.line(img, (x + 6, ty), (x + w - 6, ty), (70, 76, 90), 1)
        t0 = now - self.window
        pts = []
        for t, sc, cl in self.hist:
            px = x + 2 + int((t - t0) / self.window * (w - 4))
            py = py1 - int(sc * (py1 - py0))
            pts.append((max(x + 2, min(x + w - 2, px)), py, cl))
        poly = np.array([(pts[0][0], py1)] + [(p[0], p[1]) for p in pts]
                        + [(pts[-1][0], py1)], np.int32)
        roi = img[y:y + h, x:x + w]
        over = roi.copy()
        cv2.fillPoly(over, [poly - (x, y)], (60, 90, 70))
        cv2.addWeighted(roi, 0.45, over, 0.55, 0, dst=roi)
        for (x1, y1, c1), (x2, y2, c2) in zip(pts, pts[1:]):
            cv2.line(img, (x1, y1), (x2, y2), BAD if (c1 or c2) else GOOD, 2,
                     cv2.LINE_AA)

    def _debug_eyes(self, img, obs, probs, x_right, y, s):
        size = int(96 * s)
        gap = int(8 * s)
        x = x_right - 2 * size - gap
        panel(img, x - 10, y - 10, 2 * size + gap + 20, size + int(44 * s),
              radius=10)
        for i, (eye, name) in enumerate(((obs.eye_left, "L"),
                                         (obs.eye_right, "R"))):
            big = cv2.resize((eye * 255).astype(np.uint8), (size, size),
                             interpolation=cv2.INTER_NEAREST)
            xx = x + i * (size + gap)
            img[y:y + size, xx:xx + size] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
            used = (obs.use_left, obs.use_right)[i]
            cv2.rectangle(img, (xx, y), (xx + size - 1, y + size - 1),
                          GOOD if used else DIM, 1)
            label = name if probs is None else f"{name}  {probs[i]:.2f}"
            if not used:
                label += "  ignored"
            text(img, label, xx + 2, y + size + int(16 * s), 0.38 * max(s, 0.8),
                 INK if used else DIM)
        text(img, "what the CNN sees" + ("" if probs is None else
                                         "  -  P(closed)"),
             x, y + size + int(32 * s), 0.36 * max(s, 0.8), DIM)

    # -- the whole thing --------------------------------------------------
    def draw(self, frame, st, obs, fps, ear_thr, model_on, alarm_on, debug,
             lighting=None, calib=None, probs=None, now=None):
        now = time.monotonic() if now is None else now
        h, w = frame.shape[:2]
        # Everything is sized for 1280 wide and scaled down for smaller frames.
        s = max(0.55, min(1.0, w / 1280.0))
        col = LEVEL_COLOR[st.level]
        pad = int(14 * s)

        # ---- CRITICAL: pulsing red frame. Peripheral vision catches motion
        #      even when nobody is looking straight at the screen. ----
        flash = st.level == Level.CRITICAL and int(now * 5) % 2 == 0
        if flash:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), BAD, int(16 * s))

        # ---- top bar ----
        bar_h = int(64 * s)
        panel(frame, 0, 0, w, bar_h, alpha=0.6, radius=0)
        x = pad
        pill_bg = INK if flash else col
        pill_fg = BAD if flash else PANEL
        x += chip(frame, LEVEL_TEXT[st.level], x, int(12 * s), pill_fg, pill_bg,
                  scale=0.85 * s, pad_x=int(16 * s), h=int(40 * s), thick=2,
                  font=FONT_BOLD) + int(14 * s)

        # right-hand chips: FPS and lighting
        rx = w - pad
        fps_txt = f"{fps:4.1f} FPS"
        cw = text_size(fps_txt, 0.48 * s)[0] + int(20 * s)
        rx -= cw
        chip(frame, fps_txt, rx, int(19 * s), MUTED, TRACK, scale=0.48 * s,
             pad_x=int(10 * s), h=int(26 * s))
        if lighting is not None:
            ls = lighting.stats
            tag = LIGHT_TEXT.get(ls.condition, "?")
            if not lighting.enabled:
                tag += " (fix off)"
            good = ls.condition == Light.NORMAL and lighting.enabled
            cw = text_size(tag, 0.48 * s)[0] + int(20 * s)
            rx -= cw + int(8 * s)
            chip(frame, tag, rx, int(19 * s), PANEL if not good else MUTED,
                 WARN if not good else TRACK, scale=0.48 * s,
                 pad_x=int(10 * s), h=int(26 * s))

        # reasons, truncated to the space that is actually left
        reason = "  |  ".join(st.reasons[:2])
        reason = fit_text(reason, max(20, rx - x - int(12 * s)), 0.5 * s)
        text(frame, reason, x, int(38 * s), 0.5 * s, MUTED)

        # ---- calibration banner ----
        if calib is not None and calib.active:
            bw, bh = int(520 * s), int(58 * s)
            bx, by = (w - bw) // 2, bar_h + int(12 * s)
            panel(frame, bx, by, bw, bh, alpha=0.8, radius=12)
            msg = ("CALIBRATING  -  keep your eyes OPEN" if obs is not None
                   else "CALIBRATING  -  waiting for a face...")
            text(frame, msg, bx + int(16 * s), by + int(24 * s), 0.55 * s, WARN,
                 1, FONT_BOLD)
            rem = calib.remaining()
            frac = 1.0 - rem / max(1e-6, calib.seconds)
            bar(frame, bx + int(16 * s), by + int(36 * s), bw - int(32 * s),
                int(10 * s), frac if obs is not None else 0.0, WARN)

        # ---- metrics card, bottom left ----
        cw, ch = int(392 * s), int(224 * s)
        cx, cy = pad, h - pad - ch
        panel(frame, cx, cy, cw, ch, alpha=0.66, radius=14)
        ix = cx + int(16 * s)
        y = cy + int(28 * s)
        lab_w = int(78 * s)
        bw = cw - int(32 * s) - lab_w - int(74 * s)

        perclos_col = (BAD if st.perclos >= CFG.drowsy.perclos_critical else
                       HOT if st.perclos >= CFG.drowsy.perclos_warn else GOOD)
        text(frame, "PERCLOS", ix, y, 0.46 * s, MUTED)
        bar(frame, ix + lab_w, y - int(11 * s), bw, int(12 * s), st.perclos,
            perclos_col, warn_at=CFG.drowsy.perclos_warn)
        text(frame, f"{st.perclos * 100:4.1f}%", ix + lab_w + bw + int(10 * s),
             y, 0.46 * s, INK)
        y += int(26 * s)

        text(frame, "EYES", ix, y, 0.46 * s, MUTED)
        bar(frame, ix + lab_w, y - int(11 * s), bw, int(12 * s), st.closed_score,
            BAD if st.closed else GOOD, warn_at=0.5)
        state = ("CLOSED %.1fs" % st.closure_sec if st.closed else "open")
        text(frame, state, ix + lab_w + bw + int(10 * s), y, 0.46 * s,
             BAD if st.closed else INK)
        y += int(18 * s)

        self._sparkline(frame, ix, y, cw - int(32 * s), int(58 * s), now, s)
        y += int(58 * s) + int(24 * s)

        if obs is not None:
            text(frame, f"EAR {obs.ear:.3f}", ix, y, 0.46 * s, INK)
            text(frame, f"thr {ear_thr:.3f}", ix + int(96 * s), y, 0.42 * s, DIM)
            text(frame, f"MAR {obs.mar:.2f}", ix + int(170 * s), y, 0.46 * s, INK)
            eyes = ("L" if obs.use_left else "-") + ("R" if obs.use_right else "-")
            text(frame, f"eyes [{eyes}] {max(obs.width_left, obs.width_right):.0f}px",
                 ix + int(258 * s), y, 0.42 * s,
                 INK if st.eyes_reliable else WARN)
            y += int(24 * s)
            text(frame, f"pitch {obs.pitch:+5.1f}   yaw {obs.yaw:+5.1f}   "
                        f"roll {obs.roll:+5.1f}", ix, y, 0.44 * s, MUTED)
            y += int(24 * s)
        else:
            text(frame, "no face in view", ix, y, 0.46 * s, DIM)
            y += int(48 * s)

        counters = (f"blinks {st.blinks}    long {st.long_blinks} "
                    f"({st.long_blink_rate:.0f}/min)    yawns {st.yawns} "
                    f"({st.yawn_rate:.0f}/min)")
        text(frame, counters, ix, y, 0.44 * s, MUTED)

        # ---- controls card, bottom right ----
        kw, kh = int(300 * s), int(96 * s)
        kx, ky = w - pad - kw, h - pad - kh
        panel(frame, kx, ky, kw, kh, alpha=0.6, radius=14)
        x = kx + int(14 * s)
        y = ky + int(14 * s)
        x += chip(frame, "CNN + EAR" if model_on else "EAR only", x, y,
                  PANEL, COOL, scale=0.42 * s, pad_x=int(9 * s),
                  h=int(24 * s)) + int(8 * s)
        chip(frame, "ALARM ON" if alarm_on else "ALARM MUTED", x, y,
             PANEL if alarm_on else INK, GOOD if alarm_on else DIM,
             scale=0.42 * s, pad_x=int(9 * s), h=int(24 * s))
        text(frame, "q quit   c calibrate   r reset   a alarm",
             kx + int(14 * s), ky + int(58 * s), 0.4 * s, MUTED)
        text(frame, "d debug   l light fix   s screenshot",
             kx + int(14 * s), ky + int(80 * s), 0.4 * s, MUTED)

        # ---- debug: exactly what the CNN sees ----
        if debug and obs is not None:
            self._debug_eyes(frame, obs, probs, w - pad, bar_h + int(22 * s), s)

    def draw_signal_lost(self, frame, gone_sec: float):
        h, w = frame.shape[:2]
        s = max(0.55, min(1.0, w / 1280.0))
        bw, bh = int(460 * s), int(96 * s)
        bx, by = (w - bw) // 2, (h - bh) // 2
        panel(frame, bx, by, bw, bh, alpha=0.82, radius=16)
        text(frame, "SIGNAL LOST", bx + int(20 * s), by + int(42 * s), 1.0 * s,
             WARN, 2, FONT_BOLD)
        text(frame, f"reconnecting...  {gone_sec:.0f}s   (q to quit)",
             bx + int(20 * s), by + int(74 * s), 0.5 * s, MUTED)


def draw_face(frame, obs, closed: bool):
    """Outline the eyes and mouth, and box the eye crops, coloured by state."""
    from . import geometry as G
    col = BAD if closed else GOOD
    for idx in (G.LEFT_EYE_CONTOUR, G.RIGHT_EYE_CONTOUR):
        pts = obs.landmarks_px[idx].astype(np.int32)
        cv2.polylines(frame, [pts], True, col, 1, cv2.LINE_AA)
    mouth = obs.landmarks_px[G.MOUTH_UPPER + G.MOUTH_LOWER[::-1]].astype(np.int32)
    cv2.polylines(frame, [mouth], True, COOL, 1, cv2.LINE_AA)
    for box, used in ((obs.box_left, obs.use_left),
                      (obs.box_right, obs.use_right)):
        rounded_rect(frame, box[0], box[1], box[2], box[3], 4,
                     col if used else DIM, 1)
