"""
Live webcam drowsiness detector -- the demo you actually show people.

    python -m src.infer                     # built-in webcam
    python -m src.infer --no-model          # geometry only (EAR), no CNN
    python -m src.infer --source clip.mp4   # run on a recorded file
    python -m src.infer --record out.mp4    # save what you see

    # PHONE AS A DASHCAM (same WiFi as the PC):
    #   1. install "IP Webcam" (Android) or "DroidCam"
    #   2. start the server in the app, note the URL it shows
    #   3. mount the phone facing the driver, then:
    python -m src.infer --source http://192.168.1.7:8080/video \
                        --rotate 270 --no-mirror
    # --rotate: a phone in a car mount is sideways; MediaPipe needs it upright
    # --no-mirror: a phone pointed at you is not a mirror, unlike a webcam

Keys:
    q / ESC   quit
    c         calibrate (hold eyes OPEN and look at the camera for 3 s)
    r         reset all counters
    a         mute / unmute the alarm
    d         toggle the debug panel (eye crops + landmarks)
    s         save a screenshot to reports/

=============================================================================
HOW THE LOOP IS STRUCTURED, AND WHY
=============================================================================

    read frame -> FaceTracker -> CNN on both eyes -> DrowsinessMonitor -> draw

The two eye crops are stacked into ONE batch of 2 and sent through the model in
a single call. Two separate calls would pay the Python/GPU launch overhead
twice, and that overhead dwarfs the actual maths on a model this small.

torch.no_grad() disables gradient tracking. During training PyTorch records
every operation so it can back-propagate; at inference we never do, so
recording is pure waste - it costs memory and time. Forgetting no_grad() is a
very common cause of a "slow" inference loop.
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from .alarm import Alarm
from .config import CFG
from .drowsiness import (DrowsinessMonitor, EarCalibrator, Level,
                         LEVEL_COLOR, LEVEL_TEXT)
from .face import FaceTracker
from .lighting import LightingNormalizer, LIGHT_TEXT, Light
from .source import VideoSource
from .model import build_model

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def draw_panel(img, x, y, w, h, alpha=0.55):
    """
    Translucent dark panel behind text so it stays readable on any background.

    Trick: draw the rectangle on a COPY, then blend the copy with the original.
    OpenCV has no native alpha, so blending two full images is how you fake it.
    """
    sub = img[y:y + h, x:x + w]
    if sub.size == 0:
        return
    box = np.zeros_like(sub)
    img[y:y + h, x:x + w] = cv2.addWeighted(sub, 1 - alpha, box, alpha, 0)


def draw_bar(img, x, y, w, h, frac, color, label, warn_at=None):
    """Horizontal progress bar with an optional threshold tick mark."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
    fill = int(w * max(0.0, min(1.0, frac)))
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + fill - 1, y + h - 1), color, -1)
    if warn_at is not None:
        wx = x + int(w * warn_at)
        cv2.line(img, (wx, y - 2), (wx, y + h + 2), (255, 255, 255), 1)
    cv2.putText(img, label, (x + w + 8, y + h - 2), FONT, 0.42,
                (230, 230, 230), 1, cv2.LINE_AA)


def draw_hud(frame, st, obs, fps, ear_thr, model_on, alarm_on, debug,
             lighting=None):
    h, w = frame.shape[:2]
    color = LEVEL_COLOR[st.level]

    # --- big status banner ---
    draw_panel(frame, 0, 0, w, 62)
    cv2.putText(frame, LEVEL_TEXT[st.level], (14, 44), FONT, 1.25, color, 3,
                cv2.LINE_AA)
    reason = " | ".join(st.reasons[:2])
    cv2.putText(frame, reason, (w - 12 - 8 * len(reason), 26), FONT, 0.46,
                (215, 215, 215), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f} FPS", (w - 88, 50), FONT, 0.5,
                (180, 180, 180), 1, cv2.LINE_AA)

    # --- red border pulse on CRITICAL: peripheral vision catches motion even
    #     when you are not looking straight at the screen ---
    if st.level == Level.CRITICAL and int(time.time() * 6) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 14)

    # --- metrics panel ---
    py = h - 153
    draw_panel(frame, 0, py, 340, 153)
    y = py + 22

    draw_bar(frame, 12, y - 10, 150, 12, st.perclos,
             (0, 165, 255) if st.perclos >= CFG.drowsy.perclos_warn
             else (90, 190, 90),
             f"PERCLOS {st.perclos*100:4.1f}%",
             warn_at=CFG.drowsy.perclos_warn)
    y += 24

    draw_bar(frame, 12, y - 10, 150, 12, st.closed_score,
             (0, 0, 235) if st.closed else (90, 190, 90),
             f"closed {st.closed_score:.2f}", warn_at=0.5)
    y += 24

    if obs is not None:
        cv2.putText(frame, f"EAR {obs.ear:.3f} (thr {ear_thr:.3f})   "
                           f"MAR {obs.mar:.2f}", (12, y), FONT, 0.46,
                    (215, 215, 215), 1, cv2.LINE_AA)
        y += 21
        cv2.putText(frame, f"pitch {obs.pitch:+5.1f}  yaw {obs.yaw:+5.1f}  "
                           f"roll {obs.roll:+5.1f}", (12, y), FONT, 0.46,
                    (215, 215, 215), 1, cv2.LINE_AA)
        y += 21
        # Eye size in pixels and which eyes are being trusted. This line is
        # what makes the "turned head = false closure" class of bug visible
        # instead of mysterious.
        eyes = ("L" if obs.use_left else "-") + ("R" if obs.use_right else "-")
        wmax = max(obs.width_left, obs.width_right)
        wmin = min(obs.width_left, obs.width_right)
        cv2.putText(frame, f"eye px {wmax:.0f}/{wmin:.0f}  using [{eyes}]"
                           f"{'' if st.eyes_reliable else '  UNRELIABLE'}",
                    (12, y), FONT, 0.46,
                    (215, 215, 215) if st.eyes_reliable else (0, 200, 255),
                    1, cv2.LINE_AA)
        y += 21

    cv2.putText(frame, f"blinks {st.blinks} ({st.blink_rate:.0f}/min)   "
                       f"yawns {st.yawns} ({st.yawn_rate:.0f}/min)",
                (12, y), FONT, 0.46, (215, 215, 215), 1, cv2.LINE_AA)

    # --- lighting readout, top-left under the banner ---
    if lighting is not None:
        ls = lighting.stats
        # Amber whenever the light is genuinely difficult, so it is obvious
        # that a bad reading might be the scene rather than the driver.
        col = ((150, 210, 150) if ls.condition == Light.NORMAL
               else (0, 200, 255))
        tag = LIGHT_TEXT.get(ls.condition, "?")
        cv2.putText(frame, f"light: {tag}  mean {ls.mean:>3.0f}  "
                           f"gamma {ls.gamma:.2f}"
                           f"{'' if lighting.enabled else '  [OFF]'}",
                    (14, 82), FONT, 0.5, col, 1, cv2.LINE_AA)

    # --- footer ---
    mode = "CNN+EAR" if model_on else "EAR only"
    cv2.putText(frame, f"[{mode}]  alarm {'ON' if alarm_on else 'OFF'}   "
                       f"q quit  c calib  r reset  a alarm  d debug  l light  s shot",
                (12, h - 8), FONT, 0.40, (170, 170, 170), 1, cv2.LINE_AA)

    # --- debug: show exactly what the CNN sees ---
    if debug and obs is not None:
        for i, eye in enumerate((obs.eye_left, obs.eye_right)):
            big = cv2.resize((eye * 255).astype(np.uint8), (96, 96),
                             interpolation=cv2.INTER_NEAREST)
            big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
            x0 = w - 210 + i * 102
            frame[70:166, x0:x0 + 96] = big
            cv2.rectangle(frame, (x0, 70), (x0 + 96, 166), (110, 110, 110), 1)
        cv2.putText(frame, "what the CNN sees", (w - 210, 182), FONT, 0.40,
                    (170, 170, 170), 1, cv2.LINE_AA)


def draw_landmarks(frame, obs):
    """Outline the eyes and mouth, and colour the eyes by state."""
    from . import geometry as G
    for idx in (G.LEFT_EYE_CONTOUR, G.RIGHT_EYE_CONTOUR):
        pts = obs.landmarks_px[idx].astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 220, 120), 1, cv2.LINE_AA)
    mouth = obs.landmarks_px[G.MOUTH_UPPER + G.MOUTH_LOWER[::-1]].astype(np.int32)
    cv2.polylines(frame, [mouth], True, (200, 160, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
def load_model(device):
    """Load the trained checkpoint, or return None to fall back to EAR only."""
    p = Path(CFG.paths.best_model)
    if not p.exists():
        print(f"[warn] no model at {p} - running on EAR geometry only.")
        print("       train one with:  python -m src.train")
        return None, 0.5
    ck = torch.load(p, map_location=device)
    model = build_model(ck.get("arch", "eyenet")).to(device).eval()
    model.load_state_dict(ck["model"])

    # Use the threshold evaluate.py validated, not a guessed 0.5.
    thr = CFG.drowsy.cnn_closed_thresh
    cp = Path(CFG.paths.calibration)
    if cp.exists():
        import json
        thr = json.loads(cp.read_text()).get("cnn_closed_thresh", thr)
    print(f"[ok] model loaded (epoch {ck['epoch']}, "
          f"val balanced acc {ck['val_bal_acc']*100:.2f}%)  threshold={thr:.3f}")
    return model, thr


def load_calibration():
    p = Path(CFG.paths.calibration)
    if p.exists():
        import json
        d = json.loads(p.read_text())
        if "ear_thresh" in d:
            print(f"[ok] using calibrated EAR threshold {d['ear_thresh']:.3f}")
            return float(d["ear_thresh"]), True
    return CFG.drowsy.ear_thresh, False


def save_calibration(**kw):
    import json
    p = Path(CFG.paths.calibration)
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2))


def main():
    ap = argparse.ArgumentParser()
    # --source takes anything: a device index, a file, or a phone URL.
    #   0                                  built-in webcam
    #   http://192.168.1.7:8080/video      Android "IP Webcam" app
    #   http://192.168.1.7:4747/video      DroidCam
    #   drive.mp4                          recorded clip
    ap.add_argument("--source", type=str, default=None,
                    help="camera index, video file, or phone stream URL")
    ap.add_argument("--camera", type=int, default=0)          # legacy alias
    ap.add_argument("--video", type=str, default=None)        # legacy alias
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate frames - a phone in a car mount is sideways")
    ap.add_argument("--mirror", action="store_true",
                    help="force mirroring on")
    ap.add_argument("--no-mirror", action="store_true",
                    help="force mirroring off (correct for a phone facing you)")
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-alarm", action="store_true")
    ap.add_argument("--no-enhance", action="store_true",
                    help="disable adaptive night/sun correction")
    # 1280x720, not 960x540. Cameras do NOT error on an unsupported resolution
    # -- they silently hand back whatever they do support. This webcam quietly
    # downgraded a 960x540 request to 640x480, which halved the eye to ~23 px
    # and made the eyelid landmarks unusable. Always verify what you got.
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cnn_thr = (None, 0.5) if args.no_model else load_model(device)
    ear_thr, calibrated = load_calibration()

    tracker = FaceTracker(CFG.data.img_size, CFG.data.crop_margin)
    monitor = DrowsinessMonitor(ear_thresh=ear_thr, cnn_thresh=cnn_thr,
                                cnn_weight=CFG.drowsy.fusion_cnn_weight
                                if model is not None else 0.0)
    alarm = Alarm(enabled=not args.no_alarm)
    # Runs BEFORE MediaPipe: in the dark there is no face to find,
    # so enhancing only the eye crop would be too late.
    lighting = LightingNormalizer(enabled=not args.no_enhance)
    calib = EarCalibrator()

    # One source type for webcam, file and phone stream. See src/source.py for
    # why a network stream needs its own frame-dropping reader.
    source = args.source if args.source is not None else (
        args.video if args.video else args.camera)
    mirror = True if args.mirror else (False if args.no_mirror else None)
    cap = VideoSource(source, args.width, args.height, rotate=args.rotate,
                      mirror=mirror)
    if not cap.opened:
        raise SystemExit(
            f"could not open video source {source!r}\n"
            f"  phone: install 'IP Webcam' (Android), start the server, then\n"
            f"         python -m src.infer --source http://<phone-ip>:8080/video")
    print(f"[ok] source: {cap.describe()}")

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h = cap.size
        writer = cv2.VideoWriter(args.record, fourcc, 20.0, (w, h))

    # FPS over a rolling window of 30 frames -- a single-frame estimate is far
    # too noisy to read on screen.
    times = deque(maxlen=30)
    t_prev = None
    debug = False
    WIN = "Drowsy Driver Detection"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    # Keep the window on top so it cannot open behind the editor and look
    # like the app failed to start.
    try:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    # Auto-calibrate when no saved threshold exists. Measured on a real user:
    # their open-eye EAR was 0.181, BELOW the 0.210 default, so the geometry
    # branch called their open eyes closed all session. They were told to press
    # 'c' three times and never did -- so the system now does it itself rather
    # than depending on the user remembering.
    if not calibrated:
        calib.start()
        print("[auto-calibrate] measuring your open-eye EAR for 3 s - "
              "keep your eyes OPEN and look at the camera")
    print("[run] window open. press 'c' to recalibrate, 'q' to quit")

    # Session stats, printed on exit so a short run explains itself.
    t_session = time.time()
    n_frames = n_faces = n_alarms = 0
    max_perclos = 0.0
    peak_level = Level.AWAKE
    lost_since = None
    last_frame = None

    while True:
        # A SHORT wait, deliberately. read() blocks until a genuinely new frame
        # arrives, so a long timeout means the whole loop stalls -- and while
        # it is stalled cv2.waitKey is never called, so the window stops
        # repainting and 'q' does nothing. With the old 2 s wait and a 60-fail
        # budget, a dropped phone stream froze the app for two minutes with no
        # way to quit it.
        ok, frame = cap.read(wait=0.25)
        if not ok:
            if lost_since is None:
                lost_since = time.time()
            gone = time.time() - lost_since

            # Keep the UI alive while reconnecting: show the last good frame
            # under a banner, and keep honouring keystrokes.
            if last_frame is not None:
                shown = last_frame.copy()
                draw_panel(shown, 0, 0, shown.shape[1], 62)
                cv2.putText(shown, "SIGNAL LOST", (14, 44), FONT, 1.25,
                            (0, 165, 255), 3, cv2.LINE_AA)
                cv2.putText(shown, f"reconnecting... {gone:.0f}s",
                            (shown.shape[1] - 250, 40), FONT, 0.6,
                            (215, 215, 215), 1, cv2.LINE_AA)
                cv2.imshow(WIN, shown)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            # Give up on elapsed TIME, not on a raw retry count: the number of
            # retries depends on the timeout, so counting them measured
            # nothing meaningful.
            if gone > 30.0:
                print(f"[error] no frames for {gone:.0f}s - giving up")
                break
            continue

        lost_since = None
        last_frame = frame
        n_frames += 1

        frame = lighting.process(frame)
        obs = tracker.process(frame)

        closed_prob = None
        if obs is not None and model is not None:
            # Both eyes as ONE batch of 2 -> a single GPU call.
            batch = torch.from_numpy(
                np.stack([obs.eye_left, obs.eye_right])[:, None]
            ).to(device)
            with torch.no_grad():
                p = torch.softmax(model(batch), dim=1)[:, 1]

            # Take the max ONLY over eyes we can actually see.
            #
            # The naive max() over both eyes is safety-biased and correct when
            # facing forward, but it breaks badly on a head turn: measured at
            # yaw +70 deg the far eye was 4.6 px wide and the CNN scored that
            # sliver 0.68 "closed", while the near eye correctly said 0.03.
            # max() picked the garbage and reported a false closure.
            # face.py flags which eyes are worth trusting; we honour that.
            usable = [v for v, use in zip(p.tolist(),
                                          (obs.use_left, obs.use_right)) if use]
            closed_prob = max(usable) if usable else None

        if obs is not None and calib.active:
            new_thr = calib.feed(obs.ear)
            if new_thr is not None:
                ear_thr = new_thr
                monitor.ear_thresh = new_thr
                save_calibration(ear_thresh=new_thr)
                print(f"[calibrated] your open-eye EAR threshold -> "
                      f"{new_thr:.3f}  (was {CFG.drowsy.ear_thresh:.3f})")
            elif not calib.active and calib.failed_reason:
                # Never fail silently: a wrong threshold ruins the whole
                # session and the user has no way to know it happened.
                print(f"[calibration failed] {calib.failed_reason}")

        st = monitor.update(
            face_found=obs is not None,
            closed_prob=closed_prob,
            ear=obs.ear if obs else 0.3,
            mar=obs.mar if obs else 0.0,
            pitch=obs.pitch if obs else 0.0,
            yaw=obs.yaw if obs else 0.0,
            eyes_reliable=obs.eyes_reliable if obs else False,
        )

        if obs is not None:
            n_faces += 1
        max_perclos = max(max_perclos, st.perclos)
        peak_level = max(peak_level, st.level)
        if st.should_alarm:
            n_alarms += 1
            alarm.fire(kind=st.alarm_kind or "drowsy")

        if obs is not None:
            draw_landmarks(frame, obs)
            c = (0, 0, 235) if st.closed else (0, 220, 120)
            for box in (obs.box_left, obs.box_right):
                cv2.rectangle(frame, box[:2], box[2:], c, 1)

        # True end-to-end frame rate, measured from the start of one iteration
        # to the start of the next so it INCLUDES the camera read.
        # Timing only the processing block reports how fast our code is
        # (~145 FPS) rather than the rate the user actually sees (~20 FPS),
        # which is worse than no number at all.
        loop_now = time.time()
        if t_prev is not None:
            times.append(loop_now - t_prev)
        t_prev = loop_now
        fps = 1.0 / max(1e-6, float(np.mean(times))) if times else 0.0
        draw_hud(frame, st, obs, fps, ear_thr, model is not None,
                 alarm.enabled, debug, lighting)

        if calib.active:
            msg = ("CALIBRATING - keep eyes OPEN  "
                   f"{calib.remaining():.1f}s" if obs is not None
                   else "CALIBRATING - waiting for a face...")
            cv2.putText(frame, msg, (14, 92), FONT, 0.75, (0, 255, 255), 2,
                        cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)
        cv2.imshow(WIN, frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("c"):
            calib.start()
            print("[calibrating] hold your eyes open...")
        elif k == ord("r"):
            monitor.reset()
            print("[reset]")
        elif k == ord("a"):
            print(f"[alarm] {'on' if alarm.toggle() else 'off'}")
        elif k == ord("d"):
            debug = not debug
        elif k == ord("l"):
            print(f"[lighting] {'on' if lighting.toggle() else 'off'}")
        elif k == ord("s"):
            Path(CFG.paths.reports).mkdir(parents=True, exist_ok=True)
            p = Path(CFG.paths.reports) / f"shot_{int(time.time())}.png"
            cv2.imwrite(str(p), frame)
            print(f"[saved] {p}")

    dur = time.time() - t_session
    print("-" * 52)
    print(f"session      : {dur:.1f}s, {n_frames} frames "
          f"({n_frames/max(dur,1e-6):.1f} FPS)")
    print(f"face tracked : {n_faces}/{n_frames} frames "
          f"({100*n_faces/max(1,n_frames):.0f}%)")
    print(f"peak state   : {LEVEL_TEXT[peak_level]}")
    print(f"max PERCLOS  : {max_perclos*100:.1f}%  "
          f"(warn at {CFG.drowsy.perclos_warn*100:.0f}%)")
    print(f"blinks {monitor.blinks} | yawns {monitor.yawns} | "
          f"alarms fired {n_alarms}")
    print("-" * 52)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[saved] {args.record}")
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
