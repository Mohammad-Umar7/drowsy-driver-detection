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
    d         toggle the debug panel (the exact 32x32 crops the CNN sees)
    l         toggle the night / sun correction
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
from .drowsiness import DrowsinessMonitor, EarCalibrator, Level, LEVEL_TEXT
from .face import FaceTracker
from .hud import Hud, draw_face
from .lighting import LightingNormalizer
from .source import VideoSource
from .model import build_model


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
    # Metadata is optional: a checkpoint written by anything other than
    # src.train (a bare state_dict wrapped by hand, an older run) has none,
    # and a missing label must not stop a perfectly good model from loading.
    epoch = ck.get("epoch", "?")
    acc = ck.get("val_bal_acc")
    acc_txt = f"{acc*100:.2f}%" if isinstance(acc, (int, float)) else "n/a"
    print(f"[ok] model loaded (epoch {epoch}, val balanced acc {acc_txt})  "
          f"threshold={thr:.3f}")
    return model, thr


def load_calibration():
    """
    This person's saved EAR threshold, or the default if they have none.

    Read from the git-ignored user file. The tracked calibration.json is
    checked too, but only for an `ear_thresh` left there by older versions,
    which wrote personal calibration into the model's file.
    """
    import json
    for p in (Path(CFG.paths.user_calibration), Path(CFG.paths.calibration)):
        if p.exists():
            d = json.loads(p.read_text())
            if "ear_thresh" in d:
                print(f"[ok] using calibrated EAR threshold "
                      f"{d['ear_thresh']:.3f}  ({p.name})")
                return float(d["ear_thresh"]), True
    return CFG.drowsy.ear_thresh, False


def save_calibration(**kw):
    """Persist per-person values in the user file, never the tracked one."""
    import json
    p = Path(CFG.paths.user_calibration)
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

    # The recorder is created LAZILY, once the real frame rate is known.
    # It used to be opened up front at a hard-coded 20 fps, so a clip captured
    # at 28 fps played back 40% too fast and one captured at 13 fps played in
    # slow motion. The first ~30 frames (about a second) are not recorded;
    # that is the price of a clip that plays at the right speed.
    writer = None

    # FPS over a rolling window of 30 frames -- a single-frame estimate is far
    # too noisy to read on screen.
    times = deque(maxlen=30)
    t_prev = None
    debug = False
    hud = Hud()
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
            if cap.ended:
                # A recording that has played to the end is not a dropped
                # camera. This used to fall into the SIGNAL LOST branch below
                # and sit there "reconnecting" for 30 s before giving up.
                print("[done] end of video")
                break
            if lost_since is None:
                lost_since = time.time()
            gone = time.time() - lost_since
            # Forget the loop timer: otherwise the first frame after recovery is
            # timed against the entire outage and the FPS readout collapses.
            t_prev = None

            # Keep the UI alive while reconnecting: show the last good frame
            # under a banner, and keep honouring keystrokes.
            if last_frame is not None:
                shown = last_frame.copy()
                hud.draw_signal_lost(shown, gone)
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

        if lost_since is not None:
            # Back after an outage. The eased gamma was tuned for the scene
            # BEFORE the drop; the reconnected stream may be a different
            # exposure entirely, so start the correction from neutral.
            lighting.reset()
        lost_since = None
        last_frame = frame
        n_frames += 1

        frame = lighting.process(frame)
        obs = tracker.process(frame)

        closed_prob = None
        probs = None
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
            probs = tuple(p.tolist())
            usable = [v for v, use in zip(probs, (obs.use_left, obs.use_right))
                      if use]
            closed_prob = max(usable) if usable else None

        # Only feed the calibrator when the eyes are actually readable. With
        # the head turned, obs.ear is the "nothing trustworthy" fallback, and a
        # value like the 1.05 measured from a 4 px eye would poison the median
        # and set a threshold no real eye could ever fall below.
        if obs is not None and obs.eyes_reliable and calib.active:
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
            draw_face(frame, obs, st.closed)

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
        hud.push(st.closed_score, st.closed)
        hud.draw(frame, st, obs, fps, ear_thr, model is not None,
                 alarm.enabled, debug, lighting, calib=calib, probs=probs)

        if args.record and writer is None and len(times) >= 30:
            rec_fps = float(min(60.0, max(5.0, round(fps))))
            h_, w_ = frame.shape[:2]
            writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                     rec_fps, (w_, h_))
            print(f"[record] {args.record} at {rec_fps:.0f} fps (measured)")
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
            hud.reset()          # the strip is history too
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
    alarm.close()


if __name__ == "__main__":
    main()
