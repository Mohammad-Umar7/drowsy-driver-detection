"""
Guided perception diagnostic. Run this when the detector misbehaves.

    python scripts/diagnose_pose.py

It walks you through scripted poses with an on-screen countdown, records what
the eye geometry and the CNN report during each one, then prints a verdict
saying which pose broke and why.

WHY THIS EXISTS
A blind capture is useless -- if the script starts recording before you are in
position, you get numbers for a pose you never struck, and you will "fix" a bug
that does not exist. This shows you exactly what to do, counts you in, and only
records while you are actually holding the pose.

The columns it collects, and what each tells you:

  ear        Eye Aspect Ratio. Geometry only. Collapses when the eyelid
             landmarks lose resolution, which happens when the face is small
             in frame or turned away.
  p_closed   The CNN's probability that the eye is closed. Independent of EAR,
             so if BOTH agree the input is genuinely bad, not the maths.
  eye_w      Eye corner-to-corner width in PIXELS. This is the resolution the
             whole system rests on. Below roughly 30 px the eyelid is only a
             few pixels tall and a 2 px landmark error becomes a 30% error.
  yaw        Head rotation left/right, degrees.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG                       # noqa: E402
from src.face import FaceTracker                 # noqa: E402
from src.model import build_model                # noqa: E402
from src.source import VideoSource               # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX

PHASES = [
    ("look STRAIGHT at the camera, eyes OPEN", "straight", 5.0),
    ("turn your head LEFT, keep eyes OPEN", "left", 5.0),
    ("turn your head RIGHT, keep eyes OPEN", "right", 5.0),
    ("look STRAIGHT, now CLOSE your eyes", "closed", 4.0),
    ("look STRAIGHT, eyes OPEN, lean CLOSE to the camera", "close_up", 4.0),
    ("look STRAIGHT, eyes OPEN, lean FAR BACK", "far_back", 4.0),
]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CFG.paths.best_model, map_location=dev)
    model = build_model(ck.get("arch", "eyenet")).to(dev).eval()
    model.load_state_dict(ck["model"])

    tracker = FaceTracker(CFG.data.img_size, CFG.data.crop_margin)
    # VideoSource, not a raw VideoCapture at 960x540. This camera does not
    # support 960x540 and silently returned 640x480 instead, so the very tool
    # meant to diagnose small-eye problems was itself running at half the
    # resolution of the app it was diagnosing. VideoSource requests a mode the
    # camera supports, reports what it actually got, and mirrors for us.
    cap = VideoSource(0, 1280, 720)
    if not cap.opened:
        sys.exit("could not open the camera")

    WIN = "Diagnostic - follow the instruction"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    data = {key: [] for _, key, _ in PHASES}
    print("Follow the on-screen instructions. Press q to abort.\n")

    for label, key, hold in PHASES:
        # --- 3 second countdown so you can get into position ---
        t0 = time.time()
        while time.time() - t0 < 3.0:
            ok, frame = cap.read()
            if not ok:
                continue
            left = 3.0 - (time.time() - t0)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (0, 0, 0), -1)
            cv2.putText(frame, label, (16, 38), FONT, 0.7, (0, 230, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, f"get ready... {left:.0f}", (16, 76), FONT,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WIN, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cap.release(); cv2.destroyAllWindows(); return

        # --- record while holding the pose ---
        t0 = time.time()
        while time.time() - t0 < hold:
            ok, frame = cap.read()
            if not ok:
                continue
            obs = tracker.process(frame)

            if obs is not None:
                batch = torch.from_numpy(
                    np.stack([obs.eye_left, obs.eye_right])[:, None]).to(dev)
                with torch.no_grad():
                    p = torch.softmax(model(batch), 1)[:, 1].cpu().numpy()
                # Eye width in pixels = distance between the two eye corners.
                wl = float(np.linalg.norm(obs.landmarks_px[362] -
                                          obs.landmarks_px[263]))
                wr = float(np.linalg.norm(obs.landmarks_px[33] -
                                          obs.landmarks_px[133]))
                data[key].append((obs.ear_left, obs.ear_right,
                                  float(p[0]), float(p[1]), wl, wr,
                                  obs.yaw, obs.pitch))
                for box in (obs.box_left, obs.box_right):
                    cv2.rectangle(frame, box[:2], box[2:], (0, 220, 120), 1)

            left = hold - (time.time() - t0)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (0, 0, 0), -1)
            cv2.putText(frame, label, (16, 38), FONT, 0.7, (0, 255, 120), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, f"RECORDING  {left:.1f}s", (16, 76), FONT, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)
            if obs is not None:
                cv2.putText(frame, f"EAR {obs.ear:.3f}  yaw {obs.yaw:+.0f}",
                            (frame.shape[1] - 260, 76), FONT, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WIN, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cap.release(); cv2.destroyAllWindows(); return

        print(f"  captured {len(data[key]):3d} frames for '{key}'")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    # ---------------- analysis ----------------
    thr = CFG.drowsy.cnn_closed_thresh
    print("\n" + "=" * 78)
    print(f"{'pose':<10} {'earL':>6} {'earR':>6} {'pL':>6} {'pR':>6} "
          f"{'eye_wL':>7} {'eye_wR':>7} {'yaw':>6}   verdict")
    print("=" * 78)

    summary = {}
    for label, key, _ in PHASES:
        rows = data[key]
        if not rows:
            print(f"{key:<10}  no face detected")
            continue
        a = np.array(rows)
        med = np.median(a, axis=0)
        earL, earR, pL, pR, wl, wr, yaw = (med[0], med[1], med[2], med[3],
                                           med[4], med[5], med[6])
        says_closed = max(pL, pR) >= thr
        should_be_closed = (key == "closed")
        ok = says_closed == should_be_closed
        verdict = "OK" if ok else ("MISSED closure" if should_be_closed
                                   else "FALSE 'closed'")
        summary[key] = dict(earL=earL, earR=earR, pL=pL, pR=pR,
                            wl=wl, wr=wr, yaw=yaw, ok=ok)
        print(f"{key:<10} {earL:6.3f} {earR:6.3f} {pL:6.3f} {pR:6.3f} "
              f"{wl:7.1f} {wr:7.1f} {yaw:6.1f}   {verdict}")

    print("=" * 78)
    print("\nDIAGNOSIS")

    if "straight" in summary:
        s = summary["straight"]
        print(f"  baseline open-eye EAR : {(s['earL']+s['earR'])/2:.3f}")
        print(f"  baseline eye width    : {(s['wl']+s['wr'])/2:.0f} px")
        if (s["wl"] + s["wr"]) / 2 < 30:
            print("  -> TOO SMALL. Below ~30 px the eyelid is only a few pixels")
            print("     tall, so landmark noise swamps the measurement.")
            print("     Sit closer to the camera or raise the capture resolution.")

    for side in ("left", "right"):
        if side in summary and "straight" in summary:
            s, b = summary[side], summary["straight"]
            ratio_l = s["wl"] / max(b["wl"], 1e-6)
            ratio_r = s["wr"] / max(b["wr"], 1e-6)
            print(f"  turned {side:<5}: yaw {s['yaw']:+.0f} deg, "
                  f"eye widths shrink to {ratio_l*100:.0f}% / {ratio_r*100:.0f}%"
                  f"  -> {'FALSE closed' if not s['ok'] else 'handled fine'}")

    bad = [k for k, v in summary.items() if not v["ok"]]
    print(f"\n  poses misread: {bad if bad else 'none'}")


if __name__ == "__main__":
    main()
