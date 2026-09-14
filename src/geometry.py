"""
Pure geometry on facial landmarks. No machine learning in this file at all.

This is the "classical computer vision" half of the project:
    EAR  -> how open is an eye
    MAR  -> how open is the mouth
    pose -> where is the head pointing

Everything here is stateless -- give it landmarks, get numbers back -- with one
deliberate exception: head_pose() remembers the previous frame's solution to
seed the next solve (see _POSE_PREV), and face.py clears that memory whenever
the face is lost.
"""
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices.
# The mesh has 478 points (468 face + 10 iris). These indices are fixed by
# MediaPipe, so they can be hard-coded.
# ---------------------------------------------------------------------------

# 6 points per eye, ordered [outer, upper1, upper2, inner, lower2, lower1]
# so that (p2,p6) and (p3,p5) are vertical pairs and (p1,p4) is horizontal.
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Full eyelid contours -> used to compute the crop box fed to the CNN.
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                    466, 388, 387, 386, 385, 384, 398]

# Mouth: 3 vertical pairs + 2 corners -> more stable than a single pair.
MOUTH_UPPER = [82, 13, 312]
MOUTH_LOWER = [87, 14, 317]
MOUTH_CORNERS = [78, 308]

# 6 points used for head-pose estimation (must match MODEL_POINTS_3D below).
POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]

# A generic 3D face in millimetres, origin at the nose tip.
# This is NOT a measurement of any particular head. A rough shape is enough
# because only ANGLES are needed, not absolute size.
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),     # 1   nose tip
    (0.0,  -63.6,  -12.5),     # 199 chin
    (-43.3, 32.7,  -26.0),     # 33  right eye outer corner
    (43.3,  32.7,  -26.0),     # 263 left eye outer corner
    (-28.9, -28.9, -24.1),     # 61  right mouth corner
    (28.9,  -28.9, -24.1),     # 291 left mouth corner
], dtype=np.float64)


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Intuition: the numerator is how TALL the eye is (two vertical measurements),
    the denominator is how WIDE it is. Dividing by width makes the number
    scale-invariant -- it does not change when you lean toward the camera.

    Open eye   ~ 0.25 - 0.35
    Closed eye ~ 0.05 - 0.12
    Those values are person-specific, which is exactly why we calibrate later.

    pts: (6,2) array ordered [outer, up1, up2, inner, low2, low1]
    """
    p1, p2, p3, p4, p5, p6 = np.asarray(pts, dtype=np.float64)
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def mouth_aspect_ratio(upper: np.ndarray, lower: np.ndarray,
                       corners: np.ndarray) -> float:
    """
    Same trick as EAR but for the mouth, averaged over 3 vertical pairs.

    Talking produces short MAR spikes; a yawn is a LONG sustained high MAR.
    That is why the yawn detector also requires a minimum duration.
    """
    width = _dist(corners[0], corners[1])
    if width < 1e-6:
        return 0.0
    vertical = float(np.mean([_dist(u, l) for u, l in zip(upper, lower)]))
    return vertical / width


def head_pose(landmarks_px: np.ndarray, frame_shape) -> tuple:
    """
    Estimate head orientation with PnP ("Perspective-n-Point").

    The setup: we know 6 points in 3D (a generic head) and where those same
    6 points landed in the 2D image. solvePnP finds the rotation + translation
    that explains that projection. The rotation matrix is then converted to
    Euler angles.

    Returns (pitch, yaw, roll) in degrees:
        pitch < 0   -> head tipping DOWN   (nodding off)
        yaw   != 0  -> looking left/right  (distracted)
        roll  != 0  -> head tilting toward a shoulder
    """
    h, w = frame_shape[:2]
    image_points = np.asarray(landmarks_px, dtype=np.float64)[POSE_LANDMARKS]

    # Approximate camera intrinsics. A webcam focal length in pixels is roughly
    # the image width -- good enough here because only angles are needed.
    focal = float(w)
    cam_matrix = np.array([[focal, 0, w / 2.0],
                           [0, focal, h / 2.0],
                           [0, 0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    # Seed the iterative solver with LAST frame's answer. The head moves only a
    # little between consecutive frames, so the previous pose is an excellent
    # starting point: the solver converges in fewer iterations and, more
    # importantly, cannot flip to a mirror-image solution that is also a valid
    # local minimum. Without the seed, pitch and yaw would occasionally jump
    # by tens of degrees for a single frame - enough to fake a head nod.
    guess = _POSE_PREV["rvec"] is not None
    if guess:
        rvec, tvec = _POSE_PREV["rvec"].copy(), _POSE_PREV["tvec"].copy()
        ok, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, cam_matrix, dist_coeffs,
            rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        _POSE_PREV["rvec"] = _POSE_PREV["tvec"] = None
        return 0.0, 0.0, 0.0
    _POSE_PREV["rvec"], _POSE_PREV["tvec"] = rvec, tvec

    rot_mat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rot_mat)[0]      # degrees, (x, y, z)
    return _wrap(angles[0]), _wrap(angles[1]), _wrap(angles[2])


# Previous frame's pose, used to seed the next solve. Module-level because
# head_pose() is a plain function; there is one driver, so one slot suffices.
_POSE_PREV = {"rvec": None, "tvec": None}


def reset_pose_seed():
    """Forget the previous pose, e.g. after the face has been lost."""
    _POSE_PREV["rvec"] = _POSE_PREV["tvec"] = None


def _wrap(a) -> float:
    """
    RQDecomp3x3 can report values near +/-180 for a forward-facing head.
    Wrap them back into a human-readable [-90, 90] range.
    """
    a = float(a)
    if a > 90:
        a -= 180
    elif a < -90:
        a += 180
    return a


def bbox_from_points(pts: np.ndarray, margin: float, frame_shape) -> tuple:
    """
    Turn a set of eyelid landmarks into a SQUARE crop box.

    Square matters: the CNN is trained on square 32x32 inputs. Feeding it a
    wide rectangle squashed into a square would distort eyes at inference time
    but not during training -- a classic train/test mismatch bug.
    """
    h, w = frame_shape[:2]
    pts = np.asarray(pts, dtype=np.float64)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * margin
    half = side / 2.0
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def safe_crop(img: np.ndarray, box: tuple) -> np.ndarray:
    """
    Crop with zero-padding when the box runs off the edge of the frame.

    Naive slicing (img[y0:y1, x0:x1]) silently returns a SMALLER image when the
    box is partially outside -- which then gets resized and warps the eye.
    Padding keeps the geometry honest.
    """
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1), dtype=img.dtype)
    crop = img[y0:y1, x0:x1]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_REPLICATE)
    return crop
