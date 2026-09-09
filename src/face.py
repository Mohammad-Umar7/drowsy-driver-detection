"""
MediaPipe FaceMesh wrapper: one frame in, one FaceObs out.

Why MediaPipe instead of training our own face detector:
  - it is already trained on millions of faces and runs at 100+ FPS on CPU
  - re-training that from scratch would take weeks and a GPU farm
  - our contribution is the EYE-STATE model and the temporal logic on top

This is a normal engineering decision, not cheating: use a solved component
off the shelf, spend your effort on the part that is actually your problem.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

from . import geometry as G
from .preprocess import preprocess_eye


@dataclass
class FaceObs:
    """Everything we extract from a single frame."""
    landmarks_px: np.ndarray      # (478, 2) pixel coordinates
    ear_left: float
    ear_right: float
    ear: float                    # visibility-weighted, NOT a naive mean
    mar: float
    width_left: float             # eye corner-to-corner width in pixels
    width_right: float
    vis_left: float               # 0..1, this eye vs the wider one
    vis_right: float
    use_left: bool                # is this eye worth trusting this frame?
    use_right: bool
    eyes_reliable: bool           # is ANY eye big enough to trust at all?
    pitch: float
    yaw: float
    roll: float
    eye_left: np.ndarray          # (S, S) float32 in [0,1], ready for the CNN
    eye_right: np.ndarray
    box_left: tuple               # crop boxes, for drawing on the HUD
    box_right: tuple


class FaceTracker:
    """
    Stateful wrapper around MediaPipe FaceMesh.

    Key settings:
      refine_landmarks=True  -> adds 10 iris points (468 -> 478) and noticeably
                                sharpens the EYELID landmarks, which is what we
                                actually care about. Costs ~1 ms. Worth it.
      max_num_faces=1        -> only the driver matters; more faces = slower
      min_tracking_confidence-> MediaPipe TRACKS between detections instead of
                                re-detecting every frame, which is why it is fast
    """

    def __init__(self, img_size: int = 32, crop_margin: float = 1.35,
                 det_conf: float = 0.5, track_conf: float = 0.5,
                 vis_min_ratio: float = None, min_width_px: float = None):
        from .config import CFG
        self.img_size = img_size
        self.crop_margin = crop_margin
        self.vis_min_ratio = (CFG.drowsy.eye_vis_min_ratio
                              if vis_min_ratio is None else vis_min_ratio)
        self.min_width_px = (CFG.drowsy.eye_min_width_px
                             if min_width_px is None else min_width_px)
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceObs]:
        """Returns None when no face is found in the frame."""
        h, w = frame_bgr.shape[:2]

        # MediaPipe wants RGB. Marking the array read-only lets it skip a copy.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        # Landmarks come back NORMALISED to [0,1]; scale to pixels.
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)

        ear_l = G.eye_aspect_ratio(pts[G.LEFT_EYE_EAR])
        ear_r = G.eye_aspect_ratio(pts[G.RIGHT_EYE_EAR])
        mar = G.mouth_aspect_ratio(pts[G.MOUTH_UPPER], pts[G.MOUTH_LOWER],
                                   pts[G.MOUTH_CORNERS])
        pitch, yaw, roll = G.head_pose(pts, frame_bgr.shape)

        # ---- how much can we trust each eye this frame? -------------------
        # Corner-to-corner width in pixels. When the head turns away, the far
        # eye's width collapses by roughly cos(yaw) while the near eye keeps
        # its size -- so the ratio between them is a direct, calibration-free
        # measure of which eye is actually facing the camera.
        w_l = float(np.linalg.norm(pts[G.LEFT_EYE_EAR[0]] -
                                   pts[G.LEFT_EYE_EAR[3]]))
        w_r = float(np.linalg.norm(pts[G.RIGHT_EYE_EAR[0]] -
                                   pts[G.RIGHT_EYE_EAR[3]]))
        w_max = max(w_l, w_r, 1e-6)
        vis_l, vis_r = w_l / w_max, w_r / w_max

        # An eye counts only if it is both facing us AND big enough in pixels
        # for the eyelid landmarks to mean anything.
        use_l = (vis_l >= self.vis_min_ratio) and (w_l >= self.min_width_px)
        use_r = (vis_r >= self.vis_min_ratio) and (w_r >= self.min_width_px)
        eyes_reliable = use_l or use_r

        # Report EAR from the eyes we actually trust. Averaging in a
        # foreshortened eye is what produced false "closed" readings on turns.
        if use_l and use_r:
            ear = (ear_l + ear_r) / 2.0
        elif use_l:
            ear = ear_l
        elif use_r:
            ear = ear_r
        else:
            ear = max(ear_l, ear_r)     # nothing trustworthy; bias toward open

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        box_l = G.bbox_from_points(pts[G.LEFT_EYE_CONTOUR],
                                   self.crop_margin, frame_bgr.shape)
        box_r = G.bbox_from_points(pts[G.RIGHT_EYE_CONTOUR],
                                   self.crop_margin, frame_bgr.shape)
        eye_l = preprocess_eye(G.safe_crop(gray, box_l), self.img_size)
        eye_r = preprocess_eye(G.safe_crop(gray, box_r), self.img_size)

        return FaceObs(
            landmarks_px=pts,
            ear_left=ear_l, ear_right=ear_r, ear=ear,
            mar=mar,
            width_left=w_l, width_right=w_r,
            vis_left=vis_l, vis_right=vis_r,
            use_left=use_l, use_right=use_r, eyes_reliable=eyes_reliable,
            pitch=pitch, yaw=yaw, roll=roll,
            eye_left=eye_l, eye_right=eye_r,
            box_left=box_l, box_right=box_r,
        )

    def close(self):
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
