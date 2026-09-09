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
    ear: float                    # mean of both eyes
    mar: float
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
                 det_conf: float = 0.5, track_conf: float = 0.5):
        self.img_size = img_size
        self.crop_margin = crop_margin
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

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        box_l = G.bbox_from_points(pts[G.LEFT_EYE_CONTOUR],
                                   self.crop_margin, frame_bgr.shape)
        box_r = G.bbox_from_points(pts[G.RIGHT_EYE_CONTOUR],
                                   self.crop_margin, frame_bgr.shape)
        eye_l = preprocess_eye(G.safe_crop(gray, box_l), self.img_size)
        eye_r = preprocess_eye(G.safe_crop(gray, box_r), self.img_size)

        return FaceObs(
            landmarks_px=pts,
            ear_left=ear_l, ear_right=ear_r, ear=(ear_l + ear_r) / 2.0,
            mar=mar, pitch=pitch, yaw=yaw, roll=roll,
            eye_left=eye_l, eye_right=eye_r,
            box_left=box_l, box_right=box_r,
        )

    def close(self):
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
