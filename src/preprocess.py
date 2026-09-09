"""
The single source of truth for how an eye image becomes a model input.

THIS IS THE MOST IMPORTANT SMALL FILE IN THE PROJECT.

Training and live inference must preprocess pixels in EXACTLY the same way.
If they differ even slightly (different resize, different contrast handling),
the model scores 99% on the test set and then fails on your webcam. That bug
is called train/serve skew and it is the #1 killer of student CV projects.

Both `dataset.py` (training) and `face.py` (live) call this one function.
"""
import cv2
import numpy as np

# CLAHE = Contrast Limited Adaptive Histogram Equalization.
# Plain histogram equalization stretches contrast globally and blows out
# highlights. CLAHE does it in small tiles with a clip limit, so it survives
# the wildly varying light inside a car (tunnels, sun, night, dashboard glow).
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def preprocess_eye(crop, size: int = 32) -> np.ndarray:
    """
    BGR or grayscale eye crop  ->  (size, size) float32 array in [0, 1].

    Steps:
      1. to grayscale   - eye openness is a shape cue, colour adds nothing
                          and tripling the channels would triple compute
      2. resize         - INTER_AREA is the correct filter for downscaling
                          (it averages; INTER_LINEAR aliases and adds noise)
      3. CLAHE          - normalise contrast so bright and dark frames match
      4. scale to [0,1] - neural nets train badly on raw 0-255 inputs
    """
    if crop is None or crop.size == 0:
        return np.zeros((size, size), dtype=np.float32)

    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    crop = _CLAHE.apply(crop.astype(np.uint8))
    return crop.astype(np.float32) / 255.0
