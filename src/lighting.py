"""
Adaptive illumination normalisation: make night and direct sun both work.

=============================================================================
NIGHT AND SUNLIGHT ARE THE SAME BUG
=============================================================================

It is tempting to write two features, "night mode" and "sun mode". They are
actually one problem seen from two ends: the pixel values are bunched in the
wrong part of the range.

    NIGHT      every pixel is squashed into 0-40. The eyelid edge is still
               there, but the contrast across it is a couple of levels, which
               landmark detection cannot resolve.

    DIRECT SUN the lit side of the face saturates at 255. Once a pixel clips
               at 255 the detail is GONE - averaging or scaling cannot bring
               back a value that was never recorded.

    BACKLIT    the worst case, and the common one in a car: bright sky through
               the windscreen while the face sits in shadow. The frame AVERAGE
               looks fine, so a naive brightness check sees nothing wrong,
               while the face itself is crushed into near-black.

So we do not detect "night" or "day". We look at where the histogram actually
sits and move it to the middle. One mechanism, every condition.

TWO TOOLS, AND WHY BOTH ARE NEEDED

1. GAMMA CORRECTION -- fixes the overall level.
       out = 255 * (in/255) ** gamma
   gamma < 1 brightens, gamma > 1 darkens. It is non-linear on purpose: it
   lifts the dark end far more than the bright end, so shadow detail is
   recovered without blowing out highlights that are already fine.

   We do not guess the value. Given the current mean and a target, the gamma
   that maps one to the other is:
       gamma = log(target/255) / log(mean/255)
   That single line adapts to any lighting automatically.

2. CLAHE -- fixes LOCAL contrast, which gamma cannot.
   Backlit is the case that proves the point. Gamma applies one curve to the
   whole frame, so lifting the shadowed face also pushes the bright window
   further into clipping. CLAHE works in small tiles, so the dark half gets
   stretched while the bright half is left alone. That is exactly the
   half-lit-face situation of driving with the sun to one side.

WHY WE WORK IN LAB AND NOT BGR
Applying gamma to B, G and R separately changes their ratios, which shifts the
colours - skin goes orange or blue and the face detector suffers. LAB splits
lightness (L) from colour (A, B). Touch only L and brightness changes while
hue stays exactly as it was.

WHY THE WHOLE FRAME AND NOT JUST THE EYE CROP
preprocess.py already CLAHEs the 32x32 eye crop for the CNN. But that happens
AFTER MediaPipe has located the face. In the dark MediaPipe finds no face at
all, so there is no crop to enhance. This stage runs first, on the full frame,
so that landmark detection gets something workable.
"""
import functools
from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np


class Light(IntEnum):
    DARK = 0        # night, no cabin light
    DIM = 1         # dusk, tunnel, underground car park
    NORMAL = 2      # ordinary daylight or indoors
    BRIGHT = 3      # direct sun, washed out
    BACKLIT = 4     # bright behind, face in shadow -- the nasty one


LIGHT_TEXT = {
    Light.DARK: "NIGHT",
    Light.DIM: "DIM",
    Light.NORMAL: "NORMAL",
    Light.BRIGHT: "BRIGHT SUN",
    Light.BACKLIT: "BACKLIT",
}


@dataclass
class LightStats:
    condition: Light
    mean: float             # average luminance, 0-255
    face_mean: float        # luminance of the centre region (where a face sits)
    clipped_high: float     # fraction of pixels at/near 255 -- detail destroyed
    clipped_low: float      # fraction at/near 0
    contrast: float         # p95 - p5, how much of the range is actually used
    gamma: float            # what we applied (1.0 = untouched)


def _stats(l_chan: np.ndarray) -> tuple:
    """
    Cheap histogram summary of the luminance channel.

    Sampled on a downscaled copy: we need distribution shape, not precision,
    and doing this on a full 1280x720 frame every frame is wasted work.
    """
    small = cv2.resize(l_chan, (160, 90), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    p5, p95 = np.percentile(small, [5, 95])
    clipped_high = float((small >= 250).mean())
    clipped_low = float((small <= 8).mean())

    # Centre crop: in a driver-facing camera the face is near the middle, so
    # this approximates "how bright is the face" without running a detector.
    h, w = small.shape
    face = small[h // 5: h * 4 // 5, w // 4: w * 3 // 4]
    face_mean = float(face.mean())
    return mean, face_mean, float(p95 - p5), clipped_high, clipped_low


def classify(mean, face_mean, contrast, clipped_high, clipped_low) -> Light:
    """
    Decide which regime we are in.

    Order matters. BACKLIT is checked FIRST because it is the case that fools
    every simpler test: the frame mean can look perfectly normal while the face
    is in deep shadow. Only comparing the centre against the whole frame
    catches it.
    """
    if clipped_high > 0.12 and face_mean < mean - 25:
        return Light.BACKLIT
    if mean < 45 or (face_mean < 40 and contrast < 60):
        return Light.DARK
    if mean < 80:
        return Light.DIM
    if mean > 185 or clipped_high > 0.28:
        return Light.BRIGHT
    return Light.NORMAL


# A frame anywhere in this band is already well exposed and is LEFT ALONE.
#
# Without this dead band, auto_gamma forced every frame to exactly one target,
# so a perfectly good bright scene at mean 183 was darkened with gamma 2.19 --
# the clamp ceiling -- for no benefit. It also pushed such frames away from the
# statistics the model was trained on, which is why sunlit frames measured
# slightly WORSE with correction enabled than without it.
#
# Correcting only toward the nearest EDGE of the band also keeps gamma
# continuous: at mean == COMFORT_LOW the correction is exactly 1.0, so a frame
# hovering at the boundary cannot flicker between corrected and uncorrected.
COMFORT_LOW = 85.0
COMFORT_HIGH = 165.0


def auto_gamma(mean: float, low: float = COMFORT_LOW,
               high: float = COMFORT_HIGH) -> float:
    """
    The gamma that pulls `mean` to the nearest edge of the comfortable band,
    or 1.0 (no change) if it is already inside.

    Derivation: we want (mean/255) ** gamma == target/255, so taking logs of
    both sides and rearranging gives the expression below.

    Clamped to [0.35, 2.2]. An unclamped gamma on a nearly black frame goes
    enormous and amplifies pure sensor noise into a grey blizzard.
    """
    mean = float(np.clip(mean, 4.0, 250.0))
    if low <= mean <= high:
        return 1.0
    target = low if mean < low else high
    g = np.log(target / 255.0) / np.log(mean / 255.0)
    return float(np.clip(g, 0.35, 2.2))


@functools.lru_cache(maxsize=64)
def _gamma_lut(gamma: float) -> np.ndarray:
    """
    Precompute all 256 outputs once, then map the image through the table.

    Cached by value: the caller rounds gamma to 2 decimals first. Rounding to
    3 was measured to miss on EVERY frame while gamma eases toward its
    target, so the cache did nothing; at 2 decimals the visual difference
    is nil and consecutive frames share a key.

    A lookup table turns a per-pixel power operation (expensive, ~1M of them
    per frame) into a single memory lookup. On a 720p frame this is the
    difference between roughly 40 ms and under 1 ms.
    """
    i = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(((i ** gamma) * 255.0), 0, 255).astype(np.uint8)


class LightingNormalizer:
    """
    Stateful frame enhancer.

    Stateful for one reason: TEMPORAL SMOOTHING. If gamma were recomputed from
    scratch every frame, passing headlights or a flickering streetlight would
    make the whole image pulse, the landmarks would jitter, and EAR would
    wobble enough to fake blinks. We ease toward the new gamma instead, so
    the correction tracks real lighting changes but ignores flicker.
    """

    def __init__(self, enabled: bool = True, smooth: float = 0.12):
        self.enabled = enabled
        self.smooth = smooth            # 0..1, lower = steadier
        self._gamma = 1.0
        self._clahe_strong = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._clahe_soft = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        self.stats = LightStats(Light.NORMAL, 0, 0, 0, 0, 0, 1.0)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.process(frame)

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        mean, face_mean, contrast, hi, lo = _stats(l)
        cond = classify(mean, face_mean, contrast, hi, lo)

        if not self.enabled:
            self.stats = LightStats(cond, mean, face_mean, hi, lo, contrast, 1.0)
            return frame

        # In a backlit frame the FACE is what we need to expose correctly, not
        # the average of a scene dominated by bright sky. Aim the gamma at the
        # centre region instead and let the background clip -- we do not care
        # about the background.
        aim = face_mean if cond == Light.BACKLIT else mean
        target = auto_gamma(aim)

        # Ease toward the target rather than snapping to it.
        self._gamma += (target - self._gamma) * self.smooth
        g = self._gamma

        if abs(g - 1.0) > 0.03:
            l = cv2.LUT(l, _gamma_lut(round(g, 2)))

        # Strong local contrast where the image is genuinely poor, gentle
        # otherwise. Running the strong setting on an already-good frame just
        # adds noise and halos around the eyelids.
        if cond in (Light.DARK, Light.DIM, Light.BACKLIT, Light.BRIGHT):
            l = self._clahe_strong.apply(l)
        else:
            l = self._clahe_soft.apply(l)

        if cond == Light.DARK:
            # Brightening a night frame also brightens its sensor noise. A 3x3
            # median kills salt-and-pepper speckle while preserving edges --
            # unlike a blur, which would soften the eyelid we are measuring.
            l = cv2.medianBlur(l, 3)

        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        self.stats = LightStats(cond, mean, face_mean, hi, lo, contrast, g)
        return out

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled
