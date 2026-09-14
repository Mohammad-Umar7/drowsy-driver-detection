"""
Tests for the night / sun correction.

    python -m tests.test_lighting

scripts/test_lighting.py measures the correction on YOUR face with a camera.
These are the camera-free invariants: the classifier's regimes, the gamma
maths, the comfort band that stops good frames being touched, and the
smoothing state that must be forgotten on a reset.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import check, summary                        # noqa: E402
from src import lighting as L                                     # noqa: E402
from src.lighting import (Light, LightingNormalizer, auto_gamma,  # noqa: E402
                          classify, _gamma_lut)


def frame(mean, noise=6.0, shape=(180, 320), seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(mean, noise, (*shape, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_classifier_regimes():
    print("\n[1] the five lighting regimes")
    check("mean 30 is NIGHT", classify(30, 30, 20, 0.0, 0.5) == Light.DARK)
    check("mean 60 is DIM", classify(60, 60, 60, 0.0, 0.1) == Light.DIM)
    check("mean 128 is NORMAL", classify(128, 128, 140, 0.0, 0.0) == Light.NORMAL)
    check("mean 200 is BRIGHT SUN", classify(200, 200, 100, 0.05, 0.0) == Light.BRIGHT)
    check("30% clipped is BRIGHT SUN even at a normal mean",
          classify(150, 150, 100, 0.30, 0.0) == Light.BRIGHT)
    check("bright frame with a dark centre is BACKLIT",
          classify(150, 100, 150, 0.20, 0.0) == Light.BACKLIT)
    # The contrast term the Kotlin port once dropped: a normally lit frame
    # whose centre is merely dark (a beard, dark clothing) is NOT night.
    check("dark centre with good contrast is NOT night",
          classify(100, 30, 120, 0.0, 0.0) == Light.NORMAL)
    check("dark centre with poor contrast IS night",
          classify(100, 30, 40, 0.0, 0.0) == Light.DARK)


def test_gamma_maths():
    print("\n[2] auto_gamma: the comfort band, continuity, direction and clamp")
    for m in (85, 100, 128, 165):
        check(f"mean {m} inside the band is left alone", auto_gamma(m) == 1.0)
    check("exactly 1.0 at the lower edge (no flicker at the boundary)",
          abs(auto_gamma(L.COMFORT_LOW - 1e-9) - 1.0) < 1e-6)
    check("just below the band brightens only slightly", 0.97 < auto_gamma(84) < 1.0)
    check("a dark frame brightens (gamma < 1)", auto_gamma(20) < 1.0)
    check("a bright frame darkens (gamma > 1)", auto_gamma(240) > 1.0)
    check("darker input -> stronger correction", auto_gamma(20) < auto_gamma(40))
    check("clamped at the floor for black", auto_gamma(0) == L.GAMMA_MIN)
    check("clamped at the ceiling for white", auto_gamma(255) == L.GAMMA_MAX)


def test_lut_cache():
    print("\n[3] the gamma lookup table is built once per rounded gamma")
    _gamma_lut.cache_clear()
    _gamma_lut(round(0.7123, 2))
    _gamma_lut(round(0.7149, 2))
    info = _gamma_lut.cache_info()
    check("second call was a cache hit", info.hits == 1 and info.misses == 1,
          f"{info}")
    lut = _gamma_lut(0.5)
    check("LUT has 256 uint8 entries", lut.shape == (256,) and lut.dtype == np.uint8)
    check("gamma < 1 lifts the midtones", int(lut[64]) > 64)
    check("black stays black, white stays white", lut[0] == 0 and lut[255] == 255)


def test_dark_frame_is_brightened():
    print("\n[4] a night frame comes out brighter, with the same shape and dtype")
    norm = LightingNormalizer()
    src = frame(25)
    out = None
    for _ in range(40):                  # let the eased gamma settle
        out = norm.process(src)
    check("shape and dtype preserved", out.shape == src.shape and out.dtype == np.uint8)
    check("classified as NIGHT", norm.stats.condition == Light.DARK,
          norm.stats.condition.name)
    check("gamma below 1", norm.stats.gamma < 0.9, f"{norm.stats.gamma:.2f}")
    check("output is materially brighter", out.mean() > src.mean() + 15,
          f"{src.mean():.0f} -> {out.mean():.0f}")


def test_good_frame_is_left_alone():
    print("\n[5] a well-exposed frame is not pushed around")
    norm = LightingNormalizer()
    src = frame(128, noise=25)
    out = None
    for _ in range(10):
        out = norm.process(src)
    check("classified NORMAL", norm.stats.condition == Light.NORMAL)
    check("gamma stays at 1.0", abs(norm.stats.gamma - 1.0) < 1e-9, f"{norm.stats.gamma}")
    check("mean moves by less than 6 levels (soft CLAHE only)",
          abs(out.mean() - src.mean()) < 6, f"{src.mean():.1f} -> {out.mean():.1f}")


def test_disabled_is_a_passthrough_that_still_measures():
    print("\n[6] switched off, it returns the frame untouched but still reports the light")
    norm = LightingNormalizer(enabled=False)
    src = frame(25)
    out = norm.process(src)
    check("same array object back", out is src)
    check("condition still measured", norm.stats.condition == Light.DARK)
    check("gamma reported as 1.0", norm.stats.gamma == 1.0)
    check("toggle() re-enables", norm.toggle() is True and norm.enabled)


def test_reset_forgets_the_eased_gamma():
    print("\n[7] reset() drops the smoothing state from the previous scene")
    norm = LightingNormalizer()
    for _ in range(40):
        norm.process(frame(25))
    check("dark scene pulled gamma down", norm.stats.gamma < 0.9)
    norm.reset()
    norm.process(frame(128, noise=25))
    check("after reset a normal frame gets exactly 1.0, not a value eased from the old scene",
          abs(norm.stats.gamma - 1.0) < 1e-9, f"{norm.stats.gamma:.3f}")


def test_backlit_exposes_the_face_not_the_sky():
    print("\n[8] backlit: bright surround, dark centre -> the centre is what gets lifted")
    src = np.full((180, 320, 3), 252, np.uint8)      # sky clipped at white
    src[36:144, 80:240] = 40                          # face region in shadow
    norm = LightingNormalizer()
    out = None
    for _ in range(40):
        out = norm.process(src)
    check("classified BACKLIT", norm.stats.condition == Light.BACKLIT,
          norm.stats.condition.name)
    centre_before = src[36:144, 80:240].mean()
    centre_after = out[36:144, 80:240].mean()
    check("the shadowed centre is brighter", centre_after > centre_before + 20,
          f"{centre_before:.0f} -> {centre_after:.0f}")


if __name__ == "__main__":
    print("=" * 60)
    print("LIGHTING - regimes, gamma, comfort band, smoothing")
    print("=" * 60)
    for fn in (test_classifier_regimes, test_gamma_maths, test_lut_cache,
               test_dark_frame_is_brightened, test_good_frame_is_left_alone,
               test_disabled_is_a_passthrough_that_still_measures,
               test_reset_forgets_the_eased_gamma,
               test_backlit_exposes_the_face_not_the_sky):
        fn()
    summary("lighting")
