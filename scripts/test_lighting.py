"""
Measure how badly bad light breaks the detector, and how much the fix helps.

    python scripts/test_lighting.py

Captures one clean reference frame of your face, then SIMULATES night, direct
sun and backlighting on it, and reports for each condition whether the face is
still found and whether the eye state is still read correctly -- with the
lighting normaliser off, then on.

Why simulate instead of testing at night: it isolates one variable. Real night
footage differs in pose, distance and expression too, so you could never tell
which change caused the drop. Here the ONLY difference is the light, so any
change in the numbers is caused by the light and nothing else.

The degradations are modelled on what a camera sensor actually does:
  night    multiply everything down (less light reaches the sensor), then ADD
           noise -- because sensor noise is roughly constant, so it dominates
           once the signal shrinks. This is why night footage looks grainy.
  sun      push the exposure up and CLIP at 255. Clipping is destructive:
           values above 255 are not recorded at all, so that detail is gone
           for good and no amount of processing recovers it.
  backlit  brighten one side toward white and darken the other, the classic
           sun-through-the-side-window case.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG                                  # noqa: E402
from src.face import FaceTracker                            # noqa: E402
from src.lighting import LightingNormalizer, LIGHT_TEXT     # noqa: E402
from src.model import build_model                           # noqa: E402
from src.source import VideoSource                          # noqa: E402


def degrade_night(img, level=0.18, noise=9.0):
    """Less light on the sensor, plus the noise that then dominates."""
    rng = np.random.default_rng(0)
    out = img.astype(np.float32) * level
    out += rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_sun(img, gain=2.4):
    """Overexposure. The clip at 255 is what destroys the information."""
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def degrade_backlit(img, strength=0.75):
    """Bright on one side, face pushed into shadow on the other."""
    h, w = img.shape[:2]
    ramp = np.linspace(1.0 + strength * 1.8, 1.0 - strength, w, dtype=np.float32)
    out = img.astype(np.float32) * ramp[None, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def capture_reference(model, device, seconds=12.0):
    """
    Grab one clean frame whose eyes are provably OPEN.

    THE BUG THIS REPLACES: the first version kept the frame with the LARGEST
    face and never checked eye state. Out of ~300 captured frames several are
    mid-blink, and a blink frame can easily be the biggest one. The benchmark
    then measured every lighting condition against a reference whose eyes were
    shut, so the "clean" baseline itself scored P(closed)=0.93 and every
    subsequent number was meaningless.

    A benchmark that silently measures the wrong thing is worse than no
    benchmark, because you act on it. Select on what actually matters -- the
    model's own confidence that the eye is OPEN -- and refuse to proceed if no
    such frame was seen.
    """
    cap = VideoSource(0, 1280, 720, verbose=False)
    tr = FaceTracker()
    best, best_p, best_w = None, 1.0, 0.0
    seen = 0
    t0 = time.time()
    print("Look at the camera with your eyes OPEN ...")
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        obs = tr.process(frame)
        if obs is None:
            continue
        seen += 1
        w = max(obs.width_left, obs.width_right)
        if w < 30:                       # too small to judge anything from
            continue
        batch = torch.from_numpy(
            np.stack([obs.eye_left, obs.eye_right])[:, None]).to(device)
        with torch.no_grad():
            p = float(torch.softmax(model(batch), 1)[:, 1].max().item())
        if p < best_p:                   # most confidently OPEN wins
            best, best_p, best_w = frame.copy(), p, w
    cap.release()
    tr.close()
    print(f"  {seen} frames with a face; best P(closed)={best_p:.3f}")
    if best is not None and best_p > 0.30:
        print("  [warn] never saw a confidently open eye - results will be "
              "unreliable. Sit closer, face the camera, keep your eyes open.")
    return best, best_w, best_p


def measure(frame, tracker, model, device, normalizer=None):
    """Run one frame through the pipeline; report what survived."""
    if normalizer is not None:
        frame = normalizer.process(frame)
    obs = tracker.process(frame)
    if obs is None:
        return dict(found=False, ear=0.0, p=1.0, width=0.0,
                    cond=(normalizer.stats.condition if normalizer else None))

    batch = torch.from_numpy(
        np.stack([obs.eye_left, obs.eye_right])[:, None]).to(device)
    with torch.no_grad():
        p = torch.softmax(model(batch), 1)[:, 1].cpu().numpy()
    usable = [v for v, use in zip(p.tolist(), (obs.use_left, obs.use_right))
              if use]
    return dict(found=True, ear=obs.ear, p=(max(usable) if usable else 1.0),
                width=max(obs.width_left, obs.width_right),
                cond=(normalizer.stats.condition if normalizer else None))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CFG.paths.best_model, map_location=device)
    model = build_model(ck.get("arch", "eyenet")).to(device).eval()
    model.load_state_dict(ck["model"])

    ref, ref_w, ref_p = capture_reference(model, device)
    if ref is None:
        sys.exit("no face captured - sit in front of the camera and retry")
    print(f"reference: eye width {ref_w:.0f} px, P(closed)={ref_p:.3f} "
          f"(lower is better)\n")
    Path(CFG.paths.reports).mkdir(parents=True, exist_ok=True)

    conditions = [
        ("clean", ref),
        ("dusk", degrade_night(ref, 0.42, 5)),
        ("night", degrade_night(ref, 0.18, 9)),
        ("deep night", degrade_night(ref, 0.09, 12)),
        ("bright sun", degrade_sun(ref, 2.4)),
        ("harsh sun", degrade_sun(ref, 3.6)),
        ("backlit", degrade_backlit(ref, 0.75)),
    ]

    tracker = FaceTracker()
    norm = LightingNormalizer(enabled=True)

    print("=" * 84)
    print(f"{'condition':<12} | {'--- normaliser OFF ---':^30} | "
          f"{'--- normaliser ON ---':^32}")
    print(f"{'':<12} | {'face':>5} {'EAR':>6} {'P(closed)':>10} {'px':>5} | "
          f"{'face':>5} {'EAR':>6} {'P(closed)':>10} {'detected as':>12}")
    print("=" * 84)

    rows, tiles = [], []
    for name, img in conditions:
        off = measure(img, tracker, model, device, None)
        norm.reset()                           # fresh start per condition
        on = measure(img, tracker, model, device, norm)
        rows.append((name, off, on))
        tiles.append((name, norm.process(img)))

        print(f"{name:<12} | {str(off['found']):>5} {off['ear']:>6.3f} "
              f"{off['p']:>10.3f} {off['width']:>5.0f} | "
              f"{str(on['found']):>5} {on['ear']:>6.3f} {on['p']:>10.3f} "
              f"{LIGHT_TEXT.get(on['cond'], '-'):>12}")

    print("=" * 84)

    # A condition counts as WORKING when the face is found AND the open eye is
    # not misread as closed.
    thr = CFG.drowsy.cnn_closed_thresh
    ok_off = sum(1 for _, o, _ in rows if o["found"] and o["p"] < thr)
    ok_on = sum(1 for _, _, n in rows if n["found"] and n["p"] < thr)
    print(f"\nconditions handled correctly:  OFF {ok_off}/{len(rows)}   "
          f"ON {ok_on}/{len(rows)}")
    for name, o, n in rows:
        o_ok = o["found"] and o["p"] < thr
        n_ok = n["found"] and n["p"] < thr
        if o_ok != n_ok:
            print(f"  {name:<12} {'FIXED' if n_ok else 'BROKEN'} by the normaliser")

    # cost per frame
    t0 = time.time()
    for _ in range(60):
        norm.process(ref)
    ms = (time.time() - t0) / 60 * 1000
    print(f"\nnormaliser cost: {ms:.2f} ms/frame "
          f"({1000/ms:.0f} FPS ceiling on its own)")

    # contact sheet so the effect is visible, not just tabulated
    thumbs = []
    for name, img in tiles:
        t = cv2.resize(img, (320, 180))
        cv2.rectangle(t, (0, 0), (320, 22), (0, 0, 0), -1)
        cv2.putText(t, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
    while len(thumbs) % 4:
        thumbs.append(np.zeros((180, 320, 3), np.uint8))
    grid = np.vstack([np.hstack(thumbs[i:i + 4])
                      for i in range(0, len(thumbs), 4)])
    out = Path(CFG.paths.reports) / "lighting_normalised.png"
    cv2.imwrite(str(out), grid)

    raw = [cv2.resize(i, (320, 180)) for _, i in conditions]
    while len(raw) % 4:
        raw.append(np.zeros((180, 320, 3), np.uint8))
    grid2 = np.vstack([np.hstack(raw[i:i + 4]) for i in range(0, len(raw), 4)])
    out2 = Path(CFG.paths.reports) / "lighting_raw.png"
    cv2.imwrite(str(out2), grid2)
    print(f"[saved] {out.name} and {out2.name}")

    tracker.close()


if __name__ == "__main__":
    main()
