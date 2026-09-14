# 04 — Running it off a phone, at night, and in the sun

Three separate problems. This covers the practical setup first, then why each
one is hard.

---

## 1. Using your phone as the camera

Your phone is a better dashcam than a laptop webcam: higher resolution, better
low-light sensor, and it mounts where you actually need it.

### Setup (Android)

1. Install **IP Webcam** (free, Play Store).
2. Open it, scroll to the bottom, tap **Start server**.
3. It shows a URL like `http://192.168.1.7:8080`. Note the IP.
4. Phone and PC must be on **the same WiFi network**.
5. On the PC:

```bash
python -m src.infer --source http://192.168.1.7:8080/video --rotate 270 --no-mirror
```

**DroidCam** works too — its stream URL is `http://<ip>:4747/video`.

### The two flags you will need

| Flag | Why |
|---|---|
| `--rotate 90/180/270` | A phone in a windscreen mount films sideways. MediaPipe needs an upright face or it finds nothing at all. If no face is detected, try each rotation. |
| `--no-mirror` | A webcam is mirrored so moving right moves you right on screen. A phone pointed at you is **not** a mirror — flipping it puts the eyes on the wrong sides. |

### Settings inside IP Webcam that matter

- **Video resolution → 1280×720.** Higher is not better here: 1080p triples the
  data over WiFi for eyes that are already big enough, and the added latency
  costs you more than the pixels gain.
- **Quality → ~60%.** Above that you are spending bandwidth on JPEG detail the
  32×32 eye crop throws away anyway.
- **Focus mode → continuous**, or it will hunt and blur.
- Disable **"Video orientation: auto"** so the rotation stays fixed and your
  `--rotate` value keeps working.

### Why a network camera needs its own code path

A USB webcam gives you a frame when you ask. A phone stream arrives at its own
rate whether or not you are ready, and frames queue up inside OpenCV.

If the stream delivers 30 fps and you process 20, you fall a further 10 frames
behind **every second**. After a minute you are watching the driver from 30
seconds ago. For a drowsiness alarm that is not a glitch — the warning would
describe a microsleep that already happened.

`src/source.py` runs a background thread that reads continuously and keeps only
the **newest** frame, discarding the rest. Dropping frames is correct here: for
a real-time safety signal, fresh beats complete.

Network reads also *fail*, which USB essentially never does. WiFi in a moving
car drops. So a failed read triggers a reconnect with capped exponential
backoff instead of killing the app.

---

## 2. Night

### The problem
At night every pixel is squashed into roughly 0–40. The eyelid edge still
exists, but the contrast across it is a couple of levels — below what landmark
detection can resolve. Worse, sensor noise is roughly constant, so as the
signal shrinks the noise comes to dominate. That is why night video is grainy.

### Measured effect

With one clean reference frame artificially darkened
(`python scripts/test_lighting.py`):

| Condition | P(closed), enhancement OFF | P(closed), ON |
|---|---|---|
| clean | 0.025 ✓ | 0.019 ✓ |
| dusk | 0.064 ✓ | 0.025 ✓ |
| **night** | **0.557 ✗** | **0.169 ✓** |
| **deep night** | **0.785 ✗** | **0.274 ✓** |

The decision threshold is 0.342. Without enhancement, night pushes **open** eyes
past it — the system would report you as asleep while you are wide awake. With
it, both night cases land back on the correct side.

Conditions handled correctly: **5/7 → 7/7**, at a cost of 4.09 ms/frame.

### What actually helps in a real car

Software enhancement recovers a signal that is *there but faint*. It cannot
invent one that was never captured. In a genuinely dark car at night:

- **An IR illuminator is the real answer.** A £10 850 nm LED array plus a
  camera with the IR filter removed gives a bright, evenly lit face that is
  completely invisible to the driver. This is exactly what production
  driver-monitoring systems use, and it is why the MRL dataset this model was
  trained on is infrared.
- **Dashboard glow is not enough**, but a dim warm cabin light aimed at the
  face is usually sufficient for a phone sensor.
- **Point the phone away from the windscreen.** Oncoming headlights sweeping
  through the frame force the phone's auto-exposure to hunt, and the resulting
  brightness pulsing is worse than steady dark.

---

## 3. Sunlight

### The problem
Direct sun **clips**. Once a pixel reaches 255 the information is gone — not
compressed, not hidden, never recorded. No amount of processing brings it back.

The common in-car case is worse than plain brightness: **backlighting**. Bright
sky through the windscreen with the driver's face in shadow. The frame *average*
looks perfectly normal, so a naive brightness check sees no problem at all,
while the face itself is crushed into near-black.

### What we found
Honest result: **sunlight already worked before any of this was added.**

| Condition | OFF | ON |
|---|---|---|
| bright sun | 0.010 ✓ | 0.024 ✓ |
| harsh sun | 0.022 ✓ | 0.039 ✓ |
| backlit | 0.016 ✓ | 0.031 ✓ |

The CLAHE already applied to the 32×32 eye crop in `preprocess.py` was
absorbing overexposure on its own. Normalisation makes those frames very
slightly *worse* (0.010 → 0.024), though both sit far below the 0.342
threshold.

That is worth stating rather than hiding. The feature earns its place on night,
not on sun.

**Known limitation:** the classifier labels the synthetic backlit frame as
`BRIGHT SUN` rather than `BACKLIT`. The correction applied is still adequate,
but the detection rule needs real backlit footage to tune properly — a
one-sided brightness ramp is not a faithful model of sky-through-windscreen.

### Practical mitigations
- Mount the phone so the **windscreen is behind the camera**, not behind you.
- A visor or hood over the lens costs nothing and prevents most direct glare.
- Sunglasses defeat the eye path entirely. Nothing in software fixes that; the
  system falls back to head-pose and yawn signals, and says so on the HUD.

---

## 4. Reading the display

The lighting condition is the chip in the top bar next to the frame rate:
`NORMAL` in grey, or `NIGHT` / `DIM` / `BRIGHT SUN` / `BACKLIT` in amber
whenever the scene is genuinely difficult — so a bad reading can be blamed on
the light rather than the driver. With correction switched off it reads
`(fix off)`.

The eye line in the metrics card is the other thing to watch here:

| Field | Meaning |
|---|---|
| `eyes [LR] 53px` | which eyes are trusted, and the wider eye's width in pixels. `[L-]` means the right eye is turned away and being ignored; the line turns amber when neither eye is usable |
| under ~30 px | the eyelid is only a few pixels tall and a 2 px landmark error is a 30 % error in EAR. Sit closer, or raise the capture resolution |

Press **`l`** to toggle enhancement on and off while watching, so you can see
what it is doing rather than trusting that it works. For the numbers behind
the chip — mean luminance, the gamma being applied — run
`python scripts/test_lighting.py`, which prints them per condition.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `could not open video source` | wrong IP, different network, server not started | Open the URL in a browser on the PC first — if it does not play there, it is a network problem, not this app |
| Video plays but no face found | frame is rotated | Try `--rotate 90`, `180`, `270` |
| Face found but eyes swapped | phone stream mirrored | Add or remove `--no-mirror` |
| Video lags further behind over time | frames queueing | Already handled by the threaded reader; if it persists, lower the phone's resolution and quality |
| Stream stutters and reconnects | weak WiFi | Move the phone closer to the router, or drop to 640×480 in the app |
| `eye px` under 30 | phone too far away | Mount it closer, or raise the resolution |
| Says drowsy at night | enhancement off | Press `l`, or check you did not pass `--no-enhance` |
