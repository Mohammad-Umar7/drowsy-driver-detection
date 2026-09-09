# 01 — Foundations (assumes you know nothing)

Read this once, slowly. Everything else in the project builds on it.

---

## 1. What an image actually is to a computer

A computer has no idea what a "face" is. An image is just **a grid of numbers**.

A 640×480 webcam frame is a grid 640 wide and 480 tall = 307,200 cells. Each cell
is a **pixel**. Each pixel stores brightness as a number from **0 (black) to 255
(white)**.

```
a tiny 5x5 grayscale patch of an eyebrow:

 200 195 190 188 192
 180  90  40  35 100     <- the dark row is the eyebrow hair
 185  85  30  28  95
 190 150 120 130 160
 195 200 198 199 201
```

Colour images store **3 numbers per pixel** — Blue, Green, Red (OpenCV uses BGR
order, not RGB — a classic source of bugs). So a colour frame is
`480 × 640 × 3 = 921,600` numbers.

That is *all* your webcam gives us. Every single thing this project does is
arithmetic on that grid of numbers.

**Why we convert to grayscale:** whether an eye is open or shut is a matter of
*shape*, not *colour*. Dropping colour throws away 2/3 of the numbers, makes
everything 3× faster, and loses nothing useful.

---

## 2. The problem: 307,200 numbers, and we need "is the eye shut?"

Going straight from a grid of 307,200 numbers to "drowsy / not drowsy" is a huge
leap. So we break it into stages, each one shrinking the problem:

```
307,200 numbers (whole frame)
        │   MediaPipe FaceMesh
        ▼
      478 points (where the face parts are)          <- stage 1
        │   crop around the eyes
        ▼
   2 × 1,024 numbers (two 32x32 eye pictures)        <- stage 2
        │   our CNN
        ▼
      1 number  (probability the eye is closed)      <- stage 3
        │   watch it over 30 seconds
        ▼
      drowsy?  yes / no                              <- stage 4
```

This staged shrinking is the core idea of nearly all computer vision. Each stage
throws away information that does not matter for the next one.

---

## 3. Stage 1 — MediaPipe FaceMesh, explained properly

### What it is
**MediaPipe** is a free library from Google. **FaceMesh** is one model inside it.

You hand FaceMesh an image. It hands back **478 points** ("landmarks"), each one
an (x, y) coordinate saying *where on the image* a specific part of the face is.

Crucially, the points are **always in the same order, every time, for everyone**:

| Index | Always means |
|-------|--------------|
| 1     | tip of the nose |
| 33    | outer corner of the right eye |
| 133   | inner corner of the right eye |
| 159   | middle of the right upper eyelid |
| 145   | middle of the right lower eyelid |
| 13    | centre of the upper lip |
| 14    | centre of the lower lip |
| 199   | chin |

Point 33 is the outer corner of the right eye in **your** face, in **my** face,
in a photo of anyone, at any angle. That fixed ordering is what makes the whole
project possible — we can write `landmarks[33]` and know exactly what we get.

That is why `src/geometry.py` has hard-coded lists like
`RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]`. Those are not random numbers;
they are the six specific points around the right eye that we need.

### How it works inside (the honest version)
FaceMesh is itself a neural network, trained by Google on millions of
hand-labelled face photos. It runs in two steps:

1. **Face detector** — scans the frame, finds a box containing a face.
2. **Landmark model** — looks only inside that box and predicts the 478 points.

After the first frame it **tracks** instead of re-detecting: it assumes your face
did not teleport, so it looks near where the face was last frame. Tracking is far
cheaper than detecting, which is why it hits 100+ FPS.

### Why we do NOT train this ourselves
Training a face-landmark model from scratch needs millions of labelled images,
weeks of GPU time, and money. Google already did it and gave it away free.

**Using it is engineering, not cheating.** Nobody forges their own screws to build
a car. Our actual contribution is the eye-state model and the time-based logic —
that is where the real work is, and that is what we build ourselves.

### The coordinate detail that bites everyone
MediaPipe returns coordinates **normalised to 0–1**, not pixels.
`x = 0.5` means "halfway across the image", whatever the resolution.

To get pixels you multiply: `px = x * width`, `py = y * height`. In
`src/face.py` that is this line:

```python
pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
```

Forget it, and every downstream number is garbage.

---

## 4. EAR — turning 6 points into "how open is this eye"

We have the 6 points around an eye. Now we need one number.

```
        p2      p3          p2, p3 = upper eyelid
   p1 ·----------· p4       p1, p4 = the two corners
        p6      p5          p5, p6 = lower eyelid
```

**Eye Aspect Ratio:**

```
        |p2 − p6|  +  |p3 − p5|
EAR = ─────────────────────────
             2 × |p1 − p4|
```

`|a − b|` means the straight-line distance between two points
(Pythagoras: `√((x₁−x₂)² + (y₁−y₂)²)`).

- The **top** is how *tall* the eye is, measured in two places and added.
- The **bottom** is how *wide* the eye is, doubled (because we added two heights).

So EAR is literally **height ÷ width**.

### Why divide at all? (this is the clever bit)
If you lean toward the camera, your eye takes up more pixels — height *and* width
both grow. Because we divide one by the other, the ratio **stays the same**.

That property is called **scale invariance**, and we proved it in the test above:
the same eye scaled 3× larger still returned EAR = 1.2.

Without dividing, "eye is 8 pixels tall" would mean *open* when you sit close and
*shut* when you sit far away. Useless.

### Typical values
| State | EAR |
|-------|-----|
| Eye wide open | 0.28 – 0.35 |
| Eye half open | 0.18 – 0.22 |
| Eye shut | 0.05 – 0.12 |

**But these are person-specific.** Someone with naturally narrow eyes might sit at
0.19 wide awake, and a fixed threshold of 0.21 would scream that they are asleep
all day. That is why the app has a **calibration** step: it measures *your*
open-eye EAR for a few seconds and sets your threshold to 75% of it.

**MAR (Mouth Aspect Ratio)** is the identical idea applied to the lips — mouth
height ÷ mouth width. High and *sustained* = a yawn.

---

## 5. Stage 3 — What a CNN is (from zero)

### First: what "a model" even means
A model is a function with **adjustable knobs**. You feed it an input, it produces
an output. The knobs (called **weights** or **parameters**) start as random
numbers, so the output starts as nonsense.

**Training** = showing it thousands of examples where you already know the right
answer, and nudging the knobs a little each time so the output gets closer.

Our model has ~200,000 knobs. We never set a single one by hand — the training
loop finds them all.

### The naive approach, and why it fails
A 32×32 eye picture is 1,024 numbers. You *could* connect all 1,024 to a layer of
neurons that each multiply-and-add them. That is a "fully connected" network.

Two fatal problems:

1. **Explosion.** 1,024 inputs × 1,000 neurons = a million knobs, in one layer.
2. **No position sense.** It treats the pixel at top-left and the pixel at centre
   as unrelated inputs. If the eye shifts 2 pixels right, it sees a completely
   different input and has to re-learn from scratch.

### The CNN fix: slide a small window
**CNN = Convolutional Neural Network.** Its trick: instead of looking at the whole
image at once, slide a tiny window (a **kernel**, typically 3×3) across it, doing
the same small calculation everywhere.

A 3×3 kernel is just 9 numbers. At each position you multiply the 9 kernel numbers
against the 9 pixels underneath and add up the result:

```
image patch        kernel           multiply & sum
 10  10  10       -1  -1  -1
 10  10  10   ×    0   0   0   =  (10×-1)+(10×-1)+(10×-1)
200 200 200        1   1   1       + 0+0+0
                                   + (200×1)+(200×1)+(200×1)
                                   = -30 + 600 = 570   <- big number!
```

That kernel fires strongly on a **horizontal edge** (dark above, bright below).
Slide it over the whole image and you get a new grid — a **feature map** — that
lights up wherever a horizontal edge exists.

An eyelid *is* a horizontal edge. This is not a coincidence; it is why CNNs work
on this problem.

### Why this solves both problems
1. **Tiny.** That kernel is 9 numbers, reused at every position, instead of a
   million. This is called **weight sharing**.
2. **Position-independent.** The same kernel is applied everywhere, so an edge is
   detected whether it sits at the top-left or the centre. This is called
   **translation invariance** — and it is exactly why a CNN beats a plain network
   at images.

### Stacking layers = building up meaning
One layer finds edges. Feed those feature maps into another conv layer and it
finds combinations of edges — corners, curves. Another layer finds combinations
of those — "an eyelid crease", "a round dark iris". The last layer combines those
into "open" or "closed".

```
layer 1:  edges, gradients          (simple)
layer 2:  corners, curves, blobs
layer 3:  eyelid shapes, iris-like regions
output :  open / closed             (meaningful)
```

Nobody programs "look for an iris". The training process discovers that on its
own, because iris-detecting kernels happen to reduce the error.

### The other pieces you will see in `model.py`
- **ReLU** — `max(0, x)`. Throws away negatives. Without something non-linear
  between layers, stacking layers is pointless: a chain of multiplications
  collapses into one big multiplication, and the network could only ever learn
  straight lines.
- **MaxPool** — takes each 2×2 block and keeps only the largest value. Halves the
  width and height. Makes the network faster and a bit more tolerant of shifts.
- **BatchNorm** — re-centres the numbers flowing between layers so they do not
  drift to huge or tiny values. Makes training much faster and more stable.
- **Dropout** — during training only, randomly zeroes some neurons. Forces the
  network to not depend on any single one. Fights **overfitting**.
- **Softmax** — turns the final two raw scores into two probabilities that add to
  1.0, e.g. `[0.03, 0.97]` = "97% sure this eye is closed".

### Overfitting — the thing that will bite you
**Overfitting** = the model memorises the training images instead of learning the
concept. Symptom: 99.8% accuracy on training data, 71% on data it has never seen.

It is the single most common failure in student ML projects, and we defend against
it three ways: a held-out test set, data augmentation, and dropout. All three are
covered in `docs/03-training.md`.

---

## 6. Stage 4 — Why one frame is never enough

Here is the insight that separates a real system from a toy demo:

> **A closed eye is not drowsiness. A closed eye for 800 milliseconds is.**

You blink every few seconds. A blink takes 100–400 ms. If we alarmed on any closed
frame, it would scream constantly and you would switch it off.

So we track eye state **over time** and compute:

- **PERCLOS** — *PERcentage of eye CLOSure*: the fraction of the last 30 seconds
  your eyes were shut. Above ~15% is the real automotive-industry drowsiness
  threshold. This is the single most validated fatigue metric in the research.
- **Microsleep** — one continuous closure ≥ 0.8 s. Immediate alarm, no waiting.
- **Yawn rate** — yawns per minute.
- **Head nod** — pitch angle dropping and staying down.

This time-based layer lives in `src/drowsiness.py` and is genuinely the brain of
the project. The CNN just answers "open or closed" for one eye in one frame.

---

## 7. Where each idea lives in the code

| Concept | File |
|---------|------|
| Landmarks, eye crops | `src/face.py` |
| EAR, MAR, head pose maths | `src/geometry.py` |
| Pixels → model input | `src/preprocess.py` |
| The CNN itself | `src/model.py` |
| Training loop | `src/train.py` |
| Accuracy, confusion matrix, ROC | `src/evaluate.py` |
| PERCLOS / microsleep / fusion | `src/drowsiness.py` |
| Live webcam app | `src/infer.py` |
| Every tunable number | `src/config.py` |
