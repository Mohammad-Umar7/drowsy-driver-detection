# 03 — How training actually works

You have a network with 139,426 knobs, all set to random numbers. It currently
guesses eye states at coin-flip accuracy. This document explains exactly how
those knobs get fixed.

---

## 1. The core loop, in four steps

Training repeats these four steps thousands of times:

```
   1. FORWARD    push 256 eye images through the network -> 256 guesses
   2. LOSS       measure how wrong those guesses were -> one number
   3. BACKWARD   work out, for each of the 139,426 knobs, which way to turn it
   4. STEP       turn every knob a tiny amount in that direction
```

That is the whole algorithm. Everything else is refinement.

---

## 2. Step 2 — "Loss" is just a wrongness score

The network outputs two numbers per image, turned by softmax into probabilities:

```
image of a CLOSED eye  ->  model says  [open: 0.70, closed: 0.30]
                                                   ^^^^
                                        it gave the right answer only 0.30
```

**Cross-entropy loss** = `−log(probability it assigned to the correct answer)`.

```
correct answer got 0.30  ->  loss = −log(0.30) = 1.20     bad
correct answer got 0.90  ->  loss = −log(0.90) = 0.11     good
correct answer got 0.99  ->  loss = −log(0.99) = 0.01     great
correct answer got 0.01  ->  loss = −log(0.01) = 4.61     TERRIBLE
```

Notice the asymmetry: being confidently **wrong** is punished enormously
(4.61), while being confidently right barely registers (0.01). That is exactly
the behaviour we want — it pushes the model away from confident mistakes, which
in our case means missing a driver's closed eyes.

Loss for the batch = the average over all 256 images. **Lower is better.**
That single number is what the whole process minimises.

---

## 3. Step 3 — Gradients, without the calculus

Imagine standing on a foggy hillside, trying to reach the bottom. You cannot see
the valley. But you *can* feel the slope under your feet, so you step downhill.
Repeat until the ground is flat.

- the **hill** = the loss, as a function of all 139,426 knobs
- **your position** = the current knob values
- the **slope under your feet** = the **gradient**
- **step downhill** = adjust the knobs

The gradient answers one question per knob:

> "If I increase this knob slightly, does the loss go up or down, and how fast?"

A gradient of `+3.2` means increasing that knob raises the loss sharply, so we
should *decrease* it — and decrease it a lot, because the number is big. A
gradient of `−0.001` means it barely matters; nudge it up slightly.

### Backpropagation
Computing that for 139,426 knobs sounds impossible. It is not, because of the
**chain rule** from calculus.

PyTorch records every operation performed during the forward pass (this is
called the *computation graph*). When you call `loss.backward()`, it walks that
graph **backwards** from the loss to every knob, multiplying local derivatives
along the way. That is **backpropagation**. One backward pass costs about the
same as one forward pass, no matter how many knobs there are.

In `train.py` this is literally one line:

```python
scaler.scale(loss).backward()      # fills in .grad for all 139,426 knobs
```

---

## 4. Step 4 — The learning rate is the step size

```python
new_knob = old_knob − learning_rate × gradient
```

The learning rate controls how far you step downhill:

| Learning rate | What happens |
|---|---|
| Too large (e.g. 1.0) | You leap across the valley and land higher up. Loss becomes `nan`. Training dies. |
| Too small (e.g. 1e-7) | You inch along. Would take weeks. |
| Right (ours: 3e-3) | Steady, fast descent |

### Why our learning rate changes over time

```
LR
 |        ___________
 |       /           \___
 |      /                 \____
 |     /                        \______
 |____/                                 \________
 +----------------------------------------------- epoch
   warmup           cosine decay
   (2 epochs)
```

- **Warmup** — at step 0 the weights are random, so the first gradients are
  enormous. A full-size step would wreck the model before it starts. Ramping up
  gently avoids that.
- **Cosine decay** — start with big steps to cross the landscape quickly, then
  shrink them to settle precisely into the bottom of the valley. Reliably worth
  1–2% accuracy over a constant rate.

### AdamW, our optimizer
Plain gradient descent uses one step size for every knob. **Adam** keeps a
running memory of each knob's recent gradients and gives each its own adaptive
step size — knobs with consistently small gradients get bigger steps. The **W**
adds *weight decay*, gently pulling all knobs toward zero so no single one
dominates. This is a regulariser: it fights overfitting.

---

## 5. Vocabulary you will see in the logs

| Term | Meaning |
|---|---|
| **Batch** | 256 images processed together. GPUs are parallel, so 256 costs barely more than 1. It also averages out noise so one weird image cannot yank the weights around. |
| **Iteration / step** | One batch going through all four steps. |
| **Epoch** | One complete pass over all 59,636 training images. That is 233 batches. |
| **Parameters / weights** | The 139,426 knobs. |
| **Gradient** | Which way, and how hard, to turn each knob. |

---

## 6. Overfitting — how to spot it in your own logs

**Overfitting** = the model memorises the training images instead of learning
the concept.

Watch the two loss curves:

```
HEALTHY                            OVERFITTING
loss                               loss
 |\                                 |\
 | \  train                         | \  train
 |  \____                           |  \____
 |   \____ val                      |   \___
 |         (both fall together)     |        \____
 |                                  |    val  ____/-----   <- val rises again!
 +--------- epoch                   +--------- epoch
```

**The signal: training loss keeps falling while validation loss starts rising.**
From that point on the model is memorising, not learning. Everything after is
harmful.

### Our three defences

1. **Data augmentation** (`dataset.py`) — randomly distort every image so the
   model never sees the same picture twice. It cannot memorise what keeps
   changing.
2. **Dropout** (`model.py`) — randomly switch off 30% of neurons each step, so
   the network cannot lean on any single one.
3. **Early stopping** (`train.py`) — watch validation accuracy; when it stops
   improving for 7 epochs, stop. We also save the checkpoint from the **best**
   epoch, not the last one, so even if it starts overfitting we keep the good
   version.

---

## 7. Why three splits, not two

| Split | Used for | Touched how often |
|---|---|---|
| **train** (25 subjects) | the model learns from these | every epoch |
| **val** (6 subjects) | choose the best epoch, decide when to stop | every epoch, look only |
| **test** (6 subjects) | the final honest number | **exactly once, at the end** |

The subtle point: because we *make decisions* using validation (which checkpoint
to keep, when to stop), it is slightly "used up". Choosing the best of 30 epochs
by validation score means validation flatters us a little. The test set, touched
once, has no such contamination. **That is the number you report.**

---

## 8. Mixed precision (AMP) — free speed

Normally every number is a 32-bit float. Modern GPUs like your RTX 4070 have
dedicated hardware (Tensor Cores) that multiply **16-bit** numbers far faster.

AMP runs most operations in 16-bit while keeping the sensitive ones (the loss,
the weight updates) in 32-bit. Roughly **2× faster**, half the memory, no
measurable accuracy cost.

**The catch, and what `GradScaler` is for:** 16-bit floats cannot represent very
small numbers — anything below about `6e-8` becomes exactly 0. Many gradients
are that small, so they would vanish and those knobs would never learn.

The fix is simple: multiply the loss by a large number (say 65,536) *before*
`backward()`, so all gradients scale up into representable range, then divide
them back down before the optimizer step. `GradScaler` does exactly that, and
adjusts the factor automatically if it ever overflows.

---

## 9. How to read our training log

```
ep  7/30  train loss 0.1832 acc 94.10%  |  val loss 0.1547 acc 95.22% bal 95.05% auc 0.9891  lr 2.55e-03  12.4s  <- best, saved
```

| Field | Meaning |
|---|---|
| `ep 7/30` | epoch 7 of 30 |
| `train loss / acc` | performance on data it is learning from |
| `val loss / acc` | performance on 6 held-out people |
| `bal` | **balanced accuracy** — mean of per-class recall. Cannot be gamed by always guessing the majority class |
| `auc` | ranking quality across all thresholds. 1.0 perfect, 0.5 coin flip |
| `lr` | current learning rate (watch it warm up, then decay) |
| `<- best, saved` | validation improved, so this checkpoint was written to disk |

### What to look for
- **train acc far above val acc** (e.g. 99% vs 88%) → overfitting.
- **both stuck low** → underfitting: model too small, or learning rate wrong.
- **val loss rising while val acc also rises** → the model is getting more
  answers right but growing overconfident on the ones it gets wrong. Usually
  harmless, but it means the raw probabilities are less trustworthy — which
  matters for us, because we feed those probabilities into the fusion logic.
- **loss becomes `nan`** → learning rate too high, or a division by zero.

---

## 10. Accuracy is not the metric that matters

Covered fully in `src/evaluate.py`, but the short version:

```
                     PREDICTED
                  open      closed
TRUE   open   |    TN    |    FP      false alarm -> annoying
     closed   |    FN    |    TP      MISSED closure -> dangerous
```

- **Recall** = `TP / (TP + FN)` — of all genuinely closed eyes, how many did we
  catch? **This is the safety number.**
- **Precision** = `TP / (TP + FP)` — when we shout "closed", how often are we
  right? **This is the do-not-annoy-the-driver number.**

A false negative is much worse than a false positive here. `evaluate.py` prints
both, and picks an operating threshold rather than blindly assuming 0.5.
