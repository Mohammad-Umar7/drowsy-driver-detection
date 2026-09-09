# 02 — What a CNN is, with actual numbers

No jargon. Follow the arithmetic and you will understand it.

---

## Part 1: A picture is a grid of numbers

Here is a tiny 6×6 grayscale picture. Every number is a brightness: **0 = black,
255 = white**.

```
        c1   c2   c3   c4   c5   c6
r1     200  200  200  200  200  200      bright (skin)
r2     200  200  200  200  200  200      bright (skin)
r3      20   20   20   20   20   20      DARK  <- an eyelash line
r4     200  200  200  200  200  200      bright
r5     200  200  200  200  200  200      bright
r6     200  200  200  200  200  200      bright
```

You can see a dark horizontal line at row 3. **How do we make a computer notice
it?** That is the entire question a CNN answers.

---

## Part 2: A kernel is just 9 numbers

A **kernel** (also called a *filter*) is a small grid of numbers. Ours is
3 rows × 3 columns.

3 × 3 = **9 numbers**. That is all "9 numbers" means. Here is one:

```
        -1   -1   -1
         0    0    0
        +1   +1   +1
```

It is a little stamp. On its own it means nothing. It becomes meaningful when
you *slide it over the picture*.

---

## Part 3: Sliding it — the actual arithmetic

**The operation:** lay the 3×3 kernel on top of a 3×3 chunk of the picture.
Multiply each kernel number by the picture number underneath it. Add up all 9
results. That gives **one** output number.

### Position A — top-left (rows 1–3, columns 1–3)

The chunk of picture underneath:

```
picture chunk        kernel
200  200  200        -1  -1  -1
200  200  200         0   0   0
 20   20   20        +1  +1  +1
```

Multiply each pair, then add all nine:

```
row 1:   200×(-1) + 200×(-1) + 200×(-1)  =  -200 -200 -200  =  -600
row 2:   200×( 0) + 200×( 0) + 200×( 0)  =     0 +  0 +  0  =     0
row 3:    20×(+1) +  20×(+1) +  20×(+1)  =   +20 + 20 + 20  =   +60
                                                        ------------
                                          TOTAL          =   -540
```

**−540.** A big number. The kernel reacted strongly here.

### Position B — slide down one row (rows 2–4)

```
picture chunk        kernel
200  200  200        -1  -1  -1
 20   20   20         0   0   0
200  200  200        +1  +1  +1
```

```
row 1:   200×(-1) ×3  =  -600
row 2:    20×( 0) ×3  =     0
row 3:   200×(+1) ×3  =  +600
                        -------
              TOTAL   =      0
```

**0.** No reaction at all.

### Position C — slide down one more row (rows 3–5)

```
 20   20   20        -1  -1  -1
200  200  200         0   0   0
200  200  200        +1  +1  +1
```

```
row 1:    20×(-1) ×3  =   -60
row 2:   200×( 0) ×3  =     0
row 3:   200×(+1) ×3  =  +600
                        -------
              TOTAL   =  +540
```

**+540.** Big again, opposite sign.

### What just happened

| Where the kernel sat | Output |
|---|---|
| bright on top, dark below | **−540** |
| dark sandwiched in the middle | 0 |
| dark on top, bright below | **+540** |

The output is **large exactly where the brightness changes**, and **zero in flat
areas**. This kernel is a **horizontal edge detector**.

That is not magic — it is a direct consequence of the 9 numbers we picked. The
`-1` row subtracts what is above, the `+1` row adds what is below. If the two
are equal (flat area) they cancel to zero. If they differ (an edge) they do not.

**And an eyelid is a horizontal edge.** That is why this works for eyes.

---

## Part 4: Slide it everywhere → a "feature map"

Do that at *every* position in the picture. The outputs form a new grid called a
**feature map**:

```
original picture              feature map (edge detector output)
 bright                         ~0    ~0    ~0     <- flat, nothing
 bright                       -540  -540  -540     <- EDGE FOUND
 DARK LINE                       0     0     0
 bright                       +540  +540  +540     <- EDGE FOUND
 bright                        ~0    ~0    ~0
```

The feature map is a map of *"where is there a horizontal edge?"*. We have turned
raw brightness into **meaning**.

---

## Part 5: Why not just look at all the pixels at once?

You could connect all 1,024 pixels of a 32×32 image to a layer of 1,000 neurons.
Two things go wrong:

**1. Size.** 1,024 × 1,000 = **1,024,000 knobs** in a single layer. Our whole CNN
uses 139,426 knobs *in total*.

**2. It has no idea what "next to" means.** It sees 1,024 unrelated numbers. If
the eye shifts 2 pixels right, every input changes and it must relearn from
scratch.

The kernel fixes both:

- It is **9 numbers reused at every position** instead of a million.
  → called **weight sharing**.
- It finds the edge **wherever it appears**, because the same stamp is applied
  everywhere.
  → called **translation invariance**.

That is the whole reason CNNs beat ordinary networks at images.

---

## Part 6: We do NOT pick the 9 numbers — training finds them

Here is the part that surprises people.

I chose `-1 -1 -1 / 0 0 0 / +1 +1 +1` by hand to demonstrate the idea. **In a
real CNN, nobody does that.**

The 9 numbers start as **random junk**:

```
 0.31  -0.08   0.55
-0.42   0.13  -0.27
 0.09   0.64  -0.11
```

Then training runs: show it an eye image, see how wrong the answer was, and nudge
all 9 numbers slightly in the direction that reduces the error. Repeat ~7,000
times.

After training, those 9 numbers have *drifted on their own* into something that
detects a useful pattern — an edge, a curve, a dark blob — because those patterns
are what reduce the error.

**Nobody programmed "look for an eyelid". The training discovered it.**

---

## Part 7: One kernel is not enough — use 32

One kernel finds one kind of pattern. Our first layer has **32 different
kernels**, each with its own 9 numbers, each learning to spot something
different: horizontal edges, vertical edges, diagonals, dark blobs, bright spots.

So the first layer turns:

```
1 picture (32×32)   ->   32 feature maps (32×32 each)
```

In `model.py` that is this line:

```python
nn.Conv2d(1, 32, kernel_size=3, padding=1)
     #    |   |        |
     #    |   |        +-- 3x3 kernel = 9 numbers each
     #    |   +----------- produce 32 feature maps
     #    +--------------- take 1 input image (grayscale)
```

Parameter count: 32 kernels × 9 numbers = **288 knobs**. Compare that to the
1,024,000 the naive approach needed.

---

## Part 8: Stack layers — simple patterns become complex ones

Feed those 32 feature maps into *another* conv layer. It now looks for patterns
**in the edge maps** — combinations of edges, which are corners and curves. Feed
those into a third layer and it finds combinations of corners — eyelid shapes,
iris-like rings.

```
layer 1   edges and gradients                  (dumb, generic)
layer 2   corners, curves, blobs
layer 3   eyelid shapes, iris-like regions
output    "open"  or  "closed"                 (meaningful)
```

Each layer builds on the one below. **This is what "deep" learning means** —
deep = many stacked layers, each making the representation slightly more
abstract.

---

## Part 9: The other pieces, in one line each

| Piece | What it does | Why |
|---|---|---|
| **ReLU** | `max(0, x)` — negatives become 0 | Without something non-linear between layers, 10 stacked layers collapse into 1 and the network can only draw straight lines |
| **MaxPool** | Each 2×2 block → keep only the biggest | Halves the size. Faster, and small shifts stop mattering |
| **BatchNorm** | Re-centres numbers between layers | Stops values exploding or vanishing; lets us train much faster |
| **Dropout** | Randomly switch off 30% of neurons *while training* | Stops the network leaning on any single neuron — fights memorisation |
| **Softmax** | Two raw scores → two probabilities summing to 1 | `[0.03, 0.97]` = "97% sure this eye is closed" |

---

## The whole thing in one paragraph

A picture is a grid of numbers. A kernel is 9 numbers you slide across that grid,
multiplying and adding, which produces a new grid showing where a particular
pattern occurs. Stack several layers of these and simple patterns (edges) combine
into complex ones (eyelids). The 9 numbers in every kernel start random and are
adjusted automatically by training until they detect whatever is useful. That is
a CNN.
