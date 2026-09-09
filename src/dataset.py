"""
PyTorch Dataset + data augmentation.

A "Dataset" in PyTorch is any object that answers two questions:
    len(ds)   -> how many samples do you have?
    ds[i]     -> give me sample i as (input_tensor, label)
That is the whole interface. The DataLoader then handles batching and shuffling.

=============================================================================
DATA AUGMENTATION - why we deliberately corrupt the training images
=============================================================================

Our model has 139,426 knobs and we have ~60,000 training images. With that much
capacity it can simply MEMORISE the training set: perfect training accuracy,
useless on anything new. That failure is called OVERFITTING.

Augmentation fights it by randomly distorting each image every time it is used,
so the model never sees the exact same picture twice. It is forced to learn the
underlying concept ("the eyelid covers the iris") instead of memorising pixels.

Every augmentation here corresponds to something that genuinely happens in a
real car. That is the rule for choosing them:

    shift/scale/rotate  -> the driver moves; the eye crop is never perfectly centred
    horizontal flip     -> a left eye looks like a mirrored right eye
    brightness/contrast -> tunnels, sunlight, night, oncoming headlights
    blur                -> motion blur from road vibration
    noise               -> sensor noise in low light
    random erasing      -> glasses frames, hair, glare patches occluding the eye

NOTE WHAT IS ABSENT: no vertical flip, no 90-degree rotation. An upside-down eye
never occurs in a car, so training on one wastes capacity teaching the model to
handle a case that cannot happen. Augment to match reality, not to maximise
variety.

Augmentation is applied to the TRAINING split ONLY. Validation and test must
stay clean, otherwise your measured accuracy is measuring a random distortion
instead of the model.
"""
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import CFG


class EyeDataset(Dataset):
    """
    Loads the .npy arrays written by scripts/prepare_data.py.

    The whole split is held in RAM as uint8 (60k x 32 x 32 = about 61 MB),
    so there is no disk I/O during training at all. That is why we can keep
    num_workers=0 and still saturate the GPU.
    """

    def __init__(self, split: str, augment: bool = False,
                 root: Path = None, img_size: int = None, seed: int = 0):
        root = Path(root or CFG.paths.processed)
        self.split = split
        self.augment = augment
        self.img_size = img_size or CFG.data.img_size

        xp, yp = root / f"{split}_X.npy", root / f"{split}_y.npy"
        if not xp.exists():
            raise FileNotFoundError(
                f"{xp} missing - run:  python scripts/prepare_data.py")

        self.X = np.load(xp)                       # (N, S, S) uint8
        self.y = np.load(yp).astype(np.int64)      # (N,)  0=open 1=closed
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    # -- the augmentation pipeline ------------------------------------------
    def _augment(self, img: np.ndarray) -> np.ndarray:
        r = self.rng
        s = self.img_size

        # 1. Horizontal flip (50%). A left eye mirrored looks like a right eye,
        #    so this genuinely doubles our effective dataset for free.
        if r.random() < 0.5:
            img = img[:, ::-1]

        # 2. Small affine jitter: rotate +/-12 deg, scale +/-10%, shift +/-10%.
        #    Our eye crop comes from landmarks that jitter slightly frame to
        #    frame, so the model must tolerate imperfect centring.
        if r.random() < 0.8:
            angle = r.uniform(-12, 12)
            scale = r.uniform(0.9, 1.1)
            tx, ty = r.uniform(-0.10, 0.10, 2) * s
            M = cv2.getRotationMatrix2D((s / 2, s / 2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (s, s), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

        # 3. Brightness / contrast. alpha scales contrast, beta shifts
        #    brightness. Simulates driving from a tunnel into sunlight.
        if r.random() < 0.7:
            alpha = r.uniform(0.7, 1.3)
            beta = r.uniform(-30, 30)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255) \
                    .astype(np.uint8)

        # 4. Motion blur from road vibration.
        if r.random() < 0.25:
            k = int(r.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 5. Sensor noise in low light.
        if r.random() < 0.25:
            noise = r.normal(0, r.uniform(3, 12), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 6. Random erasing: blank out a rectangle. Forces the model to use the
        #    WHOLE eye rather than fixating on one spot, so a glasses frame or a
        #    glare patch across part of the eye does not break it.
        if r.random() < 0.25:
            eh, ew = r.integers(4, 11, 2)
            y0 = int(r.integers(0, max(1, s - eh)))
            x0 = int(r.integers(0, max(1, s - ew)))
            img = img.copy()
            img[y0:y0 + eh, x0:x0 + ew] = int(r.integers(0, 256))

        return np.ascontiguousarray(img)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        img = self.X[i]
        if self.augment:
            img = self._augment(img)

        # uint8 [0,255] -> float32 [0,1]. This mirrors preprocess_eye() exactly,
        # which is what keeps training and live inference in agreement.
        #
        # We deliberately do NOT subtract a dataset mean/std here. CLAHE already
        # normalised contrast, and the BatchNorm after the first conv layer
        # handles the rest. One less number that could drift out of sync
        # between training and the webcam.
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return x, int(self.y[i])

    # -- helpers ------------------------------------------------------------
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y, minlength=2)

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights, for use in the loss function.

        If 90% of your data were "open", a lazy model could hit 90% accuracy by
        always predicting "open" and never learning anything. Weighting the loss
        makes each mistake on the rare class cost proportionally more.

        MRL is nearly balanced (~49% closed) so these weights land near 1.0 --
        but the code belongs here because your own collected data will NOT be
        balanced.
        """
        c = self.class_counts().astype(np.float64)
        w = c.sum() / (len(c) * np.maximum(c, 1))
        return torch.tensor(w, dtype=torch.float32)


def make_loaders(batch_size: int = None, num_workers: int = None,
                 root: Path = None):
    """
    Build the three DataLoaders.

    num_workers=0 on Windows by default: spawning worker processes is slow
    there, and our data already lives in RAM so there is nothing to wait for.
    On Linux, raising it to 4 can help.

    pin_memory=True lets the GPU copy data with DMA instead of going through
    the CPU -- a small free speedup.

    drop_last=True on the training loader discards a final partial batch, which
    keeps BatchNorm statistics stable (a batch of 3 gives noisy statistics).
    """
    bs = batch_size or CFG.train.batch_size
    nw = CFG.train.num_workers if num_workers is None else num_workers
    # Hard guard: Windows worker processes are fragile and give us nothing here.
    if os.name == "nt":
        nw = 0

    train = EyeDataset("train", augment=True, root=root, seed=CFG.data.seed)
    val = EyeDataset("val", augment=False, root=root)
    test = EyeDataset("test", augment=False, root=root)

    common = dict(num_workers=nw, pin_memory=torch.cuda.is_available(),
                  persistent_workers=nw > 0)
    return (
        DataLoader(train, batch_size=bs, shuffle=True, drop_last=True, **common),
        DataLoader(val, batch_size=bs * 2, shuffle=False, **common),
        DataLoader(test, batch_size=bs * 2, shuffle=False, **common),
        train, val, test,
    )
