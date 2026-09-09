"""
Turn the raw MRL zip into clean numpy arrays ready for training.

    python scripts/prepare_data.py

MRL filename format (this is documented by the dataset authors):

    s0001_00001_0_0_0_0_0_01.png
     |     |    | | | | |  |
     |     |    | | | | |  +-- sensor id (01/02/03)
     |     |    | | | | +----- lighting     0=bad      1=good
     |     |    | | | +------- reflections  0=none 1=small 2=big
     |     |    | | +--------- EYE STATE    0=CLOSED   1=OPEN     <-- our label
     |     |    | +----------- glasses      0=no       1=yes
     |     |    +------------- gender       0=man      1=woman
     |     +------------------ image number
     +------------------------ SUBJECT ID                         <-- our split key

=============================================================================
THE MOST IMPORTANT IDEA IN THIS ENTIRE PROJECT: SUBJECT-WISE SPLITTING
=============================================================================

You must test a model on data it has never seen. Everyone knows that. But
"never seen" is subtler than it sounds.

The dataset has ~85,000 images from only ~37 people. Each person contributed
thousands of near-identical frames of THEIR eye.

If you shuffle all 85,000 images randomly and take 15% as a test set, then
person s0012 appears in the training set AND in the test set. The model does
not have to learn "what does a closed eye look like" -- it can learn
"this particular eyelid shape and this skin texture belongs to s0012, and
s0012's eye is usually open". Then it recognises s0012 in the test set and
scores brilliantly.

You would report 99.5% accuracy, feel great, then point it at your own face
and watch it fail completely. This is called DATA LEAKAGE and it is the single
most common way ML projects lie to their authors.

THE FIX: split by PERSON, not by image. Every image of s0012 goes to exactly
one split. The test set then contains only faces the model has genuinely never
seen -- which is exactly the situation on your webcam.

Subject-wise accuracy is LOWER (expect ~95-97% instead of ~99.5%) and that
lower number is the HONEST one. Report it. Being able to explain this in a
viva is worth more than a fake 99.5%.
"""
import sys
import json
import zipfile
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CFG                      # noqa: E402
from src.preprocess import preprocess_eye       # noqa: E402

ZIP_NAME = "mrlEyes_2018_01.zip"


def unzip_if_needed(raw_dir: Path) -> Path:
    """Extract the archive once; skip if already extracted."""
    extracted = raw_dir / "mrlEyes_2018_01"
    if extracted.exists() and any(extracted.rglob("*.png")):
        print(f"[skip] already extracted at {extracted}")
        return extracted

    zip_path = raw_dir / ZIP_NAME
    if not zip_path.exists():
        sys.exit(f"[error] {zip_path} not found. Download it first.")

    print(f"[unzip] {zip_path.name} ({zip_path.stat().st_size/1e6:.0f} MB) ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_dir)
    print("[unzip] done")
    return extracted


def parse_name(p: Path):
    """
    Pull (subject, eye_state, glasses, lighting) out of the filename.
    Returns None for anything that does not match the expected format,
    so one stray file cannot crash the whole run.
    """
    parts = p.stem.split("_")
    if len(parts) < 8:
        return None
    try:
        return {
            "subject": parts[0],
            "state": int(parts[4]),      # 0 closed, 1 open
            "glasses": int(parts[3]),
            "lighting": int(parts[6]),
        }
    except ValueError:
        return None


def main():
    raw = Path(CFG.paths.raw)
    out = Path(CFG.paths.processed)
    out.mkdir(parents=True, exist_ok=True)

    root = unzip_if_needed(raw)

    # ---- 1. index every image, grouped by the person it came from ----------
    print("[scan] indexing images ...")
    by_subject = defaultdict(list)
    skipped = 0
    for p in root.rglob("*.png"):
        meta = parse_name(p)
        if meta is None:
            skipped += 1
            continue
        by_subject[meta["subject"]].append((p, meta))

    total = sum(len(v) for v in by_subject.values())
    print(f"[scan] {total:,} images from {len(by_subject)} subjects "
          f"({skipped} unparseable files skipped)")
    if total == 0:
        sys.exit("[error] no images found - is the archive complete?")

    # ---- 2. split BY SUBJECT (see the essay at the top of this file) -------
    subjects = sorted(by_subject.keys())
    rng = np.random.default_rng(CFG.data.seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(round(n * CFG.data.test_frac)))
    n_val = max(1, int(round(n * CFG.data.val_frac)))
    split_of = {}
    for s in subjects[:n_test]:
        split_of[s] = "test"
    for s in subjects[n_test:n_test + n_val]:
        split_of[s] = "val"
    for s in subjects[n_test + n_val:]:
        split_of[s] = "train"

    counts = defaultdict(int)
    for s, sp in split_of.items():
        counts[sp] += 1
    print(f"[split] subjects -> train {counts['train']} | "
          f"val {counts['val']} | test {counts['test']}")
    print("[split] no subject appears in more than one split (no leakage)")

    # ---- 3. preprocess every image through the SAME function inference uses -
    size = CFG.data.img_size
    buffers = {k: {"X": [], "y": [], "subj": [], "glasses": [], "light": []}
               for k in ("train", "val", "test")}

    done = 0
    for subject, items in by_subject.items():
        sp = split_of[subject]
        b = buffers[sp]
        for path, meta in items:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # preprocess_eye returns float [0,1]; store as uint8 to keep the
            # arrays 4x smaller. Lossless here because the source was uint8.
            arr = (preprocess_eye(img, size) * 255.0).round().astype(np.uint8)
            b["X"].append(arr)
            # label: 1 = CLOSED (the thing we are detecting), 0 = open.
            # MRL encodes 1=open, so we invert it.
            b["y"].append(0 if meta["state"] == 1 else 1)
            b["subj"].append(subject)
            b["glasses"].append(meta["glasses"])
            b["light"].append(meta["lighting"])
            done += 1
            if done % 10000 == 0:
                print(f"[prep] {done:,}/{total:,}")

    # ---- 4. save ----------------------------------------------------------
    manifest = {"img_size": size, "classes": ["open", "closed"], "splits": {}}
    for sp, b in buffers.items():
        if not b["X"]:
            continue
        X = np.stack(b["X"])
        y = np.array(b["y"], dtype=np.uint8)
        np.save(out / f"{sp}_X.npy", X)
        np.save(out / f"{sp}_y.npy", y)
        np.save(out / f"{sp}_glasses.npy", np.array(b["glasses"], np.uint8))
        np.save(out / f"{sp}_light.npy", np.array(b["light"], np.uint8))

        n_closed = int(y.sum())
        manifest["splits"][sp] = {
            "n": int(len(y)),
            "open": int(len(y) - n_closed),
            "closed": n_closed,
            "pct_closed": round(100.0 * n_closed / len(y), 2),
            "subjects": sorted(set(b["subj"])),
        }
        print(f"[save] {sp:5s} {len(y):>7,} images   "
              f"open {len(y)-n_closed:>6,}  closed {n_closed:>6,}  "
              f"({100.0*n_closed/len(y):.1f}% closed)")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote arrays + manifest.json to {out}")


if __name__ == "__main__":
    main()
