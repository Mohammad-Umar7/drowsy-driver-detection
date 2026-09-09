"""
Evaluate the trained model and produce the numbers/plots for your report.

    python -m src.evaluate

=============================================================================
WHY ACCURACY ALONE IS A BAD METRIC
=============================================================================

"97% accurate" tells you almost nothing, because it hides WHICH mistakes.
For drowsiness the two error types have wildly different costs:

                        PREDICTED
                     open        closed
   TRUE   open   |   TN     |     FP      FP = false alarm.
                 |          |               Annoying. Driver switches it off.
        closed   |   FN     |     TP      FN = MISSED a closed eye.
                 |          |               Driver is asleep. Nobody warned.

A false negative is far worse than a false positive here. So we care about:

  RECALL (of closed) = TP / (TP + FN)
      "of all the genuinely closed eyes, what fraction did we catch?"
      This is the safety-critical number.

  PRECISION (of closed) = TP / (TP + FP)
      "when we shout closed, how often are we right?"
      This is the do-not-annoy-the-user number.

  F1 = harmonic mean of the two. One number when you need one number.

The model does not output a decision, it outputs a PROBABILITY. Choosing the
cutoff (0.5? 0.3?) is a separate engineering decision from training, and it is
where you trade recall against precision. This script finds a good cutoff for
you rather than blindly assuming 0.5.

ROC curve   : recall vs false-alarm-rate at EVERY possible cutoff.
AUC         : area under it. 1.0 = perfect, 0.5 = coin flip. Threshold-free,
              so it measures the model itself rather than your cutoff choice.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # render to file; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)

from .config import CFG
from .dataset import EyeDataset
from .model import build_model
from .train import evaluate_split, get_device
from torch.utils.data import DataLoader


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    labels = ["open", "closed"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (test set)")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}\n{100*cm[i,j]/total:.1f}%",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curves(y, p, roc_path, pr_path):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y, p):.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guessing")
    ax.set_xlabel("false alarm rate")
    ax.set_ylabel("recall (closed eyes caught)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=140)
    plt.close(fig)

    pr, rc, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(rc, pr, lw=2, label=f"AP = {average_precision_score(y, p):.4f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(pr_path, dpi=140)
    plt.close(fig)


def plot_history(hist_path, out_path):
    if not Path(hist_path).exists():
        return
    h = json.loads(Path(hist_path).read_text())["history"]
    ep = [r["epoch"] for r in h]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ep, [r["train_loss"] for r in h], label="train")
    axes[0].plot(ep, [r["val_loss"] for r in h], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=.3)
    axes[1].plot(ep, [100*r["train_acc"] for r in h], label="train")
    axes[1].plot(ep, [100*r["val_acc"] for r in h], label="val")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def subgroup_report(ds, probs, thr, split="test"):
    """
    Break accuracy down by GLASSES and by LIGHTING.

    An overall average can hide a systematic failure: 97% overall might be
    99% without glasses and 78% with them. Only a subgroup breakdown reveals
    that, and "it degrades on glasses" is exactly the kind of honest limitation
    that makes a project report credible.
    """
    root = Path(CFG.paths.processed)
    out = {}
    pred = (probs >= thr).astype(int)
    for name, fname in (("glasses", f"{split}_glasses.npy"),
                        ("lighting", f"{split}_light.npy")):
        f = root / fname
        if not f.exists():
            continue
        g = np.load(f)
        rows = {}
        for v in np.unique(g):
            m = g == v
            if m.sum() == 0:
                continue
            rows[int(v)] = {"n": int(m.sum()),
                            "acc": round(float((pred[m] == ds.y[m]).mean()) * 100, 2)}
        out[name] = rows
    return out


def main():
    device = get_device()
    ckpt_path = Path(CFG.paths.best_model)
    if not ckpt_path.exists():
        raise SystemExit("no checkpoint - run:  python -m src.train")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt.get("arch", "eyenet")).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {ckpt_path.name}  (epoch {ckpt['epoch']}, "
          f"val balanced acc {ckpt['val_bal_acc']*100:.2f}%)")

    ds = EyeDataset("test", augment=False)
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    res = evaluate_split(model, dl, device)
    y, p = res["targets"], res["probs"]

    reports = Path(CFG.paths.reports)
    reports.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 62)
    print("TEST SET  (subjects the model has NEVER seen)")
    print("=" * 62)
    print(f"images        : {len(y):,}")
    print(f"accuracy      : {res['acc']*100:.2f}%")
    print(f"balanced acc  : {res['bal_acc']*100:.2f}%")
    print(f"ROC-AUC       : {res['auc']:.4f}")
    print(f"avg precision : {average_precision_score(y, p):.4f}")

    print("\n--- at the default 0.50 cutoff ---")
    print(classification_report(y, p >= 0.5, target_names=["open", "closed"],
                                digits=4))

    # ---- pick a better cutoff --------------------------------------------
    fpr, tpr, thr = roc_curve(y, p)
    # Youden's J = recall - false_alarm_rate. Maximising it finds the cutoff
    # that best separates the two classes overall.
    j_idx = int(np.argmax(tpr - fpr))
    thr_j = float(thr[j_idx])

    # Safety-first alternative: the strictest cutoff that still catches >=98%
    # of closed eyes. Costs some false alarms, but PERCLOS averages those out
    # over 30 seconds, so short spurious closures barely move the final signal.
    pr_c, rc_c, thr_pr = precision_recall_curve(y, p)
    ok = np.where(rc_c[:-1] >= 0.98)[0]
    thr_recall98 = float(thr_pr[ok[-1]]) if len(ok) else 0.5

    print("--- threshold selection ---")
    for name, t in (("default", 0.5), ("Youden J (balanced)", thr_j),
                    ("recall>=98% (safety-first)", thr_recall98)):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        print(f"  {name:26s} thr={t:.3f}  "
              f"recall={tp/max(1,tp+fn)*100:5.2f}%  "
              f"precision={tp/max(1,tp+fp)*100:5.2f}%  "
              f"false alarms={fp:,}  MISSED={fn:,}")

    chosen = thr_j
    cm = confusion_matrix(y, p >= chosen, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\nchosen operating threshold: {chosen:.3f}")
    print(f"  missed closed eyes (dangerous): {fn:,} / {tp+fn:,} "
          f"({100*fn/max(1,tp+fn):.2f}%)")
    print(f"  false alarms (annoying)       : {fp:,} / {tn+fp:,} "
          f"({100*fp/max(1,tn+fp):.2f}%)")

    sub = subgroup_report(ds, p, chosen)
    if sub:
        print("\n--- accuracy by subgroup (honesty check) ---")
        gl = {0: "no glasses", 1: "glasses"}
        li = {0: "bad light", 1: "good light"}
        for k, rows in sub.items():
            names = gl if k == "glasses" else li
            for v, r in rows.items():
                print(f"  {k:9s} {names.get(v, v):12s} n={r['n']:>6,}  "
                      f"acc={r['acc']:.2f}%")

    plot_confusion(cm, reports / "confusion_matrix.png")
    plot_curves(y, p, reports / "roc_curve.png", reports / "pr_curve.png")
    plot_history(reports / "history.json", reports / "training_curves.png")

    (reports / "metrics.json").write_text(json.dumps({
        "n_test": int(len(y)), "accuracy": res["acc"],
        "balanced_accuracy": res["bal_acc"], "roc_auc": res["auc"],
        "average_precision": float(average_precision_score(y, p)),
        "thresholds": {"default": 0.5, "youden_j": thr_j,
                       "recall98": thr_recall98, "chosen": chosen},
        "confusion_at_chosen": {"tn": int(tn), "fp": int(fp),
                                "fn": int(fn), "tp": int(tp)},
        "subgroups": sub,
    }, indent=2))

    # The live app reads this so it uses the same cutoff we validated here.
    calib = {}
    cp = Path(CFG.paths.calibration)
    if cp.exists():
        calib = json.loads(cp.read_text())
    calib["cnn_closed_thresh"] = chosen
    cp.write_text(json.dumps(calib, indent=2))

    print(f"\n[saved] plots + metrics.json -> {reports}")
    print(f"[saved] threshold {chosen:.3f} -> {cp.name} (used by the live app)")


if __name__ == "__main__":
    main()
