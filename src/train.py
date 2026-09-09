"""
Training loop.

    python -m src.train                    # defaults from config.py
    python -m src.train --epochs 15 --arch mobilenet

=============================================================================
HOW TRAINING ACTUALLY WORKS (the whole thing in one page)
=============================================================================

The model has 139,426 knobs ("weights"), currently random. We want the values
that make it right most often. We find them by repeating four steps millions of
times:

  1. FORWARD PASS   Push a batch of 256 eye images through the network.
                    Out come 256 pairs of scores, currently nonsense.

  2. LOSS           Compare the scores to the true answers with a single number
                    that measures wrongness. We use cross-entropy: it is small
                    when the model is confidently right and grows very large
                    when it is confidently WRONG. That asymmetry is what pushes
                    the model away from overconfident mistakes.

  3. BACKWARD PASS  Calculus. For every one of the 139,426 knobs, compute
                    "if I nudge this knob up slightly, does the loss go up or
                    down, and by how much?" That is the GRADIENT. PyTorch does
                    this automatically (autograd) by tracking every operation
                    performed in the forward pass and applying the chain rule
                    backwards through it. This is BACKPROPAGATION.

  4. STEP           Nudge every knob a small distance in the direction that
                    lowered the loss. How far is the LEARNING RATE. Too big and
                    you overshoot and diverge; too small and training crawls.

One pass over the whole training set = one EPOCH. We do 30 of them.

Key vocabulary you will see below:
  BATCH        256 images processed at once. GPUs are parallel, so 256 images
               costs barely more than 1. It also averages out noise: a single
               weird image cannot yank the weights around.
  OPTIMIZER    The rule for step 4. We use AdamW, which adapts the step size
               per-knob based on recent gradient history. Far more forgiving
               than plain gradient descent.
  SCHEDULER    Changes the learning rate over time. We warm up (start small so
               the initially-random model does not explode) then decay along a
               cosine curve to near zero (so it settles precisely at the end).
  AMP          Automatic Mixed Precision. Does most of the maths in 16-bit
               instead of 32-bit. About 2x faster on your RTX 4070 and uses
               half the memory, with no measurable accuracy loss.
  EARLY STOP   Watch validation accuracy. When it stops improving for N epochs,
               stop -- further training would only overfit.

WHY THREE SPLITS:
  train  the model learns from these
  val    we peek at these every epoch to pick the best checkpoint and decide
         when to stop. Because we make decisions using it, it is slightly
         "used up" and no longer a fair test.
  test   touched exactly ONCE, at the very end. This is the number you report.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from .config import CFG
from .dataset import make_loaders
from .model import build_model, count_params


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scheduler(optimizer, epochs: int, steps_per_epoch: int,
                    warmup_epochs: int):
    """
    Learning-rate schedule: linear warmup, then cosine decay.

    Warmup: at step 0 the weights are random, so the first gradients are huge
    and a full-size step can wreck the model before it starts. Ramping the LR
    up over the first couple of epochs avoids that.

    Cosine decay: start big to explore quickly, end tiny to settle into a good
    minimum precisely. Reliably beats a constant LR by a percent or two.
    """
    total = epochs * steps_per_epoch
    warm = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_split(model, loader, device, criterion=None):
    """
    Run the model over a split without training. Returns metrics + raw scores.

    model.eval() matters: it switches Dropout OFF (we want the full network at
    inference) and makes BatchNorm use its accumulated running statistics
    instead of the current batch's. Forgetting model.eval() is a classic bug
    that makes validation accuracy look randomly bad.
    """
    model.eval()
    losses, probs, targets = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            losses.append(criterion(logits, y).item())
        probs.append(torch.softmax(logits, 1)[:, 1].float().cpu().numpy())
        targets.append(y.cpu().numpy())

    p = np.concatenate(probs)
    t = np.concatenate(targets)
    pred = (p >= 0.5).astype(int)

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float((pred == t).mean()),
        # Balanced accuracy = mean of per-class recall. Immune to a model that
        # wins by always guessing the majority class.
        "bal_acc": float(balanced_accuracy_score(t, pred)),
        # AUC measures ranking quality across ALL thresholds, so it tells us
        # how good the model is independently of where we set the cutoff.
        "auc": float(roc_auc_score(t, p)) if len(np.unique(t)) > 1 else float("nan"),
        "probs": p,
        "targets": t,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=CFG.train.epochs)
    ap.add_argument("--batch-size", type=int, default=CFG.train.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.train.lr)
    ap.add_argument("--arch", type=str, default=CFG.train.arch,
                    choices=["eyenet", "mobilenet"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(CFG.data.seed)
    np.random.seed(CFG.data.seed)
    # Lets cuDNN benchmark conv algorithms once and reuse the fastest. Free
    # speed when input size is fixed, which it is for us (always 32x32).
    torch.backends.cudnn.benchmark = True

    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = make_loaders(
        args.batch_size, args.workers)

    print("=" * 68)
    print(f"device        : {device}  "
          f"({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})")
    print(f"train / val / test : {len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,}")
    print(f"class balance (train) open/closed : {train_ds.class_counts().tolist()}")

    model = build_model(args.arch).to(device)
    print(f"architecture  : {args.arch}   parameters: {count_params(model):,}")

    # ---- loss --------------------------------------------------------------
    # label_smoothing=0.05 means we train toward [0.025, 0.975] instead of
    # [0, 1]. Chasing exactly 1.0 pushes the model to be blindly overconfident;
    # smoothing keeps its probabilities meaningful, which matters because we
    # FEED those probabilities into the fusion logic later.
    criterion = nn.CrossEntropyLoss(
        weight=train_ds.class_weights().to(device),
        label_smoothing=CFG.train.label_smoothing)

    # AdamW: Adam plus *decoupled* weight decay. Weight decay gently pulls all
    # weights toward zero, discouraging any single one from dominating -- a
    # regulariser that reduces overfitting.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=CFG.train.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, len(train_dl),
                                CFG.train.warmup_epochs)

    use_amp = (not args.no_amp) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"mixed precision: {use_amp}")
    print("=" * 68)

    Path(CFG.paths.checkpoints).mkdir(parents=True, exist_ok=True)
    Path(CFG.paths.reports).mkdir(parents=True, exist_ok=True)

    history, best_score, best_epoch, patience = [], -1.0, -1, 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()          # Dropout ON, BatchNorm uses batch statistics
        run_loss, run_correct, run_n = 0.0, 0, 0
        t0 = time.time()

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # set_to_none=True frees the gradient tensors instead of filling
            # them with zeros -- slightly faster and less memory.
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                logits = model(x)              # 1. forward
                loss = criterion(logits, y)    # 2. loss

            # 3. backward. In fp16, small gradients underflow to zero, so the
            # scaler multiplies the loss by a large factor first, then divides
            # it back out before the optimizer step. That is all GradScaler does.
            scaler.scale(loss).backward()

            # Unscale before clipping, or we would be clipping scaled values.
            scaler.unscale_(optimizer)
            # Gradient clipping: cap the total gradient size. Insurance against
            # one freak batch producing an enormous step that wrecks training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)             # 4. step
            scaler.update()
            scheduler.step()

            run_loss += loss.item() * y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            run_n += y.size(0)

        tr_loss = run_loss / run_n
        tr_acc = run_correct / run_n
        va = evaluate_split(model, val_dl, device, criterion)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        history.append({"epoch": epoch, "lr": lr_now,
                        "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va["loss"], "val_acc": va["acc"],
                        "val_bal_acc": va["bal_acc"], "val_auc": va["auc"],
                        "sec": dt})

        # Select on balanced accuracy, not raw accuracy: it cannot be gamed by
        # favouring the majority class.
        score = va["bal_acc"]
        flag = ""
        if score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "img_size": CFG.data.img_size, "epoch": epoch,
                        "val_bal_acc": score, "val_auc": va["auc"],
                        "classes": ["open", "closed"]},
                       CFG.paths.best_model)
            flag = "  <- best, saved"
        else:
            patience += 1

        print(f"ep {epoch:2d}/{args.epochs}  "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:5.2f}%  |  "
              f"val loss {va['loss']:.4f} acc {va['acc']*100:5.2f}% "
              f"bal {va['bal_acc']*100:5.2f}% auc {va['auc']:.4f}  "
              f"lr {lr_now:.2e}  {dt:.1f}s{flag}")

        if patience >= CFG.train.early_stop_patience:
            print(f"[early stop] no val improvement for {patience} epochs")
            break

    total = time.time() - t_start
    print("=" * 68)
    print(f"trained in {total/60:.1f} min | best epoch {best_epoch} "
          f"| best val balanced acc {best_score*100:.2f}%")

    # ---- final, single evaluation on the untouched test set ---------------
    ckpt = torch.load(CFG.paths.best_model, map_location=device)
    model.load_state_dict(ckpt["model"])
    te = evaluate_split(model, test_dl, device, criterion)
    print(f"TEST (unseen subjects): acc {te['acc']*100:.2f}%  "
          f"balanced {te['bal_acc']*100:.2f}%  auc {te['auc']:.4f}")

    Path(CFG.paths.reports, "history.json").write_text(json.dumps(
        {"history": history, "best_epoch": best_epoch,
         "test": {k: v for k, v in te.items() if k not in ("probs", "targets")},
         "arch": args.arch, "minutes": total / 60}, indent=2))
    print(f"[saved] {CFG.paths.best_model}")
    print(f"[saved] {Path(CFG.paths.reports,'history.json')}")
    print("next:  python -m src.evaluate")


if __name__ == "__main__":
    main()
