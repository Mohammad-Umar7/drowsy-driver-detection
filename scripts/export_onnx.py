"""
Export the trained CNN to ONNX so it can run on the phone.

    python scripts/export_onnx.py

=============================================================================
WHY ONNX
=============================================================================

PyTorch itself does not run on Android. The model has to be converted into a
portable format that a mobile runtime can load.

ONNX (Open Neural Network Exchange) is a file format that stores the network's
STRUCTURE and its WEIGHTS in a framework-independent way. We export from
PyTorch, and ONNX Runtime on Android loads the result. Nothing about the maths
changes; only the container does.

The alternative is TFLite, which is more common on Android but needs a
PyTorch -> ONNX -> TensorFlow -> TFLite chain, and every hop is a chance for a
silent numerical difference. ONNX Runtime has a first-class Android build, so
one hop is enough.

=============================================================================
TWO DECISIONS WORTH EXPLAINING
=============================================================================

1. SOFTMAX IS BAKED INTO THE GRAPH.
   model.py returns raw logits, because PyTorch's CrossEntropyLoss applies
   softmax internally during training. For the phone we export logits ->
   softmax -> probabilities, so the Kotlin side reads P(closed) directly.

   That is not cosmetic. Every line of maths reimplemented on the other side
   is a line that can drift from the Python. Moving softmax inside the graph
   removes one such line permanently.

2. THE BATCH AXIS IS DYNAMIC.
   We feed two eyes per frame, but the exporter would otherwise hard-code
   whatever batch size the dummy input had. Marking the axis dynamic lets the
   same file take 1 eye or 2 without re-exporting.

=============================================================================
VERIFICATION IS THE POINT OF THIS SCRIPT
=============================================================================

Exporting is one line. The rest of this file checks that the exported model
gives the SAME ANSWERS as the original on real data.

An export can succeed and still be subtly wrong - an unsupported op silently
approximated, a shape mis-inferred. If that happens, the phone app quietly
gets worse than the desktop one and nothing tells you. So we run both models
over the whole test set and compare.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG                    # noqa: E402
from src.model import build_model             # noqa: E402

OUT = Path(CFG.paths.root) / "android" / "app" / "src" / "main" / "assets"
OPSET = 17


class ProbWrapper(nn.Module):
    """Wraps the network so the exported graph outputs probabilities."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


def main():
    ckpt_path = Path(CFG.paths.best_model)
    if not ckpt_path.exists():
        sys.exit("no checkpoint - run:  python -m src.train")

    ck = torch.load(ckpt_path, map_location="cpu")
    net = build_model(ck.get("arch", "eyenet"))
    net.load_state_dict(ck["model"])
    net.eval()

    model = ProbWrapper(net).eval()
    size = CFG.data.img_size
    dummy = torch.randn(2, 1, size, size)

    OUT.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT / "eyenet.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["eye"], output_names=["prob"],
        # "eye" is (batch, 1, 32, 32) and "prob" is (batch, 2). Only the batch
        # axis varies, so only that one is declared dynamic.
        dynamic_axes={"eye": {0: "batch"}, "prob": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,      # fold BatchNorm into the conv weights
    )
    kb = onnx_path.stat().st_size / 1024
    print(f"[exported] {onnx_path.relative_to(CFG.paths.root)}  ({kb:.0f} KB)")

    # ---------- structural check ----------
    import onnx
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"[graph] {len(m.graph.node)} nodes, ops: {', '.join(ops)}")

    # ---------- numerical check on REAL data ----------
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])

    xp = Path(CFG.paths.processed) / "test_X.npy"
    if not xp.exists():
        sys.exit("no test data - run:  python scripts/prepare_data.py")
    X = np.load(xp)
    y = np.load(Path(CFG.paths.processed) / "test_y.npy")

    batch = 512
    max_diff = 0.0
    agree = total = 0
    torch_pred, onnx_pred = [], []

    for i in range(0, len(X), batch):
        chunk = X[i:i + batch].astype(np.float32) / 255.0
        t_in = torch.from_numpy(chunk).unsqueeze(1)

        with torch.no_grad():
            p_torch = model(t_in).numpy()
        p_onnx = sess.run(["prob"], {"eye": t_in.numpy()})[0]

        max_diff = max(max_diff, float(np.abs(p_torch - p_onnx).max()))
        a = (p_torch[:, 1] >= 0.5).astype(int)
        b = (p_onnx[:, 1] >= 0.5).astype(int)
        agree += int((a == b).sum())
        total += len(a)
        torch_pred.append(p_torch[:, 1])
        onnx_pred.append(p_onnx[:, 1])

    tp = np.concatenate(torch_pred)
    op = np.concatenate(onnx_pred)
    thr = CFG.drowsy.cnn_closed_thresh
    acc_t = float(((tp >= thr).astype(int) == y).mean())
    acc_o = float(((op >= thr).astype(int) == y).mean())

    print(f"\nverified on {total:,} real test images")
    print(f"  max probability difference : {max_diff:.3e}")
    print(f"  decisions agreeing         : {agree:,}/{total:,} "
          f"({100*agree/total:.4f}%)")
    print(f"  accuracy pytorch / onnx    : {acc_t*100:.2f}% / {acc_o*100:.2f}%")

    # float32 arithmetic can reorder slightly between runtimes, so exact
    # bit-equality is not expected. 1e-4 means every decision is identical.
    if max_diff > 1e-4 or agree != total:
        sys.exit("\nEXPORT MISMATCH - the phone would behave differently from "
                 "the desktop model. Do not ship this.")
    print("\nOK - the exported model is numerically identical in every decision.")

    meta = {
        "img_size": size,
        "input": "eye (batch,1,32,32) float32 in [0,1]",
        "output": "prob (batch,2), index 1 = P(closed)",
        "threshold": thr,
        "test_accuracy": acc_o,
    }
    import json
    (OUT / "eyenet.json").write_text(json.dumps(meta, indent=2))
    print(f"[wrote] {(OUT / 'eyenet.json').name}")


if __name__ == "__main__":
    main()
