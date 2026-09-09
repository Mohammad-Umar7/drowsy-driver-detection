"""
The eye-state CNN.

Input : one 32x32 grayscale eye crop   -> tensor of shape (batch, 1, 32, 32)
Output: two raw scores ("logits")      -> tensor of shape (batch, 2)
        index 0 = "open", index 1 = "closed"

Read docs/01-foundations.md section 5 first if "CNN" is new to you.
"""
import torch
import torch.nn as nn

CLASS_NAMES = ["open", "closed"]


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """
    One convolution stage: Conv -> BatchNorm -> ReLU.

    Conv2d(c_in, c_out, 3, padding=1)
        Slides c_out different 3x3 kernels over the input. Each kernel produces
        one "feature map" -- a new grid showing where that kernel fired.
        padding=1 adds a 1-pixel border of zeros so the output keeps the same
        width/height as the input (without it, every conv would shrink the image
        by 2 pixels and we would run out of image after a few layers).

        bias=False because the BatchNorm immediately after has its own shift
        parameter, so a bias here would be redundant (and wasted parameters).

    BatchNorm2d
        Rescales the numbers flowing out of the conv so they have roughly mean 0
        and variance 1 across the batch. Without it, values drift to huge or tiny
        magnitudes as they pass through layers and training becomes slow or
        diverges. This single layer is why we can use a high learning rate.

    ReLU = max(0, x)
        The non-linearity. Throws away negative values.
        WHY IT IS ESSENTIAL: stacking two linear layers is mathematically the
        same as one linear layer (A x B x = C x). Without something non-linear
        in between, a 10-layer network could still only draw straight lines.
        ReLU is what lets depth actually buy you anything.

        inplace=True overwrites the input tensor instead of allocating a new
        one -- a free memory saving.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class EyeNet(nn.Module):
    """
    A small CNN built from scratch. ~200k parameters, runs in well under 1 ms
    per eye on a GPU, and comfortably real-time even on a CPU.

    The shape of the data as it flows through (for one image):

        input          1 x 32 x 32     1024 numbers, the raw eye picture
        block1        32 x 32 x 32     32 feature maps: edges, gradients
        maxpool       32 x 16 x 16     half the size, keep the strongest signal
        block2        64 x 16 x 16     64 maps: corners, curves
        maxpool       64 x  8 x  8
        block3       128 x  8 x  8     128 maps: eyelid-ish / iris-ish shapes
        maxpool      128 x  4 x  4
        avgpool      128 x  1 x  1     average each map -> one number per map
        flatten      128               a 128-number summary of the eye
        dropout      128               randomly zero 30% (training only)
        linear         2               two scores: open, closed

    Notice the pattern: the picture gets SMALLER (32 -> 16 -> 8 -> 4) while
    getting DEEPER (1 -> 32 -> 64 -> 128 channels). That is the standard CNN
    shape. You trade spatial detail for semantic meaning as you go up.
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),        # 32 -> 16

            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),        # 16 -> 8

            conv_block(64, 128),
            nn.MaxPool2d(2),        # 8 -> 4
        )

        # Global Average Pooling: collapse each 4x4 feature map to its mean.
        # The old-school alternative is Flatten -> Linear(128*4*4, ...) which
        # needs 2048 x N weights and overfits badly on a small dataset.
        # GAP needs zero parameters and generalises much better.
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Sensible starting values for the knobs.

        Kaiming ("He") initialisation scales the initial random weights by
        sqrt(2 / fan_in), which is the value that keeps the signal variance
        stable as it passes through ReLU layers. Start with weights too large
        and activations explode; too small and they vanish and nothing learns.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns LOGITS (raw unbounded scores), not probabilities.

        Why: PyTorch's CrossEntropyLoss applies softmax internally in a
        numerically stable way. Applying softmax here too would be wrong
        (double softmax) and would train badly. Convert to probabilities only
        at inference time -- see predict_proba below.
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper: logits -> probabilities that sum to 1."""
        return torch.softmax(self.forward(x), dim=1)


def build_mobilenet(n_classes: int = 2) -> nn.Module:
    """
    Alternative: TRANSFER LEARNING from MobileNetV3-Small.

    Transfer learning = start from a network already trained on ImageNet
    (1.2 million photos, 1000 categories) and re-purpose it. Its early layers
    already know edges and textures, so it can reach high accuracy from far
    fewer eye images than training from scratch.

    Two adjustments are needed:
      1. It expects 3-channel RGB, we have 1-channel grayscale -> we duplicate
         the grayscale channel 3 times.
      2. Its final layer outputs 1000 ImageNet classes -> we replace it with a
         2-output layer.

    Train both this and EyeNet and compare. On a large dataset like MRL, the
    from-scratch EyeNet usually wins on speed at similar accuracy; on a small
    self-collected dataset, transfer learning usually wins on accuracy.
    That comparison is a genuinely good result to put in your report.
    """
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, n_classes)

    class GrayWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)     # gray -> fake RGB
            return self.net(x)

    return GrayWrapper(m)


def build_model(arch: str = "eyenet", n_classes: int = 2) -> nn.Module:
    arch = arch.lower()
    if arch == "eyenet":
        return EyeNet(n_classes)
    if arch == "mobilenet":
        return build_mobilenet(n_classes)
    raise ValueError(f"unknown arch: {arch!r} (use 'eyenet' or 'mobilenet')")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
