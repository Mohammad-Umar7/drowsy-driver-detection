"""
Central configuration.

Every magic number in this project lives HERE, not scattered through the code.
Why: when you tune the system for a new person / camera / car, you edit one file.
Thresholds you should expect to tune are marked  # TUNE
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    root: Path = ROOT
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    checkpoints: Path = ROOT / "checkpoints"
    reports: Path = ROOT / "reports"
    best_model: Path = ROOT / "checkpoints" / "eyenet_best.pt"
    calibration: Path = ROOT / "checkpoints" / "calibration.json"


@dataclass
class DataCfg:
    img_size: int = 32          # eye crops are resized to 32x32 grayscale
    grayscale: bool = True
    crop_margin: float = 1.35   # expand eye bbox by this factor before cropping
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42


@dataclass
class TrainCfg:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    # 0 = load data in the main process. On Windows each worker is a whole new
    # process that re-imports everything, which is slow and crash-prone. Our
    # dataset is already in RAM (~60 MB), so there is nothing to wait on.
    num_workers: int = 0
    amp: bool = True            # mixed precision -> ~2x faster on an RTX 4070
    early_stop_patience: int = 7
    arch: str = "eyenet"        # "eyenet" (from scratch) | "mobilenet" (transfer)


@dataclass
class DrowsyCfg:
    # ---- eye closure ----
    ear_thresh: float = 0.21            # TUNE  fallback used if not calibrated
    ear_calib_ratio: float = 0.75       # calibrated thresh = 0.75 * your open EAR
    cnn_closed_thresh: float = 0.50     # TUNE  P(closed) above this = closed
    fusion_cnn_weight: float = 0.65     # how much we trust the CNN vs EAR (0..1)

    # ---- eye visibility ----
    # When the head turns, the far eye is foreshortened into a sliver. Both EAR
    # and the CNN then read nonsense from it -- and because we take the MAX of
    # the two eyes (safety bias), that nonsense wins and we report a false
    # closure. An eye narrower than this fraction of the wider one is dropped.
    eye_vis_min_ratio: float = 0.62     # TUNE
    # Absolute garbage floor, in pixels of corner-to-corner eye width.
    # Measured on a real head turn: the far eye collapsed to 4.6 px and
    # returned EAR 1.05 -- impossible for a real eye -- while the CNN scored it
    # 0.68 "closed". Anything this narrow carries no information at all.
    # Set well below a usable eye (~23 px) so it rejects only true nonsense.
    eye_min_width_px: float = 15.0      # TUNE

    # ---- blink / microsleep ----
    # How long the eyes must stay shut before we call it a microsleep.
    # Research uses ~0.5-1.0 s, but that feels twitchy in a demo and punishes
    # a slow deliberate blink. 2.0 s is unambiguous: nobody blinks for 2 s.
    microsleep_sec: float = 2.0         # TUNE
    blink_max_sec: float = 0.45         # closures shorter than this are normal blinks

    # ---- PERCLOS (the automotive-industry metric) ----
    perclos_window_sec: float = 30.0    # rolling window length
    perclos_warn: float = 0.15          # TUNE  15% of time closed -> DROWSY
    perclos_critical: float = 0.30      # TUNE  30% of time closed -> CRITICAL
    # PERCLOS is a PERCENTAGE, so it is meaningless until the window holds
    # enough data. With only 3 s observed, a single 1 s blink reads as 33% and
    # would fire CRITICAL the moment the app starts. Below this much observed
    # time PERCLOS is displayed but never triggers. Microsleep is unaffected --
    # it is an absolute duration, so it stays instant from the first second.
    perclos_min_obs_sec: float = 12.0

    # ---- yawning ----
    mar_thresh: float = 0.60            # TUNE
    yawn_min_sec: float = 1.5           # mouth must stay open this long to count
    yawn_window_sec: float = 60.0
    yawn_rate_warn: int = 3             # >=3 yawns per minute = fatigue

    # ---- head pose ----
    pitch_nod_deg: float = -18.0        # TUNE  head tipped down this much
    nod_min_sec: float = 2.5            # glancing at the dashboard is not nodding off
    yaw_distract_deg: float = 35.0      # looking away from the road
    distract_min_sec: float = 2.5       # a shoulder check is not distraction

    # ---- alarm / robustness ----
    alarm_cooldown_sec: float = 4.0
    # Looking away is a nudge, not an emergency, so it gets a quieter sound and
    # a much longer gap between repeats. Nagging every 4 s while someone checks
    # a mirror is how the whole system ends up switched off.
    distract_alarm_cooldown_sec: float = 9.0
    face_lost_grace_sec: float = 2.0    # ignore brief tracking dropouts


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    drowsy: DrowsyCfg = field(default_factory=DrowsyCfg)

    def save(self, path):
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        Path(path).write_text(json.dumps(d, indent=2))


CFG = Config()
