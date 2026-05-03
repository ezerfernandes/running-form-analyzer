# Known parameters
import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class Config:
    side: str
    model_type: str
    runner_height: float
    sex: str = "male"  # Drives anthropometric ratios (e.g. torso length).

    @classmethod
    def from_args(cls, args):
        # getattr lets older callers (and tests) build a Config from a
        # Namespace that doesn't carry --sex; the dataclass default applies.
        return cls(
            side=args.side,
            model_type=args.model_type,
            runner_height=args.runner_height,
            sex=getattr(args, "sex", "male"),
        )

    def to_dict(self) -> Dict[str, str | float]:
        return {
            "side": self.side,
            "model_type": self.model_type,
            "runner_height": self.runner_height,
            "sex": self.sex,
        }


# Camera parameters for your XPS 15
HFOV_DEG = 74  # Horizontal field of view in degrees
IMAGE_WIDTH_PX = 1280  # Image width in pixels

# Anchor model paths to the repo root so `python main.py` works from any CWD
# (this file lives in <repo>/core/, so the repo root is one level up).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
THUNDER_PATH = os.path.join(_REPO_ROOT, "models", "thunder-float32.tflite")
LIGHTNING_PATH = os.path.join(_REPO_ROOT, "models", "lightning-float32.tflite")

# edges for the pose graph
EDGES = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}
