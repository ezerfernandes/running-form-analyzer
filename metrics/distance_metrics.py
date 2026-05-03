from typing import Dict
import numpy as np
from collections import deque
from core.config import HFOV_DEG, IMAGE_WIDTH_PX, Config
from feedback.assessment_calculator import AssessmentCalculator

HFOV_RAD = np.radians(HFOV_DEG)
# Calculate the focal length in pixels
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2) / np.tan(HFOV_RAD / 2)


# Torso-length-as-fraction-of-stature anthropometric ratios.
_TORSO_RATIO_BY_SEX = {"male": 0.30, "female": 0.29}


class DistanceMetrics:
    def __init__(self, config: Config):
        self.hip_positions = deque(maxlen=10)
        self.current_distance = 0.0
        self.runner_height_cm = config.runner_height
        self.sex = getattr(config, "sex", "male")
        self.torso_length_cm = self.calculate_torso_length(self.runner_height_cm)

    def calculate_torso_length(self, runner_height_cm):
        ratio = _TORSO_RATIO_BY_SEX.get(self.sex, _TORSO_RATIO_BY_SEX["male"])
        return runner_height_cm * ratio

    def calculate(
        self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, any]
    ):
        self.update_distance(valid_keypoints)
        self.calculate_vertical_oscillation(metrics)

    def update_distance(self, valid_keypoints: Dict[int, np.ndarray]):
        # Invalidate every frame so a stale depth from earlier never feeds
        # the cm/px scale when the current frame lacks torso keypoints.
        self.current_distance = 0.0
        if all(i in valid_keypoints for i in [5, 6, 11, 12]):
            shoulder_midpoint = (valid_keypoints[5] + valid_keypoints[6]) / 2
            hip_midpoint = (valid_keypoints[11] + valid_keypoints[12]) / 2
            torso_length_px = np.abs(hip_midpoint[1] - shoulder_midpoint[1])
            if torso_length_px < 1e-3:
                return

            self.current_distance = (
                self.torso_length_cm * FOCAL_LENGTH_PX
            ) / torso_length_px

            self.hip_positions.append(hip_midpoint[1])

    def calculate_vertical_oscillation(self, metrics: Dict[str, any]):
        kernel_size = 5
        # Need at least `kernel_size` samples for np.convolve(..., "valid") to
        # return a non-empty array, and a fresh depth estimate so the px→cm
        # scale isn't stale from a frame where the torso wasn't visible.
        if (
            len(self.hip_positions) < kernel_size
            or self.current_distance <= 0.0
        ):
            metrics["vertical_oscillation"] = 0.0
        else:
            moving_avg = (
                np.convolve(list(self.hip_positions), np.ones(kernel_size), "valid")
                / kernel_size
            )
            # Peak-to-peak hip vertical excursion (pixels) converted to cm via
            # the pinhole-camera scale at the runner's depth:
            #   cm_per_px = current_distance_cm / FOCAL_LENGTH_PX
            oscillation_px = float(np.max(moving_avg) - np.min(moving_avg))
            cm_per_px = self.current_distance / FOCAL_LENGTH_PX
            metrics["vertical_oscillation"] = oscillation_px * cm_per_px

        metrics["vertical_oscillation_assessment"] = (
            AssessmentCalculator.assess_vertical_oscillation(
                metrics["vertical_oscillation"]
            )
        )
