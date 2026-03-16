import numpy as np

from core.config import Config
from utils.angle_calculator import AngleCalculator


class TestCalculateAngle:
    def test_perpendicular_vectors(self):
        angle = AngleCalculator.calculate_angle(
            np.array([1, 0]), np.array([0, 1])
        )
        assert abs(angle - 90.0) < 1e-6

    def test_parallel_vectors(self):
        angle = AngleCalculator.calculate_angle(
            np.array([1, 0]), np.array([1, 0])
        )
        assert abs(angle) < 1e-6

    def test_opposite_vectors(self):
        angle = AngleCalculator.calculate_angle(
            np.array([1, 0]), np.array([-1, 0])
        )
        assert abs(angle - 180.0) < 1e-6

    def test_45_degree_angle(self):
        angle = AngleCalculator.calculate_angle(
            np.array([1, 0]), np.array([1, 1])
        )
        assert abs(angle - 45.0) < 1e-6


class TestCalculateAllAngles:
    def _make_keypoints(self, mapping):
        return {k: np.array(v, dtype=float) for k, v in mapping.items()}

    def test_head_angle_left_side(self):
        config = Config(side="left", model_type="blazepose", runner_height=182)
        kps = self._make_keypoints({
            1: [100, 90],   # left_eye
            3: [100, 100],  # left_ear
            5: [100, 150],  # left_shoulder
            11: [100, 300], # left_hip
        })
        angles = {}
        result = AngleCalculator.calculate_all_angles(kps, angles, config)
        assert "head_angle" in result

    def test_head_angle_right_side(self):
        config = Config(side="right", model_type="blazepose", runner_height=182)
        kps = self._make_keypoints({
            2: [100, 90],   # right_eye
            4: [100, 100],  # right_ear
            6: [100, 150],  # right_shoulder
            12: [100, 300], # right_hip
        })
        angles = {}
        result = AngleCalculator.calculate_all_angles(kps, angles, config)
        assert "head_angle" in result

    def test_missing_keypoints_skips_angle(self):
        config = Config(side="left", model_type="blazepose", runner_height=182)
        kps = self._make_keypoints({1: [100, 90]})  # only one keypoint
        angles = {}
        result = AngleCalculator.calculate_all_angles(kps, angles, config)
        assert "head_angle" not in result

    def test_knee_angles_computed(self):
        config = Config(side="left", model_type="blazepose", runner_height=182)
        kps = self._make_keypoints({
            11: [100, 200],  # left hip
            12: [200, 200],  # right hip
            13: [100, 350],  # left knee
            14: [200, 350],  # right knee
            15: [100, 500],  # left ankle
            16: [200, 500],  # right ankle
        })
        angles = {}
        result = AngleCalculator.calculate_all_angles(kps, angles, config)
        assert "left_knee_angle" in result
        assert "right_knee_angle" in result

    def test_trunk_angle_left(self):
        config = Config(side="left", model_type="blazepose", runner_height=182)
        # Shoulder directly above hip -> trunk angle ~0
        kps = self._make_keypoints({
            5: [200, 100],   # left shoulder
            11: [200, 300],  # left hip
        })
        angles = {}
        result = AngleCalculator.calculate_all_angles(kps, angles, config)
        assert "trunk_angle" in result
        assert abs(result["trunk_angle"]) < 5  # nearly vertical
