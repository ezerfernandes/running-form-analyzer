from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core.config import Config
from metrics.metrics import Metrics


@pytest.fixture
def config():
    return Config(side="left", model_type="blazepose", runner_height=182)


@pytest.fixture
def metrics_instance(config):
    with patch("metrics.metrics.AudioFeedbackProvider") as mock_audio:
        mock_audio.return_value = MagicMock()
        m = Metrics(config)
    return m


def _full_keypoints():
    """Create a full set of 17 keypoints simulating a standing pose."""
    return {
        0: np.array([300.0, 100.0]),   # nose
        1: np.array([290.0, 90.0]),    # left_eye
        2: np.array([310.0, 90.0]),    # right_eye
        3: np.array([280.0, 95.0]),    # left_ear
        4: np.array([320.0, 95.0]),    # right_ear
        5: np.array([270.0, 200.0]),   # left_shoulder
        6: np.array([330.0, 200.0]),   # right_shoulder
        7: np.array([260.0, 300.0]),   # left_elbow
        8: np.array([340.0, 300.0]),   # right_elbow
        9: np.array([255.0, 400.0]),   # left_wrist
        10: np.array([345.0, 400.0]),  # right_wrist
        11: np.array([280.0, 400.0]),  # left_hip
        12: np.array([320.0, 400.0]),  # right_hip
        13: np.array([275.0, 550.0]),  # left_knee
        14: np.array([325.0, 550.0]),  # right_knee
        15: np.array([270.0, 700.0]),  # left_ankle
        16: np.array([330.0, 700.0]),  # right_ankle
    }


class TestMetricsInit:
    def test_key_metrics_has_expected_keys(self, metrics_instance):
        km = metrics_instance.get_key_metrics()
        assert "head_angle" in km
        assert "trunk_angle" in km
        assert "vertical_oscillation" in km
        assert "steps_per_minute" in km
        assert "recommendations" in km
        assert "left_foot_strike" in km
        assert "right_foot_strike" in km

    def test_key_metrics_defaults(self, metrics_instance):
        km = metrics_instance.get_key_metrics()
        assert km["head_angle"] == 0.0
        assert km["recommendations"] == []
        assert km["left_foot_strike"] is False

    def test_start_time_initially_none(self, metrics_instance):
        assert metrics_instance.start_time is None


class TestCalculateMetrics:
    def test_returns_metrics_and_angles(self, metrics_instance):
        result = metrics_instance.calculate_metrics({}, timestamp=1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        metrics, angles = result
        assert isinstance(metrics, dict)
        assert isinstance(angles, dict)

    def test_sets_start_time_on_first_call(self, metrics_instance):
        metrics_instance.calculate_metrics({}, timestamp=5.0)
        assert metrics_instance.start_time == 5.0

    def test_start_time_not_overwritten(self, metrics_instance):
        metrics_instance.calculate_metrics({}, timestamp=5.0)
        metrics_instance.calculate_metrics({}, timestamp=10.0)
        assert metrics_instance.start_time == 5.0

    def test_elapsed_time_calculated(self, metrics_instance):
        metrics_instance.calculate_metrics({}, timestamp=5.0)
        metrics, _ = metrics_instance.calculate_metrics({}, timestamp=8.0)
        assert abs(metrics["elapsed_time"] - 3.0) < 1e-6

    def test_empty_keypoints_no_angles(self, metrics_instance):
        metrics, angles = metrics_instance.calculate_metrics({}, timestamp=0.0)
        # No keypoints -> no angles computed on first call
        assert len(angles) == 0
        assert metrics["vertical_oscillation"] == 0.0

    def test_with_full_keypoints(self, metrics_instance):
        kps = _full_keypoints()
        metrics, angles = metrics_instance.calculate_metrics(kps, timestamp=1.0)
        # With full keypoints, angles should be computed
        assert "head_angle" in angles or "left_knee_angle" in angles

    def test_head_angle_computed_left_side(self, metrics_instance):
        kps = _full_keypoints()
        metrics, angles = metrics_instance.calculate_metrics(kps, timestamp=1.0)
        # Left side config uses keypoints 1, 3, 5, 11 for head angle
        assert "head_angle" in angles
        assert metrics["head_angle"] != 0.0

    def test_head_angle_assessment_populated(self, metrics_instance):
        kps = _full_keypoints()
        metrics, _ = metrics_instance.calculate_metrics(kps, timestamp=1.0)
        assert metrics["head_angle_assessment"] in ["Good", "Need Improvement", "Bad"]

    def test_trunk_angle_computed(self, metrics_instance):
        kps = _full_keypoints()
        metrics, angles = metrics_instance.calculate_metrics(kps, timestamp=1.0)
        assert "trunk_angle" in angles

    def test_knee_angles_computed(self, metrics_instance):
        kps = _full_keypoints()
        metrics, angles = metrics_instance.calculate_metrics(kps, timestamp=1.0)
        assert "left_knee_angle" in angles
        assert "right_knee_angle" in angles

    def test_elbow_angle_computed_left(self, metrics_instance):
        kps = _full_keypoints()
        metrics, angles = metrics_instance.calculate_metrics(kps, timestamp=1.0)
        assert "left_elbow_angle" in angles
        assert metrics["left_elbow_angle_assessment"] in [
            "Good", "Need Improvement", "Bad"
        ]

    def test_vertical_oscillation_assessed(self, metrics_instance):
        kps = _full_keypoints()
        metrics, _ = metrics_instance.calculate_metrics(kps, timestamp=1.0)
        assert metrics["vertical_oscillation_assessment"] in [
            "Good", "Need Improvement", "Bad"
        ]

    def test_angles_persist_across_calls(self, metrics_instance):
        """AngleMetrics retains angles to prevent flickering when keypoints are lost."""
        kps = _full_keypoints()
        metrics1, _ = metrics_instance.calculate_metrics(kps, timestamp=1.0)
        head_angle = metrics1["head_angle"]
        metrics2, _ = metrics_instance.calculate_metrics({}, timestamp=2.0)
        # Angles persist from previous frame (by design)
        assert metrics2["head_angle"] == head_angle

    def test_multiple_frames_accumulate_distance(self, metrics_instance):
        kps = _full_keypoints()
        for t in range(5):
            metrics_instance.calculate_metrics(kps, timestamp=float(t))
        # After multiple frames, hip_positions in distance_metrics should be populated
        assert len(metrics_instance.distance_metrics.hip_positions) > 0


class TestCalculateMetricsRightSide:
    @pytest.fixture
    def right_metrics(self):
        config = Config(side="right", model_type="blazepose", runner_height=175)
        with patch("metrics.metrics.AudioFeedbackProvider") as mock_audio:
            mock_audio.return_value = MagicMock()
            m = Metrics(config)
        return m

    def test_right_side_elbow(self, right_metrics):
        kps = _full_keypoints()
        metrics, angles = right_metrics.calculate_metrics(kps, timestamp=1.0)
        assert "right_elbow_angle" in angles
        assert metrics["right_elbow_angle_assessment"] in [
            "Good", "Need Improvement", "Bad"
        ]

    def test_right_side_head_angle(self, right_metrics):
        kps = _full_keypoints()
        metrics, angles = right_metrics.calculate_metrics(kps, timestamp=1.0)
        assert "head_angle" in angles


class TestGetKeyMetrics:
    def test_returns_dict(self, metrics_instance):
        assert isinstance(metrics_instance.get_key_metrics(), dict)

    def test_contains_all_expected_metric_groups(self, metrics_instance):
        km = metrics_instance.get_key_metrics()
        # Angles
        assert "head_angle" in km
        assert "trunk_angle" in km
        assert "left_elbow_angle" in km
        assert "right_elbow_angle" in km
        # Swing metrics
        assert "max_left_arm_backward_swing" in km
        assert "max_right_hip_forward_swing" in km
        # Strike metrics
        assert "left_hip_ankle_angle_at_strike" in km
        assert "left_knee_angle_at_strike" in km
        assert "left_shank_angle_at_strike" in km
        # Assessment strings
        assert "head_angle_assessment" in km
        assert "vertical_oscillation_assessment" in km
