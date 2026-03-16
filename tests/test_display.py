import numpy as np
import pytest

from visualization.display import (
    display_angles,
    display_metrics,
    display_mode,
    display_recommendations,
)


def _make_frame():
    return np.full((600, 800, 3), 255, dtype=np.uint8)


def _full_metrics():
    return {
        "head_angle": 90.0,
        "head_angle_assessment": "Good",
        "trunk_angle": 10.0,
        "trunk_angle_assessment": "Good",
        "left_elbow_angle": 75.0,
        "right_elbow_angle": 80.0,
        "left_elbow_angle_assessment": "Good",
        "right_elbow_angle_assessment": "Good",
        "left_hip_ankle_angle_at_strike": 10.0,
        "right_hip_ankle_angle_at_strike": 12.0,
        "left_hip_ankle_angle_assessment": "Good",
        "right_hip_ankle_angle_assessment": "Good",
        "left_knee_angle_at_strike": 140.0,
        "right_knee_angle_at_strike": 138.0,
        "left_knee_assessment": "Good",
        "right_knee_assessment": "Good",
        "left_shank_angle_at_strike": 5.0,
        "right_shank_angle_at_strike": 6.0,
        "left_shank_angle_assessment": "Good",
        "right_shank_angle_assessment": "Good",
        "max_left_arm_backward_swing": 35.0,
        "max_left_arm_forward_swing": 55.0,
        "max_right_arm_backward_swing": 37.0,
        "max_right_arm_forward_swing": 53.0,
        "left_arm_backward_swing_assessment": "Good",
        "left_arm_forward_swing_assessment": "Good",
        "right_arm_backward_swing_assessment": "Good",
        "right_arm_forward_swing_assessment": "Good",
        "max_left_hip_backward_swing": 35.0,
        "max_left_hip_forward_swing": 35.0,
        "max_right_hip_backward_swing": 35.0,
        "max_right_hip_forward_swing": 35.0,
        "left_hip_backward_swing_assessment": "Good",
        "left_hip_forward_swing_assessment": "Good",
        "right_hip_backward_swing_assessment": "Good",
        "right_hip_forward_swing_assessment": "Good",
        "vertical_oscillation": 5.0,
        "vertical_oscillation_assessment": "Good",
        "left_foot_strike": False,
        "right_foot_strike": False,
        "steps_per_minute": 170.0,
        "elapsed_time": 60.0,
        "recommendations": [],
    }


_WHITE_SUM = 255 * 600 * 800 * 3


class TestDisplayAngles:
    def test_renders_text(self):
        frame = _make_frame()
        display_angles(frame, {"head_angle": 90.5, "trunk_angle": 10.2})
        # Black text on white frame reduces sum
        assert frame.sum() < _WHITE_SUM

    def test_empty_angles_unchanged(self):
        frame = _make_frame()
        display_angles(frame, {})
        assert frame.sum() == _WHITE_SUM


class TestDisplayMetrics:
    def test_left_side(self):
        frame = _make_frame()
        display_metrics(frame, _full_metrics(), "left")
        assert frame.sum() < _WHITE_SUM

    def test_right_side(self):
        frame = _make_frame()
        display_metrics(frame, _full_metrics(), "right")
        assert frame.sum() < _WHITE_SUM


class TestDisplayRecommendations:
    def test_with_recommendations(self):
        frame = _make_frame()
        metrics = {"recommendations": ["Fix posture", "Bend knees"]}
        display_recommendations(frame, metrics)
        assert frame.sum() < _WHITE_SUM

    def test_no_recommendations_shows_great_form(self):
        frame = _make_frame()
        metrics = {"recommendations": []}
        display_recommendations(frame, metrics)
        # "Great form! Keep it up!" drawn in green — some pixels change
        assert frame.sum() != _WHITE_SUM


class TestDisplayMode:
    def test_angles_mode(self):
        frame = _make_frame()
        angles = {"head_angle": 90.0}
        result = display_mode(frame, {}, angles, "angles")
        assert result is frame
        assert frame.sum() < _WHITE_SUM

    def test_metrics_mode(self):
        frame = _make_frame()
        result = display_mode(frame, _full_metrics(), {}, "metrics", "left")
        assert result is frame
        assert frame.sum() < _WHITE_SUM

    def test_recommendations_mode(self):
        frame = _make_frame()
        metrics = {"recommendations": ["Test"]}
        result = display_mode(frame, metrics, {}, "recommendations")
        assert result is frame

    def test_invalid_mode_raises(self):
        frame = _make_frame()
        with pytest.raises(ValueError, match="Invalid display mode"):
            display_mode(frame, {}, {}, "invalid")
