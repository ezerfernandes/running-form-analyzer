import numpy as np

from core.config import Config
from metrics.distance_metrics import DistanceMetrics


class TestDistanceMetrics:
    def _make_metrics(self):
        config = Config(side="left", model_type="blazepose", runner_height=180)
        return DistanceMetrics(config)

    def test_torso_length_calculation(self):
        dm = self._make_metrics()
        assert abs(dm.torso_length_cm - 54.0) < 1e-6  # 180 * 0.3

    def test_vertical_oscillation_zero_with_few_points(self):
        dm = self._make_metrics()
        metrics = {}
        dm.calculate_vertical_oscillation(metrics)
        assert metrics["vertical_oscillation"] == 0.0
        assert metrics["vertical_oscillation_assessment"] == "Good"

    def test_update_distance_with_valid_keypoints(self):
        dm = self._make_metrics()
        kps = {
            5: np.array([100.0, 100.0]),   # left shoulder
            6: np.array([200.0, 100.0]),   # right shoulder
            11: np.array([100.0, 300.0]),  # left hip
            12: np.array([200.0, 300.0]),  # right hip
        }
        dm.update_distance(kps)
        assert dm.current_distance > 0

    def test_update_distance_missing_keypoints(self):
        dm = self._make_metrics()
        dm.update_distance({5: np.array([100.0, 100.0])})
        assert dm.current_distance == 0.0

    def test_update_distance_zero_torso_length(self):
        dm = self._make_metrics()
        kps = {
            5: np.array([100.0, 200.0]),
            6: np.array([200.0, 200.0]),
            11: np.array([100.0, 200.0]),  # same y as shoulders
            12: np.array([200.0, 200.0]),
        }
        dm.update_distance(kps)
        assert dm.current_distance == 0.0  # no update, stays at default

    def test_calculate_runs_without_error(self):
        dm = self._make_metrics()
        kps = {
            5: np.array([100.0, 100.0]),   # left shoulder
            6: np.array([200.0, 100.0]),   # right shoulder
            11: np.array([100.0, 300.0]),  # left hip
            12: np.array([200.0, 300.0]),  # right hip
        }
        metrics = {}
        dm.calculate(kps, metrics)
        assert "vertical_oscillation" in metrics
