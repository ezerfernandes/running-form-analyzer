import pytest

from utils.filters import Filters, KalmanFilter, TemporalFilter


class TestKalmanFilter:
    def test_converges_to_measurement(self):
        kf = KalmanFilter(
            initial_state=0,
            initial_estimate_error=1,
            measurement_noise=0.1,
            process_noise=0.01,
        )
        for _ in range(50):
            result = kf.update(10.0)
        assert abs(result - 10.0) < 0.1

    def test_smooth_tracking(self):
        kf = KalmanFilter(
            initial_state=0,
            initial_estimate_error=1,
            measurement_noise=0.5,
            process_noise=0.01,
        )
        results = [kf.update(m) for m in [1, 2, 3, 4, 5]]
        # Should be monotonically increasing
        for i in range(1, len(results)):
            assert results[i] > results[i - 1]


class TestTemporalFilter:
    def test_returns_raw_before_window_fills(self):
        tf = TemporalFilter(window_size=5)
        assert tf.update(10.0) == 10.0
        assert tf.update(20.0) == 20.0

    def test_returns_mean_after_window_fills(self):
        tf = TemporalFilter(window_size=3)
        tf.update(1.0)
        tf.update(2.0)
        result = tf.update(3.0)
        assert abs(result - 2.0) < 1e-6

    def test_sliding_window(self):
        tf = TemporalFilter(window_size=3)
        for v in [1, 2, 3]:
            tf.update(v)
        result = tf.update(6.0)  # window: [2, 3, 6]
        assert abs(result - (2 + 3 + 6) / 3) < 1e-6


class TestFiltersFactory:
    def test_create_kalman(self):
        f = Filters.create_filter("kalman")
        assert isinstance(f, KalmanFilter)

    def test_create_temporal(self):
        f = Filters.create_filter("temporal", window_size=5)
        assert isinstance(f, TemporalFilter)

    def test_create_none(self):
        f = Filters.create_filter("none")
        assert f is None

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid filter type"):
            Filters.create_filter("invalid")
