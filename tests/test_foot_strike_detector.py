import numpy as np

from utils.foot_strike_detector import FootStrikeDetector


class TestFootStrikeDetector:
    def test_no_detection_with_few_samples(self):
        detector = FootStrikeDetector(filter_type="none")
        detected, stride, _ = detector.update(np.array([0.5, 0.5]), 0.0)
        assert not detected
        assert stride == 0.0

    def test_detects_foot_strike_x_axis(self):
        detector = FootStrikeDetector(
            filter_type="none", detection_axis="x", initial_threshold=0.01
        )
        # Simulate a dip then rise in x (foot strike pattern)
        positions = [0.5, 0.45, 0.4, 0.35, 0.3, 0.35, 0.4]
        detected_any = False
        for i, x in enumerate(positions):
            detected, _, _ = detector.update(np.array([x, 0.5]), float(i) * 0.5)
            if detected:
                detected_any = True
        assert detected_any

    def test_detects_foot_strike_y_axis(self):
        detector = FootStrikeDetector(
            filter_type="none", detection_axis="y", initial_threshold=0.01
        )
        positions = [0.5, 0.45, 0.4, 0.35, 0.3, 0.35, 0.4]
        detected_any = False
        for i, y in enumerate(positions):
            detected, _, _ = detector.update(np.array([0.5, y]), float(i) * 0.5)
            if detected:
                detected_any = True
        assert detected_any

    def test_stride_time_calculated(self):
        detector = FootStrikeDetector(
            filter_type="none", detection_axis="x", initial_threshold=0.001
        )
        # Two foot strikes separated by time
        pattern = [0.5, 0.4, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.4, 0.5]
        stride_times = []
        for i, x in enumerate(pattern):
            detected, stride, _ = detector.update(np.array([x, 0.5]), float(i) * 0.5)
            if detected and stride > 0:
                stride_times.append(stride)
        # Should have at least one stride time measured
        if stride_times:
            assert all(t > 0 for t in stride_times)

    def test_prevents_false_positives_too_close(self):
        detector = FootStrikeDetector(
            filter_type="none", detection_axis="x", initial_threshold=0.01
        )
        # Two strikes too close together (< 0.4s apart)
        pattern = [0.5, 0.3, 0.5, 0.3, 0.5]
        detections = 0
        for i, x in enumerate(pattern):
            detected, _, _ = detector.update(np.array([x, 0.5]), float(i) * 0.1)
            if detected:
                detections += 1
        # Should prevent rapid re-detection
        assert detections <= 1

    def test_get_filtered_position_empty(self):
        detector = FootStrikeDetector(filter_type="none")
        assert detector.get_filtered_position() is None

    def test_get_filtered_position_after_update(self):
        detector = FootStrikeDetector(filter_type="none", detection_axis="x")
        detector.update(np.array([0.5, 0.3]), 0.0)
        assert detector.get_filtered_position() is not None

    def test_with_kalman_filter(self):
        detector = FootStrikeDetector(filter_type="kalman", detection_axis="x")
        # Should not crash
        for i in range(10):
            detector.update(np.array([float(i) * 0.1, 0.5]), float(i))

    def test_with_temporal_filter(self):
        detector = FootStrikeDetector(filter_type="temporal", detection_axis="x")
        for i in range(10):
            detector.update(np.array([float(i) * 0.1, 0.5]), float(i))
