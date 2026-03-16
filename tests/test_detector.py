import numpy as np

from core.detector import extract_keypoints, get_valid_keypoints


class TestExtractKeypoints:
    def test_basic_extraction(self):
        # Shape: (1, 17, 3) — normalized [y, x, confidence]
        results = np.zeros((1, 17, 3))
        results[0, 0] = [0.5, 0.25, 0.9]  # nose: y=0.5, x=0.25, conf=0.9
        results[0, 5] = [0.4, 0.3, 0.8]  # left shoulder

        coords, confs = extract_keypoints(results, image_height=480, image_width=640)

        assert len(coords) == 17
        assert len(confs) == 17
        # x = col * width, y = row * height
        np.testing.assert_allclose(coords[0], [0.25 * 640, 0.5 * 480])
        assert confs[0] == 0.9

    def test_all_zeros(self):
        results = np.zeros((1, 17, 3))
        coords, confs = extract_keypoints(results, 480, 640)
        assert all(c == 0.0 for c in confs)


class TestGetValidKeypoints:
    def test_filters_by_confidence(self):
        coords = [np.array([100, 200]), np.array([150, 250]), np.array([50, 60])]
        confs = [0.9, 0.2, 0.5]
        valid = get_valid_keypoints(coords, confs, confidence_threshold=0.3)
        assert 0 in valid
        assert 1 not in valid
        assert 2 in valid

    def test_none_above_threshold(self):
        coords = [np.array([10, 20])]
        confs = [0.1]
        valid = get_valid_keypoints(coords, confs, confidence_threshold=0.5)
        assert len(valid) == 0

    def test_all_above_threshold(self):
        coords = [np.array([1, 2]), np.array([3, 4])]
        confs = [0.9, 0.8]
        valid = get_valid_keypoints(coords, confs, confidence_threshold=0.1)
        assert len(valid) == 2
