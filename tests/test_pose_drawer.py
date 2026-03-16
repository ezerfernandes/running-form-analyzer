import numpy as np

from visualization.pose_drawer import draw_connections, draw_keypoints


def _make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_keypoints(conf=0.9):
    """Create (1, 17, 3) keypoints in normalized [y, x, conf] format."""
    kps = np.zeros((1, 17, 3))
    for i in range(17):
        kps[0, i] = [0.3 + i * 0.02, 0.2 + i * 0.03, conf]
    return kps


class TestDrawKeypoints:
    def test_draws_circles_above_threshold(self):
        frame = _make_frame()
        kps = _make_keypoints(conf=0.9)
        draw_keypoints(frame, kps, confidence_threshold=0.5)
        # Frame should have non-zero pixels (green circles drawn)
        assert frame.sum() > 0

    def test_no_draw_below_threshold(self):
        frame = _make_frame()
        kps = _make_keypoints(conf=0.1)
        draw_keypoints(frame, kps, confidence_threshold=0.5)
        assert frame.sum() == 0

    def test_does_not_crash_with_zeros(self):
        frame = _make_frame()
        kps = np.zeros((1, 17, 3))
        draw_keypoints(frame, kps, confidence_threshold=0.5)

    def test_frame_shape_preserved(self):
        frame = _make_frame(100, 200)
        kps = _make_keypoints()
        draw_keypoints(frame, kps, confidence_threshold=0.3)
        assert frame.shape == (100, 200, 3)


class TestDrawConnections:
    def test_draws_lines_above_threshold(self):
        frame = _make_frame()
        kps = _make_keypoints(conf=0.9)
        result = draw_connections(frame, kps, confidence_threshold=0.5)
        assert result.sum() > 0
        assert result is frame  # modifies in place and returns

    def test_no_draw_below_threshold(self):
        frame = _make_frame()
        kps = _make_keypoints(conf=0.1)
        result = draw_connections(frame, kps, confidence_threshold=0.5)
        assert result.sum() == 0

    def test_partial_confidence(self):
        frame = _make_frame()
        kps = _make_keypoints(conf=0.0)
        # Set only nose (0) and left eye (1) above threshold — edge (0,1) exists
        kps[0, 0, 2] = 0.9
        kps[0, 1, 2] = 0.9
        result = draw_connections(frame, kps, confidence_threshold=0.5)
        assert result.sum() > 0
