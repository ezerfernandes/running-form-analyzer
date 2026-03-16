import numpy as np

from models.blazepose_model import BlazePoseModel


class TestConvertToMovenetFormat:
    def _make_landmark(self, x, y, z=0.0, visibility=1.0):
        """Create a mock landmark with x, y, z, visibility attributes."""

        class Landmark:
            pass

        lm = Landmark()
        lm.x = x
        lm.y = y
        lm.z = z
        lm.visibility = visibility
        return lm

    def test_output_shape(self):
        landmarks = [self._make_landmark(0.5, 0.5) for _ in range(33)]
        result = BlazePoseModel._convert_to_movenet_format(landmarks)
        assert result.shape == (1, 17, 3)

    def test_nose_mapping(self):
        landmarks = [self._make_landmark(0.0, 0.0) for _ in range(33)]
        landmarks[0] = self._make_landmark(0.3, 0.7, visibility=0.95)
        result = BlazePoseModel._convert_to_movenet_format(landmarks)
        # Movenet format: [y, x, confidence]
        assert abs(result[0, 0, 0] - 0.7) < 1e-6  # y
        assert abs(result[0, 0, 1] - 0.3) < 1e-6  # x
        assert abs(result[0, 0, 2] - 0.95) < 1e-6  # visibility

    def test_shoulder_mapping(self):
        landmarks = [self._make_landmark(0.0, 0.0) for _ in range(33)]
        landmarks[11] = self._make_landmark(0.4, 0.6, visibility=0.8)  # left shoulder -> movenet 5
        result = BlazePoseModel._convert_to_movenet_format(landmarks)
        assert abs(result[0, 5, 0] - 0.6) < 1e-6
        assert abs(result[0, 5, 1] - 0.4) < 1e-6
        assert abs(result[0, 5, 2] - 0.8) < 1e-6

    def test_ankle_mapping(self):
        landmarks = [self._make_landmark(0.0, 0.0) for _ in range(33)]
        landmarks[27] = self._make_landmark(0.1, 0.9, visibility=0.7)  # left ankle -> movenet 15
        result = BlazePoseModel._convert_to_movenet_format(landmarks)
        assert abs(result[0, 15, 0] - 0.9) < 1e-6
        assert abs(result[0, 15, 1] - 0.1) < 1e-6


class TestConvertBlazeposToKeypoints:
    def test_basic_conversion(self):
        # Movenet format: (1, 17, 3) with [y, x, conf]
        landmarks = np.zeros((1, 17, 3))
        landmarks[0, 0] = [0.5, 0.3, 0.9]
        coords, confs = BlazePoseModel.convert_blazepose_to_keypoints(landmarks)
        assert coords[0] == (0.3, 0.5)  # (x, y)
        assert confs[0] == 0.9

    def test_none_input(self):
        coords, confs = BlazePoseModel.convert_blazepose_to_keypoints(None)
        assert coords is None
        assert confs is None
