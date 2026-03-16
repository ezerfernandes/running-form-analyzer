import numpy as np
import torch

from models.lite_hrnet import LiteHRNetModel


class TestLiteHRNetModel:
    def test_init_no_args(self):
        model = LiteHRNetModel()
        assert model.model is not None

    def test_preprocess_shape(self):
        model = LiteHRNetModel()
        # Create a fake BGR image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = model._preprocess(image)
        assert tensor.shape == (1, 3, 256, 256)
        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0

    def test_postprocess_shape(self):
        model = LiteHRNetModel()
        # Create fake heatmaps (17 joints, 64x64 spatial)
        fake_output = torch.randn(1, 17, 64, 64)
        keypoints = model._postprocess(fake_output)
        assert keypoints.shape == (1, 17, 3)

    def test_postprocess_finds_peaks(self):
        model = LiteHRNetModel()
        fake_output = torch.zeros(1, 17, 64, 64)
        # Place a peak at position (10, 20) for joint 0
        fake_output[0, 0, 10, 20] = 5.0
        keypoints = model._postprocess(fake_output)
        assert keypoints[0, 0, 0] == 20  # x
        assert keypoints[0, 0, 1] == 10  # y
        assert keypoints[0, 0, 2] == 5.0  # confidence

    def test_predict_returns_array(self):
        model = LiteHRNetModel()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = model.predict(image)
        assert result.shape == (1, 17, 3)
