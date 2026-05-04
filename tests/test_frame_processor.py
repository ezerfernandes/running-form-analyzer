from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.config import Config


@pytest.fixture
def config():
    return Config(side="right", model_type="blazepose", runner_height=182)


def _gray_frame(h=480, w=640):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_frame_processor_disables_audio_in_server_mode(config):
    with patch("core.frame_processor.BlazePoseModel") as mock_model_cls, patch(
        "metrics.metrics.AudioFeedbackProvider"
    ) as mock_audio_cls:
        mock_model_cls.return_value = MagicMock(predict=MagicMock(return_value=None))
        from core.frame_processor import FrameProcessor

        fp = FrameProcessor(config, enable_audio=False)
        assert fp.metrics_calculator.audio_provider is None
        # AudioFeedbackProvider must not be constructed when disabled —
        # otherwise pyttsx3 starts a background thread we don't want on the
        # server side.
        mock_audio_cls.assert_not_called()


def test_frame_processor_returns_metrics_and_angles(config):
    with patch("core.frame_processor.BlazePoseModel") as mock_model_cls, patch(
        "metrics.metrics.AudioFeedbackProvider"
    ):
        # Model returns None (no pose) so Metrics fills in the zero-default
        # dict — that exercises the full pipeline shape without depending on
        # the real BlazePose model file.
        mock_model_cls.return_value = MagicMock(predict=MagicMock(return_value=None))
        from core.frame_processor import FrameProcessor

        fp = FrameProcessor(config, enable_audio=False)
        metrics, angles = fp.process(_gray_frame(), current_time=0.0)

        assert isinstance(metrics, dict)
        assert isinstance(angles, dict)
        assert "recommendations" in metrics
        assert "elapsed_time" in metrics
        assert "steps_per_minute" in metrics
        assert metrics["elapsed_time"] == 0.0


def test_frame_processor_predict_and_analyze_match_process(config):
    with patch("core.frame_processor.BlazePoseModel") as mock_model_cls, patch(
        "metrics.metrics.AudioFeedbackProvider"
    ):
        mock_model_cls.return_value = MagicMock(predict=MagicMock(return_value=None))
        from core.frame_processor import FrameProcessor

        fp = FrameProcessor(config, enable_audio=False)
        frame = _gray_frame()
        out = fp.predict(frame)
        metrics, angles = fp.analyze(out, frame.shape, current_time=0.5)
        assert "recommendations" in metrics
        assert metrics["elapsed_time"] == pytest.approx(0.0)  # start_time set on first call


def test_frame_processor_invalid_model_type():
    bad = Config(side="right", model_type="bogus", runner_height=182)
    with patch("metrics.metrics.AudioFeedbackProvider"):
        from core.frame_processor import FrameProcessor

        with pytest.raises(ValueError, match="Invalid model type"):
            FrameProcessor(bad, enable_audio=False)
