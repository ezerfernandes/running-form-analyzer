from typing import Any, Dict, Tuple

import numpy as np

from core.config import THUNDER_PATH, Config
from core.detector import extract_keypoints, get_valid_keypoints
from metrics.metrics import Metrics
from models.blazepose_model import BlazePoseModel
from models.movenet import MoveNetModel


class FrameProcessor:
    """Pure pose-and-metrics pipeline. No camera I/O, no display, no recording.

    Used by both local mode (Analyzer wraps this with cv2.VideoCapture +
    cv2.imshow) and server mode (the WebSocket handler feeds it phone-supplied
    frames). enable_audio=False suppresses the laptop's pyttsx3 speaker so
    server mode delivers cues via the phone browser instead.
    """

    def __init__(self, config: Config, enable_audio: bool = True):
        self.config = config
        if config.model_type == "blazepose":
            self.model = BlazePoseModel()
        elif config.model_type == "movenet":
            self.model = MoveNetModel(THUNDER_PATH)
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")
        self.metrics_calculator = Metrics(config, enable_audio=enable_audio)

    def predict(self, frame: np.ndarray):
        return self.model.predict(frame)

    def analyze(
        self,
        model_output,
        frame_shape: Tuple[int, ...],
        current_time: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if model_output is not None:
            kp_coords, kp_confs = extract_keypoints(
                model_output, frame_shape[0], frame_shape[1]
            )
            valid = get_valid_keypoints(kp_coords, kp_confs, confidence_threshold=0.3)
        else:
            valid = {}
        return self.metrics_calculator.calculate_metrics(valid, current_time)

    def process(
        self, frame: np.ndarray, current_time: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        model_output = self.predict(frame)
        return self.analyze(model_output, frame.shape, current_time)

    def get_key_metrics(self) -> Dict[str, Any]:
        return self.metrics_calculator.get_key_metrics()

    def close(self):
        self.model.close()
        if self.metrics_calculator.audio_provider is not None:
            self.metrics_calculator.audio_provider.stop()
