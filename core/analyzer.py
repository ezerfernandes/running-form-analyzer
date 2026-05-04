import time
from typing import Tuple

import cv2
import numpy as np

from core.config import Config
from core.frame_processor import FrameProcessor
from visualization.display import display_mode
from visualization.metric_logger import MetricsLogger
from visualization.pose_drawer import draw_connections, draw_keypoints
from visualization.video_recorder import VideoRecorder


class Analyzer:
    def __init__(self, config: Config):
        self.config = config
        self.side = config.side
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Camera unavailable: cv2.VideoCapture(0) failed to open. "
                "Check camera connection and OS permissions."
            )
        source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.start_time = time.time()
        self.frame_count = 0
        self.display_mode = "metrics"
        self.processor = FrameProcessor(config, enable_audio=True)
        self.metrics_logger = MetricsLogger()
        self.video_recorder = VideoRecorder(fps=source_fps)

        available_metrics = self.processor.get_key_metrics()
        self.metrics_logger.initialize_logging(available_metrics)

    def process_frame(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        current_time = time.time() - self.start_time

        if not self.video_recorder.recording:
            self.video_recorder.start_recording(frame)

        model_output = self.processor.predict(frame)
        if model_output is not None:
            draw_keypoints(frame, model_output, confidence_threshold=0.3)
            draw_connections(frame, model_output, confidence_threshold=0.3)

        metrics, angles = self.processor.analyze(
            model_output, frame.shape, current_time
        )

        self.metrics_logger.log_metrics(current_time, metrics)
        frame = display_mode(frame, metrics, angles, self.display_mode, self.side)
        self.video_recorder.record_frame(frame)
        return True, frame

    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.process_frame()
                if not ret:
                    break

                cv2.imshow("Running Analysis", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("a"):
                    self.display_mode = "angles"
                elif key == ord("m"):
                    self.display_mode = "metrics"
                elif key == ord("r"):
                    self.display_mode = "recommendations"

        except KeyboardInterrupt:
            pass
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.processor.close()
            self.video_recorder.stop_recording()

            print("\nPost-processing options:")
            self.video_recorder.post_recording_options()
            self.metrics_logger.close()
            self.metrics_logger.post_logging_options()
