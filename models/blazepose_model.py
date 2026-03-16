import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")


class BlazePoseModel:
    def __init__(self):
        self._ensure_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    @staticmethod
    def _ensure_model():
        if not os.path.exists(_MODEL_PATH):
            print(f"Downloading pose landmarker model to {_MODEL_PATH}...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

    def close(self):
        self.landmarker.close()

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        self._frame_timestamp_ms += 33  # ~30fps
        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
        if result.pose_landmarks:
            return self._convert_to_movenet_format(result.pose_landmarks[0])
        return None

    @staticmethod
    def _convert_to_movenet_format(landmarks):
        movenet_keypoints = np.zeros((17, 3))

        blazepose_to_movenet = {
            0: 0,  # nose
            2: 1,  # left_eye
            5: 2,  # right_eye
            7: 3,  # left_ear
            8: 4,  # right_ear
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10,  # right_wrist
            23: 11,  # left_hip
            24: 12,  # right_hip
            25: 13,  # left_knee
            26: 14,  # right_knee
            27: 15,  # left_ankle
            28: 16,  # right_ankle
        }

        for blazepose_idx, movenet_idx in blazepose_to_movenet.items():
            if blazepose_idx < len(landmarks):
                lm = landmarks[blazepose_idx]
                movenet_keypoints[movenet_idx] = [
                    lm.y,
                    lm.x,
                    lm.visibility if lm.visibility is not None else 0.0,
                ]

        return movenet_keypoints.reshape(1, 17, 3)

    @staticmethod
    def convert_blazepose_to_keypoints(landmarks):
        if landmarks is not None:
            keypoint_coords = [(lm[1], lm[0]) for lm in landmarks[0]]
            keypoint_confs = [lm[2] for lm in landmarks[0]]
            return keypoint_coords, keypoint_confs
        return None, None
