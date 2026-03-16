from collections import deque

from feedback.audio_feedback import AudioFeedbackProvider


class Recommendation:
    def __init__(
        self,
        window_size=30,
        consistency_threshold=0.7,
        min_samples=5,
        audio_provider: AudioFeedbackProvider | None = None,
    ):
        self.window_size = window_size
        self.consistency_threshold = consistency_threshold
        self.min_samples = min_samples
        self.metric_history = {}
        self.audio_provider = audio_provider
        self.recommendations = {
            "head_angle_assessment": "Adjust your head position",
            "trunk_angle_assessment": "Adjust your torso position",
            "left_elbow_angle_assessment": "Adjust your left elbow angle",
            "right_elbow_angle_assessment": "Adjust your right elbow angle",
            "left_hip_ankle_angle_assessment": "Adjust your left foot strike",
            "right_hip_ankle_angle_assessment": "Adjust your right foot strike",
            "left_knee_assessment": "Adjust your left knee position",
            "right_knee_assessment": "Adjust your right knee position",
            "left_shank_angle_assessment": "Adjust your left shank position",
            "right_shank_angle_assessment": "Adjust your right shank position",
            "left_arm_backward_swing_assessment": "Adjust your left arm backward swing",
            "left_arm_forward_swing_assessment": "Adjust your left arm forward swing",
            "right_arm_backward_swing_assessment": "Adjust your right arm backward swing",
            "right_arm_forward_swing_assessment": "Adjust your right arm forward swing",
            "left_hip_backward_swing_assessment": "Adjust your left hip backward swing",
            "left_hip_forward_swing_assessment": "Adjust your left hip forward swing",
            "right_hip_backward_swing_assessment": "Adjust your right hip backward swing",
            "right_hip_forward_swing_assessment": "Adjust your right hip forward swing",
            "vertical_oscillation_assessment": "Adjust your vertical oscillation",
        }

    def update(self, metrics):
        for metric, assessment in metrics.items():
            if metric not in self.metric_history:
                self.metric_history[metric] = deque(maxlen=self.window_size)
            if assessment:  # Only add non-empty assessments
                self.metric_history[metric].append(assessment)

    def needs_improvement(self, metric):
        if metric not in self.metric_history:
            return False
        history = self.metric_history[metric]
        if len(history) < self.min_samples:
            return False

        # Count both 'Bad' and 'Need Improvement' assessments
        count = sum(
            1 for assessment in history if assessment in ["Bad", "Need Improvement"]
        )
        return count / len(history) >= self.consistency_threshold

    def get_recommendations(self, metrics):
        self.update(metrics)
        recommendations = [
            self.recommendations[metric]
            for metric in self.recommendations
            if self.needs_improvement(metric)
        ]

        if self.audio_provider:
            for recommendation in recommendations:
                self.audio_provider.add_feedback(recommendation)

        return recommendations
