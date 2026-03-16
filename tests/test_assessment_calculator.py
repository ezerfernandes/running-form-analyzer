import pytest

from feedback.assessment_calculator import AssessmentCalculator


class TestAssessHeadAngle:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (50, "Bad"),
            (65, "Need Improvement"),
            (79, "Need Improvement"),
            (80, "Good"),
            (95, "Good"),
            (110, "Good"),
            (115, "Need Improvement"),
            (120, "Need Improvement"),
            (130, "Bad"),
        ],
    )
    def test_head_angle(self, angle, expected):
        assert AssessmentCalculator.assess_head_angle(angle) == expected


class TestAssessKneeAngle:
    @pytest.mark.parametrize(
        "angle, is_front, expected",
        [
            (140, True, "Good"),
            (135, True, "Need Improvement"),
            (130, True, "Need Improvement"),
            (120, True, "Bad"),
            (50, False, "Good"),
            (64, False, "Need Improvement"),
            (73, False, "Need Improvement"),
            (80, False, "Bad"),
        ],
    )
    def test_knee_angle(self, angle, is_front, expected):
        assert AssessmentCalculator.assess_knee_angle(angle, is_front) == expected


class TestAssessTorsoAngle:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (0.5, "Bad"),
            (3, "Need Improvement"),
            (10, "Good"),
            (16, "Need Improvement"),
            (20, "Bad"),
        ],
    )
    def test_torso_angle(self, angle, expected):
        assert AssessmentCalculator.assess_torso_angle(angle) == expected


class TestAssessElbowAngle:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (40, "Bad"),
            (55, "Need Improvement"),
            (75, "Good"),
            (92, "Need Improvement"),
            (100, "Bad"),
        ],
    )
    def test_elbow_angle(self, angle, expected):
        assert AssessmentCalculator.assess_elbow_angle(angle) == expected


class TestAssessArmSwing:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (10, "Bad"),
            (25, "Need Improvement"),
            (37, "Good"),
            (50, "Need Improvement"),
            (60, "Bad"),
        ],
    )
    def test_backward_arm_swing(self, angle, expected):
        assert AssessmentCalculator.assess_backward_arm_swing(angle) == expected

    @pytest.mark.parametrize(
        "angle, expected",
        [
            (20, "Bad"),
            (40, "Need Improvement"),
            (55, "Good"),
            (70, "Need Improvement"),
            (80, "Bad"),
        ],
    )
    def test_forward_arm_swing(self, angle, expected):
        assert AssessmentCalculator.assess_forward_arm_swing(angle) == expected


class TestAssessVerticalOscillation:
    @pytest.mark.parametrize(
        "osc, expected",
        [
            (5.0, "Good"),
            (7.0, "Need Improvement"),
            (9.0, "Bad"),
        ],
    )
    def test_vertical_oscillation(self, osc, expected):
        assert AssessmentCalculator.assess_vertical_oscillation(osc) == expected


class TestAssessHipAnkleAngle:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (10, "Good"),
            (18, "Need Improvement"),
            (25, "Bad"),
        ],
    )
    def test_hip_ankle_angle(self, angle, expected):
        assert AssessmentCalculator.assess_hip_ankle_angle(angle) == expected


class TestAssessHipSwing:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (20, "Bad"),
            (35, "Good"),
            (50, "Need Improvement"),
        ],
    )
    def test_backward_hip_swing(self, angle, expected):
        assert AssessmentCalculator.assess_backward_hip_swing(angle) == expected

    @pytest.mark.parametrize(
        "angle, expected",
        [
            (20, "Bad"),
            (35, "Good"),
            (50, "Need Improvement"),
        ],
    )
    def test_forward_hip_swing(self, angle, expected):
        assert AssessmentCalculator.assess_forward_hip_swing(angle) == expected


class TestAssessShankAngle:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (5, "Good"),
            (12, "Need Improvement"),
            (20, "Bad"),
        ],
    )
    def test_shank_angle(self, angle, expected):
        assert AssessmentCalculator.assess_shank_angle(angle) == expected
