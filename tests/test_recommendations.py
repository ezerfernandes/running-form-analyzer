from feedback.recommendations import Recommendation


class TestRecommendation:
    def test_no_recommendations_initially(self):
        rec = Recommendation(window_size=5, consistency_threshold=0.7)
        result = rec.get_recommendations({})
        assert result == []

    def test_needs_improvement_when_mostly_bad(self):
        rec = Recommendation(window_size=5, consistency_threshold=0.7)
        for _ in range(5):
            rec.update({"head_angle_assessment": "Bad"})
        assert rec.needs_improvement("head_angle_assessment")

    def test_no_improvement_when_mostly_good(self):
        rec = Recommendation(window_size=5, consistency_threshold=0.7)
        for _ in range(5):
            rec.update({"head_angle_assessment": "Good"})
        assert not rec.needs_improvement("head_angle_assessment")

    def test_mixed_assessments_below_threshold(self):
        rec = Recommendation(window_size=10, consistency_threshold=0.7)
        for _ in range(4):
            rec.update({"head_angle_assessment": "Bad"})
        for _ in range(6):
            rec.update({"head_angle_assessment": "Good"})
        # 4/10 = 0.4, below 0.7
        assert not rec.needs_improvement("head_angle_assessment")

    def test_mixed_assessments_above_threshold(self):
        rec = Recommendation(window_size=10, consistency_threshold=0.7)
        for _ in range(8):
            rec.update({"head_angle_assessment": "Need Improvement"})
        for _ in range(2):
            rec.update({"head_angle_assessment": "Good"})
        # 8/10 = 0.8, above 0.7
        assert rec.needs_improvement("head_angle_assessment")

    def test_get_recommendations_returns_messages(self):
        rec = Recommendation(window_size=3, consistency_threshold=0.5, min_samples=3)
        for _ in range(3):
            rec.get_recommendations({
                "head_angle_assessment": "Bad",
                "trunk_angle_assessment": "Good",
            })
        result = rec.get_recommendations({
            "head_angle_assessment": "Bad",
            "trunk_angle_assessment": "Good",
        })
        assert "Adjust your head position" in result
        assert "Adjust your torso position" not in result

    def test_unknown_metric_no_crash(self):
        rec = Recommendation(window_size=3, consistency_threshold=0.5, min_samples=3)
        assert not rec.needs_improvement("nonexistent_metric")

    def test_no_trigger_below_min_samples(self):
        rec = Recommendation(window_size=5, consistency_threshold=0.7, min_samples=3)
        rec.update({"head_angle_assessment": "Bad"})
        assert not rec.needs_improvement("head_angle_assessment")
        rec.update({"head_angle_assessment": "Bad"})
        assert not rec.needs_improvement("head_angle_assessment")
        # Only triggers once min_samples is reached
        rec.update({"head_angle_assessment": "Bad"})
        assert rec.needs_improvement("head_angle_assessment")

    def test_empty_assessments_skipped(self):
        rec = Recommendation(window_size=5, consistency_threshold=0.7)
        rec.update({"head_angle_assessment": ""})
        rec.update({"head_angle_assessment": None})
        assert not rec.needs_improvement("head_angle_assessment")
