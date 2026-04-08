"""Tests for the grading functions — uses spec-compliant top-level keys."""
import pytest
from grader_easy import grade_easy
from grader_medium import grade_medium
from grader_hard import grade_hard


# ---------------------------------------------------------------------------
# Helpers — build episode_info dicts in spec-compliant format
# ---------------------------------------------------------------------------

def _easy_info(response_times=None, optimal_times=None):
    """Spec format for grade_easy: lists of per-emergency times."""
    if response_times is None:
        response_times = [10.0, 8.0, 12.0]
    if optimal_times is None:
        optimal_times = [5.0, 5.0, 5.0]
    return {"response_times": response_times, "optimal_times": optimal_times}


def _medium_info(
    served=5,
    total_emergencies=10,
    avg_response_time=10.0,
    idle_steps=6,
    total_steps=60,
):
    return {
        "served": served,
        "total_emergencies": total_emergencies,
        "avg_response_time": avg_response_time,
        "idle_steps": idle_steps,
        "total_steps": total_steps,
    }


def _hard_info(
    critical_served=3,
    critical_total=4,
    served=7,
    total_emergencies=10,
    priority_correct=5,
    priority_total=7,
    capacity_violations=0,
    fairness_zone_counts=None,
):
    if fairness_zone_counts is None:
        fairness_zone_counts = {
            "zone_served": {0: 2, 1: 2, 2: 2, 3: 1},
            "zone_total":  {0: 3, 1: 3, 2: 3, 3: 1},
        }
    return {
        "critical_served": critical_served,
        "critical_total": critical_total,
        "served": served,
        "total_emergencies": total_emergencies,
        "priority_correct": priority_correct,
        "priority_total": priority_total,
        "capacity_violations": capacity_violations,
        "fairness_zone_counts": fairness_zone_counts,
    }


# ---------------------------------------------------------------------------
# grade_easy
# ---------------------------------------------------------------------------

class TestGradeEasy:
    def test_returns_float(self):
        assert isinstance(grade_easy(_easy_info()), float)

    def test_in_unit_interval(self):
        for rt, ot in [
            ([10.0], [5.0]),
            ([1.0, 2.0], [1.0, 1.0]),
            ([100.0], [5.0]),
        ]:
            score = grade_easy(_easy_info(rt, ot))
            assert 0.0 <= score <= 1.0, f"out of range: {score}"

    def test_empty_lists_returns_zero(self):
        assert grade_easy(_easy_info([], [])) == 0.0

    def test_empty_dict_does_not_crash(self):
        score = grade_easy({})
        assert 0.0 <= score <= 1.0

    def test_perfect_ratio_returns_one(self):
        # optimal == actual → ratio = 1.0 for every emergency
        score = grade_easy(_easy_info([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]))
        assert score == pytest.approx(1.0)

    def test_faster_actual_than_optimal_clamped(self):
        # optimal > actual should be clamped to 1.0 per emergency
        score = grade_easy(_easy_info([2.0], [10.0]))
        assert score == pytest.approx(1.0)

    def test_slow_actual_lowers_score(self):
        fast = grade_easy(_easy_info([5.0], [5.0]))
        slow = grade_easy(_easy_info([50.0], [5.0]))
        assert fast > slow

    def test_fallback_metrics_subdict(self):
        """grade_easy must handle legacy 'metrics' wrapper."""
        info = {"metrics": {"served": 5, "total_emergencies": 10, "avg_response_time": 5.0}}
        score = grade_easy(info)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# grade_medium
# ---------------------------------------------------------------------------

class TestGradeMedium:
    def test_returns_float(self):
        assert isinstance(grade_medium(_medium_info()), float)

    def test_in_unit_interval(self):
        for info in [
            _medium_info(served=0),
            _medium_info(served=10, total_emergencies=10, avg_response_time=0.0, idle_steps=0),
            _medium_info(idle_steps=60, total_steps=60),
        ]:
            score = grade_medium(info)
            assert 0.0 <= score <= 1.0, f"out of range: {score}"

    def test_empty_dict_does_not_crash(self):
        assert 0.0 <= grade_medium({}) <= 1.0

    def test_more_served_higher_score(self):
        low = grade_medium(_medium_info(served=1, total_emergencies=10))
        high = grade_medium(_medium_info(served=9, total_emergencies=10))
        assert high > low

    def test_faster_response_higher_score(self):
        fast = grade_medium(_medium_info(avg_response_time=2.0))
        slow = grade_medium(_medium_info(avg_response_time=14.0))
        assert fast > slow

    def test_high_idle_penalises_score(self):
        active = grade_medium(_medium_info(idle_steps=0, total_steps=60))
        lazy = grade_medium(_medium_info(idle_steps=58, total_steps=60))
        assert active > lazy

    def test_formula_weights(self):
        """Manually verify: served=10/10, rt=0, idle=0 → score=0.85."""
        score = grade_medium(_medium_info(served=10, total_emergencies=10,
                                          avg_response_time=0.0, idle_steps=0))
        assert score == pytest.approx(0.85, abs=1e-6)

    def test_fallback_metrics_subdict(self):
        info = {
            "step": 60,
            "metrics": {"served": 5, "total_emergencies": 10,
                        "avg_response_time": 10.0, "idle_steps": 6},
        }
        assert 0.0 <= grade_medium(info) <= 1.0


# ---------------------------------------------------------------------------
# grade_hard
# ---------------------------------------------------------------------------

class TestGradeHard:
    def test_returns_float(self):
        assert isinstance(grade_hard(_hard_info()), float)

    def test_in_unit_interval(self):
        for info in [
            _hard_info(critical_served=0, served=1),
            _hard_info(critical_served=4, critical_total=4, served=10),
            _hard_info(capacity_violations=100),
        ]:
            score = grade_hard(info)
            assert 0.0 <= score <= 1.0, f"out of range: {score}"

    def test_empty_dict_does_not_crash(self):
        assert 0.0 <= grade_hard({}) <= 1.0

    def test_capacity_violations_reduce_score(self):
        clean = grade_hard(_hard_info(capacity_violations=0))
        heavy = grade_hard(_hard_info(capacity_violations=10))
        assert clean >= heavy

    def test_all_critical_served_boosts_score(self):
        good = grade_hard(_hard_info(critical_served=4, critical_total=4))
        bad = grade_hard(_hard_info(critical_served=0, critical_total=4))
        assert good > bad

    def test_high_priority_accuracy_boosts_score(self):
        accurate = grade_hard(_hard_info(priority_correct=7, priority_total=7))
        inaccurate = grade_hard(_hard_info(priority_correct=1, priority_total=7))
        assert accurate > inaccurate

    def test_perfect_fairness_when_uniform_service(self):
        """All zones served equally → fairness_score close to 1."""
        info = _hard_info(
            fairness_zone_counts={
                "zone_served": {0: 5, 1: 5, 2: 5, 3: 5},
                "zone_total":  {0: 5, 1: 5, 2: 5, 3: 5},
            }
        )
        score = grade_hard(info)
        assert score > 0.5

    def test_fallback_metrics_subdict(self):
        info = {
            "metrics": {
                "critical_served": 2, "critical_total": 3,
                "served": 5, "total_emergencies": 8,
                "high_served": 2, "priority_correct": 4,
                "priority_total": 5, "hospital_overflow": 1,
            }
        }
        assert 0.0 <= grade_hard(info) <= 1.0
