"""Tests for the grading functions."""
import pytest
from grader_easy import grade_easy
from grader_medium import grade_medium
from grader_hard import grade_hard


def _make_info(
    served=5,
    total_emergencies=10,
    avg_response_time=20.0,
    critical_served=2,
    high_served=2,
    idle_fraction=0.1,
    hospital_overflow=0,
    step=60,
):
    return {
        "step": step,
        "metrics": {
            "served": served,
            "total_emergencies": total_emergencies,
            "avg_response_time": avg_response_time,
            "critical_served": critical_served,
            "high_served": high_served,
            "idle_fraction": idle_fraction,
            "hospital_overflow": hospital_overflow,
        },
    }


class TestGradeEasy:
    def test_returns_float(self):
        assert isinstance(grade_easy(_make_info()), float)

    def test_in_unit_interval(self):
        for served in [0, 1, 5, 10]:
            score = grade_easy(_make_info(served=served))
            assert 0.0 <= score <= 1.0, f"score={score} out of range for served={served}"

    def test_zero_served_returns_zero(self):
        info = _make_info(served=0)
        assert grade_easy(info) == 0.0

    def test_empty_info_does_not_crash(self):
        score = grade_easy({})
        assert 0.0 <= score <= 1.0

    def test_high_response_time_lowers_score(self):
        fast = grade_easy(_make_info(avg_response_time=5.0, served=5))
        slow = grade_easy(_make_info(avg_response_time=120.0, served=5))
        assert fast >= slow

    def test_perfect_conditions(self):
        # Very fast response, many served → score close to 1
        score = grade_easy(_make_info(served=10, avg_response_time=3.0))
        assert score >= 0.9


class TestGradeMedium:
    def test_returns_float(self):
        assert isinstance(grade_medium(_make_info()), float)

    def test_in_unit_interval(self):
        for combo in [
            _make_info(served=0),
            _make_info(served=10, total_emergencies=10),
            _make_info(idle_fraction=1.0),
        ]:
            score = grade_medium(combo)
            assert 0.0 <= score <= 1.0

    def test_empty_info_does_not_crash(self):
        score = grade_medium({})
        assert 0.0 <= score <= 1.0

    def test_more_served_higher_score(self):
        low = grade_medium(_make_info(served=1, total_emergencies=10))
        high = grade_medium(_make_info(served=9, total_emergencies=10))
        assert high > low


class TestGradeHard:
    def test_returns_float(self):
        assert isinstance(grade_hard(_make_info()), float)

    def test_in_unit_interval(self):
        for combo in [
            _make_info(critical_served=0, served=1),
            _make_info(critical_served=5, high_served=5, served=10),
            _make_info(hospital_overflow=100),
        ]:
            score = grade_hard(combo)
            assert 0.0 <= score <= 1.0

    def test_empty_info_does_not_crash(self):
        score = grade_hard({})
        assert 0.0 <= score <= 1.0

    def test_overflow_penalty_reduces_score(self):
        clean = grade_hard(_make_info(hospital_overflow=0))
        overflow = grade_hard(_make_info(hospital_overflow=50))
        assert clean >= overflow

    def test_all_critical_served_boosts_score(self):
        good = grade_hard(_make_info(critical_served=5, served=5))
        bad = grade_hard(_make_info(critical_served=0, served=5))
        assert good >= bad
