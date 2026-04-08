"""Tests for Pydantic models in env/models.py."""
import pytest
from env.models import (
    ActionModel,
    AmbulanceInfo,
    AmbulanceState,
    EmergencyInfo,
    HospitalInfo,
    ObservationModel,
    Rubric,
    Severity,
)


class TestRubric:
    def test_default_total_is_zero(self):
        r = Rubric()
        assert r.total() == 0.0

    def test_total_sums_components(self):
        r = Rubric(emergency_served=20.0, severity_bonus=10.0, idle_penalty=-5.0)
        assert r.total() == 25.0

    def test_negative_total_possible(self):
        r = Rubric(idle_penalty=-100.0)
        assert r.total() < 0


class TestActionModel:
    def test_noop_flag_default_false(self):
        a = ActionModel(ambulance_id=0, emergency_id="e1", hospital_id=0)
        assert a.is_noop is False

    def test_noop_constructor(self):
        a = ActionModel(ambulance_id=None, emergency_id="", hospital_id=None, is_noop=True)
        assert a.is_noop is True
        assert a.ambulance_id is None


class TestEmergencyInfo:
    def test_max_time_remaining_defaults(self):
        e = EmergencyInfo(
            id="e1",
            node=10,
            severity=Severity.HIGH,
            time_remaining=90,
            assigned=False,
            spawn_time=0,
        )
        assert e.max_time_remaining == 90

    def test_assigned_default_false(self):
        e = EmergencyInfo(
            id="e2",
            node=5,
            severity=Severity.CRITICAL,
            time_remaining=60,
            assigned=False,
            spawn_time=0,
        )
        assert e.assigned is False


class TestObservationModel:
    def test_defaults_are_set(self):
        obs = ObservationModel(
            ambulances=[],
            emergencies=[],
            hospitals=[],
        )
        assert obs.reward == 0.0
        assert obs.done is False
        assert obs.step == 0

    def test_rubric_embedded(self):
        r = Rubric(emergency_served=20.0)
        obs = ObservationModel(
            ambulances=[],
            emergencies=[],
            hospitals=[],
            rubric=r,
        )
        assert obs.rubric.emergency_served == 20.0
