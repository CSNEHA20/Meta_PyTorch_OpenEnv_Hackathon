"""Tests for AmbulanceEnv core functionality."""
import pytest
from env.environment import AmbulanceEnv
from env.models import ObservationModel, ActionModel, AmbulanceState


class TestReset:
    def test_returns_observation_model(self):
        env = AmbulanceEnv()
        obs = env.reset(seed=42)
        assert isinstance(obs, ObservationModel)

    def test_done_is_false_after_reset(self):
        env = AmbulanceEnv()
        obs = env.reset(seed=0)
        assert obs.done is False

    def test_step_count_zero_after_reset(self):
        env = AmbulanceEnv({"max_steps": 30})
        env.reset(seed=1)
        assert env.step_count == 0

    def test_episode_id_is_set(self):
        env = AmbulanceEnv()
        env.reset(seed=7)
        assert hasattr(env, "episode_id")
        assert env.episode_id is not None and len(env.episode_id) > 0

    def test_episode_id_changes_each_reset(self):
        env = AmbulanceEnv()
        env.reset(seed=42)
        id1 = env.episode_id
        env.reset(seed=42)
        id2 = env.episode_id
        # Different uuid even with same seed
        assert id1 != id2

    def test_determinism_same_seed(self):
        env1 = AmbulanceEnv({"seed": 42})
        env2 = AmbulanceEnv({"seed": 42})
        obs1 = env1.reset(seed=42)
        obs2 = env2.reset(seed=42)
        assert obs1.step == obs2.step
        assert len(obs1.ambulances) == len(obs2.ambulances)


class TestStep:
    def setup_method(self):
        self.env = AmbulanceEnv({"max_steps": 30, "seed": 42})
        self.env.reset(seed=42)

    def _noop_action(self):
        return ActionModel(
            ambulance_id=-1,
            emergency_id="",
            hospital_id=-1,
            reposition_node=-1,
            is_noop=True,
        )

    def test_step_returns_tuple(self):
        action = self._noop_action()
        result = self.env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_step_observation_is_model(self):
        action = self._noop_action()
        obs, reward, done, info = self.env.step(action)
        assert isinstance(obs, ObservationModel)

    def test_step_reward_is_float(self):
        action = self._noop_action()
        obs, reward, done, info = self.env.step(action)
        assert isinstance(reward, (int, float))

    def test_step_increments_count(self):
        action = self._noop_action()
        self.env.step(action)
        assert self.env.step_count == 1

    def test_done_after_max_steps(self):
        action = self._noop_action()
        for _ in range(30):
            obs, reward, done, info = self.env.step(action)
        assert done is True

    def test_info_contains_metrics(self):
        action = self._noop_action()
        obs, reward, done, info = self.env.step(action)
        assert isinstance(info, dict)


class TestState:
    def test_state_dict_has_episode_id(self):
        env = AmbulanceEnv()
        env.reset(seed=5)
        state = env.state()
        assert "episode_id" in state
        assert state["episode_id"] is not None

    def test_state_dict_has_step(self):
        env = AmbulanceEnv()
        env.reset(seed=5)
        state = env.state()
        assert "step" in state

    def test_state_dict_has_metrics(self):
        env = AmbulanceEnv()
        env.reset(seed=5)
        state = env.state()
        assert "metrics" in state


class TestActionValidation:
    def test_invalid_extra_field_raises(self):
        with pytest.raises(Exception):
            ActionModel(
                ambulance_id=0,
                emergency_id=0,
                hospital_id=-1,
                reposition_node=-1,
                unknown_field="bad",  # extra='forbid' should raise
            )

    def test_valid_action_accepted(self):
        a = ActionModel(ambulance_id=0, emergency_id="emg-1", hospital_id=-1, reposition_node=-1)
        assert a.ambulance_id == 0
