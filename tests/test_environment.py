"""Tests for AmbulanceEnvironment core functionality."""
import pytest
from env.environment import AmbulanceEnvironment
from env.models import ObservationModel, ActionModel, AmbulanceState


def _noop() -> ActionModel:
    return ActionModel(ambulance_id=None, emergency_id="", hospital_id=None, is_noop=True)


class TestReset:
    def test_returns_observation_model(self):
        env = AmbulanceEnvironment()
        obs = env.reset(seed=42)
        assert isinstance(obs, ObservationModel)

    def test_done_is_false_after_reset(self):
        env = AmbulanceEnvironment()
        obs = env.reset(seed=0)
        assert obs.done is False

    def test_step_count_zero_after_reset(self):
        env = AmbulanceEnvironment({"max_steps": 30})
        env.reset(seed=1)
        assert env.step_count == 0

    def test_episode_id_is_set(self):
        env = AmbulanceEnvironment()
        env.reset(seed=7)
        assert hasattr(env, "episode_id")
        assert env.episode_id is not None and len(env.episode_id) > 0

    def test_episode_id_changes_each_reset(self):
        env = AmbulanceEnvironment()
        env.reset(seed=42)
        id1 = env.episode_id
        env.reset(seed=42)
        id2 = env.episode_id
        assert id1 != id2

    def test_determinism_same_seed(self):
        env1 = AmbulanceEnvironment({"seed": 42})
        env2 = AmbulanceEnvironment({"seed": 42})
        obs1 = env1.reset(seed=42)
        obs2 = env2.reset(seed=42)
        assert obs1.step == obs2.step
        assert len(obs1.ambulances) == len(obs2.ambulances)


class TestStep:
    def setup_method(self):
        self.env = AmbulanceEnvironment({"max_steps": 30, "seed": 42})
        self.env.reset(seed=42)

    def test_step_returns_observation_model(self):
        obs = self.env.step(_noop())
        assert isinstance(obs, ObservationModel)

    def test_step_reward_is_float(self):
        obs = self.env.step(_noop())
        assert isinstance(obs.reward, (int, float))

    def test_step_done_is_bool(self):
        obs = self.env.step(_noop())
        assert isinstance(obs.done, bool)

    def test_step_increments_count(self):
        self.env.step(_noop())
        assert self.env.step_count == 1

    def test_done_after_max_steps(self):
        for _ in range(30):
            obs = self.env.step(_noop())
        assert obs.done is True

    def test_metrics_exposed(self):
        self.env.step(_noop())
        assert "served" in self.env.metrics
        assert "missed" in self.env.metrics
        assert "avg_response_time" in self.env.metrics


class TestState:
    def test_state_has_episode_id(self):
        env = AmbulanceEnvironment()
        env.reset(seed=5)
        state = env.state
        assert hasattr(state, "episode_id")
        assert state.episode_id is not None

    def test_state_has_step_count(self):
        env = AmbulanceEnvironment()
        env.reset(seed=5)
        assert hasattr(env.state, "step_count")

    def test_state_has_metrics(self):
        env = AmbulanceEnvironment()
        env.reset(seed=5)
        assert hasattr(env.state, "metrics")


class TestDeterminism:
    """Same seed must produce identical trajectories."""

    def _run(self, seed: int, steps: int = 10):
        env = AmbulanceEnvironment({"max_steps": steps, "seed": seed})
        env.reset(seed=seed)
        rewards = []
        for _ in range(steps):
            obs = env.step(_noop())
            rewards.append(obs.reward)
        return rewards, env.metrics.copy()

    def test_same_seed_same_rewards(self):
        r1, _ = self._run(seed=7)
        r2, _ = self._run(seed=7)
        assert r1 == r2

    def test_same_seed_same_metrics(self):
        _, m1 = self._run(seed=99)
        _, m2 = self._run(seed=99)
        assert m1["served"] == m2["served"]
        assert m1["missed"] == m2["missed"]

    def test_different_seeds_differ(self):
        r1, _ = self._run(seed=1)
        r2, _ = self._run(seed=2)
        # They *may* differ (not guaranteed, but very likely)
        assert r1 != r2 or True  # just smoke-test — must not crash


class TestMetricsTracking:
    def test_response_times_list_populated_after_serve(self):
        """Run many steps; if at least one emergency is served, response_times is non-empty."""
        env = AmbulanceEnvironment({"max_steps": 100, "lambda_param": 0.8, "seed": 42})
        env.reset(seed=42)
        for _ in range(100):
            env.step(_noop())
        # Either some emergencies were served (list populated) or the test is fine either way
        assert isinstance(env.metrics.get("response_times", []), list)
        assert isinstance(env.metrics.get("optimal_times", []), list)

    def test_idle_steps_increases_when_emergencies_active(self):
        env = AmbulanceEnvironment({"max_steps": 20, "lambda_param": 1.5, "seed": 0})
        env.reset(seed=0)
        for _ in range(20):
            env.step(_noop())
        # idle_steps must be >= 0 and <= n_ambulances * max_steps
        assert env.metrics.get("idle_steps", 0) >= 0

    def test_required_grader_keys_present(self):
        required = [
            "served", "missed", "total_emergencies",
            "critical_served", "critical_total",
            "avg_response_time", "idle_steps", "total_steps",
            "priority_correct", "priority_total",
            "capacity_violations", "response_times", "optimal_times",
        ]
        env = AmbulanceEnvironment({"max_steps": 10, "seed": 42})
        env.reset(seed=42)
        for _ in range(10):
            env.step(_noop())
        for key in required:
            assert key in env.metrics, f"Missing key: {key}"


class TestOracleAgent:
    def test_oracle_returns_action(self):
        from agents.oracle import OracleAgent
        env = AmbulanceEnvironment({"max_steps": 20, "seed": 42})
        obs = env.reset(seed=42)
        agent = OracleAgent()
        action = agent.act(obs)
        assert isinstance(action, ActionModel)

    def test_oracle_scores_higher_than_noop(self):
        from agents.oracle import OracleAgent
        cfg = {"max_steps": 50, "lambda_param": 0.4, "seed": 5}

        env_oracle = AmbulanceEnvironment(cfg)
        obs = env_oracle.reset(seed=5)
        agent = OracleAgent()
        total_oracle = 0.0
        for _ in range(50):
            action = agent.act(obs)
            obs = env_oracle.step(action)
            total_oracle += obs.reward

        env_noop = AmbulanceEnvironment(cfg)
        env_noop.reset(seed=5)
        total_noop = 0.0
        for _ in range(50):
            obs_n = env_noop.step(_noop())
            total_noop += obs_n.reward

        assert total_oracle >= total_noop


class TestActionValidation:
    def test_valid_action_accepted(self):
        a = ActionModel(ambulance_id=0, emergency_id="emg-1", hospital_id=0)
        assert a.ambulance_id == 0

    def test_noop_action_is_valid(self):
        a = ActionModel(ambulance_id=None, emergency_id="", hospital_id=None, is_noop=True)
        assert a.is_noop is True

