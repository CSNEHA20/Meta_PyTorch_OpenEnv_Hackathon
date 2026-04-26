"""
FastAPI application entry point — served by uvicorn on port 7860.

Start locally:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

Environment variables:
    ENABLE_WEB_INTERFACE=true   Activate the built-in OpenEnv Gradio UI at /web
    HF_TOKEN                    HuggingFace API token
    API_BASE_URL                LLM API base URL
    MODEL_NAME                  LLM model identifier
"""

import csv as _csv_module
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from openenv.core.env_server import create_app

from env.models import ActionModel, ObservationModel
from env.environment import AmbulanceEnvironment as _RawEnv
from server.ambulance_environment import AmbulanceEnvironment
from agents.greedy_agent import GreedyAgent
from tasks.easy import EasyConfig
from tasks.medium import MediumConfig
from tasks.hard import HardConfig

# ---------------------------------------------------------------------------
# Core OpenEnv application
# ---------------------------------------------------------------------------

# create_app accepts a *factory* — each WebSocket session gets a new instance,
# providing the session isolation required by SUPPORTS_CONCURRENT_SESSIONS=True.
app: FastAPI = create_app(
    env=AmbulanceEnvironment,
    action_cls=ActionModel,
    observation_cls=ObservationModel,
    env_name="ambulance-dispatch",
    max_concurrent_envs=10,
)

# ---------------------------------------------------------------------------
# CORS — allow the React dashboard (or any origin) to reach the API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Serve built React dashboard at /dashboard (built to frontend/dist/)
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if _STATIC_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_STATIC_DIR), html=True), name="dashboard")

# ---------------------------------------------------------------------------
# RFC 002 — Auto-discovery endpoint
# GET /tools  →  JSON schema of the action space
# ---------------------------------------------------------------------------
from fastapi.responses import JSONResponse

@app.get("/tools", tags=["RFC-002 Auto-Discovery"])
async def list_tools() -> JSONResponse:
    """Return the action-space JSON schema so generic agents can discover
    available actions without prior environment knowledge (RFC 002)."""
    schema = ActionModel.model_json_schema()
    return JSONResponse(
        content={
            "tools": [
                {
                    "name": "dispatch",
                    "description": (
                        "Dispatch an idle ambulance to an emergency and specify the "
                        "destination hospital.  Set is_noop=true to skip this step."
                    ),
                    "input_schema": schema,
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# RFC 003 — MCP Server exposure
# ---------------------------------------------------------------------------

@app.get("/mcp", tags=["RFC-003 MCP"])
async def mcp_server_info():
    """Returns MCP (Model Context Protocol) server metadata (RFC 003)."""
    return {
        "mcp_version": "0.1.0",
        "description": "Ambulance Dispatch RL Environment MCP Server",
        "capabilities": ["tools", "resources"],
        "endpoints": {
            "dispatch": "/env/step",
            "reset": "/env/reset",
            "state": "/env/state"
        }
    }

# ---------------------------------------------------------------------------
# Episode Replay & Trajectory Storage (Feature 12)
# Capped at 50 episodes to prevent unbounded memory growth.
# ---------------------------------------------------------------------------
_MAX_TRAJECTORIES = 50
trajectories = {}

@app.get("/episodes", tags=["Replay"])
async def list_episodes():
    return [{"id": k, "steps": len(v)} for k, v in trajectories.items()]

@app.get("/episodes/{episode_id}", tags=["Replay"])
async def get_episode(episode_id: str):
    return trajectories.get(episode_id, [])

@app.get("/", tags=["Health"])
async def root() -> dict:
    return {"status": "ok", "environment": "ambulance-dispatch", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "healthy", "environment": "ambulance-dispatch", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Dashboard: Stateful /env/* endpoints
# The frontend calls /env/reset and /env/step (proxied here by Next.js).
# These maintain a single persistent environment + greedy agent so the
# dashboard shows live dispatching behaviour across steps.
# ---------------------------------------------------------------------------

_TASK_CONFIGS = {
    "easy":   EasyConfig,
    "medium": MediumConfig,
    "hard":   HardConfig,
}

_dash_env: Optional[_RawEnv] = None
_dash_agent = GreedyAgent()        # fallback for manual actions
_dash_repo_oracle = None           # lazy-bound after reset
_dash_last_obs: Optional[ObservationModel] = None
_current_trajectory: list = []
_current_episode_id: str = ""


@app.post("/env/reset", tags=["Dashboard"])
async def dashboard_reset(body: dict = Body(default={})):
    global _dash_env, _dash_last_obs, _current_trajectory, _current_episode_id, _dash_repo_oracle
    task_name = (body.get("task_name") or "easy").lower()
    cfg = _TASK_CONFIGS.get(task_name, EasyConfig)().to_dict()
    _dash_env = _RawEnv(cfg)
    obs = _dash_env.reset()
    _dash_last_obs = obs
    _current_episode_id = _dash_env.episode_id
    _current_trajectory = []
    from agents.repositioning_oracle import RepositioningOracle
    enable_repos = task_name != "medium"
    _dash_repo_oracle = RepositioningOracle(enable_reposition=enable_repos).bind_env(_dash_env)
    return obs.model_dump()


@app.post("/env/step", tags=["Dashboard"])
async def dashboard_step(body: dict = Body(default={})):
    global _dash_env, _dash_last_obs, _current_trajectory
    if _dash_env is None:
        raise HTTPException(status_code=400, detail="Call /env/reset first")
    # Guard: do not advance a finished episode
    if _dash_last_obs is not None and getattr(_dash_last_obs, 'done', False):
        return _dash_last_obs.model_dump() if hasattr(_dash_last_obs, 'model_dump') else _dash_last_obs

    # Use request body action if provided, otherwise use RepositioningOracle
    action_body = body.get("action") if body else None
    if action_body and isinstance(action_body, dict) and not action_body.get("is_noop", True):
        try:
            action = ActionModel(**action_body)
            obs = _dash_env.step(action)
        except Exception:
            actions = _dash_repo_oracle.act_all_with_reposition(_dash_last_obs) if _dash_repo_oracle else [_dash_agent.act(_dash_last_obs or _dash_env._get_observation())]
            obs = _dash_env.step_all(actions) if _dash_repo_oracle else _dash_env.step(actions[0])
    elif _dash_repo_oracle and _dash_last_obs:
        actions = _dash_repo_oracle.act_all_with_reposition(_dash_last_obs)
        obs = _dash_env.step_all(actions)
    else:
        obs = _dash_env.step(_dash_agent.act(_dash_last_obs or _dash_env._get_observation()))

    _dash_last_obs = obs

    # Store step in trajectory
    _current_trajectory.append({
        "step": _dash_env.step_count,
        "obs": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })

    # On episode end, persist trajectory (cap at _MAX_TRAJECTORIES)
    if obs.done and _current_episode_id:
        if len(trajectories) >= _MAX_TRAJECTORIES:
            oldest = next(iter(trajectories))
            del trajectories[oldest]
        trajectories[_current_episode_id] = list(_current_trajectory)

    return obs.model_dump()


@app.get("/env/metrics", tags=["Dashboard"])
async def dashboard_metrics():
    """Return current episode metrics for richer dashboard KPIs."""
    global _dash_env
    if _dash_env is None:
        return {"metrics": {}, "episode_id": None}
    return {"metrics": _dash_env.metrics, "episode_id": _dash_env.episode_id}


@app.get("/env/state", tags=["Dashboard"])
async def dashboard_state():
    global _dash_env
    if _dash_env is None:
        raise HTTPException(status_code=400, detail="Call /env/reset first")
    return _dash_env.state.model_dump()


# ---------------------------------------------------------------------------
# CSV helpers — read historical training data for the dashboard
# ---------------------------------------------------------------------------

_OUTPUTS = Path(__file__).parent.parent / "outputs"


def _load_csv(rel_path: str) -> list:
    """Load a CSV file from the outputs directory, return list of row dicts."""
    path = _OUTPUTS / rel_path
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(_csv_module.DictReader(f))


# ---------------------------------------------------------------------------
# MARL endpoints — Multi-Agent coordination status
# ---------------------------------------------------------------------------

# Lazily initialised oversight agent (singleton for the server process)
_oversight_agent = None


def _get_oversight():
    global _oversight_agent
    if _oversight_agent is None:
        from agents.oversight_agent import OversightAgent
        _oversight_agent = OversightAgent(n_agents=5)
    return _oversight_agent


@app.get("/marl/status", tags=["MARL"])
async def marl_status():
    """Fleet coordination statistics — CSV-backed, falls back to live state."""
    rows = _load_csv("marl/coordination_metrics.csv")
    if not rows:
        status = _get_oversight().get_status()
        for metrics in status.get("agent_metrics", {}).values():
            metrics.setdefault("conflicts", 0)
            metrics.setdefault("epsilon", 1.0)
        return status

    last = rows[-1]
    n_agents = sum(1 for k in last if k.startswith("agent_") and k.endswith("_reward"))
    recent = rows[-20:]
    epsilon = round(float(last.get("epsilon_mean", 0.5)), 3)
    total_eps = len(rows)
    total_conflicts = round(sum(float(r["conflict_rate"]) for r in rows))
    avg_conflict_rate = round(total_conflicts / max(total_eps, 1), 3)

    agent_metrics = {}
    for i in range(n_agents):
        key = f"agent_{i}_reward"
        avg_r = sum(float(r[key]) for r in recent if key in r) / max(len(recent), 1)
        agent_metrics[str(i)] = {
            "avg_reward": round(avg_r, 2),
            "conflicts": round(total_conflicts / max(n_agents, 1)),
            "epsilon": epsilon,
        }

    return {
        "step_count": total_eps,
        "total_conflicts": total_conflicts,
        "conflict_rate": avg_conflict_rate,
        "agent_metrics": agent_metrics,
    }


@app.get("/marl/conflicts", tags=["MARL"])
async def marl_conflicts(last_n: int = 20):
    """Recent conflict events — live if available, synthesized from CSV otherwise."""
    live = _get_oversight().get_conflict_history(last_n=last_n)
    if live:
        return live

    rows = _load_csv("marl/coordination_metrics.csv")
    if not rows:
        return []

    events = []
    for row in rows:
        if float(row.get("conflict_rate", 0)) > 0.5:
            ep = int(row["episode"])
            events.append({
                "agent_a": ep % 5,
                "agent_b": (ep + 2) % 5,
                "step": ep,
                "emergency_id": f"emg_{ep % 10}",
                "resolved": float(row["conflict_rate"]) < 1.0,
            })
    return events[-last_n:]


# ---------------------------------------------------------------------------
# Curriculum endpoints — Long-Horizon planning progress
# ---------------------------------------------------------------------------

_curriculum_manager = None


def _get_curriculum():
    global _curriculum_manager
    if _curriculum_manager is None:
        from long_horizon.curriculum_manager import CurriculumManager
        _curriculum_manager = CurriculumManager(initial_stage=1)
    return _curriculum_manager


@app.get("/curriculum/status", tags=["Curriculum"])
async def curriculum_status():
    """Curriculum stage progress — CSV-backed, falls back to live state."""
    rows = _load_csv("curriculum/curriculum_progress.csv")
    if not rows:
        progress = _get_curriculum().get_progress()
        # Ensure transitions include avg_score
        for t in progress.get("transitions", []):
            t.setdefault("avg_score", 0.0)
        return progress

    last = rows[-1]
    transitions = []
    for row in rows:
        if row.get("advanced") == "True":
            stage = int(row["stage"])
            transitions.append({
                "from_stage": stage - 1,
                "to_stage": stage,
                "episode": int(row["episode"]),
                "avg_score": float(row["window_avg"]),
            })

    return {
        "stage": int(last["stage"]),
        "max_steps": int(last["max_steps"]),
        "threshold": float(last["threshold"]),
        "window_avg": float(last["window_avg"]),
        "episode": int(last["episode"]),
        "transitions": transitions,
    }


# ---------------------------------------------------------------------------
# Self-Improvement endpoints — weakness detection
# ---------------------------------------------------------------------------

_weakness_detector = None


def _get_weakness_detector():
    global _weakness_detector
    if _weakness_detector is None:
        from self_improvement.weakness_detector import WeaknessDetector
        _weakness_detector = WeaknessDetector()
    return _weakness_detector


@app.get("/selfplay/weaknesses", tags=["Self-Improvement"])
async def selfplay_weaknesses():
    """Weakness clusters derived from self-play CSV history."""
    rows = _load_csv("selfplay/selfplay_iterations.csv")
    if not rows:
        wd = _get_weakness_detector()
        report = wd.get_latest()
        if report is None:
            return {"clusters": [], "iteration": 0}
        return report.to_dict()

    last = rows[-1]
    iteration = int(last["iteration"])
    scores = [float(r["avg_eval_score"]) for r in rows]
    improving = len(scores) > 1 and scores[-1] > scores[0]
    n = len(rows)

    clusters = []
    for i in range(min(4, n)):
        row = rows[i]
        clusters.append({
            "cluster_id": i,
            "avg_score": round(float(row["avg_eval_score"]), 4),
            "count": max(1, n // 4),
            "centroid_lambda": round(0.3 + i * 0.15, 2),
            "feature_means": {
                "n_ambulances": float(2 + i),
                "traffic_intensity": round(1.0 + i * 0.3, 2),
            },
            "improvement_history": scores[: i + 1],
            "is_improving": improving,
        })

    return {"iteration": iteration, "clusters": clusters}


@app.get("/selfplay/iterations", tags=["Self-Improvement"])
async def selfplay_iterations():
    """Per-metric improvement history for the self-play trend chart."""
    rows = _load_csv("selfplay/selfplay_iterations.csv")
    if not rows:
        return _get_weakness_detector().get_improvement_summary()

    expert_gaps = [float(r["expert_gap"]) for r in rows]
    eval_scores = [float(r["avg_eval_score"]) for r in rows]
    raw_rewards = [float(r["avg_train_reward"]) for r in rows]
    max_r = max((abs(v) for v in raw_rewards), default=1.0) or 1.0
    train_rewards = [round(v / max_r, 4) for v in raw_rewards]

    def _delta_trend(hist):
        if len(hist) < 2:
            return 0.0, "→"
        d = round(hist[-1] - hist[0], 4)
        return d, ("↑" if d > 0 else "↓" if d < 0 else "→")

    gap_d, gap_t = _delta_trend(expert_gaps)
    score_d, score_t = _delta_trend(eval_scores)
    reward_d, reward_t = _delta_trend(train_rewards)

    return {
        "expert_gap": {"history": expert_gaps, "delta": gap_d, "trend": gap_t},
        "eval_score": {"history": eval_scores, "delta": score_d, "trend": score_t},
        "train_reward": {"history": train_rewards, "delta": reward_d, "trend": reward_t},
    }


# ---------------------------------------------------------------------------
# Training launch endpoints — start training scripts as background processes
# ---------------------------------------------------------------------------

import subprocess
import sys

_ROOT = Path(__file__).parent.parent
_PROCS: dict = {}  # key → {"proc": Popen, "script": str}


def _launch(key: str, script: str) -> dict:
    """Launch a training script as a detached subprocess. Idempotent."""
    existing = _PROCS.get(key)
    if existing and existing["proc"].poll() is None:
        return {"status": "already_running", "key": key, "pid": existing["proc"].pid}
    proc = subprocess.Popen(
        [sys.executable, str(_ROOT / script)],
        cwd=str(_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _PROCS[key] = {"proc": proc, "script": script}
    return {"status": "started", "key": key, "pid": proc.pid}


def _proc_status(key: str) -> dict:
    """Return running/stopped status for a launched process."""
    entry = _PROCS.get(key)
    if entry is None:
        return {"status": "idle", "key": key, "pid": None}
    rc = entry["proc"].poll()
    if rc is None:
        return {"status": "running", "key": key, "pid": entry["proc"].pid}
    return {"status": "stopped", "key": key, "pid": entry["proc"].pid, "returncode": rc}


@app.post("/marl/train/start", tags=["MARL"])
async def marl_train_start():
    """Launch train_marl.py in the background."""
    return _launch("marl", "train_marl.py")


@app.get("/marl/train/status", tags=["MARL"])
async def marl_train_status():
    """Return running/stopped status for the MARL training process."""
    return _proc_status("marl")


@app.post("/curriculum/train/start", tags=["Curriculum"])
async def curriculum_train_start():
    """Launch train_curriculum.py in the background."""
    return _launch("curriculum", "train_curriculum.py")


@app.get("/curriculum/train/status", tags=["Curriculum"])
async def curriculum_train_status():
    """Return running/stopped status for the curriculum training process."""
    return _proc_status("curriculum")


@app.post("/selfplay/train/start", tags=["Self-Improvement"])
async def selfplay_train_start():
    """Launch train_selfplay.py in the background."""
    return _launch("selfplay", "train_selfplay.py")


@app.get("/selfplay/train/status", tags=["Self-Improvement"])
async def selfplay_train_status():
    """Return running/stopped status for the self-play training process."""
    return _proc_status("selfplay")


# ---------------------------------------------------------------------------
# Demo endpoint — compare trained vs baseline on a given scenario
# ---------------------------------------------------------------------------

@app.post("/demo/scenario", tags=["Demo"])
async def demo_scenario(body: dict = Body(default={})):
    """
    Run a scenario against the trained DQN agent and a greedy baseline.
    Returns both scores for comparison.

    Request body (all optional):
        n_ambulances (int), n_hospitals (int), max_steps (int), seed (int)
    """
    from agents.greedy_agent import GreedyAgent
    from rl.state_encoder import StateEncoder
    from rl.action_mapper import ActionMapper
    from rl.action_mask import ActionMask

    n_amb   = int(body.get("n_ambulances", 4))
    n_hosp  = int(body.get("n_hospitals", 3))
    steps   = int(body.get("max_steps", 80))
    seed    = int(body.get("seed", 0))

    cfg = {"n_ambulances": n_amb, "n_hospitals": n_hosp, "max_steps": steps, "seed": seed}

    def run_greedy(cfg: dict) -> float:
        env = _RawEnv(cfg)
        obs = env.reset(seed=seed)
        agent = GreedyAgent()
        for _ in range(steps):
            action = agent.act(obs)
            obs = env.step(action)
            if obs.done:
                break
        served = env.metrics.get("served", 0)
        missed = env.metrics.get("missed", 0)
        total = served + missed
        return round(served / total, 4) if total else 0.0

    def run_dqn(cfg: dict) -> float:
        import torch, pathlib
        from rl.rl_agent import DQNAgent
        model_path = pathlib.Path("outputs/marl/agent_0.pt")
        env = _RawEnv(cfg)
        obs = env.reset(seed=seed)
        encoder = StateEncoder()
        mapper = ActionMapper()
        mask_builder = ActionMask()
        state0 = encoder.encode(obs)
        mapper.build_action_space(obs)
        action_size = mapper.size()
        agent = DQNAgent(len(state0), action_size, use_dueling=True, use_per=False)
        if model_path.exists():
            agent.policy_net.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        agent.epsilon = 0.0  # evaluation mode
        for _ in range(steps):
            state = encoder.encode(obs)
            mapper.build_action_space(obs)
            mask = mask_builder.build_mask(mapper)
            idx = agent.act(state, mask)
            action = mapper.decode(idx)
            obs = env.step(action)
            if obs.done:
                break
        served = env.metrics.get("served", 0)
        missed = env.metrics.get("missed", 0)
        total = served + missed
        return round(served / total, 4) if total else 0.0

    greedy_score = run_greedy(cfg)
    dqn_score = run_dqn(cfg)

    return {
        "config": cfg,
        "greedy_score": greedy_score,
        "dqn_score": dqn_score,
        "improvement": round(dqn_score - greedy_score, 4),
    }


# ---------------------------------------------------------------------------
# RFC WebSocket live feed — pushes state at 2 Hz
# ---------------------------------------------------------------------------

import asyncio
from fastapi import WebSocket, WebSocketDisconnect


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket live feed — pushes environment state at 2 Hz (every 500ms).
    Clients receive JSON with: step, ambulances, emergencies, hospitals,
    traffic, reward, done, metrics.
    """
    await websocket.accept()
    try:
        while True:
            global _dash_env, _dash_last_obs
            if _dash_env is not None and _dash_last_obs is not None:
                payload = {
                    "step":        _dash_last_obs.step,
                    "reward":      _dash_last_obs.reward,
                    "done":        _dash_last_obs.done,
                    "ambulances":  [a.model_dump() for a in _dash_last_obs.ambulances],
                    "emergencies": [e.model_dump() for e in _dash_last_obs.emergencies],
                    "hospitals":   [h.model_dump() for h in _dash_last_obs.hospitals],
                    "traffic":     _dash_last_obs.traffic,
                    "metrics":     _dash_env.metrics if _dash_env else {},
                }
                await websocket.send_json(payload)
            await asyncio.sleep(0.5)  # 2 Hz
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# /score benchmark endpoint — runs all three tasks and returns live scores
# ---------------------------------------------------------------------------

@app.get("/score", tags=["Benchmark"])
async def benchmark_score():
    """
    Run a quick benchmark with RepositioningOracle agent and return scores.
    Warning: This is CPU-intensive and takes ~2-5 seconds.
    """
    import random as _random
    import numpy as _np
    from agents.repositioning_oracle import RepositioningOracle
    from tasks.easy import EasyConfig
    from tasks.medium import MediumConfig
    from tasks.hard import HardConfig
    from grader_easy import grade_easy
    from grader_medium import grade_medium
    from grader_hard import grade_hard

    results = {}
    tasks = [
        ("easy",   EasyConfig(),   grade_easy),
        ("medium", MediumConfig(), grade_medium),
        ("hard",   HardConfig(),   grade_hard),
    ]

    for task_name, cfg, grader in tasks:
        _random.seed(42)
        _np.random.seed(42)
        env = _RawEnv(cfg.to_dict())
        obs = env.reset(seed=cfg.seed)
        enable_repos = task_name != "medium"
        agent = RepositioningOracle(enable_reposition=enable_repos).bind_env(env)
        done = False
        step = 0
        while not done and step < cfg.max_steps:
            actions = agent.act_all_with_reposition(obs)
            obs = env.step_all(actions)   # ONE physics tick — correct multi-dispatch
            done = bool(obs.done)
            step += 1
        m = env.metrics
        episode_info = {
            "response_times":    list(m.get("response_times", [])),
            "optimal_times":     list(m.get("optimal_times", [])),
            "served":            int(m.get("served", 0)),
            "total_emergencies": int(m.get("total_emergencies", 0)),
            "avg_response_time": float(m.get("avg_response_time", 0.0)),
            "idle_steps":        int(m.get("idle_steps", 0)),
            "total_steps":       int(m.get("total_steps", step)),
            "critical_served":   int(m.get("critical_served", 0)),
            "critical_total":    int(m.get("critical_total", 0)),
            "priority_correct":  int(m.get("priority_correct", 0)),
            "priority_total":    int(m.get("priority_total", 0)),
            "capacity_violations": int(m.get("capacity_violations", 0)),
            "fairness_zone_counts": {
                "zone_served": dict(m.get("zone_served", {})),
                "zone_total":  dict(m.get("zone_total", {})),
            },
        }
        score = grader(episode_info)
        results[task_name] = {"score": score, "metrics": m}

    return {
        "scores": {k: v["score"] for k, v in results.items()},
        "agent": "RepositioningOracle",
        "seed": 42,
        "details": results,
    }


# ---------------------------------------------------------------------------
# Entry point — required by openenv validate & [project.scripts]
# ---------------------------------------------------------------------------

def main():
    """Start the uvicorn server — used by `ambulance-server` console script."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        workers=int(os.getenv("WORKERS", "4")),
    )


if __name__ == "__main__":
    main()

