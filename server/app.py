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
_dash_agent = GreedyAgent()
_dash_last_obs: Optional[ObservationModel] = None
_current_trajectory: list = []
_current_episode_id: str = ""


@app.post("/env/reset", tags=["Dashboard"])
async def dashboard_reset(body: dict = Body(default={})):
    global _dash_env, _dash_last_obs, _current_trajectory, _current_episode_id
    task_name = (body.get("task_name") or "easy").lower()
    cfg = _TASK_CONFIGS.get(task_name, EasyConfig)().to_dict()
    _dash_env = _RawEnv(cfg)
    obs = _dash_env.reset()
    _dash_last_obs = obs
    _current_episode_id = _dash_env.episode_id
    _current_trajectory = []
    return obs.model_dump()


@app.post("/env/step", tags=["Dashboard"])
async def dashboard_step(body: dict = Body(default={})):
    global _dash_env, _dash_last_obs, _current_trajectory
    if _dash_env is None:
        raise HTTPException(status_code=400, detail="Call /env/reset first")
    # Guard: do not advance a finished episode
    if _dash_last_obs is not None and getattr(_dash_last_obs, 'done', False):
        return _dash_last_obs.model_dump() if hasattr(_dash_last_obs, 'model_dump') else _dash_last_obs

    # Use request body action if provided, otherwise use greedy agent
    action_body = body.get("action") if body else None
    if action_body and isinstance(action_body, dict) and not action_body.get("is_noop", True):
        try:
            action = ActionModel(**action_body)
        except Exception:
            action = _dash_agent.act(_dash_last_obs or _dash_env._get_observation())
    else:
        action = _dash_agent.act(_dash_last_obs or _dash_env._get_observation())

    obs = _dash_env.step(action)
    _dash_last_obs = obs

    # Store step in trajectory
    _current_trajectory.append({
        "step": _dash_env.step_count,
        "action": action.model_dump(),
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

