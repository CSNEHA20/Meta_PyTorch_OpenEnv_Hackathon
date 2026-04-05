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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from openenv.core.env_server import create_app

from env.models import ActionModel, ObservationModel
from server.ambulance_environment import AmbulanceEnvironment

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
# ---------------------------------------------------------------------------
trajectories = {}

@app.get("/episodes", tags=["Replay"])
async def list_episodes():
    return [{"id": k, "steps": len(v)} for k, v in trajectories.items()]

@app.get("/episodes/{episode_id}", tags=["Replay"])
async def get_episode(episode_id: str):
    return trajectories.get(episode_id, [])

@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "ok", "environment": "ambulance-dispatch", "version": "1.0.0"}
