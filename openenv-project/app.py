import logging
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import EcomEnv
from tasks import list_tasks, get_task
from grader import grade

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openenv")

# --- App ---
app = FastAPI(
    title="OpenEnv E-Commerce Environment",
    description="A reinforcement learning environment for e-commerce operations. Supported actions: remove_item, process_order, handle_return, wait, add_item.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session store ---
sessions: dict[str, EcomEnv] = {}


# --- Base endpoints ---
@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "OpenEnv Server Running",
        "docs_url": "/docs",
        "available_tasks_url": "/tasks"
    }

from fastapi.responses import Response

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")


# --- Request / Response models ---
class ActionRequest(BaseModel):
    session_id: str
    type: str
    index: Optional[int] = 0


class ResetResponse(BaseModel):
    session_id: str
    observation: dict
    reward: float
    total_reward: float
    done: bool


# --- Helper ---
def _get_session(session_id: str) -> EcomEnv:
    """Retrieve a session or raise 404."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return sessions[session_id]


# --- Endpoints ---

@app.get("/tasks")
def get_tasks():
    """List all available task configurations."""
    return {"tasks": list_tasks()}


@app.get("/reset")
def reset(task_id: str = "easy"):
    """
    Start a new episode. Creates a new session and returns the initial state.
    """
    # Validate task_id
    try:
        get_task(task_id)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session_id = str(uuid.uuid4())
    env = EcomEnv()
    state = env.reset(task_id)
    sessions[session_id] = env

    logger.info(f"New session {session_id[:8]}... created for task '{task_id}'")

    return {"session_id": session_id, **state}


@app.post("/step")
def step(action: ActionRequest):
    """
    Execute one action in the environment and return the updated state.
    """
    env = _get_session(action.session_id)

    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is already done. Call /reset to start a new one.",
        )

    result = env.step({"type": action.type, "index": action.index})
    return {"session_id": action.session_id, **result}


@app.get("/state")
def state(session_id: str):
    """
    Get the current state of a session without taking an action.
    """
    env = _get_session(session_id)
    return {"session_id": session_id, **env.state()}


@app.get("/grade")
def grade_episode(session_id: str):
    """
    Grade the completed episode for a session.
    """
    env = _get_session(session_id)

    if not env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is still in progress. Complete it before grading.",
        )

    result = grade(env.history, env.task_id)
    return {"session_id": session_id, **result}


@app.get("/sessions")
def get_sessions():
    """List all active sessions with their status."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "task_id": env.task_id,
                "step": env.step_count,
                "done": env.done,
                "total_reward": round(env.total_reward, 2),
            }
            for sid, env in sessions.items()
        ],
        "count": len(sessions),
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a session to free resources."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    del sessions[session_id]
    logger.info(f"Session {session_id[:8]}... deleted")
    return {"detail": "Session deleted."}
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
