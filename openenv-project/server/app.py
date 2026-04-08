import logging
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .env import EcomEnv
from .tasks import list_tasks, get_task
from .grader import grade, estimate_max_reward, estimate_min_reward

# --- Server Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openenv")

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

sessions: dict[str, EcomEnv] = {}
graded_results: dict[str, dict] = {} 

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
    thought: Optional[str] = ""


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

    result = env.step({
        "type": action.type, 
        "index": action.index,
        "thought": action.thought
    })
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
    last_thought = env.history[-1]["action"].get("thought", "") if env.history else ""
    graded_results[session_id] = {
        "task_id": env.task_id,
        "score": result["score"],
        "grade": result["grade"],
        "steps": env.step_count,
        "total_reward": round(env.total_reward, 2),
        "last_thought": last_thought
    }
    return {"session_id": session_id, **result}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """A stunning, judge-ready dashboard for monitoring agent performance and reasoning."""
    return render_dashboard(sessions, graded_results)


@app.get("/trace/{session_id}", response_class=HTMLResponse)
def get_trace(session_id: str):
    """Detailed visual trace of an episode's current state and history."""
    env = _get_session(session_id)
    return render_trace(session_id, env)


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


# --- UI Templates (Refactored for Readability) ---

def render_dashboard(sessions, graded_results):
    rows = ""
    for sid, res in graded_results.items():
        color = "#00ff88" if res['score'] > 0.8 else "#ffcc00" if res['score'] > 0.5 else "#ff4444"
        thought = res.get('last_thought', 'No reasoning provided.')
        rows += f"""
        <tr>
            <td><code style="color: #888;">{sid[:8]}</code></td>
            <td><span class="badge badge-task">{res['task_id'].upper()}</span></td>
            <td><span style="color:{color}; font-weight:bold;">{res['grade']}</span></td>
            <td>
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{res['score']*100}%; background:{color};"></div></div>
                <div style="font-size:10px; margin-top:4px;">{res['score']*100:.1f}%</div>
            </td>
            <td class="reasoning-cell">"{thought}"</td>
            <td>{res['steps']}</td>
            <td style="color:#55ff55;">{res['total_reward']}</td>
            <td><span class="status-tag status-graded">GRADED</span></td>
        </tr>
        """
    
    for sid, env in sessions.items():
        if sid in graded_results: continue
        last_thought = env.history[-1]["action"].get("thought", "Calculating next move...") if env.history else "Initializing episode..."
        max_r = estimate_max_reward(env.task_id)
        min_r = estimate_min_reward(env.task_id)
        range_r = max_r - min_r
        normalized_active_reward = max(0.01, min(1.0, (env.total_reward - min_r) / range_r)) if range_r > 0 else (1.0 if env.total_reward >= 0 else 0.0)
        
        rows += f"""
        <tr class="active-row" onclick="window.location='/trace/{sid}'" style="cursor:pointer;">
            <td><code style="color: #00d9ff;">{sid[:8]}</code></td>
            <td><span class="badge badge-task">{env.task_id.upper()}</span></td>
            <td>-</td>
            <td>-</td>
            <td class="reasoning-cell" style="color: #00d9ff; font-style: italic;">"{last_thought}"</td>
            <td>{env.step_count}</td>
            <td style="color:#00d9ff;">{round(normalized_active_reward, 3)}</td>
            <td><span class="status-tag status-active">ACTIVE</span></td>
        </tr>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>OpenEnv | Agent Strategy Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            :root {{ --bg: #050505; --panel: #111111; --accent: #00ff88; --border: #222222; --text: #fff; --text-dim: #888; }}
            body {{ font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 40px; }}
            .container {{ max-width: 1200px; margin: auto; background: var(--panel); padding: 40px; border-radius: 24px; border: 1px solid var(--border); box-shadow: 0 20px 50px rgba(0,0,0,0.8); }}
            .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 40px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
            h1 {{ margin: 0; font-size: 32px; background: linear-gradient(90deg, #fff, #888); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
            .stat-card {{ background: #161616; padding: 20px; border-radius: 16px; border: 1px solid var(--border); transition: transform 0.2s; }}
            .stat-card:hover {{ transform: translateY(-5px); border-color: var(--accent); }}
            .stat-val {{ font-size: 32px; font-weight: 600; color: var(--accent); font-family: 'JetBrains Mono'; }}
            .stat-label {{ font-size: 13px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }}
            table {{ width: 100%; border-collapse: separate; border-spacing: 0 10px; }}
            th {{ text-align: left; color: var(--text-dim); font-size: 12px; text-transform: uppercase; padding: 15px; }}
            td {{ padding: 20px 15px; background: #161616; vertical-align: middle; }}
            td:first-child {{ border-radius: 12px 0 0 12px; }}
            td:last-child {{ border-radius: 0 12px 12px 0; }}
            .badge {{ padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; background: #333; }}
            .status-tag {{ padding: 6px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; }}
            .status-graded {{ background: #00ff8815; color: #00ff88; border: 1px solid #00ff8833; }}
            .status-active {{ background: #00d9ff15; color: #00d9ff; border: 1px solid #00d9ff33; }}
            .reasoning-cell {{ max-width: 300px; font-size: 13px; color: #ccc; border-left: 2px solid #333; padding-left: 20px !important; }}
            .score-bar-bg {{ width: 60px; height: 6px; background: #222; border-radius: 3px; overflow: hidden; }}
            .score-bar-fill {{ height: 100%; }}
            .active-row td {{ background: #1a1a1a88; }}
        </style>
        <meta http-equiv="refresh" content="3">
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>OpenEnv Triage Analytics</h1>
                <div style="font-size: 12px; color: var(--text-dim);">Live Status: <span style="color: var(--accent);">STABLE</span></div>
            </div>
            <div class="stats-grid">
                <div class="stat-card"><div class="stat-label">System Sessions</div><div class="stat-val">{len(sessions)}</div></div>
                <div class="stat-card"><div class="stat-label">Graded Episodes</div><div class="stat-val">{len(graded_results)}</div></div>
                <div class="stat-card">
                    <div class="stat-label">Mean Performance</div>
                    <div class="stat-val">{sum(r['score'] for r in graded_results.values())/max(1, len(graded_results))*100:.1f}%</div>
                </div>
            </div>
            <table>
                <thead><tr><th>ID</th><th>Task</th><th>Grade</th><th>Score</th><th>Reasoning Highlight</th><th>Steps</th><th>Reward</th><th>Status</th></tr></thead>
                <tbody>{rows if rows else '<tr><td colspan="8" style="text-align:center; padding:80px; color:#444;">Waiting for agent interactions...</td></tr>'}</tbody>
            </table>
        </div>
    </body>
    </html>
    """

def render_trace(session_id, env):
    state = env.state()["observation"]
    cart_html = "".join([f'<div class="item-card"><div class="item-id">{i["id"]}</div><div class="item-price">${i["price"]}</div><div class="item-margin" style="color: {"#00ff88" if i.get("profit_margin", 5) > 5 else "#ff4444"}">Margin: {i.get("profit_margin", 0)}</div></div>' for i in state["cart"]]) or "<div style='color:#444;'>Cart is empty.</div>"
    orders_html = "".join([f'<div class="queue-item"><div style="display:flex; justify-content:space-between; margin-bottom: 8px;"><span>Order #{idx+1}: {o["id"][:8]}</span><span style="color: {"#ff4444" if o.get("SLA_breach_risk") else "#00d9ff"}; font-weight:bold; font-size: 10px;">{"URGENT" if o.get("SLA_breach_risk") else "Standard"}</span></div><div class="progress-bg"><div class="progress-fill" style="width: {"90%" if o.get("SLA_breach_risk") else "30%"}; background:{"#ff4444" if o.get("SLA_breach_risk") else "#00d9ff"};"></div></div></div>' for idx, o in enumerate(state["orders"])]) or "<div style='color:#444;'>No pending orders.</div>"
    returns_html = "".join([f'<div class="queue-item"><div style="display:flex; justify-content:space-between; align-items:center;"><span>Return: {r["id"][:8]}</span><span class="badge" style="background: {"#00ff88" if r.get("policy_compliant") else "#ffcc00"}22; color: {"#00ff88" if r.get("policy_compliant") else "#ffcc00"}; border: 1px solid {"#00ff88" if r.get("policy_compliant") else "#ffcc00"}44;">{"COMPLIANT" if r.get("policy_compliant") else "INVESTIGATE"}</span></div></div>' for r in state["returns"]]) or "<div style='color:#444;'>No pending returns.</div>"
    history_html = "".join([f'<div class="history-step"><div style="display:flex; justify-content:space-between; margin-bottom: 5px;"><span style="color:var(--accent);">Step {s["step"]}</span><span style="font-family:monospace;">{s["action"]["type"]}({s["action"].get("index", 0)})</span></div><div style="font-size: 12px; color: #aaa; font-style: italic;">"{s["action"].get("thought", "No reasoning.")}"</div></div>' for s in reversed(env.history)])

    return f"""
    <!DOCTYPE html><html><head><title>Triage Trace | {session_id[:8]}</title><link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root {{ --bg: #050505; --panel: #111111; --accent: #00ff88; --border: #222222; --text: #fff; }}
        body {{ font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 30px; }}
        .grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 30px; max-width: 1400px; margin: auto; }}
        .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 25px; }}
        h2 {{ font-size: 18px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 0; }}
        .cart-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 15px; }}
        .item-card {{ background: #161616; padding: 15px; border-radius: 12px; border: 1px solid var(--border); }}
        .item-id {{ font-weight: bold; font-size: 14px; margin-bottom: 5px; color: var(--accent); }}
        .item-price {{ font-family: 'JetBrains Mono'; font-size: 18px; }}
        .queue-item {{ background: #161616; padding: 15px; border-radius: 12px; margin-bottom: 10px; border: 1px solid var(--border); }}
        .progress-bg {{ background: #222; height: 4px; border-radius: 2px; overflow: hidden; }}
        .progress-fill {{ height: 100%; }}
        .history-step {{ border-bottom: 1px solid #222; padding: 15px 0; }}
        .nav-back {{ margin-bottom: 20px; display: inline-block; color: #888; text-decoration: none; }}
    </style><meta http-equiv="refresh" content="2"></head>
    <body><a href="/dashboard" class="nav-back">← Back to Dashboard</a><div class="grid"><div class="main-col"><div class="panel" style="margin-bottom: 30px;"><h2>Current Cart (Limit: ${state['corporate_credit_limit']})</h2><div style="font-size: 24px; margin-bottom: 20px; font-weight: 600;">Total: <span style="color: {'#ff4444' if sum(i['price'] for i in state['cart']) > state['corporate_credit_limit'] else '#00ff88'}">${sum(i['price'] for i in state['cart'])}</span></div><div class="cart-grid">{cart_html}</div></div><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;"><div class="panel"><h2>Fulfillment Queue</h2>{orders_html}</div><div class="panel"><h2>Returns Processing</h2>{returns_html}</div></div></div><div class="side-col"><div class="panel"><h2>Agent Reasoning Log</h2><div style="max-height: 80vh; overflow-y: auto;">{history_html}</div></div></div></div></body></html>
    """


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
