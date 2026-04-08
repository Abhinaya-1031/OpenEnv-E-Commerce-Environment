from fastapi.testclient import TestClient
from server import app
import pytest

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_get_tasks():
    response = client.get("/tasks")
    assert response.status_code == 200
    assert "tasks" in response.json()
    assert len(response.json()["tasks"]) >= 1

def test_reset_and_step_endpoint():
    # Reset
    response = client.get("/reset", params={"task_id": "easy"})
    assert response.status_code == 200
    data = response.json()
    session_id = data["session_id"]
    
    # Step
    response = client.post("/step", json={
        "session_id": session_id,
        "type": "remove_item",
        "index": 0
    })
    assert response.status_code == 200
    assert response.json()["session_id"] == session_id
    assert "reward" in response.json()

def test_state_endpoint():
    # Reset
    response = client.get("/reset", params={"task_id": "easy"})
    session_id = response.json()["session_id"]
    
    # Get state
    response = client.get("/state", params={"session_id": session_id})
    assert response.status_code == 200
    assert response.json()["session_id"] == session_id
    assert "observation" in response.json()

def test_grade_endpoint_failure_if_not_done():
    # Reset
    response = client.get("/reset", params={"task_id": "easy"})
    session_id = response.json()["session_id"]
    
    # Grade (Should fail because not done)
    response = client.get("/grade", params={"session_id": session_id})
    assert response.status_code == 400
    assert "detail" in response.json()
