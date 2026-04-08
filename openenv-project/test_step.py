import httpx
import json

def test_actions():
    base_url = "http://localhost:8000"
    
    # 1. Reset
    print("Testing /reset...")
    resp = httpx.get(f"{base_url}/reset", params={"task_id": "easy"})
    data = resp.json()
    session_id = data["session_id"]
    print(f"Session ID: {session_id}")
    print(f"Initial Cart Size: {len(data['observation']['cart'])}")
    
    # 2. Add Item
    print("\nTesting /step with type='add_item'...")
    resp = httpx.post(f"{base_url}/step", json={
        "session_id": session_id,
        "type": "add_item",
        "index": 0
    })
    data = resp.json()
    print(f"New Cart Size (after add): {len(data['observation']['cart'])}")
    
    # 3. Remove Item
    print("\nTesting /step with type='remove_item'...")
    resp = httpx.post(f"{base_url}/step", json={
        "session_id": session_id,
        "type": "remove_item",
        "index": 0
    })
    data = resp.json()
    print(f"New Cart Size (after remove): {len(data['observation']['cart'])}")

    # 4. Wait
    print("\nTesting /step with type='wait'...")
    initial_time = data['observation']['time_remaining']
    resp = httpx.post(f"{base_url}/step", json={
        "session_id": session_id,
        "type": "wait"
    })
    data = resp.json()
    print(f"Time Remaining (after wait): {data['observation']['time_remaining']} (was {initial_time})")

if __name__ == "__main__":
    try:
        test_actions()
    except Exception as e:
        print(f"Error: {e}")
