import pytest
from env import EcomEnv

def test_env_reset():
    env = EcomEnv()
    state = env.reset("easy")
    assert "observation" in state
    assert state["done"] is False
    assert state["total_reward"] == 0.0

def test_env_step_remove_item():
    env = EcomEnv()
    env.reset("easy")
    initial_cart_len = len(env.cart)
    
    # Take action: remove_item (valid index)
    result = env.step({"type": "remove_item", "index": 0})
    assert len(env.cart) == initial_cart_len - 1
    assert env.step_count == 1
    assert env.time_remaining == 9 # easy has 10 steps

def test_env_step_invalid_action():
    env = EcomEnv()
    env.reset("easy")
    
    # Take invalid action
    # Penalty: -2.0 (invalid) + -6.0 (over budget) + -0.5 (time tax) = -8.5
    result = env.step({"type": "invalid_type", "index": 0})
    assert result["reward"] == -8.5 
    assert env.step_count == 1

def test_env_done():
    env = EcomEnv()
    env.reset("easy")
    env.time_remaining = 1
    
    # Last step
    result = env.step({"type": "wait", "index": 0})
    assert result["done"] is True
    assert env.time_remaining == 0
