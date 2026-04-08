"""
Inference Script for OpenEnv Triage Agent
Strictly compliant with [START], [STEP], and [END] logging specifications.
"""

import asyncio
import os
import textwrap
import json
import httpx
import argparse
import sys
from typing import List, Optional, Dict, Any
from openai import OpenAI

# --- Configuration (from Environment Variables) ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "EMPTY"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Environment Server URL (Local or injected)
ENV_API_URL = os.getenv("ENV_API_URL") or "http://localhost:8000"
DEFAULT_TASK_ID = os.getenv("TASK_ID", "easy")
BENCHMARK_NAME = "openenv-triage-v1"

# LLM Parameters
TEMPERATURE = 0.7
MAX_TOKENS = 150

# --- Environment Client Bridge ---

class TriageStepResult:
    def __init__(self, observation: Dict[str, Any], reward: float, total_reward: float, done: bool, session_id: str):
        self.observation = observation
        self.reward = reward
        self.total_reward = total_reward
        self.done = done
        self.session_id = session_id

class TriageEnvClient:
    """Async bridge between the inference loop and the FastAPI environment server."""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=10.0)
        self.session_id = None

    async def reset(self, task_id: str) -> TriageStepResult:
        resp = await self.client.get(f"{self.base_url}/reset", params={"task_id": task_id})
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        return TriageStepResult(data["observation"], data.get("reward", 0.5), data.get("total_reward", 0.0), data["done"], self.session_id)

    async def step(self, action: Dict[str, Any]) -> TriageStepResult:
        resp = await self.client.post(f"{self.base_url}/step", json={
            "session_id": self.session_id,
            "type": action["type"],
            "index": action.get("index", 0),
            "thought": action.get("thought", "")
        })
        resp.raise_for_status()
        data = resp.json()
        return TriageStepResult(data["observation"], data["reward"], data["total_reward"], data["done"], self.session_id)

    async def close(self):
        if self.session_id:
            try:
                await self.client.delete(f"{self.base_url}/sessions/{self.session_id}")
            except Exception:
                pass
        await self.client.aclose()

# --- Hyper-Strict Logging Helpers (MANDATORY) ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float], total_reward: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str} total_reward={total_reward:.2f}", flush=True)

# --- LLM Reasoning & Prompting ---

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Strategic Triage AI for an enterprise e-commerce platform.
    Your PRIMARY GOAL is to keep the cart total UNDER the Corporate Credit Limit to unblock checkout.

    Strategic Priority:
    1. BUDGET: If Cart Total > Corporate Credit Limit, you MUST use 'remove_item'. 
       Prefer items with profit_margin <= 5, but remove ANY item needed to fit the budget.
    2. SLA: Process orders, prioritizing those with SLA_breach_risk=true.
    3. RETURNS: Approve returns ONLY IF policy_compliant=true. Reject others.

    Available Actions:
    - remove_item (index): Remove items to fit within Corporate Credit Limit.
    - process_order (index): Process orders (Prioritize SLA risks).
    - handle_return (index): Process returns (Check compliance).
    - wait: Use only if all queues are blocked but you cannot yet terminate.
    - terminate: Use when all tasks are complete to end the session early and avoid the Time Tax.

    Reply with valid JSON ONLY.
    Include a "thought" field explaining why your choice is the best financial or strategic move.
    """
).strip()

def build_user_prompt(observation: Dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        Current Observation:
        {json.dumps(observation, indent=2)}

        Decide the next action. Ensure the budget limit is respected.
        """
    ).strip()

async def get_llm_action(client: OpenAI, observation: Dict[str, Any], heuristic_fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Get action from LLM with a safe fallback to our heuristic logic."""
    if not API_KEY or API_KEY == "EMPTY":
        return heuristic_fallback

    try:
        user_prompt = build_user_prompt(observation)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Strip markdown if hallucinated
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        
        decision = json.loads(text.strip())
        if "type" in decision:
            return {
                "type": decision["type"],
                "index": decision.get("index", 0),
                "thought": decision.get("thought", "Strategic decision based on current state.")
            }
    except Exception:
        pass
    
    return heuristic_fallback

def get_heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Solid logic to ensure we never get stuck, prioritizing Budget and SLA."""
    cart = observation.get("cart", [])
    orders = observation.get("orders", [])
    returns = observation.get("returns", [])
    limit = observation.get("corporate_credit_limit", 100)
    
    current_total = sum(item.get("price", 0) for item in cart)

    # 1. Critical Budget Check (Fixes the Fail loop)
    if current_total > limit and cart:
        # Sort to find the lowest-margin item to sacrifice
        worst_margin_idx = min(range(len(cart)), key=lambda i: cart[i].get("profit_margin", 10))
        item = cart[worst_margin_idx]
        return {
            "type": "remove_item", "index": worst_margin_idx, 
            "thought": f"Budget Alert: Cart total ({current_total}) exceeds limit ({limit}). Removing {item.get('id')} to unblock."
        }

    # 2. SLA Priority
    for i, order in enumerate(orders):
        if order.get("SLA_breach_risk", False):
            return {
                "type": "process_order", "index": i, 
                "thought": f"SLA at risk. Processing order immediately."
            }

    # 3. Valid Returns
    for i, ret in enumerate(returns):
        if ret.get("policy_compliant", False):
            return {
                "type": "handle_return", "index": i, 
                "thought": "Processing compliant return."
            }

    # 4. Standard Queue
    if orders: return {"type": "process_order", "index": 0, "thought": "Processing next order."}
    if returns: return {"type": "handle_return", "index": 0, "thought": "Reviewing next return ticket."}
    
    return {"type": "terminate", "index": 0, "thought": "System idle. All triage items processed. Terminating to maximize efficiency."}

# --- Main Execution Loop ---

async def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEnv Triage Agent")
    parser.add_argument(
        "--task", 
        default=DEFAULT_TASK_ID, 
        choices=["easy", "medium", "hard"], 
        help="Difficulty level."
    )
    parser.add_argument("--url", default=ENV_API_URL, help="Environment API URL")
    args = parser.parse_args()

    # Initialize Clients
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = TriageEnvClient(base_url=args.url)

    rewards: List[float] = []
    total_reward = 0.0
    steps_taken = 0
    success = False
    
    log_start(task=args.task, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        result = await env_client.reset(args.task)
        
        while not result.done:
            steps_taken += 1
            obs = result.observation
            
            fallback = get_heuristic_action(obs)
            action = await get_llm_action(openai_client, obs, fallback)
            
            result = await env_client.step(action)
            
            reward = result.reward
            rewards.append(reward)
            total_reward = result.total_reward
            
            print(f"# REASONING: {action.get('thought')}", flush=True)
            action_str = f"{action['type']}({action.get('index', 0)})"
            log_step(step=steps_taken, action=action_str, reward=reward, done=result.done, error=None)

            if result.done or steps_taken >= 20:
                break

        final_resp = await env_client.client.get(f"{args.url}/grade", params={"session_id": env_client.session_id})
        final_data = final_resp.json()
        success = final_data.get("passed", False)

    except Exception:
        log_end(success=False, steps=steps_taken, rewards=rewards, total_reward=total_reward)
        return
    finally:
        await env_client.close()

    log_end(success=success, steps=steps_taken, rewards=rewards, total_reward=total_reward)

if __name__ == "__main__":
    asyncio.run(main())
