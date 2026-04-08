"""
Grader — Evaluates an agent's episode performance and returns a normalized score.
"""

from .tasks import get_task
from .env import _PRODUCTS_BY_ID


# Theoretical max rewards per action type (best case scenarios)
_ACTION_MAX_REWARDS = {
    "modify_cart_good": 0.5,    # removing low-margin addon
    "modify_cart_bad": -0.3,    # removing core margin item (unavoidable sometimes)
    "fulfill_order_high": 0.6,  # processing SLA breach risk order
    "fulfill_order_low": 0.2,   # processing standard order
    "process_rma_valid": 0.4,   # handling a policy compliant return
    "process_rma_invalid": -0.5,  # handling fraudulent return
    "credit_limit_bonus": 0.2,    # per-step bonus for staying under corporate credit limit
    "time_tax": -0.05,           # per-step cost
}

GRADE_THRESHOLDS = [
    (0.9, "A"),
    (0.75, "B"),
    (0.6, "C"),
    (0.4, "D"),
    (0.0, "F"),
]


def estimate_max_reward(task_id: str) -> float:
    """
    Estimate the theoretical maximum reward for a task.
    This gives us a ceiling to normalize against.
    """
    task = get_task(task_id)

    max_reward = 0.0
    steps_needed = 0

    # Best case: remove only low-margin items (+5 each)
    # Worst case for high-margin: -3 each (but still must be done)
    for pid in task["cart_product_ids"]:
        product = _PRODUCTS_BY_ID.get(pid, {})
        margin = product.get("profit_margin", 5)
        if margin > 5:
            max_reward += _ACTION_MAX_REWARDS["modify_cart_bad"]
        else:
            max_reward += _ACTION_MAX_REWARDS["modify_cart_good"]
        steps_needed += 1

    # Orders
    for order in task["orders"]:
        if order.get("SLA_breach_risk", False):
            max_reward += _ACTION_MAX_REWARDS["fulfill_order_high"]
        else:
            max_reward += _ACTION_MAX_REWARDS["fulfill_order_low"]
        steps_needed += 1

    # Returns — only count valid ones as positive
    for ret in task["returns"]:
        if ret.get("policy_compliant", False):
            max_reward += _ACTION_MAX_REWARDS["process_rma_valid"]
        else:
            max_reward += _ACTION_MAX_REWARDS["process_rma_invalid"]
        steps_needed += 1

    # Add budget bonus for each step (best case: always under budget)
    max_reward += _ACTION_MAX_REWARDS["credit_limit_bonus"] * steps_needed

    # Add time tax for each step
    max_reward += _ACTION_MAX_REWARDS["time_tax"] * steps_needed

    return max_reward


def estimate_min_reward(task_id: str) -> float:
    """
    Estimate the theoretical minimum reward for a task (absolute catastrophic performance).
    """
    task = get_task(task_id)
    max_steps = task["max_steps"]
    
    # Worst case:
    # 1. Invalid actions at every step (-0.2 each in env.py)
    # 2. Staying over budget at every step (-0.6 each)
    # 3. Time tax at every step (-0.05 each)
    # 4. Processing all fraudulent returns (-0.5 each)
    # 5. Removing high-profit items (-0.3 each)
    
    min_reward = 0.0
    
    # 1-3. Step penalties
    min_reward += (-0.2 - 0.6 - 0.05) * max_steps
    
    # 4. Returns
    for ret in task["returns"]:
        if not ret.get("policy_compliant", False):
            min_reward += -0.5
            
    # 5. Cart removals
    for pid in task["cart_product_ids"]:
        product = _PRODUCTS_BY_ID.get(pid, {})
        if product.get("profit_margin", 5) > 5:
            min_reward += -0.3
            
    return min_reward


def _score_to_grade(score: float) -> str:
    """Convert a 0.0–1.0 score to a letter grade."""
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def grade(history: list[dict], task_id: str) -> dict:
    """
    Grade an agent's episode performance and return strictly positive, normalized values.
    """
    if not history:
        return {
            "task_id": task_id,
            "score": 0.0,
            "total_reward": 0.0,
            "max_possible_reward": 1.0,
            "steps_used": 0,
            "max_steps": 0,
            "grade": "F",
            "passed": False,
            "breakdown": {
                "reward_score": 0.0,
                "efficiency_bonus": 0.0,
                "penalties": 0.0,
            },
        }

    task = get_task(task_id)
    max_steps = task["max_steps"]
    actual_reward = history[-1].get("raw_total_reward", history[-1]["total_reward"])
    steps_used = len(history)
    
    max_reward = estimate_max_reward(task_id)
    min_reward = estimate_min_reward(task_id)
    
    # --- Min-Max Normalization (ensures 0.0 to 1.0 range) ---
    # Formula: (actual - min) / (max - min)
    range_val = max_reward - min_reward
    if range_val > 0:
        normalized_reward = max(0.0, min(1.0, (actual_reward - min_reward) / range_val))
    else:
        normalized_reward = 1.0 if actual_reward >= 0 else 0.0

    # --- Efficiency bonus (up to +0.05, added within 1.0 limit) ---
    efficiency_bonus = 0.0
    if steps_used < max_steps:
        efficiency_bonus = 0.05 * (1.0 - steps_used / max_steps)

    # --- Penalty deductions (0.0 to 1.0) ---
    # We now normalize the penalty count so it doesn't take score below zero
    penalty_count = 0
    for entry in history:
        action = entry.get("action", {})
        action_type = action.get("type", "")
        # Penalize invalid actions
        if action_type not in {"remove_item", "process_order", "handle_return", "wait", "add_item", "terminate"}:
            penalty_count += 1
    
    # Scale penalty count relative to max steps
    penalty_deduction = min(0.2, (penalty_count / max_steps) * 0.5)

    # --- Final composite score (strictly positive) ---
    # We start with normalized_reward, add bonus, subtract penalties, then clip
    score = max(0.01, min(1.0, normalized_reward + efficiency_bonus - penalty_deduction))
    letter_grade = _score_to_grade(score)

    return {
        "task_id": task_id,
        "score": round(score, 3),
        "total_reward": round(normalized_reward, 3), # Displaying normalized reward as requested
        "max_possible_reward": 1.0, # Now normalized
        "steps_used": steps_used,
        "max_steps": max_steps,
        "grade": letter_grade,
        "passed": score >= 0.5,
        "breakdown": {
            "reward_score": round(normalized_reward, 3),
            "efficiency_bonus": round(efficiency_bonus, 3),
            "penalties": round(penalty_deduction, 3),
        },
    }