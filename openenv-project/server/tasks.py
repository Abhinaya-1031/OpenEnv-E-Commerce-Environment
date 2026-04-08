"""
Task registry for the OpenEnv e-commerce environment.
Each task defines a scenario with cart items, orders, returns, corporate_credit_limit, and step limits.
"""

TASKS = {
    "easy": {
        "description": "Triage Cart: Blocked checkout. Remove low-margin items from the queue to unblock.",
        "cart_product_ids": ["p1", "p2", "p3","p4","p5"],
        "orders": [],
        "returns": [],
        "corporate_credit_limit": 100,
        "max_steps": 10,
    },
    "medium": {
        "description": "Triage Fast: Optimize cart and fulfill one high-risk SLA order.",
        "cart_product_ids": ["p6", "p7","p8","p9","p10"],
        "orders": [
            {"SLA_breach_risk": True, "value": 100}
        ],
        "returns": [],
        "corporate_credit_limit": 100,
        "max_steps": 10,
    },
    "hard": {
        "description": "Full Triage: Optimize cart, fulfill SLA orders, and process RMA tickets (including fraudulent).",
        "cart_product_ids": ["p11", "p12","p13","p14","p15"],
        "orders": [
            {"SLA_breach_risk": True, "value": 100}
        ],
        "returns": [
            {"policy_compliant": True, "reason": "Defective product"},
            {"policy_compliant": False, "reason": "Changed mind after 90 days"}
        ],
        "corporate_credit_limit": 100,
        "max_steps": 12,
    },
}


def get_task(task_id: str) -> dict:
    """Get a task configuration by ID. Raises KeyError if not found."""
    if task_id not in TASKS:
        available = ", ".join(TASKS.keys())
        raise KeyError(f"Unknown task '{task_id}'. Available tasks: {available}")
    return TASKS[task_id]


def list_tasks() -> list[dict]:
    """Return a list of all available tasks with their IDs and descriptions."""
    return [
        {"task_id": tid, "description": task["description"]}
        for tid, task in TASKS.items()
    ]
