import json
import os
import copy
from .tasks import get_task


# Load product catalog once at module level
_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
with open(_CATALOG_PATH, "r") as f:
    _CATALOG_DATA = json.load(f)

_PRODUCTS_BY_ID = {p["id"]: p for p in _CATALOG_DATA["products"]}

VALID_ACTIONS = {"remove_item", "process_order", "handle_return", "wait", "add_item", "terminate"}


class EcomEnv:
    def __init__(self):
        self.reset()

    def reset(self, task_id: str = "easy") -> dict:
        """Initialize a new episode from a task configuration."""
        task = get_task(task_id)

        self.task_id = task_id
        self.step_count = 0
        self.done = False
        self.last_reward = 0.5  # Neutral starting point
        self.total_reward = 0.0
        self.raw_total_reward = 0.0 # Internal tracking for normalization

        self.corporate_credit_limit = task["corporate_credit_limit"]
        self.max_steps = task["max_steps"]
        self.time_remaining = self.max_steps

        # Build cart from catalog product IDs
        self.cart = []
        for pid in task["cart_product_ids"]:
            if pid in _PRODUCTS_BY_ID:
                self.cart.append(copy.deepcopy(_PRODUCTS_BY_ID[pid]))

        self.orders = copy.deepcopy(task["orders"])
        self.returns = copy.deepcopy(task["returns"])

        # Episode history for grading
        self.history = []

        return self.state()

    def state(self) -> dict:
        """Return the current observation snapshot."""
        return {
            "observation": {
                "cart": self.cart,
                "orders": self.orders,
                "returns": self.returns,
                "corporate_credit_limit": self.corporate_credit_limit,
                "time_remaining": self.time_remaining,
                "step": self.step_count,
            },
            "reward": float(max(0.0, min(1.0, self.last_reward))),
            "total_reward": float(max(0.0, min(1.0, self.total_reward))),
            "done": self.done,
        }

    def step(self, action: dict) -> dict:
        """Process one action and advance the environment by one step."""
        if self.done:
            return self.state()

        self.step_count += 1
        reward = 0.0
        action_type = action.get("type", "")
        index = action.get("index", 0)

        # --- Action validation ---
        if action_type not in VALID_ACTIONS:
            reward -= 0.2  # penalty for invalid action

        # --- Action: remove_item ---
        elif action_type == "remove_item":
            if self.cart:
                idx = max(0, min(index, len(self.cart) - 1))
                item = self.cart.pop(idx)

                if item.get("profit_margin", 5) > 5:
                    reward -= 0.3  # bad removal — core margin item
                else:
                    reward += 0.5  # good removal — low-margin add-on
            self.time_remaining -= 1

        # --- Action: process_order ---
        elif action_type == "process_order":
            if self.orders:
                idx = max(0, min(index, len(self.orders) - 1))
                order = self.orders.pop(idx)

                if order.get("SLA_breach_risk", False):
                    reward += 0.6  # high-priority SLA order
                else:
                    reward += 0.2  # standard order
            self.time_remaining -= 1

        # --- Action: handle_return ---
        elif action_type == "handle_return":
            if self.returns:
                idx = max(0, min(index, len(self.returns) - 1))
                ret = self.returns.pop(idx)

                if ret.get("policy_compliant", False):
                    reward += 0.4  # legitimate return approved
                else:
                    reward -= 0.5  # fraudulent return penalized
            self.time_remaining -= 2  # returns cost more time

        # --- Action: wait ---
        elif action_type == "wait":
            self.time_remaining -= 1
            reward -= 0.01  # small penalty for waiting

        # --- Action: add_item ---
        elif action_type == "add_item":
            # Add a random item or specific item from catalog if index provides ID
            # For simplicity, we'll use index to pick from _CATALOG_DATA
            catalog_products = _CATALOG_DATA["products"]
            idx = max(0, min(index, len(catalog_products) - 1))
            new_item = copy.deepcopy(catalog_products[idx])
            self.cart.append(new_item)
            reward += 0.1  # small bonus for adding
            self.time_remaining -= 1

        # --- Action: terminate ---
        elif action_type == "terminate":
            self.done = True
            reward += 0.5  # efficiency bonus for proactive termination

        # --- Global modifiers (applied every step) ---

        # Budget check
        total_price = sum(item["price"] for item in self.cart)
        if total_price > self.corporate_credit_limit:
            reward -= 0.6  # over corporate limit
        else:
            reward += 0.2  # under limit bonus

        # Time tax — every step has a small cost
        reward -= 0.05

        # --- Termination check ---
        if (
            self.done
            or self.step_count >= self.max_steps
            or self.time_remaining <= 0
            or (len(self.cart) == 0 and len(self.orders) == 0 and len(self.returns) == 0)
        ):
            self.done = True

        # --- Normalize and Log Step ---
        from .grader import estimate_max_reward, estimate_min_reward
        
        max_r = estimate_max_reward(self.task_id)
        min_r = estimate_min_reward(self.task_id)
        
        self.raw_total_reward += reward
        
        # Min-Max Normalization to [0, 1] range
        range_r = max_r - min_r
        if range_r > 0:
            self.total_reward = (self.raw_total_reward - min_r) / range_r
        else:
            self.total_reward = 1.0 if self.raw_total_reward >= 0 else 0.0
        
        # Shifted linear mapping for step reward: raw [-1, 1] -> [0, 1]
        self.last_reward = (reward + 1.0) / 2.0
        self.total_reward = max(0.0, min(1.0, self.total_reward))
        self.last_reward = max(0.0, min(1.0, self.last_reward))

        self.history.append({
            "step": self.step_count,
            "action": action,
            "reward": self.last_reward,
            "total_reward": self.total_reward,
            "done": self.done,
            "raw_reward": reward,
            "raw_total_reward": self.raw_total_reward
        })

        return self.state()