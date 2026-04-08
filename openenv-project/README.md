<p align="center">
  <img src="logo.png" alt="OpenEnv Triage Agent Logo" width="250" />
</p>

# OpenEnv: E-Commerce Triage Agent

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenEnv Compliance](https://img.shields.io/badge/OpenEnv-Compliant-success.svg)](https://github.com/OpenEnv)

A mission-critical, REST-driven reinforcement learning environment designed for evaluating autonomous agents in **Enterprise Customer Support & Logistics Triage** scenarios. This project simulates the complex decision-making required to optimize global supply chains and customer satisfaction.

---

## The Enterprise Challenge

Modern e-commerce giants handle millions of automated tasks daily. When human intervention is traditional, bottlenecks occur. This environment trains AI systems to resolve these bottlenecks autonomously using advanced business logic:

*   ** Cart Optimization**: Analyze and resolve blocked checkouts where totals exceed corporate credit limits. AI agents must intelligently prune low-margin add-ons to ensure core fulfillment.
*   ** Fulfillment Triage**: Dynamically prioritize shipping queues to prevent Service Level Agreement (SLA) breaches and minimize late-delivery penalties.
*   ** Intelligent RMA Control**: Automate Return Merchandise Authorization (RMA) processing by differentiating between high-value policy compliance and potential fraud.

---

##  Quick Start

### 1. Installation
Deploy the environment using `uv` for high-performance dependency management or standard `pip`:

```bash
# Recommended: Using uv
uv sync

# Alternate: Using pip
pip install -e .
```

### 2. Launch the Backend
Start the FastAPI-powered triage server:

```bash
uvicorn server:app --reload
```
*The server will be available at `http://localhost:8000`.*

---

##  Environment API (The Triage Loop)

The environment manages interactions via persistent, stateful sessions.

### 1. Discovery
Retrieve the list of active triage queues and their difficulty markers.
`GET /tasks`

### 2. Session Initialization
Begin a new triage episode. Supported task levels: `easy`, `medium`, `hard`.
`GET /reset?task_id=medium`
> **Returns**: A unique `session_id` and the initial environment observation state.

### 3. Agent Actions
Agents submit decisions based on corporate profitability and operational efficiency.

```bash
curl -X POST http://localhost:8000/step \
    -H "Content-Type: application/json" \
    -d '{
        "session_id": "YOUR_SESSION_ID",
        "type": "modify_cart",
        "index": 0
    }'
```

| Action Type | Description | Cost (Units) |
| :--- | :--- | :--- |
| `modify_cart` | Remove low-margin items to fit corporate budget limits. | 1 |
| `fulfill_order` | Process pending fulfillment (Priority: SLA adherence). | 1 |
| `process_rma` | Audit and process return requests via policy compliance. | 2 |
| `escalate` | Transfer high-complexity tickets to human operators. | 1 |

### 4. Evaluation & Grading
Upon episode completion (`done: true`), retrieve the performance analytics.
`GET /grade?session_id=YOUR_SESSION_ID`

---

##  Intelligent Inference Engine

The repository includes a **Hybrid AI Agent** (`inference.py`) that leverages both deterministic heuristics and LLM-driven decision engines.

*   **Heuristic Mode**: High-speed, rule-based triage.
*   **LLM Mode**: Activated via `HF_TOKEN`. Uses `Llama-3` (default) for complex reasoning.

**Execution Commands:**
```bash
# Run a specific task
python inference.py --task medium

# Evaluate across all difficulty tiers
python inference.py --all
```

---

##  Configuration

| Environment Variable | Description | Default |
| :--- | :--- | :--- |
| `API_BASE_URL` | Endpoint for the Triage Server. | `http://localhost:8000` |
| `HF_TOKEN` | Hugging Face Token for LLM Inference. | *N/A* |
| `MODEL_NAME` | Target Model ID for RL decisions. | `meta-llama/Llama-3-8b-chat-hf` |

---

##  Quality Assurance
We maintain >90% coverage for deterministic result validation:
```bash
uv run pytest tests/
```

---
<p align="center">Built for the future of Autonomous Commerce.</p>