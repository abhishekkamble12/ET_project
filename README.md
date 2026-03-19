## Social Media Multi‑Agent System

This project is a backend prototype for a **social media content pipeline** built with LangGraph / LangChain. It orchestrates multiple agents to generate, validate, and analyze social posts, with support for **LinkedIn** and **Instagram** and optional long‑term memory via AWS AgentCore.

### High‑Level Architecture

- **Supervisor graph (`backend/agents/Supervisor.py`)**
  - Defines the `PipelineState` and a LangGraph `StateGraph`.
  - Orchestrates four nodes:
    - **Content generation** (`content_node` → `ContentCreationAgent`)
    - **Compliance** (`compliance_node_wrapper` → `compliance_node`)
    - **Engagement analysis** (`engagement_node_wrapper` → `engagement_node`)
    - **Human review** (`human_review_node`, interactive CLI step)
  - Uses `_get_checkpointer()` to wire in an `AgentCoreMemorySaver` so successful “publish” flows can be persisted.

- **Content creation agent (`backend/agents/Content_creation.py`)**
  - Uses `ChatGroq` (`qwen/qwen3-32b`) and a Pydantic model `ContentCreationOutput`.
  - Generates a caption, image prompt, hashtags, and platform field.
  - Enforces platform‑specific hashtag rules (≤ 5 for LinkedIn, 10–30 for Instagram).
  - Optionally consults `memory_context` to avoid reusing recent hashtags.
  - Calls `backend/services/image_generation.py` to trigger image generation via AWS Bedrock (Stable Diffusion XL).

- **Compliance agent (`backend/agents/compliance_agent.py`)**
  - Uses `ChatGroq` with a structured `ComplianceResult` output (status, reason, corrected_text).
  - Ensures content follows safety and truthfulness constraints.

- **Engagement analysis service (`backend/services/Engagement.py`)**
  - Uses `ChatGroq` and `EngagementAnalysis` Pydantic schema.
  - Predicts expected engagement score \([0.0, 1.0]\), audience reaction, and impact summary.
  - Optionally conditions on historical engagement from `memory_context`.

- **Image generation service (`backend/services/image_generation.py`)**
  - Uses AWS Bedrock (`stability.stable-diffusion-xl-v1`) to generate images and returns base64‑encoded image data.

- **Tests (`backend/tests/*.py`)**
  - `test_pipeline.py` – unit tests for routing, image generation integration, AgentCore checkpointer, and publish path.
  - `test_routing.py` – property‑based tests for routing and platform validation using Hypothesis.
  - `test_content_creation.py` – property‑based tests for content completeness, hashtag rules, and hashtag novelty.
  - `test_engagement.py` – property‑based tests for `EngagementAnalysis` structural invariants.
  - `test_compliance.py` – property‑based tests for `ComplianceResult` invariants.

Design documents for this feature live under `.kiro/specs/social-media-multi-agent-system/`.

### Tech Stack

- Python backend
- LangChain, LangGraph, LangGraph AWS checkpointing
- Groq `ChatGroq` LLM (Qwen models)
- AWS Bedrock (image generation)
- Optional AWS AgentCore for memory (`amazon_agentcore`)
- Testing: `pytest`, `hypothesis`

> Note: `backend/requirements.txt` also includes a FastAPI / SQLAlchemy stack, but the current code in this repo focuses on the multi‑agent pipeline and tests; an HTTP API layer can be added on top later.

### Installation

From the project root (`d:\ET_project`):

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install -r requirements.txt
```

### Environment Configuration

Set the following environment variables (tests use dummy values, but real runs need valid ones):

- **`GROQ_API_KEY`** – API key for Groq.
- **`AGENTCORE_MEMORY_ID`** – memory ID used by `AgentCoreMemorySaver`.
- **AWS credentials** – for Bedrock image generation (`boto3` will pick up standard AWS env vars or config/credentials files).
- Optional: **`AWS_REGION`** (defaults to `us-east-1` in `_get_checkpointer()`).

Example (PowerShell):

```powershell
$env:GROQ_API_KEY = "your-groq-key"
$env:AGENTCORE_MEMORY_ID = "your-memory-id"
$env:AWS_REGION = "us-east-1"
```

### Running the Test Suite

From `backend/`:

```bash
.venv\Scripts\activate  # if not already activated
pytest
```

This runs both unit tests and property‑based tests for the pipeline and its agents.

### Running the Pipeline Manually

You can invoke the pipeline directly in a Python REPL or a small script:

```python
from agents.Supervisor import run_pipeline

state = run_pipeline(
    query="AI in healthcare",
    platform="linkedin",  # or "instagram"
)

print(state["generated_content"])
print(state["compliance_result"])
print(state["engagement_analysis"])
```

During execution, the `human_review_node` will prompt in the console for a decision (`publish` / `edit` / `no`) and optional edit instructions. When the final decision is `publish`, the compiled graph writes state to the configured AgentCore memory via the checkpointer.

### Extending the Project

- **Add new platforms**: update `SUPPORTED_PLATFORMS` in `Supervisor.py` and extend prompt logic in `ContentCreationAgent`.
- **Integrate an API**: build a FastAPI app that wraps `run_pipeline` and exposes endpoints for creating posts, previewing content, and inspecting engagement predictions.
- **Richer memory**: plug in more detailed history into `memory_context` (e.g., per‑campaign stats) and adjust prompts in `Engagement` and `Content_creation`.

