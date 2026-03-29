# EngageTech AI Backend Ecosystem

This is the backend component of the EngageTech AI Platform, built robustly using **FastAPI** and **LangGraph** to automate enterprise social media content creation.

## ⚙️ Core Architecture & Technologies

- **Framework**: FastAPI (powers endpoints, CORS, metric middlewares, Server-Sent Events).
- **AI Orchestration**: LangGraph (acts as a State Graph engine regulating the autonomous flow between AI Agents).
- **Core Integrations**: LangChain (connecting to models hosted on AWS Bedrock or APIs like Groq/OpenAI).
- **Vector Database (RAG)**: Qdrant Client / Supabase pgvector for contextual Enterprise Document Retrieval.
- **Python Stack**: Uvicorn server, Pydantic for validation, and Pytest / Asyncio toolings.

## 🤖 The 8-Agent LangGraph Pipeline

The content flow acts fundamentally as a State Machine inside `agents/Supervisor.py`. Running on a `PipelineState` dictionary:

1. **Knowledge Agent** (`knowledge_agent.py`): Connects to Qdrant/Supabase RAG Vector DB to give deep contextual backing to user prompts.
2. **Strategy Agent** (`strategy_agent.py`): Sets the tone, structural format, and direction.
3. **Content Generation Agent** (`Content_creation.py`): Formulates the raw post and metadata. 
4. **Compliance Agent** (`compliance_agent.py`): Validates brand safety. *Loops back on failure.*
5. **Engagement Analysis Agent** (`services/Engagement.py`): Scores content impact. *Loops back dynamically (max 2 tries) if the score is under a 0.65 threshold.*
6. **Localization Agent** (`localization_agent.py`): Translates data based on locale parameters.
7. **Formatter Agent** (`formatter_agent.py`): Fits data inside target platform boundaries (like LinkedIn char limits).
8. **Human Review Agent** (`human_review_node`): Pauses LangGraph execution perfectly in time. Allows the user on the Next.js frontend to trigger an endpoint, injecting an override or an approval via state updates.

## 💡 How It All Works Together

- **Routing & Streaming**: Requests hit `routers/content.py` which triggers `run_pipeline(query, platform)`. The active state yields chronological updates via `sse-starlette` down the wire.
- **Memory Checkpointing**: Utilizing `MemorySaver` (or an AWS custom saver pipeline), all conversation logs and state updates persist globally across the session allowing seamless re-connections.
- **Enterprise DB**: Use the endpoints in `routers/enterprise.py` to continuously embed PDFs or Docs into the RAG environment using locally processed Sentence Transformers.

## 🛠 To Run the Backend
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
