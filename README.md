# EngageTech AI Platform

A comprehensive AI-driven content generation and orchestration platform. EngageTech AI automates the social media content creation process using an 8-agent pipeline, featuring human-in-the-loop approval, enterprise RAG for custom knowledge bases, and real-time streaming updates.

## 🏗 System Architecture

The project consists of two main applications seamlessly integrated to provide a premium user experience:
1. **Frontend**: Next.js 15, React 18, Tailwind CSS, Framer Motion. (Located in `et-frontend-/`)
2. **Backend**: FastAPI, LangChain, LangGraph, Qdrant/Supabase, AWS Bedrock. (Located in `back/backend/`)

---

## 💻 Frontend Deep Dive (`et-frontend-`)

The frontend is a modern, high-performance web application designed with a **"Technical Luxury"** aesthetic. 

### Tech Stack
- **Framework**: Next.js 15 (Leveraging the new App Router structure).
- **Styling**: Tailwind CSS configured with a highly polished design system.
- **Animations**: Framer Motion for smooth, dynamic micro-interactions.
- **Visuals**: React Flow and Three.js (for immersive backgrounds and interactive node graph representations).

### How It Works
- The frontend connects to the backend API layer and initiates generation sequences.
- Next.js acts as the host, rendering interactive dashboards, analytics, and content approval modals.
- It leverages SSE (Server-Sent Events) to display the live pipeline process, showing output in real-time as each AI agent completes its task, displaying partial completions natively to the user.

---

## ⚙️ Backend Deep Dive (`back/backend`)

The backend is the core intelligence engine of EngageTech AI, built on **FastAPI** and **LangGraph**, providing robust endpoints for orchestration, language model communication, and enterprise document retrieval.

### Core Tech Stack
- **Framework**: FastAPI (Handles API routing, CORS, metrics middleware, SSE streaming, and async tasks).
- **Orchestration Framework**: LangGraph (State graph engine orchestrating the multi-agent AI workflow).
- **LLM Integrations**: LangChain Core & Community with integrations for AWS Bedrock, Groq, and HuggingFace.
- **RAG & Vector Database**: Qdrant Client & Supabase pgvector used to store document embeddings.
- **Data Models**: Pydantic (Strict typing, validations, and schema definitions).
- **Concurrency & Servers**: Uvicorn (ASGI web server) and asyncio for concurrent node evaluations.

### The 8-Agent LangGraph Pipeline

The content generation process is driven by a highly structured State Graph Pipeline (`agents/Supervisor.py`), operating dynamically on a shared `PipelineState` object. 

1. **Knowledge Agent** (`knowledge_agent.py`):
   Retrieves contextual information from the enterprise knowledge base (Qdrant Vector DB / Supabase) to heavily augment the query providing deep context.
2. **Strategy Agent** (`strategy_agent.py`):
   Determines the overarching content strategy, tone of voice, format, and structure targeting the specific platform (e.g., LinkedIn vs. Instagram).
3. **Content Generation Agent** (`Content_creation.py`):
   Drafts the initial semantic content, captions, and hashtags by heavily relying on the pre-determined strategy.
4. **Compliance Agent** (`compliance_agent.py`):
   Checks if the drafted content adheres strictly to safety policies, brand guidelines, and legal rules.
   - *Routing Loop*: If the content is rejected, the pipeline loops and attempts to fix the violation, or ends if unrecoverable.
5. **Engagement Analysis Agent** (`services/Engagement.py`):
   Analyzes the projected engagement score of the output post.
   - *Optimization Loop*: If the score falls below the `ENGAGEMENT_THRESHOLD` (e.g., `< 0.65`), LangGraph routes it back to the Content Generation Agent to improve the draft (up to a mapped maximum of 2 retries).
6. **Localization Agent** (`localization_agent.py`):
   Translates or adapts the approved text to the given target locale while maintaining tone.
7. **Formatter Agent** (`formatter_agent.py`):
   Performs final character counting, UI/UX structural formatting, and ensures metadata boundaries are flawlessly prepared for the exact destination platform.
8. **Human Review Agent** (`human_review_node`):
   Acts as the decisive breakpoint in the pipeline. It halts the state graph process until a human physically inputs an approval (`publish`), edits (`edit`), or rejects (`no`) the drafted copy.

### Understanding Control Flow & Pipeline Mechanics

- **Triggering the Pipeline**: The Next.js frontend calls the API endpoints defined precisely in `routers/content.py`.
- **RAG Implementation**: Documents are chunked, transformed into embeddings (using `sentence-transformers`), and continuously synced into Qdrant. The `enterprise_router.py` exposes endpoints for document upload, listing, and deletion.
- **Graph State Management**: LangGraph manages the flow state object seamlessly. When human intervention is required, execution pauses. As soon as the user submits an edit via `/approve`, `graph.update_state()` injects the decision and correctly resumes execution.
- **Real-Time Streaming**: As the nodes transition, FastAPI taps into `sse-starlette` to push state lifecycle progress chronologically down to the browser.
- **Checkpointers**: The workflow executes via `run_pipeline(query, platform)`, storing states constantly to a LangGraph Checkpointer (`MemorySaver` or an AWS Custom Core Saver), rendering graph memory perfectly persistent across API hits.

---

## 🚀 Getting Started

### Local Development Setup

**1. Running the Backend Server:**
```bash
cd back/backend
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
*(Important: Verify `.env` values are correctly pointing to Qdrant, Supabase, and AWS).*

**2. Running the Frontend Server:**
```bash
cd et-frontend-
npm install
npm run dev
```
The Frontend will be exclusively running on http://localhost:3000.
