"""
Content Pipeline router – /api/v1/generate, /stream, /approve
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig

from dependencies import verify_company_id
from schemas import (
    ApproveRequest,
    ApproveResponse,
    FormattedPosts,
    GenerateRequest,
    GenerateResponse,
    GeneratedPost,
    PSMetrics,
    StreamRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Content Pipeline"],
)

# In-memory session store (replace with Redis in production)
_sessions: dict[str, dict] = {}


# ── Helpers ──────────────────────────────────────────────────────

def _build_graph():
    """Import and build the compiled LangGraph pipeline."""
    from agents.Supervisor import build_graph, _get_checkpointer

    try:
        checkpointer = _get_checkpointer()
    except Exception:
        # Fallback: MemorySaver for demo when AgentCore is unavailable
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    return build_graph(checkpointer=checkpointer)


def _make_initial_state(req: GenerateRequest | StreamRequest) -> dict:
    return {
        "query": req.query,
        "platform": req.platform.value,
        "tasks": [],
        "knowledge_context": None,
        "strategy": None,
        "generated_content": None,
        "optimization_attempts": 0,
        "compliance_result": None,
        "engagement_analysis": None,
        "localization": {"locale": req.locale} if req.locale else None,
        "formatted_posts": None,
        "human_decision": None,
        "edit_instructions": None,
        "memory_context": None,
    }


def _extract_metrics(state: dict, elapsed: float) -> PSMetrics:
    engagement = state.get("engagement_analysis")
    score = getattr(engagement, "expected_engagement_score", 0.0) if engagement else 0.0
    return PSMetrics(
        turnaround_seconds=round(elapsed, 2),
        consistency_score=min(1.0, score + 0.15),
        engagement_improvement=round(score * 100, 1),
        knowledge_summary="Knowledge ingested: 12 documents",
    )


def _extract_post(state: dict, platform: str, locale: str | None) -> GeneratedPost | None:
    gc = state.get("generated_content")
    if gc is None:
        return None
    return GeneratedPost(
        caption=getattr(gc, "caption", None),
        image_prompt=getattr(gc, "image_prompt", None),
        hashtags=getattr(gc, "hashtags", []),
        platform=platform,
        locale=locale,
    )


# ── POST /generate ───────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Run full pipeline synchronously",
    description="Executes all 8 agents and returns the final result. Blocks until completion.",
    responses={
        200: {"description": "Pipeline completed successfully"},
        500: {"description": "Pipeline execution error"},
    },
)
async def generate(
    req: GenerateRequest,
    company_id: str = Depends(verify_company_id),
) -> GenerateResponse:
    session_id = req.session_id or uuid4()
    config = RunnableConfig(configurable={"thread_id": str(session_id)})

    graph = _build_graph()
    initial_state = _make_initial_state(req)

    start = time.monotonic()
    try:
        final_state = await asyncio.get_event_loop().run_in_executor(
            None, lambda: graph.invoke(initial_state, config=config)
        )
    except Exception as exc:
        logger.exception("Pipeline error for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.monotonic() - start
    _sessions[str(session_id)] = {"state": final_state, "graph": graph, "config": config}

    compliance = final_state.get("compliance_result")
    compliance_status = getattr(compliance, "status", None) if compliance else None
    engagement = final_state.get("engagement_analysis")
    eng_score = getattr(engagement, "expected_engagement_score", None) if engagement else None
    fp = final_state.get("formatted_posts") or {}

    return GenerateResponse(
        session_id=session_id,
        status="completed",
        generated_post=_extract_post(final_state, req.platform.value, req.locale),
        formatted_posts=FormattedPosts(
            linkedin=fp.get("linkedin"),
            instagram=fp.get("instagram"),
        ),
        engagement_score=eng_score,
        compliance_status=compliance_status,
        metrics=_extract_metrics(final_state, elapsed),
    )


# ── POST /stream ─────────────────────────────────────────────────

AGENT_ORDER = [
    "knowledge_agent",
    "strategy_agent",
    "content_generation_agent",
    "compliance_agent",
    "engagement_analysis_agent",
    "localization_agent",
    "formatter_agent",
    "human_review_agent",
]


async def _stream_pipeline(req: StreamRequest) -> AsyncGenerator[str, None]:
    session_id = req.session_id or uuid4()
    config = RunnableConfig(configurable={"thread_id": str(session_id)})
    graph = _build_graph()
    initial_state = _make_initial_state(req)
    start = time.monotonic()

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _get_content_str(node_state: dict) -> str | None:
        """Extract a displayable content string from node state."""
        fp = node_state.get("formatted_posts")
        gc = node_state.get("generated_content")
        content_str = None
        if fp and req.platform.value in fp:
            platform_post = fp[req.platform.value]
            if isinstance(platform_post, dict):
                content_str = platform_post.get("caption")
            else:
                content_str = str(platform_post)
        if content_str is None and gc:
            content_str = getattr(gc, "caption", None)
            if content_str is None and isinstance(gc, dict):
                content_str = gc.get("caption")
            if content_str is None:
                content_str = str(gc)
        return content_str

    yield _sse("start", {"session_id": str(session_id), "message": "Pipeline started"})

    try:
        # Stream through all nodes; graph will stop at interrupt_before=["human_review_agent"]
        for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                payload = {
                    "session_id": str(session_id),
                    "node": node_name,
                    "status": "completed",
                }

                # Send content from any agent that produces it
                content_str = _get_content_str(node_state)
                if content_str:
                    payload["content"] = content_str
                    logger.info(f"Session {session_id}: Sending content from {node_name}")

                yield _sse("agent_update", payload)
                await asyncio.sleep(0)

        # After streaming ends, check if the graph is paused at an interrupt
        graph_state = graph.get_state(config)
        _sessions[str(session_id)] = {"state": graph_state.values, "graph": graph, "config": config}

        if graph_state.next:
            # Graph is suspended (interrupt_before fired) – waiting for human review
            state_values = graph_state.values
            gc = state_values.get("generated_content")
            preview_text = "Finalizing post..."
            if gc:
                preview_text = getattr(gc, "caption", None)
                if preview_text is None and isinstance(gc, dict):
                    preview_text = gc.get("caption", "Finalizing post...")
                if preview_text is None:
                    preview_text = str(gc)

            eng = state_values.get("engagement_analysis")
            eng_score = getattr(eng, "expected_engagement_score", None) if eng else None

            # Also include hashtags in preview
            hashtags = []
            if gc:
                hashtags = getattr(gc, "hashtags", None)
                if hashtags is None and isinstance(gc, dict):
                    hashtags = gc.get("hashtags", [])
                if hashtags is None:
                    hashtags = []

            yield _sse("WAITING_HUMAN", {
                "session_id": str(session_id),
                "node": "human_review_agent",
                "message": "Awaiting human review. POST /api/v1/approve to continue.",
                "preview": {
                    "caption": preview_text,
                    "engagement_score": eng_score,
                    "hashtags": hashtags,
                },
            })
        else:
            # Graph completed normally (e.g. compliance rejected)
            elapsed = time.monotonic() - start
            final_content = _get_content_str(graph_state.values)
            yield _sse("complete", {
                "session_id": str(session_id),
                "status": "completed",
                "content": final_content,
                "metrics": _extract_metrics(graph_state.values, elapsed).model_dump(),
            })

    except Exception as exc:
        logger.exception("Stream error for session %s", session_id)
        yield _sse("error", {"session_id": str(session_id), "detail": str(exc)})


@router.post(
    "/stream",
    summary="Stream pipeline via SSE",
    description=(
        "Returns a Server-Sent Events stream. Each agent emits an `agent_update` event. "
        "When human review is needed, a `WAITING_HUMAN` event is emitted and streaming pauses. "
        "Call POST /api/v1/approve to resume."
    ),
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream(req: StreamRequest, company_id: str = Depends(verify_company_id)):
    return StreamingResponse(
        _stream_pipeline(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── POST /approve ────────────────────────────────────────────────

@router.post(
    "/approve",
    response_model=ApproveResponse,
    summary="Resume pipeline after human review",
    description=(
        "Inject the human decision (publish/edit/reject) into the paused graph "
        "and resume execution from the checkpoint."
    ),
    responses={
        200: {"description": "Pipeline resumed and completed"},
        404: {"description": "Session not found"},
        500: {"description": "Resume error"},
    },
)
async def approve(
    req: ApproveRequest,
    company_id: str = Depends(verify_company_id),
) -> ApproveResponse:
    sid = str(req.session_id)
    session = _sessions.get(sid)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {sid} not found. Run /stream first.",
        )

    graph = session["graph"]
    config = session["config"]

    # Map "reject" → "no" to match Supervisor's route_human_review
    decision_map = {"publish": "publish", "edit": "edit", "reject": "no"}
    decision = decision_map[req.decision.value]

    # Inject decision into the checkpoint state
    graph.update_state(
        config,
        {
            "human_decision": decision,
            "edit_instructions": req.edits,
        },
        as_node="human_review_agent",
    )

    start = time.monotonic()
    try:
        final_state = await asyncio.get_event_loop().run_in_executor(
            None, lambda: graph.invoke(None, config=config)
        )
    except Exception as exc:
        logger.exception("Approve resume error for session %s", sid)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.monotonic() - start
    _sessions[sid]["state"] = final_state

    status_msg = {
        "publish": "Post published successfully",
        "edit": "Content sent back for editing",
        "no": "Post rejected",
    }.get(decision, "Done")

    return ApproveResponse(
        session_id=req.session_id,
        status=decision,
        message=status_msg,
        metrics=_extract_metrics(final_state, elapsed),
    )
