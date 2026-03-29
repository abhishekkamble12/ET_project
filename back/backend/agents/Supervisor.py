"""
Supervisor – LangGraph pipeline orchestrator.

Pipeline order:
  knowledge → strategy → content → compliance (loop until approved/rejected)
  → engagement → [optimization loop if score < 0.65, max 2 retries]
  → localization → formatter → human_review
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from models.state import PipelineState

from agents.knowledge_agent import knowledge_node
from agents.strategy_agent import strategy_node
from agents.Content_creation import content_node
from agents.compliance_agent import compliance_node, ComplianceResult
from agents.localization_agent import localization_node
from agents.formatter_agent import formatter_node
from services.Engagement import engagement_node, ENGAGEMENT_THRESHOLD

try:
    from amazon_agentcore.memory import AgentCoreMemorySaver
except ImportError:
    AgentCoreMemorySaver = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = {"linkedin", "instagram"}
MAX_OPTIMIZATION_ATTEMPTS = 2


# ── Routing functions ────────────────────────────────────────────

def route_compliance(state: PipelineState) -> str:
    result = state.get("compliance_result")
    if result is None:
        raise ValueError("compliance_result missing from state")
    if result.status == "approved":
        return "engagement_analysis_agent"
    else:
        logger.warning("Compliance rejected/needs_fix – ending pipeline.")
        return END


def route_engagement(state: PipelineState) -> str:
    analysis = state.get("engagement_analysis")
    attempts = state.get("optimization_attempts", 0)
    if analysis is None:
        return "localization_agent"
    score = analysis.expected_engagement_score
    if score < ENGAGEMENT_THRESHOLD and attempts < MAX_OPTIMIZATION_ATTEMPTS:
        logger.info("Engagement %.2f < %.2f – retry %d/%d.", score, ENGAGEMENT_THRESHOLD, attempts + 1, MAX_OPTIMIZATION_ATTEMPTS)
        return "content_generation_agent"
    return "localization_agent"


def route_human_review(state: PipelineState) -> str:
    decision = state.get("human_decision")
    if decision == "publish":
        return END
    elif decision == "edit":
        return "content_generation_agent"
    elif decision == "no":
        return END
    raise ValueError(f"Unknown human_decision: {decision!r}")


# ── Content node with optimization counter ───────────────────────

def content_node_with_counter(state: PipelineState) -> PipelineState:
    attempts = state.get("optimization_attempts", 0)
    analysis = state.get("engagement_analysis")
    if analysis and analysis.improvements:
        strategy = dict(state.get("strategy") or {})
        use_more = dict(strategy.get("use_more") or {})
        use_more["improvements"] = analysis.improvements
        strategy["use_more"] = use_more
        state = {**state, "strategy": strategy}
    new_state = content_node(state)
    return {**new_state, "optimization_attempts": attempts + 1}


# ── Human review node ────────────────────────────────────────────

def human_review_node(state: PipelineState) -> PipelineState:
    """
    In API/SSE context the stream router intercepts BEFORE this node runs
    and pauses execution. The decision is then injected via /approve which
    calls graph.update_state() and resumes — at that point human_decision
    is already set so we just pass through.

    In non-interactive server context (stdin not a tty), auto-publish to
    prevent blocking the server process.
    """
    # Decision already injected by /approve endpoint — pass through
    if state.get("human_decision") is not None:
        return state

    # Interactive CLI fallback
    generated = state.get("generated_content")
    analysis = state.get("engagement_analysis")
    formatted = state.get("formatted_posts", {})

    print("\n" + "=" * 60)
    print("HUMAN REVIEW")
    print("=" * 60)
    if generated:
        print(f"Caption      : {generated.caption}")
        print(f"Hashtags     : {', '.join(generated.hashtags)}")
    if analysis:
        print(f"Engagement   : {analysis.expected_engagement_score:.2f}")
    if formatted:
        for platform, post in formatted.items():
            print(f"  [{platform.upper()}] {post.get('char_count', 0)} chars")
    print("=" * 60)

    valid = {"publish", "edit", "no"}
    decision = ""
    while decision not in valid:
        decision = input("\nDecision (publish/edit/no): ").strip().lower()

    edit_instructions = None
    if decision == "edit":
        edit_instructions = input("Edit instructions: ").strip()

    return {**state, "human_decision": decision, "edit_instructions": edit_instructions}


# ── Graph builder ────────────────────────────────────────────────

def build_graph(checkpointer=None) -> Any:
    graph = StateGraph(PipelineState)

    graph.add_node("knowledge_agent",           knowledge_node)
    graph.add_node("strategy_agent",            strategy_node)
    graph.add_node("content_generation_agent",  content_node_with_counter)
    graph.add_node("compliance_agent",          compliance_node)
    graph.add_node("engagement_analysis_agent", engagement_node)
    graph.add_node("localization_agent",        localization_node)
    graph.add_node("formatter_agent",           formatter_node)
    graph.add_node("human_review_agent",        human_review_node)

    graph.add_edge(START,                        "knowledge_agent")
    graph.add_edge("knowledge_agent",            "strategy_agent")
    graph.add_edge("strategy_agent",             "content_generation_agent")
    graph.add_edge("content_generation_agent",   "compliance_agent")
    graph.add_conditional_edges("compliance_agent",           route_compliance)
    graph.add_conditional_edges("engagement_analysis_agent",  route_engagement)
    graph.add_edge("localization_agent",         "formatter_agent")
    graph.add_edge("formatter_agent",            "human_review_agent")
    graph.add_conditional_edges("human_review_agent",         route_human_review)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_review_agent"])


# ── Checkpointer factory ─────────────────────────────────────────

def _get_checkpointer():
    if AgentCoreMemorySaver is None:
        from langgraph.checkpoint.memory import MemorySaver
        logger.info("Using MemorySaver (AgentCore unavailable)")
        return MemorySaver()
    memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
    if not memory_id:
        from langgraph.checkpoint.memory import MemorySaver
        logger.warning("AGENTCORE_MEMORY_ID missing; falling back to MemorySaver")
        return MemorySaver()
    return AgentCoreMemorySaver(
        memory_id=memory_id,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


# ── Public entry point ───────────────────────────────────────────

def run_pipeline(query: str, platform: str, locale: str | None = None, config: RunnableConfig | None = None) -> PipelineState:
    if platform.lower() not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported platform '{platform}'. Supported: {', '.join(sorted(SUPPORTED_PLATFORMS))}")
    checkpointer = _get_checkpointer()
    compiled_graph = build_graph(checkpointer=checkpointer)
    initial_state: PipelineState = {
        "query": query, "platform": platform.lower(), "tasks": [],
        "knowledge_context": None, "strategy": None, "generated_content": None,
        "optimization_attempts": 0, "compliance_result": None, "engagement_analysis": None,
        "localization": {"locale": locale} if locale else None, "formatted_posts": None,
        "human_decision": None, "edit_instructions": None, "memory_context": None,
    }
    return compiled_graph.invoke(initial_state, config=config)


def validate_required_field(state: PipelineState, field: str) -> None:
    if state.get(field) is None:
        raise ValueError(f"Required field '{field}' is missing from PipelineState")
