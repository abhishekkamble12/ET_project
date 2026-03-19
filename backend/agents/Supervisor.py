"""
Supervisor – LangGraph pipeline orchestrator.

Pipeline order:
  knowledge → strategy → content → compliance (loop until approved/rejected)
  → engagement → [optimization loop if score < 0.65, max 2 retries]
  → localization → formatter → human_review

Optimization loop:
  If engagement score < ENGAGEMENT_THRESHOLD and optimization_attempts < MAX_OPTIMIZATION_ATTEMPTS,
  route back to content_generation to regenerate with improvement hints injected into strategy.

Memory:
  AgentCoreMemorySaver (AWS Bedrock) used as LangGraph checkpointer.
  On "publish" decision the final state is persisted automatically.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from models.state import PipelineState

# Agent / service nodes
try:
    from backend.agents.knowledge_agent import knowledge_node
    from backend.agents.strategy_agent import strategy_node
    from backend.agents.Content_creation import content_node
    from backend.agents.compliance_agent import compliance_node, ComplianceResult
    from backend.agents.localization_agent import localization_node
    from backend.agents.formatter_agent import formatter_node
    from backend.services.Engagement import engagement_node, ENGAGEMENT_THRESHOLD
except ImportError:
    from agents.knowledge_agent import knowledge_node
    from agents.strategy_agent import strategy_node
    from agents.Content_creation import content_node
    from agents.compliance_agent import compliance_node, ComplianceResult
    from agents.localization_agent import localization_node
    from agents.formatter_agent import formatter_node
    from services.Engagement import engagement_node, ENGAGEMENT_THRESHOLD

# AgentCore memory
try:
    from amazon_agentcore.memory import AgentCoreMemorySaver
except ImportError:
    AgentCoreMemorySaver = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = {"linkedin", "instagram"}
MAX_OPTIMIZATION_ATTEMPTS = 2


# ── Routing functions ────────────────────────────────────────────

def route_compliance(state: PipelineState) -> str:
    """Route after compliance check."""
    result = state.get("compliance_result")
    if result is None:
        raise ValueError("compliance_result missing from state")

    if result.status == "approved":
        return "engagement_analysis_agent"
    elif result.status == "needs_fix":
        # compliance_node loops internally; if it exits with needs_fix
        # it means MAX_FIX_ATTEMPTS was exhausted – treat as rejected.
        logger.warning("Compliance exited with needs_fix after max attempts – rejecting.")
        return END
    else:  # rejected
        return END


def route_engagement(state: PipelineState) -> str:
    """
    Optimization loop: if score < threshold and retries remain,
    route back to content generation with improvement hints.
    """
    analysis = state.get("engagement_analysis")
    attempts = state.get("optimization_attempts", 0)

    if analysis is None:
        # Engagement failed – skip optimization, proceed to localization
        return "localization_agent"

    score = analysis.expected_engagement_score
    if score < ENGAGEMENT_THRESHOLD and attempts < MAX_OPTIMIZATION_ATTEMPTS:
        logger.info(
            "Engagement score %.2f < %.2f – optimization attempt %d/%d.",
            score, ENGAGEMENT_THRESHOLD, attempts + 1, MAX_OPTIMIZATION_ATTEMPTS,
        )
        return "content_generation_agent"

    return "localization_agent"


def route_human_review(state: PipelineState) -> str:
    """Route after human review decision."""
    decision = state.get("human_decision")
    if decision == "publish":
        return END
    elif decision == "edit":
        return "content_generation_agent"
    elif decision == "no":
        return END
    raise ValueError(f"Unknown human_decision: {decision!r}")


# ── Optimization-aware content node wrapper ──────────────────────

def content_node_with_counter(state: PipelineState) -> PipelineState:
    """
    Wraps content_node to:
      1. Increment optimization_attempts counter.
      2. Inject improvement hints from the previous engagement analysis
         into the strategy so the LLM knows what to fix.
    """
    attempts = state.get("optimization_attempts", 0)

    # Inject improvement hints into strategy before regenerating
    analysis = state.get("engagement_analysis")
    if analysis and analysis.improvements:
        strategy = dict(state.get("strategy") or {})
        existing_use_more = dict(strategy.get("use_more") or {})
        existing_use_more["improvements"] = analysis.improvements
        strategy["use_more"] = existing_use_more
        state = {**state, "strategy": strategy}

    new_state = content_node(state)
    return {**new_state, "optimization_attempts": attempts + 1}


# ── Human review node ────────────────────────────────────────────

def human_review_node(state: PipelineState) -> PipelineState:
    generated = state.get("generated_content")
    analysis = state.get("engagement_analysis")
    formatted = state.get("formatted_posts", {})

    print("\n" + "=" * 60)
    print("HUMAN REVIEW")
    print("=" * 60)

    if generated:
        print(f"Caption      : {generated.caption}")
        print(f"Image Prompt : {generated.image_prompt}")
        print(f"Hashtags     : {', '.join(generated.hashtags)}")

    if analysis:
        print(f"\nEngagement Score    : {analysis.expected_engagement_score:.2f}")
        print(f"Audience Reaction   : {analysis.predicted_audience_reaction}")
        print(f"Impact Summary      : {analysis.post_impact_summary}")
        if analysis.improvements:
            print(f"Improvements        : {'; '.join(analysis.improvements)}")

    if formatted:
        print("\nFormatted Posts:")
        for platform, post in formatted.items():
            print(f"  [{platform.upper()}] {post.get('char_count', 0)} chars, "
                  f"{len(post.get('hashtags', []))} hashtags")

    print("=" * 60)

    valid = {"publish", "edit", "no"}
    decision = ""
    while decision not in valid:
        decision = input("\nDecision (publish/edit/no): ").strip().lower()
        if decision not in valid:
            print(f"Please enter one of: {', '.join(sorted(valid))}")

    edit_instructions = None
    if decision == "edit":
        edit_instructions = input("Enter edit instructions: ").strip()

    return {**state, "human_decision": decision, "edit_instructions": edit_instructions}


# ── Graph builder ────────────────────────────────────────────────

def build_graph(checkpointer=None) -> Any:
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("knowledge_agent",          knowledge_node)
    graph.add_node("strategy_agent",           strategy_node)
    graph.add_node("content_generation_agent", content_node_with_counter)
    graph.add_node("compliance_agent",         compliance_node)
    graph.add_node("engagement_analysis_agent",engagement_node)
    graph.add_node("localization_agent",       localization_node)
    graph.add_node("formatter_agent",          formatter_node)
    graph.add_node("human_review_agent",       human_review_node)

    # Edges
    graph.add_edge(START,                       "knowledge_agent")
    graph.add_edge("knowledge_agent",           "strategy_agent")
    graph.add_edge("strategy_agent",            "content_generation_agent")
    graph.add_edge("content_generation_agent",  "compliance_agent")
    graph.add_conditional_edges("compliance_agent",          route_compliance)
    graph.add_conditional_edges("engagement_analysis_agent", route_engagement)
    graph.add_edge("localization_agent",        "formatter_agent")
    graph.add_edge("formatter_agent",           "human_review_agent")
    graph.add_conditional_edges("human_review_agent",        route_human_review)

    return graph.compile(checkpointer=checkpointer)


# ── Checkpointer factory ─────────────────────────────────────────

def _get_checkpointer():
    if AgentCoreMemorySaver is None:
        raise ImportError(
            "AgentCoreMemorySaver not available. Install the AWS AgentCore SDK."
        )
    memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
    if not memory_id:
        raise EnvironmentError("AGENTCORE_MEMORY_ID environment variable is not set.")
    return AgentCoreMemorySaver(
        memory_id=memory_id,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


# ── Public entry point ───────────────────────────────────────────

def run_pipeline(
    query: str,
    platform: str,
    locale: str | None = None,
    config: RunnableConfig | None = None,
) -> PipelineState:
    """
    Execute the full social media content pipeline.

    Args:
        query:    Topic / brief for the post.
        platform: Target platform ("linkedin" or "instagram").
        locale:   Optional ISO 639-1 locale code for localization (e.g. "es").
        config:   Optional LangGraph RunnableConfig (thread_id etc.).

    Returns:
        Final PipelineState after human review.
    """
    if platform.lower() not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Unsupported platform '{platform}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_PLATFORMS))}"
        )

    checkpointer = _get_checkpointer()
    compiled_graph = build_graph(checkpointer=checkpointer)

    initial_state: PipelineState = {
        "query":                query,
        "platform":             platform.lower(),
        "tasks":                [],
        "knowledge_context":    None,
        "strategy":             None,
        "generated_content":    None,
        "optimization_attempts": 0,
        "compliance_result":    None,
        "engagement_analysis":  None,
        "localization":         {"locale": locale} if locale else None,
        "formatted_posts":      None,
        "human_decision":       None,
        "edit_instructions":    None,
        "memory_context":       None,
    }

    result: PipelineState = compiled_graph.invoke(initial_state, config=config)

    if result.get("human_decision") == "publish":
        logger.info("Post published. State persisted to AgentCore memory via checkpointer.")

    return result


# ── Backward-compat helper (used by tests) ───────────────────────

def validate_required_field(state: PipelineState, field: str) -> None:
    """Raise ValueError if a required field is None in PipelineState."""
    if state.get(field) is None:
        raise ValueError(f"Required field '{field}' is missing from PipelineState")
