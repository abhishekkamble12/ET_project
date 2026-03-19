from __future__ import annotations

import logging
import os
from typing import Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

# Import agent modules with fallback for different run contexts
try:
    from backend.agents.compliance_agent import ComplianceResult, compliance_node
    from backend.agents.Content_creation import ContentCreationAgent, ContentCreationOutput
    from backend.services.Engagement import engagement_node, EngagementAnalysis
except ImportError:
    from agents.compliance_agent import ComplianceResult, compliance_node
    from agents.Content_creation import ContentCreationAgent, ContentCreationOutput
    from services.Engagement import engagement_node, EngagementAnalysis

# Import AgentCoreMemorySaver
try:
    from amazon_agentcore.memory import AgentCoreMemorySaver
except ImportError:
    # Fallback for environments where AgentCore SDK is not installed
    AgentCoreMemorySaver = None

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = {"linkedin", "instagram"}


# ============================================================
# PipelineState TypedDict
# ============================================================

class PipelineState(TypedDict):
    query: str
    platform: str
    tasks: list[str]
    generated_content: Optional[Any]
    compliance_result: Optional[Any]
    engagement_analysis: Optional[Any]
    human_decision: Optional[str]
    edit_instructions: Optional[str]
    memory_context: Optional[dict]


# ============================================================
# Helpers
# ============================================================

def validate_required_field(state: PipelineState, field: str) -> None:
    """Raise ValueError if a required field is missing (None) from PipelineState."""
    if state.get(field) is None:
        raise ValueError(f"Required field '{field}' is missing from PipelineState")


# ============================================================
# Node wrappers with required-field guards (8.6)
# ============================================================

def content_node(state: PipelineState) -> PipelineState:
    validate_required_field(state, "query")
    validate_required_field(state, "platform")
    agent = ContentCreationAgent()
    output = agent.create_social_post(
        topic=state["query"],
        platform=state["platform"],
        memory_context=state.get("memory_context"),
    )
    return {**state, "generated_content": output}


def compliance_node_wrapper(state: PipelineState) -> PipelineState:
    validate_required_field(state, "generated_content")
    return compliance_node(state)


def engagement_node_wrapper(state: PipelineState) -> PipelineState:
    validate_required_field(state, "compliance_result")
    return engagement_node(state)


def human_review_node(state: PipelineState) -> PipelineState:
    generated = state.get("generated_content")
    engagement = state.get("engagement_analysis")

    print("\n" + "=" * 60)
    print("HUMAN REVIEW")
    print("=" * 60)
    if generated:
        print(f"Caption: {generated.caption}")
        print(f"Image Prompt: {generated.image_prompt}")
        print(f"Hashtags: {', '.join(generated.hashtags)}")
    if engagement:
        print(f"\nEngagement Score: {engagement.expected_engagement_score:.2f}")
        print(f"Audience Reaction: {engagement.predicted_audience_reaction}")
        print(f"Impact Summary: {engagement.post_impact_summary}")
    else:
        print("\nEngagement analysis: not available")
    print("=" * 60)

    valid_decisions = {"publish", "edit", "no"}
    decision = ""
    while decision not in valid_decisions:
        decision = input("\nDecision (publish/edit/no): ").strip().lower()
        if decision not in valid_decisions:
            print(f"Invalid choice. Please enter one of: {', '.join(sorted(valid_decisions))}")

    edit_instructions = None
    if decision == "edit":
        edit_instructions = input("Enter edit instructions: ").strip()

    return {**state, "human_decision": decision, "edit_instructions": edit_instructions}


# ============================================================
# Routing functions (8.1)
# ============================================================

def route_compliance(state: PipelineState) -> str:
    compliance_result = state.get("compliance_result")
    if compliance_result is None:
        raise ValueError("Required field 'compliance_result' is missing from PipelineState")
    status = compliance_result.status
    if status == "approved":
        return "engagement_analysis_agent"
    elif status == "needs_fix":
        return "compliance_agent"
    else:  # rejected
        return END


def route_human_review(state: PipelineState) -> str:
    decision = state.get("human_decision")
    if decision == "publish":
        return END
    elif decision == "edit":
        return "content_generation_agent"
    elif decision == "no":
        return END
    else:
        raise ValueError(f"Unknown human_decision value: {decision!r}. Expected one of: publish, edit, no")


# ============================================================
# Graph builder (8.3, 8.4)
# ============================================================

def build_graph(checkpointer=None) -> Any:
    graph = StateGraph(PipelineState)
    graph.add_node("content_generation_agent", content_node)
    graph.add_node("compliance_agent", compliance_node_wrapper)
    graph.add_node("engagement_analysis_agent", engagement_node_wrapper)
    graph.add_node("human_review_agent", human_review_node)

    graph.add_edge(START, "content_generation_agent")
    graph.add_edge("content_generation_agent", "compliance_agent")
    graph.add_conditional_edges("compliance_agent", route_compliance)
    graph.add_edge("engagement_analysis_agent", "human_review_agent")
    graph.add_conditional_edges("human_review_agent", route_human_review)

    return graph.compile(checkpointer=checkpointer)


def _get_checkpointer():
    """Build AgentCoreMemorySaver from environment variable."""
    if AgentCoreMemorySaver is None:
        raise ImportError("AgentCoreMemorySaver is not available. Install the AWS AgentCore SDK.")
    memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
    if not memory_id:
        raise EnvironmentError("AGENTCORE_MEMORY_ID environment variable is not set")
    return AgentCoreMemorySaver(
        memory_id=memory_id,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


# ============================================================
# run_pipeline (8.2, 8.5)
# ============================================================

def run_pipeline(query: str, platform: str, config: RunnableConfig | None = None) -> PipelineState:
    # 8.2: Platform validation before any agent runs
    if platform.lower() not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Unsupported platform '{platform}'. Supported: {', '.join(sorted(SUPPORTED_PLATFORMS))}"
        )

    checkpointer = _get_checkpointer()
    compiled_graph = build_graph(checkpointer=checkpointer)

    initial_state: PipelineState = {
        "query": query,
        "platform": platform.lower(),
        "tasks": [],
        "generated_content": None,
        "compliance_result": None,
        "engagement_analysis": None,
        "human_decision": None,
        "edit_instructions": None,
        "memory_context": None,
    }

    result: PipelineState = compiled_graph.invoke(initial_state, config=config)

    # 8.5: Memory write on publish
    if result.get("human_decision") == "publish":
        logger.info("Post published. State persisted to AgentCore memory via checkpointer.")

    return result
