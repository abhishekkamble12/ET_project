"""
Engagement Analysis Service

Returns a numeric score (0–1) and a list of concrete improvement suggestions.
The score is used by the Supervisor's optimization loop:
  if score < 0.65 and optimization_attempts < 2 → regenerate content.

Also persists the engagement result to Supabase for future strategy signals.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser

if TYPE_CHECKING:
    from models.state import PipelineState

logger = logging.getLogger(__name__)

ENGAGEMENT_THRESHOLD = 0.65
ENGAGEMENT_TABLE = "engagement_logs"


# ── Schema ───────────────────────────────────────────────────────

class EngagementAnalysis(BaseModel):
    expected_engagement_score: float = Field(
        description="Expected engagement score between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
    )
    predicted_audience_reaction: str = Field(
        description="Predicted audience reaction to the post"
    )
    post_impact_summary: str = Field(
        description="Summary of the post's expected impact"
    )
    improvements: list[str] = Field(
        default_factory=list,
        description="Concrete suggestions to improve engagement if score is low",
    )


# ── LLM ─────────────────────────────────────────────────────────

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0,
            max_tokens=None,
            timeout=30,
            max_retries=2,
        )
    return _llm


parser = PydanticOutputParser(pydantic_object=EngagementAnalysis)


# ── Supabase persistence (best-effort) ───────────────────────────

def _persist_engagement(platform: str, state: "PipelineState", result: EngagementAnalysis) -> None:
    """Write engagement result to Supabase for future strategy lookups."""
    try:
        from services.supabase_client import supabase

        generated = state.get("generated_content")
        hashtags = generated.hashtags if generated else []
        topic = state.get("query", "")

        supabase.table(ENGAGEMENT_TABLE).insert({
            "platform": platform,
            "hashtags": hashtags,
            "topic": topic,
            "score": result.expected_engagement_score,
        }).execute()
    except Exception as exc:
        logger.warning("Could not persist engagement log: %s", exc)


# ── LangGraph node ───────────────────────────────────────────────

def engagement_node(state: "PipelineState") -> "PipelineState":
    """
    LangGraph node: analyse engagement potential of the current post.

    Stores EngagementAnalysis in state["engagement_analysis"].
    The Supervisor reads state["engagement_analysis"].expected_engagement_score
    to decide whether to trigger the optimization loop.
    """
    generated = state.get("generated_content")
    platform = state.get("platform", "unknown")
    caption = generated.caption if generated else ""
    hashtags = generated.hashtags if generated else []

    # Historical context from memory
    history_section = ""
    memory_context = state.get("memory_context")
    if memory_context:
        prior_posts = memory_context.get(f"{platform.lower()}_posts", [])
        if prior_posts:
            scores = [p.get("engagement_score") for p in prior_posts[-5:] if p.get("engagement_score") is not None]
            if scores:
                avg = sum(scores) / len(scores)
                history_section = f"\nHistorical average engagement score for {platform}: {avg:.2f}"

    prompt = f"""Analyse the engagement potential of this {platform} post.

Caption: {caption}
Hashtags: {hashtags}
{history_section}

Return JSON with:
- expected_engagement_score: float 0.0–1.0
- predicted_audience_reaction: string
- post_impact_summary: string
- improvements: list of concrete suggestions to raise the score (empty list if score >= {ENGAGEMENT_THRESHOLD})

{parser.get_format_instructions()}"""

    try:
        response = _get_llm().invoke(prompt)
        result: EngagementAnalysis = parser.parse(response.content)
        logger.info(
            "Engagement score for %s: %.2f (threshold %.2f)",
            platform, result.expected_engagement_score, ENGAGEMENT_THRESHOLD,
        )
        _persist_engagement(platform, state, result)
        return {**state, "engagement_analysis": result}
    except Exception as exc:
        logger.error("Engagement analysis failed: %s", exc)
        return {**state, "engagement_analysis": None}
