from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser

if TYPE_CHECKING:
    from backend.agents.Supervisor import PipelineState

logger = logging.getLogger(__name__)


class EngagementAnalysis(BaseModel):
    expected_engagement_score: float = Field(description="Expected engagement score between 0.0 and 1.0")
    predicted_audience_reaction: str = Field(description="Predicted audience reaction to the post")
    post_impact_summary: str = Field(description="Summary of the post's expected impact")


_llm = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="qwen/qwen3-32b", temperature=0, max_tokens=None, timeout=30, max_retries=2)
    return _llm


parser = PydanticOutputParser(pydantic_object=EngagementAnalysis)


def engagement_node(state: "PipelineState") -> "PipelineState":
    generated = state.get("generated_content")
    memory_context = state.get("memory_context")

    caption = generated.caption if generated else ""
    platform = state.get("platform", "")

    history_section = ""
    if memory_context:
        platform_lower = platform.lower()
        prior_posts = memory_context.get(f"{platform_lower}_posts", [])
        if prior_posts:
            recent = prior_posts[-3:]
            scores = [p.get("engagement_score") for p in recent if p.get("engagement_score") is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
                history_section = f"\nHistorical average engagement score for {platform}: {avg_score:.2f}"

    prompt = f"""Analyze the engagement potential of this social media post for {platform}:

Caption: {caption}
{history_section}

Predict the engagement metrics and return a JSON with:
- expected_engagement_score: float between 0.0 and 1.0
- predicted_audience_reaction: string describing how the audience will react
- post_impact_summary: string summarizing the post's expected impact

{parser.get_format_instructions()}"""

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        result = parser.parse(response.content)
        return {**state, "engagement_analysis": result}
    except Exception as e:
        logger.error(f"Engagement analysis failed: {e}")
        return {**state, "engagement_analysis": None}
