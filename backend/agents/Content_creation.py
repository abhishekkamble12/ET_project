"""
Content Creation Agent – generates platform-specific social media posts.

Incorporates:
  - RAG knowledge context (state["knowledge_context"])
  - Strategy signals (state["strategy"])
  - Memory context (state["memory_context"]) for hashtag deduplication
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

try:
    from backend.services.image_generation import generate_image
except ImportError:
    from services.image_generation import generate_image

if TYPE_CHECKING:
    from models.state import PipelineState

logger = logging.getLogger(__name__)


# ── Output schema ────────────────────────────────────────────────

class ContentCreationOutput(BaseModel):
    caption: str = Field(description="Caption for the social media post")
    image_prompt: str = Field(description="Prompt used to generate the post image")
    hashtags: list[str] = Field(description="List of hashtags for the post")
    platform: str = Field(description="Target platform (linkedin / instagram)")


# ── LLM ─────────────────────────────────────────────────────────

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0.7,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )
    return _llm


parser = PydanticOutputParser(pydantic_object=ContentCreationOutput)


# ── Agent class ──────────────────────────────────────────────────

class ContentCreationAgent:

    def create_social_post(
        self,
        topic: str,
        platform: str,
        knowledge_context: str | None = None,
        strategy: dict | None = None,
        memory_context: dict | None = None,
    ) -> ContentCreationOutput:
        platform_lower = platform.lower()

        # Platform-specific base instructions
        if platform_lower == "linkedin":
            base = (
                f"Create a professional LinkedIn post about: {topic}. "
                "Use no more than 5 hashtags. Keep a formal, insightful tone."
            )
        else:
            base = (
                f"Create an engaging Instagram post about: {topic}. "
                "Use 10 to 30 hashtags. Keep a conversational, vibrant tone."
            )

        # Inject RAG knowledge context
        knowledge_section = ""
        if knowledge_context:
            knowledge_section = (
                f"\n\nRelevant background knowledge (use this to enrich the post):\n{knowledge_context}"
            )

        # Inject strategy signals
        strategy_section = ""
        if strategy:
            use_more = strategy.get("use_more", {})
            avoid = strategy.get("avoid", {})
            parts = []
            if use_more.get("hashtags"):
                parts.append(f"Prefer these high-performing hashtags: {use_more['hashtags']}")
            if use_more.get("tones"):
                parts.append(f"Preferred tones: {use_more['tones']}")
            if avoid.get("hashtags"):
                parts.append(f"Avoid these low-performing hashtags: {avoid['hashtags']}")
            if avoid.get("tones"):
                parts.append(f"Avoid these tones: {avoid['tones']}")
            if parts:
                strategy_section = "\n\nContent strategy signals:\n" + "\n".join(f"- {p}" for p in parts)

        # Inject memory-based hashtag deduplication
        memory_section = ""
        if memory_context:
            prior_posts = memory_context.get(f"{platform_lower}_posts", [])
            if prior_posts:
                recent = prior_posts[-3:]
                excluded = []
                for post in recent:
                    excluded.extend(post.get("hashtags", []))
                if excluded:
                    memory_section = f"\n\nAvoid repeating these recently used hashtags: {excluded}"

        full_prompt = (
            base
            + knowledge_section
            + strategy_section
            + memory_section
            + f"\n\n{parser.get_format_instructions()}"
        )

        llm = _get_llm()
        response = llm.invoke(full_prompt)
        output: ContentCreationOutput = parser.parse(response.content)

        # Fire-and-forget image generation
        try:
            generate_image(output.image_prompt)
        except Exception as exc:
            logger.warning("Image generation skipped: %s", exc)

        return ContentCreationOutput(
            caption=output.caption,
            image_prompt=output.image_prompt,
            hashtags=output.hashtags,
            platform=platform,
        )


# ── LangGraph node ───────────────────────────────────────────────

def content_node(state: "PipelineState") -> "PipelineState":
    """
    LangGraph node: generate social media content.
    Reads knowledge_context, strategy, and memory_context from state.
    """
    agent = ContentCreationAgent()
    output = agent.create_social_post(
        topic=state["query"],
        platform=state["platform"],
        knowledge_context=state.get("knowledge_context"),
        strategy=state.get("strategy"),
        memory_context=state.get("memory_context"),
    )
    return {**state, "generated_content": output}
