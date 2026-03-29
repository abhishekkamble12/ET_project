"""
Multi-Platform Formatter Agent

Takes the final approved content and formats it for each supported platform,
storing the results in state["formatted_posts"].

Output structure:
    state["formatted_posts"] = {
        "linkedin": {
            "caption": str,
            "hashtags": list[str],   # max 5
            "image_prompt": str,
            "char_count": int,
        },
        "instagram": {
            "caption": str,
            "hashtags": list[str],   # 10–30
            "image_prompt": str,
            "char_count": int,
        },
    }
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_groq import ChatGroq

if TYPE_CHECKING:
    from models.state import PipelineState

logger = logging.getLogger(__name__)

LINKEDIN_MAX_HASHTAGS = 5
INSTAGRAM_MIN_HASHTAGS = 10
INSTAGRAM_MAX_HASHTAGS = 30
LINKEDIN_CHAR_LIMIT = 3000
INSTAGRAM_CHAR_LIMIT = 2200

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    return _llm


def _adapt_for_platform(caption: str, hashtags: list[str], platform: str) -> dict:
    """Use the LLM to reformat caption and hashtags for the target platform."""
    if platform == "linkedin":
        instructions = (
            f"Reformat the following social media caption for LinkedIn. "
            f"Keep it professional and concise (max {LINKEDIN_CHAR_LIMIT} chars). "
            f"Select the {LINKEDIN_MAX_HASHTAGS} most relevant hashtags from the list provided. "
            "Return only the reformatted caption followed by the hashtags on a new line, "
            "each hashtag prefixed with #."
        )
        max_tags = LINKEDIN_MAX_HASHTAGS
    else:  # instagram
        instructions = (
            f"Reformat the following social media caption for Instagram. "
            f"Keep it engaging and conversational (max {INSTAGRAM_CHAR_LIMIT} chars). "
            f"Include between {INSTAGRAM_MIN_HASHTAGS} and {INSTAGRAM_MAX_HASHTAGS} hashtags. "
            "Return only the reformatted caption followed by the hashtags on a new line, "
            "each hashtag prefixed with #."
        )
        max_tags = INSTAGRAM_MAX_HASHTAGS

    prompt = (
        f"{instructions}\n\n"
        f"Original caption:\n{caption}\n\n"
        f"Available hashtags: {hashtags}"
    )

    try:
        response = _get_llm().invoke(prompt)
        text = response.content.strip()

        # Split caption body from hashtag line(s)
        lines = text.split("\n")
        tag_lines = [l for l in lines if l.strip().startswith("#")]
        body_lines = [l for l in lines if not l.strip().startswith("#")]

        adapted_caption = "\n".join(body_lines).strip()
        adapted_tags = []
        for line in tag_lines:
            adapted_tags.extend(t.strip() for t in line.split() if t.startswith("#"))

        # Enforce limits
        adapted_tags = adapted_tags[:max_tags]

        return {
            "caption": adapted_caption,
            "hashtags": adapted_tags,
            "char_count": len(adapted_caption),
        }
    except Exception as exc:
        logger.error("Formatter failed for %s: %s", platform, exc)
        # Fallback: truncate and cap hashtags
        return {
            "caption": caption[:LINKEDIN_CHAR_LIMIT if platform == "linkedin" else INSTAGRAM_CHAR_LIMIT],
            "hashtags": hashtags[:max_tags],
            "char_count": len(caption),
        }


def formatter_node(state: "PipelineState") -> "PipelineState":
    """
    LangGraph node: produce platform-specific formatted versions of the post.
    """
    generated = state.get("generated_content")
    if not generated:
        logger.warning("formatter_node: no generated_content in state.")
        return {**state, "formatted_posts": {}}

    caption = generated.caption
    hashtags = generated.hashtags
    image_prompt = generated.image_prompt

    formatted: dict = {}
    for platform in ("linkedin", "instagram"):
        adapted = _adapt_for_platform(caption, hashtags, platform)
        formatted[platform] = {**adapted, "image_prompt": image_prompt}
        logger.info(
            "Formatted for %s: %d chars, %d hashtags.",
            platform, adapted["char_count"], len(adapted["hashtags"]),
        )

    return {**state, "formatted_posts": formatted}
