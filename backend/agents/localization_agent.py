"""
Localization Agent – translates / adapts the post caption for a target locale.

If no locale is requested (state["localization"] is None or locale == "en"),
the node is a no-op and passes state through unchanged.

Expected state input:
    state["localization"] = {"locale": "es"}   # ISO 639-1 code

State output:
    state["localization"] = {
        "locale": "es",
        "localized_caption": "...",
    }
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_groq import ChatGroq

if TYPE_CHECKING:
    from models.state import PipelineState

logger = logging.getLogger(__name__)

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


def localization_node(state: "PipelineState") -> "PipelineState":
    """
    LangGraph node: localise the post caption to the requested locale.
    Skips if locale is absent or 'en'.
    """
    localization = state.get("localization")
    if not localization:
        return state

    locale = localization.get("locale", "en").lower()
    if locale in ("en", "en-us", "en-gb"):
        return state

    generated = state.get("generated_content")
    if not generated:
        return state

    caption = generated.caption

    prompt = (
        f"Translate and culturally adapt the following social media caption to locale '{locale}'. "
        "Preserve the tone, emojis, and intent. Return only the translated caption, nothing else.\n\n"
        f"Original caption:\n{caption}"
    )

    try:
        response = _get_llm().invoke(prompt)
        localized_caption = response.content.strip()
        logger.info("Localized caption to locale '%s'.", locale)
        return {
            **state,
            "localization": {**localization, "localized_caption": localized_caption},
        }
    except Exception as exc:
        logger.error("Localization failed for locale '%s': %s", locale, exc)
        return state
