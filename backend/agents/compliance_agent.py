"""
Compliance Agent – checks generated content against policy rules.

The node loops internally until the content is COMPLIANT or the maximum
number of fix attempts is exhausted (hard cap: MAX_FIX_ATTEMPTS).
If the LLM returns "needs_fix", the corrected_text is fed back for
re-evaluation in the next iteration.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

if TYPE_CHECKING:
    from models.state import PipelineState

logger = logging.getLogger(__name__)

MAX_FIX_ATTEMPTS = 5


# ── Schema ───────────────────────────────────────────────────────

class ComplianceResult(BaseModel):
    status: Literal["approved", "rejected", "needs_fix"]
    reason: str
    corrected_text: str | None = None


# ── LLM ─────────────────────────────────────────────────────────

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    return _llm


parser = PydanticOutputParser(pydantic_object=ComplianceResult)


# ── Prompt ───────────────────────────────────────────────────────

def _compliance_prompt(content: str) -> str:
    return f"""You are a government compliance officer reviewing social media content.

CONTENT TO REVIEW:
{content}

Compliance rules:
1. No false or unverifiable claims
2. No harmful, violent, or illegal content
3. No misleading or deceptive information
4. Must be factually accurate and professionally appropriate

Return JSON with:
{{
  "status": "approved | rejected | needs_fix",
  "reason": "concise explanation",
  "corrected_text": "corrected version of the content (only when status is needs_fix, else null)"
}}
"""


# ── Internal check ───────────────────────────────────────────────

def _run_compliance_check(content: str) -> ComplianceResult:
    prompt = _compliance_prompt(content) + "\n\n" + parser.get_format_instructions()
    try:
        response = _get_llm().invoke(prompt)
        return parser.parse(response.content)
    except Exception as exc:
        logger.error("Compliance LLM call failed: %s", exc)
        return ComplianceResult(
            status="rejected",
            reason=f"Compliance check failed: {exc}",
        )


# ── LangGraph node ───────────────────────────────────────────────

def compliance_node(state: "PipelineState") -> "PipelineState":
    """
    LangGraph node: iteratively check and fix content until it is
    COMPLIANT (approved) or definitively REJECTED.

    The loop runs up to MAX_FIX_ATTEMPTS times.  On each "needs_fix"
    iteration the corrected_text replaces the caption for the next pass.
    The final ComplianceResult is stored in state["compliance_result"].
    """
    generated = state.get("generated_content")
    content: str = generated.caption if generated else state.get("query", "")

    result: ComplianceResult | None = None

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        result = _run_compliance_check(content)
        logger.info(
            "Compliance attempt %d/%d → status=%s",
            attempt, MAX_FIX_ATTEMPTS, result.status,
        )

        if result.status == "approved":
            # Patch the caption in generated_content with the last reviewed text
            if generated and content != generated.caption:
                generated = generated.model_copy(update={"caption": content})
            break

        if result.status == "rejected":
            break

        # needs_fix – use corrected_text for next iteration
        if result.corrected_text:
            content = result.corrected_text
            if generated:
                generated = generated.model_copy(update={"caption": content})
        else:
            # LLM said needs_fix but gave no correction – treat as rejected
            result = ComplianceResult(
                status="rejected",
                reason="needs_fix returned without corrected_text",
            )
            break

    return {**state, "generated_content": generated, "compliance_result": result}
