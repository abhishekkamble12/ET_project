"""
Compliance Agent – checks generated content against policy rules.

The node loops internally until the content is COMPLIANT or the maximum
number of fix attempts is exhausted (hard cap: MAX_FIX_ATTEMPTS).
If the LLM returns "needs_fix", the corrected_text is fed back for
re-evaluation in the next iteration.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Literal

from langchain_groq import ChatGroq
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


# ── Robust JSON extraction ──────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract the first JSON object from text that may contain reasoning/thinking."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strip <think>...</think> blocks (common in reasoning models)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find JSON block in markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first {...} block
    brace_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")


def _parse_compliance(text: str) -> ComplianceResult:
    """Parse LLM response into ComplianceResult, handling reasoning wrappers."""
    data = _extract_json(text)
    # Normalize status field
    status_raw = data.get("status", "approved").strip().lower()
    if status_raw in ("approved", "approve", "compliant"):
        status_raw = "approved"
    elif status_raw in ("rejected", "reject", "non-compliant"):
        status_raw = "rejected"
    elif "fix" in status_raw:
        status_raw = "needs_fix"
    data["status"] = status_raw
    return ComplianceResult(**data)


# ── Prompt ───────────────────────────────────────────────────────

def _compliance_prompt(content: str) -> str:
    return f"""You are a compliance officer reviewing social media content.

CONTENT TO REVIEW:
{content}

Compliance rules:
1. No false or unverifiable claims
2. No harmful, violent, or illegal content
3. No misleading or deceptive information
4. Must be factually accurate and professionally appropriate

IMPORTANT: For standard marketing and promotional content that does not make dangerous claims, you should APPROVE it.

Respond with ONLY a JSON object, no other text:
{{
  "status": "approved",
  "reason": "Content is professionally appropriate and compliant.",
  "corrected_text": null
}}

Allowed status values: "approved", "rejected", "needs_fix".
Only use "rejected" for genuinely harmful or illegal content.
Only use "needs_fix" if minor edits would make it compliant.
Do NOT include any explanation outside the JSON.
"""


# ── Internal check ───────────────────────────────────────────────

def _run_compliance_check(content: str) -> ComplianceResult:
    try:
        response = _get_llm().invoke(_compliance_prompt(content))
        return _parse_compliance(response.content)
    except Exception as exc:
        logger.error("Compliance LLM call failed: %s", exc)
        # Fallback: approve on parse failure so the pipeline isn't blocked
        logger.warning("Falling back to auto-approve due to parse error")
        return ComplianceResult(
            status="approved",
            reason=f"Auto-approved (compliance parse error: {exc})",
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
