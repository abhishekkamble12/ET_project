from __future__ import annotations

# goal :- to create a compliance agent
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from backend.agents.Supervisor import PipelineState

class ComplianceResult(BaseModel):
    status: Literal["approved", "rejected", "needs_fix"]
    reason: str
    corrected_text: str | None = None

def compliance_prompt(content: str) -> str:
    return f"""
You are a government compliance officer.

Check the following content:

CONTENT:
{content}

Rules:
- No false claims
- No harmful/illegal content
- No misleading information
- Must be factually correct

Return JSON:
{{
  "status": "approved | rejected | needs_fix",
  "reason": "why",
  "corrected_text": "fixed version if needed"
}}
"""

parser = PydanticOutputParser(pydantic_object=ComplianceResult)

_llm = None

def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="qwen/qwen3-32b", temperature=0, max_tokens=None, timeout=None, max_retries=2)
    return _llm

def compliance_node(state: "PipelineState") -> "PipelineState":
    generated = state.get("generated_content")
    content = generated.caption if generated else state.get("query", "")

    prompt = compliance_prompt(content) + "\n\n" + parser.get_format_instructions()

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        result = parser.parse(response.content)
    except Exception:
        result = ComplianceResult(status="rejected", reason="Compliance check failed to parse response")

    return {**state, "compliance_result": result}
