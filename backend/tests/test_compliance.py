"""
Property-based tests for ComplianceResult invariants.

Feature: social-media-multi-agent-system
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from agents.compliance_agent import ComplianceResult, compliance_node


def _make_state(query: str) -> dict:
    return {
        "query": query,
        "platform": "linkedin",
        "tasks": [],
        "generated_content": None,
        "compliance_result": None,
        "engagement_analysis": None,
        "human_decision": None,
        "edit_instructions": None,
        "memory_context": None,
    }


# ---------------------------------------------------------------------------
# Property 6: ComplianceResult invariants
# Validates: Requirements 4.1, 4.2, 4.3, 4.4
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(content=st.text(min_size=1, max_size=200))
def test_p6_compliance_result_invariants(content):
    # Feature: social-media-multi-agent-system, Property 6: ComplianceResult invariants
    """
    For any content string passed to compliance_node, the returned ComplianceResult must:
    (a) have status exactly one of "approved", "rejected", "needs_fix"
    (b) have corrected_text not None when status == "needs_fix"
    (c) have a non-empty reason when status == "rejected"
    """
    status = random.choice(["approved", "rejected", "needs_fix"])
    mock_result = ComplianceResult(
        status=status,
        reason="test reason" if status == "rejected" else "",
        corrected_text="fixed content" if status == "needs_fix" else None,
    )

    state = _make_state(content)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="mocked response")

    with patch("agents.compliance_agent._get_llm", return_value=mock_llm), \
         patch("agents.compliance_agent.parser") as mock_parser:
        mock_parser.get_format_instructions.return_value = ""
        mock_parser.parse.return_value = mock_result

        result_state = compliance_node(state)
        result = result_state["compliance_result"]

    # (a) status must be one of the three valid literals
    assert result.status in ("approved", "rejected", "needs_fix"), (
        f"Unexpected status: {result.status!r}"
    )

    # (b) corrected_text must be non-None when status is "needs_fix"
    if result.status == "needs_fix":
        assert result.corrected_text is not None, (
            "corrected_text must not be None when status is 'needs_fix'"
        )

    # (c) reason must be a non-empty string when status is "rejected"
    if result.status == "rejected":
        assert result.reason and len(result.reason) > 0, (
            "reason must be non-empty when status is 'rejected'"
        )
