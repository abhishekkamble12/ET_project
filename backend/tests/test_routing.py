"""
Property-based tests for routing functions and platform validation.

Feature: social-media-multi-agent-system
"""
import sys
import os

os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-tests")
os.environ.setdefault("AGENTCORE_MEMORY_ID", "dummy-memory-id-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from agents.Supervisor import route_compliance, route_human_review, run_pipeline, SUPPORTED_PLATFORMS
from agents.compliance_agent import ComplianceResult
from langgraph.graph import END


def _make_state(**overrides) -> dict:
    base = {
        "query": "test query",
        "platform": "linkedin",
        "tasks": [],
        "generated_content": None,
        "compliance_result": None,
        "engagement_analysis": None,
        "human_decision": None,
        "edit_instructions": None,
        "memory_context": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Property 1: Compliance routing is exhaustive and correct
# Validates: Requirements 1.4, 1.5, 1.6
# ---------------------------------------------------------------------------

VALID_COMPLIANCE_STATUSES = ["approved", "needs_fix", "rejected"]
EXPECTED_COMPLIANCE_ROUTES = {
    "approved": "engagement_analysis_agent",
    "needs_fix": "compliance_agent",
    "rejected": END,
}

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(status=st.sampled_from(VALID_COMPLIANCE_STATUSES))
def test_p1_compliance_routing_exhaustive(status):
    # Feature: social-media-multi-agent-system, Property 1: Compliance routing is exhaustive and correct
    compliance_result = ComplianceResult(
        status=status,
        reason="test reason" if status == "rejected" else "",
        corrected_text="fixed" if status == "needs_fix" else None,
    )
    state = _make_state(compliance_result=compliance_result)
    result = route_compliance(state)
    expected = EXPECTED_COMPLIANCE_ROUTES[status]
    assert result == expected, f"For status={status!r}, expected {expected!r}, got {result!r}"
    # Must never return an unrecognized node name
    recognized = {"engagement_analysis_agent", "compliance_agent", END}
    assert result in recognized, f"route_compliance returned unrecognized value: {result!r}"


# ---------------------------------------------------------------------------
# Property 2: Human review routing is exhaustive and correct
# Validates: Requirements 1.8, 1.9, 6.2
# ---------------------------------------------------------------------------

VALID_HUMAN_DECISIONS = ["publish", "edit", "no"]
EXPECTED_HUMAN_ROUTES = {
    "publish": END,
    "edit": "content_generation_agent",
    "no": END,
}

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(decision=st.sampled_from(VALID_HUMAN_DECISIONS))
def test_p2_human_review_routing_valid(decision):
    # Feature: social-media-multi-agent-system, Property 2: Human review routing is exhaustive and correct
    state = _make_state(human_decision=decision)
    result = route_human_review(state)
    expected = EXPECTED_HUMAN_ROUTES[decision]
    assert result == expected, f"For decision={decision!r}, expected {expected!r}, got {result!r}"


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(decision=st.text(min_size=1, max_size=50).filter(lambda s: s not in VALID_HUMAN_DECISIONS))
def test_p2_human_review_routing_invalid_raises(decision):
    # Feature: social-media-multi-agent-system, Property 2: Human review routing raises ValueError for invalid input
    state = _make_state(human_decision=decision)
    with pytest.raises(ValueError):
        route_human_review(state)


# ---------------------------------------------------------------------------
# Property 8: Unsupported platform raises validation error before any agent runs
# Validates: Requirements 7.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(platform=st.text(min_size=1, max_size=50).filter(lambda s: s.lower() not in SUPPORTED_PLATFORMS))
def test_p8_unsupported_platform_raises_before_agents(platform):
    # Feature: social-media-multi-agent-system, Property 8: Unsupported platform raises validation error before any agent runs
    content_agent_called = []

    def mock_content_node(state):
        content_agent_called.append(True)
        return state

    with patch("agents.Supervisor._get_checkpointer") as mock_checkpointer, \
         patch("agents.Supervisor.content_node", side_effect=mock_content_node):
        mock_checkpointer.return_value = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            run_pipeline(query="test", platform=platform)

    assert len(content_agent_called) == 0, "No agent node should be invoked for unsupported platform"
    assert platform.lower() in str(exc_info.value).lower() or "unsupported" in str(exc_info.value).lower()
