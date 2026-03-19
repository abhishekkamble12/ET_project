# Feature: social-media-multi-agent-system, Property 7: EngagementAnalysis structural invariant
import sys
import os

os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-tests")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from services.Engagement import EngagementAnalysis, engagement_node


def _make_state(caption: str, platform: str) -> dict:
    generated = MagicMock()
    generated.caption = caption
    return {
        "query": "test",
        "platform": platform,
        "tasks": [],
        "generated_content": generated,
        "compliance_result": None,
        "engagement_analysis": None,
        "human_decision": None,
        "edit_instructions": None,
        "memory_context": None,
    }


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    caption=st.text(min_size=1, max_size=200),
    platform=st.sampled_from(["linkedin", "instagram"]),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    reaction=st.text(min_size=1, max_size=100),
    summary=st.text(min_size=1, max_size=100),
)
def test_p7_engagement_analysis_structural_invariant(caption, platform, score, reaction, summary):
    # Feature: social-media-multi-agent-system, Property 7: EngagementAnalysis structural invariant
    # Validates: Requirements 5.1
    mock_result = EngagementAnalysis(
        expected_engagement_score=score,
        predicted_audience_reaction=reaction,
        post_impact_summary=summary,
    )
    state = _make_state(caption, platform)

    with patch("services.Engagement._get_llm") as mock_get_llm, \
         patch("services.Engagement.parser") as mock_parser:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="mocked")
        mock_parser.get_format_instructions.return_value = ""
        mock_parser.parse.return_value = mock_result

        result_state = engagement_node(state)
        result = result_state["engagement_analysis"]

    assert result is not None
    assert 0.0 <= result.expected_engagement_score <= 1.0
    assert result.predicted_audience_reaction and len(result.predicted_audience_reaction) > 0
    assert result.post_impact_summary and len(result.post_impact_summary) > 0
