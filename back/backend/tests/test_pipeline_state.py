"""
Property-based tests for PipelineState TypedDict and validate_required_field.

Feature: social-media-multi-agent-system
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from agents.Supervisor import validate_required_field
from models.state import PipelineState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPTIONAL_FIELDS = [
    "knowledge_context",
    "strategy",
    "generated_content",
    "compliance_result",
    "engagement_analysis",
    "localization",
    "formatted_posts",
    "human_decision",
    "edit_instructions",
    "memory_context",
]

_ALL_FIELDS = ["query", "platform", "tasks", "optimization_attempts"] + _OPTIONAL_FIELDS


def _make_state(**overrides) -> dict:
    base: dict = {
        "query": "test query",
        "platform": "linkedin",
        "tasks": [],
        "knowledge_context": None,
        "strategy": None,
        "generated_content": None,
        "optimization_attempts": 0,
        "compliance_result": None,
        "engagement_analysis": None,
        "localization": None,
        "formatted_posts": None,
        "human_decision": None,
        "edit_instructions": None,
        "memory_context": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Property 9: Pipeline state merge completeness
# ---------------------------------------------------------------------------

_optional_value_st = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=50),
    st.dictionaries(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=10)),
)

_partial_update_st = st.fixed_dictionaries(
    {},
    optional={field: _optional_value_st for field in _OPTIONAL_FIELDS},
)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(partial_update=_partial_update_st)
def test_p9_state_merge_completeness(partial_update):
    state = _make_state()
    merged = {**state, **partial_update}

    for field in _ALL_FIELDS:
        assert field in merged, f"Field '{field}' was dropped after merge"


# ---------------------------------------------------------------------------
# Property 10: Missing required field raises ValueError
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(field=st.sampled_from(_OPTIONAL_FIELDS))
def test_p10_missing_field_raises_value_error(field):
    state = _make_state(**{field: None})

    with pytest.raises(ValueError) as exc_info:
        validate_required_field(state, field)

    assert field in str(exc_info.value), (
        f"ValueError message did not name the missing field '{field}': {exc_info.value}"
    )
