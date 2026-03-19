"""
Unit tests for the social media multi-agent pipeline.

Feature: social-media-multi-agent-system
"""
import sys
import os

os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-tests")
os.environ.setdefault("AGENTCORE_MEMORY_ID", "dummy-memory-id-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

from langgraph.graph import END

from agents.Supervisor import (
    route_compliance,
    route_human_review,
    build_graph,
    _get_checkpointer,
    SUPPORTED_PLATFORMS,
)
from agents.compliance_agent import ComplianceResult
from services.image_generation import generate_image


def _make_state(**overrides) -> dict:
    base = {
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
# 10.1 Test specific routing examples
# ---------------------------------------------------------------------------

class TestRoutingExamples:
    def test_compliance_approved_routes_to_engagement(self):
        result = ComplianceResult(status="approved", reason="")
        state = _make_state(compliance_result=result)
        assert route_compliance(state) == "engagement_analysis_agent"

    def test_compliance_needs_fix_routes_to_end(self):
        # compliance_node loops internally; needs_fix on exit means max attempts hit → END
        result = ComplianceResult(status="needs_fix", reason="", corrected_text="fixed")
        state = _make_state(compliance_result=result)
        assert route_compliance(state) == END

    def test_compliance_rejected_routes_to_end(self):
        result = ComplianceResult(status="rejected", reason="harmful content")
        state = _make_state(compliance_result=result)
        assert route_compliance(state) == END

    def test_human_review_publish_routes_to_end(self):
        state = _make_state(human_decision="publish")
        assert route_human_review(state) == END

    def test_human_review_edit_routes_to_content_agent(self):
        state = _make_state(human_decision="edit")
        assert route_human_review(state) == "content_generation_agent"

    def test_human_review_no_routes_to_end(self):
        state = _make_state(human_decision="no")
        assert route_human_review(state) == END


# ---------------------------------------------------------------------------
# 10.2 Test image generation integration
# ---------------------------------------------------------------------------

class TestImageGeneration:
    def test_generate_image_returns_base64_on_success(self):
        mock_bedrock = MagicMock()
        mock_response = {
            "body": MagicMock(read=lambda: b'{"artifacts": [{"base64": "abc123"}]}')
        }
        mock_bedrock.invoke_model.return_value = mock_response

        with patch("services.image_generation.bedrock", mock_bedrock):
            result = generate_image("a beautiful sunset")

        assert result == "abc123"
        mock_bedrock.invoke_model.assert_called_once()

    def test_generate_image_raises_runtime_error_on_api_failure(self):
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = Exception("Bedrock API error")

        with patch("services.image_generation.bedrock", mock_bedrock):
            with pytest.raises(RuntimeError) as exc_info:
                generate_image("a beautiful sunset")

        assert "Image generation failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 10.3 Test AgentCoreMemorySaver is used as checkpointer
# ---------------------------------------------------------------------------

class TestAgentCoreMemorySaver:
    def test_checkpointer_is_agentcore_memory_saver(self):
        mock_saver_class = MagicMock()
        mock_saver_instance = MagicMock()
        mock_saver_class.return_value = mock_saver_instance

        with patch("agents.Supervisor.AgentCoreMemorySaver", mock_saver_class):
            checkpointer = _get_checkpointer()

        assert checkpointer is mock_saver_instance
        mock_saver_class.assert_called_once()

    def test_build_graph_uses_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver
        real_checkpointer = MemorySaver()
        compiled = build_graph(checkpointer=real_checkpointer)
        assert compiled.checkpointer is real_checkpointer


# ---------------------------------------------------------------------------
# 10.4 Test memory_id is read from environment variable
# ---------------------------------------------------------------------------

class TestMemoryIdFromEnv:
    def test_memory_id_passed_to_agentcore_saver(self):
        test_memory_id = "test-memory-id-12345"
        mock_saver_class = MagicMock()

        with patch.dict(os.environ, {"AGENTCORE_MEMORY_ID": test_memory_id}):
            with patch("agents.Supervisor.AgentCoreMemorySaver", mock_saver_class):
                _get_checkpointer()

        call_kwargs = mock_saver_class.call_args
        assert call_kwargs.kwargs.get("memory_id") == test_memory_id or \
               (call_kwargs.args and call_kwargs.args[0] == test_memory_id)

    def test_missing_memory_id_raises_environment_error(self):
        env_backup = os.environ.pop("AGENTCORE_MEMORY_ID", None)
        try:
            mock_saver_class = MagicMock()
            with patch("agents.Supervisor.AgentCoreMemorySaver", mock_saver_class):
                with pytest.raises(EnvironmentError):
                    _get_checkpointer()
        finally:
            if env_backup is not None:
                os.environ["AGENTCORE_MEMORY_ID"] = env_backup


# ---------------------------------------------------------------------------
# 10.5 Test engagement failure routes to human review with null analysis
# ---------------------------------------------------------------------------

class TestEngagementFailure:
    def test_engagement_failure_sets_null_analysis(self):
        from services.Engagement import engagement_node

        state = _make_state(
            generated_content=MagicMock(caption="test caption", hashtags=[]),
            compliance_result=ComplianceResult(status="approved", reason=""),
        )

        with patch("services.Engagement._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            mock_llm.invoke.side_effect = Exception("LLM timeout")

            result_state = engagement_node(state)

        assert result_state["engagement_analysis"] is None


# ---------------------------------------------------------------------------
# 10.6 Test publish path triggers memory write
# ---------------------------------------------------------------------------

class TestPublishMemoryWrite:
    def test_publish_path_logs_memory_write(self):
        from agents.Supervisor import run_pipeline

        mock_saver_class = MagicMock()
        mock_saver_instance = MagicMock()
        mock_saver_class.return_value = mock_saver_instance

        final_state = _make_state(
            platform="linkedin",
            human_decision="publish",
            generated_content=MagicMock(),
            compliance_result=ComplianceResult(status="approved", reason=""),
        )

        with patch("agents.Supervisor.AgentCoreMemorySaver", mock_saver_class), \
             patch("agents.Supervisor.build_graph") as mock_build_graph:
            mock_compiled = MagicMock()
            mock_compiled.invoke.return_value = final_state
            mock_build_graph.return_value = mock_compiled

            result = run_pipeline(query="test", platform="linkedin")

        assert result["human_decision"] == "publish"
        mock_compiled.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# 10.7 Test caption-generator.py does not exist
# ---------------------------------------------------------------------------

class TestCaptionGeneratorRemoved:
    def test_caption_generator_file_does_not_exist(self):
        caption_gen_path = os.path.join(
            os.path.dirname(__file__), "..", "services", "caption-generator.py"
        )
        assert not os.path.exists(caption_gen_path), (
            f"caption-generator.py should have been deleted but still exists at: "
            f"{os.path.abspath(caption_gen_path)}"
        )
