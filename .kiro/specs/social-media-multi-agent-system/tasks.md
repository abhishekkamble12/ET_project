# Implementation Plan: Social Media Multi-Agent System

## Overview

Incremental implementation of the LangGraph multi-agent pipeline. Tasks start with code quality fixes on existing files, then build each agent/service module, wire the StateGraph, and finish with property-based and unit tests.

## Tasks

- [x] 1. Code quality fixes — rename, remove, and repair existing files
  - [x] 1.1 Rename `backend/agents/governent-compliance-agent.py` to `backend/agents/compliance_agent.py`
    - Python module naming convention; fixes broken import in Supervisor
    - _Requirements: 10.4_

  - [x] 1.2 Delete `backend/services/caption-generator.py`
    - Duplicate of `image_generation.py`; only one canonical module should exist
    - _Requirements: 8.3_

  - [x] 1.3 Fix `ContentCreationOutput` fields in `backend/agents/Content_creation.py`
    - Replace `content`, `response` fields with `caption: str`, `image_prompt: str`, `hashtags: list[str]`; keep `platform: str`
    - _Requirements: 3.3, 9.1_

  - [x] 1.4 Fix `create_social_post` indentation in `backend/agents/Content_creation.py`
    - Add `self` as first parameter; indent body inside the class; remove stale `generate_caption` call
    - _Requirements: 3.6, 10.5_

- [x] 2. Define `PipelineState` TypedDict in `backend/agents/supervisor.py`
  - Create (or overwrite) `backend/agents/supervisor.py` with the `PipelineState` TypedDict containing all 9 fields: `query`, `platform`, `tasks`, `generated_content`, `compliance_result`, `engagement_analysis`, `human_decision`, `edit_instructions`, `memory_context`
  - All `Optional` fields default to `None`
  - _Requirements: 9.1_

  - [x] 2.1 Write property test for state merge completeness (P9)
    - **Property 9: Pipeline state merge completeness**
    - Generate random partial state dicts; assert no fields are dropped after merging into `PipelineState`
    - **Validates: Requirements 9.2**

  - [x] 2.2 Write property test for missing-field ValueError (P10)
    - **Property 10: Missing required field raises ValueError**
    - For each required field, set it to `None` and assert `ValueError` names the missing field
    - **Validates: Requirements 9.3**

- [x] 3. Implement `image_generation.py` service with error handling
  - Add try/except around the Bedrock `invoke_model` call; raise `RuntimeError(f"Image generation failed: {e}")` on failure
  - No partial results returned
  - _Requirements: 8.1, 8.2_

- [x] 4. Implement `compliance_agent.py`
  - [x] 4.1 Write `compliance_node(state: PipelineState) -> PipelineState`
    - Invoke LLM with `compliance_prompt` + `parser.get_format_instructions()`
    - Parse response with `PydanticOutputParser(pydantic_object=ComplianceResult)`
    - On parse failure return `ComplianceResult(status="rejected", reason="Compliance check failed to parse response")`
    - Merge result into state and return
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 4.2 Write property test for ComplianceResult invariants (P6)
    - **Property 6: ComplianceResult invariants**
    - Generate random content strings; assert status is one of the three literals, `corrected_text` non-None when `needs_fix`, `reason` non-empty when `rejected`
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [x] 5. Implement `ContentCreationAgent.create_social_post` in `backend/agents/content_creation.py`
  - [x] 5.1 Build platform-specific prompt logic
    - LinkedIn: professional tone, ≤ 5 hashtags
    - Instagram: conversational tone, 10–30 hashtags
    - Accept `topic: str`, `platform: str`, `memory_context: dict | None`
    - _Requirements: 3.1, 3.4, 3.5_

  - [x] 5.2 Add memory context awareness
    - When `memory_context` contains prior posts for the same platform, extract hashtags from the 3 most recent and exclude them from the new post
    - _Requirements: 3.2_

  - [x] 5.3 Call `generate_image` and return `ContentCreationOutput`
    - Import `generate_image` from `backend.services.image_generation`
    - Populate `caption`, `image_prompt`, `hashtags`, `platform`
    - _Requirements: 3.3, 10.2_

  - [x] 5.4 Write property test for content output completeness (P3)
    - **Property 3: Content generation output is structurally complete**
    - Generate random topic/platform/memory_context; assert all four output fields non-null and non-empty
    - **Validates: Requirements 3.1, 3.3**

  - [x] 5.5 Write property test for platform hashtag count invariant (P4)
    - **Property 4: Platform hashtag count invariant**
    - Generate random topics for LinkedIn and Instagram; assert `len(hashtags) <= 5` for LinkedIn and `10 <= len(hashtags) <= 30` for Instagram
    - **Validates: Requirements 3.4, 3.5**

  - [x] 5.6 Write property test for hashtag novelty (P5)
    - **Property 5: Hashtag novelty relative to recent memory**
    - Generate memory context with prior posts; assert no hashtag in new post appears in the union of the 3 most recent prior posts for that platform
    - **Validates: Requirements 3.2**

- [x] 6. Implement `engagement_node` in `backend/services/engagement.py`
  - [x] 6.1 Define `EngagementAnalysis` Pydantic model
    - Fields: `expected_engagement_score: float`, `predicted_audience_reaction: str`, `post_impact_summary: str`
    - _Requirements: 5.1_

  - [x] 6.2 Write `engagement_node(state: PipelineState) -> PipelineState`
    - Invoke LLM with engagement prompt; incorporate historical data from `memory_context` when available
    - Wrap in try/except with 30-second timeout; on failure log error and set `engagement_analysis=None`
    - _Requirements: 5.2, 5.3, 5.4_

  - [x] 6.3 Write property test for EngagementAnalysis structural invariant (P7)
    - **Property 7: EngagementAnalysis structural invariant**
    - Generate random approved content; assert `0.0 <= expected_engagement_score <= 1.0` and both string fields non-empty
    - **Validates: Requirements 5.1**

- [x] 7. Implement `human_review_node` in `backend/agents/supervisor.py`
  - Present `caption`, `image_prompt`, `hashtags`, and `engagement_analysis` to the user via stdout/input
  - Accept exactly one of `publish`, `edit`, `no`; collect free-text edit instructions when `edit` is chosen
  - Write result into `human_decision` and `edit_instructions` fields of state
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 8. Wire the StateGraph and AgentCore memory in `backend/agents/supervisor.py`
  - [x] 8.1 Implement routing functions `route_compliance` and `route_human_review`
    - `route_compliance`: `approved` → `engagement_analysis_agent`, `needs_fix` → `compliance_agent`, `rejected` → `END`
    - `route_human_review`: `publish` → `END`, `edit` → `content_generation_agent`, `no` → `END`; raise `ValueError` for unknown values
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [x] 8.2 Add platform validation in `run_pipeline`
    - Before invoking any node, check `platform` against `{"linkedin", "instagram"}`; raise `ValueError` with supported list if invalid
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 8.3 Wire all nodes and edges
    - `add_node` for all four agents; `add_edge(START, "content_generation_agent")`; `add_edge("content_generation_agent", "compliance_agent")`; `add_conditional_edges` for compliance and human review; `add_edge("engagement_analysis_agent", "human_review_agent")`
    - _Requirements: 1.2, 1.3, 1.7_

  - [x] 8.4 Integrate `AgentCoreMemorySaver` as checkpointer
    - Read `memory_id` from `os.environ["AGENTCORE_MEMORY_ID"]`; raise `EnvironmentError` if unset
    - Import `AgentCoreMemorySaver` from the correct AWS AgentCore SDK package
    - Pass checkpointer to `graph.compile(checkpointer=checkpointer)`
    - _Requirements: 2.1, 2.2, 10.1_

  - [x] 8.5 Implement memory write on publish
    - After `human_decision == "publish"`, write completed `PipelineState` to AgentCore memory via the checkpointer config
    - _Requirements: 2.4, 1.10_

  - [x] 8.6 Add required-field validation guard before each node
    - Raise `ValueError(f"Required field '{field}' is missing from PipelineState")` when a node's required input field is `None`
    - _Requirements: 9.3_

  - [x] 8.7 Fix all imports in `supervisor.py`
    - Import `ComplianceResult`, `compliance_node` from `backend.agents.compliance_agent`
    - Import `ContentCreationAgent` from `backend.agents.content_creation`
    - Import `engagement_node`, `EngagementAnalysis` from `backend.services.engagement`
    - _Requirements: 10.1, 10.3_

  - [x] 8.8 Write property test for compliance routing exhaustiveness (P1)
    - **Property 1: Compliance routing is exhaustive and correct**
    - Generate random `ComplianceResult` status values from the three literals; assert `route_compliance` returns the correct node name and never an unrecognized string
    - **Validates: Requirements 1.4, 1.5, 1.6**

  - [x] 8.9 Write property test for human review routing exhaustiveness (P2)
    - **Property 2: Human review routing is exhaustive and correct**
    - Generate random decision strings; assert correct return for the three valid values and `ValueError` for any other string
    - **Validates: Requirements 1.8, 1.9, 6.2**

  - [x] 8.10 Write property test for unsupported platform validation (P8)
    - **Property 8: Unsupported platform raises validation error before any agent runs**
    - Generate random strings not in `{"linkedin", "instagram"}`; assert `ValueError` is raised and no agent node is invoked
    - **Validates: Requirements 7.3**

- [x] 9. Checkpoint — ensure all tests pass
  - Run the full test suite; confirm no import errors, all property tests pass at ≥ 100 iterations, all unit tests green. Ask the user if questions arise.

- [x] 10. Write unit tests in `backend/tests/test_pipeline.py`
  - [x] 10.1 Test specific routing examples
    - `approved` → `"engagement_analysis_agent"`, `needs_fix` → `"compliance_agent"`, `rejected` → `END`
    - `publish` → `END`, `edit` → `"content_generation_agent"`, `no` → `END`
    - _Requirements: 1.4, 1.5, 1.6, 1.8, 1.9_

  - [x] 10.2 Test image generation integration
    - Mock Bedrock client; assert `generate_image` returns base64 string on success and raises `RuntimeError` on API failure
    - _Requirements: 8.1, 8.2_

  - [x] 10.3 Test `AgentCoreMemorySaver` is used as checkpointer
    - Compile graph and inspect `compiled_graph.checkpointer`; assert it is an instance of `AgentCoreMemorySaver`
    - _Requirements: 2.1_

  - [x] 10.4 Test `memory_id` is read from environment variable
    - Set `AGENTCORE_MEMORY_ID` env var in test; assert the value is passed to `AgentCoreMemorySaver`
    - _Requirements: 2.2_

  - [x] 10.5 Test engagement failure routes to human review with null analysis
    - Mock `engagement_node` to raise an exception; assert `PipelineState.engagement_analysis` is `None` and next node is `human_review_agent`
    - _Requirements: 5.4_

  - [x] 10.6 Test publish path triggers memory write
    - Mock `AgentCoreMemorySaver`; run pipeline to `publish` decision; assert memory write was called with the completed state
    - _Requirements: 2.4, 1.10_

  - [x] 10.7 Test `caption-generator.py` does not exist
    - Assert `os.path.exists("backend/services/caption-generator.py")` is `False`
    - _Requirements: 8.3_

- [x] 11. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use `hypothesis` and run a minimum of 100 iterations each
- Each property test is tagged with `# Feature: social-media-multi-agent-system, Property {N}: {property_text}`
- All code examples use Python
- Checkpoints ensure incremental validation before moving to the next phase
