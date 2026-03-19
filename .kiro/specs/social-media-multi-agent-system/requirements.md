# Requirements Document

## Introduction

A context-aware social media multi-agent system built on AWS AgentCore memory and LangGraph. A Supervisor Agent orchestrates a pipeline of four specialized sub-agents: content generation, compliance checking, engagement analysis, and human review. The system persists cross-session memory so it can learn from past posts, user preferences, and platform history to improve future content.

## Glossary

- **Supervisor_Agent**: The top-level orchestrator that routes tasks between sub-agents and manages the overall pipeline state.
- **Content_Generation_Agent**: Sub-agent responsible for producing captions, image prompts, and hashtags for a given topic and platform.
- **Compliance_Agent**: Sub-agent that checks generated content for harmful material, policy violations, and factual inaccuracies.
- **Engagement_Analysis_Agent**: Sub-agent that predicts expected engagement, audience reaction, and post impact before publishing.
- **Human_Review_Agent**: Sub-agent that presents the finalized content to the user and collects a publish/edit/reject decision.
- **AgentCore_Memory**: AWS AgentCore persistent memory store used to retain cross-session context such as past posts, user preferences, and platform history.
- **Pipeline_State**: The shared LangGraph state object passed between all agents in a single run.
- **Platform**: A supported social media destination (LinkedIn, Instagram, etc.).
- **Session**: A single end-to-end run of the pipeline from user request to final publish or rejection.

---

## Requirements

### Requirement 1: Supervisor Agent Orchestration

**User Story:** As a developer, I want a Supervisor Agent that routes work to the correct sub-agent at each pipeline stage, so that the overall workflow is coordinated without manual intervention.

#### Acceptance Criteria

1. THE Supervisor_Agent SHALL maintain a Pipeline_State that includes: the original user request, target Platform, generated content, compliance result, engagement analysis, and human decision.
2. WHEN a user submits a social media post request, THE Supervisor_Agent SHALL invoke the Content_Generation_Agent as the first step.
3. WHEN the Content_Generation_Agent returns output, THE Supervisor_Agent SHALL route the Pipeline_State to the Compliance_Agent.
4. WHEN the Compliance_Agent returns an `approved` status, THE Supervisor_Agent SHALL route the Pipeline_State to the Engagement_Analysis_Agent.
5. WHEN the Compliance_Agent returns a `needs_fix` status, THE Supervisor_Agent SHALL apply the corrected text and re-route to the Compliance_Agent for re-evaluation.
6. WHEN the Compliance_Agent returns a `rejected` status, THE Supervisor_Agent SHALL terminate the pipeline and return a rejection reason to the user.
7. WHEN the Engagement_Analysis_Agent returns output, THE Supervisor_Agent SHALL route the Pipeline_State to the Human_Review_Agent.
8. WHEN the Human_Review_Agent returns an `edit` decision, THE Supervisor_Agent SHALL route the Pipeline_State back to the Content_Generation_Agent with the user's edit instructions.
9. WHEN the Human_Review_Agent returns a `no` decision, THE Supervisor_Agent SHALL terminate the pipeline without publishing.
10. WHEN the Human_Review_Agent returns a `publish` decision, THE Supervisor_Agent SHALL mark the post as published and persist the post record to AgentCore_Memory.

---

### Requirement 2: AWS AgentCore Memory Integration

**User Story:** As a product owner, I want the system to use AWS AgentCore memory instead of in-process memory, so that context is preserved across sessions and deployments.

#### Acceptance Criteria

1. THE Supervisor_Agent SHALL use `AgentCoreMemorySaver` as the LangGraph checkpointer, replacing `InMemorySaver`.
2. THE Supervisor_Agent SHALL derive the `memory_id` from a configuration value or environment variable, not from a hardcoded literal string.
3. WHEN a new session starts, THE Supervisor_Agent SHALL load prior context (past posts, user preferences, platform history) from AgentCore_Memory using the resolved `memory_id`.
4. WHEN a session ends with a `publish` decision, THE Supervisor_Agent SHALL write the completed Pipeline_State to AgentCore_Memory.
5. WHILE a session is active, THE Supervisor_Agent SHALL pass the AgentCore_Memory checkpointer to the compiled LangGraph graph via `RunnableConfig`.

---

### Requirement 3: Context-Aware Content Generation

**User Story:** As a social media manager, I want the Content_Generation_Agent to use past post history and user preferences from memory, so that new content is consistent with my brand voice and avoids repeating past topics.

#### Acceptance Criteria

1. WHEN generating content, THE Content_Generation_Agent SHALL accept a topic, target Platform, and a memory context object as inputs.
2. WHEN memory context contains prior posts for the same Platform, THE Content_Generation_Agent SHALL avoid repeating hashtags or phrasing used in the three most recent posts.
3. THE Content_Generation_Agent SHALL return a structured output containing: `caption` (string), `image_prompt` (string), `hashtags` (list of strings), and `platform` (string).
4. WHERE the Platform is LinkedIn, THE Content_Generation_Agent SHALL produce a professional tone with no more than 5 hashtags.
5. WHERE the Platform is Instagram, THE Content_Generation_Agent SHALL produce a conversational tone with 10 to 30 hashtags.
6. THE Content_Generation_Agent SHALL use a single, correctly indented `create_social_post` method on the `ContentCreationAgent` class.

---

### Requirement 4: Compliance Checking

**User Story:** As a compliance officer, I want every piece of generated content reviewed for harmful material and policy violations before it reaches the user, so that the brand is protected.

#### Acceptance Criteria

1. WHEN content is submitted for review, THE Compliance_Agent SHALL invoke an LLM with the compliance prompt and return a `ComplianceResult` Pydantic object.
2. THE Compliance_Agent SHALL classify each submission as exactly one of: `approved`, `rejected`, or `needs_fix`.
3. WHEN the status is `needs_fix`, THE Compliance_Agent SHALL populate the `corrected_text` field with a revised version of the content.
4. WHEN the status is `rejected`, THE Compliance_Agent SHALL populate the `reason` field with a human-readable explanation.
5. IF the LLM response cannot be parsed into a `ComplianceResult`, THEN THE Compliance_Agent SHALL return a `rejected` status with reason "Compliance check failed to parse response".

---

### Requirement 5: Engagement Analysis

**User Story:** As a social media manager, I want predicted engagement metrics before I decide to publish, so that I can make an informed decision.

#### Acceptance Criteria

1. WHEN approved content is received, THE Engagement_Analysis_Agent SHALL return a structured `EngagementAnalysis` object containing: `expected_engagement_score` (float 0.0–1.0), `predicted_audience_reaction` (string), and `post_impact_summary` (string).
2. WHEN memory context contains historical engagement data for the same Platform, THE Engagement_Analysis_Agent SHALL factor that history into the `expected_engagement_score`.
3. THE Engagement_Analysis_Agent SHALL complete analysis and return output within 30 seconds.
4. IF the Engagement_Analysis_Agent fails to produce a result, THEN THE Supervisor_Agent SHALL log the error and route the Pipeline_State directly to the Human_Review_Agent with a null engagement analysis.

---

### Requirement 6: Human-in-the-Loop Review

**User Story:** As a user, I want to review the generated post and engagement prediction before it goes live, so that I retain final control over what is published.

#### Acceptance Criteria

1. WHEN the Human_Review_Agent is invoked, THE Human_Review_Agent SHALL present the user with the caption, image prompt, hashtags, and engagement analysis.
2. THE Human_Review_Agent SHALL accept exactly one of three decisions: `publish`, `edit`, or `no`.
3. WHEN the user selects `edit`, THE Human_Review_Agent SHALL collect free-text edit instructions and include them in the Pipeline_State before routing back to the Content_Generation_Agent.
4. WHEN the user selects `publish`, THE Human_Review_Agent SHALL confirm the decision and signal the Supervisor_Agent to finalize the post.
5. WHEN the user selects `no`, THE Human_Review_Agent SHALL confirm cancellation and signal the Supervisor_Agent to terminate the pipeline.

---

### Requirement 7: Multi-Platform Support

**User Story:** As a social media manager, I want to create posts for multiple platforms in a single system, so that I don't need separate tools per platform.

#### Acceptance Criteria

1. THE Supervisor_Agent SHALL accept a `platform` field in the initial user request.
2. THE system SHALL support at minimum the following platforms: LinkedIn and Instagram.
3. WHEN an unsupported platform is specified, THE Supervisor_Agent SHALL return a validation error listing the supported platforms before invoking any sub-agent.
4. WHERE platform-specific formatting rules exist, THE Content_Generation_Agent SHALL apply them as defined in Requirement 3.

---

### Requirement 8: Image Generation

**User Story:** As a content creator, I want the system to generate an image for each post using the image prompt, so that posts are visually complete.

#### Acceptance Criteria

1. WHEN an `image_prompt` is produced by the Content_Generation_Agent, THE Image_Generation_Service SHALL invoke the AWS Bedrock Stable Diffusion XL model and return a base64-encoded image string.
2. IF the Bedrock API call fails, THEN THE Image_Generation_Service SHALL raise an exception with a descriptive error message rather than returning a partial result.
3. THE system SHALL use a single canonical `image_generation.py` module; the duplicate `caption-generator.py` file SHALL be removed.

---

### Requirement 9: Structured Pipeline State and Outputs

**User Story:** As a developer, I want every agent to consume and produce well-typed structured data, so that the pipeline is reliable and easy to debug.

#### Acceptance Criteria

1. THE Pipeline_State TypedDict SHALL include fields: `query` (str), `platform` (str), `tasks` (list[str]), `generated_content` (ContentCreationOutput | None), `compliance_result` (ComplianceResult | None), `engagement_analysis` (EngagementAnalysis | None), `human_decision` (str | None), `edit_instructions` (str | None), and `memory_context` (dict | None).
2. WHEN any agent returns output, THE Supervisor_Agent SHALL merge that output into the Pipeline_State before routing to the next node.
3. IF a required field is missing from the Pipeline_State when a node is invoked, THEN THE Supervisor_Agent SHALL raise a descriptive `ValueError` identifying the missing field.

---

### Requirement 10: Code Quality and Import Correctness

**User Story:** As a developer, I want all modules to have correct imports and no broken references, so that the system runs without import errors on startup.

#### Acceptance Criteria

1. THE Supervisor_Agent module SHALL import `AgentCoreMemorySaver` from the correct AWS AgentCore SDK package.
2. THE Content_Creation module SHALL import `generate_image` from `backend.services.image_generation`.
3. THE Supervisor_Agent module SHALL import `ComplianceResult` and `compliance_prompt` from `backend.agents.compliance_agent` (correctly named module).
4. THE `governent-compliance-agent.py` file SHALL be renamed to `compliance_agent.py` to match Python module naming conventions.
5. THE `create_social_post` function in `ContentCreationAgent` SHALL be correctly indented as an instance method with a `self` parameter.
