from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    # ── Core inputs ──────────────────────────────────────────────
    query: str
    platform: str
    tasks: list[str]

    # ── Knowledge / Strategy ─────────────────────────────────────
    knowledge_context: Optional[str]       # RAG-retrieved context from Supabase pgvector
    strategy: Optional[dict]               # {"use_more": [...], "avoid": [...]}

    # ── Generated content ────────────────────────────────────────
    generated_content: Optional[Any]       # ContentCreationOutput
    optimization_attempts: int             # number of regeneration retries so far

    # ── Compliance ───────────────────────────────────────────────
    compliance_result: Optional[Any]       # ComplianceResult

    # ── Engagement ───────────────────────────────────────────────
    engagement_analysis: Optional[Any]     # EngagementAnalysis

    # ── Localization ─────────────────────────────────────────────
    localization: Optional[dict]           # {"locale": str, "localized_caption": str}

    # ── Multi-platform formatted output ──────────────────────────
    formatted_posts: Optional[dict]        # {"linkedin": {...}, "instagram": {...}}

    # ── Human review ─────────────────────────────────────────────
    human_decision: Optional[str]          # "publish" | "edit" | "no"
    edit_instructions: Optional[str]

    # ── Memory ───────────────────────────────────────────────────
    memory_context: Optional[dict]         # loaded from AgentCore / Supabase
