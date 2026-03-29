"""
Pydantic v2 schemas for the FastAPI backend.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────

class Platform(str, Enum):
    linkedin = "linkedin"
    instagram = "instagram"


class HumanDecision(str, Enum):
    publish = "publish"
    edit = "edit"
    reject = "reject"


# ── Request schemas ──────────────────────────────────────────────

class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, examples=["Write a post about our new AI product launch"])
    platform: Platform = Field(..., examples=["linkedin"])
    locale: Optional[str] = Field(None, examples=["es"], description="ISO 639-1 locale for localization")
    session_id: Optional[UUID] = Field(default_factory=uuid4, description="Reuse to resume a session")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Announce our new AI-powered analytics dashboard",
                "platform": "linkedin",
                "locale": "en",
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
    }


class StreamRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    platform: Platform
    locale: Optional[str] = None
    session_id: Optional[UUID] = Field(default_factory=uuid4)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Promote our sustainability initiative",
                "platform": "instagram",
                "locale": "en",
                "session_id": "550e8400-e29b-41d4-a716-446655440001"
            }
        }
    }


class ApproveRequest(BaseModel):
    session_id: UUID = Field(..., description="Session ID returned from /stream or /generate")
    decision: HumanDecision = Field(..., description="publish | edit | reject")
    edits: Optional[str] = Field(None, description="Edit instructions when decision=edit")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440001",
                "decision": "publish",
                "edits": None
            }
        }
    }


# ── Response schemas ─────────────────────────────────────────────

class PSMetrics(BaseModel):
    turnaround_seconds: float
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    engagement_improvement: float
    knowledge_summary: str = Field(default="Knowledge ingested: 12 documents")


class GeneratedPost(BaseModel):
    caption: Optional[str] = None
    image_prompt: Optional[str] = None
    hashtags: list[str] = []
    platform: str
    locale: Optional[str] = None


class FormattedPosts(BaseModel):
    linkedin: Optional[dict[str, Any]] = None
    instagram: Optional[dict[str, Any]] = None


class GenerateResponse(BaseModel):
    session_id: UUID
    status: str = Field(..., examples=["completed", "waiting_human"])
    generated_post: Optional[GeneratedPost] = None
    formatted_posts: Optional[FormattedPosts] = None
    engagement_score: Optional[float] = None
    compliance_status: Optional[str] = None
    metrics: PSMetrics


class ApproveResponse(BaseModel):
    session_id: UUID
    status: str
    message: str
    metrics: PSMetrics


# ── Enterprise schemas ───────────────────────────────────────────

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    company_id: str
    department: Optional[str] = None
    chunk_count: int
    indexed_at: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str = "indexed"
    message: str


class ListDocumentsResponse(BaseModel):
    documents: list[DocumentMetadata]
    total: int


class DeleteResponse(BaseModel):
    doc_id: str
    status: str = "deleted"
    message: str
