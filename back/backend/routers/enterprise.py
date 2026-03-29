"""
Enterprise Knowledge router – upload, list, delete documents in Qdrant.
"""
from __future__ import annotations

import io
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList, PointStruct

from dependencies import COLLECTION_NAME, get_qdrant, verify_company_id
from schemas import DeleteResponse, DocumentMetadata, ListDocumentsResponse, UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/enterprise",
    tags=["Enterprise Knowledge"],
)

SUPPORTED_TYPES = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# ── Text extraction ──────────────────────────────────────────────

def _extract_text(content: bytes, content_type: str, filename: str) -> str:
    """Extract plain text from uploaded file bytes."""
    if content_type == "application/pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            return content.decode("utf-8", errors="ignore")

    if content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return content.decode("utf-8", errors="ignore")

    return content.decode("utf-8", errors="ignore")


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Simple sliding-window chunker."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks or [text]


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with sentence-transformers all-MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=False).tolist()


# ── Background indexing task ─────────────────────────────────────

async def _index_to_qdrant(
    client: AsyncQdrantClient,
    doc_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: dict,
) -> None:
    """Upsert all chunks into Qdrant (runs as BackgroundTask)."""
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "doc_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                **metadata,
            },
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    await client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Indexed %d chunks for doc_id=%s", len(points), doc_id)


# ── POST /enterprise/data/upload ─────────────────────────────────

@router.post(
    "/data/upload",
    response_model=UploadResponse,
    summary="Upload and index a document",
    description=(
        "Upload PDF, CSV, TXT, or DOCX. The file is chunked, embedded with "
        "sentence-transformers, and stored in Qdrant with company/department metadata. "
        "Indexing runs as a background task."
    ),
    responses={
        202: {"description": "File accepted and indexing started"},
        400: {"description": "Unsupported file type"},
    },
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF, CSV, TXT, or DOCX file")],
    company_id: Annotated[str, Form(description="Company identifier")],
    department: Annotated[str | None, Form(description="Department name (optional)")] = None,
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    _auth: str = Depends(verify_company_id),
) -> UploadResponse:
    content_type = file.content_type or ""
    if content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Allowed: PDF, CSV, TXT, DOCX",
        )

    content = await file.read()
    text = _extract_text(content, content_type, file.filename or "upload")
    chunks = _chunk_text(text)
    embeddings = _embed(chunks)

    doc_id = str(uuid.uuid4())
    metadata = {
        "company_id": company_id,
        "department": department or "general",
        "filename": file.filename or "upload",
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }

    background_tasks.add_task(
        _index_to_qdrant, qdrant, doc_id, chunks, embeddings, metadata
    )

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename or "upload",
        chunk_count=len(chunks),
        status="indexed",
        message=f"Indexing {len(chunks)} chunks in background. doc_id={doc_id}",
    )


# ── GET /enterprise/data/list ────────────────────────────────────

@router.get(
    "/data/list",
    response_model=ListDocumentsResponse,
    summary="List all indexed documents",
    description="Returns all documents stored in the Qdrant enterprise_knowledge collection.",
)
async def list_documents(
    company_id: str = Depends(verify_company_id),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
) -> ListDocumentsResponse:
    # Scroll through all points and deduplicate by doc_id
    seen: dict[str, DocumentMetadata] = {}
    offset = None

    while True:
        result, next_offset = await qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in result:
            payload = point.payload or {}
            did = payload.get("doc_id", str(point.id))
            if did not in seen:
                seen[did] = DocumentMetadata(
                    doc_id=did,
                    filename=payload.get("filename", "unknown"),
                    company_id=payload.get("company_id", "unknown"),
                    department=payload.get("department"),
                    chunk_count=1,
                    indexed_at=payload.get("indexed_at", ""),
                )
            else:
                seen[did].chunk_count += 1

        if next_offset is None:
            break
        offset = next_offset

    docs = list(seen.values())
    return ListDocumentsResponse(documents=docs, total=len(docs))


# ── DELETE /enterprise/data/{doc_id} ─────────────────────────────

@router.delete(
    "/data/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete a document by doc_id",
    description="Removes all Qdrant points associated with the given doc_id.",
    responses={
        200: {"description": "Document deleted"},
        404: {"description": "Document not found"},
    },
)
async def delete_document(
    doc_id: str,
    company_id: str = Depends(verify_company_id),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
) -> DeleteResponse:
    # Find all point IDs for this doc_id
    results, _ = await qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=1000,
        with_payload=False,
        with_vectors=False,
    )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in Qdrant.",
        )

    point_ids = [str(p.id) for p in results]
    await qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=point_ids),
    )

    return DeleteResponse(
        doc_id=doc_id,
        status="deleted",
        message=f"Deleted {len(point_ids)} chunks for doc_id={doc_id}",
    )
