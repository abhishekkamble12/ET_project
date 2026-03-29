"""
Shared FastAPI dependencies: Qdrant client, checkpointer, auth header.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Header, HTTPException, status
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = "enterprise_knowledge"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2


@lru_cache(maxsize=1)
def get_sync_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)


@lru_cache(maxsize=1)
def get_async_qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)


async def ensure_collection(client: AsyncQdrantClient) -> None:
    """Create the Qdrant collection if it doesn't exist."""
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


async def get_qdrant() -> AsyncQdrantClient:
    client = get_async_qdrant()
    await ensure_collection(client)
    return client


def verify_company_id(x_company_id: Annotated[str | None, Header()] = None) -> str:
    """Demo auth: require X-Company-ID header."""
    if not x_company_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Company-ID header is required",
        )
    return x_company_id
