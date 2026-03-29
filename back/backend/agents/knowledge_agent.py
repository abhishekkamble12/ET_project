"""
Knowledge Agent – RAG retrieval using Qdrant.

Qdrant Collection Schema:
    collection_name = "Knowledge_agent"
    vector size = 384
    payload = {
        "text": str,
        "source": str (optional)
    }
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from qdrant_client import QdrantClient
from services.embedding import get_embedding

if TYPE_CHECKING:
    from models.state import PipelineState

logger = logging.getLogger(__name__)

COLLECTION_NAME = "Knowledge_agent"
DEFAULT_TOP_K = 5


def _get_client() -> QdrantClient:
    import os
    return QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
    )


class KnowledgeAgent:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection = collection_name

    # ── Store ────────────────────────────────────────────────────

    def store_knowledge(self, text: str, source: str = "unknown") -> bool:
        """Embed and store text into Qdrant collection."""
        if not text or not text.strip():
            raise ValueError("Empty text provided")

        embedding = get_embedding(text)
        if not embedding:
            raise ValueError("Embedding generation failed")

        try:
            import uuid
            point_id = str(uuid.uuid4())
            client = _get_client()
            client.upsert(
                collection_name=self.collection,
                points=[
                    {
                        "id": point_id,
                        "vector": embedding,
                        "payload": {
                            "text": text,
                            "source": source
                        }
                    }
                ]
            )

            return True

        except Exception as exc:
            logger.error("store_knowledge failed: %s", exc)
            return False

    # ── Retrieve ─────────────────────────────────────────────────

    def retrieve_knowledge(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[str]:
        """Retrieve top-k relevant chunks from Qdrant."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = get_embedding(query)
        if not query_embedding:
            raise ValueError("Query embedding failed")

        try:
            client = _get_client()
            results = client.search(
                collection_name=self.collection,
                query_vector=query_embedding,
                limit=top_k
            )

            if not results:
                logger.warning("No relevant documents found for query: %s", query[:80])
                return []

            return [
                r.payload.get("text", "")
                for r in results
                if r.payload and r.payload.get("text")
            ]

        except Exception as exc:
            logger.error("retrieve_knowledge failed: %s", exc)
            return []


# ── LangGraph node ───────────────────────────────────────────────

def knowledge_node(state: "PipelineState") -> "PipelineState":
    """
    LangGraph node:
    Retrieves relevant knowledge and injects into state.
    Gracefully skips if Qdrant is unavailable.
    """
    query = state.get("query", "")
    if not query:
        return {**state, "knowledge_context": None}

    try:
        agent = KnowledgeAgent()
        docs = agent.retrieve_knowledge(query)
        context = "\n\n".join(docs) if docs else None
        logger.info("Knowledge node retrieved %d document(s).", len(docs))
    except Exception as exc:
        logger.warning("Knowledge node skipped (Qdrant unavailable): %s", exc)
        context = None

    return {**state, "knowledge_context": context}