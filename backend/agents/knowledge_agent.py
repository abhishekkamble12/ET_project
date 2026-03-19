"""
Knowledge Agent – RAG retrieval using Supabase pgvector.

Supabase table expected schema:
    knowledge_base (id uuid, content text, embedding vector(384))

Supabase RPC expected:
    match_documents(query_embedding vector, match_count int) -> table(content text, similarity float)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from services.embedding import get_embedding

if TYPE_CHECKING:
    from models.state import PipelineState

logger = logging.getLogger(__name__)

TABLE_NAME = "knowledge_base"
DEFAULT_TOP_K = 5


class KnowledgeAgent:
    def __init__(self, table_name: str = TABLE_NAME):
        self.table = table_name

    # ── Store ────────────────────────────────────────────────────

    def store_knowledge(self, text: str) -> bool:
        """Embed and persist a document to the knowledge base."""
        from services.supabase_client import supabase  # lazy import

        if not text or not text.strip():
            raise ValueError("Empty text provided")

        embedding = get_embedding(text)
        if not embedding:
            raise ValueError("Embedding generation failed")

        try:
            response = supabase.table(self.table).insert(
                {"content": text, "embedding": embedding}
            ).execute()

            if hasattr(response, "error") and response.error:
                logger.error("Supabase insert error: %s", response.error)
                return False
            return True
        except Exception as exc:
            logger.error("store_knowledge failed: %s", exc)
            return False

    # ── Retrieve ─────────────────────────────────────────────────

    def retrieve_knowledge(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
        """Return the top-k most relevant document chunks for *query*."""
        from services.supabase_client import supabase  # lazy import

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = get_embedding(query)
        if not query_embedding:
            raise ValueError("Query embedding failed")

        try:
            response = supabase.rpc(
                "match_documents",
                {"query_embedding": query_embedding, "match_count": top_k},
            ).execute()

            if not response or not response.data:
                logger.warning("No relevant documents found for query: %s", query[:80])
                return []

            return [item["content"] for item in response.data if item.get("content")]
        except Exception as exc:
            logger.error("retrieve_knowledge failed: %s", exc)
            return []


# ── LangGraph node ───────────────────────────────────────────────

def knowledge_node(state: "PipelineState") -> "PipelineState":
    """
    LangGraph node: retrieve relevant knowledge for the current query
    and store the concatenated context in state["knowledge_context"].
    """
    query = state.get("query", "")
    if not query:
        return {**state, "knowledge_context": None}

    agent = KnowledgeAgent()
    docs = agent.retrieve_knowledge(query)

    context = "\n\n".join(docs) if docs else None
    logger.info("Knowledge node retrieved %d document(s).", len(docs))

    return {**state, "knowledge_context": context}
