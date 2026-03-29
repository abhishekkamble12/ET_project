from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not installed – embeddings disabled.")
            return None
    return _model


def get_embedding(text: str) -> list[float]:
    model = _get_model()
    if model is None:
        # Return a zero vector of the expected size so callers don't crash
        return [0.0] * 384
    return model.encode(text).tolist()
