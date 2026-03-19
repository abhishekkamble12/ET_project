"""
Strategy Agent – fetches historical engagement logs from Supabase
and derives content strategy signals: what to use more of and what to avoid.

Supabase table expected schema:
    engagement_logs (
        id          uuid,
        platform    text,
        hashtags    text[],
        tone        text,
        topic       text,
        score       float,
        created_at  timestamptz
    )
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING



if TYPE_CHECKING:
    from models.state import PipelineState

logger = logging.getLogger(__name__)

ENGAGEMENT_TABLE = "engagement_logs"
HIGH_SCORE_THRESHOLD = 0.65
LOW_SCORE_THRESHOLD = 0.40
RECENT_LIMIT = 50          # rows to fetch per platform
TOP_N = 10                 # items to surface in use_more / avoid lists


def _fetch_logs(platform: str) -> list[dict]:
    """Fetch the most recent engagement logs for *platform* from Supabase."""
    from services.supabase_client import supabase  # lazy import

    try:
        response = (
            supabase.table(ENGAGEMENT_TABLE)
            .select("hashtags, tone, topic, score")
            .eq("platform", platform.lower())
            .order("created_at", desc=True)
            .limit(RECENT_LIMIT)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.error("Failed to fetch engagement logs: %s", exc)
        return []


def _build_strategy(logs: list[dict]) -> dict:
    """
    Derive use_more / avoid signals from engagement logs.

    Returns:
        {
            "use_more": {"hashtags": [...], "tones": [...], "topics": [...]},
            "avoid":    {"hashtags": [...], "tones": [...], "topics": [...]},
        }
    """
    high: list[dict] = [r for r in logs if (r.get("score") or 0) >= HIGH_SCORE_THRESHOLD]
    low:  list[dict] = [r for r in logs if (r.get("score") or 0) <= LOW_SCORE_THRESHOLD]

    def _top_hashtags(rows: list[dict]) -> list[str]:
        counter: Counter = Counter()
        for row in rows:
            for tag in (row.get("hashtags") or []):
                counter[tag] += 1
        return [tag for tag, _ in counter.most_common(TOP_N)]

    def _top_values(rows: list[dict], field: str) -> list[str]:
        counter: Counter = Counter(
            row[field] for row in rows if row.get(field)
        )
        return [v for v, _ in counter.most_common(TOP_N)]

    return {
        "use_more": {
            "hashtags": _top_hashtags(high),
            "tones":    _top_values(high, "tone"),
            "topics":   _top_values(high, "topic"),
        },
        "avoid": {
            "hashtags": _top_hashtags(low),
            "tones":    _top_values(low, "tone"),
            "topics":   _top_values(low, "topic"),
        },
    }


# ── LangGraph node ───────────────────────────────────────────────

def strategy_node(state: "PipelineState") -> "PipelineState":
    """
    LangGraph node: load engagement history from Supabase and compute
    use_more / avoid strategy signals stored in state["strategy"].
    """
    platform = state.get("platform", "")
    if not platform:
        logger.warning("strategy_node: no platform in state, skipping.")
        return {**state, "strategy": {"use_more": {}, "avoid": {}}}

    logs = _fetch_logs(platform)
    strategy = _build_strategy(logs)

    logger.info(
        "Strategy node: %d logs processed for platform '%s'.",
        len(logs), platform,
    )
    return {**state, "strategy": strategy}
