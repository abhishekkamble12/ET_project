"""
Strategy Agent – Supabase + AWS Bedrock (Nova-2)

Enhancement:
- Uses Nova-2 LLM to generate intelligent strategy insights
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from models.state import PipelineState

logger = logging.getLogger(__name__)

ENGAGEMENT_TABLE = "engagement_logs"
HIGH_SCORE_THRESHOLD = 0.65
LOW_SCORE_THRESHOLD = 0.40
RECENT_LIMIT = 50
TOP_N = 10


def _get_bedrock():
    try:
        import boto3
        return boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    except Exception:
        return None


# ───────────────────────────────────────────────
# FETCH DATA
# ───────────────────────────────────────────────

def _fetch_logs(platform: str) -> list[dict]:
    from services.supabase_client import supabase

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


# ───────────────────────────────────────────────
# BASIC ANALYSIS (FAST)
# ───────────────────────────────────────────────

def _build_strategy(logs: list[dict]) -> dict:
    high = [r for r in logs if (r.get("score") or 0) >= HIGH_SCORE_THRESHOLD]
    low = [r for r in logs if (r.get("score") or 0) <= LOW_SCORE_THRESHOLD]

    def _top_hashtags(rows):
        counter = Counter()
        for row in rows:
            for tag in (row.get("hashtags") or []):
                counter[tag] += 1
        return [tag for tag, _ in counter.most_common(TOP_N)]

    def _top_values(rows, field):
        counter = Counter(row[field] for row in rows if row.get(field))
        return [v for v, _ in counter.most_common(TOP_N)]

    return {
        "use_more": {
            "hashtags": _top_hashtags(high),
            "tones": _top_values(high, "tone"),
            "topics": _top_values(high, "topic"),
        },
        "avoid": {
            "hashtags": _top_hashtags(low),
            "tones": _top_values(low, "tone"),
            "topics": _top_values(low, "topic"),
        },
    }


# ───────────────────────────────────────────────
# 🔥 NOVA-2 STRATEGY GENERATION
# ───────────────────────────────────────────────

def _generate_ai_strategy(strategy_data: dict, platform: str) -> dict:
    """Use Nova-2 to generate smart strategy insights"""
    bedrock = _get_bedrock()
    if bedrock is None:
        return {"ai_strategy": "AI strategy unavailable (AWS not configured)"}

    prompt = f"""
You are a social media strategist.

Based on the following engagement insights:

{json.dumps(strategy_data, indent=2)}

Generate:
1. Key insights
2. Content strategy recommendations
3. What to focus more on
4. What to avoid
5. 3 content ideas

Platform: {platform}

Return in JSON format.
"""

    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.5
            })
        )

        result = json.loads(response["body"].read())
        text_output = result["output"]["message"]["content"][0]["text"]

        return {
            "ai_strategy": text_output
        }

    except Exception as e:
        logger.error("Nova strategy generation failed: %s", e)
        return {"ai_strategy": "AI strategy generation failed"}


# ───────────────────────────────────────────────
# LANGGRAPH NODE
# ───────────────────────────────────────────────

def strategy_node(state: "PipelineState") -> "PipelineState":
    platform = state.get("platform", "")

    if not platform:
        logger.warning("strategy_node: no platform provided")
        return {**state, "strategy": {}}

    # Step 1: Fetch logs
    logs = _fetch_logs(platform)

    # Step 2: Basic strategy
    base_strategy = _build_strategy(logs)

    # Step 3: AI-enhanced strategy (🔥 new)
    ai_strategy = _generate_ai_strategy(base_strategy, platform)

    final_strategy = {
        "base": base_strategy,
        "ai": ai_strategy
    }

    logger.info("Strategy node processed %d logs", len(logs))

    return {**state, "strategy": final_strategy}