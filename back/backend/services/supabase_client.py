"""
Supabase client singleton.

Required environment variables:
    SUPABASE_URL   – project URL  (e.g. https://xxxx.supabase.co)
    SUPABASE_KEY   – service-role or anon key

The 'supabase' package is an optional runtime dependency.
ImportError is deferred until get_supabase() is actually called.
"""
from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None


def get_supabase() -> Any:
    """Return a cached Supabase client, creating it on first call."""
    global _client
    if _client is None:
        try:
            from supabase import create_client
        except ImportError as exc:
            raise ImportError(
                "The 'supabase' package is required. Install it with: pip install supabase"
            ) from exc

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set."
            )
        _client = create_client(url, key)
        logger.info("Supabase client initialised.")
    return _client


class _SupabaseProxy:
    """Transparent proxy – initialises the real client on first attribute access."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_supabase(), name)


# Module-level alias: `from services.supabase_client import supabase`
supabase = _SupabaseProxy()
