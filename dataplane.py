#!/usr/bin/env python3
"""AI Data Plane integration helpers (phase 2).

Thin layer over the real `couchbase-agent-memory` SDK + the native memory server, used by:
- the RAG scripts, to recall prior context and remember new turns (when the global toggle is on);
- app.py, to report live status/usage for the diagram overlay and to flip the toggle.

Everything is best-effort: if the memory server is down or the SDK isn't installed, the helpers
degrade gracefully so the plain RAG path always still works.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import config

DEFAULT_RECALL_K = 5


def _client(settings: "config.Settings"):
    from agentmemory import AgentMemoryClient
    return AgentMemoryClient(base_url=settings.agent_memory_base_url)


def ensure_session(client, user_id: str, session_id: str) -> None:
    """Get-or-create the user and session so add/search have somewhere to write."""
    from agentmemory.exceptions import NotFoundError, ConflictError
    try:
        user = client.get_user(user_id)
    except NotFoundError:
        user = client.create_user(user_id, name=user_id)
    try:
        user.create_session(session_id)
    except ConflictError:
        pass


def recall(settings: "config.Settings", user_id: str, session_id: str, query: str,
           k: int = DEFAULT_RECALL_K) -> Tuple[str, int]:
    """Return (context_text, n_blocks) of relevant prior memory for the query. Empty on any error."""
    try:
        with _client(settings) as client:
            ensure_session(client, user_id, session_id)
            sess = client.get_user(user_id).get_session(session_id)
            res = sess.search_memory(query=query, filters={"relevant_k": k})
            lines = []
            for b in res.memory_blocks:
                if b.message:
                    txt = " ".join(x for x in (b.message.user_content, b.message.assistant_content) if x)
                elif b.fact:
                    txt = b.fact
                else:
                    continue
                if txt.strip():
                    lines.append(f"- {txt.strip()}")
            return ("\n".join(lines), len(lines))
    except Exception:  # noqa: BLE001 - memory is best-effort; never break RAG
        return ("", 0)


def remember(settings: "config.Settings", user_id: str, session_id: str,
             user_content: str, assistant_content: str) -> bool:
    """Store a conversation turn as memory. Returns True on success, False on any error."""
    try:
        with _client(settings) as client:
            ensure_session(client, user_id, session_id)
            sess = client.get_user(user_id).get_session(session_id)
            sess.add_memory(messages=[{
                "user_content": user_content or "",
                "assistant_content": assistant_content or "",
            }])
            return True
    except Exception:  # noqa: BLE001
        return False


def memory_health(settings: "config.Settings") -> Tuple[bool, str]:
    """(active, detail) for the agent-memory server via the SDK's health check."""
    try:
        with _client(settings) as client:
            status = client.health_ping().overall_status.value
            return (status == "healthy", f"server healthy ({status})")
    except Exception as e:  # noqa: BLE001
        return (False, f"unreachable at {settings.agent_memory_base_url}: {e}")


def memory_stats(settings: "config.Settings") -> dict:
    """Live usage counters from the memory server's /stats endpoint ({} on error)."""
    try:
        import httpx
        r = httpx.get(settings.agent_memory_base_url.rstrip("/") + "/stats", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return {}


def status(settings: "config.Settings") -> dict:
    """Aggregate status for the diagram overlay: toggle + memory active/used/tokens-served."""
    active, detail = memory_health(settings)
    stats = memory_stats(settings) if active else {}
    return {
        "enabled": settings.ai_dataplane_enabled,
        "capabilities": {
            "agent_memory": {
                "active": active,
                "detail": detail,
                "base_url": settings.agent_memory_base_url,
                "used": bool(stats.get("memory_added") or stats.get("searches")),
                "memory_added": stats.get("memory_added", 0),
                "searches": stats.get("searches", 0),
                "blocks_recalled": stats.get("blocks_recalled", 0),
                "tokens_served_from_memory_est": stats.get("tokens_served_from_memory_est", 0),
            },
        },
    }
