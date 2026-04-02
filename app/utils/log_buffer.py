"""In-memory ring buffer for capturing recent log entries.

Adds a loguru sink that stores the last N log records so they can be
served via the /logs API endpoint — useful on Railway where you may
not want to open the platform dashboard just to check recent activity.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone


class LogBuffer:
    """Thread-safe ring buffer that stores recent log records."""

    def __init__(self, max_size: int = 2000) -> None:
        self._buffer: deque[dict] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def sink(self, message) -> None:
        """Loguru sink callable — receives each log message."""
        record = message.record
        entry = {
            "timestamp": record["time"].astimezone(timezone.utc).isoformat(),
            "level": record["level"].name,
            "module": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
        }
        if record["exception"] is not None:
            entry["exception"] = str(record["exception"])
        with self._lock:
            self._buffer.append(entry)

    def get_entries(
        self,
        limit: int = 200,
        level: str | None = None,
        search: str | None = None,
    ) -> list[dict]:
        """Return recent log entries, newest first.

        Args:
            limit: Max entries to return.
            level: Filter by minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
            search: Case-insensitive substring match on message text.
        """
        level_order = {
            "TRACE": 0, "DEBUG": 1, "INFO": 2, "SUCCESS": 3,
            "WARNING": 4, "ERROR": 5, "CRITICAL": 6,
        }
        min_level = level_order.get((level or "").upper(), 0)

        with self._lock:
            entries = list(self._buffer)

        # Filter
        if min_level > 0:
            entries = [e for e in entries if level_order.get(e["level"], 0) >= min_level]
        if search:
            needle = search.lower()
            entries = [e for e in entries if needle in e["message"].lower()]

        # Newest first, capped
        entries.reverse()
        return entries[:limit]


# Singleton instance
log_buffer = LogBuffer()
