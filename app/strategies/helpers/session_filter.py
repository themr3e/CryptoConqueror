"""Forex trading session detection utilities.

Determines which trading sessions (Asian, London, New York, Overlap) are active
for a given UTC timestamp.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Session hours in UTC (start_hour, end_hour)
_SESSION_HOURS: dict[str, tuple[int, int]] = {
    "asian": (0, 9),       # 00:00 – 09:00 UTC
    "london": (7, 16),     # 07:00 – 16:00 UTC
    "new_york": (12, 21),  # 12:00 – 21:00 UTC
}

_OVERLAP_START = 12  # London/NY overlap starts 12:00 UTC
_OVERLAP_END = 16    # London/NY overlap ends   16:00 UTC


def get_active_sessions(timestamp: datetime | None = None) -> list[str]:
    """Return list of active session names for the given UTC timestamp.

    The special value ``"overlap"`` is included when both London and New York
    are simultaneously open (12:00–16:00 UTC).

    Args:
        timestamp: UTC-aware datetime; defaults to ``datetime.now(timezone.utc)``.

    Returns:
        List of active session names, e.g. ``["london", "new_york", "overlap"]``.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    hour = timestamp.hour

    active: list[str] = []

    for name, (start, end) in _SESSION_HOURS.items():
        if start <= hour < end:
            active.append(name)

    # Overlap = London + New York simultaneously open
    if _OVERLAP_START <= hour < _OVERLAP_END:
        active.append("overlap")

    return active


def is_trading_hours(timestamp: datetime | None = None) -> bool:
    """Return True if any major session is active."""
    return bool(get_active_sessions(timestamp))


def get_primary_session(timestamp: datetime | None = None) -> str:
    """Return the primary active session name, or 'off_hours'."""
    sessions = get_active_sessions(timestamp)
    if not sessions:
        return "off_hours"
    # Overlap takes priority
    if "overlap" in sessions:
        return "overlap"
    return sessions[0]
