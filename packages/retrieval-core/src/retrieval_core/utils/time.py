"""UTC timestamp helpers used by run and sweep metadata."""

from datetime import datetime, timezone


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""

    return datetime.now(timezone.utc).isoformat()


def utc_timestamp() -> str:
    """Return a compact, sortable UTC timestamp for identifiers."""

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
