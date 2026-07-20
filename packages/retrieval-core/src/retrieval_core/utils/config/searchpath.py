"""Per-composition state shared with the Hydra search-path plugin."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

_ACTIVE_FALLBACK_PATHS: ContextVar[tuple[tuple[str, str], ...] | None] = ContextVar(
    "retrieval_core_config_fallbacks",
    default=None,
)


@contextmanager
def use_config_fallbacks(paths: Sequence[tuple[str, str]]) -> Iterator[None]:
    """Temporarily configure fallback paths for one Hydra composition."""

    context = _ACTIVE_FALLBACK_PATHS
    token = context.set(tuple(paths))
    try:
        yield
    finally:
        context.reset(token)


def active_config_fallbacks() -> tuple[tuple[str, str], ...] | None:
    """Return the fallback paths for the active Hydra composition."""

    return _ACTIVE_FALLBACK_PATHS.get()
