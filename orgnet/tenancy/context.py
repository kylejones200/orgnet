"""Request- or task-scoped tenant identifier (async-safe via contextvars)."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional

_tenant_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "orgnet_tenant_id", default=None
)


def current_tenant_id() -> Optional[str]:
    """Return the active tenant id for this context, if any."""
    return _tenant_id.get()


@contextmanager
def tenant_scope(tenant_id: Optional[str]) -> Iterator[None]:
    """
    Set ``current_tenant_id()`` for the duration of the block.

    Pass ``None`` to clear the tenant for nested scopes.
    """
    token = _tenant_id.set(tenant_id)
    try:
        yield
    finally:
        _tenant_id.reset(token)
