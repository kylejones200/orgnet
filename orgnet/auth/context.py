"""Principal / identity context for authorization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AuthContext:
    """
    Represents an authenticated subject (user or service) inside your app.

    ``roles`` are opaque strings (e.g. ``analyst``, ``admin``) mapped to
    permissions in :mod:`orgnet.auth.rbac`.
    """

    subject: str
    roles: frozenset[str]
    tenant_id: str | None = None
    claims: Mapping[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        return role in self.roles
