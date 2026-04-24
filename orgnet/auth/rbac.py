"""Simple role-to-permission checks (host apps extend the permission map)."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Set

from orgnet.auth.context import AuthContext

# Reasonable defaults for ONA-style deployments; override or extend at runtime.
DEFAULT_ROLE_PERMISSIONS: Dict[str, frozenset[str]] = {
    "viewer": frozenset({"graph.read", "metrics.read"}),
    "analyst": frozenset(
        {
            "graph.read",
            "metrics.read",
            "export.run",
            "insights.read",
        }
    ),
    "admin": frozenset(
        {
            "graph.read",
            "graph.write",
            "metrics.read",
            "export.run",
            "insights.read",
            "admin.manage",
        }
    ),
}


class AuthorizationError(Exception):
    """Raised when an action is not permitted for the current :class:`AuthContext`."""


def attach_role_permissions(
    mapping: Mapping[str, Iterable[str]],
    *,
    replace: bool = False,
) -> None:
    """
    Register additional role → permissions (mutates ``DEFAULT_ROLE_PERMISSIONS``).

    Args:
        mapping: role name to iterable of permission strings.
        replace: If True, replace the entire map; otherwise update entries.
    """
    if replace:
        DEFAULT_ROLE_PERMISSIONS.clear()
    for role, perms in mapping.items():
        DEFAULT_ROLE_PERMISSIONS[role] = frozenset(perms)


def _permissions_for_roles(
    roles: Iterable[str],
    role_permissions: Optional[Mapping[str, frozenset[str]]] = None,
) -> Set[str]:
    rp: MutableMapping[str, frozenset[str]] = dict(role_permissions or DEFAULT_ROLE_PERMISSIONS)
    out: Set[str] = set()
    for r in roles:
        out.update(rp.get(r, frozenset()))
    return out


def is_allowed(
    ctx: AuthContext,
    permission: str,
    *,
    role_permissions: Optional[Mapping[str, frozenset[str]]] = None,
) -> bool:
    """Return True if ``ctx`` holds ``permission`` via any of its roles."""
    return permission in _permissions_for_roles(ctx.roles, role_permissions)


def require_permission(
    ctx: AuthContext,
    permission: str,
    *,
    role_permissions: Optional[Mapping[str, frozenset[str]]] = None,
) -> None:
    """Raise :class:`AuthorizationError` if the permission is missing."""
    if not is_allowed(ctx, permission, role_permissions=role_permissions):
        raise AuthorizationError(
            f"Subject {ctx.subject!r} lacks permission {permission!r} (roles={sorted(ctx.roles)})"
        )
