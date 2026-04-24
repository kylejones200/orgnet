"""Authentication helpers (Bearer tokens, JWT claims) and role checks."""

from orgnet.auth.context import AuthContext
from orgnet.auth.rbac import AuthorizationError
from orgnet.auth.rbac import attach_role_permissions, is_allowed, require_permission
from orgnet.auth.tokens import bearer_headers, decode_jwt_unverified

__all__ = [
    "AuthContext",
    "AuthorizationError",
    "attach_role_permissions",
    "is_allowed",
    "require_permission",
    "bearer_headers",
    "decode_jwt_unverified",
]
