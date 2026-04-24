"""Bearer headers and optional JWT decoding."""

from __future__ import annotations

from typing import Any, Dict


def bearer_headers(access_token: str) -> Dict[str, str]:
    """Return ``Authorization: Bearer …`` headers for HTTP clients."""
    if not access_token or not str(access_token).strip():
        raise ValueError("access_token is required")
    return {"Authorization": f"Bearer {str(access_token).strip()}"}


def decode_jwt_unverified(token: str) -> Dict[str, Any]:
    """
    Decode JWT claims **without** signature verification.

    Intended only for trusted internal debugging or when verification is handled
    elsewhere. For production verification use PyJWT with keys (``orgnet[auth]``).
    """
    try:
        import jwt
    except ImportError as e:  # pragma: no cover - exercised when PyJWT missing
        raise ImportError("Install PyJWT: pip install 'orgnet[auth]'") from e

    return jwt.decode(
        token,
        options={"verify_signature": False},
        algorithms=["HS256", "RS256", "ES256"],
    )
