"""Microsoft Graph REST client (Bearer delegated or application token)."""

from __future__ import annotations

from typing import Any, Dict

from orgnet.integrations._http import json_request


class MicrosoftGraphClient:
    """
    Minimal Microsoft Graph v1.0 client.

    Supply an OAuth 2.0 access token (delegated or application). Token acquisition
    (MSAL, device code, client credentials) is intentionally left to the host
    application so secrets and tenant-specific policy stay outside orgnet.
    """

    def __init__(
        self,
        access_token: str,
        *,
        base_url: str = "https://graph.microsoft.com/v1.0",
    ):
        if not access_token or not access_token.strip():
            raise ValueError("access_token is required")
        self._token = access_token.strip()
        self._base = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def get(self, path: str, *, timeout: float = 60.0) -> Any:
        """GET a Graph-relative path (e.g. ``/me`` or ``/users``)."""
        if not path.startswith("/"):
            path = "/" + path
        url = self._base + path
        status, payload = json_request("GET", url, headers=self._headers(), timeout=timeout)
        if status != 200:
            raise RuntimeError(f"Graph HTTP {status}: {payload}")
        return payload

    def me(self) -> Dict[str, Any]:
        """Return the signed-in user profile (``/me``)."""
        data = self.get("/me")
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected /me response: {data!r}")
        return data

    def list_users(self, *, top: int = 25) -> Dict[str, Any]:
        """First page of users (``/users``); follow ``@odata.nextLink`` for more pages."""
        return self.get(f"/users?$top={top}")
