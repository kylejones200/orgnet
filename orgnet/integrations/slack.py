"""Slack Web API client (Bearer bot token)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from orgnet.integrations._http import json_request


class SlackWebClient:
    """
    Thin wrapper around Slack's ``api.*`` methods.

    Pass a bot token from a Slack app (``xoxb-...``). OAuth installation flows
    are out of scope here; obtain a token via Slack app management UI or your
    identity provider, then store it securely (e.g. environment / secret manager).
    """

    def __init__(self, bot_token: str, *, base_url: str = "https://slack.com/api"):
        if not bot_token or not bot_token.strip():
            raise ValueError("bot_token is required")
        self._token = bot_token.strip()
        self._base = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def call(self, method: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke a Slack Web API method (POST with JSON body).

        Args:
            method: Method name without ``api.`` prefix, e.g. ``auth.test``.
            params: Optional JSON body fields.
        """
        url = f"{self._base}/{method}"
        body = dict(params or {})
        status, payload = json_request("POST", url, headers=self._headers(), json_body=body)
        if status != 200:
            raise RuntimeError(f"Slack HTTP {status}: {payload}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Slack response: {payload!r}")
        if not payload.get("ok", False):
            err = payload.get("error", "unknown_error")
            raise RuntimeError(f"Slack API error: {err}")
        return payload

    def auth_test(self) -> Dict[str, Any]:
        """Verify the token and return workspace metadata."""
        return self.call("auth.test")

    def conversations_history(
        self,
        channel: str,
        *,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Fetch a page of messages from a channel (``conversations.history``)."""
        params: Dict[str, Any] = {"channel": channel, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self.call("conversations.history", params=params)
