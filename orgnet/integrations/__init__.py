"""HTTP clients for enterprise collaboration APIs (Slack, Microsoft Graph)."""

from orgnet.integrations.microsoft_graph import MicrosoftGraphClient
from orgnet.integrations.slack import SlackWebClient

__all__ = ["SlackWebClient", "MicrosoftGraphClient"]
