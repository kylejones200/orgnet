"""Configuration management for ONA platform."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for ONA platform."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file.
                        If None, uses default config.yaml in project root.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'data_sources.email.enabled')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_data_source_config(self, source: str) -> Dict[str, Any]:
        """Get configuration for a specific data source."""
        return self.get(f"data_sources.{source}", {})

    def is_data_source_enabled(self, source: str) -> bool:
        """Check if a data source is enabled."""
        return self.get(f"data_sources.{source}.enabled", False)

    @property
    def graph_config(self) -> Dict[str, Any]:
        """Get graph construction configuration."""
        return self.get("graph", {})

    @property
    def analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return self.get("analysis", {})

    @property
    def privacy_config(self) -> Dict[str, Any]:
        """Get privacy configuration."""
        return self.get("privacy", {})

    @property
    def api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get("api", {})
