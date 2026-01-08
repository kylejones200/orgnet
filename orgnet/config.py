"""Configuration management for ONA platform."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


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
        self._validate_config()

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

    @property
    def config_dict(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config

    def get_column_mapping(self, source: str) -> Dict[str, str]:
        """
        Get column mapping for a data source.

        Args:
            source: Data source name (e.g., 'email', 'slack', 'hris')

        Returns:
            Dictionary mapping standard column names to actual column names
        """
        mapping = self.get(f"data_sources.{source}.column_mapping", {})
        return mapping

    def _validate_config(self):
        """Validate configuration and raise errors for missing required fields."""
        errors = []

        # Validate data sources have required fields
        data_sources = self.get("data_sources", {})
        for source, config in data_sources.items():
            if config.get("enabled", False):
                # Check for required column mappings if specified
                if "column_mapping" in config:
                    required_columns = self._get_required_columns(source)
                    mapping = config.get("column_mapping", {})
                    missing = [col for col in required_columns if col not in mapping]
                    if missing:
                        errors.append(
                            f"Data source '{source}' missing required column mappings: {missing}"
                        )

        # Validate graph config
        graph_config = self.graph_config
        if not graph_config:
            errors.append("Graph configuration is missing")

        # Validate analysis config
        analysis_config = self.analysis_config
        if not analysis_config:
            errors.append("Analysis configuration is missing")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _get_required_columns(self, source: str) -> List[str]:
        """
        Get required columns for a data source.

        Args:
            source: Data source name

        Returns:
            List of required column names
        """
        required = {
            "email": ["sender_id", "receiver_id", "timestamp"],
            "slack": ["sender_id", "receiver_id", "timestamp"],
            "calendar": ["organizer_id", "attendee_id", "start_time"],
            "hris": ["person_id"],
            "code": ["author_id", "timestamp"],
            "documents": ["author_id", "timestamp"],
        }
        return required.get(source, [])
