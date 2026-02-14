"""Configuration utilities."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {output_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Configuration to override base

    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged
