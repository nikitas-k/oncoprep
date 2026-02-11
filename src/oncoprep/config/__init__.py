"""Configuration files for OncoPrep segmentation models."""

from pathlib import Path

CONFIG_DIR = Path(__file__).parent


def get_config_path(filename: str) -> Path:
    """Get path to a configuration file."""
    return CONFIG_DIR / filename
