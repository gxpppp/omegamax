"""Configuration loading utilities for omigamax.

Loads YAML configuration (config/default.yaml) and provides path helpers.
All file I/O uses explicit ``encoding="utf-8"``.
"""

from pathlib import Path

import yaml

DEFAULT_CONFIG_FILE = "default.yaml"


def get_default_config_path() -> Path:
    """Return the absolute path to the bundled config/default.yaml."""
    return Path(__file__).resolve().parent.parent / "config" / DEFAULT_CONFIG_FILE


def load_config(path: "str | Path | None" = None) -> dict:
    """Load a YAML config file into a dict (defaults to config/default.yaml)."""
    config_path = Path(path) if path is not None else get_default_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
