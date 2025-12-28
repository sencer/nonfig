"""Optional config loaders for YAML, TOML, and JSON formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["load_json", "load_toml", "load_yaml"]


def load_json(path: str | Path) -> dict[str, Any]:
  """Load config from JSON file.

  Args:
    path: Path to the JSON file.

  Returns:
    Parsed configuration dictionary.
  """
  with Path(path).open(encoding="utf-8") as f:
    result: dict[str, Any] = json.load(f)
    return result


def load_toml(path: str | Path) -> dict[str, Any]:
  """Load config from TOML file.

  Uses tomllib from Python 3.11+ stdlib.

  Args:
    path: Path to the TOML file.

  Returns:
    Parsed configuration dictionary.
  """
  import tomllib

  with Path(path).open("rb") as f:
    return tomllib.load(f)


def load_yaml(path: str | Path) -> dict[str, Any]:
  """Load config from YAML file.

  Requires optional PyYAML dependency.

  Args:
    path: Path to the YAML file.

  Returns:
    Parsed configuration dictionary.

  Raises:
    ImportError: If PyYAML is not installed.
  """
  try:
    import yaml
  except ImportError as e:
    raise ImportError(
      "PyYAML is required for YAML support. Install with: pip install pyyaml"
    ) from e

  with Path(path).open(encoding="utf-8") as f:
    result: dict[str, Any] = yaml.safe_load(f)
    return result
