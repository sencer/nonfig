"""CLI runner for configurable functions and classes."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TypeVar, get_args, get_origin

from pydantic import ValidationError

from nonfig.models import ConfigValidationError

if TYPE_CHECKING:
  from collections.abc import Callable

  from nonfig.models import MakeableModel

__all__ = ["run_cli"]

T = TypeVar("T")


def _print_help(
  config_cls: type[MakeableModel[Any]],
  prefix: str = "",
  visited: set[type[Any]] | None = None,
) -> None:
  """Recursively print configuration options."""
  if visited is None:
    visited = set()

  if config_cls in visited:
    return
  visited.add(config_cls)

  for name, field in config_cls.model_fields.items():
    if name.startswith("_"):
      continue

    # Determine if it's a nested config
    field_type = field.annotation
    nested_config_cls = _get_nested_config_cls(field_type)

    # Get default value string
    default = field.default
    default_str = str(default)
    if hasattr(default, "__repr__"):
      default_str = default.__repr__()

    key = f"{prefix}{name}"

    if nested_config_cls:
      print(f"{key} (nested config):")
      _print_help(nested_config_cls, prefix=f"{key}.", visited=visited)
    else:
      type_name = str(field_type)
      # Clean up type name for better readability
      type_name = type_name.replace("typing.", "").replace("nonfig.typedefs.", "")
      print(f"{key} [{type_name}] = {default_str}")


def _parse_value(value_str: str, target_type: type[Any] | None) -> Any:
  """Parse a string value into the appropriate type.

  Args:
    value_str: String value from CLI.
    target_type: Target type for coercion.

  Returns:
    Coerced value.
  """
  # Handle None type
  if value_str.lower() == "none":
    return None

  # Handle booleans
  if value_str.lower() in ("true", "yes", "1"):
    return True
  if value_str.lower() in ("false", "no", "0"):
    return False

  # Try type coercion if target type is known
  if target_type is not None:
    origin = get_origin(target_type)

    # Handle Optional/Union - try the first non-None type
    if origin is not None:
      args = get_args(target_type)
      for arg in args:
        if arg is not type(None):
          return _parse_value(value_str, arg)

    # Direct type coercion
    if target_type in (int, float, str, bool):
      try:
        return target_type(value_str)
      except (ValueError, TypeError):
        pass

  # Try numeric coercion
  try:
    if "." in value_str:
      return float(value_str)
    return int(value_str)
  except ValueError:
    pass

  # Fall back to string
  return value_str


def _parse_overrides(args: list[str]) -> dict[str, Any]:
  """Parse CLI arguments into a nested dictionary.

  Args:
    args: List of "key=value" strings. Dot notation supported.

  Returns:
    Nested dictionary of overrides.
  """
  result: dict[str, Any] = {}

  for arg in args:
    if "=" not in arg:
      continue

    key, value = arg.split("=", 1)
    parts = key.split(".")

    # Build nested dict
    current = result
    for part in parts[:-1]:
      if part not in current:
        current[part] = {}
      current = current[part]

    current[parts[-1]] = value

  return result


def _get_nested_config_cls(field_type: type[Any] | None) -> type[Any] | None:
  """Get nested Config class from a field type if available."""
  if field_type is None:
    return None

  # Check if it's a union with a Config class
  origin = get_origin(field_type)
  if origin is not None:
    for arg in get_args(field_type):
      if hasattr(arg, "model_fields"):
        return arg
    return None

  # Direct Config class
  if hasattr(field_type, "model_fields"):
    return field_type

  return None


def _apply_type_coercion(
  overrides: dict[str, Any],
  config_cls: type[Any],
) -> dict[str, Any]:
  """Apply type coercion based on Config field types.

  Args:
    overrides: Raw overrides with string values.
    config_cls: The Config class to get field types from.

  Returns:
    Overrides with coerced values.
  """
  result: dict[str, Any] = {}

  for key, value in overrides.items():
    # Get field type from Config
    field_type: type[Any] | None = None
    if hasattr(config_cls, "model_fields") and key in config_cls.model_fields:
      field_type = config_cls.model_fields[key].annotation

    if isinstance(value, dict):
      nested_config_cls = _get_nested_config_cls(field_type)
      if nested_config_cls is not None:
        from typing import cast

        result[key] = _apply_type_coercion(
          cast("dict[str, Any]", value), nested_config_cls
        )
      else:
        result[key] = value
    elif isinstance(value, str):
      result[key] = _parse_value(value, field_type)
    else:
      result[key] = value

  return result


def run_cli(
  target: Callable[..., T] | type[T],
  args: list[str] | None = None,
) -> T:
  """Run a configurable target with CLI overrides.

  Usage:
    ```python
    from nonfig import configurable, Hyper, run_cli

    @configurable
    def train(*, epochs: Hyper[int] = 10, lr: Hyper[float] = 0.01) -> dict:
        return {"epochs": epochs, "lr": lr}

    if __name__ == "__main__":
        result = run_cli(train)
        print(result)
    ```

    ```bash
    python train.py epochs=100 lr=0.001
    ```

  Args:
    target: A @configurable function or class.
    args: CLI arguments (defaults to sys.argv[1:]).

  Returns:
    Result of calling target (for functions) or instance (for classes).

  Raises:
    TypeError: If target is not configurable.
  """
  # Default to sys.argv[1:]
  if args is None:
    args = sys.argv[1:]

  # Validate target is configurable
  config_cls = getattr(target, "Config", None)
  if config_cls is None:
    raise TypeError(f"Target {target} is not configurable (no .Config attribute)")

  # Parse CLI arguments
  # Check for help
  if any(arg in ("--help", "-h") for arg in args):
    print(f"Configuration for {target.__name__}:")
    # Cast to MakeableModel type for the helper
    from typing import cast

    from nonfig.models import MakeableModel

    if issubclass(config_cls, MakeableModel):
      _print_help(cast("type[MakeableModel[Any]]", config_cls))
    else:
      # Fallback for non-MakeableModel configs (shouldn't happen with @configurable)
      print("  (Help not available for this config type)")
    sys.exit(0)

  raw_overrides = _parse_overrides(args)

  # Apply type coercion
  typed_overrides = _apply_type_coercion(raw_overrides, config_cls)

  try:
    # Create config and make instance
    config = config_cls(**typed_overrides)
    result: T = config.make()
    return result
  except ValidationError as e:
    # Wrap and print user-friendly error
    readable_error = ConfigValidationError(e, config_cls.__name__)
    print(f"\n{readable_error}", file=sys.stderr)
    sys.exit(1)
