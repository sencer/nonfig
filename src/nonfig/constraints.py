"""Constraint marker classes for Pydantic Field validation.

This module provides type-safe constraint markers that can be used in type annotations
with the Hyper type system. Each constraint class uses __class_getitem__ to create
runtime constraint objects (from annotated_types) while appearing as valid type
expressions to type checkers.

Example:
    from nonfig.constraints import Ge, Le
    from nonfig import Hyper

    def my_function(
        period: Hyper[int, Ge[2], Le[100]] = 14,
        threshold: Hyper[float, Ge[0.0], Le[1.0]] = 0.5,
    ) -> float:
        return period * threshold
"""

import re
from typing import Any

from annotated_types import (
  Ge as _Ge,
  Gt as _Gt,
  Le as _Le,
  Lt as _Lt,
  MaxLen as _MaxLen,
  MinLen as _MinLen,
  MultipleOf as _MultipleOf,
)

__all__ = [
  "Ge",
  "Gt",
  "InvalidPatternError",
  "Le",
  "Lt",
  "MaxLen",
  "MinLen",
  "MultipleOf",
  "Pattern",
  "PatternConstraint",
  "validate_constraint_conflicts",
]


class InvalidPatternError(ValueError):
  """Raised when a regex pattern is invalid."""

  pass


def _check_numeric(value: Any, name: str) -> None:
  """Validate that a constraint value is numeric."""
  if not isinstance(value, (int, float)):
    raise TypeError(f"{name} requires a numeric value, got {type(value).__name__}")


def _check_int(value: Any, name: str) -> None:
  """Validate that a constraint value is an integer."""
  if not isinstance(value, int):
    raise TypeError(f"{name} requires an integer value, got {type(value).__name__}")


def _check_non_negative_int(value: Any, name: str) -> None:
  """Validate that a constraint value is a non-negative integer."""
  _check_int(value, name)
  if value < 0:
    raise ValueError(f"{name} requires a non-negative integer, got {value}")


class Ge:
  """Greater than or equal constraint marker (for numbers).

  Usage: Ge[value] in type annotations creates a constraint that the
  runtime value must be >= the specified value.

  Example:
      age: Hyper[int, Ge[0]] = 25  # age must be >= 0
  """

  __slots__ = ()

  def __class_getitem__(cls, value: Any) -> _Ge:
    """Support Ge[2] syntax."""
    _check_numeric(value, "Ge")
    return _Ge(value)


class Gt:
  """Greater than constraint marker (for numbers).

  Usage: Gt[value] in type annotations creates a constraint that the
  runtime value must be > the specified value (strict inequality).

  Example:
      epsilon: Hyper[float, Gt[0.0]] = 1e-9  # epsilon must be > 0
  """

  __slots__ = ()

  def __class_getitem__(cls, value: Any) -> _Gt:
    """Support Gt[0] syntax."""
    _check_numeric(value, "Gt")
    return _Gt(value)


class Le:
  """Less than or equal constraint marker (for numbers).

  Usage: Le[value] in type annotations creates a constraint that the
  runtime value must be <= the specified value.

  Example:
      percentage: Hyper[float, Ge[0.0], Le[100.0]] = 50.0
  """

  __slots__ = ()

  def __class_getitem__(cls, value: Any) -> _Le:
    """Support Le[100] syntax."""
    _check_numeric(value, "Le")
    return _Le(value)


class Lt:
  """Less than constraint marker (for numbers).

  Usage: Lt[value] in type annotations creates a constraint that the
  runtime value must be < the specified value (strict inequality).

  Example:
      probability: Hyper[float, Ge[0.0], Lt[1.0]] = 0.5
  """

  __slots__ = ()

  def __class_getitem__(cls, value: Any) -> _Lt:
    """Support Lt[10] syntax."""
    _check_numeric(value, "Lt")
    return _Lt(value)


class MinLen:
  """Minimum length constraint marker (for strings, lists, etc.).

  Usage: MinLen[value] in type annotations creates a constraint that the
  runtime value's length must be >= the specified value.

  Example:
      name: Hyper[str, MinLen[1]] = "default"
  """

  __slots__ = ()

  def __class_getitem__(cls, value: Any) -> _MinLen:
    """Support MinLen[5] syntax."""
    _check_non_negative_int(value, "MinLen")
    return _MinLen(value)


class MaxLen:
  """Maximum length constraint marker (for strings, lists, etc.).

  Usage: MaxLen[value] in type annotations creates a constraint that the
  runtime value's length must be <= the specified value.

  Example:
      name: Hyper[str, MinLen[1], MaxLen[100]] = "default"
  """

  __slots__ = ()

  def __class_getitem__(cls, value: Any) -> _MaxLen:
    """Support MaxLen[100] syntax."""
    _check_non_negative_int(value, "MaxLen")
    return _MaxLen(value)


class MultipleOf:
  """Multiple of constraint marker (for numbers).

  Usage: MultipleOf[value] in type annotations creates a constraint that the
  runtime value must be a multiple of the specified value.

  Example:
      batch_size: Hyper[int, Ge[1], MultipleOf[8]] = 32
  """

  __slots__ = ()

  def __class_getitem__(cls, value: Any) -> _MultipleOf:
    """Support MultipleOf[5] syntax."""
    _check_numeric(value, "MultipleOf")
    if value <= 0:
      raise ValueError(f"MultipleOf requires a positive value, got {value}")
    return _MultipleOf(value)


class PatternConstraint:
  """Represents a compiled regex pattern constraint."""

  __slots__ = ("pattern",)

  def __init__(self, pattern: str) -> None:
    self.pattern = pattern


class Pattern:
  """Regex pattern constraint marker (for strings).

  Usage: Pattern[regex] in type annotations creates a constraint that the
  runtime string value must match the specified regex pattern.

  Example:
      code: Hyper[str, Pattern[r'^[A-Z]{3}$']] = "USD"
  """

  __slots__ = ()

  def __class_getitem__(cls, pattern: Any) -> PatternConstraint:
    """Support Pattern[r'^[a-z]+$'] syntax."""
    # Handle Literal types (e.g., Pattern[Literal["..."]])
    from typing import get_args, get_origin

    try:
      from typing import Literal
    except ImportError:
      pass
    else:
      if get_origin(pattern) is Literal:
        args = get_args(pattern)
        if args and isinstance(args[0], str):
          pattern = args[0]

    if not isinstance(pattern, str):
      raise TypeError(f"Pattern expects a string, got {type(pattern).__name__}")

    # Validate regex at decoration time
    try:
      re.compile(pattern)
    except re.error as e:
      raise InvalidPatternError(f"Invalid regex pattern: {e}") from e
    return PatternConstraint(pattern)


def validate_constraint_conflicts(
  constraints: dict[str, Any],
  param_name: str,
  func_name: str | None = None,
) -> None:
  """
  Validate that constraints don't conflict with each other.

  Args:
    constraints: Dictionary of constraint names to values
    param_name: Name of the parameter (for error messages)
    func_name: Optional function/class name (for richer error messages)

  Raises:
    ValueError: If impossible constraints are detected.
  """
  # Build context string for error messages
  if func_name:
    context = f" (in '{func_name}')"
  else:
    context = ""

  # Check numeric bound conflicts
  lower_bound = None
  upper_bound = None
  lower_exclusive = False
  upper_exclusive = False

  if "ge" in constraints:
    lower_bound = constraints["ge"]
    lower_exclusive = False
  if "gt" in constraints:
    if lower_bound is not None:
      lower_bound = max(lower_bound, constraints["gt"])
    else:
      lower_bound = constraints["gt"]
      lower_exclusive = True

  if "le" in constraints:
    upper_bound = constraints["le"]
    upper_exclusive = False
  if "lt" in constraints:
    if upper_bound is not None:
      upper_bound = min(upper_bound, constraints["lt"])
    else:
      upper_bound = constraints["lt"]
      upper_exclusive = True

  if lower_bound is not None and upper_bound is not None:
    if lower_bound > upper_bound:
      raise ValueError(
        f"Conflicting constraints for '{param_name}'{context}: lower bound ({lower_bound}) > upper bound ({upper_bound})"
      )
    if lower_bound == upper_bound and (lower_exclusive or upper_exclusive):
      raise ValueError(
        f"Conflicting constraints for '{param_name}'{context}: bounds equal at {lower_bound} but one is exclusive"
      )

  # Check length bound conflicts
  if (
    "min_length" in constraints
    and "max_length" in constraints
    and constraints["min_length"] > constraints["max_length"]
  ):
    raise ValueError(
      f"Conflicting constraints for '{param_name}'{context}: min_length ({constraints['min_length']}) > max_length ({constraints['max_length']})"
    )
