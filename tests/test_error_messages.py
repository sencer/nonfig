"""Test error message quality and debugging information."""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError
import pytest

from nonfig import Ge, Hyper, Le, Pattern, configurable


def test_validation_error_contains_field_name() -> None:
  """Test that validation errors include parameter names."""

  @configurable
  def process(
    period: Hyper[int, Ge[2], Le[100]] = 14,
  ) -> int:
    return period

  with pytest.raises(ValidationError) as exc_info:
    process.Config(period=1)

  error_str = str(exc_info.value)
  # Error should mention the field name
  assert "period" in error_str.lower()


def test_validation_error_shows_constraint() -> None:
  """Test that validation errors show which constraint failed."""

  @configurable
  def process(
    value: Hyper[int, Ge[10]] = 20,
  ) -> int:
    return value

  with pytest.raises(ValidationError) as exc_info:
    process.Config(value=5)

  error_str = str(exc_info.value)
  # Error should mention the constraint
  assert "10" in error_str or "greater" in error_str.lower()


def test_multiple_validation_errors() -> None:
  """Test that multiple field errors are reported together."""

  @configurable
  def process(
    period: Hyper[int, Ge[1]] = 14,
    alpha: Hyper[float, Ge[0.0], Le[1.0]] = 0.5,
  ) -> float:
    return period * alpha

  # Both fields invalid
  with pytest.raises(ValidationError) as exc_info:
    process.Config(period=-5, alpha=2.0)

  error_str = str(exc_info.value)
  # Should mention both fields
  assert "period" in error_str.lower()
  assert "alpha" in error_str.lower()


def test_pattern_error_shows_pattern() -> None:
  """Test that pattern validation error shows the expected pattern."""

  @configurable
  def process(
    code: Hyper[str, Pattern[Literal["^[A-Z]{3}$"]]] = "USD",
  ) -> str:
    return code

  with pytest.raises(ValidationError) as exc_info:
    process.Config(code="invalid")

  error_str = str(exc_info.value)
  # Error should show the pattern or mention pattern matching
  assert "pattern" in error_str.lower() or "[A-Z]{3}" in error_str


def test_missing_required_param_error_is_clear() -> None:
  """Test that missing required param error message is helpful."""

  @configurable
  def process(period: Hyper[int]) -> int:
    return period

  with pytest.raises(ValidationError) as exc_info:
    process.Config()  # Missing required param

  error_msg = str(exc_info.value)
  # Should mention the field is required
  assert "period" in error_msg.lower()
  assert "required" in error_msg.lower() or "missing" in error_msg.lower()


def test_constraint_validation_at_decoration_time() -> None:
  """Test that invalid constraints are caught at creation time."""
  from nonfig import Ge

  # Invalid constraint value type should fail immediately
  with pytest.raises(TypeError, match="numeric value"):
    Ge["invalid"]


def test_invalid_regex_pattern_caught_early() -> None:
  """Test that invalid regex patterns are caught at decoration time."""
  from nonfig.constraints import InvalidPatternError

  with pytest.raises(InvalidPatternError, match="Invalid regex"):
    Pattern["[invalid"]  # Unclosed bracket - caught at Pattern creation
