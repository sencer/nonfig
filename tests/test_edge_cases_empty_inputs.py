"""Test edge cases with empty, None, and null inputs."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest

from nonfig import Hyper, MinLen, Pattern, configurable


def test_empty_string_with_minlen_constraint() -> None:
  """Test that empty string fails MinLen constraint."""

  @configurable
  def process(text: Hyper[str, MinLen[1]] = "default") -> str:
    return text.upper()

  # Empty string should fail validation
  with pytest.raises(ValidationError, match="at least 1"):
    process.Config(text="")


def test_empty_list_parameter() -> None:
  """Test function with empty list as default."""

  @configurable
  def process(data: list[float], items: Hyper[list[int]] = None) -> float:
    if items is None:
      items = []
    return sum(data) + len(items)

  # Should work with empty list default
  config = process.Config(items=[])
  assert config.items == []

  fn = config.make()
  result = fn(data=[1.0, 2.0])
  assert result == 3.0


def test_none_value_rejection() -> None:
  """Test that None is rejected for non-optional Hyper parameters."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value * 2

  # None should fail validation
  with pytest.raises(ValidationError):
    process.Config(value=None)


def test_optional_hyper_parameter() -> None:
  """Test Hyper parameter with Optional type."""

  @configurable
  def process(
    data: list[float],
    threshold: Hyper[float | None] = None,
  ) -> bool:
    if threshold is None:
      return True
    return max(data) > threshold

  # Should accept None
  config = process.Config(threshold=None)
  assert config.threshold is None

  # Should accept value
  config2 = process.Config(threshold=10.0)
  assert config2.threshold == 10.0


def test_empty_pandas_series() -> None:
  """Test handling of empty pandas Series."""

  @configurable
  def process(
    data: pd.Series,
    multiplier: Hyper[float] = 2.0,
  ) -> float:
    if data.empty:
      return 0.0
    return float(data.sum()) * multiplier

  config = process.Config(multiplier=3.0)
  fn = config.make()

  # Empty series should return 0.0
  result = fn(data=pd.Series([], dtype=float))
  assert result == 0.0


def test_whitespace_only_string_with_pattern() -> None:
  """Test that whitespace-only string fails pattern constraint."""

  @configurable
  def process(
    code: Hyper[str, Pattern[Literal["^[A-Z]{3}$"]]] = "USD",
  ) -> str:
    return code

  # Whitespace should fail pattern
  with pytest.raises(ValidationError):
    process.Config(code="   ")

  # Empty string should fail pattern
  with pytest.raises(ValidationError):
    process.Config(code="")


def test_empty_dict_parameter() -> None:
  """Test function with empty dict as default."""

  @configurable
  def process(
    data: int,
    mapping: Hyper[dict[str, int]] = None,
  ) -> int:
    if mapping is None:
      mapping = {}
    return data + sum(mapping.values())

  # Should work with empty dict
  config = process.Config(mapping={})
  assert config.mapping == {}

  fn = config.make()
  result = fn(data=10)
  assert result == 10


def test_nan_in_optional_float() -> None:
  """Test handling of NaN for optional float parameter."""

  @configurable
  def process(
    data: list[float],
    fill_value: Hyper[float | None] = None,
  ) -> list[float]:
    if fill_value is None:
      return data
    return [fill_value if np.isnan(x) else x for x in data]

  # None should work
  config1 = process.Config(fill_value=None)
  assert config1.fill_value is None

  # NaN should work (it's a valid float)
  config2 = process.Config(fill_value=float("nan"))
  assert np.isnan(config2.fill_value)
