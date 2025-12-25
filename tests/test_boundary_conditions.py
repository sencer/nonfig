"""Test boundary conditions for all constraint types."""

from __future__ import annotations

import sys

from pydantic import ValidationError
import pytest

from nonfig import Ge, Gt, Hyper, Le, Lt, MaxLen, MinLen, MultipleOf, configurable


def test_float_precision_boundaries() -> None:
  """Test floating point precision at boundaries."""

  @configurable
  def process(
    value: Hyper[float, Ge[0.0], Le[1.0]] = 0.5,
  ) -> float:
    return value

  # Test exact boundary values
  config1 = process.Config(value=0.0)
  assert config1.value == 0.0

  config2 = process.Config(value=1.0)
  assert config2.value == 1.0

  # Test very close to boundary (floating point edge case)
  config3 = process.Config(value=0.9999999999999999)
  assert config3.value <= 1.0

  # Just outside boundary should fail
  with pytest.raises(ValidationError):
    process.Config(value=1.0000001)


def test_integer_overflow_boundaries() -> None:
  """Test integer boundaries at extremes."""

  @configurable
  def process(
    large_value: Hyper[int, Ge[0]] = 1000,
  ) -> int:
    return large_value

  # Test very large integers
  max_int = sys.maxsize
  config = process.Config(large_value=max_int)
  assert config.large_value == max_int


def test_zero_boundary_for_gt_constraint() -> None:
  """Test Gt constraint at zero boundary."""

  @configurable
  def process(value: Hyper[float, Gt[0.0]] = 1.0) -> float:
    return value

  # Zero should fail (strict inequality)
  with pytest.raises(ValidationError):
    process.Config(value=0.0)

  # Smallest positive float should pass
  config = process.Config(value=sys.float_info.min)
  assert config.value > 0.0


def test_negative_zero() -> None:
  """Test handling of negative zero (-0.0)."""

  @configurable
  def process(value: Hyper[float, Ge[0.0]] = 1.0) -> float:
    return value

  # -0.0 should be treated as 0.0 and pass Ge[0.0]
  config = process.Config(value=-0.0)
  assert config.value == 0.0


def test_maxlen_at_exact_limit() -> None:
  """Test MaxLen constraint at exact boundary."""

  @configurable
  def process(text: Hyper[str, MaxLen[5]] = "hello") -> str:
    return text

  # Exactly at limit should pass
  config = process.Config(text="12345")
  assert len(config.text) == 5

  # One over should fail
  with pytest.raises(ValidationError):
    process.Config(text="123456")


def test_minlen_at_exact_limit() -> None:
  """Test MinLen constraint at exact boundary."""

  @configurable
  def process(text: Hyper[str, MinLen[3]] = "abc") -> str:
    return text

  # Exactly at limit should pass
  config = process.Config(text="abc")
  assert len(config.text) == 3

  # One under should fail
  with pytest.raises(ValidationError):
    process.Config(text="ab")


def test_multiple_of_with_zero() -> None:
  """Test MultipleOf constraint with zero value."""

  @configurable
  def process(value: Hyper[int, MultipleOf[5], Ge[0]] = 10) -> int:
    return value

  # Zero is a multiple of everything
  config = process.Config(value=0)
  assert config.value == 0


def test_multiple_of_with_negative() -> None:
  """Test MultipleOf constraint with negative values."""

  @configurable
  def process(value: Hyper[int, MultipleOf[5]] = 10) -> int:
    return value

  # Negative multiples should work
  config = process.Config(value=-15)
  assert config.value == -15

  # Non-multiple should fail
  with pytest.raises(ValidationError):
    process.Config(value=-7)


def test_lt_vs_le_boundary() -> None:
  """Test difference between Lt and Le at boundary."""

  @configurable
  def with_le(value: Hyper[int, Le[10]] = 5) -> int:
    return value

  @configurable
  def with_lt(value: Hyper[int, Lt[10]] = 5) -> int:
    return value

  # Le should accept 10
  config_le = with_le.Config(value=10)
  assert config_le.value == 10

  # Lt should reject 10
  with pytest.raises(ValidationError):
    with_lt.Config(value=10)

  # Both should accept 9
  config_le2 = with_le.Config(value=9)
  config_lt2 = with_lt.Config(value=9)
  assert config_le2.value == 9
  assert config_lt2.value == 9


def test_gt_vs_ge_boundary() -> None:
  """Test difference between Gt and Ge at boundary."""

  @configurable
  def with_ge(value: Hyper[int, Ge[10]] = 15) -> int:
    return value

  @configurable
  def with_gt(value: Hyper[int, Gt[10]] = 15) -> int:
    return value

  # Ge should accept 10
  config_ge = with_ge.Config(value=10)
  assert config_ge.value == 10

  # Gt should reject 10
  with pytest.raises(ValidationError):
    with_gt.Config(value=10)

  # Both should accept 11
  config_ge2 = with_ge.Config(value=11)
  config_gt2 = with_gt.Config(value=11)
  assert config_ge2.value == 11
  assert config_gt2.value == 11


def test_combined_boundary_constraints() -> None:
  """Test multiple constraints at boundaries."""

  @configurable
  def process(
    value: Hyper[int, Ge[0], Le[100]] = 50,
  ) -> int:
    return value

  # Both boundaries should work
  config0 = process.Config(value=0)
  assert config0.value == 0

  config100 = process.Config(value=100)
  assert config100.value == 100

  # Outside boundaries should fail
  with pytest.raises(ValidationError):
    process.Config(value=-1)

  with pytest.raises(ValidationError):
    process.Config(value=101)


def test_minlen_maxlen_at_same_boundary() -> None:
  """Test MinLen and MaxLen at same value."""

  @configurable
  def process(text: Hyper[str, MinLen[5], MaxLen[5]] = "hello") -> str:
    return text

  # Exactly 5 chars should work
  config = process.Config(text="12345")
  assert len(config.text) == 5

  # 4 chars should fail
  with pytest.raises(ValidationError):
    process.Config(text="1234")

  # 6 chars should fail
  with pytest.raises(ValidationError):
    process.Config(text="123456")


def test_empty_string_with_maxlen() -> None:
  """Test that empty string passes MaxLen but may fail MinLen."""

  @configurable
  def with_maxlen_only(text: Hyper[str, MaxLen[10]] = "default") -> str:
    return text

  # Empty string should pass MaxLen
  config = with_maxlen_only.Config(text="")
  assert not config.text


def test_list_length_boundaries() -> None:
  """Test MinLen/MaxLen work on lists at boundaries."""

  @configurable
  def process(items: Hyper[list[int], MinLen[1], MaxLen[3]] = None) -> list[int]:
    if items is None:
      items = [1, 2]
    return items

  # Exactly 1 item (min boundary)
  config1 = process.Config(items=[10])
  assert len(config1.items) == 1

  # Exactly 3 items (max boundary)
  config3 = process.Config(items=[10, 20, 30])
  assert len(config3.items) == 3

  # 0 items should fail
  with pytest.raises(ValidationError):
    process.Config(items=[])

  # 4 items should fail
  with pytest.raises(ValidationError):
    process.Config(items=[10, 20, 30, 40])
