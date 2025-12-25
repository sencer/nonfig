"""Test type coercion and type mismatch handling."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from nonfig import Ge, Hyper, Le, configurable


def test_string_to_int_coercion() -> None:
  """Test that numeric strings are coerced to int."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value * 2

  # Pydantic should coerce string to int
  config = process.Config(value="42")
  assert config.value == 42
  assert isinstance(config.value, int)


def test_invalid_string_to_int_fails() -> None:
  """Test that non-numeric strings fail validation."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value

  with pytest.raises(ValidationError):
    process.Config(value="not_a_number")


def test_float_to_int_coercion() -> None:
  """Test float to int coercion behavior."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value

  # Pydantic should coerce clean float to int
  config = process.Config(value=42.0)
  assert config.value == 42

  # Fractional float should fail
  with pytest.raises(ValidationError):
    process.Config(value=42.5)


def test_int_to_float_coercion() -> None:
  """Test int to float coercion."""

  @configurable
  def process(value: Hyper[float] = 1.0) -> float:
    return value

  # Int should coerce to float
  config = process.Config(value=42)
  assert config.value == 42.0
  assert isinstance(config.value, float)


def test_bool_to_int_coercion() -> None:
  """Test bool to int coercion behavior."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value

  # True should become 1, False should become 0
  config_true = process.Config(value=True)
  assert config_true.value == 1

  config_false = process.Config(value=False)
  assert config_false.value == 0


def test_list_type_mismatch() -> None:
  """Test that wrong list element types fail validation."""

  @configurable
  def process(values: Hyper[list[int]] = None) -> int:
    if values is None:
      values = [1, 2, 3]
    return sum(values)

  # List of strings should fail
  with pytest.raises(ValidationError):
    process.Config(values=["a", "b", "c"])

  # Mixed types should coerce if possible
  config = process.Config(values=[1, 2.0, "3"])
  assert config.values == [1, 2, 3]


def test_dict_type_mismatch() -> None:
  """Test dict type validation."""

  @configurable
  def process(
    mapping: Hyper[dict[str, int]] = None,
  ) -> int:
    if mapping is None:
      mapping = {"a": 1}
    return sum(mapping.values())

  # Wrong value types should coerce
  config = process.Config(mapping={"x": "10", "y": 20})
  assert config.mapping == {"x": 10, "y": 20}


def test_constraint_on_coerced_value() -> None:
  """Test that constraints apply after type coercion."""

  @configurable
  def process(value: Hyper[int, Ge[0], Le[100]] = 50) -> int:
    return value

  # String "10" should coerce to int 10 and pass constraints
  config = process.Config(value="10")
  assert config.value == 10

  # String "200" should coerce to int 200 but fail Le[100] constraint
  with pytest.raises(ValidationError):
    process.Config(value="200")


def test_numeric_string_with_whitespace() -> None:
  """Test that numeric strings with whitespace are handled."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value

  # Pydantic should strip whitespace and coerce
  config = process.Config(value=" 42 ")
  assert config.value == 42


def test_empty_string_to_int_fails() -> None:
  """Test that empty string to int fails."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value

  with pytest.raises(ValidationError):
    process.Config(value="")


def test_list_coercion_from_tuple() -> None:
  """Test that tuples are coerced to lists."""

  @configurable
  def process(values: Hyper[list[int]] = None) -> int:
    if values is None:
      values = [1, 2, 3]
    return sum(values)

  # Tuple should coerce to list
  config = process.Config(values=(10, 20, 30))
  assert config.values == [10, 20, 30]
  assert isinstance(config.values, list)


def test_float_string_to_float() -> None:
  """Test float string coercion."""

  @configurable
  def process(value: Hyper[float] = 1.0) -> float:
    return value

  # Float string should coerce
  config = process.Config(value="3.14")
  assert config.value == pytest.approx(3.14)
