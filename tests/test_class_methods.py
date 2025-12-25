"""Test configurable decorator with class methods and Field constraints."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from nonfig import Ge, Hyper, Le, configurable


class Calculator:
  """Test class with configurable methods."""

  def __init__(self, base: float):
    super().__init__()
    self.base = base

  @configurable
  def multiply(self, value: float, factor: Hyper[int, Ge[1], Le[10]] = 2) -> float:
    """Multiply value by factor and add base."""
    return self.base + (value * factor)


def test_class_method_basic() -> None:
  """Test that @configurable works with class methods."""
  calc = Calculator(base=10.0)

  # Test direct call still works
  result = calc.multiply(5.0)
  assert result == 20.0  # 10 + (5 * 2)

  # Test with custom factor
  result = calc.multiply(5.0, factor=3)
  assert result == 25.0  # 10 + (5 * 3)


def test_class_method_config() -> None:
  """Test that Config.make() works with class methods."""
  calc = Calculator(base=10.0)

  # Create a configured version
  config = calc.multiply.Config(factor=4)
  configured_fn = config.make()

  # Should work when called with instance
  result = configured_fn(calc, value=5.0)
  assert result == 30.0  # 10 + (5 * 4)


def test_field_constraints() -> None:
  """Test that Field constraints are preserved."""
  # Valid values should work
  config = Calculator.multiply.Config(factor=5)
  assert config.factor == 5

  # Test lower bound
  with pytest.raises(ValidationError):
    Calculator.multiply.Config(factor=0)

  # Test upper bound
  with pytest.raises(ValidationError):
    Calculator.multiply.Config(factor=11)


def test_default_values() -> None:
  """Test that default values from Hyper are used."""
  config = Calculator.multiply.Config()
  assert config.factor == 2
