"""Test required Hyper parameters without default values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic import ValidationError
import pytest

from nonfig import Ge, Hyper, MakeableModel, configurable

# ============= Basic required params =============


def test_basic_required_hyper_param() -> None:
  """Test function with a required Hyper parameter."""

  @configurable
  def multiply(factor: Hyper[int]) -> int:
    return factor * 10

  # Must provide required param
  config = multiply.Config(factor=5)
  fn = config.make()
  assert fn() == 50


def test_mixed_required_and_optional() -> None:
  """Test function with both required and optional Hyper params."""

  @configurable
  def calculate(
    required: Hyper[int],
    optional: Hyper[int] = 10,
  ) -> int:
    return required + optional

  # Provide only required
  config1 = calculate.Config(required=5)
  fn1 = config1.make()
  assert fn1() == 15

  # Provide both
  config2 = calculate.Config(required=5, optional=20)
  fn2 = config2.make()
  assert fn2() == 25


def test_required_with_constraints() -> None:
  """Test required Hyper param with constraints."""

  @configurable
  def process(value: Hyper[int, Ge[0]]) -> int:
    return value

  # Valid value
  config = process.Config(value=10)
  fn = config.make()
  assert fn() == 10

  # Invalid value fails validation
  with pytest.raises(ValidationError):
    process.Config(value=-5)  # Less than Ge[0]


# ============= Nested required configs =============


@configurable
class Inner:
  """Simple inner class for nesting tests."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(self, value: Hyper[int] = 10) -> None:
    super().__init__()
    self.value = value

  def get_value(self) -> int:
    return self.value


def test_required_nested_config() -> None:
  """Test function with required nested Config param."""

  @configurable
  def outer(
    inner_instance: Hyper[Inner.Config],  # Required nested config
    multiplier: Hyper[int] = 2,
  ) -> int:
    # inner_instance is auto-made by _recursive_make
    return inner_instance.get_value() * multiplier

  # Must provide the nested config
  config = outer.Config(inner_instance=Inner.Config(value=5))
  fn = config.make()
  assert fn() == 10

  # Using default for inner value
  config2 = outer.Config(inner_instance=Inner.Config())
  fn2 = config2.make()
  assert fn2() == 20  # 10 * 2


# ============= Class with required params =============


def test_class_with_required_hyper() -> None:
  """Test class with required Hyper parameters in __init__."""

  @configurable
  class Processor:
    Config: ClassVar[type[MakeableModel[object]]]

    def __init__(self, factor: Hyper[int]) -> None:
      super().__init__()
      self.factor = factor

    def process(self, x: int) -> int:
      return x * self.factor

  # Must provide required param
  config = Processor.Config(factor=3)
  instance = config.make()
  assert instance.process(10) == 30


# ============= Dataclass with required fields =============


@dataclass
class _DataProcessor:
  """Test dataclass for required fields."""

  Config: ClassVar[type[MakeableModel[object]]]

  multiplier: Hyper[int]  # Required
  offset: Hyper[int] = 0  # Optional

  def process(self, x: int) -> int:
    return x * self.multiplier + self.offset


# Apply @configurable after dataclass definition
_DataProcessor = configurable(_DataProcessor)


def test_dataclass_with_required_hyper() -> None:
  """Test dataclass with required Hyper-marked fields."""
  # Must provide required field
  config = _DataProcessor.Config(multiplier=5)
  instance = config.make()
  assert instance.process(10) == 50

  # With optional
  config2 = _DataProcessor.Config(multiplier=5, offset=10)
  instance2 = config2.make()
  assert instance2.process(10) == 60


# ============= Validation errors for missing required =============


def test_missing_required_raises_validation_error() -> None:
  """Test that missing required params raise Pydantic ValidationError."""

  @configurable
  def func(required: Hyper[int]) -> int:
    return required

  with pytest.raises(ValidationError):
    func.Config()  # Missing required field
