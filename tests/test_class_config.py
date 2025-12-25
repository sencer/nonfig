"""Test @configurable decorator on classes (dataclasses and regular classes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic import ValidationError
import pytest

from nonfig import Ge, Gt, Hyper, Le, MakeableModel, configurable


# Test with dataclass
# Note: @configurable must be on top (applied last)
@configurable
@dataclass
class DataclassModel:
  """A dataclass with Hyper parameters."""

  Config: ClassVar[type[MakeableModel[object]]]

  learning_rate: Hyper[float, Gt[0.0], Le[1.0]] = 0.01
  batch_size: Hyper[int, Ge[1]] = 32
  name: str = "default"  # Regular field, not Hyper


# Test with regular class
@configurable
class RegularClass:
  """A regular class with __init__ containing Hyper parameters."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(
    self,
    threshold: Hyper[float, Ge[0.0], Le[1.0]] = 0.5,
    max_iterations: Hyper[int, Ge[1]] = 100,
    verbose: bool = False,  # Regular parameter
  ):
    super().__init__()
    self.threshold = threshold
    self.max_iterations = max_iterations
    self.verbose = verbose

  def run(self) -> str:
    """Example method."""
    return f"Running with threshold={self.threshold}, iterations={self.max_iterations}"


# Test with callable class (has __call__)
@configurable
class CallableModel:
  """A callable class that can be configured."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(
    self,
    multiplier: Hyper[float, Ge[1.0]] = 2.0,
    offset: Hyper[float] = 0.0,
  ):
    super().__init__()
    self.multiplier = multiplier
    self.offset = offset

  def __call__(self, value: float) -> float:
    """Transform a value."""
    return value * self.multiplier + self.offset


def test_dataclass_basic() -> None:
  """Test basic dataclass instantiation with defaults."""
  model = DataclassModel()
  assert model.learning_rate == 0.01
  assert model.batch_size == 32
  assert model.name == "default"


def test_dataclass_with_params() -> None:
  """Test dataclass with custom parameters."""
  model = DataclassModel(learning_rate=0.001, batch_size=64, name="custom")
  assert model.learning_rate == 0.001
  assert model.batch_size == 64
  assert model.name == "custom"


def test_dataclass_config_class() -> None:
  """Test that Config class is created for dataclass."""
  assert hasattr(DataclassModel, "Config")
  config = DataclassModel.Config(learning_rate=0.005, batch_size=16)
  data = config.model_dump()
  assert data["learning_rate"] == 0.005
  assert data["batch_size"] == 16


def test_dataclass_config_make() -> None:
  """Test Config.make() for dataclass returns instance directly."""
  config = DataclassModel.Config(learning_rate=0.1, batch_size=8)
  model = config.make()  # Now returns instance directly
  assert isinstance(model, DataclassModel)
  assert model.learning_rate == 0.1
  assert model.batch_size == 8


def test_dataclass_validation() -> None:
  """Test that constraints are validated for dataclass."""
  # Valid values
  config = DataclassModel.Config(learning_rate=0.5, batch_size=10)
  assert config.learning_rate == 0.5

  # Invalid: learning_rate > 1.0
  with pytest.raises(ValidationError):
    DataclassModel.Config(learning_rate=1.5, batch_size=10)

  # Invalid: batch_size < 1
  with pytest.raises(ValidationError):
    DataclassModel.Config(learning_rate=0.1, batch_size=0)


def test_regular_class_basic() -> None:
  """Test basic regular class instantiation."""
  obj = RegularClass()
  assert obj.threshold == 0.5
  assert obj.max_iterations == 100
  assert obj.verbose is False


def test_regular_class_with_params() -> None:
  """Test regular class with custom parameters."""
  obj = RegularClass(threshold=0.8, max_iterations=50, verbose=True)
  assert obj.threshold == 0.8
  assert obj.max_iterations == 50
  assert obj.verbose is True


def test_regular_class_config() -> None:
  """Test Config class for regular class."""
  assert hasattr(RegularClass, "Config")
  config = RegularClass.Config(threshold=0.3, max_iterations=200)
  obj = config.make()  # Now returns instance directly
  assert isinstance(obj, RegularClass)
  assert obj.threshold == 0.3
  assert obj.max_iterations == 200
  assert obj.run() == "Running with threshold=0.3, iterations=200"


def test_regular_class_validation() -> None:
  """Test validation for regular class."""
  # Invalid: threshold > 1.0
  with pytest.raises(ValidationError):
    RegularClass.Config(threshold=1.5, max_iterations=100)

  # Invalid: max_iterations < 1
  with pytest.raises(ValidationError):
    RegularClass.Config(threshold=0.5, max_iterations=0)


def test_callable_class() -> None:
  """Test callable class (has __call__)."""
  model = CallableModel()
  result = model(10.0)
  assert result == 20.0  # 10 * 2.0 + 0.0

  model2 = CallableModel(multiplier=3.0, offset=5.0)
  result2 = model2(10.0)
  assert result2 == 35.0  # 10 * 3.0 + 5.0


def test_callable_class_config() -> None:
  """Test Config for callable class."""
  config = CallableModel.Config(multiplier=4.0, offset=10.0)
  model = config.make()  # Now returns instance directly
  assert isinstance(model, CallableModel)
  result = model(5.0)
  assert result == 30.0  # 5 * 4.0 + 10.0


# Define nested config classes at module level for proper type resolution
@configurable
class InnerModel:
  """Inner model for testing nested configs."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(self, factor: Hyper[float, Ge[1.0]] = 2.0):
    super().__init__()
    self.factor = factor


@configurable
class OuterModel:
  """Outer model that contains an inner config."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(
    self,
    multiplier: Hyper[float] = 1.0,
    inner: Hyper[InnerModel.Config] = None,
  ):
    super().__init__()
    self.multiplier = multiplier
    if inner is None:
      self.inner = InnerModel.Config()
    else:
      self.inner: MakeableModel[object] = inner


def test_nested_class_configs() -> None:
  """Test using a Config class as a Hyper parameter in another class."""
  # Test that we can create and use nested configs
  outer_config = OuterModel.Config(
    multiplier=3.0,
    inner=InnerModel.Config(factor=5.0),
  )
  assert outer_config.multiplier == 3.0
  assert outer_config.inner.factor == 5.0
