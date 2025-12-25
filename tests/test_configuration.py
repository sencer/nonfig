from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ValidationError
import pytest

from nonfig import (
  Ge,
  Hyper,
  Le,
  MakeableModel,
  configurable,
)

"""Consolidated configuration tests."""

"""Tests for the configurable decorator and configuration system."""


def test_configurable_decorator_creates_correct_wrapper_and_config() -> None:
  """Verify @configurable creates a wrapper and a Pydantic model with correct fields."""

  @configurable
  def dummy_indicator(
    data: pd.Series,
    window: Hyper[int] = 10,
    name: Hyper[str] = "default_name",
  ) -> str:
    return f"{name}:{data.sum()}:{window}"

  # 1. Assert that the decorator returns a callable with a Config attribute
  assert callable(dummy_indicator)
  assert hasattr(dummy_indicator, "Config")

  # 2. Assert that the .Config class was created and attached
  config_cls = dummy_indicator.Config
  assert issubclass(config_cls, MakeableModel)
  assert issubclass(config_cls, BaseModel)

  # 3. Assert that the model has the correct fields and types
  fields = config_cls.model_fields
  assert list(fields.keys()) == ["window", "name"]
  assert fields["window"].annotation is int
  assert fields["name"].annotation is str
  # FieldInfo.default returns Any in pydantic-stubs, so we accept it
  window_default: int = fields["window"].default
  name_default: str = fields["name"].default
  assert window_default == 10
  assert name_default == "default_name"


def test_config_validation_is_enforced() -> None:
  """Verify that Pydantic validation catches invalid input."""

  @configurable
  def dummy_indicator(
    data: pd.Series,
    window: Hyper[int] = 10,
  ) -> int:
    # pandas .item() returns Any in stubs, so we annotate it
    sum_value: int = data.sum().item()
    return sum_value + window

  # This should raise a ValidationError because window is not an int
  with pytest.raises(ValidationError):
    dummy_indicator.Config(window="not_an_int")


def test_configurable_indicator_is_callable() -> None:
  """Verify the wrapper object is callable and passes through calls correctly."""

  @configurable
  def dummy_indicator(
    data: pd.Series,
    window: Hyper[int] = 10,
  ) -> int:
    sum_value: int = data.sum().item()
    return sum_value + window

  # The decorated object should be directly callable
  result: int | float = dummy_indicator(data=pd.Series([1, 2, 3]))
  assert result == (1 + 2 + 3) + 10


def test_config_make_returns_partial_function() -> None:
  """Verify .make() returns a partial function with config values bound."""

  @configurable
  def dummy_indicator(
    data: pd.Series,
    window: Hyper[int] = 10,
  ) -> float:
    sum_value: float = data.sum().item()
    return sum_value * window

  # Create a config instance with window=5
  config = dummy_indicator.Config(window=5)
  partial_fn = config.make()

  test_series = pd.Series([1.0, 2.0, 3.0])
  result: float = partial_fn(data=test_series)
  assert result == test_series.sum() * 5.0


def test_make_method_produces_correct_function() -> None:
  """Verify that the .make() method creates a correctly configured function."""

  @configurable
  def dummy_indicator(
    data: pd.Series,
    multiplier: Hyper[float] = 2.0,
  ) -> float:
    sum_value: float = data.sum().item()
    return sum_value * multiplier

  # Use Config to create an instance
  my_config = dummy_indicator.Config(multiplier=5.0)
  made_function = my_config.make()

  test_series = pd.Series([1.0, 2.0, 3.0])
  result: float = made_function(data=test_series)
  assert result == test_series.sum() * 5.0


def test_configurable_with_required_hyper_param() -> None:
  """Verify that Hyper parameters without defaults work as required fields."""

  @configurable
  def process(
    window: Hyper[int],
  ) -> int:
    return window * 2

  # Config class should have a required 'window' field
  config_cls = process.Config
  assert "window" in config_cls.model_fields

  # Must provide required param
  config = process.Config(window=5)
  fn = config.make()
  assert fn() == 10


def test_configurable_with_no_hyperparameters() -> None:
  """Verify correct behavior for a function with no hyperparameters."""

  @configurable
  def simple_function(
    data: pd.Series,
  ) -> pd.Series:
    return data

  assert hasattr(simple_function, "Config")
  config_cls = simple_function.Config
  assert not config_cls.model_fields

  config_instance = simple_function.Config()
  made_function = config_instance.make()

  test_series = pd.Series([1, 2])
  result_series: pd.Series = made_function(data=test_series)
  assert result_series.equals(test_series)


"""Regression tests for runtime attributes on configurable functions."""


def test_function_runtime_attributes_defaults():
  """Test that default hyperparameters are exposed as attributes on the function wrapper."""

  @configurable
  def my_func(x: Hyper[int] = 10, y: Hyper[str] = "test"):
    return x

  # Verify attributes exist and match defaults
  assert hasattr(my_func, "x")
  assert my_func.x == 10
  assert hasattr(my_func, "y")
  assert my_func.y == "test"

  # Verify we can access them without calling the function
  assert isinstance(my_func.x, int)


def test_function_runtime_attributes_no_defaults():
  """Test that parameters without defaults are not exposed."""

  @configurable
  def my_func(x: Hyper[int]):
    return x

  # Should not have attribute x as it has no default
  assert not hasattr(my_func, "x")


def test_function_runtime_attributes_bound_function():
  """Test that attributes are also available on bound functions (from make())."""

  @configurable
  def my_func(x: Hyper[int] = 10):
    return x

  config = my_func.Config(x=20)
  bound_fn = config.make()

  # Bound function should have the configured value
  assert bound_fn.x == 20
  # Original function should still have default
  assert my_func.x == 10


def test_function_runtime_attributes_are_readonly() -> None:
  """Test that bound function attributes are read-only."""

  @configurable
  def my_func(x: Hyper[int] = 10, y: Hyper[str] = "test") -> int:
    return x

  config = my_func.Config(x=20)
  bound_fn = config.make()

  # Reading should work
  assert bound_fn.x == 20

  # Writing should fail
  with pytest.raises(AttributeError, match="Hyperparameter 'x' is read-only"):
    bound_fn.x = 30

  # Writing to unknown attribute should generic error
  with pytest.raises(AttributeError, match="'BoundFunction' has no attribute 'z'"):
    bound_fn.z = 100


def test_config_str_matches_repr() -> None:
  @configurable
  @dataclass
  class MyModel:
    x: Hyper[int] = 1
    y: Hyper[str] = "foo"

  config = MyModel.Config(x=10)

  # Check that repr output is as expected (Pydantic style)
  repr_output = repr(config)
  assert "MyModelConfig" in repr_output
  assert "x=10" in repr_output

  # Check that str output matches repr output
  # This is what the user requested ("as pretty as __repr__")
  assert str(config) == repr(config)


def test_function_config_str_matches_repr() -> None:
  @configurable
  def my_func(a: Hyper[int] = 1) -> int:
    return a

  func_config = my_func.Config(a=5)
  assert "MyFuncConfig" in repr(func_config)
  assert str(func_config) == repr(func_config)


"""Test runtime config modification patterns."""


def test_config_dict_roundtrip() -> None:
  """Test config -> dict -> config roundtrip."""

  @configurable
  def process(
    period: Hyper[int, Ge[1]] = 14,
    alpha: Hyper[float, Ge[0.0], Le[1.0]] = 0.5,
  ) -> float:
    return period * alpha

  # Create config
  config1 = process.Config(period=20, alpha=0.8)

  # Convert to dict
  config_dict = config1.model_dump()
  assert config_dict == {"period": 20, "alpha": 0.8}

  # Recreate from dict
  config2 = process.Config(**config_dict)
  assert config2.period == 20
  assert config2.alpha == 0.8

  # Should produce same function behavior
  fn1 = config1.make()
  fn2 = config2.make()
  assert fn1() == fn2()


def test_config_copy_and_modify() -> None:
  """Test creating modified copies of configs."""

  @configurable
  def process(
    value: Hyper[int] = 10,
    multiplier: Hyper[float] = 2.0,
  ) -> float:
    return value * multiplier

  config1 = process.Config(value=10, multiplier=2.0)

  # Create a modified copy
  config2 = config1.model_copy(update={"value": 20})

  assert config1.value == 10
  assert config2.value == 20
  assert config2.multiplier == 2.0


def test_partial_config_update() -> None:
  """Test updating only some fields of a config."""

  @configurable
  def process(
    a: Hyper[int] = 1,
    b: Hyper[int] = 2,
    c: Hyper[int] = 3,
  ) -> int:
    return a + b + c

  config = process.Config(a=10, b=20, c=30)

  # Update only 'b'
  updated_config = config.model_copy(update={"b": 50})

  assert updated_config.a == 10
  assert updated_config.b == 50
  assert updated_config.c == 30


def test_config_validation_on_creation() -> None:
  """Test that validation happens at config creation time."""

  @configurable
  def process(value: Hyper[int, Ge[0], Le[100]] = 50) -> int:
    return value

  # Valid value should work
  config = process.Config(value=75)
  assert config.value == 75

  # Invalid value should fail immediately
  with pytest.raises(ValidationError):
    process.Config(value=200)


def test_json_serialization() -> None:
  """Test JSON serialization of configs."""

  @configurable
  def process(
    period: Hyper[int] = 14,
    threshold: Hyper[float] = 0.5,
  ) -> float:
    return period * threshold

  config = process.Config(period=20, threshold=0.8)

  # Convert to JSON
  json_str = config.model_dump_json()
  assert isinstance(json_str, str)
  assert "20" in json_str
  assert "0.8" in json_str

  # Parse from JSON
  import json

  parsed = json.loads(json_str)
  config2 = process.Config(**parsed)
  assert config2.period == 20
  assert config2.threshold == 0.8


def test_config_equality() -> None:
  """Test that configs with same values are equal."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value

  config1 = process.Config(value=20)
  config2 = process.Config(value=20)
  config3 = process.Config(value=30)

  # Same values should be equal
  assert config1 == config2

  # Different values should not be equal
  assert config1 != config3


def test_config_immutability() -> None:
  """Test that config instances are immutable (frozen)."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value

  config = process.Config(value=20)

  # Direct mutation should raise ValidationError
  with pytest.raises(ValidationError):
    config.value = 30

  # Value should remain unchanged
  assert config.value == 20


def test_config_hashability() -> None:
  """Test that config instances are hashable and can be used as dict keys."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value

  config1 = process.Config(value=20)
  config2 = process.Config(value=20)
  config3 = process.Config(value=30)

  # Should be hashable
  assert hash(config1) == hash(config2)
  assert hash(config1) != hash(config3)

  # Should work as dict keys
  d = {config1: "first", config3: "third"}
  assert d[config2] == "first"  # config2 equals config1
  assert d[config3] == "third"

  # Should work in sets
  s = {config1, config2, config3}
  assert len(s) == 2  # config1 and config2 are equal


def test_model_config_parameter_raises_error():
  """Test that defining a parameter named 'model_config' raises ValueError.

  'model_config' is reserved by Pydantic for model configuration and
  cannot be used as a Config field name.
  """

  # Test with function
  with pytest.raises(ValueError, match="reserved by Pydantic"):

    @configurable
    def my_func(model_config: Hyper[int] = 1):
      pass

  # Test with class
  with pytest.raises(ValueError, match="reserved by Pydantic"):

    @configurable
    class MyClass:
      def __init__(self, model_config: Hyper[int] = 1):
        pass
