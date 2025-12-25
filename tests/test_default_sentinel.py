"""Test DEFAULT sentinel for nested configs."""

from __future__ import annotations

import pandas as pd
from pydantic import ValidationError
import pytest

from nonfig import (
  DEFAULT,
  Hyper,
  configurable,
)


@configurable
def inner_transform(
  data: pd.Series,
  multiplier: Hyper[float] = 2.0,
) -> float:
  sum_value: float = data.sum().item()
  return sum_value * multiplier


@configurable
def outer_transform(
  data: pd.Series,
  offset: Hyper[float] = 10.0,
  inner_fn: Hyper[inner_transform.Config] = DEFAULT,
) -> float:
  """Pipeline that uses DEFAULT for nested config."""
  # inner_fn is auto-made by _recursive_make
  inner_result: float = inner_fn(data=data)
  return inner_result + offset


@configurable
class InnerOptimizer:
  def __init__(
    self,
    learning_rate: Hyper[float] = 0.01,
    momentum: Hyper[float] = 0.9,
  ):
    self.learning_rate = learning_rate
    self.momentum = momentum


@configurable
class OuterModel:
  def __init__(
    self,
    hidden_size: Hyper[int] = 128,
    # Nested configs remain as Config objects for later inspection/make()
    optimizer_config: Hyper[InnerOptimizer.Config] = DEFAULT,
  ):
    self.hidden_size = hidden_size
    # Config objects are preserved - call .make() when you need the instance
    self.optimizer_config = optimizer_config


def test_default_with_function_config() -> None:
  """Test DEFAULT works with function-based nested configs."""
  # Test with defaults
  config = outer_transform.Config()
  fn = config.make()
  test_series = pd.Series([1.0, 2.0, 3.0])
  result: float = fn(data=test_series)
  # inner: (1+2+3) * 2.0 = 12.0, outer: 12.0 + 10.0 = 22.0
  assert result == 22.0

  # Test with overrides
  config2 = outer_transform.Config(
    offset=5.0,
    inner_fn=inner_transform.Config(multiplier=3.0),
  )
  fn2 = config2.make()
  result2: float = fn2(data=test_series)
  # inner: (1+2+3) * 3.0 = 18.0, outer: 18.0 + 5.0 = 23.0
  assert result2 == 23.0


def test_default_with_class_config() -> None:
  """Test DEFAULT works with class-based nested configs."""
  # Test with defaults
  config = OuterModel.Config()
  assert config.hidden_size == 128
  assert config.optimizer_config.learning_rate == 0.01
  assert config.optimizer_config.momentum == 0.9

  model = config.make()
  assert isinstance(model, OuterModel)
  assert model.hidden_size == 128
  # With _recursive_make, nested configs are automatically made into instances
  assert isinstance(model.optimizer_config, InnerOptimizer)
  assert model.optimizer_config.learning_rate == 0.01
  assert model.optimizer_config.momentum == 0.9

  # Test with overrides
  config2 = OuterModel.Config(
    hidden_size=256,
    optimizer_config=InnerOptimizer.Config(learning_rate=0.001, momentum=0.95),
  )
  model2 = config2.make()
  assert isinstance(model2, OuterModel)
  assert model2.hidden_size == 256
  # Nested config is automatically made into an instance
  assert isinstance(model2.optimizer_config, InnerOptimizer)
  assert model2.optimizer_config.learning_rate == 0.001
  assert model2.optimizer_config.momentum == 0.95


def test_default_serialization() -> None:
  """Test that configs with DEFAULT serialize and deserialize correctly."""
  # Create config with defaults
  config = outer_transform.Config()

  # Serialize to dict
  config_dict = config.model_dump()
  assert "inner_fn" in config_dict
  assert config_dict["inner_fn"]["multiplier"] == 2.0
  assert config_dict["offset"] == 10.0

  # Serialize to JSON
  config_json = config.model_dump_json()
  assert "inner_fn" in config_json
  assert "multiplier" in config_json

  # Deserialize from JSON
  loaded_config = outer_transform.Config.model_validate_json(config_json)
  assert loaded_config.offset == 10.0
  assert loaded_config.inner_fn.multiplier == 2.0

  # Verify it still works
  fn = loaded_config.make()
  test_series = pd.Series([1.0, 2.0, 3.0])
  result: float = fn(data=test_series)
  assert result == 22.0


def test_default_with_non_config_type_raises_error() -> None:
  """Test that using DEFAULT with non-Config types raises an error."""
  with pytest.raises(
    TypeError, match="DEFAULT can only be used with nested Config types"
  ):

    @configurable
    def bad_function(
      data: pd.Series,
      bad_param: Hyper[int] = DEFAULT,
    ) -> float:
      return float(data.sum().item())


@configurable
class Layer:
  def __init__(self, units: Hyper[int] = 64):
    self.units = units


@configurable
class Network:
  def __init__(
    self,
    num_layers: Hyper[int] = 3,
    layer_config: Hyper[Layer.Config] = DEFAULT,
  ):
    self.num_layers = num_layers
    self.layer_config = layer_config


def test_default_nested_in_class() -> None:
  """Test DEFAULT in nested class configs."""
  config = Network.Config()
  assert config.num_layers == 3
  assert config.layer_config.units == 64

  # Verify serialization
  config_dict = config.model_dump()
  assert config_dict["num_layers"] == 3
  assert config_dict["layer_config"]["units"] == 64


@configurable
class ValidatedInner:
  def __init__(
    self,
    value: Hyper[int] = 10,
  ):
    self.value = value


@configurable
class ValidatedOuter:
  def __init__(
    self,
    inner_config: Hyper[ValidatedInner.Config] = DEFAULT,
  ):
    self.inner_config = inner_config


def test_default_with_validation() -> None:
  """Test that nested configs with DEFAULT still validate properly."""
  # Valid config
  config = ValidatedOuter.Config()
  assert config.inner_config.value == 10

  # Invalid nested config should raise ValidationError
  with pytest.raises(ValidationError):
    ValidatedOuter.Config(inner_config=ValidatedInner.Config(value="not_an_int"))
