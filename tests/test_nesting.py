from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, cast

import pandas as pd

from nonfig import DEFAULT, Ge, Gt, Hyper, MakeableModel, configurable

"""Consolidated nesting tests."""

"""Test nested configs at module level (real-world usage)."""


@configurable
def inner_fn(
  data: pd.Series,
  multiplier: Hyper[float] = 2.0,
) -> float:
  sum_value: float = data.sum().item()
  return sum_value * multiplier


InnerConfig = inner_fn.Config

_DEFAULT_INNER_CONFIG = InnerConfig()


@configurable
def outer_fn(
  data: pd.Series,
  offset: Hyper[float] = 10.0,
  inner_fn_made: Hyper[InnerConfig] = _DEFAULT_INNER_CONFIG,
) -> float:
  # inner_fn_made is auto-made by _recursive_make
  inner_result: float = inner_fn_made(data=data)
  return inner_result + offset


def test_module_level_nested_config() -> None:
  """Verify nested configs work at module level."""
  # Test with defaults
  config = outer_fn.Config()
  fn = config.make()
  test_series = pd.Series([1.0, 2.0, 3.0])
  result: float = fn(data=test_series)
  # inner: (1+2+3) * 2.0 = 12.0, outer: 12.0 + 10.0 = 22.0
  assert result == 22.0

  # Test with overrides
  # Note: Type checker doesn't know about dynamically generated Config classes,
  # so the parameters below will show as type errors, but they work at runtime
  config2 = outer_fn.Config(
    offset=5.0,
    inner_fn_made=inner_fn.Config(multiplier=3.0),
  )
  fn2 = config2.make()
  result2: float = fn2(data=test_series)
  # inner: (1+2+3) * 3.0 = 18.0, outer: 18.0 + 5.0 = 23.0
  assert result2 == 23.0


# --- Direct call pattern tests ---


@configurable
def direct_inner(data: pd.Series, multiplier: Hyper[float] = 2.0) -> float:
  """Inner function for direct call tests."""
  return data.sum().item() * multiplier


@configurable
def direct_outer_unified(
  data: pd.Series,
  nested: direct_inner.Type = direct_inner,  # Unified pattern: function as default
) -> float:
  """Unified pattern: fn: inner.Type = inner works for both direct and Config."""
  return nested(data) + 100.0


def test_unified_pattern_direct_call() -> None:
  """Test unified pattern: nested: inner.Type = inner - direct call."""
  test_data = pd.Series([1.0, 2.0, 3.0])

  # Direct call uses the function itself - works correctly
  result = direct_outer_unified(test_data)
  assert result == (6.0 * 2.0) + 100.0  # 112.0

  # Direct call with explicit function
  result2 = direct_outer_unified(test_data, nested=direct_inner)
  assert result2 == 112.0


def test_unified_pattern_config() -> None:
  """Test unified pattern: nested: inner.Type = inner - Config support."""
  test_data = pd.Series([1.0, 2.0, 3.0])

  # Config now has 'nested' field (with inner.Config() as default)
  assert "nested" in direct_outer_unified.Config.model_fields

  # Via Config.make() with default
  fn_default = direct_outer_unified.Config().make()
  result_default = fn_default(test_data)
  assert result_default == 112.0

  # Via Config.make() with override
  fn_override = direct_outer_unified.Config(
    nested=direct_inner.Config(multiplier=3.0)
  ).make()
  result_override = fn_override(test_data)
  assert result_override == 118.0  # (6.0 * 3.0) + 100.0


"""Test nested configs in containers - list, dict, Sequence, Mapping, etc."""


@configurable
@dataclass
class Layer:
  scale: float = 1.0

  def __call__(self, data: float) -> float:
    return data * self.scale


# Test with list[T]
@configurable
@dataclass
class NetworkWithList:
  layers: list[Layer] = DEFAULT

  def __call__(self, data: float) -> float:
    for layer in self.layers:
      data = layer(data)
    return data


# Test with Sequence[T]
@configurable
@dataclass
class NetworkWithSequence:
  layers: Sequence[Layer] = DEFAULT

  def __call__(self, data: float) -> float:
    for layer in self.layers:
      data = layer(data)
    return data


# Test with dict[str, T]
@configurable
@dataclass
class NetworkWithDict:
  layers: dict[str, Layer] = DEFAULT

  def __call__(self, data: float) -> float:
    for layer in self.layers.values():
      data = layer(data)
    return data


# Test with Mapping[str, T]
@configurable
@dataclass
class NetworkWithMapping:
  layers: Mapping[str, Layer] = DEFAULT

  def __call__(self, data: float) -> float:
    for layer in self.layers.values():
      data = layer(data)
    return data


def test_list_with_default() -> None:
  """list[Layer] = DEFAULT uses empty list."""
  config = NetworkWithList.Config()
  network = config.make()
  assert network.layers == []
  assert network(10.0) == 10.0


def test_list_with_explicit_configs() -> None:
  """Explicit list of configs works."""
  config = NetworkWithList.Config(
    layers=[Layer.Config(scale=2.0), Layer.Config(scale=3.0)]
  )
  network = config.make()
  assert network(10.0) == 60.0  # 10 * 2 * 3


def test_sequence_with_default() -> None:
  """Sequence[Layer] = DEFAULT uses empty list."""
  config = NetworkWithSequence.Config()
  network = config.make()
  assert network.layers == []
  assert network(10.0) == 10.0


def test_sequence_with_explicit_configs() -> None:
  """Explicit sequence of configs works."""
  config = NetworkWithSequence.Config(
    layers=[Layer.Config(scale=2.0), Layer.Config(scale=0.5)]
  )
  network = config.make()
  assert network(10.0) == 10.0  # 10 * 2 * 0.5


def test_dict_with_default() -> None:
  """dict[str, Layer] = DEFAULT uses empty dict."""
  config = NetworkWithDict.Config()
  network = config.make()
  assert network.layers == {}
  assert network(10.0) == 10.0


def test_dict_with_explicit_configs() -> None:
  """Explicit dict of configs works."""
  config = NetworkWithDict.Config(
    layers={"first": Layer.Config(scale=2.0), "second": Layer.Config(scale=5.0)}
  )
  network = config.make()
  assert network(1.0) == 10.0  # 1 * 2 * 5


def test_mapping_with_default() -> None:
  """Mapping[str, Layer] = DEFAULT uses empty dict."""
  config = NetworkWithMapping.Config()
  network = config.make()
  assert network.layers == {}
  assert network(10.0) == 10.0


def test_mapping_with_explicit_configs() -> None:
  """Explicit mapping of configs works."""
  config = NetworkWithMapping.Config(
    layers={"a": Layer.Config(scale=3.0), "b": Layer.Config(scale=2.0)}
  )
  network = config.make()
  assert network(5.0) == 30.0  # 5 * 3 * 2


"""Regression tests for nested dictionary handling in configurations."""


@configurable
@dataclass
class InnerConfig:
  x: int = 10
  y: int = 20


@configurable
@dataclass
class OuterConfig:
  inner: Hyper[InnerConfig] = DEFAULT
  extra: int = 5


@configurable
@dataclass
class DeepConfig:
  outer: Hyper[OuterConfig] = DEFAULT


@configurable
@dataclass
class ListConfig:
  items: Hyper[list[InnerConfig]] = DEFAULT


def test_dict_assignment_basic():
  """Test assigning a dict to a nested Config field."""
  # Should automatically convert dict to InnerConfig
  cfg = OuterConfig.Config(inner={"x": 99})  # type: ignore
  model = cfg.make()

  assert isinstance(model.inner, InnerConfig)
  assert model.inner.x == 99
  assert model.inner.y == 20  # Default preserved
  assert model.extra == 5


def test_dict_assignment_recursive():
  """Test converting nested dicts through multiple levels."""
  # Dict -> DeepConfig -> OuterConfig -> InnerConfig
  cfg = DeepConfig.Config(
    outer={"inner": {"x": 123, "y": 456}, "extra": 7}  # type: ignore
  )
  model = cfg.make()

  assert isinstance(model.outer, OuterConfig)
  assert isinstance(model.outer.inner, InnerConfig)
  assert model.outer.inner.x == 123
  assert model.outer.inner.y == 456
  assert model.outer.extra == 7


def test_dict_assignment_list():
  """Test assigning a list of dicts to a Hyper[List[Config]]."""
  # This relies on _recursive_make handling lists
  cfg = ListConfig.Config(
    items=[
      {"x": 1},  # type: ignore
      InnerConfig.Config(x=2),
      {"x": 3, "y": 33},  # type: ignore
    ]
  )
  model = cfg.make()

  assert len(model.items) == 3
  assert isinstance(model.items[0], InnerConfig)
  assert model.items[0].x == 1
  assert model.items[1].x == 2
  assert model.items[2].x == 3
  assert model.items[2].y == 33


def test_dict_assignment_extra_keys_ignored():
  """Test that extra keys in dict are ignored (Pydantic default behavior)."""
  cfg = OuterConfig.Config(inner={"x": 1, "unknown_key": 999})  # type: ignore
  model = cfg.make()

  # Validation should pass and ignore unknown_key
  assert model.inner.x == 1
  assert not hasattr(model.inner, "unknown_key")


"""Test nested configs with mixed configurables (dataclass, class, function)."""

# ============= Leaf-level configurables =============


@configurable
def base_transform(
  data: pd.Series,
  multiplier: Hyper[float, Gt[0.0]] = 2.0,
) -> float:
  """Basic transformation function."""
  sum_value: float = data.sum().item()
  return sum_value * multiplier


BaseTransformConfig = base_transform.Config


@configurable
class Scaler:
  """Simple scaling class."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(self, scale: Hyper[float, Gt[0.0]] = 1.0):
    super().__init__()
    self.scale = scale

  def apply(self, value: float) -> float:
    """Scale a value."""
    return value * self.scale


ScalerConfig = Scaler.Config


@dataclass
class Offsetter:
  """Simple offset dataclass."""

  Config: ClassVar[type[MakeableModel[object]]]

  offset: Hyper[float] = 10.0

  def apply(self, value: float) -> float:
    """Add offset to value."""
    return value + self.offset


# Apply @configurable AFTER dataclass definition
Offsetter = configurable(Offsetter)
OffsetterConfig = Offsetter.Config

# ============= Default configs as module-level constants =============
# Must be created after all leaf-level configurables are defined

_DEFAULT_SCALER_CONFIG = Scaler.Config()
_DEFAULT_OFFSETTER_CONFIG = Offsetter.Config()
_DEFAULT_BASE_TRANSFORM_CONFIG = base_transform.Config()

# ============= Mid-level configurables (nesting leaf-level) =============


@configurable
def function_nesting_class(
  data: pd.Series,
  scaler_config: Hyper[ScalerConfig] = _DEFAULT_SCALER_CONFIG,
  extra: Hyper[float] = 5.0,
) -> float:
  """Function that nests a class config."""
  scaler = cast("Scaler", scaler_config)
  sum_value: float = data.sum().item()
  return scaler.apply(sum_value) + extra


@configurable
def function_nesting_dataclass(
  data: pd.Series,
  offsetter_config: Hyper[OffsetterConfig] = _DEFAULT_OFFSETTER_CONFIG,
  multiplier: Hyper[float] = 2.0,
) -> float:
  """Function that nests a dataclass config."""
  offsetter = cast("Offsetter", offsetter_config)
  sum_value: float = data.sum().item()
  return offsetter.apply(sum_value * multiplier)


@configurable
class ClassNestingFunction:
  """Class that nests a function config."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(
    self,
    transform_config: Hyper[BaseTransformConfig] = _DEFAULT_BASE_TRANSFORM_CONFIG,
    bonus: Hyper[float] = 100.0,
  ):
    super().__init__()
    self.transform_config = transform_config
    self.bonus = bonus

  def process(self, data: pd.Series) -> float:
    """Process data using nested function."""
    # Config fields stay as Config - call make() when needed
    transform_fn = cast("base_transform", self.transform_config)
    result: float = transform_fn(data=data)
    return result + self.bonus


@configurable
class ClassNestingDataclass:
  """Class that nests a dataclass config."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(
    self,
    offsetter_config: Hyper[OffsetterConfig] = _DEFAULT_OFFSETTER_CONFIG,
    scale: Hyper[float, Gt[0.0]] = 3.0,
  ):
    super().__init__()
    self.offsetter_config = offsetter_config
    self.scale = scale

  def process(self, value: float) -> float:
    """Process value using nested dataclass."""
    # Config fields stay as Config - call make() when needed
    offsetter = cast("Offsetter", self.offsetter_config)
    return offsetter.apply(value * self.scale)


@dataclass
class DataclassNestingFunction:
  """Dataclass that nests a function config."""

  Config: ClassVar[type[MakeableModel[object]]]

  transform_config: Hyper[BaseTransformConfig] = field(
    default_factory=base_transform.Config
  )
  threshold: Hyper[float, Ge[0.0]] = 50.0

  def evaluate(self, data: pd.Series) -> bool:
    """Evaluate if transformed data exceeds threshold."""
    # Config fields stay as Config - call make() when needed
    transform_fn = cast("base_transform", self.transform_config)
    result: float = transform_fn(data=data)
    return result > self.threshold


DataclassNestingFunction = configurable(DataclassNestingFunction)


@dataclass
class DataclassNestingClass:
  """Dataclass that nests a class config."""

  Config: ClassVar[type[MakeableModel[object]]]

  scaler_config: Hyper[ScalerConfig] = field(default_factory=Scaler.Config)
  minimum: Hyper[float] = 0.0

  def calculate(self, value: float) -> float:
    """Calculate using nested class."""
    # Config fields stay as Config - call make() when needed
    scaler = cast("Scaler", self.scaler_config)
    result = scaler.apply(value)
    return max(result, self.minimum)


DataclassNestingClass = configurable(DataclassNestingClass)

# ============= Default configs for mid-level configurables =============
# Must be created after all mid-level configurables are defined

_DEFAULT_CLASS_NESTING_FN_CONFIG = ClassNestingFunction.Config()
_DEFAULT_FUNCTION_NESTING_CLASS_CONFIG = function_nesting_class.Config()

# ============= Top-level configurable (nesting mid-level) =============

ClassNestingFunctionConfig = ClassNestingFunction.Config
FunctionNestingClassConfig = function_nesting_class.Config


@configurable
def top_level_orchestrator(
  data: pd.Series,
  class_nesting_fn_instance: Hyper[
    ClassNestingFunctionConfig
  ] = _DEFAULT_CLASS_NESTING_FN_CONFIG,
  function_nesting_class_fn: Hyper[
    FunctionNestingClassConfig
  ] = _DEFAULT_FUNCTION_NESTING_CLASS_CONFIG,
  final_multiplier: Hyper[float, Gt[0.0]] = 1.5,
) -> float:
  """Top-level function that orchestrates multiple nested configs."""
  # Nested configs are auto-made - class_nesting_fn_instance is already a ClassNestingFunction
  result1: float = class_nesting_fn_instance.process(data)

  # function_nesting_class_fn is already a bound function
  result2: float = function_nesting_class_fn(data=data)

  return (result1 + result2) * final_multiplier


# ============= Tests =============


def test_function_nesting_class() -> None:
  """Test function with nested class config."""
  test_data = pd.Series([1.0, 2.0, 3.0])  # sum = 6.0

  # Test with defaults
  config = function_nesting_class.Config()
  fn = config.make()
  result: float = fn(data=test_data)
  # scaler applies scale=1.0: 6.0 * 1.0 = 6.0, extra=5.0: 6.0 + 5.0 = 11.0
  assert result == 11.0

  # Test with custom configs
  config2 = function_nesting_class.Config(
    scaler_config=Scaler.Config(scale=2.0),
    extra=10.0,
  )
  fn2 = config2.make()
  result2: float = fn2(data=test_data)
  # scaler applies scale=2.0: 6.0 * 2.0 = 12.0, extra=10.0: 12.0 + 10.0 = 22.0
  assert result2 == 22.0


def test_function_nesting_dataclass() -> None:
  """Test function with nested dataclass config."""
  test_data = pd.Series([1.0, 2.0, 3.0])  # sum = 6.0

  # Test with defaults
  config = function_nesting_dataclass.Config()
  fn = config.make()
  result: float = fn(data=test_data)
  # sum * multiplier: 6.0 * 2.0 = 12.0, offset=10.0: 12.0 + 10.0 = 22.0
  assert result == 22.0

  # Test with custom configs
  config2 = function_nesting_dataclass.Config(
    offsetter_config=Offsetter.Config(offset=20.0),
    multiplier=3.0,
  )
  fn2 = config2.make()
  result2: float = fn2(data=test_data)
  # sum * multiplier: 6.0 * 3.0 = 18.0, offset=20.0: 18.0 + 20.0 = 38.0
  assert result2 == 38.0


def test_class_nesting_function() -> None:
  """Test class with nested function config."""
  test_data = pd.Series([1.0, 2.0, 3.0])  # sum = 6.0

  # Test with defaults
  config = ClassNestingFunction.Config()
  instance = cast("ClassNestingFunction", config.make())
  result: float = instance.process(test_data)
  # transform: 6.0 * 2.0 = 12.0, bonus=100.0: 12.0 + 100.0 = 112.0
  assert result == 112.0

  # Test with custom configs
  config2 = ClassNestingFunction.Config(
    transform_config=base_transform.Config(multiplier=5.0),
    bonus=50.0,
  )
  instance2 = cast("ClassNestingFunction", config2.make())
  result2: float = instance2.process(test_data)
  # transform: 6.0 * 5.0 = 30.0, bonus=50.0: 30.0 + 50.0 = 80.0
  assert result2 == 80.0


def test_class_nesting_dataclass() -> None:
  """Test class with nested dataclass config."""
  # Test with defaults
  config = ClassNestingDataclass.Config()
  instance = cast("ClassNestingDataclass", config.make())
  result: float = instance.process(10.0)
  # value * scale: 10.0 * 3.0 = 30.0, offset=10.0: 30.0 + 10.0 = 40.0
  assert result == 40.0

  # Test with custom configs
  config2 = ClassNestingDataclass.Config(
    offsetter_config=Offsetter.Config(offset=5.0),
    scale=2.0,
  )
  instance2 = cast("ClassNestingDataclass", config2.make())
  result2: float = instance2.process(10.0)
  # value * scale: 10.0 * 2.0 = 20.0, offset=5.0: 20.0 + 5.0 = 25.0
  assert result2 == 25.0


def test_dataclass_nesting_function() -> None:
  """Test dataclass with nested function config."""
  test_data = pd.Series([1.0, 2.0, 3.0])  # sum = 6.0

  # Test with defaults
  config = DataclassNestingFunction.Config()
  instance = cast("DataclassNestingFunction", config.make())
  result: bool = instance.evaluate(test_data)
  # transform: 6.0 * 2.0 = 12.0, threshold=50.0: 12.0 > 50.0 = False
  assert result is False

  # Test with custom configs
  config2 = DataclassNestingFunction.Config(
    transform_config=base_transform.Config(multiplier=10.0),
    threshold=50.0,
  )
  instance2 = cast("DataclassNestingFunction", config2.make())
  result2: bool = instance2.evaluate(test_data)
  # transform: 6.0 * 10.0 = 60.0, threshold=50.0: 60.0 > 50.0 = True
  assert result2 is True


def test_dataclass_nesting_class() -> None:
  """Test dataclass with nested class config."""
  # Test with defaults
  config = DataclassNestingClass.Config()
  instance = cast("DataclassNestingClass", config.make())
  result: float = instance.calculate(10.0)
  # scaler: 10.0 * 1.0 = 10.0, max(10.0, 0.0) = 10.0
  assert result == 10.0

  # Test with custom configs
  config2 = DataclassNestingClass.Config(
    scaler_config=Scaler.Config(scale=0.5),
    minimum=7.0,
  )
  instance2 = cast("DataclassNestingClass", config2.make())
  result2: float = instance2.calculate(10.0)
  # scaler: 10.0 * 0.5 = 5.0, max(5.0, 7.0) = 7.0
  assert result2 == 7.0


def test_triple_level_nesting() -> None:
  """Test three levels of nesting with mixed types."""
  test_data = pd.Series([1.0, 2.0, 3.0])  # sum = 6.0

  # Test with all defaults
  config = top_level_orchestrator.Config()
  fn = config.make()
  result: float = fn(data=test_data)

  # Calculate expected result:
  # ClassNestingFunction path:
  #   - base_transform: 6.0 * 2.0 = 12.0
  #   - bonus: 12.0 + 100.0 = 112.0
  # function_nesting_class path:
  #   - Scaler: 6.0 * 1.0 = 6.0
  #   - extra: 6.0 + 5.0 = 11.0
  # Combined: (112.0 + 11.0) * 1.5 = 184.5
  assert result == 184.5

  # Test with deeply nested custom configs
  config2 = top_level_orchestrator.Config(
    class_nesting_fn_instance=ClassNestingFunction.Config(
      transform_config=base_transform.Config(multiplier=3.0),
      bonus=50.0,
    ),
    function_nesting_class_fn=function_nesting_class.Config(
      scaler_config=Scaler.Config(scale=2.0),
      extra=10.0,
    ),
    final_multiplier=2.0,
  )
  fn2 = config2.make()
  result2: float = fn2(data=test_data)

  # Calculate expected result:
  # ClassNestingFunction path:
  #   - base_transform: 6.0 * 3.0 = 18.0
  #   - bonus: 18.0 + 50.0 = 68.0
  # function_nesting_class path:
  #   - Scaler: 6.0 * 2.0 = 12.0
  #   - extra: 12.0 + 10.0 = 22.0
  # Combined: (68.0 + 22.0) * 2.0 = 180.0
  assert result2 == 180.0


def test_config_serialization() -> None:
  """Test that nested configs can be serialized/deserialized."""
  # Create a config with nested values
  config = top_level_orchestrator.Config(
    class_nesting_fn_instance=ClassNestingFunction.Config(
      transform_config=base_transform.Config(multiplier=4.0),
      bonus=75.0,
    ),
    function_nesting_class_fn=function_nesting_class.Config(
      scaler_config=Scaler.Config(scale=1.5),
      extra=15.0,
    ),
    final_multiplier=2.5,
  )

  # Serialize to dict
  config_dict = config.model_dump()
  print(f"Serialized config: {config_dict}")

  # Verify structure
  assert "class_nesting_fn_instance" in config_dict
  assert "function_nesting_class_fn" in config_dict
  assert "final_multiplier" in config_dict

  # Deserialize back
  config2 = top_level_orchestrator.Config(**config_dict)

  # Test that it produces same results
  test_data = pd.Series([2.0, 3.0, 5.0])  # sum = 10.0
  fn1 = config.make()
  fn2 = config2.make()

  result1: float = fn1(data=test_data)
  result2: float = fn2(data=test_data)
  assert result1 == result2
