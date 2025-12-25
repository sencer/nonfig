"""Static typing tests for @configurable decorator.

These tests verify that the Configurable Protocol provides correct type inference.
They are checked by basedpyright during CI - if types are wrong, basedpyright fails.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, assert_type

from nonfig import DEFAULT, Ge, Hyper, Le, MakeableModel, configurable

# --- Basic classes ---


@configurable
class Model:
  def __init__(self, x: int = 10, y: str = "default") -> None:
    self.x = x
    self.y = y


@configurable
@dataclass
class DataModel:
  x: int
  y: str = "default"
  computed: str = field(init=False, default="computed_value")


# --- For nested configs ---


@configurable
class Optimizer:
  def __init__(self, lr: float = 0.01) -> None:
    self.lr = lr


@configurable
@dataclass
class Pipeline:
  model: Model = DEFAULT
  optimizer: Optimizer = DEFAULT


# --- Function with Hyper ---


@configurable
def process(data: list[int], window: Hyper[int] = 10) -> float:
  return sum(data[:window]) / window


# --- Dataclass applied with = configurable() pattern ---


@dataclass
class Layer:
  size: int = 64


Layer = configurable(Layer)


# --- Literal and Enum types ---


class Mode(StrEnum):
  FAST = "fast"
  SLOW = "slow"


@configurable
class Processor:
  def __init__(
    self,
    mode: Literal["train", "eval"] = "train",
    priority: Mode = Mode.FAST,
  ) -> None:
    self.mode = mode
    self.priority = priority


# --- Constraints ---


@configurable
@dataclass
class Constrained:
  value: Hyper[int, Ge[0], Le[100]] = 50


# =============================================================================
# Tests
# =============================================================================


class TestClassTyping:
  def test_config_returns_makeable_model(self) -> None:
    config = Model.Config(x=5, y="hello")
    assert_type(config, MakeableModel[Model])

  def test_make_returns_instance(self) -> None:
    config = Model.Config(x=5)
    instance = config.make()
    assert_type(instance, Model)

  def test_direct_instantiation(self) -> None:
    instance = Model(x=10, y="world")
    assert_type(instance, Model)


class TestDataclassTyping:
  def test_config_returns_makeable_model(self) -> None:
    config = DataModel.Config(x=5, y="hello")
    assert_type(config, MakeableModel[DataModel])

  def test_make_returns_instance(self) -> None:
    config = DataModel.Config(x=5)
    instance = config.make()
    assert_type(instance, DataModel)

  def test_direct_instantiation(self) -> None:
    instance = DataModel(x=10, y="world")
    assert_type(instance, DataModel)

  def test_init_false_field_excluded_from_config(self) -> None:
    config = DataModel.Config(x=42)
    instance = config.make()
    assert instance.computed == "computed_value"


class TestFunctionTyping:
  def test_config_returns_makeable_model(self) -> None:
    # Note: For functions, ParamSpec means Config appears to accept all params.
    # This is a known trade-off - stubs fix it.
    config = process.Config(data=[], window=20)
    assert_type(config, MakeableModel[float])

  def test_direct_call(self) -> None:
    result = process([1, 2, 3], window=2)
    assert_type(result, float)


class TestNestedConfigTyping:
  def test_nested_config_fields(self) -> None:
    config = Pipeline.Config(
      model=Model.Config(x=10),
      optimizer=Optimizer.Config(lr=0.001),
    )
    assert_type(config, MakeableModel[Pipeline])

  def test_nested_make_returns_instances(self) -> None:
    config = Pipeline.Config()
    pipeline = config.make()
    assert_type(pipeline, Pipeline)
    # After make(), nested fields should be instances
    assert_type(pipeline.model, Model)
    assert_type(pipeline.optimizer, Optimizer)

  def test_default_creates_nested_with_defaults(self) -> None:
    config = Pipeline.Config()  # Uses DEFAULT for nested
    pipeline = config.make()
    assert pipeline.model.x == 10  # Model's default
    assert pipeline.optimizer.lr == 0.01  # Optimizer's default


class TestConfigurableEqualsPattern:
  """Test the recommended `Model = configurable(Model)` pattern."""

  def test_config_returns_makeable_model(self) -> None:
    config = Layer.Config(size=128)
    assert_type(config, MakeableModel[Layer])

  def test_make_returns_instance(self) -> None:
    config = Layer.Config(size=256)
    instance = config.make()
    assert_type(instance, Layer)


class TestLiteralAndEnumTyping:
  def test_literal_in_config(self) -> None:
    config = Processor.Config(mode="eval")
    assert_type(config, MakeableModel[Processor])
    instance = config.make()
    assert instance.mode == "eval"

  def test_enum_in_config(self) -> None:
    config = Processor.Config(priority=Mode.SLOW)
    instance = config.make()
    assert instance.priority == Mode.SLOW


class TestConstraintTyping:
  def test_constrained_field_accepts_valid(self) -> None:
    config = Constrained.Config(value=75)
    assert_type(config, MakeableModel[Constrained])
    instance = config.make()
    assert instance.value == 75

  def test_constraint_default(self) -> None:
    config = Constrained.Config()
    assert config.value == 50
