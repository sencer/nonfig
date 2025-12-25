from __future__ import annotations

from typing import Annotated, Literal, NewType

from annotated_types import Ge as AnnotatedGe
from pydantic import ValidationError
import pytest

from nonfig import DEFAULT, Ge, Hyper, Le, configurable
from nonfig.extraction import has_hyper_marker, unwrap_hyper
from nonfig.models import BoundFunction

"""Consolidated extraction tests."""

"""Tests for extraction edge cases."""

# Define a NewType
UserId = NewType("UserId", int)


def test_new_type_extraction():
  """Test extracting parameters typed with NewType."""

  @configurable
  def fn_uid(uid: Hyper[UserId] = UserId(1)):
    return uid

  config = fn_uid.Config()
  assert config.uid == 1


# Annotated[int, "metadata"]
TaggedInt = Annotated[int, "tag"]


def test_annotated_nested_extraction():
  """Test Hyper wrapping an already Annotated type."""

  @configurable
  def fn_annotated_nested_extraction(x: Hyper[TaggedInt, Ge[0]] = 1):
    return x

  config = fn_annotated_nested_extraction.Config(x=5)
  assert config.x == 5

  # Check constraints applied
  with pytest.raises(ValidationError):  # Pydantic validation error
    fn_annotated_nested_extraction.Config(x=-1)


def test_hyper_marker_check():
  assert has_hyper_marker(Hyper[int])
  assert not has_hyper_marker(int)
  assert not has_hyper_marker(Annotated[int, "tag"])


def test_unwrap_hyper_logic():
  # Unwrap Hyper[int, Ge[1]]
  inner, constraints = unwrap_hyper(Hyper[int, Ge[1]])
  assert inner is int
  assert len(constraints) == 1
  assert isinstance(constraints[0], AnnotatedGe)

  # Unwrap plain type
  inner, constraints = unwrap_hyper(int)
  assert inner is int
  assert len(constraints) == 0

  # Unwrap Annotated but not Hyper
  inner, constraints = unwrap_hyper(Annotated[int, Ge[1]])
  # Note: extraction logic currently filters out HyperMarker.
  # If using unwrap_hyper on generic Annotated, it returns inner + constraints
  # as long as they are not HyperMarker.
  assert inner is int
  # Ge[1] is a constraint, so it should be returned
  print(f"DEBUG: constraints={constraints}")
  assert len(constraints) == 1
  # nonfig.Ge[...] produces an annotated_types.Ge instance
  assert isinstance(constraints[0], AnnotatedGe)


"""Test the new Hyper[T, Ge[2], Le[100]] syntax in type annotations."""


@configurable
def constrained_function(
  value: float,
  period: Hyper[int, Ge[2], Le[100]] = 14,
  threshold: Hyper[float, Ge[0.0], Le[1.0]] = 0.5,
) -> float:
  """Test function with constraints in type annotations."""
  return value * period * threshold


def test_annotated_syntax_basic() -> None:
  """Test that the new syntax works for basic calls."""
  result = constrained_function(10.0)
  assert result == 70.0  # 10 * 14 * 0.5


def test_annotated_syntax_with_params() -> None:
  """Test calling with custom parameters."""
  result = constrained_function(10.0, period=5, threshold=0.2)
  assert result == 10.0  # 10 * 5 * 0.2


def test_annotated_syntax_config() -> None:
  """Test that Config class works with annotated syntax."""
  config = constrained_function.Config(period=10, threshold=0.8)
  assert config.period == 10
  assert config.threshold == 0.8


def test_annotated_syntax_validation() -> None:
  """Test that Field constraints from annotations are enforced."""
  # Valid values should work
  config = constrained_function.Config(period=50, threshold=0.5)
  assert config.period == 50

  # Test lower bound for period
  with pytest.raises(ValidationError):
    constrained_function.Config(period=1, threshold=0.5)

  # Test upper bound for period
  with pytest.raises(ValidationError):
    constrained_function.Config(period=101, threshold=0.5)

  # Test lower bound for threshold
  with pytest.raises(ValidationError):
    constrained_function.Config(period=10, threshold=-0.1)

  # Test upper bound for threshold
  with pytest.raises(ValidationError):
    constrained_function.Config(period=10, threshold=1.1)


def test_annotated_syntax_make() -> None:
  """Test that Config.make() works with annotated syntax."""
  config = constrained_function.Config(period=20, threshold=0.25)
  fn = config.make()
  result = fn(value=10.0)
  assert result == 50.0  # 10 * 20 * 0.25


@configurable
def nested(x: Hyper[int] = 1) -> int:
  return x


@configurable
def target_func(data: list[int], m: nested.Type = DEFAULT) -> int:
  """A function with implicit hyper parameter via DEFAULT."""
  return sum(data) + m()


@configurable
def target_func_explicit_config(
  data: list[int],
  m: nested.Type = nested.Config(x=10),  # noqa: B008
) -> int:
  """A function with implicit hyper parameter via Config object."""
  return sum(data) + m()


def test_implicit_hyper_via_default():
  # Verify it's in the Config fields
  assert "m" in target_func.Config.model_fields

  # Test direct call
  assert target_func([1, 2], m=lambda: 5) == 8

  # Test via Config.make()
  cfg = target_func.Config(m={"x": 5})
  fn = cfg.make()
  assert isinstance(fn, BoundFunction)
  assert fn.m.x == 5
  assert fn([1, 2]) == 8


def test_implicit_hyper_via_config_object():
  # Verify it's in the Config fields
  assert "m" in target_func_explicit_config.Config.model_fields

  # Test direct call
  assert target_func_explicit_config([1, 2], m=lambda: 5) == 8

  # Test default behavior
  cfg = target_func_explicit_config.Config()
  fn = cfg.make()
  assert fn.m.x == 10
  assert fn([1, 2]) == 13


def test_regular_params_still_ignored():
  @configurable
  def func_with_mixed(data: list[int], scale: float = 1.0, bias: Hyper[float] = 0.0):
    pass

  # 'scale' should NOT be in Config because it's not Hyper and doesn't use DEFAULT/Config
  assert "bias" in func_with_mixed.Config.model_fields
  assert "scale" not in func_with_mixed.Config.model_fields
  assert "data" not in func_with_mixed.Config.model_fields


"""Test support for Literal types in hyperparameters."""


def test_literal_type_basic() -> None:
  """Test that Literal types work for string enums."""

  @configurable
  def process(
    mode: Hyper[Literal["fast", "slow", "medium"]] = "fast",
  ) -> str:
    return f"Processing in {mode} mode"

  # Default should work
  assert process() == "Processing in fast mode"

  # Valid values should work
  assert process(mode="slow") == "Processing in slow mode"
  assert process(mode="medium") == "Processing in medium mode"

  # Config should work
  config = process.Config(mode="slow")
  fn = config.make()
  assert fn() == "Processing in slow mode"


def test_literal_validation() -> None:
  """Test that Literal types validate correctly."""

  @configurable
  def process(
    mode: Hyper[Literal["fast", "slow"]] = "fast",
  ) -> str:
    return mode

  # Invalid value should raise ValidationError
  with pytest.raises(ValidationError) as exc_info:
    process.Config(mode="invalid")

  error_str = str(exc_info.value)
  assert "mode" in error_str.lower()


def test_literal_int_types() -> None:
  """Test Literal with integer values."""

  @configurable
  def process(
    level: Hyper[Literal[1, 2, 3]] = 1,
  ) -> int:
    return level * 10

  assert process() == 10
  assert process(level=2) == 20
  assert process(level=3) == 30

  # Invalid value
  with pytest.raises(ValidationError):
    process.Config(level=4)


def test_literal_mixed_types() -> None:
  """Test Literal with mixed types."""

  @configurable
  def process(
    value: Hyper[Literal[1, "auto", True]] = "auto",
  ) -> str:
    return str(value)

  assert process() == "auto"
  assert process(value=1) == "1"
  assert process(value=True) == "True"

  config = process.Config(value=1)
  fn = config.make()
  assert fn() == "1"


def test_literal_in_class() -> None:
  """Test Literal types in class configurations."""

  @configurable
  class Optimizer:
    def __init__(
      self,
      algorithm: Hyper[Literal["sgd", "adam", "rmsprop"]] = "adam",
    ) -> None:
      self.algorithm = algorithm

  # Default
  opt = Optimizer()
  assert opt.algorithm == "adam"

  # Custom config
  config = Optimizer.Config(algorithm="sgd")
  opt2 = config.make()  # Now returns instance directly
  assert isinstance(opt2, Optimizer)
  assert opt2.algorithm == "sgd"


def test_literal_serialization() -> None:
  """Test that Literal types serialize correctly."""

  @configurable
  def process(
    mode: Hyper[Literal["fast", "slow"]] = "fast",
  ) -> str:
    return mode

  config = process.Config(mode="slow")

  # Serialize to dict
  config_dict = config.model_dump()
  assert config_dict == {"mode": "slow"}

  # Serialize to JSON
  config_json = config.model_dump_json()
  assert '"mode":"slow"' in config_json or '"mode": "slow"' in config_json

  # Deserialize
  loaded = process.Config.model_validate_json(config_json)
  assert loaded.mode == "slow"
