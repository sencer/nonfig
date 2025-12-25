"""Test non-primitive Python types as Hyper parameters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from nonfig import Hyper, configurable


# Module-level enum for proper type hint resolution
class ProcessMode(StrEnum):
  FAST = "fast"
  ACCURATE = "accurate"
  BALANCED = "balanced"


def test_enum_as_hyper_parameter() -> None:
  """Test using Enum as a Hyper parameter type.

  Note: Enums defined at module level can be used directly as Hyper types.
  For enums defined locally, use .value to get the string/int value.
  """

  @configurable
  def process(mode: Hyper[ProcessMode] = ProcessMode.BALANCED) -> ProcessMode:
    return mode

  # Can use enum directly
  config = process.Config(mode=ProcessMode.FAST)
  assert config.mode == ProcessMode.FAST

  fn = config.make()
  result = fn()
  assert result == ProcessMode.FAST

  # Can also use string value (Pydantic coerces)
  config2 = process.Config(mode="accurate")
  assert config2.mode == ProcessMode.ACCURATE

  # Test with default
  config3 = process.Config()
  assert config3.mode == ProcessMode.BALANCED


def test_path_as_hyper_parameter() -> None:
  """Test pathlib.Path as a Hyper parameter."""

  @configurable
  def process(path: Hyper[Path] = Path("/tmp")) -> str:
    return str(path)

  # Test with Path object
  config = process.Config(path=Path("/home"))
  assert config.path == Path("/home")

  fn = config.make()
  result = fn()
  assert result == "/home"

  # Test coercion from string
  config2 = process.Config(path="/var/log")
  assert config2.path == Path("/var/log")


def test_tuple_as_hyper_parameter() -> None:
  """Test tuple as a Hyper parameter."""

  @configurable
  def process(dimensions: Hyper[tuple[int, int]] = (640, 480)) -> int:
    return dimensions[0] * dimensions[1]

  config = process.Config(dimensions=(1920, 1080))
  assert config.dimensions == (1920, 1080)

  fn = config.make()
  result = fn()
  assert result == 1920 * 1080


def test_set_as_hyper_parameter() -> None:
  """Test set as a Hyper parameter."""

  @configurable
  def process(tags: Hyper[set[str]] = None) -> int:
    if tags is None:
      tags = {"default"}
    return len(tags)

  config = process.Config(tags={"a", "b", "c"})
  assert config.tags == {"a", "b", "c"}

  fn = config.make()
  result = fn()
  assert result == 3


def test_callable_as_hyper_parameter() -> None:
  """Test callable/function as a Hyper parameter."""

  def default_transform(x: int) -> int:
    return x * 2

  @configurable
  def process(
    value: int,
    transform: Hyper[object] = default_transform,  # Use object for callable
  ) -> int:
    return transform(value)

  # Test with default function
  config = process.Config()
  fn = config.make()
  result = fn(value=5)
  assert result == 10

  # Test with custom function
  def custom(x: int) -> int:
    return x * 3

  config2 = process.Config(transform=custom)
  fn2 = config2.make()
  result2 = fn2(value=5)
  assert result2 == 15


def test_nested_dataclass_as_hyper() -> None:
  """Test dataclass as a Hyper parameter."""

  @dataclass
  class Settings:
    width: int
    height: int

  default_settings = Settings(640, 480)

  @configurable
  def process(settings: Hyper[object] = default_settings) -> int:
    return settings.width * settings.height

  config = process.Config(settings=Settings(1920, 1080))
  assert config.settings.width == 1920
  assert config.settings.height == 1080

  fn = config.make()
  result = fn()
  assert result == 1920 * 1080


def test_custom_class_instance() -> None:
  """Test custom class objects as Hyper parameters."""

  class CustomConfig:
    def __init__(self, value: int) -> None:
      self.value = value

    def __eq__(self, other: object) -> bool:
      if not isinstance(other, CustomConfig):
        return False
      return self.value == other.value

  default_config = CustomConfig(10)

  @configurable
  def process(config: Hyper[object] = default_config) -> int:
    return config.value * 2

  # Test with custom object
  custom = CustomConfig(5)
  pconfig = process.Config(config=custom)
  assert pconfig.config == custom

  fn = pconfig.make()
  result = fn()
  assert result == 10


def test_special_type_serialization() -> None:
  """Test that model_dump() works with special types."""

  @configurable
  def process(
    mode: Hyper[str] = "a",
    path: Hyper[Path] = Path("/tmp"),
    tags: Hyper[set[str]] = None,
  ) -> str:
    if tags is None:
      tags = {"default"}
    return f"{mode}-{path}-{len(tags)}"

  config = process.Config(
    mode="b",
    path=Path("/home"),
    tags={"x", "y"},
  )

  # model_dump should work
  data = config.model_dump()
  assert data["mode"] == "b"
  assert "path" in data
  assert "tags" in data

  # Should be able to recreate from dump
  config2 = process.Config(**data)
  assert config2.mode == "b"
