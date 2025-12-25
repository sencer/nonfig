"""Simple tests for @configurable on classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from nonfig import Ge, Gt, Hyper, Le, MakeableModel, configurable


@dataclass
class SimpleDataclass:
  """Simple dataclass example."""

  Config: ClassVar[type[MakeableModel[object]]]

  value: Hyper[int, Ge[0], Le[100]] = 10


# Apply @configurable AFTER the class is defined
SimpleDataclass = configurable(SimpleDataclass)


@configurable
class SimpleClass:
  """Simple class example."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(self, multiplier: Hyper[float, Gt[0.0]] = 2.0):
    super().__init__()
    self.multiplier = multiplier

  def transform(self, x: float) -> float:
    """Transform a value."""
    return x * self.multiplier


def test_dataclass_instantiation() -> None:
  """Test creating a dataclass instance."""
  obj = SimpleDataclass()
  assert obj.value == 10

  obj2 = SimpleDataclass(value=50)
  assert obj2.value == 50


def test_dataclass_config() -> None:
  """Test dataclass Config."""
  config = SimpleDataclass.Config(value=30)
  print(f"Config dump: {config.model_dump()}")
  obj = config.make()  # make() now returns instance directly
  print(f"Object: {obj}")
  print(f"Object value: {obj.value}")
  assert isinstance(obj, SimpleDataclass)
  assert obj.value == 30


def test_class_instantiation() -> None:
  """Test creating a class instance."""
  obj = SimpleClass()
  assert obj.multiplier == 2.0
  assert obj.transform(10.0) == 20.0

  obj2 = SimpleClass(multiplier=3.0)
  assert obj2.multiplier == 3.0
  assert obj2.transform(10.0) == 30.0


def test_class_config() -> None:
  """Test class Config."""
  config = SimpleClass.Config(multiplier=4.0)
  obj = config.make()  # make() now returns instance directly
  assert isinstance(obj, SimpleClass)
  assert obj.multiplier == 4.0
  assert obj.transform(10.0) == 40.0
