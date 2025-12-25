"""Test @configurable with class inheritance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from nonfig import Ge, Hyper, Le, MakeableModel, configurable


def test_configurable_with_inheritance() -> None:
  """Test @configurable on a class with inheritance."""

  class BaseClass:
    def __init__(self, base_value: int = 1) -> None:
      super().__init__()
      self.base_value = base_value

  @configurable
  class DerivedClass(BaseClass):
    Config: ClassVar[type[MakeableModel[object]]]

    def __init__(
      self,
      base_value: int = 1,
      derived_value: Hyper[int, Ge[0]] = 10,
    ) -> None:
      super().__init__(base_value)
      self.derived_value = derived_value

  # Should work normally
  obj = DerivedClass(base_value=5, derived_value=20)
  assert obj.base_value == 5
  assert obj.derived_value == 20

  # Config now has ALL params (not just Hyper) for classes
  config = DerivedClass.Config(base_value=7, derived_value=30)
  assert config.derived_value == 30
  assert config.base_value == 7

  # make() returns instance directly
  obj2 = config.make()
  assert isinstance(obj2, DerivedClass)
  assert obj2.base_value == 7
  assert obj2.derived_value == 30


def test_multiple_inheritance() -> None:
  """Test @configurable with multiple base classes."""

  class Mixin1:
    def method1(self) -> str:
      return "mixin1"

  class Mixin2:
    def method2(self) -> str:
      return "mixin2"

  @configurable
  class MultiClass(Mixin1, Mixin2):
    Config: ClassVar[type[MakeableModel[object]]]

    def __init__(self, value: Hyper[int] = 10) -> None:
      super().__init__()
      self.value = value

  # Should work with multiple inheritance
  obj = MultiClass(value=5)
  assert obj.value == 5
  assert obj.method1() == "mixin1"
  assert obj.method2() == "mixin2"

  # Config should work normally
  config = MultiClass.Config(value=20)
  obj2 = config.make()  # Returns instance directly
  assert isinstance(obj2, MultiClass)
  assert obj2.value == 20


def test_abstract_base_class() -> None:
  """Test @configurable with ABC."""

  class BaseProcessor(ABC):
    @abstractmethod
    def process(self) -> int:
      pass

  @configurable
  class ConcreteProcessor(BaseProcessor):
    Config: ClassVar[type[MakeableModel[object]]]

    def __init__(self, multiplier: Hyper[int] = 2) -> None:
      super().__init__()
      self.multiplier = multiplier

    def process(self) -> int:
      return self.multiplier * 10

  # Should work with abstract base classes
  obj = ConcreteProcessor(multiplier=3)
  assert obj.process() == 30

  # Config should work
  config = ConcreteProcessor.Config(multiplier=5)
  obj2 = config.make()  # Returns instance directly
  assert isinstance(obj2, ConcreteProcessor)
  assert obj2.process() == 50


def test_super_call_with_hyper_params() -> None:
  """Test that super().__init__() works correctly with Hyper params."""

  @configurable
  class BaseClass:
    Config: ClassVar[type[MakeableModel[object]]]

    def __init__(self, base_param: Hyper[int] = 5) -> None:
      super().__init__()
      self.base_param = base_param

  @configurable
  class DerivedClass(BaseClass):
    Config: ClassVar[type[MakeableModel[object]]]

    def __init__(
      self,
      base_param: Hyper[int] = 5,
      derived_param: Hyper[int] = 10,
    ) -> None:
      super().__init__(base_param=base_param)
      self.derived_param = derived_param

  # Create derived class directly
  obj = DerivedClass(base_param=7, derived_param=14)
  assert obj.base_param == 7
  assert obj.derived_param == 14

  # Create via config
  config = DerivedClass.Config(base_param=3, derived_param=6)
  obj2 = config.make()  # Returns instance directly
  assert isinstance(obj2, DerivedClass)
  assert obj2.base_param == 3
  assert obj2.derived_param == 6


def test_method_resolution_order() -> None:
  """Test that MRO is respected with multiple inheritance."""

  class A:
    def __init__(self) -> None:
      super().__init__()
      self.a_value = "A"

  class B(A):
    def __init__(self) -> None:
      super().__init__()
      self.b_value = "B"

  class C(A):
    def __init__(self) -> None:
      super().__init__()
      self.c_value = "C"

  @configurable
  class D(B, C):
    Config: ClassVar[type[MakeableModel[object]]]

    def __init__(self, value: Hyper[int] = 10) -> None:
      super().__init__()
      self.value = value

  # Should follow MRO: D -> B -> C -> A
  obj = D(value=5)
  assert obj.value == 5
  assert obj.a_value == "A"
  assert obj.b_value == "B"
  assert obj.c_value == "C"

  # Config should work
  config = D.Config(value=20)
  obj2 = config.make()  # Returns instance directly
  assert isinstance(obj2, D)
  assert obj2.value == 20
  assert obj2.a_value == "A"
  assert obj2.b_value == "B"
  assert obj2.c_value == "C"


def test_inheritance_constraint_override() -> None:
  """Test overriding Hyper constraints in subclasses."""
  from dataclasses import dataclass

  from pydantic import ValidationError
  import pytest

  @configurable
  @dataclass
  class Base:
    x: Hyper[int, Ge[0], Le[10]] = 5
    y: int = 1

  @configurable
  @dataclass
  class Derived(Base):
    # Override constraints: tighter lower bound, higher upper bound
    x: Hyper[int, Ge[20], Le[30]] = 25

  # Base validation
  assert Base.Config(x=0).x == 0
  assert Base.Config(x=10).x == 10

  with pytest.raises(ValidationError):
    Base.Config(x=-1)

  with pytest.raises(ValidationError):
    Base.Config(x=11)

  # Derived validation
  assert Derived.Config(x=20).x == 20
  assert Derived.Config(x=30).x == 30

  with pytest.raises(ValidationError):
    Derived.Config(x=19)

  with pytest.raises(ValidationError):
    Derived.Config(x=31)

  # Check inherited field 'y' works in Config
  assert Derived.Config(y=100).y == 100
  obj = Derived.Config(y=100).make()
  assert obj.y == 100
