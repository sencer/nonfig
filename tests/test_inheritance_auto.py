"""Test recursive configurability via __init_subclass__."""

from __future__ import annotations

from typing import ClassVar

from nonfig import MakeableModel, configurable


def test_recursive_inheritance_simple() -> None:
  """Test that a subclass of a configurable class is automatically configurable."""

  @configurable
  class Base:
    Config: ClassVar[type[MakeableModel[object]]]

    def __init__(self, x: int = 1) -> None:
      self.x = x

  class Sub(Base):
    def __init__(self, y: int = 2) -> None:
      super().__init__()
      self.y = y

  # Sub should be automatically configured
  assert hasattr(Sub, "Config")

  # Config should be for Sub, not Base
  config = Sub.Config(y=10)
  assert config.y == 10

  # make() should return Sub instance
  obj = config.make()
  assert isinstance(obj, Sub)
  assert obj.y == 10
  assert obj.x == 1  # Default from Base


def test_multilevel_inheritance() -> None:
  """Test inheritance chain: Base -> Sub -> SubSub."""

  @configurable
  class Base:
    def __init__(self, val: int = 1) -> None:
      self.val = val

  class Sub(Base):
    pass

  class SubSub(Sub):
    def __init__(self, extra: int = 3) -> None:
      super().__init__()
      self.extra = extra

  # SubSub should be configured
  assert hasattr(SubSub, "Config")

  # Check extraction
  config = SubSub.Config(extra=30)
  obj = config.make()

  assert isinstance(obj, SubSub)
  assert obj.extra == 30
  assert obj.val == 1


def test_mixin_inheritance() -> None:
  """Test inheritance with mixins."""

  class Mixin:
    def mixin_method(self) -> str:
      return "mixin"

  @configurable
  class Base:
    pass

  class Sub(Mixin, Base):
    def __init__(self, m: int = 5) -> None:
      self.m = m

  # Sub should be configured
  config = Sub.Config(m=50)
  obj = config.make()

  assert isinstance(obj, Sub)
  assert obj.m == 50
  assert obj.mixin_method() == "mixin"


def test_explicit_decoration_idempotency() -> None:
  """Test that explicit @configurable on subclass works fine (idempotent)."""

  @configurable
  class Base:
    pass

  @configurable
  class Sub(Base):
    def __init__(self, z: int = 9) -> None:
      self.z = z

  # Should still work and not double-configure or error
  config = Sub.Config(z=90)
  obj = config.make()

  assert isinstance(obj, Sub)
  assert obj.z == 90


def test_init_subclass_is_preserved() -> None:
  """Test that user-defined __init_subclass__ is preserved and called."""

  events = []

  @configurable
  class Base:
    def __init_subclass__(cls, **kwargs) -> None:
      super().__init_subclass__(**kwargs)
      events.append(f"Base init_subclass for {cls.__name__}")

  class Sub(Base):
    pass

  class SubSub(Sub):
    pass

  # Events should have fired during class definition
  assert "Base init_subclass for Sub" in events
  assert "Base init_subclass for SubSub" in events

  # And they should be configurable
  assert hasattr(Sub, "Config")
  assert hasattr(SubSub, "Config")
