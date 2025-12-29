"""Test smart parameter propagation via **kwargs."""

from __future__ import annotations

from typing import ClassVar

from nonfig import MakeableModel, configurable


def test_kwargs_propagation() -> None:
  """Test that **kwargs pulls in base class params."""

  @configurable
  class Base:
    Config: ClassVar[type[MakeableModel[object]]]

    def __init__(self, x: int = 1) -> None:
      self.x = x

  # Sub accepts kwargs -> should get 'x'
  class Sub(Base):
    def __init__(self, y: int = 2, **kwargs) -> None:
      super().__init__(**kwargs)
      self.y = y

  assert hasattr(Sub, "Config")
  assert "x" in Sub.Config.model_fields

  config = Sub.Config(x=10, y=20)
  obj = config.make()

  assert isinstance(obj, Sub)
  assert obj.x == 10  # Correctly passed to base
  assert obj.y == 20


def test_no_kwargs_no_propagation() -> None:
  """Test that WITHOUT **kwargs, we strictly respect signature (no propagation)."""

  @configurable
  class Base:
    def __init__(self, x: int = 1) -> None:
      self.x = x

  class Sub(Base):
    def __init__(self, y: int = 2) -> None:
      super().__init__()
      self.y = y

  # Sub config should NOT have 'x' because it can't accept it at runtime
  assert "x" not in Sub.Config.model_fields


def test_explicit_override_precedence() -> None:
  """Test that explicit parameters override propagated ones."""

  @configurable
  class Base:
    def __init__(self, x: int = 1) -> None:
      self.x = x

  class SubOverride(Base):
    # Override default of x to 100
    def __init__(self, x: int = 100, **kwargs) -> None:
      super().__init__(x=x, **kwargs)

  assert "x" in SubOverride.Config.model_fields

  # Check default value
  field_info = SubOverride.Config.model_fields["x"]
  assert field_info.default == 100

  # Functionality
  config = SubOverride.Config()
  obj = config.make()
  assert obj.x == 100


def test_nested_propagation() -> None:
  """Test multi-level propagation."""

  @configurable
  class Level1:
    def __init__(self, p1: int = 1) -> None:
      self.p1 = p1

  @configurable
  class Level2(Level1):
    def __init__(self, p2: int = 2, **kwargs) -> None:
      super().__init__(**kwargs)
      self.p2 = p2

  class Level3(Level2):
    def __init__(self, p3: int = 3, **kwargs) -> None:
      super().__init__(**kwargs)
      self.p3 = p3

  # Level3 should have p1, p2, p3
  fields = Level3.Config.model_fields
  assert "p1" in fields
  assert "p2" in fields
  assert "p3" in fields

  config = Level3.Config(p1=10, p2=20, p3=30)
  obj = config.make()
  assert obj.p1 == 10
  assert obj.p2 == 20
  assert obj.p3 == 30
