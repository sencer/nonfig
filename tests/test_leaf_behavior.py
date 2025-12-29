from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, get_args

import pytest

from nonfig import Hyper, Leaf, configurable
from nonfig.extraction import transform_type_for_nesting


@configurable
class Internal:
  def __init__(self, x: int = 1):
    self.x = x


def test_leaf_prevents_transformation() -> None:
  """Test that Leaf marker prevents T -> T | T.Config transformation."""
  # Without Leaf
  t1 = transform_type_for_nesting(Internal)
  assert t1 == Internal | Internal.Config

  # With Leaf marker in Hyper
  t2 = transform_type_for_nesting(Annotated[Internal, Leaf])
  assert t2 == Annotated[Internal, Leaf]

  # With Leaf[T] syntax
  t3 = transform_type_for_nesting(Leaf[Internal])
  assert t3 == Leaf[Internal]


def test_leaf_nested_in_hyper() -> None:
  """Test that Leaf works when nested inside Hyper."""

  @configurable
  class Outer:
    def __init__(self, item: Hyper[Leaf[Internal]]):
      self.item = item

  # Config field should be Internal, not Internal | Internal.Config
  assert Outer.Config.model_fields["item"].annotation == Internal


def test_reserved_names_errors() -> None:
  """Test that clashing reserved names raise descriptive errors."""

  # Test 'make' clash
  with pytest.raises(ValueError, match="Parameter 'make' is reserved by nonfig"):

    @configurable
    def func_make(make: Hyper[int] = 1):
      pass

  # Test 'fast_make' clash

  # Test 'fast_make' clash
  with pytest.raises(ValueError, match="Parameter 'fast_make' is reserved by nonfig"):

    @configurable
    class ClassFastMake:
      def __init__(self, fast_make: int = 1):
        pass

  # Test Pydantic reserved name
  with pytest.raises(
    ValueError, match="Parameter 'model_config' is reserved by Pydantic"
  ):

    @configurable
    @dataclass
    class DataReserved:
      model_config: Hyper[int] = 1


def test_leaf_runtime_behavior() -> None:
  """Test that Leaf-marked parameters expect the raw instance at runtime."""

  @configurable
  class Processor:
    def __init__(self, internal: Hyper[Leaf[Internal]]):
      self.internal = internal

  obj = Internal(x=10)
  # Should accept Internal instance
  config = Processor.Config(internal=obj)
  proc = config.make()
  assert proc.internal.x == 10

  # Should NOT accept Internal.Config instance (validation error from Pydantic)
  # because annotation is exactly 'Internal', and Internal is not a Pydantic model.
  # create_model uses Internal as annotation.
  from pydantic import ValidationError

  with pytest.raises(ValidationError):
    # This should fail because InternalConfig is not an Internal instance
    Processor.Config(internal=Internal.Config(x=20))


def test_leaf_in_containers() -> None:
  """Test that Leaf prevents transformation even inside containers."""

  # List of Leaf[Internal]
  t1 = transform_type_for_nesting(list[Leaf[Internal]])
  assert t1 == list[Leaf[Internal]]

  # Union with Leaf[Internal]
  t2 = transform_type_for_nesting(Internal | int | Leaf[Internal])
  # The first Internal gets transformed, the Leaf[Internal] does not.
  # So we get (Internal | Internal.Config | int | Leaf[Internal])
  # Note: Internal | Leaf[Internal] might be simplified by Python to just Annotated[Internal, Leaf]
  # if Python's type system handles it that way, but here they are distinct.
  assert Internal.Config in get_args(t2)
  assert Leaf[Internal] in get_args(t2)


def test_nested_leaf_markers() -> None:
  """Test that nested Leaf markers work correctly."""
  # Hyper[Annotated[Internal, Leaf]]
  t1 = transform_type_for_nesting(Annotated[Annotated[Internal, Leaf], "other"])
  assert t1 == Annotated[Annotated[Internal, Leaf], "other"]

  # Annotated[Hyper[Internal], Leaf] -> transform_type_for_nesting is called on the type
  # after Hyper is stripped by unwrap_hyper.
  # This is handled by passing is_leaf=True to transform_type_for_nesting.
  t2 = transform_type_for_nesting(Internal, is_leaf=True)
  assert t2 == Internal
