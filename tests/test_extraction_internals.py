from __future__ import annotations

import dataclasses
from typing import Annotated, Any
from unittest.mock import patch

import pytest

from nonfig import DEFAULT, Hyper, MakeableModel
from nonfig.extraction import (
  _config_creation_stack,
  _instantiate_default_config,
  create_field_info,
  get_type_hints_safe,
  unwrap_hyper,
)
from nonfig.typedefs import HyperMarker


def test_get_type_hints_name_error_on_class() -> None:
  """Test fallback logic when get_type_hints raises NameError for a class."""

  class MyClass:
    pass

  # Mock get_type_hints to raise NameError
  call_count = 0

  def side_effect(*args, **kwargs):
    nonlocal call_count
    call_count += 1
    raise NameError("Mocked NameError")

  with patch("nonfig.extraction.get_type_hints", side_effect=side_effect):
    hints = get_type_hints_safe(MyClass)

  assert hints == {}  # Falls back to __annotations__ which is empty
  assert call_count == 1


def test_nested_config_instantiation_failure() -> None:
  """Test error handling when nested config instantiation fails."""

  class BrokenConfig(MakeableModel[Any]):
    def make(self) -> Any:
      return None

    def __init__(self, **data: Any):
      raise RuntimeError("Intentional failure")

  # Create a field with DEFAULT - this should fail because BrokenConfig raises
  inner_type, constraints = unwrap_hyper(Annotated[BrokenConfig, HyperMarker])

  with pytest.raises(TypeError, match="Failed to instantiate"):
    create_field_info("param", inner_type, DEFAULT, constraints)


def test_unwrap_hyper() -> None:
  """Test unwrap_hyper function."""
  ann = Hyper[int]
  inner, constraints = unwrap_hyper(ann)
  assert inner is int
  assert constraints == ()

  from nonfig import Ge, Le

  ann2 = Hyper[int, Ge[0], Le[100]]
  inner2, constraints2 = unwrap_hyper(ann2)
  assert inner2 is int
  assert len(constraints2) == 2


def test_circular_dependency_detection_manual() -> None:
  """Test that manual recursion in _instantiate_default_config detects cycles."""

  class A(MakeableModel[Any]):
    pass

  config_name = f"{A.__module__}.{A.__qualname__}"

  # Use .set() to manipulate the context variable
  token = _config_creation_stack.set([config_name, "B"])
  try:
    with pytest.raises(ValueError, match="Circular dependency detected"):
      _instantiate_default_config(A, "p")
  finally:
    _config_creation_stack.reset(token)


def test_dataclass_default_factory() -> None:
  """Test extraction of dataclass fields with default_factory."""

  @dataclasses.dataclass
  class Container:
    items: list[int] = dataclasses.field(default_factory=list)

  get_type_hints_safe(Container)
  # This just ensures our extraction logic doesn't crash on default_factory
  # Real test is via extract_class_params, which is covered by higher level tests,
  # but we can verify the low level utility here if needed.
  pass
