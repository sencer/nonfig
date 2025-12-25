"""Test circular dependency error paths to improve coverage.

This file tests the specific error conditions in circular dependency detection
that are hard to trigger through normal usage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

# fmt: off
# fmt: on
from nonfig import (
  DEFAULT,
  Hyper,
  MakeableModel,
  configurable,
)

if TYPE_CHECKING:
  from collections.abc import Callable

# Module-level configs for testing


class BrokenConfigWithStringMake(MakeableModel[int]):
  """Config with non-callable make attribute (string)."""

  value: int = 10


# Override make after class definition to avoid Pydantic field issues
BrokenConfigWithStringMake.make = "not_callable"


class ConfigWithInvalidMakeType(MakeableModel[int]):
  """Config where make is an integer."""

  value: int = 10


# Override after definition
ConfigWithInvalidMakeType.make = 42


class FailingConfigOnInit(MakeableModel[int]):
  """Config that fails during instantiation."""

  def __init__(self, **data: object) -> None:
    if not data:  # Fails when instantiated with no arguments (DEFAULT case)
      raise RuntimeError("Intentional failure for testing")
    super().__init__(**data)

  def make(self) -> Callable[[], int]:
    """Dummy make method."""
    return lambda: 42


# Tests


def test_nested_config_without_callable_make_method() -> None:
  """Test error when nested config doesn't have a callable make method.

  This tests the code path where we check for a callable make method.
  """
  with pytest.raises(TypeError, match="does not have a callable 'make' method"):

    @configurable
    def broken_processor(
      data: int,
      broken_config: Hyper[BrokenConfigWithStringMake] = DEFAULT,
    ) -> int:
      return data


def test_nested_config_instantiation_failure() -> None:
  """Test error handling when nested config instantiation fails.

  This tests the code path where we catch exceptions
  during config instantiation (wrapped as TypeError).
  """
  with pytest.raises(
    TypeError, match="Failed to instantiate nested config type 'FailingConfigOnInit'"
  ):

    @configurable
    def failing_processor(
      data: int,
      failing_config: Hyper[FailingConfigOnInit] = DEFAULT,
    ) -> int:
      return data


def test_nested_config_with_invalid_make_attribute_type() -> None:
  """Test error when make attribute exists but is not callable (integer).

  This tests the code path for non-callable make method detection.
  """
  with pytest.raises(TypeError, match="does not have a callable 'make' method"):

    @configurable
    def invalid_make_processor(
      data: int,
      config: Hyper[ConfigWithInvalidMakeType] = DEFAULT,
    ) -> int:
      return data
