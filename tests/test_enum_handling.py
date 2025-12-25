"""Tests for enum handling.

These tests verify that:
- Enums used as Hyper type annotations work correctly at runtime
"""

from __future__ import annotations

from enum import Enum, StrEnum

from nonfig import Hyper, configurable


class Mode(StrEnum):
  """Example enum for testing."""

  FAST = "fast"
  SLOW = "slow"
  MEDIUM = "medium"


class Priority(int, Enum):
  """Integer enum for testing."""

  LOW = 1
  MEDIUM = 2
  HIGH = 3


class TestEnumRuntime:
  """Runtime tests for enum handling."""

  def test_enum_hyper_param(self) -> None:
    """Enum can be used as Hyper type annotation."""

    @configurable
    def process(
      mode: Hyper[Mode] = Mode.FAST,
    ) -> str:
      return f"Processing in {mode.value} mode"

    config = process.Config(mode=Mode.SLOW)
    fn = config.make()
    result = fn()
    assert result == "Processing in slow mode"

  def test_enum_string_coercion(self) -> None:
    """Pydantic should coerce string to enum value."""

    @configurable
    def process(
      mode: Hyper[Mode] = Mode.FAST,
    ) -> str:
      return mode.value

    # Pass string value instead of enum
    config = process.Config(mode="slow")  # type: ignore[arg-type]
    fn = config.make()
    result = fn()
    assert result == "slow"

  def test_int_enum(self) -> None:
    """Integer enums should work with Hyper."""

    @configurable
    def prioritize(
      priority: Hyper[Priority] = Priority.MEDIUM,
    ) -> int:
      return priority.value

    config = prioritize.Config(priority=Priority.HIGH)
    fn = config.make()
    assert fn() == 3

  def test_enum_in_class(self) -> None:
    """Enum Hyper param in class __init__."""

    @configurable
    class Processor:
      def __init__(self, mode: Hyper[Mode] = Mode.FAST) -> None:
        self.mode = mode

    config = Processor.Config(mode=Mode.SLOW)
    instance = config.make()
    assert instance.mode == Mode.SLOW
