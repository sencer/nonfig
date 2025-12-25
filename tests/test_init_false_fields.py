"""Tests for init=False dataclass field handling.

These tests verify that fields marked with init=False are correctly
excluded from Config generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nonfig import Hyper, configurable


class TestInitFalseRuntime:
  """Runtime tests for init=False field handling."""

  def test_init_false_field_excluded_from_config(self) -> None:
    """Fields with init=False should not appear in Config."""

    @configurable
    @dataclass
    class MyDataclass:
      included: Hyper[int] = 10
      excluded: int = field(default=5, init=False)

    # Config should only have 'included'
    config = MyDataclass.Config(included=20)
    assert hasattr(config, "included")
    assert not hasattr(config, "excluded")

  def test_init_false_field_not_in_config_init(self) -> None:
    """Config __init__ should not accept init=False fields."""

    @configurable
    @dataclass
    class MyDataclass:
      a: Hyper[int] = 1
      computed: str = field(default="computed", init=False)

    # Should work - only 'a' is accepted
    config = MyDataclass.Config(a=5)
    instance = config.make()
    assert instance.a == 5
    # The computed field gets its default value
    assert instance.computed == "computed"

  def test_init_false_with_factory(self) -> None:
    """Fields with init=False and default_factory should be excluded."""

    @configurable
    @dataclass
    class WithFactory:
      items: Hyper[int] = 10
      internal_list: list = field(default_factory=list, init=False)

    config = WithFactory.Config(items=20)
    instance = config.make()
    assert instance.items == 20
    assert instance.internal_list == []

  def test_mixed_hyper_and_regular_with_init_false(self) -> None:
    """Test mixture of Hyper, regular fields, and init=False."""

    @configurable
    @dataclass
    class MixedClass:
      name: Hyper[str] = "default"
      count: Hyper[int] = 0
      internal_state: str = field(default="state", init=False)
      computed_value: int = field(default=0, init=False)

    config = MixedClass.Config(name="test", count=5)
    instance = config.make()

    assert instance.name == "test"
    assert instance.count == 5
    assert instance.internal_state == "state"
    assert instance.computed_value == 0
