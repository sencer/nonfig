"""Tests for the run_cli function."""

import pytest

from nonfig import Hyper, configurable, run_cli


@configurable
def simple_func(*, name: Hyper[str] = "World", count: Hyper[int] = 1) -> str:
  return f"{name}:{count}"


@configurable
class SimpleClass:
  def __init__(self, *, value: int = 10, label: str = "default") -> None:
    self.value = value
    self.label = label


class TestRunCli:
  """Tests for run_cli function."""

  def test_run_cli_with_no_args(self) -> None:
    """run_cli with no args uses defaults."""
    result = run_cli(simple_func, [])
    bound = result
    output = bound()
    assert output == "World:1"

  def test_run_cli_with_simple_override(self) -> None:
    """run_cli parses key=value overrides."""
    result = run_cli(simple_func, ["name=Claude"])
    output = result()
    assert output == "Claude:1"

  def test_run_cli_with_int_coercion(self) -> None:
    """run_cli coerces int values."""
    result = run_cli(simple_func, ["count=5"])
    output = result()
    assert output == "World:5"

  def test_run_cli_with_multiple_overrides(self) -> None:
    """run_cli handles multiple overrides."""
    result = run_cli(simple_func, ["name=Test", "count=42"])
    output = result()
    assert output == "Test:42"

  def test_run_cli_with_class(self) -> None:
    """run_cli works with configurable classes."""
    result = run_cli(SimpleClass, ["value=99", "label=custom"])
    assert result.value == 99
    assert result.label == "custom"

  def test_run_cli_raises_for_non_configurable(self) -> None:
    """run_cli raises TypeError for non-configurable targets."""

    def plain_func() -> None:
      pass

    with pytest.raises(TypeError, match="not configurable"):
      run_cli(plain_func, [])


class TestCliValueParsing:
  """Tests for CLI value parsing."""

  def test_parse_none_value(self) -> None:
    """Parsing 'none' returns None."""
    from nonfig.cli.runner import _parse_value

    assert _parse_value("none", None) is None
    assert _parse_value("None", None) is None
    assert _parse_value("NONE", None) is None

  def test_parse_bool_true(self) -> None:
    """Parsing true-like values returns True."""
    from nonfig.cli.runner import _parse_value

    assert _parse_value("true", None) is True
    assert _parse_value("True", None) is True
    assert _parse_value("yes", None) is True
    assert _parse_value("1", None) is True

  def test_parse_bool_false(self) -> None:
    """Parsing false-like values returns False."""
    from nonfig.cli.runner import _parse_value

    assert _parse_value("false", None) is False
    assert _parse_value("False", None) is False
    assert _parse_value("no", None) is False
    assert _parse_value("0", None) is False

  def test_parse_int(self) -> None:
    """Parsing numeric strings without decimal returns int."""
    from nonfig.cli.runner import _parse_value

    assert _parse_value("42", None) == 42
    assert _parse_value("-10", None) == -10

  def test_parse_float(self) -> None:
    """Parsing numeric strings with decimal returns float."""
    from nonfig.cli.runner import _parse_value

    assert _parse_value("3.14", None) == 3.14
    assert _parse_value("-0.5", None) == -0.5

  def test_parse_with_type_hint(self) -> None:
    """Parsing with type hint uses that type."""
    from nonfig.cli.runner import _parse_value

    assert _parse_value("42", int) == 42
    assert _parse_value("3.14", float) == 3.14
    assert _parse_value("hello", str) == "hello"

  def test_parse_string_fallback(self) -> None:
    """Non-numeric strings return as-is."""
    from nonfig.cli.runner import _parse_value

    assert _parse_value("hello", None) == "hello"
    assert _parse_value("path/to/file", None) == "path/to/file"


class TestCliOverrideParsing:
  """Tests for CLI override parsing."""

  def test_parse_simple_overrides(self) -> None:
    """Parse simple key=value pairs."""
    from nonfig.cli.runner import _parse_overrides

    result = _parse_overrides(["a=1", "b=hello"])
    assert result == {"a": "1", "b": "hello"}

  def test_parse_nested_overrides(self) -> None:
    """Parse dot-notation for nested structures."""
    from nonfig.cli.runner import _parse_overrides

    result = _parse_overrides(["optimizer.lr=0.01", "optimizer.momentum=0.9"])
    assert result == {"optimizer": {"lr": "0.01", "momentum": "0.9"}}

  def test_parse_ignores_non_keyvalue(self) -> None:
    """Non key=value args are ignored."""
    from nonfig.cli.runner import _parse_overrides

    result = _parse_overrides(["--help", "a=1", "positional"])
    assert result == {"a": "1"}

  def test_parse_value_with_equals(self) -> None:
    """Values can contain equals signs."""
    from nonfig.cli.runner import _parse_overrides

    result = _parse_overrides(["url=http://example.com?a=b&c=d"])
    assert result == {"url": "http://example.com?a=b&c=d"}


class TestNestedConfigCli:
  """Tests for nested config handling in CLI."""

  def test_nested_config_overrides(self) -> None:
    """Nested config values are properly coerced."""
    from nonfig import configurable, run_cli

    @configurable
    class Inner:
      def __init__(self, x: int = 0) -> None:
        self.x = x

    @configurable
    class Outer:
      def __init__(
        self, inner: Inner.Config | None = None, name: str = "default"
      ) -> None:
        self.inner = inner
        self.name = name

    # Test with simple override (name)
    result = run_cli(Outer, ["name=custom"])
    assert result.name == "custom"

  def test_parse_optional_type(self) -> None:
    """Parse value with Optional[int] type hint."""
    from nonfig.cli.runner import _parse_value

    # Test with optional int
    result = _parse_value("42", int | None)
    assert result == 42

  def test_get_nested_config_cls_with_union(self) -> None:
    """_get_nested_config_cls finds config in union types."""
    from nonfig import configurable
    from nonfig.cli.runner import _get_nested_config_cls

    @configurable
    class Sub:
      def __init__(self, val: int = 0) -> None:
        self.val = val

    # Test with union containing Config
    result = _get_nested_config_cls(Sub.Config | None)
    assert result is Sub.Config

    # Test with non-config type
    result = _get_nested_config_cls(int | None)
    assert result is None

    # Test with direct config type
    result = _get_nested_config_cls(Sub.Config)
    assert result is Sub.Config

  def test_apply_type_coercion_nested(self) -> None:
    """_apply_type_coercion handles nested dicts."""
    from nonfig import configurable
    from nonfig.cli.runner import _apply_type_coercion

    @configurable
    class Sub:
      def __init__(self, count: int = 0) -> None:
        self.count = count

    @configurable
    class Parent:
      def __init__(self, sub: Sub.Config | None = None) -> None:
        self.sub = sub

    # Test nested dict coercion
    overrides = {"sub": {"count": "42"}}
    result = _apply_type_coercion(overrides, Parent.Config)
    assert result["sub"]["count"] == 42
