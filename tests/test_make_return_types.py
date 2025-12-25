"""Tests for make() return type handling.

These tests verify that:
- Functions: make() returns Callable[[DataParams], ReturnType]
- Classes: make() returns the instance directly
"""

from __future__ import annotations

from dataclasses import dataclass

from nonfig import Hyper, configurable


class TestMakeReturnTypes:
  """Tests for make() return type behavior."""

  def test_function_make_returns_callable(self) -> None:
    """For functions, make() returns a callable that accepts data params."""

    @configurable
    def process(
      data: list[float],
      multiplier: Hyper[float] = 2.0,
    ) -> float:
      return sum(data) * multiplier

    config = process.Config(multiplier=3.0)
    fn = config.make()

    # make() returns a callable
    assert callable(fn)

    # Call with data param
    result = fn(data=[1.0, 2.0, 3.0])
    assert result == 18.0  # (1+2+3) * 3

  def test_class_make_returns_instance(self) -> None:
    """For classes, make() returns the instance directly."""

    @configurable
    class MyClass:
      def __init__(self, value: Hyper[int] = 10) -> None:
        self.value = value

    config = MyClass.Config(value=42)
    instance = config.make()

    # make() returns instance, not callable
    assert isinstance(instance, MyClass)
    assert instance.value == 42

  def test_dataclass_make_returns_instance(self) -> None:
    """For dataclasses, make() returns the instance directly."""

    @configurable
    @dataclass
    class MyDataclass:
      name: Hyper[str] = "default"
      count: Hyper[int] = 0

    config = MyDataclass.Config(name="test", count=5)
    instance = config.make()

    assert isinstance(instance, MyDataclass)
    assert instance.name == "test"
    assert instance.count == 5

  def test_function_no_data_params(self) -> None:
    """Functions with only Hyper params still return callable (no args)."""

    @configurable
    def compute(
      a: Hyper[int] = 1,
      b: Hyper[int] = 2,
    ) -> int:
      return a + b

    config = compute.Config(a=10, b=20)
    fn = config.make()

    # Returns callable that takes no args
    assert callable(fn)
    result = fn()
    assert result == 30


class TestMakeStubGeneration:
  """Tests for make() stub generation."""

  def test_function_stub_has_callable_return(self) -> None:
    """Function config stubs should have make() -> Callable."""
    from nonfig.stubs import ConfigurableInfo, HyperParam
    from nonfig.stubs.generator import _generate_config_class

    info = ConfigurableInfo(
      name="process",
      is_class=False,
      params=[HyperParam("multiplier", "float", "2.0")],
      call_params=[("data", "list[float]", None)],
      return_type="float",
    )

    stub = _generate_config_class(info)

    # For functions, make() returns a Callable with the data param signature
    assert "def make(self) -> Callable[[list[float]], float]" in stub

  def test_class_stub_has_instance_return(self) -> None:
    """Class config stubs should have make() -> ClassName."""
    from nonfig.stubs import ConfigurableInfo, HyperParam
    from nonfig.stubs.generator import _generate_config_class

    info = ConfigurableInfo(
      name="MyClass",
      is_class=True,
      params=[HyperParam("value", "int", "10")],
      return_type="MyClass",
    )

    stub = _generate_config_class(info)

    # For classes, make() returns the class instance directly
    assert "def make(self) -> MyClass" in stub

  def test_multiple_data_params_in_callable(self) -> None:
    """Data params are included in the Callable signature."""
    from nonfig.stubs import ConfigurableInfo, HyperParam
    from nonfig.stubs.generator import _generate_config_class

    info = ConfigurableInfo(
      name="transform",
      is_class=False,
      params=[HyperParam("scale", "float", "1.0")],
      call_params=[
        ("x", "float", None),
        ("y", "float", None),
        ("z", "float", "0.0"),
      ],
      return_type="tuple[float, float, float]",
    )

    stub = _generate_config_class(info)

    # make() returns Callable with data param types
    assert (
      "def make(self) -> Callable[[float, float, float], tuple[float, float, float]]"
      in stub
    )
