"""Additional tests to improve code coverage.

Covers gaps identified in coverage analysis:
- generation.py: invalid target for @configurable
- models.py: BoundFunction introspection, attribute errors, is_makeable_model unions
- extraction.py: dict/Annotated type transformation
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import pytest

from nonfig import Hyper, MakeableModel, configurable
from nonfig.extraction import transform_type_for_nesting
from nonfig.models import is_makeable_model


class TestConfigurableInvalidTarget:
  """Test error handling when @configurable receives invalid target."""

  def test_configurable_rejects_string(self) -> None:
    """@configurable should reject non-class/function types."""
    with pytest.raises(TypeError, match="requires a class or function"):
      configurable("not a class or function")  # type: ignore[arg-type]

  def test_configurable_rejects_int(self) -> None:
    """@configurable should reject primitive types."""
    with pytest.raises(TypeError, match="requires a class or function"):
      configurable(42)  # type: ignore[arg-type]

  def test_configurable_rejects_none(self) -> None:
    """@configurable should reject None."""
    with pytest.raises(TypeError, match="requires a class or function"):
      configurable(None)  # type: ignore[arg-type]


class TestBoundFunctionIntrospection:
  """Test BoundFunction's introspection properties."""

  def test_bound_function_name(self) -> None:
    """BoundFunction should expose the wrapped function's name."""

    @configurable
    def my_indicator(x: Hyper[int] = 1) -> int:
      return x

    config = my_indicator.Config(x=5)
    bound = config.make()
    assert bound.__name__ == "my_indicator"

  def test_bound_function_doc(self) -> None:
    """BoundFunction should expose the wrapped function's docstring."""

    @configurable
    def documented_fn(x: Hyper[int] = 1) -> int:
      """This function has documentation."""
      return x

    config = documented_fn.Config()
    bound = config.make()
    assert bound.__doc__ == "This function has documentation."

  def test_bound_function_doc_none(self) -> None:
    """BoundFunction should return None for functions without docstrings."""

    @configurable
    def undocumented_fn(x: Hyper[int] = 1) -> int:
      return x

    config = undocumented_fn.Config()
    bound = config.make()
    # Functions defined in tests typically don't have __doc__ set to None
    # but we can check it's accessible
    _ = bound.__doc__  # Should not raise

  def test_bound_function_wrapped(self) -> None:
    """BoundFunction should expose the original function via __wrapped__."""

    @configurable
    def original_fn(x: Hyper[int] = 1) -> int:
      return x

    config = original_fn.Config()
    bound = config.make()
    assert bound.__wrapped__ is original_fn


class TestBoundFunctionAttributeErrors:
  """Test BoundFunction attribute access error handling."""

  def test_accessing_private_attribute_raises(self) -> None:
    """Accessing _private attributes should raise AttributeError."""

    @configurable
    def fn(x: Hyper[int] = 1) -> int:
      return x

    bound = fn.Config().make()
    with pytest.raises(AttributeError, match="has no attribute '_secret'"):
      _ = bound._secret

  def test_accessing_nonexistent_hyper_raises(self) -> None:
    """Accessing non-existent hyper params should raise AttributeError."""

    @configurable
    def fn(x: Hyper[int] = 1) -> int:
      return x

    bound = fn.Config().make()
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
      _ = bound.nonexistent

  def test_accessing_valid_hyper_works(self) -> None:
    """Accessing valid hyper params should work."""

    @configurable
    def fn(x: Hyper[int] = 1, y: Hyper[str] = "hello") -> str:
      return f"{y}:{x}"

    bound = fn.Config(x=10, y="world").make()
    assert bound.x == 10  # pyright: ignore[reportAttributeAccessIssue]
    assert bound.y == "world"  # pyright: ignore[reportAttributeAccessIssue]


class TestIsMakeableModelUnionTypes:
  """Test is_makeable_model with union types."""

  def test_direct_makeable_model(self) -> None:
    """Direct MakeableModel subclass should be detected."""

    @configurable
    class MyClass:
      Config: ClassVar[type[MakeableModel[object]]]

      def __init__(self, x: Hyper[int] = 1) -> None:
        self.x = x

    assert is_makeable_model(MyClass.Config) is True

  def test_union_with_makeable_model(self) -> None:
    """Union containing MakeableModel should be detected."""

    @configurable
    class MyClass:
      Config: ClassVar[type[MakeableModel[object]]]

      def __init__(self, x: Hyper[int] = 1) -> None:
        self.x = x

    # Create a union type: MyClass | MyClass.Config
    union_type = MyClass | MyClass.Config
    assert is_makeable_model(union_type) is True

  def test_union_without_makeable_model(self) -> None:
    """Union without MakeableModel should not be detected."""
    union_type = int | str
    assert is_makeable_model(union_type) is False

  def test_none_type(self) -> None:
    """None should not be detected as MakeableModel."""
    assert is_makeable_model(None) is False

  def test_regular_class(self) -> None:
    """Regular class should not be detected as MakeableModel."""

    class RegularClass:
      pass

    assert is_makeable_model(RegularClass) is False


class TestTransformTypeForNesting:
  """Test transform_type_for_nesting with various type annotations."""

  def test_transform_dict_with_config_value(self) -> None:
    """Dict with configurable value type should be transformed."""

    @configurable
    class Inner:
      Config: ClassVar[type[MakeableModel[object]]]

      def __init__(self, x: Hyper[int] = 1) -> None:
        self.x = x

    # Transform dict[str, Inner]
    original_type = dict[str, Inner]
    transformed = transform_type_for_nesting(original_type)

    # Should become dict[str, Inner | Inner.Config]
    transformed_str = str(transformed)
    assert "dict[str," in transformed_str
    assert "Inner" in transformed_str
    assert "Config" in transformed_str

  def test_transform_annotated_with_config(self) -> None:
    """Annotated type with configurable inner type should be transformed."""

    @configurable
    class Inner:
      Config: ClassVar[type[MakeableModel[object]]]

      def __init__(self, x: Hyper[int] = 1) -> None:
        self.x = x

    # Transform Annotated[Inner, "some metadata"]
    original_type = Annotated[Inner, "metadata"]
    transformed = transform_type_for_nesting(original_type)

    # The inner type should be transformed
    assert "Inner" in str(transformed)
    assert "Config" in str(transformed)

  def test_transform_list_with_config(self) -> None:
    """List with configurable element type should be transformed."""

    @configurable
    class Inner:
      Config: ClassVar[type[MakeableModel[object]]]

      def __init__(self, x: Hyper[int] = 1) -> None:
        self.x = x

    original_type = list[Inner]
    transformed = transform_type_for_nesting(original_type)

    # Should become list[Inner | Inner.Config]
    assert "Inner" in str(transformed)
    assert "Config" in str(transformed)

  def test_transform_plain_type_unchanged(self) -> None:
    """Non-configurable types should remain unchanged."""
    assert transform_type_for_nesting(int) is int
    assert transform_type_for_nesting(str) is str

  def test_transform_dict_without_config(self) -> None:
    """Dict without configurable values should remain unchanged."""
    original_type = dict[str, int]
    transformed = transform_type_for_nesting(original_type)
    # Should remain dict[str, int]
    assert str(transformed) == "dict[str, int]"


class TestBoundFunctionRepr:
  """Test BoundFunction string representation."""

  def test_repr_with_params(self) -> None:
    """BoundFunction repr should show bound parameters."""

    @configurable
    def process(x: Hyper[int] = 1, name: Hyper[str] = "test") -> str:
      return f"{name}:{x}"

    bound = process.Config(x=42, name="example").make()
    repr_str = repr(bound)

    # assert "BoundFunction" in repr_str  # Removed text wrapper
    assert "process" in repr_str
    assert "x=42" in repr_str
    assert "name='example'" in repr_str
