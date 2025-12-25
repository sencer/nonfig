"""Tests for behaviors that were fixed and should not regress.

These tests cover specific fixes made during code review to ensure
they are not accidentally removed in future refactoring.
"""

from __future__ import annotations

from pydantic import Field
import pytest

from nonfig import Ge, Hyper, Le, configurable


class TestFieldInfoPreservation:
  """Test that custom FieldInfo in Hyper annotations is preserved."""

  def test_field_description_preserved(self) -> None:
    """FieldInfo with description should be preserved in Config."""

    @configurable
    def func_with_description(
      x: Hyper[int, Field(description="My custom description")] = 1,
    ) -> int:
      return x

    field_info = func_with_description.Config.model_fields["x"]
    assert field_info.description == "My custom description"

  def test_field_title_preserved(self) -> None:
    """FieldInfo with title should be preserved in Config."""

    @configurable
    def func_with_title(
      x: Hyper[int, Field(title="My Title")] = 1,
    ) -> int:
      return x

    field_info = func_with_title.Config.model_fields["x"]
    assert field_info.title == "My Title"

  def test_field_with_constraints_and_description(self) -> None:
    """FieldInfo should work alongside constraints."""

    @configurable
    def func_mixed(
      x: Hyper[int, Ge[0], Le[100], Field(description="Value between 0 and 100")] = 50,
    ) -> int:
      return x

    # The FieldInfo takes precedence, so description should be set
    field_info = func_mixed.Config.model_fields["x"]
    assert field_info.description == "Value between 0 and 100"

  def test_class_field_description_preserved(self) -> None:
    """FieldInfo should also work for class parameters."""

    @configurable
    class MyClass:
      def __init__(
        self,
        value: Hyper[float, Field(description="A float value")] = 1.0,
      ) -> None:
        self.value = value

    field_info = MyClass.Config.model_fields["value"]
    assert field_info.description == "A float value"


class TestConfigNaming:
  """Test that Config classes are named in PascalCase."""

  def test_snake_case_function_gets_pascal_case_config(self) -> None:
    """snake_case function names should produce PascalCase Config names."""

    @configurable
    def my_snake_case_function(x: Hyper[int] = 1) -> int:
      return x

    assert my_snake_case_function.Config.__name__ == "MySnakeCaseFunctionConfig"

  def test_single_word_function(self) -> None:
    """Single word function names should also be PascalCase."""

    @configurable
    def process(x: Hyper[int] = 1) -> int:
      return x

    assert process.Config.__name__ == "ProcessConfig"

  def test_class_config_naming(self) -> None:
    """Class Config names should use the class name directly."""

    @configurable
    class MyProcessor:
      def __init__(self, x: Hyper[int] = 1) -> None:
        self.x = x

    assert MyProcessor.Config.__name__ == "MyProcessorConfig"


class TestErrorMessageContext:
  """Test that error messages include function/class name context."""

  def test_function_name_in_constraint_error(self) -> None:
    """Constraint errors should include the function name."""
    with pytest.raises(ValueError, match=r"in 'bad_function'"):

      @configurable
      def bad_function(x: Hyper[int, Ge[10], Le[5]] = 7) -> int:
        return x

  def test_class_name_in_constraint_error(self) -> None:
    """Constraint errors should include the class name."""
    with pytest.raises(ValueError, match=r"in 'BadClass'"):

      @configurable
      class BadClass:
        def __init__(self, x: Hyper[int, Ge[10], Le[5]] = 7) -> None:
          self.x = x

  def test_error_still_contains_conflicting_constraints(self) -> None:
    """Error messages should still contain 'Conflicting constraints'."""
    with pytest.raises(ValueError, match=r"Conflicting constraints"):

      @configurable
      def another_bad(x: Hyper[int, Ge[10], Le[5]] = 7) -> int:
        return x


class TestConfigurableFunctionAttributes:
  """Test that decorated functions have Config and Type attributes."""

  def test_function_has_config(self) -> None:
    """Decorated function should have .Config attribute."""

    @configurable
    def my_func(x: Hyper[int] = 1) -> int:
      return x

    assert hasattr(my_func, "Config")
    assert my_func.Config.__name__ == "MyFuncConfig"

  def test_function_has_type(self) -> None:
    """Decorated function should have .Type attribute."""

    @configurable
    def func_with_params(
      count: Hyper[int] = 1,
      rate: Hyper[float] = 0.5,
    ) -> float:
      return count * rate

    assert hasattr(func_with_params, "Type")
    assert hasattr(func_with_params.Type, "Config")
    assert func_with_params.Type.Config is func_with_params.Config

  def test_function_still_callable(self) -> None:
    """Decorated function should remain directly callable."""

    @configurable
    def simple(x: Hyper[int] = 1) -> int:
      return x * 2

    assert simple() == 2  # default x=1, returns 1*2=2

  def test_function_no_hyper_params(self) -> None:
    """Decorated function with no Hyper params should still work."""

    @configurable
    def no_hyper_params() -> int:
      return 42

    assert no_hyper_params() == 42
    assert hasattr(no_hyper_params, "Config")


class TestLruCacheBehavior:
  """Test that type hint resolution is cached."""

  def test_same_function_uses_cache(self) -> None:
    """Calling get_type_hints_safe multiple times should use cache."""
    from nonfig.extraction import get_type_hints_safe

    def sample_func(x: int, y: str) -> bool:
      return True

    # Call twice
    result1 = get_type_hints_safe(sample_func)
    result2 = get_type_hints_safe(sample_func)

    # Should return same cached result
    assert result1 is result2

  def test_cache_info_available(self) -> None:
    """The cache should have cache_info available (proving it's lru_cached)."""
    from nonfig.extraction import get_type_hints_safe

    # lru_cache adds cache_info method
    assert hasattr(get_type_hints_safe, "cache_info")
    info = get_type_hints_safe.cache_info()
    assert hasattr(info, "hits")
    assert hasattr(info, "misses")


class TestInstanceCaching:
  """Test that make() caches instances in the base class."""

  def test_function_make_returns_same_instance(self) -> None:
    """Calling make() twice should return the same BoundFunction."""

    @configurable
    def cached_func(x: Hyper[int] = 1) -> int:
      return x

    config = cached_func.Config(x=5)
    result1 = config.make()
    result2 = config.make()

    assert result1 is result2

  def test_class_make_returns_same_instance(self) -> None:
    """Calling make() twice should return the same class instance."""

    @configurable
    class CachedClass:
      def __init__(self, x: Hyper[int] = 1) -> None:
        self.x = x

    config = CachedClass.Config(x=5)
    result1 = config.make()
    result2 = config.make()

    assert result1 is result2

  def test_different_configs_have_different_instances(self) -> None:
    """Different config instances should produce different made instances."""

    @configurable
    def func(x: Hyper[int] = 1) -> int:
      return x

    config1 = func.Config(x=1)
    config2 = func.Config(x=2)

    assert config1.make() is not config2.make()


class TestConfigurableFunctionMetadata:
  """Test that ConfigurableFunction exposes function metadata correctly."""

  def test_name_attribute(self) -> None:
    """__name__ should return the original function name."""

    @configurable
    def my_function(x: Hyper[int] = 1) -> int:
      return x

    assert my_function.__name__ == "my_function"

  def test_qualname_attribute(self) -> None:
    """__qualname__ should return the original function qualname."""

    @configurable
    def my_function(x: Hyper[int] = 1) -> int:
      return x

    assert "my_function" in my_function.__qualname__

  def test_module_attribute(self) -> None:
    """__module__ should return the original function module."""

    @configurable
    def my_function(x: Hyper[int] = 1) -> int:
      return x

    assert my_function.__module__ == __name__

  def test_doc_attribute(self) -> None:
    """__doc__ should return the original function docstring."""

    @configurable
    def documented_function(x: Hyper[int] = 1) -> int:
      """This is my docstring."""
      return x

    assert documented_function.__doc__ == "This is my docstring."

  def test_annotations_attribute(self) -> None:
    """__annotations__ should return the original function annotations."""

    @configurable
    def annotated_function(x: Hyper[int] = 1) -> int:
      return x

    assert "x" in annotated_function.__annotations__
    assert "return" in annotated_function.__annotations__

  def test_decorated_is_same_function(self) -> None:
    """Decorated function should be the same object (monkey-patched)."""

    def original_func(x: int = 1) -> int:
      return x

    decorated = configurable(original_func)

    # With monkey-patching, decorated IS the original function
    assert decorated is original_func
    assert hasattr(decorated, "Config")


class TestBoundConfigurableMethod:
  """Test BoundConfigurableMethod behavior."""

  def test_method_has_config(self) -> None:
    """Bound method should have .Config attribute."""

    class MyClass:
      @configurable
      def my_method(self, x: Hyper[int] = 1) -> int:
        return x

    instance = MyClass()
    bound = instance.my_method

    assert hasattr(bound, "Config")
    assert bound.Config is MyClass.my_method.Config

  def test_method_has_type(self) -> None:
    """Bound method should have .Type attribute."""

    class MyClass:
      @configurable
      def my_method(self, x: Hyper[int] = 1) -> int:
        return x

    instance = MyClass()
    bound = instance.my_method

    assert hasattr(bound, "Type")
    assert hasattr(bound.Type, "Config")
    assert bound.Type.Config is bound.Config

  def test_bound_method_callable(self) -> None:
    """Bound method should be callable."""

    class MyClass:
      @configurable
      def my_method(self, x: Hyper[int] = 1) -> int:
        return x * 2

    instance = MyClass()
    result = instance.my_method()

    assert result == 2  # 1 * 2

  def test_unbound_method_has_config(self) -> None:
    """Accessing method on class should have .Config attribute."""

    class MyClass:
      @configurable
      def my_method(self, x: Hyper[int] = 1) -> int:
        return x

    # With monkey-patching, method has Config attribute
    assert hasattr(MyClass.my_method, "Config")
    assert callable(MyClass.my_method)
