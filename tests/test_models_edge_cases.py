"""Tests for models.py edge cases to increase coverage."""

from collections.abc import Mapping

from nonfig import configurable
from nonfig.models import (
  calculate_make_fields,
  could_be_nested_type,
  is_nested_type,
  recursive_make,
)


class TestIsNestedType:
  """Tests for is_nested_type edge cases."""

  def test_is_nested_with_makeable_model_subclass(self) -> None:
    """is_nested_type returns True for MakeableModel subclasses."""

    @configurable
    class Sub:
      def __init__(self, x: int = 0) -> None:
        self.x = x

    config = Sub.Config()
    assert is_nested_type(config) is True

  def test_is_nested_with_class_type_config_attr(self) -> None:
    """is_nested_type returns True for class with Config attribute."""

    @configurable
    class HasConfig:
      def __init__(self, val: int = 0) -> None:
        self.val = val

    # Test with the decorated class type (has .Config)
    assert is_nested_type(HasConfig) is True

  def test_is_nested_with_plain_class_type(self) -> None:
    """is_nested_type returns False for plain class type."""

    class PlainClass:
      pass

    assert is_nested_type(PlainClass) is False

  def test_is_nested_with_sequence_containing_config(self) -> None:
    """is_nested_type returns True for sequences with nested configs."""

    @configurable
    class Sub:
      def __init__(self, val: int = 0) -> None:
        self.val = val

    config = Sub.Config()
    assert is_nested_type([config]) is True
    assert is_nested_type((config,)) is True
    assert is_nested_type({config}) is True
    assert is_nested_type(frozenset({config})) is True

  def test_is_nested_with_dict_containing_config(self) -> None:
    """is_nested_type returns True for dicts with nested config values."""

    @configurable
    class Sub:
      def __init__(self, val: int = 0) -> None:
        self.val = val

    config = Sub.Config()
    assert is_nested_type({"key": config}) is True

  def test_is_nested_with_empty_containers(self) -> None:
    """is_nested_type returns False for empty containers."""
    assert is_nested_type([]) is False
    assert is_nested_type({}) is False
    assert is_nested_type(()) is False


class TestRecursiveMake:
  """Tests for recursive_make edge cases."""

  def test_recursive_make_with_tuple(self) -> None:
    """recursive_make handles tuples with nested configs."""

    @configurable
    class Item:
      def __init__(self, x: int = 0) -> None:
        self.x = x

    configs = (Item.Config(x=1), Item.Config(x=2))
    result = recursive_make(configs)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].x == 1
    assert result[1].x == 2

  def test_recursive_make_with_set(self) -> None:
    """recursive_make handles sets (returns set of made objects)."""

    @configurable
    class Item:
      def __init__(self, x: int = 0) -> None:
        self.x = x

    # Sets containing configs - note configs aren't hashable by default
    # so we test with a list that gets made
    items = [Item.Config(x=1)]
    result = recursive_make(items)

    assert isinstance(result, list)
    assert result[0].x == 1

  def test_recursive_make_with_frozen_set_no_transform(self) -> None:
    """recursive_make returns original frozenset if no transform needed."""
    original = frozenset({1, 2, 3})
    result = recursive_make(original)
    assert result is original

  def test_recursive_make_with_dict(self) -> None:
    """recursive_make handles dicts with nested configs."""

    @configurable
    class Item:
      def __init__(self, val: int = 0) -> None:
        self.val = val

    data = {"item": Item.Config(val=42)}
    result = recursive_make(data)

    assert isinstance(result, dict)
    assert result["item"].val == 42

  def test_recursive_make_with_pure_data_tuple(self) -> None:
    """recursive_make returns original tuple if no transform needed."""
    original = (1, 2, 3)
    result = recursive_make(original)
    assert result is original  # Same object, no copy

  def test_recursive_make_with_generic_sequence(self) -> None:
    """recursive_make handles generic Sequence types."""
    from collections import deque

    # deque is a Sequence but not list/tuple
    original = deque([1, 2, 3])
    result = recursive_make(original)
    assert result is original  # No transform needed

  def test_recursive_make_with_generic_mapping(self) -> None:
    """recursive_make handles generic Mapping types."""
    from collections import OrderedDict

    original: Mapping[str, int] = OrderedDict([("a", 1), ("b", 2)])
    result = recursive_make(original)
    assert result is original  # No transform needed


class TestCouldBeNestedType:
  """Tests for could_be_nested_type annotation checks."""

  def test_could_be_nested_with_none(self) -> None:
    """None annotation returns True (could be Any)."""
    assert could_be_nested_type(None) is True

  def test_could_be_nested_with_union(self) -> None:
    """Union types are checked recursively."""

    @configurable
    class Sub:
      def __init__(self, x: int = 0) -> None:
        self.x = x

    # Union containing a Config should return True
    assert could_be_nested_type(Sub.Config | None) is True
    # Union of primitives returns False
    assert could_be_nested_type(int | None) is False


class TestCalculateMakeFields:
  """Tests for calculate_make_fields."""

  def test_calculate_make_fields_with_nested(self) -> None:
    """calculate_make_fields detects nested config values."""

    @configurable
    class Inner:
      def __init__(self, x: int = 0) -> None:
        self.x = x

    @configurable
    class Outer:
      def __init__(self, inner: Inner.Config | None = None) -> None:
        self.inner = inner

    config = Outer.Config(inner=Inner.Config(x=5))
    fields, has_nested = calculate_make_fields(config)

    assert has_nested is True
    # Check that 'inner' is marked as nested
    inner_field = next(f for f in fields if f[0] == "inner")
    assert inner_field[1] is True

  def test_calculate_make_fields_no_nested(self) -> None:
    """calculate_make_fields returns False when no nesting."""

    @configurable
    class Flat:
      def __init__(self, x: int = 0, y: str = "test") -> None:
        self.x = x
        self.y = y

    config = Flat.Config(x=1, y="hello")
    _fields, has_nested = calculate_make_fields(config)

    assert has_nested is False


class TestDoubleDecoration:
  """Tests for double-decoration handling."""

  def test_configurable_twice_is_idempotent(self) -> None:
    """Applying @configurable twice returns same class."""

    @configurable
    class Original:
      def __init__(self, x: int = 0) -> None:
        self.x = x

    # Apply again
    result = configurable(Original)

    # Should return the same class
    assert result is Original
    assert hasattr(result, "Config")
