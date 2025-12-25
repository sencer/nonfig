"""Base model classes for nonfig."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import UnionType
from typing import Any, cast, get_args, get_origin, override

from pydantic import BaseModel, ConfigDict, PrivateAttr

__all__ = [
  "BoundFunction",
  "MakeableModel",
  "is_makeable_model",
]


from nonfig.typedefs import DefaultSentinel


class MakeableModel[R](BaseModel):
  """Base class for all Config models."""

  model_config = ConfigDict(
    arbitrary_types_allowed=True,
    validate_assignment=True,
    extra="ignore",
    frozen=True,
  )

  # Internal metadata for optimized make()
  # _make_fields: list of (field_name, is_nested)
  # _has_nested: bool indicating if any field is nested
  _make_fields: list[tuple[str, bool]] = PrivateAttr(default_factory=list)
  _has_nested: bool = PrivateAttr(default=False)

  def make(self) -> R:
    """Create an instance of the target from this config.

    Subclasses should override _make_impl().
    """
    # Optimized access to private metadata to bypass Pydantic overhead
    private = cast("dict[str, Any]", self.__pydantic_private__)
    assert private is not None  # noqa: S101

    # Initialize metadata if not already done (once per instance)
    if not private["_make_fields"]:
      config_cls = type(self)
      # If class is guaranteed to have no nested fields, we can bypass calculation
      if config_cls.__dict__.get("_is_always_leaf"):
        # Set a sentinel to avoid repeating this check
        private["_make_fields"] = [("_", False)]
        return self._make_impl()

      fields, has_nested = calculate_make_fields(self)
      private["_make_fields"] = fields
      private["_has_nested"] = has_nested

    return self._make_impl()

  @override
  def __str__(self) -> str:
    return self.__repr__()

  def _make_impl(self) -> R:
    """Override this method to implement instance creation."""
    raise NotImplementedError("Subclasses must implement _make_impl()")


def calculate_class_make_fields(
  config_cls: type[MakeableModel[Any]],
) -> set[str]:
  """Identify fields that COULD be nested based on type hints.

  This is called once per Config class at decoration time.
  """
  maybe_nested: set[str] = set()
  for name, field in config_cls.model_fields.items():
    if name.startswith("_"):
      continue

    # Check if the type hint suggests it could be a Config object
    if could_be_nested_type(field.annotation):
      maybe_nested.add(name)

  return maybe_nested


def calculate_make_fields(
  model: MakeableModel[Any],
) -> tuple[list[tuple[str, bool]], bool]:
  """Calculate which fields need recursive make() based on actual values.

  Uses class-level 'maybe_nested' hint to avoid checking every field.
  """
  fields: list[tuple[str, bool]] = []
  has_nested = False
  data = model.__dict__
  config_cls = type(model)

  # Get hint from class if available, otherwise check all fields
  # Accessing via __dict__ to avoid descriptor overhead if any
  maybe_nested = config_cls.__dict__.get("_maybe_nested_fields")

  if maybe_nested is not None:
    # Optimized path: only check fields that COULD be nested
    for name in config_cls.model_fields:
      if name.startswith("_"):
        continue

      if name in maybe_nested:
        is_nested = is_nested_type(data.get(name))
        if is_nested:
          has_nested = True
        fields.append((name, is_nested))
      else:
        fields.append((name, False))
  else:
    # Fallback: check all fields
    for name in config_cls.model_fields:
      if name.startswith("_"):
        continue
      is_nested = is_nested_type(data.get(name))
      if is_nested:
        has_nested = True
      fields.append((name, is_nested))

  return fields, has_nested


def could_be_nested_type(annotation: Any) -> bool:
  """Check if a type annotation could potentially contain a Config object."""
  if annotation is None:
    return True  # Any

  origin = get_origin(annotation)
  if origin is not None:
    # Check args for Unions/Sequences/etc.
    return any(could_be_nested_type(arg) for arg in get_args(annotation))

  # Simple types
  return annotation not in (int, float, str, bool, type(None))


def is_nested_type(value: Any) -> bool:
  """Check if a value might contain nested Config objects."""
  # Handle instances
  if isinstance(value, MakeableModel | DefaultSentinel):
    return True

  # Handle classes (for default values that might be the class itself)
  if isinstance(value, type):
    try:
      # Use cast to Any to avoid "type[Unknown]" issues in basedpyright
      v_type = cast("Any", value)
      if issubclass(v_type, MakeableModel):
        return True
    except TypeError:
      pass
    if hasattr(value, "Config"):
      return True

  # Handle sequences/mappings
  if isinstance(value, list | tuple | set | frozenset):
    return any(is_nested_type(item) for item in cast("Sequence[Any]", value))

  if isinstance(value, dict):
    return any(
      is_nested_type(v) for v in cast("dict[Any, Any]", value).values()
    )

  # Check if it's a configurable function (has .Config)
  return bool(callable(value) and hasattr(cast("Any", value), "Config"))


def recursive_make(value: Any) -> Any:
  """
  Recursively make nested config objects.

  If value is a MakeableModel, call make() on it.
  If value is a list/dict, recursively process elements.
  """
  # Fast path for common types
  if value is None or isinstance(value, int | float | str | bool):
    return value

  # Make MakeableModel instances
  if isinstance(value, MakeableModel):
    res_make: Any = cast("MakeableModel[Any]", value).make()
    return res_make

  # Recursively handle sequences (list, tuple, set, etc.)
  # Use unique names to avoid basedpyright redeclaration errors
  v_type = type(value)  # type: ignore[reportUnknownVariableType]
  if v_type is list:
    res_list: Any = [recursive_make(item) for item in cast("list[Any]", value)]
    return res_list

  if v_type is dict:
    res_dict: Any = {
      k: recursive_make(v) for k, v in cast("dict[Any, Any]", value).items()
    }
    return res_dict

  if v_type is tuple:
    res_tuple: Any = tuple(
      recursive_make(item) for item in cast("tuple[Any, ...]", value)
    )
    return res_tuple

  if v_type is set:
    res_set: Any = {recursive_make(item) for item in cast("set[Any]", value)}
    return res_set

  if v_type is frozenset:
    res_fset: Any = frozenset(
      recursive_make(item) for item in cast("frozenset[Any]", value)
    )
    return res_fset

  return value


class BoundFunction[R]:
  """
  A callable wrapper that exposes hyperparameters as attributes.

  When a @configurable function's Config.make() is called, it returns
  a BoundFunction instead of a plain function. This allows accessing
  the bound hyperparameters:

      @configurable
      def process(data: pd.Series, window: Hyper[int] = 10) -> pd.Series:
          return data.rolling(window).mean()

      config = process.Config(window=20)
      fn = config.make()
      result = fn(data)      # Call it like a function
      print(fn.window)       # Access the bound hyperparameter (20)
  """

  __slots__ = ("_doc", "_func", "_hyper_kwargs", "_name")

  def __init__(
    self,
    func: Callable[..., R],
    hyper_kwargs: dict[str, Any],
  ) -> None:
    self._func = func
    self._hyper_kwargs = hyper_kwargs
    self._name = func.__name__
    self._doc = func.__doc__

  @property
  def __name__(self) -> str:
    return self._name

  @property
  @override
  def __doc__(self) -> str | None:  # pyright: ignore[reportIncompatibleVariableOverride]
    return self._doc

  @property
  def __wrapped__(self) -> Callable[..., R]:
    return self._func

  def __call__(self, *args: Any, **kwargs: Any) -> R:
    """Call the function with bound hyperparameters."""
    return self._func(*args, **self._hyper_kwargs, **kwargs)

  def __getattr__(self, name: str) -> Any:
    """Access bound hyperparameters as attributes."""
    if name.startswith("_"):
      raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    try:
      return self._hyper_kwargs[name]
    except KeyError:
      raise AttributeError(
        f"'{type(self).__name__}' has no attribute '{name}'"
      ) from None

  @override
  def __setattr__(self, name: str, value: Any) -> None:
    """Prevent modification of bound hyperparameters."""
    # Allow setting slots (internal use only)
    if name in self.__slots__:
      object.__setattr__(self, name, value)
      return

    # Check if it's a known hyperparameter
    if hasattr(self, "_hyper_kwargs") and name in self._hyper_kwargs:
      raise AttributeError(f"Hyperparameter '{name}' is read-only")

    # Otherwise it's an unknown attribute
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

  @override
  def __repr__(self) -> str:
    params = ", ".join(f"{k}={v!r}" for k, v in self._hyper_kwargs.items())
    return f"{self._name}({params})"


def is_makeable_model(type_ann: Any) -> bool:
  """Check if a type annotation represents a MakeableModel."""
  if type_ann is None:
    return False

  # Direct class check
  if isinstance(type_ann, type) and issubclass(type_ann, MakeableModel):
    return True

  # Check Union types (T | T.Config)
  origin = get_origin(type_ann)
  if origin is UnionType:
    args = get_args(type_ann)
    return any(is_makeable_model(arg) for arg in args)

  return False
