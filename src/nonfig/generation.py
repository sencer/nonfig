"""The configurable decorator and related utilities."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import threading
from typing import (
  TYPE_CHECKING,
  Any,
  cast,
  overload,
)

from pydantic import create_model
from pydantic_core import PydanticUndefined

from nonfig.extraction import (
  extract_class_params,
  extract_function_hyper_params,
)
from nonfig.models import (
  BoundFunction,
  MakeableModel,
  calculate_class_make_fields,
  recursive_make,
)

if TYPE_CHECKING:
  from pydantic.fields import FieldInfo

  from nonfig.typedefs import Configurable, ConfigurableFunc

__all__ = ["configurable"]

# Lock for thread-safe decoration
_decoration_lock = threading.Lock()

# Reserved Pydantic field names that cannot be used as Config field names
_RESERVED_PYDANTIC_NAMES = {"model_config", "model_fields", "model_computed_fields"}


def _to_pascal_case(name: str) -> str:
  """Convert snake_case or other naming to PascalCase.

  Examples:
    my_func -> MyFunc
    myFunc -> Myfunc
    my_function_name -> MyFunctionName
  """
  return "".join(word.capitalize() for word in name.split("_"))


# Note on Overlapping Overloads:
# type[T] is a subtype of Callable[..., T] because classes are callable.
# This causes pyright to warn about overlapping overloads. However, pyright
# correctly resolves which overload to use: type[T] matches classes,
# Callable[P, T] matches functions. The ignore comment is safe because:
# 1. The more specific overload (type[T]) is listed first
# 2. Runtime correctly dispatches via isinstance(target, type)
# 3. Pyright uses the correct overload for both classes and functions


# Overload for classes: captures __init__ signature via ParamSpec
@overload
def configurable[T, **P](target: type[T]) -> Configurable[T, P, T]: ...  # pyright: ignore[reportOverlappingOverload]


# Overload for functions: returns a Phantom Class type to enable .Type access
@overload
def configurable[T, **P](target: Callable[P, T]) -> type[ConfigurableFunc[P, T]]: ...


# Fallback overload for edge cases
@overload
def configurable(target: Callable[..., Any]) -> Callable[..., Any]: ...


def configurable[T](
  target: type[T] | Callable[..., Any],
) -> type[T] | Callable[..., Any]:
  """
  Make a class or function configurable.

  For classes/dataclasses:
      - Creates a Config class as target.Config
      - Config.make() returns an instance of the class
      - The original class can still be instantiated directly

  For functions:
      - Separates Hyper-annotated args from regular args
      - Hyper args become Config fields
      - Regular args remain call-time arguments
      - Config.make() returns a callable with Hyper args bound

  Examples:
      @configurable
      @dataclass
      class Model:
          learning_rate: Hyper[float, Ge[0], Le[1]] = 0.01
          num_layers: Hyper[int, Ge[1]] = 3

      config = Model.Config(learning_rate=0.001)
      model = config.make()

      @configurable
      def train(
          data: Dataset,  # Call-time arg
          epochs: Hyper[int, Ge[1]] = 10,  # Config arg
      ) -> Metrics:
          ...

      config = train.Config(epochs=20)
      trainer = config.make()  # Returns callable
      result = trainer(my_dataset)  # Calls train(my_dataset, epochs=20)
  """
  if isinstance(target, type):
    return _configurable_class(target)  # pyright: ignore[reportUnknownVariableType]
  if callable(target):
    return _configurable_function(target)
  raise TypeError(f"configurable requires a class or function, got {type(target)}")


def _configurable_class[T](cls: type[T]) -> type[T]:
  """Apply configurable to a class or dataclass."""
  # Fast path: check if THIS class has its OWN Config (not inherited)
  if "Config" in cls.__dict__:
    config_attr = cls.__dict__["Config"]
    if isinstance(config_attr, type) and issubclass(config_attr, MakeableModel):
      return cls

  with _decoration_lock:
    # Double-check after acquiring lock - check cls.__dict__ to avoid inherited Config
    if "Config" in cls.__dict__:
      config_attr = cls.__dict__["Config"]
      if isinstance(config_attr, type) and issubclass(config_attr, MakeableModel):
        return cls

    # Extract parameters
    params = extract_class_params(cls)

    # Create the Config class
    config_cls = _create_class_config(cls, params)

    # Attach to the original class
    cls.Config = config_cls  # type: ignore[attr-defined]

    return cls


def _create_class_config[T](
  cls: type[T], params: dict[str, tuple[Any, FieldInfo]]
) -> type[MakeableModel[T]]:
  """Create a Config class for a target class."""
  # Check for reserved Pydantic field names
  for name in params:
    if name in _RESERVED_PYDANTIC_NAMES:
      raise ValueError(
        f"Parameter '{name}' is reserved by Pydantic and cannot be used as a Config field"
      )

  # Create the model dynamically - pyright can't infer the type here
  config_cls = cast(
    type[MakeableModel[T]],
    create_model(
      f"{cls.__name__}Config",
      __base__=MakeableModel[cls],  # type: ignore[valid-type]
      **params,  # type: ignore[arg-type]
    ),
  )

  # Propagate docstring
  if cls.__doc__:
    config_cls.__doc__ = f"Configuration for {cls.__name__}.\n\n{cls.__doc__}"

  # Pre-calculate which fields could be nested
  maybe_nested = calculate_class_make_fields(config_cls)
  is_leaf = not maybe_nested
  config_cls._maybe_nested_fields = maybe_nested
  config_cls._is_always_leaf = is_leaf

  # Add the _make_impl method
  config_cls._make_impl = _create_class_make_impl(
    cls, is_leaf=is_leaf, maybe_nested=maybe_nested
  )

  # Add fast_make static method for high-performance direct creation
  config_cls.fast_make = _create_class_fast_make(
    cls, is_leaf=is_leaf, maybe_nested=maybe_nested
  )

  return cast("type[MakeableModel[T]]", config_cls)


def _create_class_fast_make[T](
  cls: type[T],
  is_leaf: bool = False,
  maybe_nested: set[str] | None = None,
) -> Callable[..., T]:
  """Create a static fast_make method that bypasses Pydantic."""
  if is_leaf:

    def fast_make_leaf(**kwargs: Any) -> T:
      return cls(**kwargs)

    return fast_make_leaf

  nested_names = maybe_nested or set()

  def fast_make_nested(**kwargs: Any) -> T:
    # Recursively make only the fields that MIGHT be nested
    for name in nested_names:
      if name in kwargs:
        val = kwargs[name]
        made = recursive_make(val)
        if made is not val:
          kwargs[name] = made
    return cls(**kwargs)

  return fast_make_nested


def _create_class_make_impl[T](
  cls: type[T],
  is_leaf: bool = False,
  maybe_nested: set[str] | None = None,
) -> Callable[[MakeableModel[T]], T]:
  """Create the _make_impl() method for a class Config."""
  if is_leaf:

    def _make_impl_leaf(self: MakeableModel[T]) -> T:
      # Ultra-fast leaf path: direct instantiation from __dict__
      return cls(**self.__dict__)

    return _make_impl_leaf

  # Specialized nested implementation to avoid runtime loops
  nested_names = list(maybe_nested or set())

  def _make_impl_nested(self: MakeableModel[T]) -> T:
    data = self.__dict__
    private = cast("dict[str, Any]", self.__pydantic_private__)

    # If no actual nested values at runtime, use fast path
    if not private["_has_nested"]:
      return cls(**data)

    # Calculate updates only for nested fields
    updates: dict[str, Any] = {}
    for name in nested_names:
      value = data[name]
      made = recursive_make(value)
      if made is not value:
        updates[name] = made

    if not updates:
      return cls(**data)

    return cls(**(data | updates))

  return _make_impl_nested


def _configurable_function(func: Callable[..., Any]) -> Callable[..., Any]:
  """Apply configurable to a function."""
  # Check if it's a method (has self parameter)
  sig = inspect.signature(func)
  params = list(sig.parameters.keys())
  skip_first = bool(params and params[0] in ("self", "cls"))

  # Extract only Hyper parameters
  hyper_params = extract_function_hyper_params(func, skip_first=skip_first)

  # Create Config class
  config_cls = _create_function_config(func, hyper_params)

  # Create Type proxy
  type_proxy = _create_type_proxy(func, config_cls)

  # Attach Config and Type directly to the function
  func.Config = config_cls  # type: ignore[attr-defined]
  func.Type = type_proxy  # type: ignore[attr-defined]

  # Expose default hyperparameters as attributes on the function
  for name, (_, field_info) in hyper_params.items():
    # Only expose if we have a concrete default value
    if (
      field_info.default is not None
      and field_info.default is not ...
      and field_info.default is not PydanticUndefined
    ):
      # If default is a MakeableModel, we might want to expose it or its make() result?
      # The user request implies accessing attributes like 'window', which are primitive values usually.
      # For now, just expose the default value as is.
      setattr(func, name, field_info.default)

  # Return original function (monkey-patched)
  return func


def _create_function_config(
  func: Callable[..., Any],
  params: dict[str, tuple[Any, FieldInfo]],
) -> type[MakeableModel[Callable[..., Any]]]:
  """Create a Config class for a function."""
  # Check for reserved Pydantic field names
  for name in params:
    if name in _RESERVED_PYDANTIC_NAMES:
      raise ValueError(
        f"Parameter '{name}' is reserved by Pydantic and cannot be used as a Config field"
      )

  # The return type is a callable that takes non-Hyper args
  return_type = Callable[..., Any]

  # Create PascalCase config name (my_func -> MyFuncConfig)
  config_name = _to_pascal_case(func.__name__) + "Config"

  # Create the model dynamically - pyright can't infer the type here
  config_cls = cast(
    type[MakeableModel[Callable[..., Any]]],
    create_model(
      config_name,
      __base__=MakeableModel[return_type],  # type: ignore[valid-type]
      **params,  # type: ignore[arg-type]
    ),
  )

  # Propagate docstring
  if func.__doc__:
    config_cls.__doc__ = f"Configuration for {func.__name__}.\n\n{func.__doc__}"

  # Pre-calculate which fields could be nested
  maybe_nested = calculate_class_make_fields(config_cls)
  is_leaf = not maybe_nested
  config_cls._maybe_nested_fields = maybe_nested
  config_cls._is_always_leaf = is_leaf

  # Add the _make_impl method
  config_cls._make_impl = _create_function_make_impl(
    func, is_leaf=is_leaf, maybe_nested=maybe_nested
  )

  # Add fast_make static method
  config_cls.fast_make = _create_function_fast_make(
    func, is_leaf=is_leaf, maybe_nested=maybe_nested
  )

  return cast("type[MakeableModel[Callable[..., Any]]]", config_cls)


def _create_function_fast_make(
  func: Callable[..., Any],
  is_leaf: bool = False,
  maybe_nested: set[str] | None = None,
) -> Callable[..., BoundFunction[Any]]:
  """Create a static fast_make method for functions."""
  if is_leaf:

    def fast_make_leaf(**kwargs: Any) -> BoundFunction[Any]:
      return BoundFunction(func, kwargs)

    return fast_make_leaf

  nested_names = maybe_nested or set()

  def fast_make_nested(**kwargs: Any) -> BoundFunction[Any]:
    for name in nested_names:
      if name in kwargs:
        val = kwargs[name]
        made = recursive_make(val)
        if made is not val:
          kwargs[name] = made
    return BoundFunction(func, kwargs)

  return fast_make_nested


def _create_function_make_impl(
  func: Callable[..., Any],
  is_leaf: bool = False,
  maybe_nested: set[str] | None = None,
) -> Callable[[MakeableModel[Any]], BoundFunction[Any]]:
  """Create the _make_impl() method for a function Config."""
  if is_leaf:

    def _make_impl_leaf(self: MakeableModel[Any]) -> BoundFunction[Any]:
      return BoundFunction(func, self.__dict__)

    return _make_impl_leaf

  # Specialized nested implementation to avoid runtime loops
  nested_names = list(maybe_nested or set())

  def _make_impl_nested(self: MakeableModel[Any]) -> BoundFunction[Any]:
    data = self.__dict__
    private = cast("dict[str, Any]", self.__pydantic_private__)

    if not private["_has_nested"]:
      return BoundFunction(func, data)

    updates: dict[str, Any] = {}
    for name in nested_names:
      value = data[name]
      made = recursive_make(value)
      if made is not value:
        updates[name] = made

    if not updates:
      return BoundFunction(func, data)

    return BoundFunction(func, data | updates)

  return _make_impl_nested


def _create_type_proxy(
  func: Callable[..., Any],
  config_cls: type[MakeableModel[Any]],
) -> type:
  """Create a proxy type for type hinting nested function configs."""

  class _TypeProxy:
    Config = config_cls

  _TypeProxy.__name__ = f"{func.__name__}Type"
  _TypeProxy.__qualname__ = f"{func.__qualname__}.Type"
  return _TypeProxy