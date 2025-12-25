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
from nonfig.models import BoundFunction, MakeableModel

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
    "Any",
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
  config_cls._make_impl = _create_class_make_impl(cls, is_leaf=is_leaf)

  return cast("type[MakeableModel[T]]", config_cls)


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
    if _could_be_nested_type(field.annotation):
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
        is_nested = _is_nested_type(data.get(name))
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
      is_nested = _is_nested_type(data.get(name))
      if is_nested:
        has_nested = True
      fields.append((name, is_nested))

  return fields, has_nested


def _could_be_nested_type(annotation: Any) -> bool:
  """Check if a type annotation could potentially contain a Config object."""
  if annotation is None:
    return True  # Any

  from typing import get_args, get_origin

  origin = get_origin(annotation)
  if origin is not None:
    # Check args for Unions/Sequences/etc.
    return any(_could_be_nested_type(arg) for arg in get_args(annotation))

  # Simple types
  return annotation not in (int, float, str, bool, type(None))


def _is_nested_type(value: Any) -> bool:
  """Check if a value might contain nested Config objects."""
  from nonfig.typedefs import DefaultSentinel

  # Handle instances
  if isinstance(value, MakeableModel | DefaultSentinel):
    return True

  # Handle classes (for default values that might be the class itself)
  if isinstance(value, type):
    try:
      if issubclass(value, MakeableModel):
        return True
    except TypeError:
      pass
    if hasattr(value, "Config"):  # type: ignore[reportUnknownArgumentType]
      return True

  # Handle sequences/mappings
  if isinstance(value, list | tuple | set | frozenset):
    return any(_is_nested_type(item) for item in value)  # type: ignore[reportUnknownVariableType]

  if isinstance(value, dict):
    return any(_is_nested_type(v) for v in value.values())  # type: ignore[reportUnknownVariableType]

  # Check if it's a configurable function (has .Config)
  return bool(
    callable(value) and hasattr(value, "Config")  # type: ignore[reportUnknownArgumentType]
  )


def _create_class_make_impl[T](cls: type[T], is_leaf: bool = False) -> Callable[[MakeableModel[T]], T]:
  """Create the _make_impl() method for a class Config."""
  if is_leaf:

    def _make_impl_leaf(self: MakeableModel[T]) -> T:
      # Ultra-fast leaf path: direct instantiation from __dict__
      return cls(**self.__dict__)

    return _make_impl_leaf

  def _make_impl(self: MakeableModel[T]) -> T:
    # Optimized instantiation:
    # 1. Use pre-calculated fields to avoid discovery loop
    # 2. Use __dict__ to bypass Pydantic's getattr overhead
    # 3. Bypass loop entirely if no nested fields
    data = self.__dict__
    private = cast("dict[str, Any]", self.__pydantic_private__)

    if not private["_has_nested"]:
      return cls(**data)

    made_kwargs: dict[str, Any] = {}
    for name, is_nested in cast("list[tuple[str, bool]]", private["_make_fields"]):
      value = data[name]
      if is_nested:
        made_kwargs[name] = _recursive_make(value)
      else:
        made_kwargs[name] = value

    # Create instance
    return cls(**made_kwargs)

  return _make_impl


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
    "Any",
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
  config_cls._make_impl = _create_function_make_impl(func, is_leaf=is_leaf)

  return cast("type[MakeableModel[Callable[..., Any]]]", config_cls)


def _create_function_make_impl(
  func: Callable[..., Any],
  is_leaf: bool = False,
) -> Callable[[MakeableModel[Any]], BoundFunction[Any]]:
  """Create the _make_impl() method for a function Config."""
  if is_leaf:

    def _make_impl_leaf(self: MakeableModel[Any]) -> BoundFunction[Any]:
      return BoundFunction(func, self.__dict__)

    return _make_impl_leaf

  def _make_impl(self: MakeableModel[Any]) -> BoundFunction[Any]:
    # Optimized instantiation:
    # 1. Use pre-calculated fields to avoid discovery loop
    # 2. Use __dict__ to bypass Pydantic's getattr overhead
    # 3. Bypass loop entirely if no nested fields
    data = self.__dict__
    private = cast("dict[str, Any]", self.__pydantic_private__)

    if not private["_has_nested"]:
      return BoundFunction(func, data)

    made_kwargs: dict[str, Any] = {}
    for name, is_nested in cast("list[tuple[str, bool]]", private["_make_fields"]):
      value = data[name]
      if is_nested:
        made_kwargs[name] = _recursive_make(value)
      else:
        made_kwargs[name] = value

    # Create BoundFunction that exposes hyper params as attributes
    return BoundFunction(func, made_kwargs)

  return _make_impl


def _recursive_make(value: Any) -> Any:
  """
  Recursively make nested config objects.

  If value is a MakeableModel, call make() on it.
  If value is a list/dict, recursively process elements.
  """
  # Make MakeableModel instances
  if isinstance(value, MakeableModel):
    return value.make()  # pyright: ignore[reportUnknownVariableType]

  # Recursively handle sequences (list, tuple, set, etc.)
  # Preserve the original container type
  if isinstance(value, list):
    return [_recursive_make(item) for item in value]  # pyright: ignore[reportUnknownVariableType]

  if isinstance(value, tuple):
    return tuple(_recursive_make(item) for item in value)  # pyright: ignore[reportUnknownVariableType]

  if isinstance(value, set):
    return {_recursive_make(item) for item in value}  # pyright: ignore[reportUnknownVariableType]

  if isinstance(value, frozenset):
    return frozenset(_recursive_make(item) for item in value)  # pyright: ignore[reportUnknownVariableType]

  # Recursively handle mappings (dict, etc.)
  if isinstance(value, dict):
    result_dict: dict[Any, Any] = {}
    for k, v in value.items():  # pyright: ignore[reportUnknownVariableType]
      result_dict[k] = _recursive_make(v)
    return result_dict

  return value


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
