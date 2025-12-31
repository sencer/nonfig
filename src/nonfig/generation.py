"The configurable decorator and related utilities."

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportArgumentType=false, reportFunctionMemberAccess=false

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

__all__ = ["configurable", "wrap_external"]

# Lock for thread-safe decoration
_decoration_lock = threading.Lock()

# Reserved Pydantic field names that cannot be used as Config field names
_RESERVED_PYDANTIC_NAMES = {"model_config", "model_fields", "model_computed_fields"}

# Names reserved by nonfig on the Config class
_NONFIG_RESERVED_NAMES = {
  "make",
  "_make_impl",
  "_is_always_leaf",
  "_maybe_nested_fields",
  "func",
  "args",
  "keywords",
  "__wrapped__",
}

_ALL_RESERVED_NAMES = _RESERVED_PYDANTIC_NAMES | _NONFIG_RESERVED_NAMES


def _to_pascal_case(name: str) -> str:
  """Convert snake_case or other naming to PascalCase.

  Examples:
    my_func -> MyFunc
    _private -> _Private
    __dunder -> __Dunder
    CallableObj -> CallableObj
  """
  # Preserve leading underscores
  stripped = name.lstrip("_")
  leading_underscores = "_" * (len(name) - len(stripped))

  # PascalCase the rest, preserving existing uppercase (e.g., URL -> URL, myFunc -> MyFunc)
  pascal = "".join(w[:1].upper() + w[1:] for w in stripped.split("_") if w)

  return leading_underscores + pascal


def _get_type_params(cls: type[Any]) -> tuple[Any, ...] | None:
  """Get type parameters from a class (PEP 695 or Generic[T] style).

  Args:
    cls: The class to check for type parameters.

  Returns:
    Tuple of type parameters if class is generic, None otherwise.
  """
  from typing import Generic, get_args, get_origin

  # PEP 695 style: class Foo[T]
  type_params = getattr(cls, "__type_params__", None)
  if type_params:
    return type_params

  # Generic[T] style: class Foo(Generic[T])
  for base in getattr(cls, "__orig_bases__", ()):
    if get_origin(base) is Generic:
      args = get_args(base)
      if args:
        return args

  return None


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
    ```python
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
    ```
  """
  if isinstance(target, type):
    return _configurable_class(target)
  if callable(target):
    return _configurable_function(target)
  raise TypeError(
    f"@configurable requires a class or function, got {type(target).__name__}: {target!r}"
  )


@overload
def wrap_external[T](target: type[T], *, overrides: dict[str, Any] | None = None) -> type[MakeableModel[T]]: ...  # pyright: ignore[reportOverlappingOverload] # fmt: skip


@overload
def wrap_external[T, **P](
  target: Callable[P, T], *, overrides: dict[str, Any] | None = None
) -> type[MakeableModel[Callable[P, T]]]: ...


def wrap_external(
  target: type[Any] | Callable[..., Any],
  *,
  overrides: dict[str, Any] | None = None,
) -> type[MakeableModel[Any]]:
  """
  Wrap an external class or function to make it configurable without modification.

  Returns a Config class (MakeableModel) that can be used in other configs.
  Calling .make() on the returned config will instantiate the original target.

  Args:
    target: The class or function to wrap.
    overrides: Optional dictionary mapping parameter names to types/configs
               to override the extracted ones.

  Example:
    ```python
    from torch.optim import Adam
    AdamConfig = wrap_external(Adam)

    @configurable
    @dataclass
    class TrainConfig:
      optimizer: AdamConfig = DEFAULT

    config = TrainConfig(optimizer=AdamConfig(lr=0.01))
    trainer = config.make()
    ```
  """
  target_name = getattr(target, "__name__", type(target).__name__)
  if isinstance(target, type):
    # Check for positional-only arguments which we cannot support as keyword-only fields
    sig = inspect.signature(target)
    if any(
      p.kind == inspect.Parameter.POSITIONAL_ONLY for p in sig.parameters.values()
    ):
      raise ValueError(
        f"Cannot wrap '{target_name}' because it contains positional-only parameters. "
        + "Please provide a wrapper function that accepts these parameters as regular arguments "
        + "and passes them positionally to the target."
      )

    params = extract_class_params(target)
    if overrides:
      _apply_wrap_overrides(target, params, overrides)
    return cast("type[MakeableModel[Any]]", _create_class_config(target, params))

  if callable(target):
    # Check for positional-only arguments which we cannot support as keyword-only fields
    sig = inspect.signature(target)
    if any(
      p.kind == inspect.Parameter.POSITIONAL_ONLY for p in sig.parameters.values()
    ):
      raise ValueError(
        f"Cannot wrap '{target_name}' because it contains positional-only parameters. "
        + "Please provide a wrapper function that accepts these parameters as regular arguments "
        + "and passes them positionally to the target."
      )

    params = extract_function_hyper_params(target, all_params=True)
    if overrides:
      _apply_wrap_overrides(target, params, overrides)
    return cast("type[MakeableModel[Any]]", _create_function_config(target, params))

  raise TypeError(
    f"wrap_external requires a class or function, got {type(target).__name__}: {target!r}"
  )


def _apply_wrap_overrides(
  target: Any,
  params: dict[str, tuple[Any, FieldInfo]],
  overrides: dict[str, Any],
) -> None:
  """Apply parameter overrides to extracted parameters."""
  from nonfig.extraction import create_field_info, unwrap_hyper

  target_name = getattr(target, "__name__", type(target).__name__)
  sig = inspect.signature(target)
  has_kwargs = any(
    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
  )

  for name, new_type in overrides.items():
    if name in params:
      # Re-create field info with new type
      # We preserve the existing default if possible
      _, field_info = params[name]
      default = field_info.default
      # Pydantic uses PydanticUndefined for required fields, but create_field_info expects inspect.Parameter.empty
      if default is PydanticUndefined:
        default = inspect.Parameter.empty

      try:
        inner_type, constraints, is_leaf = unwrap_hyper(new_type)
      except Exception as e:
        raise TypeError(
          f"Invalid type override for parameter '{name}': {new_type!r}. "
          + "Expected a type or Hyper[T] annotation."
        ) from e

      params[name] = create_field_info(
        name, inner_type, default, constraints, target_name, is_leaf=is_leaf
      )
    elif has_kwargs:
      # Add new field for **kwargs
      try:
        inner_type, constraints, is_leaf = unwrap_hyper(new_type)
      except Exception as e:
        raise TypeError(
          f"Invalid type override for parameter '{name}': {new_type!r}. "
          + "Expected a type or Hyper[T] annotation."
        ) from e

      params[name] = create_field_info(
        name,
        inner_type,
        inspect.Parameter.empty,
        constraints,
        target_name,
        is_leaf=is_leaf,
      )
    else:
      raise ValueError(
        f"Cannot override parameter '{name}' for '{target_name}' because it is not in the signature and the target does not accept **kwargs."
      )


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

    # Create the Config class (includes validation hook if present)
    config_cls = _create_class_config(cls, params)

    # Attach to the original class
    cls.Config = config_cls

    # Inject __init_subclass__ for recursive configurability
    # This ensures subclasses are also marked as configurable
    _inject_init_subclass(cls)

    return cls


def _inject_init_subclass(cls: type[Any]) -> None:
  """Inject a custom __init_subclass__ to recursively configure subclasses."""
  # Check if __init_subclass__ is already defined on THIS class
  original = cls.__dict__.get("__init_subclass__")

  def auto_config_init_subclass(sub_cls: type[Any], **kwargs: Any) -> None:
    # 1. Call the original __init_subclass__ (maintain MRO chain)
    if original:
      # If original was a classmethod (which it effectively is), unwrap it
      if isinstance(original, classmethod):
        cast("Any", original).__func__(sub_cls, **kwargs)
      else:
        # Static method or raw function
        original(sub_cls, **kwargs)
    else:
      # Call super().__init_subclass__ explicitly
      # We use super(cls, sub_cls) to skip 'cls' and go to the next in MRO
      # using cast(Any, ...) to avoid "Type of super_cls is unknown"
      super_cls = cast("Any", super(cls, sub_cls))
      if hasattr(super_cls, "__init_subclass__"):
        super_cls.__init_subclass__(**kwargs)

    # 2. Automatically make the subclass configurable
    # The configurable() function is idempotent (checks for Config existence)
    # We must call it on sub_cls, not cls
    configurable(sub_cls)

  # Attach as a classmethod
  # Explicitly use setattr because __init_subclass__ type is special and direct assignment fails type check
  setattr(cls, "__init_subclass__", classmethod(auto_config_init_subclass))


def _create_class_config[T](
  cls: type[T], params: dict[str, tuple[Any, FieldInfo]]
) -> type[MakeableModel[T]]:
  """Create a Config class for a target class."""
  # Check for reserved names
  for name in params:
    if name in _ALL_RESERVED_NAMES:
      source = "Pydantic" if name in _RESERVED_PYDANTIC_NAMES else "nonfig"
      raise ValueError(
        f"Parameter '{name}' is reserved by {source} and cannot be used as a Config field. Please rename this parameter in your __init__ or dataclass."
      )

  # Check for __config_validate__ hook and create validator if present
  validators: dict[str, Any] = {}
  validate_hook = getattr(cls, "__config_validate__", None)
  if validate_hook is not None:
    from pydantic import model_validator

    @model_validator(mode="after")
    def _config_validate(self: MakeableModel[T]) -> MakeableModel[T]:
      return validate_hook(self)

    validators["_config_validate"] = _config_validate

  # Create the model dynamically - pyright can't infer the type here
  config_cls = cast(
    "type[MakeableModel[T]]",
    create_model(
      f"{cls.__name__}Config",
      __base__=MakeableModel[cls],
      __validators__=validators,
      **params,
    ),
  )

  # Propagate docstring
  if cls.__doc__:
    config_cls.__doc__ = f"Configuration for {cls.__name__}.\n\n{cls.__doc__}"

  # Propagate type parameters for generic classes (PEP 695 or Generic[T] style)
  type_params = _get_type_params(cls)
  if type_params:
    # Set __type_params__ to make Config subscribable like the original class
    config_cls.__type_params__ = type_params

  # Pre-calculate which fields could be nested
  maybe_nested = calculate_class_make_fields(config_cls)
  is_leaf = not maybe_nested
  config_cls._maybe_nested_fields = maybe_nested
  config_cls._is_always_leaf = is_leaf

  # Add the make method directly
  config_cls.make = _create_class_make_method(
    cls, is_leaf=is_leaf, maybe_nested=maybe_nested
  )

  return config_cls


def _create_class_make_method[T](
  cls: type[T],
  is_leaf: bool = False,
  maybe_nested: set[str] | None = None,
) -> Callable[[MakeableModel[T]], T]:
  """Create the make() method for a class Config."""
  if is_leaf:

    def make_leaf(self: MakeableModel[T]) -> T:
      # Ultra-fast leaf path: direct instantiation from __dict__
      return cls(**self.__dict__)

    return make_leaf

  # Specialized nested implementation to avoid runtime loops
  nested_names = list(maybe_nested or set())

  def make_nested(self: MakeableModel[T]) -> T:
    data = self.__dict__
    # Optimize access to private storage
    private = cast("dict[str, Any]", self.__pydantic_private__)

    # Calculate _has_nested lazily if not present (using _make_fields as flag)
    if not private.get("_make_fields"):
      has_nested = False
      for name in nested_names:
        # Check if the value is actually nested
        # We access via data to avoid descriptors
        val = data.get(name)
        from nonfig.models import is_nested_type

        if is_nested_type(val):
          has_nested = True
          break
      private["_has_nested"] = has_nested
      # Mark as done using a sentinel
      private["_make_fields"] = [("DONE", False)]
    else:
      has_nested = private["_has_nested"]

    # Fast path if no nested objects found
    if not has_nested:
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

  return make_nested


def _configurable_function(func: Callable[..., Any]) -> Callable[..., Any]:
  """Apply configurable to a function."""
  # Check if it's a method (has self parameter)
  sig = inspect.signature(func)
  params = list(sig.parameters.keys())
  skip_first = bool(params and params[0] in ("self", "cls"))

  # Extract only Hyper parameters
  hyper_params = extract_function_hyper_params(func, skip_first=skip_first)

  # Create Config class (includes validation hook if present)
  config_cls = _create_function_config(func, hyper_params)

  # Create Type proxy
  type_proxy = _create_type_proxy(func, config_cls)

  # Attach Config and Type directly to the function
  func.Config = config_cls
  func.Type = type_proxy

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
  func_name = getattr(func, "__name__", type(func).__name__)

  # Check for reserved names
  for name in params:
    if name in _ALL_RESERVED_NAMES:
      source = "Pydantic" if name in _RESERVED_PYDANTIC_NAMES else "nonfig"
      raise ValueError(
        f"Parameter '{name}' is reserved by {source} and cannot be used as a Config field. Please rename this parameter in function '{func_name}'."
      )

  # The return type is a callable that takes non-Hyper args
  return_type = Callable[..., Any]

  # Create PascalCase config name (my_func -> MyFuncConfig)
  config_name = _to_pascal_case(func_name) + "Config"

  # Create the model dynamically - pyright can't infer the type here
  config_cls = cast(
    "type[MakeableModel[Callable[..., Any]]]",
    create_model(
      config_name,
      __base__=MakeableModel[return_type],
      **params,
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

  # Add the make method directly
  config_cls.make = _create_function_make_method(
    func, is_leaf=is_leaf, maybe_nested=maybe_nested
  )

  return config_cls


def _create_function_make_method(
  func: Callable[..., Any],
  is_leaf: bool = False,
  maybe_nested: set[str] | None = None,
) -> Callable[[MakeableModel[Any]], BoundFunction[Any, Any]]:
  """Create the make() method for a function Config."""
  # func_name and func_doc are no longer needed here as they are retrieved dynamically by BoundFunction property proxies

  if is_leaf:

    def make_leaf(self: MakeableModel[Any]) -> BoundFunction[Any, Any]:
      return BoundFunction(func, **self.__dict__)

    return make_leaf

  # Specialized nested implementation to avoid runtime loops
  nested_names = list(maybe_nested or set())

  def make_nested(self: MakeableModel[Any]) -> BoundFunction[Any, Any]:
    data = self.__dict__
    private = cast("dict[str, Any]", self.__pydantic_private__)

    # Calculate _has_nested lazily if not present (using _make_fields as flag)
    if not private.get("_make_fields"):
      has_nested = False
      for name in nested_names:
        val = data.get(name)
        from nonfig.models import is_nested_type

        if is_nested_type(val):
          has_nested = True
          break
      private["_has_nested"] = has_nested
      # Mark as done using a sentinel
      private["_make_fields"] = [("DONE", False)]
    else:
      has_nested = private["_has_nested"]

    if not has_nested:
      return BoundFunction(func, **data)

    updates: dict[str, Any] = {}
    for name in nested_names:
      value = data[name]
      made = recursive_make(value)
      if made is not value:
        updates[name] = made

    if not updates:
      return BoundFunction(func, **data)

    return BoundFunction(func, **(data | updates))

  return make_nested


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
