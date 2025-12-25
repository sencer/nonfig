"""Parameter extraction utilities for nonfig."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
import dataclasses
import functools
import inspect
from typing import (
  Annotated,
  Any,
  get_args,
  get_origin,
  get_type_hints,
)

from annotated_types import BaseMetadata, Ge, Gt, Le, Lt, MaxLen, MinLen, MultipleOf
from pydantic import Field
from pydantic.fields import FieldInfo

from nonfig.constraints import PatternConstraint, validate_constraint_conflicts
from nonfig.models import MakeableModel
from nonfig.typedefs import DefaultSentinel, HyperMarker

__all__ = [
  "extract_class_params",
  "extract_function_hyper_params",
  "get_public_fields",
  "is_mapping_origin",
  "is_sequence_origin",
  "transform_type_for_nesting",
]

# Track config creation stack for circular dependency detection
_config_creation_stack: ContextVar[list[str] | None] = ContextVar(
  "_config_creation_stack", default=None
)


def has_hyper_marker(type_ann: Any) -> bool:
  """Check if a type annotation has the HyperMarker."""
  if get_origin(type_ann) is Annotated:
    args = get_args(type_ann)
    return any(isinstance(arg, type) and arg is HyperMarker for arg in args[1:])
  return False


def unwrap_hyper(type_ann: Any) -> tuple[Any, tuple[Any, ...]]:
  """
  Unwrap a Hyper annotation to get the inner type and constraints.

  Returns (inner_type, constraints_tuple)
  """
  if get_origin(type_ann) is Annotated:
    args = get_args(type_ann)
    inner_type = args[0]
    metadata = args[1:]
    # Filter out HyperMarker, keep constraints
    constraints = tuple(
      m for m in metadata if not (isinstance(m, type) and m is HyperMarker)
    )
    return inner_type, constraints
  return type_ann, ()


def extract_constraints(metadata: tuple[Any, ...]) -> dict[str, Any]:
  """Extract Pydantic field constraints from annotation metadata."""
  constraints: dict[str, Any] = {}

  for item in metadata:
    if isinstance(item, Ge):
      constraints["ge"] = item.ge
    elif isinstance(item, Gt):
      constraints["gt"] = item.gt
    elif isinstance(item, Le):
      constraints["le"] = item.le
    elif isinstance(item, Lt):
      constraints["lt"] = item.lt
    elif isinstance(item, MinLen):
      constraints["min_length"] = item.min_length
    elif isinstance(item, MaxLen):
      constraints["max_length"] = item.max_length
    elif isinstance(item, MultipleOf):
      constraints["multiple_of"] = item.multiple_of
    elif isinstance(item, PatternConstraint):
      constraints["pattern"] = item.pattern
    elif isinstance(item, BaseMetadata):
      # Handle other annotated_types constraints
      for attr in ("ge", "gt", "le", "lt", "min_length", "max_length", "multiple_of"):
        if hasattr(item, attr):
          val = getattr(item, attr)
          if val is not None:
            constraints[attr] = val

  return constraints


def is_sequence_origin(origin: Any) -> bool:
  """Check if origin is a sequence-like type (but not str/bytes).

  Args:
    origin: Result of get_origin(type_ann). None for non-generic types,
            a class for generic containers, or a special form for Union etc.
  """
  # Guard: None for non-generics (int, str), not a type for Union/Annotated/Literal
  if not isinstance(origin, type):
    return False
  # Exclude str/bytes - they're sequences but shouldn't be treated as containers
  if origin in (str, bytes):
    return False
  return issubclass(origin, Sequence)


def is_mapping_origin(origin: Any) -> bool:
  """Check if origin is a mapping-like type.

  Args:
    origin: Result of get_origin(type_ann). None for non-generic types,
            a class for generic containers, or a special form for Union etc.
  """
  if not isinstance(origin, type):
    return False
  return issubclass(origin, Mapping)


def transform_type_for_nesting(type_ann: Any) -> Any:
  """
  Transform a type to allow nested configs.

  If the type is a configurable class, transforms T -> T | T.Config
  so the field can accept either an instance or a config.
  """
  # Handle generic types recursively
  origin = get_origin(type_ann)

  if is_sequence_origin(origin):
    args = get_args(type_ann)
    if len(args) >= 1:
      first_arg = args[0]
      # For tuple with multiple args like tuple[A, B, C], transform each
      if origin is tuple and len(args) > 1 and args[-1] is not ...:
        transformed = tuple(transform_type_for_nesting(arg) for arg in args)
        return tuple[transformed]
      # For homogeneous sequences, transform the inner type but preserve container
      transformed_inner = transform_type_for_nesting(first_arg)
      if origin is list:
        return list[transformed_inner]
      if origin is tuple:
        return tuple[transformed_inner, ...]
      if origin is set:
        return set[transformed_inner]
      if origin is frozenset:
        return frozenset[transformed_inner]
      # For abstract types (Sequence, MutableSequence), use list
      return list[transformed_inner]
    return type_ann

  if is_mapping_origin(origin):
    args = get_args(type_ann)
    if len(args) == 2:  # noqa: PLR2004
      transformed_value = transform_type_for_nesting(args[1])
      if origin is dict:
        return dict[args[0], transformed_value]
      # For abstract types (Mapping, MutableMapping), use dict
      return dict[args[0], transformed_value]
    return type_ann

  if origin is Annotated:
    args = get_args(type_ann)
    inner = transform_type_for_nesting(args[0])
    return Annotated[inner, *args[1:]]

  # Check if it's a configurable class
  if isinstance(type_ann, type):
    config_cls = getattr(type_ann, "Config", None)
    if isinstance(config_cls, type) and issubclass(config_cls, MakeableModel):
      return type_ann | config_cls | dict[str, Any]

  return type_ann


def _is_configurable_callable(value: Any) -> bool:
  """Check if value is a configurable function/class (has .Config that is MakeableModel)."""
  if not callable(value):
    return False
  config_cls = getattr(value, "Config", None)
  return isinstance(config_cls, type) and issubclass(config_cls, MakeableModel)


def _get_config_class(field_type: Any) -> type[MakeableModel[Any]] | None:
  """Get the Config class for a field type, if available."""
  if isinstance(field_type, type):
    # First check if field_type itself is a MakeableModel (e.g., SomeClass.Config)
    if issubclass(field_type, MakeableModel):
      return field_type  # pyright: ignore[reportUnknownVariableType]
    # Then check if field_type has a Config attribute (e.g., configurable class)
    config_cls = getattr(field_type, "Config", None)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(config_cls, type) and issubclass(config_cls, MakeableModel):
      return config_cls  # pyright: ignore[reportUnknownVariableType]
  return None


def _instantiate_default_config(
  config_cls: type[MakeableModel[Any]],
  param_name: str,
) -> MakeableModel[Any]:
  """Instantiate a default config, checking for circular dependencies.

  Args:
    config_cls: The Config class to instantiate
    param_name: Name of the parameter (for error messages)

  Returns:
    An instance of the config class with default values

  Raises:
    ValueError: If circular dependency is detected
    TypeError: If config doesn't have a callable 'make' method or instantiation fails
  """
  config_name = f"{config_cls.__module__}.{config_cls.__qualname__}"

  # Check for circular dependency
  stack = _config_creation_stack.get() or []
  if config_name in stack:
    cycle_path = " -> ".join([*stack, config_name])
    raise ValueError(
      f"Circular dependency detected: {cycle_path}. Parameter '{param_name}' creates a cycle."
    )

  # Track this config creation
  new_stack = [*stack, config_name]
  token = _config_creation_stack.set(new_stack)

  try:
    instance = config_cls()
    # Verify the instance has a callable make method
    if not hasattr(instance, "make") or not callable(instance.make):
      raise TypeError(
        f"Nested config type '{config_cls.__name__}' for parameter '{param_name}' does not have a callable 'make' method"
      )
    return instance
  except (ValueError, TypeError):
    # Re-raise ValueError (circular dep) and TypeError (our own) as-is
    raise
  except Exception as e:
    raise TypeError(
      f"Failed to instantiate nested config type '{config_cls.__name__}' for parameter '{param_name}': {e}"
    ) from e
  finally:
    _config_creation_stack.reset(token)


def create_field_info(
  param_name: str,
  field_type: Any,
  default_value: Any,
  constraints: tuple[Any, ...],
  func_name: str | None = None,
) -> tuple[Any, FieldInfo]:
  """Create a Pydantic FieldInfo from extracted parameter information."""
  # Check for existing FieldInfo in constraints (e.g., Hyper[int, Field(description=...)])
  existing_field_info = next(
    (item for item in constraints if isinstance(item, FieldInfo)),
    None,
  )

  # Extract constraints from metadata
  constraint_kwargs = extract_constraints(constraints)

  # Validate for conflicts
  validate_constraint_conflicts(constraint_kwargs, param_name, func_name)

  # Transform type for nested configs
  transformed_type = transform_type_for_nesting(field_type)

  # If there's an existing FieldInfo, use it directly
  if existing_field_info is not None:
    return (transformed_type, existing_field_info)

  # Handle default value
  if default_value is inspect.Parameter.empty:
    # Required field
    return (transformed_type, Field(..., **constraint_kwargs))

  if isinstance(default_value, DefaultSentinel):
    # DEFAULT sentinel - instantiate the type's Config if available
    config_cls = _get_config_class(field_type)
    if config_cls is not None:
      default_config = _instantiate_default_config(config_cls, param_name)
      return (transformed_type, Field(default=default_config, **constraint_kwargs))

    # Handle container types with DEFAULT: use empty container
    origin = get_origin(field_type)

    if is_sequence_origin(origin):
      # If origin is a concrete class (like list, tuple, deque), use it as factory
      if isinstance(origin, type) and not inspect.isabstract(origin):
        return (transformed_type, Field(default_factory=origin, **constraint_kwargs))
      # Fallback for abstract types (Sequence, MutableSequence)
      return (transformed_type, Field(default_factory=list, **constraint_kwargs))

    if is_mapping_origin(origin):
      # If origin is a concrete class (like dict, OrderedDict), use it as factory
      if isinstance(origin, type) and not inspect.isabstract(origin):
        return (transformed_type, Field(default_factory=origin, **constraint_kwargs))
      # Fallback for abstract types (Mapping, MutableMapping)
      return (transformed_type, Field(default_factory=dict, **constraint_kwargs))

    # DEFAULT used with non-Config type - this is an error
    type_name = getattr(field_type, "__name__", str(field_type))
    raise TypeError(
      f"DEFAULT can only be used with nested Config types, but parameter '{param_name}' has type '{type_name}'. Use a concrete default value instead."
    )

  # Handle configurable callable as default (fn: inner.Type = inner)
  if _is_configurable_callable(default_value):
    config_cls = default_value.Config
    default_config = _instantiate_default_config(config_cls, param_name)
    return (transformed_type, Field(default=default_config, **constraint_kwargs))

  return (transformed_type, Field(default=default_value, **constraint_kwargs))


@functools.lru_cache(maxsize=256)
def get_type_hints_safe(obj: Any) -> dict[str, Any]:
  """Get type hints with fallback for forward references.

  Note: Cached for performance. The obj must be hashable (functions/classes are).
  """
  try:
    # Use obj's __globals__ if available (for functions) or get it from the object
    globalns: dict[str, Any] | None = None
    if hasattr(obj, "__globals__"):
      globalns = obj.__globals__
    elif hasattr(obj, "__module__"):
      import sys

      module = sys.modules.get(obj.__module__)
      if module is not None:
        globalns = vars(module)

    # Add Hyper to namespace if needed
    ns = globalns or {}
    if "Hyper" not in ns:
      from nonfig.typedefs import Hyper

      ns = {**ns, "Hyper": Hyper}
    return get_type_hints(obj, globalns=ns, include_extras=True)
  except Exception:  # noqa: BLE001
    # Fallback to annotations without resolution
    return getattr(obj, "__annotations__", {})


def extract_class_params(cls: type[Any]) -> dict[str, tuple[Any, FieldInfo]]:
  """
  Extract config parameters from a class or dataclass.

  For dataclasses: uses field definitions
  For regular classes: uses __init__ signature
  """
  if dataclasses.is_dataclass(cls):
    return _extract_dataclass_params(cls, cls.__name__)
  return _extract_init_params(cls, cls.__name__)


def _extract_dataclass_params(
  cls: type, class_name: str
) -> dict[str, tuple[Any, FieldInfo]]:
  """Extract parameters from a dataclass."""
  params: dict[str, tuple[Any, FieldInfo]] = {}
  hints = get_type_hints_safe(cls)

  for field in dataclasses.fields(cls):
    # Skip non-init fields
    if not field.init:
      continue

    field_type = hints.get(field.name, field.type)
    inner_type, constraints = unwrap_hyper(field_type)

    # Determine default value
    if field.default is not dataclasses.MISSING:
      default = field.default
    elif field.default_factory is not dataclasses.MISSING:
      # Use factory result as default
      default = field.default_factory()
    else:
      default = inspect.Parameter.empty

    params[field.name] = create_field_info(
      field.name, inner_type, default, constraints, class_name
    )

  return params


def _extract_init_params(
  cls: type, class_name: str
) -> dict[str, tuple[Any, FieldInfo]]:
  """Extract parameters from a class's __init__ method."""
  params: dict[str, tuple[Any, FieldInfo]] = {}

  init_method = cls.__init__
  sig = inspect.signature(init_method)
  hints = get_type_hints_safe(init_method)

  for name, param in sig.parameters.items():
    # Skip self
    if name == "self":
      continue

    # Skip *args and **kwargs
    if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
      continue

    field_type = hints.get(name, param.annotation)
    if field_type is inspect.Parameter.empty:
      field_type = Any

    inner_type, constraints = unwrap_hyper(field_type)
    default = param.default

    params[name] = create_field_info(name, inner_type, default, constraints, class_name)

  return params


def extract_function_hyper_params(
  func: Callable[..., Any],
  skip_first: bool = False,
) -> dict[str, tuple[Any, FieldInfo]]:
  """
  Extract Hyper-annotated parameters from a function.

  Only parameters annotated with Hyper[T] are extracted.
  Regular parameters are ignored (they become call-time args).

  Args:
      func: The function to extract from
      skip_first: If True, skip the first parameter (for methods)
  """
  params: dict[str, tuple[Any, FieldInfo]] = {}
  func_name = func.__name__

  sig = inspect.signature(func)
  hints = get_type_hints_safe(func)

  param_list = list(sig.parameters.items())
  if skip_first and param_list:
    param_list = param_list[1:]

  for name, param in param_list:
    # Skip *args and **kwargs
    if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
      continue

    field_type = hints.get(name, param.annotation)
    if field_type is inspect.Parameter.empty:
      continue

    # Only extract Hyper-annotated parameters OR implicit hyper params
    is_hyper = has_hyper_marker(field_type)

    # Check for implicit hyper: default is DEFAULT, MakeableModel, or configurable callable
    if not is_hyper and isinstance(param.default, (DefaultSentinel, MakeableModel)):
      is_hyper = True
    if not is_hyper and _is_configurable_callable(param.default):  # pyright: ignore[reportUnknownMemberType]
      is_hyper = True

    if not is_hyper:
      continue

    inner_type, constraints = unwrap_hyper(field_type)
    default = param.default  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    params[name] = create_field_info(name, inner_type, default, constraints, func_name)

  return params


def get_public_fields(model: MakeableModel[Any]) -> dict[str, Any]:
  """Extract non-private fields from a MakeableModel instance."""
  return {
    name: getattr(model, name)
    for name in type(model).model_fields
    if not name.startswith("_")
  }
