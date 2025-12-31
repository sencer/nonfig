"""Parameter extraction utilities for nonfig."""

# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar, Token
import dataclasses
import functools
import inspect
from typing import (
  Annotated,
  Any,
  cast,
  get_args,
  get_origin,
  get_type_hints,
)

from annotated_types import (
  BaseMetadata,
  Ge,
  Gt,
  Le,
  Lt,
  MaxLen,
  MinLen,
  MultipleOf,
)
from pydantic import Field
from pydantic.fields import FieldInfo

from nonfig.constraints import PatternConstraint, validate_constraint_conflicts
from nonfig.models import DefaultSentinel, HyperMarker, LeafMarker, MakeableModel

__all__ = [
  "extract_class_params",
  "extract_function_hyper_params",
  "get_public_fields",
  "is_mapping_origin",
  "is_sequence_origin",
  "is_set_origin",
  "transform_type_for_nesting",
]

# Track config creation stack for circular dependency detection
_config_creation_stack: ContextVar[list[str] | None] = ContextVar(
  "_config_creation_stack", default=None
)


class ConfigCreationContext:
  """Context manager for tracking config creation depth and detecting cycles."""

  __slots__ = ("config_name", "context_var", "param_name", "token")

  def __init__(
    self,
    config_name: str,
    context_var: ContextVar[list[str] | None],
    param_name: str,
  ) -> None:
    self.config_name = config_name
    self.context_var = context_var
    self.param_name = param_name
    self.token: Token[list[str] | None] | None = None

  def __enter__(self) -> None:
    stack = self.context_var.get() or []
    if self.config_name in stack:
      cycle_path = " -> ".join([*stack, self.config_name])
      raise ValueError(
        f"Circular dependency detected: {cycle_path}. Parameter '{self.param_name}' creates a cycle."
      )

    new_stack = [*stack, self.config_name]
    self.token = self.context_var.set(new_stack)

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    if self.token:
      self.context_var.reset(self.token)


def has_hyper_marker(type_ann: Any) -> bool:
  """Check if a type annotation has the HyperMarker."""
  if get_origin(type_ann) is Annotated:
    args = get_args(type_ann)
    return any(isinstance(arg, type) and arg is HyperMarker for arg in args[1:])
  return False


def unwrap_hyper(type_ann: Any) -> tuple[Any, tuple[Any, ...], bool]:
  """
  Unwrap a Hyper annotation to get the inner type and constraints.

  Returns:
    Tuple of (inner_type, constraints_tuple, is_leaf)
  """
  if get_origin(type_ann) is Annotated:
    args = get_args(type_ann)
    inner_type = args[0]
    metadata = args[1:]

    # Check for LeafMarker at this level
    is_leaf = any(
      isinstance(m, type) and issubclass(m, LeafMarker) for m in metadata
    ) or any(isinstance(m, LeafMarker) for m in metadata)

    # Filter out HyperMarker and LeafMarker from constraints
    constraints = tuple(
      m
      for m in metadata
      if not (isinstance(m, type) and issubclass(m, (HyperMarker, LeafMarker)))
      and not isinstance(m, LeafMarker)
    )

    # Recursively unwrap if inner is also Annotated
    if get_origin(inner_type) is Annotated:
      inner_inner, inner_constraints, inner_leaf = unwrap_hyper(inner_type)
      return cast(
        "tuple[Any, tuple[Any, ...], bool]",
        (inner_inner, constraints + inner_constraints, is_leaf or inner_leaf),
      )

    return cast("tuple[Any, tuple[Any, ...], bool]", (inner_type, constraints, is_leaf))
  return type_ann, (), False


def extract_constraints(
  metadata: tuple[Any, ...],
) -> tuple[dict[str, Any], tuple[Any, ...]]:
  """Extract Pydantic field constraints from annotation metadata."""
  constraints: dict[str, Any] = {}
  leftovers: list[Any] = []

  for item in metadata:
    it_type = type(item)
    is_constraint = False

    # Fast path for common annotated_types (Ge, Le, etc.)
    if it_type is Ge:
      constraints["ge"] = item.ge
      is_constraint = True
    elif it_type is Gt:
      constraints["gt"] = item.gt
      is_constraint = True
    elif it_type is Le:
      constraints["le"] = item.le
      is_constraint = True
    elif it_type is Lt:
      constraints["lt"] = item.lt
      is_constraint = True
    elif it_type is MinLen:
      constraints["min_length"] = item.min_length
      is_constraint = True
    elif it_type is MaxLen:
      constraints["max_length"] = item.max_length
      is_constraint = True
    elif it_type is MultipleOf:
      constraints["multiple_of"] = item.multiple_of
      is_constraint = True
    elif it_type is PatternConstraint:
      constraints["pattern"] = item.pattern
      is_constraint = True
    elif isinstance(item, BaseMetadata):
      # Fallback for other annotated_types
      is_constraint = True
      mapping = {
        "ge": "ge",
        "gt": "gt",
        "le": "le",
        "lt": "lt",
        "min_length": "min_length",
        "max_length": "max_length",
        "multiple_of": "multiple_of",
      }
      found_any = False
      for attr, pydantic_name in mapping.items():
        if hasattr(item, attr):
          val = getattr(item, attr)
          if val is not None:
            constraints[pydantic_name] = val
            found_any = True

      if not found_any:
        # If it's BaseMetadata but we don't handle it, keep it in leftovers
        leftovers.append(item)

    if not is_constraint:
      leftovers.append(item)

  return constraints, tuple(leftovers)


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


def is_set_origin(origin: Any) -> bool:
  """Check if origin is a set-like type.

  Args:
    origin: Result of get_origin(type_ann).
  """
  if not isinstance(origin, type):
    return False
  return issubclass(origin, (set, frozenset))


def is_mapping_origin(origin: Any) -> bool:
  """Check if origin is a mapping-like type.

  Args:
    origin: Result of get_origin(type_ann). None for non-generic types,
            a class for generic containers, or a special form for Union etc.
  """
  if not isinstance(origin, type):
    return False
  return issubclass(origin, Mapping)


def is_leaf_annotation(type_ann: Any) -> bool:
  """Check if a type annotation has the LeafMarker."""
  if get_origin(type_ann) is Annotated:
    args = get_args(type_ann)
    # Check current level
    for arg in args[1:]:
      # Check if it's the LeafMarker class itself, a subclass (like Leaf), or an instance
      if (
        arg is LeafMarker
        or (isinstance(arg, type) and issubclass(arg, LeafMarker))
        or isinstance(arg, LeafMarker)
      ):
        return True
    # Check recursively
    return is_leaf_annotation(args[0])
  return False


def transform_type_for_nesting(type_ann: Any, is_leaf: bool = False) -> Any:
  """
  Transform a type to allow nested configs.

  If the type is a configurable class, transforms T -> T | T.Config
  so the field can accept either an instance or a config.

  Args:
    type_ann: The type annotation to transform.
    is_leaf: If True, skip transformation (used when Leaf[T] is detected).
  """
  # If explicitly marked as Leaf, return as is
  if is_leaf or is_leaf_annotation(type_ann):
    return type_ann

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
      # For abstract types (Sequence, MutableSequence), use list
      return list[transformed_inner]
    return type_ann

  if is_set_origin(origin):
    args = get_args(type_ann)
    if len(args) >= 1:
      transformed_inner = transform_type_for_nesting(args[0])
      if origin is frozenset:
        return frozenset[transformed_inner]
      # set, MutableSet, or abstract
      return set[transformed_inner]
    return type_ann

  if is_mapping_origin(origin):
    args = get_args(type_ann)
    if len(args) == 2:
      transformed_value = transform_type_for_nesting(args[1])
      if origin is dict:
        return dict[args[0], transformed_value]
      # For abstract types (Mapping, MutableMapping), use dict
      return dict[args[0], transformed_value]
    return type_ann

  if origin is Annotated:
    args = get_args(type_ann)
    # The Leaf check at the start handles Annotated[T, Leaf]
    inner = transform_type_for_nesting(args[0])
    return Annotated[inner, *args[1:]]

  # Handle Union types (both `X | Y` syntax and `Union[X, Y]`)
  # Import here to avoid circular imports at module level
  from types import UnionType
  from typing import Union

  if origin is Union or isinstance(type_ann, UnionType):
    args = get_args(type_ann)
    transformed_args = tuple(transform_type_for_nesting(arg) for arg in args)
    if transformed_args != args:
      # Reconstruct union with transformed types using | operator
      from functools import reduce

      return reduce(lambda a, b: a | b, transformed_args)
    return type_ann

  # Transform configurable classes to T | T.Config
  config_cls = _get_config_class(type_ann)
  if config_cls is not None and config_cls is not type_ann:
    return type_ann | config_cls

  return type_ann


@functools.lru_cache(maxsize=128)
def _get_config_class_for_type(cls: type) -> type[MakeableModel[Any]] | None:
  """Cached helper for types."""
  # Case 1: value is already a Config class (MakeableModel subclass)
  if issubclass(cls, MakeableModel):
    return cls

  # Case 3: value has a .Config attribute (configurable class or function)
  config_cls = getattr(cls, "Config", None)
  if isinstance(config_cls, type) and issubclass(config_cls, MakeableModel):
    return config_cls
  return None


def _get_config_class(value: Any) -> type[MakeableModel[Any]] | None:
  """Get the Config class for a field type or default value, if available."""
  # Optimization: early return for common primitive types
  if value is None:
    return None
  if isinstance(value, type) and value in {int, float, str, bool}:
    return None

  # Handle types (cached)
  if isinstance(value, type):
    return _get_config_class_for_type(value)

  # Case 2: value is a Config instance (MakeableModel instance)
  if isinstance(value, MakeableModel):
    return type(value)

  # Optimization: only check types and callables for the rest
  if not callable(value):
    return None

  # Case 3: value has a .Config attribute (configurable function)
  # For callables, we don't cache as they might be dynamically created/modified or unhashable closures
  config_cls = getattr(value, "Config", None)
  if isinstance(config_cls, type) and issubclass(config_cls, MakeableModel):
    return config_cls

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

  with ConfigCreationContext(config_name, _config_creation_stack, param_name):
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


def create_field_info(
  param_name: str,
  field_type: Any,
  default_value: Any,
  constraints: tuple[Any, ...],
  func_name: str | None = None,
  *,
  is_leaf: bool = False,
) -> tuple[Any, FieldInfo]:
  """Create a Pydantic FieldInfo from extracted parameter information.

  Args:
    param_name: Name of the parameter.
    field_type: Type of the parameter.
    default_value: Default value from signature.
    constraints: Extracted constraints (Ge, Le, etc).
    func_name: Optional name of the function for error reporting.
    is_leaf: If True, force the field to be a leaf (skip transformation).
  """
  # Check for existing FieldInfo in constraints (e.g., Hyper[int, Field(description=...)])
  existing_field_info = next(
    (item for item in constraints if isinstance(item, FieldInfo)),
    None,
  )

  # Extract constraints from metadata
  constraint_kwargs, leftovers = extract_constraints(constraints)

  # Validate for conflicts
  validate_constraint_conflicts(constraint_kwargs, param_name, func_name)

  # Transform type for nested configs
  transformed_type = transform_type_for_nesting(field_type, is_leaf=is_leaf)

  # If there's an existing FieldInfo, use it directly
  if existing_field_info is not None:
    # Remove existing FieldInfo from leftovers to avoid duplication
    leftovers = tuple(x for x in leftovers if x is not existing_field_info)
    if leftovers:
      transformed_type = Annotated[transformed_type, *leftovers]
    return (transformed_type, existing_field_info)

  # Re-apply non-constraint metadata (e.g. jaxtyping info)
  if leftovers:
    transformed_type = Annotated[transformed_type, *leftovers]

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

    if is_set_origin(origin):
      if origin is frozenset:
        return (transformed_type, Field(default_factory=frozenset, **constraint_kwargs))
      # set or abstract
      return (transformed_type, Field(default_factory=set, **constraint_kwargs))

    # DEFAULT used with non-Config type - this is an error
    type_name = getattr(field_type, "__name__", str(field_type))
    raise TypeError(
      f"DEFAULT can only be used with nested Config types, but parameter '{param_name}' has type '{type_name}'. Use a concrete default value instead."
    )

  # Handle configurable items or direct Config classes as defaults
  if isinstance(default_value, MakeableModel):
    return (transformed_type, Field(default=default_value, **constraint_kwargs))

  config_cls = _get_config_class(default_value)

  if config_cls is not None:
    default_config = _instantiate_default_config(config_cls, param_name)
    return (transformed_type, Field(default=default_config, **constraint_kwargs))

  return (transformed_type, Field(default=default_value, **constraint_kwargs))


@functools.lru_cache(maxsize=256)
def get_type_hints_safe(obj: Any) -> dict[str, Any]:
  """Get type hints with fallback for forward references.

  Note: Cached for performance. The obj must be hashable (functions/classes are).
  """
  try:
    # Unwrap decorators (like jaxtyping, beartype) to get the original function
    # This ensures we get the correct __globals__ for resolving types
    base_obj = inspect.unwrap(obj)

    # Use base_obj's __globals__ if available
    globalns: dict[str, Any] | None = None
    if hasattr(base_obj, "__globals__"):
      globalns = base_obj.__globals__
    elif hasattr(base_obj, "__module__"):
      import sys

      module = sys.modules.get(base_obj.__module__)
      if module is not None:
        globalns = vars(module)

    # Add Hyper and Leaf to namespace if needed
    ns = globalns or {}
    if "Hyper" not in ns or "Leaf" not in ns:
      from nonfig.typedefs import Hyper, Leaf

      ns = {**ns, "Hyper": Hyper, "Leaf": Leaf}

    # Use base_obj for get_type_hints to get original annotations
    return get_type_hints(base_obj, globalns=ns, include_extras=True)
  except Exception:
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
    inner_type, constraints, is_leaf = unwrap_hyper(field_type)

    # Determine default value
    if field.default is not dataclasses.MISSING:
      default = field.default
    elif field.default_factory is not dataclasses.MISSING:
      # Use factory result as default
      default = field.default_factory()
    else:
      default = inspect.Parameter.empty

    params[field.name] = create_field_info(
      field.name, inner_type, default, constraints, class_name, is_leaf=is_leaf
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

  has_kwargs = False

  for name, param in sig.parameters.items():
    # Skip self
    if name == "self":
      continue

    # Skip *args
    if param.kind == param.VAR_POSITIONAL:
      continue

    # Check for **kwargs
    if param.kind == param.VAR_KEYWORD:
      has_kwargs = True
      continue

    field_type = hints.get(name, param.annotation)
    if field_type is inspect.Parameter.empty:
      field_type = Any

    inner_type, constraints, is_leaf = unwrap_hyper(field_type)
    default = param.default

    params[name] = create_field_info(
      name, inner_type, default, constraints, class_name, is_leaf=is_leaf
    )

  # Smart Parameter Propagation: If **kwargs is present, inherit params from base Configs
  if has_kwargs:
    # Iterate MRO (skipping self)
    for base in cls.mro()[1:]:
      # Check if base has a valid Config
      config_cls = getattr(base, "Config", None)
      if (
        isinstance(config_cls, type)
        and issubclass(config_cls, MakeableModel)
        and hasattr(cast("Any", config_cls), "model_fields")
      ):
        # Merge fields from base config
        for field_name, field_info in config_cls.model_fields.items():
          # Skip if already defined in subclass (override takes precedence)
          if field_name in params:
            continue

          # Skip internal fields
          if field_name.startswith("_"):
            continue

          # Add inherited field
          # We use the annotation and FieldInfo directly from the base Config
          params[field_name] = (field_info.annotation, field_info)

  return params


def extract_function_hyper_params(
  func: Callable[..., Any],
  skip_first: bool = False,
  *,
  all_params: bool = False,
) -> dict[str, tuple[Any, FieldInfo]]:
  """
  Extract Hyper-annotated parameters from a function.

  If all_params is True, all parameters are extracted regardless of Hyper annotation.
  Otherwise, only parameters annotated with Hyper[T] (or implicit hyper) are extracted.
  Regular parameters are ignored (they become call-time args).

  Args:
      func: The function to extract from
      skip_first: If True, skip the first parameter (for methods)
      all_params: If True, extract all parameters.
  """
  params: dict[str, tuple[Any, FieldInfo]] = {}
  func_name = getattr(func, "__name__", type(func).__name__)

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
      if all_params:
        field_type = Any
      else:
        continue

    # Only extract Hyper-annotated parameters OR implicit hyper params
    is_hyper = all_params or has_hyper_marker(field_type)

    # Check for implicit hyper: default is DEFAULT, MakeableModel, or configurable item
    if not is_hyper and (
      isinstance(param.default, DefaultSentinel) or _get_config_class(param.default)
    ):
      is_hyper = True

    if not is_hyper:
      continue

    inner_type, constraints, is_leaf = unwrap_hyper(field_type)
    default = param.default

    params[name] = create_field_info(
      name, inner_type, default, constraints, func_name, is_leaf=is_leaf
    )

  return params


def get_public_fields(model: MakeableModel[Any]) -> dict[str, Any]:
  """Extract non-private fields from a MakeableModel instance."""
  return {
    name: getattr(model, name)
    for name in type(model).model_fields
    if not name.startswith("_")
  }
