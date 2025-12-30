"""AST scanner for finding @configurable decorated items."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from nonfig.constraints import validate_constraint_conflicts

__all__ = ["ConfigurableInfo", "HyperParam", "extract_primitive_aliases", "scan_module"]


@dataclass
class HyperParam:
  """Information about a Hyper-annotated parameter."""

  name: str
  type_annotation: str
  default_value: str | None = None
  is_leaf: bool = False


@dataclass
class ConfigurableInfo:
  """Information about a @configurable decorated item."""

  name: str
  is_class: bool
  params: list[HyperParam] = field(default_factory=list)
  # For functions: parameters that are NOT Hyper (call-time args)
  call_params: list[tuple[str, str, str | None]] = field(default_factory=list)
  return_type: str = "Any"
  docstring: str | None = None
  is_wrapped: bool = False


def _is_configurable_decorator(node: ast.expr, aliases: set[str]) -> bool:
  """Check if a decorator is @configurable."""
  if isinstance(node, ast.Name):
    return node.id in aliases
  if isinstance(node, ast.Attribute):
    return node.attr in aliases
  return False


def _get_annotation_str(node: ast.expr | None) -> str:
  """Convert an AST annotation to a string."""
  if node is None:
    return "Any"
  return ast.unparse(node)


def _is_hyper_annotation(node: ast.expr) -> bool:
  """Check if an annotation is Hyper[...]."""
  if isinstance(node, ast.Subscript):
    # Check for Hyper[...]
    if isinstance(node.value, ast.Name) and node.value.id == "Hyper":
      return True
    if isinstance(node.value, ast.Attribute) and node.value.attr == "Hyper":
      return True

    # Check for Annotated[..., Hyper, ...]
    is_annotated = False
    if (isinstance(node.value, ast.Name) and node.value.id == "Annotated") or (
      isinstance(node.value, ast.Attribute) and node.value.attr == "Annotated"
    ):
      is_annotated = True

    if is_annotated and isinstance(node.slice, ast.Tuple):
      for elt in node.slice.elts[1:]:
        if isinstance(elt, ast.Name) and elt.id == "Hyper":
          return True
        if isinstance(elt, ast.Attribute) and elt.attr == "Hyper":
          return True

  return False


def _is_leaf_marker(node: ast.expr) -> bool:
  """Check if an annotation part is the Leaf marker or Leaf[T] subscript."""
  if isinstance(node, ast.Name) and node.id == "Leaf":
    return True
  if isinstance(node, ast.Attribute) and node.attr == "Leaf":
    return True
  # Handle Leaf[T]
  if isinstance(node, ast.Subscript):
    if isinstance(node.value, ast.Name) and node.value.id == "Leaf":
      return True
    if isinstance(node.value, ast.Attribute) and node.value.attr == "Leaf":
      return True
  return False


def _unwrap_hyper(node: ast.expr) -> tuple[str, bool]:
  """Extract the inner type from Hyper[T, ...] -> (T, is_leaf)."""
  if isinstance(node, ast.Subscript):
    slice_node = node.slice

    # Handle Annotated[T, ...]
    is_annotated = False
    if (isinstance(node.value, ast.Name) and node.value.id == "Annotated") or (
      isinstance(node.value, ast.Attribute) and node.value.attr == "Annotated"
    ):
      is_annotated = True

    # Handle Leaf[T]
    is_leaf_subscript = False
    if (isinstance(node.value, ast.Name) and node.value.id == "Leaf") or (
      isinstance(node.value, ast.Attribute) and node.value.attr == "Leaf"
    ):
      is_leaf_subscript = True

    # Handle Hyper[...]
    is_hyper = False
    if (isinstance(node.value, ast.Name) and node.value.id == "Hyper") or (
      isinstance(node.value, ast.Attribute) and node.value.attr == "Hyper"
    ):
      is_hyper = True

    if is_annotated:
      if isinstance(slice_node, ast.Tuple):
        inner_type_node = slice_node.elts[0]
        # Check markers in this Annotated
        is_leaf = any(_is_leaf_marker(elt) for elt in slice_node.elts[1:])
        # Recursively unwrap inner
        inner_type, inner_leaf = _unwrap_hyper(inner_type_node)
        return inner_type, is_leaf or inner_leaf
      # Single arg Annotated[T]
      return _unwrap_hyper(slice_node)

    if is_leaf_subscript:
      # Leaf[T] -> unwrap T and set leaf=True
      inner_type, _ = _unwrap_hyper(slice_node)
      return inner_type, True

    if is_hyper:
      if isinstance(slice_node, ast.Tuple):
        # Multiple args: Hyper[T, Ge[0], Le[100]]
        inner_type_node = slice_node.elts[0]
        is_leaf = any(_is_leaf_marker(elt) for elt in slice_node.elts[1:])
        inner_type, inner_leaf = _unwrap_hyper(inner_type_node)
        return inner_type, is_leaf or inner_leaf
      # Single arg: Hyper[T]
      return _unwrap_hyper(slice_node)

  return _get_annotation_str(node), False


def _extract_default(node: ast.expr | None) -> str | None:
  """Extract default value as string."""
  if node is None:
    return None
  return ast.unparse(node)


def _is_default_sentinel(node: ast.expr | None) -> bool:
  """Check if a default value is the DEFAULT sentinel."""
  if node is None:
    return False
  if isinstance(node, ast.Name):
    return node.id == "DEFAULT"
  if isinstance(node, ast.Attribute):
    return node.attr == "DEFAULT"
  return False


def _is_configurable_callable_default(
  default: ast.expr | None, ann: ast.expr | None
) -> bool:
  """Check if a default is a configurable callable (fn: inner.Type = inner).

  This heuristically detects the pattern where:
  - The annotation contains '.Type' (e.g., inner.Type)
  - The default is a simple name (the function itself)
  """
  if default is None or ann is None:
    return False
  # Check if default is a simple name (like 'inner')
  if not isinstance(default, ast.Name):
    return False
  # Check if annotation contains .Type
  ann_str = ast.unparse(ann)
  return ".Type" in ann_str


def _extract_constraints_from_hyper(node: ast.expr, param_name: str) -> None:
  """Extract and validate constraints from a Hyper annotation.

  Args:
    node: The Hyper[T, ...] subscript node
    param_name: Name of the parameter (for error messages)

  Raises:
    ValueError: If constraints are contradictory
  """
  if not isinstance(node, ast.Subscript):
    return

  slice_node = node.slice
  if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) < 2:
    return  # No constraints, just Hyper[T]

  constraints: dict[str, float | int] = {}

  # Process each constraint (skip first element which is the type)
  for elt in slice_node.elts[1:]:
    if isinstance(elt, ast.Subscript) and isinstance(elt.value, ast.Name):
      constraint_name = elt.value.id.lower()
      # Map Ge -> ge, Le -> le, etc.
      constraint_key = {
        "ge": "ge",
        "gt": "gt",
        "le": "le",
        "lt": "lt",
        "minlen": "min_length",
        "maxlen": "max_length",
        "multipleof": "multiple_of",
      }.get(constraint_name)

      if constraint_key and isinstance(elt.slice, ast.Constant):
        value = elt.slice.value
        if isinstance(value, (int, float)):
          constraints[constraint_key] = value

  if constraints:
    validate_constraint_conflicts(constraints, param_name)


def _is_implicit_hyper_default(
  default: ast.expr | None, ann: ast.expr | None, config_names: set[str]
) -> bool:
  """Check if a default value implies a hyperparameter."""
  if default is None:
    return False

  # Case 1: DEFAULT sentinel
  if _is_default_sentinel(default):
    return True

  # Case 2: Configurable callable pattern (fn: inner.Type = inner)
  if _is_configurable_callable_default(default, ann):
    return True

  # Case 3: Reference to a known configuration in this module
  # default = MyConfig
  if isinstance(default, ast.Name) and default.id in config_names:
    return True

  # default = MyConfig.Config
  return (
    isinstance(default, ast.Attribute)
    and isinstance(default.value, ast.Name)
    and default.value.id in config_names
    and default.attr == "Config"
  )


def _scan_function(
  node: ast.FunctionDef | ast.AsyncFunctionDef,
  configurable_aliases: set[str],
  config_names: set[str] | None = None,
) -> ConfigurableInfo | None:
  """Scan a function for @configurable decorator."""
  has_configurable = any(
    _is_configurable_decorator(d, configurable_aliases) for d in node.decorator_list
  )
  if not has_configurable:
    return None

  params: list[HyperParam] = []
  call_params: list[tuple[str, str, str | None]] = []

  # Skip self/cls
  args_to_check = node.args.args
  if args_to_check and args_to_check[0].arg in ("self", "cls"):
    args_to_check = args_to_check[1:]

  # Build defaults mapping (defaults align to the end of args)
  num_defaults = len(node.args.defaults)
  num_args = len(args_to_check)
  defaults_offset = num_args - num_defaults

  for i, arg in enumerate(args_to_check):
    ann = arg.annotation
    default_idx = i - defaults_offset
    default = node.args.defaults[default_idx] if default_idx >= 0 else None

    is_hyper = False
    if (ann and _is_hyper_annotation(ann)) or _is_implicit_hyper_default(
      default, ann, config_names or set()
    ):
      is_hyper = True

    if is_hyper:
      inner_type, is_leaf = (
        _unwrap_hyper(ann)
        if ann and _is_hyper_annotation(ann)
        else (_get_annotation_str(ann), False)
      )
      if ann and _is_hyper_annotation(ann):
        _extract_constraints_from_hyper(ann, arg.arg)

      params.append(
        HyperParam(
          name=arg.arg,
          type_annotation=inner_type,
          default_value=_extract_default(default),
          is_leaf=is_leaf,
        )
      )
    else:
      call_params.append((
        arg.arg,
        _get_annotation_str(ann),
        _extract_default(default),
      ))

  return_type = _get_annotation_str(node.returns)

  return ConfigurableInfo(
    name=node.name,
    is_class=False,
    params=params,
    call_params=call_params,
    return_type=return_type,
    docstring=ast.get_docstring(node),
  )


def _scan_class(
  node: ast.ClassDef,
  configurable_aliases: set[str],
) -> ConfigurableInfo | None:
  """Scan a class for @configurable decorator."""
  has_configurable = any(
    _is_configurable_decorator(d, configurable_aliases) for d in node.decorator_list
  )
  if not has_configurable:
    return None

  params: list[HyperParam] = []

  # Check for dataclass fields in class body
  for item in node.body:
    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
      # Skip 'Config' field - we generate our own Config class
      if item.target.id == "Config":
        continue

      ann = item.annotation
      default = item.value

      # Skip fields marked with init=False in field() call
      if (
        isinstance(default, ast.Call)
        and isinstance(default.func, ast.Name)
        and default.func.id == "field"
        and any(
          kw.arg == "init"
          and isinstance(kw.value, ast.Constant)
          and kw.value.value is False
          for kw in default.keywords
        )
      ):
        continue

      if _is_hyper_annotation(ann):
        inner_type, is_leaf = _unwrap_hyper(ann)
        _extract_constraints_from_hyper(ann, item.target.id)
      else:
        inner_type, is_leaf = _get_annotation_str(ann), False

      params.append(
        HyperParam(
          name=item.target.id,
          type_annotation=inner_type,
          default_value=_extract_default(default),
          is_leaf=is_leaf,
        )
      )

  # Also check __init__ for regular classes
  for item in node.body:
    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
      args_to_check = item.args.args[1:]  # Skip self
      num_defaults = len(item.args.defaults)
      num_args = len(args_to_check)
      defaults_offset = num_args - num_defaults

      for i, arg in enumerate(args_to_check):
        # Skip if already found in class body
        if any(p.name == arg.arg for p in params):
          continue

        ann = arg.annotation
        default_idx = i - defaults_offset
        default = item.args.defaults[default_idx] if default_idx >= 0 else None
        if ann and _is_hyper_annotation(ann):
          inner_type, is_leaf = _unwrap_hyper(ann)
          _extract_constraints_from_hyper(ann, arg.arg)
        else:
          inner_type, is_leaf = _get_annotation_str(ann), False

        params.append(
          HyperParam(
            name=arg.arg,
            type_annotation=inner_type,
            default_value=_extract_default(default),
            is_leaf=is_leaf,
          )
        )

  return ConfigurableInfo(
    name=node.name,
    is_class=True,
    params=params,
    return_type=node.name,
    docstring=ast.get_docstring(node),
  )


def _extract_configurable_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
  """Identify names that refer to 'configurable' and 'wrap_external'."""
  configurable_aliases = {"configurable"}
  wrap_external_aliases = {"wrap_external"}

  for node in tree.body:
    if isinstance(node, ast.ImportFrom) and node.module in {
      "nonfig",
      "nonfig.generation",
    }:
      for name in node.names:
        if name.name == "configurable":
          configurable_aliases.add(name.asname or name.name)
        if name.name == "wrap_external":
          wrap_external_aliases.add(name.asname or name.name)

  return configurable_aliases, wrap_external_aliases


def extract_primitive_aliases(tree: ast.Module) -> set[str]:
  """Identify type aliases to primitive types or containers.

  Example:
    ```python
    Vector = list[float]  # 'Vector' is an alias
    UserID = int  # 'UserID' is an alias
    ```
  """
  aliases: set[str] = set()

  # Primitive types and container names we recognize
  # Note: generator.py has _PRIMITIVE_TYPES and _CONTAINER_PREFIXES
  # We duplicate a minimal set here for the scanner
  primitives = {
    "int",
    "float",
    "str",
    "bool",
    "None",
    "bytes",
    "complex",
    "Any",
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
    "Sequence",
    "Mapping",
    "Iterable",
    "Collection",
    "Optional",
    "Union",
    "Callable",
  }

  for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target = node.targets[0]
      if isinstance(target, ast.Name):
        value = node.value

        is_alias = False

        # Check simple assignment: Alias = type
        if isinstance(value, ast.Name):
          if value.id in primitives:
            is_alias = True

        # Check subscript: Alias = list[...]
        elif (
          isinstance(value, ast.Subscript)
          and isinstance(value.value, ast.Name)
          and value.value.id in primitives
        ):
          is_alias = True

        # Check Annotated[T, Leaf]
        if (
          isinstance(value, ast.Subscript)
          and isinstance(value.value, ast.Name)
          and value.value.id == "Annotated"
        ):
          if isinstance(value.slice, ast.Tuple):
            # Annotated[T, Marker1, Marker2]
            markers = value.slice.elts[1:]
            if any(_is_leaf_marker(m) for m in markers):
              is_alias = True
          elif _is_leaf_marker(value.slice):
            # Annotated[T, Leaf]
            is_alias = True

        if is_alias:
          aliases.add(target.id)

  return aliases


def _collect_config_names(
  tree: ast.Module, configurable_aliases: set[str], wrap_aliases: set[str]
) -> set[str]:
  """Collect all names that refer to configurations in this module (including imports)."""
  config_names: set[str] = set()

  # Pass 1: Explicitly defined configurations and direct imports
  for node in tree.body:
    # 1. Decorated items
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      if any(
        _is_configurable_decorator(d, configurable_aliases) for d in node.decorator_list
      ):
        config_names.add(node.name)
    # 2. wrap_external assignments
    elif isinstance(node, ast.Assign):
      for target_node in node.targets:
        if isinstance(target_node, (ast.Tuple, ast.List)):
          targets = target_node.elts
          if isinstance(node.value, (ast.Tuple, ast.List)) and len(
            node.value.elts
          ) == len(targets):
            values = node.value.elts
          else:
            values = [node.value] * len(
              targets
            )  # Fallback for scalar assignments to multiple targets
        else:
          targets = [target_node]
          values = [node.value]

        for t, v in zip(targets, values, strict=False):
          if isinstance(t, ast.Name) and isinstance(v, ast.Call):
            func = v.func
            if (isinstance(func, ast.Name) and func.id in wrap_aliases) or (
              isinstance(func, ast.Attribute) and func.attr in wrap_aliases
            ):
              config_names.add(t.id)

    # 3. Imports from nonfig (likely configurations)
    elif isinstance(node, ast.ImportFrom) and node.module in {
      "nonfig",
      "nonfig.generation",
    }:
      for name in node.names:
        config_names.add(name.asname or name.name)

  # Pass 2: Infer configurations from usage patterns
  for node in ast.walk(tree):
    # Check function/method parameters
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      num_defaults = len(node.args.defaults)
      num_args = len(node.args.args)
      defaults_offset = num_args - num_defaults

      for i, arg in enumerate(node.args.args):
        default_idx = i - defaults_offset
        default = node.args.defaults[default_idx] if default_idx >= 0 else None
        ann = arg.annotation

        if ann is None or default is None:
          continue

        # Pattern: x: T = DEFAULT
        if _is_default_sentinel(default):
          if isinstance(ann, ast.Name):
            config_names.add(ann.id)
          elif isinstance(ann, ast.Attribute) and isinstance(ann.value, ast.Name):
            # T.Type = DEFAULT or T.Config = DEFAULT
            config_names.add(ann.value.id)

        # Pattern: x: T.Type = T or x: T.Config = T
        if (
          isinstance(default, ast.Name)
          and isinstance(ann, ast.Attribute)
          and isinstance(ann.value, ast.Name)
          and ann.value.id == default.id
          and ann.attr in {"Type", "Config"}
        ):
          config_names.add(default.id)

    # Check class attributes (dataclasses)
    elif isinstance(node, ast.ClassDef):
      for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
          if item.value is None:
            continue

          if _is_default_sentinel(item.value):
            if isinstance(item.annotation, ast.Name):
              config_names.add(item.annotation.id)
            elif isinstance(item.annotation, ast.Attribute) and isinstance(
              item.annotation.value, ast.Name
            ):
              config_names.add(item.annotation.value.id)

  return config_names


def _extract_wrapped_configs(
  tree: ast.Module, wrap_aliases: set[str]
) -> list[ConfigurableInfo]:
  """Extract information from wrap_external assignments."""
  results: list[ConfigurableInfo] = []
  for node in tree.body:
    if isinstance(node, ast.Assign):
      targets: list[ast.AST] = []
      values: list[ast.AST] = []

      for target_node in node.targets:
        if isinstance(target_node, (ast.Tuple, ast.List)):
          if isinstance(node.value, (ast.Tuple, ast.List)) and len(
            node.value.elts
          ) == len(target_node.elts):
            targets.extend(target_node.elts)
            values.extend(node.value.elts)
        else:
          targets.append(target_node)
          values.append(node.value)

      for t, v in zip(targets, values, strict=False):
        if isinstance(t, ast.Name) and isinstance(v, ast.Call):
          func = v.func
          is_wrap = False
          if (isinstance(func, ast.Name) and func.id in wrap_aliases) or (
            isinstance(func, ast.Attribute) and func.attr in wrap_aliases
          ):
            is_wrap = True

          if is_wrap and v.args:
            target_arg = v.args[0]
            target_name = ast.unparse(target_arg)

            results.append(
              ConfigurableInfo(
                name=t.id,
                is_class=True,
                params=[],
                return_type=target_name,
                is_wrapped=True,
              )
            )
  return results


def scan_module(path: Path) -> tuple[list[ConfigurableInfo], set[str]]:
  """
  Scan a Python module for @configurable items, wrap_external calls, and type aliases.

  Returns:
      Tuple of (list of ConfigurableInfo, set of alias names)
  """
  source = path.read_text(encoding="utf-8")
  tree = ast.parse(source)

  configurable_aliases, wrap_aliases = _extract_configurable_aliases(tree)
  type_aliases = extract_primitive_aliases(tree)

  # Pass 1: Collect all names that refer to configurations in this module
  config_names = _collect_config_names(tree, configurable_aliases, wrap_aliases)

  # Pass 2: Extract information for each configurable item (wrapped or decorated)
  results = _extract_wrapped_configs(tree, wrap_aliases)

  # Pass 3: find decorated items (using config_names for implicit hyper detection)
  for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      info = _scan_function(node, configurable_aliases, config_names)
      if info:
        results.append(info)
    elif isinstance(node, ast.ClassDef):
      info = _scan_class(node, configurable_aliases)
      if info:
        results.append(info)

  return results, type_aliases
