"""AST scanner for finding @configurable decorated items."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from nonfig.constraints import validate_constraint_conflicts

__all__ = ["ConfigurableInfo", "HyperParam", "scan_module"]


@dataclass
class HyperParam:
  """Information about a Hyper-annotated parameter."""

  name: str
  type_annotation: str
  default_value: str | None = None


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


def _is_configurable_decorator(node: ast.expr) -> bool:
  """Check if a decorator is @configurable."""
  if isinstance(node, ast.Name):
    return node.id == "configurable"
  if isinstance(node, ast.Attribute):
    return node.attr == "configurable"
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


def _unwrap_hyper(node: ast.expr) -> str:
  """Extract the inner type from Hyper[T, ...] -> T."""
  if isinstance(node, ast.Subscript):
    slice_node = node.slice

    # Handle Annotated[T, ...] -> T
    is_annotated = False
    if (isinstance(node.value, ast.Name) and node.value.id == "Annotated") or (
      isinstance(node.value, ast.Attribute) and node.value.attr == "Annotated"
    ):
      is_annotated = True

    if is_annotated:
      if isinstance(slice_node, ast.Tuple):
        return _get_annotation_str(slice_node.elts[0])
      return _get_annotation_str(slice_node)

    # Handle Hyper[...]
    if isinstance(slice_node, ast.Tuple):
      # Multiple args: Hyper[T, Ge[0], Le[100]]
      return _get_annotation_str(slice_node.elts[0])
    # Single arg: Hyper[T]
    return _get_annotation_str(slice_node)
  return _get_annotation_str(node)


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
  if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) < 2:  # noqa: PLR2004
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


def _scan_function(
  node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ConfigurableInfo | None:
  """Scan a function for @configurable decorator."""
  has_configurable = any(_is_configurable_decorator(d) for d in node.decorator_list)
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
    if (
      (ann and _is_hyper_annotation(ann))
      or _is_default_sentinel(default)
      or _is_configurable_callable_default(default, ann)
    ):
      is_hyper = True

    if is_hyper:
      inner_type = (
        _unwrap_hyper(ann)
        if ann and _is_hyper_annotation(ann)
        else _get_annotation_str(ann)
      )
      if ann and _is_hyper_annotation(ann):
        _extract_constraints_from_hyper(ann, arg.arg)

      params.append(
        HyperParam(
          name=arg.arg,
          type_annotation=inner_type,
          default_value=_extract_default(default),
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


def _scan_class(node: ast.ClassDef) -> ConfigurableInfo | None:
  """Scan a class for @configurable decorator."""
  has_configurable = any(_is_configurable_decorator(d) for d in node.decorator_list)
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
      if _is_hyper_annotation(ann):
        inner_type = _unwrap_hyper(ann)
        _extract_constraints_from_hyper(ann, item.target.id)
      else:
        inner_type = _get_annotation_str(ann)

      params.append(
        HyperParam(
          name=item.target.id,
          type_annotation=inner_type,
          default_value=_extract_default(default),
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
          inner_type = _unwrap_hyper(ann)
          _extract_constraints_from_hyper(ann, arg.arg)
        else:
          inner_type = _get_annotation_str(ann)

        params.append(
          HyperParam(
            name=arg.arg,
            type_annotation=inner_type,
            default_value=_extract_default(default),
          )
        )

  return ConfigurableInfo(
    name=node.name,
    is_class=True,
    params=params,
    return_type=node.name,
    docstring=ast.get_docstring(node),
  )


def scan_module(path: Path) -> list[ConfigurableInfo]:
  """Scan a Python module for @configurable decorated items."""
  source = path.read_text(encoding="utf-8")
  tree = ast.parse(source)

  results: list[ConfigurableInfo] = []

  for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      info = _scan_function(node)
      if info:
        results.append(info)
    elif isinstance(node, ast.ClassDef):
      info = _scan_class(node)
      if info:
        results.append(info)

  return results
