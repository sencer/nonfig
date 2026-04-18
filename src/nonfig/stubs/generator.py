"""Stub file generator for @configurable decorated items."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

from nonfig.stubs.scanner import ConfigurableInfo, HyperParam, scan_module

__all__ = ["generate_stub_for_file", "generate_stubs_for_directory"]


def _format_docstring(
  docstring: str | None,
  extra_sections: list[str] | None = None,
  indent: int = 4,
) -> str:
  """Format a docstring for inclusion in a stub file."""
  parts: list[str] = []

  if docstring:
    parts.append(docstring.expandtabs())

  if extra_sections:
    if parts:
      parts.append("")  # Blank line between description and sections
    parts.extend(extra_sections)

  if not parts:
    return ""

  full_doc = "\n".join(parts)
  lines = full_doc.splitlines()

  if not lines:
    return ""

  prefix = " " * indent

  # For single-line docstrings (only if no extra sections and short)
  if len(lines) == 1 and not extra_sections:
    return f'{prefix}"""{lines[0]}"""\n'

  # For multi-line docstrings
  formatted_lines = [f'{prefix}"""{lines[0]}']
  for line in lines[1:]:
    if not line.strip():
      formatted_lines.append("")
    else:
      formatted_lines.append(f"{prefix}{line}")

  # Ensure the closing quotes are on a new line and indented
  formatted_lines.append(f'{prefix}"""\n')

  return "\n".join(formatted_lines)


def _generate_params_doc_section(
  params: list[HyperParam],
  _indent: int = 4,
  aliases: set[str] | None = None,
  configurable_names: set[str] | None = None,
) -> str | None:
  """Generate documentation for parameters."""
  if aliases is None:
    aliases = set()
  lines: list[str] = []

  if params:
    lines.append("Configuration:")
    for param in params:
      config_type = _transform_to_config_type(
        param.type_annotation,
        aliases,
        is_leaf=param.is_leaf,
        configurable_names=configurable_names,
      )
      lines.append(f"    {param.name} ({config_type})")

  return "\n".join(lines)


def _collect_used_names(stub_content: str) -> set[str]:
  """Collect all names referenced in stub content.

  Parses the stub and collects Name nodes and the root of Attribute chains.
  Also recursively scans string constants which might contain type references.
  """
  used_names: set[str] = set()
  try:
    tree = ast.parse(stub_content)
  except SyntaxError:
    return used_names

  def _scan_node(node: ast.AST) -> None:
    if isinstance(node, ast.Name):
      used_names.add(node.id)
    elif isinstance(node, ast.Attribute):
      # For A.B.C, we want to collect 'A'
      curr = node.value
      while isinstance(curr, ast.Attribute):
        curr = curr.value
      if isinstance(curr, ast.Name):
        used_names.add(curr.id)
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
      # Try to parse string content as a type expression
      try:
        # We wrap it in a dummy expression to handle things like "int | str"
        sub_tree = ast.parse(node.value, mode="eval")
        for sub_node in ast.walk(sub_tree):
          _scan_node(sub_node)
      except SyntaxError:
        pass

  for node in ast.walk(tree):
    _scan_node(node)

  return used_names


def _filter_import_names(
  node: ast.ImportFrom, used_names: set[str]
) -> ast.ImportFrom | None:
  """Filter 'from x import a, b, c' to only keep used names."""
  kept_aliases: list[ast.alias] = []
  for alias in node.names:
    # Check the name that would be used in code (asname or name)
    name_in_code = alias.asname if alias.asname else alias.name
    # Skip TYPE_CHECKING - stubs don't need conditional imports
    if name_in_code == "TYPE_CHECKING":
      continue
    if name_in_code in used_names:
      kept_aliases.append(alias)

  if not kept_aliases:
    return None

  return ast.ImportFrom(
    module=node.module,
    names=kept_aliases,
    level=node.level,
  )


def _should_keep_import_alias(
  alias: ast.alias, used_names: set[str], already_imported: set[str]
) -> bool:
  """Check if an import alias should be kept."""
  name = alias.asname if alias.asname else alias.name.split(".")[0]
  return name in used_names and name not in already_imported


def _filter_type_checking_block(
  import_stmt: str, used_names: set[str], already_imported: set[str]
) -> str | None:
  """Filter a TYPE_CHECKING block to only keep used imports."""
  try:
    tree = ast.parse(import_stmt)
    if_node = tree.body[0]
  except SyntaxError:
    return import_stmt

  if not isinstance(if_node, ast.If):
    return import_stmt

  allowed_names = used_names - already_imported
  new_body: list[ast.stmt] = []

  for stmt in if_node.body:
    if isinstance(stmt, ast.ImportFrom):
      filtered = _filter_import_names(stmt, allowed_names)
      if filtered:
        new_body.append(filtered)
    elif isinstance(stmt, ast.Import):
      if any(
        _should_keep_import_alias(a, used_names, already_imported) for a in stmt.names
      ):
        new_body.append(stmt)

  if not new_body:
    return None

  if_node.body = new_body
  return ast.unparse(if_node)


def _filter_imports(
  imports: list[str], used_names: set[str], already_imported: set[str]
) -> list[str]:
  """Filter imports to only those whose names are used."""
  filtered: list[str] = []

  for import_stmt in imports:
    if "__future__" in import_stmt:
      filtered.append(import_stmt)
      continue

    # Handle 'if TYPE_CHECKING:' blocks (but not 'from typing import TYPE_CHECKING')
    # Note: import_stmt is unparsed code, so it might span multiple lines
    if import_stmt.lstrip().startswith("if ") and "TYPE_CHECKING" in import_stmt:
      result = _filter_type_checking_block(import_stmt, used_names, already_imported)
      if result:
        filtered.append(result)
      continue

    try:
      tree = ast.parse(import_stmt)
      node = tree.body[0]
    except SyntaxError:
      filtered.append(import_stmt)
      continue

    if isinstance(node, ast.Import):
      if any(
        _should_keep_import_alias(a, used_names, already_imported) for a in node.names
      ):
        filtered.append(import_stmt)
    elif isinstance(node, ast.ImportFrom):
      filtered_node = _filter_import_names(node, used_names - already_imported)
      if filtered_node:
        filtered.append(ast.unparse(filtered_node))

  return filtered


def _format_default(default_value: str | None) -> str:
  """Format a default value for use in a stub.

  Preserves special values like DEFAULT, uses ... for others.
  """
  if default_value is None:
    return ""
  if default_value == "DEFAULT":
    return " = DEFAULT"
  return " = ..."


# Primitive types that should NOT be transformed to .Config
_PRIMITIVE_TYPES = frozenset({
  "int",
  "float",
  "str",
  "bool",
  "None",
  "bytes",
  "complex",
  "Any",
  "dict",
  "list",
  "set",
  "tuple",
  "type",
  "object",
  "frozenset",
  "slice",
  "range",
  "pd.Timedelta",
  "pd.Timestamp",
  "pd.Series",
  "pd.DataFrame",
  "np.ndarray",
  "datetime",
  "date",
  "time",
  "timedelta",
  "Path",
})

# Generic container prefixes that should NOT be transformed
_CONTAINER_PREFIXES = (
  "list[",
  "dict[",
  "set[",
  "tuple[",
  "frozenset[",
  "Sequence[",
  "Mapping[",
  "Iterable[",
  "Collection[",
  "Optional[",
  "Union[",
  "Callable[",
  "Annotated[",
  "Literal[",
  "Type[",
)


def _is_primitive_or_container(type_str: str) -> bool:
  """Check if a type is primitive or a generic container.

  Primitive types and containers don't have .Config - only configurable classes do.
  """
  # Direct primitive match
  if type_str in _PRIMITIVE_TYPES:
    return True

  # Generic container (list[int], dict[str, float], etc.)
  type_lower = type_str.lower()
  return any(type_lower.startswith(prefix.lower()) for prefix in _CONTAINER_PREFIXES)


def _is_pascal_case(s: str) -> bool:
  """Check if a string is PascalCase (e.g. MyClass, not my_func or CONSTANT)."""
  # Handle nested names like A.B.C
  parts = s.split(".")
  last_part = parts[-1]
  # PascalCase: starts with uppercase, and is not ALL_UPPER (which usually means constant)
  return bool(last_part and last_part[0].isupper() and not last_part.isupper())


def _should_transform_to_config(
  type_str: str, aliases: set[str], configurable_names: set[str] | None = None
) -> bool:
  """Determine if a Hyper type should be transformed to .Config.

  Non-primitive types in Hyper[] must be configurable classes, so they
  should use .Config in the Config class's __init__ signature.
  """

  if type_str in aliases:
    return False
  if _is_primitive_or_container(type_str):
    return False
  if configurable_names and type_str in configurable_names:
    return True
  return _is_pascal_case(type_str)


def _is_likely_type_ref(s: str, configurable_names: set[str] | None = None) -> bool:
  """Check if a string literal is likely a type reference (forward ref)."""
  if _is_pascal_case(s) or "[" in s or "|" in s or s.endswith(".Type"):
    return True
  return bool(configurable_names and s in configurable_names)


def _transform_ast_node(
  node: ast.expr, aliases: set[str], configurable_names: set[str] | None = None
) -> ast.expr:
  """Recursively transform an AST type node."""
  # Handle A | B (Python 3.10+ unions)
  if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
    return ast.BinOp(
      left=_transform_ast_node(node.left, aliases, configurable_names),
      op=ast.BitOr(),
      right=_transform_ast_node(node.right, aliases, configurable_names),
    )

  # Handle containers: list[T], Union[T1, T2], etc.
  if isinstance(node, ast.Subscript):
    # Transform the slice (the part inside [])
    if isinstance(node.slice, ast.Tuple):
      new_elts = [
        _transform_ast_node(elt, aliases, configurable_names) for elt in node.slice.elts
      ]
      new_slice: ast.expr = ast.Tuple(elts=new_elts, ctx=ast.Load())
    else:
      new_slice = _transform_ast_node(node.slice, aliases, configurable_names)

    return ast.Subscript(
      value=node.value,
      slice=new_slice,
      ctx=ast.Load(),
    )

  # Handle Dict literals (common in Overrides)
  if isinstance(node, ast.Dict):
    return ast.Dict(
      keys=[
        _transform_ast_node(k, aliases, configurable_names) if k else None
        for k in node.keys
      ],
      values=[_transform_ast_node(v, aliases, configurable_names) for v in node.values],
    )

  # Handle List literals
  if isinstance(node, ast.List):
    return ast.List(
      elts=[_transform_ast_node(e, aliases, configurable_names) for e in node.elts],
      ctx=node.ctx,
    )

  # Handle Tuple literals
  if isinstance(node, ast.Tuple):
    return ast.Tuple(
      elts=[_transform_ast_node(e, aliases, configurable_names) for e in node.elts],
      ctx=node.ctx,
    )

  # Leaf type (Name or Attribute)
  if isinstance(node, (ast.Name, ast.Attribute)):
    type_str = ast.unparse(node)

    # Strip .Type suffix before transforming to .Config
    if type_str.endswith(".Type"):
      type_str = type_str.removesuffix(".Type")
      import contextlib

      with contextlib.suppress(Exception):
        # Re-parse to get the base name without .Type
        # This handles cases like 'foo.bar.Type' -> 'foo.bar'
        new_node = ast.parse(type_str, mode="eval").body
        if isinstance(new_node, (ast.Name, ast.Attribute)):
          node = new_node

    if _should_transform_to_config(type_str, aliases, configurable_names):
      # Transform to T.Config | T.ConfigDict
      config_attr = ast.Attribute(value=node, attr="Config", ctx=ast.Load())
      dict_attr = ast.Attribute(value=node, attr="ConfigDict", ctx=ast.Load())
      return ast.BinOp(left=config_attr, op=ast.BitOr(), right=dict_attr)

  # Handle forward references (string literals) inside containers
  if (
    isinstance(node, ast.Constant)
    and isinstance(node.value, str)
    and _is_likely_type_ref(node.value, configurable_names)
  ):
    try:
      sub_node = ast.parse(node.value, mode="eval").body
      transformed_sub = _transform_ast_node(sub_node, aliases, configurable_names)
      # Wrap back in a Constant with the unparsed (transformed) string
      return ast.Constant(value=ast.unparse(transformed_sub))
    except SyntaxError:
      pass

  return node


def _transform_to_config_type(
  type_str: str,
  aliases: set[str] | None = None,
  is_leaf: bool = False,
  configurable_names: set[str] | None = None,
) -> str:
  """Transform a type to its .Config variant.

  For Config class __init__ parameters, nested configurable types should
  be their .Config type since you pass configs, not instances.

  Examples:
    ```
    DataPreprocessor -> DataPreprocessor.Config | DataPreprocessor.ConfigDict
    Model -> Model.Config | Model.ConfigDict
    int -> int (unchanged, primitive)
    list[Model] -> list[Model.Config | Model.ConfigDict]
    str | Model -> str | Model.Config | Model.ConfigDict
    ```
  """
  if is_leaf:
    return type_str

  # Handle .Type suffix (e.g. cap.Type -> cap)
  if type_str.endswith(".Type"):
    type_str = type_str.removesuffix(".Type")

  if aliases is None:
    aliases = set()

  # Use AST to parse and transform the type string safely
  try:
    # mode='eval' is for single expressions
    node = ast.parse(type_str, mode="eval").body
    transformed = _transform_ast_node(node, aliases, configurable_names)
    return ast.unparse(transformed)
  except Exception:  # noqa: BLE001
    # Fallback to simple logic if parsing fails
    if _should_transform_to_config(type_str, aliases, configurable_names):
      return f"{type_str}.Config | {type_str}.ConfigDict"
    return type_str


def _generate_config_dict(
  info: ConfigurableInfo,
  indent: int = 4,
  name: str = "ConfigDict",
  aliases: set[str] | None = None,
  configurable_names: set[str] | None = None,
) -> str:
  """Generate the ConfigDict TypedDict."""
  if aliases is None:
    aliases = set()

  lines: list[str] = []
  prefix = " " * indent
  noqa = "  # noqa: N801" if name.startswith("_") else ""
  lines.append(f"{prefix}class {name}(TypedDict, total=False):{noqa}")

  # Generate docstring for ConfigDict
  doc_section = _generate_params_doc_section(
    info.params, aliases=aliases, configurable_names=configurable_names
  )
  if doc_section:
    lines.append(
      _format_docstring(
        f"Configuration dictionary for {info.name}.",
        [doc_section],
        indent=indent + 4,
      )
    )

  if not info.params:
    lines.append(f"{prefix}    pass")
  else:
    for param in info.params:
      config_type = _transform_to_config_type(
        param.type_annotation,
        aliases,
        is_leaf=param.is_leaf,
        configurable_names=configurable_names,
      )
      lines.append(f"{prefix}    {param.name}: {config_type}")

  return "\n".join(lines)


def _generate_config_class(
  info: ConfigurableInfo, aliases: set[str], configurable_names: set[str]
) -> str:
  """Generate the Config class stub."""
  lines: list[str] = []

  # Determine make() return type
  if info.is_class:
    make_return = info.return_type
  # For functions, make() returns a callable
  elif info.call_params:
    call_param_types = ", ".join(t for _, t, _ in info.call_params)
    make_return = f"Callable[[{call_param_types}], {info.return_type}]"
  else:
    make_return = f"Callable[[], {info.return_type}]"

  lines.append(f"    class Config(_NCMakeableModel[{make_return}]):")

  # Generate docstring for Config class: params + original docstring
  doc_section = _generate_params_doc_section(
    info.params, aliases=aliases, configurable_names=configurable_names
  )

  # Prepare the description part: "Configuration class for X.\n\nOriginal docstring"
  description = f"Configuration class for {info.name}."
  if info.docstring:
    description += f"\n\n{info.docstring}"

  lines.append(
    _format_docstring(description, [doc_section] if doc_section else None, indent=8)
  )

  if not info.params:
    lines.append("        def __init__(self) -> None: ...")
    # make() method - marked as override since MakeableModel defines it
    lines.append("        @override")
    lines.append(f"        def make(self) -> {make_return}: ...")
  else:
    # Field declarations - transform non-primitive types to .Config
    for param in info.params:
      config_type = _transform_to_config_type(
        param.type_annotation,
        aliases,
        is_leaf=param.is_leaf,
        configurable_names=configurable_names,
      )
      lines.append(f"        {param.name}: {config_type}")

    # __init__ signature - transform non-primitive types to .Config
    init_params: list[str] = []
    for param in info.params:
      config_type = _transform_to_config_type(
        param.type_annotation,
        aliases,
        is_leaf=param.is_leaf,
        configurable_names=configurable_names,
      )
      default_str = _format_default(param.default_value)
      init_params.append(f"{param.name}: {config_type}{default_str}")

    init_sig = ", ".join(init_params)
    lines.append(f"        def __init__(self, *, {init_sig}) -> None: ...")

    # Add docstring to __init__ to prevent fallback to BaseModel.__init__ docstring
    if doc_section:
      lines.append(
        _format_docstring(
          f"Initialize configuration for {info.name}.", [doc_section], indent=8
        )
      )

    # make() method - marked as override since MakeableModel defines it
    lines.append("        @override")
    lines.append(f"        def make(self) -> {make_return}: ...")

  return "\n".join(lines)


def _generate_class_stub(
  info: ConfigurableInfo, aliases: set[str], configurable_names: set[str]
) -> str:
  """Generate stub for a @configurable decorated class."""
  lines: list[str] = []

  lines.append(f"class {info.name}:")

  doc_section = _generate_params_doc_section(
    info.params, aliases=aliases, configurable_names=configurable_names
  )
  extra = [doc_section] if doc_section else None
  lines.append(_format_docstring(info.docstring, extra, indent=4))
  lines.append(
    _generate_config_dict(info, aliases=aliases, configurable_names=configurable_names)
  )
  lines.append(_generate_config_class(info, aliases, configurable_names))

  # Instance attribute declarations (for dataclasses)
  if info.params:
    for param in info.params:
      lines.append(f"    {param.name}: {param.type_annotation}")

  # __init__ signature
  if info.params:
    init_params: list[str] = []
    for param in info.params:
      default_str = _format_default(param.default_value)
      init_params.append(f"{param.name}: {param.type_annotation}{default_str}")
    init_sig = ", ".join(init_params)
    lines.append(f"    def __init__(self, {init_sig}) -> None: ...")
  else:
    lines.append("    def __init__(self) -> None: ...")

  return "\n".join(lines)


def _generate_function_stub(
  info: ConfigurableInfo, aliases: set[str], configurable_names: set[str]
) -> str:
  """Generate stub for a @configurable decorated function."""
  lines: list[str] = []

  # Generate the BoundFunction Protocol
  lines.append(f"class _{info.name}_Bound(Protocol):  # noqa: N801")
  lines.append('    """Bound function with hyperparameters as attributes."""')

  for param in info.params:
    lines.append("    @property")
    lines.append(f"    def {param.name}(self) -> {param.type_annotation}: ...")

  call_params: list[str] = []
  for name, type_ann, default in info.call_params:
    if default:
      call_params.append(f"{name}: {type_ann} = ...")
    else:
      call_params.append(f"{name}: {type_ann}")
  call_sig = ", ".join(call_params)
  lines.append(f"    def __call__(self, {call_sig}) -> {info.return_type}: ...")

  lines.append("")

  # Generate the ConfigDict
  config_dict_name = f"_{info.name}_ConfigDict"
  lines.append(
    _generate_config_dict(
      info,
      indent=0,
      name=config_dict_name,
      aliases=aliases,
      configurable_names=configurable_names,
    )
  )

  lines.append("")

  # Generate Config class
  lines.append(
    f"class _{info.name}_Config(_NCMakeableModel[_{info.name}_Bound]):  # noqa: N801"
  )

  doc_section = _generate_params_doc_section(
    info.params, aliases=aliases, configurable_names=configurable_names
  )
  description = f"Configuration class for {info.name}."
  if info.docstring:
    description += f"\n\n{info.docstring}"
  lines.append(
    _format_docstring(description, [doc_section] if doc_section else None, indent=4)
  )

  if not info.params:
    lines.append("    pass")
  else:
    for param in info.params:
      # Config fields
      config_type = _transform_to_config_type(
        param.type_annotation,
        aliases,
        is_leaf=param.is_leaf,
        configurable_names=configurable_names,
      )
      lines.append(f"    {param.name}: {config_type}")

    # __init__
    init_params: list[str] = []
    for param in info.params:
      config_type = _transform_to_config_type(
        param.type_annotation,
        aliases,
        is_leaf=param.is_leaf,
        configurable_names=configurable_names,
      )
      default_str = _format_default(param.default_value)
      init_params.append(f"{param.name}: {config_type}{default_str}")

    init_sig = ", ".join(init_params)
    lines.append(f"    def __init__(self, *, {init_sig}) -> None: ...")

    if doc_section:
      lines.append(
        _format_docstring(
          f"Initialize configuration for {info.name}.", [doc_section], indent=4
        )
      )

    lines.append("    @override")
    lines.append(f"    def make(self) -> _{info.name}_Bound: ...")

  lines.append("")

  # Generate the function wrapper as a CLASS matching the runtime behavior
  # of type[ConfigurableFunc] but with specific attributes.
  lines.append(f"class {info.name}:  # noqa: N801")
  lines.append(f"    Type = _{info.name}_Bound")
  lines.append(f"    Config = _{info.name}_Config")
  lines.append(f"    ConfigDict = {config_dict_name}")

  # Expose default hyperparameters as CLASS attributes
  for param in info.params:
    if param.default_value is not None:
      lines.append(f"    {param.name}: ClassVar[{param.type_annotation}]")

  # __new__ to handle call behavior
  all_params: list[str] = []
  for name, type_ann, default in info.call_params:
    if default:
      all_params.append(f"{name}: {type_ann} = ...")
    else:
      all_params.append(f"{name}: {type_ann}")
  for param in info.params:
    default_str = _format_default(param.default_value)
    all_params.append(f"{param.name}: {param.type_annotation}{default_str}")

  all_call_sig = ", ".join(all_params)
  lines.append(f"    def __new__(cls, {all_call_sig}) -> {info.return_type}: ...")

  return "\n".join(lines)


def _extract_imports(tree: ast.Module) -> tuple[list[str], list[str]]:
  """Extract imports from AST tree."""
  future_imports: list[str] = []
  other_imports: list[str] = []

  for node in tree.body:
    if isinstance(node, ast.ImportFrom):
      if node.module == "__future__":
        future_imports.append(ast.unparse(node))
      else:
        other_imports.append(ast.unparse(node))
    elif isinstance(node, ast.Import):
      other_imports.append(ast.unparse(node))
    elif isinstance(node, ast.If):
      # Include TYPE_CHECKING blocks
      test_source = ast.unparse(node.test)
      if "TYPE_CHECKING" in test_source:
        other_imports.append(ast.unparse(node))

  return future_imports, other_imports


def _is_wrap_external_call(
  node: ast.expr, wrap_aliases: set[str] | None = None
) -> bool:
  """Check if an expression is a call to wrap_external or wrap."""
  aliases = wrap_aliases or {"wrap_external", "wrap"}
  if not isinstance(node, ast.Call):
    return False
  func = node.func
  return (isinstance(func, ast.Name) and func.id in aliases) or (
    isinstance(func, ast.Attribute) and func.attr in aliases
  )


def _is_allowed_decorator(node: ast.expr, allowed_names: set[str]) -> bool:
  """Check if a decorator is in the allowed list, handling calls like @dataclass()."""
  if isinstance(node, ast.Name):
    return node.id in allowed_names
  if isinstance(node, ast.Attribute):
    return node.attr in allowed_names
  if isinstance(node, ast.Call):
    return _is_allowed_decorator(node.func, allowed_names)
  return False


def _generate_non_configurable_class_stub(node: ast.ClassDef) -> str:
  """Generate a stub for a non-configurable class."""
  # Common class decorators that affect typing/structure
  allowed_class_decorators = {
    "dataclass",
    "final",
    "runtime_checkable",
    "sealed",
  }
  filtered_class_decorators: list[ast.expr] = [
    d for d in node.decorator_list if _is_allowed_decorator(d, allowed_class_decorators)
  ]

  stub_body: list[ast.stmt] = []
  for item in node.body:
    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
      # Filter decorators to keep only standard ones in stubs
      allowed_decorators = {
        "property",
        "classmethod",
        "staticmethod",
        "abstractmethod",
        "override",
      }
      filtered_decorators: list[ast.expr] = [
        d for d in item.decorator_list if _is_allowed_decorator(d, allowed_decorators)
      ]

      # Replace method body with ... but keep docstring
      method_body: list[ast.stmt] = []
      if (
        item.body
        and isinstance(item.body[0], ast.Expr)
        and isinstance(item.body[0].value, ast.Constant)
      ):
        # Keep docstring
        method_body.append(item.body[0])
      method_body.append(ast.Expr(value=ast.Constant(value=...)))

      # We use cast(Any, ...) because basedpyright cannot statically verify that type(item)
      # is a constructor that accepts these specific arguments.
      stub_method = cast("Any", type(item))(
        name=item.name,
        args=item.args,
        body=method_body,
        decorator_list=filtered_decorators,
        returns=item.returns,
        type_comment=item.type_comment,
        type_params=getattr(item, "type_params", []),
        lineno=item.lineno,
        col_offset=item.col_offset,
      )
      stub_body.append(stub_method)
    elif isinstance(item, ast.AnnAssign):
      # Skip 'Config' annotations - they're for @configurable classes
      if isinstance(item.target, ast.Name) and item.target.id == "Config":
        continue
      stub_item = ast.AnnAssign(
        target=item.target,
        annotation=item.annotation,
        value=ast.Constant(value=...) if item.value else None,
        simple=item.simple,
        lineno=item.lineno,
        col_offset=item.col_offset,
      )
      stub_body.append(stub_item)
    elif isinstance(item, ast.Assign):
      stub_item = ast.Assign(
        targets=item.targets,
        value=ast.Constant(value=...),
        type_comment=item.type_comment,
        lineno=item.lineno,
        col_offset=item.col_offset,
      )
      stub_body.append(stub_item)
    elif isinstance(item, ast.Pass):
      stub_body.append(item)
    elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
      # Keep docstrings
      stub_body.append(item)

  if not stub_body:
    stub_body = [ast.Pass()]

  stub_class = ast.ClassDef(
    name=node.name,
    bases=node.bases,
    keywords=node.keywords,
    body=stub_body,
    decorator_list=filtered_class_decorators,
    type_params=getattr(node, "type_params", []),
    lineno=node.lineno,
    col_offset=node.col_offset,
  )
  return ast.unparse(stub_class)


def _extract_public_items(
  tree: ast.Module,
  configurable_names: set[str],
  aliases: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
  """Extract all public (non-configurable) items from source.

  Returns:
      (class_stubs, function_stubs, constant_stubs)
  """
  if aliases is None:
    aliases = set()

  class_stubs: list[str] = []
  function_stubs: list[str] = []
  constant_stubs: list[str] = []

  for node in tree.body:
    # Skip private items
    if isinstance(node, ast.ClassDef):
      if node.name.startswith("_") or node.name in configurable_names:
        continue
      class_stubs.append(_generate_non_configurable_class_stub(node))

    elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
      # Skip private and configurable functions
      if node.name.startswith("_") or node.name in configurable_names:
        continue
      # Generate stub for non-configurable function
      stub_func = cast("Any", type(node))(
        name=node.name,
        args=node.args,
        body=[ast.Expr(value=ast.Constant(value=...))],
        decorator_list=[],
        returns=node.returns,
        type_comment=node.type_comment,
        type_params=getattr(node, "type_params", []),
        lineno=node.lineno,
        col_offset=node.col_offset,
      )
      function_stubs.append(ast.unparse(stub_func))

    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      # Type-annotated constants/type aliases
      if not node.target.id.startswith("_"):
        # For stubs: just "name: type" without value, or "name: type = ..." if has default
        name = node.target.id
        type_str = ast.unparse(node.annotation)
        if node.value is not None:
          # Always use ... for annotated constants to avoid value dependencies
          constant_stubs.append(f"{name}: {type_str} = ...")
        else:
          constant_stubs.append(f"{name}: {type_str}")

    elif isinstance(node, ast.Assign):
      # Simple assignments (constants/type aliases)
      for target in node.targets:
        if not isinstance(target, ast.Name) or target.id.startswith("_"):
          continue

        # Skip wrap_external calls - they are handled in configurable_stubs
        if _is_wrap_external_call(node.value, aliases):
          continue

        # Check if we know this is a type alias
        is_alias = target.id in aliases
        value_str = ast.unparse(node.value)

        if is_alias:
          constant_stubs.append(f"{target.id} = {value_str}")
        elif "[" in value_str or (
          value_str and value_str[0].isupper() and not value_str.startswith(('"', "'"))
        ):
          # Fallback heuristic: exclude string literals starting with quote
          constant_stubs.append(f"{target.id} = {value_str}")
        else:
          constant_stubs.append(f"{target.id}: ...")
        break

    # Skip if __name__ == "__main__": blocks (they're ast.If nodes, handled implicitly by not matching)

  return class_stubs, function_stubs, constant_stubs


def _generate_configurable_stubs(
  infos: list[ConfigurableInfo], aliases: set[str], configurable_names: set[str]
) -> list[str]:

  stubs: list[str] = []
  for info in infos:
    if info.is_wrapped:
      # For wrapped items, the target IS the config class itself
      lines = [
        f"class {info.name}(_NCMakeableModel[{info.return_type}]):",
        "    def __init__(self, **kwargs: Any) -> None: ...",
        "    @override",
        f"    def make(self) -> {info.return_type}: ...",
      ]
      stubs.append("\n".join(lines))
    elif info.is_class:
      stubs.append(_generate_class_stub(info, aliases, configurable_names))
    else:
      stubs.append(_generate_function_stub(info, aliases, configurable_names))
  return stubs


def _assemble_stub_content(
  import_lines: list[str],
  public_constants: list[str],
  public_funcs: list[str],
  public_classes: list[str],
  configurable_stubs: list[str],
) -> str:
  """Assemble the final stub file content."""
  lines: list[str] = []
  lines.append('"""Auto-generated type stubs for @configurable decorators.')
  lines.append("")
  lines.append("Do not edit manually - regenerate with: nonfig-stubgen <path>")
  lines.append('"""')
  lines.append("")

  lines.extend(import_lines)
  lines.append("")

  if public_constants:
    lines.append("")
    lines.extend(public_constants)

  for func_stub in public_funcs:
    lines.append("")
    lines.append(func_stub)

  for class_stub in public_classes:
    lines.append("")
    lines.append(class_stub)

  for stub in configurable_stubs:
    lines.append("")
    lines.append(stub)

  lines.append("")

  return "\n".join(lines)


def _build_import_section(
  tree: ast.Module,
  all_stubs: str,
  used_names: set[str],
) -> list[str]:
  """Build the import section for the stub file."""
  lines: list[str] = []

  has_callable = "Callable" in all_stubs
  has_default = "DEFAULT" in all_stubs

  # Extract and filter source imports
  future_imports, other_imports = _extract_imports(tree)

  # Filter out top-level nonfig imports that we regenerate, but keep TYPE_CHECKING blocks
  # regardless of what they contain (they will be filtered by usage later)
  kept_other_imports: list[str] = []
  for imp in other_imports:
    if "from nonfig import" in imp:
      if imp.lstrip().startswith("if ") and "TYPE_CHECKING" in imp:
        kept_other_imports.append(imp)
    else:
      kept_other_imports.append(imp)
  other_imports = kept_other_imports

  # Track what we're already importing
  # Note: We alias MakeableModel to _NCMakeableModel, so we don't
  # add MakeableModel to already_imported. If the user imports it,
  # it's distinct from our internal base class alias.
  already_imported: set[str] = set()
  if has_callable:
    already_imported.add("Callable")
  if has_default:
    already_imported.add("DEFAULT")

  filtered_imports = _filter_imports(other_imports, used_names, already_imported)

  # Check if TYPE_CHECKING is used in filtered imports
  has_type_checking_block = any(
    "TYPE_CHECKING" in imp and imp.lstrip().startswith("if ")
    for imp in filtered_imports
  )

  # Future imports first
  if future_imports:
    lines.extend(future_imports)
    lines.append("")

  # Standard library imports
  std_imports: list[str] = []
  if has_callable:
    std_imports.append("from collections.abc import Callable")

  # Detect common typing decorators/types
  typing_names_to_check = {
    "Annotated",
    "Any",
    "ClassVar",
    "Literal",
    "Optional",
    "Protocol",
    "TypedDict",
    "Union",
    "final",
    "override",
    "runtime_checkable",
    "sealed",
  }

  used_typing_names: set[str] = set()
  for name in typing_names_to_check:
    if name in used_names:
      used_typing_names.add(name)

  if has_type_checking_block:
    used_typing_names.add("TYPE_CHECKING")

  if used_typing_names:
    std_imports.append(f"from typing import {', '.join(sorted(used_typing_names))}")

  if "dataclass" in used_names:
    std_imports.append("from dataclasses import dataclass")

  if std_imports:
    lines.extend(std_imports)
    lines.append("")

  # nonfig imports
  # Use alias to avoid collision with user's own MakeableModel
  lines.append("from nonfig import MakeableModel as _NCMakeableModel")

  # Detect common nonfig exports by usage
  nonfig_exports_to_check = {"DEFAULT", "Overrides", "Hyper", "Leaf"}
  used_nonfig_exports: set[str] = set()

  for name in nonfig_exports_to_check:
    if name in used_names:
      used_nonfig_exports.add(name)

  if has_default and "DEFAULT" not in used_nonfig_exports:
    # Fallback check for DEFAULT since it's used in default_str logic
    used_nonfig_exports.add("DEFAULT")

  if used_nonfig_exports:
    lines.append(f"from nonfig import {', '.join(sorted(used_nonfig_exports))}")

  # Other filtered imports
  if filtered_imports:
    lines.append("")
    lines.extend(filtered_imports)

  return lines


def generate_stub_content(
  infos: list[ConfigurableInfo],
  source_path: Path,
  aliases: set[str] | None = None,
) -> str:
  """Generate complete stub file content."""
  if aliases is None:
    aliases = set()

  source = source_path.read_text(encoding="utf-8")
  tree = ast.parse(source)

  configurable_names = {info.name for info in infos}

  public_classes, public_funcs, public_constants = _extract_public_items(
    tree, configurable_names, aliases
  )

  if not infos and not public_classes and not public_funcs and not public_constants:
    return ""

  configurable_stubs = _generate_configurable_stubs(infos, aliases, configurable_names)

  # Combine all stubs to collect used names
  all_stubs = "\n\n".join(
    configurable_stubs + public_classes + public_funcs + public_constants
  )
  used_names = _collect_used_names(all_stubs)
  used_names.update({"MakeableModel"})

  import_lines = _build_import_section(tree, all_stubs, used_names)

  return _assemble_stub_content(
    import_lines, public_constants, public_funcs, public_classes, configurable_stubs
  )


def generate_stub_for_file(path: Path) -> Path | None:
  """Generate a .pyi stub file for a Python source file.

  Only generates stubs for files that have @configurable decorated items.
  When a stub is generated, it includes ALL public items (not just configurables)
  to ensure the stub is complete for type checkers.

  Args:
      path: Path to the Python source file

  Returns:
      Path to generated stub file, or None if no configurables found
  """
  infos, aliases = scan_module(path)
  if not infos:
    return None

  content = generate_stub_content(infos, path, aliases)
  if not content:
    return None

  stub_path = path.with_suffix(".pyi")
  stub_path.write_text(content, encoding="utf-8")
  return stub_path


def generate_stubs_for_directory(directory: Path, recursive: bool = True) -> list[Path]:
  """Generate stub files for all Python files in a directory.

  Args:
      directory: Directory to scan
      recursive: Whether to scan subdirectories

  Returns:
      List of generated stub file paths
  """
  generated: list[Path] = []

  pattern = "**/*.py" if recursive else "*.py"
  for py_file in directory.glob(pattern):
    # Skip private modules but not dunder files like __init__.py
    if py_file.name.startswith("_") and not py_file.name.startswith("__"):
      continue
    stub_path = generate_stub_for_file(py_file)
    if stub_path:
      generated.append(stub_path)

  return generated
