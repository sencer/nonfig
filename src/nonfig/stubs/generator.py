"""Stub file generator for @configurable decorated items."""

from __future__ import annotations

import ast
from pathlib import Path

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
  hyper_params: list[HyperParam],
  call_params: list[tuple[str, str, str | None]] | None = None,
) -> str:
  """Generate documentation for parameters.

  Args:
      hyper_params: list of hyper parameters
      call_params: optional list of call parameters (name, type, default)
  """
  lines: list[str] = []

  # If we have both call params and hyper params, distinguish them
  if call_params:
    lines.append("Call Arguments:")
    for name, type_ann, _ in call_params:
      lines.append(f"    {name} ({type_ann})")

    if hyper_params:
      lines.append("")
      lines.append("Hyperparameters:")
      for param in hyper_params:
        config_type = _transform_to_config_type(param.type_annotation)
        lines.append(f"    {param.name} ({config_type})")

  elif hyper_params:
    # Only hyper params - just use "Configuration" or "Parameters"
    lines.append("Configuration:")
    for param in hyper_params:
      config_type = _transform_to_config_type(param.type_annotation)
      lines.append(f"    {param.name} ({config_type})")

  return "\n".join(lines)


def _collect_used_names(stub_content: str) -> set[str]:
  """Collect all names referenced in stub content.

  Parses the stub and collects Name nodes and the root of Attribute chains.
  """
  used_names: set[str] = set()
  try:
    tree = ast.parse(stub_content)
  except SyntaxError:
    return used_names

  for node in ast.walk(tree):
    if isinstance(node, ast.Name):
      used_names.add(node.id)
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
      used_names.add(node.value.id)

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


def _should_transform_to_config(type_str: str) -> bool:
  """Determine if a Hyper type should be transformed to .Config.

  Non-primitive types in Hyper[] must be configurable classes, so they
  should use .Config in the Config class's __init__ signature.
  """
  return not _is_primitive_or_container(type_str)


def _transform_to_config_type(type_str: str) -> str:
  """Transform a type to its .Config variant.

  For Config class __init__ parameters, nested configurable types should
  be their .Config type since you pass configs, not instances.

  Examples:
    DataPreprocessor -> DataPreprocessor.Config
    Model -> Model.Config
    int -> int (unchanged, primitive)
    list[float] -> list[float] (unchanged, container)
    cap.Type -> cap.Config (Type suffix removed)
  """
  # Handle .Type suffix (e.g. cap.Type -> cap)
  if type_str.endswith(".Type"):
    type_str = type_str.removesuffix(".Type")

  if _should_transform_to_config(type_str):
    # Allow passing a TypedDict for nested configs
    return f"{type_str}.Config | {type_str}.ConfigDict"
  return type_str


def _generate_config_dict(
  info: ConfigurableInfo, indent: int = 4, name: str = "ConfigDict"
) -> str:
  """Generate the ConfigDict TypedDict."""
  lines: list[str] = []
  prefix = " " * indent
  lines.append(f"{prefix}class {name}(TypedDict, total=False):")

  # Generate docstring for ConfigDict
  doc_section = _generate_params_doc_section(info.params)
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
      config_type = _transform_to_config_type(param.type_annotation)
      lines.append(f"{prefix}    {param.name}: {config_type}")

  return "\n".join(lines)


def _generate_config_class(info: ConfigurableInfo) -> str:
  """Generate the Config class stub."""
  lines: list[str] = []

  # Determine make() return type
  if info.is_class:
    make_return = info.name
  # For functions, make() returns a callable
  elif info.call_params:
    call_param_types = ", ".join(t for _, t, _ in info.call_params)
    make_return = f"Callable[[{call_param_types}], {info.return_type}]"
  else:
    make_return = f"Callable[[], {info.return_type}]"

  lines.append(f"    class Config(_NCMakeableModel[{make_return}]):")

  # Generate docstring for Config class: params + original docstring
  doc_section = _generate_params_doc_section(info.params)

  # Prepare the description part: "Configuration class for X.\n\nOriginal docstring"
  description = f"Configuration class for {info.name}."
  if info.docstring:
    description += f"\n\n{info.docstring}"

  lines.append(
    _format_docstring(description, [doc_section] if doc_section else None, indent=8)
  )

  if not info.params:
    lines.append("        pass")
  else:
    # Field declarations - transform non-primitive types to .Config
    for param in info.params:
      config_type = _transform_to_config_type(param.type_annotation)
      lines.append(f"        {param.name}: {config_type}")

    # __init__ signature - transform non-primitive types to .Config
    init_params: list[str] = []
    for param in info.params:
      config_type = _transform_to_config_type(param.type_annotation)
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


def _generate_class_stub(info: ConfigurableInfo) -> str:
  """Generate stub for a @configurable decorated class."""
  lines: list[str] = []

  lines.append(f"class {info.name}:")

  doc_section = _generate_params_doc_section(info.params)
  extra = [doc_section] if doc_section else None
  lines.append(_format_docstring(info.docstring, extra, indent=4))
  lines.append(_generate_config_dict(info))
  lines.append(_generate_config_class(info))

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


def _generate_function_stub(info: ConfigurableInfo) -> str:  # noqa: PLR0915
  """Generate stub for a @configurable decorated function."""
  lines: list[str] = []

  # Generate the BoundFunction Protocol
  lines.append(f"class _{info.name}_Bound(Protocol):")
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
  lines.append(_generate_config_dict(info, indent=0, name=config_dict_name))
  lines.append("")

  # Generate Config class
  lines.append(f"class _{info.name}_Config(_NCMakeableModel[_{info.name}_Bound]):")

  doc_section = _generate_params_doc_section(info.params)
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
      config_type = _transform_to_config_type(param.type_annotation)
      lines.append(f"    {param.name}: {config_type}")

    # __init__
    init_params: list[str] = []
    for param in info.params:
      config_type = _transform_to_config_type(param.type_annotation)
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
  lines.append(f"class {info.name}:")
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


def _extract_public_items(
  tree: ast.Module, configurable_names: set[str]
) -> tuple[list[str], list[str], list[str]]:
  """Extract all public (non-configurable) items from source.

  Returns:
    Tuple of (class_stubs, function_stubs, constant_stubs)
  """
  class_stubs: list[str] = []
  function_stubs: list[str] = []
  constant_stubs: list[str] = []

  for node in tree.body:
    # Skip private items
    if isinstance(node, ast.ClassDef):
      if node.name.startswith("_") or node.name in configurable_names:
        continue
      # Generate stub for non-configurable class
      stub_body: list[ast.stmt] = []
      for item in node.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
          # Replace method body with ...
          stub_method = ast.FunctionDef(
            name=item.name,
            args=item.args,
            body=[ast.Expr(value=ast.Constant(value=...))],
            decorator_list=[],
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
          stub_body.append(item)
        elif isinstance(item, ast.Assign | ast.Pass):
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
        decorator_list=[],
        type_params=getattr(node, "type_params", []),
        lineno=node.lineno,
        col_offset=node.col_offset,
      )
      class_stubs.append(ast.unparse(stub_class))

    elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
      # Skip private and configurable functions
      if node.name.startswith("_") or node.name in configurable_names:
        continue
      # Generate stub for non-configurable function
      stub_func = ast.FunctionDef(
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
          constant_stubs.append(f"{name}: {type_str} = ...")
        else:
          constant_stubs.append(f"{name}: {type_str}")

    elif isinstance(node, ast.Assign):
      # Simple assignments (constants/type aliases)
      for target in node.targets:
        if isinstance(target, ast.Name) and not target.id.startswith("_"):
          # Type aliases like: MyType = list[int] - keep the value for type aliases
          # but use ... for regular constants
          value_str = ast.unparse(node.value)
          # Heuristic: if RHS looks like a type (contains brackets or is a name),
          # it's likely a type alias - keep the value
          if "[" in value_str or value_str[0].isupper():
            constant_stubs.append(f"{target.id} = {value_str}")
          else:
            # Regular constant - just declare the name
            constant_stubs.append(f"{target.id}: ...")
          break

    # Skip if __name__ == "__main__": blocks (they're ast.If nodes, handled implicitly by not matching)

  return class_stubs, function_stubs, constant_stubs


def _generate_configurable_stubs(infos: list[ConfigurableInfo]) -> list[str]:
  """Generate stubs for @configurable decorated items."""
  stubs: list[str] = []
  for info in infos:
    if info.is_class:
      stubs.append(_generate_class_stub(info))
    else:
      stubs.append(_generate_function_stub(info))
  return stubs


def _assemble_stub_content(
  import_lines: list[str],
  public_constants: list[str],
  public_funcs: list[str],
  public_classes: list[str],
  configurable_stubs: list[str],
) -> str:
  """Assemble the final stub file content."""
  lines: list[str] = [
    '"""Auto-generated type stubs for @configurable decorators.',
    "",
    "Do not edit manually - regenerate with: nonfig-stubgen <path>",
    '"""',
    "",
  ]

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
  other_imports = [imp for imp in other_imports if "from nonfig import" not in imp]

  # Track what we're already importing
  # Note: We alias MakeableModel to _NCMakeableModel, so we don't
  # add MakeableModel to already_imported. If the user imports it,
  # it's distinct from our internal base class alias.
  already_imported: set[str] = {"override"}
  if has_callable:
    already_imported.add("Callable")
  if has_default:
    already_imported.add("DEFAULT")

  filtered_imports = _filter_imports(other_imports, used_names, already_imported)

  # Future imports first
  if future_imports:
    lines.extend(future_imports)
    lines.append("")

  # Standard library imports
  std_imports: list[str] = []
  if has_callable:
    std_imports.append("from collections.abc import Callable")
  if "override" in used_names:
    std_imports.append("from typing import override")
  if "Any" in used_names:
    std_imports.append("from typing import Any")
  if "TypedDict" in used_names:
    std_imports.append("from typing import TypedDict")
  if "Protocol" in used_names:
    std_imports.append("from typing import Protocol")
  if "ClassVar" in used_names:
    std_imports.append("from typing import ClassVar")

  if std_imports:
    lines.extend(std_imports)
    lines.append("")

  # nonfig imports
  # Use alias to avoid collision with user's own MakeableModel
  lines.append("from nonfig import MakeableModel as _NCMakeableModel")

  nonfig_imports: list[str] = []
  if has_default:
    nonfig_imports.append("DEFAULT")

  if nonfig_imports:
    lines.append(f"from nonfig import {', '.join(nonfig_imports)}")

  # Other filtered imports
  if filtered_imports:
    lines.append("")
    lines.extend(filtered_imports)

  return lines


def generate_stub_content(infos: list[ConfigurableInfo], source_path: Path) -> str:
  """Generate complete stub file content."""
  source = source_path.read_text(encoding="utf-8")
  tree = ast.parse(source)

  configurable_names = {info.name for info in infos}
  public_classes, public_funcs, public_constants = _extract_public_items(
    tree, configurable_names
  )

  if not infos and not public_classes and not public_funcs and not public_constants:
    return ""

  configurable_stubs = _generate_configurable_stubs(infos)

  # Combine all stubs to collect used names
  all_stubs = "\n\n".join(
    configurable_stubs + public_classes + public_funcs + public_constants
  )
  used_names = _collect_used_names(all_stubs)
  used_names.update({"MakeableModel", "override"})

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
  infos = scan_module(path)
  if not infos:
    return None

  content = generate_stub_content(infos, path)
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
    if py_file.name.startswith("_"):
      continue
    stub_path = generate_stub_for_file(py_file)
    if stub_path:
      generated.append(stub_path)

  return generated
