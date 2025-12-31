from __future__ import annotations

from pathlib import Path

from nonfig.stubs.generator import (
  _filter_type_checking_block,
  _format_docstring,
  _should_transform_to_config,
  _transform_to_config_type,
)


def test_format_docstring_variants():
  """Test various docstring formatting scenarios."""
  # Empty
  assert not _format_docstring(None)
  assert not _format_docstring("")

  # Single line
  assert _format_docstring("Hello") == '    """Hello"""\n'

  # Multi-line
  doc = "Line 1\nLine 2"
  formatted = _format_docstring(doc)
  assert '"""Line 1' in formatted
  assert "    Line 2" in formatted
  assert '    """' in formatted

  # With extra sections
  formatted = _format_docstring("Desc", ["Extra"], indent=0)
  assert '"""Desc' in formatted
  assert "Extra" in formatted


def test_filter_type_checking_mixed_content():
  """Test filtering TYPE_CHECKING block with mixed content."""
  block = "if TYPE_CHECKING:\n    x = 1\n    from foo import bar"
  # Used names doesn't include bar, so import should be removed
  # x = 1 is not an import, so loop in _filter_type_checking_block ignores it?
  # Actually checking code: it iterates body. If Import/ImportFrom, it filters.
  # Other nodes are ignored (dropped from new_body).

  # Case 1: Import 'bar' is used
  filtered = _filter_type_checking_block(block, {"bar"}, set())
  # The assignment 'x=1' is dropped because the function only reconstructs imports
  assert "from foo import bar" in filtered
  assert "x = 1" not in filtered


def test_transform_to_config_leaf():
  """Test _transform_to_config_type with is_leaf=True."""
  assert _transform_to_config_type("MyClass", is_leaf=True) == "MyClass"
  assert (
    _transform_to_config_type("MyClass", is_leaf=False)
    == "MyClass.Config | MyClass.ConfigDict"
  )


def test_should_transform_primitive_containers():
  """Test _should_transform_to_config for containers."""
  assert not _should_transform_to_config("list[int]", set())
  assert not _should_transform_to_config("dict[str, Any]", set())
  assert not _should_transform_to_config("Optional[MyClass]", set())
  # Custom type
  assert _should_transform_to_config("MyClass", set())


def test_stub_docstring_preservation():
  """Issue: Docstrings of methods in non-configurable classes are stripped in stubs."""
  import ast
  from textwrap import dedent

  from nonfig.stubs.generator import _generate_non_configurable_class_stub

  source = dedent("""
        class Processor:
            def process(self, x: int) -> int:
                \"\"\"Process the input.\"\"\"
                return x + 1
    """)
  tree = ast.parse(source)
  class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
  stub = _generate_non_configurable_class_stub(class_node)

  assert '"""Process the input."""' in stub


def test_scanner_unpacking_assignment_crash(tmp_path: Path):
  """
  Issue 1: Scanner crashes on unpacking assignments like A, B = func().

  due to zip(..., strict=True) mismatch.
  """
  from nonfig.stubs.scanner import scan_module

  source_file = tmp_path / "unpacking.py"
  source_file.write_text(
    "from nonfig import wrap_external\nA, B = some_function()\n", encoding="utf-8"
  )

  # This currently crashes with ValueError: zip() argument 2 is shorter than argument 1
  # We want it to NOT crash.
  infos, _ = scan_module(source_file)
  assert len(infos) == 0
