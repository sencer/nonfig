from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

from nonfig.stubs.generator import (
  _generate_non_configurable_class_stub,
  generate_stub_content,
)
from nonfig.stubs.scanner import scan_module


def test_generate_stub_for_empty_file(tmp_path: Path) -> None:
  """Test generating stubs for an empty file (should return empty string)."""
  p = tmp_path / "empty.py"
  p.write_text("")

  infos, aliases = scan_module(p)
  content = generate_stub_content(infos, p, aliases)
  assert not content


def test_generate_stub_for_file_with_no_public_symbols(tmp_path: Path) -> None:
  """Test file with only private symbols."""
  p = tmp_path / "private.py"
  p.write_text(
    dedent("""
        def _private(): pass
        class _Secret: pass
        _CONST = 1
    """)
  )

  infos, aliases = scan_module(p)
  content = generate_stub_content(infos, p, aliases)
  assert not content


def test_configurable_with_no_hyperparams(tmp_path: Path) -> None:
  """Test a @configurable function that has no Hyper parameters."""
  p = tmp_path / "no_hyper.py"
  p.write_text(
    dedent("""
        from nonfig import configurable

        @configurable
        def simple(x: int):
            pass
    """)
  )

  infos, aliases = scan_module(p)
  content = generate_stub_content(infos, p, aliases)

  assert "class _simple_Config" in content
  assert "pass" in content  # The config class body should be pass or have just init


def test_docstring_formatting_edge_cases(tmp_path: Path) -> None:
  """Test complex docstring formatting in stubs."""
  p = tmp_path / "docstrings.py"
  # Use concatenation to avoid nesting triple quotes which causes SyntaxError
  source = (
    "from nonfig import configurable, Hyper\n"
    "\n"
    "@configurable\n"
    "def complex_docs(x: Hyper[int] = 1):\n"
    '    """\n'
    "    This is a multi-line docstring.\n"
    "    \n"
    "    It has multiple paragraphs.\n"
    '    """\n'
    "    pass\n"
    "    \n"
    "@configurable\n"
    "def one_line(x: Hyper[int] = 1):\n"
    '    """One liner."""\n'
    "    pass\n"
  )
  p.write_text(source)

  infos, aliases = scan_module(p)
  content = generate_stub_content(infos, p, aliases)

  # The generator might wrap single line docstrings if they have extra sections
  assert "One liner." in content
  # The generator indents docstrings - functions use 4 spaces, classes use 8
  assert '    """' in content
  assert "    This is a multi-line docstring." in content


def test_import_filtering(tmp_path: Path) -> None:
  """Test that stubs filter unused imports and handle TYPE_CHECKING blocks."""
  p = tmp_path / "imports.py"
  p.write_text(
    dedent("""
        from __future__ import annotations
        from typing import TYPE_CHECKING
        from nonfig import configurable, Hyper

        if TYPE_CHECKING:
            from unused import Unused
            import unused_module
            from used_mod import UsedType

        import os  # Unused
        import sys # Used

        @configurable
        def func(param: Hyper[UsedType]):
            return sys.platform
    """)
  )

  # Note: scan_module doesn't extract types used in Hyper pointers if they aren't standard.
  # But the generator *reads* the file and parses imports.
  # The generator's _collect_used_names scans the *generated stub content*.
  # Wait, _collect_used_names scans the STUB content?
  # Yes, generator.py:77: _collect_used_names(stub_content: str)
  # But wait, how do we know what imports to KEEP if we scan the stub content *before* we put imports in?
  # Ah, `generate_stub_content` calls `_generate_configurable_stubs` first.
  # Then `_build_import_section` calls `_collect_used_names` on the *body* of the stub (classes/funcs).
  # So if the stub body uses `UsedType`, then `UsedType` should be kept in imports.

  infos, aliases = scan_module(p)
  content = generate_stub_content(infos, p, aliases)

  # __future__ should be kept
  assert "from __future__ import annotations" in content

  # TYPE_CHECKING block should be kept if it has used imports
  assert "if TYPE_CHECKING:" in content
  assert "from used_mod import UsedType" in content

  # Unused imports inside TYPE_CHECKING should be gone
  assert "from unused import Unused" not in content
  assert "import unused_module" not in content

  # Unused top-level imports should be gone
  assert "import os" not in content

  # Used top-level imports *might* be gone if they are used in BODY of function but not signature?
  # Stubs usually only contain signature. `sys.platform` is in body.
  # So `sys` should be REMOVED if it's not in the signature.
  assert "import sys" not in content


def test_stub_generation_dotted_names(tmp_path: Path) -> None:
  """Verify that dotted expressions in wrap_external are cleaned up for return types in stubs."""
  source_file = tmp_path / "complex_wrap.py"
  source_file.write_text(
    dedent("""
        from nonfig import wrap_external
        MyConfig = wrap_external(torch.optim.Adam)
    """),
    encoding="utf-8",
  )

  infos, aliases = scan_module(source_file)
  content = generate_stub_content(infos, source_file, aliases)

  # Should preserve the full dotted name for correctness in stubs
  assert "class MyConfig(_NCMakeableModel[torch.optim.Adam]):" in content


def test_stub_method_decorator_filtering():
  """
  Verify that external decorators are filtered out of method stubs.

  to prevent NameErrors in .pyi files.
  """
  source = """
class DataProcessor:
    @expensive_computation_tracker
    @property
    def data(self):
        return []

    @validate_call
    def process(self, x: int):
        pass
"""
  tree = ast.parse(source)
  class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
  stub = _generate_non_configurable_class_stub(class_node)

  assert "@expensive_computation_tracker" not in stub
  assert "@validate_call" not in stub
  assert "@property" in stub


def test_scanner_multiple_assignments(tmp_path: Path):
  """Verify that the scanner detects wrap_external calls in multiple assignments."""
  source_file = tmp_path / "multi_wrap.py"
  source_file.write_text(
    """
from nonfig import wrap_external
ConfigA, ConfigB = wrap_external(int), wrap_external(float)
""",
    encoding="utf-8",
  )

  infos, _ = scan_module(source_file)
  names = {info.name for info in infos}
  assert "ConfigA" in names
  assert "ConfigB" in names
  assert len(infos) == 2


def test_preserve_essential_class_decorators(tmp_path: Path):
  source = dedent("""
        from dataclasses import dataclass
        from typing import final, runtime_checkable, Protocol

        @dataclass
        class MyData:
            x: int

        @final
        class Locked:
            pass

        @runtime_checkable
        class MyProto(Protocol):
            def execute(self) -> None: ...

        @custom_decorator
        class Custom:
            pass
    """)

  source_file = tmp_path / "deco_test.py"
  source_file.write_text(source, encoding="utf-8")

  # We scan with empty infos because these are non-configurable classes
  # scan_module returns (infos, aliases)
  infos, aliases = scan_module(source_file)
  content = generate_stub_content(infos, source_file, aliases)

  print("\nGenerated Stub Content:")
  print(content)

  # Check preserved
  assert "@dataclass" in content
  assert "@final" in content
  assert "@runtime_checkable" in content

  # Check stripped
  assert "@custom_decorator" not in content

  # Check imports
  assert "from dataclasses import dataclass" in content
  assert "from typing import final" in content or "from typing import" in content
  assert "runtime_checkable" in content


def test_stub_generator_import_handling_for_decorators(tmp_path: Path):
  """Verify that using decorators adds necessary imports to the stub."""
  source = dedent("""
        from typing import final
        @final
        class Secret:
            pass
    """)
  source_file = tmp_path / "secret.py"
  source_file.write_text(source, encoding="utf-8")

  infos, aliases = scan_module(source_file)
  content = generate_stub_content(infos, source_file, aliases)

  assert "from typing import final" in content or "from typing import" in content


def test_stub_dataclass_import_preservation(tmp_path: Path):
  """
  Issue 3: Verify that dataclass import is correctly added to stubs.

  when non-configurable dataclasses are present.
  """
  source_file = tmp_path / "data.py"
  source_file.write_text(
    dedent("""
            from dataclasses import dataclass
            @dataclass
            class Point:
                x: int
                y: int
        """),
    encoding="utf-8",
  )

  infos, _ = scan_module(source_file)
  content = generate_stub_content(infos, source_file, _)

  # Check that dataclass is imported
  assert "from dataclasses import dataclass" in content
  assert "@dataclass" in content


def test_stub_generator_custom_wrap_alias(tmp_path: Path):
  """
  Verify that custom aliases for wrap_external are correctly handled.

  and don't leak into public constants.
  """
  source_file = tmp_path / "custom_alias.py"
  source_file.write_text(
    dedent("""
            from nonfig import wrap_external as wrap
            MyConfig = wrap(int)
        """),
    encoding="utf-8",
  )

  infos, aliases = scan_module(source_file)
  assert any(info.name == "MyConfig" for info in infos)

  content = generate_stub_content(infos, source_file, aliases)

  # MyConfig should be in configurable stubs, not constants
  assert "class MyConfig" in content
  # It should NOT be in constants as 'MyConfig: ...' or similar
  assert content.count("class MyConfig") == 1
