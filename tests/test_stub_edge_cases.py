from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from nonfig.stubs.generator import generate_stub_content
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
