from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from nonfig.stubs.generator import generate_stub_content
from nonfig.stubs.scanner import scan_module


def test_generate_stub_for_empty_file(tmp_path: Path) -> None:
  """Test generating stubs for an empty file (should return empty string)."""
  p = tmp_path / "empty.py"
  p.write_text("")

  infos = scan_module(p)
  content = generate_stub_content(infos, p)
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

  infos = scan_module(p)
  content = generate_stub_content(infos, p)
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

  infos = scan_module(p)
  content = generate_stub_content(infos, p)

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

  infos = scan_module(p)
  content = generate_stub_content(infos, p)

  # The generator might wrap single line docstrings if they have extra sections
  assert "One liner." in content
  # The generator indents docstrings - functions use 4 spaces, classes use 8
  assert '    """' in content
  assert "    This is a multi-line docstring." in content
