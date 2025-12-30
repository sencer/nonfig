from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from nonfig.stubs.generator import generate_stub_content
from nonfig.stubs.scanner import scan_module


def test_stub_async_function_preservation(tmp_path: Path) -> None:
  """Verify that async keyword is preserved in stubs for non-configurable functions."""
  source_file = tmp_path / "async_test.py"
  source_file.write_text(
    dedent("""
        async def my_async_func(x: int) -> int:
            return x
    """),
    encoding="utf-8",
  )

  infos, aliases = scan_module(source_file)
  content = generate_stub_content(infos, source_file, aliases)

  assert "async def my_async_func" in content


def test_stub_dataclass_init_false_exclusion(tmp_path: Path) -> None:
  """Verify that init=False fields are excluded from Config class in stubs."""
  source_file = tmp_path / "dataclass_test.py"
  source_file.write_text(
    dedent("""
        from dataclasses import dataclass, field
        from nonfig import configurable, Hyper

        @configurable
        @dataclass
        class MyDataclass:
            included: Hyper[int] = 10
            excluded: int = field(default=5, init=False)
    """),
    encoding="utf-8",
  )

  infos, aliases = scan_module(source_file)
  content = generate_stub_content(infos, source_file, aliases)

  # 'included' should be in Config
  assert "included: int = ..." in content
  # 'excluded' should NOT be in Config (it will be in the original class stub though)

  # Check Config class body
  config_lines = [
    line
    for line in content.splitlines()
    if "class Config" in line or (line.startswith("        ") and ":" in line)
  ]
  assert any("included: int" in line for line in config_lines)
  assert not any("excluded: int" in line for line in config_lines)


def test_stub_optional_union_imports(tmp_path: Path) -> None:
  """Verify that Optional and Union are imported when used in stubs."""
  source_file = tmp_path / "typing_test.py"
  source_file.write_text(
    dedent("""
        from typing import Optional, Union
        from nonfig import configurable, Hyper

        @configurable
        def my_func(
            x: Hyper[Optional[int]] = None,
            y: Hyper[Union[int, str]] = 1
        ):
            pass
    """),
    encoding="utf-8",
  )

  infos, aliases = scan_module(source_file)
  content = generate_stub_content(infos, source_file, aliases)

  assert "from typing import" in content
  assert "Optional" in content
  assert "Union" in content
