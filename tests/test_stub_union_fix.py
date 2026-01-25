from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from nonfig.stubs import generate_stub_for_file


def test_stub_union_with_hyper(tmp_path: Path) -> None:
  """Regression test: Ensure Hyper[...] inside Union or Optional is detected as configurable."""
  source_file = tmp_path / "union_hyper_reg.py"
  source_file.write_text(
    dedent("""
            from typing import Union, Optional
            from nonfig import configurable, Hyper

            @configurable
            def my_func(
                a: Hyper[int] | None = None,
                b: Optional[Hyper[str]] = None,
                c: Union[Hyper[float], int] = 1.0
            ) -> None:
                pass
        """),
    encoding="utf-8",
  )

  stub_path = generate_stub_for_file(source_file)
  assert stub_path is not None

  content = stub_path.read_text(encoding="utf-8")

  # Verify that params are recognized as configurable (present in ConfigDict/Config)
  assert "a: int | None" in content
  assert "b: Optional[str]" in content
  assert "c: Union[float, int]" in content

  # Verify they are in the __init__ of Config
  assert "a: int | None =" in content
  assert "b: Optional[str] =" in content
  assert "c: Union[float, int] =" in content
