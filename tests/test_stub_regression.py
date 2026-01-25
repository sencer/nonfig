from pathlib import Path
import textwrap

from nonfig.stubs import generate_stub_for_file


def test_stub_regression_quoted_type_checking(tmp_path: Path) -> None:
  """Regression test: Ensure quoted types inside TYPE_CHECKING blocks are preserved."""
  source_file = tmp_path / "quoted_regression.py"
  source_file.write_text(
    textwrap.dedent("""
            from typing import TYPE_CHECKING
            from nonfig import configurable

            if TYPE_CHECKING:
                from decimal import Decimal as MyResult

            @configurable
            def my_func(x: int) -> "MyResult":
                return 0  # type: ignore
        """),
    encoding="utf-8",
  )

  stub_path = generate_stub_for_file(source_file)
  assert stub_path is not None

  content = stub_path.read_text(encoding="utf-8")

  # 1. Verify the TYPE_CHECKING block is preserved
  assert "if TYPE_CHECKING:" in content
  assert "from decimal import Decimal as MyResult" in content

  # 2. Verify the return type is unquoted in the stub (standard practice)
  assert "-> MyResult:" in content or "-> 'MyResult':" in content

  # 3. Verify it appears in the Bound protocol
  assert (
    "def __call__(self, x: int) -> MyResult:" in content
    or "def __call__(self, x: int) -> 'MyResult':" in content
  )


def test_stub_regression_nested_quoted_type(tmp_path: Path) -> None:
  """Regression test: Ensure quoted types nested in containers are preserved."""
  source_file = tmp_path / "nested_quoted.py"
  source_file.write_text(
    textwrap.dedent("""
            from typing import TYPE_CHECKING, List
            from nonfig import configurable

            if TYPE_CHECKING:
                from decimal import Decimal as MyResult

            @configurable
            def my_func(x: int) -> list["MyResult"]:
                return []
        """),
    encoding="utf-8",
  )

  stub_path = generate_stub_for_file(source_file)
  assert stub_path is not None

  content = stub_path.read_text(encoding="utf-8")

  # Verify import is kept
  assert "from decimal import Decimal as MyResult" in content

  # Verify usage
  assert (
    "list['MyResult']" in content
    or 'list["MyResult"]' in content
    or "list[MyResult]" in content
  )


def test_stub_regression_string_literal_constant(tmp_path: Path) -> None:
  """Regression test: Ensure string literals that look like types are not accidentally pruned if they are just values."""
  # This tests the safety of our 'recursive string scanning'.
  # If we have a string "int", it shouldn't cause weird behavior,
  # but more importantly, if we have a type alias defined as a string, it should work.
  source_file = tmp_path / "string_alias.py"
  source_file.write_text(
    textwrap.dedent("""
            from nonfig import configurable

            # This is a type alias defined as a string (forward ref style)
            MyType = "int"

            @configurable
            def my_func(x: MyType) -> None:
                pass
        """),
    encoding="utf-8",
  )

  stub_path = generate_stub_for_file(source_file)
  assert stub_path is not None
  content = stub_path.read_text(encoding="utf-8")

  # The alias should be preserved
  assert 'MyType = "int"' in content or "MyType = 'int'" in content
