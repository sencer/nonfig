from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

import pytest

from nonfig.cli.generate_stubs import main
from nonfig.constraints import validate_constraint_conflicts
from nonfig.stubs import (
  ConfigurableInfo,
  HyperParam,
  generate_stub_for_file,
  scan_module,
)
from nonfig.stubs.generator import (
  _collect_used_names,
  _filter_imports,
  _filter_type_checking_block,
  _generate_class_stub,
  _generate_config_class,
  _generate_function_stub,
  generate_stub_content,
)
from nonfig.stubs.scanner import (
  _extract_default,
  _get_annotation_str,
  _is_configurable_callable_default,
  _is_configurable_decorator,
  _is_hyper_annotation,
  _scan_class,
  _scan_function,
  _unwrap_hyper,
)

"""Consolidated stub tests."""

"""Tests for stub generation."""


@pytest.fixture
def temp_source_file(tmp_path: Path) -> Path:
  source_code = dedent("""
        from nonfig import configurable, Hyper, Ge

        @configurable
        def my_func(
            data: list[float],
            period: Hyper[int, Ge[1]] = 14,
            alpha: float = 0.5,
        ) -> list[float]:
            return [x * alpha for x in data]

        @configurable
        class MyIndicator:
            def __init__(self, window: Hyper[int] = 10):
                self.window = window

            def __call__(self, data: list[float]) -> list[float]:
                return data
    """)
  file_path = tmp_path / "test_source.py"
  file_path.write_text(source_code, encoding="utf-8")
  return file_path


def test_is_configurable_decorated():
  code = "@configurable\ndef foo(): pass"
  tree = ast.parse(code)
  func_def = tree.body[0]
  assert isinstance(func_def, ast.FunctionDef)
  assert _is_configurable_decorator(func_def.decorator_list[0])

  code = "def bar(): pass"
  tree = ast.parse(code)
  func_def = tree.body[0]
  assert isinstance(func_def, ast.FunctionDef)
  assert len(func_def.decorator_list) == 0


def test_is_hyper_annotation():
  code = "def f(x: Hyper[int]): pass"
  tree = ast.parse(code)
  func = tree.body[0]
  assert isinstance(func, ast.FunctionDef)
  arg = func.args.args[0]
  assert _is_hyper_annotation(arg.annotation)


def test_unwrap_hyper():
  code = "def f(x: Hyper[int, Ge[0], Le[10]]): pass"
  tree = ast.parse(code)
  func = tree.body[0]
  assert isinstance(func, ast.FunctionDef)
  arg = func.args.args[0]
  inner = _unwrap_hyper(arg.annotation)
  assert inner == "int"


def test_scan_module(temp_source_file: Path):
  infos = scan_module(temp_source_file)
  # Should find both the function and the class
  assert len(infos) == 2

  # Check the function
  func_info = infos[0]
  assert func_info.name == "my_func"
  assert len(func_info.params) == 1
  assert func_info.params[0].name == "period"
  assert len(func_info.call_params) == 2  # data and alpha

  # Check the class
  class_info = infos[1]
  assert class_info.name == "MyIndicator"
  assert len(class_info.params) == 1
  assert class_info.params[0].name == "window"
  assert class_info.params[0].type_annotation == "int"
  assert class_info.params[0].default_value == "10"


def test_generate_config_class():
  info = ConfigurableInfo(
    name="my_func",
    is_class=False,
    params=[HyperParam("p1", "int", "1")],
    call_params=[("data", "list", None)],
    return_type="list[float]",
  )
  code = _generate_config_class(info)
  assert "class Config(_NCMakeableModel" in code
  assert "p1: int" in code
  assert "def __init__(self, *, p1: int = ...) -> None: ..." in code


def test_generate_class_stub():
  info = ConfigurableInfo(
    name="MyClass",
    is_class=True,
    params=[HyperParam("window", "int", "10")],
    return_type="MyClass",
  )
  code = _generate_class_stub(info)
  assert "class MyClass:" in code
  assert "class Config(_NCMakeableModel[MyClass]):" in code
  assert "window: int" in code


def test_generate_function_stub():
  info = ConfigurableInfo(
    name="my_func",
    is_class=False,
    params=[HyperParam("window", "int", "10")],
    call_params=[("data", "list", None)],
    return_type="float",
  )
  code = _generate_function_stub(info)
  assert "class _my_func_Bound(Protocol):" in code
  assert "class _my_func_Config(_NCMakeableModel[_my_func_Bound]):" in code
  assert "class my_func:" in code
  assert "def __new__(cls, data: list, window: int = ...) -> float: ..." in code


def test_generate_function_stub_properties_readonly():
  """Test that hyperparameters in bound function stub are read-only properties."""
  info = ConfigurableInfo(
    name="my_func",
    is_class=False,
    params=[HyperParam("window", "int", "10")],
    call_params=[("data", "list", None)],
    return_type="float",
  )
  code = _generate_function_stub(info)

  # Should use @property and def, not simple annotation
  assert "@property" in code
  assert "def window(self) -> int: ..." in code
  assert (
    "window: int"
    not in code.split("class _my_func_Bound(Protocol):")[1].split("def __call__")[0]
  )


def test_generate_stub_content(temp_source_file: Path):
  infos = scan_module(temp_source_file)
  content = generate_stub_content(infos, temp_source_file)
  assert (
    "class UsingConfig(_NCMakeableModel" not in content
  )  # Config classes generated with _Config prefix
  assert "from nonfig import MakeableModel as _NCMakeableModel" in content


def test_validate_constraint_conflicts_catches_contradictions():
  """Test that validation catches contradictory numeric bounds."""
  with pytest.raises(ValueError, match="Conflicting constraints"):
    validate_constraint_conflicts({"ge": 100, "le": 50}, "x")


def test_validate_constraint_conflicts_catches_strict_bounds():
  """Test that validation catches impossible strict/non-strict bounds."""
  with pytest.raises(ValueError, match="exclusive"):
    validate_constraint_conflicts({"ge": 50, "lt": 50}, "x")


def test_validate_constraint_conflicts_catches_length_issues():
  """Test that validation catches MinLen > MaxLen."""
  with pytest.raises(ValueError, match="Conflicting constraints"):
    validate_constraint_conflicts({"min_length": 10, "max_length": 5}, "text")


def test_validate_constraint_conflicts_allows_valid_constraints():
  """Test that validation allows valid constraints."""
  # These should not raise
  validate_constraint_conflicts({"ge": 10, "le": 100}, "x")
  validate_constraint_conflicts({"gt": 0, "lt": 100}, "x")
  validate_constraint_conflicts({"min_length": 5, "max_length": 50}, "text")


def test_scan_module_with_dataclass(tmp_path: Path):
  """Test that scan_module properly handles @dataclass decorated classes."""
  dataclass_file = tmp_path / "dataclass_test.py"
  dataclass_file.write_text(
    dedent("""
        from dataclasses import dataclass
        from nonfig import configurable, Hyper, Ge

        @configurable
        @dataclass
        class DataConfig:
            learning_rate: Hyper[float, Ge[0.0]] = 0.01
            batch_size: Hyper[int, Ge[1]] = 32
            epochs: Hyper[int, Ge[1]] = 10
    """)
  )

  infos = scan_module(dataclass_file)
  assert len(infos) == 1

  class_info = infos[0]
  assert class_info.name == "DataConfig"
  assert len(class_info.params) == 3

  # Check learning_rate param
  assert class_info.params[0].name == "learning_rate"
  assert class_info.params[0].type_annotation == "float"
  assert class_info.params[0].default_value == "0.01"


def test_generate_stub_for_file(tmp_path: Path):
  """Test generate_stub_for_file creates stub."""
  source_file = tmp_path / "source.py"
  source_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper

        @configurable
        def my_func(x: Hyper[int] = 10) -> int:
            return x
    """)
  )

  result = generate_stub_for_file(source_file)
  assert result is not None

  stub_file = tmp_path / "source.pyi"
  assert stub_file.exists()
  stub_content = stub_file.read_text()
  assert "class _my_func_Config(_NCMakeableModel" in stub_content


def test_generate_stub_for_file_without_configurable(tmp_path: Path):
  """Test generate_stub_for_file returns None for file without @configurable."""
  source_file = tmp_path / "plain.py"
  source_file.write_text(
    dedent("""
        def regular_func(x: int) -> int:
            return x
    """)
  )

  result = generate_stub_for_file(source_file)
  assert result is None

  stub_file = tmp_path / "plain.pyi"
  assert not stub_file.exists()


def test_main_with_directory(tmp_path: Path):
  """Test main() with directory."""
  test_dir = tmp_path / "custom"
  test_dir.mkdir()

  test_file = test_dir / "module.py"
  test_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper

        @configurable
        def func(x: Hyper[int] = 5) -> int:
            return x
    """)
  )

  exit_code = main([str(test_dir)])
  assert exit_code == 0

  stub_file = test_dir / "module.pyi"
  assert stub_file.exists()


def test_main_nonexistent_directory(tmp_path: Path):
  """Test main() with nonexistent pattern."""
  nonexistent = tmp_path / "does_not_exist"

  exit_code = main([str(nonexistent)])
  assert exit_code == 1  # No files found


def test_main_single_file(tmp_path: Path):
  """Test main() with a single Python file."""
  test_file = tmp_path / "single_module.py"
  test_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper

        @configurable
        def single_func(x: Hyper[int] = 10) -> int:
            return x
    """)
  )

  exit_code = main([str(test_file)])
  assert exit_code == 0

  stub_file = tmp_path / "single_module.pyi"
  assert stub_file.exists()


def test_main_handles_syntax_errors(tmp_path: Path):
  """Test main() handles files with syntax errors gracefully."""
  test_dir = tmp_path / "src"
  test_dir.mkdir()

  bad_file = test_dir / "bad_syntax.py"
  bad_file.write_text("def broken( # invalid syntax")

  # Should not crash, just skip the file
  exit_code = main([str(test_dir)])
  assert exit_code == 0


def test_main_skips_test_files(tmp_path: Path):
  """Test main() skips test_*.py files."""
  test_dir = tmp_path / "src"
  test_dir.mkdir()

  test_file = test_dir / "test_something.py"
  test_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper

        @configurable
        def test_func(x: Hyper[int] = 1) -> int:
            return x
    """)
  )

  exit_code = main([str(test_dir)])
  assert exit_code == 0

  # No stub should be created for test files
  stub_file = test_dir / "test_something.pyi"
  assert not stub_file.exists()


def test_extract_default():
  """Test _extract_default."""
  assert _extract_default(None) is None

  code = "10"
  tree = ast.parse(code, mode="eval")
  assert _extract_default(tree.body) == "10"


def test_get_annotation_str():
  """Test _get_annotation_str."""
  assert _get_annotation_str(None) == "Any"

  code = "int"
  tree = ast.parse(code, mode="eval")
  assert _get_annotation_str(tree.body) == "int"


def test_scan_function():
  """Test _scan_function."""
  code = dedent("""
        @configurable
        def my_func(x: Hyper[int] = 1) -> int:
            return x
    """)
  tree = ast.parse(code)
  func = tree.body[0]
  assert isinstance(func, ast.FunctionDef)

  info = _scan_function(func)
  assert info is not None
  assert info.name == "my_func"
  assert len(info.params) == 1
  assert info.params[0].name == "x"


def test_scan_class():
  """Test _scan_class."""
  code = dedent("""
        @configurable
        class MyClass:
            def __init__(self, x: Hyper[int] = 1):
                self.x = x
    """)
  tree = ast.parse(code)
  cls = tree.body[0]
  assert isinstance(cls, ast.ClassDef)

  info = _scan_class(cls)
  assert info is not None
  assert info.name == "MyClass"
  assert info.is_class is True
  assert len(info.params) == 1


def test_scan_class_with_dataclass_fields():
  """Test _scan_class with dataclass-style fields."""
  code = dedent("""
        @configurable
        @dataclass
        class MyClass:
            x: int = 1
            y: float = 0.5
    """)
  tree = ast.parse(code)
  cls = tree.body[0]
  assert isinstance(cls, ast.ClassDef)

  info = _scan_class(cls)
  assert info is not None
  assert len(info.params) == 2
  assert info.params[0].name == "x"
  assert info.params[1].name == "y"


def test_default_sentinel_in_stub_generation(tmp_path: Path):
  """Test that DEFAULT sentinel is properly handled in stub generation."""
  source_file = tmp_path / "with_default.py"
  source_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper, DEFAULT

        @configurable
        class InnerConfig:
            def __init__(self, param: Hyper[int] = 10):
                self.param = param

        @configurable
        def outer_func(
            data: list[float],
            inner: Hyper[InnerConfig.Config] = DEFAULT,
        ) -> float:
            return sum(data)
    """)
  )

  # Generate stub content
  infos = scan_module(source_file)
  stub_content = generate_stub_content(infos, source_file)

  # Verify DEFAULT is imported
  assert "DEFAULT" in stub_content

  # Verify the default value is preserved in the stub
  assert "inner: InnerConfig.Config = DEFAULT" in stub_content


def test_scan_module_rejects_contradictory_constraints(tmp_path: Path):
  """Test that scan_module raises error for contradictory constraints."""
  bad_file = tmp_path / "bad.py"
  bad_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper, Ge, Le

        @configurable
        def bad_func(x: Hyper[int, Ge[100], Le[50]] = 75) -> int:
            return x
    """)
  )

  with pytest.raises(ValueError, match="[Cc]onflict"):
    scan_module(bad_file)


def test_scan_module_accepts_valid_constraints(tmp_path: Path):
  """Test that scan_module works with valid constraints."""
  good_file = tmp_path / "good.py"
  good_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper, Ge, Le

        @configurable
        def good_func(x: Hyper[int, Ge[10], Le[100]] = 50) -> int:
            return x
    """)
  )

  functions = scan_module(good_file)
  assert len(functions) == 1
  assert functions[0].name == "good_func"


def test_scan_module_with_methods(tmp_path: Path):
  """Test that scan_module properly handles instance, class, and static methods."""
  methods_file = tmp_path / "methods_test.py"
  methods_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper, Ge

        class MyClass:
            @configurable
            def instance_method(
                self, data: list[float], window: Hyper[int, Ge[1]] = 10
            ) -> float:
                return sum(data) / window

            @classmethod
            @configurable
            def class_method(
                cls, data: list[float], alpha: Hyper[float, Ge[0.0]] = 0.5
            ) -> float:
                return sum(data) * alpha

            @staticmethod
            @configurable
            def static_method(
                data: list[float], beta: Hyper[float] = 1.0
            ) -> float:
                return sum(data) + beta
    """)
  )

  functions = scan_module(methods_file)
  # Should find all 3 methods
  assert len(functions) == 3
  names = {f.name for f in functions}
  assert "instance_method" in names
  assert "class_method" in names
  assert "static_method" in names


def test_main_verbose_mode(tmp_path: Path):
  """Test main() with verbose flag."""
  test_dir = tmp_path / "src"
  test_dir.mkdir()

  test_file = test_dir / "verbose_test.py"
  test_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper

        @configurable
        def verbose_func(x: Hyper[int] = 1) -> int:
            return x
    """)
  )

  # Run with verbose flag
  exit_code = main([str(test_dir), "--verbose"])
  assert exit_code == 0

  stub_file = test_dir / "verbose_test.pyi"
  assert stub_file.exists()


def test_nested_config_type_transformation(tmp_path: Path):
  """Test that nested configurable types are transformed to .Config in Config class."""
  source_file = tmp_path / "nested_configs.py"
  source_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper, DEFAULT

        @configurable
        class InnerOptimizer:
            learning_rate: Hyper[float] = 0.01

        @configurable
        class OuterModel:
            hidden_size: Hyper[int] = 128
            optimizer: Hyper[InnerOptimizer] = DEFAULT

        @configurable
        def train(
            data: list[float],
            epochs: Hyper[int] = 10,
            model: Hyper[OuterModel] = DEFAULT,
        ) -> float:
            return sum(data)
    """)
  )

  infos = scan_module(source_file)
  stub_content = generate_stub_content(infos, source_file)

  # The Config class's __init__ should use .Config types for nested configurables
  # This is critical for type-correct nested config instantiation

  # For OuterModel.Config, the optimizer param should be InnerOptimizer.Config
  assert "optimizer: InnerOptimizer.Config" in stub_content
  assert (
    "optimizer: InnerOptimizer.Config | InnerOptimizer.ConfigDict = DEFAULT"
    in stub_content
  )

  # For train.Config, the model param should be OuterModel.Config
  assert "model: OuterModel.Config" in stub_content
  assert "model: OuterModel.Config | OuterModel.ConfigDict = DEFAULT" in stub_content
  assert "class ConfigDict(TypedDict, total=False):" in stub_content

  # Primitive types should NOT be transformed
  assert "hidden_size: int" in stub_content
  assert "epochs: int" in stub_content
  assert "learning_rate: float" in stub_content


def test_required_nested_config_type_transformation(tmp_path: Path):
  """Test that REQUIRED nested configurables (no DEFAULT) are also transformed."""
  source_file = tmp_path / "required_nested.py"
  source_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper

        @configurable
        class Optimizer:
            lr: Hyper[float] = 0.01

        @configurable
        class Model:
            # Required nested config - no DEFAULT
            optimizer: Hyper[Optimizer]
            hidden_size: Hyper[int] = 128
    """)
  )

  infos = scan_module(source_file)
  stub_content = generate_stub_content(infos, source_file)

  # Even without DEFAULT, non-primitive Hyper types should become .Config
  # because you always pass Config objects to Config.__init__
  assert "optimizer: Optimizer.Config | Optimizer.ConfigDict" in stub_content

  # Primitive types unchanged
  assert "hidden_size: int" in stub_content
  assert "lr: float" in stub_content


def test_stub_class_var_for_wrapper(tmp_path: Path):
  """Regression test: Ensure wrapper class attributes use ClassVar for defaults."""
  source_file = tmp_path / "wrapper_defaults.py"
  source_file.write_text(
    dedent("""
        from nonfig import configurable, Hyper
        
        @configurable
        def my_func(x: Hyper[int] = 10):
            pass
    """)
  )

  infos = scan_module(source_file)
  content = generate_stub_content(infos, source_file)

  # Must import ClassVar
  assert "from typing import ClassVar" in content
  # Must use ClassVar for the attribute on the wrapper class
  assert "x: ClassVar[int]" in content


"""Tests for stub generation and scanning edge cases."""


def test_collect_names_syntax_error():
  """Test resilience against syntax errors in collect_used_names."""
  # Should return empty set for invalid python
  assert _collect_used_names("def broken_syntax(") == set()


def test_filter_type_checking_syntax_error():
  """Test resilience against syntax errors in filter_type_checking_block."""
  # Should return original string if parsing fails
  bad_source = "if TYPE_CHECKING: def foo("
  assert _filter_type_checking_block(bad_source, set(), set()) == bad_source


def test_filter_type_checking_not_if():
  """Test filter returns original if not an If node (unlikely but robust)."""
  source = "import os"
  # Even if we pass something that isn't 'if ...', if it parses,
  # the function checks `isinstance(if_node, ast.If)`.
  assert _filter_type_checking_block(source, set(), set()) == source


def test_filter_imports_syntax_error():
  """Test filter_imports handles syntax errors gracefully."""
  # Should keep the line if it can't parse it
  imports = ["import valid", "this is invalid python"]
  filtered = _filter_imports(imports, {"valid"}, set())
  assert "import valid" in filtered
  assert "this is invalid python" in filtered


def test_is_hyper_annotation_edge_cases():
  """Test _is_hyper_annotation with unexpected AST nodes."""
  # Simple Name not Hyper
  assert not _is_hyper_annotation(ast.Name(id="int"))

  # Subscript but not Hyper (e.g. List[int])
  node = ast.parse("List[int]").body[0].value
  assert not _is_hyper_annotation(node)  # type: ignore

  # Annotated but not Hyper
  node = ast.parse("Annotated[int, 'tag']").body[0].value
  assert not _is_hyper_annotation(node)  # type: ignore


def test_unwrap_hyper_annotated_edge_cases():
  """Test _unwrap_hyper with Annotated/Hyper variations."""
  # Hyper[T]
  node = ast.parse("Hyper[int]").body[0].value
  assert _unwrap_hyper(node) == "int"  # type: ignore

  # Annotated[T, Hyper]
  # Note: scanner logic checks for Hyper in Annotated args
  node = ast.parse("Annotated[int, Hyper]").body[0].value
  assert _unwrap_hyper(node) == "int"  # type: ignore

  # Hyper[T, Ge[1]]
  node = ast.parse("Hyper[int, Ge[1]]").body[0].value
  assert _unwrap_hyper(node) == "int"  # type: ignore

  # Fallback
  node = ast.parse("int").body[0].value
  assert _unwrap_hyper(node) == "int"  # type: ignore


def test_extract_default_none():
  assert _extract_default(None) is None


def test_is_configurable_callable_default_edge_cases():
  # Both None
  assert not _is_configurable_callable_default(None, None)

  # Not a name default
  default = ast.parse("1").body[0].value
  ann = ast.Name(id="Any")
  assert not _is_configurable_callable_default(default, ann)  # type: ignore


def test_generate_stub_content_copies_imports(tmp_path: Path):
  """Test that imports are copied from source to stub."""
  source_file = tmp_path / "imports_test.py"
  source_file.write_text(
    dedent(
      """
        import numpy as np
        from typing import List, Optional
        from my_module import MyType
        from nonfig import configurable, Hyper

        @configurable
        def my_func(
            x: Hyper[int] = 10,
            y: Optional[List[MyType]] = None
        ) -> np.ndarray:
            return np.array([x])
    """
    )
  )

  functions = scan_module(source_file)
  content = generate_stub_content(functions, source_file)

  # Check that imports are present
  assert "import numpy as np" in content
  assert "from typing import List, Optional" in content
  assert "from my_module import MyType" in content

  # Check that nonfig imports are handled correctly
  assert "from nonfig import MakeableModel" in content


"""Tests for stub generation improvements.

Tests for:
- @override decorator on make() methods
- Import filtering (only used imports)
- TYPE_CHECKING block handling
- Public items inclusion (classes, functions, constants)
- Config field skipping
- Constants and type alias formatting
"""


class TestOverrideDecorator:
  """Tests for @override decorator on make() methods."""

  def test_class_config_has_override(self, tmp_path: Path) -> None:
    """Test that class Config.make() has @override decorator."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        @configurable
        class MyClass:
            def __init__(self, x: Hyper[int] = 10):
                self.x = x
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    assert "@override" in content
    assert "from typing import override" in content
    # Check override appears before make
    lines = content.split("\n")
    for i, line in enumerate(lines):
      if "def make(self)" in line and i > 0:
        assert "@override" in lines[i - 1]

  def test_function_config_has_override(self, tmp_path: Path) -> None:
    """Test that function Config.make() has @override decorator."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        @configurable
        def my_func(data: list, x: Hyper[int] = 10) -> int:
            return x
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    assert "@override" in content
    # Check override appears before make
    lines = content.split("\n")
    for i, line in enumerate(lines):
      if "def make(self)" in line and i > 0:
        assert "@override" in lines[i - 1]


class TestImportFiltering:
  """Tests for import filtering - only keep used imports."""

  def test_unused_imports_removed(self, tmp_path: Path) -> None:
    """Test that unused imports are not included in stub."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from typing import List, Dict, Optional, Any
        from collections import OrderedDict
        from nonfig import configurable, Hyper, Ge, Le, DEFAULT

        @configurable
        def my_func(x: Hyper[int] = 10) -> int:
            return x
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    # These should NOT be in the stub (unused)
    assert "List" not in content
    # assert "Dict" not in content  # Skipped because TypedDict contains Dict
    assert "Optional" not in content
    assert "Any" not in content
    assert "OrderedDict" not in content
    assert "Ge" not in content
    assert "Le" not in content

    # These SHOULD be in the stub
    assert "MakeableModel" in content
    assert "override" in content

  def test_used_imports_kept(self, tmp_path: Path) -> None:
    """Test that used imports are kept in stub."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        import numpy as np
        from typing import Optional
        from nonfig import configurable, Hyper

        @configurable
        def my_func(x: Hyper[int] = 10) -> Optional[np.ndarray]:
            return None
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    # These should be in the stub (used in return type)
    assert "import numpy as np" in content
    assert "Optional" in content

  def test_type_checking_block_filtered(self, tmp_path: Path) -> None:
    """Test that TYPE_CHECKING imports don't duplicate top-level imports."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from typing import TYPE_CHECKING
        from collections.abc import Callable
        from nonfig import configurable, Hyper

        if TYPE_CHECKING:
            from collections.abc import Callable  # duplicate

        @configurable
        def my_func(x: Hyper[int] = 10, cb: Callable[[int], int] | None = None) -> int:
            return x
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    # Callable should appear only once (not duplicated from TYPE_CHECKING)
    assert content.count("from collections.abc import Callable") == 1

  def test_nonfig_imports_not_duplicated(self, tmp_path: Path) -> None:
    """Test that nonfig imports from source don't duplicate our generated ones."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper, DEFAULT, MakeableModel

        @configurable
        class MyClass:
            def __init__(self, x: Hyper[int] = 10):
                self.x = x
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    # Should only have one nonfig import line
    assert content.count("from nonfig import") == 1


class TestPublicItemsInclusion:
  """Tests for including all public items in stubs."""

  def test_public_function_included(self, tmp_path: Path) -> None:
    """Test that non-configurable public functions are included."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        def helper_func(x: int) -> int:
            return x * 2

        @configurable
        def my_func(x: Hyper[int] = 10) -> int:
            return helper_func(x)
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    assert "def helper_func(x: int) -> int:" in content
    assert "..." in content  # Body should be ellipsis

  def test_private_function_excluded(self, tmp_path: Path) -> None:
    """Test that private functions are not included."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        def _private_helper(x: int) -> int:
            return x * 2

        @configurable
        def my_func(x: Hyper[int] = 10) -> int:
            return _private_helper(x)
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    assert "_private_helper" not in content

  def test_public_class_included(self, tmp_path: Path) -> None:
    """Test that non-configurable public classes are included."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        class HelperClass:
            x: int
            def do_something(self, y: int) -> int:
                return self.x + y

        @configurable
        def my_func(helper: HelperClass, x: Hyper[int] = 10) -> int:
            return helper.do_something(x)
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    assert "class HelperClass:" in content
    assert "x: int" in content
    assert "def do_something(self, y: int) -> int:" in content

  def test_main_function_included(self, tmp_path: Path) -> None:
    """Test that main() function is included (it's public, not private)."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        @configurable
        def my_func(x: Hyper[int] = 10) -> int:
            return x

        def main() -> None:
            result = my_func.Config().make()(10)
            print(result)
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    assert "def main() -> None:" in content


class TestConfigFieldSkipping:
  """Tests for skipping Config: ClassVar fields."""

  def test_config_classvar_skipped_in_configurable(self, tmp_path: Path) -> None:
    """Test that Config: ClassVar fields are skipped in @configurable classes."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from typing import ClassVar
        from nonfig import configurable, Hyper, MakeableModel

        @configurable
        class MyClass:
            Config: ClassVar[type[MakeableModel[object]]]

            def __init__(self, x: Hyper[int] = 10):
                self.x = x
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    # Should have class Config (our generated one)
    assert "class Config(_NCMakeableModel[MyClass]):" in content
    # Should NOT have Config: ClassVar (the field declaration)
    assert "Config: ClassVar" not in content

  def test_config_classvar_skipped_in_public_class(self, tmp_path: Path) -> None:
    """Test that Config: ClassVar fields are skipped in non-configurable classes too."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from typing import ClassVar
        from nonfig import configurable, Hyper, MakeableModel

        class HelperClass:
            Config: ClassVar[type[MakeableModel[object]]]
            x: int

        @configurable
        def my_func(x: Hyper[int] = 10) -> int:
            return x
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    # Should have x: int but NOT Config: ClassVar
    assert "x: int" in content
    assert "Config: ClassVar" not in content


class TestConstantsAndTypeAliases:
  """Tests for constants and type alias handling."""

  def test_constants_use_ellipsis(self, tmp_path: Path) -> None:
    """Test that constants use name: ... format."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        WINDOW_SIZE = 10
        THRESHOLD = 0.75
        MODE = "fast"

        @configurable
        def my_func(x: Hyper[int] = WINDOW_SIZE) -> int:
            return x
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    # Constants should use ellipsis, not actual values
    assert "WINDOW_SIZE: ..." in content
    assert "THRESHOLD: ..." in content
    assert "MODE: ..." in content
    # Should NOT have actual values
    assert "= 10" not in content or "int = ..." in content
    assert "= 0.75" not in content
    assert '= "fast"' not in content

  def test_annotated_constants_use_ellipsis(self, tmp_path: Path) -> None:
    """Test that annotated constants use name: type = ... format."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        MAX_SIZE: int = 100
        RATIO: float = 0.5

        @configurable
        def my_func(x: Hyper[int] = 10) -> int:
            return x
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    assert "MAX_SIZE: int = ..." in content
    assert "RATIO: float = ..." in content

  def test_type_aliases_preserved(self, tmp_path: Path) -> None:
    """Test that type aliases are preserved with their values."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        MyList = list[int]
        ResultDict = dict[str, list[float]]

        @configurable
        def my_func(x: Hyper[int] = 10) -> ResultDict:
            return {}
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    # Type aliases should preserve the value (needed for type checking)
    assert "MyList = list[int]" in content
    assert "ResultDict = dict[str, list[float]]" in content

  def test_private_constants_excluded(self, tmp_path: Path) -> None:
    """Test that private constants are not included."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        _INTERNAL_VALUE = 42
        PUBLIC_VALUE = 100

        @configurable
        def my_func(x: Hyper[int] = 10) -> int:
            return x
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    assert "_INTERNAL_VALUE" not in content
    assert "PUBLIC_VALUE" in content


class TestIfNameMainExcluded:
  """Tests for excluding if __name__ == '__main__' blocks."""

  def test_if_name_main_excluded(self, tmp_path: Path) -> None:
    """Test that if __name__ == '__main__' blocks are not in stub."""
    source_file = tmp_path / "test.py"
    source_file.write_text(
      dedent("""
        from nonfig import configurable, Hyper

        @configurable
        def my_func(x: Hyper[int] = 10) -> int:
            return x

        if __name__ == "__main__":
            result = my_func.Config().make()(10)
            print(result)
      """)
    )

    infos = scan_module(source_file)
    content = generate_stub_content(infos, source_file)

    assert '__name__ == "__main__"' not in content
    assert "__name__" not in content or "def __" in content  # Allow dunder methods


def test_scan_implicit_hyper():
  source = """

@configurable
def my_func(x: int = DEFAULT):
    pass
"""
  # Create a temp file
  p = Path("temp_stub_test.py")
  p.write_text(source, encoding="utf-8")

  try:
    results = scan_module(p)
    assert len(results) == 1
    info = results[0]
    assert info.name == "my_func"

    # Should have found 'x' as a HyperParam because default is DEFAULT
    assert len(info.params) == 1
    assert info.params[0].name == "x"
    assert info.params[0].default_value == "DEFAULT"

    # Should NOT be in call_params
    assert len(info.call_params) == 0

  finally:
    if p.exists():
      p.unlink()


def test_scan_mixed_params():
  source = """

@configurable
def mixed(
    a: int, 
    b: int = DEFAULT, 
    c: Hyper[int] = 1
):
    pass
"""
  p = Path("temp_stub_test_mixed.py")
  p.write_text(source, encoding="utf-8")

  try:
    results = scan_module(p)
    assert len(results) == 1
    info = results[0]

    # 'a' -> call param
    # 'b' -> hyper (implicit)
    # 'c' -> hyper (explicit)

    hyper_names = {p.name for p in info.params}
    assert "b" in hyper_names
    assert "c" in hyper_names
    assert "a" not in hyper_names

    call_names = {cp[0] for cp in info.call_params}
    assert "a" in call_names
    assert "b" not in call_names
    assert "c" not in call_names

  finally:
    if p.exists():
      p.unlink()
