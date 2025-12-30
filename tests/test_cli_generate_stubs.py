from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from nonfig.cli.generate_stubs import _process_files, _resolve_paths, main


def test_resolve_paths_glob(tmp_path: Path):
  """Test _resolve_paths with glob pattern."""
  # Create some files
  (tmp_path / "a.py").touch()
  (tmp_path / "b.py").touch()
  (tmp_path / "c.txt").touch()

  # Change to tmp_path to test globbing relative to CWD
  original_cwd = Path.cwd()
  os.chdir(tmp_path)
  try:
    paths = _resolve_paths("*.py")
    names = {p.name for p in paths}
    assert "a.py" in names
    assert "b.py" in names
    assert "c.txt" not in names
  finally:
    os.chdir(original_cwd)


def test_resolve_paths_absolute_nonexistent():
  """Test _resolve_paths with absolute nonexistent path returns empty."""
  path = Path(Path.cwd()) / "nonexistent_absolute_path.py"
  assert _resolve_paths(str(path)) == []


def test_process_files_exceptions(tmp_path: Path, caplog):
  """Test exception handling in _process_files."""
  f1 = tmp_path / "f1.py"
  f2 = tmp_path / "f2.py"
  f3 = tmp_path / "f3.py"
  f1.touch()
  f2.touch()
  f3.touch()

  with patch("nonfig.cli.generate_stubs.generate_stub_for_file") as mock_gen:
    # File 1: OSError
    # File 2: ValueError
    # File 3: SyntaxError
    mock_gen.side_effect = [
      OSError("Disk full"),
      ValueError("Bad constraint"),
      SyntaxError("Bad syntax"),
    ]

    count = _process_files([f1, f2, f3])

    assert count == 0
    assert "I/O error reading" in caplog.text
    assert "Invalid constraint in" in caplog.text
    assert "Syntax error in" in caplog.text


def test_process_files_skips(tmp_path: Path):
  """Test skipping __pycache__ and test_ files."""
  cache_file = tmp_path / "__pycache__" / "cached.py"
  test_file = tmp_path / "test_foo.py"

  with patch("nonfig.cli.generate_stubs.generate_stub_for_file") as mock_gen:
    count = _process_files([cache_file, test_file])
    assert count == 0
    mock_gen.assert_not_called()


def test_main_no_files(capsys):
  """Test main returns 1 if no files found."""
  with patch("nonfig.cli.generate_stubs._resolve_paths", return_value=[]):
    ret = main(["nonexistent"])
    assert ret == 1
