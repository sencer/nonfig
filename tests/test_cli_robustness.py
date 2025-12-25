import logging
from pathlib import Path

import pytest

from nonfig.cli.generate_stubs import main


def test_cli_ignores_pycache_and_handles_bad_files(
  tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
  """Verify CLI robustness against directory clutter and syntax errors."""
  src = tmp_path / "src"
  src.mkdir()

  # 1. A valid file
  (src / "valid.py").write_text("@configurable\ndef f(x: Hyper[int]=1): pass")

  # 2. A syntax error file
  (src / "broken.py").write_text("def broken(")

  # 3. A __pycache__ directory that should be skipped
  pycache = src / "__pycache__"
  pycache.mkdir()
  (pycache / "ignored.py").write_text("@configurable\ndef secret(): pass")

  # 4. A file that causes an OSError (simulated by non-readable permissions if possible,
  # but hard to do reliably cross-platform in tmp. We'll rely on the syntax error for now
  # to trigger exception handling paths).

  # Capture logs at DEBUG level
  with caplog.at_level(logging.DEBUG):
    exit_code = main([str(src), "--verbose"])

  assert exit_code == 0

  # Check logs
  assert "Skipping cache directory" in caplog.text
  assert "Syntax error in" in caplog.text

  # Check outputs
  assert (src / "valid.pyi").exists()
  assert not (pycache / "ignored.pyi").exists()


def test_cli_handles_invalid_paths(caplog: pytest.LogCaptureFixture) -> None:
  """Test CLI behavior with non-existent paths."""
  with caplog.at_level(logging.ERROR):
    exit_code = main(["/non/existent/path/12345"])

  assert exit_code == 1
  assert "No Python files found" in caplog.text
