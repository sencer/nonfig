#!/usr/bin/env python3
"""Generate .pyi stub files for @configurable decorated functions and classes.

This script scans Python files for @configurable decorators on functions,
methods, classes (with __init__), and dataclasses. It generates corresponding
type stub files with Config class stubs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from nonfig.stubs import generate_stub_for_file

logger = logging.getLogger(__name__)


def _process_files(files: list[Path]) -> int:
  """Process a list of Python files and count generated stubs.

  Args:
    files: List of Python files to process

  Returns:
    Number of stub files generated
  """
  generated_count = 0

  for py_file in files:
    if "__pycache__" in str(py_file):
      logger.debug("Skipping cache directory: %s", py_file)
      continue

    if py_file.name.startswith("test_"):
      logger.debug("Skipping test file: %s", py_file)
      continue

    try:
      if generate_stub_for_file(py_file):
        generated_count += 1
    except SyntaxError as e:
      logger.error("Syntax error in %s: %s", py_file, e)
    except OSError as e:
      logger.error("I/O error reading %s: %s", py_file, e)
    except ValueError as e:
      logger.error("Invalid constraint in %s: %s", py_file, e)

  return generated_count


def _resolve_paths(pattern: str) -> list[Path]:
  """Resolve a path pattern to a list of Python files."""
  path = Path(pattern)

  if path.is_dir():
    return list(path.rglob("*.py"))

  if path.is_file() and path.suffix == ".py":
    return [path]

  # For nonexistent paths, return empty list
  if path.is_absolute() or not path.exists():
    return []

  # Glob pattern
  return [p for p in Path().glob(pattern) if p.suffix == ".py" and p.is_file()]


def main(argv: list[str] | None = None) -> int:
  """Generate stubs for @configurable decorated items."""
  parser = argparse.ArgumentParser(
    description="Generate .pyi stub files for @configurable decorators",
  )
  parser.add_argument(
    "patterns",
    nargs="*",
    default=["src"],
    help="Files, directories, or glob patterns to process (default: src)",
  )
  parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Enable verbose logging",
  )

  args = parser.parse_args(argv)
  verbose: bool = bool(args.verbose)
  patterns: list[str] = list(args.patterns)

  # Configure logging
  log_level = logging.DEBUG if verbose else logging.INFO
  logging.basicConfig(level=log_level, format="%(message)s")

  # Resolve all patterns to a list of files
  all_files: list[Path] = []
  for pattern in patterns:
    all_files.extend(_resolve_paths(pattern))

  # Remove duplicates
  files = list(dict.fromkeys(all_files))

  if not files:
    logger.error("No Python files found matching: %s", ", ".join(patterns))
    return 1

  logger.debug("Found %d Python file(s) to process", len(files))

  generated_count = _process_files(files)
  logger.info("Generated %d stub file(s)", generated_count)
  return 0


if __name__ == "__main__":
  sys.exit(main())
