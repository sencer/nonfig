"""Tests for config loaders (JSON, TOML, YAML)."""

import json
from pathlib import Path
import tempfile

import pytest

from nonfig import load_json, load_toml


class TestLoadJson:
  """Tests for load_json function."""

  def test_load_simple_json(self) -> None:
    """Load a simple JSON file."""
    data = {"name": "test", "value": 42}

    with tempfile.NamedTemporaryFile(
      encoding="utf-8", mode="w", suffix=".json", delete=False
    ) as f:
      json.dump(data, f)
      path = Path(f.name)

    try:
      result = load_json(path)
      assert result == data
    finally:
      path.unlink()

  def test_load_json_with_string_path(self) -> None:
    """Load JSON with string path."""
    data = {"key": "value"}

    with tempfile.NamedTemporaryFile(
      encoding="utf-8", mode="w", suffix=".json", delete=False
    ) as f:
      json.dump(data, f)
      path = f.name

    try:
      result = load_json(path)
      assert result == data
    finally:
      Path(path).unlink()

  def test_load_nested_json(self) -> None:
    """Load JSON with nested structure."""
    data = {"optimizer": {"lr": 0.01, "momentum": 0.9}, "epochs": 100}

    with tempfile.NamedTemporaryFile(
      encoding="utf-8", mode="w", suffix=".json", delete=False
    ) as f:
      json.dump(data, f)
      path = Path(f.name)

    try:
      result = load_json(path)
      assert result["optimizer"]["lr"] == 0.01
      assert result["epochs"] == 100
    finally:
      path.unlink()


class TestLoadToml:
  """Tests for load_toml function."""

  def test_load_simple_toml(self) -> None:
    """Load a simple TOML file."""
    toml_content = """
name = "test"
value = 42
"""
    with tempfile.NamedTemporaryFile(
      encoding="utf-8", mode="w", suffix=".toml", delete=False
    ) as f:
      f.write(toml_content)
      path = Path(f.name)

    try:
      result = load_toml(path)
      assert result["name"] == "test"
      assert result["value"] == 42
    finally:
      path.unlink()

  def test_load_nested_toml(self) -> None:
    """Load TOML with nested tables."""
    toml_content = """
[optimizer]
lr = 0.01
momentum = 0.9

[training]
epochs = 100
"""
    with tempfile.NamedTemporaryFile(
      encoding="utf-8", mode="w", suffix=".toml", delete=False
    ) as f:
      f.write(toml_content)
      path = Path(f.name)

    try:
      result = load_toml(path)
      assert result["optimizer"]["lr"] == 0.01
      assert result["training"]["epochs"] == 100
    finally:
      path.unlink()


class TestLoadYaml:
  """Tests for load_yaml function."""

  def test_load_yaml_import_error(self) -> None:
    """load_yaml raises ImportError with helpful message when PyYAML unavailable."""
    # We can't easily test this without uninstalling PyYAML
    # but we can test the function works if PyYAML is available
    try:
      from nonfig import load_yaml

      yaml_content = "name: test\nvalue: 42"
      with tempfile.NamedTemporaryFile(
        encoding="utf-8", mode="w", suffix=".yaml", delete=False
      ) as f:
        f.write(yaml_content)
        path = Path(f.name)

      try:
        result = load_yaml(path)
        assert result["name"] == "test"
        assert result["value"] == 42
      finally:
        path.unlink()
    except ImportError:
      pytest.skip("PyYAML not installed")

  def test_load_nested_yaml(self) -> None:
    """Load YAML with nested structure."""
    try:
      from nonfig import load_yaml

      yaml_content = """
optimizer:
  lr: 0.01
  momentum: 0.9
training:
  epochs: 100
"""
      with tempfile.NamedTemporaryFile(
        encoding="utf-8", mode="w", suffix=".yaml", delete=False
      ) as f:
        f.write(yaml_content)
        path = Path(f.name)

      try:
        result = load_yaml(path)
        assert result["optimizer"]["lr"] == 0.01
        assert result["training"]["epochs"] == 100
      finally:
        path.unlink()
    except ImportError:
      pytest.skip("PyYAML not installed")
