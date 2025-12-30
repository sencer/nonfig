from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field, ValidationError
from pydantic_core import PydanticUndefined
import pytest

from nonfig import DEFAULT, Hyper, configurable, wrap_external
from nonfig.constraints import Ge
from nonfig.models import HyperMarker


# External components to wrap
class ExternalModel:
  def __init__(self, name: str, layers: int = 3):
    self.name = name
    self.layers = layers

  def __repr__(self):
    return f"ExternalModel(name={self.name!r}, layers={self.layers})"


class ExternalLayer:
  def __init__(self, size: int = 128):
    self.size = size


class ExternalModelWithLayer:
  def __init__(self, layer: ExternalLayer | None = None, name: str = "model"):
    self.layer = layer or ExternalLayer()
    self.name = name


def external_factory(x: float = 0.0, y: float = 1.0):
  return x + y


# Module level wrapped configs for correct type resolution
FactoryConfig = wrap_external(external_factory)
LayerConfig = wrap_external(ExternalLayer)
ModelWithLayerConfig = wrap_external(
  ExternalModelWithLayer, overrides={"layer": LayerConfig}
)


@configurable
@dataclass
class AppConfig:
  factory: FactoryConfig = DEFAULT


@configurable
@dataclass
class Trainer:
  model: ModelWithLayerConfig = DEFAULT
  lr: float = 0.001


def test_wrap_external_class():
  model_config_cls = wrap_external(ExternalModel)

  # Basic usage
  config = model_config_cls(name="test", layers=5)
  model = config.make()
  assert isinstance(model, ExternalModel)
  assert model.name == "test"
  assert model.layers == 5

  # Default values
  config_default = model_config_cls(name="default")
  model_default = config_default.make()
  assert model_default.layers == 3

  # Validation
  with pytest.raises(ValidationError):
    model_config_cls()  # name is required


def test_wrap_external_function():
  config = FactoryConfig(x=10.0, y=2.0)
  bound_fn = config.make()
  assert bound_fn() == 12.0

  # Nested in another config
  app_config = AppConfig.Config(factory=FactoryConfig(x=5.0))
  app = app_config.make()
  assert app.factory() == 6.0


def test_serialization():
  model_config_cls = wrap_external(ExternalModel)
  config = model_config_cls(name="serialized", layers=10)

  # To dict
  data = config.model_dump()
  assert data == {"name": "serialized", "layers": 10}

  # From dict
  config2 = model_config_cls(**data)
  assert config2.name == "serialized"
  assert config2.layers == 10

  # To JSON
  json_data = config.model_dump_json()
  assert '"name":"serialized"' in json_data
  assert '"layers":10' in json_data


def test_wrap_external_overrides():
  class Inner:
    def __init__(self, a: int = 1):
      self.a = a

  class Outer:
    def __init__(self, inner: Inner):
      self.inner = inner

  inner_config_cls = wrap_external(Inner)
  outer_config_cls = wrap_external(Outer, overrides={"inner": inner_config_cls})

  config = outer_config_cls(inner=inner_config_cls(a=42))
  obj = config.make()
  assert isinstance(obj.inner, Inner)
  assert obj.inner.a == 42


def test_wrap_external_error():
  with pytest.raises(TypeError, match="requires a class or function"):
    wrap_external(123)  # type: ignore


def test_wrap_external_with_complex_types():
  class ComplexExternal:
    def __init__(self, mapping: dict[str, int], items: list[str]):
      self.mapping = mapping
      self.items = items

  complex_config_cls = wrap_external(ComplexExternal)
  config = complex_config_cls(mapping={"a": 1}, items=["x", "y"])
  obj = config.make()
  assert obj.mapping == {"a": 1}
  assert obj.items == ["x", "y"]


def test_deep_nesting_recursive_make():
  config = Trainer.Config(
    model=ModelWithLayerConfig(layer=LayerConfig(size=256), name="deep_model"), lr=0.01
  )

  trainer = config.make()
  assert isinstance(trainer, Trainer)
  assert isinstance(trainer.model, ExternalModelWithLayer)
  assert isinstance(trainer.model.layer, ExternalLayer)
  assert trainer.model.layer.size == 256
  assert trainer.model.name == "deep_model"


def test_scan_wrap_external(tmp_path: Path):
  """Test that scan_module detects wrap_external assignments."""
  from textwrap import dedent

  from nonfig.stubs.scanner import scan_module

  source_file = tmp_path / "wrapped.py"
  source_file.write_text(
    dedent("""
        from nonfig import wrap_external
        class Adam: pass
        AdamConfig = wrap_external(Adam)
    """),
    encoding="utf-8",
  )

  infos, _ = scan_module(source_file)
  assert len(infos) == 1
  assert infos[0].name == "AdamConfig"
  assert infos[0].return_type == "Adam"


def test_stub_generation_for_wrap_external(tmp_path: Path):
  from textwrap import dedent

  from nonfig.stubs import generate_stub_content, scan_module

  source_file = tmp_path / "wrap_stub.py"
  source_file.write_text(
    dedent("""
        from nonfig import wrap_external
        class MyExternal:
            def __init__(self, a: int = 1): pass

        MyConfig = wrap_external(MyExternal)
    """),
    encoding="utf-8",
  )

  infos, aliases = scan_module(source_file)
  content = generate_stub_content(infos, source_file, aliases)

  assert "class MyConfig(_NCMakeableModel[MyExternal]):" in content
  assert "def __init__(self, **kwargs: Any) -> None: ..." in content
  assert "def make(self) -> MyExternal: ..." in content
  assert "class Config(" not in content


def test_typing_inference():
  # This is for static analysis check
  config = LayerConfig(size=64)
  layer = config.make()

  if TYPE_CHECKING:
    assert isinstance(layer, ExternalLayer)


def test_wrap_external_hyper_override_metadata():
  """
  Verify that wrap_external with Hyper override doesn't leak HyperMarker
  into Pydantic metadata and correctly applies constraints.
  """

  def external_func(x: int, y: int = 5):
    return x + y

  # This should enforce x >= 10
  config_cls = wrap_external(external_func, overrides={"x": Hyper[int, Ge[10]]})

  field_x = config_cls.model_fields["x"]

  # Check if HyperMarker leaked into metadata
  has_leak = any(isinstance(m, type) and m is HyperMarker for m in field_x.metadata)
  assert not has_leak, "HyperMarker leaked into Pydantic metadata"

  # Verify constraint is actually applied
  with pytest.raises(ValidationError):
    config_cls(x=5)

  config = config_cls(x=15)
  assert config.x == 15


def test_wrap_external_required_field_preservation():
  """
  Verify that wrap_external correctly identifies and preserves required fields
  even when overridden with a simple type.
  """

  def external_func(x: int, y: int = 5):
    return x + y

  # 'x' is required in external_func.
  config_cls = wrap_external(external_func, overrides={"y": int})

  field_x = config_cls.model_fields["x"]
  # In Pydantic V2, required fields have default=PydanticUndefined
  assert field_x.default is PydanticUndefined

  # This should fail because 'x' is missing
  with pytest.raises(ValidationError, match="Field required"):
    config_cls(y=10)


def target_with_kwargs(a: int, **kwargs: Any):
  return {"a": a, "kwargs": kwargs}


def test_kwargs_override_optionality():
  """Verify that parameters added via overrides for **kwargs can be made optional."""
  from pydantic import Field

  # Use Annotated to specify the default value for the new parameter
  config_cls = wrap_external(
    target_with_kwargs, overrides={"extra": Annotated[int | None, Field(default=None)]}
  )

  # 1. 'a' is still required
  with pytest.raises(ValidationError, match="a"):
    config_cls(extra=10)

  # 2. 'extra' is now optional because we gave it a default
  config = config_cls(a=1)
  assert config.extra is None

  bound_fn = config.make()
  result = bound_fn()
  assert result["a"] == 1
  assert result["kwargs"]["extra"] is None


def test_kwargs_override_mandatory_by_default():
  """Verify that additions to **kwargs are mandatory if no default is provided."""
  config_cls = wrap_external(target_with_kwargs, overrides={"extra": int})

  with pytest.raises(ValidationError, match="extra"):
    config_cls(a=1)


def test_multiple_kwargs_overrides():
  config_cls = wrap_external(
    target_with_kwargs,
    overrides={
      "extra_int": Annotated[int, Field(default=0)],
      "extra_str": Annotated[str | None, Field(default=None)],
    },
  )

  config = config_cls(a=5, extra_int=10)
  assert config.extra_int == 10
  assert config.extra_str is None

  bound_fn = config.make()
  result = bound_fn()
  assert result["kwargs"]["extra_int"] == 10


def test_override_existing_parameter_remains_required():
  """Verify that overriding an existing required parameter doesn't make it optional."""
  config_cls = wrap_external(target_with_kwargs, overrides={"a": float})

  with pytest.raises(ValidationError, match="a"):
    config_cls()


def test_wrap_external_positional_only_error():
  """
  Issue 2: wrap_external fails at runtime when target has positional-only args.
  We want to catch this early and provide a helpful error message.
  """

  def pos_only(a: int, /, b: int = 1):
    return a + b

  # Currently this succeeds but .make() will fail.
  # We want this to raise a ValueError during wrap_external.
  with pytest.raises(ValueError, match="positional-only parameters") as excinfo:
    wrap_external(pos_only)

  assert "wrapper function" in str(excinfo.value)


def test_wrap_external_callable_instance():
  """
  Issue: wrap_external crashes when target has no __name__ (e.g., callable instance).
  """

  class CallableObj:
    def __call__(self, x: int = 1):
      return x

  obj = CallableObj()
  # This should not crash and should produce a working config class
  config_cls = wrap_external(obj)
  assert config_cls.__name__ == "CallableObjConfig"

  bound_fn = config_cls(x=10).make()
  assert bound_fn() == 10


def test_wrap_external_overrides_callable_instance():
  """
  Issue: _apply_wrap_overrides crashes when target has no __name__.
  """

  class CallableObj:
    def __call__(self, x: int = 1):
      return x

  obj = CallableObj()
  # Overriding a parameter on a callable instance
  config_cls = wrap_external(obj, overrides={"x": int})
  assert "x" in config_cls.model_fields
