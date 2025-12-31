from pathlib import Path
from textwrap import dedent

from nonfig.stubs.scanner import scan_module


def test_multipass_implicit_hyper_detection(tmp_path: Path):
  """
  Verify that the multi-pass scanner detects implicit hyperparameters.

  referring to local configurations (decorated items or wrap_external).
  """
  source_file = tmp_path / "app.py"
  source_file.write_text(
    dedent("""
            from nonfig import configurable, wrap_external, Hyper

            @configurable
            class LocalClass:
                def __init__(self, x: int = 1):
                    self.x = x

            @configurable
            def local_func(y: int = 2):
                return y

            Wrapped = wrap_external(dict)

            @configurable
            def consumer(
                p1 = LocalClass,         # Implicit hyper (decorated class)
                p2 = local_func,         # Implicit hyper (decorated func)
                p3 = Wrapped,            # Implicit hyper (wrap_external)
                p4 = LocalClass.Config,  # Implicit hyper (.Config attr)
                p5: int = 42             # Regular param
            ):
                pass
        """),
    encoding="utf-8",
  )

  infos, _ = scan_module(source_file)

  # Find consumer info
  consumer_info = next(i for i in infos if i.name == "consumer")

  param_names = {p.name for p in consumer_info.params}
  call_param_names = {p[0] for p in consumer_info.call_params}

  # Verify implicit hypers are detected
  assert "p1" in param_names
  assert "p2" in param_names
  assert "p3" in param_names
  assert "p4" in param_names

  # Verify regular param is NOT a hyper
  assert "p5" in call_param_names
  assert "p5" not in param_names


def test_multipass_multiple_assignments(tmp_path: Path):
  """Verify implicit hyper detection with multiple assignments."""
  source_file = tmp_path / "multi.py"
  source_file.write_text(
    dedent("""
            from nonfig import wrap_external, configurable

            CfgA, CfgB = wrap_external(int), wrap_external(float)

            @configurable
            def func(a = CfgA, b = CfgB):
                pass
        """),
    encoding="utf-8",
  )

  infos, _ = scan_module(source_file)
  func_info = next(i for i in infos if i.name == "func")

  param_names = {p.name for p in func_info.params}
  assert "a" in param_names
  assert "b" in param_names


def test_aliased_config_import_detection(tmp_path: Path):
  """Issue: Aliased imports not ending in 'Config' should be detected via usage patterns."""
  source_file = tmp_path / "aliased_import.py"
  source_file.write_text(
    dedent("""
            from other import SomeConfig as MyItem
            from nonfig import configurable, DEFAULT

            @configurable
            def process(
                cfg: MyItem = DEFAULT,  # Usage of DEFAULT guarantees MyItem is a config
                other = MyItem          # This should now be detected as implicit hyper
            ):
                pass
        """),
    encoding="utf-8",
  )

  # This should detect MyItem as a config name via the DEFAULT usage,
  # and then 'other' as a hyper parameter because it uses MyItem as default.
  infos, _ = scan_module(source_file)
  process_info = next(i for i in infos if i.name == "process")
  param_names = {p.name for p in process_info.params}
  assert "cfg" in param_names
  assert "other" in param_names
