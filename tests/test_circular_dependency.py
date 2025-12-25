"""Test circular dependency detection for nested configs.

Note: Python's name resolution naturally prevents most circular dependencies
at decoration time. Our detection mechanism provides an additional safety net
for cases where instantiation with DEFAULT could cause infinite recursion.
"""

from __future__ import annotations

import concurrent.futures

from nonfig import DEFAULT, Hyper, configurable

# Module-level function definitions for proper forward reference handling


@configurable
def inner_processor(
  data: int,
  multiplier: Hyper[int] = 2,
) -> int:
  """Simple processor that multiplies data."""
  return data * multiplier


@configurable
def outer_processor(
  data: int,
  offset: Hyper[int] = 10,
  inner_fn: Hyper[inner_processor.Config] = DEFAULT,
) -> int:
  """Processor that uses nested config with DEFAULT."""
  # inner_fn is auto-made by _recursive_make, so it's already a function
  inner_result: int = inner_fn(data=data)
  return inner_result + offset


@configurable
def level_3_processor(
  data: int,
  factor: Hyper[int] = 1,
) -> int:
  """Third level in deep nesting."""
  return data * factor


@configurable
def level_2_processor(
  data: int,
  offset: Hyper[int] = 0,
  l3_fn: Hyper[level_3_processor.Config] = DEFAULT,
) -> int:
  """Second level in deep nesting."""
  # l3_fn is auto-made by _recursive_make
  return l3_fn(data=data) + offset


@configurable
def level_1_processor(
  data: int,
  multiplier: Hyper[int] = 1,
  l2_fn: Hyper[level_2_processor.Config] = DEFAULT,
) -> int:
  """First level in deep nesting."""
  # l2_fn is auto-made by _recursive_make
  return l2_fn(data=data) * multiplier


@configurable
def processor_a(
  data: int,
  factor_a: Hyper[int] = 2,
) -> int:
  """Processor A for multi-config test."""
  return data * factor_a


@configurable
def processor_b(
  data: int,
  factor_b: Hyper[int] = 3,
) -> int:
  """Processor B for multi-config test."""
  return data * factor_b


@configurable
def combiner_processor(
  data: int,
  fn_a: Hyper[processor_a.Config] = DEFAULT,
  fn_b: Hyper[processor_b.Config] = DEFAULT,
) -> int:
  """Combines multiple independent configs."""
  # fn_a, fn_b are auto-made by _recursive_make
  result_a = fn_a(data=data)
  result_b = fn_b(data=data)
  return result_a + result_b


@configurable
class InnerComponent:
  """Inner class for testing class-based configs."""

  def __init__(
    self,
    multiplier: Hyper[float] = 2.0,
  ):
    self.multiplier = multiplier


@configurable
class OuterComponent:
  """Outer class that uses nested config."""

  def __init__(
    self,
    offset: Hyper[float] = 10.0,
    inner_config: Hyper[InnerComponent.Config] = DEFAULT,
  ):
    self.offset = offset
    self.inner_config = inner_config


# Tests


def test_valid_nested_configs_still_work() -> None:
  """Test that valid nested configs without cycles work correctly."""
  config = outer_processor.Config()
  fn = config.make()
  result = fn(data=5)
  # inner: 5 * 2 = 10, outer: 10 + 10 = 20
  assert result == 20

  # Test with custom config
  config2 = outer_processor.Config(
    offset=5,
    inner_fn=inner_processor.Config(multiplier=3),
  )
  fn2 = config2.make()
  result2 = fn2(data=5)
  # inner: 5 * 3 = 15, outer: 15 + 5 = 20
  assert result2 == 20


def test_deep_valid_nesting_works() -> None:
  """Test that deep nesting without cycles works correctly."""
  config = level_1_processor.Config()
  fn = config.make()
  result = fn(data=10)
  # l3: 10 * 1 = 10, l2: 10 + 0 = 10, l1: 10 * 1 = 10
  assert result == 10

  # Test with custom configs
  config2 = level_1_processor.Config(
    multiplier=2,
    l2_fn=level_2_processor.Config(
      offset=5,
      l3_fn=level_3_processor.Config(factor=3),
    ),
  )
  fn2 = config2.make()
  result2 = fn2(data=10)
  # l3: 10 * 3 = 30, l2: 30 + 5 = 35, l1: 35 * 2 = 70
  assert result2 == 70


def test_concurrent_config_creation_no_false_positives() -> None:
  """Test that concurrent config creation doesn't cause false positive cycles."""

  def create_and_call(val: int) -> int:
    config = outer_processor.Config(offset=val)
    fn = config.make()
    return fn(data=val)

  # Create configs from multiple threads - should not cause false positives
  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(create_and_call, i) for i in range(50)]
    results: list[int] = [f.result() for f in futures]

  # All results should be correct
  expected = [i * 2 + i for i in range(50)]  # (i * 2) + i
  assert results == expected


def test_multiple_independent_nested_configs() -> None:
  """Test that multiple independent nested configs don't trigger false cycles."""
  config = combiner_processor.Config()
  fn = config.make()
  result = fn(data=10)
  # processor_a: 10 * 2 = 20, processor_b: 10 * 3 = 30, total: 50
  assert result == 50


def test_class_nested_configs_work() -> None:
  """Test that class-based configs work correctly."""
  config = OuterComponent.Config()

  assert config is not None
  assert hasattr(config, "inner_config")


def test_context_var_isolation_across_calls() -> None:
  """Test that ContextVar properly isolates config creation across different calls."""
  # Multiple separate instantiations should not interfere with each other
  config1 = outer_processor.Config()
  config2 = outer_processor.Config()
  config3 = outer_processor.Config()

  assert config1 is not None
  assert config2 is not None
  assert config3 is not None

  # They should be independent instances
  assert config1 is not config2
  assert config2 is not config3


def test_nested_config_with_custom_values() -> None:
  """Test that nested configs with custom values work correctly."""
  # Create a custom inner config
  custom_inner = inner_processor.Config(multiplier=5)

  # Use it in the outer config
  outer_config = outer_processor.Config(
    offset=20,
    inner_fn=custom_inner,
  )

  fn = outer_config.make()
  result = fn(data=10)
  # inner: 10 * 5 = 50, outer: 50 + 20 = 70
  assert result == 70


def test_config_instantiation_multiple_times() -> None:
  """Test that configs can be instantiated multiple times without issues."""
  # This tests that the ContextVar stack is properly cleaned up after each instantiation
  for i in range(10):
    config = outer_processor.Config(offset=i)
    assert config is not None
    assert config.offset == i


def test_circular_dependency_detected() -> None:
  """Test that circular dependencies are detected and raise clear errors."""
  import pytest

  # Define two classes that reference each other with DEFAULT
  # This should fail when we try to instantiate DEFAULT for the circular reference

  @configurable
  class ClassA:
    def __init__(self, value: Hyper[int] = 1) -> None:
      self.value = value

  # Now try to create a class that references ClassA which references back
  # We simulate this by trying to create a config that would create infinite recursion

  with pytest.raises(ValueError, match="Circular dependency detected"):
    # Create a mock scenario where a Config tries to instantiate itself
    from nonfig.extraction import _config_creation_stack, _instantiate_default_config

    # Manually set up a circular situation
    token = _config_creation_stack.set([
      f"{ClassA.Config.__module__}.{ClassA.Config.__qualname__}"
    ])
    try:
      # This should detect the cycle
      _instantiate_default_config(ClassA.Config, "test_param")
    finally:
      _config_creation_stack.reset(token)
