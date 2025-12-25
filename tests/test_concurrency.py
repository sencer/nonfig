"""Test thread safety and concurrent usage of @configurable."""

from __future__ import annotations

import concurrent.futures
import threading

from nonfig import Ge, Hyper, Le, configurable


def test_config_creation_thread_safety() -> None:
  """Test that creating configs from multiple threads is safe."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value * 2

  def create_config(val: int) -> int:
    config = process.Config(value=val)
    return config.value

  # Create configs from multiple threads
  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(create_config, i) for i in range(100)]
    results: list[int] = [f.result() for f in futures]

  # All values should be correct
  assert results == list(range(100))


def test_make_function_thread_safety() -> None:
  """Test that calling .make() from multiple threads is safe."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value * 2

  def create_and_call(val: int) -> int:
    config = process.Config(value=val)
    fn = config.make()
    return fn()

  # Create and call from multiple threads
  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(create_and_call, i) for i in range(100)]
    results: list[int] = [f.result() for f in futures]

  # All results should be value * 2
  assert results == [i * 2 for i in range(100)]


def test_shared_config_across_threads() -> None:
  """Test that using same config in different threads is safe."""

  @configurable
  def process(value: Hyper[int] = 10) -> int:
    return value * 2

  # Create one config
  config = process.Config(value=5)
  fn = config.make()

  def call_fn() -> int:
    return fn()

  # Call from multiple threads
  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(call_fn) for _ in range(100)]
    results: list[int] = [f.result() for f in futures]

  # All results should be the same
  assert all(r == 10 for r in results)


def test_decorator_application_is_not_racy() -> None:
  """Test that applying decorator concurrently doesn't cause issues."""

  results: list[int] = []
  lock = threading.Lock()

  def apply_decorator(i: int) -> None:
    @configurable
    def process(value: Hyper[int] = i) -> int:
      return value * 2

    config = process.Config(value=i * 10)
    fn = config.make()
    result = fn()

    with lock:
      results.append(result)

  # Apply decorator from multiple threads
  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(apply_decorator, i) for i in range(20)]
    for f in futures:
      f.result()

  # All results should be (i * 10) * 2
  expected = [(i * 10) * 2 for i in range(20)]
  assert sorted(results) == sorted(expected)


def test_concurrent_validation() -> None:
  """Test that validating different configs concurrently works."""

  @configurable
  def process(
    value: Hyper[int, Ge[0], Le[100]] = 50,
  ) -> int:
    return value

  def create_and_validate(val: int) -> bool:
    try:
      config = process.Config(value=val)
      return config.value >= 0 and config.value <= 100
    except Exception:
      return False

  # Test with mix of valid and invalid values
  test_values = list(range(-10, 110))

  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(create_and_validate, v) for v in test_values]
    results: list[bool] = [f.result() for f in futures]

  # Only values in range [0, 100] should be valid
  expected = [0 <= v <= 100 for v in test_values]
  assert results == expected
