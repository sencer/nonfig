"""Benchmark comparing class instantiation patterns."""

import os
import time

# Disable debug logging for benchmarks
os.environ["LOGURU_LEVEL"] = "WARNING"

from nonfig import Hyper, configurable


class RawClass:
  def __init__(self, x: int = 10, y: float = 0.5):
    self.x = x
    self.y = y


@configurable
class ConfigurableClass:
  def __init__(self, x: Hyper[int] = 10, y: Hyper[float] = 0.5):
    self.x = x
    self.y = y


def benchmark():
  iterations = 100000
  print(f"Running benchmarks with {iterations:,} iterations...\n")

  # 1. Raw class instantiation (baseline)
  start = time.perf_counter()
  for _ in range(iterations):
    _ = RawClass(x=50, y=0.75)
  raw_us = (time.perf_counter() - start) / iterations * 1_000_000

  # 2. Configurable class direct instantiation
  start = time.perf_counter()
  for _ in range(iterations):
    _ = ConfigurableClass(x=50, y=0.75)
  direct_us = (time.perf_counter() - start) / iterations * 1_000_000

  # 3. Configurable class via Config().make()
  # This includes config creation and the make() call
  start = time.perf_counter()
  for _ in range(iterations):
    _ = ConfigurableClass.Config(x=50, y=0.75).make()
  make_us = (time.perf_counter() - start) / iterations * 1_000_000

  # 4. Configurable class via cached make()
  config = ConfigurableClass.Config(x=50, y=0.75)
  start = time.perf_counter()
  for _ in range(iterations):
    _ = config.make()
  cached_make_us = (time.perf_counter() - start) / iterations * 1_000_000

  print(f"{'Pattern':<30} {'Time (μs)':>12}")
  print("-" * 43)
  print(f"{'Raw Class Instantiation':<30} {raw_us:>11.3f}μs")
  print(f"{'Configurable Direct':<30} {direct_us:>11.3f}μs")
  print(f"{'Configurable Config().make()':<30} {make_us:>11.3f}μs")
  print(f"{'Configurable Cached make()':<30} {cached_make_us:>11.3f}μs")
  print("-" * 43)

  overhead = make_us - raw_us
  print(
    f"\nOverhead of Config().make() vs Raw: {overhead:.3f}μs ({(make_us / raw_us):.1f}x)"
  )


if __name__ == "__main__":
  benchmark()
