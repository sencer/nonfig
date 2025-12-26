"""Benchmark comparing class instantiation patterns."""

import os
import time

# Disable debug logging for benchmarks
os.environ["LOGURU_LEVEL"] = "WARNING"

from dataclasses import dataclass

from nonfig import DEFAULT, configurable
from nonfig.typedefs import Hyper


class RawClass:
  def __init__(self, x: int = 10, y: float = 0.5):
    self.x = x
    self.y = y


@configurable
@dataclass
class ConfigurableClass:
  x: Hyper[int] = 10
  y: Hyper[float] = 0.5


@configurable
@dataclass
class Inner:
  x: int = 1


@configurable
@dataclass
class Outer:
  inner: Hyper[Inner] = DEFAULT


def run_benchmark(iterations: int = 100_000):
  print(f"Running benchmarks with {iterations:,} iterations...")

  # Pre-create configs
  config = ConfigurableClass.Config(x=10, y=0.5)
  outer_cfg = Outer.Config(inner=Inner.Config(x=10))

  print(
    f"ConfigurableClass is leaf: {getattr(ConfigurableClass.Config, '_is_always_leaf', False)}"
  )
  print(f"Outer is leaf: {getattr(Outer.Config, '_is_always_leaf', False)}")
  print(f"Inner is leaf: {getattr(Inner.Config, '_is_always_leaf', False)}")

  # Raw instantiation
  start = time.perf_counter()
  for _ in range(iterations):
    RawClass(x=10, y=0.5)
  raw_time = (time.perf_counter() - start) / iterations * 1e6

  # Configurable direct
  start = time.perf_counter()
  for _ in range(iterations):
    ConfigurableClass(x=10, y=0.5)
  direct_time = (time.perf_counter() - start) / iterations * 1e6

  # Config.make() (Full Pydantic lifecycle)
  start = time.perf_counter()
  for _ in range(iterations):
    ConfigurableClass.Config(x=10, y=0.5).make()
  make_time = (time.perf_counter() - start) / iterations * 1e6

  # Config.fast_make() (Bypass Pydantic)
  start = time.perf_counter()
  for _ in range(iterations):
    ConfigurableClass.Config.fast_make(x=10, y=0.5)
  fast_make_time = (time.perf_counter() - start) / iterations * 1e6

  # Reused Config object
  start = time.perf_counter()
  for _ in range(iterations):
    config.make()
  reused_time = (time.perf_counter() - start) / iterations * 1e6

  # Nested make() (metadata reused)
  start = time.perf_counter()
  for _ in range(iterations):
    outer_cfg.make()
  nested_time = (time.perf_counter() - start) / iterations * 1e6

  # Nested fast_make()
  start = time.perf_counter()
  inner_config = Inner.Config(x=10)
  for _ in range(iterations):
    Outer.Config.fast_make(inner=inner_config)
  nested_fast_time = (time.perf_counter() - start) / iterations * 1e6

  print("\nPattern                           Time (μs)")
  print("-" * 43)
  print(f"Raw Class Instantiation           {raw_time:8.3f}μs")
  print(f"Configurable Direct               {direct_time:8.3f}μs")
  print(f"Configurable Config().make()      {make_time:8.3f}μs")
  print(f"Configurable Config.fast_make()   {fast_make_time:8.3f}μs")
  print(f"Configurable Reused Config        {reused_time:8.3f}μs")
  print(f"Configurable Nested make()        {nested_time:8.3f}μs")
  print(f"Configurable Nested fast_make()   {nested_fast_time:8.3f}μs")
  print("-" * 43)

  overhead = reused_time - raw_time
  print(
    f"\nOverhead of Reused Config vs Raw: {overhead:8.3f}μs ({reused_time / raw_time:.1f}x)"
  )
  print(
    f"Overhead of fast_make() vs Raw:   {fast_make_time - raw_time:8.3f}μs ({fast_make_time / raw_time:.1f}x)"
  )


if __name__ == "__main__":
  run_benchmark()
