"""Benchmark usage patterns: functions, methods, classes, dataclasses.

Each section benchmarks a different usage pattern independently.
"""

from dataclasses import dataclass
import os
import time

# Disable debug logging for benchmarks
os.environ["LOGURU_LEVEL"] = "WARNING"

from nonfig import Hyper, configurable


def benchmark_function_pattern():
  """Benchmark configurable function usage."""

  @configurable
  def compute(x: Hyper[int] = 10, y: Hyper[float] = 0.5) -> float:
    return x * y

  iterations = 100000

  # Config creation
  start = time.perf_counter()
  for _ in range(iterations):
    config = compute.Config(x=50, y=0.75)
  elapsed = time.perf_counter() - start
  config_us = elapsed / iterations * 1_000_000

  # make()
  config = compute.Config(x=50, y=0.75)
  start = time.perf_counter()
  for _ in range(iterations):
    _ = config.make()
  elapsed = time.perf_counter() - start
  make_us = elapsed / iterations * 1_000_000

  # Full pattern
  start = time.perf_counter()
  for _ in range(iterations):
    _ = compute.Config(x=50, y=0.75).make()()
  elapsed = time.perf_counter() - start
  full_us = elapsed / iterations * 1_000_000

  return {"config_us": config_us, "make_us": make_us, "full_us": full_us}


def benchmark_method_pattern():
  """Benchmark configurable instance method usage."""

  class Calculator:
    def __init__(self, multiplier: float = 1.0) -> None:
      self.multiplier = multiplier

    @configurable
    def compute(self, x: Hyper[int] = 10, y: Hyper[float] = 0.5) -> float:
      return x * y * self.multiplier

  iterations = 50000
  calc = Calculator(multiplier=2.0)

  # Config creation
  start = time.perf_counter()
  for _ in range(iterations):
    config = calc.compute.Config(x=50, y=0.75)
  elapsed = time.perf_counter() - start
  config_us = elapsed / iterations * 1_000_000

  # make()
  config = calc.compute.Config(x=50, y=0.75)
  start = time.perf_counter()
  for _ in range(iterations):
    fn = config.make()
  elapsed = time.perf_counter() - start
  make_us = elapsed / iterations * 1_000_000

  # Full pattern - made function needs to be called with instance
  start = time.perf_counter()
  for _ in range(iterations):
    fn = calc.compute.Config(x=50, y=0.75).make()
    _ = fn(calc)
  elapsed = time.perf_counter() - start
  full_us = elapsed / iterations * 1_000_000

  return {"config_us": config_us, "make_us": make_us, "full_us": full_us}


def benchmark_class_pattern():
  """Benchmark configurable class usage."""

  @configurable
  class Processor:
    def __init__(self, x: Hyper[int] = 10, y: Hyper[float] = 0.5) -> None:
      self.x = x
      self.y = y

    def compute(self) -> float:
      return self.x * self.y

  iterations = 50000

  # Config creation
  start = time.perf_counter()
  for _ in range(iterations):
    config = Processor.Config(x=50, y=0.75)
  elapsed = time.perf_counter() - start
  config_us = elapsed / iterations * 1_000_000

  # make()
  config = Processor.Config(x=50, y=0.75)
  start = time.perf_counter()
  for _ in range(iterations):
    _ = config.make()
  elapsed = time.perf_counter() - start
  make_us = elapsed / iterations * 1_000_000

  # Full overhead (make() directly returns instance for classes)
  start = time.perf_counter()
  for _ in range(iterations):
    _ = Processor.Config(x=50, y=0.75).make()
  elapsed = time.perf_counter() - start
  full_us = elapsed / iterations * 1_000_000

  return {"config_us": config_us, "make_us": make_us, "full_us": full_us}


def benchmark_dataclass_pattern():
  """Benchmark configurable dataclass usage."""

  @configurable
  @dataclass
  class Data:
    x: Hyper[int] = 10
    y: Hyper[float] = 0.5

    def compute(self) -> float:
      return self.x * self.y

  iterations = 50000

  # Config creation
  start = time.perf_counter()
  for _ in range(iterations):
    config = Data.Config(x=50, y=0.75)
  elapsed = time.perf_counter() - start
  config_us = elapsed / iterations * 1_000_000

  # make()
  config = Data.Config(x=50, y=0.75)
  start = time.perf_counter()
  for _ in range(iterations):
    _ = config.make()
  elapsed = time.perf_counter() - start
  make_us = elapsed / iterations * 1_000_000

  # Full pattern (make() directly returns instance for dataclasses)
  start = time.perf_counter()
  for _ in range(iterations):
    _ = Data.Config(x=50, y=0.75).make()
  elapsed = time.perf_counter() - start
  full_us = elapsed / iterations * 1_000_000

  return {"config_us": config_us, "make_us": make_us, "full_us": full_us}


def print_results():
  """Run all benchmarks and print formatted results."""
  print("=" * 80)
  print("nonfig Usage Pattern Benchmarks")
  print("=" * 80)
  print()

  # Function pattern
  print("1. Function Pattern")
  print("-" * 80)
  func_result = benchmark_function_pattern()
  print(f"   Config creation:    {func_result['config_us']:.3f}μs")
  print(f"   make():             {func_result['make_us']:.3f}μs")
  print(f"   Full pattern:       {func_result['full_us']:.3f}μs")
  print()

  # Method pattern
  print("2. Method Pattern (instance method)")
  print("-" * 80)
  method_result = benchmark_method_pattern()
  print(f"   Config creation:    {method_result['config_us']:.3f}μs")
  print(f"   make():             {method_result['make_us']:.3f}μs")
  print(f"   Full pattern:       {method_result['full_us']:.3f}μs")
  print()

  # Class pattern
  print("3. Class Pattern")
  print("-" * 80)
  class_result = benchmark_class_pattern()
  print(f"   Config creation:    {class_result['config_us']:.3f}μs")
  print(f"   make():             {class_result['make_us']:.3f}μs")
  print(f"   Full pattern:       {class_result['full_us']:.3f}μs")
  print()

  # Dataclass pattern
  print("4. Dataclass Pattern")
  print("-" * 80)
  dataclass_result = benchmark_dataclass_pattern()
  print(f"   Config creation:    {dataclass_result['config_us']:.3f}μs")
  print(f"   make():             {dataclass_result['make_us']:.3f}μs")
  print(f"   Full pattern:       {dataclass_result['full_us']:.3f}μs")
  print()

  # Summary table
  print("=" * 80)
  print("Summary Table")
  print("=" * 80)
  print()
  print(f"{'Pattern':<20} {'Config':>12} {'make()':>12} {'Full':>12}")
  print("-" * 80)
  print(
    f"{'Function':<20} {func_result['config_us']:>11.3f}μs "
    f"{func_result['make_us']:>11.3f}μs {func_result['full_us']:>11.3f}μs"
  )
  print(
    f"{'Method':<20} {method_result['config_us']:>11.3f}μs "
    f"{method_result['make_us']:>11.3f}μs {method_result['full_us']:>11.3f}μs"
  )
  print(
    f"{'Class':<20} {class_result['config_us']:>11.3f}μs "
    f"{class_result['make_us']:>11.3f}μs {class_result['full_us']:>11.3f}μs"
  )
  print(
    f"{'Dataclass':<20} {dataclass_result['config_us']:>11.3f}μs "
    f"{dataclass_result['make_us']:>11.3f}μs {dataclass_result['full_us']:>11.3f}μs"
  )
  print("=" * 80)


if __name__ == "__main__":
  print_results()
