"""Comprehensive benchmark of core nonfig performance.

Measures:
- Config creation cost
- make() overhead
- Function call patterns (raw, configurable, made)
"""

import os
import time

# Disable debug logging for benchmarks
os.environ["LOGURU_LEVEL"] = "WARNING"

from nonfig import Ge, Hyper, Le, configurable


def benchmark_config_creation():
  """Benchmark just creating config objects."""

  @configurable
  def func(x: Hyper[int] = 10, y: Hyper[float] = 0.5) -> float:
    return x * y

  iterations = 100000

  start = time.perf_counter()
  for _ in range(iterations):
    _ = func.Config(x=50, y=0.75)
  elapsed = time.perf_counter() - start
  per_call = elapsed / iterations * 1_000_000  # microseconds

  return {"time_us": per_call, "iterations": iterations}


def benchmark_make_overhead():
  """Benchmark make() overhead (assuming config already created)."""

  @configurable
  def func(x: Hyper[int] = 10, y: Hyper[float] = 0.5) -> float:
    return x * y

  config = func.Config(x=50, y=0.75)
  iterations = 100000

  start = time.perf_counter()
  for _ in range(iterations):
    _ = config.make()
  elapsed = time.perf_counter() - start
  per_call = elapsed / iterations * 1_000_000  # microseconds

  return {"time_us": per_call, "iterations": iterations}


def benchmark_function_calls():
  """Compare different ways of calling functions."""

  # Raw function (baseline)
  def raw_func(x: int = 10, y: float = 0.5) -> float:
    return x * y

  # Configurable function
  @configurable
  def config_func(x: Hyper[int] = 10, y: Hyper[float] = 0.5) -> float:
    return x * y

  iterations = 100000

  # 1. Raw function call
  start = time.perf_counter()
  for _ in range(iterations):
    _ = raw_func(x=50, y=0.75)
  elapsed_raw = time.perf_counter() - start
  raw_us = elapsed_raw / iterations * 1_000_000

  # 2. Configurable direct call
  start = time.perf_counter()
  for _ in range(iterations):
    _ = config_func(x=50, y=0.75)
  elapsed_direct = time.perf_counter() - start
  direct_us = elapsed_direct / iterations * 1_000_000

  # 3. Made function call
  made_fn = config_func.Config(x=50, y=0.75).make()
  start = time.perf_counter()
  for _ in range(iterations):
    _ = made_fn()
  elapsed_made = time.perf_counter() - start
  made_us = elapsed_made / iterations * 1_000_000

  return {
    "raw": {"time_us": raw_us, "iterations": iterations},
    "configurable_direct": {"time_us": direct_us, "iterations": iterations},
    "made": {"time_us": made_us, "iterations": iterations},
  }


def benchmark_full_pattern():
  """Benchmark the full Config().make()() pattern."""

  @configurable
  def func(x: Hyper[int] = 10, y: Hyper[float] = 0.5) -> float:
    return x * y

  iterations = 100000

  # Full pattern: create config, make, and call all together
  start = time.perf_counter()
  for _ in range(iterations):
    _ = func.Config(x=50, y=0.75).make()()
  elapsed = time.perf_counter() - start
  per_call = elapsed / iterations * 1_000_000  # microseconds

  return {"time_us": per_call, "iterations": iterations}


def benchmark_with_validation():
  """Benchmark with validation constraints."""

  @configurable
  def func(
    x: Hyper[int, Ge[0], Le[100]] = 10, y: Hyper[float, Ge[0.0], Le[1.0]] = 0.5
  ) -> float:
    return x * y

  iterations = 100000

  # Config creation with validation
  start = time.perf_counter()
  for _ in range(iterations):
    _ = func.Config(x=50, y=0.75)
  elapsed_config = time.perf_counter() - start
  config_us = elapsed_config / iterations * 1_000_000

  # Full pattern with validation
  start = time.perf_counter()
  for _ in range(iterations):
    _ = func.Config(x=50, y=0.75).make()()
  elapsed_full = time.perf_counter() - start
  full_us = elapsed_full / iterations * 1_000_000

  return {
    "config_creation": {"time_us": config_us, "iterations": iterations},
    "full_pattern": {"time_us": full_us, "iterations": iterations},
  }


def print_results():
  """Run all benchmarks and print formatted results."""
  print("=" * 80)
  print("nonfig Core Performance Benchmarks")
  print("=" * 80)
  print()

  # Config creation
  print("1. Config Creation Cost")
  print("-" * 80)
  config_result = benchmark_config_creation()
  print(f"   Time: {config_result['time_us']:.3f}μs per config")
  print(f"   Iterations: {config_result['iterations']:,}")
  print()

  # make() overhead
  print("2. make() Overhead")
  print("-" * 80)
  make_result = benchmark_make_overhead()
  print(f"   Time: {make_result['time_us']:.3f}μs per make()")
  print(f"   Iterations: {make_result['iterations']:,}")
  print()

  # Function calls comparison
  print("3. Function Call Comparison")
  print("-" * 80)
  call_results = benchmark_function_calls()
  raw_time = call_results["raw"]["time_us"]
  direct_time = call_results["configurable_direct"]["time_us"]
  made_time = call_results["made"]["time_us"]

  print(f"   Raw function call:              {raw_time:.3f}μs")
  print(f"   Configurable direct call:       {direct_time:.3f}μs")
  print(
    f"     Overhead vs raw:              +{direct_time - raw_time:.3f}μs "
    f"({(direct_time / raw_time - 1) * 100:.1f}%)"
  )
  print(f"   Made function call:             {made_time:.3f}μs")
  print(
    f"     Overhead vs raw:              +{made_time - raw_time:.3f}μs "
    f"({(made_time / raw_time - 1) * 100:.1f}%)"
  )
  print()

  # Full pattern
  print("4. Full Pattern: Config().make()()")
  print("-" * 80)
  full_result = benchmark_full_pattern()
  full_time = full_result["time_us"]
  print(f"   Time: {full_time:.3f}μs per call")
  print()
  print("   Breakdown:")
  print(
    f"     Config creation:  ~{config_result['time_us']:.3f}μs "
    f"({config_result['time_us'] / full_time * 100:.1f}%)"
  )
  print(
    f"     make():           ~{make_result['time_us']:.3f}μs "
    f"({make_result['time_us'] / full_time * 100:.1f}%)"
  )
  print(
    f"     Function call:    ~{made_time:.3f}μs ({made_time / full_time * 100:.1f}%)"
  )
  print()

  # With validation
  print("5. With Validation Constraints")
  print("-" * 80)
  validation_results = benchmark_with_validation()
  val_config = validation_results["config_creation"]["time_us"]
  val_full = validation_results["full_pattern"]["time_us"]
  print(f"   Config creation:    {val_config:.3f}μs")
  print(
    f"     Overhead vs no validation:    +{val_config - config_result['time_us']:.3f}μs"
  )
  print(f"   Full pattern:       {val_full:.3f}μs")
  print(
    f"     Overhead vs no validation:    +{val_full - full_result['time_us']:.3f}μs"
  )
  print()

  print("=" * 80)
  print("Summary")
  print("=" * 80)
  print(
    f"Creating a config and calling via make() adds ~{full_time - raw_time:.1f}μs "
    "overhead"
  )
  print(f"  ({(full_time / raw_time - 1) * 100:.0f}% more than raw function call)")
  print()
  print("Breakdown of overhead:")
  print(f"  Config creation: {config_result['time_us'] / full_time * 100:.0f}%")
  print(f"  make():          {make_result['time_us'] / full_time * 100:.0f}%")
  print(f"  Function call:   {made_time / full_time * 100:.0f}%")
  print("=" * 80)


if __name__ == "__main__":
  print_results()
