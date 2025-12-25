# Performance Benchmarks

This directory contains performance benchmarks for nonfig.

## Benchmarks

### Core Performance (`bench_core_performance.py`)

Comprehensive analysis of nonfig's core overhead:

- **Config creation cost**: Time to create `func.Config(**kwargs)`
- **make() overhead**: Time to call `config.make()`
- **Function call patterns**: Raw function vs configurable function vs made
  function
- **Full pattern analysis**: Breakdown of `Config().make()()` overhead
- **Validation impact**: Performance with constraint validation

### Usage Patterns (`bench_usage_patterns.py`)

Benchmark each usage pattern independently:

- **Function pattern**: Configurable functions
- **Method pattern**: Configurable instance methods
- **Class pattern**: Configurable class constructors
- **Dataclass pattern**: Configurable dataclasses

## Running Benchmarks

```bash
# Run from project root
uv run python benchmarks/bench_core_performance.py
uv run python benchmarks/bench_usage_patterns.py
```

## Understanding Results

### Config Creation

Creating a config object involves Pydantic model instantiation and validation.
Target: \<5µs

### make() Overhead

The `make()` function creates a closure binding the config parameters. Target:
~2.0µs

### Function Call Overhead

There are two ways to call a configurable function:

1. **Direct call**: `config_func(x=50, y=0.75)` - goes through the wrapper each
   time
1. **Made call**: `config.make()()` - pre-bound closure, faster for repeated
   calls

Calling a "made" function has minimal overhead over raw function calls. Target:
\<0.5µs additional overhead

### Full Pattern Cost

`Config().make()()` combines all overhead components. Most cost is in config
creation.

## Performance Targets

Based on typical development machines:

| Operation | Target | Notes |
|-----------|--------|-------|
| Config creation (simple) | <5µs | Pydantic model instantiation |
| Config creation (validated) | <5µs | Constraint checking adds minimal overhead |
| make() | <0.5µs | Closure creation |
| Made function call | <0.5µs overhead | vs raw function |
| Configurable direct call | <1µs overhead | vs raw function (wrapper overhead) |
| Full pattern | <10µs | Config + make + call |

## Tracking Performance

Run benchmarks before and after changes to detect regressions:

```bash
# Before changes
uv run python benchmarks/bench_core_performance.py > before.txt

# Make your changes
# ...

# After changes
uv run python benchmarks/bench_core_performance.py > after.txt

# Compare
diff before.txt after.txt
```

## Adding New Benchmarks

When adding benchmarks:

1. Use `time.perf_counter()` for accurate timing
1. Run enough iterations to get stable results (≥10,000 for µs-level operations)
1. Report in appropriate units (ms for slow, µs for fast)
1. Include iteration count in output
1. Test realistic scenarios
1. Provide baseline/raw comparisons
