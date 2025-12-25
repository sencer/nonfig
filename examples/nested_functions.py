"""Example: Nested Configurable Functions with nonfig.

This example shows how to nest configurable functions inside other configurable
functions or classes using the unified pattern: `fn: inner.Type = inner`.
"""

from __future__ import annotations

from dataclasses import dataclass

from nonfig import Hyper, configurable

# ==============================================================================
# Configurable Functions
# ==============================================================================


@configurable
def normalize(x: float, scale: Hyper[float] = 1.0, offset: Hyper[float] = 0.0) -> float:
  """Normalize a value with configurable scale and offset."""
  return (x * scale) + offset


@configurable
def clip(x: float, min_val: Hyper[float] = 0.0, max_val: Hyper[float] = 1.0) -> float:
  """Clip a value to a configurable range."""
  return max(min_val, min(x, max_val))


# ==============================================================================
# Nesting Functions in Classes
# ==============================================================================


@configurable
@dataclass
class Processor:
  """Processes values using configurable functions.

  Uses the unified pattern: `fn: inner.Type = inner` which works for
  both direct instantiation and via Config.make().
  """

  normalize_fn: normalize.Type = normalize
  clip_fn: clip.Type = clip

  def process(self, x: float) -> float:
    """Apply normalization then clipping."""
    normalized = self.normalize_fn(x)
    return self.clip_fn(normalized)


# ==============================================================================
# Nesting Functions in Functions
# ==============================================================================


@configurable
def process_pipeline(
  data: list[float],
  normalize_fn: normalize.Type = normalize,
  clip_fn: clip.Type = clip,
) -> list[float]:
  """Process a list of values through normalize then clip.

  The nested functions are configurable via Config, or you can pass
  them directly when calling.
  """
  return [clip_fn(normalize_fn(x)) for x in data]


# ==============================================================================
# Usage Examples
# ==============================================================================


def example_direct_instantiation() -> None:
  """Example 1: Direct instantiation uses functions with defaults."""
  print("\n" + "=" * 60)
  print("Example 1: Direct Instantiation")
  print("=" * 60)

  # Direct instantiation - uses the default functions
  processor = Processor()
  result = processor.process(5.0)
  print(f"Processor with defaults: process(5.0) = {result}")
  # normalize: (5.0 * 1.0) + 0.0 = 5.0
  # clip: max(0.0, min(5.0, 1.0)) = 1.0

  # Direct function call
  result2 = process_pipeline([0.5, 1.5, 2.5])
  print(f"Pipeline with defaults: {result2}")
  # Each value normalized then clipped to [0, 1]


def example_via_config() -> None:
  """Example 2: Via Config.make() with defaults."""
  print("\n" + "=" * 60)
  print("Example 2: Via Config.make() with defaults")
  print("=" * 60)

  # Config uses inner.Config() as default, make() converts to BoundFunction
  config = Processor.Config()
  processor = config.make()
  result = processor.process(5.0)
  print(f"Processor via Config: process(5.0) = {result}")


def example_with_overrides() -> None:
  """Example 3: Override nested function configs."""
  print("\n" + "=" * 60)
  print("Example 3: Override nested configs")
  print("=" * 60)

  # Override the nested function configurations
  config = Processor.Config(
    normalize_fn=normalize.Config(scale=2.0, offset=-1.0),
    clip_fn=clip.Config(min_val=0.0, max_val=10.0),
  )
  processor = config.make()

  result = processor.process(5.0)
  print(f"Custom Processor: process(5.0) = {result}")
  # normalize: (5.0 * 2.0) + (-1.0) = 9.0
  # clip: max(0.0, min(9.0, 10.0)) = 9.0

  # Same for functions
  fn_config = process_pipeline.Config(
    normalize_fn=normalize.Config(scale=0.5),
    clip_fn=clip.Config(max_val=0.5),
  )
  fn = fn_config.make()
  result2 = fn([1.0, 2.0, 3.0])
  print(f"Custom Pipeline: {result2}")
  # normalize: [0.5, 1.0, 1.5], clip to [0, 0.5]: [0.5, 0.5, 0.5]


def example_serialization() -> None:
  """Example 4: Serialize nested function configs."""
  print("\n" + "=" * 60)
  print("Example 4: Serialization")
  print("=" * 60)

  config = Processor.Config(
    normalize_fn=normalize.Config(scale=2.0),
    clip_fn=clip.Config(max_val=5.0),
  )

  # Serialize to JSON
  json_str = config.model_dump_json(indent=2)
  print("Config as JSON:")
  print(json_str)

  # Deserialize
  loaded = Processor.Config.model_validate_json(json_str)
  processor = loaded.make()
  result = processor.process(10.0)
  print(f"\nLoaded and executed: process(10.0) = {result}")


if __name__ == "__main__":
  example_direct_instantiation()
  example_via_config()
  example_with_overrides()
  example_serialization()

  print("\n" + "=" * 60)
  print("All examples completed!")
  print("=" * 60)
