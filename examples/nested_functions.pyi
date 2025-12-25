"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel

def example_direct_instantiation() -> None: ...
def example_via_config() -> None: ...
def example_with_overrides() -> None: ...
def example_serialization() -> None: ...

class normalize:  # noqa: N801
  """Normalize a value with configurable scale and offset.

  Call Arguments:
      x (float)

  Hyperparameters:
      scale (float)
      offset (float)
  """

  class _BoundFunction:
    """Bound function with hyperparameters as attributes."""

    scale: float
    offset: float
    def __call__(self, x: float) -> float: ...

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for normalize.

    Configuration:
        scale (float)
        offset (float)
    """

    scale: float
    offset: float

  class Config(_NCMakeableModel[_BoundFunction]):
    """Configuration class for normalize.

    Normalize a value with configurable scale and offset.

    Configuration:
        scale (float)
        offset (float)
    """

    scale: float
    offset: float
    def __init__(self, *, scale: float = ..., offset: float = ...) -> None: ...
    """Initialize configuration for normalize.

        Configuration:
            scale (float)
            offset (float)
        """

    @override
    def make(self) -> normalize._BoundFunction: ...

  Type = _BoundFunction
  def __call__(self, x: float, scale: float = ..., offset: float = ...) -> float: ...

class clip:  # noqa: N801
  """Clip a value to a configurable range.

  Call Arguments:
      x (float)

  Hyperparameters:
      min_val (float)
      max_val (float)
  """

  class _BoundFunction:
    """Bound function with hyperparameters as attributes."""

    min_val: float
    max_val: float
    def __call__(self, x: float) -> float: ...

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for clip.

    Configuration:
        min_val (float)
        max_val (float)
    """

    min_val: float
    max_val: float

  class Config(_NCMakeableModel[_BoundFunction]):
    """Configuration class for clip.

    Clip a value to a configurable range.

    Configuration:
        min_val (float)
        max_val (float)
    """

    min_val: float
    max_val: float
    def __init__(self, *, min_val: float = ..., max_val: float = ...) -> None: ...
    """Initialize configuration for clip.

        Configuration:
            min_val (float)
            max_val (float)
        """

    @override
    def make(self) -> clip._BoundFunction: ...

  Type = _BoundFunction
  def __call__(self, x: float, min_val: float = ..., max_val: float = ...) -> float: ...

class Processor:
  """Processes values using configurable functions.

  Uses the unified pattern: `fn: inner.Type = inner` which works for
  both direct instantiation and via Config.make().

  Configuration:
      normalize_fn (normalize.Config | normalize.ConfigDict)
      clip_fn (clip.Config | clip.ConfigDict)
  """

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for Processor.

    Configuration:
        normalize_fn (normalize.Config | normalize.ConfigDict)
        clip_fn (clip.Config | clip.ConfigDict)
    """

    normalize_fn: normalize.Config | normalize.ConfigDict
    clip_fn: clip.Config | clip.ConfigDict

  class Config(_NCMakeableModel[Processor]):
    """Configuration class for Processor.

    Processes values using configurable functions.

    Uses the unified pattern: `fn: inner.Type = inner` which works for
    both direct instantiation and via Config.make().

    Configuration:
        normalize_fn (normalize.Config | normalize.ConfigDict)
        clip_fn (clip.Config | clip.ConfigDict)
    """

    normalize_fn: normalize.Config | normalize.ConfigDict
    clip_fn: clip.Config | clip.ConfigDict
    def __init__(
      self,
      *,
      normalize_fn: normalize.Config | normalize.ConfigDict = ...,
      clip_fn: clip.Config | clip.ConfigDict = ...,
    ) -> None: ...
    """Initialize configuration for Processor.

        Configuration:
            normalize_fn (normalize.Config | normalize.ConfigDict)
            clip_fn (clip.Config | clip.ConfigDict)
        """

    @override
    def make(self) -> Processor: ...

  normalize_fn: normalize.Type
  clip_fn: clip.Type
  def __init__(
    self, normalize_fn: normalize.Type = ..., clip_fn: clip.Type = ...
  ) -> None: ...

class process_pipeline:  # noqa: N801
  """Process a list of values through normalize then clip.

  The nested functions are configurable via Config, or you can pass
  them directly when calling.

  Call Arguments:
      data (list[float])

  Hyperparameters:
      normalize_fn (normalize.Config | normalize.ConfigDict)
      clip_fn (clip.Config | clip.ConfigDict)
  """

  class _BoundFunction:
    """Bound function with hyperparameters as attributes."""

    normalize_fn: normalize.Type
    clip_fn: clip.Type
    def __call__(self, data: list[float]) -> list[float]: ...

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for process_pipeline.

    Configuration:
        normalize_fn (normalize.Config | normalize.ConfigDict)
        clip_fn (clip.Config | clip.ConfigDict)
    """

    normalize_fn: normalize.Config | normalize.ConfigDict
    clip_fn: clip.Config | clip.ConfigDict

  class Config(_NCMakeableModel[_BoundFunction]):
    """Configuration class for process_pipeline.

    Process a list of values through normalize then clip.

    The nested functions are configurable via Config, or you can pass
    them directly when calling.

    Configuration:
        normalize_fn (normalize.Config | normalize.ConfigDict)
        clip_fn (clip.Config | clip.ConfigDict)
    """

    normalize_fn: normalize.Config | normalize.ConfigDict
    clip_fn: clip.Config | clip.ConfigDict
    def __init__(
      self,
      *,
      normalize_fn: normalize.Config | normalize.ConfigDict = ...,
      clip_fn: clip.Config | clip.ConfigDict = ...,
    ) -> None: ...
    """Initialize configuration for process_pipeline.

        Configuration:
            normalize_fn (normalize.Config | normalize.ConfigDict)
            clip_fn (clip.Config | clip.ConfigDict)
        """

    @override
    def make(self) -> process_pipeline._BoundFunction: ...

  Type = _BoundFunction
  def __call__(
    self,
    data: list[float],
    normalize_fn: normalize.Type = ...,
    clip_fn: clip.Type = ...,
  ) -> list[float]: ...
