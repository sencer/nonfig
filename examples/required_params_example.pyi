"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import TypedDict, override

from nonfig import DEFAULT, MakeableModel

class train_model:  # noqa: N801
  """Train a model with the given hyperparameters.

  Configuration:
      learning_rate (float)
      epochs (int)
  """

  class _BoundFunction:
    """Bound function with hyperparameters as attributes."""

    learning_rate: float
    epochs: int
    def __call__(
      self,
    ) -> dict[str, float]: ...

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for train_model.

    Configuration:
        learning_rate (float)
        epochs (int)
    """

    learning_rate: float
    epochs: int

  class Config(MakeableModel[_BoundFunction]):
    """Configuration class for train_model.

    Train a model with the given hyperparameters.

    Configuration:
        learning_rate (float)
        epochs (int)
    """

    learning_rate: float
    epochs: int
    def __init__(self, *, learning_rate: float, epochs: int = ...) -> None: ...
    """Initialize configuration for train_model.

        Configuration:
            learning_rate (float)
            epochs (int)
        """

    @override
    def make(self) -> train_model._BoundFunction: ...

  Type = _BoundFunction
  def __call__(self, learning_rate: float, epochs: int = ...) -> dict[str, float]: ...

class OptionalOptimizer:
  """Optimizer where all params have defaults - works with DEFAULT.

  Configuration:
      learning_rate (float)
      momentum (float)
  """

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for OptionalOptimizer.

    Configuration:
        learning_rate (float)
        momentum (float)
    """

    learning_rate: float
    momentum: float

  class Config(MakeableModel[OptionalOptimizer]):
    """Configuration class for OptionalOptimizer.

    Optimizer where all params have defaults - works with DEFAULT.

    Configuration:
        learning_rate (float)
        momentum (float)
    """

    learning_rate: float
    momentum: float
    def __init__(
      self, *, learning_rate: float = ..., momentum: float = ...
    ) -> None: ...
    """Initialize configuration for OptionalOptimizer.

        Configuration:
            learning_rate (float)
            momentum (float)
        """

    @override
    def make(self) -> OptionalOptimizer: ...

  learning_rate: float
  momentum: float
  def __init__(self, learning_rate: float = ..., momentum: float = ...) -> None: ...

class Model:
  """Model using nested config with DEFAULT.

  Configuration:
      optimizer (OptionalOptimizer.Config | OptionalOptimizer.ConfigDict)
      hidden_size (int)
  """

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for Model.

    Configuration:
        optimizer (OptionalOptimizer.Config | OptionalOptimizer.ConfigDict)
        hidden_size (int)
    """

    optimizer: OptionalOptimizer.Config | OptionalOptimizer.ConfigDict
    hidden_size: int

  class Config(MakeableModel[Model]):
    """Configuration class for Model.

    Model using nested config with DEFAULT.

    Configuration:
        optimizer (OptionalOptimizer.Config | OptionalOptimizer.ConfigDict)
        hidden_size (int)
    """

    optimizer: OptionalOptimizer.Config | OptionalOptimizer.ConfigDict
    hidden_size: int
    def __init__(
      self,
      *,
      optimizer: OptionalOptimizer.Config | OptionalOptimizer.ConfigDict = DEFAULT,
      hidden_size: int = ...,
    ) -> None: ...
    """Initialize configuration for Model.

        Configuration:
            optimizer (OptionalOptimizer.Config | OptionalOptimizer.ConfigDict)
            hidden_size (int)
        """

    @override
    def make(self) -> Model: ...

  optimizer: OptionalOptimizer
  hidden_size: int
  def __init__(
    self, optimizer: OptionalOptimizer = DEFAULT, hidden_size: int = ...
  ) -> None: ...

class RequiredOptimizer:
  """Optimizer with required learning rate.

  Configuration:
      learning_rate (float)
      momentum (float)
  """

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for RequiredOptimizer.

    Configuration:
        learning_rate (float)
        momentum (float)
    """

    learning_rate: float
    momentum: float

  class Config(MakeableModel[RequiredOptimizer]):
    """Configuration class for RequiredOptimizer.

    Optimizer with required learning rate.

    Configuration:
        learning_rate (float)
        momentum (float)
    """

    learning_rate: float
    momentum: float
    def __init__(self, *, learning_rate: float, momentum: float = ...) -> None: ...
    """Initialize configuration for RequiredOptimizer.

        Configuration:
            learning_rate (float)
            momentum (float)
        """

    @override
    def make(self) -> RequiredOptimizer: ...

  learning_rate: float
  momentum: float
  def __init__(self, learning_rate: float, momentum: float = ...) -> None: ...

class Pipeline:
  """Pipeline that requires an optimizer config (no default).

  Configuration:
      optimizer (RequiredOptimizer.Config | RequiredOptimizer.ConfigDict)
      learning_rate_scale (float)
  """

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for Pipeline.

    Configuration:
        optimizer (RequiredOptimizer.Config | RequiredOptimizer.ConfigDict)
        learning_rate_scale (float)
    """

    optimizer: RequiredOptimizer.Config | RequiredOptimizer.ConfigDict
    learning_rate_scale: float

  class Config(MakeableModel[Pipeline]):
    """Configuration class for Pipeline.

    Pipeline that requires an optimizer config (no default).

    Configuration:
        optimizer (RequiredOptimizer.Config | RequiredOptimizer.ConfigDict)
        learning_rate_scale (float)
    """

    optimizer: RequiredOptimizer.Config | RequiredOptimizer.ConfigDict
    learning_rate_scale: float
    def __init__(
      self,
      *,
      optimizer: RequiredOptimizer.Config | RequiredOptimizer.ConfigDict,
      learning_rate_scale: float = ...,
    ) -> None: ...
    """Initialize configuration for Pipeline.

        Configuration:
            optimizer (RequiredOptimizer.Config | RequiredOptimizer.ConfigDict)
            learning_rate_scale (float)
        """

    @override
    def make(self) -> Pipeline: ...

  optimizer: RequiredOptimizer
  learning_rate_scale: float
  def __init__(
    self, optimizer: RequiredOptimizer, learning_rate_scale: float = ...
  ) -> None: ...
