"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import TypedDict, override

from nonfig import DEFAULT, MakeableModel

def example_basic_usage() -> None: ...
def example_custom_config() -> None: ...
def example_config_serialization() -> None: ...
def example_hyperparameter_search() -> None: ...

class DataPreprocessor:
  """Preprocesses input data with normalization and augmentation.

  Configuration:
      normalize (bool)
      augment (bool)
      train_test_split (float)
  """

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for DataPreprocessor.

    Configuration:
        normalize (bool)
        augment (bool)
        train_test_split (float)
    """

    normalize: bool
    augment: bool
    train_test_split: float

  class Config(MakeableModel[DataPreprocessor]):
    """Configuration class for DataPreprocessor.

    Preprocesses input data with normalization and augmentation.

    Configuration:
        normalize (bool)
        augment (bool)
        train_test_split (float)
    """

    normalize: bool
    augment: bool
    train_test_split: float
    def __init__(
      self, *, normalize: bool = ..., augment: bool = ..., train_test_split: float = ...
    ) -> None: ...
    """Initialize configuration for DataPreprocessor.

        Configuration:
            normalize (bool)
            augment (bool)
            train_test_split (float)
        """

    @override
    def make(self) -> DataPreprocessor: ...

  normalize: bool
  augment: bool
  train_test_split: float
  def __init__(
    self, normalize: bool = ..., augment: bool = ..., train_test_split: float = ...
  ) -> None: ...

class Optimizer:
  """Gradient descent optimizer with momentum.

  Configuration:
      learning_rate (float)
      momentum (float)
      weight_decay (float)
  """

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for Optimizer.

    Configuration:
        learning_rate (float)
        momentum (float)
        weight_decay (float)
    """

    learning_rate: float
    momentum: float
    weight_decay: float

  class Config(MakeableModel[Optimizer]):
    """Configuration class for Optimizer.

    Gradient descent optimizer with momentum.

    Configuration:
        learning_rate (float)
        momentum (float)
        weight_decay (float)
    """

    learning_rate: float
    momentum: float
    weight_decay: float
    def __init__(
      self,
      *,
      learning_rate: float = ...,
      momentum: float = ...,
      weight_decay: float = ...,
    ) -> None: ...
    """Initialize configuration for Optimizer.

        Configuration:
            learning_rate (float)
            momentum (float)
            weight_decay (float)
        """

    @override
    def make(self) -> Optimizer: ...

  learning_rate: float
  momentum: float
  weight_decay: float
  def __init__(
    self, learning_rate: float = ..., momentum: float = ..., weight_decay: float = ...
  ) -> None: ...

class Model:
  """Simple neural network model.

  Configuration:
      hidden_size (int)
      num_layers (int)
      dropout (float)
      optimizer_config (Optimizer.Config | Optimizer.ConfigDict)
  """

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for Model.

    Configuration:
        hidden_size (int)
        num_layers (int)
        dropout (float)
        optimizer_config (Optimizer.Config | Optimizer.ConfigDict)
    """

    hidden_size: int
    num_layers: int
    dropout: float
    optimizer_config: Optimizer.Config | Optimizer.ConfigDict

  class Config(MakeableModel[Model]):
    """Configuration class for Model.

    Simple neural network model.

    Configuration:
        hidden_size (int)
        num_layers (int)
        dropout (float)
        optimizer_config (Optimizer.Config | Optimizer.ConfigDict)
    """

    hidden_size: int
    num_layers: int
    dropout: float
    optimizer_config: Optimizer.Config | Optimizer.ConfigDict
    def __init__(
      self,
      *,
      hidden_size: int = ...,
      num_layers: int = ...,
      dropout: float = ...,
      optimizer_config: Optimizer.Config | Optimizer.ConfigDict = DEFAULT,
    ) -> None: ...
    """Initialize configuration for Model.

        Configuration:
            hidden_size (int)
            num_layers (int)
            dropout (float)
            optimizer_config (Optimizer.Config | Optimizer.ConfigDict)
        """

    @override
    def make(self) -> Model: ...

  hidden_size: int
  num_layers: int
  dropout: float
  optimizer_config: Optimizer
  def __init__(
    self,
    hidden_size: int = ...,
    num_layers: int = ...,
    dropout: float = ...,
    optimizer_config: Optimizer = DEFAULT,
  ) -> None: ...

class train_model:  # noqa: N801
  """Complete training pipeline.

  Args:
    data: Raw training data
    epochs: Number of training epochs
    batch_size: Batch size for training
    preprocessor: Data preprocessing (nested configs are auto-instantiated)
    model: Model architecture and optimizer

  Returns:
    Training results with final loss and metrics

  Call Arguments:
      data (list[float])

  Hyperparameters:
      epochs (int)
      batch_size (int)
      preprocessor (DataPreprocessor.Config | DataPreprocessor.ConfigDict)
      model (Model.Config | Model.ConfigDict)
  """

  class _BoundFunction:
    """Bound function with hyperparameters as attributes."""

    epochs: int
    batch_size: int
    preprocessor: DataPreprocessor
    model: Model
    def __call__(self, data: list[float]) -> dict[str, object]: ...

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for train_model.

    Configuration:
        epochs (int)
        batch_size (int)
        preprocessor (DataPreprocessor.Config | DataPreprocessor.ConfigDict)
        model (Model.Config | Model.ConfigDict)
    """

    epochs: int
    batch_size: int
    preprocessor: DataPreprocessor.Config | DataPreprocessor.ConfigDict
    model: Model.Config | Model.ConfigDict

  class Config(MakeableModel[_BoundFunction]):
    """Configuration class for train_model.

    Complete training pipeline.

    Args:
      data: Raw training data
      epochs: Number of training epochs
      batch_size: Batch size for training
      preprocessor: Data preprocessing (nested configs are auto-instantiated)
      model: Model architecture and optimizer

    Returns:
      Training results with final loss and metrics

    Configuration:
        epochs (int)
        batch_size (int)
        preprocessor (DataPreprocessor.Config | DataPreprocessor.ConfigDict)
        model (Model.Config | Model.ConfigDict)
    """

    epochs: int
    batch_size: int
    preprocessor: DataPreprocessor.Config | DataPreprocessor.ConfigDict
    model: Model.Config | Model.ConfigDict
    def __init__(
      self,
      *,
      epochs: int = ...,
      batch_size: int = ...,
      preprocessor: DataPreprocessor.Config | DataPreprocessor.ConfigDict = DEFAULT,
      model: Model.Config | Model.ConfigDict = DEFAULT,
    ) -> None: ...
    """Initialize configuration for train_model.

        Configuration:
            epochs (int)
            batch_size (int)
            preprocessor (DataPreprocessor.Config | DataPreprocessor.ConfigDict)
            model (Model.Config | Model.ConfigDict)
        """

    @override
    def make(self) -> train_model._BoundFunction: ...

  Type = _BoundFunction
  def __call__(
    self,
    data: list[float],
    epochs: int = ...,
    batch_size: int = ...,
    preprocessor: DataPreprocessor = DEFAULT,
    model: Model = DEFAULT,
  ) -> dict[str, object]: ...
