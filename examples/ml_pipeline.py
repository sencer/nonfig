"""Example: Machine Learning Pipeline with nonfig.

This example shows how to use nonfig to configure a complete ML training pipeline
with nested configurations for optimizer, model, and data processing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
import random

from nonfig import (
  DEFAULT,
  Ge,
  Gt,
  Hyper,
  Le,
  Lt,
  configurable,
)

# ==============================================================================
# Data Processing
# ==============================================================================


@configurable
@dataclass
class DataPreprocessor:
  """Preprocesses input data with normalization and augmentation."""

  normalize: Hyper[bool] = True
  augment: Hyper[bool] = False
  train_test_split: Hyper[float, Gt[0.0], Lt[1.0]] = 0.8

  def process(self, data: list[float]) -> dict[str, list[float]]:
    """Split and process data."""
    split_idx = int(len(data) * self.train_test_split)

    train_data = data[:split_idx]
    test_data = data[split_idx:]

    if self.normalize:
      mean = sum(train_data) / len(train_data)
      train_data = [x - mean for x in train_data]
      test_data = [x - mean for x in test_data]

    if self.augment:
      # Simple augmentation: add noise
      train_data = [x + random.uniform(-0.1, 0.1) for x in train_data]  # noqa: S311

    return {"train": train_data, "test": test_data}


# ==============================================================================
# Optimizer
# ==============================================================================


@configurable
@dataclass
class Optimizer:
  """Gradient descent optimizer with momentum."""

  learning_rate: Hyper[float, Gt[0.0], Le[1.0]] = 0.01
  momentum: Hyper[float, Ge[0.0], Le[1.0]] = 0.9
  weight_decay: Hyper[float, Ge[0.0]] = 0.0001

  def step(self, loss: float) -> float:
    """Simulate an optimization step."""
    return loss * (1 - self.learning_rate) + self.weight_decay


# ==============================================================================
# Model
# ==============================================================================


@configurable
@dataclass
class Model:
  """Simple neural network model."""

  hidden_size: Hyper[int, Ge[1], Le[1000]] = 128
  num_layers: Hyper[int, Ge[1], Le[10]] = 3
  dropout: Hyper[float, Ge[0.0], Lt[1.0]] = 0.1
  optimizer_config: Hyper[Optimizer] = DEFAULT

  def __post_init__(self) -> None:
    """Initialize the optimizer."""
    # After make(), optimizer_config is already an Optimizer instance!
    # Use underscore to mark as "set at runtime, not in __init__"
    object.__setattr__(self, "_optimizer", self.optimizer_config)

  @property
  def optimizer(self) -> Optimizer:
    """Get the optimizer (set in __post_init__)."""
    return self._optimizer  # type: ignore[attr-defined]

  def forward(self, data: list[float]) -> float:
    """Simulate forward pass."""
    # Simplified computation
    return sum(data) / self.hidden_size

  def train_step(self, data: list[float]) -> dict[str, float]:
    """Simulate a training step."""
    output = self.forward(data)
    loss = abs(output - 1.0)  # Simplified loss
    new_loss = self.optimizer.step(loss)

    return {"loss": loss, "optimized_loss": new_loss}


# ==============================================================================
# Training Pipeline
# ==============================================================================


@configurable
def train_model(
  data: list[float],
  epochs: Hyper[int, Ge[1], Le[1000]] = 100,
  batch_size: Hyper[int, Ge[1], Le[512]] = 32,
  preprocessor: Hyper[DataPreprocessor] = DEFAULT,
  model: Hyper[Model] = DEFAULT,
) -> dict[str, object]:
  """Complete training pipeline.

  Args:
    data: Raw training data
    epochs: Number of training epochs
    batch_size: Batch size for training
    preprocessor: Data preprocessing (nested configs are auto-instantiated)
    model: Model architecture and optimizer

  Returns:
    Training results with final loss and metrics
  """
  # Nested Hyper params are already instantiated by make()
  # No need to call .make() on them!

  # Preprocess data
  processed = preprocessor.process(data)
  train_data = processed["train"]
  test_data = processed["test"]

  # Training loop
  print(f"Training for {epochs} epochs with batch size {batch_size}")
  print(f"Model: {model.num_layers} layers, {model.hidden_size} hidden units")
  print(
    f"Optimizer: LR={model.optimizer.learning_rate}, "
    + f"momentum={model.optimizer.momentum}"
  )

  losses: list[float] = []
  for epoch in range(epochs):
    # Simplified batch processing
    epoch_losses: list[float] = []
    for i in range(0, len(train_data), batch_size):
      batch = train_data[i : i + batch_size]
      result = model.train_step(batch)
      epoch_losses.append(result["optimized_loss"])

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(avg_loss)

    if epoch % 10 == 0:
      print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

  # Evaluate on test set
  test_output = model.forward(test_data)
  test_loss = abs(test_output - 1.0)

  return {
    "final_train_loss": losses[-1],
    "test_loss": test_loss,
    "total_epochs": epochs,
    "model_config": {
      "hidden_size": model.hidden_size,
      "num_layers": model.num_layers,
      "dropout": model.dropout,
    },
  }


# ==============================================================================
# Usage Examples
# ==============================================================================


def example_basic_usage() -> None:
  """Example 1: Basic usage with default configuration."""
  print("\n" + "=" * 80)
  print("Example 1: Basic Usage")
  print("=" * 80)

  # Generate sample data
  data = list(range(100))

  # Create config with defaults
  config = train_model.Config()
  fn = config.make()
  result = fn(data)
  print(f"\nResults: {result}")


def example_custom_config() -> None:
  """Example 2: Custom configuration."""
  print("\n" + "=" * 80)
  print("Example 2: Custom Configuration")
  print("=" * 80)

  data = list(range(100))

  # Create custom configuration
  config = train_model.Config(
    epochs=50,
    batch_size=16,
    preprocessor=DataPreprocessor.Config(
      normalize=True,
      augment=True,
      train_test_split=0.7,
    ),
    model=Model.Config(
      hidden_size=256,
      num_layers=5,
      dropout=0.2,
      optimizer_config=Optimizer.Config(
        learning_rate=0.001,
        momentum=0.95,
        weight_decay=0.0001,
      ),
    ),
  )

  # Execute with custom config
  fn = config.make()
  result = fn(data)
  print(f"\nResults: {result}")


def example_config_serialization() -> None:
  """Example 3: Configuration serialization for experiment tracking."""
  print("\n" + "=" * 80)
  print("Example 3: Configuration Serialization")
  print("=" * 80)

  # Create configuration
  config = train_model.Config(
    epochs=20,
    batch_size=32,
    model=Model.Config(
      hidden_size=128,
      num_layers=3,
      optimizer_config=Optimizer.Config(learning_rate=0.01),
    ),
  )

  # Serialize to JSON
  config_json = config.model_dump_json(indent=2)
  print("\nSerialized configuration:")
  print(config_json)

  # Save to file
  with pathlib.Path("/tmp/experiment_config.json").open("w", encoding="utf-8") as f:
    f.write(config_json)
  print("\nConfiguration saved to /tmp/experiment_config.json")

  # Load from file
  with pathlib.Path("/tmp/experiment_config.json").open(encoding="utf-8") as f:
    config_dict = json.load(f)

  loaded_config = train_model.Config(**config_dict)
  print("\nConfiguration loaded successfully!")

  # Run experiment with loaded config
  data = list(range(100))
  fn = loaded_config.make()
  result = fn(data)
  print(f"\nResults: {result}")


def example_hyperparameter_search() -> None:
  """Example 4: Simple hyperparameter search."""
  print("\n" + "=" * 80)
  print("Example 4: Hyperparameter Search")
  print("=" * 80)

  data = list(range(100))

  # Try different learning rates
  learning_rates = [0.001, 0.01, 0.1]
  results: list[tuple[float, object]] = []

  for lr in learning_rates:
    print(f"\nTrying learning_rate={lr}")

    config = train_model.Config(
      epochs=20,
      model=Model.Config(optimizer_config=Optimizer.Config(learning_rate=lr)),
    )

    fn = config.make()
    result = fn(data)
    results.append((lr, result["final_train_loss"]))

  print("\n" + "-" * 80)
  print("Hyperparameter Search Results:")
  for lr, loss in results:
    print(f"  learning_rate={lr:5.3f} -> final_loss={loss:.6f}")

  best_lr, best_loss = min(results, key=lambda x: x[1])
  print(f"\nBest learning_rate: {best_lr} (loss={best_loss:.6f})")


if __name__ == "__main__":
  example_basic_usage()
  example_custom_config()
  example_config_serialization()
  example_hyperparameter_search()

  print("\n" + "=" * 80)
  print("All examples completed successfully!")
  print("=" * 80)
