"""Example demonstrating required Hyper parameters (no defaults).

This example shows how to create @configurable functions and classes
with required parameters that must be provided at config creation time.
"""

from __future__ import annotations

from typing import ClassVar

from nonfig import DEFAULT, Ge, Gt, Hyper, MakeableModel, configurable

# =============================================================================
# Basic required parameters
# =============================================================================


@configurable
def train_model(
  learning_rate: Hyper[float, Gt[0.0]],  # Required - no default
  epochs: Hyper[int, Ge[1]] = 100,  # Optional - has default
) -> dict[str, float]:
  """Train a model with the given hyperparameters."""
  return {"learning_rate": learning_rate, "epochs": epochs}


# =============================================================================
# Nested config with all-optional params
# =============================================================================


@configurable
class OptionalOptimizer:
  """Optimizer where all params have defaults - works with DEFAULT."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(
    self,
    learning_rate: Hyper[float, Gt[0.0]] = 0.01,  # Has default
    momentum: Hyper[float, Ge[0.0]] = 0.9,  # Has default
  ) -> None:
    self.learning_rate = learning_rate
    self.momentum = momentum


@configurable
class Model:
  """Model using nested config with DEFAULT."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(
    self,
    optimizer: Hyper[OptionalOptimizer] = DEFAULT,  # Works - all params optional
    hidden_size: Hyper[int, Ge[1]] = 128,
  ) -> None:
    # After make(), optimizer is already an OptionalOptimizer instance!
    self.optimizer = optimizer
    self.hidden_size = hidden_size


# =============================================================================
# Required nested configuration
# =============================================================================


@configurable
class RequiredOptimizer:
  """Optimizer with required learning rate."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(
    self,
    learning_rate: Hyper[float, Gt[0.0]],  # Required - no default
    momentum: Hyper[float, Ge[0.0]] = 0.9,  # Optional
  ) -> None:
    self.learning_rate = learning_rate
    self.momentum = momentum


@configurable
class Pipeline:
  """Pipeline that requires an optimizer config (no default)."""

  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(
    self,
    optimizer: Hyper[RequiredOptimizer],  # Required - no DEFAULT possible!
    learning_rate_scale: Hyper[float] = 1.0,
  ) -> None:
    # optimizer is already a RequiredOptimizer instance after make()
    self.optimizer = optimizer
    self.learning_rate_scale = learning_rate_scale
