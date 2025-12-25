from nonfig import (
  DEFAULT,
  Hyper,
  configurable,
)


def test_model_config_name_collision():
  """Test that a parameter named 'model_config' does not crash Pydantic."""

  @configurable
  class Optimizer:
    def __init__(self, learning_rate: Hyper[float] = 0.01):
      self.learning_rate = learning_rate

  @configurable
  class Model:
    def __init__(
      self,
      hidden_size: Hyper[int] = 128,
      optimizer_config: Hyper[Optimizer.Config] = DEFAULT,
    ):
      self.hidden_size = hidden_size

  # This should now raise ValueError due to "model_config" parameter name
  import pytest

  with pytest.raises(ValueError, match="reserved by Pydantic"):

    @configurable
    def train_model(
      data: list[float],
      epochs: Hyper[int] = 100,
      model_config: Hyper[Model.Config] = DEFAULT,
    ) -> dict:
      return {"epochs": epochs}
