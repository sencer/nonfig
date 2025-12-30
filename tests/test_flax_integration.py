from __future__ import annotations

from typing import Any

import pytest

try:
  import flax.linen as nn
  from jax import random
  import jax.numpy as jnp
  from jaxtyping import Array, Float, jaxtyped

  HAS_JAX_FLAX = True
except ImportError:
  HAS_JAX_FLAX = False

from beartype import beartype

from nonfig import Hyper, configurable

pytestmark = pytest.mark.skipif(not HAS_JAX_FLAX, reason="flax/jax not installed")


@configurable
class MyLayer(nn.Module):  # type: ignore
  features: Hyper[int]
  dtype: Hyper[Any] = jnp.float32
  # Verify jaxtyping metadata preservation in Flax field
  kernel_size: Hyper[tuple[int, int]] = (3, 3)

  @nn.compact
  @jaxtyped(typechecker=beartype)
  def __call__(
    self,
    x: Float[Array, "batch in_features"],  # noqa: F722
  ) -> Float[Array, "batch out_features"]:  # noqa: F722
    # Standard Flax pattern
    kernel = self.param(
      "kernel", nn.initializers.lecun_normal(), (x.shape[-1], self.features)
    )
    return jnp.dot(x, kernel)


def test_flax_integration():
  # 1. Check Config
  config_cls = MyLayer.Config
  assert "features" in config_cls.model_fields

  # 2. Instantiate and Make
  conf = config_cls(features=32)
  layer = conf.make()

  assert isinstance(layer, MyLayer)
  assert layer.features == 32

  # 3. Initialize Flax module
  key = random.PRNGKey(0)
  x = jnp.ones((4, 16))
  variables = layer.init(key, x)

  # 4. Apply
  out = layer.apply(variables, x)
  assert out.shape == (4, 32)
  print("Flax integration successful.")


if __name__ == "__main__":
  test_flax_integration()
