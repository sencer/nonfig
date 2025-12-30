from __future__ import annotations

import pytest

try:
  from jax import numpy as jnp
  from jaxtyping import Array, Float, jaxtyped

  HAS_JAX = True
except ImportError:
  HAS_JAX = False

from beartype import beartype

from nonfig import Hyper, configurable

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="jaxtyping/jax not installed")

# Mock dimensions for jaxtyping strings to avoid F821
b, c = "b", "c"


@configurable
@jaxtyped(typechecker=beartype)
def scale_tensor(
  tensor: Float[Array, "b c"],  # noqa: F722
  scale: Hyper[float] = 1.0,
) -> Float[Array, "b c"]:  # noqa: F722
  return tensor * scale


@configurable
@jaxtyped(typechecker=beartype)
def add_bias(
  tensor: Float[Array, "b c"],  # noqa: F722
  bias: Hyper[Float[Array, c]],
) -> Float[Array, "b c"]:  # noqa: F722
  return tensor + bias


def test_jaxtyping_valid():
  x = jnp.zeros((2, 3))
  res = scale_tensor(x, scale=2.0)
  assert res.shape == (2, 3)

  config_cls = scale_tensor.Config
  configured_fn = config_cls(scale=3.0).make()
  res2 = configured_fn(x)
  assert jnp.allclose(res2, x * 3.0)


def test_jaxtyping_invalid_shape():
  x_bad = jnp.zeros((2, 3, 4))

  with pytest.raises(Exception):  # noqa: B017
    scale_tensor(x_bad, scale=1.0)

  configured_fn = scale_tensor.Config().make()
  with pytest.raises(Exception):  # noqa: B017
    configured_fn(x_bad)


def test_hyper_with_jaxtyping_metadata():
  x = jnp.zeros((2, 3))
  bias = jnp.ones((3,))

  # Verify extraction
  config_cls = add_bias.Config
  assert "bias" in config_cls.model_fields

  # Valid
  fn = config_cls(bias=bias).make()
  res = fn(x)
  assert res.shape == (2, 3)

  # Invalid bias shape caught at runtime
  bias_bad = jnp.ones((4,))
  fn_bad = config_cls(bias=bias_bad).make()
  with pytest.raises(Exception):  # noqa: B017
    fn_bad(x)
