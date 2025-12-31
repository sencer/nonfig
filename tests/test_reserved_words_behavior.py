import pytest

from nonfig import Hyper, configurable


def test_args_reserved_word_protection():
  """
  Realistic scenario: A CLI wrapper where 'args' is a natural parameter name.

  This test verifies that 'args' is now a reserved name to prevent shadowing
  by functools.partial.args, which would return an empty tuple instead of the configured list.
  """
  with pytest.raises(ValueError, match="Parameter 'args' is reserved"):

    @configurable
    def cli_wrapper(
      command: Hyper[str] = "echo", args: Hyper[tuple[str, ...]] = ("hello",)
    ) -> str:
      return f"{command} {' '.join(args)}"


def test_func_reserved_word_protection():
  """Verify 'func' is protected."""
  with pytest.raises(ValueError, match="Parameter 'func' is reserved"):

    @configurable
    def wrapper(func: Hyper[str] = "test"):
      pass


def test_keywords_reserved_word_protection():
  """Verify 'keywords' is protected."""
  with pytest.raises(ValueError, match="Parameter 'keywords' is reserved"):

    @configurable
    def wrapper(keywords: Hyper[dict] = None):
      pass


def test_wrapped_reserved_word_protection():
  """Verify '__wrapped__' is protected."""
  with pytest.raises(ValueError, match="Parameter '__wrapped__' is reserved"):

    @configurable
    def wrapper(__wrapped__: Hyper[str] = "val"):
      pass
