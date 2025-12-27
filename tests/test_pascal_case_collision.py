from nonfig.generation import _to_pascal_case


def test_pascal_case_collision():
  # New behavior: preserves leading underscores
  assert _to_pascal_case("my_func") == "MyFunc"
  assert _to_pascal_case("_my_func") == "_MyFunc"
  assert _to_pascal_case("__my_func") == "__MyFunc"

  # Verify no collision
  assert _to_pascal_case("my_func") != _to_pascal_case("_my_func")


def test_pascal_case_desired_behavior():
  # If we want to support strict preservation of underscores
  # expected:
  # _my_func -> _MyFunc
  # __my_func -> __MyFunc

  # We can check what it resolves to currently vs what we might want
  assert (
    _to_pascal_case("camelCase") == "Camelcase"
  )  # Current behavior on camelCase is also weird: split('_') does nothing so it just capitalizes the first letter.
