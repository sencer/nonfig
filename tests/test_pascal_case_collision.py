from nonfig.generation import _to_pascal_case


def test_pascal_case_collision():
  # New behavior: preserves leading underscores
  assert _to_pascal_case("my_func") == "MyFunc"
  assert _to_pascal_case("_my_func") == "_MyFunc"
  assert _to_pascal_case("__my_func") == "__MyFunc"

  # Verify no collision
  assert _to_pascal_case("my_func") != _to_pascal_case("_my_func")


def test_pascal_case_desired_behavior():
  """
  Verify that _to_pascal_case handles existing camel/pascal case gracefully.
  """
  from nonfig.generation import _to_pascal_case

  assert _to_pascal_case("my_func") == "MyFunc"
  assert _to_pascal_case("CamelCase") == "CamelCase"
  assert _to_pascal_case("_private_var") == "_PrivateVar"
