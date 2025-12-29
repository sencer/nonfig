"""
Example demonstrating inheritance limitations in nonfig.

nonfig relies on static analysis of function signatures (using `inspect`) to generate
configuration classes. It does not inspect the runtime behavior of `super().__init__`.
Therefore, if a subclass accepts arguments via `**kwargs` and passes them to the base class,
those arguments will NOT appear in the subclass's Config.
"""

from typing import Any

from nonfig import configurable


# 1. The Base Class
# This works fine: config has 'val'.
class Base:
  def __init__(self, val: int = 1) -> None:
    self.val = val


# 2. The "Broken" Subclass
# Here, we accept **kwargs and pass them up.
# Syntactically valid Python, but nonfig can't read 'val' from `**kwargs`.
@configurable
class ImplicitSubclass(Base):
  def __init__(self, extra: int = 2, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self.extra = extra


# 3. The "Working" Subclass
# We explicitly repeat the arguments we want to expose.
@configurable
class ExplicitSubclass(Base):
  def __init__(self, val: int = 1, extra: int = 2) -> None:
    super().__init__(val=val)
    self.extra = extra


def main() -> None:
  print("--- Implicit Subclass (Broken) ---")
  # implicit_config has 'extra', but 'val' is missing!
  try:
    # This will fail or simply not set val in the config way
    conf = ImplicitSubclass.Config(extra=10, val=5)
  except Exception as e:  # noqa: BLE001
    print(f"Error creating config: {e}")
    print(
      "Reason: ImplicitSubclass.Config only knows about arguments explicitly defined in __init__."
    )
    print(f"Known fields: {ImplicitSubclass.Config.model_fields.keys()}")

  print("\n--- Explicit Subclass (Working) ---")
  # This works as expected
  conf = ExplicitSubclass.Config(extra=10, val=5)
  obj = conf.make()
  print(f"Success! val={obj.val}, extra={obj.extra}")


if __name__ == "__main__":
  main()
