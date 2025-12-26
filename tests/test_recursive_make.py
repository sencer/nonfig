"""Test recursive make behavior and strict Config types."""

from __future__ import annotations

from typing import ClassVar

from nonfig import DEFAULT, Hyper, MakeableModel, configurable


@configurable
class Leaf:
  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(self, value: int = 1):
    self.value = value


@configurable
class Node:
  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(self, leaf: Hyper[Leaf] = DEFAULT):
    self.leaf = leaf


@configurable
class Tree:
  Config: ClassVar[type[MakeableModel[object]]]

  def __init__(self, node: Hyper[Node] = DEFAULT):
    self.node = node


def test_recursive_make_defaults() -> None:
  """Test that make() recursively instantiates defaults."""
  config = Tree.Config()
  tree = config.make()

  assert isinstance(tree, Tree)
  assert isinstance(tree.node, Node)
  assert isinstance(tree.node.leaf, Leaf)
  assert tree.node.leaf.value == 1


def test_recursive_make_overrides() -> None:
  """Test that make() recursively instantiates with overrides."""
  config = Tree.Config(node=Node.Config(leaf=Leaf.Config(value=10)))
  tree = config.make()

  assert isinstance(tree, Tree)
  assert isinstance(tree.node, Node)
  assert isinstance(tree.node.leaf, Leaf)
  assert tree.node.leaf.value == 10

from collections import deque, OrderedDict
from collections.abc import Sequence, Mapping
from typing import Any, ClassVar
from nonfig import configurable, MakeableModel, Hyper, DEFAULT
from nonfig.models import recursive_make

@configurable
class Item:
    Config: ClassVar[type[MakeableModel[object]]]
    def __init__(self, value: int = 1):
        self.value = value

class MySeq(Sequence):
    def __init__(self, items):
        self.items = items
    def __getitem__(self, index):
        return self.items[index]
    def __len__(self):
        return len(self.items)

class MyMap(Mapping):
    def __init__(self, data):
        self._data = data
    def __getitem__(self, key):
        return self._data[key]
    def __iter__(self):
        return iter(self._data)
    def __len__(self):
        return len(self._data)

def test_recursive_make_generic_sequence() -> None:
    """Test recursive_make with generic Sequence types."""
    config = Item.Config(value=10)
    
    # Test deque (stdlib) - conceptually a Sequence but check registration
    # Note: deque registers as Sequence but we want to confirm it works
    dq = deque([config])
    made_dq = recursive_make(dq)
    assert isinstance(made_dq, list)  # Should convert to list
    assert isinstance(made_dq[0], Item)
    assert made_dq[0].value == 10

    # Test custom Sequence
    seq = MySeq([config])
    made_seq = recursive_make(seq)
    assert isinstance(made_seq, list)
    assert isinstance(made_seq[0], Item)
    assert made_seq[0].value == 10

def test_recursive_make_generic_mapping() -> None:
    """Test recursive_make with generic Mapping types."""
    config = Item.Config(value=20)
    
    # Test OrderedDict
    od = OrderedDict([("a", config)])
    made_od = recursive_make(od)
    # Our fallback logic currently returns dict for generic Mapping
    # Note: OrderedDict IS a dict subclass, so it hits 'v_type is dict' check? No, 'v_type is dict' is exact check.
    # Ah, v_type is type(value). type(OrderedDict()) is OrderedDict.
    # So it falls through to MappingABC check.
    assert isinstance(made_od, dict) 
    assert isinstance(made_od["a"], Item)
    assert made_od["a"].value == 20

    # Test custom Mapping
    m = MyMap({"b": config})
    made_m = recursive_make(m)
    assert isinstance(made_m, dict)
    assert isinstance(made_m["b"], Item)
    assert made_m["b"].value == 20

def test_recursive_make_excludes_strings() -> None:
    """Ensure strings/bytes are not treated as Sequences."""
    s = "hello"
    assert recursive_make(s) is s
    
    b = b"bytes"
    assert recursive_make(b) is b
