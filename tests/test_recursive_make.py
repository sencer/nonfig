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
