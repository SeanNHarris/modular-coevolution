#  Copyright 2026 BONSAI Lab at Auburn University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2026, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

from collections.abc import Hashable
from typing import TypeVar, Sequence, Optional, Callable

T = TypeVar('T', bound=Hashable)


def tree_to_string(
        root: T,
        children: dict[T, Sequence[T]],
        node_to_string: Optional[Callable[[T], str]] = None,
        skip_repeated: bool = False,
        displayed: Optional[set[T]] = None
) -> str:
    """
    Prints a fancy string representation of a tree.

    Any hashable type can be used as a node.
    If you have a tree of non-hashable objects, you can pass a tree of their IDs,
        and use the node_to_string parameter to get the correct string representation for each ID.

    Args:
        root: The root node of the tree.
        children: A dictionary mapping each node to a sequence of its children.
        node_to_string: A function that takes a node and returns a string representation.
            If omitted, uses `str(node)`.
        skip_repeated: If True, only the first occurrence of each node will have its children displayed.
            Use this if the tree is not a true tree and has can converge in places.
            The default `node_to_string` will append " (displayed above)" to repeated nodes in this case.
        displayed: If provided, this set will be updated with all nodes that have been displayed in the tree so far.
            This is mainly for custom `node_to_string` functions when `skip_repeated` is used.

    Returns:
        A string representation of the tree.
    """
    if displayed is None:
        displayed = set()

    if node_to_string is None:
        def node_to_string(node: T) -> str:
            if skip_repeated and node in displayed:
                return f"{node} (displayed above)"
            else:
                return f"{node}"

    def _build_tree(node: T, depth: int, last_child: bool, pipe_depths: list[int]) -> str:
        """
        Recursively build the tree at the current node.

        Args:
            node: The current node for this line.
            depth: The depth of this genotype in the tree.
            last_child: Whether this genotype is the last child (tree-wise) of its parent, for drawing.
            pipe_depths: Which depths should have a vertical pipe drawn because they are in between children.

        Returns:
            A string representation of the tree rooted at this node, in the context of the whole tree.
        """
        line = ""
        if depth > 0:
            for space_depth in range(depth - 1):
                if space_depth in pipe_depths:
                    line += "│ "
                else:
                    line += "  "
            if last_child:
                line += "└╴"
            else:
                line += "├╴"

        new_node = node not in displayed

        line += node_to_string(node)

        substrings = [line]
        displayed.add(node)

        if new_node and node in children:
            # Skips if repeated or depth-limited.
            child_nodes = children[node]
            for index, child in enumerate(child_nodes):
                last_child = index == len(child_nodes) - 1
                sub_pipe_depths = pipe_depths.copy()
                if not last_child:
                    sub_pipe_depths.append(depth)
                substrings.append(_build_tree(child, depth + 1, last_child, sub_pipe_depths))

        return "\n".join(substrings)

    return _build_tree(root, depth=0, last_child=True, pipe_depths=[])