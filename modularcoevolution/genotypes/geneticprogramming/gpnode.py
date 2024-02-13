# uncompyle6 version 2.13.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 25 2016, 22:18:55) [MSC v.1900 64 bit (AMD64)]
# Embedded file name: S:\My Documents\ceads\ceads\CEADS-LIN\geneticprogramming\GPNodes.py
# Compiled at: 2017-03-21 14:16:04
# Size of source mod 2**32: 7822 bytes
from functools import cache
from typing import Callable, Any, Union, Optional, Generator

from modularcoevolution.genotypes.geneticprogramming.gpnodetyperegistry import GPNodeTypeRegistry

import random

NodeType = int
NodeFunction = Callable[[list[Any], dict[str, Any]], Any]
LiteralFunction = Callable[[dict[str, Any]], Any]
FunctionEntry = tuple[NodeFunction | LiteralFunction, NodeType, tuple[NodeType, ...]]


class GPNode(metaclass=GPNodeTypeRegistry):
    functions: dict[str, FunctionEntry] = {}
    literals: list[str] = []

    type_functions: dict[NodeType, list[str]] = None
    terminal_list: list[str] = None
    non_terminal_list: list[str] = None
    branch_list: list[str] = None
    semiterminal_table: dict[str, set[int]] = None

    DATA_TYPES = None  # TODO: Use an abstract function for this instead.

    function_id: str
    """The ID of the node function, which is the string name of the Python function."""
    function: NodeFunction | LiteralFunction
    """The function that this node represents."""
    literal: Any
    """The literal value of this node, if it is a literal node."""
    output_type: NodeType
    """The type of the output of this node (in the internal typing system)."""
    input_types: tuple[NodeType, ...]
    """The types of the inputs to this node in order,
    i.e. the output types of its children (in the internal typing system)."""
    input_nodes: list['GPNode']
    """The children of this node in order."""
    parent: Optional['GPNode']
    """The parent of this node. If this node is the root of a tree, this will be None."""
    depth: int | None
    """The depth of this node in the tree from the root. Cached for :meth:`GPNode.get_depth`.
    Invalidated by :meth:`GPNode.set_parent`."""
    height: int | None
    """The height of this node in the tree from its deepest child. Cached for :meth:`GPNode.get_height`.
    Invalidated by :meth:`GPNode.add_input`."""
    fixed_context: dict[str, Any]
    """A dictionary of fixed context values that are used to parameterize literal generation."""

    def __init__(self, function_id, literal=None, fixed_context=None):
        self.function_id = function_id
        func_data = type(self).get_function_data(function_id)
        self.function = func_data[0]
        self.output_type = func_data[1]
        self.input_types = func_data[2]
        self.input_nodes = list()
        self.literal = literal
        self.fixed_context = fixed_context
        if function_id in type(self).literals and literal is None:
            self.literal = self.function(fixed_context)
        self.parent = None
        self.depth = None
        self.height = None

    def execute(self, context) -> Any:
        """Executes the function represented by this node, using the given context.

        Args:
            context: A dictionary of context values that are used to parameterize the function or its children.

        Returns:
            The output of the function represented by this node,
            or the literal value of this node if it is a literal node.
        """
        output = None
        if self.function_id not in type(self).literals:
            output = self.function(self.input_nodes, context)
        else:
            output = self.literal
        return output

    def add_input(self, input_node: 'GPNode') -> None:
        """Adds a child to this node as an input.

        Args:
            input_node: The child node to add. The output type should match the input type of this node.
                This type constraint is not checked here. Does not automatically set this node as the parent.
        """
        self.input_nodes.append(input_node)
        self.height = None

    def set_parent(self, parent: 'GPNode') -> None:
        """Sets the parent of this node.

        Args:
            parent: The parent node to set. Does not automatically add this node as a child of the parent.
        """
        self.parent = parent
        self.depth = None

    def get_depth(self) -> int:
        """Gets the depth of this node in the tree from the root. The root node has a depth of 0.
        Cached in the :attr:`GPNode.depth` attribute.

        Returns: The depth of this node in the tree.
        """
        if self.depth is not None:
            return self.depth

        if self.parent is None:
            self.depth = 0
            return 0
        else:
            self.depth = self.parent.get_depth() + 1
            return self.depth

    def get_height(self) -> int:
        """Gets the height of this node in the tree from its deepest child,
        i.e. the height of the subtree rooted at this node. A single node has a height of 1.
        Cached in the :attr:`GPNode.height` attribute.

        Returns: The height of this node in the tree.
        """

        if self.height is not None:
            return self.height

        if len(self.input_nodes) == 0:
            self.height = 1
            return 1
        else:
            max_length = 0
            for node in self.input_nodes:
                max_length = max(max_length, node.get_height())

            self.height = max_length + 1
            return self.height

    def get_size(self) -> int:
        return sum((1 for _ in self.traverse_post_order()))

    def get_node_list(self) -> list['GPNode']:
        node_list = [self]
        for input_node in self.input_nodes:
            node_list.extend(input_node.get_node_list())

        return node_list

    def tree_string(self, max_height, depth) -> str:
        string = ''
        for _ in range(depth):
            string += '|\t'

        string += str(self)
        string += '\n'
        for input_node in self.input_nodes:
            string += input_node.tree_string(max_height, depth + 1)

        return string

    def traverse_post_order(self) -> Generator['GPNode', None, None]:
        for child in self.input_nodes:
            for node in child.traverse_post_order():
                yield node

        yield self

    def traverse_pre_order(self) -> Generator['GPNode', None, None]:
        yield self
        for child in self.input_nodes:
            for node in child.traverse_pre_order():
                yield node

    def __str__(self) -> str:
        if self.function_id not in type(self).literals:
            return self.function.__name__
        else:
            return self.function.__name__ + '(' + str(self.literal) + ')'

    @classmethod
    def initialize_class(cls) -> None:
        """Performs any class-level initialization that is necessary. Does nothing if called after the first time.
        This is called by :class:`GPTree` when it is initialized.
        Do not call this before the primitive function decorators have been applied."""
        if cls.type_functions is None:
            cls.build_data_type_tables()

    @classmethod
    def build_data_type_tables(cls) -> None:
        cls.type_functions = dict()
        for data_type in cls.DATA_TYPES:
            cls.type_functions[data_type] = list()

        cls.terminal_list = list()
        cls.non_terminal_list = list()
        cls.branch_list = list()
        for node_id, node_data in cls.functions.items():
            node_data = cls.get_function_data(node_id)
            cls.type_functions[node_data[1]].append(node_id)
            if len(node_data[2]) == 0:
                cls.terminal_list.append(node_id)
            else:
                cls.non_terminal_list.append(node_id)
                if len(node_data[2]) > 1:
                    cls.branch_list.append(node_id)

    @classmethod
    def build_height_tables(cls, height, forbidden_nodes=None):
        if forbidden_nodes is None:
            forbidden_nodes = []
        grow_table = dict()
        full_table = dict()
        grow_table[1] = set()
        full_table[1] = set()
        for function in cls.terminal_list:
            if function in forbidden_nodes:
                continue
            grow_table[1].add(cls.get_function_data(function)[1])
            full_table[1].add(cls.get_function_data(function)[1])

        for i in range(2, height + 1):
            grow_table[i] = set()
            full_table[i] = set()
            grow_table[i] |= grow_table[i - 1]
            for function in cls.non_terminal_list:
                if function in forbidden_nodes:
                    continue
                grow_valid = True
                full_valid = True
                for inputType in cls.get_function_data(function)[2]:
                    if inputType not in grow_table[i - 1]:
                        grow_valid = False
                    if inputType not in full_table[i - 1]:
                        full_valid = False

                if grow_valid:
                    grow_table[i].add(cls.get_function_data(function)[1])
                if full_valid:
                    full_table[i].add(cls.get_function_data(function)[1])

        return grow_table, full_table

    @classmethod
    def build_semiterminal_table(cls, height, forbidden_nodes=None):
        if forbidden_nodes is None:
            forbidden_nodes = []
        if cls.semiterminal_table is None:
            cls.semiterminal_table = dict()
            cls.semiterminal_table[1] = set(
                [function for function in cls.terminal_list if function not in forbidden_nodes])
            for i in range(2, height + 1):
                cls.semiterminal_table[i] = set()
            dead_types = set()
            for i in range(2, height + 1):
                cls.semiterminal_table[i] = set()
                live_functions = [function for function in cls.non_terminal_list if
                                  function not in cls.semiterminal_table[i - 1] and function not in forbidden_nodes]
                for dataType in cls.DATA_TYPES:
                    if len(set(live_functions) & set(cls.type_functions[dataType])) == 0:
                        dead_types.add(dataType)

                modify = False
                for function in live_functions:
                    still_alive = False
                    for inputType in cls.get_function_data(function)[2]:
                        if inputType not in dead_types:
                            still_alive = True

                    if not still_alive:
                        cls.semiterminal_table[i].add(function)
                        modify = True

                if not modify:
                    break

            print(cls.semiterminal_table)
        return cls.semiterminal_table

    # Generates a table describing how deep it's possible to construct a tree with a given type.
    @classmethod
    def build_depth_table(cls, max_height, forbidden_nodes=None):
        # It's clearly impossible to go more than this depth without hitting a dead end or loop
        # max_height = len(cls.DATA_TYPES)

        if forbidden_nodes is None:
            forbidden_nodes = []
        type_tables = dict()
        type_depths = dict()
        for type in cls.DATA_TYPES:
            type_tables[type] = {1: {type}}
            type_depths[type] = 1
        # Construct the type tables describing which types can be present at each depth by taking the list of children of possible types at the previous depth
        for depth in range(2, max_height + 1):
            for type in cls.DATA_TYPES:
                children_types = set()
                for parent_type in type_tables[type][depth - 1]:
                    edge_functions = [function for function in cls.functions if
                                      function in cls.type_functions[parent_type] and function not in forbidden_nodes]
                    for function in edge_functions:
                        children_types |= set(cls.get_function_data(function)[2])
                type_tables[type][depth] = children_types
        # Construct the depth table, listing the highest depth that can be reached from a type.
        for depth in range(2, max_height + 1):
            for type in cls.DATA_TYPES:
                # Loops indicate that infinite depth can be reached
                if type in type_tables[type][depth]:
                    type_depths[type] = 1000000
                    continue
                # If anything can be reached at the current depth, then that depth can be reached
                if len(type_tables[type][depth]) > 0:
                    type_depths[type] = max(type_depths[type], depth)
                # If a type can be reached is known to have an even higher max depth (usually due to a loop) then that higher depth can be reached
                for childType in type_tables[type][depth]:
                    type_depths[type] = max(type_depths[type], type_depths[childType])
        return type_depths

    @classmethod
    @cache
    def get_functions(cls, output_type: int, terminal=False, non_terminal=False, branch=False, num_children: int = None, child_types=None, forbidden_nodes=None):
        if forbidden_nodes is None:
            forbidden_nodes = []
        possible = list([function for function in cls.type_functions[output_type] if function not in forbidden_nodes])
        if terminal:
            possible = [function for function in possible if function in cls.terminal_list]
        if non_terminal:
            possible = [function for function in possible if function in cls.non_terminal_list]
        if branch:
            possible = [function for function in possible if function in cls.branch_list]
        if num_children is not None:
            possible = [function for function in possible if len(cls.get_function_data(function)[2]) == num_children]
        if child_types is not None:
            possible = [function for function in possible if cls.get_function_data(function)[2] == child_types]
        return possible

    @classmethod
    def random_function(cls, output_type, terminal=False, non_terminal=False, branch=False, num_children=None, child_types=None, forbidden_nodes=None):
        possible_functions = cls.get_functions(output_type, terminal, non_terminal, branch, num_children, child_types, forbidden_nodes)
        return random.choice(possible_functions)

    @classmethod
    def get_function_data(cls, func_id):
        return cls.functions[func_id]

    @classmethod
    def get_id(cls, function):
        return [func_id for func_id, func_tuple in cls.functions.items() if func_tuple[0] == function][0]

    @classmethod
    def gp_primitive(cls, output_type: Any, input_types: tuple[Any, ...]):
        def internal_decorator(function):
            function_id = function.__name__
            if function_id in cls.functions:
                raise ValueError(f"GPNode ID conflict: a function with name {function_id} was already registered!")
            cls.functions[function_id] = (function, output_type, input_types)
            return function
        return internal_decorator

    @classmethod
    def gp_literal(cls, output_type: Any):
        def internal_decorator(function):
            function_id = function.__name__
            if function_id in cls.functions:
                raise ValueError(f"GPNode ID conflict: a function with name {function_id} was already registered!")
            cls.functions[function_id] = (function, output_type, ())
            cls.literals.append(function_id)
            return function
        return internal_decorator
