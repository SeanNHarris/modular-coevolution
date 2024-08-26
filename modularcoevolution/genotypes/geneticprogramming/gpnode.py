from abc import abstractmethod
from functools import cache
from typing import Callable, Any, Union, Optional, Generator, Sequence, Protocol, TypeVar

from modularcoevolution.genotypes.geneticprogramming.gpnodetyperegistry import GPNodeTypeRegistry

import random

NodeType = Any
NodeFunction = Callable[[list[Any], dict[str, Any]], Any]
LiteralFunction = Callable[[dict[str, Any]], Any]
FunctionEntry = tuple[NodeFunction | LiteralFunction, NodeType, tuple[NodeType, ...]]
T = TypeVar('T')


class GPPrimitiveProtocol(Protocol):
    def __call__(self, input_nodes: list['GPNode'], context: dict[str, Any]) -> Any:
        ...


class GPLiteralProtocol(Protocol):
    def __call__(self, fixed_context: dict[str, Any]) -> Any:
        ...


class GPLiteralMutatorProtocol(Protocol):
    def __call__(self, value: T, fixed_context: dict[str, Any]) -> T:
        ...


class GPNodeType(GPNodeTypeRegistry):
    """Metaclass for GPNode classes which ensures that subclass members are copies, not references,
    of the superclass members.

    Needed when two GPNode subclasses are sharing primitives from a superclass."""
    def __init__(cls: type['GPNode'], name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls.functions = {}
        cls.literals = set()
        for parent in bases:
            cls.functions.update(parent.functions)
            cls.literals.update(parent.literals)

        # Leave these uninitialized until all the primitives function decorators have been applied.
        type_functions: dict[NodeType, list[str]] = None
        terminal_list: list[str] = None
        non_terminal_list: list[str] = None
        branch_list: list[str] = None
        semiterminal_table: dict[str, set[int]] = None


class GPNode(metaclass=GPNodeType):
    functions: dict[str, FunctionEntry] = {}
    """A dictionary of functions usable as primitives and their properties.
    This dictionary is keyed by the function id,
    and stores a tuple of the function, the output type, and the input types.
    Populated by the :meth:`gp_primitive` and :meth:`gp_literal` decorators."""
    literals: set[str] = []
    """A set denoting which primitives are literal nodes.
    Literal nodes store a constant value rather than a function.
    The function associated with a literal node is used to
    randomly generate the literal value when the node is created.
    Populated by the :meth:`gp_literal` decorator."""

    literal_mutators: dict[NodeType, 'GPLiteralMutatorProtocol'] = {}
    """If the GP tree attempts to mutate a literal node with a given data type,
    and that type is in this dictionary, the associated function will be used to mutate the literal value.
    Otherwise, the normal mutation process will be used.
    Populated by the :meth:`gp_literal_mutator` decorator."""
    literal_serializers: dict[NodeType, Callable[[Any], Any]] = {}
    """When the GP tree constructs a node id list and reaches a literal node with a given data type,
    and that type is in this dictionary, the associated function will applied to the literal value.
    Otherwise, the literal value will be used as-is.
    This is necessary for literals that cannot be hashed or stored in JSON format.
    Populated by the :meth:`gp_literal_serializer` decorator.
    You must provide a matching deserializer with :meth:`gp_literal_deserializer`."""
    literal_deserializers: dict[NodeType, Callable[[Any], Any]] = {}
    """When the GP tree is constructed from a node id list and reaches a literal node with a given data type,
    and that type is in this dictionary, the associated function will applied to the stored literal value.
    Otherwise, the stored value will be used as-is.
    This is necessary for literals that cannot be hashed or stored in JSON format.
    Populated by the :meth:`:meth:`gp_literal_deserializer` decorator.
    You must provide a matching serializer with :meth:`gp_literal_serializer`."""

    type_functions: dict[NodeType, list[str]] = None
    """A dictionary mapping data types to lists of function IDs that output that data type."""
    terminal_list: list[str] = None
    """A list of function IDs for terminal nodes, i.e., nodes with no inputs."""
    non_terminal_list: list[str] = None
    """A list of function IDs for non-terminal nodes, i.e., nodes with inputs."""
    branch_list: list[str] = None
    """A list of function IDs for nodes with more than one input."""
    semiterminal_table: dict[str, set[int]] = None

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
    _depth: int | None
    """The depth of this node in the tree from the root. Cached for :meth:`GPNode.get_depth`.
    Invalidated by :meth:`GPNode.set_parent`."""
    _height: int | None
    """The height of this node in the tree from its deepest child. Cached for :meth:`GPNode.get_height`.
    Invalidated by :meth:`GPNode.add_input`."""
    fixed_context: dict[str, Any]
    """A dictionary of fixed context values that are used to parameterize literal generation."""

    @classmethod
    @abstractmethod
    def data_types(cls) -> Sequence[NodeType]:
        """Returns a list of data types to be used for this set of GP nodes.
        These values will be passed to the :meth:`gp_primitive` and :meth:`gp_literal` decorators.

        Returns:
            A list of values representing different node types.
        """
        pass

    def __init__(self, function_id, literal=None, fixed_context=None):
        self.function_id = function_id
        func_data = type(self).get_function_data(function_id)
        self.function = func_data[0]
        self.output_type = func_data[1]
        self.input_types = func_data[2]
        self.input_nodes = [None for _ in self.input_types]
        self.fixed_context = fixed_context

        if function_id in type(self).literals:
            if literal is None:
                self.literal = self.function(fixed_context)
            else:
                self.literal = self._deserialize_literal(literal)
        else:
            self.literal = None

        self.parent = None
        self._depth = None
        self._height = None

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

    def set_input(self, index: int, input_node: 'GPNode') -> None:
        """Sets the child of this node at the given index to the given node, and sets this node as the parent of the child.

        Args:
            index: The index of the child to set.
            input_node: The child node to set. The output type should match the input type of this node at the given index,
                but this type constraint is not checked here.
        """
        self.input_nodes[index] = input_node
        self._height = None  # Invalidate height cache
        input_node.set_parent(self)

    def add_input(self, input_node: 'GPNode') -> None:
        """Adds a child to this node as an input, and sets this node as the parent of the child.

        Args:
            input_node: The child node to add. The output type should match corresponding input type of this node,
                but this type constraint is not checked here.
        """
        next_input = self.input_nodes.index(None)
        self.set_input(next_input, input_node)

    def set_parent(self, parent: 'GPNode') -> None:
        """Sets the parent of this node.

        Args:
            parent: The parent node to set. Does not automatically add this node as a child of the parent.
        """
        self.parent = parent
        self._depth = None  # Invalidate depth cache

    def get_depth(self) -> int:
        """Gets the depth of this node in the tree from the root. The root node has a depth of 0.
        Cached in the :attr:`GPNode._depth` attribute.

        Returns: The depth of this node in the tree.
        """
        if self._depth is not None:
            return self._depth

        if self.parent is None:
            self._depth = 0
            return 0
        else:
            self._depth = self.parent.get_depth() + 1
            return self._depth

    def get_height(self) -> int:
        """Gets the height of this node in the tree from its deepest child,
        i.e. the height of the subtree rooted at this node. A single node has a height of 1.
        Cached in the :attr:`GPNode._height` attribute.

        Returns: The height of this node in the tree.
        """

        if self._height is not None:
            return self._height

        if len(self.input_nodes) == 0:
            self._height = 1
            return 1
        else:
            max_length = 0
            for node in self.input_nodes:
                max_length = max(max_length, node.get_height())

            self._height = max_length + 1
            return self._height

    def get_size(self) -> int:
        return sum((1 for _ in self.traverse_post_order()))

    def get_node_list(self) -> list['GPNode']:
        node_list = [self]
        for input_node in self.input_nodes:
            node_list.extend(input_node.get_node_list())

        return node_list

    def tree_string(self):
        max_height = self.get_height()
        depth = 0
        return '\n'.join(self._build_tree_string(max_height, depth))

    def _build_tree_string(self, max_height: int, depth: int, substring_list: list[str] = None) -> list[str]:
        if substring_list is None:
            substring_list = []
        string = ''
        for _ in range(depth):
            string += '|\t'

        string += str(self)
        substring_list.append(string)
        for input_node in self.input_nodes:
            input_node._build_tree_string(max_height, depth + 1, substring_list)

        return substring_list

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

    def serialize_literal(self) -> Any:
        if self.output_type in type(self).literal_serializers:
            return self.literal_serializers[self.output_type](self.literal)
        return self.literal

    def _deserialize_literal(self, literal: Any) -> Any:
        if self.output_type in type(self).literal_deserializers:
            return self.literal_deserializers[self.output_type](literal)
        return literal

    def __str__(self) -> str:
        if self.function_id not in type(self).literals:
            return self.function.__name__
        else:
            return self.function.__name__ + '(' + str(self.serialize_literal()) + ')'

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
        for data_type in cls.data_types():
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
                for dataType in cls.data_types():
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
        if forbidden_nodes is None:
            forbidden_nodes = []
        type_tables = dict()
        type_depths = dict()
        for data_type in cls.data_types():
            type_tables[data_type] = {1: {data_type}}
            type_depths[data_type] = 1
        # Construct the type tables describing which types can be present at each depth by taking the list of children of possible types at the previous depth
        for depth in range(2, max_height + 1):
            for data_type in cls.data_types():
                children_types = set()
                for parent_type in type_tables[data_type][depth - 1]:
                    edge_functions = [function for function in cls.functions if
                                      function in cls.type_functions[parent_type] and function not in forbidden_nodes]
                    for function in edge_functions:
                        children_types |= set(cls.get_function_data(function)[2])
                type_tables[data_type][depth] = children_types
        # Construct the depth table, listing the highest depth that can be reached from a type.
        for depth in range(2, max_height + 1):
            for data_type in cls.data_types():
                # Loops indicate that infinite depth can be reached
                if data_type in type_tables[data_type][depth]:
                    type_depths[data_type] = 1000000
                    continue
                # If anything can be reached at the current depth, then that depth can be reached
                if len(type_tables[data_type][depth]) > 0:
                    type_depths[data_type] = max(type_depths[data_type], depth)
                # If a type can be reached is known to have an even higher max depth (usually due to a loop) then that higher depth can be reached
                for childType in type_tables[data_type][depth]:
                    type_depths[data_type] = max(type_depths[data_type], type_depths[childType])
        return type_depths

    @classmethod
    @cache
    def _get_functions(
            cls,
            output_type: NodeType,
            terminal: bool = False,
            non_terminal: bool = False,
            branch: bool = False,
            num_children: int = None,
            child_types: Sequence[NodeType] = None,
            has_child: NodeType = None,
            forbidden_nodes: Sequence[str] = None
    ) -> list[str]:
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
        if has_child is not None:
            possible = [function for function in possible if has_child in cls.get_function_data(function)[2]]
        return possible

    @classmethod
    def random_function(
            cls,
            output_type: NodeType,
            terminal: bool = False,
            non_terminal: bool = False,
            branch: bool = False,
            num_children: int = None,
            child_types: Sequence[NodeType] = None,
            has_child: NodeType = None,
            forbidden_nodes: Sequence[str] = None
    ):
        """Returns a random function ID out of those that meet the provided constraints.
        Use this when generating random nodes.

        Args:
            output_type: The output type of the function must match this argument.
            terminal: If True, the function must be a terminal node.
            non_terminal: If True, the function must be a non-terminal node.
            branch: If True, the function must be a node with more than one input.
            num_children: The number of children that the function must match.
            child_types: An list of input types that the function must match in order.
            has_child: At least one input to this function must match the given type.
            forbidden_nodes: A list of function IDs to exclude from selection.

        Returns:
            A random function ID that meets the given constraints.

        Raises:
            ValueError: If no functions meet the given constraints.
        """
        possible_functions = cls._get_functions(output_type, terminal, non_terminal, branch, num_children, child_types, has_child, forbidden_nodes)
        if len(possible_functions) == 0:
            raise ValueError("No functions meet the given constraints.")
        return random.choice(possible_functions)

    @classmethod
    def get_function_data(cls, func_id):
        return cls.functions[func_id]

    @classmethod
    def get_id(cls, function):
        return [func_id for func_id, func_tuple in cls.functions.items() if func_tuple[0] == function][0]

    @classmethod
    def gp_primitive(cls, output_type: Any, input_types: tuple[Any, ...]):
        def internal_decorator(function: 'GPPrimitiveProtocol'):
            function_id = function.__name__
            if function_id in cls.functions:
                raise ValueError(f"GPNode ID conflict: a function with name {function_id} was already registered!")
            cls.functions[function_id] = (function, output_type, input_types)
            return function
        return internal_decorator

    @classmethod
    def gp_literal(cls, output_type: Any):
        def internal_decorator(function: 'GPLiteralProtocol'):
            function_id = function.__name__
            if function_id in cls.functions:
                raise ValueError(f"GPNode ID conflict: a function with name {function_id} was already registered!")
            cls.functions[function_id] = (function, output_type, ())
            cls.literals.add(function_id)
            return function
        return internal_decorator

    @classmethod
    def gp_literal_mutator(cls, literal_type: Any):
        def internal_decorator(function: 'GPLiteralMutatorProtocol'):
            if literal_type in cls.literal_mutators:
                raise ValueError(f"Literal mutator conflict: a mutator for type {literal_type} was already registered!")
            cls.literal_mutators[literal_type] = function
            return function
        return internal_decorator

    @classmethod
    def gp_literal_serializer(cls, literal_type: Any):
        def internal_decorator(function: Callable[[Any], Any]):
            if literal_type in cls.literal_serializers:
                raise ValueError(f"Literal serializer conflict: a serializer for type {literal_type} was already registered!")
            cls.literal_serializers[literal_type] = function
            return function
        return internal_decorator

    @classmethod
    def gp_literal_deserializer(cls, literal_type: Any):
        def internal_decorator(function: Callable[[Any], Any]):
            if literal_type in cls.literal_deserializers:
                raise ValueError(f"Literal deserializer conflict: a deserializer for type {literal_type} was already registered!")
            cls.literal_deserializers[literal_type] = function
            return function
        return internal_decorator
