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

import copy
import itertools
import random
from collections import deque
from typing import Any, TypedDict

from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.genotypes.geneticprogramming.gpnode import GPNode, NodeType
from modularcoevolution.genotypes.geneticprogramming.gpnodetyperegistry import GPNodeTypeRegistry
from modularcoevolution.genotypes.geneticprogramming.gptree import GPTreeParameters, GPTree
from modularcoevolution.utilities.commandlineutils import color_string_256


class CartesianGPParameters(TypedDict, total=False):
    node_class: type[GPNode] | str
    return_types: list[NodeType]
    columns: int
    type_rows: dict[NodeType, int]
    levels_back: int
    fixed_context: dict[str, Any]
    nodes: list[str | tuple[str, Any]]
    connections: list[list[int]]
    roots: list[int]


class CartesianGP(BaseGenotype):
    """An implementation of Cartesian Genetic Programming compatible with :class:`.GPNode`."""

    node_class: type[GPNode]
    """The type of genetic programming node used in the genotype, which defines the available primitives and types."""
    return_types: list[NodeType]
    """An ordered list of types to return from the genotype."""

    columns: int
    """The number of columns in the genotype, shared across all types."""
    levels_back: int
    """The maximum number of levels back a node may receive inputs from."""
    type_rows: dict[NodeType, int]
    """The number of rows in the genotype for each type.
    This can vary by type to bias the genotype towards certain types."""

    nodes: list[GPNode | None]
    """A flat list of nodes in the genotype."""
    connections: list[list[int]]
    """A list of input indices for each node."""
    roots: list[int]
    """For each return value, identifies the node to return a value from."""

    fixed_context: dict[str, Any]
    """A dictionary of information made available to all nodes in the genotype.
    :meth:`GPNode.execute` passes a context dictionary to each node that it receives as a parameter, which is updated with this dictionary.
    The main purpose of this dictionary is to provide context to literal nodes, whose values can not depend on the dynamic context."""

    min_depth_table: dict[str, int]
    """The minimum depth of a tree rooted at each function ID."""
    # min_type_depth_table: dict[NodeType, int]
    # """The minimum depth of a tree with a given node type."""

    _cumulative_row_sum: dict[str, int]
    """Used to calculate the node index, since node types have varying numbers of rows."""
    _rows: int
    """The number of rows in the genotype across all types, used to calculate the node index."""

    def __init__(self, parameters: CartesianGPParameters):
        super().__init__(parameters)

        if "node_class" in parameters:
            if isinstance(parameters["node_class"], type):
                self.node_class = parameters["node_class"]
            elif isinstance(parameters["node_class"], str):
                self.node_class = GPNodeTypeRegistry.name_lookup[parameters["node_class"]]
            else:
                raise TypeError("node_class must be a type or a type name.")
            self.node_class.initialize_class()
        else:
            raise ValueError("'node_class' must be supplied as a parameter.")

        if "return_types" in parameters:
            self.return_types = parameters["return_types"]
        else:
            raise ValueError("'return_types' must be supplied as a parameter.")

        if 'columns' in parameters:
            self.columns = parameters['columns']
            self.min_depth_table, _ = self.node_class.build_min_depth_table()
        else:
            raise ValueError("'columns' must be supplied as a parameter.")

        if 'type_rows' in parameters:
            self.type_rows = parameters['type_rows']
        else:
            self.type_rows = {}
        self._initialize_type_rows()

        if 'levels_back' in parameters:
            self.levels_back = parameters['levels_back']
        else:
            # Default to unrestricted connection distance.
            self.levels_back = self.columns

        if 'fixed_context' in parameters:
            self.fixed_context = parameters['fixed_context']
        else:
            self.fixed_context = {}

        if 'nodes' in parameters and 'connections' in parameters and 'roots' in parameters:
            self.nodes = []
            self.connections = []
            for node_id, connections in zip(parameters['nodes'], parameters['connections']):
                if isinstance(node_id, tuple):
                    node = self.node_class(node_id[0], node_id[1], fixed_context=self.fixed_context)
                elif node_id == "":
                    node = None
                else:
                    node = self.node_class(node_id, fixed_context=self.fixed_context)
                self.nodes.append(node)
                self.connections.append(connections.copy())
                for connection in connections:
                    node.add_input(self.nodes[connection], ignore_parent=True)
            self.roots = parameters['roots'].copy()
        elif 'nodes' in parameters or 'connections' in parameters or 'roots' in parameters:
            raise ValueError("Must supply all of 'nodes', 'connections', and 'roots' to initialize from a raw genotype.")
        else:
            self._initialize()

    def _initialize_type_rows(self):
        row_sum = 0
        self._cumulative_type_rows = {}

        for node_type in self.node_class.data_types():
            # Types default to having one row each.
            if node_type not in self.type_rows:
                self.type_rows[node_type] = 1

            self._cumulative_type_rows[node_type] = row_sum
            row_sum += self.type_rows[node_type]
        self._rows = row_sum

    def mutate(self):
        super().mutate()
        active_nodes, active_connections = self.get_active()

        def mutate_inner():
            random_index = random.randrange(-len(self.roots), len(self.nodes))
            if random_index < 0:
                return self._mutate_root_connection(-random_index - 1)

            column, node_type, row = self.index_position(random_index)
            if random.random() < 0.5:
                return self._mutate_node_function(column, node_type, row, active_nodes)
            else:
                return self._mutate_node_connection(column, node_type, row, active_connections)

        result = False
        while result is False:
            # Parameterless mutation: keep mutating until the phenotype is modified.
            result = mutate_inner()

    def recombine(self, donor: 'CartesianGP'):
        # CGP doesn't typically use recombination.
        # I've implemented one-point crossover here as a test.
        super().recombine(donor)
        crossover_point = random.randrange(0, len(self.nodes))

        self.nodes = donor.nodes[:crossover_point] + self.nodes[crossover_point:]
        # Everything else is already either deep-copied or value-based anyway, but the donor connections are not.
        self.connections = copy.deepcopy(donor.connections[:crossover_point]) + self.connections[crossover_point:]
        # Assume the roots always come with the high-valued nodes.

        # Reassign GPNode children
        for node_index in range(crossover_point, len(self.nodes)):
            node = self.nodes[node_index]
            for child_index, connection in enumerate(self.connections[node_index]):
                node.set_input(child_index, self.nodes[connection], ignore_parent=True)

    def clone(self) -> 'CartesianGP':
        return super().clone()

    def __hash__(self):
        string_list = []
        for node, connections in zip(self.nodes, self.connections):
            if node is None:
                node_string = ""
            else:
                node_string = node.function_id + " " + str(node.literal)
            string_list.append(node_string + " " + str(connections))
        string_list.append(str(self.roots))
        return hash(tuple(string_list))

    def _get_id_list(self) -> list[str | tuple[str, Any]]:
        id_list = []
        for node in self.nodes:
            if node is None:
                id_list.append("")
            elif node.literal is None:
                id_list.append(node.function_id)
            else:
                id_list.append((node.function_id, node.literal))
        return id_list

    def get_raw_genotype(self) -> CartesianGPParameters:
        return {
            'node_class': self.node_class.__name__,
            'return_types': self.return_types,
            'columns': self.columns,
            'type_rows': self.type_rows,
            'levels_back': self.levels_back,
            'fixed_context': self.fixed_context,
            'nodes': self._get_id_list(),
            'connections': self.connections,
            'roots': self.roots,
        }

    def diversity_function(self, population, reference=None, samples=None):
        # TODO: Implement this.
        return 0

    def execute(self, context: dict[str, Any]) -> list[Any]:
        # Clear saved values first.
        for node in self.nodes:
            if node is not None:
                node.saved_value = None

        context = context.copy()
        context.update(self.fixed_context)
        # save_values is always on for CGP.
        context['save_values'] = True

        result = []
        for root in self.roots:
            try:
                # Roots can't connect to empty nodes.
                # use_cached enables the desired CGP behavior.
                result.append(self.nodes[root].execute(context, use_cached=True))
            except Exception as error:
                error.add_note(f"Error while executing the following subtree:\n{self.nodes[root]}")
                raise error
        return result

    def node_index(self, column: int, node_type: NodeType, row: int):
        """
        Returns the index of the node with the given coordinates.

        Args:
            column: The column of the node.
            node_type: The output type of the node.
            row: The row of the node.
        """
        full_row = self._cumulative_type_rows[node_type] + row
        index = column * self._rows + full_row
        return index

    def index_position(self, index: int) -> tuple[int, NodeType, int]:
        """
        Returns the coordinates of the node with the given index.

        Args:
            index: The index of the node.

        Returns:
            A tuple of (column, node_type, row) for the node with the given index.
        """
        types = self.node_class.data_types()
        column = index // self._rows
        full_row = index % self._rows
        for type_index, data_type in enumerate(types):
            if full_row < self._cumulative_type_rows[data_type]:
                node_type = types[type_index - 1]
                break
        else:
            node_type = types[-1]

        row = full_row - self._cumulative_type_rows[node_type]
        return column, node_type, row

    def _random_valid_child_index(self, column: int, child_type: NodeType) -> int:
        """
        Returns a random node index of the given child type that would be a valid input to a node in the given column.

        Args:
            column: The column of the node.
            child_type: The input type to select a matching child for.

        Returns:
            A node index of type `child_type` within :attr:`CartesianGP.levels_back` of `column`
            which is not empty.
        """
        child = None
        attempt_count = 0
        while child is None:
            random_column = random.randint(max(0, column - self.levels_back), column - 1)
            random_row = random.randrange(self.type_rows[child_type])
            index = self.node_index(random_column, child_type, random_row)
            child = self.nodes[index]
            attempt_count += 1
            if attempt_count > 100:
                raise RuntimeError(f"No valid child index of type {child_type} from column {column}.")
        return index

    def _random_valid_function(self, column: int, node_type: NodeType) -> str | None:
        """
        Returns a random function with the given output type that could be used in the given column,
        based on the minimum depth table (which ensures there is enough space to complete the subtree).
        Returns `None` if there is no valid function.
        """
        valid_functions = self.node_class._get_functions(output_type=node_type)
        valid_functions = [function for function in valid_functions if column >= self.min_depth_table[function]]
        if len(valid_functions) == 0:
            return None
        return random.choice(valid_functions)

    def _initialize(self):
        """
        Initializes the genotype with random functions and connections.
        """
        self.nodes = []
        self.connections = []
        for column in range(self.columns):
            for node_type in self.type_rows.keys():
                for row in range(self.type_rows[node_type]):
                    node_index = self.node_index(column, node_type, row)
                    function = self._random_valid_function(column, node_type)
                    if function is None:
                        self.nodes.append(None)
                        self.connections.append([])
                        continue

                    node = self.node_class(function, fixed_context=self.fixed_context)
                    self.nodes.append(node)
                    node_connections = []
                    for child_index, child_type in enumerate(node.input_types):
                        connection_index = self._random_valid_child_index(column, child_type)
                        input_node = self.nodes[connection_index]
                        node.set_input(child_index, input_node, ignore_parent=True)
                        node_connections.append(connection_index)
                    self.connections.append(node_connections)

        self.roots = []
        for return_type in self.return_types:
            root_index = self._random_valid_child_index(self.columns, return_type)
            self.roots.append(root_index)

    def get_active(self) -> tuple[set[int], set[tuple[int, int]]]:
        """
        Returns the sets of active nodes and connections in the genotype.

        Returns:
            - A set of node indices that are used in the output.
            - A set of (parent_index, child_index) tuples for each connection used in the output.
        """
        active_nodes = set()
        active_connections = set()

        frontier = deque()
        visited = set()
        frontier.extend(self.roots)
        while frontier:
            node_index = frontier.pop()
            visited.add(node_index)
            active_nodes.add(node_index)
            for child in self.connections[node_index]:
                if child not in visited:
                    frontier.append(child)
                active_connections.add((node_index, child))
        return active_nodes, active_connections


    def _mutate_node_function(self, column: int, node_type: NodeType, row: int, active_nodes: set[int]) -> bool:
        """
        Mutates the function of the node at the given location,
        keeping any valid inputs and selecting random inputs otherwise.

        Args:
            column: The column of the node.
            node_type: The output type of the node.
            row: The row of the node.
        """
        node_index = self.node_index(column, node_type, row)
        old_node = self.nodes[node_index]
        if old_node is None:
            return False

        # This should never return None, as there was a valid function here to begin with.
        new_function = self._random_valid_function(column, node_type)
        new_node = self.node_class(new_function, fixed_context=self.fixed_context)
        new_connections = []

        for child_index, (old_type, new_type) in enumerate(itertools.zip_longest(old_node.input_types, new_node.input_types)):
            if old_type == new_type:
                new_node.set_input(child_index, old_node.input_nodes[child_index], ignore_parent=True)
                new_connections.append(self.connections[node_index][child_index])
            elif new_type is not None:
                new_input_index = self._random_valid_child_index(column, new_type)
                new_node.set_input(child_index, self.nodes[new_input_index], ignore_parent=True)
                new_connections.append(new_input_index)

        self.nodes[node_index] = new_node
        self.connections[node_index] = new_connections
        assert old_node.output_type == new_node.output_type
        return node_index in active_nodes

    def _mutate_node_connection(self, column: int, node_type: NodeType, row: int, active_connections: set[tuple[int, int]]) -> bool:
        """
        Mutates one of the input connections of the node at the given location, selected at random.

        Args:
            column: The column of the node.
            node_type: The output type of the node.
            row: The row of the node.
        """
        node_index = self.node_index(column, node_type, row)
        node = self.nodes[node_index]
        if node is None:
            return False
        if len(node.input_types) == 0:
            return False
        child_index = random.randrange(len(node.input_types))
        child_type = node.input_types[child_index]
        new_input_index = self._random_valid_child_index(column, child_type)
        node.set_input(child_index, self.nodes[new_input_index], ignore_parent=True)
        self.connections[node_index][child_index] = new_input_index
        return (node_index, new_input_index) in active_connections

    def _mutate_root_connection(self, root_index: int) -> bool:
        """
        Mutates the given output connection of the genotype.

        Args:
            root_index: The index of the output.

        Returns:
            True (mutating the root always modifies the phenotype).
        """
        return_type = self.return_types[root_index]
        new_root_index = self._random_valid_child_index(self.columns, return_type)
        self.roots[root_index] = new_root_index
        return True

    def to_table_string(self) -> str:
        active_nodes, _ = self.get_active()

        columns = [[] for _ in range(self.columns)]
        for node_type, rows in self.type_rows.items():
            for row in range(rows):
                for column in range(self.columns):
                    node_index = self.node_index(column, node_type, row)
                    node = self.nodes[node_index]
                    if node is None:
                        columns[column].append("")
                        continue

                    if node.literal is not None:
                        suffix_string = str(node.literal)
                        if len(suffix_string) > 16:
                            suffix_string = suffix_string[:13] + "..."
                    else:
                        suffix_string = ", ".join(str(connections) for connections in self.connections[node_index])
                    node_string = f"{node_index}: {node.function_id}({suffix_string})"

                    if node_index not in active_nodes:
                        node_string = color_string_256(node_string, 2, 2, 2)
                    else:
                        node_string = color_string_256(node_string, 5, 5, 5)
                    columns[column].append(node_string)

        for column in columns:
            max_length = max(len(string) for string in column)
            for index, string in enumerate(column):
                column[index] = string.ljust(max_length)

        row_strings = []
        for row in zip(*columns):
            row_strings.append(" ".join(row))
        root_string = "Outputs: " + ", ".join((str(root) for root in self.roots))
        table = "\n".join(row_strings)
        return root_string + "\n" + table

    def to_trees(self) -> list[GPTree]:
        trees = []
        for root_index, root in enumerate(self.roots):
            root_node = self.nodes[root]
            id_list = []
            for node in root_node.traverse_pre_order():
                id_list.append(node.function_id)
                if node.function_id in self.node_class.literals:
                    id_list.append(node.serialize_literal())
            return_type = self.return_types[root_index]

            tree_parameters: GPTreeParameters = {
                'node_type': self.node_class,
                'return_type': return_type,
                'fixed_context': self.fixed_context,
                'id_list': id_list,
            }
            tree = GPTree(tree_parameters)
            trees.append(tree)
        return trees

    def to_tree_string(self) -> str:
        trees = self.to_trees()
        tree_strings = []
        for index, tree in enumerate(trees):
            tree_string = f"Output {index} ({self.return_types[index]})\n{tree}"
            tree_strings.append(tree_string)
        return "\n".join(tree_strings)

    def __str__(self) -> str:
        return self.to_table_string() + "\nRepresented trees:\n" + self.to_tree_string()





