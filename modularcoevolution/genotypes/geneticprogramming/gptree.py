# Represents a genetic programming tree. Nodes for the tree are in GPNodes.py.
from warnings import warn

from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.genotypes.geneticprogramming.gpnode import GPNodeTypeRegistry, GPNode
from modularcoevolution.genotypes.diversity.gpdiversity import *

from typing import Any, TypedDict, Union

import random

# TODO: Rename variables


MAXIMUM_HEIGHT = 15
"""Raises an exception if a tree is generated that exceeds this height."""


# Hardcoded parameters
SUBTREE_MUTATE_HEIGHT_SIGMA = 2


class GPTreeParameters(TypedDict, total=False):
    node_type: Union[type, str]
    """See :attr:`GPTree.node_type`. This can be a type or a string name of a type.
    Using a string name only works if the type has been imported somewhere, such as in the experiment definition."""
    return_type: int
    """See :attr:`GPTree.return_type`."""
    min_height: int
    """See :attr:`GPTree.min_height`."""
    max_height: int
    """See :attr:`GPTree.max_height`."""
    parsimony_weight: float
    """See :attr:`GPTree.parsimony_weight`."""
    forbidden_nodes: tuple[str]
    """See :attr:`GPTree.forbidden_nodes`."""
    fixed_context: list[str, Any]  # Can we do more typing enforcement with this?
    """See :attr:`GPTree.fixed_context`."""
    id_list: list[str | Any]
    """A list of node IDs and literal values to be used to generate the tree, instead of using random generation.
    This list should match the output of :meth:`GPTree.getNodeIDList`."""


class GPTree(BaseGenotype):
    node_type: type[GPNode]
    """The type of genetic programming node used in the tree, which defines the available primitives and types."""
    return_type: int
    """The return type of the tree's root node (using the types defined by :attr:`node_type`).
    This is enforced at all times."""
    min_height: int
    """The minimum height of the tree for random generation. Not enforced for recombination or mutation.
    Defaults to 3."""
    max_height: int
    """The maximum height of the tree for random generation. Not enforced for recombination or mutation.
    Defaults to 7."""
    parsimony_weight: float
    """A percentage of fitness removed per node in the tree.
    Should be on the order of 0.01 = 1% depending on the expected tree size.
    Defaults to 0."""
    forbidden_nodes: tuple[str]
    """A tuple of node IDs that are not allowed to be used in the tree.
    This is useful when using the same node type for multiple populations or chromosomes with different restrictions."""
    fixed_context: dict[str, Any]
    """A dictionary of information made available to all nodes in the tree.
    :meth:`GPNode.execute` passes a context dictionary to each node that it receives as a parameter, which is updated with this dictionary.
    The main purpose of this dictionary is to provide context to literal nodes, whose values can not depend on the dynamic context."""

    root: GPNode
    """The root node of the tree."""
    node_list: list[GPNode] | None
    """Cached output for :meth:`GPTree.getNodeList`."""
    node_id_list: list[str | Any] | None
    """Cached output for :meth:`GPTree.getNodeIDList`."""

    # Generate a tree either at random or from a list of expansions.
    def __init__(self, parameters):
        super().__init__()

        self.parameters = parameters
        if "node_type" in parameters:
            if isinstance(parameters["node_type"], type):
                self.node_type = parameters["node_type"]
            elif isinstance(parameters["node_type"], str):
                self.node_type = GPNodeTypeRegistry.name_lookup[parameters["node_type"]]
            else:
                raise TypeError("node_type must be a type or a type name.")
            self.node_type.initialize_class()
        else:
            raise TypeError("A node type must be supplied as a parameter.")

        if "return_type" in parameters:
            self.return_type = parameters["return_type"]
        else:
            raise TypeError("A return type must be supplied as a parameter.")

        if "min_height" in parameters:
            self.min_height = parameters["min_height"]
        else:
            self.min_height = 3

        if "max_height" in parameters:
            self.max_height = parameters["max_height"]
        else:
            self.max_height = 7

        if "parsimony_weight" in parameters:
            self.parsimony_weight = parameters["parsimony_weight"]
        else:
            self.parsimony_weight = 0.0

        if "forbidden_nodes" in parameters:
            self.forbidden_nodes = parameters["forbidden_nodes"]
        else:
            self.forbidden_nodes = tuple()

        if "fixed_context" in parameters:
            self.fixed_context = parameters["fixed_context"]
        else:
            self.fixed_context = dict()

        # self.node_type.build_data_type_tables()
        # self.growTable, self.fullTable = self.node_type.build_height_tables(MAXIMUM_HEIGHT,
        #                                                                     self.forbidden_nodes)  # Todo: Cache these per node type, forbidden_nodes
        # self.depthTable = self.node_type.build_depth_table(MAXIMUM_HEIGHT, self.forbidden_nodes)

        if "id_list" in parameters:
            self.root = self._generate_from_list(list(parameters["id_list"]))
        else:
            height = random.randint(self.min_height, self.max_height)
            self.root = self.random_subtree(height, self.return_type)

        self.node_list = None
        self.node_id_list = None

    # Run the program the tree represents
    def execute(self, context: dict[str, Any]) -> Any:
        """Executes the tree with the given context and returns the result.
        The context is updated with the fixed context before execution.

        Args:
            context: A dictionary of information to be used by the tree.
            The necessary values are defined by the node type."""
        context = context.copy()
        context.update(self.fixed_context)
        return self.root.execute(context)

    def _replace_subtree(self, node: GPNode, replacement: GPNode) -> None:
        """Replaces a subtree with a replacement subtree.

        Args:
            node: The root of the subtree to be replaced.
            replacement: The root of the new subtree."""

        if node is self.root:
            self.root = replacement
            if self.root.get_height() > MAXIMUM_HEIGHT:
                raise Exception(f"New tree exceeds maximum height of {MAXIMUM_HEIGHT}.")
        else:
            parent = node.parent
            index = parent.input_nodes.index(node)
            parent.input_nodes[index] = replacement
            replacement.set_parent(parent)
            if self.root.get_height() > MAXIMUM_HEIGHT:
                raise Exception(f"New tree exceeds maximum height of {MAXIMUM_HEIGHT}.")
        self.node_list = None  # Invalidate the cached node list
        self.node_id_list = None  # Invalidate the cached node ID list

    def _replace_point(self, node: GPNode, replacement: GPNode) -> None:
        """Replaces a node with a new node, transferring its children.
        Type and -arity compatibility is not checked here.

        Args:
            node: The node to be replaced
            replacement: A node with the same input and output types as `node`. It should not have any children yet.
        """
        replacement.input_nodes = node.input_nodes
        for child in replacement.input_nodes:
            child.set_parent(replacement)
        if node is self.root:
            self.root = replacement
        else:
            parent = node.parent
            index = parent.input_nodes.index(node)
            parent.input_nodes[index] = replacement
            replacement.set_parent(parent)
        self.node_list = None  # Invalidate the cached node list
        self.node_id_list = None  # Invalidate the cached node ID list

    def _random_node(self, output_type: int = None, max_depth: int = None) -> GPNode:
        """Selects a random node from the tree.

        Args:
            output_type: The type of the node to be selected. If None, any type is allowed.
            max_depth: The maximum depth of the node to be selected. If None, any depth is allowed.

        Returns:
            A random node from the tree, matching any parameters specified.

        Raises:
            ValueError: If no nodes matching the parameters are found in the tree.
        """
        node_list = self.get_node_list()
        if output_type is not None:
            node_list = [node for node in node_list if node.output_type == output_type]
            if len(node_list) == 0:
                raise ValueError(f"No valid nodes of type {output_type} found in the tree.")
        if max_depth is not None:
            node_list = [node for node in node_list if node.get_depth() <= max_depth]
            if len(node_list) == 0:
                raise ValueError(f"No valid nodes found at depth {max_depth} or less in the tree.")
        return random.choice(node_list)

    def mutate(self) -> None:
        """Mutates the tree in place.
        The mutation method is chosen randomly between subtree mutation and point mutation."""
        mutation_function = random.choices((self._subtree_mutate, self._point_mutate), (0.5, 0.5), k=1)[0]
        mutation_function()

    def _subtree_mutate(self) -> None:
        """Selects a random subtree in the tree, and randomly regenerates it.
        The height of the new subtree is chosen randomly around the height of the old subtree."""
        old_subtree = self._random_node()
        mean_height = old_subtree.get_height()
        generate_height = int(round(random.gauss(mean_height, SUBTREE_MUTATE_HEIGHT_SIGMA)))
        # Don't allow the new subtree to exceed the maximum height
        generate_height = min(max(1, generate_height), MAXIMUM_HEIGHT - old_subtree.get_depth())
        new_subtree = self.random_subtree(generate_height, old_subtree.output_type)
        try:
            self._replace_subtree(old_subtree, new_subtree)
        except Exception as e:
            warn(f"Subtree mutation failed; retrying: {e}")
            self._subtree_mutate()
            return
        self.creation_method = "Mutation"

    def _point_mutate(self) -> None:
        """Selects a random node in the tree and replaces it with a random node with the same input and output types,
        preserving the original node's children."""
        node = self._random_node()
        new_node = self.node_type(self.node_type.random_function(node.output_type, child_types=node.input_types), fixed_context=self.fixed_context)
        self._replace_point(node, new_node)
        self.creation_method = "Mutation"

    # Subtree recombination
    def recombine(self, donor: 'GPTree') -> None:
        """Subtree crossover: replaces a random subtree of this tree with a random subtree from the donor.

        Args:
            donor: The other parent tree to recombine with.
        """
        # Retry if a node has no valid recombination point.
        for _ in range(100):
            donor_subtree = donor._random_node()
            # Don't place the donor subtree in a position that would exceed the maximum height
            maximum_depth = MAXIMUM_HEIGHT - donor_subtree.get_height()
            try:
                parent_subtree = self._random_node(output_type=donor_subtree.output_type, max_depth=maximum_depth)
                break
            except ValueError:
                continue
        else:
            warn("Recombination failed to find a valid pair of nodes to recombine. Using mutation instead.")
            self.mutate()
            return

        self._replace_subtree(parent_subtree, donor_subtree)

        self.parent_ids.append(donor.id)
        self.creation_method = "Recombination"

    # Creates a deep copy of the tree
    def clone(self) -> 'GPTree':
        cloned_genotype = GPTree({**self.parameters, "id_list": self.get_node_id_list()})
        cloned_genotype.parent_ids.append(self.id)
        cloned_genotype.creation_method = "Cloning"
        return cloned_genotype

    def get_node_list(self) -> list[GPNode]:
        """Generates a preorder list of nodes for the tree.
        Cached in :attr:`node_list`, and invalidated by any method that changes the tree.

        Returns:
            A preorder list of nodes for the tree.
        """
        if self.node_list is not None:
            return self.node_list
        node_list = self.root.get_node_list()
        self.node_list = node_list
        return node_list

    def get_node_id_list(self) -> list[str | Any]:
        """Generates a preorder list of node IDs and literals for the tree.
        This list can be used as the :attr:`GPTreeParameters.id_list` parameter to regenerate the tree.
        All elements are string ids, except for literals,
        which are string ids followed by the literal values themselves.
        Cached in :attr:`node_id_list`, and invalidated by any method that changes the tree.

        Returns:
            A preorder list of node IDs for the tree.
        """
        if self.node_id_list is not None:
            return self.node_id_list
        id_list = list()
        for node in self.root.traverse_pre_order():
            id_list.append(node.function_id)
            if node.function_id in self.node_type.literals:
                id_list.append(node.literal)
        self.node_id_list = id_list
        return id_list

    def __str__(self):
        tree_string = self.root.tree_string(self.root.get_height(), 0)
        return tree_string + "ID List: " + str(self.get_node_id_list()) + "\n"

    def __len__(self):
        return len(self.get_node_id_list())

    # Positive is good, negative is bad
    def get_fitness_modifier(self, raw_fitness):
        # Fitness modifier is added, so negate to make it a penalty
        penalty = -self.parsimony_weight * len(self)
        return penalty

    # Generate a tree given an id list
    def _generate_from_list(self, id_list):
        primitive_function = id_list.pop(0)
        literal = None
        if primitive_function in self.node_type.literals:
            literal = id_list.pop(0)
            if isinstance(literal, list):
                # TODO: Consider storing a type conversion function for literals
                literal = tuple(literal)

        node = self.node_type(primitive_function, literal=literal, fixed_context=self.fixed_context)
        for input_type in node.input_types:
            child = self._generate_from_list(id_list)
            child.set_parent(node)
            node.add_input(child)
        return node

    # Generate a random tree with the requested height and output type
    def random_subtree(self, height, output_type):
        if random.random() < 0.5:
            return self.generate_grow_soft(height, output_type)
        else :
            return self.generate_full_soft(height, output_type)

    def generate_grow_soft(self, height, output_type):
        """Generates a tree using a relaxed version of the Grow method, without strictly enforcing the height limit
        (computationally expensive for strongly-typed GP).

        Args:
            height: The desired height of the tree to be generated.
            Due to the relaxed restrictions, the actual height may differ.
            output_type: The type of the root node of the tree to be generated.

        Returns:
            A :class:`GPNode` representing the root of the generated tree.
        """
        if height <= 1:
            node_function = self.node_type.random_function(output_type, terminal=True, forbidden_nodes=self.forbidden_nodes)
            if node_function is None:
                return self.generate_grow_soft(height + 1, output_type)
        else:
            node_function = self.node_type.random_function(output_type, forbidden_nodes=self.forbidden_nodes)

        node = self.node_type(node_function, fixed_context=self.fixed_context)
        for input_type in node.input_types:
            child = self.generate_grow_soft(height - 1, input_type)
            child.set_parent(node)
            node.add_input(child)
        return node

    def generate_full_soft(self, height, output_type):
        """Generates a tree using a relaxed version of the Full method, without strictly enforcing the height limit
        (computationally expensive for strongly-typed GP).

        Args:
            height: The desired height of the tree to be generated.
            Due to the relaxed restrictions, the actual height may differ.
            output_type: The type of the root node of the tree to be generated.

        Returns:
            A :class:`GPNode` representing the root of the generated tree.
        """
        if height <= 1:
            node_function = self.node_type.random_function(output_type, terminal=True, forbidden_nodes=self.forbidden_nodes)
            if node_function is None:
                return self.generate_grow_soft(height + 1, output_type)
        else:
            node_function = self.node_type.random_function(output_type, terminal=False, forbidden_nodes=self.forbidden_nodes)
            if node_function is None:
                node_function = self.node_type.random_function(output_type, terminal=True, forbidden_nodes=self.forbidden_nodes)

        node = self.node_type(node_function, fixed_context=self.fixed_context)
        for input_type in node.input_types:
            child = self.generate_grow_soft(height - 1, input_type)
            child.set_parent(node)
            node.add_input(child)
        return node

    # Generates a tree guaranteed to be a certain depth
    def generateDeep(self, height, outputType):
        node = None
        if height > 1:
            # Find the list of possible functions to choose that can reach the desired depth
            typeChildren = self.node_type.type_functions[outputType]
            deepChildren = list()
            for function in typeChildren:
                if function in self.forbidden_nodes:
                    continue
                functionInfo = self.node_type.get_function_data(function)
                # Such functions have an input of a type that can reach exactly one less than that depth
                if self.growTable[height - 1].issuperset(functionInfo[2]):  # Ensure that the function can stop at that depth (Using the depth table)
                    for inputType in functionInfo[2]:
                        if self.depthTable[inputType] >= height - 1:  # Ensure that the function can reach that depth (Using the Grow restrictions for typed GP)
                            deepChildren.append(function)
                            break

            # This can only happen if it is literally impossible to build a tree of the desired height. Relax height limit. #Todo: Incorporate more detailed checks into depth table
            if len(deepChildren) == 0:
                print("Failure in full table (type 1). Relaxing restrictions.")
                return self.generateGrow(height, outputType)

            # Select a random function
            node = self.node_type(random.choice(deepChildren), fixed_context=self.fixed_context)

            # Assign the inputs with a random start, picking one to generate to the desired depth and generating the rest through grow
            randStart = 0
            if (len(node.input_types) > 0):
                randStart = random.randrange(len(node.input_types))
            deepComplete = False
            inputs = [None] * len(node.input_types)
            for i in range(randStart, randStart + len(node.input_types)):
                inputType = node.input_types[i % len(node.input_types)]
                child = None
                if not deepComplete and self.depthTable[inputType] >= height - 1:
                    child = self.generateDeep(height - 1, inputType)
                    deepComplete = True
                else:
                    child = self.generateGrow(height - 1, inputType)
                inputs[i % len(node.input_types)] = child

            for child in inputs:
                child.set_parent(node)
                node.add_input(child)

        # Use terminal nodes at the depth limit
        else:
            if self.node_type.random_function(outputType, terminal=True, forbidden_nodes=self.forbidden_nodes) is None:
                print("Failure in full table (type 2). Relaxing restrictions.")
                return self.generateGrow(height, outputType)
            node = self.node_type(outputType, terminal=True, forbidden_nodes=self.forbidden_nodes, fixed_context=self.fixed_context)

        return node

    # Generates a tree using the grow method for strongly-typed GP
    def generateGrow(self, height, outputType):
        node = None
        if height > 1:
            typeList = self.node_type.type_functions[outputType]
            randStart = random.randrange(len(typeList))
            for i in range(randStart, randStart + len(typeList)):
                function = typeList[i % len(typeList)]
                if function in self.forbidden_nodes:
                    continue
                functionInfo = self.node_type.get_function_data(function)
                if self.growTable[height - 1].issuperset(functionInfo[2]):
                    node = self.node_type(outputType, function, fixed_context=self.fixed_context)
                    break
            if node is None:
                print("Failure in grow table (type 1). Relaxing restrictions.")
                return self.generateGrow(height + 1, outputType)

        else:
            if self.node_type.random_function(outputType, terminal=True, forbidden_nodes=self.forbidden_nodes) is None:
                print("Failure in grow table (type 2). Relaxing restrictions.")
                return self.generateGrow(height + 1, outputType)
            node = self.node_type(outputType, terminal=True, forbidden_nodes=self.forbidden_nodes, fixed_context=self.fixed_context)

        for inputType in node.input_types:
            child = self.generateGrow(height - 1, inputType)
            child.set_parent(node)
            node.add_input(child)
        return node

    def get_raw_genotype(self):
        return {"node_type": str(self.node_type), "id_list": self.get_node_id_list()}

    def __hash__(self):
        return tuple(self.get_node_id_list()).__hash__()

    def diversity_function(self, population, reference=None, samples=None):
        return edit_diversity(population, reference, samples)
